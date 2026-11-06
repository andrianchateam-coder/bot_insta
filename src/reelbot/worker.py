"""Restart-safe sequential media and publishing worker."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aiogram import Bot

from .ai import OpenAIReelAI
from .bot import build_public_media_url, extract_google_drive_file_id, send_preview
from .config import Settings
from .db import Database
from .instagram import InstagramPublisher
from .media import burn_subtitles, create_cover, extract_audio, write_ass, write_srt
from .models import Job, JobStatus, SourceType

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessedArtifacts:
    output_path: Path
    cover_path: Path
    subtitle_path: Path
    transcript_path: Path
    language: str
    hook: str
    caption: str


Processor = Callable[[Path, Path], ProcessedArtifacts]
DriveDownloader = Callable[[str, Path], Any]
PreviewSender = Callable[..., Awaitable[None]]


def _download_drive_file(file_id: str, destination: Path) -> None:
    import gdown

    result = gdown.download(id=file_id, output=str(destination), quiet=True)
    if not result or not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("Google Drive did not return a video file")


class ReelWorker:
    """Process one durable job at a time and publish only approved jobs."""

    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        bot: Bot,
        instagram_publisher: InstagramPublisher | None = None,
        ai: OpenAIReelAI | None = None,
        processor: Processor | None = None,
        drive_downloader: DriveDownloader = _download_drive_file,
        preview_sender: PreviewSender = send_preview,
    ) -> None:
        self.settings = settings
        self.database = database
        self.bot = bot
        self.instagram_publisher = instagram_publisher
        self.drive_downloader = drive_downloader
        self.preview_sender = preview_sender
        self.media_root = settings.media_root.resolve()

        if processor is not None:
            self.processor = processor
            self.ai = ai
        else:
            if ai is None:
                if settings.openai_api_key is None:
                    raise ValueError(
                        "REELBOT_OPENAI_API_KEY is required to process videos"
                    )
                ai = OpenAIReelAI(
                    api_key=settings.openai_api_key.get_secret_value(),
                    transcription_model=settings.openai_transcription_model,
                    copy_model=settings.openai_chat_model,
                )
            self.ai = ai
            self.processor = self._process_sync

    def _job_directory(self, job_id: str) -> Path:
        directory = (self.media_root / job_id).resolve()
        if self.media_root not in directory.parents:
            raise ValueError("Unsafe job directory")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _process_sync(self, source: Path, directory: Path) -> ProcessedArtifacts:
        assert self.ai is not None
        audio = extract_audio(
            source,
            directory / "speech.mp3",
            ffmpeg=self.settings.ffmpeg_path,
        )
        transcription = self.ai.transcribe(audio)
        metadata = self.ai.generate_metadata(
            transcription.text,
            language=transcription.language,
        )
        transcript = directory / "transcript.txt"
        transcript.write_text(transcription.text, encoding="utf-8")
        srt = write_srt(transcription.segments, directory / "subtitles.srt")
        ass = write_ass(transcription.segments, directory / "subtitles.ass")
        output = burn_subtitles(
            source,
            ass,
            directory / "ready.mp4",
            ffmpeg=self.settings.ffmpeg_path,
            preset=self.settings.video_preset,
            crf=self.settings.video_crf,
        )
        cover = create_cover(
            source,
            directory / "cover.jpg",
            metadata.cover_hook,
            ffmpeg=self.settings.ffmpeg_path,
            font_path=self.settings.font_path,
        )
        return ProcessedArtifacts(
            output_path=output,
            cover_path=cover,
            subtitle_path=srt,
            transcript_path=transcript,
            language=transcription.language,
            hook=metadata.cover_hook,
            caption=metadata.instagram_caption,
        )

    async def _source_for_job(self, job: Job, directory: Path) -> Path:
        if job.source_type is SourceType.TELEGRAM:
            if not job.source_path:
                raise RuntimeError("Telegram source path is missing")
            source = Path(job.source_path).resolve()
            if self.media_root not in source.parents or not source.is_file():
                raise RuntimeError("Telegram source file is unavailable")
            return source

        file_id = extract_google_drive_file_id(job.source_url)
        if not file_id:
            raise RuntimeError("Stored Google Drive URL is invalid")
        source = directory / "source.mp4"
        if not source.is_file() or source.stat().st_size == 0:
            await asyncio.to_thread(self.drive_downloader, file_id, source)
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError("Google Drive source file is unavailable")
        await self.database.update_job(job.id, source_path=source)
        return source

    def _safe_error(self, error: BaseException) -> str:
        detail = f"{type(error).__name__}: {error}"
        for secret in (
            self.settings.telegram_bot_token,
            self.settings.openai_api_key,
            self.settings.meta_access_token,
        ):
            if secret is not None:
                value = secret.get_secret_value()
                if value:
                    detail = detail.replace(value, "[REDACTED]")
        return detail[:1000]

    async def _notify(self, chat_id: int, text: str) -> None:
        try:
            await self.bot.send_message(chat_id, text)
        except Exception as error:
            LOGGER.warning(
                "Telegram notification failed (%s)", type(error).__name__
            )

    async def _process(self, job: Job) -> None:
        try:
            directory = self._job_directory(job.id)
            source = await self._source_for_job(job, directory)
            artifacts = await asyncio.to_thread(self.processor, source, directory)
            ready = await self.database.update_job(
                job.id,
                status=JobStatus.READY,
                output_path=artifacts.output_path,
                cover_path=artifacts.cover_path,
                subtitle_path=artifacts.subtitle_path,
                transcript_path=artifacts.transcript_path,
                language=artifacts.language,
                hook=artifacts.hook,
                caption=artifacts.caption,
                error=None,
            )
        except Exception as error:
            LOGGER.error("Processing job %s failed (%s)", job.id, type(error).__name__)
            await self.database.update_job(
                job.id,
                status=JobStatus.FAILED,
                error=self._safe_error(error),
            )
            await self._notify(
                job.chat_id,
                f"Не удалось обработать ролик {job.id}. Попробуйте отправить его ещё раз.",
            )
            return

        # A Telegram outage must not discard successfully produced artifacts.
        try:
            preview = asdict(ready)
            preview["cover_hook"] = ready.hook
            await self.preview_sender(
                self.bot,
                preview,
                max_telegram_bytes=self.settings.max_preview_mb * 1024 * 1024,
                public_base_url=self.settings.public_base_url,
            )
        except Exception as error:
            LOGGER.warning(
                "Preview delivery for job %s failed (%s)",
                job.id,
                type(error).__name__,
            )

    async def _publish(self, job: Job) -> None:
        try:
            if self.instagram_publisher is None or not self.settings.public_base_url:
                raise RuntimeError("Instagram publishing is not configured")
            container_id = job.instagram_container_id
            if not container_id:
                video_url = build_public_media_url(
                    self.settings.public_base_url,
                    job_id=job.id,
                    public_token=job.public_token,
                    kind="video",
                )
                cover_url = build_public_media_url(
                    self.settings.public_base_url,
                    job_id=job.id,
                    public_token=job.public_token,
                    kind="cover",
                )
                if not video_url or not cover_url or not job.caption:
                    raise RuntimeError("Processed media or caption is missing")
                container_id = (
                    await self.instagram_publisher.create_reel_container(
                        video_url=video_url,
                        cover_url=cover_url,
                        caption=job.caption,
                        share_to_feed=True,
                    )
                )
                # Persist before waiting so a restart resumes this exact Meta
                # container rather than creating a duplicate Reel.
                job = await self.database.update_job(
                    job.id,
                    instagram_container_id=container_id,
                    error=None,
                )

            media_id = job.instagram_media_id
            if not media_id:
                await self.instagram_publisher.wait_until_finished(container_id)
                media_id = await self.instagram_publisher.publish_container(
                    container_id
                )
                job = await self.database.update_job(
                    job.id,
                    instagram_media_id=media_id,
                    error=None,
                )

            permalink = job.instagram_permalink
            if not permalink:
                permalink = await self.instagram_publisher.fetch_permalink(media_id)
            await self.database.update_job(
                job.id,
                status=JobStatus.PUBLISHED,
                instagram_container_id=container_id,
                instagram_media_id=media_id,
                instagram_permalink=permalink,
                error=None,
            )
            message = "Рилс опубликован."
            if permalink:
                message += f"\n{permalink}"
            await self._notify(job.chat_id, message)
        except Exception as error:
            LOGGER.error("Publishing job %s failed (%s)", job.id, type(error).__name__)
            await self.database.update_job(
                job.id,
                status=JobStatus.FAILED,
                error=self._safe_error(error),
            )
            await self._notify(
                job.chat_id,
                f"Не удалось опубликовать ролик {job.id}. "
                "Файл сохранён; проверьте настройки Instagram.",
            )

    async def run_once(self) -> bool:
        """Process at most one job, prioritising an explicit publication approval."""

        approved = await self.database.claim_next(
            [JobStatus.APPROVED], JobStatus.PUBLISHING
        )
        if approved is not None:
            await self._publish(approved)
            return True

        queued = await self.database.claim_next(
            [JobStatus.QUEUED], JobStatus.PROCESSING
        )
        if queued is not None:
            await self._process(queued)
            return True
        return False

    async def run(self, stop_event: asyncio.Event) -> None:
        await self.database.recover_interrupted_jobs()
        while not stop_event.is_set():
            if await self.run_once():
                continue
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=self.settings.worker_poll_seconds
                )
            except TimeoutError:
                pass


__all__ = ["ProcessedArtifacts", "ReelWorker"]
