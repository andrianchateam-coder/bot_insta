"""Restart-safe sequential media and publishing worker."""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
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


async def _download_drive_file(
    file_id: str,
    destination: Path,
    *,
    max_bytes: int,
    timeout_seconds: float,
) -> None:
    """Run gdown in a killable process with an OS-enforced file-size quota."""

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "gdown",
        "--id",
        file_id,
        "--output",
        str(destination),
        "--quiet",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        # Linux enforces the quota in the child even if it downloads faster
        # than the parent can poll. Polling still protects other platforms.
        try:
            import resource

            if hasattr(resource, "prlimit"):
                resource.prlimit(
                    process.pid,
                    resource.RLIMIT_FSIZE,
                    (max_bytes, max_bytes),
                )
        except (OSError, ValueError):
            pass

        communicate = asyncio.create_task(process.communicate())
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        output: tuple[bytes, bytes] | None = None
        while output is None:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Google Drive download timed out")
            try:
                output = await asyncio.wait_for(
                    asyncio.shield(communicate),
                    timeout=min(0.25, remaining),
                )
            except TimeoutError:
                try:
                    downloaded = sum(
                        item.stat().st_size
                        for item in destination.parent.iterdir()
                        if item.is_file()
                    )
                except OSError:
                    continue
                if downloaded > max_bytes:
                    raise RuntimeError(
                        "Google Drive source exceeds the configured size limit"
                    ) from None

        _stdout, stderr = output
        if process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()[:500]
            raise RuntimeError(
                "Google Drive download failed"
                + (f": {detail}" if detail else "")
            )
    except BaseException:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await asyncio.gather(communicate, return_exceptions=True)
        for partial in destination.parent.glob(f"{destination.name}*.part"):
            partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise

    if not destination.is_file() or destination.stat().st_size == 0:
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
        self._last_cleanup_monotonic = 0.0

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
            timeout_seconds=self.settings.media_command_timeout_seconds,
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
            timeout_seconds=self.settings.media_command_timeout_seconds,
        )
        cover = create_cover(
            source,
            directory / "cover.jpg",
            metadata.cover_hook,
            ffmpeg=self.settings.ffmpeg_path,
            font_path=self.settings.font_path,
            timeout_seconds=self.settings.media_command_timeout_seconds,
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
            max_bytes = self.settings.max_drive_download_mb * 1024 * 1024
            if self.drive_downloader is _download_drive_file:
                await _download_drive_file(
                    file_id,
                    source,
                    max_bytes=max_bytes,
                    timeout_seconds=self.settings.drive_download_timeout_seconds,
                )
            else:
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self.drive_downloader, file_id, source),
                        timeout=self.settings.drive_download_timeout_seconds,
                    )
                except TimeoutError as exc:
                    source.unlink(missing_ok=True)
                    raise RuntimeError("Google Drive download timed out") from exc
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError("Google Drive source file is unavailable")
        if source.stat().st_size > self.settings.max_drive_download_mb * 1024 * 1024:
            source.unlink(missing_ok=True)
            raise RuntimeError("Google Drive source exceeds the configured size limit")
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
                f"Не удалось обработать ролик {job.id}. "
                f"Для повтора без новой загрузки отправьте: /retry {job.id}",
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
                "Файл сохранён; проверьте настройки Instagram и отправьте: "
                f"/retry {job.id}",
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

    async def cleanup_expired_media(self, *, now: datetime | None = None) -> int:
        """Remove old terminal-job files while retaining their SQLite history."""

        cutoff = (now or datetime.now(UTC)) - timedelta(
            days=self.settings.media_retention_days
        )
        jobs = await self.database.list_jobs(
            statuses=[
                JobStatus.PUBLISHED,
                JobStatus.REJECTED,
                JobStatus.FAILED,
            ],
            limit=1_000,
        )
        removed = 0
        for job in jobs:
            if job.updated_at > cutoff:
                continue
            directory = (self.media_root / job.id).resolve()
            if self.media_root not in directory.parents:
                LOGGER.warning("Refusing to clean unsafe media path for job %s", job.id)
                continue
            try:
                if directory.is_dir():
                    await asyncio.to_thread(shutil.rmtree, directory)
                await self.database.update_job(
                    job.id,
                    source_path=None,
                    output_path=None,
                    cover_path=None,
                    subtitle_path=None,
                    transcript_path=None,
                )
                removed += 1
            except OSError as error:
                LOGGER.warning(
                    "Could not clean media for job %s (%s)",
                    job.id,
                    type(error).__name__,
                )
        return removed

    async def _maybe_cleanup(self) -> None:
        current = time.monotonic()
        if current - self._last_cleanup_monotonic < 86_400:
            return
        self._last_cleanup_monotonic = current
        try:
            removed = await self.cleanup_expired_media()
            if removed:
                LOGGER.info("Cleaned media for %d expired job(s)", removed)
        except Exception as error:
            LOGGER.warning("Media cleanup failed (%s)", type(error).__name__)

    async def run(self, stop_event: asyncio.Event) -> None:
        await self.database.recover_interrupted_jobs()
        while not stop_event.is_set():
            await self._maybe_cleanup()
            if await self.run_once():
                continue
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=self.settings.worker_poll_seconds
                )
            except TimeoutError:
                pass


__all__ = ["ProcessedArtifacts", "ReelWorker"]
