"""Application service used by the Telegram handlers.

The UI deliberately does not write files or mutate queue state itself. This
adapter keeps those operations small, validated and independently testable.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from aiogram import Bot

from .db import Database
from .models import Job, JobStatus, SourceType

_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".webm"})
_DRIVE_ID_RE = re.compile(r"[A-Za-z0-9_-]{10,200}\Z")


class JobStateError(RuntimeError):
    """A review action was invalid for the current durable state."""


class PublishingNotConfigured(JobStateError):
    """Instagram publishing has not been connected for this runtime."""


class ReelService:
    """Download submissions and expose safe queue transitions to Aiogram."""

    def __init__(
        self,
        database: Database,
        bot: Bot,
        media_root: str | Path,
        *,
        admin_ids: frozenset[int] | set[int] = frozenset(),
        max_upload_bytes: int | None = None,
        publishing_enabled: bool = True,
    ) -> None:
        self.database = database
        self.bot = bot
        self.media_root = Path(media_root).resolve()
        self.admin_ids = frozenset(admin_ids)
        self.max_upload_bytes = max_upload_bytes
        self.publishing_enabled = publishing_enabled

    def _job_directory(self, job_id: str) -> Path:
        directory = (self.media_root / job_id).resolve()
        if self.media_root not in directory.parents:
            raise ValueError("Unsafe job directory")
        directory.mkdir(parents=True, exist_ok=False)
        return directory

    async def enqueue_telegram_file(
        self,
        *,
        chat_id: int,
        user_id: int,
        file_id: str,
        file_unique_id: str,
        file_name: str,
        file_size: int | None,
        mime_type: str | None,
    ) -> Job:
        del file_unique_id, mime_type
        if (
            self.max_upload_bytes is not None
            and file_size is not None
            and file_size > self.max_upload_bytes
        ):
            raise ValueError("Telegram file exceeds the configured upload limit")
        job_id = uuid.uuid4().hex
        directory = self._job_directory(job_id)
        suffix = Path(file_name).suffix.lower()
        if suffix not in _VIDEO_SUFFIXES:
            suffix = ".mp4"
        source = directory / f"source{suffix}"
        try:
            await self.bot.download(file_id, destination=source)
            if not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError("Telegram returned an empty file")
            if (
                self.max_upload_bytes is not None
                and source.stat().st_size > self.max_upload_bytes
            ):
                raise ValueError("Telegram file exceeds the configured upload limit")
            return await self.database.create_job(
                job_id=job_id,
                chat_id=chat_id,
                user_id=user_id,
                source_type=SourceType.TELEGRAM,
                original_name=Path(file_name).name[:255] or f"video{suffix}",
                source_path=source,
            )
        except Exception:
            # The directory contains only this attempted download because its
            # random ID was not visible to any worker before create_job.
            if source.exists():
                source.unlink()
            try:
                directory.rmdir()
            except OSError:
                pass
            raise

    async def enqueue_drive_file(
        self,
        *,
        chat_id: int,
        user_id: int,
        drive_url: str,
        drive_file_id: str,
    ) -> Job:
        del drive_url  # Store a canonical URL rather than arbitrary chat text.
        if not _DRIVE_ID_RE.fullmatch(drive_file_id):
            raise ValueError("Invalid Google Drive file ID")
        return await self.database.create_job(
            chat_id=chat_id,
            user_id=user_id,
            source_type=SourceType.GOOGLE_DRIVE,
            original_name=f"{drive_file_id}.mp4",
            source_url=f"https://drive.google.com/uc?id={drive_file_id}",
        )

    async def recent_jobs(self, *, chat_id: int, limit: int = 5) -> list[Job]:
        return await self.database.list_jobs(chat_id=chat_id, limit=limit)

    async def get_job(self, job_id: str) -> Job | None:
        return await self.database.get_job(job_id)

    async def _review_transition(
        self,
        *,
        job_id: str,
        actor_id: int,
        new_status: JobStatus,
    ) -> Job:
        job = await self.database.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if actor_id != job.user_id and actor_id not in self.admin_ids:
            raise PermissionError("The actor cannot manage this job")
        if new_status is JobStatus.APPROVED and not self.publishing_enabled:
            raise PublishingNotConfigured(
                "Instagram publishing is not configured"
            )
        changed = await self.database.transition_job(
            job_id,
            [JobStatus.READY],
            new_status,
            error=None,
        )
        if changed is None:
            raise JobStateError(f"Job {job_id} is already reviewed")
        return changed

    async def approve_job(self, *, job_id: str, actor_id: int) -> Job:
        return await self._review_transition(
            job_id=job_id,
            actor_id=actor_id,
            new_status=JobStatus.APPROVED,
        )

    async def reject_job(self, *, job_id: str, actor_id: int) -> Job:
        return await self._review_transition(
            job_id=job_id,
            actor_id=actor_id,
            new_status=JobStatus.REJECTED,
        )

    async def retry_job(self, *, job_id: str, actor_id: int) -> Job:
        """Retry a failed processing or publishing stage without re-uploading.

        A job with complete processed artifacts resumes publication; otherwise
        it returns to the processing queue. The compare-and-set transition
        keeps repeated commands from scheduling the same job twice.
        """

        job = await self.database.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if actor_id != job.user_id and actor_id not in self.admin_ids:
            raise PermissionError("The actor cannot manage this job")

        processed_paths = (job.output_path, job.cover_path)
        processed_ready = bool(job.caption)
        for stored_path in processed_paths:
            if not stored_path:
                processed_ready = False
                break
            path = Path(stored_path).resolve()
            if self.media_root not in path.parents or not path.is_file():
                processed_ready = False
                break
        target = JobStatus.APPROVED if processed_ready else JobStatus.QUEUED
        if target is JobStatus.APPROVED and not self.publishing_enabled:
            raise PublishingNotConfigured(
                "Instagram publishing is not configured"
            )
        changed = await self.database.transition_job(
            job.id,
            [JobStatus.FAILED],
            target,
            error=None,
        )
        if changed is None:
            raise JobStateError(f"Job {job_id} is not retryable")
        return changed


__all__ = ["JobStateError", "PublishingNotConfigured", "ReelService"]
