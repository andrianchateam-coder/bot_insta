"""Domain models shared by the Telegram UI and background worker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class SourceType(StrEnum):
    TELEGRAM = "telegram"
    GOOGLE_DRIVE = "google_drive"


class JobStatus(StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    READY = "ready"
    APPROVED = "approved"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    chat_id: int
    user_id: int
    source_type: SourceType
    original_name: str
    source_path: str | None
    source_url: str | None
    status: JobStatus
    output_path: str | None
    cover_path: str | None
    subtitle_path: str | None
    transcript_path: str | None
    language: str | None
    hook: str | None
    caption: str | None
    public_token: str
    instagram_container_id: str | None
    instagram_media_id: str | None
    instagram_permalink: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime
