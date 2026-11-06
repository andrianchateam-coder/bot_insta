"""Durable SQLite job queue.

Every public method uses a short-lived connection.  ``claim_next`` takes an
immediate transaction and performs a compare-and-set update, so two worker
loops cannot process the same job.
"""

from __future__ import annotations

import secrets
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from .models import Job, JobStatus, SourceType

_UPDATABLE_FIELDS = frozenset(
    {
        "source_path",
        "source_url",
        "status",
        "output_path",
        "cover_path",
        "subtitle_path",
        "transcript_path",
        "language",
        "hook",
        "caption",
        "instagram_container_id",
        "instagram_media_id",
        "instagram_permalink",
        "error",
    }
)


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def _connect(self) -> aiosqlite.Connection:
        connection = await aiosqlite.connect(self.path)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA busy_timeout = 5000")
        await connection.execute("PRAGMA foreign_keys = ON")
        return connection

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = await self._connect()
        try:
            await connection.execute("PRAGMA journal_mode = WAL")
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    source_path TEXT,
                    source_url TEXT,
                    status TEXT NOT NULL,
                    output_path TEXT,
                    cover_path TEXT,
                    subtitle_path TEXT,
                    transcript_path TEXT,
                    language TEXT,
                    hook TEXT,
                    caption TEXT,
                    public_token TEXT NOT NULL UNIQUE,
                    instagram_container_id TEXT,
                    instagram_media_id TEXT,
                    instagram_permalink TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK (source_type IN ('telegram', 'google_drive')),
                    CHECK (status IN (
                        'queued', 'downloading', 'processing', 'ready',
                        'approved', 'publishing', 'published', 'rejected',
                        'failed'
                    ))
                )
                """
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_status_created_idx "
                "ON jobs(status, created_at)"
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_chat_created_idx "
                "ON jobs(chat_id, created_at DESC)"
            )
            await connection.commit()
        finally:
            await connection.close()

    async def create_job(
        self,
        *,
        chat_id: int,
        user_id: int,
        source_type: SourceType | str,
        original_name: str,
        source_path: str | Path | None = None,
        source_url: str | None = None,
        job_id: str | None = None,
    ) -> Job:
        source_type = SourceType(source_type)
        normalized_path = str(source_path) if source_path is not None else None
        normalized_url = source_url.strip() if source_url else None
        self._validate_source(source_type, normalized_path, normalized_url)

        now = datetime.now(UTC)
        identifier = job_id or uuid.uuid4().hex
        public_token = secrets.token_urlsafe(32)
        status = JobStatus.QUEUED

        connection = await self._connect()
        try:
            await connection.execute(
                """
                INSERT INTO jobs (
                    id, chat_id, user_id, source_type, original_name,
                    source_path, source_url, status, public_token,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    chat_id,
                    user_id,
                    source_type.value,
                    original_name,
                    normalized_path,
                    normalized_url,
                    status.value,
                    public_token,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            await connection.commit()
        finally:
            await connection.close()

        job = await self.get_job(identifier)
        assert job is not None
        return job

    @staticmethod
    def _validate_source(
        source_type: SourceType,
        source_path: str | None,
        source_url: str | None,
    ) -> None:
        if source_type is SourceType.TELEGRAM:
            if not source_path or source_url:
                raise ValueError("Telegram jobs require source_path only")
        elif not source_url or source_path:
            raise ValueError("Google Drive jobs require source_url only")

    async def get_job(self, job_id: str) -> Job | None:
        connection = await self._connect()
        try:
            cursor = await connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            )
            row = await cursor.fetchone()
        finally:
            await connection.close()
        return self._row_to_job(row) if row is not None else None

    async def update_job(self, job_id: str, **fields: Any) -> Job:
        if not fields:
            job = await self.get_job(job_id)
            if job is None:
                raise KeyError(job_id)
            return job

        unknown = set(fields) - _UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Fields cannot be updated: {', '.join(sorted(unknown))}")

        normalized: dict[str, Any] = {}
        for name, value in fields.items():
            if name == "status" and value is not None:
                value = JobStatus(value).value
            elif isinstance(value, Path):
                value = str(value)
            normalized[name] = value
        normalized["updated_at"] = datetime.now(UTC).isoformat()

        assignments = ", ".join(f"{name} = ?" for name in normalized)
        values = [*normalized.values(), job_id]
        connection = await self._connect()
        try:
            cursor = await connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?", values
            )
            if cursor.rowcount != 1:
                await connection.rollback()
                raise KeyError(job_id)
            await connection.commit()
        finally:
            await connection.close()

        job = await self.get_job(job_id)
        assert job is not None
        return job

    async def claim_next(
        self,
        statuses: Iterable[JobStatus | str],
        new_status: JobStatus | str,
    ) -> Job | None:
        """Atomically move the oldest matching job into ``new_status``."""

        old_statuses = tuple(JobStatus(status).value for status in statuses)
        if not old_statuses:
            raise ValueError("At least one source status is required")
        target_status = JobStatus(new_status).value
        placeholders = ", ".join("?" for _ in old_statuses)
        now = datetime.now(UTC).isoformat()

        connection = await self._connect()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            cursor = await connection.execute(
                f"SELECT id, status FROM jobs WHERE status IN ({placeholders}) "
                "ORDER BY created_at ASC, id ASC LIMIT 1",
                old_statuses,
            )
            selected = await cursor.fetchone()
            if selected is None:
                await connection.commit()
                return None

            cursor = await connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ? "
                "WHERE id = ? AND status = ?",
                (target_status, now, selected["id"], selected["status"]),
            )
            if cursor.rowcount != 1:
                await connection.rollback()
                return None
            await connection.commit()
            job_id = selected["id"]
        finally:
            await connection.close()

        return await self.get_job(job_id)

    async def transition_job(
        self,
        job_id: str,
        expected_statuses: Iterable[JobStatus | str],
        new_status: JobStatus | str,
        **fields: Any,
    ) -> Job | None:
        """Compare-and-set a job status, returning ``None`` after a lost race.

        Telegram approval/rejection callbacks use this instead of a blind
        update so repeated button presses can never reverse an earlier choice.
        """

        expected = tuple(JobStatus(status).value for status in expected_statuses)
        if not expected:
            raise ValueError("At least one expected status is required")
        unknown = set(fields) - (_UPDATABLE_FIELDS - {"status"})
        if unknown:
            raise ValueError(f"Fields cannot be updated: {', '.join(sorted(unknown))}")

        normalized: dict[str, Any] = {}
        for name, value in fields.items():
            normalized[name] = str(value) if isinstance(value, Path) else value
        normalized["status"] = JobStatus(new_status).value
        normalized["updated_at"] = datetime.now(UTC).isoformat()
        assignments = ", ".join(f"{name} = ?" for name in normalized)
        placeholders = ", ".join("?" for _ in expected)
        values = [*normalized.values(), job_id, *expected]

        connection = await self._connect()
        try:
            cursor = await connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ? "
                f"AND status IN ({placeholders})",
                values,
            )
            await connection.commit()
            changed = cursor.rowcount == 1
        finally:
            await connection.close()
        return await self.get_job(job_id) if changed else None

    async def recover_interrupted_jobs(self) -> int:
        """Make in-flight work claimable after a clean process restart.

        Instagram identifiers are deliberately preserved; the publisher can
        resume/poll an existing container instead of creating another Reel.
        """

        now = datetime.now(UTC).isoformat()
        connection = await self._connect()
        try:
            first = await connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ? "
                "WHERE status IN (?, ?)",
                (
                    JobStatus.QUEUED.value,
                    now,
                    JobStatus.DOWNLOADING.value,
                    JobStatus.PROCESSING.value,
                ),
            )
            second = await connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE status = ?",
                (
                    JobStatus.APPROVED.value,
                    now,
                    JobStatus.PUBLISHING.value,
                ),
            )
            await connection.commit()
            return max(first.rowcount, 0) + max(second.rowcount, 0)
        finally:
            await connection.close()

    async def list_jobs(
        self,
        *,
        chat_id: int | None = None,
        statuses: Iterable[JobStatus | str] | None = None,
        limit: int = 20,
    ) -> list[Job]:
        if limit < 1 or limit > 1_000:
            raise ValueError("limit must be between 1 and 1000")

        clauses: list[str] = []
        params: list[Any] = []
        if chat_id is not None:
            clauses.append("chat_id = ?")
            params.append(chat_id)
        if statuses is not None:
            values = tuple(JobStatus(status).value for status in statuses)
            if not values:
                return []
            clauses.append(f"status IN ({', '.join('?' for _ in values)})")
            params.extend(values)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        connection = await self._connect()
        try:
            cursor = await connection.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at DESC LIMIT ?",
                params,
            )
            rows = await cursor.fetchall()
        finally:
            await connection.close()
        return [self._row_to_job(row) for row in rows]

    @staticmethod
    def _row_to_job(row: aiosqlite.Row) -> Job:
        return Job(
            id=row["id"],
            chat_id=row["chat_id"],
            user_id=row["user_id"],
            source_type=SourceType(row["source_type"]),
            original_name=row["original_name"],
            source_path=row["source_path"],
            source_url=row["source_url"],
            status=JobStatus(row["status"]),
            output_path=row["output_path"],
            cover_path=row["cover_path"],
            subtitle_path=row["subtitle_path"],
            transcript_path=row["transcript_path"],
            language=row["language"],
            hook=row["hook"],
            caption=row["caption"],
            public_token=row["public_token"],
            instagram_container_id=row["instagram_container_id"],
            instagram_media_id=row["instagram_media_id"],
            instagram_permalink=row["instagram_permalink"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
