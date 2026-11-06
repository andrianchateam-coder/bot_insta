from pathlib import Path

import pytest

from reelbot.db import Database
from reelbot.models import JobStatus
from reelbot.service import JobStateError, PublishingNotConfigured, ReelService


class FakeBot:
    async def download(self, file_id: str, *, destination: Path):
        assert file_id == "telegram-file"
        destination.write_bytes(b"video bytes")


@pytest.fixture
async def service(tmp_path: Path):
    database = Database(tmp_path / "jobs.sqlite3")
    await database.initialize()
    return ReelService(database, FakeBot(), tmp_path / "media", admin_ids={99})


@pytest.mark.asyncio
async def test_enqueue_telegram_and_review_once(service: ReelService) -> None:
    job = await service.enqueue_telegram_file(
        chat_id=10,
        user_id=20,
        file_id="telegram-file",
        file_unique_id="unique",
        file_name="../../unsafe.MP4",
        file_size=100,
        mime_type="video/mp4",
    )

    assert job.source_path is not None
    source = Path(job.source_path)
    assert source.name == "source.mp4"
    assert source.read_bytes() == b"video bytes"
    assert service.media_root in source.resolve().parents

    await service.database.update_job(job.id, status=JobStatus.READY)
    approved = await service.approve_job(job_id=job.id, actor_id=20)
    assert approved.status is JobStatus.APPROVED
    with pytest.raises(JobStateError):
        await service.approve_job(job_id=job.id, actor_id=20)


@pytest.mark.asyncio
async def test_admin_can_reject_but_stranger_cannot(service: ReelService) -> None:
    job = await service.enqueue_drive_file(
        chat_id=10,
        user_id=20,
        drive_url="ignored chat text",
        drive_file_id="1Abc_def-GHIjklMNop",
    )
    await service.database.update_job(job.id, status=JobStatus.READY)
    with pytest.raises(PermissionError):
        await service.reject_job(job_id=job.id, actor_id=21)
    rejected = await service.reject_job(job_id=job.id, actor_id=99)
    assert rejected.status is JobStatus.REJECTED


@pytest.mark.asyncio
async def test_drive_id_is_validated(service: ReelService) -> None:
    with pytest.raises(ValueError, match="Drive"):
        await service.enqueue_drive_file(
            chat_id=10,
            user_id=20,
            drive_url="https://evil.test/",
            drive_file_id="../../bad",
        )


@pytest.mark.asyncio
async def test_service_enforces_upload_limit(tmp_path: Path) -> None:
    database = Database(tmp_path / "jobs.sqlite3")
    await database.initialize()
    service = ReelService(
        database,
        FakeBot(),
        tmp_path / "media",
        max_upload_bytes=5,
    )
    with pytest.raises(ValueError, match="upload limit"):
        await service.enqueue_telegram_file(
            chat_id=10,
            user_id=20,
            file_id="telegram-file",
            file_unique_id="unique",
            file_name="video.mp4",
            file_size=100,
            mime_type="video/mp4",
        )


@pytest.mark.asyncio
async def test_retry_processing_failure_without_reupload(service: ReelService) -> None:
    job = await service.enqueue_drive_file(
        chat_id=10,
        user_id=20,
        drive_url="ignored",
        drive_file_id="1Abc_def-GHIjklMNop",
    )
    await service.database.update_job(job.id, status=JobStatus.FAILED, error="temporary")

    retried = await service.retry_job(job_id=job.id, actor_id=20)

    assert retried.status is JobStatus.QUEUED
    assert retried.error is None
    with pytest.raises(JobStateError):
        await service.retry_job(job_id=job.id, actor_id=20)


@pytest.mark.asyncio
async def test_retry_publishing_resumes_processed_artifacts(service: ReelService) -> None:
    job = await service.enqueue_drive_file(
        chat_id=10,
        user_id=20,
        drive_url="ignored",
        drive_file_id="1Abc_def-GHIjklMNop",
    )
    artifact_directory = service.media_root / job.id
    artifact_directory.mkdir(parents=True)
    output = artifact_directory / "ready.mp4"
    cover = artifact_directory / "cover.jpg"
    output.write_bytes(b"video")
    cover.write_bytes(b"cover")
    await service.database.update_job(
        job.id,
        status=JobStatus.FAILED,
        output_path=output,
        cover_path=cover,
        caption="caption",
        error="temporary Meta outage",
    )

    retried = await service.retry_job(job_id=job.id, actor_id=99)
    assert retried.status is JobStatus.APPROVED


@pytest.mark.asyncio
async def test_publishing_actions_are_gated_until_configured(tmp_path: Path) -> None:
    database = Database(tmp_path / "jobs.sqlite3")
    await database.initialize()
    service = ReelService(
        database,
        FakeBot(),
        tmp_path / "media",
        publishing_enabled=False,
    )
    job = await service.enqueue_drive_file(
        chat_id=10,
        user_id=20,
        drive_url="ignored",
        drive_file_id="1Abc_def-GHIjklMNop",
    )
    await database.update_job(job.id, status=JobStatus.READY)
    with pytest.raises(PublishingNotConfigured):
        await service.approve_job(job_id=job.id, actor_id=20)

    artifact_directory = service.media_root / job.id
    artifact_directory.mkdir(parents=True)
    output = artifact_directory / "ready.mp4"
    cover = artifact_directory / "cover.jpg"
    output.write_bytes(b"video")
    cover.write_bytes(b"cover")
    await database.update_job(
        job.id,
        status=JobStatus.FAILED,
        output_path=output,
        cover_path=cover,
        caption="caption",
    )
    with pytest.raises(PublishingNotConfigured):
        await service.retry_job(job_id=job.id, actor_id=20)
