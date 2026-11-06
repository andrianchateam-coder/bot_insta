import asyncio
import resource
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from reelbot.config import Settings
from reelbot.db import Database
from reelbot.models import JobStatus, SourceType
from reelbot.worker import ProcessedArtifacts, ReelWorker, _download_drive_file


class FakeBot:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.messages.append((chat_id, text))


@pytest.mark.asyncio
async def test_drive_downloader_enforces_os_quota_and_kills_timeout(
    monkeypatch, tmp_path: Path
) -> None:
    destination = tmp_path / "source.mp4"
    partial = tmp_path / "source.mp4.random.part"
    partial.write_bytes(b"partial")
    limit_calls = []

    class FakeProcess:
        pid = 12345
        returncode = None

        def __init__(self) -> None:
            self.finished = asyncio.Event()
            self.killed = False

        async def communicate(self):
            await self.finished.wait()
            return b"", b""

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9
            self.finished.set()

    process = FakeProcess()

    async def fake_subprocess(*args, **kwargs):
        assert "--no-check-certificate" not in args
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
    monkeypatch.setattr(
        resource,
        "prlimit",
        lambda pid, kind, limits: limit_calls.append((pid, kind, limits)),
    )

    with pytest.raises(TimeoutError, match="timed out"):
        await _download_drive_file(
            "1Abc_def-GHIjklMNop",
            destination,
            max_bytes=1024,
            timeout_seconds=0.01,
        )

    assert limit_calls == [(12345, resource.RLIMIT_FSIZE, (1024, 1024))]
    assert process.killed
    assert not partial.exists()


class FakePublisher:
    def __init__(self) -> None:
        self.calls = []

    async def create_reel_container(self, **kwargs):
        self.calls.append(kwargs)
        return "container-1"

    async def wait_until_finished(self, container_id: str) -> None:
        assert container_id == "container-1"

    async def publish_container(self, container_id: str) -> str:
        assert container_id == "container-1"
        return "media-1"

    async def fetch_permalink(self, media_id: str) -> str:
        assert media_id == "media-1"
        return "https://www.instagram.com/reel/example/"


def fake_processor(source: Path, directory: Path) -> ProcessedArtifacts:
    assert source.is_file()
    files = {
        "output_path": directory / "ready.mp4",
        "cover_path": directory / "cover.jpg",
        "subtitle_path": directory / "subtitles.srt",
        "transcript_path": directory / "transcript.txt",
    }
    for path in files.values():
        path.write_bytes(b"artifact")
    return ProcessedArtifacts(
        **files,
        language="kk",
        hook="Пайдалы кеңес",
        caption="Қазақша\n\nРусский\n\n#reels",
    )


@pytest.fixture
async def worker_parts(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        database_path=tmp_path / "jobs.sqlite3",
        media_root=tmp_path / "media",
        public_base_url="https://media.example.test",
        worker_poll_seconds=0.1,
    )
    settings.ensure_directories()
    database = Database(settings.database_path)
    await database.initialize()
    bot = FakeBot()
    publisher = FakePublisher()
    previews = []

    async def preview_sender(*args, **kwargs):
        previews.append((args, kwargs))

    worker = ReelWorker(
        settings=settings,
        database=database,
        bot=bot,
        instagram_publisher=publisher,
        processor=fake_processor,
        preview_sender=preview_sender,
    )
    return settings, database, bot, publisher, previews, worker


@pytest.mark.asyncio
async def test_process_preview_then_publish_only_after_approval(worker_parts) -> None:
    settings, database, bot, publisher, previews, worker = worker_parts
    directory = settings.media_root / "job-one"
    directory.mkdir()
    source = directory / "source.mp4"
    source.write_bytes(b"video")
    job = await database.create_job(
        job_id="job-one",
        chat_id=10,
        user_id=20,
        source_type=SourceType.TELEGRAM,
        original_name="source.mp4",
        source_path=source,
    )

    assert await worker.run_once()
    ready = await database.get_job(job.id)
    assert ready is not None and ready.status is JobStatus.READY
    assert ready.language == "kk"
    assert len(previews) == 1
    assert publisher.calls == []

    await database.transition_job(
        job.id, [JobStatus.READY], JobStatus.APPROVED
    )
    assert await worker.run_once()
    published = await database.get_job(job.id)
    assert published is not None and published.status is JobStatus.PUBLISHED
    assert published.instagram_media_id == "media-1"
    assert publisher.calls[0]["video_url"].startswith("https://media.example.test/media/")
    assert publisher.calls[0]["cover_url"].endswith("/cover")
    assert "опубликован" in bot.messages[-1][1].lower()


@pytest.mark.asyncio
async def test_drive_download_uses_only_extracted_id(worker_parts) -> None:
    settings, database, bot, publisher, previews, _worker = worker_parts
    downloaded = []

    def downloader(file_id: str, destination: Path) -> None:
        downloaded.append(file_id)
        destination.write_bytes(b"drive video")

    worker = ReelWorker(
        settings=settings,
        database=database,
        bot=bot,
        instagram_publisher=publisher,
        processor=fake_processor,
        drive_downloader=downloader,
        preview_sender=lambda *args, **kwargs: _async_none(),
    )
    job = await database.create_job(
        chat_id=10,
        user_id=20,
        source_type=SourceType.GOOGLE_DRIVE,
        original_name="drive.mp4",
        source_url="https://drive.google.com/uc?id=1Abc_def-GHIjklMNop",
    )
    assert await worker.run_once()
    ready = await database.get_job(job.id)
    assert ready is not None and ready.status is JobStatus.READY
    assert downloaded == ["1Abc_def-GHIjklMNop"]


@pytest.mark.asyncio
async def test_publish_resumes_persisted_container_without_duplicate(worker_parts) -> None:
    settings, database, _bot, publisher, _previews, worker = worker_parts
    directory = settings.media_root / "resume-job"
    directory.mkdir()
    source = directory / "source.mp4"
    source.write_bytes(b"video")
    job = await database.create_job(
        job_id="resume-job",
        chat_id=10,
        user_id=20,
        source_type=SourceType.TELEGRAM,
        original_name="source.mp4",
        source_path=source,
    )
    await database.update_job(
        job.id,
        status=JobStatus.APPROVED,
        caption="caption",
        instagram_container_id="container-1",
    )

    assert await worker.run_once()
    published = await database.get_job(job.id)
    assert published is not None and published.status is JobStatus.PUBLISHED
    assert published.instagram_media_id == "media-1"
    assert publisher.calls == []  # No second create_reel_container call.


async def _async_none() -> None:
    return None


@pytest.mark.asyncio
async def test_processing_failure_is_durable_and_redacts_secret(tmp_path: Path) -> None:
    secret = "never-log-this"
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        database_path=tmp_path / "jobs.sqlite3",
        media_root=tmp_path / "media",
        openai_api_key=secret,
    )
    settings.ensure_directories()
    database = Database(settings.database_path)
    await database.initialize()
    bot = FakeBot()

    def broken_processor(_source: Path, _directory: Path):
        raise RuntimeError(f"upstream rejected {secret}")

    worker = ReelWorker(
        settings=settings,
        database=database,
        bot=bot,
        processor=broken_processor,
    )
    directory = settings.media_root / "failed-job"
    directory.mkdir()
    source = directory / "source.mp4"
    source.write_bytes(b"video")
    job = await database.create_job(
        job_id="failed-job",
        chat_id=10,
        user_id=20,
        source_type=SourceType.TELEGRAM,
        original_name="source.mp4",
        source_path=source,
    )

    assert await worker.run_once()
    failed = await database.get_job(job.id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert failed.error is not None
    assert secret not in failed.error
    assert "[REDACTED]" in failed.error


@pytest.mark.asyncio
async def test_cleanup_removes_old_terminal_media_but_keeps_history(worker_parts) -> None:
    settings, database, _bot, _publisher, _previews, worker = worker_parts
    directory = settings.media_root / "old-published-job"
    directory.mkdir()
    source = directory / "source.mp4"
    output = directory / "ready.mp4"
    source.write_bytes(b"source")
    output.write_bytes(b"ready")
    job = await database.create_job(
        job_id="old-published-job",
        chat_id=10,
        user_id=20,
        source_type=SourceType.TELEGRAM,
        original_name="source.mp4",
        source_path=source,
    )
    await database.update_job(
        job.id,
        status=JobStatus.PUBLISHED,
        output_path=output,
        instagram_media_id="media-1",
    )
    old_timestamp = datetime.now(UTC) - timedelta(days=31)
    connection = await database._connect()
    try:
        await connection.execute(
            "UPDATE jobs SET updated_at = ? WHERE id = ?",
            (old_timestamp.isoformat(), job.id),
        )
        await connection.commit()
    finally:
        await connection.close()

    assert await worker.cleanup_expired_media(now=datetime.now(UTC)) == 1

    cleaned = await database.get_job(job.id)
    assert cleaned is not None
    assert cleaned.status is JobStatus.PUBLISHED
    assert cleaned.source_path is None
    assert cleaned.output_path is None
    assert not directory.exists()
