import asyncio
from pathlib import Path

import pytest

from reelbot.db import Database
from reelbot.models import JobStatus, SourceType


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "jobs.sqlite3")
    await db.initialize()
    return db


@pytest.mark.asyncio
async def test_create_update_and_list_job(database: Database, tmp_path: Path):
    source = tmp_path / "input.mp4"
    job = await database.create_job(
        chat_id=101,
        user_id=202,
        source_type=SourceType.TELEGRAM,
        original_name="input.mp4",
        source_path=source,
    )

    assert job.status is JobStatus.QUEUED
    assert job.source_path == str(source)
    assert len(job.public_token) >= 32

    updated = await database.update_job(
        job.id,
        status=JobStatus.READY,
        language="kk",
        hook="Күшті бастау",
        caption="Қазақша\n\nРусский",
    )
    assert updated.status is JobStatus.READY
    assert updated.language == "kk"

    jobs = await database.list_jobs(chat_id=101)
    assert [item.id for item in jobs] == [job.id]


@pytest.mark.asyncio
async def test_atomic_claim_allows_only_one_worker(database: Database):
    job = await database.create_job(
        chat_id=1,
        user_id=2,
        source_type=SourceType.GOOGLE_DRIVE,
        original_name="drive-video.mp4",
        source_url="https://drive.google.com/file/d/example/view",
    )

    first, second = await asyncio.gather(
        database.claim_next([JobStatus.QUEUED], JobStatus.PROCESSING),
        database.claim_next([JobStatus.QUEUED], JobStatus.PROCESSING),
    )
    claimed = [item for item in (first, second) if item is not None]
    assert len(claimed) == 1
    assert claimed[0].id == job.id
    assert claimed[0].status is JobStatus.PROCESSING


@pytest.mark.asyncio
async def test_source_validation_and_unknown_update(database: Database):
    with pytest.raises(ValueError, match="source_path"):
        await database.create_job(
            chat_id=1,
            user_id=2,
            source_type=SourceType.TELEGRAM,
            original_name="missing.mp4",
            source_url="https://example.com/video.mp4",
        )

    job = await database.create_job(
        chat_id=1,
        user_id=2,
        source_type=SourceType.TELEGRAM,
        original_name="ok.mp4",
        source_path="/tmp/ok.mp4",
    )
    with pytest.raises(ValueError, match="cannot be updated"):
        await database.update_job(job.id, id="replacement")


@pytest.mark.asyncio
async def test_compare_and_set_transition_and_recovery(database: Database):
    job = await database.create_job(
        chat_id=1,
        user_id=2,
        source_type=SourceType.TELEGRAM,
        original_name="ok.mp4",
        source_path="/tmp/ok.mp4",
    )
    ready = await database.transition_job(
        job.id, [JobStatus.QUEUED], JobStatus.PROCESSING
    )
    assert ready is not None
    assert ready.status is JobStatus.PROCESSING
    assert (
        await database.transition_job(
            job.id, [JobStatus.QUEUED], JobStatus.REJECTED
        )
        is None
    )

    assert await database.recover_interrupted_jobs() == 1
    recovered = await database.get_job(job.id)
    assert recovered is not None
    assert recovered.status is JobStatus.QUEUED
