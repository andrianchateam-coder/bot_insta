from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from reelbot.web import create_web_app


@dataclass
class FakeJob:
    public_token: str
    output_path: str
    cover_path: str
    updated_at: datetime


class FakeDatabase:
    def __init__(self, job=None):
        self.job = job

    async def get_job(self, job_id: str):
        return self.job if job_id == "job-1" else None


def test_healthz(tmp_path):
    client = TestClient(create_web_app(FakeDatabase(), tmp_path))
    assert client.get("/healthz").json() == {"status": "ok"}


def test_media_requires_matching_token(tmp_path):
    video = tmp_path / "video.mp4"
    cover = tmp_path / "cover.jpg"
    video.write_bytes(b"video")
    cover.write_bytes(b"cover")
    job = FakeJob("right-token", str(video), str(cover), datetime.now(UTC))
    client = TestClient(create_web_app(FakeDatabase(job), tmp_path))

    assert client.get("/media/job-1/wrong/video").status_code == 404
    response = client.get("/media/job-1/right-token/video")
    assert response.status_code == 200
    assert response.content == b"video"


def test_media_rejects_path_outside_root(tmp_path):
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"private")
    job = FakeJob("token", str(outside), str(outside), datetime.now(UTC))
    client = TestClient(create_web_app(FakeDatabase(job), tmp_path))

    assert client.get("/media/job-1/token/video").status_code == 404


def test_media_link_expires_after_ttl(tmp_path):
    video = tmp_path / "video.mp4"
    cover = tmp_path / "cover.jpg"
    video.write_bytes(b"video")
    cover.write_bytes(b"cover")
    job = FakeJob(
        "token",
        str(video),
        str(cover),
        datetime.now(UTC) - timedelta(hours=73),
    )
    client = TestClient(
        create_web_app(FakeDatabase(job), tmp_path, public_media_ttl_hours=72)
    )

    assert client.get("/media/job-1/token/video").status_code == 404


def test_media_link_handles_naive_database_timestamp_as_utc(tmp_path):
    video = tmp_path / "video.mp4"
    cover = tmp_path / "cover.jpg"
    video.write_bytes(b"video")
    cover.write_bytes(b"cover")
    job = FakeJob("token", str(video), str(cover), datetime.now(UTC).replace(tzinfo=None))
    client = TestClient(create_web_app(FakeDatabase(job), tmp_path))

    assert client.get("/media/job-1/token/video").status_code == 200
