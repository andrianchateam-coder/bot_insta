from enum import Enum

import pytest

from reelbot.bot import (
    build_public_media_url,
    extract_google_drive_file_id,
    format_job_status,
    is_allowed_user,
    may_manage_job,
    parse_review_callback,
    review_keyboard,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://drive.google.com/file/d/1Abc_def-GHIjklMNop/view?usp=sharing",
            "1Abc_def-GHIjklMNop",
        ),
        (
            "Вот ссылка: https://drive.google.com/open?id=1Abc_def-GHIjklMNop.",
            "1Abc_def-GHIjklMNop",
        ),
        (
            "https://drive.usercontent.google.com/download?id=1Abc_def-GHIjklMNop&export=download",
            "1Abc_def-GHIjklMNop",
        ),
    ],
)
def test_extract_google_drive_file_id(url: str, expected: str) -> None:
    assert extract_google_drive_file_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://drive.google.com/file/d/1Abc_def-GHIjklMNop/view",
        "https://drive.google.com.evil.test/file/d/1Abc_def-GHIjklMNop/view",
        "https://evil.test/?id=1Abc_def-GHIjklMNop",
        "https://user@drive.google.com/file/d/1Abc_def-GHIjklMNop/view",
        "https://drive.google.com/file/d/short/view",
        "not a link",
    ],
)
def test_drive_parser_rejects_unsafe_or_invalid_urls(url: str) -> None:
    assert extract_google_drive_file_id(url) is None


def test_allowlist_is_closed_by_default() -> None:
    assert not is_allowed_user(1, frozenset())
    assert is_allowed_user(42, frozenset({42}))
    assert not is_allowed_user(None, frozenset({42}))


def test_owner_or_admin_can_manage_job() -> None:
    assert may_manage_job(7, 7, frozenset())
    assert may_manage_job(9, 7, frozenset({9}))
    assert not may_manage_job(8, 7, frozenset({9}))


def test_callback_round_trip() -> None:
    markup = review_keyboard("abc_123")
    approve = markup.inline_keyboard[0][0].callback_data
    reject = markup.inline_keyboard[0][1].callback_data
    assert parse_review_callback(approve) == ("approve", "abc_123")
    assert parse_review_callback(reject) == ("reject", "abc_123")
    assert parse_review_callback("review:approve:../unsafe") is None
    with pytest.raises(ValueError):
        review_keyboard("x" * 41)


def test_public_media_url() -> None:
    assert build_public_media_url(
        "https://bot.example.test/",
        job_id="abc123",
        public_token="secret-token",
    ) == "https://bot.example.test/media/abc123/secret-token/video"
    assert (
        build_public_media_url(
            "https://bot.example.test",
            job_id="../bad",
            public_token="token",
        )
        is None
    )


class Status(str, Enum):
    PUBLISHED = "published"


def test_format_job_status_accepts_mapping_and_enum() -> None:
    text = format_job_status(
        {
            "id": "job-1",
            "status": Status.PUBLISHED,
            "instagram_permalink": "https://instagram.com/reel/example",
        }
    )
    assert "опубликован" in text
    assert "https://instagram.com/reel/example" in text


@pytest.mark.parametrize(
    ("status", "command"),
    [("ready", "/preview job-1"), ("failed", "/retry job-1")],
)
def test_format_job_status_suggests_recovery_command(status: str, command: str) -> None:
    text = format_job_status({"id": "job-1", "status": status})
    assert command in text
