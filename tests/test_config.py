from pathlib import Path

import pytest
from pydantic import ValidationError

from reelbot.config import Settings


def test_settings_parse_admin_ids_and_normalize_url(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TELEGRAM_ADMIN_IDS", "123, 456;789")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://bot.example.com/")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "top-secret")
    monkeypatch.setenv("REELBOT_OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("OPENAI_TEXT_MODEL", "test-model")
    monkeypatch.setenv("MAX_TELEGRAM_UPLOAD_MB", "42")
    monkeypatch.setenv("MAX_DRIVE_DOWNLOAD_MB", "750")
    monkeypatch.setenv("DRIVE_DOWNLOAD_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("MEDIA_COMMAND_TIMEOUT_SECONDS", "700")
    monkeypatch.setenv("PUBLIC_MEDIA_TTL_HOURS", "48")
    monkeypatch.setenv("MEDIA_RETENTION_DAYS", "45")
    monkeypatch.setenv("FFMPEG_BINARY", "/opt/ffmpeg")

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        database_path=tmp_path / "db.sqlite3",
        media_root=tmp_path / "media",
    )

    assert settings.admin_ids == frozenset({123, 456, 789})
    assert settings.public_base_url == "https://bot.example.com"
    assert settings.telegram_bot_token is not None
    assert "top-secret" not in repr(settings.telegram_bot_token)
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "openai-secret"
    assert settings.openai_chat_model == "test-model"
    assert settings.max_upload_mb == 42
    assert settings.max_drive_download_mb == 750
    assert settings.drive_download_timeout_seconds == 600
    assert settings.media_command_timeout_seconds == 700
    assert settings.public_media_ttl_hours == 48
    assert settings.media_retention_days == 45
    assert settings.ffmpeg_path == "/opt/ffmpeg"


def test_settings_reject_bad_admin_id(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_IDS", "123,not-an-id")
    with pytest.raises(ValidationError, match="integer IDs"):
        Settings(_env_file=None)


def test_ensure_directories(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "state",
        database_path=tmp_path / "db" / "jobs.sqlite3",
        media_root=tmp_path / "media",
    )
    settings.ensure_directories()
    assert settings.data_dir.is_dir()
    assert settings.database_path.parent.is_dir()
    assert settings.media_root.is_dir()
