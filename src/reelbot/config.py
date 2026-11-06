"""Application configuration loaded from environment variables.

Secrets intentionally use :class:`pydantic.SecretStr` so an accidental model
dump or log statement cannot reveal their values.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the bot, worker, and public media endpoint."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    telegram_bot_token: SecretStr | None = None
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("REELBOT_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    meta_access_token: SecretStr | None = None

    # A comma/space-separated string is friendlier in environment settings than
    # JSON, while ``admin_ids`` below exposes the parsed immutable set.
    telegram_admin_ids: str = ""
    instagram_user_id: str | None = None
    public_base_url: str | None = None

    data_dir: Path = Path("data")
    database_path: Path = Path("data/reelbot.sqlite3")
    media_root: Path = Path("data/media")

    ffmpeg_path: str = Field(
        default="ffmpeg",
        validation_alias=AliasChoices("FFMPEG_BINARY", "FFMPEG_PATH"),
    )
    ffprobe_path: str = Field(
        default="ffprobe",
        validation_alias=AliasChoices("FFPROBE_BINARY", "FFPROBE_PATH"),
    )
    font_path: Path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    )

    openai_transcription_model: str = "whisper-1"
    openai_chat_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("OPENAI_TEXT_MODEL", "OPENAI_CHAT_MODEL"),
    )
    meta_api_version: str = "v22.0"

    max_upload_mb: int = Field(
        default=20,
        ge=1,
        le=2_000,
        validation_alias=AliasChoices("MAX_TELEGRAM_UPLOAD_MB", "MAX_UPLOAD_MB"),
    )
    max_preview_mb: int = Field(default=48, ge=1, le=2_000)
    video_crf: int = Field(default=21, ge=0, le=51)
    video_preset: str = "medium"
    worker_poll_seconds: float = Field(default=1.0, ge=0.1, le=60)
    instagram_poll_seconds: float = Field(default=5.0, ge=0.1, le=60)
    instagram_timeout_seconds: float = Field(default=300.0, ge=10, le=3_600)

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65_535)
    log_level: str = "INFO"

    @field_validator(
        "instagram_user_id",
        "public_base_url",
        "telegram_admin_ids",
        mode="before",
    )
    @classmethod
    def _empty_strings_are_unset_or_normalized(cls, value: object, info):
        if value is None:
            return "" if info.field_name == "telegram_admin_ids" else None
        if isinstance(value, str):
            value = value.strip()
            if info.field_name == "telegram_admin_ids":
                return value
            return value or None
        return value

    @field_validator("public_base_url")
    @classmethod
    def _validate_public_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.startswith(("https://", "http://")):
            raise ValueError("PUBLIC_BASE_URL must start with http:// or https://")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _validate_admin_ids(self) -> Settings:
        # Force eager validation so a typo does not silently lock the owner out.
        self._parse_admin_ids(self.telegram_admin_ids)
        return self

    @staticmethod
    def _parse_admin_ids(raw: str) -> frozenset[int]:
        if not raw:
            return frozenset()
        normalized = raw.replace(";", ",").replace(" ", ",")
        tokens = [token for token in normalized.split(",") if token]
        try:
            ids = frozenset(int(token) for token in tokens)
        except ValueError as exc:
            raise ValueError(
                "TELEGRAM_ADMIN_IDS must contain integer IDs separated by commas"
            ) from exc
        if any(value <= 0 for value in ids):
            raise ValueError("TELEGRAM_ADMIN_IDS must contain positive integers")
        return ids

    @property
    def admin_ids(self) -> frozenset[int]:
        """Telegram user IDs allowed to use the bot."""

        return self._parse_admin_ids(self.telegram_admin_ids)

    @property
    def instagram_configured(self) -> bool:
        return bool(
            self.meta_access_token
            and self.instagram_user_id
            and self.public_base_url
        )

    def ensure_directories(self) -> None:
        """Create only the local state directories required at runtime."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.media_root.mkdir(parents=True, exist_ok=True)


def get_settings(**overrides: object) -> Settings:
    """Small factory that is convenient to override in tests."""

    return Settings(**overrides)
