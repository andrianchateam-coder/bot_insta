from pathlib import Path

from reelbot.config import Settings
from reelbot.runtime import (
    create_runtime,
    instagram_configuration_issues,
    pipeline_configuration_issues,
)


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "jobs.sqlite3",
        media_root=tmp_path / "data" / "media",
        **overrides,
    )


def test_pipeline_configuration_is_closed_by_default(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    assert pipeline_configuration_issues(settings) == (
        "TELEGRAM_BOT_TOKEN",
        "REELBOT_OPENAI_API_KEY",
        "TELEGRAM_ADMIN_IDS",
    )


def test_instagram_requires_public_https_origin(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        meta_access_token="secret",
        instagram_user_id="123",
        public_base_url="http://localhost:8000",
    )

    assert instagram_configuration_issues(settings) == ("PUBLIC_BASE_URL_HTTPS",)


async def test_health_only_runtime_initializes_database(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    runtime = await create_runtime(settings)

    assert not runtime.pipeline_enabled
    assert settings.database_path.is_file()
    assert settings.media_root.is_dir()


async def test_configured_runtime_builds_pipeline_without_network(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        telegram_bot_token="123456:" + "A" * 35,
        openai_api_key="sk-test-only",
        telegram_admin_ids="42",
    )

    runtime = await create_runtime(settings)
    try:
        assert runtime.pipeline_enabled
        assert runtime.publisher is None
    finally:
        assert runtime.bot is not None
        await runtime.bot.session.close()
        client = runtime.worker.ai.client
        client.close()
