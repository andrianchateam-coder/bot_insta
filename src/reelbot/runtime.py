"""Application lifecycle for HTTP, Telegram polling, and the durable worker."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any

import uvicorn
from aiogram import Bot, Dispatcher

from .bot import create_dispatcher
from .config import Settings, get_settings
from .db import Database
from .instagram import InstagramPublisher
from .web import create_web_app

LOGGER = logging.getLogger(__name__)


def _secret_value(value: Any | None) -> str | None:
    if value is None:
        return None
    secret = value.get_secret_value()
    return secret if secret else None


def pipeline_configuration_issues(settings: Settings) -> tuple[str, ...]:
    """Return missing settings that prevent accepting Telegram jobs.

    The HTTP service intentionally remains usable without these values.  That
    makes installation and container health checks possible before credentials
    are supplied, without exposing a half-configured Telegram bot.
    """

    missing: list[str] = []
    if not _secret_value(settings.telegram_bot_token):
        missing.append("TELEGRAM_BOT_TOKEN")
    if not _secret_value(settings.openai_api_key):
        missing.append("REELBOT_OPENAI_API_KEY")
    if not settings.admin_ids:
        missing.append("TELEGRAM_ADMIN_IDS")
    return tuple(missing)


def instagram_configuration_issues(settings: Settings) -> tuple[str, ...]:
    """Return settings that prevent official Instagram publishing."""

    missing: list[str] = []
    if not _secret_value(settings.meta_access_token):
        missing.append("META_ACCESS_TOKEN")
    if not settings.instagram_user_id:
        missing.append("INSTAGRAM_USER_ID")
    if not settings.public_base_url:
        missing.append("PUBLIC_BASE_URL")
    elif not settings.public_base_url.startswith("https://"):
        missing.append("PUBLIC_BASE_URL_HTTPS")
    return tuple(missing)


@dataclass(slots=True)
class RuntimeComponents:
    settings: Settings
    database: Database
    app: Any
    bot: Bot | None = None
    dispatcher: Dispatcher | None = None
    worker: Any | None = None
    publisher: InstagramPublisher | None = None

    @property
    def pipeline_enabled(self) -> bool:
        return self.bot is not None and self.dispatcher is not None and self.worker is not None


async def create_runtime(settings: Settings) -> RuntimeComponents:
    """Initialize durable state and construct enabled runtime components."""

    settings.ensure_directories()
    database = Database(settings.database_path)
    await database.initialize()
    recovered = await database.recover_interrupted_jobs()
    if recovered:
        LOGGER.info("Recovered %d interrupted job(s)", recovered)

    app = create_web_app(database, settings.media_root)
    issues = pipeline_configuration_issues(settings)
    if issues:
        LOGGER.warning(
            "Telegram processing is disabled until these settings are supplied: %s",
            ", ".join(issues),
        )
        return RuntimeComponents(settings=settings, database=database, app=app)

    # Imports are delayed so health-only mode can start even while optional
    # processing integrations are being diagnosed.
    from .service import ReelService
    from .worker import ReelWorker

    telegram_token = _secret_value(settings.telegram_bot_token)
    assert telegram_token is not None
    bot = Bot(token=telegram_token)
    service = ReelService(
        database,
        bot,
        settings.media_root,
        admin_ids=settings.admin_ids,
        max_upload_bytes=settings.max_upload_mb * 1024 * 1024,
    )
    dispatcher = create_dispatcher(
        service,
        admin_ids=settings.admin_ids,
        max_upload_bytes=settings.max_upload_mb * 1024 * 1024,
    )

    publisher: InstagramPublisher | None = None
    instagram_issues = instagram_configuration_issues(settings)
    if instagram_issues:
        LOGGER.warning(
            "Instagram publishing is disabled until these settings are supplied: %s",
            ", ".join(instagram_issues),
        )
    else:
        meta_token = _secret_value(settings.meta_access_token)
        assert meta_token is not None and settings.instagram_user_id is not None
        publisher = InstagramPublisher(
            access_token=meta_token,
            instagram_user_id=settings.instagram_user_id,
            api_version=settings.meta_api_version,
            poll_interval_seconds=settings.instagram_poll_seconds,
            processing_timeout_seconds=settings.instagram_timeout_seconds,
        )

    worker = ReelWorker(
        settings=settings,
        database=database,
        bot=bot,
        instagram_publisher=publisher,
    )
    return RuntimeComponents(
        settings=settings,
        database=database,
        app=app,
        bot=bot,
        dispatcher=dispatcher,
        worker=worker,
        publisher=publisher,
    )


async def _close_components(components: RuntimeComponents) -> None:
    if components.publisher is not None:
        await components.publisher.aclose()
    ai = getattr(components.worker, "ai", None)
    ai_client = getattr(ai, "client", None)
    close_ai = getattr(ai_client, "close", None)
    if callable(close_ai):
        result = close_ai()
        if inspect.isawaitable(result):
            await result
    if components.bot is not None:
        await components.bot.session.close()


async def serve(settings: Settings | None = None) -> None:
    """Run one HTTP server and, when configured, one polling/worker pair."""

    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    components = await create_runtime(settings)
    config = uvicorn.Config(
        components.app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        # Media capability tokens live in URL paths while Meta downloads the
        # files. Standard access logs would persist those secrets.
        access_log=False,
    )
    server = uvicorn.Server(config)

    if not components.pipeline_enabled:
        try:
            await server.serve()
        finally:
            await _close_components(components)
        return

    assert components.bot is not None
    assert components.dispatcher is not None
    assert components.worker is not None
    stop_event = asyncio.Event()
    tasks = {
        "http": asyncio.create_task(server.serve(), name="reelbot-http"),
        "telegram": asyncio.create_task(
            components.dispatcher.start_polling(
                components.bot,
                handle_signals=False,
                close_bot_session=False,
            ),
            name="reelbot-telegram",
        ),
        "worker": asyncio.create_task(
            components.worker.run(stop_event), name="reelbot-worker"
        ),
    }

    failure: BaseException | None = None
    try:
        done, _pending = await asyncio.wait(
            tasks.values(), return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            try:
                exception = task.exception()
            except asyncio.CancelledError:
                exception = None
            if exception is not None:
                failure = exception
                break
        if failure is None and tasks["http"] not in done and not server.should_exit:
            failure = RuntimeError("A background ReelBot task stopped unexpectedly")
    finally:
        stop_event.set()
        server.should_exit = True
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        await _close_components(components)

    if failure is not None:
        raise failure


def main() -> None:
    asyncio.run(serve())


__all__ = [
    "RuntimeComponents",
    "create_runtime",
    "instagram_configuration_issues",
    "main",
    "pipeline_configuration_issues",
    "serve",
]
