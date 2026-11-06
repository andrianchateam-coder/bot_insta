"""Telegram user interface for the Reel bot.

The module deliberately depends on a small, duck-typed service protocol.  The
service owns persistence, downloads and background processing; this module only
validates user input and presents state to Telegram users.  Keeping those
responsibilities separate makes updates and restarts much less likely to lose a
job.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, quote, urlsplit

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

LOGGER = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
_DRIVE_HOSTS = {
    "drive.google.com",
    "docs.google.com",
    "drive.usercontent.google.com",
}
_DRIVE_ID_RE = re.compile(r"[A-Za-z0-9_-]{10,200}\Z")
_URL_RE = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
_CALLBACK_RE = re.compile(r"review:(approve|reject):([A-Za-z0-9_-]{1,40})\Z")


class TelegramJobService(Protocol):
    """Boundary implemented by the application/workflow layer.

    Returned jobs may be dataclasses, Pydantic models or dictionaries.  The UI
    accesses their fields through :func:`job_value`.
    """

    async def enqueue_telegram_file(
        self,
        *,
        chat_id: int,
        user_id: int,
        file_id: str,
        file_unique_id: str,
        file_name: str,
        file_size: int | None,
        mime_type: str | None,
    ) -> Any: ...

    async def enqueue_drive_file(
        self,
        *,
        chat_id: int,
        user_id: int,
        drive_url: str,
        drive_file_id: str,
    ) -> Any: ...

    async def recent_jobs(self, *, chat_id: int, limit: int = 5) -> Sequence[Any]: ...

    async def get_job(self, job_id: str) -> Any | None: ...

    async def approve_job(self, *, job_id: str, actor_id: int) -> Any: ...

    async def reject_job(self, *, job_id: str, actor_id: int) -> Any: ...


def is_allowed_user(user_id: int | None, admin_ids: frozenset[int] | set[int]) -> bool:
    """Return whether a Telegram user may submit work.

    An empty allow-list denies access.  Accidentally exposing an expensive media
    processing bot to everyone is a much worse default than requiring one ID.
    """

    return user_id is not None and user_id in admin_ids


def may_manage_job(
    actor_id: int | None,
    owner_id: int | None,
    admin_ids: frozenset[int] | set[int],
) -> bool:
    """Allow a job owner or an administrator to act on a review button."""

    if actor_id is None:
        return False
    return actor_id == owner_id or actor_id in admin_ids


def extract_google_drive_file_id(text: str | None) -> str | None:
    """Extract a Drive file ID from a supported HTTPS sharing URL.

    Only well-known Google Drive hosts are accepted.  This keeps a later
    downloader from becoming a generic server-side request forgery primitive.
    """

    if not text:
        return None

    match = _URL_RE.search(text)
    if not match:
        return None
    candidate = match.group(0).rstrip(".,);]}")

    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError:
        return None

    if (
        parsed.scheme.lower() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or (parsed.hostname or "").lower() not in _DRIVE_HOSTS
    ):
        return None

    parts = [part for part in parsed.path.split("/") if part]
    file_id: str | None = None
    if len(parts) >= 3 and parts[0] == "file" and parts[1] == "d":
        file_id = parts[2]
    else:
        values = parse_qs(parsed.query).get("id", [])
        if values:
            file_id = values[0]

    if file_id and _DRIVE_ID_RE.fullmatch(file_id):
        return file_id
    return None


def parse_review_callback(data: str | None) -> tuple[str, str] | None:
    """Parse and validate compact callback data."""

    if not data:
        return None
    match = _CALLBACK_RE.fullmatch(data)
    if not match:
        return None
    return match.group(1), match.group(2)


def review_keyboard(job_id: str) -> InlineKeyboardMarkup:
    """Build approval controls while respecting Telegram's 64-byte limit."""

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", job_id):
        raise ValueError("job_id is not safe for Telegram callback data")
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Одобрить и опубликовать",
                    callback_data=f"review:approve:{job_id}",
                ),
                InlineKeyboardButton(
                    text="❌ Отклонить",
                    callback_data=f"review:reject:{job_id}",
                ),
            ]
        ]
    )


def job_value(job: Any, name: str, default: Any = None) -> Any:
    """Read a job field from a mapping or an object."""

    if isinstance(job, dict):
        return job.get(name, default)
    return getattr(job, name, default)


def _status_value(status: Any) -> str:
    value = getattr(status, "value", status)
    return str(value or "unknown").lower()


def format_job_status(job: Any) -> str:
    """Format one persisted job for the ``/status`` command."""

    labels = {
        "queued": "в очереди",
        "downloading": "скачивается",
        "processing": "обрабатывается",
        "ready": "готов к проверке",
        "approved": "одобрен",
        "publishing": "публикуется",
        "published": "опубликован",
        "rejected": "отклонён",
        "failed": "ошибка",
    }
    job_id = str(job_value(job, "id", "?"))
    status = _status_value(job_value(job, "status"))
    label = labels.get(status, status)
    suffix = ""
    if status == "published" and job_value(job, "instagram_permalink"):
        suffix = f" — {job_value(job, 'instagram_permalink')}"
    return f"• {job_id}: {label}{suffix}"


def build_public_media_url(
    public_base_url: str | None,
    *,
    job_id: str,
    public_token: str,
    kind: str = "video",
) -> str | None:
    """Build an opaque public media URL consumed by Telegram or Instagram."""

    if not public_base_url or not public_token or kind not in {"video", "cover"}:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_id):
        return None
    base = public_base_url.rstrip("/")
    return (
        f"{base}/media/{quote(job_id, safe='')}/"
        f"{quote(public_token, safe='')}/{kind}"
    )


def _job_caption(job: Any) -> str:
    explicit = job_value(job, "caption")
    if explicit:
        return str(explicit).strip()

    parts: list[str] = []
    for field in ("caption_kk", "caption_ru"):
        value = job_value(job, field)
        if value:
            parts.append(str(value).strip())
    hashtags = job_value(job, "hashtags")
    if isinstance(hashtags, list | tuple | set):
        hashtags = " ".join(str(tag) for tag in hashtags)
    if hashtags:
        parts.append(str(hashtags).strip())
    return "\n\n".join(part for part in parts if part)


def _text_chunks(text: str, size: int = 4000) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)] or [""]


async def send_preview(
    bot: Bot,
    job: Any,
    *,
    max_telegram_bytes: int,
    public_base_url: str | None = None,
) -> None:
    """Send cover, processed video/caption and approval controls.

    Large files are linked through the authenticated public media endpoint.  A
    Telegram upload error also falls back to that URL when one is available.
    """

    chat_id = int(job_value(job, "chat_id"))
    job_id = str(job_value(job, "id"))
    public_token = str(job_value(job, "public_token", ""))
    video_url = build_public_media_url(
        public_base_url,
        job_id=job_id,
        public_token=public_token,
        kind="video",
    )
    keyboard = review_keyboard(job_id)
    hook = str(
        job_value(
            job,
            "cover_hook",
            job_value(job, "hook", job_value(job, "title", "Готово")),
        )
    )
    caption = _job_caption(job)

    cover_path_value = job_value(job, "cover_path")
    if cover_path_value:
        cover_path = Path(str(cover_path_value))
        if cover_path.is_file():
            await bot.send_photo(
                chat_id,
                FSInputFile(cover_path),
                caption=f"Обложка: {hook}"[:1024],
            )

    output_path_value = job_value(job, "output_path")
    output_path = Path(str(output_path_value)) if output_path_value else None
    sent_video = False
    if output_path and output_path.is_file() and output_path.stat().st_size <= max_telegram_bytes:
        try:
            short_caption = caption if len(caption) <= 900 else None
            await bot.send_video(
                chat_id,
                FSInputFile(output_path),
                caption=short_caption,
                reply_markup=keyboard if short_caption is not None else None,
                supports_streaming=True,
            )
            sent_video = True
            if short_caption is not None:
                return
        except TelegramAPIError:
            LOGGER.warning("Telegram could not upload preview for job %s", job_id)
            if not video_url:
                raise

    intro = "Монтаж готов. Проверьте ролик перед публикацией."
    if not sent_video:
        if video_url:
            intro += f"\n\nВидео: {video_url}"
        else:
            intro += "\n\nВидео слишком велико для Telegram; публичная ссылка пока не настроена."
    await bot.send_message(chat_id, intro)

    chunks = _text_chunks(caption)
    for index, chunk in enumerate(chunks):
        await bot.send_message(
            chat_id,
            chunk or "Описание не сгенерировано.",
            reply_markup=keyboard if index == len(chunks) - 1 else None,
        )


def _telegram_file(message: Message) -> tuple[Any, str, str | None] | None:
    if message.video:
        name = message.video.file_name or f"{message.video.file_unique_id}.mp4"
        return message.video, name, message.video.mime_type
    if message.document:
        name = message.document.file_name or message.document.file_unique_id
        suffix = Path(name).suffix.lower()
        mime = message.document.mime_type
        if (mime and mime.startswith("video/")) or suffix in _VIDEO_EXTENSIONS:
            return message.document, name, mime
    return None


def build_router(
    service: TelegramJobService,
    *,
    admin_ids: frozenset[int] | set[int],
    max_upload_bytes: int,
) -> Router:
    """Create an isolated Aiogram router; no global bot or token is retained."""

    allowed = frozenset(admin_ids)
    router = Router(name="reelbot")

    async def guard_message(message: Message) -> bool:
        user_id = message.from_user.id if message.from_user else None
        if is_allowed_user(user_id, allowed):
            return True
        await message.answer("У вас нет доступа к этому боту.")
        return False

    @router.message(Command("start"))
    async def start(message: Message) -> None:
        if not await guard_message(message):
            return
        await message.answer(
            "Отправьте видеофайл или HTTPS-ссылку Google Drive. "
            "Я добавлю субтитры и обложку, затем пришлю результат на проверку."
        )

    @router.message(Command("help"))
    async def help_command(message: Message) -> None:
        if not await guard_message(message):
            return
        await message.answer(
            "1. Пришлите видео или ссылку Google Drive.\n"
            "2. Дождитесь предпросмотра.\n"
            "3. Нажмите «Одобрить и опубликовать» либо «Отклонить».\n\n"
            "Команда /status показывает последние задания."
        )

    @router.message(Command("status"))
    async def status_command(message: Message) -> None:
        if not await guard_message(message):
            return
        try:
            jobs = await service.recent_jobs(chat_id=message.chat.id, limit=5)
        except Exception:
            LOGGER.exception("Could not load recent jobs")
            await message.answer("Не удалось получить статус. Попробуйте немного позже.")
            return
        if not jobs:
            await message.answer("Заданий пока нет.")
            return
        await message.answer("Последние задания:\n" + "\n".join(map(format_job_status, jobs)))

    @router.message(F.video | F.document)
    async def media_message(message: Message) -> None:
        if not await guard_message(message):
            return
        media = _telegram_file(message)
        if not media:
            await message.answer("Поддерживаются видео MP4, MOV, M4V, MKV и WEBM.")
            return
        file, file_name, mime_type = media
        if file.file_size and file.file_size > max_upload_bytes:
            await message.answer(
                "Этот файл слишком велик для обычной загрузки Telegram. "
                "Загрузите его в Google Drive, включите доступ по ссылке и пришлите ссылку сюда."
            )
            return
        try:
            job = await service.enqueue_telegram_file(
                chat_id=message.chat.id,
                user_id=message.from_user.id,
                file_id=file.file_id,
                file_unique_id=file.file_unique_id,
                file_name=file_name,
                file_size=file.file_size,
                mime_type=mime_type,
            )
        except Exception:
            LOGGER.exception("Could not enqueue Telegram video")
            await message.answer("Не удалось принять видео. Попробуйте ещё раз.")
            return
        await message.answer(f"Видео принято. Номер задания: {job_value(job, 'id', '—')}.")

    @router.message(F.text)
    async def drive_message(message: Message) -> None:
        if not await guard_message(message):
            return
        file_id = extract_google_drive_file_id(message.text)
        if not file_id:
            await message.answer(
                "Пришлите видеофайл или HTTPS-ссылку на файл Google Drive "
                "с включённым доступом по ссылке."
            )
            return
        try:
            job = await service.enqueue_drive_file(
                chat_id=message.chat.id,
                user_id=message.from_user.id,
                drive_url=message.text.strip(),
                drive_file_id=file_id,
            )
        except Exception:
            LOGGER.exception("Could not enqueue Google Drive video")
            await message.answer(
                "Не удалось принять ссылку. Проверьте доступ "
                "«всем, у кого есть ссылка» и повторите."
            )
            return
        await message.answer(f"Ссылка принята. Номер задания: {job_value(job, 'id', '—')}.")

    @router.callback_query(F.data.startswith("review:"))
    async def review_callback(callback: CallbackQuery) -> None:
        parsed = parse_review_callback(callback.data)
        actor_id = callback.from_user.id
        if not parsed:
            await callback.answer("Кнопка устарела.", show_alert=True)
            return
        action, job_id = parsed
        try:
            job = await service.get_job(job_id)
        except Exception:
            LOGGER.exception("Could not load job %s", job_id)
            await callback.answer("Не удалось проверить задание.", show_alert=True)
            return
        if not job:
            await callback.answer("Задание не найдено.", show_alert=True)
            return
        owner_id = job_value(job, "user_id", job_value(job, "owner_id"))
        if not may_manage_job(actor_id, owner_id, allowed):
            await callback.answer("Это не ваше задание.", show_alert=True)
            return
        try:
            if action == "approve":
                await service.approve_job(job_id=job_id, actor_id=actor_id)
                text = "Одобрено. Начинаю публикацию в Instagram."
            else:
                await service.reject_job(job_id=job_id, actor_id=actor_id)
                text = "Задание отклонено. Публикации не будет."
        except Exception:
            LOGGER.exception("Could not apply %s to job %s", action, job_id)
            await callback.answer("Не удалось выполнить действие.", show_alert=True)
            return
        if callback.message:
            await callback.message.edit_reply_markup(reply_markup=None)
            await callback.message.answer(text)
        await callback.answer()

    return router


def create_dispatcher(
    service: TelegramJobService,
    *,
    admin_ids: frozenset[int] | set[int],
    max_upload_bytes: int,
) -> Dispatcher:
    """Create a dispatcher for callers that do not need additional routers."""

    dispatcher = Dispatcher()
    dispatcher.include_router(
        build_router(
            service,
            admin_ids=admin_ids,
            max_upload_bytes=max_upload_bytes,
        )
    )
    return dispatcher

