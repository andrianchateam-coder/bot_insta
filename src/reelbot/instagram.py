"""Publishing processed videos through the Instagram Graph API.

The publisher deliberately keeps the access token out of URLs used in error
messages and never logs request data.  Callers are expected to pass public
HTTPS URLs that Meta can fetch without authentication.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

Sleep = Callable[[float], Awaitable[None]]


class InstagramError(RuntimeError):
    """Base class for failures reported by the Instagram publisher."""


class InstagramAPIError(InstagramError):
    """The Graph API rejected a request or returned malformed data."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: int | str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class InstagramContainerError(InstagramError):
    """Meta could not finish processing the uploaded Reel container."""

    def __init__(self, container_id: str, status: str, detail: str = "") -> None:
        message = f"Instagram container {container_id} entered {status} state"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.container_id = container_id
        self.status = status
        self.detail = detail


class InstagramProcessingTimeout(InstagramError):
    """Meta did not finish processing within the configured polling window."""

    def __init__(self, container_id: str, timeout_seconds: float) -> None:
        super().__init__(
            f"Instagram container {container_id} was not ready within "
            f"{timeout_seconds:g} seconds"
        )
        self.container_id = container_id
        self.timeout_seconds = timeout_seconds


@dataclass(frozen=True, slots=True)
class PublishResult:
    """Identifiers returned after a Reel has been published."""

    container_id: str
    media_id: str
    permalink: str | None


class InstagramPublisher:
    """Small async client for the Instagram Content Publishing API."""

    TERMINAL_FAILURES = frozenset({"ERROR", "EXPIRED"})

    def __init__(
        self,
        *,
        access_token: str,
        instagram_user_id: str,
        api_version: str = "v22.0",
        client: httpx.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
        poll_interval_seconds: float = 5.0,
        processing_timeout_seconds: float = 300.0,
        permalink_attempts: int = 3,
    ) -> None:
        if not access_token.strip():
            raise ValueError("access_token is required")
        if not instagram_user_id.strip():
            raise ValueError("instagram_user_id is required")
        if not api_version.strip():
            raise ValueError("api_version is required")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero")
        if processing_timeout_seconds <= 0:
            raise ValueError("processing_timeout_seconds must be greater than zero")
        if permalink_attempts < 1:
            raise ValueError("permalink_attempts must be at least one")

        self._token = access_token
        self.instagram_user_id = instagram_user_id
        self.api_version = api_version.strip("/")
        self.base_url = f"https://graph.facebook.com/{self.api_version}"
        self.poll_interval_seconds = poll_interval_seconds
        self.processing_timeout_seconds = processing_timeout_seconds
        self.permalink_attempts = permalink_attempts
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    async def __aenter__(self) -> InstagramPublisher:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def publish_reel(
        self,
        *,
        video_url: str,
        caption: str,
        cover_url: str | None = None,
        share_to_feed: bool = True,
    ) -> PublishResult:
        """Create, process, and publish a Reel, returning its permalink."""

        container_id = await self.create_reel_container(
            video_url=video_url,
            caption=caption,
            cover_url=cover_url,
            share_to_feed=share_to_feed,
        )
        await self.wait_until_finished(container_id)
        media_id = await self.publish_container(container_id)
        permalink = await self.fetch_permalink(media_id)
        return PublishResult(
            container_id=container_id,
            media_id=media_id,
            permalink=permalink,
        )

    async def create_reel_container(
        self,
        *,
        video_url: str,
        caption: str,
        cover_url: str | None = None,
        share_to_feed: bool = True,
    ) -> str:
        """Create a Reel container and return its ID without polling it.

        Keeping this operation separate lets a worker persist ``container_id``
        before waiting, so a process restart does not create a duplicate upload.
        """

        self._validate_public_url(video_url, field="video_url")
        if cover_url:
            self._validate_public_url(cover_url, field="cover_url")
        if len(caption) > 2_200:
            raise ValueError("caption must not exceed 2200 characters")

        create_data: dict[str, str] = {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": str(share_to_feed).lower(),
        }
        if cover_url:
            create_data["cover_url"] = cover_url

        created = await self._request_json(
            "POST", f"/{self.instagram_user_id}/media", data=create_data
        )
        return self._required_string(created, "id", "create container")

    async def publish_container(self, container_id: str) -> str:
        """Publish an already processed container and return the media ID."""

        if not container_id.strip():
            raise ValueError("container_id is required")
        published = await self._request_json(
            "POST",
            f"/{self.instagram_user_id}/media_publish",
            data={"creation_id": container_id},
        )
        return self._required_string(published, "id", "publish container")

    async def wait_until_finished(self, container_id: str) -> None:
        """Wait until Meta finishes processing a previously created container."""

        if not container_id.strip():
            raise ValueError("container_id is required")
        max_polls = max(
            1,
            math.ceil(
                self.processing_timeout_seconds / self.poll_interval_seconds
            ),
        )
        for attempt in range(max_polls):
            payload = await self._request_json(
                "GET",
                f"/{container_id}",
                params={"fields": "status_code,status"},
            )
            status_code = str(payload.get("status_code", "")).upper()
            if status_code == "FINISHED":
                return
            if status_code in self.TERMINAL_FAILURES:
                detail = str(payload.get("status") or "")
                raise InstagramContainerError(container_id, status_code, detail)
            if attempt + 1 < max_polls:
                await self._sleep(self.poll_interval_seconds)

        raise InstagramProcessingTimeout(
            container_id, self.processing_timeout_seconds
        )

    async def fetch_permalink(self, media_id: str) -> str | None:
        """Fetch a published media permalink, retrying eventual consistency."""

        if not media_id.strip():
            raise ValueError("media_id is required")
        for attempt in range(self.permalink_attempts):
            payload = await self._request_json(
                "GET",
                f"/{media_id}",
                params={"fields": "permalink"},
            )
            permalink = payload.get("permalink")
            if isinstance(permalink, str) and permalink:
                return permalink
            if attempt + 1 < self.permalink_attempts:
                await self._sleep(self.poll_interval_seconds)
        return None

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        data: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        request_params = dict(params or {})
        request_data = dict(data or {})
        # Graph accepts bearer authentication. Keeping the token out of query
        # strings prevents ordinary HTTP access/request logs from persisting it.
        headers = {"Authorization": f"Bearer {self._token}"}

        try:
            response = await self._client.request(
                method,
                f"{self.base_url}{path}",
                params=request_params,
                data=request_data,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise InstagramAPIError(
                f"Instagram request failed: {self._redact(str(exc))}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise InstagramAPIError(
                "Instagram returned a non-JSON response",
                status_code=response.status_code,
            ) from exc

        if response.is_error:
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            if not isinstance(error, dict):
                error = {}
            message = self._redact(str(error.get("message") or "request rejected"))
            raise InstagramAPIError(
                f"Instagram API error: {message}",
                status_code=response.status_code,
                error_code=error.get("code"),
            )
        if not isinstance(payload, dict):
            raise InstagramAPIError(
                "Instagram returned an unexpected JSON response",
                status_code=response.status_code,
            )
        return payload

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "[REDACTED]")

    @staticmethod
    def _required_string(
        payload: Mapping[str, Any], key: str, operation: str
    ) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise InstagramAPIError(
                f"Instagram did not return {key!r} while trying to {operation}"
            )
        return value

    @staticmethod
    def _validate_public_url(url: str, *, field: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"{field} must be a public HTTPS URL")


__all__ = [
    "InstagramAPIError",
    "InstagramContainerError",
    "InstagramError",
    "InstagramProcessingTimeout",
    "InstagramPublisher",
    "PublishResult",
]
