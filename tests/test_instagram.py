from __future__ import annotations

from urllib.parse import parse_qs

import httpx
import pytest

from reelbot.instagram import (
    InstagramAPIError,
    InstagramContainerError,
    InstagramProcessingTimeout,
    InstagramPublisher,
)


def _form(request: httpx.Request) -> dict[str, list[str]]:
    return parse_qs(request.content.decode())


@pytest.mark.asyncio
async def test_publish_reel_happy_path() -> None:
    statuses = iter(["IN_PROGRESS", "FINISHED"])
    seen_create: dict[str, list[str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_create
        if request.method == "POST" and request.url.path.endswith("/123/media"):
            seen_create = _form(request)
            return httpx.Response(200, json={"id": "container-1"})
        if request.method == "GET" and request.url.path.endswith("/container-1"):
            return httpx.Response(200, json={"status_code": next(statuses)})
        if request.method == "POST" and request.url.path.endswith(
            "/123/media_publish"
        ):
            assert _form(request)["creation_id"] == ["container-1"]
            return httpx.Response(200, json={"id": "media-1"})
        if request.method == "GET" and request.url.path.endswith("/media-1"):
            return httpx.Response(
                200, json={"permalink": "https://www.instagram.com/reel/abc/"}
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    async def no_sleep(_: float) -> None:
        return None

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        publisher = InstagramPublisher(
            access_token="top-secret",
            instagram_user_id="123",
            client=client,
            sleep=no_sleep,
            poll_interval_seconds=0.01,
            processing_timeout_seconds=0.05,
        )
        result = await publisher.publish_reel(
            video_url="https://media.example/video.mp4",
            cover_url="https://media.example/cover.jpg",
            caption="Қазақша\n\nРусский",
        )

    assert result.container_id == "container-1"
    assert result.media_id == "media-1"
    assert result.permalink == "https://www.instagram.com/reel/abc/"
    assert seen_create["media_type"] == ["REELS"]
    assert seen_create["share_to_feed"] == ["true"]
    assert seen_create["video_url"] == ["https://media.example/video.mp4"]
    assert seen_create["cover_url"] == ["https://media.example/cover.jpg"]
    assert seen_create["access_token"] == ["top-secret"]


@pytest.mark.asyncio
async def test_checkpointable_step_api() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path.endswith("/123/media"):
            assert _form(request)["caption"] == ["caption"]
            return httpx.Response(200, json={"id": "checkpoint-container"})
        if request.method == "GET" and request.url.path.endswith(
            "/checkpoint-container"
        ):
            return httpx.Response(200, json={"status_code": "FINISHED"})
        if request.method == "POST" and request.url.path.endswith(
            "/123/media_publish"
        ):
            assert _form(request)["creation_id"] == ["checkpoint-container"]
            return httpx.Response(200, json={"id": "checkpoint-media"})
        if request.method == "GET" and request.url.path.endswith(
            "/checkpoint-media"
        ):
            return httpx.Response(
                200, json={"permalink": "https://www.instagram.com/reel/step/"}
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        publisher = InstagramPublisher(
            access_token="secret",
            instagram_user_id="123",
            client=client,
            poll_interval_seconds=0.01,
            processing_timeout_seconds=0.01,
        )

        container_id = await publisher.create_reel_container(
            video_url="https://media.example/video.mp4",
            cover_url="https://media.example/cover.jpg",
            caption="caption",
            share_to_feed=False,
        )
        # A worker can persist container_id here before continuing after restart.
        assert container_id == "checkpoint-container"
        await publisher.wait_until_finished(container_id)
        media_id = await publisher.publish_container(container_id)
        # A worker can likewise persist media_id before permalink lookup.
        assert media_id == "checkpoint-media"
        permalink = await publisher.fetch_permalink(media_id)

    assert permalink == "https://www.instagram.com/reel/step/"
    assert requests == [
        ("POST", "/v22.0/123/media"),
        ("GET", "/v22.0/checkpoint-container"),
        ("POST", "/v22.0/123/media_publish"),
        ("GET", "/v22.0/checkpoint-media"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["ERROR", "EXPIRED"])
async def test_terminal_container_status_is_reported(terminal_status: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "container-1"})
        return httpx.Response(
            200,
            json={"status_code": terminal_status, "status": "encoding failed"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        publisher = InstagramPublisher(
            access_token="secret",
            instagram_user_id="123",
            client=client,
            poll_interval_seconds=0.01,
            processing_timeout_seconds=0.01,
        )
        with pytest.raises(InstagramContainerError) as exc_info:
            await publisher.publish_reel(
                video_url="https://media.example/video.mp4", caption="caption"
            )

    assert exc_info.value.status == terminal_status
    assert "encoding failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_processing_timeout_is_bounded() -> None:
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.method == "POST":
            return httpx.Response(200, json={"id": "container-1"})
        polls += 1
        return httpx.Response(200, json={"status_code": "IN_PROGRESS"})

    async def no_sleep(_: float) -> None:
        return None

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        publisher = InstagramPublisher(
            access_token="secret",
            instagram_user_id="123",
            client=client,
            sleep=no_sleep,
            poll_interval_seconds=0.01,
            processing_timeout_seconds=0.03,
        )
        with pytest.raises(InstagramProcessingTimeout):
            await publisher.publish_reel(
                video_url="https://media.example/video.mp4", caption="caption"
            )

    assert polls == 3


@pytest.mark.asyncio
async def test_api_errors_do_not_expose_access_token() -> None:
    token = "do-not-print-me"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": f"bad token {token}",
                    "code": 190,
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        publisher = InstagramPublisher(
            access_token=token,
            instagram_user_id="123",
            client=client,
        )
        with pytest.raises(InstagramAPIError) as exc_info:
            await publisher.publish_reel(
                video_url="https://media.example/video.mp4", caption="caption"
            )

    assert token not in str(exc_info.value)
    assert exc_info.value.error_code == 190


@pytest.mark.asyncio
async def test_requires_public_https_urls() -> None:
    publisher = InstagramPublisher(
        access_token="secret", instagram_user_id="123"
    )
    with pytest.raises(ValueError, match="public HTTPS"):
        # Validation happens before the publisher performs any network request.
        await publisher.publish_reel(
            video_url="http://localhost/video.mp4", caption="caption"
        )
    await publisher.aclose()
