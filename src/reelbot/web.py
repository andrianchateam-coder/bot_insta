"""Small HTTP surface used for health checks and Meta media ingestion."""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


class JobReader(Protocol):
    async def get_job(self, job_id: str) -> Any | None: ...


def _safe_media_path(value: str | Path | None, media_root: Path) -> Path:
    """Resolve a stored path while preventing accidental directory traversal."""

    if not value:
        raise HTTPException(status_code=404, detail="Media is not available")
    candidate = Path(value).resolve()
    root = media_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise HTTPException(status_code=404, detail="Media is not available")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Media is not available")
    return candidate


def create_web_app(database: JobReader, media_root: str | Path) -> FastAPI:
    """Create routes without starting Telegram polling or background workers."""

    root = Path(media_root)
    app = FastAPI(title="Reel Bot", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/media/{job_id}/{token}/{kind}", response_class=FileResponse)
    async def media(job_id: str, token: str, kind: str) -> FileResponse:
        job = await database.get_job(job_id)
        stored_token = getattr(job, "public_token", "") if job else ""
        if not stored_token or not hmac.compare_digest(token, stored_token):
            raise HTTPException(status_code=404, detail="Media is not available")

        if kind == "video":
            path = _safe_media_path(getattr(job, "output_path", None), root)
            return FileResponse(path, media_type="video/mp4", filename=path.name)
        if kind == "cover":
            path = _safe_media_path(getattr(job, "cover_path", None), root)
            return FileResponse(path, media_type="image/jpeg", filename=path.name)
        raise HTTPException(status_code=404, detail="Media is not available")

    return app
