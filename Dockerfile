# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 reelbot \
    && useradd --uid 10001 --gid reelbot --create-home --shell /usr/sbin/nologin reelbot

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN --mount=type=secret,id=ca_bundle \
    if [ -f /run/secrets/ca_bundle ]; then \
        export PIP_CERT=/run/secrets/ca_bundle; \
    fi \
    && python -m pip install . \
    && mkdir -p /app/data \
    && chown -R reelbot:reelbot /app/data

USER reelbot

VOLUME ["/app/data"]
EXPOSE 8000

CMD ["python", "-m", "reelbot"]

