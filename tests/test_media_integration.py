from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from reelbot.media import (
    SubtitleSegment,
    burn_subtitles,
    create_cover,
    extract_audio,
    write_ass,
)

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg and ffprobe are required for the media integration test",
)


def _run(*command: str) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def _probe(path: Path) -> dict:
    result = subprocess.run(
        (
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height",
            "-of",
            "json",
            str(path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _video_stream(probe: dict) -> dict:
    return next(stream for stream in probe["streams"] if stream["codec_type"] == "video")


def test_real_ffmpeg_pipeline_preserves_video_and_creates_assets(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    subtitles = tmp_path / "captions.ass"
    output = tmp_path / "ready.mp4"
    audio = tmp_path / "speech.mp3"
    cover = tmp_path / "cover.jpg"

    _run(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=0x315a7d:s=720x1280:r=24:d=1.2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=1.2",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(source),
    )

    write_ass([SubtitleSegment(0.05, 1.1, "Сәлем! Проверка субтитров")], subtitles)
    burn_subtitles(source, subtitles, output, preset="ultrafast", crf=28)
    extract_audio(source, audio)
    create_cover(source, cover, "Қазақша және русский", frame_second=0.2)

    source_probe = _probe(source)
    output_probe = _probe(output)
    source_video = _video_stream(source_probe)
    output_video = _video_stream(output_probe)

    assert abs(
        float(source_probe["format"]["duration"])
        - float(output_probe["format"]["duration"])
    ) <= 0.1
    assert (output_video["width"], output_video["height"]) == (
        source_video["width"],
        source_video["height"],
    )
    assert any(stream["codec_type"] == "audio" for stream in output_probe["streams"])
    assert any(stream["codec_type"] == "audio" for stream in _probe(audio)["streams"])
    assert audio.stat().st_size > 0
    with Image.open(cover) as image:
        assert image.size == (1080, 1920)
