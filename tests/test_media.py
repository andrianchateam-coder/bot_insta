import subprocess
from pathlib import Path

import pytest

from reelbot.media import (
    MediaProcessingError,
    SubtitleSegment,
    burn_subtitles,
    extract_audio,
    write_ass,
    write_srt,
)


def test_write_srt_uses_timestamp_and_utf8_kazakh(tmp_path: Path) -> None:
    destination = tmp_path / "captions.srt"

    write_srt([SubtitleSegment(1.234, 3.5, "Қазақша мәтін")], destination)

    assert destination.read_text(encoding="utf-8") == (
        "1\n00:00:01,234 --> 00:00:03,500\nҚазақша мәтін\n"
    )


def test_write_ass_contains_readable_style_and_dialogue(tmp_path: Path) -> None:
    destination = tmp_path / "captions.ass"

    write_ass([SubtitleSegment(0, 2, "Привет, әлем!")], destination)

    content = destination.read_text(encoding="utf-8")
    assert "PlayResX: 1080" in content
    assert "Style: Reel,DejaVu Sans,64" in content
    assert "Dialogue: 0,0:00:00.00,0:00:02.00" in content
    assert "Привет, әлем!" in content


def test_extract_audio_builds_mono_16khz_command(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    output = tmp_path / "audio.mp3"

    extract_audio("source.mp4", output, runner=lambda command: commands.append(list(command)))

    assert commands[0][0] == "ffmpeg"
    assert commands[0][commands[0].index("-ac") + 1] == "1"
    assert commands[0][commands[0].index("-ar") + 1] == "16000"
    assert "-vn" in commands[0]
    assert commands[0][-1] == str(output)


def test_burn_subtitles_does_not_crop_or_trim_video(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    output = tmp_path / "ready.mp4"

    burn_subtitles(
        "source.mp4",
        tmp_path / "captions.ass",
        output,
        runner=lambda command: commands.append(list(command)),
    )

    command = commands[0]
    assert "-t" not in command
    assert "-ss" not in command
    assert "-shortest" not in command
    filter_value = command[command.index("-vf") + 1]
    assert filter_value.startswith("ass=filename=")
    assert "crop" not in filter_value
    assert command[-1] == str(output)


def test_default_media_runner_enforces_timeout(monkeypatch, tmp_path: Path) -> None:
    def timed_out(command, **kwargs):
        assert kwargs["timeout"] == 12
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timed_out)

    with pytest.raises(MediaProcessingError, match="timed out after 12 seconds"):
        extract_audio(
            "source.mp4",
            tmp_path / "audio.mp3",
            timeout_seconds=12,
        )

