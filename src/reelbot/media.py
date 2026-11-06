"""Deterministic media operations used by the Reel bot.

The functions in this module deliberately execute ffmpeg without a shell.  This
keeps user-supplied filenames from becoming shell syntax and also makes every
command straightforward to inspect in tests.
"""

from __future__ import annotations

import subprocess
import tempfile
import textwrap
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


class MediaProcessingError(RuntimeError):
    """Raised when an external media tool cannot complete an operation."""


@dataclass(frozen=True, slots=True)
class SubtitleSegment:
    """A single subtitle cue in seconds."""

    start: float
    end: float
    text: str


CommandRunner = Callable[[Sequence[str]], object]
DEFAULT_MEDIA_COMMAND_TIMEOUT_SECONDS = 900.0


def _default_runner(
    command: Sequence[str],
    *,
    timeout_seconds: float = DEFAULT_MEDIA_COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise MediaProcessingError(f"Required executable is missing: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaProcessingError(
            f"Media command timed out after {timeout_seconds:g} seconds"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise MediaProcessingError(f"Media command failed: {detail}") from exc


def _run_command(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    timeout_seconds: float,
) -> object:
    """Apply the production timeout without breaking simple injected test runners."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if runner is _default_runner:
        return _default_runner(command, timeout_seconds=timeout_seconds)
    return runner(command)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _clean_segments(segments: Iterable[SubtitleSegment]) -> list[SubtitleSegment]:
    cleaned: list[SubtitleSegment] = []
    for segment in sorted(segments, key=lambda item: item.start):
        text = " ".join(segment.text.split()).strip()
        if not text:
            continue
        start = max(0.0, float(segment.start))
        end = max(start + 0.05, float(segment.end))
        cleaned.append(SubtitleSegment(start=start, end=end, text=text))
    return cleaned


def extract_audio(
    video_path: str | Path,
    audio_path: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    timeout_seconds: float = DEFAULT_MEDIA_COMMAND_TIMEOUT_SECONDS,
    runner: CommandRunner = _default_runner,
) -> Path:
    """Extract speech-friendly mono MP3 audio without changing the source video."""

    source = Path(video_path)
    destination = Path(audio_path)
    _ensure_parent(destination)
    _run_command(
        runner,
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "64k",
            str(destination),
        ],
        timeout_seconds=timeout_seconds,
    )
    return destination


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: Iterable[SubtitleSegment], destination: str | Path) -> Path:
    """Write standard UTF-8 SRT subtitles, useful as a downloadable artifact."""

    path = Path(destination)
    _ensure_parent(path)
    blocks: list[str] = []
    for index, segment in enumerate(_clean_segments(segments), start=1):
        wrapped = "\n".join(textwrap.wrap(segment.text, width=42) or [segment.text])
        blocks.append(
            f"{index}\n{_srt_timestamp(segment.start)} --> "
            f"{_srt_timestamp(segment.end)}\n{wrapped}"
        )
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return path


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centis = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape_ass_text(value: str) -> str:
    value = value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return "\\N".join(textwrap.wrap(value, width=32) or [value])


def write_ass(
    segments: Iterable[SubtitleSegment],
    destination: str | Path,
    *,
    font_name: str = "DejaVu Sans",
    font_size: int = 64,
) -> Path:
    """Write styled ASS captions designed for a vertical 1080x1920 canvas."""

    path = Path(destination)
    _ensure_parent(path)
    style_format = (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding"
    )
    style = (
        f"Style: Reel,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00111111,"
        "&H80000000,-1,0,0,0,100,100,0,0,1,5,1,2,80,80,170,1"
    )
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
{style_format}
{style}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dialogue = [
        "Dialogue: 0,"
        f"{_ass_timestamp(segment.start)},{_ass_timestamp(segment.end)},"
        f"Reel,,0,0,0,,{_escape_ass_text(segment.text)}"
        for segment in _clean_segments(segments)
    ]
    path.write_text(header + "\n".join(dialogue) + ("\n" if dialogue else ""), encoding="utf-8")
    return path


def _escape_ffmpeg_filter_path(path: Path) -> str:
    # libavfilter parses this argument itself even though subprocess uses no shell.
    value = str(path.resolve())
    replacements = (
        ("\\", "\\\\"),
        (":", "\\:"),
        ("'", "\\'"),
        (",", "\\,"),
        ("[", "\\["),
        ("]", "\\]"),
    )
    for original, escaped in replacements:
        value = value.replace(original, escaped)
    return value


def burn_subtitles(
    video_path: str | Path,
    subtitle_path: str | Path,
    output_path: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    preset: str = "medium",
    crf: int = 20,
    timeout_seconds: float = DEFAULT_MEDIA_COMMAND_TIMEOUT_SECONDS,
    runner: CommandRunner = _default_runner,
) -> Path:
    """Burn subtitles while retaining the complete source duration and frame."""

    source = Path(video_path)
    subtitles = Path(subtitle_path)
    destination = Path(output_path)
    _ensure_parent(destination)
    ass_filter = f"ass=filename='{_escape_ffmpeg_filter_path(subtitles)}'"
    _run_command(
        runner,
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-map_metadata",
            "0",
            "-vf",
            ass_filter,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        timeout_seconds=timeout_seconds,
    )
    return destination


def _load_cover_font(size: int, preferred_path: str | Path | None = None):
    from PIL import ImageFont

    candidates = [
        preferred_path,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSans-Bold.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(str(candidate), size=size)
    # The bitmap fallback is less attractive but lets processing finish on a
    # misconfigured machine; health/setup checks should ensure DejaVu is present.
    return ImageFont.load_default()


def _wrap_by_pixels(text: str, *, draw: object, font: object, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        proposed = f"{current} {word}"
        left, _top, right, _bottom = draw.textbbox(  # type: ignore[attr-defined]
            (0, 0), proposed, font=font
        )
        if right - left <= max_width:
            current = proposed
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines[:4]


def create_cover(
    video_path: str | Path,
    output_path: str | Path,
    hook: str,
    *,
    ffmpeg: str = "ffmpeg",
    frame_second: float = 0.5,
    font_path: str | Path | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_COMMAND_TIMEOUT_SECONDS,
    runner: CommandRunner = _default_runner,
) -> Path:
    """Create an Instagram-ready 9:16 cover without altering the video itself."""

    from PIL import Image, ImageDraw

    source = Path(video_path)
    destination = Path(output_path)
    _ensure_parent(destination)
    with tempfile.TemporaryDirectory(prefix="reel-cover-") as temporary_directory:
        frame_path = Path(temporary_directory) / "frame.jpg"
        _run_command(
            runner,
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{max(0.0, frame_second):.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                "-q:v",
                "2",
                str(frame_path),
            ],
            timeout_seconds=timeout_seconds,
        )

        image = Image.open(frame_path).convert("RGB").resize((1080, 1920))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        # A bottom-up gradient keeps the face/background visible while making
        # white text legible on arbitrary footage.
        for y in range(700, 1920):
            alpha = min(205, int((y - 700) / 1220 * 205))
            overlay_draw.line((0, y, 1080, y), fill=(0, 0, 0, alpha))
        image = Image.alpha_composite(image.convert("RGBA"), overlay)
        draw = ImageDraw.Draw(image)
        font = _load_cover_font(90, font_path)
        cleaned_hook = " ".join(hook.split()).strip() or "Новое видео"
        lines = _wrap_by_pixels(cleaned_hook[:140], draw=draw, font=font, max_width=920)
        line_height = 112
        total_height = len(lines) * line_height
        y = min(1570, 1660 - total_height)
        for line in lines:
            draw.text(
                (80, y),
                line,
                font=font,
                fill="white",
                stroke_width=4,
                stroke_fill="#111111",
            )
            y += line_height
        image.convert("RGB").save(destination, format="JPEG", quality=94, optimize=True)
    return destination

