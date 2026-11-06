from pathlib import Path
from types import SimpleNamespace

from reelbot.ai import OpenAIReelAI, fallback_metadata


class _FakeTranscriptions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return {
            "text": "Сәлем, Қазақстан!",
            "language": "kazakh",
            "segments": [{"start": 0.0, "end": 1.8, "text": "Сәлем, Қазақстан!"}],
        }


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content

    def create(self, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def _client(copy_content: str = "{}"):
    transcriptions = _FakeTranscriptions()
    client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=transcriptions),
        chat=SimpleNamespace(completions=_FakeCompletions(copy_content)),
    )
    return client, transcriptions


def test_transcribe_preserves_detected_kazakh(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")
    client, fake_transcriptions = _client()

    result = OpenAIReelAI(client=client).transcribe(audio)

    assert result.language == "kk"
    assert result.text == "Сәлем, Қазақстан!"
    assert result.segments[0].end == 1.8
    assert fake_transcriptions.request["response_format"] == "verbose_json"
    assert fake_transcriptions.request["timestamp_granularities"] == ["segment"]


def test_generate_metadata_parses_and_normalises_json() -> None:
    client, _ = _client(
        '{"cover_hook":"Неге бұл маңызды?","caption_kk":"Қазақша мәтін.",'
        '"caption_ru":"Русский текст.","hashtags":["reels","#білім","reels"]}'
    )

    result = OpenAIReelAI(client=client).generate_metadata(
        "Маңызды тақырып", language="kk"
    )

    assert result.cover_hook == "Неге бұл маңызды?"
    assert result.caption_kk == "Қазақша мәтін."
    assert result.caption_ru == "Русский текст."
    assert result.hashtags == ("#reels", "#білім")
    assert result.instagram_caption.startswith(
        "Қазақша мәтін.\n\nРусский текст."
    )


def test_generate_metadata_falls_back_on_bad_json() -> None:
    client, _ = _client("not JSON")

    result = OpenAIReelAI(client=client).generate_metadata(
        "Полезный совет для видео", language="ru"
    )

    assert result.cover_hook == "Полезный совет для видео"
    assert result.caption_ru == "Полезный совет для видео"
    assert "#reels" in result.hashtags


def test_fallback_is_bilingual_for_empty_transcript() -> None:
    result = fallback_metadata("", "kk")

    assert result.caption_kk
    assert result.caption_ru
    assert result.cover_hook == "Соңына дейін көріңіз"
