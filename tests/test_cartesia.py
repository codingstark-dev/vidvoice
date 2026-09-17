"""Cartesia client tests.

Driven against `httpx.MockTransport`, so these assert the **exact wire format**
without touching the network. That matters because the request body is the part
most likely to drift: the voice specifier, the emotion enum and the API version
have all changed shape across recent revisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from vidvoice.cartesia import (
    SAFE_CHUNK_CHARS,
    CartesiaClient,
    Voice,
    chunk_text,
    summarize_voices,
)
from vidvoice.config import EMOTIONS, Settings
from vidvoice.errors import CartesiaError

#: A tiny valid WAV, so duration probing has something real to read.
def build_wav_bytes(seconds: float = 0.5, rate: int = 48_000) -> bytes:
    import io
    import struct
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = int(rate * seconds)
        handle.writeframes(struct.pack("<h", 0) * frames * 2)
    return buffer.getvalue()


def make_client(settings: Settings, handler) -> CartesiaClient:
    transport = httpx.MockTransport(handler)
    return CartesiaClient(settings, client=httpx.Client(transport=transport))


class TestSynthesizeRequest:
    def test_headers_pin_the_api_version_and_use_bearer_auth(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=build_wav_bytes())

        with make_client(live_settings, handler) as client:
            client.synthesize("Hello there.", tmp_path / "out.wav")

        headers = captured["headers"]
        assert headers["cartesia-version"] == live_settings.cartesia_version
        assert headers["authorization"] == "Bearer test-cartesia-key"
        assert captured["url"] == "https://api.cartesia.ai/tts/bytes"

    def test_body_shape_matches_the_current_spec(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=build_wav_bytes())

        with make_client(live_settings, handler) as client:
            client.synthesize("Testing one two three.", tmp_path / "out.wav")

        body = captured["body"]
        assert body["model_id"] == live_settings.cartesia_model
        # `transcript` is the field name; `text` never existed.
        assert body["transcript"] == "Testing one two three."
        assert "text" not in body
        # Current spec: a bare id or {"id": ...} -- no `mode` key.
        assert body["voice"] == {"id": live_settings.cartesia_voice_id}
        assert "mode" not in body["voice"]
        assert body["output_format"] == {
            "container": "wav",
            "sample_rate": 48_000,
            "encoding": "pcm_s16le",
        }
        assert body["generation_config"]["speed"] == 1.0

    def test_emotion_is_sent_when_given(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=build_wav_bytes())

        with make_client(live_settings, handler) as client:
            client.synthesize("Excited!", tmp_path / "out.wav", emotion="excited")

        assert captured["body"]["generation_config"]["emotion"] == "excited"

    def test_emotion_omitted_when_not_given(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=build_wav_bytes())

        with make_client(live_settings, handler) as client:
            client.synthesize("Neutral.", tmp_path / "out.wav")

        # Absent means "let the model read the text's own subtext".
        assert "emotion" not in captured["body"]["generation_config"]

    def test_writes_the_audio_bytes(self, live_settings: Settings, tmp_path: Path) -> None:
        payload = build_wav_bytes(0.25)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=payload)

        out = tmp_path / "nested" / "out.wav"
        with make_client(live_settings, handler) as client:
            client.synthesize("Hi.", out)

        assert out.read_bytes() == payload

    def test_rejects_empty_text(self, live_settings: Settings, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("should not reach the network")

        with make_client(live_settings, handler) as client:
            with pytest.raises(CartesiaError, match="empty text"):
                client.synthesize("   ", tmp_path / "out.wav")

    def test_rejects_oversized_text(self, live_settings: Settings, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("should not reach the network")

        with make_client(live_settings, handler) as client:
            with pytest.raises(CartesiaError, match="safety chunk size"):
                client.synthesize("x" * (SAFE_CHUNK_CHARS + 1), tmp_path / "out.wav")

    def test_rejects_both_language_and_locale(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        """The API 400s when both are set, even to the same value."""

        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("should not reach the network")

        with make_client(live_settings, handler) as client:
            with pytest.raises(CartesiaError, match="never both"):
                client.synthesize(
                    "Hi.", tmp_path / "out.wav", language="en", locale="en-US"
                )

    def test_locale_alone_is_allowed(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=build_wav_bytes())

        with make_client(live_settings, handler) as client:
            client.synthesize("Hi.", tmp_path / "out.wav", locale="en-GB")

        assert captured["body"]["locale"] == "en-GB"
        assert "language" not in captured["body"]

    def test_rejects_unsupported_sample_rate(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        settings = live_settings.with_overrides(sample_rate=12345)

        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("should not reach the network")

        with make_client(settings, handler) as client:
            with pytest.raises(CartesiaError, match="Unsupported sample rate"):
                client.synthesize("Hi.", tmp_path / "out.wav")


class TestSpeedAndVolumeClamping:
    def test_speed_above_max_is_clamped(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=build_wav_bytes())

        with make_client(live_settings, handler) as client:
            client.synthesize("Fast.", tmp_path / "out.wav", speed=9.0)

        assert captured["body"]["generation_config"]["speed"] == 1.5

    def test_speed_below_min_is_clamped(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=build_wav_bytes())

        with make_client(live_settings, handler) as client:
            client.synthesize("Slow.", tmp_path / "out.wav", speed=0.1)

        assert captured["body"]["generation_config"]["speed"] == 0.6

    def test_volume_is_clamped(self, live_settings: Settings, tmp_path: Path) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=build_wav_bytes())

        with make_client(live_settings, handler) as client:
            client.synthesize("Loud.", tmp_path / "out.wav", volume=10.0)

        assert captured["body"]["generation_config"]["volume"] == 2.0


class TestErrors:
    def test_structured_error_is_surfaced(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={
                    "error_code": "unauthorized",
                    "title": "Unauthorized",
                    "message": "Invalid API key",
                    "request_id": "abc",
                },
            )

        with make_client(live_settings, handler) as client:
            with pytest.raises(CartesiaError) as excinfo:
                client.synthesize("Hi.", tmp_path / "out.wav")

        assert "Invalid API key" in excinfo.value.message
        assert excinfo.value.context["status_code"] == 401
        assert excinfo.value.hint and "CARTESIA_API_KEY" in excinfo.value.hint

    def test_plain_text_error_is_surfaced(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        """Pre-2026-03-01 (and bad version headers) return plain text."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="Bad Request: unsupported version")

        with make_client(live_settings, handler) as client:
            with pytest.raises(CartesiaError) as excinfo:
                client.synthesize("Hi.", tmp_path / "out.wav")

        assert "unsupported version" in excinfo.value.message

    def test_empty_audio_body_is_an_error(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        with make_client(live_settings, handler) as client:
            with pytest.raises(CartesiaError, match="empty audio body"):
                client.synthesize("Hi.", tmp_path / "out.wav")

    def test_rate_limit_is_retried_then_succeeds(
        self, live_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("vidvoice.cartesia.time.sleep", lambda _s: None)
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] < 3:
                return httpx.Response(429, json={"message": "concurrency_limited"})
            return httpx.Response(200, content=build_wav_bytes())

        with make_client(live_settings, handler) as client:
            client.synthesize("Hi.", tmp_path / "out.wav")

        assert attempts["count"] == 3

    def test_persistent_failure_eventually_raises(
        self, live_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("vidvoice.cartesia.time.sleep", lambda _s: None)
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            return httpx.Response(503, json={"message": "unavailable"})

        with make_client(live_settings, handler) as client:
            with pytest.raises(CartesiaError):
                client.synthesize("Hi.", tmp_path / "out.wav")

        assert attempts["count"] == 4  # _MAX_ATTEMPTS


class TestVoiceListing:
    def test_parses_the_current_voice_object(
        self, live_settings: Settings
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/voices"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "v1",
                            "name": "Skylar",
                            "description": "Calm and clear",
                            "language": "en",
                            "gender": "feminine",
                            "access": "public",
                            "visibility": "all",
                            "is_owner": False,
                            "is_pro": False,
                            "status": "ready",
                        }
                    ],
                    "has_more": False,
                },
            )

        with make_client(live_settings, handler) as client:
            voices = client.list_voices()

        assert len(voices) == 1
        voice = voices[0]
        assert voice.id == "v1"
        assert voice.name == "Skylar"
        assert voice.access == "public"     # string enum, replaced is_public
        assert voice.visibility == "all"
        assert voice.language == "en"

    def test_paginates_using_next_page(self, live_settings: Settings) -> None:
        pages = {
            None: {"data": [{"id": "a"}, {"id": "b"}], "has_more": True, "next_page": "b"},
            "b": {"data": [{"id": "c"}], "has_more": False},
        }
        seen: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            cursor = request.url.params.get("starting_after")
            seen.append(cursor)
            return httpx.Response(200, json=pages[cursor])

        with make_client(live_settings, handler) as client:
            voices = client.list_voices(limit=10)

        assert [v.id for v in voices] == ["a", "b", "c"]
        assert seen == [None, "b"]

    def test_passes_filters_to_the_api(self, live_settings: Settings) -> None:
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(dict(request.url.params))
            return httpx.Response(200, json={"data": [], "has_more": False})

        with make_client(live_settings, handler) as client:
            client.list_voices(search="calm", gender="feminine", language="en")

        assert captured["q"] == "calm"
        assert captured["gender"] == "feminine"
        assert captured["language"] == "en"

    def test_respects_the_limit(self, live_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"data": [{"id": str(i)} for i in range(50)], "has_more": False}
            )

        with make_client(live_settings, handler) as client:
            assert len(client.list_voices(limit=5)) == 5


class TestVoiceModel:
    def test_label_includes_useful_bits(self) -> None:
        voice = Voice(id="x", name="Skylar", gender="feminine", language="en")
        assert "Skylar" in voice.label
        assert "feminine" in voice.label
        assert "en" in voice.label

    def test_label_falls_back_to_id(self) -> None:
        assert Voice(id="abc123").label == "abc123"

    def test_summarize_renders_rows(self) -> None:
        text = summarize_voices([Voice(id="v1", name="Skylar", language="en")])
        assert "v1" in text
        assert "Skylar" in text


class TestChunkText:
    def test_short_text_is_one_chunk(self) -> None:
        assert chunk_text("Hello.") == ["Hello."]

    def test_splits_on_sentence_boundaries(self) -> None:
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunk_text(text, limit=40)
        assert len(chunks) > 1
        assert all(len(chunk) <= 40 for chunk in chunks)
        assert "".join(c.replace(" ", "") for c in chunks) == text.replace(" ", "")

    def test_hard_wraps_a_sentence_longer_than_the_limit(self) -> None:
        chunks = chunk_text("x" * 100, limit=30)
        assert all(len(chunk) <= 30 for chunk in chunks)


class TestEmotionPalette:
    def test_config_palette_matches_the_spec_count(self) -> None:
        # The 2026-08-14 Emotion enum has 58 values.
        assert len(EMOTIONS) == 58

    def test_no_duplicates(self) -> None:
        assert len(set(EMOTIONS)) == len(EMOTIONS)

    def test_documented_primaries_are_present(self) -> None:
        for primary in ("neutral", "calm", "angry", "content", "sad", "scared"):
            assert primary in EMOTIONS
