"""Gemini client tests.

The wire format is asserted against a mock transport, because that is the part
with the most ways to be subtly wrong: a schema keyword the API rejects, a
camelCase field that must not be renamed, a deprecated generation parameter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from vidvoice.config import Settings
from vidvoice.errors import GeminiError
from vidvoice.gemini import (
    GeminiClient,
    build_prompt,
    dry_run_script,
    parse_script_response,
    to_rest_schema,
)
from vidvoice.models import build_script_schema


def make_client(settings: Settings, handler) -> GeminiClient:
    return GeminiClient(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))


def gemini_response(payload: dict[str, Any], finish: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": json.dumps(payload)}]},
                "finishReason": finish,
            }
        ]
    }


# --------------------------------------------------------------------------
# Schema conversion
# --------------------------------------------------------------------------


class TestToRestSchema:
    def test_types_become_uppercase(self) -> None:
        converted = to_rest_schema({"type": "object", "properties": {"a": {"type": "string"}}})
        assert converted["type"] == "OBJECT"
        assert converted["properties"]["a"]["type"] == "STRING"

    def test_nested_arrays_are_converted(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"n": {"type": "integer"}}},
                }
            },
        }
        converted = to_rest_schema(schema)
        items = converted["properties"]["items"]
        assert items["type"] == "ARRAY"
        assert items["items"]["type"] == "OBJECT"
        assert items["items"]["properties"]["n"]["type"] == "INTEGER"

    @pytest.mark.parametrize(
        "keyword",
        ["minimum", "maximum", "minLength", "maxLength", "pattern", "minItems", "maxItems"],
    )
    def test_unsupported_keywords_are_stripped(self, keyword: str) -> None:
        """Gemini rejects these outright, so sending them is a hard 400."""
        schema = {"type": "string", keyword: 1}
        assert keyword not in to_rest_schema(schema)

    def test_additional_properties_is_stripped(self) -> None:
        assert "additionalProperties" not in to_rest_schema(
            {"type": "object", "additionalProperties": False}
        )

    def test_property_ordering_is_preserved(self) -> None:
        """`propertyOrdering` is supported and camelCase; renaming it breaks."""
        schema = {"type": "object", "propertyOrdering": ["a", "b"]}
        converted = to_rest_schema(schema)
        assert converted["propertyOrdering"] == ["a", "b"]
        assert "property_ordering" not in converted

    def test_arrays_without_items_get_them(self) -> None:
        converted = to_rest_schema({"type": "array"})
        assert converted["items"] == {"type": "STRING"}

    def test_descriptions_survive(self) -> None:
        converted = to_rest_schema({"type": "string", "description": "keep me"})
        assert converted["description"] == "keep me"

    def test_the_real_script_schema_converts_cleanly(self) -> None:
        converted = to_rest_schema(build_script_schema(mode="narration"))
        assert converted["type"] == "OBJECT"
        segment = converted["properties"]["segments"]["items"]
        assert segment["type"] == "OBJECT"
        assert set(segment["properties"]) == {"index", "start", "end", "visual", "text"}

    def test_transcribe_schema_omits_visual(self) -> None:
        converted = to_rest_schema(build_script_schema(mode="transcribe"))
        segment = converted["properties"]["segments"]["items"]
        assert "visual" not in segment["properties"]


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


class TestBuildPrompt:
    def test_narration_asks_for_a_voice_over(self) -> None:
        prompt = build_prompt(mode="narration", video_duration_ms=60_000)
        assert "SILENT" in prompt
        assert "voice-over" in prompt.lower() or "voice-over" in prompt

    def test_transcribe_asks_for_verbatim(self) -> None:
        prompt = build_prompt(mode="transcribe", video_duration_ms=60_000)
        assert "Transcribe" in prompt
        assert "Do NOT paraphrase" in prompt

    def test_narration_includes_the_word_budget(self) -> None:
        prompt = build_prompt(
            mode="narration", video_duration_ms=60_000, target_wpm=150, word_budget=112
        )
        assert "112" in prompt
        assert "150" in prompt

    def test_transcribe_has_no_word_budget(self) -> None:
        """A transcript cannot be shortened, so a budget would be misleading."""
        prompt = build_prompt(
            mode="transcribe", video_duration_ms=60_000, word_budget=112
        )
        assert "word budget stated above" not in prompt

    def test_asks_for_whole_second_timestamps(self) -> None:
        prompt = build_prompt(mode="narration", video_duration_ms=60_000)
        assert "MM:SS" in prompt
        assert "Whole seconds" in prompt or "whole seconds" in prompt

    def test_includes_the_video_duration(self) -> None:
        prompt = build_prompt(mode="narration", video_duration_ms=83_000)
        assert "01:23" in prompt

    def test_context_is_included_and_labelled(self) -> None:
        prompt = build_prompt(
            mode="narration", video_duration_ms=10_000, context="This is about billing."
        )
        assert "This is about billing." in prompt
        assert "AUTHOR" in prompt

    def test_audience_is_included(self) -> None:
        prompt = build_prompt(
            mode="narration", video_duration_ms=10_000, audience="new developers"
        )
        assert "new developers" in prompt

    def test_max_words_is_respected(self) -> None:
        prompt = build_prompt(mode="narration", video_duration_ms=10_000, max_words=17)
        assert "17 words" in prompt


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


class TestParseScriptResponse:
    def test_parses_a_well_formed_response(self) -> None:
        payload = gemini_response(
            {
                "summary": "A demo.",
                "segments": [
                    {"index": 0, "start": "00:01", "end": "00:04", "text": "Hello.", "visual": "a cursor"},
                    {"index": 1, "start": "00:05", "end": "00:09", "text": "World.", "visual": "a menu"},
                ],
            }
        )
        script = parse_script_response(payload, mode="narration", model="test")

        assert script.summary == "A demo."
        assert len(script.segments) == 2
        assert script.segments[0].start_ms == 1_000
        assert script.segments[0].end_ms == 4_000
        assert script.segments[0].visual == "a cursor"

    def test_parses_millisecond_timestamps_if_the_model_emits_them(self) -> None:
        """The prompt asks for whole seconds, but tolerance costs nothing."""
        payload = gemini_response(
            {
                "summary": "s",
                "segments": [
                    {"index": 0, "start": "00:01.250", "end": "00:04.750", "text": "Hi."}
                ],
            }
        )
        script = parse_script_response(payload, mode="narration", model="test")
        assert script.segments[0].start_ms == 1_250

    def test_renumbers_segments_densely(self) -> None:
        payload = gemini_response(
            {
                "summary": "s",
                "segments": [
                    {"index": 7, "start": "00:01", "end": "00:02", "text": "a"},
                    {"index": 9, "start": "00:03", "end": "00:04", "text": "b"},
                ],
            }
        )
        script = parse_script_response(payload, mode="narration", model="test")
        assert [s.index for s in script.segments] == [0, 1]

    def test_sorts_out_of_order_segments(self) -> None:
        payload = gemini_response(
            {
                "summary": "s",
                "segments": [
                    {"index": 0, "start": "00:10", "end": "00:12", "text": "later"},
                    {"index": 1, "start": "00:01", "end": "00:03", "text": "earlier"},
                ],
            }
        )
        script = parse_script_response(payload, mode="narration", model="test")
        assert [s.text for s in script.segments] == ["earlier", "later"]

    def test_skips_segments_with_empty_text(self) -> None:
        payload = gemini_response(
            {
                "summary": "s",
                "segments": [
                    {"index": 0, "start": "00:01", "end": "00:02", "text": "keep"},
                    {"index": 1, "start": "00:03", "end": "00:04", "text": "   "},
                ],
            }
        )
        script = parse_script_response(payload, mode="narration", model="test")
        assert len(script.segments) == 1

    def test_swaps_reversed_times_rather_than_failing(self) -> None:
        payload = gemini_response(
            {
                "summary": "s",
                "segments": [{"index": 0, "start": "00:09", "end": "00:04", "text": "odd"}],
            }
        )
        script = parse_script_response(payload, mode="narration", model="test")
        assert script.segments[0].end_ms >= script.segments[0].start_ms

    def test_handles_json_wrapped_in_a_code_fence(self) -> None:
        body = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '```json\n{"summary": "s", "segments": '
                                '[{"index": 0, "start": "00:01", "end": "00:02", "text": "hi"}]}\n```'
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        script = parse_script_response(body, mode="narration", model="test")
        assert script.segments[0].text == "hi"

    def test_handles_json_with_surrounding_prose(self) -> None:
        body = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": 'Here you go: {"summary": "s", "segments": '
                                '[{"index": 0, "start": "00:01", "end": "00:02", "text": "hi"}]} Hope that helps!'
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        script = parse_script_response(body, mode="narration", model="test")
        assert script.segments[0].text == "hi"

    def test_raises_on_no_candidates(self) -> None:
        with pytest.raises(GeminiError, match="no candidates"):
            parse_script_response({}, mode="narration", model="test")

    def test_reports_a_blocked_request(self) -> None:
        with pytest.raises(GeminiError, match="blocked"):
            parse_script_response(
                {"promptFeedback": {"blockReason": "SAFETY"}},
                mode="narration",
                model="test",
            )

    def test_reports_a_safety_finish(self) -> None:
        body = {"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]}
        with pytest.raises(GeminiError, match="refused"):
            parse_script_response(body, mode="narration", model="test")

    def test_reports_max_tokens_as_a_hint(self) -> None:
        body = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}
        with pytest.raises(GeminiError) as excinfo:
            parse_script_response(body, mode="narration", model="test")
        assert "MAX_TOKENS" in str(excinfo.value.context)

    def test_raises_on_missing_segments(self) -> None:
        payload = gemini_response({"summary": "s"})
        with pytest.raises(GeminiError, match="no 'segments' array"):
            parse_script_response(payload, mode="narration", model="test")

    def test_raises_on_malformed_json(self) -> None:
        body = {
            "candidates": [
                {"content": {"parts": [{"text": "{not json at all"}]}, "finishReason": "STOP"}
            ]
        }
        with pytest.raises(GeminiError):
            parse_script_response(body, mode="narration", model="test")

    def test_raises_when_every_segment_is_unusable(self) -> None:
        payload = gemini_response({"summary": "s", "segments": [{"start": "x", "end": "y"}]})
        with pytest.raises(GeminiError, match="no usable segments"):
            parse_script_response(payload, mode="narration", model="test")


# --------------------------------------------------------------------------
# Client wire format
# --------------------------------------------------------------------------


class TestAnalyzeWireFormat:
    def test_inline_upload_and_request_body(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        video = tmp_path / "small.mp4"
        video.write_bytes(b"fake video bytes")
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=gemini_response({
                "summary": "s",
                "segments": [{"index": 0, "start": "00:01", "end": "00:03", "text": "Hi."}],
            }))

        with make_client(live_settings, handler) as client:
            script = client.analyze(video, video_duration_ms=10_000)

        assert script.segments[0].text == "Hi."
        body = captured["body"]
        parts = body["contents"][0]["parts"]
        # Small files go inline, base64 encoded.
        assert "inline_data" in parts[0]
        assert "generationConfig" in body

    def test_generation_config_uses_the_correct_field_names(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        video = tmp_path / "small.mp4"
        video.write_bytes(b"x")
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=gemini_response({
                "summary": "s",
                "segments": [{"index": 0, "start": "00:01", "end": "00:03", "text": "Hi."}],
            }))

        with make_client(live_settings, handler) as client:
            client.analyze(video, video_duration_ms=10_000)

        config = captured["body"]["generationConfig"]
        # responseJsonSchema, NOT responseSchema: the SDK serialises
        # propertyOrdering incorrectly for the latter.
        assert "responseJsonSchema" in config
        assert "responseSchema" not in config
        assert config["responseMimeType"] == "application/json"
        # camelCase is required in the REST body.
        assert "response_mime_type" not in config

    def test_does_not_send_deprecated_sampling_parameters(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        video = tmp_path / "small.mp4"
        video.write_bytes(b"x")
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=gemini_response({
                "summary": "s",
                "segments": [{"index": 0, "start": "00:01", "end": "00:03", "text": "Hi."}],
            }))

        with make_client(live_settings, handler) as client:
            client.analyze(video, video_duration_ms=10_000)

        config = captured["body"]["generationConfig"]
        # temperature/top_p/top_k are deprecated on Gemini 3.x.
        for banned in ("temperature", "topP", "topK", "top_p", "top_k"):
            assert banned not in config

    def test_api_key_is_sent(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        video = tmp_path / "small.mp4"
        video.write_bytes(b"x")
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["key"] = request.url.params.get("key")
            return httpx.Response(200, json=gemini_response({
                "summary": "s",
                "segments": [{"index": 0, "start": "00:01", "end": "00:03", "text": "Hi."}],
            }))

        with make_client(live_settings, handler) as client:
            client.analyze(video, video_duration_ms=10_000)

        assert captured["key"] == "test-gemini-key"

    def test_model_is_in_the_url(self, live_settings: Settings, tmp_path: Path) -> None:
        video = tmp_path / "small.mp4"
        video.write_bytes(b"x")
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json=gemini_response({
                "summary": "s",
                "segments": [{"index": 0, "start": "00:01", "end": "00:03", "text": "Hi."}],
            }))

        with make_client(live_settings, handler) as client:
            client.analyze(video, video_duration_ms=10_000)

        assert f"/models/{live_settings.gemini_model}:generateContent" in captured["url"]


class TestFileUpload:
    def test_large_files_use_the_resumable_protocol(
        self, live_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video = tmp_path / "big.mp4"
        video.write_bytes(b"0" * 64)
        # Pretend the file is over the inline threshold without writing 18MB.
        monkeypatch.setattr("vidvoice.gemini.INLINE_LIMIT_BYTES", 8)

        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if request.url.path.endswith("/files") and request.method == "POST":
                return httpx.Response(
                    200,
                    headers={"x-goog-upload-url": "https://upload.example/session"},
                    json={},
                )
            if "upload.example" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "file": {
                            "name": "files/abc",
                            "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc",
                            "mimeType": "video/mp4",
                            "state": "ACTIVE",
                        }
                    },
                )
            # The upload call sends raw bytes, so only parse real JSON bodies.
            if b"contents" in request.content:
                return httpx.Response(200, json=gemini_response({
                    "summary": "s",
                    "segments": [{"index": 0, "start": "00:01", "end": "00:03", "text": "Hi."}],
                }))
            return httpx.Response(200, json={})

        with make_client(live_settings, handler) as client:
            script = client.analyze(video, video_duration_ms=10_000)

        assert script.segments[0].text == "Hi."
        start = calls[0]
        assert start.headers["x-goog-upload-protocol"] == "resumable"
        assert start.headers["x-goog-upload-command"] == "start"
        assert "display_name" in json.loads(start.content)["file"]

        upload = calls[1]
        assert upload.headers["x-goog-upload-command"] == "upload, finalize"
        assert upload.headers["x-goog-upload-offset"] == "0"

        # The generateContent call must reference the file URI. Find it by its
        # body rather than by position: the cleanup DELETE comes last.
        generate = next(call for call in calls if b"contents" in call.content)
        parts = json.loads(generate.content)["contents"][0]["parts"]
        assert parts[0]["file_data"]["file_uri"].endswith("/files/abc")

    def test_uploaded_file_is_deleted_afterwards(
        self, live_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video = tmp_path / "big.mp4"
        video.write_bytes(b"0" * 64)
        monkeypatch.setattr("vidvoice.gemini.INLINE_LIMIT_BYTES", 8)

        methods: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            methods.append((request.method, request.url.path))
            if request.url.path.endswith("/files") and request.method == "POST":
                return httpx.Response(
                    200, headers={"x-goog-upload-url": "https://upload.example/s"}, json={}
                )
            if "upload.example" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "file": {
                            "name": "files/abc",
                            "uri": "https://x/files/abc",
                            "mimeType": "video/mp4",
                            "state": "ACTIVE",
                        }
                    },
                )
            if b"contents" in request.content:
                return httpx.Response(200, json=gemini_response({
                    "summary": "s",
                    "segments": [{"index": 0, "start": "00:01", "end": "00:03", "text": "Hi."}],
                }))
            return httpx.Response(200, json={})

        with make_client(live_settings, handler) as client:
            client.analyze(video, video_duration_ms=10_000)

        assert ("DELETE", "/v1beta/files/abc") in methods

    def test_polls_until_active(
        self, live_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video = tmp_path / "big.mp4"
        video.write_bytes(b"0" * 64)
        monkeypatch.setattr("vidvoice.gemini.INLINE_LIMIT_BYTES", 8)
        monkeypatch.setattr("vidvoice.gemini.time.sleep", lambda _s: None)

        polls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/files"):
                return httpx.Response(
                    200, headers={"x-goog-upload-url": "https://upload.example/s"}, json={}
                )
            if "upload.example" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "file": {
                            "name": "files/abc",
                            "uri": "https://x/files/abc",
                            "mimeType": "video/mp4",
                            "state": "PROCESSING",
                        }
                    },
                )
            if request.method == "GET":
                polls["count"] += 1
                state = "ACTIVE" if polls["count"] >= 2 else "PROCESSING"
                return httpx.Response(
                    200, json={"name": "files/abc", "uri": "https://x/files/abc", "state": state}
                )
            return httpx.Response(200, json=gemini_response({
                "summary": "s",
                "segments": [{"index": 0, "start": "00:01", "end": "00:03", "text": "Hi."}],
            }))

        with make_client(live_settings, handler) as client:
            client.analyze(video, video_duration_ms=10_000)

        assert polls["count"] >= 2

    def test_failed_processing_is_reported(
        self, live_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video = tmp_path / "big.mp4"
        video.write_bytes(b"0" * 64)
        monkeypatch.setattr("vidvoice.gemini.INLINE_LIMIT_BYTES", 8)
        monkeypatch.setattr("vidvoice.gemini.time.sleep", lambda _s: None)

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/files"):
                return httpx.Response(
                    200, headers={"x-goog-upload-url": "https://upload.example/s"}, json={}
                )
            if "upload.example" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "file": {
                            "name": "files/abc",
                            "uri": "https://x/files/abc",
                            "mimeType": "video/mp4",
                            "state": "FAILED",
                        }
                    },
                )
            return httpx.Response(200, json={})

        with make_client(live_settings, handler) as client:
            with pytest.raises(GeminiError, match="failed to process"):
                client.analyze(video, video_duration_ms=10_000)

    def test_oversized_file_is_rejected_before_upload(
        self, live_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video = tmp_path / "huge.mp4"
        video.write_bytes(b"0" * 64)
        monkeypatch.setattr("vidvoice.gemini.MAX_UPLOAD_BYTES", 8)
        monkeypatch.setattr("vidvoice.gemini.INLINE_LIMIT_BYTES", 4)

        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("must not reach the network")

        with make_client(live_settings, handler) as client:
            with pytest.raises(GeminiError, match="per-file limit"):
                client.analyze(video, video_duration_ms=10_000)


class TestGeminiErrors:
    def test_invalid_key_is_reported(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        video = tmp_path / "small.mp4"
        video.write_bytes(b"x")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400, json={"error": {"code": 400, "message": "API key not valid"}}
            )

        with make_client(live_settings, handler) as client:
            with pytest.raises(GeminiError) as excinfo:
                client.analyze(video, video_duration_ms=10_000)

        assert "API key not valid" in excinfo.value.message

    def test_transient_failure_is_retried(
        self, live_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video = tmp_path / "small.mp4"
        video.write_bytes(b"x")
        monkeypatch.setattr("vidvoice.gemini.time.sleep", lambda _s: None)
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] < 3:
                return httpx.Response(503, json={"error": {"message": "unavailable"}})
            return httpx.Response(200, json=gemini_response({
                "summary": "s",
                "segments": [{"index": 0, "start": "00:01", "end": "00:03", "text": "Hi."}],
            }))

        with make_client(live_settings, handler) as client:
            client.analyze(video, video_duration_ms=10_000)

        assert attempts["count"] == 3

    def test_quota_error_mentions_the_quota(
        self, live_settings: Settings, tmp_path: Path
    ) -> None:
        video = tmp_path / "small.mp4"
        video.write_bytes(b"x")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"message": "Quota exceeded"}})

        with make_client(live_settings, handler) as client:
            with pytest.raises(GeminiError) as excinfo:
                client.analyze(video, video_duration_ms=10_000)

        assert excinfo.value.hint and "Quota" in excinfo.value.hint


# --------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------


class TestDryRunScript:
    def test_covers_the_video(self) -> None:
        script = dry_run_script(60_000)
        assert script.segments
        assert script.duration_ms <= 60_000
        assert script.coverage(60_000) > 0.3

    def test_segments_do_not_overlap(self) -> None:
        script = dry_run_script(60_000)
        for previous, current in zip(script.segments, script.segments[1:]):
            assert current.start_ms >= previous.end_ms

    def test_respects_max_words(self) -> None:
        script = dry_run_script(120_000, max_words=10)
        for segment in script.segments:
            assert segment.word_count <= 10

    def test_handles_a_very_short_video(self) -> None:
        script = dry_run_script(2_000)
        assert script.duration_ms <= 2_000

    def test_handles_a_zero_duration_video(self) -> None:
        script = dry_run_script(0)
        assert isinstance(script.segments, list)

    def test_marks_itself_as_a_dry_run(self) -> None:
        assert dry_run_script(30_000).meta["dry_run"] is True

    def test_is_deterministic(self) -> None:
        first = dry_run_script(45_000)
        second = dry_run_script(45_000)
        assert [s.to_dict() for s in first.segments] == [s.to_dict() for s in second.segments]
