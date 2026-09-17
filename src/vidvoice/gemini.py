"""Gemini video-understanding client.

Talks to the Generative Language REST API directly over httpx. Two reasons not
to use `google-genai` here:

1. One fewer dependency in a tool meant to run on any machine.
2. The wire format for schema-constrained output is the part most likely to
   drift between SDK versions. Keeping it visible in this file means a future
   change is a one-line fix, not a mystery.

The two jobs this client does:

``narration``
    Watch a *silent* screen recording and write what a presenter would say,
    anchored to the moments on screen. This is the cap.so use case.

``transcribe``
    Transcribe speech that is already in the video, with timings, so it can be
    re-voiced in a different voice.

Both return a :class:`~vidvoice.models.Script` whose segment times are what the
rest of the pipeline treats as ground truth.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import Settings
from .errors import GeminiError
from .models import Script, ScriptMode, Segment, build_script_schema, format_timestamp, parse_timestamp

log = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com"
API_VERSION = "v1beta"
UPLOAD_BASE = "https://generativelanguage.googleapis.com/upload/v1beta"

#: Inline base64 is only used below this size; the Files API is the real path.
#: Inline requests cap at 100 MB total, and base64 inflates payloads by ~33%,
#: so this stays comfortably clear of that ceiling.
INLINE_LIMIT_BYTES = 18 * 1024 * 1024

#: Documented per-file ceiling for the Files API (20 GB is the per-project cap).
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

#: Uploaded files are deleted by Gemini after this long; we clean up sooner.
FILE_EXPIRY_HOURS = 48

_RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 4

#: Generous, because a schema-constrained script for a long video is a big response.
DEFAULT_MAX_OUTPUT_TOKENS = 32_768

#: Narration pace assumed when synthesising dry-run placeholder lengths.
_DRY_RUN_WPM = 150.0

#: Spacing between dry-run placeholder lines. Roughly one narration beat.
_DRY_RUN_CADENCE_MS = 5_000


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

_COMMON_RULES = """\
TIMESTAMP RULES (critical -- these timings drive an automated voice-over):
- Use MM:SS format, e.g. 00:04 or 01:23. Whole seconds only -- do not use \
fractions of a second and do not use seconds alone.
- `end` must be strictly greater than `start`.
- Segments must be chronological and must NOT overlap each other.
- Timestamps must refer to real moments in THIS video, at its real length of \
{duration}. Do not invent timings that fall beyond the end.
- Leave a natural pause between segments rather than making one run into the next.

NOTE ON PRECISION: your view of this video has one-second granularity, so a \
timestamp may be up to a second out. Aim for the nearest whole second rather than \
trying to be more precise than you can be.

CONTENT RULES:
- Split at natural pauses or changes of on-screen focus, not at a fixed interval.
- One sentence per segment, and give each segment enough time to be spoken: a \
segment should last at least 2 seconds and contain at most {max_words} words.
- Write plain prose: no markdown, no bullet points, no emoji, no stage directions \
in square brackets, no speaker labels.
- Keep product names, menu items and on-screen labels verbatim and correctly spelled.
{budget_note}
"""

_NARRATION_RULES = """\
You are writing the voice-over for a screen recording. The recording is SILENT: there \
is no speech in it to transcribe. Your job is to watch what happens on screen and \
write what a clear, friendly presenter would SAY to explain it.

LENGTH IS PART OF THE BRIEF. The video is {duration} long and the finished narration \
will be spoken aloud at roughly {wpm} words per minute. Your total script must therefore \
be about {word_budget} words -- no more than {word_budget_max}. A script that is too long \
cannot be used: it would have to be sped up until it sounds rushed, or it would run past \
the end of the video. Count your words before answering, and cut material rather than \
exceeding the budget. It is much better to say less, well, than to cram.

How to write it:
- Describe what the viewer is looking at and why it matters, as it happens.
- Be concrete and grounded in what is actually visible. Refer to the real names of \
buttons, panels, files and fields that appear on screen.
- Do not read the screen aloud word for word. Explain it the way a person would.
- Do not invent features, numbers or outcomes that are not visible in the video.
- Do not narrate the cursor itself ("I move my mouse to..."). Describe the task.
- Distribute the words across the whole video. Do not front-load everything into the \
first few seconds and leave the rest silent.
- If the video is too short to say anything meaningful, return a single short segment \
rather than many rushed ones.
"""

_TRANSCRIBE_RULES = """\
Transcribe the speech in this video, with accurate timing.

- Transcribe exactly what is said, in the language it is spoken.
- Do NOT paraphrase, summarise, correct grammar, or add anything that was not said.
- Do not describe visuals, actions or sounds. Speech only.
- Use normal punctuation and capitalisation so the text can be re-voiced naturally.
- If a stretch of the video has no speech at all, simply do not emit a segment for it.
- If a word is genuinely unintelligible, write [inaudible] for just that word.
"""


def build_prompt(
    *,
    mode: ScriptMode = "narration",
    video_duration_ms: int = 0,
    context: str = "",
    max_words: int = 26,
    audience: str = "",
    target_wpm: float = 150.0,
    word_budget: int = 0,
) -> str:
    """Compose the analysis prompt for the requested mode.

    ``word_budget`` is the key input for length matching: telling the model how
    many words fit is dramatically more effective than trimming afterwards,
    because it never generates the excess in the first place.
    """
    duration = format_timestamp(video_duration_ms) if video_duration_ms else "unknown"

    parts: list[str] = []
    if mode == "narration":
        budget = word_budget or 0
        parts.append(
            _NARRATION_RULES.format(
                duration=duration,
                wpm=int(round(target_wpm)),
                word_budget=budget or "as many as fit naturally",
                word_budget_max=int(round(budget * 1.1)) if budget else "that figure",
            )
        )
    else:
        parts.append(_TRANSCRIBE_RULES)

    parts.append(
        _COMMON_RULES.format(
            duration=duration,
            max_words=max_words,
            # Transcription cannot be shortened -- it must be verbatim -- so the
            # prompt for that mode must not imply a word budget.
            budget_note=""
            if mode == "transcribe"
            else "Do not exceed the total word budget stated above.",
        )
    )

    if audience:
        parts.append(f"AUDIENCE: write for {audience.strip()}.")
    if context:
        parts.append(
            "ADDITIONAL CONTEXT FROM THE VIDEO'S AUTHOR (treat as authoritative, but "
            f"never contradict what is actually visible):\n{context.strip()}"
        )

    parts.append(
        "Return the `summary` field as one or two sentences describing the whole video, "
        "and the `segments` array as described by the response schema."
    )
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Schema conversion
# --------------------------------------------------------------------------


def to_rest_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON-Schema-ish dict to the OpenAPI subset Gemini's REST API wants.

    The REST API requires uppercase type names (``OBJECT``, ``STRING``). It also
    rejects JSON Schema keywords outside its supported subset -- notably
    ``minimum``, ``maximum``, ``minLength``, ``maxLength``, ``pattern`` and
    ``additionalProperties`` -- so those are stripped rather than sent and 400'd.
    """
    unsupported = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "additionalProperties",
        "$schema",
        "$id",
    }

    def convert(node: Any) -> Any:
        if isinstance(node, list):
            return [convert(item) for item in node]
        if not isinstance(node, dict):
            return node

        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in unsupported:
                continue
            if key == "type" and isinstance(value, str):
                out[key] = value.upper()
            elif key in {"properties"} and isinstance(value, dict):
                out[key] = {name: convert(sub) for name, sub in value.items()}
            elif key == "items":
                out[key] = convert(value)
            elif key == "propertyOrdering" and isinstance(value, list):
                out[key] = [str(item) for item in value]
            elif key == "required" and isinstance(value, list):
                out[key] = [str(item) for item in value]
            else:
                out[key] = convert(value)
        return out

    converted = convert(schema)
    # Gemini requires `items` to be present on every array.
    return _ensure_array_items(converted)


def _ensure_array_items(node: Any) -> Any:
    if isinstance(node, list):
        return [_ensure_array_items(item) for item in node]
    if not isinstance(node, dict):
        return node
    if node.get("type") == "ARRAY" and "items" not in node:
        node["items"] = {"type": "STRING"}
    if "properties" in node and isinstance(node["properties"], dict):
        node["properties"] = {
            name: _ensure_array_items(sub) for name, sub in node["properties"].items()
        }
    if "items" in node:
        node["items"] = _ensure_array_items(node["items"])
    return node


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


def parse_script_response(
    payload: dict[str, Any],
    *,
    mode: ScriptMode,
    model: str,
    source_video: str = "",
) -> Script:
    """Turn a schema-constrained Gemini response into a validated Script."""
    candidates = payload.get("candidates") or []
    if not candidates:
        feedback = payload.get("promptFeedback") or {}
        reason = feedback.get("blockReason")
        if reason:
            raise GeminiError(
                f"Gemini blocked the request ({reason}).",
                hint="Try a different video, or loosen the safety settings.",
            )
        raise GeminiError("Gemini returned no candidates.", body=_truncate(json.dumps(payload)))

    candidate = candidates[0]
    finish = candidate.get("finishReason")
    if finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"}:
        raise GeminiError(
            f"Gemini refused to analyse the video (finishReason={finish})."
        )

    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    if not text.strip():
        raise GeminiError(
            f"Gemini returned an empty response (finishReason={finish}).",
            hint=(
                "If finishReason is MAX_TOKENS, raise --max-output-tokens or shorten "
                "the video."
            ),
            finish_reason=finish,
        )

    data = _extract_json(text)

    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list):
        raise GeminiError(
            "Gemini's JSON response had no 'segments' array.",
            body=_truncate(text),
        )

    segments: list[Segment] = []
    for position, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            continue
        try:
            start = parse_timestamp(raw.get("start", 0))
            end = parse_timestamp(raw.get("end", start))
        except ValueError as exc:
            log.warning("Skipping segment %d with unparseable time: %s", position, exc)
            continue

        text_value = str(raw.get("text") or "").strip()
        if not text_value:
            continue

        segments.append(
            Segment(
                index=int(raw.get("index", position)),
                start_ms=start,
                end_ms=max(start, end),
                text=text_value,
                visual=str(raw.get("visual") or ""),
                tone=str(raw.get("tone") or ""),
            )
        )

    if not segments:
        raise GeminiError(
            "Gemini returned no usable segments.",
            hint="Check that the video actually contains visible activity or speech.",
            body=_truncate(text),
        )

    # Re-number densely: the model's own indices often skip or repeat.
    ordered = sorted(segments, key=lambda s: (s.start_ms, s.end_ms))
    for new_index, segment in enumerate(ordered):
        segment.index = new_index

    return Script(
        segments=ordered,
        mode=mode,
        summary=str(data.get("summary") or ""),
        source_video=source_video,
        model=model,
        meta={"finish_reason": finish},
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a response that may be fenced or chatty."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        # ```json ... ``` or ``` ... ```
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]

    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to the outermost balanced braces.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise GeminiError(
                "Gemini did not return JSON.",
                hint="Re-run; if it persists, lower --max-words so the response is shorter.",
                body=_truncate(text),
            )
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise GeminiError(
                f"Gemini returned malformed JSON: {exc}",
                body=_truncate(text),
            ) from exc

    if not isinstance(parsed, dict):
        raise GeminiError("Gemini returned JSON that is not an object.", body=_truncate(text))
    return parsed


def _truncate(text: str, limit: int = 600) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... [{len(text)} chars total]"


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class GeminiClient:
    """Upload a video, wait for processing, and get a timed script back."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.settings = settings
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> GeminiClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- plumbing -----------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        merged_params = {"key": self.settings.require_gemini_key()}
        if params:
            merged_params.update(params)

        last_error: Exception | None = None
        last_hint: str | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.request(
                    method,
                    url,
                    params=merged_params,
                    json=json_body,
                    content=content,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                log.warning("Gemini %s failed (%s), attempt %d", method, exc, attempt)
            else:
                if response.status_code < 400:
                    return response

                detail = _error_detail(response)
                if response.status_code not in _RETRY_STATUSES:
                    raise GeminiError(
                        f"Gemini returned {response.status_code}: {detail}",
                        hint=_hint_for_status(response.status_code),
                        status_code=response.status_code,
                        body=detail,
                    )
                # Remember why this attempt failed: if we exhaust our retries,
                # the *specific* reason (quota, model missing) is far more
                # useful than a generic "check your network".
                last_error = GeminiError(
                    f"Gemini returned {response.status_code}: {detail}",
                    status_code=response.status_code,
                    body=detail,
                )
                last_hint = _hint_for_status(response.status_code)
                retry_after = response.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(float(retry_after), 60.0))
                    continue

            if attempt < _MAX_ATTEMPTS:
                time.sleep(min(2.0 ** (attempt - 1), 15.0))

        raise GeminiError(
            f"Gemini request failed after {_MAX_ATTEMPTS} attempts: {last_error}",
            hint=last_hint or "Check your network and that GEMINI_API_KEY is valid.",
        )

    # -- files --------------------------------------------------------------

    def upload(self, path: Path | str) -> dict[str, Any]:
        """Upload a video via the resumable Files API and return its metadata.

        Single-shot: we send all the bytes with ``upload, finalize`` in one
        request rather than chunking. The protocol permits it, and it keeps the
        code simple. Video files large enough to warrant 8 MiB chunking would
        normally be over the per-file ceiling anyway.
        """
        path = Path(path)
        if not path.is_file():
            raise GeminiError(f"Video not found: {path}")

        size = path.stat().st_size
        _check_upload_size(path, size)

        mime = mimetypes.guess_type(path.name)[0] or "video/mp4"

        # Step 1: start a resumable session.
        start = self._request(
            "POST",
            f"{UPLOAD_BASE}/files",
            headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime,
                "Content-Type": "application/json",
            },
            json_body={"file": {"display_name": path.name}},
        )

        upload_url = start.headers.get("x-goog-upload-url")
        if not upload_url:
            raise GeminiError(
                "Gemini did not return an upload URL.",
                hint="The API key may be invalid or the Files API unavailable for it.",
            )

        # Step 2: upload the bytes and finalise in one command.
        log.info("Uploading %s (%.1f MB) to Gemini...", path.name, size / 1_048_576)
        finished = self._request(
            "POST",
            upload_url,
            headers={
                "Content-Length": str(size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=path.read_bytes(),
        )

        try:
            metadata = finished.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise GeminiError("Gemini returned unparseable upload metadata.") from exc

        file_info = metadata.get("file") or metadata
        if not file_info.get("uri"):
            raise GeminiError("Upload succeeded but no file URI was returned.", body=str(metadata))
        return file_info

    def wait_until_active(
        self,
        file_info: dict[str, Any],
        *,
        poll_seconds: float = 3.0,
        timeout_seconds: float = 600.0,
    ) -> dict[str, Any]:
        """Poll a freshly uploaded file until Gemini finishes processing it."""
        name = file_info.get("name")
        if not name:
            return file_info

        deadline = time.monotonic() + timeout_seconds
        current = file_info
        while True:
            state = ((current.get("state") or {}).get("name") if isinstance(current.get("state"), dict) else current.get("state")) or ""
            state = str(state).upper()

            if state in {"ACTIVE", ""}:
                return current
            if state == "FAILED":
                raise GeminiError(
                    f"Gemini failed to process {file_info.get('displayName', name)}.",
                    hint="The file may be corrupt or in an unsupported format.",
                )
            if time.monotonic() > deadline:
                raise GeminiError(
                    f"Timed out waiting for Gemini to process the video "
                    f"(last state: {state}).",
                    hint="Try a shorter video, or a lower-bitrate export.",
                )

            time.sleep(poll_seconds)
            response = self._request("GET", f"{API_BASE}/{API_VERSION}/{name}")
            current = response.json()

    def delete(self, file_info: dict[str, Any]) -> None:
        """Best-effort cleanup of an uploaded file."""
        name = file_info.get("name")
        if not name:
            return
        try:
            self._request("DELETE", f"{API_BASE}/{API_VERSION}/{name}")
        except (GeminiError, httpx.HTTPError) as exc:
            log.debug("Could not delete %s from Gemini: %s", name, exc)

    # -- analysis -----------------------------------------------------------

    def analyze(
        self,
        path: Path | str,
        *,
        mode: ScriptMode = "narration",
        video_duration_ms: int = 0,
        context: str = "",
        audience: str = "",
        max_words: int = 26,
        target_wpm: float = 150.0,
        word_budget: int = 0,
        thinking_level: str = "medium",
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        cleanup: bool = True,
        on_status: Callable[[str], None] | None = None,
    ) -> Script:
        """Upload *path*, analyse it, and return a timed :class:`Script`."""
        path = Path(path)
        notify = on_status or (lambda _msg: None)

        size = path.stat().st_size if path.is_file() else 0
        _check_upload_size(path, size)

        file_info: dict[str, Any] | None = None
        try:
            if size <= INLINE_LIMIT_BYTES:
                notify(f"Sending {path.name} inline ({size / 1_048_576:.1f} MB)")
                video_part: dict[str, Any] = {
                    "inline_data": {
                        "mime_type": mimetypes.guess_type(path.name)[0] or "video/mp4",
                        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                    }
                }
            else:
                notify(f"Uploading {path.name} ({size / 1_048_576:.1f} MB)")
                file_info = self.upload(path)
                file_info = self.wait_until_active(file_info)
                notify("Video processed; analysing")
                video_part = {
                    "file_data": {
                        "mime_type": file_info.get("mimeType") or "video/mp4",
                        "file_uri": file_info["uri"],
                    }
                }

            prompt = build_prompt(
                mode=mode,
                video_duration_ms=video_duration_ms,
                context=context,
                audience=audience,
                max_words=max_words,
                target_wpm=target_wpm,
                word_budget=word_budget,
            )

            body = {
                "contents": [
                    {"role": "user", "parts": [video_part, {"text": prompt}]}
                ],
                "generationConfig": {
                    "maxOutputTokens": max_output_tokens,
                    # `responseJsonSchema` passes schema keys through verbatim.
                    # Do NOT switch to `responseSchema`: the google-genai SDK
                    # serialises `propertyOrdering` as `property_ordering` for
                    # that field, which the API rejects.
                    "responseMimeType": "application/json",
                    "responseJsonSchema": to_rest_schema(build_script_schema(mode=mode)),
                    # `temperature`, `top_p` and `top_k` are deprecated on
                    # Gemini 3.x and must not be sent.
                    "thinkingConfig": {"thinkingLevel": thinking_level},
                },
            }

            response = self._request(
                "POST",
                f"{API_BASE}/{API_VERSION}/models/{self.settings.gemini_model}:generateContent",
                json_body=body,
            )
            return parse_script_response(
                response.json(),
                mode=mode,
                model=self.settings.gemini_model,
                source_video=str(path),
            )
        finally:
            if cleanup and file_info:
                self.delete(file_info)


# --------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------


def dry_run_script(
    video_duration_ms: int,
    *,
    mode: ScriptMode = "narration",
    max_words: int = 26,
    seed_text: str = "",
) -> Script:
    """Synthesise a plausible script without calling any API.

    This exists so the whole pipeline -- timing maths, ffmpeg mixing, muxing,
    output -- can be exercised end to end with no keys and no spend. Placeholder
    lines are written to a realistic word budget and spread over the video, so
    the fitting logic meets the same kind of disagreement between predicted and
    actual speech length that a real render produces.
    """
    if video_duration_ms <= 0:
        video_duration_ms = 30_000

    sentences = [part.strip() for part in seed_text.split(". ") if part.strip()]
    generated = not sentences
    if generated:
        sentences = [
            "This is a dry run, so no real narration was written for this video.",
            "The pipeline is placing placeholder lines at even intervals instead.",
            "Timing, mixing and muxing are all exercised by these stand-in tones.",
            "Add an API key to replace them with a script Gemini writes for you.",
        ]

    # Roughly how many words fit, matching vidvoice.timing.word_budget so the
    # dry run produces the same pressure a real script would.
    minutes = video_duration_ms / 60_000
    budget = max(8, int(minutes * _DRY_RUN_WPM * 0.75))

    # Choose a segment count by how many comfortable line windows the video
    # holds, not by the word budget: deriving slots from the budget creates a
    # feedback loop (fewer words -> fewer slots -> longer windows -> yet fewer
    # words) that leaves the video mostly silent.
    max_words_per_segment = max(6, max_words)
    slots = max(1, min(12, int(video_duration_ms // _DRY_RUN_CADENCE_MS))) if generated else 1

    cadence_ms = video_duration_ms // slots
    # Leave a breath between lines, but keep most of the cadence usable.
    window_ms = max(1_000, int(cadence_ms * 0.82))

    # Words that fill this window at the assumed pace, which is also what the
    # real budget formula implies -- so the dry run exercises the same fitting
    # pressure a genuine script produces.
    words_per_window = max(4, int(window_ms / 60_000 * _DRY_RUN_WPM))
    target_words = max(4, min(max_words_per_segment, words_per_window, budget))

    segments: list[Segment] = []
    for index in range(slots):
        sentence = sentences[index % len(sentences)]
        if generated:
            # Trim each placeholder to the words its window can actually hold.
            words = sentence.split()
            if len(words) > target_words:
                sentence = " ".join(words[:target_words]).rstrip(",;:") + "."
        if not sentence.endswith("."):
            sentence += "."

        start = index * cadence_ms + 250
        end = min(start + window_ms, video_duration_ms)
        if end - start < 800:
            break

        segments.append(
            Segment(
                index=index,
                start_ms=start,
                end_ms=end,
                text=sentence,
                visual="(dry run -- no video was analysed)",
            )
        )

    return Script(
        segments=segments,
        mode=mode,
        summary="Dry-run placeholder script (no API call was made).",
        source_video="(dry run)",
        model="dry-run",
        meta={"dry_run": True, "video_duration_ms": video_duration_ms},
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _check_upload_size(path: Path, size: int) -> None:
    """Refuse files above the per-file ceiling, before reading any bytes.

    Called from both :meth:`GeminiClient.analyze` and :meth:`GeminiClient.upload`
    because the inline path would otherwise base64-encode a huge file into
    memory before anyone noticed it was too big.
    """
    if size > MAX_UPLOAD_BYTES:
        raise GeminiError(
            f"{path.name} is {size / 1_073_741_824:.2f} GB, over the "
            f"{MAX_UPLOAD_BYTES / 1_073_741_824:.0f} GB per-file limit.",
            hint=(
                "Re-export at a lower bitrate or shorter length. Screen "
                "recordings compress well at CRF 23 / 1080p."
            ),
        )


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return (response.text or "").strip()[:300]
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return _truncate(json.dumps(payload))


def _hint_for_status(status: int) -> str:
    if status == 400:
        return (
            "Gemini rejected the request. The video may be an unsupported codec, or "
            "the response schema contained a keyword Gemini does not support."
        )
    if status == 401 or status == 403:
        return "GEMINI_API_KEY looks invalid, or the Generative Language API is not enabled."
    if status == 404:
        return f"Model '{Settings().gemini_model}' was not found. Set VIDVOICE_GEMINI_MODEL to a current model id."
    if status == 413:
        return "The video is too large for inline upload; the Files API path should have been used."
    if status == 429:
        return "Quota exceeded. Wait, or use a key with a higher tier."
    return "See https://ai.google.dev/gemini-api/docs/troubleshooting"


def list_models(settings: Settings) -> list[str]:
    """Model ids available to this key, for `vidvoice doctor`."""
    with GeminiClient(settings) as client:
        response = client._request("GET", f"{API_BASE}/{API_VERSION}/models", params={"pageSize": 200})
        payload = response.json()
    models = payload.get("models") or []
    return sorted(
        str(model.get("name", "")).removeprefix("models/")
        for model in models
        if isinstance(model, dict)
    )
