"""Cartesia text-to-speech client (raw HTTP, no SDK dependency).

Why hand-rolled rather than the official SDK: this pipeline needs exactly two
calls, and keeping them as plain HTTP means the request/response contract is
visible in one file instead of hidden behind a library version. Swapping to the
SDK later is a small, contained change.

API notes (verified against the raw OpenAPI spec served at
``docs.cartesia.ai/api-reference/tts/bytes.md``, 2026-08-14):

* ``Cartesia-Version`` is required and the spec's enum accepts exactly one
  value. We pin it explicitly so a server-side default change cannot silently
  alter our output format.
* Auth: ``Authorization: Bearer`` is the documented scheme for this version
  (``X-Api-Key`` still works but is the older form).
* ``POST /tts/bytes`` streams raw audio in whatever container we ask for. We
  request WAV so downstream ffmpeg work needs no decoding.
* The voice specifier is a bare id string or ``{"id": ...}`` -- the older
  ``{"mode": "id", ...}`` form and inline embeddings were both removed.
* **``/tts/bytes`` cannot return word timestamps.** Timestamps exist only on
  ``/tts/sse`` and the WebSocket, where they arrive as columnar arrays. That is
  exactly why this pipeline synthesises *per segment* and derives timing from
  real audio durations instead of trusting the model's estimates.
* Errors are structured JSON: ``{error_code, title, message, request_id}``.
  There is no per-request character limit documented, so :func:`chunk_text`
  exists as a defensive measure, not because a specific ceiling is known.

Voice clips are cached on disk by a hash of (text, voice, model, speed), so
re-running a render after tweaking one line does not re-bill every other line.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx

from .config import SPEED_RANGE, VOLUME_RANGE, Settings
from .errors import CartesiaError
from .models import Segment

log = logging.getLogger(__name__)

BASE_URL = "https://api.cartesia.ai"

#: WAV/PCM is the safest intermediate: no lossy generation, no decode step.
#: The spec is self-contradictory about whether encoding/sample_rate are
#: required (they have defaults but inherit `required` via allOf), so we always
#: send both explicitly.
OUTPUT_CONTAINER = "wav"
OUTPUT_ENCODING = "pcm_s16le"

#: Valid sample rates per spec; anything else is a 400.
VALID_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)

#: Statuses worth retrying: rate limit + transient server faults.
#: 429 on this API means the plan's *concurrency* limit was hit, not a quota.
_RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 4

#: Conservative chunk size for very long text. No documented limit exists; this
#: simply keeps any single request small enough to be unremarkable.
SAFE_CHUNK_CHARS = 4_000


@dataclass(slots=True)
class Voice:
    """A Cartesia voice available for synthesis.

    Field names track the 2026-08-14 Voice object, where ``is_public`` was
    removed in favour of ``access`` (string enum) plus ``visibility``.
    """

    id: str
    name: str = ""
    description: str = ""
    language: str = ""
    gender: str = ""
    #: "private" | "public"
    access: str = ""
    #: "owner" | "all"
    visibility: str = ""
    is_owner: bool = False
    is_pro: bool = False
    status: str = ""
    tagline: str = ""

    @property
    def label(self) -> str:
        bits = [self.name or self.id]
        if self.gender:
            bits.append(self.gender)
        if self.language:
            bits.append(self.language)
        if self.is_pro:
            bits.append("pro")
        return " - ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "gender": self.gender,
            "access": self.access,
            "visibility": self.visibility,
            "is_owner": self.is_owner,
            "is_pro": self.is_pro,
            "status": self.status,
        }

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Voice:
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name") or ""),
            description=str(data.get("description") or data.get("tagline") or ""),
            language=str(data.get("language") or ""),
            gender=str(data.get("gender") or ""),
            access=str(data.get("access") or ""),
            visibility=str(data.get("visibility") or ""),
            is_owner=bool(data.get("is_owner", False)),
            is_pro=bool(data.get("is_pro", False)),
            status=str(data.get("status") or ""),
            tagline=str(data.get("tagline") or ""),
        )


class CartesiaClient:
    """Minimal, resilient Cartesia TTS client."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
        cache_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)
        self.cache_dir = cache_dir

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> CartesiaClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- request plumbing ---------------------------------------------------

    @property
    def _headers(self) -> dict[str, str]:
        return {
            # `Authorization: Bearer` is the scheme declared by the 2026-03-01+
            # specs; X-Api-Key is the older documented form.
            "Authorization": f"Bearer {self.settings.require_cartesia_key()}",
            "Cartesia-Version": self.settings.cartesia_version,
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Issue a request with bounded exponential backoff on transient faults."""
        url = f"{BASE_URL}{path}"
        last_error: Exception | None = None
        last_hint: str | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.request(
                    method,
                    url,
                    headers=self._headers,
                    json=json_body,
                    params=params,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                log.warning("Cartesia %s %s failed (%s), attempt %d", method, path, exc, attempt)
            else:
                if response.status_code < 400:
                    return response

                detail = _error_detail(response)
                if response.status_code not in _RETRY_STATUSES:
                    raise CartesiaError(
                        f"Cartesia returned {response.status_code}: {detail}",
                        hint=_hint_for_status(response.status_code, detail),
                        status_code=response.status_code,
                        body=detail,
                    )
                # Keep the specific reason for the final error message; if we
                # run out of retries, "concurrency limit" beats "check network".
                last_error = CartesiaError(
                    f"Cartesia returned {response.status_code}: {detail}",
                    status_code=response.status_code,
                )
                last_hint = _hint_for_status(response.status_code, detail)
                # Honour Retry-After when the server tells us how long to wait.
                retry_after = response.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(float(retry_after), 30.0))
                    continue

            if attempt < _MAX_ATTEMPTS:
                time.sleep(min(2.0 ** (attempt - 1), 8.0))

        raise CartesiaError(
            f"Cartesia request to {path} failed after {_MAX_ATTEMPTS} attempts: {last_error}",
            hint=last_hint
            or "Check your network connection and that CARTESIA_API_KEY is valid.",
        )

    # -- API ----------------------------------------------------------------

    def list_voices(
        self,
        *,
        limit: int = 100,
        language: str | None = None,
        search: str | None = None,
        gender: str | None = None,
        owned_only: bool = False,
    ) -> list[Voice]:
        """Fetch available voices, paging until exhausted or *limit* is reached.

        Filtering is pushed to the API where the spec supports it (`q`, `gender`,
        `language`, `is_owner`), because server-side filtering keeps paging
        correct -- filtering a single page locally can silently drop matches
        that live on later pages.
        """
        voices: list[Voice] = []
        starting_after: str | None = None
        page_size = min(max(limit, 1), 100)

        while len(voices) < limit:
            params: dict[str, Any] = {"limit": page_size}
            if starting_after:
                params["starting_after"] = starting_after
            if search:
                params["q"] = search
            if gender:
                params["gender"] = gender
            if language:
                params["language"] = language
            if owned_only:
                params["is_owner"] = "true"

            response = self._request("GET", "/voices", params=params)
            payload = _safe_json(response)
            items = payload if isinstance(payload, list) else payload.get("data") or []

            if not items:
                break

            for item in items:
                if isinstance(item, dict):
                    voices.append(Voice.from_api(item))

            if not isinstance(payload, dict) or not payload.get("has_more"):
                break
            # Prefer the server's own cursor; fall back to the last id.
            starting_after = str(payload.get("next_page") or items[-1].get("id", ""))
            if not starting_after:
                break

        return voices[:limit]

    def synthesize(
        self,
        text: str,
        out_path: Path | str,
        *,
        voice_id: str | None = None,
        speed: float | None = None,
        volume: float | None = None,
        emotion: str | None = None,
        language: str | None = None,
        locale: str | None = None,
    ) -> Path:
        """Render *text* to a WAV file at *out_path*, using the cache when possible."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        text = text.strip()
        if not text:
            raise CartesiaError("Cannot synthesise empty text.")
        if len(text) > SAFE_CHUNK_CHARS:
            raise CartesiaError(
                f"Text is {len(text)} characters, above the {SAFE_CHUNK_CHARS}-char "
                "safety chunk size.",
                hint="Split this segment with vidvoice.cartesia.chunk_text().",
            )
        if language and locale:
            # The API explicitly rejects both, even when they are identical.
            raise CartesiaError(
                "Set either `language` or `locale`, never both (the API returns 400)."
            )

        voice_id = voice_id or self.settings.cartesia_voice_id
        speed = self.settings.speech_speed if speed is None else speed
        volume = self.settings.speech_volume if volume is None else volume

        speed = _clamp("speed", speed, SPEED_RANGE)
        volume = _clamp("volume", volume, VOLUME_RANGE)

        sample_rate = self.settings.sample_rate
        if sample_rate not in VALID_SAMPLE_RATES:
            raise CartesiaError(
                f"Unsupported sample rate {sample_rate}.",
                hint=f"Cartesia accepts one of {list(VALID_SAMPLE_RATES)}.",
            )

        generation_config: dict[str, Any] = {"speed": speed, "volume": volume}
        if emotion:
            generation_config["emotion"] = emotion

        body: dict[str, Any] = {
            "model_id": self.settings.cartesia_model,
            "transcript": text,
            # Current spec: a bare id string, or an object with a required `id`.
            # The old {"mode": "id", ...} form is gone in this API version.
            "voice": {"id": voice_id},
            "output_format": {
                "container": OUTPUT_CONTAINER,
                "sample_rate": sample_rate,
                "encoding": OUTPUT_ENCODING,
            },
            "generation_config": generation_config,
        }
        # `language` and `locale` take the same values; send at most one.
        if locale:
            body["locale"] = locale
        elif language:
            body["language"] = language

        key = self._cache_key(body)
        cached = self._cache_path(key)
        if cached and cached.is_file() and cached.stat().st_size > 0:
            log.debug("Cartesia cache hit for %s", key[:12])
            _copy(cached, out_path)
            return out_path

        response = self._request("POST", "/tts/bytes", json_body=body)
        audio = response.content
        if not audio:
            raise CartesiaError(
                "Cartesia returned an empty audio body.",
                hint="The text may have been rejected as unpronounceable.",
            )

        out_path.write_bytes(audio)
        if cached:
            try:
                _copy(out_path, cached)
            except OSError as exc:  # cache is best-effort, never fatal
                log.debug("Could not write Cartesia cache entry: %s", exc)
        return out_path

    def synthesize_segments(
        self,
        segments: Sequence[Segment],
        work_dir: Path,
        *,
        voice_id: str | None = None,
        on_progress: Any = None,
    ) -> dict[int, Path]:
        """Render every speakable segment to ``work_dir/seg_NNNN.wav``.

        Returns ``{segment_index: path}``. Segments with no text are skipped
        rather than sent as empty requests.
        """
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        speakable = [s for s in segments if s.is_speakable]
        outputs: dict[int, Path] = {}

        for position, segment in enumerate(speakable, start=1):
            out_path = work_dir / f"seg_{segment.index:04d}.wav"
            self.synthesize(
                segment.text,
                out_path,
                voice_id=voice_id,
                emotion=segment.tone or None,
            )
            outputs[segment.index] = out_path
            if on_progress:
                on_progress(position, len(speakable), segment)

        return outputs

    # -- caching ------------------------------------------------------------

    def _cache_key(self, body: dict[str, Any]) -> str:
        payload = json.dumps(body, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return Path(self.cache_dir) / key[:2] / f"{key}.wav"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _error_detail(response: httpx.Response) -> str:
    """Extract a readable message from Cartesia's structured error body.

    Shape (version >= 2026-03-01)::

        {"error_code": "...", "title": "...", "message": "...", "request_id": "..."}

    Older versions and invalid version headers return plain text ``Title: Message``.
    """
    payload = _safe_json(response)
    if isinstance(payload, dict):
        message = payload.get("message")
        title = payload.get("title")
        code = payload.get("error_code")
        parts = [str(p) for p in (title, message) if p]
        detail = ": ".join(parts) if parts else ""
        if code:
            detail = f"{detail} [{code}]" if detail else f"[{code}]"
        if detail:
            return detail
    text = (response.text or "").strip()
    return text[:300] if text else "(no detail)"


def _clamp(name: str, value: float, bounds: tuple[float, float]) -> float:
    """Clamp to the API's documented range, warning when we actually change a value."""
    low, high = bounds
    if value < low:
        log.warning("%s=%.3f is below the supported minimum %.2f; clamping.", name, value, low)
        return low
    if value > high:
        log.warning("%s=%.3f is above the supported maximum %.2f; clamping.", name, value, high)
        return high
    return value


def chunk_text(text: str, *, limit: int = SAFE_CHUNK_CHARS) -> list[str]:
    """Split *text* into sentence-aligned chunks of at most *limit* characters.

    Cartesia publishes no per-request character limit, so this is insurance
    rather than a workaround for a known ceiling. Splitting on sentence
    boundaries keeps each chunk pronounceable as a standalone utterance.
    """
    import re

    if len(text) <= limit:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        # A single sentence longer than the limit gets hard-wrapped.
        while len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:limit])
            sentence = sentence[limit:]

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError):
        return {}


def _hint_for_status(status: int, body: str = "") -> str:
    if status == 401:
        return "CARTESIA_API_KEY looks invalid or expired. Check .env."
    if status == 403:
        return (
            "This key lacks access to the requested model or voice. Note that "
            "admin keys (sk_car_admin_...) are rejected on the TTS endpoint."
        )
    if status == 404:
        return "Check the voice id, and that Cartesia-Version matches a live API version."
    if status == 422 or status == 400:
        if "concurrency" in body.lower():
            return "Concurrency limit reached. Retry shortly, or reduce parallelism."
        return (
            "Cartesia rejected the request body. Common causes: an unsupported "
            "sample_rate (allowed: 8000/16000/22050/24000/44100/48000), an invalid "
            "voice id, or both `language` and `locale` set."
        )
    if status == 429:
        return (
            "Concurrency limit reached (each HTTP request holds one slot). "
            "Lower --concurrency or wait and retry."
        )
    return "See https://docs.cartesia.ai for API details."


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def probe_voice(settings: Settings, voice_id: str) -> Voice | None:
    """Look up a single voice by id, or None if the key/voice is unavailable."""
    with CartesiaClient(settings) as client:
        try:
            voices = client.list_voices(limit=100)
        except CartesiaError:
            return None
    for voice in voices:
        if voice.id == voice_id:
            return voice
    return None


def summarize_voices(voices: Iterable[Voice], *, limit: int = 20) -> str:
    """Human-readable list for CLI output."""
    rows = [f"{'id':<40} {'name':<24} {'gender':<16} lang"]
    rows.append("-" * 92)
    for voice in list(voices)[:limit]:
        rows.append(
            f"{voice.id:<40} {voice.name[:24]:<24} {voice.gender[:16]:<16} {voice.language}"
        )
    return "\n".join(rows)
