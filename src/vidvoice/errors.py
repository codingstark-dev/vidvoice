"""Typed, validated errors.

Everything the pipeline raises deliberately derives from :class:`VidvoiceError`
so the CLI and web layer can turn failures into clean messages instead of
tracebacks, while genuine programming bugs still surface as real tracebacks.
"""

from __future__ import annotations

from typing import Any


class VidvoiceError(Exception):
    """Base class for all expected, user-facing failures."""

    #: Short machine-readable tag, useful in JSON error payloads.
    code = "error"

    def __init__(self, message: str, *, hint: str | None = None, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        #: Optional actionable next step shown to the user.
        self.hint = hint
        #: Arbitrary structured detail (status codes, paths, response bodies).
        self.context = context

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        if self.context:
            payload["context"] = self.context
        return payload


class ConfigError(VidvoiceError):
    """Missing or malformed configuration (usually a bad .env)."""

    code = "config_error"


class MediaError(VidvoiceError):
    """ffmpeg/ffprobe failed, or the input is not usable media."""

    code = "media_error"


class GeminiError(VidvoiceError):
    """The Gemini call failed or returned something unusable."""

    code = "gemini_error"


class CartesiaError(VidvoiceError):
    """The Cartesia call failed."""

    code = "cartesia_error"


class AlignmentError(VidvoiceError):
    """Timed audio could not be fitted onto the video timeline."""

    code = "alignment_error"
