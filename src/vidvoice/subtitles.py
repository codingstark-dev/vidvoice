"""Subtitle and transcript export.

Produces three artefacts from the same timed script:

* **SRT** -- the subtitle format every editor and player accepts. Cue times come
  from the *placed* timeline when it is available, so the subtitles line up with
  the audio you actually hear rather than the window the model asked for.
* **VTT** -- same timings, WebVTT syntax, for browsers and web players.
* **TXT** -- a readable transcript. Plain by default, with timestamps on request.

All three are written by default alongside the rendered video, because the
transcript is usually wanted separately from the video (chapters, blog posts,
accessibility review) and costs nothing extra to emit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .models import Placement, Script, Segment, format_timestamp

#: Subtitle lines are conventionally wrapped at ~42 characters for readability.
DEFAULT_LINE_WIDTH = 42


def _cue_time(ms: int, *, separator: str = ",") -> str:
    """``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (VTT)."""
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def wrap_text(text: str, *, width: int = DEFAULT_LINE_WIDTH, max_lines: int = 2) -> str:
    """Wrap *text* for display, keeping at most *max_lines* lines."""
    words = text.split()
    if not words:
        return ""

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
            if len(lines) >= max_lines:
                break

    if current and len(lines) < max_lines:
        lines.append(current)

    if len(lines) == max_lines and len(" ".join(lines)) < len(text):
        # Signal that the cue continues rather than silently dropping words.
        lines[-1] = f"{lines[-1].rstrip()}..."

    return "\n".join(lines)


def _cues_from_script(
    script: Script,
    placements: Sequence[Placement] | None,
) -> list[tuple[int, int, str]]:
    """Build (start_ms, end_ms, text) cues, preferring placed times."""
    by_index = {p.index: p for p in (placements or []) if p.natural_ms > 0}
    cues: list[tuple[int, int, str]] = []

    for segment in script.speakable:
        placement = by_index.get(segment.index)
        if placement is not None:
            start, end = placement.start_ms, placement.end_ms
        else:
            start, end = segment.start_ms, segment.end_ms

        if end <= start:
            # A zero-length cue is invalid in every subtitle format; give it a
            # minimal readable duration rather than emitting a broken entry.
            end = start + 800
        cues.append((start, end, segment.text))

    return cues


def _repair_overlaps(cues: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Trim each cue so it ends before the next one begins.

    Overlapping cues are legal in some players and render as stacked garbage in
    others. Since this export is meant to be handed to other tools, we make it
    unambiguous. Genuine overlaps (two lines spoken at once) keep the later
    start and clip the earlier end by a frame-ish margin.
    """
    ordered = sorted(cues, key=lambda cue: (cue[0], cue[1]))
    fixed: list[tuple[int, int, str]] = []

    for position, (start, end, text) in enumerate(ordered):
        if position + 1 < len(ordered):
            next_start = ordered[position + 1][0]
            if end > next_start:
                end = max(start + 200, next_start - 40)
        fixed.append((start, end, text))

    return fixed


def to_srt(
    script: Script,
    *,
    placements: Sequence[Placement] | None = None,
    wrap: bool = True,
    line_width: int = DEFAULT_LINE_WIDTH,
) -> str:
    """Render the script as SRT."""
    cues = _repair_overlaps(_cues_from_script(script, placements))

    blocks: list[str] = []
    for number, (start, end, text) in enumerate(cues, start=1):
        body = wrap_text(text, width=line_width) if wrap else text
        blocks.append(
            f"{number}\n"
            f"{_cue_time(start)} --> {_cue_time(end)}\n"
            f"{body}\n"
        )
    return "\n".join(blocks)


def to_vtt(
    script: Script,
    *,
    placements: Sequence[Placement] | None = None,
    wrap: bool = True,
    line_width: int = DEFAULT_LINE_WIDTH,
) -> str:
    """Render the script as WebVTT."""
    cues = _repair_overlaps(_cues_from_script(script, placements))

    blocks: list[str] = ["WEBVTT", ""]
    for number, (start, end, text) in enumerate(cues, start=1):
        body = wrap_text(text, width=line_width) if wrap else text
        blocks.append(
            f"{number}\n"
            f"{_cue_time(start, separator='.')} --> {_cue_time(end, separator='.')}\n"
            f"{body}\n"
        )
    return "\n".join(blocks)


def to_txt(
    script: Script,
    *,
    placements: Sequence[Placement] | None = None,
    timestamps: bool = False,
    include_visuals: bool = False,
) -> str:
    """Render a plain-text transcript, optionally with timestamps."""
    by_index = {p.index: p for p in (placements or []) if p.natural_ms > 0}
    lines: list[str] = []

    if script.summary:
        lines.append(script.summary)
        lines.append("")

    for segment in script.speakable:
        placement = by_index.get(segment.index)
        start = placement.start_ms if placement else segment.start_ms

        if timestamps:
            lines.append(f"[{format_timestamp(start)}] {segment.text}")
        else:
            lines.append(segment.text)

        if include_visuals and segment.visual:
            lines.append(f"    ({segment.visual})")

    return "\n".join(lines).strip() + "\n"


def write_subtitles(
    script: Script,
    out_base: Path | str,
    *,
    placements: Sequence[Placement] | None = None,
    formats: Iterable[str] = ("srt", "txt", "vtt"),
    timestamps_in_txt: bool = False,
) -> dict[str, Path]:
    """Write every requested format next to *out_base*.

    ``out_base`` is used with the extension replaced, so ``demo_narrated.mp4``
    produces ``demo_narrated.srt`` and so on. Returns ``{format: path}``.
    """
    base = Path(out_base)
    base = base.with_suffix("") if base.suffix else base
    written: dict[str, Path] = {}

    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        if fmt == "srt":
            content = to_srt(script, placements=placements)
        elif fmt == "vtt":
            content = to_vtt(script, placements=placements)
        elif fmt == "txt":
            content = to_txt(script, placements=placements, timestamps=timestamps_in_txt)
        else:
            continue

        path = base.with_suffix(f".{fmt}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written[fmt] = path

    return written


def parse_srt(text: str) -> list[Segment]:
    """Parse SRT back into segments -- used by `vidvoice render --script-file`."""
    blocks = [b for b in text.replace("\r\n", "\n").split("\n\n") if b.strip()]
    segments: list[Segment] = []

    for position, block in enumerate(blocks):
        lines = [line for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue

        # The timing line may or may not be preceded by an index line.
        timing_at = 0
        if "-->" not in lines[0] and len(lines) > 1:
            timing_at = 1
        if timing_at >= len(lines):
            continue

        timing = lines[timing_at]
        if "-->" not in timing:
            continue

        start_text, _, end_text = timing.partition("-->")
        try:
            start = _parse_cue_time(start_text.strip())
            end = _parse_cue_time(end_text.strip())
        except ValueError:
            continue

        body = " ".join(lines[timing_at + 1 :]).strip()
        if not body:
            continue

        segments.append(
            Segment(index=position, start_ms=start, end_ms=max(start, end), text=body)
        )

    return segments


def _parse_cue_time(value: str) -> int:
    """``HH:MM:SS,mmm`` -> milliseconds."""
    from .models import parse_timestamp

    cleaned = value.replace(",", ".")
    return parse_timestamp(cleaned)
