"""Domain model: a timed script and its placement plan.

Units: everything time-related is **integer milliseconds**. Floats accumulate
error across dozens of segments and make equality comparisons in tests fragile.
Conversion to seconds happens once, at the ffmpeg boundary (`render.py`).

The one piece of non-obvious design here is the separation between:

* :class:`Segment` -- what the *model* said (intent), and
* :class:`Placement` -- what we can *actually* fit onto the timeline (reality).

Gemini's idea of how long a sentence takes to say is routinely off by 2x
versus Cartesia's actual output. Keeping the two apart means we can report,
log and tune that mismatch instead of silently producing drift.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence


ScriptMode = Literal["narration", "transcribe"]

# --------------------------------------------------------------------------
# Timestamp parsing
# --------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(
    r"^\s*(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[.,](?P<ms>\d{1,3}))?\s*$"
)


def parse_timestamp(value: str | int | float) -> int:
    """Parse a timestamp into integer milliseconds.

    Accepts ``"90"``/``90`` (seconds), ``"MM:SS"``, ``"HH:MM:SS"``, ``"MM:SS.mmm"``
    and ``"HH:MM:SS,mmm"``. Models emit all of these depending on mood, so
    normalising here keeps the rest of the pipeline honest.
    """
    if isinstance(value, bool):  # bool is an int subclass; almost certainly a bug
        raise ValueError(f"Invalid timestamp: {value!r}")
    if isinstance(value, (int, float)):
        return int(round(float(value) * 1000))

    match = _TIMESTAMP_RE.match(str(value))
    if not match:
        # Bare seconds, e.g. "12.5"
        try:
            return int(round(float(str(value).strip()) * 1000))
        except ValueError as exc:
            raise ValueError(f"Unrecognised timestamp format: {value!r}") from exc

    hours = int(match.group("h") or 0)
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    fraction = match.group("ms") or ""
    millis = int(fraction.ljust(3, "0")) if fraction else 0

    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def format_timestamp(ms: int, *, always_hours: bool = False) -> str:
    """Render milliseconds as ``MM:SS.mmm`` (or ``HH:MM:SS.mmm``)."""
    negative = ms < 0
    ms = abs(int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)

    if hours or always_hours:
        text = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    else:
        text = f"{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"-{text}" if negative else text


# --------------------------------------------------------------------------
# Core entities
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Segment:
    """One spoken beat, anchored to a window of the source video."""

    index: int
    start_ms: int
    end_ms: int
    text: str
    #: What is happening on screen; narrated-only, used for review/debugging.
    visual: str = ""
    #: Optional per-segment delivery hint, e.g. "excited", "calm".
    tone: str = ""

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"Segment {self.index}: end_ms ({self.end_ms}) precedes "
                f"start_ms ({self.start_ms})"
            )
        self.text = self.text.strip()

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def is_speakable(self) -> bool:
        """False for segments the model used to mark a silent beat."""
        return bool(self.text) and not self.text.startswith("[")

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        """Serialise with `start`/`end` keys.

        These deliberately match both the schema Gemini is given and
        :meth:`Script.from_dict`, so a saved script round-trips exactly and can
        be hand-edited without surprising key names.
        """
        return {
            "index": self.index,
            "start": format_timestamp(self.start_ms),
            "end": format_timestamp(self.end_ms),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "text": self.text,
            "visual": self.visual,
            "tone": self.tone,
        }


def build_script_schema(*, mode: ScriptMode = "narration") -> dict[str, Any]:
    """JSON Schema handed to Gemini's structured-output mode.

    Two constraints shape this:

    * **Whole seconds.** Gemini's video understanding samples at 1 FPS and
      attaches timestamps per second, so a request for milliseconds would be
      false precision -- the model cannot see finer than a second. Asking for
      ``MM:SS`` matches the documented convention and the model's real
      resolution. (Parsing still accepts fractions, so nothing breaks if a
      model emits them anyway.)
    * **No unsupported keywords.** Gemini's schema subset rejects ``minimum``,
      ``maxLength`` and friends, so ranges are expressed in prose in the prompt
      rather than in the schema. ``propertyOrdering`` is supported and is used
      to keep key order stable and diffable.
    """
    segment_properties: dict[str, Any] = {
        "index": {
            "type": "integer",
            "description": "0-based position of this segment in the timeline.",
        },
        "start": {
            "type": "string",
            "description": (
                "Start time of this beat, formatted exactly as MM:SS with whole "
                "seconds, for example 01:23. No fractional seconds."
            ),
        },
        "end": {
            "type": "string",
            "description": (
                "End time of this beat, formatted exactly as MM:SS with whole "
                "seconds, strictly greater than `start`."
            ),
        },
        "text": {
            "type": "string",
            "description": (
                "The words to be spoken during this window, with no stage "
                "directions, no speaker labels and no markdown."
            ),
        },
    }
    ordering = ["index", "start", "end", "text"]

    if mode == "narration":
        segment_properties["visual"] = {
            "type": "string",
            "description": "Brief description of what is on screen during this beat.",
        }
        ordering.insert(3, "visual")

    return {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "description": "Chronological, non-overlapping narration beats.",
                "items": {
                    "type": "object",
                    "properties": segment_properties,
                    "required": ordering,
                    "propertyOrdering": ordering,
                },
            },
            "summary": {
                "type": "string",
                "description": "One or two sentences describing the whole video.",
            },
        },
        "required": ["segments", "summary"],
        "propertyOrdering": ["summary", "segments"],
    }


@dataclass(slots=True)
class Script:
    """A full timed script plus provenance, for reproducibility."""

    segments: list[Segment]
    mode: ScriptMode = "narration"
    summary: str = ""
    source_video: str = ""
    model: str = ""
    #: Free-form extras from the model or from post-processing.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.segments = sorted(self.segments, key=lambda s: (s.start_ms, s.index))

    # -- derived ------------------------------------------------------------

    @property
    def duration_ms(self) -> int:
        return max((s.end_ms for s in self.segments), default=0)

    @property
    def speakable(self) -> list[Segment]:
        return [s for s in self.segments if s.is_speakable]

    @property
    def total_words(self) -> int:
        return sum(s.word_count for s in self.speakable)

    def coverage(self, video_duration_ms: int) -> float:
        """Fraction of the video covered by speech windows (0.0-1.0)."""
        if video_duration_ms <= 0:
            return 0.0
        covered = sum(
            max(0, min(s.end_ms, video_duration_ms) - max(0, s.start_ms))
            for s in self.speakable
        )
        return min(1.0, covered / video_duration_ms)

    def validate(self, video_duration_ms: int | None = None) -> list[str]:
        """Return a list of human-readable warnings. Never raises."""
        warnings: list[str] = []

        if not self.segments:
            warnings.append("Script contains no segments.")
            return warnings

        if not self.speakable:
            warnings.append("Every segment is silent/marked as non-speech.")

        overlaps = 0
        for previous, current in zip(self.segments, self.segments[1:]):
            if current.start_ms < previous.end_ms:
                overlaps += 1
        if overlaps:
            warnings.append(
                f"{overlaps} segment(s) overlap the previous one; the mixed audio "
                "may talk over itself. Consider merging them."
            )

        zero_length = [s.index for s in self.segments if s.duration_ms == 0]
        if zero_length:
            warnings.append(
                f"{len(zero_length)} segment(s) have zero duration: {zero_length[:5]}"
            )

        empty = [s.index for s in self.segments if not s.text]
        if empty:
            warnings.append(f"{len(empty)} segment(s) have empty text: {empty[:5]}")

        if video_duration_ms:
            overrun = [s.index for s in self.segments if s.end_ms > video_duration_ms]
            if overrun:
                warnings.append(
                    f"{len(overrun)} segment(s) extend past the end of the video "
                    f"({format_timestamp(video_duration_ms)}): {overrun[:5]}"
                )

        words_per_minute = _wpm(self)
        if words_per_minute:
            if words_per_minute > 210:
                warnings.append(
                    f"Dense script ({words_per_minute:.0f} wpm) -- speech will need "
                    "heavy time-compression to fit."
                )
            elif words_per_minute < 60 and len(self.speakable) > 2:
                warnings.append(
                    f"Sparse script ({words_per_minute:.0f} wpm) -- there may be "
                    "long silences between lines."
                )

        return warnings

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "summary": self.summary,
            "source_video": self.source_video,
            "model": self.model,
            "duration_ms": self.duration_ms,
            "meta": self.meta,
            "segments": [s.to_dict() for s in self.segments],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Script:
        raw_segments = data.get("segments")
        if not isinstance(raw_segments, list):
            raise ValueError("Script JSON must contain a 'segments' array")

        segments: list[Segment] = []
        for position, raw in enumerate(raw_segments):
            if not isinstance(raw, dict):
                raise ValueError(f"Segment {position} is not an object")
            try:
                start = parse_timestamp(raw.get("start", 0))
                end = parse_timestamp(raw.get("end", 0))
            except ValueError as exc:
                raise ValueError(f"Segment {position}: {exc}") from exc
            segments.append(
                Segment(
                    index=int(raw.get("index", position)),
                    start_ms=start,
                    end_ms=max(start, end),
                    text=str(raw.get("text", "")),
                    visual=str(raw.get("visual", "")),
                    tone=str(raw.get("tone", "")),
                )
            )

        return cls(
            segments=segments,
            mode=data.get("mode", "narration"),
            summary=str(data.get("summary", "")),
            source_video=str(data.get("source_video", "")),
            model=str(data.get("model", "")),
            meta=dict(data.get("meta") or {}),
        )

    @classmethod
    def load(cls, path: Path | str) -> Script:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _wpm(script: Script) -> float:
    """Words per minute across the script's *spoken windows*.

    This measures script density -- how many words the model asked us to fit
    into the time it allocated -- not the speed of the finished voice. A high
    value here means the script is too dense for its windows, which is a prompt
    or budget problem, not a rendering one. For the delivered rate, use
    :func:`delivered_wpm`.
    """
    minutes = sum(s.duration_ms for s in script.speakable) / 60_000
    if minutes <= 0:
        return 0.0
    return script.total_words / minutes


def delivered_wpm(script: Script, audio_duration_ms: int) -> float:
    """Words per minute actually spoken, against the real length of the audio.

    This is the number that tells you how fast the finished narration sounds.
    Compare it with the target pace: a delivered rate well above target means
    the voice was sped up to fit.
    """
    minutes = audio_duration_ms / 60_000
    if minutes <= 0:
        return 0.0
    return script.total_words / minutes


def merge_short_segments(
    segments: Sequence[Segment], *, min_duration_ms: int = 900
) -> list[Segment]:
    """Fold very short segments into the previous one.

    A 400 ms window is not enough time to say anything, and forcing a whole
    sentence into it produces comically fast speech. Merging keeps delivery
    natural and reduces the number of audio streams we have to mix.
    """
    merged: list[Segment] = []
    for segment in segments:
        if (
            merged
            and segment.duration_ms < min_duration_ms
            and not merged[-1].text.endswith((".", "!", "?"))
        ):
            previous = merged[-1]
            joiner = " " if previous.text and segment.text else ""
            merged[-1] = Segment(
                index=previous.index,
                start_ms=previous.start_ms,
                end_ms=max(previous.end_ms, segment.end_ms),
                text=f"{previous.text}{joiner}{segment.text}".strip(),
                visual=previous.visual or segment.visual,
                tone=previous.tone or segment.tone,
            )
        else:
            merged.append(segment)

    return [
        Segment(
            index=index,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            text=segment.text,
            visual=segment.visual,
            tone=segment.tone,
        )
        for index, segment in enumerate(merged)
    ]


def script_stats(
    script: Script, video_duration_ms: int = 0, audio_duration_ms: int = 0
) -> dict[str, Any]:
    """Summary numbers for logs, the GUI, and the report sidecar.

    ``words_per_minute`` is script density against the allocated windows.
    Pass ``audio_duration_ms`` to also get ``delivered_wpm``, the speed the
    finished narration actually plays at.
    """
    durations = [s.duration_ms for s in script.speakable]
    stats: dict[str, Any] = {
        "segments": len(script.segments),
        "speakable_segments": len(script.speakable),
        "total_words": script.total_words,
        "script_span_ms": script.duration_ms,
        "speech_window_ms": sum(durations),
        "median_window_ms": int(statistics.median(durations)) if durations else 0,
        "words_per_minute": round(_wpm(script), 1),
        "coverage": round(script.coverage(video_duration_ms), 3) if video_duration_ms else 0.0,
    }
    if audio_duration_ms:
        stats["delivered_wpm"] = round(delivered_wpm(script, audio_duration_ms), 1)
    return stats


# --------------------------------------------------------------------------
# Placement: intent -> reality
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Placement:
    """Where one synthesised clip actually lands on the timeline."""

    index: int
    #: The window the model asked for.
    target_start_ms: int
    target_end_ms: int
    #: Where we actually place it (>= target unless shifted).
    start_ms: int
    #: Length of the generated speech, before any speed change.
    natural_ms: int
    #: Playback rate applied to make it fit (1.0 = untouched).
    speed: float
    #: Resulting audible length after speed change, before padding.
    fitted_ms: int
    #: True when the clip had to be sped up to fit.
    compressed: bool
    #: True when the clip could not fit even at max speed and overhangs.
    overflowed: bool
    text: str = ""

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.fitted_ms

    @property
    def gap_after_ms(self) -> int:
        return max(0, self.target_end_ms - self.end_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "target_start": format_timestamp(self.target_start_ms),
            "target_end": format_timestamp(self.target_end_ms),
            "start": format_timestamp(self.start_ms),
            "end": format_timestamp(self.end_ms),
        }


@dataclass(slots=True)
class TimelinePlan:
    """The full set of placements for a video, plus how healthy it is."""

    placements: list[Placement]
    video_duration_ms: int
    #: Seconds we had to stretch/squeeze, as a share of total speech time.
    mean_abs_speed_delta: float = 0.0

    @property
    def overflow_count(self) -> int:
        return sum(1 for p in self.placements if p.overflowed)

    @property
    def compressed_count(self) -> int:
        return sum(1 for p in self.placements if p.compressed)

    @property
    def ends_at_ms(self) -> int:
        return max((p.end_ms for p in self.placements), default=0)

    def warnings(self) -> list[str]:
        notes: list[str] = []
        if self.overflow_count:
            worst = max(
                (p for p in self.placements if p.overflowed),
                key=lambda p: p.end_ms - p.target_end_ms,
            )
            notes.append(
                f"{self.overflow_count} segment(s) could not fit and will run long "
                f"(worst: +{format_timestamp(worst.end_ms - worst.target_end_ms)} "
                f"on segment {worst.index})."
            )
        if self.compressed_count:
            notes.append(
                f"{self.compressed_count} segment(s) were sped up to fit "
                f"(mean rate change {self.mean_abs_speed_delta:.0%})."
            )
        if self.video_duration_ms and self.ends_at_ms > self.video_duration_ms:
            notes.append(
                f"Speech ends at {format_timestamp(self.ends_at_ms)}, "
                f"{format_timestamp(self.ends_at_ms - self.video_duration_ms)} past "
                "the end of the video. Use --tail pad to extend the video, or "
                "--tail trim to cut the overflow."
            )
        return notes


def plan_timeline(
    script: Script,
    durations_ms: dict[int, int],
    *,
    video_duration_ms: int,
    max_speed: float = 1.35,
    min_speed: float = 0.9,
    shift_on_overflow: bool = True,
    protect_gap_ms: int = 120,
    tolerance_ms: int = 150,
) -> TimelinePlan:
    """Decide how each synthesised clip lands on the timeline.

    The core problem: Gemini thinks a sentence fits in 4 s; Cartesia may take
    6 s to say it. Something has to give. Per segment, in order of preference:

    1. **It fits** -- leave it alone (within ``tolerance_ms``).
    2. **Speed it up** to fit, up to ``max_speed``. Mild compression is far
       less noticeable than clipping words off the end, and up to ~1.35x is
       essentially inaudible for narration.
    3. **Speed it down** if it is much shorter than the window and there is
       room, so short lines do not sound rushed. Never below ``min_speed``.
    4. **Push the start later** so it does not collide with the previous line,
       and let it overrun its own window.
    5. **Overflow** -- accept the overhang and flag it loudly.

    Two invariants this function guarantees:

    * **No two clips overlap.** Speech that talks over itself is unusable, so a
      segment whose predecessor ran long always starts after it finishes (plus
      ``protect_gap_ms``), even if that means starting much later than planned.
      This is what ``shift_on_overflow`` controls; disabling it is only sensible
      when you have separately verified that the windows are wide enough.
    * **Placements never move backwards.** The returned list is chronological in
      both target and actual time.

    Args:
        durations_ms: segment index -> natural speech length in ms.
        video_duration_ms: used for reporting, and as a soft ceiling.
        protect_gap_ms: minimum silence to keep between consecutive lines.
    """
    if not script.segments:
        return TimelinePlan(placements=[], video_duration_ms=video_duration_ms)

    placements: list[Placement] = []
    speed_deltas: list[float] = []

    for segment in script.segments:
        natural = durations_ms.get(segment.index, 0)
        if natural <= 0:
            # Nothing to place (silent marker or failed synthesis).
            placements.append(
                Placement(
                    index=segment.index,
                    target_start_ms=segment.start_ms,
                    target_end_ms=segment.end_ms,
                    start_ms=segment.start_ms,
                    natural_ms=0,
                    speed=1.0,
                    fitted_ms=0,
                    compressed=False,
                    overflowed=False,
                    text=segment.text,
                )
            )
            continue

        # Earliest this clip may start without colliding with the previous one.
        earliest = 0
        if placements and placements[-1].fitted_ms > 0:
            earliest = placements[-1].end_ms + max(0, protect_gap_ms)

        start = segment.start_ms
        if shift_on_overflow and start < earliest:
            start = earliest

        # Width of the window this clip must fit into, measured from where it
        # actually starts so a shifted clip must still fit in the room it has.
        available = max(0, segment.end_ms - start)

        if available <= 0:
            # No room left: play it at natural pace and let it overrun.
            speed = 1.0
        elif natural <= available + tolerance_ms:
            # Fits, or close enough. Gently slow down if there is lots of room.
            if natural > 0 and available > natural * 1.6:
                speed = max(min_speed, natural / available)
            else:
                speed = 1.0
        else:
            speed = min(max_speed, natural / available)

        fitted = int(round(natural / speed)) if speed > 0 else natural
        compressed = speed > 1.0 + 1e-9
        overflowed = fitted > available + tolerance_ms
        speed_deltas.append(abs(speed - 1.0))

        placements.append(
            Placement(
                index=segment.index,
                target_start_ms=segment.start_ms,
                target_end_ms=segment.end_ms,
                start_ms=start,
                natural_ms=natural,
                speed=round(speed, 4),
                fitted_ms=fitted,
                compressed=compressed,
                overflowed=overflowed,
                text=segment.text,
            )
        )

    mean_delta = sum(speed_deltas) / len(speed_deltas) if speed_deltas else 0.0
    return TimelinePlan(
        placements=placements,
        video_duration_ms=video_duration_ms,
        mean_abs_speed_delta=mean_delta,
    )


def iter_speakable(script: Script) -> Iterable[Segment]:
    return (segment for segment in script.segments if segment.is_speakable)
