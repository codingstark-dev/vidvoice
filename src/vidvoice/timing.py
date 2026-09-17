"""Voice performance budgeting and length fitting.

This module answers the question the whole tool lives or dies on: *how do we
make the narration match the video?*

There are three independent levers, and they are best applied in this order:

1. **Budget the script** (:func:`word_budget`). Tell Gemini how many words the
   video can hold *before* it writes, so the script arrives approximately right.
   This is by far the cheapest fix -- no audio is regenerated, nothing is
   time-stretched.

2. **Global rate** (:func:`fit_to_length`). Cartesia's ``speed`` parameter
   re-renders the voice at a different pace. Because the whole script is spoken
   at one consistent rate, this sounds entirely natural -- unlike per-line
   stretching, which makes individual sentences lurch. Cartesia accepts
   0.6-1.5, so this covers +-50%.

3. **Per-segment stretch** (see :func:`vidvoice.models.plan_timeline`). The last
   resort, for individual lines that still do not fit their own window. Applied
   to as few segments as possible.

Applying (2) *before* synthesis is what makes (3) rare in practice, which is why
the pipeline plans the global rate up front rather than fixing it after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import SPEED_RANGE
from .models import Script, TimelinePlan, format_timestamp, plan_timeline

#: Fraction of the video we expect to be filled with speech. Narration with no
#: gaps sounds relentless; 0.75 leaves a breath between lines.
DEFAULT_SPEECH_RATIO = 0.75

#: Never speed a script up beyond this to make it fit. Past ~1.4x a voice starts
#: to sound like a disclaimer read at the end of a radio advert.
NATURAL_SPEED_CEILING = 1.4
#: Never slow below this; the voice starts to sound sedated.
NATURAL_SPEED_FLOOR = 0.9


def word_budget(
    video_duration_ms: int,
    *,
    wpm: float = 150.0,
    speech_ratio: float = DEFAULT_SPEECH_RATIO,
) -> int:
    """How many words fit comfortably in *video_duration_ms*.

    At the defaults, a 60 s video yields ~112 words -- roughly 26 seconds of
    speech spread across the minute.
    """
    if video_duration_ms <= 0 or wpm <= 0:
        return 0
    minutes = video_duration_ms / 60_000
    return max(10, int(round(minutes * wpm * max(0.05, min(1.0, speech_ratio)))))


def words_per_segment(word_budget_total: int, segment_count: int) -> int:
    """Split a word budget across segments, clamped to a sane range."""
    if segment_count <= 0:
        return 0
    return max(6, min(45, word_budget_total // segment_count))


@dataclass(slots=True)
class FitReport:
    """The outcome of :func:`fit_to_length`."""

    speed: float
    end_ms: int
    fits: bool
    #: True when we hit the API's speed ceiling and the script still runs long.
    clipped_by_limit: bool
    #: Per-segment stretch needed *after* the global rate; populated when the
    #: caller supplies durations.
    plan: TimelinePlan | None = None

    def describe(self, video_duration_ms: int) -> str:
        if self.fits:
            return (
                f"fits the video at {self.speed:.2f}x "
                f"(speech ends {format_timestamp(self.end_ms)} of "
                f"{format_timestamp(video_duration_ms)})"
            )
        return (
            f"runs {format_timestamp(self.end_ms - video_duration_ms)} past the video "
            f"even at {self.speed:.2f}x"
        )


def fit_to_length(
    script: Script,
    video_duration_ms: int,
    durations_ms: dict[int, int] | None,
    *,
    min_speed: float = NATURAL_SPEED_FLOOR,
    max_speed: float = NATURAL_SPEED_CEILING,
    max_segment_speed: float = 1.35,
    protect_gap_ms: int = 120,
) -> FitReport:
    """Find the global speaking rate that lands the narration inside the video.

    ``durations_ms`` maps segment index -> natural speech length at 1.0x. It may
    be ``None`` (before synthesis), in which case we fall back to a text-based
    estimate so the prompt can still be budgeted sensibly.

    The rate is found by bisection on a monotonic predicate: a higher speed
    always produces a shorter timeline, so there is exactly one crossover point.
    """
    if not script.speakable or video_duration_ms <= 0:
        return FitReport(speed=1.0, end_ms=0, fits=True, clipped_by_limit=False)

    estimates = durations_ms or estimate_durations(script)

    def end_at(speed: float) -> int:
        plan = plan_timeline(
            script,
            {index: int(round(value / speed)) for index, value in estimates.items()},
            video_duration_ms=video_duration_ms,
            # The global rate already did the work; keep per-segment stretch
            # gentle so we do not compound two aggressive transforms.
            max_speed=max_segment_speed,
            min_speed=1.0,
            protect_gap_ms=protect_gap_ms,
        )
        return plan.ends_at_ms

    # Already fits at natural pace: consider slowing down to fill the video.
    if end_at(1.0) <= video_duration_ms:
        slowest = max(min_speed, SPEED_RANGE[0])
        if end_at(slowest) <= video_duration_ms:
            return _report(script, estimates, video_duration_ms, slowest, max_segment_speed, protect_gap_ms, False)
        # Bisect between `slowest` (too long) and 1.0 (fits) for the slowest
        # rate that still fits -- the longest, most relaxed delivery possible.
        low, high = slowest, 1.0
        for _ in range(24):
            mid = (low + high) / 2
            if end_at(mid) <= video_duration_ms:
                low = mid
            else:
                high = mid
        return _report(script, estimates, video_duration_ms, low, max_segment_speed, protect_gap_ms, False)

    # Too long at natural pace: speed up, up to the allowed ceiling.
    ceiling = min(max_speed, SPEED_RANGE[1])
    if end_at(ceiling) > video_duration_ms:
        return _report(script, estimates, video_duration_ms, ceiling, max_segment_speed, protect_gap_ms, True)

    low, high = 1.0, ceiling
    for _ in range(24):
        mid = (low + high) / 2
        if end_at(mid) <= video_duration_ms:
            high = mid
        else:
            low = mid
    return _report(script, estimates, video_duration_ms, high, max_segment_speed, protect_gap_ms, False)


def _report(
    script: Script,
    estimates: dict[int, int],
    video_duration_ms: int,
    speed: float,
    max_segment_speed: float,
    protect_gap_ms: int,
    clipped: bool,
) -> FitReport:
    scaled = {index: int(round(value / speed)) for index, value in estimates.items()}
    plan = plan_timeline(
        script,
        scaled,
        video_duration_ms=video_duration_ms,
        max_speed=max_segment_speed,
        min_speed=1.0,
        protect_gap_ms=protect_gap_ms,
    )
    return FitReport(
        speed=round(speed, 4),
        end_ms=plan.ends_at_ms,
        fits=plan.ends_at_ms <= video_duration_ms,
        clipped_by_limit=clipped,
        plan=plan,
    )


def estimate_durations(script: Script, *, wpm: float = 150.0) -> dict[int, int]:
    """Estimate speech length per segment from its word count.

    Only used before real audio exists (to budget the prompt and to plan the
    global rate). Once Cartesia has spoken, its measured durations replace these
    entirely -- this is an estimate, never a substitute for measurement.
    """
    if wpm <= 0:
        wpm = 150.0
    return {
        segment.index: max(400, int(round(segment.word_count / wpm * 60_000)))
        for segment in script.speakable
    }


def verify_sync(
    video_duration_ms: int,
    audio_duration_ms: int,
    plan: TimelinePlan,
    *,
    tolerance_ms: int = 250,
) -> list[str]:
    """Post-render sanity check that the narration is where it claims to be.

    Returns human-readable notes; an empty list means the render is clean. This
    runs against the *real* measured durations, so it catches the case where
    ffmpeg or the codec round-tripped audio to a different length than planned.
    """
    notes: list[str] = []

    drift = audio_duration_ms - video_duration_ms
    if abs(drift) > tolerance_ms:
        if drift < 0:
            notes.append(
                f"Narration is {format_timestamp(-drift)} shorter than the video; "
                "the final moments are silent."
            )
        else:
            notes.append(
                f"Narration is {format_timestamp(drift)} longer than the video."
            )

    for placement in plan.placements:
        if placement.natural_ms <= 0:
            continue
        if placement.overflowed:
            notes.append(
                f"Segment {placement.index} overruns its window by "
                f"{format_timestamp(placement.end_ms - placement.target_end_ms)}."
            )

    return notes
