"""Word budgeting, global-rate fitting, and sync verification."""

from __future__ import annotations

import pytest

from vidvoice.models import Script, Segment, plan_timeline
from vidvoice.timing import (
    estimate_durations,
    fit_to_length,
    verify_sync,
    word_budget,
    words_per_segment,
)


def make_script(spans: list[tuple[int, int, str]]) -> Script:
    return Script(
        segments=[
            Segment(index=i, start_ms=start, end_ms=end, text=text)
            for i, (start, end, text) in enumerate(spans)
        ]
    )


class TestWordBudget:
    def test_scales_with_duration(self) -> None:
        # 150 wpm * 0.75 speech ratio = 112.5 words per minute.
        assert word_budget(60_000) == 112
        assert word_budget(120_000) == 225
        assert word_budget(30_000) == 56

    def test_has_a_floor(self) -> None:
        assert word_budget(500) >= 10

    def test_zero_duration(self) -> None:
        assert word_budget(0) == 0

    def test_faster_pace_allows_more_words(self) -> None:
        assert word_budget(60_000, wpm=180) > word_budget(60_000, wpm=120)

    def test_words_per_segment_is_clamped(self) -> None:
        assert words_per_segment(100, 4) == 25
        assert words_per_segment(1000, 4) == 45   # upper clamp
        assert words_per_segment(10, 4) == 6      # lower clamp
        assert words_per_segment(100, 0) == 0


class TestEstimateDurations:
    def test_proportional_to_word_count(self) -> None:
        script = make_script(
            [(0, 10_000, "one two three four five"), (10_000, 20_000, "one two")]
        )
        estimates = estimate_durations(script, wpm=120)
        # 5 words at 120 wpm = 2.5s
        assert estimates[0] == pytest.approx(2_500, abs=10)
        assert estimates[1] == pytest.approx(1_000, abs=10)

    def test_has_a_floor(self) -> None:
        script = make_script([(0, 10_000, "hi")])
        assert estimate_durations(script, wpm=150)[0] >= 400


class TestFitToLength:
    def test_speeds_up_when_too_long(self) -> None:
        script = make_script([(0, 10_000, "a b c d e f g h i j")])
        # 14s of speech for a 10s video.
        report = fit_to_length(script, 10_000, {0: 14_000})

        assert report.speed > 1.0
        assert report.end_ms <= 10_000
        assert report.fits

    def test_slows_down_when_there_is_room(self) -> None:
        script = make_script([(0, 20_000, "short")])
        # 4s of speech for a 20s video: plenty of room to relax.
        report = fit_to_length(script, 20_000, {0: 4_000})

        assert report.speed < 1.0
        assert report.fits

    def test_respects_the_speed_ceiling(self) -> None:
        script = make_script([(0, 10_000, "x")])
        # Wildly too long: cannot be fixed within the allowed range.
        report = fit_to_length(script, 10_000, {0: 60_000}, max_speed=1.4)

        assert report.speed <= 1.4
        assert not report.fits
        assert report.clipped_by_limit

    def test_reports_an_end_time(self) -> None:
        script = make_script([(0, 5_000, "hello there")])
        report = fit_to_length(script, 10_000, {0: 3_000})
        assert report.end_ms > 0

    def test_describe_mentions_fitting(self) -> None:
        script = make_script([(0, 10_000, "a b c")])
        fitting = fit_to_length(script, 10_000, {0: 4_000})
        assert "fits" in fitting.describe(10_000)

        failing = fit_to_length(script, 2_000, {0: 40_000}, max_speed=1.4)
        assert "past the video" in failing.describe(2_000)

    def test_empty_script_is_a_no_op(self) -> None:
        report = fit_to_length(Script(segments=[]), 10_000, {})
        assert report.speed == 1.0
        assert report.fits

    def test_zero_duration_video_is_a_no_op(self) -> None:
        script = make_script([(0, 5_000, "hi")])
        assert fit_to_length(script, 0, {0: 3_000}).speed == 1.0

    def test_works_without_measured_durations(self) -> None:
        """Before synthesis there is no audio, so the estimate must carry it."""
        script = make_script([(0, 10_000, " ".join(["word"] * 50))])
        report = fit_to_length(script, 10_000, None)
        assert report.speed > 1.0

    def test_result_plan_does_not_overlap(self) -> None:
        script = make_script(
            [
                (0, 4_000, "first line here"),
                (4_500, 8_500, "second line here"),
                (9_000, 13_000, "third line here"),
            ]
        )
        durations = {0: 9_000, 1: 9_000, 2: 9_000}
        report = fit_to_length(script, 13_000, durations, max_speed=1.4)

        assert report.plan is not None
        for previous, current in zip(report.plan.placements, report.plan.placements[1:]):
            assert current.start_ms >= previous.end_ms


class TestVerifySync:
    def test_clean_render_produces_no_notes(self) -> None:
        script = make_script([(0, 4_000, "a")])
        plan = plan_timeline(script, {0: 3_000}, video_duration_ms=10_000)
        assert verify_sync(10_000, 10_050, plan) == []

    def test_flags_short_narration(self) -> None:
        script = make_script([(0, 4_000, "a")])
        plan = plan_timeline(script, {0: 3_000}, video_duration_ms=10_000)
        notes = " ".join(verify_sync(10_000, 5_000, plan))
        assert "shorter than the video" in notes

    def test_flags_long_narration(self) -> None:
        script = make_script([(0, 4_000, "a")])
        plan = plan_timeline(script, {0: 3_000}, video_duration_ms=10_000)
        notes = " ".join(verify_sync(10_000, 15_000, plan))
        assert "longer than the video" in notes

    def test_flags_overflowing_segment(self) -> None:
        script = make_script([(0, 2_000, "a")])
        plan = plan_timeline(script, {0: 9_000}, video_duration_ms=10_000, max_speed=1.35)
        notes = " ".join(verify_sync(10_000, 10_000, plan))
        assert "overruns its window" in notes

    def test_tolerance_is_respected(self) -> None:
        script = make_script([(0, 4_000, "a")])
        plan = plan_timeline(script, {0: 3_000}, video_duration_ms=10_000)
        # Inside the default 250ms tolerance.
        assert verify_sync(10_000, 10_200, plan) == []
