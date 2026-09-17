"""Timeline solver tests.

These are the highest-value tests in the suite: the solver is pure, it holds the
two invariants that make output usable, and it is where the subtlest bugs live.
"""

from __future__ import annotations

import pytest

from vidvoice.models import (
    Script,
    Segment,
    format_timestamp,
    merge_short_segments,
    parse_timestamp,
    plan_timeline,
    script_stats,
)


def make_script(spans: list[tuple[int, int, str]]) -> Script:
    return Script(
        segments=[
            Segment(index=i, start_ms=start, end_ms=end, text=text)
            for i, (start, end, text) in enumerate(spans)
        ]
    )


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------


class TestTimestamps:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("00:04", 4_000),
            ("00:04.250", 4_250),
            ("01:23", 83_000),
            ("01:23.900", 83_900),
            ("01:02:03", 3_723_000),
            ("01:02:03.500", 3_723_500),
            ("01:02:03,500", 3_723_500),  # SRT comma form
            ("90", 90_000),              # bare seconds
            (90, 90_000),
            (12.5, 12_500),
        ],
    )
    def test_parses_all_forms(self, value: object, expected: int) -> None:
        assert parse_timestamp(value) == expected  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", ["", "abc", "1:2:3:4", "--:--", None])
    def test_rejects_garbage(self, value: object) -> None:
        with pytest.raises(ValueError):
            parse_timestamp(value)  # type: ignore[arg-type]

    def test_rejects_bool(self) -> None:
        # bool is an int subclass; silently treating True as 1s would be a bug.
        with pytest.raises(ValueError):
            parse_timestamp(True)  # type: ignore[arg-type]

    def test_round_trips(self) -> None:
        for ms in (0, 250, 4_250, 83_900, 3_723_500):
            assert parse_timestamp(format_timestamp(ms)) == ms

    def test_format_hours(self) -> None:
        assert format_timestamp(3_723_500) == "01:02:03.500"
        assert format_timestamp(3_723_500, always_hours=False) == "01:02:03.500"
        assert format_timestamp(83_900) == "01:23.900"


# --------------------------------------------------------------------------
# Timeline planning
# --------------------------------------------------------------------------


class TestPlanTimeline:
    def test_empty_script(self) -> None:
        plan = plan_timeline(Script(segments=[]), {}, video_duration_ms=10_000)
        assert plan.placements == []
        assert plan.warnings() == []

    def test_clip_that_fits_is_left_alone(self) -> None:
        script = make_script([(1_000, 6_000, "short line")])
        plan = plan_timeline(script, {0: 4_000}, video_duration_ms=10_000)

        placement = plan.placements[0]
        assert placement.start_ms == 1_000
        assert placement.speed == 1.0
        assert placement.fitted_ms == 4_000
        assert not placement.compressed
        assert not placement.overflowed

    def test_long_clip_is_sped_up_to_fit(self) -> None:
        # 8s of speech in a 4s window needs 2x; capped at max_speed=1.35.
        script = make_script([(0, 4_000, "a long line")])
        plan = plan_timeline(script, {0: 8_000}, video_duration_ms=20_000, max_speed=1.35)

        placement = plan.placements[0]
        assert placement.speed == pytest.approx(1.35)
        assert placement.compressed
        assert placement.overflowed  # 8000/1.35 = 5926 > 4000 + tolerance

    def test_mild_overflow_within_tolerance_is_not_flagged(self) -> None:
        script = make_script([(0, 4_000, "x")])
        # 4100ms in a 4000ms window: 100ms over, inside the 150ms tolerance.
        plan = plan_timeline(script, {0: 4_100}, video_duration_ms=20_000)
        assert not plan.placements[0].overflowed

    def test_short_clip_in_a_large_window_is_slowed_down(self) -> None:
        script = make_script([(0, 20_000, "brief")])
        plan = plan_timeline(script, {0: 5_000}, video_duration_ms=30_000, min_speed=0.9)

        placement = plan.placements[0]
        assert placement.speed < 1.0
        assert placement.speed >= 0.9
        # Never stretched below the floor.
        assert placement.fitted_ms <= 5_600

    def test_clips_never_overlap(self) -> None:
        """The invariant that matters most: narration must not talk over itself."""
        script = make_script(
            [
                (0, 3_000, "first"),
                (3_500, 6_500, "second"),
                (7_000, 10_000, "third"),
            ]
        )
        # Every clip is far too long for its window and will overflow.
        durations = {0: 9_000, 1: 9_000, 2: 9_000}
        plan = plan_timeline(
            script, durations, video_duration_ms=12_000, max_speed=1.35
        )

        for previous, current in zip(plan.placements, plan.placements[1:]):
            assert current.start_ms >= previous.end_ms, (
                f"segment {current.index} starts at {current.start_ms}ms but "
                f"segment {previous.index} runs until {previous.end_ms}ms"
            )

    def test_gap_is_preserved_between_clips(self) -> None:
        script = make_script([(0, 5_000, "first"), (5_000, 10_000, "second")])
        # The first overruns, so the second must be pushed back.
        plan = plan_timeline(
            script,
            {0: 8_000, 1: 2_000},
            video_duration_ms=15_000,
            protect_gap_ms=200,
        )
        first, second = plan.placements
        assert second.start_ms >= first.end_ms + 200

    def test_placements_are_chronological(self) -> None:
        script = make_script([(0, 3_000, "a"), (4_000, 7_000, "b"), (8_000, 11_000, "c")])
        plan = plan_timeline(
            script, {0: 6_000, 1: 6_000, 2: 6_000}, video_duration_ms=12_000
        )
        starts = [p.start_ms for p in plan.placements]
        assert starts == sorted(starts)

    def test_shift_can_be_disabled(self) -> None:
        script = make_script([(0, 3_000, "a"), (3_200, 6_000, "b")])
        plan = plan_timeline(
            script,
            {0: 8_000, 1: 1_000},
            video_duration_ms=10_000,
            shift_on_overflow=False,
        )
        # Without shifting, the second keeps its planned start even though the
        # first is still speaking.
        assert plan.placements[1].start_ms == 3_200

    def test_zero_duration_segment_is_placed_silently(self) -> None:
        script = make_script([(0, 3_000, "spoken"), (4_000, 6_000, "[silence]")])
        plan = plan_timeline(script, {0: 2_000}, video_duration_ms=10_000)

        silent = plan.placements[1]
        assert silent.natural_ms == 0
        assert silent.fitted_ms == 0
        assert silent.speed == 1.0

    def test_zero_width_window_plays_at_natural_pace(self) -> None:
        script = make_script([(0, 1_000, "a"), (5_000, 5_000, "b")])
        plan = plan_timeline(
            script, {0: 4_000, 1: 2_000}, video_duration_ms=10_000, max_speed=1.35
        )
        # available <= 0, so no speed change is attempted.
        assert plan.placements[1].speed == 1.0

    def test_warnings_mention_overflow(self) -> None:
        script = make_script([(0, 2_000, "far too long for this window")])
        plan = plan_timeline(script, {0: 10_000}, video_duration_ms=5_000)
        notes = " ".join(plan.warnings())
        assert "could not fit" in notes

    def test_warnings_mention_overrun_past_video_end(self) -> None:
        script = make_script([(0, 4_000, "a")])
        plan = plan_timeline(script, {0: 4_000}, video_duration_ms=2_000)
        notes = " ".join(plan.warnings())
        assert "past the end" in notes

    def test_mean_speed_delta_reported(self) -> None:
        script = make_script([(0, 10_000, "a")])
        plan = plan_timeline(script, {0: 5_000}, video_duration_ms=20_000)
        assert plan.mean_abs_speed_delta > 0


# --------------------------------------------------------------------------
# Script helpers
# --------------------------------------------------------------------------


class TestMergeShortSegments:
    def test_merges_short_unterminated_segment(self) -> None:
        script = make_script([(0, 3_000, "A complete thought"), (3_000, 3_400, "and more")])
        merged = merge_short_segments(script.segments, min_duration_ms=900)

        assert len(merged) == 1
        assert merged[0].text == "A complete thought and more"
        assert merged[0].index == 0

    def test_keeps_short_segment_after_sentence_end(self) -> None:
        script = make_script([(0, 3_000, "Done."), (3_000, 3_400, "New thought")])
        merged = merge_short_segments(script.segments, min_duration_ms=900)
        assert len(merged) == 2

    def test_reindexes_densely(self) -> None:
        script = make_script([(0, 1_000, "a"), (5_000, 6_000, "b"), (9_000, 10_000, "c")])
        merged = merge_short_segments(script.segments, min_duration_ms=100)
        assert [s.index for s in merged] == [0, 1, 2]


class TestScript:
    def test_sorts_segments_by_time(self) -> None:
        script = Script(
            segments=[
                Segment(index=0, start_ms=5_000, end_ms=6_000, text="second"),
                Segment(index=1, start_ms=1_000, end_ms=2_000, text="first"),
            ]
        )
        assert [s.text for s in script.segments] == ["first", "second"]

    def test_speakable_excludes_silent_markers(self) -> None:
        script = make_script([(0, 1_000, "spoken"), (1_000, 2_000, "[pause]")])
        assert len(script.speakable) == 1

    def test_coverage(self) -> None:
        script = make_script([(0, 5_000, "a")])
        assert script.coverage(10_000) == pytest.approx(0.5)
        assert script.coverage(0) == 0.0

    def test_validate_flags_overlap(self) -> None:
        script = make_script([(0, 5_000, "a"), (3_000, 8_000, "b")])
        notes = " ".join(script.validate(10_000))
        assert "overlap" in notes

    def test_validate_flags_overrun(self) -> None:
        script = make_script([(0, 15_000, "a")])
        notes = " ".join(script.validate(10_000))
        assert "past the end" in notes

    def test_validate_flags_empty_text(self) -> None:
        script = Script(segments=[Segment(index=0, start_ms=0, end_ms=1_000, text="")])
        notes = " ".join(script.validate())
        assert "empty text" in notes

    def test_json_round_trip(self, tmp_path) -> None:
        original = Script(
            segments=[Segment(index=0, start_ms=1_000, end_ms=2_500, text="hello")],
            mode="narration",
            summary="a summary",
            model="test-model",
        )
        path = original.save(tmp_path / "script.json")
        restored = Script.load(path)

        assert restored.summary == original.summary
        assert restored.mode == original.mode
        assert len(restored.segments) == 1
        assert restored.segments[0].text == "hello"
        assert restored.segments[0].start_ms == 1_000

    def test_end_before_start_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            Segment(index=0, start_ms=5_000, end_ms=1_000, text="bad")

    def test_stats_include_delivered_wpm_when_audio_known(self) -> None:
        script = make_script([(0, 10_000, "one two three four five six")])
        stats = script_stats(script, 10_000, audio_duration_ms=12_000)
        assert "delivered_wpm" in stats
        assert stats["delivered_wpm"] == pytest.approx(30.0)

    def test_stats_omit_delivered_wpm_without_audio(self) -> None:
        script = make_script([(0, 10_000, "one two")])
        assert "delivered_wpm" not in script_stats(script, 10_000)
