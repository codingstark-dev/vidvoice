"""Subtitle and transcript export."""

from __future__ import annotations

from pathlib import Path

from vidvoice.models import Placement, Script, Segment, parse_timestamp
from vidvoice.subtitles import (
    parse_srt,
    to_srt,
    to_txt,
    to_vtt,
    wrap_text,
    write_subtitles,
)


def make_script(spans: list[tuple[int, int, str]]) -> Script:
    return Script(
        segments=[
            Segment(index=i, start_ms=start, end_ms=end, text=text)
            for i, (start, end, text) in enumerate(spans)
        ]
    )


def make_placement(
    index: int, target: tuple[int, int], start: int, fitted: int
) -> Placement:
    return Placement(
        index=index,
        target_start_ms=target[0],
        target_end_ms=target[1],
        start_ms=start,
        natural_ms=fitted,
        speed=1.0,
        fitted_ms=fitted,
        compressed=False,
        overflowed=False,
    )


class TestSrt:
    def test_basic_shape(self) -> None:
        script = make_script([(1_000, 4_000, "Hello world.")])
        srt = to_srt(script, wrap=False)

        lines = srt.strip().splitlines()
        assert lines[0] == "1"
        assert lines[1] == "00:00:01,000 --> 00:00:04,000"
        assert lines[2] == "Hello world."

    def test_cues_are_numbered_sequentially(self) -> None:
        script = make_script(
            [(0, 1_000, "one"), (2_000, 3_000, "two"), (4_000, 5_000, "three")]
        )
        blocks = [b for b in to_srt(script, wrap=False).strip().split("\n\n")]
        assert [b.splitlines()[0] for b in blocks] == ["1", "2", "3"]

    def test_uses_placed_times_when_available(self) -> None:
        """Subtitles must match the audio you hear, not the window requested."""
        script = make_script([(1_000, 4_000, "Shifted line.")])
        placements = [make_placement(0, (1_000, 4_000), start=2_500, fitted=2_000)]

        srt = to_srt(script, placements=placements, wrap=False)
        assert "00:00:02,500 --> 00:00:04,500" in srt

    def test_overlapping_cues_are_trimmed(self) -> None:
        script = make_script([(0, 5_000, "first"), (3_000, 8_000, "second")])
        srt = to_srt(script, wrap=False)

        # Second cue starts at 3s, so the first must be shortened to clear it.
        blocks = srt.strip().split("\n\n")
        first_end = blocks[0].splitlines()[1].split("-->")[1].strip()
        assert parse_timestamp(first_end) < 3_000

    def test_zero_length_cue_gets_a_minimum_duration(self) -> None:
        script = make_script([(2_000, 2_000, "instant")])
        srt = to_srt(script, wrap=False)
        assert "00:00:02,000 --> 00:00:02,800" in srt

    def test_hours_are_formatted(self) -> None:
        script = make_script([(3_723_000, 3_724_000, "late")])
        assert "01:02:03,000 --> 01:02:04,000" in to_srt(script, wrap=False)

    def test_wrapping_splits_long_lines(self) -> None:
        long_text = " ".join(["word"] * 30)
        script = make_script([(0, 5_000, long_text)])
        srt = to_srt(script, wrap=True, line_width=30)

        body = srt.strip().splitlines()[2:]
        assert len(body) <= 2  # at most two lines per cue
        assert len(body[0]) <= 30

    def test_wrapping_marks_truncation(self) -> None:
        long_text = " ".join(["word"] * 60)
        assert "..." in wrap_text(long_text, width=20, max_lines=2)


class TestVtt:
    def test_has_the_required_header(self) -> None:
        script = make_script([(0, 1_000, "hi")])
        assert to_vtt(script, wrap=False).startswith("WEBVTT")

    def test_uses_dot_separator(self) -> None:
        script = make_script([(1_500, 2_500, "hi")])
        assert "00:00:01.500 --> 00:00:02.500" in to_vtt(script, wrap=False)


class TestTxt:
    def test_plain_by_default(self) -> None:
        script = make_script([(1_000, 2_000, "First line."), (3_000, 4_000, "Second.")])
        text = to_txt(script)
        assert "First line." in text
        assert "[00:" not in text

    def test_timestamps_on_request(self) -> None:
        script = make_script([(1_000, 2_000, "First line.")])
        assert "[00:01.000] First line." in to_txt(script, timestamps=True)

    def test_includes_summary(self) -> None:
        script = Script(
            segments=[Segment(index=0, start_ms=0, end_ms=1_000, text="hi")],
            summary="A short summary.",
        )
        assert "A short summary." in to_txt(script)

    def test_visuals_are_optional(self) -> None:
        script = Script(
            segments=[
                Segment(index=0, start_ms=0, end_ms=1_000, text="hi", visual="a cursor")
            ]
        )
        assert "a cursor" not in to_txt(script)
        assert "a cursor" in to_txt(script, include_visuals=True)


class TestWriteSubtitles:
    def test_writes_all_formats(self, tmp_path: Path) -> None:
        script = make_script([(0, 1_000, "hi")])
        written = write_subtitles(script, tmp_path / "demo_narrated.mp4")

        assert set(written) == {"srt", "txt", "vtt"}
        for path in written.values():
            assert path.is_file()
            assert path.stat().st_size > 0

    def test_replaces_the_extension(self, tmp_path: Path) -> None:
        script = make_script([(0, 1_000, "hi")])
        written = write_subtitles(script, tmp_path / "demo_narrated.mp4")
        assert written["srt"].name == "demo_narrated.srt"

    def test_can_write_a_subset(self, tmp_path: Path) -> None:
        script = make_script([(0, 1_000, "hi")])
        written = write_subtitles(script, tmp_path / "demo.mp4", formats=("srt",))
        assert set(written) == {"srt"}


class TestParseSrt:
    def test_round_trips(self) -> None:
        script = make_script([(1_000, 4_000, "Hello world."), (5_000, 8_000, "Again.")])
        restored = parse_srt(to_srt(script, wrap=False))

        assert len(restored) == 2
        assert restored[0].text == "Hello world."
        assert restored[0].start_ms == 1_000
        assert restored[0].end_ms == 4_000

    def test_handles_missing_index_lines(self) -> None:
        text = "00:00:01,000 --> 00:00:02,000\nNo index here.\n"
        parsed = parse_srt(text)
        assert len(parsed) == 1
        assert parsed[0].text == "No index here."

    def test_ignores_malformed_blocks(self) -> None:
        assert parse_srt("not a subtitle at all") == []

    def test_unwraps_multiline_cues(self) -> None:
        text = "1\n00:00:01,000 --> 00:00:03,000\nfirst line\nsecond line\n"
        parsed = parse_srt(text)
        assert parsed[0].text == "first line second line"
