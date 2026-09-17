"""End-to-end pipeline tests.

Real ffmpeg, real video, real mixing and muxing -- but fake Gemini and fake
Cartesia, injected through the Pipeline's factories. That combination is the
point: the expensive, non-deterministic parts are replaced while everything that
actually decides whether the output is usable (timing, placement, caching, the
fitting loop) runs for real.

The fakes deliberately misbehave in realistic ways: the script generator puts
more words in a window than will fit, and the "voice" takes a predictable but
different amount of time than the words suggest. That is exactly the mismatch
the real pipeline has to absorb.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidvoice import media
from vidvoice.errors import MediaError
from vidvoice.models import Script, Segment
from vidvoice.pipeline import Pipeline, RenderOptions, discover_videos
from vidvoice.subtitles import parse_srt

pytestmark = pytest.mark.skipif(
    not (media.shutil.which("ffmpeg") and media.shutil.which("ffprobe")),
    reason="ffmpeg is not installed",
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeGemini:
    """Stands in for GeminiClient, returning a script built from fixed spans."""

    def __init__(self, spans: list[tuple[int, int, str]], calls: list[str] | None = None):
        self.spans = spans
        self.calls = calls if calls is not None else []

    def __enter__(self) -> FakeGemini:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def analyze(self, path: Path, **kwargs: object) -> Script:
        self.calls.append(str(path))
        return Script(
            segments=[
                Segment(index=i, start_ms=start, end_ms=end, text=text)
                for i, (start, end, text) in enumerate(self.spans)
            ],
            mode=str(kwargs.get("mode", "narration")),
            summary="fake summary",
            model="fake-gemini",
        )


class FakeCartesia:
    """Stands in for CartesiaClient, writing tones of a predictable length.

    ``ms_per_word`` is the fake voice's pace, so tests can make the synthesised
    audio deliberately disagree with the segment windows.
    """

    def __init__(
        self,
        ms_per_word: int = 400,
        calls: list[tuple[int, float, str]] | None = None,
        fail_on: int | None = None,
    ):
        self.ms_per_word = ms_per_word
        self.calls = calls if calls is not None else []
        self.fail_on = fail_on
        self._counter = 0

    def __enter__(self) -> FakeCartesia:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def synthesize(
        self,
        text: str,
        out_path: Path,
        *,
        voice_id: str | None = None,
        speed: float | None = None,
        emotion: str | None = None,
        **kwargs: object,
    ) -> Path:
        self._counter += 1
        if self.fail_on is not None and self._counter == self.fail_on:
            raise MediaError("simulated TTS failure")

        rate = speed or 1.0
        words = max(1, len(text.split()))
        duration_ms = int(words * self.ms_per_word / rate)
        self.calls.append((words, rate, emotion or ""))

        media.make_tone(
            out_path,
            duration_ms=max(200, duration_ms),
            frequency=200.0 + words * 10,
        )
        return out_path


def build_pipeline(settings, gemini: FakeGemini, cartesia: FakeCartesia) -> Pipeline:
    return Pipeline(
        settings,
        gemini_factory=lambda _s: gemini,
        cartesia_factory=lambda _s, _cache: cartesia,
    )


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


class TestFullRender:
    def test_produces_a_narrated_video(self, settings, test_video: Path, tmp_path: Path) -> None:
        gemini = FakeGemini([(1_000, 5_000, "First line of narration here.")])
        cartesia = FakeCartesia(ms_per_word=300)
        settings = settings.with_overrides(dry_run=False)

        result = build_pipeline(settings, gemini, cartesia).run(
            test_video,
            RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work"),
        )

        assert result.output_path.is_file()
        info = media.probe(result.output_path)
        assert info.has_audio
        assert info.video_codec == "h264"
        assert info.duration_ms == pytest.approx(24_000, abs=600)

    def test_writes_every_subtitle_format(self, settings, test_video: Path, tmp_path: Path) -> None:
        gemini = FakeGemini([(1_000, 5_000, "First line.")])
        settings = settings.with_overrides(dry_run=False)

        result = build_pipeline(settings, gemini, FakeCartesia()).run(
            test_video,
            RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work"),
        )

        assert set(result.subtitles) == {"srt", "txt", "vtt"}
        for path in result.subtitles.values():
            assert path.is_file()
        assert result.subtitles["srt"].read_text().strip().startswith("1")

    def test_subtitles_match_the_placed_audio(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        """Cue times must come from the timeline, not the model's request."""
        # The model asks for 1s-5s; the voice will need shifting/speeding.
        gemini = FakeGemini([(1_000, 5_000, "A line that will not fit its window at all.")])
        settings = settings.with_overrides(dry_run=False)

        result = build_pipeline(settings, gemini, FakeCartesia(ms_per_word=600)).run(
            test_video,
            RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work"),
        )

        cues = parse_srt(result.subtitles["srt"].read_text())
        assert cues
        placement = result.plan.placements[0]
        assert cues[0].start_ms == placement.start_ms

    def test_writes_script_and_timeline_artifacts(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        gemini = FakeGemini([(0, 4_000, "Hello.")])
        settings = settings.with_overrides(dry_run=False)

        result = build_pipeline(settings, gemini, FakeCartesia()).run(
            test_video,
            RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work"),
        )

        assert result.script_path.is_file()
        assert result.timeline_path and result.timeline_path.is_file()
        assert result.report_path and result.report_path.is_file()

        timeline = json.loads(result.timeline_path.read_text())
        assert "placements" in timeline
        assert "global_speech_speed" in timeline

    def test_report_is_valid_json(self, settings, test_video: Path, tmp_path: Path) -> None:
        gemini = FakeGemini([(0, 4_000, "Hello.")])
        settings = settings.with_overrides(dry_run=False)

        result = build_pipeline(settings, gemini, FakeCartesia()).run(
            test_video,
            RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work"),
        )

        report = json.loads(result.report_path.read_text())
        assert report["output"] == str(result.output_path)
        assert "stats" in report

    def test_audio_lands_at_the_planned_timestamps(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        """The end-to-end promise, verified against the rendered audio."""
        spans = [
            (1_000, 4_000, "First line here."),
            (10_000, 13_000, "Second line here."),
        ]
        gemini = FakeGemini(spans)
        settings = settings.with_overrides(dry_run=False)

        result = build_pipeline(settings, gemini, FakeCartesia(ms_per_word=400)).run(
            test_video,
            RenderOptions(
                output=tmp_path / "out",
                work_dir=tmp_path / "work",
                keep_temp=True,
                normalize=False,
            ),
        )

        mix = result.work_dir / "narration.wav"
        assert mix.is_file()

        for placement in result.plan.placements:
            if placement.fitted_ms <= 0:
                continue
            # A window inside the placed clip should have sound.
            middle = (placement.start_ms + placement.fitted_ms // 2) / 1000
            assert not _is_silent(mix, middle - 0.1, middle + 0.1), (
                f"segment {placement.index} has no audio at {middle:.2f}s "
                f"where the plan places it"
            )

    def test_narration_never_overlaps_itself(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        spans = [
            (0, 3_000, "First line here that is quite long."),
            (3_000, 6_000, "Second line here that is also long."),
            (6_000, 9_000, "Third line here and equally wordy."),
        ]
        gemini = FakeGemini(spans)
        # 900ms per word makes every line far too long for its 3s window.
        settings = settings.with_overrides(dry_run=False)

        result = build_pipeline(settings, gemini, FakeCartesia(ms_per_word=900)).run(
            test_video,
            RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work"),
        )

        speakable = [p for p in result.plan.placements if p.fitted_ms > 0]
        for previous, current in zip(speakable, speakable[1:]):
            assert current.start_ms >= previous.end_ms


# --------------------------------------------------------------------------
# Caching and the fitting loop
# --------------------------------------------------------------------------


class TestCaching:
    def test_script_is_reused_on_a_second_run(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        gemini = FakeGemini([(0, 4_000, "Hello.")])
        cartesia = FakeCartesia()
        settings = settings.with_overrides(dry_run=False)
        pipeline = build_pipeline(settings, gemini, cartesia)

        options = RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work")
        pipeline.run(test_video, options)
        assert len(gemini.calls) == 1

        pipeline.run(test_video, options)
        assert len(gemini.calls) == 1, "Gemini should not be called twice"

    def test_voice_clips_are_reused(self, settings, test_video: Path, tmp_path: Path) -> None:
        gemini = FakeGemini([(0, 4_000, "Hello there friend.")])
        cartesia = FakeCartesia()
        settings = settings.with_overrides(dry_run=False)
        pipeline = build_pipeline(settings, gemini, cartesia)

        options = RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work")
        pipeline.run(test_video, options)
        first = len(cartesia.calls)

        pipeline.run(test_video, options)
        assert len(cartesia.calls) == first, "clips should come from cache"

    def test_no_resume_forces_re_synthesis(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        gemini = FakeGemini([(0, 4_000, "Hello there friend.")])
        cartesia = FakeCartesia()
        settings = settings.with_overrides(dry_run=False)
        pipeline = build_pipeline(settings, gemini, cartesia)

        work = tmp_path / "work"
        pipeline.run(test_video, RenderOptions(output=tmp_path / "out", work_dir=work))
        first = len(cartesia.calls)

        pipeline.run(
            test_video,
            RenderOptions(output=tmp_path / "out", work_dir=work, resume=False),
        )
        assert len(cartesia.calls) > first

    def test_dry_run_makes_no_api_calls(self, settings, test_video: Path, tmp_path: Path) -> None:
        gemini = FakeGemini([(0, 4_000, "Should not be used.")])
        cartesia = FakeCartesia()
        # settings fixture already has dry_run=True

        result = build_pipeline(settings, gemini, cartesia).run(
            test_video,
            RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work"),
        )

        assert gemini.calls == []
        assert cartesia.calls == []
        assert result.output_path.is_file()


class TestFittingLoop:
    def test_speeds_up_a_script_that_is_too_long(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        """A script too long at natural pace must be compressed to fit."""
        # 8 lines x 6 words x 500ms = 24s of raw speech for a 24s video, spread
        # over windows totalling only 12s. Fixable, but only by speeding up.
        spans = [
            (i * 3_000, i * 3_000 + 1_500, " ".join(["word"] * 6))
            for i in range(8)
        ]
        gemini = FakeGemini(spans)
        cartesia = FakeCartesia(ms_per_word=500)
        settings = settings.with_overrides(dry_run=False)

        result = build_pipeline(settings, gemini, cartesia).run(
            test_video,
            RenderOptions(
                output=tmp_path / "out",
                work_dir=tmp_path / "work",
                fit_mode="exact",
                fit_passes=2,
            ),
        )

        stats = result.stats
        assert stats["global_speed"] > 1.0 or result.plan.compressed_count > 0
        assert abs(stats["sync_delta_ms"]) < 3_000

    def test_impossible_script_still_renders_and_warns(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        """When the script cannot possibly fit, degrade loudly, not silently.

        60s of speech cannot fit a 24s video even at the maximum combined
        compression, so the render must still complete, keep the audio intact,
        and tell the user exactly what happened.
        """
        spans = [
            (i * 2_000, i * 2_000 + 1_500, " ".join(["word"] * 10))
            for i in range(12)
        ]
        gemini = FakeGemini(spans)
        settings = settings.with_overrides(dry_run=False)

        result = build_pipeline(settings, gemini, FakeCartesia(ms_per_word=500)).run(
            test_video,
            RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work"),
        )

        # It rendered, and the narration is fully present rather than cut off.
        assert result.output_path.is_file()
        assert result.stats["sync_delta_ms"] > 0

        notes = " ".join(result.warnings)
        assert "longer than the video" in notes

    def test_adapts_per_segment_when_only_the_windows_are_tight(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        """Speech that fits the video overall still has to fit each window.

        Here the total speech is shorter than the video, so the global rate can
        relax -- but every line is individually too long for its own window, so
        per-segment compression must pick up the slack.
        """
        spans = [
            (0, 4_000, " ".join(["word"] * 10)),
            (8_000, 12_000, " ".join(["word"] * 10)),
            (16_000, 20_000, " ".join(["word"] * 10)),
        ]
        gemini = FakeGemini(spans)
        settings = settings.with_overrides(dry_run=False)

        result = build_pipeline(settings, gemini, FakeCartesia(ms_per_word=500)).run(
            test_video,
            RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work"),
        )

        assert result.plan.compressed_count == 3
        assert result.stats["sync_delta_ms"] == 0

    def test_converges_towards_the_video_length(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        """The point of the fitting loop: extra passes must not make it worse."""
        spans = [
            (i * 3_000, i * 3_000 + 1_500, " ".join(["word"] * 6))
            for i in range(8)
        ]
        gemini = FakeGemini(spans)
        settings = settings.with_overrides(dry_run=False)

        single = build_pipeline(settings, gemini, FakeCartesia(ms_per_word=500)).run(
            test_video,
            RenderOptions(
                output=tmp_path / "one", work_dir=tmp_path / "work_one", fit_passes=1
            ),
        )
        double = build_pipeline(settings, gemini, FakeCartesia(ms_per_word=500)).run(
            test_video,
            RenderOptions(
                output=tmp_path / "two", work_dir=tmp_path / "work_two", fit_passes=2
            ),
        )

        assert abs(double.stats["sync_delta_ms"]) <= abs(single.stats["sync_delta_ms"]) + 250

    def test_fit_off_leaves_the_rate_alone(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        gemini = FakeGemini([(0, 4_000, "A short line.")])
        cartesia = FakeCartesia(ms_per_word=300)
        settings = settings.with_overrides(dry_run=False)

        result = build_pipeline(settings, gemini, cartesia).run(
            test_video,
            RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work", fit_mode="off"),
        )

        assert result.stats["global_speed"] == 1.0

    def test_emotion_is_passed_through(self, settings, test_video: Path, tmp_path: Path) -> None:
        gemini = FakeGemini([(0, 4_000, "Excited line.")])
        cartesia = FakeCartesia()
        settings = settings.with_overrides(dry_run=False)

        build_pipeline(settings, gemini, cartesia).run(
            test_video,
            RenderOptions(
                output=tmp_path / "out",
                work_dir=tmp_path / "work",
                emotion="excited",
            ),
        )

        assert cartesia.calls
        assert cartesia.calls[0][2] == "excited"


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------


class TestFailures:
    def test_missing_video_raises_clearly(self, settings, tmp_path: Path) -> None:
        pipeline = build_pipeline(settings, FakeGemini([]), FakeCartesia())
        with pytest.raises(MediaError, match="not found"):
            pipeline.run(tmp_path / "nope.mp4")

    def test_empty_script_is_rejected(self, settings, test_video: Path, tmp_path: Path) -> None:
        gemini = FakeGemini([])
        settings = settings.with_overrides(dry_run=False)

        with pytest.raises(MediaError, match="nothing to say"):
            build_pipeline(settings, gemini, FakeCartesia()).run(
                test_video,
                RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work"),
            )

    def test_synthesis_failure_propagates(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        gemini = FakeGemini([(0, 4_000, "Line one."), (5_000, 9_000, "Line two.")])
        cartesia = FakeCartesia(fail_on=2)
        settings = settings.with_overrides(dry_run=False)

        with pytest.raises(MediaError, match="simulated TTS failure"):
            build_pipeline(settings, gemini, cartesia).run(
                test_video,
                RenderOptions(output=tmp_path / "out", work_dir=tmp_path / "work"),
            )

    def test_unknown_emotion_is_rejected_before_any_work(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        from vidvoice.errors import ConfigError

        gemini = FakeGemini([(0, 4_000, "Hello.")])
        pipeline = build_pipeline(settings, gemini, FakeCartesia())

        with pytest.raises(ConfigError, match="Unknown emotion"):
            pipeline.run(test_video, RenderOptions(emotion="not-a-real-emotion"))

        assert gemini.calls == [], "validation must happen before any API call"

    def test_tail_error_refuses_a_too_long_script(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        # 60 words at 600ms is 36s of speech for a 24s video, past the speed cap.
        spans = [(0, 22_000, " ".join(["word"] * 60))]
        gemini = FakeGemini(spans)
        cartesia = FakeCartesia(ms_per_word=600)
        settings = settings.with_overrides(dry_run=False)

        with pytest.raises(MediaError):
            build_pipeline(settings, gemini, cartesia).run(
                test_video,
                RenderOptions(
                    output=tmp_path / "out",
                    work_dir=tmp_path / "work",
                    tail="error",
                    fit_mode="off",
                ),
            )


# --------------------------------------------------------------------------
# Script-only path
# --------------------------------------------------------------------------


class TestBuildScript:
    def test_returns_script_without_synthesising(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        gemini = FakeGemini([(0, 4_000, "Hello.")])
        cartesia = FakeCartesia()
        settings = settings.with_overrides(dry_run=False)

        script, info, budget = build_pipeline(settings, gemini, cartesia).build_script(
            test_video, RenderOptions(work_dir=tmp_path / "work")
        )

        assert script.segments
        assert info.duration_ms > 0
        assert budget > 0
        assert cartesia.calls == [], "analyze must not synthesise"

    def test_word_budget_tracks_video_length(
        self, settings, test_video: Path, tmp_path: Path
    ) -> None:
        gemini = FakeGemini([(0, 4_000, "Hello.")])
        settings = settings.with_overrides(dry_run=False)
        pipeline = build_pipeline(settings, gemini, FakeCartesia())

        _, _, slow = pipeline.build_script(
            test_video, RenderOptions(work_dir=tmp_path / "a", target_wpm=120)
        )
        _, _, fast = pipeline.build_script(
            test_video, RenderOptions(work_dir=tmp_path / "b", target_wpm=180)
        )

        assert fast > slow


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


class TestDiscoverVideos:
    def test_finds_videos_by_extension(self, tmp_path: Path) -> None:
        for name in ("a.mp4", "b.mov", "c.webm", "notes.txt"):
            (tmp_path / name).write_bytes(b"x")

        found = {p.name for p in discover_videos(tmp_path)}
        assert found == {"a.mp4", "b.mov", "c.webm"}

    def test_skips_already_narrated_outputs(self, tmp_path: Path) -> None:
        (tmp_path / "demo.mp4").write_bytes(b"x")
        (tmp_path / "demo_narrated.mp4").write_bytes(b"x")

        found = {p.name for p in discover_videos(tmp_path)}
        assert found == {"demo.mp4"}

    def test_recursive_option(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep.mp4").write_bytes(b"x")

        assert discover_videos(tmp_path) == []
        assert len(discover_videos(tmp_path, recursive=True)) == 1

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MediaError):
            discover_videos(tmp_path / "nope")


def _is_silent(path: Path, start_s: float, end_s: float, threshold_db: float = -50.0) -> bool:
    import subprocess

    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-ss", str(start_s), "-t", str(end_s - start_s),
            "-i", str(path), "-af", "volumedetect", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stderr.splitlines():
        if "mean_volume:" in line:
            try:
                return float(line.split("mean_volume:")[1].strip().split()[0]) < threshold_db
            except (ValueError, IndexError):
                return False
    return False
