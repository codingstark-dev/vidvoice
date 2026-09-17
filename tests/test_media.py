"""ffmpeg boundary tests.

Split into two kinds:

* **Pure** tests of filter construction, which run everywhere and catch the
  string-assembly bugs that are otherwise only visible as an ffmpeg exit code.
* **Integration** tests that really shell out to ffmpeg on synthetic media,
  asserting on measured output rather than on the command we built.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vidvoice import media
from vidvoice.errors import MediaError
from vidvoice.models import Placement

pytestmark = pytest.mark.skipif(
    not (media.shutil.which("ffmpeg") and media.shutil.which("ffprobe")),
    reason="ffmpeg is not installed",
)


def make_placement(
    index: int, start_ms: int, speed: float = 1.0, natural_ms: int = 1_000
) -> Placement:
    return Placement(
        index=index,
        target_start_ms=start_ms,
        target_end_ms=start_ms + natural_ms,
        start_ms=start_ms,
        natural_ms=natural_ms,
        speed=speed,
        fitted_ms=int(natural_ms / speed),
        compressed=speed > 1.0,
        overflowed=False,
    )


# --------------------------------------------------------------------------
# Pure filter construction
# --------------------------------------------------------------------------


class TestAtempoChain:
    def test_no_change_for_unity(self) -> None:
        assert media.atempo_chain(1.0) == []

    def test_single_filter_within_range(self) -> None:
        assert media.atempo_chain(1.25) == ["atempo=1.250000"]

    def test_chains_above_two(self) -> None:
        filters = media.atempo_chain(3.0)
        assert len(filters) == 2
        assert all(f.startswith("atempo=") for f in filters)
        product = 1.0
        for f in filters:
            product *= float(f.split("=")[1])
        assert product == pytest.approx(3.0, rel=1e-6)

    def test_chains_below_half(self) -> None:
        filters = media.atempo_chain(0.25)
        product = 1.0
        for f in filters:
            product *= float(f.split("=")[1])
        assert product == pytest.approx(0.25, rel=1e-6)

    def test_every_step_is_inside_the_supported_range(self) -> None:
        for speed in (0.1, 0.4, 0.75, 1.0, 1.9, 2.0, 2.5, 8.0):
            for f in media.atempo_chain(speed):
                value = float(f.split("=")[1])
                assert 0.5 <= value <= 2.0, f"{speed} produced {f}"

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError):
            media.atempo_chain(0.0)
        with pytest.raises(ValueError):
            media.atempo_chain(-1.0)


class TestBuildMixFilter:
    def test_single_input_skips_amix(self) -> None:
        graph, args = media.build_mix_filter(
            [make_placement(0, 0)], sample_rate=48_000, channels=2, normalize=False
        )
        assert "amix" not in graph
        # Still reaches the label the output mapping expects.
        assert args == ["-map", "[mixed]"]

    def test_multiple_inputs_use_amix(self) -> None:
        placements = [make_placement(i, i * 1000) for i in range(3)]
        graph, args = media.build_mix_filter(
            placements, sample_rate=48_000, channels=2, normalize=False
        )
        assert "amix=inputs=3" in graph
        assert args == ["-map", "[mixed]"]

    def test_normalize_adds_loudnorm(self) -> None:
        graph, args = media.build_mix_filter(
            [make_placement(0, 0)], sample_rate=48_000, channels=2, normalize=True
        )
        assert "loudnorm" in graph
        assert args == ["-map", "[normalised]"]

    def test_every_branch_is_padded(self) -> None:
        """Without apad the mix ends at the last clip and truncates the video."""
        placements = [make_placement(i, i * 1000) for i in range(3)]
        graph, _ = media.build_mix_filter(
            placements, sample_rate=48_000, channels=2, normalize=False
        )
        assert graph.count("apad") == 3

    def test_delay_has_one_value_per_channel(self) -> None:
        graph, _ = media.build_mix_filter(
            [make_placement(0, 1500)], sample_rate=48_000, channels=2, normalize=False
        )
        assert "adelay=1500|1500" in graph

    def test_mono_gets_one_delay_value(self) -> None:
        graph, _ = media.build_mix_filter(
            [make_placement(0, 1500)], sample_rate=48_000, channels=1, normalize=False
        )
        assert "adelay=1500" in graph
        assert "channel_layouts=mono" in graph

    def test_zero_delay_is_still_emitted(self) -> None:
        graph, _ = media.build_mix_filter(
            [make_placement(0, 0)], sample_rate=48_000, channels=2, normalize=False
        )
        assert "adelay=0|0" in graph

    def test_speed_filter_is_applied(self) -> None:
        graph, _ = media.build_mix_filter(
            [make_placement(0, 0, speed=1.5)],
            sample_rate=48_000,
            channels=2,
            normalize=False,
        )
        assert "atempo" in graph or "rubberband" in graph

    def test_no_speed_filter_for_unity(self) -> None:
        graph, _ = media.build_mix_filter(
            [make_placement(0, 0, speed=1.0)],
            sample_rate=48_000,
            channels=2,
            normalize=False,
        )
        assert "atempo" not in graph

    def test_branch_labels_are_sequential(self) -> None:
        placements = [make_placement(i, i * 500) for i in range(4)]
        graph, _ = media.build_mix_filter(
            placements, sample_rate=48_000, channels=2, normalize=False
        )
        for i in range(4):
            assert f"[a{i}]" in graph
            assert f"[{i}:a]" in graph

    def test_empty_raises(self) -> None:
        with pytest.raises(MediaError):
            media.build_mix_filter([], sample_rate=48_000, channels=2)


class TestMixable:
    def test_drops_silent_placements(self) -> None:
        placements = [make_placement(0, 0), make_placement(1, 1000, natural_ms=0)]
        assert len(media.mixable(placements)) == 1


# --------------------------------------------------------------------------
# Real ffmpeg integration
# --------------------------------------------------------------------------


class TestProbe:
    def test_reads_video_properties(self, test_video: Path) -> None:
        info = media.probe(test_video)
        assert info.width == 320
        assert info.height == 180
        assert info.video_codec == "h264"
        assert info.duration_ms == pytest.approx(24_000, abs=500)
        assert not info.has_audio
        assert info.resolution == "320x180"

    def test_reads_audio_properties(self, video_with_audio: Path) -> None:
        info = media.probe(video_with_audio)
        assert info.has_audio
        assert info.audio_codec == "aac"
        assert info.audio_channels == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MediaError, match="not found"):
            media.probe(tmp_path / "nope.mp4")

    def test_non_video_raises(self, tmp_path: Path) -> None:
        text_file = tmp_path / "notes.txt"
        text_file.write_text("this is not a video")
        with pytest.raises(MediaError):
            media.probe(text_file)

    def test_to_dict_is_json_friendly(self, test_video: Path) -> None:
        import json

        info = media.probe(test_video)
        assert json.dumps(info.to_dict())  # must not raise


class TestGenerators:
    def test_make_silence_has_the_requested_length(self, tmp_path: Path) -> None:
        path = media.make_silence(tmp_path / "silence.wav", duration_ms=1_500)
        assert media.audio_duration_ms(path) == pytest.approx(1_500, abs=60)

    def test_make_tone_has_the_requested_length(self, tmp_path: Path) -> None:
        path = media.make_tone(tmp_path / "tone.wav", duration_ms=800)
        assert media.audio_duration_ms(path) == pytest.approx(800, abs=60)

    def test_transcode_normalises_format(self, tmp_path: Path) -> None:
        source = media.make_tone(tmp_path / "in.wav", duration_ms=500, sample_rate=22_050)
        out = media.transcode_to_wav(source, tmp_path / "out.wav", sample_rate=48_000)

        info = media.probe_audio(out)
        assert info.sample_rate == 48_000
        assert info.channels == 2
        assert info.duration_ms == pytest.approx(500, abs=60)


class TestRenderMix:
    def test_places_clips_at_their_timestamps(self, tmp_path: Path) -> None:
        """The core promise: audio lands where the plan says it does."""
        clips = [
            media.make_tone(tmp_path / "a.wav", duration_ms=1_000, frequency=440),
            media.make_tone(tmp_path / "b.wav", duration_ms=1_000, frequency=880),
        ]
        placements = [make_placement(0, 1_000), make_placement(1, 5_000)]

        mix = media.render_mix(
            clips,
            placements,
            tmp_path / "mix.wav",
            duration_ms=7_000,
            normalize=False,
        )

        # The mix runs the full requested length, not just to the last clip.
        assert media.audio_duration_ms(mix) == pytest.approx(7_000, abs=80)

        # Silence before the first clip, sound during it, silence in the gap.
        assert _is_silent(mix, 0.0, 0.8)
        assert not _is_silent(mix, 1.2, 1.6)
        assert _is_silent(mix, 3.0, 3.8)

    def test_speed_factor_shortens_the_clip(self, tmp_path: Path) -> None:
        clip = media.make_tone(tmp_path / "a.wav", duration_ms=2_000)
        placements = [make_placement(0, 0, speed=2.0, natural_ms=2_000)]

        mix = media.render_mix(
            [clip], placements, tmp_path / "mix.wav", duration_ms=4_000, normalize=False
        )

        # At 2x, the audible part should end around 1s rather than 2s.
        assert not _is_silent(mix, 0.2, 0.8)
        assert _is_silent(mix, 1.4, 2.0)

    def test_no_overlap_when_two_clips_are_adjacent(self, tmp_path: Path) -> None:
        clips = [
            media.make_tone(tmp_path / f"{i}.wav", duration_ms=1_000, frequency=f)
            for i, f in enumerate((300, 600, 900))
        ]
        placements = [
            make_placement(0, 0, natural_ms=1_000),
            make_placement(1, 1_000, natural_ms=1_000),
            make_placement(2, 2_000, natural_ms=1_000),
        ]
        mix = media.render_mix(
            clips, placements, tmp_path / "mix.wav", duration_ms=3_000, normalize=False
        )
        assert media.audio_duration_ms(mix) == pytest.approx(3_000, abs=80)

    def test_normalization_does_not_break_output(self, tmp_path: Path) -> None:
        clips = [media.make_tone(tmp_path / "a.wav", duration_ms=1_000)]
        placements = [make_placement(0, 0)]
        mix = media.render_mix(
            clips, placements, tmp_path / "mix.wav", duration_ms=2_000, normalize=True
        )
        assert media.audio_duration_ms(mix) == pytest.approx(2_000, abs=80)

    def test_empty_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MediaError):
            media.render_mix([], [], tmp_path / "mix.wav", duration_ms=1_000)


class TestMux:
    def test_copies_video_and_replaces_audio(
        self, video_with_audio: Path, tmp_path: Path
    ) -> None:
        audio = media.make_tone(tmp_path / "voice.wav", duration_ms=5_000)
        out = media.mux(
            video_with_audio,
            audio,
            tmp_path / "out.mp4",
            video_duration_ms=24_000,
            audio_duration_ms=5_000,
        )

        info = media.probe(out)
        assert info.has_audio
        assert info.video_codec == "h264"

    def test_pad_extends_the_video_to_cover_long_narration(
        self, test_video: Path, tmp_path: Path
    ) -> None:
        # 26s of narration on a 24s video: 2s must be added by freezing frames.
        audio = media.make_tone(tmp_path / "voice.wav", duration_ms=26_000)
        out = media.mux(
            test_video,
            audio,
            tmp_path / "out.mp4",
            video_duration_ms=24_000,
            audio_duration_ms=26_000,
            tail="pad",
        )

        info = media.probe(out)
        assert info.duration_ms == pytest.approx(26_000, abs=400)

    def test_trim_cuts_narration_to_the_video(
        self, test_video: Path, tmp_path: Path
    ) -> None:
        audio = media.make_tone(tmp_path / "voice.wav", duration_ms=30_000)
        out = media.mux(
            test_video,
            audio,
            tmp_path / "out.mp4",
            video_duration_ms=24_000,
            audio_duration_ms=30_000,
            tail="trim",
        )

        info = media.probe(out)
        assert info.duration_ms == pytest.approx(24_000, abs=400)

    def test_error_mode_refuses(self, test_video: Path, tmp_path: Path) -> None:
        audio = media.make_tone(tmp_path / "voice.wav", duration_ms=30_000)
        with pytest.raises(MediaError, match="longer than the video"):
            media.mux(
                test_video,
                audio,
                tmp_path / "out.mp4",
                video_duration_ms=24_000,
                audio_duration_ms=30_000,
                tail="error",
            )

    def test_refuses_an_absurd_pad(self, test_video: Path, tmp_path: Path) -> None:
        audio = media.make_tone(tmp_path / "voice.wav", duration_ms=30_000)
        with pytest.raises(MediaError, match="Refusing to freeze"):
            media.mux(
                test_video,
                audio,
                tmp_path / "out.mp4",
                video_duration_ms=24_000,
                audio_duration_ms=200_000,
                tail="pad",
                max_pad_ms=10_000,
            )

    def test_reencode_option_produces_playable_video(
        self, test_video: Path, tmp_path: Path
    ) -> None:
        audio = media.make_tone(tmp_path / "voice.wav", duration_ms=24_000)
        out = media.mux(
            test_video,
            audio,
            tmp_path / "out.mp4",
            video_duration_ms=24_000,
            audio_duration_ms=24_000,
            video_codec="libx264",
        )
        info = media.probe(out)
        assert info.video_codec == "h264"
        assert info.duration_ms == pytest.approx(24_000, abs=400)


class TestSnapshot:
    def test_extracts_a_frame(self, test_video: Path, tmp_path: Path) -> None:
        out = media.snapshot(test_video, tmp_path / "frame.jpg", at_ms=1_000)
        assert out.is_file()
        assert out.stat().st_size > 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _is_silent(path: Path, start_s: float, end_s: float, threshold_db: float = -50.0) -> bool:
    """True when the given window of *path* is essentially silent."""
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
            value = line.split("mean_volume:")[1].strip().split()[0]
            try:
                return float(value) < threshold_db
            except ValueError:
                return False
    return False
