"""Pipeline orchestration: video in, narrated video out.

The stages are deliberately separable and each one writes its artifact to the
run directory, so a failed render can be inspected and resumed rather than
restarted from scratch:

    1. probe      -> MediaInfo            (what are we working with?)
    2. analyse    -> script.json          (the expensive, cached Gemini call)
    3. synthesise -> audio/seg_NNNN.wav   (the expensive, cached Cartesia calls)
    4. plan       -> timeline.json        (how do the clips fit the video?)
    5. mix        -> narration.wav        (placement + loudness)
    6. mux        -> <output>.mp4         (video + narration)

Stages 2 and 3 are the ones that cost money and time, so both are skipped when
their outputs already exist (``--resume``). Stages 4-6 are pure ffmpeg and are
cheap enough to always redo.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from . import media
from .cartesia import CartesiaClient
from .config import EMOTIONS, SPEED_RANGE, Settings
from .errors import ConfigError, MediaError, VidvoiceError
from .gemini import GeminiClient, dry_run_script
from .models import (
    Script,
    ScriptMode,
    Segment,
    TimelinePlan,
    format_timestamp,
    merge_short_segments,
    plan_timeline,
    script_stats,
)
from .subtitles import write_subtitles
from .timing import (
    NATURAL_SPEED_CEILING,
    NATURAL_SPEED_FLOOR,
    estimate_durations,
    fit_to_length,
    verify_sync,
    word_budget,
)

log = logging.getLogger(__name__)

#: Rough narration pace used only for dry-run placeholder lengths.
_DRY_RUN_WPM = 150.0

#: If the measured audio differs from target by less than this, accept it
#: rather than paying for another synthesis pass.
_FIT_TOLERANCE_MS = 750

#: Only re-synthesise when the corrected rate differs from the rate we used by
#: at least this much. Prevents a pointless second pass over a 2% correction.
_FIT_RATE_EPSILON = 0.04

#: Shortest window we will let a segment occupy after clamping to the video.
_MIN_SEGMENT_MS = 500


@dataclass(slots=True)
class RenderOptions:
    """Everything that changes the output, in one place."""

    mode: ScriptMode = "narration"
    voice_id: str | None = None
    output: Path | None = None
    work_dir: Path | None = None
    context: str = ""
    audience: str = ""
    max_words: int = 26
    #: Cartesia emotion from the documented palette; empty = model's own reading.
    emotion: str = ""
    #: Narration pace used to derive the word budget from the video length.
    target_wpm: float = 150.0
    #: exact | natural | off -- how hard to match the video's length.
    fit_mode: str = "exact"
    #: How many synthesis passes to spend converging on the length.
    fit_passes: int = 2
    #: Rewrite the script to fit (merge short segments, re-time to the video).
    edit: bool = True
    merge_under_ms: int = 900
    max_speed: float = 1.35
    min_speed: float = 0.9
    tail: str = "pad"  # pad | trim | error
    max_pad_ms: int = 60_000
    normalize: bool = True
    video_codec: str = "copy"
    audio_bitrate: str = "192k"
    #: Emit .srt/.txt/.vtt next to the output.
    subtitles: bool = True
    resume: bool = True
    dry_run: bool = False
    keep_temp: bool = False

    def validate(self) -> None:
        if self.emotion and self.emotion not in EMOTIONS:
            raise ConfigError(
                f"Unknown emotion '{self.emotion}'.",
                hint=f"Choose one of: {', '.join(EMOTIONS)}",
            )
        if self.fit_mode not in {"exact", "natural", "off"}:
            raise ConfigError(
                f"Unknown fit mode '{self.fit_mode}'.",
                hint="Use exact, natural, or off.",
            )


@dataclass(slots=True)
class RenderResult:
    """What came out the other end."""

    output_path: Path
    script_path: Path
    timeline_path: Path | None
    report_path: Path | None
    work_dir: Path
    media: media.MediaInfo
    script: Script
    plan: TimelinePlan
    stats: dict[str, Any]
    #: {format: path} for the .srt/.txt/.vtt files written next to the output.
    subtitles: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "output": str(self.output_path),
            "script": str(self.script_path),
            "timeline": str(self.timeline_path) if self.timeline_path else None,
            "subtitles": {fmt: str(path) for fmt, path in self.subtitles.items()},
            "duration_ms": self.media.duration_ms,
            "voice_id": self.script.meta.get("voice_id", ""),
            "emotion": self.script.meta.get("emotion", ""),
            "speech_speed": self.script.meta.get("speech_speed", 1.0),
            "stats": self.stats,
            "warnings": self.warnings,
            "elapsed_s": round(self.elapsed_s, 2),
        }


ProgressFn = Callable[[str, float | None], None]


def _noop(stage: str, fraction: float | None = None) -> None:
    return None


class Pipeline:
    """Runs the six stages. One instance per render.

    ``gemini_factory`` and ``cartesia_factory`` exist so the pipeline can be
    driven end to end against fakes. Without them, testing the real orchestration
    (caching, the length-fitting loop, cache invalidation) would require either
    live API keys or monkeypatching module internals.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        gemini_factory: Callable[[Settings], Any] | None = None,
        cartesia_factory: Callable[[Settings, Path | None], Any] | None = None,
    ) -> None:
        self.settings = settings
        self._gemini_factory = gemini_factory or (
            lambda s: GeminiClient(s)
        )
        self._cartesia_factory = cartesia_factory or (
            lambda s, cache: CartesiaClient(s, cache_dir=cache)
        )

    # -- public -------------------------------------------------------------

    def build_script(
        self,
        video: Path | str,
        options: RenderOptions | None = None,
        *,
        on_progress: ProgressFn | None = None,
    ) -> tuple[Script, media.MediaInfo, int]:
        """Produce the timed script only -- no speech, no muxing.

        This is the cheap half of the pipeline: one Gemini call, no TTS spend.
        Use it to iterate on the prompt and to see the word budget before
        committing to a full render.

        Returns ``(script, media_info, word_budget)``.
        """
        options = options or RenderOptions()
        options.validate()
        progress = on_progress or _noop

        media.require_ffmpeg()
        video = Path(video).expanduser().resolve()
        info = media.probe(video)

        budget = word_budget(info.duration_ms, wpm=options.target_wpm)
        work_dir = Path(options.work_dir or self.settings.work_dir) / _slug(video.stem)
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / "script.json"

        script = self._load_or_analyse(video, info, options, script_path, budget, progress)

        if options.edit:
            script = self._edit_script(script, info.duration_ms, options)
            script.save(script_path)

        return script, info, budget

    def run(
        self,
        video: Path | str,
        options: RenderOptions | None = None,
        *,
        on_progress: ProgressFn | None = None,
    ) -> RenderResult:
        options = options or RenderOptions()
        options.validate()
        progress = on_progress or _noop
        started = time.monotonic()

        media.require_ffmpeg()
        video = Path(video).expanduser().resolve()
        if not video.is_file():
            raise MediaError(
                f"Video not found: {video}",
                hint="Export the recording from cap.so, then pass its path.",
            )

        progress("Probing video", 0.02)
        info = media.probe(video)
        log.info(
            "Input: %s (%s, %.2fs, %s)",
            info.path.name,
            info.resolution,
            info.duration_s,
            info.video_codec,
        )
        if info.has_audio:
            log.info(
                "Input already has an audio track (%s); it will be replaced.",
                info.audio_codec,
            )

        work_dir = Path(options.work_dir or self.settings.work_dir) / _slug(video.stem)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Budget the script from the video length, so Gemini writes something
        # that fits instead of us compressing it afterwards.
        budget = word_budget(info.duration_ms, wpm=options.target_wpm)
        if options.mode == "narration" and budget:
            log.info(
                "Word budget: ~%d words for %.1fs at %.0f wpm.",
                budget,
                info.duration_s,
                options.target_wpm,
            )

        # ---- stage 2: script ------------------------------------------------
        script_path = work_dir / "script.json"
        script = self._load_or_analyse(video, info, options, script_path, budget, progress)

        if options.edit:
            progress("Editing script", 0.42)
            script = self._edit_script(script, info.duration_ms, options)
            script.save(script_path)

        warnings = script.validate(info.duration_ms)
        for warning in warnings:
            log.warning("Script: %s", warning)

        if not script.speakable:
            raise MediaError(
                "The script has nothing to say.",
                hint=(
                    "Every segment was empty or marked as silent. Re-run with "
                    "--no-resume to analyse the video again, or check that the "
                    "recording actually shows some activity."
                ),
            )

        # ---- stage 3: synthesis, converging on the video's length -----------
        durations, applied_speed = self._synthesise_to_length(
            script, info, work_dir, options, progress
        )
        script.meta["voice_id"] = options.voice_id or self.settings.cartesia_voice_id
        script.meta["emotion"] = options.emotion or self.settings.emotion or "neutral"
        script.meta["speech_speed"] = round(applied_speed, 4)
        script.save(script_path)

        # ---- stage 4: plan --------------------------------------------------
        progress("Planning timeline", 0.85)
        # The global rate already accounts for the overall length, so individual
        # segments should now need only gentle correction.
        plan = plan_timeline(
            script,
            durations,
            video_duration_ms=info.duration_ms,
            max_speed=options.max_speed,
            min_speed=options.min_speed,
        )
        for note in plan.warnings():
            log.warning("Timeline: %s", note)
        warnings.extend(plan.warnings())

        timeline_path = work_dir / "timeline.json"
        timeline_path.write_text(
            json.dumps(
                {
                    "video_duration_ms": info.duration_ms,
                    "global_speech_speed": round(applied_speed, 4),
                    "emotion": script.meta.get("emotion"),
                    "placements": [p.to_dict() for p in plan.placements],
                    "warnings": plan.warnings(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        audio_total_ms = self._resolve_tail(plan, info.duration_ms, options)

        # ---- stage 5: mix ---------------------------------------------------
        progress("Mixing narration", 0.88)
        clip_paths = [
            work_dir / "audio" / f"seg_{p.index:04d}.wav" for p in plan.placements
        ]
        mix_path = work_dir / "narration.wav"
        media.render_mix(
            clip_paths,
            plan.placements,
            mix_path,
            duration_ms=audio_total_ms,
            sample_rate=self.settings.sample_rate,
            channels=self.settings.channels,
            normalize=options.normalize,
        )
        actual_audio_ms = media.audio_duration_ms(mix_path) or audio_total_ms

        # ---- stage 6: mux ---------------------------------------------------
        progress("Muxing video", 0.94)
        output_path = self._output_path(video, options)
        media.mux(
            video,
            mix_path,
            output_path,
            video_duration_ms=info.duration_ms,
            audio_duration_ms=actual_audio_ms,
            tail=options.tail,
            max_pad_ms=options.max_pad_ms,
            video_codec=options.video_codec,
            audio_bitrate=options.audio_bitrate,
        )

        # ---- subtitles ------------------------------------------------------
        subtitle_paths: dict[str, Path] = {}
        if options.subtitles:
            progress("Writing subtitles", 0.97)
            subtitle_paths = write_subtitles(
                script, output_path, placements=plan.placements
            )
            for fmt, path in subtitle_paths.items():
                log.info("Wrote %s", path)

        # ---- verification ---------------------------------------------------
        sync_notes = verify_sync(info.duration_ms, actual_audio_ms, plan)
        for note in sync_notes:
            log.warning("Sync: %s", note)
        warnings.extend(sync_notes)

        stats = script_stats(script, info.duration_ms, actual_audio_ms)
        stats["audio_ms"] = actual_audio_ms
        stats["output_ms"] = media.audio_duration_ms(output_path) or 0
        stats["global_speed"] = round(applied_speed, 4)
        stats["sync_delta_ms"] = actual_audio_ms - info.duration_ms
        stats["video_ms"] = info.duration_ms

        report_path = work_dir / "report.json"
        result = RenderResult(
            output_path=output_path,
            script_path=script_path,
            timeline_path=timeline_path,
            report_path=report_path,
            work_dir=work_dir,
            media=info,
            script=script,
            plan=plan,
            stats=stats,
            subtitles=subtitle_paths,
            warnings=_dedupe(warnings),
            elapsed_s=time.monotonic() - started,
        )
        report_path.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")

        if not options.keep_temp:
            self._cleanup(work_dir, keep=(script_path, timeline_path, report_path))

        progress("Done", 1.0)
        return result

    # -- stage helpers ------------------------------------------------------

    def _load_or_analyse(
        self,
        video: Path,
        info: media.MediaInfo,
        options: RenderOptions,
        script_path: Path,
        budget: int,
        progress: ProgressFn,
    ) -> Script:
        cached = _read_json(script_path)
        if options.resume and cached and (cached.get("meta") or {}).get("mode") == options.mode:
            log.info("Reusing cached script at %s (delete it to re-analyse).", script_path)
            progress("Using cached script", 0.3)
            return Script.from_dict(cached)

        if options.dry_run or self.settings.dry_run:
            log.info("Dry run: generating a placeholder script instead of calling Gemini.")
            progress("Dry run: placeholder script", 0.3)
            script = dry_run_script(
                info.duration_ms, mode=options.mode, max_words=options.max_words
            )
            script.meta["mode"] = options.mode
        else:
            progress(f"Asking Gemini for a {options.mode} script", 0.1)

            def status(message: str) -> None:
                progress(message, None)

            with self._gemini_factory(self.settings) as client:
                script = client.analyze(
                    video,
                    mode=options.mode,
                    video_duration_ms=info.duration_ms,
                    context=options.context,
                    audience=options.audience,
                    max_words=options.max_words,
                    target_wpm=options.target_wpm,
                    word_budget=budget,
                    on_status=status,
                )
            script.meta["mode"] = options.mode

        script.save(script_path)
        return script

    def _edit_script(self, script: Script, video_duration_ms: int, options: RenderOptions) -> Script:
        """Light, predictable edits. Anything cleverer belongs in the prompt."""
        original_count = len(script.segments)
        segments = merge_short_segments(script.segments, min_duration_ms=options.merge_under_ms)

        clamped: list[Segment] = []
        for segment in segments:
            # Clamp into [0, video_duration]. When the video length is unknown,
            # leave the timings alone rather than guessing a ceiling.
            if video_duration_ms > 0:
                start = max(0, min(segment.start_ms, video_duration_ms))
                end = max(start, min(segment.end_ms, video_duration_ms))
                # A segment squeezed to nothing cannot be spoken; nudge it to a
                # workable minimum, pulling the start back if we are at the end.
                if end - start < _MIN_SEGMENT_MS:
                    end = min(video_duration_ms, start + _MIN_SEGMENT_MS)
                    start = max(0, end - _MIN_SEGMENT_MS)
            else:
                start, end = segment.start_ms, segment.end_ms

            clamped.append(
                Segment(
                    index=segment.index,
                    start_ms=start,
                    end_ms=end,
                    text=segment.text,
                    visual=segment.visual,
                    tone=segment.tone,
                )
            )

        for new_index, segment in enumerate(clamped):
            segment.index = new_index

        if len(clamped) != original_count:
            log.info("Merged %d segment(s) into %d.", original_count, len(clamped))

        return Script(
            segments=clamped,
            mode=script.mode,
            summary=script.summary,
            source_video=script.source_video,
            model=script.model,
            meta=dict(script.meta),
        )

    def _synthesise_to_length(
        self,
        script: Script,
        info: media.MediaInfo,
        work_dir: Path,
        options: RenderOptions,
        progress: ProgressFn,
    ) -> tuple[dict[int, int], float]:
        """Synthesise the voice and converge on the video's length.

        Returns ``(durations_ms, applied_speed)``.

        The loop:

        1. Predict the global rate from word counts (free -- no audio needed).
        2. Synthesise at that rate. Clips are cached by (text, voice, speed), so
           a repeat render at the same rate costs nothing.
        3. Measure the *real* durations and ask: does this now fit?
        4. If not, and we have passes left, correct the rate from the measured
           overhang and go again. Only the clips whose rate actually changed get
           re-synthesised.

        Two passes is enough in practice because step 1 is already close: the
        estimator and the TTS engine disagree by a roughly constant factor, and
        one measurement pins that factor down.
        """
        audio_dir = work_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        if not script.speakable:
            return {}, 1.0

        estimates = estimate_durations(script)
        target_ms = info.duration_ms
        passes = max(1, options.fit_passes) if options.fit_mode == "exact" else 1

        # Pass 1: pick a starting rate from the text estimate.
        if options.fit_mode == "off":
            rate = 1.0
            report = None
        else:
            report = fit_to_length(
                script,
                target_ms,
                estimates,
                min_speed=max(NATURAL_SPEED_FLOOR, options.min_speed)
                if options.fit_mode == "natural"
                else NATURAL_SPEED_FLOOR,
                max_speed=min(NATURAL_SPEED_CEILING, SPEED_RANGE[1]),
            )
            rate = report.speed
            log.info("Predicted speaking rate: %.3f (%s)", rate, report.describe(target_ms))

        durations: dict[int, int] = {}

        if not options.resume:
            # --no-resume means "redo the expensive stage", so drop stale clips
            # rather than silently reusing audio rendered at a different rate.
            for stale in audio_dir.glob("seg_*.wav"):
                stale.unlink(missing_ok=True)

        for attempt in range(1, passes + 1):
            progress(
                f"Synthesising speech at {rate:.2f}x"
                + (f" (pass {attempt}/{passes})" if passes > 1 else ""),
                0.45 + (attempt - 1) * 0.1,
            )
            durations = self._render_clips(
                script, audio_dir, options, rate, progress
            )

            if options.fit_mode != "exact" or attempt == passes:
                break

            measured_end = _measure_end(script, durations, target_ms, rate=rate)
            overhang = measured_end - target_ms
            log.info(
                "Measured at %.3fx: speech ends %s (target %s), %s",
                rate,
                format_timestamp(measured_end),
                format_timestamp(target_ms),
                "fits" if overhang <= 0 else f"+{format_timestamp(overhang)}",
            )

            if abs(overhang) <= _FIT_TOLERANCE_MS:
                break

            corrected = _next_rate(
                script, durations, target_ms, current=rate, end_ms=measured_end
            )
            if abs(corrected - rate) < _FIT_RATE_EPSILON:
                log.info("Rate correction below threshold; accepting %.3fx.", rate)
                break

            log.info("Adjusting speaking rate %.3fx -> %.3fx", rate, corrected)
            rate = round(corrected, 4)

        return durations, rate

    def _render_clips(
        self,
        script: Script,
        audio_dir: Path,
        options: RenderOptions,
        rate: float,
        progress: ProgressFn,
    ) -> dict[int, int]:
        """Render every speakable segment at *rate*; return index -> measured ms."""
        durations: dict[int, int] = {}
        pending: list[Segment] = []

        for segment in script.speakable:
            clip = audio_dir / f"seg_{segment.index:04d}.wav"
            existing = media.audio_duration_ms(clip) if clip.is_file() else 0
            if existing > 0:
                durations[segment.index] = existing
            else:
                pending.append(segment)

        if durations:
            log.info("Reusing %d cached voice clip(s).", len(durations))

        if not pending:
            return durations

        if options.dry_run or self.settings.dry_run:
            self._dry_run_synthesis(pending, audio_dir, durations, rate, progress)
        else:
            self._synthesise(pending, audio_dir, durations, options, rate, progress)

        return durations

    def _dry_run_synthesis(
        self,
        pending: Sequence[Segment],
        audio_dir: Path,
        durations: dict[int, int],
        rate: float,
        progress: ProgressFn,
    ) -> None:
        """Write a tone whose length matches an estimated speaking time.

        Using a per-word estimate rather than the segment window is what makes
        this a real test: the tone lengths then disagree with the windows in
        exactly the way real speech does, so the fitting maths gets exercised.
        """
        log.info(
            "Dry run: generating %d placeholder tone(s) instead of calling Cartesia.",
            len(pending),
        )
        for position, segment in enumerate(pending, start=1):
            words = max(1, segment.word_count)
            estimate_ms = int(words / _DRY_RUN_WPM * 60_000 / max(0.01, rate))
            clip = audio_dir / f"seg_{segment.index:04d}.wav"
            media.make_tone(
                clip,
                duration_ms=max(300, estimate_ms),
                frequency=180.0 + (segment.index % 8) * 40.0,
                sample_rate=self.settings.sample_rate,
                channels=self.settings.channels,
            )
            durations[segment.index] = media.audio_duration_ms(clip)
            progress(f"Dry run: {position}/{len(pending)} placeholders", None)

    def _synthesise(
        self,
        pending: Sequence[Segment],
        audio_dir: Path,
        durations: dict[int, int],
        options: RenderOptions,
        rate: float,
        progress: ProgressFn,
    ) -> None:
        cache_dir = Path(self.settings.work_dir) / ".cache" / "cartesia"
        emotion = options.emotion or self.settings.emotion or None
        total = len(pending)

        with self._cartesia_factory(self.settings, cache_dir) as client:
            for position, segment in enumerate(pending, start=1):
                clip = audio_dir / f"seg_{segment.index:04d}.wav"
                client.synthesize(
                    segment.text,
                    clip,
                    voice_id=options.voice_id,
                    speed=rate,
                    # A per-segment tone wins over the run-wide emotion; both
                    # fall back to letting the model read the text itself.
                    emotion=segment.tone or emotion,
                )
                duration = media.audio_duration_ms(clip)
                if duration <= 0:
                    raise MediaError(
                        f"Cartesia produced unreadable audio for segment {segment.index}.",
                        hint=f"Text was: {segment.text[:120]!r}",
                    )
                durations[segment.index] = duration
                progress(f"Synthesising {position}/{total}", None)

    def _resolve_tail(self, plan: TimelinePlan, video_duration_ms: int, options: RenderOptions) -> int:
        """Decide how long the narration track should be."""
        speech_end = plan.ends_at_ms

        if options.tail == "trim":
            return video_duration_ms
        if options.tail == "error" and speech_end > video_duration_ms:
            raise MediaError(
                f"Narration ends at {format_timestamp(speech_end)}, "
                f"{format_timestamp(speech_end - video_duration_ms)} past the video.",
                hint=(
                    "Shorten the script (lower --max-words), raise --max-speed, or "
                    "use --tail pad / --tail trim."
                ),
            )
        # Pad: the mix must run to the end of the speech so nothing is clipped;
        # the mux stage freezes the last frame to cover the overhang.
        return max(video_duration_ms, speech_end)

    def _output_path(self, video: Path, options: RenderOptions) -> Path:
        """Resolve where the narrated video goes.

        ``-o`` may be a file, an existing directory, or a directory that does not
        exist yet. A path with no suffix is treated as a directory, because
        "output to a folder" is what someone means by ``-o ./out``, whereas
        ffmpeg would otherwise fail with "unable to choose an output format".
        """
        if options.output:
            out = Path(options.output).expanduser()
            if out.is_dir() or out.suffix == "":
                out.mkdir(parents=True, exist_ok=True)
                return out / f"{video.stem}_narrated.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            return out

        return video.with_name(f"{video.stem}_narrated.mp4")

    @staticmethod
    def _cleanup(work_dir: Path, *, keep: Sequence[Path]) -> None:
        """Remove only the safely-regenerable intermediates.

        The per-segment clips in ``audio/`` are deliberately **kept**. They are
        what makes a re-render cheap: changing the voice, emotion or timing
        options still hits the clip cache for every unchanged segment, and the
        length-fitting loop can re-measure without paying to synthesise again.
        Deleting them here would quietly turn every tweak into a full re-render.
        """
        keep_set = {p.resolve() for p in keep}
        mix = work_dir / "narration.wav"
        if mix.is_file():
            mix.unlink(missing_ok=True)
        for stray in work_dir.glob("*.tmp"):
            if stray.resolve() not in keep_set:
                stray.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in text).strip("-")
    return (cleaned or "video")[:64]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Ignoring unreadable %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _measure_end(
    script: Script,
    durations_ms: dict[int, int],
    target_ms: int,
    *,
    rate: float = 1.0,
) -> int:
    """Where the narration ends, given clip lengths measured at *rate*.

    Places the clips with the same rules the real render will use -- so gaps and
    any shifting are accounted for -- and reports the end of the last audible
    clip, which is the thing that must fit inside the video.

    Per-segment stretching is deliberately disabled (``max_speed=1.0``): this
    measures whether the *global* rate is right, and letting individual clips
    stretch too would mask an incorrect global rate.
    """
    if not durations_ms:
        return 0
    plan = plan_timeline(
        script,
        _at_rate(durations_ms, rate),
        video_duration_ms=target_ms,
        max_speed=1.0,
        min_speed=1.0,
    )
    return plan.ends_at_ms


def _next_rate(
    script: Script,
    durations_ms: dict[int, int],
    target_ms: int,
    *,
    current: float,
    end_ms: int,
) -> float:
    """Pick the next global speaking rate to try, from real measurements.

    A proportional correction -- ``new = current * end / target`` -- is the
    obvious approach and it does not work here: a segment's own speed cap and
    the shifting rule both do part of the work, so the global rate is not
    linearly related to where the audio ends. The proportional estimate can even
    move the rate in the wrong direction.

    The one property that does hold is monotonicity: raising the global rate
    never makes the narration longer. So this brackets the answer from the
    direction of the error and bisects inside the bracket. One pass gets close;
    the caller's next measurement finishes the job.
    """
    low = max(SPEED_RANGE[0], NATURAL_SPEED_FLOOR)
    high = min(NATURAL_SPEED_CEILING, SPEED_RANGE[1])
    if target_ms <= 0 or low >= high:
        return current

    if end_ms > target_ms:
        # Too long: the answer is faster than what we just measured with.
        lo = min(max(current, low), high)
        if lo >= high:
            return high
        if _measure_end(script, durations_ms, target_ms, rate=high) > target_ms:
            # Even the fastest allowed rate overruns. Use it anyway: it is the
            # closest we can get, and the tail strategy covers the overhang.
            return high
        for _ in range(24):
            mid = (lo + high) / 2
            if _measure_end(script, durations_ms, target_ms, rate=mid) > target_ms:
                lo = mid
            else:
                high = mid
        return high

    # Too short: the answer is slower, which fills more of the video.
    hi = min(max(current, low), high)
    if hi <= low:
        return low
    if _measure_end(script, durations_ms, target_ms, rate=low) <= target_ms:
        # Even the slowest allowed rate still fits; that is the best fill.
        return low
    for _ in range(24):
        mid = (low + hi) / 2
        if _measure_end(script, durations_ms, target_ms, rate=mid) > target_ms:
            low = mid
        else:
            hi = mid
    return low


def _at_rate(durations_ms: dict[int, int], rate: float) -> dict[int, int]:
    """Clip lengths as they would be if rendered at *rate*.

    ``durations_ms`` are the lengths measured at whatever rate produced them, so
    the new length is the old one divided by the rate. (Speech length is
    inversely proportional to speaking rate.) This is a prediction used only for
    bracketing; the next synthesis pass replaces it with a real measurement.
    """
    if rate <= 0:
        return dict(durations_ms)
    return {index: int(round(value / rate)) for index, value in durations_ms.items()}


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def render(
    video: Path | str,
    *,
    settings: Settings | None = None,
    on_progress: ProgressFn | None = None,
    **option_overrides: Any,
) -> RenderResult:
    """Convenience one-shot API: ``render("demo.mp4", voice_id="...", emotion="happy")``."""
    settings = settings or Settings.load()
    options = RenderOptions(**option_overrides)
    return Pipeline(settings).run(video, options, on_progress=on_progress)


#: The RenderOptions fields render_many forwards. Kept explicit so a typo is an
#: error at import time rather than a silently-ignored keyword argument.
_FORWARDED_OPTIONS = (
    "mode",
    "voice_id",
    "output",
    "work_dir",
    "context",
    "audience",
    "max_words",
    "emotion",
    "target_wpm",
    "fit_mode",
    "fit_passes",
    "edit",
    "merge_under_ms",
    "max_speed",
    "min_speed",
    "tail",
    "max_pad_ms",
    "normalize",
    "video_codec",
    "audio_bitrate",
    "subtitles",
    "resume",
    "dry_run",
    "keep_temp",
)


def render_many(
    videos: Sequence[Path | str],
    *,
    settings: Settings | None = None,
    on_progress: ProgressFn | None = None,
    **option_overrides: Any,
) -> list[RenderResult | VidvoiceError]:
    """Render a batch, continuing past individual failures.

    Returns results in input order; failures come back as the exception object
    so the caller can report them all at the end rather than dying on the first
    bad file.
    """
    settings = settings or Settings.load()
    unknown = set(option_overrides) - set(_FORWARDED_OPTIONS)
    if unknown:
        raise ConfigError(f"Unknown render option(s): {', '.join(sorted(unknown))}")

    results: list[RenderResult | VidvoiceError] = []

    for position, video in enumerate(videos, start=1):
        if on_progress:
            on_progress(f"[{position}/{len(videos)}] {Path(video).name}", None)
        try:
            results.append(
                render(
                    video,
                    settings=settings,
                    on_progress=on_progress,
                    **option_overrides,
                )
            )
        except VidvoiceError as exc:
            log.error("Failed on %s: %s", video, exc.message)
            results.append(exc)

    return results


def discover_videos(directory: Path | str, *, recursive: bool = False) -> list[Path]:
    """Find renderable videos in a directory, newest first."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise MediaError(f"Not a directory: {directory}")

    patterns = ("*.mp4", "*.mov", "*.mkv", "*.webm", "*.avi", "*.m4v")
    found: list[Path] = []
    for pattern in patterns:
        iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
        found.extend(p for p in iterator if p.is_file() and "_narrated" not in p.stem)

    return sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)
