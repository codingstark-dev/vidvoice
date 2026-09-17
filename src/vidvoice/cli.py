"""Command-line interface.

Subcommands, each thin over the library so behaviour cannot drift between the
CLI, the Python API and the web GUI:

    render    one video -> one narrated video
    batch     a directory of videos, continuing past failures
    analyze   script only (no audio), for inspecting Gemini's output cheaply
    voices    list Cartesia voices
    preview   voice one segment, to audition a voice before committing
    doctor    check ffmpeg, keys, models, and that the APIs actually answer
    gui       launch the local web interface

Exit codes: 0 success, 1 pipeline error, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__, media
from .config import Settings
from .errors import VidvoiceError
from .models import format_timestamp, script_stats
from .pipeline import (
    _FORWARDED_OPTIONS,
    Pipeline,
    RenderOptions,
    discover_videos,
    render_many,
)

#: Windows/ANSI colour, disabled when not a TTY or when NO_COLOR is set.
_COLOR = sys.stdout.isatty()


def _supports_color() -> bool:
    import os

    return _COLOR and not os.environ.get("NO_COLOR")


def _c(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def green(text: str) -> str:
    return _c(text, "32")


def yellow(text: str) -> str:
    return _c(text, "33")


def red(text: str) -> str:
    return _c(text, "31")


def dim(text: str) -> str:
    return _c(text, "2")


def bold(text: str) -> str:
    return _c(text, "1")


# --------------------------------------------------------------------------
# Logging + progress
# --------------------------------------------------------------------------


class _ProgressPrinter:
    """Progress output that stays on one line on a TTY and logs plain lines when piped."""

    def __init__(self, *, quiet: bool = False) -> None:
        self.quiet = quiet
        self.last_width = 0
        self.inline = False

    def __call__(self, stage: str, fraction: float | None) -> None:
        if self.quiet:
            return
        if fraction is None:
            self._write(stage, inline=False)
            return
        bar_width = 24
        filled = int(bar_width * max(0.0, min(1.0, fraction)))
        bar = "#" * filled + "-" * (bar_width - filled)
        self._write(f"[{bar}] {fraction * 100:5.1f}%  {stage}", inline=True)

    def _write(self, text: str, *, inline: bool) -> None:
        if not _supports_color():
            print(text, file=sys.stderr)
            return

        if self.inline and not inline:
            # Leaving inline mode: close the current line first.
            print(file=sys.stderr)
            self.last_width = 0

        padding = " " * max(0, self.last_width - len(text))
        print(f"\r{text}{padding}", end="" if inline else "\n", file=sys.stderr)
        self.last_width = len(text) if inline else 0
        self.inline = inline

    def finish(self) -> None:
        if self.inline:
            print(file=sys.stderr)
        self.last_width = 0
        self.inline = False


def setup_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s" if verbose else "%(message)s",
        stream=sys.stderr,
    )


# --------------------------------------------------------------------------
# Shared argument groups
# --------------------------------------------------------------------------


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to a .env file (default: search upwards from the current directory)",
    )
    parser.add_argument("--voice", dest="voice_id", default=None, help="Cartesia voice id")
    parser.add_argument(
        "--emotion",
        default=None,
        help=(
            "Cartesia emotion for the delivery, e.g. calm, excited, confident. "
            "Run `vidvoice emotions` for the full list."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("narration", "transcribe"),
        default="narration",
        help=(
            "narration: write a voice-over describing the screen (default). "
            "transcribe: re-voice speech already in the video."
        ),
    )
    parser.add_argument("--context", default="", help="Extra context to steer the script")
    parser.add_argument("--audience", default="", help="Who the video is for")
    parser.add_argument(
        "--max-words",
        type=int,
        default=26,
        help="Maximum words per spoken segment (default: 26)",
    )
    parser.add_argument(
        "--target-wpm",
        type=float,
        default=None,
        help=(
            "Narration pace used to budget the script against the video length "
            "(default: 150). Lower it for a calmer, slower read."
        ),
    )
    parser.add_argument(
        "--fit",
        dest="fit_mode",
        choices=("exact", "natural", "off"),
        default=None,
        help=(
            "How hard to match the video length. exact (default) converges on the "
            "video's duration; natural only makes gentle adjustments; off never "
            "changes the speaking rate."
        ),
    )
    parser.add_argument(
        "--fit-passes",
        type=int,
        default=None,
        help="Synthesis passes spent converging on the length (default: 2)",
    )
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Do not write .srt/.vtt/.txt next to the output",
    )
    parser.add_argument(
        "--model", dest="gemini_model", default=None, help="Gemini model id override"
    )
    parser.add_argument(
        "--cartesia-model", dest="cartesia_model", default=None, help="Cartesia model id override"
    )


def add_render_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output file or directory")
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="Where to keep intermediate artefacts"
    )
    parser.add_argument(
        "--no-edit",
        action="store_true",
        help="Do not merge/re-time segments before synthesis",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=1.35,
        help="Fastest playback rate used to make a line fit its window (default: 1.35)",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=0.9,
        help="Slowest playback rate used to pad a short line (default: 0.9)",
    )
    parser.add_argument(
        "--tail",
        choices=("pad", "trim", "error"),
        default="pad",
        help=(
            "When narration outlasts the video: pad freezes the last frame "
            "(default), trim cuts the audio, error refuses."
        ),
    )
    parser.add_argument(
        "--max-pad",
        type=float,
        default=60.0,
        help="Refuse to freeze the last frame for longer than this many seconds (default: 60)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip loudness normalisation of the final mix",
    )
    parser.add_argument(
        "--video-codec",
        default="copy",
        help="Video codec for the output; 'copy' (default) avoids re-encoding",
    )
    parser.add_argument(
        "--audio-bitrate", default="192k", help="AAC bitrate for the output (default: 192k)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore cached script and voice clips, and redo the expensive stages",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep per-segment audio and the mixed track in the work directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise the full pipeline with placeholder script and tones; no API calls",
    )


def add_runtime_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only log errors")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings.load(
        env_file=getattr(args, "env_file", None),
        work_dir=getattr(args, "work_dir", None),
        gemini_model=getattr(args, "gemini_model", None),
        cartesia_model=getattr(args, "cartesia_model", None),
        emotion=getattr(args, "emotion", None),
        target_wpm=getattr(args, "target_wpm", None),
        fit_mode=getattr(args, "fit_mode", None),
        dry_run=True if getattr(args, "dry_run", False) else None,
    )


def options_from_args(args: argparse.Namespace) -> RenderOptions:
    """Build RenderOptions from parsed args.

    Subcommands expose different subsets of the render flags (`analyze` has no
    `--tail`, for instance), so each value is read defensively and falls back to
    the RenderOptions default rather than requiring every subparser to declare
    every flag.
    """
    defaults = RenderOptions()

    def pick(name: str, fallback: Any) -> Any:
        value = getattr(args, name, None)
        return fallback if value is None else value

    kwargs: dict[str, Any] = {
        "mode": pick("mode", defaults.mode),
        "voice_id": pick("voice_id", defaults.voice_id),
        "output": pick("output", defaults.output),
        "work_dir": pick("work_dir", defaults.work_dir),
        "context": pick("context", defaults.context),
        "audience": pick("audience", defaults.audience),
        "max_words": pick("max_words", defaults.max_words),
        "emotion": pick("emotion", defaults.emotion),
        "target_wpm": pick("target_wpm", defaults.target_wpm),
        "fit_mode": pick("fit_mode", defaults.fit_mode),
        "fit_passes": pick("fit_passes", defaults.fit_passes),
        # These are store_true flags, so "absent" means the feature is on.
        "edit": not getattr(args, "no_edit", False),
        "normalize": not getattr(args, "no_normalize", False),
        "resume": not getattr(args, "no_resume", False),
        "subtitles": not getattr(args, "no_subtitles", False),
        "max_speed": pick("max_speed", defaults.max_speed),
        "min_speed": pick("min_speed", defaults.min_speed),
        "tail": pick("tail", defaults.tail),
        "max_pad_ms": int(pick("max_pad", defaults.max_pad_ms / 1000) * 1000),
        "video_codec": pick("video_codec", defaults.video_codec),
        "audio_bitrate": pick("audio_bitrate", defaults.audio_bitrate),
        "dry_run": bool(getattr(args, "dry_run", False)),
        "keep_temp": bool(getattr(args, "keep_temp", False)),
    }

    return RenderOptions(**kwargs)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_render(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    options = options_from_args(args)
    progress = _ProgressPrinter(quiet=args.quiet or args.json)

    try:
        result = Pipeline(settings).run(args.video, options, on_progress=progress)
    finally:
        progress.finish()

    if args.json:
        print(json.dumps(result.summary(), indent=2))
        return 0

    _print_result(result)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    options = options_from_args(args)
    progress = _ProgressPrinter(quiet=args.quiet or args.json)

    videos = discover_videos(args.directory, recursive=args.recursive)
    if not videos:
        print(red(f"No videos found in {args.directory}"), file=sys.stderr)
        return 1

    print(bold(f"Found {len(videos)} video(s) in {args.directory}"))
    try:
        results = render_many(
            videos,
            settings=settings,
            on_progress=progress,
            **{name: getattr(options, name) for name in _FORWARDED_OPTIONS},
        )
    finally:
        progress.finish()

    failures = [r for r in results if isinstance(r, VidvoiceError)]
    successes = [r for r in results if not isinstance(r, VidvoiceError)]

    if args.json:
        print(
            json.dumps(
                {
                    "succeeded": [r.summary() for r in successes],
                    "failed": [
                        {"error": exc.message, "context": exc.context} for exc in failures
                    ],
                },
                indent=2,
            )
        )
        return 1 if failures else 0

    for result in successes:
        _print_result(result, compact=True)

    if failures:
        print()
        print(red(f"{len(failures)} failed:"))
        for exc in failures:
            print(f"  - {exc.message}")
            if exc.hint:
                print(dim(f"    {exc.hint}"))
        return 1

    print()
    print(green(f"All {len(successes)} render(s) complete."))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Script only -- no TTS, no muxing. Cheap way to iterate on the prompt."""
    settings = settings_from_args(args)
    options = options_from_args(args)
    progress = _ProgressPrinter(quiet=args.quiet or args.json)

    try:
        script, info, budget = Pipeline(settings).build_script(
            args.video, options, on_progress=progress
        )
    finally:
        progress.finish()

    if args.json:
        print(
            json.dumps(
                {
                    "media": info.to_dict(),
                    "word_budget": budget,
                    "stats": script_stats(script, info.duration_ms),
                    "warnings": script.validate(info.duration_ms),
                    "script": script.to_dict(),
                },
                indent=2,
            )
        )
        return 0

    print(dim(f"{info.path.name}: {info.resolution}, {info.duration_s:.2f}s"))
    print()
    print(bold(script.summary or "(no summary)"))
    print()
    print(f"{'#':>4}  {'start':>11}  {'end':>11}  {'dur':>7}  text")
    print("-" * 100)
    for segment in script.segments:
        duration = f"{segment.duration_ms / 1000:.1f}s"
        text = segment.text if len(segment.text) <= 62 else f"{segment.text[:59]}..."
        print(
            f"{segment.index:>4}  {format_timestamp(segment.start_ms):>11}  "
            f"{format_timestamp(segment.end_ms):>11}  {duration:>7}  {text}"
        )

    stats = script_stats(script, info.duration_ms)
    print()
    print(bold("Length fit"))
    if budget:
        print(f"  word budget      {budget} words (at {options.target_wpm:.0f} wpm)")
    print(f"  words written    {stats['total_words']}")
    print(f"  speech windows   {stats['speech_window_ms'] / 1000:.1f}s of {info.duration_s:.1f}s video")
    print(f"  coverage         {stats['coverage'] * 100:.0f}% of the video")
    print(f"  words per minute {stats['words_per_minute']}")

    if budget and stats["total_words"] > budget * 1.25:
        print(
            yellow(
                f"  over budget by {stats['total_words'] - budget} words; the render "
                "will speed the voice up to fit. Lower --target-wpm to accept a "
                "slower read, or re-run to regenerate a shorter script."
            )
        )

    for warning in script.validate(info.duration_ms):
        print(yellow(f"warning: {warning}"))

    work_dir = Path(options.work_dir or settings.work_dir) / args.video.stem
    print()
    print(f"Saved to {work_dir / 'script.json'}")
    return 0


def cmd_emotions(args: argparse.Namespace) -> int:
    """List the Cartesia emotion palette."""
    from .config import EMOTIONS, PRIMARY_EMOTIONS

    if args.json:
        print(json.dumps({"primary": list(PRIMARY_EMOTIONS), "all": list(EMOTIONS)}, indent=2))
        return 0

    print(bold("Cartesia emotions"))
    print()
    print(dim("Primary:"))
    _print_wrapped(PRIMARY_EMOTIONS, indent=2)
    print()
    print(dim("Extended:"))
    _print_wrapped([e for e in EMOTIONS if e not in PRIMARY_EMOTIONS], indent=2)
    print()
    print(f"Pass one with {bold('--emotion <name>')}, e.g. {bold('--emotion confident')}")
    print(dim("Omit it to let the model interpret the emotional subtext itself."))
    if args.preview:
        print()
        print(dim("Auditioning each primary emotion..."))
        out_dir = Path(args.output or "emotion-previews")
        out_dir.mkdir(parents=True, exist_ok=True)
        for emotion in PRIMARY_EMOTIONS:
            out = out_dir / f"{emotion}.wav"
            code = cmd_preview(
                argparse.Namespace(
                    **{
                        **vars(args),
                        "emotion": emotion,
                        "output": out,
                        "no_play": True,
                        "text": args.text,
                        "speed": None,
                    }
                )
            )
            if code != 0:
                return code
        print()
        print(green(f"Wrote {len(PRIMARY_EMOTIONS)} previews to {out_dir}"))
    return 0


def _print_wrapped(items: Any, *, indent: int = 0, width: int = 88) -> None:
    """Print a list of short words, wrapped to *width*."""
    line = " " * indent
    for item in items:
        if len(line) + len(item) + 2 > width and line.strip():
            print(line.rstrip())
            line = " " * indent
        line += f"{item}  "
    if line.strip():
        print(line.rstrip())


def cmd_voices(args: argparse.Namespace) -> int:
    from .cartesia import CartesiaClient, summarize_voices

    settings = settings_from_args(args)
    try:
        with CartesiaClient(settings) as client:
            voices = client.list_voices(
                limit=args.limit,
                language=args.language,
                search=args.search,
                gender=args.gender,
            )
    except VidvoiceError as exc:
        print(red(exc.message), file=sys.stderr)
        if exc.hint:
            print(dim(exc.hint), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([v.to_dict() for v in voices], indent=2))
        return 0

    if not voices:
        print(yellow("No voices matched."))
        return 0

    print(summarize_voices(voices, limit=args.limit))
    print()
    print(dim(f"{len(voices)} voice(s). Pass one with --voice <id>."))
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    """Synthesise one line so you can audition a voice before a full render."""
    from .cartesia import CartesiaClient

    settings = settings_from_args(args)
    out = Path(args.output or "preview.wav")

    text = args.text or (
        "Here is how this sounds. The pipeline places each line at the moment "
        "it belongs, so the narration stays in step with what is on screen."
    )

    if args.dry_run or settings.dry_run:
        media.make_tone(out, duration_ms=4000, sample_rate=settings.sample_rate)
        print(dim(f"Dry run: wrote a 4s tone to {out}"))
        return 0

    try:
        with CartesiaClient(settings) as client:
            client.synthesize(text, out, voice_id=args.voice_id, speed=args.speed)
    except VidvoiceError as exc:
        print(red(exc.message), file=sys.stderr)
        if exc.hint:
            print(dim(exc.hint), file=sys.stderr)
        return 1

    duration = media.audio_duration_ms(out)
    print(green(f"Wrote {out} ({duration / 1000:.2f}s)"))
    if not args.no_play:
        _play(out)
    return 0


def _play(path: Path) -> None:
    """Play an audio file with whatever the OS provides. Best effort."""
    import shutil
    import subprocess

    for player, argv in (
        ("afplay", ["afplay", str(path)]),
        ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]),
        ("aplay", ["aplay", "-q", str(path)]),
    ):
        if shutil.which(player):
            try:
                subprocess.run(argv, check=False, timeout=120)
            except (OSError, subprocess.SubprocessError):
                return
            return
    print(dim("(no audio player found; open the file manually)"))


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the environment before spending money on a render."""
    settings = settings_from_args(args)
    problems: list[str] = []
    notes: list[str] = []

    print(bold("vidvoice doctor"))
    print()

    # Python + package
    print(f"  python            {sys.version.split()[0]}")
    print(f"  vidvoice          {__version__}")

    # ffmpeg
    import shutil

    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        if found:
            print(f"  {tool:<17} {found}")
        else:
            print(f"  {tool:<17} {red('MISSING')}")
            problems.append(f"{tool} not found on PATH")

    try:
        version = media._run(["ffmpeg", "-version"]).stdout.splitlines()[0]
        print(dim(f"    {version}"))
        print(f"  rubberband        {'yes' if media.has_rubberband() else 'no (atempo fallback)'}")
    except VidvoiceError:
        pass

    print()
    print(bold("Configuration"))
    for key, value in settings.describe().items():
        print(f"  {key:<20} {value}")

    if not settings.gemini_api_key:
        problems.append("GEMINI_API_KEY is not set")
    if not settings.cartesia_api_key:
        problems.append("CARTESIA_API_KEY is not set")

    print()
    print(bold("Connectivity"))

    if settings.gemini_api_key:
        from .gemini import list_models

        try:
            models = list_models(settings)
            available = [m for m in models if settings.gemini_model in m]
            if available:
                print(f"  gemini            {green('ok')} ({len(models)} models visible)")
                print(dim(f"    using: {settings.gemini_model}"))
            else:
                print(f"  gemini            {yellow('model not listed')}")
                problems.append(
                    f"Model '{settings.gemini_model}' is not in the list for this key"
                )
                close = [m for m in models if "gemini" in m and "flash" in m][:5]
                if close:
                    notes.append(f"Available flash models include: {', '.join(close)}")
        except VidvoiceError as exc:
            print(f"  gemini            {red('failed')}: {exc.message}")
            problems.append(f"Gemini check failed: {exc.message}")
    else:
        print(f"  gemini            {dim('skipped (no key)')}")

    if settings.cartesia_api_key:
        from .cartesia import CartesiaClient

        try:
            with CartesiaClient(settings) as client:
                voices = client.list_voices(limit=1)
            print(f"  cartesia          {green('ok')} ({len(voices)} voice(s) on first page)")
        except VidvoiceError as exc:
            print(f"  cartesia          {red('failed')}: {exc.message}")
            problems.append(f"Cartesia check failed: {exc.message}")
    else:
        print(f"  cartesia          {dim('skipped (no key)')}")

    print()
    if problems:
        print(red(f"{len(problems)} problem(s):"))
        for problem in problems:
            print(f"  - {problem}")
    else:
        print(green("Everything looks good. Try: vidvoice render <video.mp4>"))

    for note in notes:
        print(dim(f"note: {note}"))

    if problems and not settings.dry_run:
        print()
        print(dim("Tip: `vidvoice render <video> --dry-run` works without any keys."))
        return 1
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .web.app import serve
    except ImportError as exc:
        print(red("The web GUI needs extra dependencies."), file=sys.stderr)
        print(dim(f"  {exc}"), file=sys.stderr)
        print(dim("  Install them with: pip install 'vidvoice[web]'"), file=sys.stderr)
        return 1

    settings = settings_from_args(args)
    serve(settings, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------


def _print_result(result: Any, *, compact: bool = False) -> None:
    plan = result.plan
    stats = result.stats or {}
    sync = stats.get("sync_delta_ms", 0)

    print()
    print(green(f"Done in {result.elapsed_s:.1f}s") + f"  ->  {bold(str(result.output_path))}")
    print(
        dim(
            f"  {result.media.duration_s:.1f}s video, "
            f"{stats.get('speakable_segments', 0)} spoken segment(s), "
            f"{stats.get('total_words', 0)} words, "
            f"{stats.get('words_per_minute', 0)} wpm"
        )
    )
    print(
        dim(
            f"  speaking rate {stats.get('global_speed', 1.0):.2f}x, "
            f"sync offset {sync / 1000:+.2f}s"
        )
    )

    if not compact:
        print(dim(f"  script:    {result.script_path}"))
        if result.timeline_path:
            print(dim(f"  timeline:  {result.timeline_path}"))
        for fmt, path in sorted(getattr(result, "subtitles", {}).items()):
            print(dim(f"  {fmt + ':':<10} {path}"))

    if plan.compressed_count or plan.overflow_count:
        print(
            dim(
                f"  {plan.compressed_count} sped up, {plan.overflow_count} overran "
                f"their window"
            )
        )

    for warning in result.warnings:
        print(yellow(f"  warning: {warning}"))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidvoice",
        description=(
            "Turn a silent screen recording into a narrated video. Gemini writes "
            "or transcribes a timed script, Cartesia voices it, ffmpeg merges it back."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  vidvoice doctor\n"
            "  vidvoice render demo.mp4 --voice db6b0ed5-d5d3-463d-ae85-518a07d3c2b4\n"
            "  vidvoice render demo.mp4 --dry-run          # no API keys needed\n"
            "  vidvoice analyze demo.mp4                   # script only, no audio\n"
            "  vidvoice batch ./recordings --recursive\n"
            "  vidvoice gui\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"vidvoice {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # render
    p_render = subparsers.add_parser("render", help="Narrate one video")
    p_render.add_argument("video", type=Path, help="Path to the source video")
    add_common_options(p_render)
    add_render_options(p_render)
    add_runtime_flags(p_render)
    p_render.set_defaults(func=cmd_render)

    # batch
    p_batch = subparsers.add_parser("batch", help="Narrate every video in a directory")
    p_batch.add_argument("directory", type=Path, help="Directory containing videos")
    p_batch.add_argument("-r", "--recursive", action="store_true", help="Search subdirectories")
    add_common_options(p_batch)
    add_render_options(p_batch)
    add_runtime_flags(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    # analyze
    p_analyze = subparsers.add_parser(
        "analyze", help="Produce/refresh the script only (no speech, no muxing)"
    )
    p_analyze.add_argument("video", type=Path, help="Path to the source video")
    add_common_options(p_analyze)
    p_analyze.add_argument("--work-dir", type=Path, default=None)
    p_analyze.add_argument("--no-edit", action="store_true", help="Skip segment merging")
    p_analyze.add_argument("--no-resume", action="store_true", help="Re-analyse even if cached")
    p_analyze.add_argument("--dry-run", action="store_true", help="No API calls")
    add_runtime_flags(p_analyze)
    p_analyze.set_defaults(func=cmd_analyze, resume=True, edit=True, tail="pad")

    # voices
    p_voices = subparsers.add_parser("voices", help="List Cartesia voices")
    add_common_options(p_voices)
    p_voices.add_argument("--limit", type=int, default=25, help="How many to show")
    p_voices.add_argument("--language", default=None, help="Filter by language, e.g. en")
    p_voices.add_argument("--gender", default=None, help="masculine|feminine|gender_neutral")
    p_voices.add_argument("--search", default=None, help="Free-text search")
    add_runtime_flags(p_voices)
    p_voices.set_defaults(func=cmd_voices)

    # preview
    p_preview = subparsers.add_parser("preview", help="Audition a voice on one line")
    add_common_options(p_preview)
    p_preview.add_argument("--text", default="", help="Text to speak")
    p_preview.add_argument("-o", "--output", type=Path, default=Path("preview.wav"))
    p_preview.add_argument("--speed", type=float, default=None, help="Speaking rate, 0.6-1.5")
    p_preview.add_argument("--no-play", action="store_true", help="Do not play it back")
    p_preview.add_argument("--dry-run", action="store_true", help="Write a tone instead")
    add_runtime_flags(p_preview)
    p_preview.set_defaults(func=cmd_preview)

    # emotions
    p_emotions = subparsers.add_parser("emotions", help="List Cartesia's emotion palette")
    add_common_options(p_emotions)
    p_emotions.add_argument("--text", default="", help="Text used by --preview")
    p_emotions.add_argument(
        "--preview",
        action="store_true",
        help="Render a sample of every primary emotion into a directory",
    )
    p_emotions.add_argument(
        "-o", "--output", type=Path, default=Path("emotion-previews"), help="Preview directory"
    )
    p_emotions.add_argument("--no-play", action="store_true", help="(accepted for symmetry)")
    p_emotions.add_argument("--dry-run", action="store_true", help="Write tones instead")
    add_runtime_flags(p_emotions)
    p_emotions.set_defaults(func=cmd_emotions, no_play=True)

    # doctor
    p_doctor = subparsers.add_parser("doctor", help="Check tools, keys and connectivity")
    add_common_options(p_doctor)
    p_doctor.add_argument("--dry-run", action="store_true", help="Skip network checks")
    add_runtime_flags(p_doctor)
    p_doctor.set_defaults(func=cmd_doctor)

    # gui
    p_gui = subparsers.add_parser("gui", help="Launch the local web interface")
    add_common_options(p_gui)
    p_gui.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Where uploads, outputs and intermediates are kept",
    )
    p_gui.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p_gui.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    p_gui.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    p_gui.add_argument("--dry-run", action="store_true", help="Force dry-run renders")
    add_runtime_flags(p_gui)
    p_gui.set_defaults(func=cmd_gui)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", False), getattr(args, "quiet", False))

    try:
        return int(args.func(args))
    except VidvoiceError as exc:
        if getattr(args, "json", False):
            print(json.dumps({"error": exc.as_dict()}, indent=2))
        else:
            print(red(f"error: {exc.message}"), file=sys.stderr)
            if exc.hint:
                print(dim(f"  {exc.hint}"), file=sys.stderr)
            if getattr(args, "verbose", False) and exc.context:
                print(dim(json.dumps(exc.context, indent=2)), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(yellow("\ninterrupted"), file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
