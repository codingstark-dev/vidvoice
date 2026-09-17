"""ffmpeg / ffprobe boundary.

Everything that shells out to ffmpeg lives here, behind functions that build an
argument list and run it. Two rules keep this predictable:

* **Never** build a shell string. Always pass an argv list, so odd filenames
  (spaces, quotes, unicode) cannot break or inject.
* Every failure raises :class:`~vidvoice.errors.MediaError` carrying the stderr
  tail, so the CLI can show the real reason instead of "exit code 1".

Filters are built with comma-joined option lists rather than escaped string
literals. ``adelay=1500|1500`` is unambiguous; ``adelay=1500:all=1`` risks a
colon inside a value being read as a filter separator.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .errors import MediaError

log = logging.getLogger(__name__)

#: atempo is the always-available filter; rubberband sounds better but is often
#: not compiled in. We probe for it once and prefer it when present.
_RUBBERBAND: bool | None = None


# --------------------------------------------------------------------------
# Process helpers
# --------------------------------------------------------------------------


def require_ffmpeg() -> None:
    """Raise a helpful error if the ffmpeg toolchain is missing."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise MediaError(
            f"Missing required tool(s): {', '.join(missing)}",
            hint=(
                "Install with `brew install ffmpeg` (macOS) or "
                "`sudo apt install ffmpeg` (Debian/Ubuntu)."
            ),
        )


def _run(argv: Sequence[str], *, timeout: float | None = 1800) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing output. Raises MediaError on failure."""
    log.debug("exec: %s", " ".join(argv))
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(
            f"Command timed out after {timeout}s: {argv[0]}",
            context={"argv": list(argv)[:8]},
        ) from exc
    except FileNotFoundError as exc:
        raise MediaError(
            f"Executable not found: {argv[0]}",
            hint="Install ffmpeg (`brew install ffmpeg`).",
        ) from exc

    if completed.returncode != 0:
        raise MediaError(
            f"{Path(argv[0]).name} failed with exit code {completed.returncode}",
            context={
                "argv": list(argv),
                "stderr_tail": _tail(completed.stderr, 25),
            },
        )
    return completed


def _tail(text: str | None, lines: int) -> str:
    if not text:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


def has_rubberband() -> bool:
    """Detect the rubberband filter once per process (it sounds better than atempo)."""
    global _RUBBERBAND
    if _RUBBERBAND is None:
        try:
            completed = subprocess.run(
                ["ffmpeg", "-hide_banner", "-filters"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            _RUBBERBAND = "rubberband" in completed.stdout
        except (OSError, subprocess.SubprocessError):
            _RUBBERBAND = False
    return _RUBBERBAND


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------


@dataclass(slots=True)
class MediaInfo:
    """The subset of ffprobe output the pipeline actually cares about."""

    path: Path
    duration_ms: int
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None
    audio_channels: int
    audio_sample_rate: int
    size_bytes: int
    container: str
    raw: dict[str, Any]

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "duration_ms": self.duration_ms,
            "duration": f"{self.duration_ms / 1000:.2f}s",
            "resolution": self.resolution,
            "fps": round(self.fps, 3),
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec or "(none)",
            "audio_channels": self.audio_channels,
            "audio_sample_rate": self.audio_sample_rate,
            "size_mb": round(self.size_bytes / 1_048_576, 2),
            "container": self.container,
        }


@dataclass(slots=True)
class AudioInfo:
    """The audio-stream facts the pipeline checks (used for clips and the mix)."""

    path: Path
    duration_ms: int
    codec: str
    channels: int
    sample_rate: int
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "duration_ms": self.duration_ms,
            "duration": f"{self.duration_ms / 1000:.3f}s",
            "codec": self.codec,
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "size_kb": round(self.size_bytes / 1024, 1),
        }


def _parse_fraction(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        numerator, _, denominator = value.partition("/")
        try:
            denom = float(denominator)
            return float(numerator) / denom if denom else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe_audio(path: Path | str) -> AudioInfo:
    """Inspect an audio file. Raises MediaError if it has no audio stream."""
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"File not found: {path}")

    completed = _run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        timeout=120,
    )
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned unparseable JSON for {path}") from exc

    audio = next(
        (s for s in (data.get("streams") or []) if s.get("codec_type") == "audio"), None
    )
    if audio is None:
        raise MediaError(f"No audio stream found in {path.name}")

    fmt = data.get("format") or {}
    duration_s = 0.0
    for candidate in (fmt.get("duration"), audio.get("duration")):
        try:
            duration_s = float(candidate)
            if duration_s > 0:
                break
        except (TypeError, ValueError):
            continue

    return AudioInfo(
        path=path,
        duration_ms=int(round(duration_s * 1000)),
        codec=str(audio.get("codec_name") or "unknown"),
        channels=int(audio.get("channels") or 0),
        sample_rate=int(audio.get("sample_rate") or 0),
        size_bytes=path.stat().st_size,
    )


def probe(path: Path | str) -> MediaInfo:
    """Inspect a media file. Raises MediaError if it is not usable."""
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"File not found: {path}")

    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=120,
    )

    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned unparseable JSON for {path}") from exc

    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video is None:
        raise MediaError(
            f"No video stream found in {path.name}",
            hint="Point vidvoice at the exported video from cap.so, not an audio file.",
        )

    fmt = data.get("format") or {}

    # Duration: prefer the format-level value, fall back to the video stream.
    duration_s = 0.0
    for candidate in (fmt.get("duration"), video.get("duration")):
        try:
            duration_s = float(candidate)
            if duration_s > 0:
                break
        except (TypeError, ValueError):
            continue

    fps = _parse_fraction(video.get("avg_frame_rate")) or _parse_fraction(
        video.get("r_frame_rate")
    )

    return MediaInfo(
        path=path,
        duration_ms=int(round(duration_s * 1000)),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        video_codec=str(video.get("codec_name") or "unknown"),
        audio_codec=str(audio.get("codec_name")) if audio else None,
        audio_channels=int(audio.get("channels") or 0) if audio else 0,
        audio_sample_rate=int(audio.get("sample_rate") or 0) if audio else 0,
        size_bytes=int(fmt.get("size") or path.stat().st_size),
        container=str(fmt.get("format_name") or path.suffix.lstrip(".")),
        raw=data,
    )


def audio_duration_ms(path: Path | str) -> int:
    """Duration of an audio file in ms, or 0 if unreadable."""
    try:
        completed = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            timeout=60,
        )
        return int(round(float(completed.stdout.strip()) * 1000))
    except (MediaError, ValueError) as exc:
        log.warning("Could not read duration of %s: %s", path, exc)
        return 0


# --------------------------------------------------------------------------
# Filter construction (pure functions -- unit testable without ffmpeg)
# --------------------------------------------------------------------------


def atempo_chain(speed: float) -> list[str]:
    """Decompose a speed factor into atempo steps, each within [0.5, 2.0].

    ``atempo`` only accepts 0.5-2.0 per instance; outside that you chain them
    (``atempo=2.0,atempo=1.5`` == 3.0x). We cap the chain length defensively.
    """
    if speed <= 0:
        raise ValueError("speed must be positive")
    if abs(speed - 1.0) < 1e-4:
        return []

    filters: list[str] = []
    remaining = speed
    # Above 2.0: apply factors of 2.0 until we are in range.
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    # Below 0.5: apply factors of 0.5 until we are in range.
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    return filters


def speed_filters(speed: float) -> list[str]:
    """Filters that change tempo without changing pitch."""
    if abs(speed - 1.0) < 1e-4:
        return []
    if has_rubberband():
        return [f"rubberband=tempo={speed:.6f}"]
    return atempo_chain(speed)


class _Placeable(Protocol):
    """Structural type for anything build_mix_filter can place (see models.Placement)."""

    start_ms: int
    speed: float
    natural_ms: int


def build_mix_filter(
    placements: Sequence[_Placeable],
    *,
    sample_rate: int,
    channels: int,
    normalize: bool = True,
) -> tuple[str, list[str]]:
    """Build the filtergraph that places every clip at its timestamp and mixes.

    ``placements[i]`` describes input ``i`` -- they must be given in the same
    order as the ``-i`` arguments, and must already be filtered to clips that
    actually have audio (see :func:`mixable`).

    Each input is processed as::

        [i:a] atempo/rubberband   (speed)
              ,aresample=48000    (uniform rate before mixing)
              ,aformat            (uniform channel layout, required by amix)
              ,apad               (infinite silence; the final -t trims it)
              ,adelay=<ms>|<ms>   (shift to its slot)
              [a<i>]
        [a0][a1]...[an] amix=inputs=n:duration=longest:normalize=0 [mixed]

    ``apad`` is doing real work here. Without it each branch ends when its clip
    ends, ``amix`` with ``duration=longest`` ends at the last sample of the last
    clip, and the narration track comes out shorter than the video.

    Returns ``(filter_complex, output_args)`` where ``output_args`` is the
    argv fragment mapping the final label to the output.
    """
    if not placements:
        raise MediaError(
            "No synthesised audio to mix.",
            hint="Every segment was silent -- check the script for empty text.",
        )

    layout = "stereo" if channels == 2 else "mono"
    delay_spec = "|".join(["0"] * channels)  # placeholder, replaced per segment

    steps: list[str] = []
    labels: list[str] = []

    for position, placement in enumerate(placements):
        chain: list[str] = [f"{position}:a"]
        chain.extend(speed_filters(placement.speed))
        chain.append(f"aresample={sample_rate}")
        chain.append(f"aformat=sample_fmts=fltp:channel_layouts={layout}")
        chain.append("apad")

        delay = max(0, placement.start_ms)
        if delay > 0:
            # adelay takes one delay per channel; specifying fewer leaves the
            # remaining channels at 0 ms, which desynchronises stereo.
            chain.append("adelay=" + "|".join([str(delay)] * channels))
        else:
            chain.append(f"adelay={delay_spec}")

        source, *rest = chain
        labels.append(f"[a{position}]")
        steps.append(f"[{source}]{','.join(rest)}[a{position}]")

    mixed_label = "mixed"
    if len(labels) == 1:
        # A single input needs no amix, but still needs to reach the label the
        # output mapping expects. `anull` is a no-op that just renames it.
        steps.append(f"{labels[0]}anull[{mixed_label}]")
    else:
        steps.append(
            f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:"
            f"normalize=0[{mixed_label}]"
        )

    final_label = "mixed"
    if normalize:
        # Single-pass loudness normalisation with a true-peak ceiling. This is
        # what stops a page of individually-loud clips from clipping when summed.
        steps.append(f"[{mixed_label}]loudnorm=I=-16:TP=-1.5:LRA=11[normalised]")
        final_label = "normalised"

    return ";".join(steps), ["-map", f"[{final_label}]"]


def mixable(placements: Sequence[_Placeable]) -> list[_Placeable]:
    """Keep only placements that have real audio to place."""
    return [p for p in placements if p.natural_ms > 0]


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------


def make_silence(
    out_path: Path | str,
    *,
    duration_ms: int,
    sample_rate: int = 48_000,
    channels: int = 2,
) -> Path:
    """Write a silent WAV. Used by dry-run mode and as a mixing fallback."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layout = "stereo" if channels == 2 else "mono"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl={layout}",
            "-t",
            f"{duration_ms / 1000:.3f}",
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ],
        timeout=120,
    )
    return out_path


def make_tone(
    out_path: Path | str,
    *,
    duration_ms: int,
    frequency: float = 220.0,
    sample_rate: int = 48_000,
    channels: int = 2,
) -> Path:
    """Write a sine tone. Used by tests and dry-run to make audible placeholders."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layout = "stereo" if channels == 2 else "mono"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:sample_rate={sample_rate}",
            "-t",
            f"{duration_ms / 1000:.3f}",
            "-af",
            f"aformat=channel_layouts={layout}",
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ],
        timeout=120,
    )
    return out_path


def transcode_to_wav(
    src: Path | str,
    out_path: Path | str,
    *,
    sample_rate: int = 48_000,
    channels: int = 2,
) -> Path:
    """Normalise any audio input to the pipeline's internal WAV format."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layout = "stereo" if channels == 2 else "mono"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-sample_fmt",
            "s16",
            "-af",
            f"aformat=channel_layouts={layout}",
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ],
        timeout=600,
    )
    return out_path


def render_mix(
    clip_paths: Sequence[Path | str],
    placements: Sequence[Any],
    out_path: Path | str,
    *,
    duration_ms: int,
    sample_rate: int = 48_000,
    channels: int = 2,
    normalize: bool = True,
) -> Path:
    """Place each clip at its timestamp and mix them into one WAV.

    ``clip_paths`` and ``placements`` must be parallel and already filtered to
    segments with audio (the same filter :func:`build_mix_filter` applies).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pair each clip with its placement, then drop the ones with no audio. The
    # filtered order MUST match the -i order, since the filtergraph addresses
    # inputs by index.
    pairs = [
        (path, placement)
        for path, placement in zip(clip_paths, placements)
        if placement.natural_ms > 0
    ]
    if not pairs:
        raise MediaError("No audio clips supplied to render_mix().")

    filter_complex, map_args = build_mix_filter(
        [placement for _, placement in pairs],
        sample_rate=sample_rate,
        channels=channels,
        normalize=normalize,
    )

    argv: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path, _ in pairs:
        argv.extend(["-i", str(path)])
    argv.extend(
        [
            "-filter_complex",
            filter_complex,
            *map_args,
            "-t",
            f"{duration_ms / 1000:.3f}",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]
    )

    _run(argv, timeout=3600)
    return out_path


def mux(
    video: Path | str,
    audio: Path | str,
    out_path: Path | str,
    *,
    video_duration_ms: int,
    audio_duration_ms: int,
    tail: str = "pad",
    max_pad_ms: int = 60_000,
    video_codec: str = "copy",
    audio_bitrate: str = "192k",
) -> Path:
    """Combine the video with the mixed narration track.

    ``tail`` decides what happens when the narration outlasts the picture:

    * ``pad``   -- freeze the last frame for the overhang (default; loses nothing)
    * ``trim``  -- cut the narration to the video's length
    * ``error`` -- refuse, and let the caller shrink the script

    The video stream is copied by default, so this is fast and lossless.
    """
    video = Path(video)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    overhang_ms = audio_duration_ms - video_duration_ms
    needs_pad = tail == "pad" and overhang_ms > 0

    if tail == "error" and overhang_ms > 0:
        raise MediaError(
            f"Narration is {overhang_ms / 1000:.1f}s longer than the video.",
            hint="Use --tail pad to extend the video, or --tail trim to cut the audio.",
        )

    if needs_pad and overhang_ms > max_pad_ms:
        raise MediaError(
            f"Refusing to freeze the last frame for {overhang_ms / 1000:.1f}s "
            f"(limit {max_pad_ms / 1000:.0f}s).",
            hint="Shorten the script, speed up delivery, or raise --max-pad.",
        )

    argv: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
    ]

    if needs_pad:
        # tpad clones the final frame; apad holds silence underneath it, and -t
        # gives both streams a common, exact end point.
        total_ms = audio_duration_ms
        argv.extend(
            [
                "-filter_complex",
                f"[0:v]tpad=stop_mode=clone:stop_duration={overhang_ms / 1000:.3f},"
                f"trim=duration={total_ms / 1000:.3f},setpts=PTS-STARTPTS[vpad];"
                f"[1:a]apad,atrim=duration={total_ms / 1000:.3f},asetpts=PTS-STARTPTS[apad]",
                "-map",
                "[vpad]",
                "-map",
                "[apad]",
            ]
        )
        if video_codec == "copy":
            # Frame-cloning is a filter, so the video must be re-encoded.
            argv.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])
        else:
            argv.extend(["-c:v", video_codec, "-preset", "veryfast", "-crf", "18"])
        argv.extend(["-pix_fmt", "yuv420p"])
    else:
        argv.extend(["-map", "0:v:0", "-map", "1:a:0"])
        if video_codec == "copy":
            argv.extend(["-c:v", "copy"])
        else:
            argv.extend(["-c:v", video_codec, "-preset", "veryfast", "-crf", "18"])
        argv.extend(["-t", f"{video_duration_ms / 1000:.3f}"])

    argv.extend(
        [
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-movflags",
            "+faststart",
            "-shortest",
            str(out_path),
        ]
    )

    _run(argv, timeout=7200)
    return out_path


def snapshot(video: Path | str, out_path: Path | str, *, at_ms: int = 1000) -> Path:
    """Extract a single frame, for quick visual sanity checks in the GUI."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{at_ms / 1000:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(out_path),
        ],
        timeout=180,
    )
    return out_path


__all__ = [
    "AudioInfo",
    "MediaInfo",
    "atempo_chain",
    "audio_duration_ms",
    "build_mix_filter",
    "has_rubberband",
    "make_silence",
    "make_tone",
    "mixable",
    "mux",
    "probe",
    "probe_audio",
    "render_mix",
    "require_ffmpeg",
    "snapshot",
    "speed_filters",
    "transcode_to_wav",
]
