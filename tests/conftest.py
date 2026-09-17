"""Shared fixtures.

Tests never touch the network and never need API keys. Where a real API would be
called, the pipeline runs in dry-run mode; where the API client itself is under
test, it is driven against a `httpx.MockTransport` so the exact wire format can
be asserted.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vidvoice.config import Settings

#: Long enough to need several segments, short enough to render in a second.
TEST_VIDEO_SECONDS = 24


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="session")
def ffmpeg_available() -> bool:
    from shutil import which

    return which("ffmpeg") is not None and which("ffprobe") is not None


@pytest.fixture(scope="session")
def test_video(tmp_path_factory: pytest.TempPathFactory, ffmpeg_available: bool) -> Path:
    """A small silent H.264 clip, standing in for a cap.so export.

    Deliberately silent so the narration track is the only audio and tests can
    assert on exactly where speech lands.
    """
    if not ffmpeg_available:
        pytest.skip("ffmpeg is not installed")

    path = tmp_path_factory.mktemp("media") / "demo.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i",
        f"testsrc2=size=320x180:rate=15:duration={TEST_VIDEO_SECONDS}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(path),
    )
    return path


@pytest.fixture(scope="session")
def video_with_audio(tmp_path_factory: pytest.TempPathFactory, ffmpeg_available: bool) -> Path:
    """A clip that already has an audio track, to prove it gets replaced."""
    if not ffmpeg_available:
        pytest.skip("ffmpeg is not installed")

    path = tmp_path_factory.mktemp("media") / "with_audio.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i",
        f"testsrc2=size=320x180:rate=15:duration={TEST_VIDEO_SECONDS}",
        "-f", "lavfi", "-i",
        # `sine` is mono, so force stereo to mirror a real recording.
        f"sine=frequency=440:sample_rate=48000:duration={TEST_VIDEO_SECONDS}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-ac", "2", "-c:a", "aac", "-shortest",
        str(path),
    )
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Dry-run settings pointing at a throwaway work directory."""
    return Settings(
        gemini_api_key="test-gemini-key",
        cartesia_api_key="test-cartesia-key",
        work_dir=tmp_path / "work",
        dry_run=True,
    )


@pytest.fixture
def live_settings(tmp_path: Path) -> Settings:
    """Settings that will attempt real API calls (used with a mock transport)."""
    return Settings(
        gemini_api_key="test-gemini-key",
        cartesia_api_key="test-cartesia-key",
        work_dir=tmp_path / "work",
        dry_run=False,
    )
