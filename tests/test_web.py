"""Web GUI tests.

Exercised through FastAPI's TestClient, so the real routes, request parsing and
validation run. Renders use dry-run mode, which keeps the suite fast while still
exercising upload -> submit -> poll -> download end to end.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi", reason="the web extra is not installed")

from fastapi.testclient import TestClient  # noqa: E402

from vidvoice.config import Settings  # noqa: E402
from vidvoice.web.app import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        gemini_api_key="test-gemini",
        cartesia_api_key="test-cartesia",
        work_dir=tmp_path / "work",
        dry_run=True,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def uploaded(client, test_video):
    with test_video.open("rb") as handle:
        response = client.post(
            "/api/upload", files={"file": ("demo.mp4", handle, "video/mp4")}
        )
    assert response.status_code == 200, response.text
    return response.json()


class TestPages:
    def test_index_is_served(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "vidvoice" in response.text

    def test_index_has_the_emotion_picker(self, client) -> None:
        assert 'id="emotions"' in client.get("/").text

    def test_index_has_preview_controls(self, client) -> None:
        body = client.get("/").text
        assert 'id="preview-btn"' in body
        assert 'id="preview-audio"' in body

    def test_index_has_drop_target(self, client) -> None:
        assert 'id="drop"' in client.get("/").text


class TestConfig:
    def test_reports_models_and_palette(self, client) -> None:
        payload = client.get("/api/config").json()
        assert payload["cartesia_version"] == "2026-08-14"
        assert len(payload["emotions"]) == 58
        assert payload["primary_emotions"]

    def test_reports_key_presence_not_the_keys(self, client) -> None:
        payload = client.get("/api/config").json()
        assert payload["has_gemini_key"] is True
        # The raw key must never be served to the browser.
        assert "test-gemini" not in str(payload)

    def test_reports_dry_run(self, client) -> None:
        assert client.get("/api/config").json()["dry_run"] is True


class TestVoices:
    def test_handles_missing_credentials_gracefully(self, client) -> None:
        """With no reachable voice list the picker still gets a valid answer."""
        response = client.get("/api/voices")
        assert response.status_code in (200, 502)
        if response.status_code == 200:
            assert "voices" in response.json()


class TestUpload:
    def test_accepts_a_video(self, client, test_video) -> None:
        with test_video.open("rb") as handle:
            response = client.post(
                "/api/upload", files={"file": ("demo.mp4", handle, "video/mp4")}
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["media"]["duration_ms"] > 0
        assert payload["name"] == "demo.mp4"

    def test_rejects_a_non_video_extension(self, client) -> None:
        response = client.post(
            "/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_rejects_an_empty_file(self, client) -> None:
        response = client.post(
            "/api/upload", files={"file": ("empty.mp4", b"", "video/mp4")}
        )
        assert response.status_code == 400

    def test_rejects_a_file_that_is_not_really_video(self, client) -> None:
        response = client.post(
            "/api/upload", files={"file": ("fake.mp4", b"not a video", "video/mp4")}
        )
        assert response.status_code == 400

    def test_sanitises_the_uploaded_filename(self, client, test_video) -> None:
        """A path in the filename must not escape the uploads directory."""
        with test_video.open("rb") as handle:
            response = client.post(
                "/api/upload",
                files={"file": ("../../evil.mp4", handle, "video/mp4")},
            )

        assert response.status_code == 200
        assert ".." not in response.json()["name"]


class TestRenderValidation:
    def test_rejects_a_missing_path(self, client) -> None:
        response = client.post("/api/render", json={})
        assert response.status_code == 400

    def test_rejects_files_outside_the_upload_directory(self, client, test_video) -> None:
        """Containment: the endpoint must not read arbitrary paths."""
        response = client.post("/api/render", json={"path": str(test_video)})
        assert response.status_code == 400
        assert "uploaded" in response.json()["detail"].lower()

    def test_rejects_an_unknown_emotion(self, client, uploaded) -> None:
        response = client.post(
            "/api/render",
            json={"path": uploaded["path"], "emotion": "not-an-emotion"},
        )
        assert response.status_code == 400
        assert "emotion" in response.json()["error"].lower()

    def test_rejects_a_missing_uploaded_file(self, client, uploaded) -> None:
        import os

        os.unlink(uploaded["path"])
        response = client.post("/api/render", json={"path": uploaded["path"]})
        assert response.status_code == 404


class TestRenderFlow:
    def test_completes_and_exposes_downloads(self, client, uploaded) -> None:
        start = client.post(
            "/api/render",
            json={
                "path": uploaded["path"],
                "emotion": "confident",
                "target_wpm": 150,
                "subtitles": True,
            },
        )
        assert start.status_code == 200, start.text
        job_id = start.json()["job_id"]

        job = _wait_for(client, job_id)
        assert job["status"] == "done", job.get("error")

        result = job["result"]
        assert result["output"].endswith(".mp4")
        assert set(result["subtitles"]) == {"srt", "txt", "vtt"}

        # Every advertised file must actually be downloadable.
        for entry in [result["output_url"]] + [
            info["url"] for info in result["subtitles"].values()
        ]:
            response = client.get(entry)
            assert response.status_code == 200, entry
            assert len(response.content) > 0

    def test_reports_progress_fields(self, client, uploaded) -> None:
        job_id = client.post("/api/render", json={"path": uploaded["path"]}).json()["job_id"]
        job = _wait_for(client, job_id)

        assert 0.0 <= job["progress"] <= 1.0
        assert job["stage"]
        assert job["elapsed_s"] >= 0

    def test_job_listing_includes_the_run(self, client, uploaded) -> None:
        job_id = client.post("/api/render", json={"path": uploaded["path"]}).json()["job_id"]
        _wait_for(client, job_id)

        jobs = client.get("/api/jobs").json()["jobs"]
        assert any(job["id"] == job_id for job in jobs)

    def test_unknown_job_is_404(self, client) -> None:
        assert client.get("/api/job/nope").status_code == 404

    def test_render_options_reach_the_pipeline(self, client, uploaded) -> None:
        job_id = client.post(
            "/api/render",
            json={"path": uploaded["path"], "emotion": "calm", "fit_mode": "off"},
        ).json()["job_id"]
        job = _wait_for(client, job_id)

        assert job["status"] == "done"
        # fit_mode=off means the global rate is never adjusted.
        assert job["result"]["stats"]["global_speed"] == 1.0


class TestDownloadSafety:
    def test_traversal_is_refused(self, client) -> None:
        for attempt in (
            "../../../../etc/passwd",
            "..%2F..%2Fetc%2Fpasswd",
            "....//....//etc/passwd",
        ):
            response = client.get(f"/api/download/{attempt}")
            assert response.status_code in (400, 404), attempt
            assert b"root:" not in response.content

    def test_nested_paths_are_refused(self, client) -> None:
        response = client.get("/api/download/outputs%2Fwhatever.srt")
        assert response.status_code in (400, 404)

    def test_missing_file_is_404(self, client) -> None:
        assert client.get("/api/download/nothing-here.mp4").status_code == 404


class TestPreview:
    def test_dry_run_returns_a_playable_tone(self, client) -> None:
        response = client.post("/api/preview", json={"emotion": "excited"})
        assert response.status_code == 200

        url = response.json()["url"]
        audio = client.get(url)
        assert audio.status_code == 200
        assert len(audio.content) > 0

    def test_accepts_a_custom_line(self, client) -> None:
        response = client.post(
            "/api/preview", json={"text": "A custom preview line.", "emotion": "calm"}
        )
        assert response.status_code == 200


class TestHealth:
    def test_reports_ffmpeg(self, client) -> None:
        payload = client.get("/api/health").json()
        assert payload["ffmpeg"] is True


def _wait_for(client, job_id: str, timeout_s: float = 60.0) -> dict:
    """Poll a job until it settles."""
    deadline = time.monotonic() + timeout_s
    job: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/job/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"done", "failed"}:
            return job
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish within {timeout_s}s: {job}")
