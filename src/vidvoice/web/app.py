"""FastAPI application backing the local web GUI.

Design notes:

* **Local tool, not a service.** It binds to 127.0.0.1 and is meant to be run by
  one person. It is not hardened for exposure to a network.
* **Jobs run off the event loop.** Renders are CPU- and IO-bound ffmpeg work, so
  they execute in a small thread pool and the HTTP layer only ever reads job
  state. That keeps the UI responsive and progress updates flowing.
* **Progress is polled, not pushed.** Server-sent events would be tidier, but a
  short-interval poll is far easier to debug and is plenty responsive for a
  progress bar. The job registry is the single source of truth either way.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import EMOTIONS, PRIMARY_EMOTIONS, Settings
from ..errors import VidvoiceError
from ..pipeline import Pipeline, RenderOptions

log = logging.getLogger(__name__)

#: Where the GUI keeps uploads, outputs, and job state.
GUI_ROOT_NAME = "gui"

#: Renders are heavy; two at once is already generous for a laptop.
MAX_WORKERS = 2

#: Uploads accepted by the browser form.
ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@dataclass
class Job:
    """One render, from queued to finished."""

    id: str
    video_path: Path
    options: dict[str, Any]
    status: str = "queued"  # queued | running | done | failed
    stage: str = "Queued"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    hint: str | None = None
    warnings: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, *, stage: str | None = None, progress: float | None = None) -> None:
        with self._lock:
            if stage is not None:
                self.stage = stage
            if progress is not None:
                self.progress = progress

    def finish(self, *, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        with self._lock:
            self.finished_at = time.time()
            if error is not None:
                self.status = "failed"
                self.error = str(error)
                self.hint = getattr(error, "hint", None)
            else:
                self.status = "done"
                self.progress = 1.0
                self.stage = "Done"
                self.result = result or {}

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "stage": self.stage,
                "progress": round(self.progress, 4),
                "error": self.error,
                "hint": self.hint,
                "warnings": list(self.warnings),
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "elapsed_s": round((self.finished_at or time.time()) - self.created_at, 1),
                "video": self.video_path.name,
                "result": self.result,
            }


class JobRegistry:
    """Thread-safe job store plus the worker pool that runs them."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="vidvoice")

    # -- lifecycle ----------------------------------------------------------

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- jobs ---------------------------------------------------------------

    def submit(self, video: Path, options: dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], video_path=Path(video), options=dict(options))
        with self._lock:
            self._jobs[job.id] = job

        self._pool.submit(self._run, job)
        log.info("Queued job %s for %s", job.id, job.video_path.name)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def _run(self, job: Job) -> None:
        job.update(stage="Starting", progress=0.01)

        def on_progress(stage: str, fraction: float | None) -> None:
            job.update(stage=stage, progress=fraction)

        try:
            render_options = RenderOptions(**job.options)
            result = Pipeline(self.settings).run(
                job.video_path, render_options, on_progress=on_progress
            )
            job.warnings = list(result.warnings)
            job.finish(result=_result_payload(result))
        except VidvoiceError as exc:
            log.warning("Job %s failed: %s", job.id, exc.message)
            job.finish(error=exc)
        except Exception as exc:  # noqa: BLE001 - report anything unexpected to the UI
            log.exception("Job %s crashed", job.id)
            job.finish(error=exc)


def _result_payload(result: Any) -> dict[str, Any]:
    """Shape a RenderResult for JSON, exposing only files the UI can fetch."""
    return {
        "output": result.output_path.name,
        "output_url": f"/api/download/{result.output_path.name}",
        "script": result.script_path.name,
        "stats": result.stats,
        "subtitles": {
            fmt: {"name": path.name, "url": f"/api/download/{path.name}"}
            for fmt, path in result.subtitles.items()
        },
        "summary": result.summary(),
    }


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------


def create_app(settings: Settings) -> Any:
    """Build the FastAPI app. Imported lazily so the CLI works without extras."""
    try:
        from contextlib import asynccontextmanager

        from fastapi import Body, FastAPI, File, HTTPException
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "The web GUI requires fastapi, uvicorn and python-multipart. "
            "Install them with: pip install 'vidvoice[web]'"
        ) from exc

    gui_root = Path(settings.work_dir) / GUI_ROOT_NAME
    uploads = gui_root / "uploads"
    outputs = gui_root / "outputs"
    uploads.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    registry = JobRegistry(settings)

    @asynccontextmanager
    async def lifespan(_app: Any):
        """Own the worker pool for exactly as long as the server runs."""
        try:
            yield
        finally:
            registry.shutdown()

    app = FastAPI(
        title="vidvoice", docs_url="/api/docs", redoc_url=None, lifespan=lifespan
    )

    # Cached so the voice picker does not hammer the API on every page load.
    voice_cache: dict[str, Any] = {"at": 0.0, "voices": []}
    VOICE_CACHE_TTL = 300.0

    # -- pages --------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        page = Path(__file__).parent / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=500, detail="index.html is missing")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    # -- configuration ------------------------------------------------------

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            "gemini_model": settings.gemini_model,
            "cartesia_model": settings.cartesia_model,
            "cartesia_version": settings.cartesia_version,
            "default_voice": settings.cartesia_voice_id,
            "default_emotion": settings.emotion or "neutral",
            "default_wpm": settings.target_wpm,
            "sample_rate": settings.sample_rate,
            "dry_run": settings.dry_run,
            "has_gemini_key": bool(settings.gemini_api_key),
            "has_cartesia_key": bool(settings.cartesia_api_key),
            "env_file": str(settings.env_file) if settings.env_file else None,
            "emotions": list(EMOTIONS),
            "primary_emotions": list(PRIMARY_EMOTIONS),
        }

    @app.get("/api/voices")
    def voices(refresh: bool = False) -> dict[str, Any]:
        from ..cartesia import CartesiaClient

        fresh = refresh or (time.time() - voice_cache["at"]) > VOICE_CACHE_TTL
        if fresh and settings.cartesia_api_key:
            try:
                with CartesiaClient(settings) as client:
                    found = client.list_voices(limit=100)
                voice_cache["voices"] = [v.to_dict() for v in found]
                voice_cache["at"] = time.time()
            except VidvoiceError as exc:
                # Serve stale data if we have any; only fail when we have none.
                if not voice_cache["voices"]:
                    return JSONResponse(
                        status_code=502,
                        content={"error": exc.message, "hint": exc.hint},
                    )

        return {
            "voices": voice_cache["voices"],
            "stale": (time.time() - voice_cache["at"]) > VOICE_CACHE_TTL,
            "source": "cartesia" if voice_cache["voices"] else "unavailable",
        }

    @app.post("/api/preview")
    async def preview(payload: dict[str, Any] = Body(...)) -> Any:
        """Synthesise a short sample so a voice/emotion can be auditioned."""
        from .. import media
        from ..cartesia import CartesiaClient

        text = str(payload.get("text") or "").strip() or (
            "Here is how this sounds. The narration is placed at the moment it "
            "belongs, so it stays in step with what is on screen."
        )
        out = gui_root / f"preview_{uuid.uuid4().hex[:8]}.wav"

        if settings.dry_run:
            await asyncio.to_thread(
                media.make_tone, out, duration_ms=4000, sample_rate=settings.sample_rate
            )
        else:
            def render_preview() -> None:
                with CartesiaClient(settings) as client:
                    client.synthesize(
                        text,
                        out,
                        voice_id=payload.get("voice_id") or None,
                        emotion=payload.get("emotion") or None,
                        speed=payload.get("speed") or None,
                    )

            try:
                await asyncio.to_thread(render_preview)
            except VidvoiceError as exc:
                return JSONResponse(
                    status_code=502, content={"error": exc.message, "hint": exc.hint}
                )

        return {"url": f"/api/download/{out.name}", "name": out.name}

    # -- jobs ---------------------------------------------------------------

    @app.post("/api/upload")
    async def upload(file: Any = File(...)) -> Any:
        """Accept a video upload and probe it.

        ``file`` is annotated ``Any`` rather than ``UploadFile`` on purpose: this
        module uses ``from __future__ import annotations``, so FastAPI has to
        resolve the annotation at runtime, and by then the locally-imported
        ``UploadFile`` is out of scope. The runtime contract is unchanged -- the
        object is a Starlette ``UploadFile``.
        """
        name = Path(getattr(file, "filename", "") or "upload.mp4").name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{suffix}'. "
                    f"Expected one of: {', '.join(sorted(ALLOWED_SUFFIXES))}"
                ),
            )

        target = uploads / f"{uuid.uuid4().hex[:8]}_{name}"
        size = 0
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                handle.write(chunk)
                size += len(chunk)

        if size == 0:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="The uploaded file was empty.")

        from .. import media

        try:
            info = await asyncio.to_thread(media.probe, target)
        except VidvoiceError as exc:
            target.unlink(missing_ok=True)
            return JSONResponse(status_code=400, content={"error": exc.message, "hint": exc.hint})

        return {"path": str(target), "name": name, "media": info.to_dict()}

    @app.post("/api/render")
    async def render(payload: dict[str, Any] = Body(...)) -> Any:
        raw_path = str(payload.get("path") or "")
        if not raw_path:
            raise HTTPException(status_code=400, detail="No video path supplied.")

        video = Path(raw_path)
        # Containment: only render files the GUI itself uploaded. Without this
        # the endpoint would happily read any path the process can see.
        try:
            video.resolve().relative_to(uploads.resolve())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Only files uploaded through this page can be rendered.",
            ) from None

        if not video.is_file():
            raise HTTPException(status_code=404, detail=f"Uploaded file is gone: {video.name}")

        options: dict[str, Any] = {
            # Write into gui_root so the download route can serve the result.
            # Without this the output lands beside the upload, inside uploads/,
            # which the download route deliberately refuses to reach into.
            "output": outputs,
            "mode": payload.get("mode", "narration"),
            "voice_id": payload.get("voice_id") or None,
            "emotion": payload.get("emotion") or "",
            "target_wpm": float(payload.get("target_wpm") or settings.target_wpm),
            "fit_mode": payload.get("fit_mode", "exact"),
            "max_words": int(payload.get("max_words") or 26),
            "context": str(payload.get("context") or ""),
            "audience": str(payload.get("audience") or ""),
            "max_speed": float(payload.get("max_speed") or 1.35),
            "tail": payload.get("tail", "pad"),
            "subtitles": bool(payload.get("subtitles", True)),
            "normalize": bool(payload.get("normalize", True)),
            "dry_run": bool(payload.get("dry_run", settings.dry_run)),
            "edit": bool(payload.get("edit", True)),
        }

        # Fail fast on bad options rather than letting a worker die later.
        try:
            RenderOptions(**options).validate()
        except VidvoiceError as exc:
            return JSONResponse(status_code=400, content={"error": exc.message, "hint": exc.hint})

        job = registry.submit(video, options)
        return {"job_id": job.id, "state": job.to_dict()}

    @app.get("/api/job/{job_id}")
    def job_status(job_id: str) -> Any:
        job = registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id.")
        return job.to_dict()

    @app.get("/api/jobs")
    def jobs() -> dict[str, Any]:
        return {"jobs": [j.to_dict() for j in registry.all()[:20]]}

    # -- files --------------------------------------------------------------

    @app.get("/api/download/{name}")
    def download(name: str) -> Any:
        """Serve a file produced by the GUI.

        ``name`` is attacker-controlled, so the basename is stripped first (which
        defeats ``../`` traversal by construction) and the result is then
        resolved and confirmed to still sit inside ``gui_root``.
        """
        safe_name = Path(name).name
        if not safe_name or safe_name in {".", ".."}:
            raise HTTPException(status_code=400, detail="Invalid file name.")

        root = gui_root.resolve()
        candidate: Path | None = None
        # Outputs live in a subdirectory so that rendering never writes beside
        # its own input; check there first, then the other known locations.
        for directory in (outputs, gui_root, uploads):
            attempt = (directory / safe_name).resolve()
            try:
                attempt.relative_to(root)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid path.") from None
            if attempt.is_file():
                candidate = attempt
                break

        if candidate is None:
            raise HTTPException(status_code=404, detail=f"No such file: {safe_name}")

        media_types = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".wav": "audio/wav",
            ".srt": "application/x-subrip",
            ".vtt": "text/vtt",
            ".txt": "text/plain; charset=utf-8",
        }
        return FileResponse(
            candidate,
            media_type=media_types.get(candidate.suffix.lower(), "application/octet-stream"),
            filename=candidate.name,
        )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        from .. import media as media_mod

        checks: dict[str, Any] = {"ffmpeg": shutil.which("ffmpeg") is not None}
        try:
            media_mod.require_ffmpeg()
            checks["rubberband"] = media_mod.has_rubberband()
        except VidvoiceError as exc:
            checks["ffmpeg_error"] = exc.message
        return checks

    return app


# --------------------------------------------------------------------------
# Server entry point
# --------------------------------------------------------------------------


def serve(
    settings: Settings,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Run the GUI until interrupted."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required for the web GUI. Install with: pip install 'vidvoice[web]'"
        ) from exc

    app = create_app(settings)
    url = f"http://{host}:{port}"

    print(f"vidvoice GUI  ->  {url}")
    if settings.dry_run:
        print("  (dry-run mode: renders use placeholder script and tones)")
    if not settings.gemini_api_key or not settings.cartesia_api_key:
        missing = [
            name
            for name, present in (
                ("GEMINI_API_KEY", bool(settings.gemini_api_key)),
                ("CARTESIA_API_KEY", bool(settings.cartesia_api_key)),
            )
            if not present
        ]
        print(f"  missing: {', '.join(missing)} -- enable dry-run to test without them")
    print("  press Ctrl+C to stop")

    if open_browser:
        # Delay slightly so the socket is listening before the browser connects.
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")
