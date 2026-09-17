"""vidvoice -- narrated videos from silent screen recordings.

Quick start::

    from vidvoice import Settings, render

    result = render("demo.mp4", settings=Settings.load())
    print(result.output_path)

Or from the shell::

    vidvoice render demo.mp4 --voice <cartesia-voice-id>
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Settings
from .errors import (
    AlignmentError,
    CartesiaError,
    ConfigError,
    GeminiError,
    MediaError,
    VidvoiceError,
)
from .models import (
    Placement,
    Script,
    ScriptMode,
    Segment,
    TimelinePlan,
    format_timestamp,
    parse_timestamp,
    plan_timeline,
    script_stats,
)
from .pipeline import (
    Pipeline,
    RenderOptions,
    RenderResult,
    discover_videos,
    render,
    render_many,
)

__all__ = [
    "__version__",
    # config
    "Settings",
    # errors
    "AlignmentError",
    "CartesiaError",
    "ConfigError",
    "GeminiError",
    "MediaError",
    "VidvoiceError",
    # model
    "Placement",
    "Script",
    "ScriptMode",
    "Segment",
    "TimelinePlan",
    "format_timestamp",
    "parse_timestamp",
    "plan_timeline",
    "script_stats",
    # pipeline
    "Pipeline",
    "RenderOptions",
    "RenderResult",
    "discover_videos",
    "render",
    "render_many",
]
