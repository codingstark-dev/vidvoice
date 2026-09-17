"""Configuration, loaded from environment / .env.

Kept dependency-free: a tiny .env parser instead of python-dotenv, because the
format we need (KEY=VALUE, # comments, optional quotes) is trivial and this
keeps `pip install vidvoice` to a single hard dependency.

Precedence, lowest to highest:
    built-in defaults  <  .env file  <  real environment variables  <  explicit overrides
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import ConfigError

# --------------------------------------------------------------------------
# Defaults. These are the values that were verified against the live API docs;
# keep the docstring next to each in sync if you bump them.
# --------------------------------------------------------------------------

#: 48 kHz stereo is the universally safe intermediate for ffmpeg mixing.
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_CHANNELS = 2

#: Cartesia's current recommended model. Verified against the 2026-08-14 spec:
#: TTSModelID enum is [sonic-3.6, sonic-3.5, sonic-3, sonic-latest], default sonic-3.6.
#: (sonic-2 / sonic-turbo were sunset and are rejected.)
DEFAULT_CARTESIA_MODEL = "sonic-3.6"
#: Cartesia pins behaviour via a required `Cartesia-Version` header. The spec's
#: enum accepts exactly one value, so this is not a free-form choice.
DEFAULT_CARTESIA_VERSION = "2026-08-14"
DEFAULT_CARTESIA_VOICE = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"  # "Skylar" (official en-US voice)

#: Documented generation_config bounds. Enforced in CartesiaClient, since the API
#: returns a 400 rather than clamping.
SPEED_RANGE = (0.6, 1.5)
VOLUME_RANGE = (0.5, 2.0)

#: The complete emotion enum from the 2026-08-14 Emotion schema, in spec order.
#: Cartesia documents `neutral`, `calm`, `angry`, `content`, `sad` and `scared`
#: as the primary set; the rest are the extended palette.
EMOTIONS: tuple[str, ...] = (
    "neutral", "happy", "excited", "enthusiastic", "elated", "euphoric",
    "triumphant", "amazed", "surprised", "flirtatious", "curious", "content",
    "peaceful", "serene", "calm", "grateful", "affectionate", "trust",
    "sympathetic", "anticipation", "mysterious", "angry", "mad", "outraged",
    "frustrated", "agitated", "threatened", "disgusted", "contempt", "envious",
    "sarcastic", "ironic", "sad", "dejected", "melancholic", "disappointed",
    "hurt", "guilty", "bored", "tired", "rejected", "nostalgic", "wistful",
    "apologetic", "hesitant", "insecure", "confused", "resigned", "anxious",
    "panicked", "alarmed", "scared", "proud", "confident", "distant",
    "skeptical", "contemplative", "determined",
)

#: The six Cartesia calls out as primary -- shown first in the GUI.
PRIMARY_EMOTIONS: tuple[str, ...] = (
    "neutral", "calm", "content", "happy", "sad", "angry", "scared",
)

#: Default narration pace. Used to derive a word budget from the video length so
#: Gemini writes a script that actually fits, rather than one we must compress.
DEFAULT_TARGET_WPM = 150.0

#: Accepted by `--fit`: how hard to work to match the video's length.
FIT_MODES = ("exact", "natural", "off")

#: Gemini model used to watch the video.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

#: Where .env is looked up within a directory, in order.
_ENV_SEARCH = (".env", ".env.local")

#: A user-level config location, consulted before walking up from the cwd.
#: Needed because vidvoice is normally installed as a global command, so its
#: working directory is wherever the user happens to be -- not the project.
USER_CONFIG_DIR = Path.home() / ".config" / "vidvoice"


def default_work_dir() -> Path:
    """Where intermediates live when VIDVOICE_WORK_DIR is not set.

    vidvoice is normally installed as a global command, so defaulting to
    ``./vidvoice-work`` would scatter a work directory into every directory it
    is ever run from. A stable per-user location keeps the filesystem tidy and
    makes the clip cache actually reusable across runs.

    ``XDG_CACHE_HOME`` is honoured on Linux; macOS and Windows get the
    conventional per-user cache directory.
    """
    override = os.environ.get("XDG_CACHE_HOME")
    if override:
        return Path(override).expanduser() / "vidvoice"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "vidvoice"
    if os.name == "nt":  # pragma: no cover - platform specific
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "vidvoice" / "Cache"
    return Path.home() / ".cache" / "vidvoice"


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict.

    Supports:
      * ``KEY=value``
      * ``export KEY=value``
      * ``#`` comments and blank lines
      * single- or double-quoted values (which may contain ``#``)
      * ``\\n`` escape sequences inside double quotes

    Values are NOT interpolated (no ``${OTHER}`` expansion) -- keep it simple
    and predictable.
    """
    result: dict[str, str] = {}
    if not path.is_file():
        return result

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            quote = value[0]
            value = value[1:-1]
            if quote == '"':
                value = value.replace("\\n", "\n").replace('\\"', '"')
        elif " #" in value:
            # Unquoted trailing comment.
            value = value.split(" #", 1)[0].strip()

        result[key] = value

    return result


def find_env_file(start: Path | None = None) -> Path | None:
    """Locate the .env file to use.

    Checked in order:

    1. ``$VIDVOICE_ENV_FILE``, if set -- an explicit pointer always wins.
    2. ``~/.config/vidvoice/.env`` -- the user-level location. This is checked
       before the upward search because vidvoice is typically installed as a
       global command, so its working directory is wherever you happen to be,
       and walking up from there would usually find nothing.
    3. ``.env`` / ``.env.local``, searching upwards from *start* (the working
       directory by default), so a project-local .env still works when you run
       vidvoice from inside that project.
    """
    explicit = os.environ.get("VIDVOICE_ENV_FILE")
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate

    user_level = USER_CONFIG_DIR / ".env"
    if user_level.is_file():
        return user_level

    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        for name in _ENV_SEARCH:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


@dataclass(slots=True)
class Settings:
    """Resolved runtime settings. Immutable in spirit; use :meth:`with_overrides`."""

    gemini_api_key: str = ""
    cartesia_api_key: str = ""

    gemini_model: str = DEFAULT_GEMINI_MODEL
    cartesia_model: str = DEFAULT_CARTESIA_MODEL
    cartesia_version: str = DEFAULT_CARTESIA_VERSION
    cartesia_voice_id: str = DEFAULT_CARTESIA_VOICE

    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS

    #: Cartesia speaking rate, 0.6-1.5 typical. 1.0 = natural.
    speech_speed: float = 1.0
    #: Voice volume multiplier, 0.5-2.0.
    speech_volume: float = 1.0
    #: Default emotion from the Cartesia palette; empty means "let the model
    #: interpret the emotional subtext of the transcript".
    emotion: str = ""
    #: Narration pace used to derive a word budget from the video length.
    target_wpm: float = DEFAULT_TARGET_WPM
    #: exact | natural | off -- see vidvoice.timing.
    fit_mode: str = "exact"

    #: Directory for intermediate artefacts (per-run subdirectories).
    work_dir: Path = field(default_factory=default_work_dir)

    #: Source of the .env file, for `vidvoice doctor` to report.
    env_file: Path | None = None

    #: Populated only in dry-run mode; replaces real API calls.
    dry_run: bool = False

    # -- construction -------------------------------------------------------

    @classmethod
    def load(
        cls,
        env_file: Path | str | None = None,
        *,
        use_process_env: bool = True,
        work_dir: Path | str | None = None,
        **overrides: Any,
    ) -> Settings:
        """Build settings from .env + os.environ + explicit keyword overrides.

        Keyword overrides are applied last, so a CLI flag always beats both the
        environment and the .env file.
        """
        explicit = Path(env_file).expanduser() if env_file else find_env_file()
        file_values = parse_env_file(explicit) if explicit else {}

        def lookup(name: str, default: Any) -> Any:
            if use_process_env and name in os.environ and os.environ[name] != "":
                return os.environ[name]
            if name in file_values and file_values[name] != "":
                return file_values[name]
            return default

        settings = cls(
            gemini_api_key=lookup("GEMINI_API_KEY", "") or lookup("GOOGLE_API_KEY", ""),
            cartesia_api_key=lookup("CARTESIA_API_KEY", ""),
            gemini_model=lookup("VIDVOICE_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            cartesia_model=lookup("VIDVOICE_CARTESIA_MODEL", DEFAULT_CARTESIA_MODEL),
            cartesia_version=lookup("VIDVOICE_CARTESIA_VERSION", DEFAULT_CARTESIA_VERSION),
            cartesia_voice_id=lookup("VIDVOICE_CARTESIA_VOICE", DEFAULT_CARTESIA_VOICE),
            sample_rate=int(lookup("VIDVOICE_SAMPLE_RATE", DEFAULT_SAMPLE_RATE)),
            channels=int(lookup("VIDVOICE_CHANNELS", DEFAULT_CHANNELS)),
            speech_speed=float(lookup("VIDVOICE_SPEECH_SPEED", 1.0)),
            speech_volume=float(lookup("VIDVOICE_SPEECH_VOLUME", 1.0)),
            emotion=lookup("VIDVOICE_EMOTION", ""),
            target_wpm=float(lookup("VIDVOICE_TARGET_WPM", DEFAULT_TARGET_WPM)),
            fit_mode=lookup("VIDVOICE_FIT_MODE", "exact"),
            work_dir=Path(lookup("VIDVOICE_WORK_DIR", default_work_dir())).expanduser(),
            env_file=explicit if explicit and explicit.is_file() else None,
        )

        if work_dir is not None:
            overrides["work_dir"] = Path(work_dir).expanduser()

        return settings.with_overrides(**overrides) if overrides else settings

    def with_overrides(self, **overrides: Any) -> Settings:
        """Return a copy with non-None overrides applied."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        unknown = set(clean) - {f.name for f in self.__dataclass_fields__.values()}
        if unknown:
            raise ConfigError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
        return replace(self, **clean)

    # -- validation ---------------------------------------------------------

    def require_gemini_key(self) -> str:
        if not self.gemini_api_key:
            raise ConfigError(
                "GEMINI_API_KEY is not set.",
                hint=(
                    "Get a key at https://aistudio.google.com/apikey then put "
                    "GEMINI_API_KEY=... in your .env (see .env.example). "
                    "Or pass --dry-run to exercise the pipeline without API keys."
                ),
            )
        return self.gemini_api_key

    def require_cartesia_key(self) -> str:
        if not self.cartesia_api_key:
            raise ConfigError(
                "CARTESIA_API_KEY is not set.",
                hint=(
                    "Get a key at https://play.cartesia.ai/keys then put "
                    "CARTESIA_API_KEY=... in your .env (see .env.example). "
                    "Or pass --dry-run to exercise the pipeline without API keys."
                ),
            )
        return self.cartesia_api_key

    def describe(self) -> dict[str, Any]:
        """Redacted view, safe to print or log."""

        def mask(secret: str) -> str:
            if not secret:
                return "(unset)"
            if len(secret) <= 8:
                return "*" * len(secret)
            return f"{secret[:4]}...{secret[-4:]}"

        return {
            "env_file": str(self.env_file) if self.env_file else "(none found)",
            "gemini_api_key": mask(self.gemini_api_key),
            "cartesia_api_key": mask(self.cartesia_api_key),
            "gemini_model": self.gemini_model,
            "cartesia_model": self.cartesia_model,
            "cartesia_version": self.cartesia_version,
            "cartesia_voice_id": self.cartesia_voice_id,
            "emotion": self.emotion or "(model decides)",
            "target_wpm": self.target_wpm,
            "fit_mode": self.fit_mode,
            "speech_speed": self.speech_speed,
            "speech_volume": self.speech_volume,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "work_dir": str(self.work_dir),
            "dry_run": self.dry_run,
        }
