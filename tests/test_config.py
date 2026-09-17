"""Configuration loading: .env parsing, precedence, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from vidvoice.config import (
    EMOTIONS,
    default_work_dir,
    PRIMARY_EMOTIONS,
    Settings,
    find_env_file,
    parse_env_file,
)
from vidvoice.errors import ConfigError


class TestParseEnvFile:
    def test_basic_pairs(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("A=1\nB=two\n")
        assert parse_env_file(path) == {"A": "1", "B": "two"}

    def test_ignores_comments_and_blanks(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("# a comment\n\nA=1\n   \n# another\nB=2\n")
        assert parse_env_file(path) == {"A": "1", "B": "2"}

    def test_handles_export_prefix(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("export A=1\n")
        assert parse_env_file(path) == {"A": "1"}

    def test_strips_quotes(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("A='single'\nB=\"double\"\n")
        assert parse_env_file(path) == {"A": "single", "B": "double"}

    def test_quoted_values_keep_hashes(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text('A="pass#word"\n')
        assert parse_env_file(path)["A"] == "pass#word"

    def test_unquoted_trailing_comment_is_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("A=value # trailing note\n")
        assert parse_env_file(path)["A"] == "value"

    def test_escapes_in_double_quotes(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text('A="line1\\nline2"\n')
        assert parse_env_file(path)["A"] == "line1\nline2"

    def test_empty_value(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("A=\n")
        assert parse_env_file(path)["A"] == ""

    def test_skips_lines_without_equals(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("GARBAGE\nA=1\n")
        assert parse_env_file(path) == {"A": "1"}

    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert parse_env_file(tmp_path / "nope") == {}


class TestFindEnvFile:
    """Resolution order matters: vidvoice is installed as a global command, so
    the working directory is not the project directory."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Point the user-level location somewhere empty and clear the override."""
        monkeypatch.delenv("VIDVOICE_ENV_FILE", raising=False)
        monkeypatch.setattr(
            "vidvoice.config.USER_CONFIG_DIR", tmp_path / "no-user-config"
        )
        return tmp_path

    def test_finds_in_current_directory(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("A=1\n")
        assert find_env_file(tmp_path) == tmp_path / ".env"

    def test_searches_upwards(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("A=1\n")
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        assert find_env_file(nested) == tmp_path / ".env"

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert find_env_file(tmp_path) is None

    def test_prefers_env_local_over_env(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("A=1\n")
        (tmp_path / ".env.local").write_text("A=2\n")
        assert find_env_file(tmp_path) == tmp_path / ".env"

    def test_user_level_config_wins_over_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running the global command from ~ must still find your config."""
        user_config = tmp_path / "user-config"
        user_config.mkdir()
        (user_config / ".env").write_text("A=user\n")
        monkeypatch.setattr("vidvoice.config.USER_CONFIG_DIR", user_config)

        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text("A=project\n")

        assert find_env_file(project) == user_config / ".env"

    def test_finds_user_config_when_no_project_env_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        user_config = tmp_path / "user-config"
        user_config.mkdir()
        (user_config / ".env").write_text("A=user\n")
        monkeypatch.setattr("vidvoice.config.USER_CONFIG_DIR", user_config)

        empty = tmp_path / "elsewhere"
        empty.mkdir()
        assert find_env_file(empty) == user_config / ".env"

    def test_explicit_override_wins_over_everything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        user_config = tmp_path / "user-config"
        user_config.mkdir()
        (user_config / ".env").write_text("A=user\n")
        monkeypatch.setattr("vidvoice.config.USER_CONFIG_DIR", user_config)

        explicit = tmp_path / "explicit.env"
        explicit.write_text("A=explicit\n")
        monkeypatch.setenv("VIDVOICE_ENV_FILE", str(explicit))

        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text("A=project\n")

        assert find_env_file(project) == explicit

    def test_ignores_a_missing_explicit_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bad VIDVOICE_ENV_FILE must fall through, not crash."""
        monkeypatch.setenv("VIDVOICE_ENV_FILE", str(tmp_path / "nope.env"))
        (tmp_path / ".env").write_text("A=1\n")
        assert find_env_file(tmp_path) == tmp_path / ".env"


class TestDefaultWorkDir:
    def test_is_a_stable_per_user_location(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A global command must not scatter work dirs into the cwd."""
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        resolved = default_work_dir()

        assert resolved.is_absolute()
        assert resolved != Path.cwd() / "vidvoice-work"
        assert "vidvoice" in str(resolved)

    def test_honours_xdg_cache_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        assert default_work_dir() == tmp_path / "cache" / "vidvoice"

    def test_is_not_shared_between_different_cwds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.chdir(tmp_path)
        first = Settings.load(env_file=tmp_path / "none", use_process_env=False).work_dir
        monkeypatch.chdir(tmp_path.parent if tmp_path.parent != tmp_path else tmp_path)
        second = Settings.load(env_file=tmp_path / "none", use_process_env=False).work_dir
        assert first == second

    def test_env_var_still_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "custom"
        monkeypatch.setenv("VIDVOICE_WORK_DIR", str(target))
        settings = Settings.load(env_file=tmp_path / "none")
        assert settings.work_dir == target


class TestSettingsLoad:
    def test_reads_from_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("GEMINI_API_KEY", "CARTESIA_API_KEY", "VIDVOICE_EMOTION"):
            monkeypatch.delenv(name, raising=False)

        path = tmp_path / ".env"
        path.write_text(
            "GEMINI_API_KEY=from-file\n"
            "CARTESIA_API_KEY=cart-file\n"
            "VIDVOICE_EMOTION=calm\n"
            "VIDVOICE_TARGET_WPM=160\n"
        )
        settings = Settings.load(env_file=path, use_process_env=False)

        assert settings.gemini_api_key == "from-file"
        assert settings.cartesia_api_key == "cart-file"
        assert settings.emotion == "calm"
        assert settings.target_wpm == 160.0

    def test_process_env_overrides_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / ".env"
        path.write_text("GEMINI_API_KEY=from-file\n")
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")

        assert Settings.load(env_file=path).gemini_api_key == "from-env"

    def test_google_api_key_is_accepted_as_a_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        path = tmp_path / ".env"
        path.write_text("GOOGLE_API_KEY=google-style\n")

        settings = Settings.load(env_file=path, use_process_env=False)
        assert settings.gemini_api_key == "google-style"

    def test_empty_env_values_do_not_override_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VIDVOICE_GEMINI_MODEL", raising=False)
        path = tmp_path / ".env"
        path.write_text("VIDVOICE_GEMINI_MODEL=\n")

        settings = Settings.load(env_file=path, use_process_env=False)
        assert settings.gemini_model == "gemini-3.8-flash"

    def test_defaults_are_current_api_values(self, tmp_path: Path) -> None:
        settings = Settings.load(env_file=tmp_path / "nonexistent", use_process_env=False)
        # Pinned to the verified 2026-08-14 Cartesia spec.
        assert settings.cartesia_version == "2026-08-14"
        assert settings.cartesia_model == "sonic-3.6"
        assert settings.gemini_model == "gemini-3.8-flash"
        assert settings.sample_rate in (8000, 16000, 22050, 24000, 44100, 48000)

    def test_explicit_overrides_win(self, tmp_path: Path) -> None:
        settings = Settings.load(
            env_file=tmp_path / "none", use_process_env=False, gemini_model="custom-model"
        )
        assert settings.gemini_model == "custom-model"

    def test_unknown_override_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="Unknown setting"):
            Settings.load(env_file=tmp_path / "none", nonsense_option=1)


class TestSettingsValidation:
    def test_require_gemini_key_raises_when_unset(self) -> None:
        with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
            Settings().require_gemini_key()

    def test_require_cartesia_key_raises_when_unset(self) -> None:
        with pytest.raises(ConfigError, match="CARTESIA_API_KEY"):
            Settings().require_cartesia_key()

    def test_hint_mentions_dry_run(self) -> None:
        """The error should tell people they can proceed without keys."""
        with pytest.raises(ConfigError) as excinfo:
            Settings().require_gemini_key()
        assert "--dry-run" in (excinfo.value.hint or "")

    def test_returns_the_key_when_set(self) -> None:
        assert Settings(gemini_api_key="k").require_gemini_key() == "k"


class TestSettingsDescribe:
    def test_masks_secrets(self) -> None:
        described = Settings(
            gemini_api_key="AIzaSyABCDEFGHIJKLMNOP", cartesia_api_key="sk_car_1234567890"
        ).describe()

        assert "AIzaSyABCDEFGHIJKLMNOP" not in str(described)
        assert "sk_car_1234567890" not in str(described)
        assert described["gemini_api_key"].startswith("AIza")
        assert described["gemini_api_key"].endswith("MNOP")

    def test_short_secrets_are_fully_masked(self) -> None:
        assert Settings(gemini_api_key="short").describe()["gemini_api_key"] == "*****"

    def test_reports_unset(self) -> None:
        assert Settings().describe()["gemini_api_key"] == "(unset)"

    def test_includes_the_new_settings(self) -> None:
        described = Settings(emotion="calm", target_wpm=140, fit_mode="natural").describe()
        assert described["emotion"] == "calm"
        assert described["target_wpm"] == 140
        assert described["fit_mode"] == "natural"

    def test_emotion_default_is_described_readably(self) -> None:
        assert Settings().describe()["emotion"] == "(model decides)"


class TestEmotionPalette:
    def test_full_palette_matches_the_api_spec(self) -> None:
        assert len(EMOTIONS) == 58

    def test_primaries_are_a_subset(self) -> None:
        assert set(PRIMARY_EMOTIONS).issubset(set(EMOTIONS))

    def test_documented_primary_six_are_present(self) -> None:
        for name in ("neutral", "calm", "angry", "content", "sad", "scared"):
            assert name in EMOTIONS
