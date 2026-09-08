from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pytest

from segmentwright_scribe.config import (
    ConfigError,
    build_config,
    build_parser,
    resolve_openrouter_api_key,
)


def make_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"wav")
    values: dict[str, object] = {
        "input": input_path,
        "output": tmp_path / "out.json",
        "language": "ja",
        "model": "microsoft/mai-transcribe-2",
        "artifacts": None,
        "env_file": tmp_path / ".env",
        "openrouter_api_key_env": "OPENROUTER_API_KEY",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "target_chunk_sec": 180.0,
        "max_chunk_sec": 180.0,
        "min_retry_split_sec": 30.0,
        "firered_vad_model_dir": tmp_path / "FireRedVAD" / "VAD",
        "no_firered_vad_auto_download": False,
        "device": "auto",
        "timeout_sec": None,
        "max_request_retries": 3,
        "cache_dir": None,
        "no_cache": False,
        "overwrite": False,
        "verbose": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_config_applies_defaults(tmp_path: Path) -> None:
    config = build_config(
        make_args(tmp_path),
        now=datetime(2026, 6, 27, 1, 2, 3),
    )

    assert config.language == "ja"
    assert config.model == "microsoft/mai-transcribe-2"
    assert config.openrouter_api_key_env == "OPENROUTER_API_KEY"
    assert config.artifacts_dir.name == "input-20260627-010203"
    assert config.cache_dir == config.artifacts_dir / "cache"


def test_help_text_distinguishes_required_inputs_and_optional_defaults() -> None:
    help_text = build_parser().format_help()

    assert "required positional arguments" in help_text
    assert "INPUT" in help_text
    assert "OUTPUT" in help_text
    assert "optional flags" in help_text
    assert "--model" in help_text
    assert "microsoft/mai-" in help_text
    assert "transcribe-2" in help_text
    assert "--openrouter-api-key-env" in help_text
    assert "OPENROUTER_API_KEY" in help_text
    assert "Credential default" in help_text


def test_input_output_same_file_is_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"wav")

    with pytest.raises(ConfigError, match="same file"):
        build_config(make_args(tmp_path, input=input_path, output=input_path))


def test_existing_output_requires_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "out.json"
    output_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigError, match="already exists"):
        build_config(make_args(tmp_path, output=output_path))

    assert build_config(make_args(tmp_path, output=output_path, overwrite=True)).overwrite is True


def test_output_must_not_collide_with_cache_dir(tmp_path: Path) -> None:
    output_path = tmp_path / "out.json"

    with pytest.raises(ConfigError, match="cache-dir"):
        build_config(make_args(tmp_path, output=output_path, cache_dir=output_path))


def test_artifacts_must_not_be_output_path(tmp_path: Path) -> None:
    output_path = tmp_path / "out.json"

    with pytest.raises(ConfigError, match="artifacts"):
        build_config(make_args(tmp_path, output=output_path, artifacts=output_path))


def test_resolve_api_key_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")
    config = build_config(make_args(tmp_path, env_file=env_file))

    assert resolve_openrouter_api_key(config) == "test-key"


def test_environment_wins_over_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "process-key")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=file-key\n", encoding="utf-8")
    config = build_config(make_args(tmp_path, env_file=env_file))

    assert resolve_openrouter_api_key(config) == "process-key"


def test_missing_api_key_fails_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = build_config(make_args(tmp_path))

    with pytest.raises(ConfigError, match="OpenRouter API key not found"):
        resolve_openrouter_api_key(config)


def test_parser_selects_new_default_and_allows_previous_model():
    parser = build_parser()
    assert parser.parse_args(["input.wav", "out.json"]).model == "microsoft/mai-transcribe-2"
    assert parser.parse_args([
        "input.wav", "out.json", "--model", "microsoft/mai-transcribe-1.5"
    ]).model == "microsoft/mai-transcribe-1.5"
