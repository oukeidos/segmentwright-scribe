from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

from . import __version__
from .defaults import (
    DEFAULT_ENV_FILE,
    DEFAULT_FIRE_RED_VAD_MODEL_DIR,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CHUNK_SEC,
    DEFAULT_MIN_RETRY_SPLIT_SEC,
    DEFAULT_MODEL,
    DEFAULT_OPENROUTER_API_KEY_ENV,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_REQUEST_RETRIES,
    DEFAULT_TARGET_CHUNK_SEC,
    SUPPORTED_INPUT_EXTENSIONS,
    SUPPORTED_LANGUAGES,
)


class ConfigError(ValueError):
    """Raised when CLI configuration is invalid."""


@dataclass(frozen=True)
class AppConfig:
    input_path: Path
    output_path: Path
    artifacts_dir: Path
    env_file: Path
    language: str
    model: str
    openrouter_base_url: str
    openrouter_api_key_env: str
    firered_vad_model_dir: Path
    firered_vad_auto_download: bool
    device: str
    target_chunk_sec: float
    max_chunk_sec: float
    min_retry_split_sec: float
    timeout_sec: float | None
    max_request_retries: int
    cache_dir: Path
    cache_enabled: bool
    overwrite: bool
    verbose: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segmentwright-scribe",
        description=(
            "Generate Segmentwright-compatible transcript chunk JSON. Required inputs are the "
            "positional INPUT media file and OUTPUT .json path; every flag below is optional."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Credential default: the API key is read from the environment variable named by "
            "--openrouter-api-key-env. If it is not in the process environment, the same name is "
            "looked up in --env-file."
        ),
    )
    required = parser.add_argument_group("required positional arguments")
    optional = parser.add_argument_group("optional flags")
    optional.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the installed package version and exit.",
    )
    required.add_argument(
        "input",
        type=Path,
        metavar="INPUT",
        help="Required media file to transcribe.",
    )
    required.add_argument(
        "output",
        type=Path,
        metavar="OUTPUT",
        help="Required Segmentwright transcript JSON path to write; must end with .json.",
    )
    optional.add_argument(
        "--language",
        choices=sorted(SUPPORTED_LANGUAGES),
        default=DEFAULT_LANGUAGE,
        help="Language hint sent to the ASR model.",
    )
    optional.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter ASR model id.")
    optional.add_argument(
        "--artifacts",
        type=Path,
        help="Directory for run artifacts; omitted means artifacts/<input-stem>-<timestamp>.",
    )
    optional.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Dotenv file used after process environment lookup.",
    )
    optional.add_argument(
        "--openrouter-api-key-env",
        default=DEFAULT_OPENROUTER_API_KEY_ENV,
        help="Environment variable name that contains the OpenRouter API key.",
    )
    optional.add_argument(
        "--openrouter-base-url",
        default=DEFAULT_OPENROUTER_BASE_URL,
        help="OpenRouter API base URL.",
    )
    optional.add_argument(
        "--target-chunk-sec",
        type=float,
        default=DEFAULT_TARGET_CHUNK_SEC,
        help="Preferred chunk duration for the Segmentwright-equivalent planner.",
    )
    optional.add_argument(
        "--max-chunk-sec",
        type=float,
        default=DEFAULT_MAX_CHUNK_SEC,
        help="Hard maximum chunk duration before forced splitting.",
    )
    optional.add_argument(
        "--min-retry-split-sec",
        type=float,
        default=DEFAULT_MIN_RETRY_SPLIT_SEC,
        help="Minimum duration considered when forcing fallback chunk splits.",
    )
    optional.add_argument(
        "--firered-vad-model-dir",
        type=Path,
        default=DEFAULT_FIRE_RED_VAD_MODEL_DIR,
        help="Directory containing FireRedVAD model files.",
    )
    optional.add_argument(
        "--no-firered-vad-auto-download",
        action="store_true",
        help="Fail if FireRedVAD files are missing instead of downloading them.",
    )
    optional.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device used by FireRedVAD.",
    )
    optional.add_argument(
        "--timeout-sec",
        type=float,
        default=None,
        help="Per-request OpenRouter timeout in seconds; omitted uses httpx default behavior.",
    )
    optional.add_argument(
        "--max-request-retries",
        type=int,
        default=DEFAULT_REQUEST_RETRIES,
        help="Retry count for transient OpenRouter failures.",
    )
    optional.add_argument(
        "--cache-dir",
        type=Path,
        help="Transcript cache directory; omitted means <artifacts>/cache.",
    )
    optional.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable transcript cache reads and writes.",
    )
    optional.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing OUTPUT file.",
    )
    optional.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress details to stderr.",
    )
    return parser


def build_config(args: argparse.Namespace, *, now: datetime | None = None) -> AppConfig:
    input_path = normalize_path(args.input)
    output_path = normalize_path(args.output)
    artifacts_dir = normalize_path(args.artifacts) if args.artifacts else default_artifacts_dir(
        input_path, now=now
    )
    cache_dir = normalize_path(args.cache_dir) if args.cache_dir else artifacts_dir / "cache"
    config = AppConfig(
        input_path=input_path,
        output_path=output_path,
        artifacts_dir=artifacts_dir,
        env_file=normalize_path(args.env_file),
        language=args.language,
        model=args.model,
        openrouter_base_url=args.openrouter_base_url.rstrip("/"),
        openrouter_api_key_env=args.openrouter_api_key_env,
        firered_vad_model_dir=normalize_path(args.firered_vad_model_dir),
        firered_vad_auto_download=not args.no_firered_vad_auto_download,
        device=args.device,
        target_chunk_sec=args.target_chunk_sec,
        max_chunk_sec=args.max_chunk_sec,
        min_retry_split_sec=args.min_retry_split_sec,
        timeout_sec=args.timeout_sec,
        max_request_retries=args.max_request_retries,
        cache_dir=cache_dir,
        cache_enabled=not args.no_cache,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    validate_config(config)
    return config


def normalize_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return Path(os.path.realpath(expanded))
    return Path(os.path.realpath(Path.cwd() / expanded))


def default_artifacts_dir(input_path: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    stem = input_path.stem or "input"
    return normalize_path(Path("artifacts") / f"{stem}-{stamp}")


def validate_config(config: AppConfig) -> None:
    if config.language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ConfigError(f"Invalid language {config.language!r}; expected one of: {supported}")
    if not config.model.strip():
        raise ConfigError("--model must not be empty")
    if not config.openrouter_api_key_env.strip():
        raise ConfigError("--openrouter-api-key-env must not be empty")
    if not config.input_path.exists():
        raise ConfigError(f"Input file does not exist: {config.input_path}")
    if not config.input_path.is_file():
        raise ConfigError(f"Input path must be a file: {config.input_path}")
    suffix = config.input_path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_INPUT_EXTENSIONS))
        raise ConfigError(f"Unsupported input extension {suffix!r}; supported: {supported}")
    validate_path_collisions(config)
    if config.output_path.suffix.lower() != ".json":
        raise ConfigError(f"Output path must end with .json: {config.output_path}")
    if config.output_path.exists() and config.output_path.is_dir():
        raise ConfigError(f"Output path is a directory: {config.output_path}")
    if config.output_path.exists() and not config.overwrite:
        raise ConfigError(f"Output file already exists; pass --overwrite: {config.output_path}")
    if config.artifacts_dir.exists() and config.artifacts_dir.is_file():
        raise ConfigError(f"Artifacts path is an existing file: {config.artifacts_dir}")
    if config.target_chunk_sec <= 0:
        raise ConfigError("--target-chunk-sec must be greater than 0")
    if config.max_chunk_sec <= 0:
        raise ConfigError("--max-chunk-sec must be greater than 0")
    if config.min_retry_split_sec <= 0:
        raise ConfigError("--min-retry-split-sec must be greater than 0")
    if config.max_chunk_sec < config.min_retry_split_sec:
        raise ConfigError("--max-chunk-sec must be >= --min-retry-split-sec")
    if config.timeout_sec is not None and config.timeout_sec <= 0:
        raise ConfigError("--timeout-sec must be greater than 0")
    if config.max_request_retries < 0:
        raise ConfigError("--max-request-retries must be >= 0")


def validate_path_collisions(config: AppConfig) -> None:
    input_path = config.input_path
    output_path = config.output_path
    artifacts_dir = config.artifacts_dir
    cache_dir = config.cache_dir

    if same_path(input_path, output_path):
        raise ConfigError("Input and output paths must not refer to the same file")
    if same_path(artifacts_dir, output_path):
        raise ConfigError("--artifacts must not be the same path as OUTPUT")
    if same_path(cache_dir, output_path):
        raise ConfigError("--cache-dir must not be the same path as OUTPUT")
    if path_is_relative_to(output_path, cache_dir) or path_is_relative_to(cache_dir, output_path):
        raise ConfigError("--cache-dir must not contain OUTPUT or be inside OUTPUT")

    reserved_artifact_output = artifacts_dir / "transcript_chunks.json"
    if path_is_relative_to(output_path, artifacts_dir) and same_path(
        output_path, reserved_artifact_output
    ):
        # This is allowed. The pipeline reserves this path and does not write a second copy.
        pass


def same_path(left: Path, right: Path) -> bool:
    if left.exists() and right.exists():
        try:
            return left.samefile(right)
        except OSError:
            pass
    return Path(os.path.realpath(left)) == Path(os.path.realpath(right))


def path_is_relative_to(path: Path, maybe_parent: Path) -> bool:
    try:
        path.relative_to(maybe_parent)
        return True
    except ValueError:
        return False


def resolve_openrouter_api_key(config: AppConfig) -> str:
    env_values = load_env_file(config.env_file)
    value = os.environ.get(config.openrouter_api_key_env)
    if value is None:
        value = env_values.get(config.openrouter_api_key_env)
    if value is None or not value.strip():
        raise ConfigError(
            f"OpenRouter API key not found; expected non-empty "
            f"{config.openrouter_api_key_env} in environment or {config.env_file}"
        )
    return value.strip()


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values = dotenv_values(path)
    return {key: value for key, value in values.items() if value is not None}
