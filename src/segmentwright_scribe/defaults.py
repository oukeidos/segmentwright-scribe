from __future__ import annotations

from pathlib import Path

SUPPORTED_INPUT_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".mp3",
    ".m4a",
    ".wav",
}

SUPPORTED_LANGUAGES = {"ja", "en"}

DEFAULT_LANGUAGE = "ja"
DEFAULT_MODEL = "microsoft/mai-transcribe-2"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_ENV_FILE = Path("~/.segmentwright-scribe/.env")
DEFAULT_FIRE_RED_VAD_MODEL_DIR = Path("~/.cache/segmentwright-scribe/FireRedVAD/VAD")

DEFAULT_TARGET_CHUNK_SEC = 180.0
DEFAULT_MAX_CHUNK_SEC = 180.0
DEFAULT_MIN_RETRY_SPLIT_SEC = 30.0
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_REQUEST_RETRIES = 3
