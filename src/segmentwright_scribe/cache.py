from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TranscriptCacheKey:
    source_id: str
    chunk_start_sec: float
    chunk_end_sec: float
    chunk_audio_hash: str
    model: str
    language: str
    audio_format: str = "wav"
    request_profile: str = "openrouter_audio_transcriptions_v1"

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedTranscript:
    text: str
    metadata: dict[str, Any]
    path: Path


class TranscriptCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def path_for(self, key: TranscriptCacheKey) -> Path:
        return self.cache_dir / "openrouter_transcripts" / f"{key.digest()}.json"

    def load(self, key: TranscriptCacheKey) -> CachedTranscript | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        text = data.get("text")
        if not isinstance(text, str):
            return None
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        return CachedTranscript(text=text, metadata=metadata, path=path)

    def save(
        self,
        key: TranscriptCacheKey,
        *,
        text: str,
        metadata: dict[str, Any],
    ) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "segmentwright_scribe.openrouter_transcript_cache.v1",
            "key": asdict(key),
            "metadata": metadata,
            "text": text,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
