from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from segmentwright_scribe.cache import TranscriptCache
from segmentwright_scribe.chunking import PlannedChunk
from segmentwright_scribe.config import build_config
from segmentwright_scribe.openrouter import AsrResult
from segmentwright_scribe.pipeline import timeout_for_chunk, transcribe_chunks


def make_config(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"wav")
    args = argparse.Namespace(
        input=input_path,
        output=tmp_path / "out.json",
        language="ja",
        model="microsoft/mai-transcribe-2",
        artifacts=tmp_path / "artifacts",
        env_file=tmp_path / ".env",
        openrouter_api_key_env="OPENROUTER_API_KEY",
        openrouter_base_url="https://openrouter.ai/api/v1",
        target_chunk_sec=180.0,
        max_chunk_sec=180.0,
        min_retry_split_sec=30.0,
        firered_vad_model_dir=tmp_path / "FireRedVAD" / "VAD",
        no_firered_vad_auto_download=False,
        device="auto",
        timeout_sec=None,
        max_request_retries=3,
        cache_dir=None,
        no_cache=False,
        overwrite=False,
        verbose=False,
    )
    return build_config(args, now=datetime(2026, 6, 27, 1, 2, 3))


def test_transcribe_chunks_writes_call_artifact_and_output_chunk(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"audio")
    chunk = PlannedChunk(
        chunk_id="1",
        parent_id="1",
        source_id="source",
        start_sec=0.0,
        end_sec=2.0,
        start_sample=0,
        end_sample=32_000,
        depth=0,
        retry_count=0,
        original_index=0,
        cut_reason="source_start",
        cut_quality=None,
        cut_source="source_start",
        audio_path=audio_path,
    )
    transcriber = FakeTranscriber()

    output = transcribe_chunks(
        config=config,
        source_id="source",
        chunks=[chunk],
        cache=TranscriptCache(config.cache_dir),
        transcriber=transcriber,
    )

    assert output == [
        {
            "id": "chunk_0001",
            "start": 0.0,
            "end": 2.0,
            "text": "hello",
            "source": "openrouter",
            "confidence": None,
            "metadata": {
                "model": "microsoft/mai-transcribe-2",
                "call_artifact": "openrouter_calls/chunk_0001.json",
                "cache_hit": False,
                "planner_chunk_id": "1",
            },
        }
    ]
    assert (config.artifacts_dir / "openrouter_calls/chunk_0001.json").exists()
    assert transcriber.requests[0].timeout_sec == 300.0


def test_timeout_for_chunk_uses_explicit_override(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config = config.__class__(**{**config.__dict__, "timeout_sec": 12.0})
    chunk = PlannedChunk(
        "1",
        "1",
        "source",
        0.0,
        100.0,
        0,
        1_600_000,
        0,
        0,
        0,
        "source_start",
        None,
        "source_start",
    )

    assert timeout_for_chunk(config, chunk) == 12.0


class FakeTranscriber:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def transcribe(self, request: Any) -> AsrResult:
        self.requests.append(request)
        return AsrResult(
            request_id=request.request_id,
            text="hello",
            raw_response={"text": "hello"},
            elapsed_sec=0.1,
            attempts=1,
            status_code=200,
        )
