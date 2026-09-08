from __future__ import annotations

from pathlib import Path

import pytest

from segmentwright_scribe.transcript_chunks import (
    TranscriptChunksError,
    load_transcript_chunks_payload,
    validate_transcript_chunks_payload,
    write_transcript_chunks,
)


def test_write_transcript_chunks_outputs_segmentwright_schema(tmp_path: Path) -> None:
    path = tmp_path / "transcript.json"

    payload = write_transcript_chunks(
        path,
        language="ja",
        metadata={"producer": "segmentwright-scribe"},
        chunks=[{"id": "chunk_0001", "start": 0.0, "end": 1.0, "text": "こんにちは"}],
    )

    assert payload["schema"] == "segmentwright.transcript_chunks.v1"
    loaded = load_transcript_chunks_payload(path)
    assert loaded["language"] == "ja"
    assert loaded["chunks"][0]["text"] == "こんにちは"


def test_validate_transcript_chunks_rejects_overlap() -> None:
    payload = {
        "version": 1,
        "language": "en",
        "chunks": [
            {"id": "a", "start": 0.0, "end": 2.0, "text": "one"},
            {"id": "b", "start": 1.5, "end": 3.0, "text": "two"},
        ],
    }

    with pytest.raises(TranscriptChunksError, match="overlaps"):
        validate_transcript_chunks_payload(
            payload,
            duration_sec=3.0,
            supported_languages={"en"},
        )


def test_validate_transcript_chunks_warns_all_empty() -> None:
    payload = {
        "version": 1,
        "language": "en",
        "chunks": [{"id": "a", "start": 0.0, "end": 2.0, "text": ""}],
    }

    validation = validate_transcript_chunks_payload(
        payload,
        duration_sec=2.0,
        supported_languages={"en"},
    )

    assert validation.warnings == ["All transcript chunks are empty"]
