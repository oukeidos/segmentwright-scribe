from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class TranscriptChunksError(ValueError):
    """Raised when a transcript chunks document is invalid."""


class TranscriptChunkModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    start: float
    end: float
    text: str
    source: str | None = None
    confidence: float | None = None
    notes: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class TranscriptChunksModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_: str | None = Field(default=None, alias="schema")
    version: int
    language: str | None = None
    metadata: dict[str, Any] | None = None
    chunks: list[TranscriptChunkModel]


@dataclass(frozen=True)
class TranscriptChunksValidation:
    warnings: list[str]
    chunk_warnings: dict[str, list[str]]

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings or self.chunk_warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "segmentwright_scribe.transcript_chunks_validation.v1",
            "warnings": self.warnings,
            "chunk_warnings": self.chunk_warnings,
            "has_warnings": self.has_warnings,
        }


def write_transcript_chunks(
    path: Path,
    *,
    language: str,
    metadata: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema": "segmentwright.transcript_chunks.v1",
        "version": 1,
        "language": language,
        "metadata": metadata,
        "chunks": chunks,
    }
    validate_transcript_chunks_payload(
        payload,
        duration_sec=max((float(chunk["end"]) for chunk in chunks), default=0.0),
        supported_languages={language},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_transcript_chunks_payload(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TranscriptChunksError(f"Transcript file is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise TranscriptChunksError(f"Could not read transcript file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TranscriptChunksError("Transcript file must contain a JSON object")
    validate_model(raw)
    return raw


def validate_model(payload: dict[str, Any]) -> TranscriptChunksModel:
    try:
        model = TranscriptChunksModel.model_validate(payload)
    except ValidationError as exc:
        raise TranscriptChunksError(f"Transcript file schema validation failed: {exc}") from exc
    if model.version != 1:
        raise TranscriptChunksError(f"Unsupported transcript version: {model.version!r}")
    return model


def validate_transcript_chunks_payload(
    payload: dict[str, Any],
    *,
    duration_sec: float,
    supported_languages: set[str],
    warn_chunk_sec: float = 180.0,
    strict: bool = False,
    epsilon_sec: float = 0.05,
) -> TranscriptChunksValidation:
    model = validate_model(payload)
    errors: list[str] = []
    warnings: list[str] = []
    chunk_warnings: dict[str, list[str]] = {}

    if model.language is None:
        errors.append("Transcript language is required")
    elif model.language not in supported_languages:
        supported = ", ".join(sorted(supported_languages))
        errors.append(f"Unsupported transcript language {model.language!r}; expected: {supported}")

    if not model.chunks:
        errors.append("Transcript file must contain at least one chunk")

    seen_ids: set[str] = set()
    previous_end: float | None = None
    empty_count = 0
    for index, chunk in enumerate(model.chunks, start=1):
        chunk_id = chunk.id or f"chunk_{index:04d}"
        chunk_errors: list[str] = []
        local_warnings: list[str] = []
        if chunk_id in seen_ids:
            chunk_errors.append(f"Duplicate chunk id: {chunk_id}")
        seen_ids.add(chunk_id)

        if not math.isfinite(chunk.start) or not math.isfinite(chunk.end):
            chunk_errors.append("Chunk start/end must be finite numbers")
        elif chunk.end <= chunk.start:
            chunk_errors.append("Chunk end must be greater than start")
        else:
            if chunk.start < -epsilon_sec or chunk.end > duration_sec + epsilon_sec:
                chunk_errors.append(
                    f"Chunk range {chunk.start:.3f}-{chunk.end:.3f}s is outside "
                    f"media duration {duration_sec:.3f}s"
                )
            if previous_end is not None and chunk.start < previous_end - epsilon_sec:
                chunk_errors.append(f"Chunk overlaps previous end {previous_end:.3f}s")
            if previous_end is not None and chunk.start > previous_end + epsilon_sec:
                local_warnings.append(
                    f"gap_from_previous_chunk_sec={chunk.start - previous_end:.3f}"
                )
            if chunk.end - chunk.start > warn_chunk_sec:
                local_warnings.append(
                    f"chunk_duration_sec={chunk.end - chunk.start:.3f} exceeds "
                    f"warning threshold {warn_chunk_sec:.3f}s"
                )
            if chunk.text.strip() == "":
                empty_count += 1
        previous_end = chunk.end
        if chunk_errors:
            errors.extend(f"{chunk_id}: {error}" for error in chunk_errors)
        if local_warnings:
            chunk_warnings[chunk_id] = local_warnings

    if model.chunks and empty_count == len(model.chunks):
        warnings.append("All transcript chunks are empty")

    if strict and (warnings or chunk_warnings):
        errors.extend(warnings)
        for chunk_id, items in chunk_warnings.items():
            errors.extend(f"{chunk_id}: {warning}" for warning in items)

    if errors:
        raise TranscriptChunksError("; ".join(errors))
    return TranscriptChunksValidation(warnings=warnings, chunk_warnings=chunk_warnings)
