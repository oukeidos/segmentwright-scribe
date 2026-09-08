from __future__ import annotations

import json
import traceback
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

from .cache import TranscriptCache, TranscriptCacheKey
from .chunking import ChunkPlannerConfig, PlannedChunk, plan_chunks
from .config import AppConfig, resolve_openrouter_api_key
from .media import hash_file, normalize_to_wav, probe_duration, require_ffmpeg, split_wav
from .model_download import ensure_firered_vad_model
from .openrouter import AsrRequest, OpenRouterTranscriber
from .transcript_chunks import validate_transcript_chunks_payload, write_transcript_chunks
from .vad import FireRedVadBackend


def run_pipeline(config: AppConfig) -> None:
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    stale_error = config.artifacts_dir / "errors.json"
    if stale_error.exists():
        stale_error.unlink()
    try:
        _run_pipeline(config)
    except Exception as exc:
        write_json(
            config.artifacts_dir / "errors.json",
            {
                "schema": "segmentwright_scribe.error.v1",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _run_pipeline(config: AppConfig) -> None:
    require_ffmpeg()
    api_key = resolve_openrouter_api_key(config)
    model_dir = ensure_firered_vad_model(
        config.firered_vad_model_dir,
        auto_download=config.firered_vad_auto_download,
    )
    write_json(config.artifacts_dir / "run_config.json", build_run_config(config))

    normalized_wav = config.artifacts_dir / "source.wav"
    normalize_to_wav(config.input_path, normalized_wav)
    source_id = hash_file(normalized_wav)
    duration_sec = probe_duration(normalized_wav)
    write_json(
        config.artifacts_dir / "media.json",
        {
            "schema": "segmentwright_scribe.media.v1",
            "input_path": str(config.input_path),
            "normalized_wav": str(normalized_wav),
            "source_id": source_id,
            "duration_sec": duration_sec,
        },
    )

    vad_backend = FireRedVadBackend(model_dir=model_dir, device=config.device)
    try:
        vad_result = vad_backend.detect(normalized_wav)
    finally:
        vad_backend.close()
    write_json(config.artifacts_dir / "vad.json", serializable(vad_result))

    chunk_plan = plan_chunks(
        duration_sec=duration_sec,
        speech_segments=vad_result.speech_segments,
        source_id=source_id,
        config=ChunkPlannerConfig(
            target_chunk_sec=config.target_chunk_sec,
            max_chunk_sec=config.max_chunk_sec,
            min_chunk_sec=config.min_retry_split_sec,
        ),
    )
    chunk_audio_paths = split_wav(
        normalized_wav,
        chunk_plan.split_times,
        config.artifacts_dir / "chunks",
    )
    chunks = attach_audio_paths(chunk_plan.chunks, chunk_audio_paths)
    write_json(
        config.artifacts_dir / "chunk_manifest.json",
        {
            "schema": "segmentwright_scribe.chunk_manifest.v1",
            "split_times": chunk_plan.split_times,
            "forced_count": chunk_plan.forced_count,
            "candidates": [serializable(candidate) for candidate in chunk_plan.candidates],
            "chunks": [serializable(chunk) for chunk in chunks],
        },
    )

    cache = TranscriptCache(config.cache_dir)
    transcriber = OpenRouterTranscriber(
        api_key=api_key,
        base_url=config.openrouter_base_url,
        max_retries=config.max_request_retries,
    )
    try:
        transcript_chunks = transcribe_chunks(
            config=config,
            source_id=source_id,
            chunks=chunks,
            cache=cache,
            transcriber=transcriber,
        )
    finally:
        transcriber.close()

    metadata = {
        "producer": "segmentwright-scribe",
        "backend": "openrouter",
        "model": config.model,
        "source_id": source_id,
        "input_path": str(config.input_path),
        "artifacts_dir": str(config.artifacts_dir),
    }
    payload = write_transcript_chunks(
        config.output_path,
        language=config.language,
        metadata=metadata,
        chunks=transcript_chunks,
    )
    validate_transcript_chunks_payload(
        payload,
        duration_sec=duration_sec,
        supported_languages={config.language},
        warn_chunk_sec=config.max_chunk_sec,
    )
    artifact_transcript = config.artifacts_dir / "transcript_chunks.json"
    if not same_path(config.output_path, artifact_transcript):
        write_json(artifact_transcript, payload)
    write_json(
        config.artifacts_dir / "run_summary.json",
        {
            "schema": "segmentwright_scribe.run_summary.v1",
            "status": "success",
            "input_path": str(config.input_path),
            "output_path": str(config.output_path),
            "artifacts_dir": str(config.artifacts_dir),
            "chunk_count": len(transcript_chunks),
            "model": config.model,
            "language": config.language,
        },
    )


def transcribe_chunks(
    *,
    config: AppConfig,
    source_id: str,
    chunks: list[PlannedChunk],
    cache: TranscriptCache,
    transcriber: OpenRouterTranscriber,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    calls_dir = config.artifacts_dir / "openrouter_calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    for index, chunk in enumerate(chunks, start=1):
        if chunk.audio_path is None:
            raise ValueError(f"Chunk has no audio path: {chunk.chunk_id}")
        public_id = f"chunk_{index:04d}"
        audio_hash = hash_file(chunk.audio_path)
        key = TranscriptCacheKey(
            source_id=source_id,
            chunk_start_sec=chunk.start_sec,
            chunk_end_sec=chunk.end_sec,
            chunk_audio_hash=audio_hash,
            model=config.model,
            language=config.language,
        )
        cached = cache.load(key) if config.cache_enabled else None
        call_path = calls_dir / f"{public_id}.json"
        if cached is not None:
            text = cached.text
            call_artifact = {
                "schema": "segmentwright_scribe.openrouter_call.v1",
                "request_id": public_id,
                "status": "success",
                "cache_hit": True,
                "cache_path": str(cached.path),
                "audio_path": str(chunk.audio_path),
                "chunk": chunk_metadata(chunk),
                "request": request_metadata(config=config, public_id=public_id),
                "response": {"text": text, "raw": cached.metadata.get("raw_response")},
                "timing": cached.metadata.get("timing", {}),
                "empty_text": text.strip() == "",
            }
        else:
            request = AsrRequest(
                request_id=public_id,
                audio_path=chunk.audio_path,
                model=config.model,
                language=config.language,
                timeout_sec=timeout_for_chunk(config, chunk),
            )
            result = transcriber.transcribe(request)
            text = result.text
            call_artifact = {
                "schema": "segmentwright_scribe.openrouter_call.v1",
                "request_id": public_id,
                "status": "success",
                "cache_hit": False,
                "audio_path": str(chunk.audio_path),
                "chunk": chunk_metadata(chunk),
                "request": request_metadata(config=config, public_id=public_id),
                "response": {"text": text, "raw": result.raw_response},
                "usage": result.raw_response.get("usage"),
                "timing": {"elapsed_sec": result.elapsed_sec},
                "http": {"status_code": result.status_code, "attempts": result.attempts},
                "empty_text": result.empty_text,
            }
            if config.cache_enabled:
                cache_path = cache.save(
                    key,
                    text=text,
                    metadata={
                        "raw_response": result.raw_response,
                        "timing": {"elapsed_sec": result.elapsed_sec},
                    },
                )
                call_artifact["cache_path"] = str(cache_path)
        write_json(call_path, call_artifact)
        output.append(
            {
                "id": public_id,
                "start": chunk.start_sec,
                "end": chunk.end_sec,
                "text": text,
                "source": "openrouter",
                "confidence": None,
                "metadata": {
                    "model": config.model,
                    "call_artifact": str(call_path.relative_to(config.artifacts_dir)),
                    "cache_hit": cached is not None,
                    "planner_chunk_id": chunk.chunk_id,
                },
            }
        )
    return output


def attach_audio_paths(chunks: list[PlannedChunk], audio_paths: list[Path]) -> list[PlannedChunk]:
    if len(chunks) != len(audio_paths):
        raise RuntimeError(
            f"Chunk/audio count mismatch: chunks={len(chunks)} audio={len(audio_paths)}"
        )
    return [
        replace(chunk, audio_path=audio_path)
        for chunk, audio_path in zip(chunks, audio_paths, strict=True)
    ]


def timeout_for_chunk(config: AppConfig, chunk: PlannedChunk) -> float:
    if config.timeout_sec is not None:
        return config.timeout_sec
    return max(300.0, chunk.duration_sec * 8.0 + 180.0)


def request_metadata(*, config: AppConfig, public_id: str) -> dict[str, Any]:
    return {
        "request_id": public_id,
        "model": config.model,
        "language": config.language,
        "base_url": config.openrouter_base_url,
        "api_key_env": config.openrouter_api_key_env,
        "audio_format": "wav",
    }


def chunk_metadata(chunk: PlannedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "start_sec": chunk.start_sec,
        "end_sec": chunk.end_sec,
        "duration_sec": chunk.duration_sec,
        "start_sample": chunk.start_sample,
        "end_sample": chunk.end_sample,
        "audio_path": None if chunk.audio_path is None else str(chunk.audio_path),
    }


def build_run_config(config: AppConfig) -> dict[str, Any]:
    return {
        "schema": "segmentwright_scribe.run_config.v1",
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "artifacts_dir": str(config.artifacts_dir),
        "cache_dir": str(config.cache_dir),
        "env_file": str(config.env_file),
        "language": config.language,
        "model": config.model,
        "openrouter_base_url": config.openrouter_base_url,
        "openrouter_api_key_env": config.openrouter_api_key_env,
        "firered_vad_model_dir": str(config.firered_vad_model_dir),
        "firered_vad_auto_download": config.firered_vad_auto_download,
        "device": config.device,
        "target_chunk_sec": config.target_chunk_sec,
        "max_chunk_sec": config.max_chunk_sec,
        "min_retry_split_sec": config.min_retry_split_sec,
        "timeout_sec": config.timeout_sec,
        "max_request_retries": config.max_request_retries,
        "cache_enabled": config.cache_enabled,
        "overwrite": config.overwrite,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serializable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def serializable(value: Any) -> Any:
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)
