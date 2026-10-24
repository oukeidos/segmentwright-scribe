"""Bounded local preset orchestration; ML dependencies live in isolated workers."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from ..media import hash_file, normalize_to_wav, probe_duration, require_ffmpeg
from ..transcript_chunks import validate_transcript_chunks_payload, write_transcript_chunks
from ..vad import FireRedVadBackend
from .runtime import POLICY, atomic_json, execute, home, load_installation, supported
from .worker_core import core, norm


def merge(base: dict, repairs: list[dict]) -> tuple[list[dict], list[dict]]:
    chunks = [dict(c) for c in base["chunks"]]
    edits: dict[int, list] = {}
    audit = []
    repaired = {r["candidate"]["id"]: r for r in repairs}
    for c in base["candidates"]:
        text = ""
        if c["insert_at"] is None:
            reason = "ambiguous_anchor"
        else:
            row = repaired[c["id"]]
            text, detail = core(row, c)
            value = norm(text)
            left, right = norm(c["left"]), norm(c["right"])
            reason = detail["reason"]
            if row["token_cap"]:
                reason = "token_cap"
            elif not reason:
                if len(value) < 2:
                    reason = "empty_or_short"
                elif len(value) > 20 * (c["end_sample"] - c["start_sample"]) / 16000:
                    reason = "excessive_rate"
                elif value in left + right:
                    reason = "duplicate"
                elif any(
                    value[:k] == left[-k:] for k in range(2, min(len(value), len(left)) + 1)
                ) or any(
                    value[-k:] == right[:k] for k in range(2, min(len(value), len(right)) + 1)
                ):
                    reason = "boundary_overlap"
            if not reason:
                edits.setdefault(c["chunk_index"], []).append((c["insert_at"], c["id"], text))
        audit.append({"id": c["id"], "accepted": reason is None, "reason": reason, "text": text})
    for i, chunk in enumerate(chunks):
        old = chunk["text"]
        cursor = 0
        parts = []
        for pos, _, text in sorted(edits.get(i, [])):
            if pos < cursor or pos > len(old):
                raise ValueError("Invalid local insertion anchor.")
            parts.extend([old[cursor:pos], "\n" + text + "\n"])
            cursor = pos
        parts.append(old[cursor:])
        combined = "".join(parts)
        chunk["text"] = "\n".join(
            s.strip() for s in re.split(r"(?<=[.!?。！？])\s*|\n+", combined) if s.strip()
        )
        if re.sub(r"\s+", "", combined) != re.sub(r"\s+", "", chunk["text"]):
            raise ValueError("Local formatting changed source text.")
        chunk["metadata"] = {"policy": POLICY, "repairs": len(edits.get(i, []))}
    return chunks, audit


def validate_coverage(chunks: list[dict], duration: float) -> None:
    if not chunks or abs(chunks[0]["start"]) > 0.001 or abs(chunks[-1]["end"] - duration) > 0.05:
        raise ValueError("Local chunks must cover the complete source duration.")
    for left, right in zip(chunks, chunks[1:], strict=False):
        if abs(left["end"] - right["start"]) > 0.001:
            raise ValueError("Local chunks must be contiguous in original source time.")


def run(config) -> None:
    supported()
    require_ffmpeg()
    if config.device == "cpu":
        raise ValueError("--preset local requires CUDA; use --device auto or cuda.")
    installation = load_installation()
    # Never allow job cleanup/output writes to collide with managed installations.
    root = home()
    for path in [config.input_path, config.output_path, config.artifacts_dir, config.cache_dir]:
        resolved = path.resolve()
        if resolved.is_relative_to(root) or root.is_relative_to(resolved):
            raise ValueError(
                "Input, output, artifacts and cache must not overlap the local runtime directory."
            )
    audio = config.artifacts_dir / "source.wav"
    normalize_to_wav(config.input_path, audio)
    duration = probe_duration(audio)
    source_id = hash_file(audio)
    key = {
        "source": source_id,
        "language": config.language,
        "profile": installation["profile"],
        "policy": POLICY,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    cache = config.cache_dir / "local_transcripts" / f"{digest}.json"
    cached = json.loads(cache.read_text()) if config.cache_enabled and cache.exists() else None
    if cached is not None:
        if cached.get("key") != key:
            raise ValueError("Local cache key mismatch.")
        chunks = cached["chunks"]
        audit = cached["audit"]
    else:
        vad = FireRedVadBackend(
            model_dir=Path(installation["models"]["vad"]["path"]) / "VAD", device="cpu"
        )
        print("Local preset: detecting speech...", file=sys.stderr)
        try:
            speech = [[s.start_sec, s.end_sec] for s in vad.detect(audio).speech_segments]
        finally:
            vad.close()
        atomic_json(config.artifacts_dir / "vad.json", {"speech": speech})
        job = {
            "audio": str(audio),
            "language": config.language,
            "models": installation["models"],
            "speech": speech,
        }
        directory = Path(installation["directory"])
        for worker in ("granite", "qwen"):
            print(f"Local preset: running {worker}...", file=sys.stderr)
            request = config.artifacts_dir / f"{worker}-request.json"
            output = config.artifacts_dir / f"{worker}-result.json"
            atomic_json(request, job)
            execute(
                [
                    str(directory / worker / "bin/python"),
                    "-I",
                    str(directory / f"worker_{worker}.py"),
                    str(request),
                    str(output),
                ],
                log=config.artifacts_dir / f"{worker}.log",
                timeout=config.timeout_sec or 3600,
            )
            result = json.loads(output.read_text())
            if worker == "granite":
                base = result
                job["candidates"] = base["candidates"]
            else:
                chunks, audit = merge(base, result)
        validate_transcript_chunks_payload(
            {"version": 1, "language": config.language, "chunks": chunks},
            duration_sec=duration,
            supported_languages={config.language},
        )
        validate_coverage(chunks, duration)
        if config.cache_enabled:
            atomic_json(cache, {"key": key, "chunks": chunks, "audit": audit})
    metadata = {
        "producer": "segmentwright-scribe",
        "backend": "local",
        "preset": "local",
        "policy": POLICY,
        "runtime_profile": installation["profile"],
        "source_id": source_id,
        "models": {
            k: {"repo": v["repo"], "revision": v["revision"]}
            for k, v in installation["models"].items()
        },
        "cache_hit": cached is not None,
        "beta": True,
    }
    payload = {"version": 1, "language": config.language, "chunks": chunks}
    validate_transcript_chunks_payload(
        payload, duration_sec=duration, supported_languages={config.language}
    )
    validate_coverage(chunks, duration)
    write_transcript_chunks(
        config.output_path, language=config.language, metadata=metadata, chunks=chunks
    )
    atomic_json(config.artifacts_dir / "repair-audit.json", audit)
    atomic_json(
        config.artifacts_dir / "run_summary.json",
        {
            **metadata,
            "status": "success",
            "duration_sec": duration,
            "chunk_count": len(chunks),
            "accepted_repairs": sum(a["accepted"] for a in audit),
            "candidate_count": len(audit),
        },
    )
    print(f"Local preset: wrote {len(chunks)} chunks to {config.output_path}", file=sys.stderr)
