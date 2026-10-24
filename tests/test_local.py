"""Optional runtime isolation, safe routing, and repair regression tests."""

import importlib.abc
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from segmentwright_scribe.cli import main
from segmentwright_scribe.config import build_config, build_parser
from segmentwright_scribe.local.pipeline import merge
from segmentwright_scribe.local.runtime import atomic_json, clean_environment, load_installation
from segmentwright_scribe.local.worker_units import in_core, units_from_ids


@pytest.mark.parametrize("args", [["--help"], ["--list-presets"], ["--version"]])
def test_public_cli_never_imports_optional_runtime(tmp_path, args):
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.startswith(("segmentwright_scribe.local", "transformers", "qwen_asr", "nemo")):
   raise AssertionError("Optional import on cloud CLI: " + fullname)
sys.meta_path.insert(0, Block())
from segmentwright_scribe.cli import main
main(sys.argv[1:])
"""
    env = dict(os.environ, SEGMENTWRIGHT_SCRIBE_LOCAL_HOME=str(tmp_path / "absent"))
    result = subprocess.run(
        [sys.executable, "-c", script, *args], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "absent").exists()


def test_missing_doctor_is_actionable_and_read_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SEGMENTWRIGHT_SCRIBE_LOCAL_HOME", str(tmp_path / "missing"))
    assert main(["local", "doctor"]) == 1
    assert "local install" in capsys.readouterr().err
    assert not (tmp_path / "missing").exists()


def test_local_config_and_pipeline_do_not_require_cloud_key(tmp_path, monkeypatch):
    import segmentwright_scribe.local.pipeline as local
    import segmentwright_scribe.pipeline as pipeline

    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    config = build_config(
        build_parser().parse_args([str(source), str(tmp_path / "out.json"), "--preset", "local"])
    )
    monkeypatch.setattr(
        pipeline, "resolve_openrouter_api_key", lambda _: pytest.fail("cloud key read")
    )
    calls = []
    monkeypatch.setattr(local, "run", lambda config: calls.append(config.preset))
    pipeline.run_pipeline(config)
    assert calls == ["local"]
    assert not config.artifacts_dir.exists()


def test_cloud_transcription_with_local_runtime_blocked(tmp_path, monkeypatch):
    from test_pipeline import FakeTranscriber, make_config

    from segmentwright_scribe.cache import TranscriptCache
    from segmentwright_scribe.chunking import PlannedChunk
    from segmentwright_scribe.pipeline import transcribe_chunks

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, *args):
            if fullname.startswith(
                ("segmentwright_scribe.local", "qwen_asr", "transformers", "nemo")
            ):
                raise AssertionError(fullname)

    blocker = Block()
    monkeypatch.setenv("SEGMENTWRIGHT_SCRIBE_LOCAL_HOME", str(tmp_path / "uninstalled"))
    config = make_config(tmp_path)
    wav = tmp_path / "chunk.wav"
    wav.write_bytes(b"audio")
    chunk = PlannedChunk(
        "1", "1", "source", 0, 2, 0, 32000, 0, 0, 0, "source_start", None, "source_start", wav
    )
    sys.meta_path.insert(0, blocker)
    try:
        output = transcribe_chunks(
            config=config,
            source_id="source",
            chunks=[chunk],
            cache=TranscriptCache(config.cache_dir),
            transcriber=FakeTranscriber(),
        )
    finally:
        sys.meta_path.remove(blocker)
    assert output[0]["text"] == "hello"
    assert not (tmp_path / "uninstalled").exists()


def test_workers_receive_no_cloud_keys_or_python_overlay(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/unrelated/python")
    env = clean_environment()
    assert not {"OPENROUTER_API_KEY", "PYTHONPATH", "PYTHONHOME"} & env.keys()
    assert env["HF_HUB_OFFLINE"] == "1"


def test_old_profile_is_not_silently_used(tmp_path):
    atomic_json(tmp_path / "active.json", {"profile": "obsolete"})
    with pytest.raises(RuntimeError, match="outdated"):
        load_installation(tmp_path)


def test_lexical_timing_excludes_distant_punctuation_and_silent_interior():
    def decode(ids):
        return "".join({1: "Come", 2: ".", 3: "hel", 4: "lo", 5: ""}[i] for i in ids)

    full, units, error = units_from_ids([1] + [0] * 112 + [2], decode, "en", 0)
    assert full == "Come." and not error
    assert units[0]["end"] == 0.08 and not in_core(units[0], 3, 6)
    _, units, _ = units_from_ids([3] + [0] * 10 + [4], decode, "en", 0)
    assert not in_core(units[0], 0.3, 0.7)
    _, units, _ = units_from_ids([5, 0, 0, 1], decode, "en", 0)
    assert units[0]["start"] == 0.24


def test_merge_preserves_original_and_rejects_capped_output():
    candidate = {
        "id": 0,
        "chunk_index": 0,
        "insert_at": 5,
        "start_sample": 16000,
        "end_sample": 48000,
        "left": "Hello",
        "right": "world",
    }
    base = {"chunks": [{"text": "Helloworld", "start": 0, "end": 4}], "candidates": [candidate]}
    row = {
        "candidate": candidate,
        "text": "there",
        "start_sample": 0,
        "audio_duration": 4,
        "words": [{"text": "there", "start_time": 1, "end_time": 2}],
        "token_cap": False,
    }
    chunks, audit = merge(base, [row])
    assert chunks[0]["text"] == "Hello\nthere\nworld" and audit[0]["accepted"]
    chunks, audit = merge(base, [{**row, "token_cap": True}])
    assert chunks[0]["text"] == "Helloworld" and audit[0]["reason"] == "token_cap"
    assert base["chunks"][0]["text"] == "Helloworld"


def test_install_failure_keeps_active_manifest(tmp_path, monkeypatch):
    import segmentwright_scribe.local.runtime as runtime

    monkeypatch.setenv("SEGMENTWRIGHT_SCRIBE_LOCAL_HOME", str(tmp_path))
    monkeypatch.setattr(runtime, "supported", lambda: None)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/fake/uv")
    original = {"profile": "old", "directory": "old-directory"}
    atomic_json(tmp_path / "active.json", original)

    def fail(*args, **kwargs):
        raise RuntimeError("Simulated package installation failure")

    monkeypatch.setattr(runtime, "execute", fail)
    with pytest.raises(RuntimeError, match="Simulated"):
        runtime.install()
    assert json.loads((tmp_path / "active.json").read_text()) == original


def test_partial_coverage_is_not_accepted_as_full_transcript():
    from segmentwright_scribe.local.pipeline import validate_coverage

    with pytest.raises(ValueError, match="complete"):
        validate_coverage([{"start": 0, "end": 2}], 3)
    with pytest.raises(ValueError, match="contiguous"):
        validate_coverage([{"start": 0, "end": 1}, {"start": 2, "end": 3}], 3)
    validate_coverage([{"start": 0, "end": 1}, {"start": 1, "end": 3}], 3)


def test_runtime_collision_rejected_before_writing(tmp_path, monkeypatch):
    monkeypatch.setenv("SEGMENTWRIGHT_SCRIBE_LOCAL_HOME", str(tmp_path / "runtime"))
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    with pytest.raises(ValueError, match="overlap"):
        build_config(
            build_parser().parse_args(
                [str(source), str(tmp_path / "runtime/out.json"), "--preset", "local"]
            )
        )
    assert not (tmp_path / "runtime").exists()


def test_local_cache_hit_skips_all_workers(tmp_path, monkeypatch):
    import segmentwright_scribe.local.pipeline as local

    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    config = build_config(
        build_parser().parse_args(
            [
                str(source),
                str(tmp_path / "out.json"),
                "--preset",
                "local",
                "--artifacts",
                str(tmp_path / "artifacts"),
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
    )
    config.artifacts_dir.mkdir()
    monkeypatch.setenv("SEGMENTWRIGHT_SCRIBE_LOCAL_HOME", str(tmp_path / "runtime"))
    monkeypatch.setattr(local, "supported", lambda: None)
    monkeypatch.setattr(local, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(local, "normalize_to_wav", lambda _, p: p.write_bytes(b"normalized"))
    monkeypatch.setattr(local, "probe_duration", lambda _: 2.0)
    installation = {
        "profile": "test",
        "directory": str(tmp_path / "workers"),
        "models": {"vad": {"path": str(tmp_path / "vad"), "repo": "vad", "revision": "rev"}},
    }
    monkeypatch.setattr(local, "load_installation", lambda: installation)

    class VAD:
        def __init__(self, **kwargs):
            pass

        def detect(self, audio):
            from segmentwright_scribe.vad import VadResult

            return VadResult(2, [], {})

        def close(self):
            pass

    monkeypatch.setattr(local, "FireRedVadBackend", VAD)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        result = (
            {
                "chunks": [
                    {"id": "chunk_1", "start": 0, "end": 2, "text": "hello", "source": "local"}
                ],
                "candidates": [],
            }
            if "worker_granite.py" in command[2]
            else []
        )
        Path(command[-1]).write_text(json.dumps(result))

    monkeypatch.setattr(local, "execute", execute)
    local.run(config)
    assert len(calls) == 2
    monkeypatch.setattr(local, "execute", lambda *a, **k: pytest.fail("worker on cache hit"))
    monkeypatch.setattr(local, "FireRedVadBackend", lambda **k: pytest.fail("VAD on cache hit"))
    local.run(config)
    assert json.loads(config.output_path.read_text())["metadata"]["cache_hit"]


def test_local_failure_cleans_temporary_work(tmp_path, monkeypatch):
    import segmentwright_scribe.local.pipeline as local
    import segmentwright_scribe.pipeline as pipeline

    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    config = build_config(
        build_parser().parse_args([str(source), str(tmp_path / "out.json"), "--preset", "local"])
    )
    used = []

    def fail(config):
        used.append(config.artifacts_dir)
        (config.artifacts_dir / "audio.wav").write_bytes(b"private audio")
        raise RuntimeError("Simulated worker failure")

    monkeypatch.setattr(local, "run", fail)
    with pytest.raises(RuntimeError, match="Simulated"):
        pipeline.run_pipeline(config)
    assert used and not used[0].exists()


def test_ctc_prefix_beam_matches_exhaustive_paths():
    from segmentwright_scribe.local.worker_beam import test

    test()


def test_utf8_and_blank_separated_repeats_are_preserved():
    full, units, error = units_from_ids(
        [1, 0, 2], lambda ids: ("全" if len(ids) == 2 else "\ufffd") if ids else "", "ja", 0
    )
    assert full == "全" and not error and len(units) == 1 and units[0]["start"] == 0
    full, units, error = units_from_ids([1, 1, 0, 1], lambda ids: "あ" * len(ids), "ja", 0)
    assert full == "ああ" and len(units) == 2 and not error
