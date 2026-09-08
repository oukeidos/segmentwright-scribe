from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import segmentwright_scribe.model_download as model_download
from segmentwright_scribe.model_download import (
    FIRERED_VAD_REQUIRED_FILES,
    ensure_firered_vad_model,
    inspect_firered_vad_model_dir,
)
from segmentwright_scribe.vad import FireRedVadBackend


def test_firered_vad_backend_parses_runtime_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = make_complete_model_dir(tmp_path)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"wav")
    fake_module = types.ModuleType("fireredvad")
    fake_module.FireRedVad = FakeFireRedVad
    fake_module.FireRedVadConfig = FakeFireRedVadConfig
    monkeypatch.setitem(sys.modules, "fireredvad", fake_module)

    result = FireRedVadBackend(model_dir=model_dir, device="cpu").detect(audio_path)

    assert result.duration_sec == 2.0
    assert [(segment.start_sec, segment.end_sec) for segment in result.speech_segments] == [
        (0.1, 0.5)
    ]
    assert result.raw["engine"] == "fireredvad"


def test_missing_model_dir_downloads_required_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for file_name in FIRERED_VAD_REQUIRED_FILES:
        (source_dir / file_name).write_text(f"{file_name}\n", encoding="utf-8")
    requested: list[str] = []

    def fake_download(*, repo_id: str, filename: str) -> Path:
        del repo_id
        requested.append(filename)
        return source_dir / Path(filename).name

    model_dir = tmp_path / "downloaded" / "VAD"
    monkeypatch.setattr(model_download, "hf_download", fake_download)

    assert ensure_firered_vad_model(model_dir, auto_download=True) == model_dir
    assert requested == ["VAD/model.pth.tar", "VAD/cmvn.ark"]
    assert inspect_firered_vad_model_dir(model_dir).complete


def test_auto_download_disabled_fails_for_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="auto-download"):
        ensure_firered_vad_model(tmp_path / "missing" / "VAD", auto_download=False)


def make_complete_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "FireRedVAD" / "VAD"
    model_dir.mkdir(parents=True)
    for file_name in FIRERED_VAD_REQUIRED_FILES:
        (model_dir / file_name).write_text(f"{file_name}\n", encoding="utf-8")
    return model_dir


class FakeFireRedVadConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeFireRedVad:
    @classmethod
    def from_pretrained(cls, model_dir: str, config: FakeFireRedVadConfig) -> FakeFireRedVad:
        assert model_dir.endswith("FireRedVAD/VAD")
        assert config.kwargs["speech_threshold"] == 0.4
        return cls()

    def detect(self, audio_path: str) -> tuple[dict, object]:
        assert audio_path.endswith("audio.wav")
        return ({"dur": 2.0, "timestamps": [(0.1, 0.5), (1.0, 0.9)]}, object())
