from __future__ import annotations

import json
import os

import pytest

from segmentwright_scribe import workdir


def marked(root, name, created_at):
    path = root / name
    path.mkdir()
    (path / workdir.TEMP_MARKER).write_text(json.dumps({
        "schema": workdir.MARKER_SCHEMA, "created_at": created_at,
    }))
    (path / "source.wav").write_bytes(b"audio")
    return path


def test_cleanup_uses_creation_time_and_only_app_markers(tmp_path):
    now = 200_000
    old = marked(tmp_path, "segmentwright-scribe-old", now - 86400)
    fresh = marked(tmp_path, "segmentwright-scribe-fresh", now - 86399)
    # Marker mtime is deliberately unrelated to recorded creation time.
    os.utime(fresh / workdir.TEMP_MARKER, (1, 1))
    other = marked(tmp_path, "other-app-old", 1)
    unmarked = tmp_path / "segmentwright-scribe-unmarked"
    unmarked.mkdir()
    assert workdir.cleanup_stale_temp_workdirs(temp_root=tmp_path, now=now) == [old]
    assert all(path.exists() for path in (fresh, other, unmarked))


@pytest.mark.parametrize("value", [None, True, "1", -1, 0, float("nan"), float("inf"), 300000])
def test_invalid_or_future_creation_time_is_preserved(tmp_path, value):
    path = marked(tmp_path, "segmentwright-scribe-invalid", value)
    assert workdir.cleanup_stale_temp_workdirs(temp_root=tmp_path, now=200000) == []
    assert path.exists()


@pytest.mark.parametrize("content", ["broken", "[]", '{"schema": "other", "created_at": 1}'])
def test_invalid_marker_is_preserved(tmp_path, content):
    path = marked(tmp_path, "segmentwright-scribe-invalid", 1)
    (path / workdir.TEMP_MARKER).write_text(content)
    assert workdir.cleanup_stale_temp_workdirs(temp_root=tmp_path, now=200000) == []
    assert path.exists()


def test_symlinks_are_preserved(tmp_path):
    target = marked(tmp_path, "other-target", 1)
    link = tmp_path / "segmentwright-scribe-link"
    link.symlink_to(target, target_is_directory=True)
    linked_marker = tmp_path / "segmentwright-scribe-linked-marker"
    linked_marker.mkdir()
    (linked_marker / workdir.TEMP_MARKER).symlink_to(target / workdir.TEMP_MARKER)
    assert workdir.cleanup_stale_temp_workdirs(temp_root=tmp_path, now=200000) == []
    assert target.exists() and link.is_symlink() and linked_marker.exists()


@pytest.mark.parametrize("relation", ["same", "child", "parent"])
def test_explicit_paths_are_protected_even_if_marked(tmp_path, relation):
    path = marked(tmp_path, "segmentwright-scribe-explicit", 1)
    protected = {"same": path, "child": path / "cache", "parent": tmp_path}[relation]
    assert workdir.cleanup_stale_temp_workdirs(
        temp_root=tmp_path, now=200000, protected_paths=(protected,),
    ) == []
    assert path.exists()


def test_cleanup_failure_does_not_abort_run(tmp_path, monkeypatch):
    path = marked(tmp_path, "segmentwright-scribe-old", 1)

    def denied(path):
        raise PermissionError("denied")

    monkeypatch.setattr(workdir.shutil, "rmtree", denied)
    assert workdir.cleanup_stale_temp_workdirs(temp_root=tmp_path, now=200000) == []
    assert path.exists()


def test_temporary_directory_is_marked_and_removed(monkeypatch):
    monkeypatch.setattr(workdir.time, "time", lambda: 12345)
    with workdir.temporary_artifacts_dir() as path:
        data = json.loads((path / workdir.TEMP_MARKER).read_text())
        assert data == {"schema": workdir.MARKER_SCHEMA, "created_at": 12345}
    assert not path.exists()
