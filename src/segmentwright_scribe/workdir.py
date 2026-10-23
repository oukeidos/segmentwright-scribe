from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

TEMP_PREFIX = "segmentwright-scribe-"
TEMP_MARKER = ".segmentwright-scribe-workdir.json"
MARKER_SCHEMA = "segmentwright_scribe.temporary_workdir.v1"
STALE_AGE_SEC = 24 * 60 * 60


@contextmanager
def temporary_artifacts_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX) as name:
        path = Path(name).resolve()
        (path / TEMP_MARKER).write_text(
            json.dumps({"schema": MARKER_SCHEMA, "created_at": time.time()}),
            encoding="utf-8",
        )
        yield path


def cleanup_stale_temp_workdirs(
    *,
    temp_root: Path | None = None,
    now: float | None = None,
    protected_paths: tuple[Path, ...] = (),
) -> list[Path]:
    """Remove only marked app workdirs, measuring age from recorded creation time."""
    root = temp_root if temp_root is not None else Path(tempfile.gettempdir())
    current_time = time.time() if now is None else now
    protected = tuple(path.resolve() for path in protected_paths)
    removed = []
    for candidate in root.glob(f"{TEMP_PREFIX}*"):
        try:
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if any(
                path == resolved or path.is_relative_to(resolved) or resolved.is_relative_to(path)
                for path in protected
            ):
                continue
            marker = candidate / TEMP_MARKER
            if marker.is_symlink() or not marker.is_file():
                continue
            data = json.loads(marker.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("schema") != MARKER_SCHEMA:
                continue
            created_at = data.get("created_at")
            if (
                type(created_at) not in (int, float)
                or not math.isfinite(created_at)
                or created_at <= 0
                or current_time - created_at < STALE_AGE_SEC
            ):
                continue
            shutil.rmtree(candidate)
            removed.append(candidate)
        except (OSError, ValueError, OverflowError):
            # Stale cleanup must not prevent a new transcription from starting.
            continue
    return removed
