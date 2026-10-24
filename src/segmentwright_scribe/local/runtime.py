"""Install and inspect isolated, versioned local workers without importing ML packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path

POLICY = "granite-qwen-single-repair-v1"
MODELS = {
    "granite": (
        "ibm-granite/granite-speech-5.0-470m-turboctc",
        "1dd23f46f6d4e6eae2531982ba2e33220086dd97",
    ),
    "qwen": ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5"),
    "aligner": ("Qwen/Qwen3-ForcedAligner-0.6B", "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"),
    "vad": ("FireRedTeam/FireRedVAD", "7990aaccc6b7aec1e527743bd30201f2c4a03b8c"),
}
PACKAGE = Path(__file__).parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def profile() -> str:
    files = sorted(p for p in PACKAGE.iterdir() if p.suffix in {".py", ".lock"})
    return hashlib.sha256(
        json.dumps({p.name: digest(p) for p in files}, sort_keys=True).encode()
    ).hexdigest()[:20]


def home() -> Path:
    default = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return (
        Path(
            os.environ.get(
                "SEGMENTWRIGHT_SCRIBE_LOCAL_HOME", default / "segmentwright-scribe/local"
            )
        )
        .expanduser()
        .resolve()
    )


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def supported() -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            "The local preset currently supports Linux x86_64 with NVIDIA CUDA only."
        )


def clean_environment(*, offline: bool = True) -> dict[str, str]:
    allowed = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_HUB_CACHE",
    }
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env.update(PYTHONNOUSERSITE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def execute(command: list[str], *, log: Path, timeout: float = 3600, offline: bool = True) -> None:
    """Bound a process group and reap it on cancellation, timeout, or failure."""
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=clean_environment(offline=offline),
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        tail = log.read_text(errors="replace")[-2500:]
        raise RuntimeError(f"Local worker failed (exit {code}). Log: {log}\n{tail}")


def load_installation(root: Path | None = None) -> dict:
    root = root or home()
    active = root / "active.json"
    if not active.is_file():
        raise RuntimeError(
            "Local runtime is not installed. Run: segmentwright-scribe local install"
        )
    data = json.loads(active.read_text())
    if data.get("profile") != profile():
        raise RuntimeError("Local runtime is outdated. Run: segmentwright-scribe local install")
    directory = Path(data["directory"])
    if not directory.resolve().is_relative_to(root / "installs"):
        raise RuntimeError("Invalid local installation path.")
    manifest = directory / "manifest.json"
    if digest(manifest) != data["manifest_sha256"]:
        raise RuntimeError(
            "Local installation manifest is damaged. Run: segmentwright-scribe local install"
        )
    result = json.loads(manifest.read_text())
    for key in ("granite", "qwen"):
        if not (directory / key / "bin/python").is_file():
            raise RuntimeError(f"Missing {key} worker. Run: segmentwright-scribe local install")
    result["directory"] = str(directory)
    return result


def probe(installation: dict, output: Path) -> dict:
    directory = Path(installation["directory"])
    checks = {}
    for worker in ("granite", "qwen"):
        dest = output / f"{worker}-probe.json"
        execute(
            [
                str(directory / worker / "bin/python"),
                "-I",
                str(directory / "worker_probe.py"),
                worker,
                str(dest),
            ],
            log=output / f"{worker}-probe.log",
            timeout=180,
        )
        checks[worker] = json.loads(dest.read_text())
    return checks


def install() -> dict:
    supported()
    from ..model_download import locked_file

    root = home()
    root.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError(
            "uv is required to install local workers. Install uv, then retry local install."
        )
    with locked_file(root / "install.lock"):
        expected_profile = profile()
        directory = root / "installs" / f"{profile()}-{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        try:
            print(
                "Installing isolated local workers. Model downloads may require several GB.",
                file=sys.stderr,
            )
            for src in PACKAGE.glob("*.py"):
                shutil.copy2(src, directory / src.name)
            for worker in ("granite", "qwen"):
                print(f"Preparing {worker} environment...", file=sys.stderr)
                execute(
                    [uv, "venv", "--python", "3.12", str(directory / worker)],
                    log=directory / f"{worker}-venv.log",
                    offline=False,
                )
                execute(
                    [
                        uv,
                        "pip",
                        "sync",
                        "--python",
                        str(directory / worker / "bin/python"),
                        "--require-hashes",
                        str(PACKAGE / f"{worker}.lock"),
                    ],
                    log=directory / f"{worker}-install.log",
                    offline=False,
                )
            from huggingface_hub import snapshot_download

            models = {}
            for key, (repo, revision) in MODELS.items():
                print(f"Preparing model: {repo}@{revision}", file=sys.stderr)
                target = root / "models" / key / revision
                snapshot_download(
                    repo,
                    revision=revision,
                    local_dir=target,
                    allow_patterns=(
                        ["VAD/model.pth.tar", "VAD/cmvn.ark"]
                        if key == "vad"
                        else [
                            "*.json",
                            "*.safetensors",
                            "*.txt",
                            "*.model",
                            "*.tiktoken",
                            "README.md",
                            "LICENSE*",
                        ]
                    ),
                )
                files = {
                    str(p.relative_to(target)): digest(p)
                    for p in target.rglob("*")
                    if p.is_file() and ".cache" not in p.parts
                }
                if (key == "vad" and not {"VAD/model.pth.tar", "VAD/cmvn.ark"} <= files.keys()) or (
                    key != "vad"
                    and (
                        not any(p.endswith(".safetensors") for p in files)
                        or "config.json" not in files
                    )
                ):
                    raise RuntimeError(f"Incomplete model download: {repo}")
                models[key] = {
                    "repo": repo,
                    "revision": revision,
                    "path": str(target),
                    "files": files,
                }
            installation = {
                "profile": profile(),
                "policy": POLICY,
                "models": models,
                "directory": str(directory),
                "workers": {p.name: digest(p) for p in directory.glob("*.py")},
            }
            installation["probe"] = probe(installation, directory)
            if profile() != expected_profile:
                raise RuntimeError("Package changed during installation; rerun local install.")
            atomic_json(directory / "manifest.json", installation)
            atomic_json(
                root / "active.json",
                {
                    "profile": profile(),
                    "directory": str(directory),
                    "manifest_sha256": digest(directory / "manifest.json"),
                },
            )
            return {"status": "ready", "profile": profile(), "home": str(root)}
        except BaseException:
            # Preserve failed logs for diagnosis; never replace the active installation.
            print(
                f"Installation failed; previous runtime is unchanged. Logs: {directory}",
                file=sys.stderr,
            )
            raise


def doctor() -> dict:
    supported()
    installation = load_installation()
    for name, expected in installation["workers"].items():
        if digest(Path(installation["directory"]) / name) != expected:
            raise RuntimeError(f"Worker integrity check failed: {name}")
    for spec in installation["models"].values():
        for name, expected in spec["files"].items():
            if digest(Path(spec["path"]) / name) != expected:
                raise RuntimeError(f"Model integrity check failed: {spec['repo']}/{name}")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH.")
    import tempfile

    with tempfile.TemporaryDirectory(prefix="scribe-local-doctor-") as directory:
        checks = probe(installation, Path(directory))
    return {
        "status": "ready",
        "profile": profile(),
        "checks": checks,
        "free_disk_bytes": shutil.disk_usage(home()).free,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="segmentwright-scribe local",
        description="Manage optional local transcription workers.",
    )
    parser.add_argument("command", choices=("install", "doctor"))
    args = parser.parse_args(argv)
    try:
        result = install() if args.command == "install" else doctor()
        print(json.dumps(result, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"segmentwright-scribe local: {exc or 'Cancelled.'}", file=sys.stderr)
        return 1
