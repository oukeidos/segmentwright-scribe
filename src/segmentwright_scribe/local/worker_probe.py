"""Isolated dependency and CUDA operation smoke test."""

import json
import sys
from pathlib import Path

import torch
import torchaudio
import transformers

if sys.argv[1] == "granite":
    from transformers import AutoModelForCTC, AutoProcessor  # noqa: F401
else:
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner  # noqa: F401
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. The local preset requires a supported NVIDIA GPU.")
x = torch.ones((32, 32), device="cuda")
assert float((x @ x)[0, 0]) == 32
Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "transformers": transformers.__version__,
            "python": sys.version,
            "gpu": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "free_vram_bytes": torch.cuda.mem_get_info()[0],
            "status": "ok",
        }
    )
)
