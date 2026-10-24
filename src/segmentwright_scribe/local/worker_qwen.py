"""One Qwen recognition pass, followed by independently loaded forced alignment."""

import dataclasses
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner


def run(job):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    audio, sr = sf.read(job["audio"], dtype="float32")
    if sr != 16000 or audio.ndim != 1:
        raise ValueError("Expected mono 16 kHz audio.")
    candidates = [c for c in job["candidates"] if c["insert_at"] is not None]
    if not candidates:
        return []
    model = Qwen3ASRModel.from_pretrained(
        job["models"]["qwen"]["path"],
        forced_aligner=None,
        max_inference_batch_size=1,
        max_new_tokens=256,
        local_files_only=True,
        device_map="cuda",
        dtype=torch.float16,
        trust_remote_code=False,
        use_safetensors=True,
    )
    # Observe generated token lengths without changing model decoding behavior.
    original_generate = model.model.generate
    lengths = []

    def generate(*args, _original=original_generate, **kwargs):
        output = _original(*args, **kwargs)
        sequences = output.sequences if hasattr(output, "sequences") else output
        source = kwargs.get("input_ids")
        if source is None:
            raise RuntimeError("Cannot validate Qwen generation length.")
        lengths.append(sequences.shape[-1] - source.shape[-1])
        return output

    model.model.generate = generate
    rows = []
    for c in candidates:
        a, b = max(0, c["start_sample"] - 3840), min(len(audio), c["end_sample"] + 3840)
        lengths.clear()
        output = model.transcribe(
            (audio[a:b], sr),
            language="English" if job["language"] == "en" else "Japanese",
            context="",
            return_time_stamps=False,
        )
        rows.append(
            {
                "candidate": c,
                "text": "".join(x.text for x in output),
                "start_sample": a,
                "end_sample": b,
                "audio_duration": (b - a) / sr,
                "token_cap": not lengths or max(lengths) >= 256,
            }
        )
        print(f"Qwen repair {c['id']}: generated", flush=True)
    # Remove the bound method reference as well as the model before loading the aligner.
    del generate, original_generate, model
    gc.collect()
    torch.cuda.empty_cache()
    aligner = Qwen3ForcedAligner.from_pretrained(
        job["models"]["aligner"]["path"],
        device_map="cuda",
        dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    for row in rows:
        if not row["text"].strip() or row["token_cap"]:
            row["words"] = []
            continue
        result = aligner.align(
            audio=(audio[row["start_sample"] : row["end_sample"]], sr),
            text=row["text"],
            language="English" if job["language"] == "en" else "Japanese",
        )[0]
        row["words"] = [dataclasses.asdict(x) for x in result.items]
    return rows


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(json.dumps(run(job), ensure_ascii=False))
