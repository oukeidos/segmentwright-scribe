"""Granite context decoding and conservative speech-gap proposals."""

import difflib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor
from worker_beam import decode
from worker_units import units_from_ids


def run(job):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("The local preset requires CUDA.")
    audio, sr = sf.read(job["audio"], dtype="float32")
    if sr != 16000 or audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Expected finite, nonempty mono 16 kHz audio.")
    proc = AutoProcessor.from_pretrained(job["models"]["granite"]["path"], local_files_only=True)
    model = (
        AutoModelForCTC.from_pretrained(
            job["models"]["granite"]["path"], local_files_only=True, dtype=torch.float32
        )
        .to("cuda")
        .eval()
    )
    chunks, candidates = [], []
    started = time.monotonic()
    for i, a in enumerate(range(0, len(audio), 60 * sr)):
        b = min(a + 60 * sr, len(audio))
        left, right = max(0, a - 10 * sr), min(len(audio), b + 10 * sr)
        with torch.inference_mode():
            batch = proc(
                [audio[left:right]], sampling_rate=sr, device="cuda", return_tensors="pt"
            ).to("cuda", dtype=torch.float32)
            result = model.generate(**batch, return_dict_in_generate=True)
            valid = int(
                model._get_output_attention_mask(
                    batch["attention_mask"], target_length=result.logits.shape[1]
                )[0].sum()
            )
            start = (a - left) // 1280
            end = (b - left) // 1280 if b < len(audio) else valid
            logits = result.logits[0, start:end]
            lp = logits.log_softmax(-1)
            values, indices = lp.topk(20, dim=-1)
            blank = lp[:, 0].cpu().numpy()
            ids = logits.argmax(-1).tolist()
        beam_ids, _ = decode(indices.cpu().numpy(), values.cpu().numpy(), blank)
        text = proc.tokenizer.decode(list(beam_ids), skip_special_tokens=True, group_tokens=False)
        greedy, units, error = units_from_ids(
            ids,
            lambda ts: proc.tokenizer.decode(ts, skip_special_tokens=True, group_tokens=False),
            job["language"],
            a / sr,
        )
        chunks.append(
            {
                "id": f"chunk_{i + 1}",
                "start": a / sr,
                "end": b / sr,
                "text": text,
                "source": "local",
            }
        )
        occupied = np.zeros(len(ids), dtype=bool)
        for u in units:
            for ua, ub in u["lexical_support"]:
                lo = round((ua - a / sr) / 0.08)
                hi = round((ub - a / sr) / 0.08)
                occupied[max(0, lo - 2) : min(len(ids), hi + 2)] = True
        if error:
            continue
        blocks = difflib.SequenceMatcher(None, greedy, text, autojunk=False).get_matching_blocks()
        for va, vb in job["speech"]:
            lo = max(0, math.ceil((max(a / sr, va) - a / sr) / 0.08))
            hi = min(len(ids), math.floor((min(b / sr, vb) - a / sr) / 0.08))
            t = lo
            while t < hi:
                if occupied[t]:
                    t += 1
                    continue
                first = t
                while t < hi and not occupied[t]:
                    t += 1
                if (t - first) * 0.08 < 1:
                    continue
                prefix = proc.tokenizer.decode(ids[:first], skip_special_tokens=True)
                pos = len(prefix)
                mapped = {
                    blk.b + pos - blk.a
                    for blk in blocks
                    if blk.size and blk.a <= pos <= blk.a + blk.size
                }
                idx = next(iter(mapped)) if len(mapped) == 1 and greedy.startswith(prefix) else None
                if idx is not None:
                    anchor = text[max(0, idx - 8) : idx + 8]
                    if not anchor or text.count(anchor) != 1:
                        idx = None
                candidates.append(
                    {
                        "id": len(candidates),
                        "chunk_index": i,
                        "start_sample": a + first * 1280,
                        "end_sample": a + t * 1280,
                        "insert_at": idx,
                        "left": text[max(0, idx - 40) : idx] if idx is not None else "",
                        "right": text[idx : idx + 40] if idx is not None else "",
                    }
                )
        print(f"Granite chunk {i + 1}: {a / sr:.2f}-{b / sr:.2f}s", flush=True)
    return {"chunks": chunks, "candidates": candidates, "seconds": time.monotonic() - started}


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[1]).read_text())
    result = run(job)
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False))
