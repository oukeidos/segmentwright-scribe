"""Reference-free extraction of ASR words in a fixed acoustic core."""

import math
import unicodedata


def norm(s):
    return "".join(
        c for c in unicodedata.normalize("NFKC", s).lower() if unicodedata.category(c)[0] in "LN"
    )


def core(view, candidate):
    words = view["words"]
    source = view["text"]
    prev = -math.inf
    selected = []
    for i, w in enumerate(words):
        a, b = w["start_time"], w["end_time"]
        if (
            not (math.isfinite(a) and math.isfinite(b))
            or a < 0
            or b > view["audio_duration"]
            or b < a
            or a < prev
        ):
            return "", {"reason": "invalid_timestamps"}
        prev = a
        t = view["start_sample"] / 16000 + (a + b) / 2
        if (
            norm(w["text"])
            and candidate["start_sample"] / 16000 <= t < candidate["end_sample"] / 16000
        ):
            selected.append(i)
    if not selected:
        return "", {"reason": "empty_core"}
    if not any(words[i]["end_time"] > words[i]["start_time"] for i in selected):
        return "", {"reason": "all_point_core"}
    if selected != list(range(selected[0], selected[-1] + 1)):
        return "", {"reason": "noncontiguous_core", "selected": selected}
    canonical = norm(source)
    mapped = ""
    positions = []
    for i, ch in enumerate(source):
        for x in norm(ch):
            mapped += x
            positions.append(i)
    lengths = [len(norm(w["text"])) for w in words]
    if canonical != mapped or "".join(norm(w["text"]) for w in words) != canonical:
        return "", {"reason": "text_mapping_mismatch"}
    lo = sum(lengths[: selected[0]])
    hi = sum(lengths[: selected[-1] + 1])
    if hi <= lo:
        return "", {"reason": "empty_core"}
    start = positions[lo]
    end = positions[hi] if hi < len(positions) else len(source)
    text = source[start:end].strip()
    assert norm(text) == "".join(norm(words[i]["text"]) for i in selected)
    return text, {
        "reason": None,
        "selected": selected,
        "source_character_range": [start, end],
        "words": [words[i] for i in selected],
    }


def test():
    v = {
        "text": "前。中央です。後。",
        "start_sample": 0,
        "audio_duration": 4,
        "words": [
            {"text": "前", "start_time": 0, "end_time": 1},
            {"text": "中央", "start_time": 1, "end_time": 2},
            {"text": "です", "start_time": 2, "end_time": 3},
            {"text": "後", "start_time": 3, "end_time": 4},
        ],
    }
    assert core(v, {"start_sample": 16000, "end_sample": 48000})[0] == "中央です。"
    assert core(v, {"start_sample": 8000, "end_sample": 40000})[0] == "前。中央"
    assert core(v, {"start_sample": 64000, "end_sample": 80000})[1]["reason"] == "empty_core"
    w = {**v, "words": [dict(x) for x in v["words"]]}
    w["words"][2]["start_time"] = float("nan")
    assert core(w, {"start_sample": 0, "end_sample": 64000})[1]["reason"] == "invalid_timestamps"
    print("core extraction checks passed")


if __name__ == "__main__":
    test()
