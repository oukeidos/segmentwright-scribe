import re
import unicodedata


def lexical(c):
    return unicodedata.category(c)[0] in "LN"


def units_from_ids(ids, decode, lang, offset, frame_sec=0.08):
    tokens = []
    for i, t in enumerate(ids):
        if i and t == ids[i - 1]:
            if t:
                tokens[-1]["end"] = i + 1
        elif t:
            tokens.append({"id": t, "start": i, "end": i + 1})
    full = decode([t["id"] for t in tokens])
    parts = []
    last = 0
    pending = 0
    for j in range(1, len(tokens) + 1):
        prefix = decode([t["id"] for t in tokens[:j]])
        if "\ufffd" in prefix or not full.startswith(prefix):
            continue
        if len(prefix) == last:
            pending = j
        if len(prefix) > last:
            parts.append(
                {
                    "text": prefix[last:],
                    "lo": last,
                    "hi": len(prefix),
                    "start": offset + tokens[pending]["start"] * frame_sec,
                    "end": offset + tokens[j - 1]["end"] * frame_sec,
                }
            )
            last = len(prefix)
            pending = j
    if last != len(full) or "\ufffd" in full:
        return full, [], "invalid_unicode"
    units = []
    spans = (
        [(m.start(), m.end()) for m in re.finditer(r"\S+", full)]
        if lang == "en"
        else [(p["lo"], p["hi"]) for p in parts]
    )
    for lo, hi in spans:
        touched = [
            p
            for p in parts
            if any(lexical(full[i]) for i in range(max(lo, p["lo"]), min(hi, p["hi"])))
        ]
        if touched:
            units.append(
                {
                    "text": full[lo:hi],
                    "start": touched[0]["start"],
                    "end": touched[-1]["end"],
                    "lexical_support": [[p["start"], p["end"]] for p in touched],
                }
            )
    return full, units, None


def in_core(unit, start, end):
    return start <= (unit["start"] + unit["end"]) / 2 < end and any(
        start <= (a + b) / 2 < end for a, b in unit["lexical_support"]
    )
