"""CTC prefix beam: log-space marginalization; no LM or reference vocabulary."""

import itertools
import math

NEG = -math.inf


def add(a, b):
    if a == NEG:
        return b
    if b == NEG:
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(-abs(a - b)))


def decode(indices, logprobs, blanks, width=32):
    beams = {(): [0.0, NEG]}
    for ids, ps, blank in zip(indices, logprobs, blanks, strict=False):
        nxt = {}
        for prefix, (pb, pn) in beams.items():
            total = add(pb, pn)
            v = nxt.setdefault(prefix, [NEG, NEG])
            v[0] = add(v[0], total + float(blank))
            for token, p in zip(ids, ps, strict=False):
                token = int(token)
                p = float(p)
                if token == 0:
                    continue
                if prefix and prefix[-1] == token:
                    v[1] = add(v[1], pn + p)
                    base = pb
                else:
                    base = total
                q = nxt.setdefault(prefix + (token,), [NEG, NEG])
                q[1] = add(q[1], base + p)
        beams = dict(sorted(nxt.items(), key=lambda kv: add(*kv[1]), reverse=True)[:width])
    return max(beams, key=lambda p: add(*beams[p])), beams


def test():
    import random

    random.seed(0)
    for n in range(1, 7):
        raw = [[random.random() + 0.01 for _ in range(3)] for t in range(n)]
        ps = [[v / sum(r) for v in r] for r in raw]
        logs = [[math.log(v) for v in r] for r in ps]
        expected = {}
        for path in itertools.product(range(3), repeat=n):
            prefix = tuple(x for x, _ in itertools.groupby(path) if x != 0)
            p = math.prod(ps[t][x] for t, x in enumerate(path))
            expected[prefix] = expected.get(prefix, 0) + p
        _, actual = decode([[0, 1, 2]] * n, logs, [r[0] for r in logs], width=10000)
        for prefix, p in expected.items():
            assert abs(math.exp(add(*actual[prefix])) - p) < 1e-12
    print("PASS: exhaustive CTC path sums, 1..6 frames, repeats and blanks")


if __name__ == "__main__":
    test()
