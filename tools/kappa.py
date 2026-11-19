"""Cohen's kappa as used by results/judge_calibration/inter_rater_summary.json (binary at >= 0.5; weighted over the
five score levels 0/.25/.5/.75/1 with linear or quadratic weights)."""
from __future__ import annotations

LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _snap(x: float) -> float:
    return min(LEVELS, key=lambda l: abs(l - float(x)))


def kappa_binary(a, b, thr=0.5) -> float:
    a = [x >= thr for x in a]; b = [x >= thr for x in b]
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def kappa_weighted(a, b, power=1) -> float:
    a = [_snap(x) for x in a]; b = [_snap(x) for x in b]
    n, k = len(a), len(LEVELS)
    idx = {l: i for i, l in enumerate(LEVELS)}
    w = [[(abs(i - j) / (k - 1)) ** power for j in range(k)] for i in range(k)]
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1 / n
    ra = [sum(r) for r in obs]; cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    do = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    de = sum(w[i][j] * ra[i] * cb[j] for i in range(k) for j in range(k))
    return 1.0 if de == 0 else 1 - do / de
