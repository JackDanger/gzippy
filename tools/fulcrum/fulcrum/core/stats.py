"""Sample statistics + distribution health (the SPREAD-RESOLUTION invariant).

A delta smaller than the arms' spread is not a finding, and wall
distributions go bimodal under scheduling regimes (a median can sit on
either mode) — whole sessions have been spent "measuring" such ties; see
docs/CASE-STUDIES.md ("the bimodal comparator"). Every verdict therefore
carries RESOLVED/UNRESOLVED with N-needed, a sub-spread delta is NEVER
presented as a finding, and bimodality is detected on every sample set.
"""

import math
import os

BIMODAL_K = 3.0


def read_samples(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [float(x) for x in f.read().split() if x.strip()]


def sample_stats(xs):
    """min, median, iqr, spread_pct over wall samples (seconds)."""
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)

    def q(p):
        # linear interpolation percentile
        k = (n - 1) * p
        lo, hi = int(math.floor(k)), int(math.ceil(k))
        if lo == hi:
            return s[lo]
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    med = q(0.5)
    iqr = q(0.75) - q(0.25)
    spread_pct = (s[-1] - s[0]) / s[0] * 100 if s[0] > 0 else 0.0
    return {"n": n, "min": s[0], "med": med, "max": s[-1],
            "iqr": iqr, "spread_pct": spread_pct}


def bimodal(xs, k=BIMODAL_K):
    """Largest-gap heuristic (the N=21 lesson). Flag iff the largest internal
    gap > k x median of the remaining gaps AND each side keeps >=2 samples."""
    s = sorted(xs)
    if len(s) < 5:
        return False
    gaps = [(s[i + 1] - s[i], i) for i in range(len(s) - 1)]
    g, i = max(gaps)
    others = sorted(x for x, j in gaps if j != i)
    if not others:
        return False
    med_other = others[len(others) // 2]
    left, right = i + 1, len(s) - (i + 1)
    if med_other <= 0:
        # Degenerate: all other gaps are zero (all other samples identical).
        # Still require both sides have >=2 samples — a single-sample "mode"
        # is not bimodal (repro: [1,1,1,1,1.01] left=4 right=1 => False).
        return g > 0 and left >= 2 and right >= 2
    return g > k * med_other and left >= 2 and right >= 2


def resolution(delta_s, spread_a_s, spread_b_s, n):
    """RESOLVED iff |delta| exceeds the larger arm spread (absolute seconds);
    else UNRESOLVED with N-needed ~ ceil(n * (spread/|delta|)^2), capped 99."""
    margin = max(spread_a_s, spread_b_s)
    if abs(delta_s) > margin:
        return ("RESOLVED", None)
    if delta_s == 0:
        return ("UNRESOLVED", 99)
    need = min(99, max(n + 2, math.ceil(n * (margin / abs(delta_s)) ** 2)))
    return ("UNRESOLVED", need)


def dist_health_str(xs):
    st = sample_stats(xs)
    if not st:
        return "no-data"
    parts = [f"n={st['n']}", f"spread={st['spread_pct']:.1f}%"]
    if bimodal(xs):
        parts.append("BIMODAL")
    return " ".join(parts)
