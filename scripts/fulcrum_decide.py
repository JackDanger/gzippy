#!/usr/bin/env python3
"""fulcrum_decide — the ONE-RUN decision instrument (plans/fulcrum2-charter.md).

Consumes the artifact directory produced by scripts/bench/decide.sh /
_decide_guest.sh and renders ONE ranked component table where every row carries:
  - cells affected + wall-ms attribution (canonical-mask trace decomposition),
  - CAUSAL STATUS: tool-executed kill-switch A/B verdict (CAUSAL-VERIFIED /
    CAUSAL-NULL) for knob-covered components; HYPOTHESIS + the exact suggested
    perturbation for everything else — NEVER a recommendation without a knob,
  - DISTRIBUTION HEALTH: spread, bimodality (the N=21 lesson), RESOLVED /
    UNRESOLVED with N-needed; a sub-spread delta is never presented as a finding,
  - the EXACT re-verify command,
and a final `DO THIS NEXT:` line.

Builds ON fulcrum_total.py (imported, not forked): trace decomposition, the
busy+idle==span / no-double-count assertions, and the RE-DERIVED routing guard
(production-seeded accepted; only ACTUAL oracle contamination refused).

WHY (the failure mode this kills): this campaign repeatedly produced attribution
that did NOT convert at the wall (377ms pair-drain phantom, per-EOB stop cost,
KEY-MISMATCH re-key) and real levers found only after manual falsifier chains.
The instrument closes the attribution->causal gap ITSELF by running the in-tree
kill-switch A/Bs and refusing to rank anything the data cannot support.

Usage:
  python3 scripts/fulcrum_decide.py <artifact-dir> [--allow-thaw]
  python3 scripts/fulcrum_decide.py --selftest
"""

import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fulcrum_total as ft  # noqa: E402  (the validated core — extended, not forked)


# ---------------------------------------------------------------------------
# Banked comparators (provenance-pinned). If a live measurement CONTRADICTS a
# bank row by more than the stated tolerance, the row is flagged
# DIVERGES-FROM-BANK and the table says so instead of silently ranking —
# either the tool or the bank is wrong, and that must be investigated.
# ---------------------------------------------------------------------------
BANK = {
    # plans/orchestrator-status.md "P3.5 OFFICIAL MATRIX @ a9fe662c" (2026-06-10),
    # CONTIG_PROF @ silesia T8 on bin-p35-native:
    "silesia_T8_backref_share": 62.6,   # % of classed cycles
    "silesia_T8_backref_cyc": 34.9,     # cyc/iter
    "silesia_T8_litchn_share": 22.9,    # % of classed cycles
    # NATIVE T1 trajectory (frozen): gz 1375ms vs rg ~914-921 => ~1.5x symbol rate.
    "silesia_T1_rg_ratio_band": (0.60, 0.72),
    # T8 was 16ms from rg (0.959); T16 0.956 with a known scheduling-state mode.
}
BANK_REL_TOL = 0.25  # >25% relative divergence => flag

# The binding TIE bar (project_tie_bar_99pct_all_threadcounts): >=0.99x at EVERY T.
TIE_BAR = 0.99
BIMODAL_K = 3.0


# ---------------------------------------------------------------------------
# Sample statistics + distribution health.
# ---------------------------------------------------------------------------

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
    """Largest-gap heuristic (the N=21 lesson: rg distributions go bimodal /
    quantized; a median can sit on either mode). Flag iff the largest internal
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


# ---------------------------------------------------------------------------
# Knob A/B verdicts (the causal core). Convention mirrors the parity spine's
# validated min-based ratio + max-spread margin (the same-binary kill-switch
# instrument that separated layout wobble from behavior in P3.4).
# ---------------------------------------------------------------------------

def knob_verdict(base, knob):
    """base/knob: wall-sample lists (s). Returns dict with status in
    {CAUSAL-VERIFIED-COSTS, CAUSAL-VERIFIED-PAYS, CAUSAL-NULL} + numbers.
    knob arm = the FEATURE-ALTERED arm (kill-switch thrown / opt-in enabled).
    delta = knob_min - base_min: delta < 0 => altered arm faster => the shipped
    default COSTS wall in this cell (actionable)."""
    sb, sk = sample_stats(base), sample_stats(knob)
    if not sb or not sk:
        return {"status": "NO-DATA"}
    delta = sk["min"] - sb["min"]
    margin = max(sb["spread_pct"], sk["spread_pct"]) / 100.0 * sb["min"]
    res, n_need = resolution(delta, margin, margin, sb["n"])
    if abs(delta) > margin:
        status = "CAUSAL-VERIFIED-COSTS" if delta < 0 else "CAUSAL-VERIFIED-PAYS"
    else:
        status = "CAUSAL-NULL"
    return {"status": status, "delta_ms": delta * 1000.0,
            "margin_ms": margin * 1000.0, "base": sb, "knob": sk,
            "bimodal": bimodal(base) or bimodal(knob),
            "resolution": res, "n_needed": n_need}


# Effect predicates: prove the kill-switch actually disabled the feature
# ("kill-switches verified to disable fully"). Input: the knob-arm and base-arm
# effect-capture stderr texts. Return (verified: bool|None, note). None = no
# in-tree counter => EFFECT-UNVERIFIED label (never silently trusted).
def effect_check(pred, base_txt, knob_txt):
    if pred == "none":
        return (None, "no in-tree counter; A/B is wall-only (EFFECT-UNVERIFIED)")
    if pred == "verbose_seeded":
        m = re.search(r"seeded_block=(\d+) seeded_wrapper=(\d+)", knob_txt)
        if not m:
            return (False, "seeded_block counter line absent in knob arm")
        blk, wrp = int(m.group(1)), int(m.group(2))
        if blk == 0 and wrp > 0:
            return (True, f"knob arm: seeded_block=0, seeded_wrapper={wrp} (switch effective)")
        return (False, f"knob arm still seeded_block={blk} (switch INEFFECTIVE)")
    if pred == "verbose_exact":
        m = re.search(r"exact_block=(\d+) exact_wrapper=(\d+)", knob_txt)
        if not m:
            return (False, "exact_block counter line absent in knob arm")
        blk, wrp = int(m.group(1)), int(m.group(2))
        if blk == 0 and wrp > 0:
            return (True, f"knob arm: exact_block=0, exact_wrapper={wrp} (switch effective)")
        if blk == 0 and wrp == 0:
            return (None, "no until-exact chunks in this cell (predicate vacuous)")
        return (False, f"knob arm still exact_block={blk} (switch INEFFECTIVE)")
    if pred == "prof_dist":
        # C_N_DISTBUILD / C_N_DISTREUSE are incremented ONLY in the amortized
        # (default) arm (marker_inflate.rs:2262/2266); the GZIPPY_DIST_AMORT=0
        # kill-switch arm (marker_inflate.rs:2226-2247) does fresh per-block
        # builds WITHOUT touching them. So the EFFECTIVE-switch signature is:
        # base arm counters alive, knob arm counters DEAD (builds=0, reuses=0).
        mb = re.search(r"disttbl: builds=(\d+) reuses=(\d+)", base_txt)
        mk = re.search(r"disttbl: builds=(\d+) reuses=(\d+)", knob_txt)
        if not (mb and mk):
            return (False, "disttbl prof line absent (capture without GZIPPY_CONTIG_PROF?)")
        bb, br = map(int, mb.groups())
        kb, kr = map(int, mk.groups())
        if bb == 0 and br == 0:
            return (None, "base arm never hit the amortized build path "
                          "(no dynamic blocks?) — predicate vacuous")
        if kb == 0 and kr == 0:
            return (True, f"base builds={bb}/reuses={br} alive, knob arm counters "
                          f"dead (P3.4 path bypassed — switch effective)")
        return (False, f"knob arm still on the amortized path (builds={kb} "
                       f"reuses={kr}) — switch INEFFECTIVE")
    if pred == "rpmalloc_stats":
        # rpmalloc_alloc.rs:523 prints:
        #   "[rpmalloc {tag}] mapped_peak=XM mapped_total=YM ..."
        # ONLY when GZIPPY_RPMALLOC_STATS is set AND the arena-allocator feature
        # is active. The effect capture for slab_alloc sets GZIPPY_RPMALLOC_STATS=1
        # in both arms; GZIPPY_SLAB_ALLOC=1 (knob arm) activates the slab which
        # uses rpmalloc. EFFECT-VERIFIED iff: knob arm HAS the stats line (slab
        # engaged) AND base arm does NOT (slab disabled by default).
        has_base = bool(re.search(r"\[rpmalloc ", base_txt))
        has_knob = bool(re.search(r"\[rpmalloc ", knob_txt))
        if not has_knob:
            return (False,
                    "no [rpmalloc] stats line in knob arm — GZIPPY_RPMALLOC_STATS "
                    "not set in effect capture, or arena-allocator feature not built")
        if has_base:
            return (False,
                    "rpmalloc stats line ALSO present in base arm — switch not "
                    "exclusive (slab may be default-on in this build)")
        m = re.search(r"\[rpmalloc[^\]]*\] mapped_peak=([\d.]+)M", knob_txt)
        mp = m.group(1) if m else "?"
        return (True, f"knob arm: rpmalloc stats present (mapped_peak={mp}M, slab "
                      f"engaged); absent in base arm — switch effective")
    return (False, f"unknown predicate '{pred}'")


# ---------------------------------------------------------------------------
# contig_prof parser (the [contig-prof] stderr block, contig_prof.rs:195-299).
# ---------------------------------------------------------------------------

PROF_CLASS_RE = re.compile(
    r"^\s+(lit1|litpack|litchn|backref)\s*: iters=\s*(\d+) cyc=\s*(\d+)\s+"
    r"([\d.]+)% of classed,\s+([\d.]+) cyc/iter", re.M)
PROF_HEAD_RE = re.compile(
    r"calls=(\d+) total_cyc=(\d+) classed_cyc=(\d+) \(([\d.]+)% of total")
PROF_DIST_RE = re.compile(r"disttbl: builds=(\d+) reuses=(\d+)")
PROF_WRAPPER_CALLS_RE = re.compile(
    r"\[contig-prof\] WRAPPER.*?calls=(\d+)", re.S)


def parse_prof(text):
    out = {"classes": {}, "head": None, "disttbl": None, "wrapper_calls": None}
    m = PROF_HEAD_RE.search(text)
    if m:
        out["head"] = {"calls": int(m.group(1)), "total_cyc": int(m.group(2)),
                       "classed_cyc": int(m.group(3)),
                       "classed_pct": float(m.group(4))}
    for cm in PROF_CLASS_RE.finditer(text):
        out["classes"][cm.group(1)] = {
            "iters": int(cm.group(2)), "cyc": int(cm.group(3)),
            "share_pct": float(cm.group(4)), "cyc_iter": float(cm.group(5))}
    dm = PROF_DIST_RE.search(text)
    if dm:
        out["disttbl"] = (int(dm.group(1)), int(dm.group(2)))
    wm = PROF_WRAPPER_CALLS_RE.search(text)
    if wm:
        out["wrapper_calls"] = int(wm.group(1))
    return out


def bank_divergence(cell, prof):
    """Compare a silesia T8 prof against the banked P3.5 comparator. Returns a
    list of human-readable divergence strings (empty == consistent)."""
    if cell != ("silesia", 8) or not prof["classes"]:
        return []
    div = []
    br = prof["classes"].get("backref")
    if br:
        share_ok = abs(br["share_pct"] - BANK["silesia_T8_backref_share"]) \
            / BANK["silesia_T8_backref_share"] <= BANK_REL_TOL
        for key, val in (("share_pct", BANK["silesia_T8_backref_share"]),
                         ("cyc_iter", BANK["silesia_T8_backref_cyc"])):
            got = br[key]
            if val and abs(got - val) / val > BANK_REL_TOL:
                msg = (f"backref.{key}={got:.1f} vs banked {val} "
                       f"(>±{BANK_REL_TOL:.0%}) — DIVERGES-FROM-BANK")
                if key == "cyc_iter" and share_ok:
                    msg += (" [shares MATCH the bank => structure consistent; "
                            "absolute TSC-cyc/iter scales with core-clock state "
                            "(TSC is fixed-rate) — suspect a frequency-state "
                            "mismatch between captures (frozen no_turbo here vs "
                            "the bank's capture), not a code change]")
                div.append(msg)
    lc = prof["classes"].get("litchn")
    if lc:
        val = BANK["silesia_T8_litchn_share"]
        if abs(lc["share_pct"] - val) / val > BANK_REL_TOL:
            div.append(f"litchn.share={lc['share_pct']:.1f} vs banked {val} "
                       f"— DIVERGES-FROM-BANK")
    return div


# ---------------------------------------------------------------------------
# Artifact-dir loading.
# ---------------------------------------------------------------------------

def parse_manifest(path):
    mf = {"cells_done": [], "knobs_done": [], "knob_sha_fail": []}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k == "cell_done":
                mf["cells_done"].append(v)
            elif k == "knob_done":
                mf["knobs_done"].append(v)
            elif k == "knob_sha_fail":
                mf["knob_sha_fail"].append(v)
            else:
                mf[k] = v
    return mf


def cell_key(corpus, t):
    return (corpus, int(t))


def load_run(art_dir):
    man_path = os.path.join(art_dir, "manifest.txt")
    if not os.path.exists(man_path):
        raise ft.InstrumentError(f"no manifest.txt in {art_dir} — not a decide "
                                 f"artifact dir")
    man = parse_manifest(man_path)
    run = {"manifest": man, "cells": {}, "dir": art_dir}
    for name in sorted(os.listdir(art_dir)):
        m = re.match(r"cell_([a-z0-9]+)_T(\d+)$", name)
        if not m:
            continue
        cdir = os.path.join(art_dir, name)
        ck = cell_key(m.group(1), m.group(2))
        cell = {"dir": cdir,
                "gz": read_samples(os.path.join(cdir, "wall_gz.txt")),
                "rg": read_samples(os.path.join(cdir, "wall_rg.txt")),
                "knobs": {}}
        ptxt = os.path.join(cdir, "prof.txt")
        cell["prof"] = parse_prof(open(ptxt).read()) if os.path.exists(ptxt) else None
        cell["trace"] = os.path.join(cdir, "trace.json")
        cell["verbose"] = os.path.join(cdir, "verbose.txt")
        for kn in sorted(os.listdir(cdir)):
            km = re.match(r"knob_(\w+)$", kn)
            if not km:
                continue
            kd = os.path.join(cdir, kn)
            cell["knobs"][km.group(1)] = {
                "base": read_samples(os.path.join(kd, "base.txt")),
                "knob": read_samples(os.path.join(kd, "knob.txt")),
                "meta": dict(
                    l.strip().split("=", 1)
                    for l in open(os.path.join(kd, "meta.txt"))
                    if "=" in l) if os.path.exists(os.path.join(kd, "meta.txt"))
                else {},
            }
        run["cells"][ck] = cell
    # knob effect captures
    run["effects"] = {}
    for name in sorted(os.listdir(art_dir)):
        m = re.match(r"knob_effects_([a-z0-9]+)_T(\d+)$", name)
        if not m:
            continue
        edir = os.path.join(art_dir, name)
        for f in os.listdir(edir):
            fm = re.match(r"effect_(base|knob)_(\w+)\.txt$", f)
            if fm:
                run["effects"].setdefault(fm.group(2), {})[fm.group(1)] = \
                    open(os.path.join(edir, f)).read()
    return run


# ---------------------------------------------------------------------------
# Freeze / contamination gate for ranking.
# ---------------------------------------------------------------------------

def frozen_ok(man):
    return (man.get("freeze_state") in ("frozen", "acknowledged")
            and man.get("quiet_state") == "quiet")


# Knob registry mirror (env + predicate; MUST match _decide_guest.sh).
KNOB_ENV = {
    "dist_amort": ("GZIPPY_DIST_AMORT=0", "prof_dist",
                   "P3.4 DistTable amortization"),
    "stored_flip": ("GZIPPY_NO_STORED_FLIP=1", "none", "M2b stored early-flip"),
    "seeded_block": ("GZIPPY_SEEDED_BLOCK=0", "verbose_seeded",
                     "M3 seeded chunks on Block"),
    "exact_block": ("GZIPPY_EXACT_BLOCK=0", "verbose_exact",
                    "M4 until-exact on Block"),
    "hit_drive": ("GZIPPY_NO_HIT_DRIVE=1", "none",
                  "confirmed-offset hit-drive prefetch"),
    "slab_alloc": ("GZIPPY_SLAB_ALLOC=1", "rpmalloc_stats",
                   "slab allocator (opt-in; the reverted lever)"),
    "eager_postproc": ("GZIPPY_EAGER_POSTPROC=1", "none",
                       "eager consumer post-processing (opt-in)"),
}

SUGGESTED_PERTURBATION = {
    "compute": ("GZIPPY_SLOW_MODE=50 [GZIPPY_SLOW_KIND=sleep control] via "
                "scripts/bench/oracle.sh --kind perturb (clean-loop slow-inject, "
                "slow_knob.rs)"),
    "output": "GZIPPY_SKIP_WRITEV_SYSCALL=1 A/B (output-stage removal probe)",
    "wait": ("worker-side lever — perturb the ENGINE (slow_knob) and watch this "
             "wait shrink/grow; the wait itself is not the cause"),
    "idle": "scheduling-state probe: N=21 re-measure (bimodal check) before anything",
}


# ---------------------------------------------------------------------------
# The decision table.
# ---------------------------------------------------------------------------

def fmt_cell(ck):
    return f"{ck[0]}:T{ck[1]}"


def dist_health_str(xs, other=None):
    st = sample_stats(xs)
    if not st:
        return "no-data"
    parts = [f"n={st['n']}", f"spread={st['spread_pct']:.1f}%"]
    if bimodal(xs):
        parts.append("BIMODAL")
    return " ".join(parts)


def analyze_run(run, allow_thaw=False, feature=None):
    man = run["manifest"]
    feature = feature or man.get("feature")
    rows = []          # each: dict(component, cells, attrib, status, dist, verify, rank_ms, tier)
    header = []
    anomalies = []

    ok_frozen = frozen_ok(man)
    if not ok_frozen and not allow_thaw:
        raise ft.InstrumentError(
            f"run NOT frozen/quiet (freeze_state={man.get('freeze_state')}, "
            f"quiet_state={man.get('quiet_state')}) — REFUSING to rank wall "
            f"numbers. Pass --allow-thaw to label instead.")
    unfrozen_tag = "" if ok_frozen else " [UNFROZEN — ratio-only, do not bank]"

    header.append(f"run        : {man.get('runid')}  bin={man.get('bin')} "
                  f"sha={str(man.get('bin_sha'))[:16]} feature={feature}")
    header.append(f"box        : freeze={man.get('freeze_state')} "
                  f"quiet={man.get('quiet_state')} governor={man.get('governor')} "
                  f"no_turbo={man.get('no_turbo')} "
                  f"runnable_avg={man.get('runnable_avg')}{unfrozen_tag}")
    header.append(f"rapidgzip  : {man.get('rg_version')}")
    header.append(f"sha-verify : every measured run checked against the corpus pin "
                  f"(guest aborts on mismatch); cells_done={len(man['cells_done'])}")

    # ---- per-cell wall scoreboard + worst failing cell (for engine bounds) ----
    cell_walls = {}
    scoreboard = []
    for ck, cell in sorted(run["cells"].items()):
        sg, sr = sample_stats(cell["gz"]), sample_stats(cell["rg"])
        if not sg or not sr:
            continue
        ratio = sr["min"] / sg["min"] if sg["min"] else 0.0
        delta_s = sg["min"] - sr["min"]
        res, n_need = resolution(delta_s,
                                 sg["spread_pct"] / 100 * sg["min"],
                                 sr["spread_pct"] / 100 * sr["min"], sg["n"])
        verdict = "PASS" if ratio >= TIE_BAR else "FAIL"
        bm = ("gz" if bimodal(cell["gz"]) else "") + \
             ("+rg" if bimodal(cell["rg"]) else "")
        cell_walls[ck] = {"gz": sg, "rg": sr, "ratio": ratio,
                          "gap_ms": delta_s * 1000.0, "resolution": res,
                          "n_needed": n_need, "verdict": verdict}
        scoreboard.append(
            f"  {fmt_cell(ck):13s} gz={sg['min']*1000:7.1f}ms rg={sr['min']*1000:7.1f}ms "
            f"ratio={ratio:.3f} {verdict:4s} {res}"
            + (f"(N->{n_need})" if n_need else "")
            + (f" spread gz={sg['spread_pct']:.1f}%/rg={sr['spread_pct']:.1f}%")
            + (f" BIMODAL[{bm}]" if bm else ""))

    # ---- trace decomposition per cell (fulcrum_total, canonical mask) ----------
    trace_components = {}   # comp -> {cell: ms}
    for ck, cell in sorted(run["cells"].items()):
        if not os.path.exists(cell["trace"]) or os.path.getsize(cell["trace"]) == 0:
            anomalies.append(f"{fmt_cell(ck)}: trace absent/empty — attribution "
                             f"rows skipped for this cell")
            continue
        try:
            b = ft.analyze(cell["trace"], counter_path=cell["verbose"],
                           declared_T=str(ck[1]), feature=feature)
        except ft.InstrumentError as e:
            anomalies.append(f"{fmt_cell(ck)}: fulcrum_total REFUSED trace: {e}")
            continue
        if b["is_production"] is False:
            anomalies.append(f"{fmt_cell(ck)}: routing guard REFUSED "
                             f"({b['seed_reason']}) — attribution rows dropped")
            continue
        cons = b["consumer"]
        span = cons.get("span", 0)
        if not span:
            continue
        for cls in ("compute", "output", "wait", "idle"):
            ms = cons.get(cls, 0.0) / 1000.0
            trace_components.setdefault(cls, {})[ck] = (ms, span / 1000.0)

    # ---- knob rows (the causal tier) -------------------------------------------
    for ck, cell in sorted(run["cells"].items()):
        for kname, kdata in sorted(cell["knobs"].items()):
            envkv, pred, desc = KNOB_ENV.get(kname, ("?", "none", kname))
            v = knob_verdict(kdata["base"], kdata["knob"])
            if v.get("status") == "NO-DATA":
                continue
            eff = run["effects"].get(kname)
            ev, enote = effect_check(pred, eff.get("base", "") if eff else "",
                                     eff.get("knob", "") if eff else "")
            if ev is False:
                status = f"EFFECT-CHECK-FAILED ({enote}) — A/B NOT causal"
                tier, rank = 5, 0.0
            else:
                d, mg = v["delta_ms"], v["margin_ms"]
                if v["status"] == "CAUSAL-VERIFIED-COSTS":
                    status = (f"CAUSAL-VERIFIED: shipped default COSTS "
                              f"{-d:.1f}ms max-arm-spread={mg:.1f}ms here (alt arm faster)")
                    tier, rank = 1, -d
                elif v["status"] == "CAUSAL-VERIFIED-PAYS":
                    status = (f"CAUSAL-VERIFIED: feature PAYS {d:.1f}ms "
                              f"max-arm-spread={mg:.1f}ms (disabling it loses)")
                    tier, rank = 3, d
                else:
                    status = (f"CAUSAL-NULL: |Δ|={abs(d):.1f}ms ≤ "
                              f"max-arm-spread={mg:.1f}ms (bounded)")
                    tier, rank = 4, abs(d)
                if ev is None:
                    status += f" [{enote}]"
                elif ev is True:
                    status += f" [effect-verified: {enote}]"
            dist = (f"base[{dist_health_str(kdata['base'])}] "
                    f"knob[{dist_health_str(kdata['knob'])}] "
                    f"{v.get('resolution','')}"
                    + (f"(N->{v['n_needed']})" if v.get("n_needed") else ""))
            # RSS per arm from meta.txt (written by timed_masked RSS extension).
            meta = kdata.get("meta", {})
            rss_base = meta.get("rss_base_mb", "")
            rss_knob = meta.get("rss_knob_mb", "")
            if rss_base and rss_knob:
                try:
                    rb_f, rk_f = float(rss_base), float(rss_knob)
                    pct = (rk_f - rb_f) / rb_f * 100 if rb_f else 0.0
                    sign = "+" if pct >= 0 else ""
                    rss_str = (f"rss base={rb_f:.0f}MB knob={rk_f:.0f}MB "
                               f"({sign}{pct:.0f}%)")
                except (ValueError, ZeroDivisionError):
                    rss_str = f"rss base={rss_base}MB knob={rss_knob}MB"
            else:
                rss_str = "rss N/A (pre-RSS capture run)"
            rows.append({
                "component": f"knob.{kname} ({desc})",
                "cells": fmt_cell(ck),
                "attrib": f"Δ(alt-base)={v['delta_ms']:+.1f}ms @ canonical mask",
                "status": status + unfrozen_tag,
                "dist": dist,
                "rss": rss_str,
                "verify": (f"scripts/bench/decide.sh --cells {ck[0]}:{ck[1]} "
                           f"--knob-cells {ck[0]}:{ck[1]} --knobs {kname} "
                           f"--knob-n 21 --bin {run['manifest'].get('bin')}"),
                "tier": tier, "rank_ms": rank,
            })

    # ---- engine micro-profile rows (per corpus; HYPOTHESIS tier) ----------------
    for ck, cell in sorted(run["cells"].items()):
        prof = cell.get("prof")
        if not prof or not prof["classes"]:
            continue
        for d in bank_divergence(ck, prof):
            anomalies.append(f"{fmt_cell(ck)}: {d}")
        w = cell_walls.get(ck)
        gap_ms = max(w["gap_ms"], 0.0) if w else 0.0
        for cls_name, c in sorted(prof["classes"].items(),
                                  key=lambda kv: -kv[1]["share_pct"]):
            bounded = gap_ms * c["share_pct"] / 100.0
            rows.append({
                "component": f"engine.{cls_name}",
                "cells": fmt_cell(ck),
                "attrib": (f"{c['share_pct']:.1f}% of classed cyc, "
                           f"{c['cyc_iter']:.1f} cyc/iter, iters={c['iters']:,}"),
                "status": (f"HYPOTHESIS: bounded ≤{bounded:.0f}ms ESTIMATE "
                           f"(= cell rg-gap {gap_ms:.0f}ms × class share; a "
                           f"partition, not a promise). Perturb: "
                           f"{SUGGESTED_PERTURBATION['compute']}"),
                "dist": "prof=1-shot counters (unfrozen-counters label)",
                "verify": (f"GZIPPY_CONTIG_PROF=1 GZIPPY_VERBOSE=1 taskset -c <mask> "
                           f"{run['manifest'].get('bin')} -d -c -p {ck[1]} "
                           f"/root/{ck[0]}.gz >/dev/null"),
                "tier": 2, "rank_ms": bounded,
            })
        if prof.get("wrapper_calls"):
            anomalies.append(f"{fmt_cell(ck)}: WRAPPER calls="
                             f"{prof['wrapper_calls']} (expected 0 — contig should "
                             f"be the sole production engine)")

    # ---- trace-component rows (HYPOTHESIS tier) ---------------------------------
    for cls, cells in sorted(trace_components.items()):
        worst_ck, (worst_ms, span_ms) = max(cells.items(), key=lambda kv: kv[1][0])
        cells_str = ",".join(fmt_cell(c) for c in sorted(cells))
        share = 100.0 * worst_ms / span_ms if span_ms else 0
        rows.append({
            "component": f"pipeline.consumer.{cls}",
            "cells": cells_str,
            "attrib": (f"worst {fmt_cell(worst_ck)}: {worst_ms:.1f}ms "
                       f"({share:.0f}% of wall-critical span)"),
            "status": (f"HYPOTHESIS (attribution only — NOT causal). Perturb: "
                       f"{SUGGESTED_PERTURBATION.get(cls, 'design a knob first')}"),
            "dist": "trace=1-shot (unfrozen-counters label)",
            "verify": (f"python3 scripts/fulcrum_total.py "
                       f"<artdir>/cell_{worst_ck[0]}_T{worst_ck[1]}/trace.json "
                       f"--feature {feature}"),
            "tier": 2, "rank_ms": worst_ms if cls != "wait" else worst_ms * 0.25,
            # wait is demoted: it is a SYMPTOM of the producer, not a lever site.
        })

    rows.sort(key=lambda r: (r["tier"], -r["rank_ms"]))

    # ---- DO THIS NEXT ------------------------------------------------------------
    do_next = None
    for r in rows:
        if r["tier"] == 1:
            if "reverted" in r["component"]:
                action = ("reconcile with the prior gated revert + check RSS "
                          "before flipping")
            else:
                action = "fix/condition the feature"
            do_next = (f"{r['component']} on {r['cells']} — the shipped default "
                       f"measurably COSTS wall ({r['status'].split(':')[1].strip()}). "
                       f"Re-verify at N=21 then {action}: {r['verify']}")
            break
    if do_next is None:
        for r in rows:
            if r["tier"] == 2 and r["component"].startswith("engine."):
                do_next = (f"{r['component']} on {r['cells']} — top bounded "
                           f"HYPOTHESIS ({r['attrib']}). Run the pre-registered "
                           f"perturbation BEFORE any work-stretch: "
                           f"{SUGGESTED_PERTURBATION['compute']}")
                break
    if do_next is None and rows:
        do_next = f"{rows[0]['component']} — see its row; no causal action surfaced."
    if do_next is None:
        do_next = "no rankable rows (all captures refused?) — fix the run first."

    return {"header": header, "scoreboard": scoreboard, "rows": rows,
            "anomalies": anomalies, "do_next": do_next,
            "cell_walls": cell_walls}


def print_report(rep):
    print("=" * 100)
    print("fulcrum decide — ONE-RUN decision table (plans/fulcrum2-charter.md)")
    print("=" * 100)
    for h in rep["header"]:
        print(h)
    print("\n-- CELL SCOREBOARD (wall, interleaved, sha-verified; bar = 0.99x EVERY T) --")
    for s in rep["scoreboard"]:
        print(s)
    print("\n-- RANKED COMPONENTS (tier 1 causal-COSTS > tier 2 hypotheses > "
          "tier 3 confirms > tier 4 null) --")
    for i, r in enumerate(rep["rows"], 1):
        print(f"\n[{i:2d}] {r['component']}   cells: {r['cells']}")
        print(f"     attribution : {r['attrib']}")
        print(f"     status      : {r['status']}")
        print(f"     distribution: {r['dist']}")
        if "rss" in r:
            print(f"     rss         : {r['rss']}")
        print(f"     re-verify   : {r['verify']}")
    if rep["anomalies"]:
        print("\n-- ANOMALIES (verbatim; investigate before trusting affected rows) --")
        for a in rep["anomalies"]:
            print(f"  !! {a}")
    print("\n" + "=" * 100)
    print(f"DO THIS NEXT: {rep['do_next']}")
    print("=" * 100)


# ---------------------------------------------------------------------------
# SELF-TEST (charter section "Self-tests"; synthetic artifacts, KNOWN structure).
# ---------------------------------------------------------------------------

def selftest():
    import tempfile
    failures = []

    def check(cond, msg):
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    print("=== fulcrum_decide --selftest ===")

    # 1. knob harness: known-null knob => CAUSAL-NULL (the directive's must-have).
    base = [1.000, 1.002, 1.001, 1.003, 1.002, 1.001, 1.004]
    v = knob_verdict(base, list(base))
    check(v["status"] == "CAUSAL-NULL",
          f"known-null knob -> CAUSAL-NULL (got {v['status']})")
    # shifted beyond spread => VERIFIED with correct sign
    v2 = knob_verdict(base, [x - 0.050 for x in base])  # alt arm 50ms faster
    check(v2["status"] == "CAUSAL-VERIFIED-COSTS" and v2["delta_ms"] < 0,
          f"-50ms shift -> CAUSAL-VERIFIED-COSTS (got {v2['status']})")
    v3 = knob_verdict(base, [x + 0.050 for x in base])
    check(v3["status"] == "CAUSAL-VERIFIED-PAYS",
          f"+50ms shift -> CAUSAL-VERIFIED-PAYS (got {v3['status']})")
    # shift INSIDE spread => NULL with bound (sub-spread never a finding)
    wide = [1.00, 1.05, 1.02, 1.08, 1.01, 1.06, 1.03]
    v4 = knob_verdict(wide, [x + 0.01 for x in wide])
    check(v4["status"] == "CAUSAL-NULL",
          f"sub-spread +10ms on 8% spread -> CAUSAL-NULL (got {v4['status']})")

    # 2. bimodality: two-mode flagged, unimodal control not.
    bi = [1.00, 1.01, 1.005, 1.30, 1.31, 1.305, 1.302]
    uni = [1.00, 1.01, 1.02, 1.03, 1.015, 1.025, 1.005]
    check(bimodal(bi) is True, "bimodal sample FLAGGED")
    check(bimodal(uni) is False, "unimodal control NOT flagged")
    # Degenerate branch: [1,1,1,1,1.01] has right-side size=1 -> NOT bimodal.
    # (gate repro: the old code returned True here; left=4 right=1 fails >=2).
    deg = [1, 1, 1, 1, 1.01]
    check(bimodal(deg) is False,
          "degenerate [1,1,1,1,1.01] NOT flagged (right side has only 1 sample)")

    # 3. N-needed monotone: smaller delta -> larger N; resolved -> no N.
    r1 = resolution(0.001, 0.010, 0.010, 9)
    r2 = resolution(0.005, 0.010, 0.010, 9)
    r3 = resolution(0.020, 0.010, 0.010, 9)
    check(r1[0] == "UNRESOLVED" and r2[0] == "UNRESOLVED" and r1[1] > r2[1],
          f"N-needed monotone (Δ1ms->{r1[1]}, Δ5ms->{r2[1]})")
    check(r3 == ("RESOLVED", None), "supra-spread delta -> RESOLVED, no N-needed")

    # 4. guard matrix (the re-derived fulcrum_total.seeding_guard).
    g1 = ft.seeding_guard({"window_seeded": 16, "finished_no_flip": 4,
                           "flip_to_clean": 12, "seeded_block": 16},
                          feature="gzippy-native")
    check(g1[0] is True, "guard: production-seeded native ACCEPT")
    g2 = ft.seeding_guard({"window_seeded": 16, "seed_replay_hits": 16})
    check(g2[0] is False, "guard: SEED_WINDOWS replay REFUSE")
    g3 = ft.seeding_guard({"isal_chunks": 14, "finished_no_flip": 2},
                          feature="gzippy-native")
    check(g3[0] is False, "guard: isal_chunks>0 on native REFUSE (oracle)")
    g4 = ft.seeding_guard({"isal_chunks": 14, "finished_no_flip": 2,
                           "window_seeded": 9}, feature="gzippy-isal")
    check(g4[0] is True, "guard: isal_chunks>0 on isal ACCEPT (production)")
    g5 = ft.seeding_guard({})
    check(g5[0] is None, "guard: no sidecar INCONCLUSIVE")

    # 5. prof parser on a synthetic dump in the exact binary format.
    prof_txt = (
        "[contig-prof] CONTIG (Block::decode_clean_into_contig):\n"
        "  calls=1768 total_cyc=1000000 classed_cyc=900000 (90.0% of total; rest=careful+entry/exit+unchained tail)\n"
        "  lit1   : iters=      100000 cyc=        90000   10.0% of classed,    0.9 cyc/iter\n"
        "  litpack: iters=       50000 cyc=        45000    5.0% of classed,    0.9 cyc/iter, lits=120000\n"
        "  litchn : iters=      200000 cyc=       206000   22.9% of classed,    1.0 cyc/iter, lits=500000\n"
        "  backref: iters=       16140 cyc=       563400   62.6% of classed,   34.9 cyc/iter, bytes=900000 dist_long=3\n"
        "  careful: cyc=50000 (5.0% of total) outer_iters=123\n"
        "  disttbl: builds=1765 reuses=3 (P3.4 dynamic-block dist_table amortization)\n"
        "[contig-prof] WRAPPER (decode_huffman_body_resumable):\n"
        "  calls=0 total_cyc=0 classed_cyc=0 (0.0%)\n")
    p = parse_prof(prof_txt)
    check(p["classes"]["backref"]["cyc_iter"] == 34.9
          and p["classes"]["backref"]["share_pct"] == 62.6
          and p["disttbl"] == (1765, 3) and p["wrapper_calls"] == 0,
          "prof parser: classes/cyc-iter/disttbl/wrapper parsed exactly")
    # bank check: consistent prof -> no divergence; moved share -> flagged
    check(bank_divergence(("silesia", 8), p) == [],
          "bank comparator: banked-consistent prof -> no divergence")
    p_moved = parse_prof(prof_txt.replace("  62.6% of classed,   34.9",
                                          "  30.0% of classed,   34.9"))
    check(any("DIVERGES-FROM-BANK" in d
              for d in bank_divergence(("silesia", 8), p_moved)),
          "bank comparator: >25% share move -> DIVERGES-FROM-BANK")
    p_small = parse_prof(prof_txt.replace("  62.6% of classed,   34.9",
                                          "  60.0% of classed,   34.9"))
    check(bank_divergence(("silesia", 8), p_small) == [],
          "bank comparator: 4% share move NOT flagged")

    # 6. effect predicates.
    ok, _ = effect_check("verbose_seeded", "",
                         "seeded_block=0 seeded_wrapper=16 ")
    check(ok is True, "effect: seeded_block kill-switch verified")
    bad, _ = effect_check("verbose_seeded", "",
                          "seeded_block=16 seeded_wrapper=0 ")
    check(bad is False, "effect: ineffective switch CAUGHT")
    okd, _ = effect_check("prof_dist",
                          "disttbl: builds=2790 reuses=7 ",
                          "disttbl: builds=0 reuses=0 ")
    check(okd is True, "effect: dist_amort off => P3.4 counters dead "
                       "(builds=0/reuses=0; the counting sites live only in "
                       "the amortized arm, marker_inflate.rs:2262/2266)")
    badd, _ = effect_check("prof_dist",
                           "disttbl: builds=2790 reuses=7 ",
                           "disttbl: builds=2790 reuses=7 ")
    check(badd is False, "effect: dist_amort knob arm still amortized => CAUGHT")
    vacd, _ = effect_check("prof_dist",
                           "disttbl: builds=0 reuses=0 ",
                           "disttbl: builds=0 reuses=0 ")
    check(vacd is None, "effect: no dynamic blocks => predicate vacuous (None)")
    none_v, _ = effect_check("none", "", "")
    check(none_v is None, "effect: knob without counter => EFFECT-UNVERIFIED (None)")
    # rpmalloc_stats predicate (slab_alloc): knob arm must have [rpmalloc], base not.
    rp_ok, rp_note = effect_check(
        "rpmalloc_stats",
        "some base output without rpmalloc line",
        "[rpmalloc final] mapped_peak=48M mapped_total=192M unmapped_total=144M "
        "cached=0.0M huge_alloc_peak=0M")
    check(rp_ok is True and "48" in rp_note,
          "effect: rpmalloc_stats knob-arm has stats, base absent => EFFECT-VERIFIED")
    rp_fail_base, _ = effect_check(
        "rpmalloc_stats",
        "[rpmalloc final] mapped_peak=48M mapped_total=192M unmapped_total=144M "
        "cached=0.0M huge_alloc_peak=0M",
        "[rpmalloc final] mapped_peak=48M mapped_total=192M unmapped_total=144M "
        "cached=0.0M huge_alloc_peak=0M")
    check(rp_fail_base is False, "effect: rpmalloc_stats both arms have stats => CAUGHT")
    rp_fail_missing, _ = effect_check("rpmalloc_stats", "base output", "knob output")
    check(rp_fail_missing is False,
          "effect: rpmalloc_stats no stats line in knob arm => CAUGHT")

    # 7. end-to-end on a synthetic artifact dir: ranked table + DO-THIS-NEXT.
    d = tempfile.mkdtemp(prefix="fulcrum_decide_st_")
    cdir = os.path.join(d, "cell_silesia_T1")
    kdir = os.path.join(cdir, "knob_hit_drive")
    k2dir = os.path.join(cdir, "knob_dist_amort")
    os.makedirs(kdir); os.makedirs(k2dir)
    with open(os.path.join(d, "manifest.txt"), "w") as f:
        f.write("runid=st\nbin=/root/bin-test\nbin_sha=deadbeef\n"
                "feature=gzippy-native\nrg_version=rapidgzip 0.16.0\n"
                "freeze_state=acknowledged\nquiet_state=quiet\n"
                "governor=performance\nno_turbo=1\nrunnable_avg=1.0\n"
                "cell_done=silesia:1:mask=0:sha_ok=1\n")
    gz = [1.380, 1.382, 1.379, 1.385, 1.381, 1.383, 1.380]
    rg = [0.920, 0.922, 0.918, 0.925, 0.921, 0.919, 0.923]
    with open(os.path.join(cdir, "wall_gz.txt"), "w") as f:
        f.write("\n".join(map(str, gz)))
    with open(os.path.join(cdir, "wall_rg.txt"), "w") as f:
        f.write("\n".join(map(str, rg)))
    with open(os.path.join(cdir, "prof.txt"), "w") as f:
        f.write(prof_txt)
    # hit_drive: disabling it makes the cell FASTER by 60ms (feature COSTS).
    with open(os.path.join(kdir, "base.txt"), "w") as f:
        f.write("\n".join(map(str, gz)))
    with open(os.path.join(kdir, "knob.txt"), "w") as f:
        f.write("\n".join(str(x - 0.060) for x in gz))
    with open(os.path.join(kdir, "meta.txt"), "w") as f:
        f.write("knob=hit_drive\nenv=GZIPPY_NO_HIT_DRIVE=1\npred=none\n"
                "cell=silesia:1\nmask=0\nsha_ok=1\n")
    # dist_amort: null.
    with open(os.path.join(k2dir, "base.txt"), "w") as f:
        f.write("\n".join(map(str, gz)))
    with open(os.path.join(k2dir, "knob.txt"), "w") as f:
        f.write("\n".join(map(str, gz)))
    with open(os.path.join(k2dir, "meta.txt"), "w") as f:
        f.write("knob=dist_amort\nenv=GZIPPY_DIST_AMORT=0\npred=prof_dist\n"
                "cell=silesia:1\nmask=0\nsha_ok=1\n")
    edir = os.path.join(d, "knob_effects_silesia_T1")
    os.makedirs(edir)
    with open(os.path.join(edir, "effect_base_dist_amort.txt"), "w") as f:
        f.write("disttbl: builds=2790 reuses=7 \n")
    with open(os.path.join(edir, "effect_knob_dist_amort.txt"), "w") as f:
        f.write("disttbl: builds=0 reuses=0 \n")
    run = load_run(d)
    rep = analyze_run(run)
    check(any("knob.hit_drive" in r["component"] and r["tier"] == 1
              for r in rep["rows"]),
          "e2e: hit_drive-COSTS row lands in tier 1")
    check(any("knob.dist_amort" in r["component"] and r["tier"] == 4
              and "CAUSAL-NULL" in r["status"] for r in rep["rows"]),
          "e2e: dist_amort null -> tier 4 CAUSAL-NULL")
    check(rep["rows"][0]["component"].startswith("knob.hit_drive"),
          "e2e: ranking puts the causal-COSTS row first")
    check("knob.hit_drive" in rep["do_next"],
          "e2e: DO-THIS-NEXT picks the top CAUSAL-VERIFIED-COSTS row")
    check(any(r["component"] == "engine.backref" for r in rep["rows"]),
          "e2e: engine.backref hypothesis row present from prof")
    # determinism
    rep2 = analyze_run(load_run(d))
    check([r["component"] for r in rep["rows"]]
          == [r["component"] for r in rep2["rows"]],
          "e2e: ranked table deterministic")

    # 8. UNFROZEN refusal + --allow-thaw label.
    with open(os.path.join(d, "manifest.txt"), "a") as f:
        f.write("")  # rewrite below
    man_path = os.path.join(d, "manifest.txt")
    txt = open(man_path).read().replace("freeze_state=acknowledged",
                                        "freeze_state=thawed")
    open(man_path, "w").write(txt)
    raised = False
    try:
        analyze_run(load_run(d))
    except ft.InstrumentError:
        raised = True
    check(raised, "UNFROZEN run REFUSED without --allow-thaw")
    rep3 = analyze_run(load_run(d), allow_thaw=True)
    check(any("UNFROZEN" in r["status"] for r in rep3["rows"]),
          "--allow-thaw labels every wall-derived row UNFROZEN")

    # 9. DO-THIS-NEXT without any causal action -> top engine hypothesis.
    import shutil
    shutil.rmtree(kdir)
    txt = open(man_path).read().replace("freeze_state=thawed",
                                        "freeze_state=frozen")
    open(man_path, "w").write(txt)
    rep4 = analyze_run(load_run(d))
    check("engine.backref" in rep4["do_next"],
          "no causal action -> DO-THIS-NEXT = top bounded engine HYPOTHESIS "
          "with its perturbation")

    # 10. "reverted" knob DO-THIS-NEXT uses "reconcile" phrasing (not fix/condition).
    d2 = tempfile.mkdtemp(prefix="fulcrum_decide_st10_")
    cdir2 = os.path.join(d2, "cell_silesia_T1")
    kslab = os.path.join(cdir2, "knob_slab_alloc")
    os.makedirs(kslab)
    with open(os.path.join(d2, "manifest.txt"), "w") as f:
        f.write("runid=st10\nbin=/root/bin-test\nbin_sha=deadbeef\n"
                "feature=gzippy-native\nrg_version=rapidgzip 0.16.0\n"
                "freeze_state=frozen\nquiet_state=quiet\n"
                "governor=performance\nno_turbo=1\nrunnable_avg=1.0\n"
                "cell_done=silesia:1:mask=0:sha_ok=1\n")
    gz2 = [1.380, 1.382, 1.379, 1.385, 1.381, 1.383, 1.380]
    rg2 = [0.920, 0.922, 0.918, 0.925, 0.921, 0.919, 0.923]
    with open(os.path.join(cdir2, "wall_gz.txt"), "w") as f:
        f.write("\n".join(map(str, gz2)))
    with open(os.path.join(cdir2, "wall_rg.txt"), "w") as f:
        f.write("\n".join(map(str, rg2)))
    # slab_alloc knob arm is 60ms FASTER (feature COSTS — enabling slab costs wall
    # means the DEFAULT (off) costs; wait, the knob arm is GZIPPY_SLAB_ALLOC=1 which
    # ENABLES the slab; if enabling it makes it faster, the default (off) COSTS wall).
    with open(os.path.join(kslab, "base.txt"), "w") as f:
        f.write("\n".join(map(str, gz2)))
    with open(os.path.join(kslab, "knob.txt"), "w") as f:
        f.write("\n".join(str(x - 0.060) for x in gz2))  # knob (slab ON) is faster
    with open(os.path.join(kslab, "meta.txt"), "w") as f:
        f.write("knob=slab_alloc\nenv=GZIPPY_SLAB_ALLOC=1\npred=rpmalloc_stats\n"
                "cell=silesia:1\nmask=0\nsha_ok=1\n")
    # Effect capture: knob arm has rpmalloc stats, base does not.
    edir2 = os.path.join(d2, "knob_effects_silesia_T1")
    os.makedirs(edir2)
    with open(os.path.join(edir2, "effect_base_slab_alloc.txt"), "w") as f:
        f.write("no rpmalloc output in base arm\n")
    with open(os.path.join(edir2, "effect_knob_slab_alloc.txt"), "w") as f:
        f.write("[rpmalloc final] mapped_peak=64M mapped_total=256M "
                "unmapped_total=192M cached=0.1M huge_alloc_peak=0M\n")
    rep10 = analyze_run(load_run(d2))
    check(any("slab_alloc" in r["component"] and r["tier"] == 1
              for r in rep10["rows"]),
          "e2e-reverted: slab_alloc CAUSAL-VERIFIED-COSTS lands tier 1")
    check("reconcile" in rep10["do_next"],
          "e2e-reverted: DO-THIS-NEXT for 'reverted' knob uses 'reconcile' phrasing")
    check("fix/condition" not in rep10["do_next"],
          "e2e-reverted: 'fix/condition' phrasing absent for reverted knob")
    check(any("rpmalloc" in r.get("status", "") for r in rep10["rows"]
              if "slab_alloc" in r["component"]),
          "e2e-reverted: rpmalloc effect-verified note in slab_alloc status")

    print(f"\n=== SELFTEST {'PASSED' if not failures else 'FAILED'} "
          f"({len(failures)} failure(s)) ===")
    return 0 if not failures else 1


def main():
    argv = sys.argv[1:]
    if "--selftest" in argv:
        sys.exit(selftest())
    allow_thaw = "--allow-thaw" in argv
    dirs = [a for a in argv if not a.startswith("--")]
    if not dirs:
        print(__doc__)
        sys.exit(1)
    try:
        run = load_run(dirs[0])
        rep = analyze_run(run, allow_thaw=allow_thaw)
    except ft.InstrumentError as e:
        print(f"\n[INSTRUMENT REFUSED] {e}")
        sys.exit(2)
    print_report(rep)


if __name__ == "__main__":
    main()
