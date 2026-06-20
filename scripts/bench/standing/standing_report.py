#!/usr/bin/env python3
"""standing_report.py — render the ONE ground-truth table from _standing_guest.sh CSVs.

GATE-1 (significance): best-of-N with inter-run spread; a Delta smaller than the
combined spread is reported as TIE, never a win (CLAUDE.md Measurement PROTOCOL).

Self-validation that LICENSES trusting ratios on a loaded shared box:
  - RG2/RG and GZ2/GZ A/A self-tests must sit at ~1.0; |ratio-1| is reported and
    a cell whose self-test exceeds SELFTEST_TOL is stamped UNTRUSTED (box too noisy).

For each (corpus, T) cell it prints, per tool:
  cyc/B (best, P-core cycles / decompressed byte), wall best+spread,
  ratio-vs-rg, ratio-vs-igzip, achieved-GHz spread, LLC-miss%.

Wall is the parity metric for ratios (decode is wall-bound at T>1); cyc/B is the
single-core work metric (most meaningful at T1).  Usage: standing_report.py <art-dir>
"""
import sys, os, glob, statistics

SELFTEST_TOL = 0.05   # |A/A ratio - 1| above this => cell ratios untrusted (noisy box)

def parse_csv(path):
    """Return dict of metric->float from a `perf stat -x,` file. Missing => None."""
    vals = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                raw, _unit, ev = parts[0], parts[1], parts[2]
                try:
                    v = float(raw)
                except ValueError:
                    continue
                if "duration_time" in ev:
                    vals["wall_ns"] = v
                elif "cycles" in ev:
                    vals["cycles"] = v
                elif "instructions" in ev:
                    vals["instr"] = v
                elif "task-clock" in ev:
                    vals["taskclock_ms"] = v
                elif "cache-references" in ev:
                    vals["cache_ref"] = v
                elif "cache-misses" in ev:
                    vals["cache_miss"] = v
    except FileNotFoundError:
        return None
    return vals or None

def load_meta(art):
    meta = {"bytes": {}}
    with open(os.path.join(art, "meta.txt")) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "bytes":
                meta["bytes"][t[1]] = int(t[2])
            elif len(t) >= 2:
                meta[t[0]] = " ".join(t[1:])
    return meta

def collect(art):
    """cells[(corp,T)][arm] = list of per-rep metric dicts."""
    cells = {}
    for p in glob.glob(os.path.join(art, "*.T*.*.csv")):
        base = os.path.basename(p)[:-4]              # corp.Tn.ARM.rep
        parts = base.split(".")
        if len(parts) < 4:
            continue
        rep = parts[-1]; arm = parts[-2]; tcol = parts[-3]; corp = ".".join(parts[:-3])
        if not tcol.startswith("T"):
            continue
        T = int(tcol[1:])
        v = parse_csv(p)
        if v is None:
            continue
        cells.setdefault((corp, T), {}).setdefault(arm, []).append(v)
    return cells

def best_wall(samples):
    walls = [s["wall_ns"] for s in samples if s.get("wall_ns")]
    return min(walls) if walls else None

def wall_spread(samples):
    walls = sorted(s["wall_ns"] for s in samples if s.get("wall_ns"))
    if len(walls) < 2:
        return 0.0
    # spread = (median - min)/min : a robust one-sided jitter estimate vs the best.
    med = statistics.median(walls)
    return (med - walls[0]) / walls[0] if walls[0] else 0.0

def best_cycb(samples, nbytes):
    cs = [s["cycles"] for s in samples if s.get("cycles")]
    return (min(cs) / nbytes) if cs and nbytes else None

def ghz_spread(samples):
    g = []
    for s in samples:
        if s.get("cycles") and s.get("taskclock_ms"):
            g.append(s["cycles"] / (s["taskclock_ms"] * 1e6))   # cyc / ms -> GHz
    if len(g) < 2:
        return (g[0] if g else None, 0.0)
    return (statistics.median(g), (max(g) - min(g)) / statistics.median(g))

def llc_pct(samples):
    refs = [s["cache_ref"] for s in samples if s.get("cache_ref")]
    mis = [s["cache_miss"] for s in samples if s.get("cache_miss")]
    if refs and mis and statistics.median(refs) > 0:
        return 100.0 * statistics.median(mis) / statistics.median(refs)
    return None

def main():
    art = sys.argv[1]
    meta = load_meta(art)
    cells = collect(art)
    sha = meta.get("sha", "?"); flavor = meta.get("flavor", "?")
    nt = meta.get("no_turbo", "?"); gov = meta.get("gov", "?"); load = meta.get("load_start", "?")
    n = meta.get("n", "?")

    print("=" * 100)
    print(f"GROUND-TRUTH STANDING — gzippy-native vs rapidgzip(rg) vs igzip(ig)")
    print(f"sha={sha[:12]} flavor={flavor} N={n} | box: no_turbo={nt} gov={gov} load_start={load}")
    print("  STAMP: Intel i7-13700T LXC ONLY — NOT-YET-LAW (AMD/Zen2 owed for LAW).")
    if nt == "0":
        print("  NOTE: turbo ON + governor not pinned => ABSOLUTE cyc/B/GHz drift; trust RELATIVE ratios gated by the A/A self-test.")
    print("=" * 100)

    hdr = (f"{'cell':<16}{'tool':<5}{'cyc/B':>9}{'wall_ms':>10}{'spread':>8}"
           f"{'/rg':>7}{'/ig':>7}{'GHz':>6}{'GHzσ':>7}{'LLC%':>7}  verdict")
    any_untrusted = False
    forks = []
    for (corp, T) in sorted(cells.keys()):
        nbytes = meta["bytes"].get(corp)
        arms = cells[(corp, T)]
        print("-" * 100)
        print(hdr)
        # self-tests
        st = {}
        for a, b in (("RG", "RG2"), ("GZ", "GZ2")):
            wa, wb = best_wall(arms.get(a, [])), best_wall(arms.get(b, []))
            st[a] = abs(wb / wa - 1.0) if (wa and wb) else None
        rg_st = st.get("RG"); gz_st = st.get("GZ")
        untrusted = (rg_st is not None and rg_st > SELFTEST_TOL) or \
                    (gz_st is not None and gz_st > SELFTEST_TOL)
        any_untrusted = any_untrusted or untrusted

        rg_wall = best_wall(arms.get("RG", []))
        ig_wall = best_wall(arms.get("IG", []))
        rows = {}
        for arm, label in (("GZ", "gz"), ("RG", "rg"), ("IG", "ig")):
            s = arms.get(arm, [])
            if not s:
                continue
            bw = best_wall(s); cb = best_cycb(s, nbytes); sp = wall_spread(s)
            ghz, gsp = ghz_spread(s); llc = llc_pct(s)
            r_rg = bw / rg_wall if (bw and rg_wall) else None
            r_ig = bw / ig_wall if (bw and ig_wall) else None
            rows[arm] = dict(label=label, bw=bw, cb=cb, sp=sp, r_rg=r_rg, r_ig=r_ig,
                             ghz=ghz, gsp=gsp, llc=llc)

        def fmt(x, f="{:.3f}"):
            return f.format(x) if x is not None else "  -"
        for arm in ("GZ", "RG", "IG"):
            if arm not in rows:
                continue
            r = rows[arm]
            verdict = ""
            if arm == "GZ" and r["r_rg"] is not None:
                # TIE if |ratio-1| within the combined wall spread of the two arms
                comb = r["sp"] + rows.get("RG", {}).get("sp", 0.0)
                if untrusted:
                    verdict = "UNTRUSTED(box)"
                elif abs(r["r_rg"] - 1.0) <= comb:
                    verdict = f"TIE vs rg (±{comb*100:.1f}%)"
                elif r["r_rg"] < 1.0:
                    verdict = f"gz FASTER than rg by {(1-r['r_rg'])*100:.1f}%"
                else:
                    verdict = f"gz SLOWER than rg by {(r['r_rg']-1)*100:.1f}%"
            print(f"{corp+' T'+str(T):<16}{r['label']:<5}"
                  f"{fmt(r['cb'],'{:.2f}'):>9}"
                  f"{fmt(r['bw']/1e6 if r['bw'] else None,'{:.2f}'):>10}"
                  f"{fmt(r['sp']*100,'{:.1f}'):>7}%"
                  f"{fmt(r['r_rg']):>7}{fmt(r['r_ig']):>7}"
                  f"{fmt(r['ghz'],'{:.2f}'):>6}{fmt(r['gsp']*100,'{:.1f}'):>6}%"
                  f"{fmt(r['llc'],'{:.1f}'):>6}%  {verdict}")
        st_s = (f"  self-test A/A: rg|gz = "
                f"{('%.1f%%'%(rg_st*100)) if rg_st is not None else '-'} / "
                f"{('%.1f%%'%(gz_st*100)) if gz_st is not None else '-'}"
                f"  (tol {SELFTEST_TOL*100:.0f}% => {'UNTRUSTED' if untrusted else 'OK'})")
        print(st_s)
        if "GZ" in rows and rows["GZ"]["r_rg"] is not None and not untrusted:
            forks.append((corp, T, rows["GZ"]["r_rg"]))

    print("=" * 100)
    print("FORK SUMMARY (trusted cells only; ratio = gz_wall / rg_wall, <1.0 = gz wins):")
    if not forks:
        print("  (no trusted cells — box too loaded; re-run when load drops)")
    worst = None
    for corp, T, r in forks:
        tag = "WIN/TIE" if r <= 1.02 else "LOSS"
        print(f"  {corp:<10} T{T:<3} gz/rg={r:.3f}  {tag}")
        if r > 1.02 and (worst is None or r > worst[2]):
            worst = (corp, T, r)
    multi = [r for c, t, r in forks if t >= 2]
    if multi:
        wm = max(multi)
        if wm <= 1.02:
            print(f"  => FORK-A: T>=2 ~parity-or-better vs rg confirmed (worst T>=2 gz/rg={wm:.3f}). "
                  f"Intel-only / NOT-YET-LAW (AMD owed).")
        else:
            print(f"  => FORK-B: a T>=2 cell genuinely loses (worst={worst}). Rule out fingerprint/artifact, "
                  f"then follow T2-LOCATE decision tree.")
    if any_untrusted:
        print("  WARNING: >=1 cell failed the A/A self-test (loaded box) — those ratios were withheld.")
    print("=" * 100)

if __name__ == "__main__":
    main()
