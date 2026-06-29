#!/usr/bin/env python3
"""Parse `fulcrum wall --steady` stdout into a machine-readable JSON record.

Reads one fulcrum-steady log on stdin (or a file arg) plus metadata flags and
emits a JSON object with the verdict fields. Pure stdlib, no deps.
"""
import sys, re, json, argparse

def parse(text):
    rec = {}
    # Gate-0 native + sha
    rec["gate0_native_gz"] = "PASS :: gz = native Mach-O" in text
    rec["gate0_native_ref"] = "PASS :: ld = native Mach-O" in text
    m = re.search(r"oracle sha \(gzip -dc\) = ([0-9a-f]+)", text)
    rec["oracle_sha12"] = m.group(1) if m else None
    rec["gate0_gz_sha_eq_oracle"] = "gz output sha == oracle" in text
    rec["gate0_ref_sha_eq_oracle"] = "ld output sha == oracle" in text
    # warmup
    m = re.search(r"steady≈([0-9.]+) GHz", text)
    rec["steady_ghz"] = float(m.group(1)) if m else None
    rec["warmup_converged"] = "CONVERGED" in text and "BUDGET-LIMITED" not in text.split("CONVERGED")[0][-40:]
    # cohort rows: "c  gz/ld  A/A  effGHz  kept  drop  freqΔ%"
    cohorts = []
    for mm in re.finditer(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s*$",
                          text, re.M):
        cohorts.append({
            "cohort": int(mm.group(1)),
            "gz_ref": float(mm.group(2)),
            "aa": float(mm.group(3)),
            "eff_ghz": float(mm.group(4)),
            "kept": int(mm.group(5)),
            "drop": int(mm.group(6)),
            "freq_delta_pct": float(mm.group(7)),
        })
    rec["cohorts"] = cohorts
    rec["n_kept_total"] = sum(c["kept"] for c in cohorts)
    rec["n_valid_cohorts"] = len(cohorts)
    # verdict line
    m = re.search(r"cross-cohort gz/ld = ([0-9.]+)\s+\(effect ([+-][0-9.]+)%\)\s+cross-cohort spread ([0-9.]+)%", text)
    if m:
        rec["overall_ratio"] = float(m.group(1))
        rec["effect_pct"] = float(m.group(2))
        rec["cross_cohort_spread_pct"] = float(m.group(3))
    m = re.search(r"FLOOR \(A/A cross-cohort spread\) = ([0-9.]+)%\s+min per-cohort \|effect\| = ([0-9.]+)%\s+sign-consistent=(\w+)", text)
    if m:
        rec["aa_floor_pct"] = float(m.group(1))
        rec["min_cohort_effect_pct"] = float(m.group(2))
        rec["sign_consistent"] = (m.group(3) == "true")
    # verdict class
    if "REPRODUCIBLE:" in text and "BEATS" in text:
        rec["verdict"] = "WIN"
    elif "REPRODUCIBLE:" in text and "LOSES TO" in text:
        rec["verdict"] = "LOSS"
    elif "REPRODUCIBLE TIE" in text:
        rec["verdict"] = "TIE"
    elif "NOT RESOLVABLE" in text:
        rec["verdict"] = "NOT_RESOLVABLE"
    else:
        rec["verdict"] = "UNKNOWN"
    # sign-consistent direction of the effect: <0 means gz(default,change-present) faster
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", nargs="?", help="fulcrum steady log (default stdin)")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--gz-arm", required=True, help="what the gz arm IS (change present)")
    ap.add_argument("--ref-arm", required=True, help="what the ref arm IS (change removed)")
    ap.add_argument("--non-inert", required=True, help="yes|no — Gate-0 path-flip proof")
    ap.add_argument("--ratio", default=None, help="corpus isize/deflate ratio")
    ap.add_argument("--commit", required=True)
    args = ap.parse_args()
    text = open(args.logfile).read() if args.logfile else sys.stdin.read()
    rec = parse(text)
    rec0 = {
        "oracle": args.oracle,
        "corpus": args.corpus,
        "threads": args.threads,
        "gz_arm_change_present": args.gz_arm,
        "ref_arm_change_removed": args.ref_arm,
        "gate0_non_inert_pathflip": args.non_inert,
        "corpus_ratio_isize_over_deflate": (float(args.ratio) if args.ratio else None),
        "binary_commit": args.commit,
        "arch": "aarch64-apple-m1pro",
    }
    rec0.update(rec)
    # interpretation
    if rec.get("verdict") == "WIN":
        rec0["interpretation"] = ("CONFIRMED wall win: default (change present) is faster than "
                                  "the removed arm beyond the A/A floor in every cohort.")
    elif rec.get("verdict") == "LOSS":
        rec0["interpretation"] = ("REGRESSION: default (change present) is SLOWER than the removed arm "
                                  "beyond the A/A floor in every cohort — the change HURTS on this cell.")
    elif rec.get("verdict") == "TIE":
        rec0["interpretation"] = ("TIE: |effect| <= A/A floor; no wall win resolvable above the noise "
                                  "floor on this laptop, stably a tie (honest).")
    elif rec.get("verdict") == "NOT_RESOLVABLE":
        rec0["interpretation"] = "NOT RESOLVABLE on this laptop (freq/thermal-bound); needs a quiet box."
    print(json.dumps(rec0, indent=2))

if __name__ == "__main__":
    main()
