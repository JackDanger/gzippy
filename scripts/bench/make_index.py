#!/usr/bin/env python3
"""Roll all per-cell steady JSONs in a dir into one index.json with the
per-win verdict. Pure stdlib."""
import json, glob, sys, os

d = sys.argv[1]
cells = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    if os.path.basename(f) == "index.json":
        continue
    cells.append(json.load(open(f)))

def cell(oracle, corpus, thr):
    for c in cells:
        if c["oracle"] == oracle and c["corpus"] == corpus and c["threads"] == thr:
            return c
    return None

def brief(c):
    if not c:
        return None
    return {
        "corpus": c["corpus"], "threads": c["threads"],
        "non_inert": c["gate0_non_inert_pathflip"],
        "verdict": c.get("verdict"), "effect_pct": c.get("effect_pct"),
        "aa_floor_pct": c.get("aa_floor_pct"),
        "cross_cohort_spread_pct": c.get("cross_cohort_spread_pct"),
        "n_kept": c.get("n_kept_total"), "sign_consistent": c.get("sign_consistent"),
    }

wins = {
    "pmull": {
        "claim": "aarch64 default CRC = PMULL carry-less fold (vs crc32x 3-way fold3)",
        "oracle": "GZIPPY_CRC_PMULL=0 forces fold3 (change removed); default = PMULL (present)",
        "gate0_non_inert": "CONFIRMED (probe: default->crc_path=PMULL, removed->crc_path=FOLD3)",
        "cells": [brief(cell("pmull", "silesia-gzip.tar.gz", 1)),
                  brief(cell("pmull", "logsbig.gz", 1))],
    },
    "tablebuild": {
        "claim": "aarch64 default litlen table build = SINGLE-symbol (vs x86 TRIPLE-pack)",
        "oracle": "GZIPPY_LITLEN_MULTISYM=triple forces x86 default (removed); default = SINGLE (present)",
        "gate0_non_inert": "CONFIRMED (probe: default->multisym=SINGLE, removed->multisym=TRIPLE)",
        "cells": [brief(cell("tablebuild", "silesia-gzip.tar.gz", 1)),
                  brief(cell("tablebuild", "logsbig.gz", 1))],
    },
    "crossover": {
        "claim": "aarch64 default parallel-crossover selector = OFF (margin 0.0; vs x86/Zen2 ON margin 1.0)",
        "oracle": "GZIPPY_PARALLEL_CROSSOVER_MARGIN=1.0 forces x86/Zen2 selector ON (removed=selector-on); default = OFF (present)",
        "gate0_non_inert": ("T1 INERT (effective_parallel_threads early-returns at num_threads<=1, env never read) "
                            "-> measured at T2 where CONFIRMED non-inert (default->2 parallel, removed->1 serial)"),
        "cells": [brief(cell("crossover", "silesia-gzip.tar.gz", 1)),
                  brief(cell("crossover", "logsbig.gz", 1)),
                  brief(cell("crossover", "silesia-gzip.tar.gz", 2)),
                  brief(cell("crossover", "logsbig.gz", 2))],
    },
}

index = {
    "harness": "scripts/bench/m1_steady_wins.sh",
    "instrument": "fulcrum wall --steady (~/www/fulcrum-mac feat/macos-kpc-measurement)",
    "binary_commit": "b2d1a2db",
    "build": "--no-default-features --features pure-rust-inflate, RUSTFLAGS=-C target-cpu=native",
    "arch": "aarch64-apple-m1pro",
    "method": ("single-binary env-toggle removal-oracle: identical gzippy run under two native "
               "Mach-O wrappers; env flips ONE code path; output byte-identical (sha==gzip -dc on both arms); "
               "freq-pinned warmed multi-cohort paired A/B; verdict REPRODUCIBLE only if effect clears the "
               "gz-vs-gz A/A cross-cohort floor in every cohort with consistent sign."),
    "params": {"cohorts": 5, "pairs": 9, "warmup_secs": 20, "cooldown_secs": 5, "sink": "/dev/null"},
    "verdicts": {
        "pmull": "CONFIRMED wall win (silesia-T1 +1.22%, logsbig-T1 +1.89%; both clear A/A floor every cohort). "
                 "NOT microbench-only.",
        "tablebuild": "CONFIRMED wall win (silesia-T1 +0.89%, logsbig-T1 +4.71%; both clear floor every cohort).",
        "crossover": "CORPUS-DEPENDENT, NOT a clean win. T1 cells INERT (env never consulted). At T2: silesia "
                     "WIN +1.58% (default-OFF parallel beats forced-serial) but logsbig REGRESSION -8.54% "
                     "(default-OFF parallel LOSES to forced-serial). The ratio proxy cannot separate them "
                     "(silesia ratio~3.1, logsbig ratio~3.7) yet their optimal T2 routing is opposite.",
    },
    "wins": wins,
}
print(json.dumps(index, indent=2))
