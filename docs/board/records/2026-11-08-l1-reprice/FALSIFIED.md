# 2026-11-08 — L1 exception re-adjudicated under BYTE-FREEDOM — flip FALSIFIED; legacy arm stands on 024.bin

Probe branch `probe/l1-reprice` `c0a49c6b` (instrument `examples/l1_time.rs`; binaries stamped; corpus recipe-verified 16,085,866 B; ldc -1 built fresh).

**Wall (T1, in-process interleaved, hypothesis-tier):** port (`LdxCompressor::new(1)`, (0, 32, 0)) = **0.748× of legacy** on the aggregate (48.99 vs 65.50 ms), per-member 0.58-0.79×. The wall bar passes decisively.

**Size (the binding bar — exact bytes):**
- port == `libdeflate-gzip -1` byte-exact on all 31 citable members + aggregate (4,546,439).
- vs min(gzip -n -1, pigz -1 -n, ldc -1): **30/31 PASS** — the historical "port loses to pigz -1 on text" blocker is **gone** on today's code (text-1MB: port 441,787 ≤ pigz 479,561 / gzip 479,469).
- **`024.bin` FAILS**: port 190,920 vs pigz -1 188,977 (**+1,028‰, ~10× the tie band**); also > gzip -1 190,594. The legacy arm's per-member output (`prod_gz`: text 432,408 / 024 185,788; aggregate 4,424,116) is still the strongest L1 everywhere.

**T>1 leg:** legacy `-1 -p4` runs the pipelined chunked arm (4,427,406 B, 36 ms); a port-Fastest chunked path would need new wiring (not measured — the T1 falsifier already decides).

`fast_l1_ratio_multi_corpus` (src/tests/deflate_encoder_matches.rs:960): shipped legacy arm passes; the flip arm was barred before the run by the failing size bar.

**VERDICT: the L1 exception survives re-measurement — nearly closeable (30/31 members pass; wall is 25-35% our-side), but `024.bin`-class (photo-like, 1.6 MB) needs port-L1 to shed ≥1,943 B vs its own libdeflate -1 algorithm before a flip is adjudicable.** The charter-era "we beat pigz -1 on text" clause is stale wording — the surviving reason is the 024 member cell. CLARIFICATION for the low-band ledger: at matched -1 L1 the port is 11.9% SMALLER than igzip CLI (wave-4) but the comparison "port ≈ igzip config" is not meaningful — different cost models; cite member sizes, not derivation.
