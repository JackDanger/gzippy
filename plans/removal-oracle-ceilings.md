# Removal-oracle ceilings — contig clean loop STORE vs DECODE (FROZEN, banked)

Date: 2026-06-11. Branch `bench/removal-oracles` @ base `8fa2042f`.
Pre-registration: plans/orchestrator-status.md "LOCALIZATION SPLIT BANKED" — the
9-arm perturbation found DECODE ≈ STORE slow-down slopes (+576 vs +526 ms at
N=50); Rule 3 forbids reading a ceiling off a slope, so these two REMOVAL
oracles size the actual speed-up bounds.

## Instruments (built this session; `src/decompress/parallel/removal_oracle.rs`)

1. **STORE-removal** — `GZIPPY_ORACLE_NOSTORE=1`. The contig clean loop
   (`Block::decode_clean_into_contig`) decodes every symbol normally (bit
   reads, litlen/dist LUT lookups, range checks all identical) but ELIDES the
   output stores at all six sites (fast lit1 / chain / litpack packed-u64 /
   backref `emit_backref_contig`, careful lit / backref). `*pos`/`emitted`
   advance identically, so termination, lengths, CRC bookkeeping, window
   extraction, and the write path all run (over garbage bytes — same CPU
   cost). Output is GARBAGE: loud banner, CRC/ISIZE verification skipped
   (sm_driver), regular-file output REFUSED (fstat S_IFREG → hard error;
   `-c … > /dev/null` is the sanctioned shape). `base − nostore` is an UPPER
   bound on any store/copy optimization (no real implementation can skip
   stores; eliding also removes the copy's dependent loads).

2. **DECODE-removal** — symbol-stream record/replay, `GZIPPY_ORACLE_RECORD=<f>`
   then `GZIPPY_ORACLE_NODECODE=<f>`. Record pass captures, per contig call
   (keyed on deterministic entry state: slice len, entry bit, 8-byte input
   fingerprint, out pos, n_max), the literal bytes + (len,dist) pairs + end
   bit-cursor state. Replay performs the STORES (byte literal writes + the
   production `emit_backref_contig` kernel + sparsity bookkeeping) WITHOUT
   Huffman decode / bit reads / per-block LUT builds, then restores the bit
   cursor so headers, block transitions, CRC, and coordination run genuinely.
   Replay output is BYTE-CORRECT (sha-verified); misses fall back to real
   decode (counted; all runs here were 100% hits). Map load+parse is reported
   out-of-wall (`warm_replay`) and subtracted.
   - **decode_bypass was NOT reused** (decision pre-authorized either way):
     it replays WHOLE CHUNKS as bulk memcpy — removing header parsing, LUT
     builds AND the per-symbol store stream along with decode — so its delta
     is not scope-matched to the NOSTORE arm and the shares-sum sanity check
     would be meaningless. The new instrument removes ONLY the decode half of
     the same region NOSTORE removes the store half of.
   - Known approximation: replay writes multi-literal packets as single-byte
     stores where the litpack arm uses one packed-u64 store (byte count
     identical; the lit1/chain single-byte arm dominates production literal
     traffic, ~2.57 lits/iter). Replay overhead (key lookup, op decode) is
     small and, if anything, UNDER-states the decode ceiling.

Both knobs are OnceLock env reads (slow_knob/contig_prof pattern); OFF-state is
a predictable branch — proven byte-exact by the sha grid below.

## Measurement (FROZEN, banked)

Host freeze: bench-lock acquire TTL 2400, BENCH_LOCK=quiet runnable_avg=1.00,
no_turbo=1/governor=performance readback on guest; released after (RESTORE
VERIFIED no_turbo=0). Binary `/root/bin-oracle-native` (gzippy-native,
sha 68a963b77bfb9503), `GZIPPY_FORCE_PARALLEL_SM=1`, all arms sink /dev/null
(uniform sink; NOSTORE refuses files — correctness of base+nodecode verified in
separate untimed sha runs). Interleaved best-of-9 (base/nostore/nodecode per
rep). Driver: `scripts/bench/removal_oracle_run.sh` →
`scripts/bench/_removal_oracle_arms_guest.sh`.

### silesia T1 (taskset -c 0), medians of 9 (spreads in parens)

| arm                  | wall ms             | ceiling = base − arm |
|----------------------|---------------------|----------------------|
| base (sha-verified)  | 1263.0 (1260–1270)  | —                    |
| NOSTORE              | 1169.0 (1169–1174)  | **94.0 ms (7.4%)**   |
| NODECODE adj¹        |  620.4 (619.9–625)  | **642.6 ms (50.9%)** |

¹ raw 895.0 − warm 274.8 (172 MB capture, 2798 calls, hits 2798/0 every rep).

Sum of ceilings 736.6 ms (58.3%) + remainder 526.4 ms (41.7%) = whole. The
remainder = irreducible loop residue (replayed stores at zero decode ≈ 620 −
non-loop pipeline) + everything outside the contig loop (marker bootstrap,
headers, CRC, window publish, output write, coordination).

### model T8 (taskset 0,2,…,14), medians of 9

| arm                  | wall ms          | ceiling = base − arm |
|----------------------|------------------|----------------------|
| base (sha-verified)  | 662.0 (658–677)  | —                    |
| NOSTORE              | 639.0 (637–647)  | **23.0 ms (3.5%)²**  |
| NODECODE adj³        | 350.4 (347–359)  | **311.6 ms (47.1%)** |

² barely above the 19 ms base spread — treat as ≤~25 ms, direction-only.
³ raw 961.0 − warm 610.4 (397 MB capture, 6224 calls, hits 6224/0 every rep).

## VERDICT (for the asm memo)

**DECODE is the lever — by ~6.8× (T1) to ~13.5× (model T8).** The Huffman
decode + bit-read dependent chain owns ~51% of the T1 silesia wall; the
store/copy half is bounded at 94 ms (7.4%) — the packed-store/copy-shape
family is exhausted as a gap-closer. Against rapidgzip (banked T1 ~926.6 ms,
gzippy base here 1263 ms, gap ~336 ms): removing contig decode compute alone
lands gzippy at 620 ms — far past rg — so the ENTIRE T1 gap (and more) sits
inside decode compute. Any inner-loop asm/rewrite effort should target the
decode chain (litlen LUT dependent load, bit-extract, dist decode), not store
bandwidth. This also RESOLVES the perturbation's DECODE ≈ STORE tie: both
slopes serialized the loop equally (slow-injection adds latency to the chain
either way), but the store work itself is almost fully absorbed by the OoO
core/store buffers — the Rule-3 lesson measured end-to-end.

Reconciliation with the asm-bounded 0.667× engine-W estimate: 0.667× of the
decode-owned 51% ≈ 0.33 × 1263 ≈ +417 ms of headroom — consistent in sign and
order with the 336 ms rg gap; the removal data says the headroom is real and
lives where the asm work would go.

## Honesty ledger / anomalies (verbatim)

- First "frozen" attempt was NOT frozen: zsh failed to word-split `$SSH_JUMP`
  inside `lib_hostlock.sh` (`timeout: exec: ssh -o ConnectTimeout=15 neurotic:
  not found`) → acquire never ran, arms executed at no_turbo=0 (~3× faster
  walls). Caught by the no_turbo readback; re-run under bash with the freeze
  verified. Unfrozen numbers discarded (same ordering: base 421 / nostore 381
  / nodecode adj ~219).
- Guest / was 96% full (1.7 G free — the known full-disk trap). Freed
  regenerable scratch (per-checkout benchmark_data copies + perf .debug cache)
  before building; checked df before/after.
- x86_64 local build initially failed: `.cargo/config.toml` target-cpu=native
  (apple-m1) is invalid for the cross target → `RUSTFLAGS="-C
  target-cpu=x86-64-v2"` per the Rosetta recipe.
- model-T8 NODECODE RAW wall (961 ms) exceeds base (662 ms) because the 610 ms
  single-threaded map load dominates; only the warm-adjusted number is the
  drive wall.
- Record/replay determinism held even at T8 under freeze: 6224/6224 hits
  (chunk clean/marker mix did not shift between record and replay runs).

## Correctness gauntlet

- OFF-state sha grid {silesia, model} × T{1,8}, gzippy-native on guest,
  oracles unset: 4/4 OK.
- NODECODE replay sha: silesia T1 OK, model T8 OK (in-process CRC verification
  also left ON for this arm).
- NOSTORE: exact output LENGTH preserved (11432735/11432735 local corpus),
  bytes garbage as designed, banner present, regular-file refusal verified.
- Local Rosetta suite (x86-64-v2): 946 pass / 1 fail — the environmental
  `test_avx2_detected_on_x86` (Rosetta 2 exposes no AVX2; pre-existing).
- Guest suite: see commit message (run post-measurement on the box).
- fmt clean; clippy default-features: 0 warnings; gzippy-native: 56 warnings ==
  baseline 56 (no new); gzippy-isal x86_64 compiles.


---

## GATE TRIMS (2026-06-11, Opus disproof gate — BINDING on the asm memo)

Verdict SOUND-WITH-CHANGES; the decision (decode chain is the lever, 6.8-13.5x
over store) STANDS. Three sub-claims trimmed:
1. Cross-tool "gap entirely in decode" demoted to the INTERNAL ratio unless
   rapidgzip is re-measured under the same freeze (turbo ~3x confound).
2. "Store side exhausted" -> "CONTIG-CLEAN store/copy exhausted (<=94ms)";
   marker-mode/ring stores + the flip are outside this instrument's scope.
3. Scope: gzippy-NATIVE engine cells at T1 + masked model-T8; real-T8
   consumer-coordinated magnitude is an extrapolation (direction holds);
   does not address the isal-build low-T non-engine residual.
Disjointness proof banked: nostore + nodecode - base = +526ms remainder
(instruments don't double-count); DECODE ceiling is a conservative LOWER
bound (replay adds streaming reads production lacks).
