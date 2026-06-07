# ENGINE ISOLATION BENCH — PRE-REGISTERED falsifier + verdict criteria

Pre-registered BEFORE building/running the technique variants (CLAUDE.md rule 5,
charter §"PRE-REGISTER the plateau falsifier"). HEAD 249f25b5, branch reimplement-isa-l.
Written 2026-06-07 by the engine-bench leader. Authority: plans/engine-bench-authorization.md,
plans/tier1-design-v2.md §2.3.

## What the bench measures
A STANDALONE microbench (NOT the production pipeline) decodes ONE known-window CLEAN
silesia deflate chunk on the locked guest x86_64, three ways, byte-exact each:
- (i)  gzippy's CURRENT clean inner loop — `marker_inflate::Block` u16-ring path
       (`set_initial_window` + `read()` driving `read_internal_compressed_canonical`,
       the production clean decode after FlipToClean). This is the SCALAR u16 baseline.
- (ii) the proposed engine technique(s) — START with E1 (u8-direct clean write), add
       E2/E3/E4 as tractable, inline-ASM where Rust codegen lags.
- (iii) ISA-L `isal_inflate` itself (via `backends::isal_decompress::decompress_deflate_
       from_bit` with the 32 KiB dict) — the upper-bound ORACLE. C lib called IN THE
       BENCH only; FFI stays OFF the native decode graph.

The chunk + window come from a `GZSEEDW2` seed file captured by a real p=1 decode
(`GZIPPY_SEED_WINDOWS_CAPTURE`), which records true (start_bit → 32 KiB predecessor
window) boundaries. The bench picks one such (start_bit, window), feeds all three
variants the identical (encoded-bytes-from-start_bit, 32 KiB window, byte-cap N), and
sha-compares the decoded output across all three.

Metric: clean decode RATE (MB/s of decoded output, best-of-N, on the locked guest under
the build lock). Report each variant's rate and the ratios (ii)/(i), (iii)/(i), (ii)/(iii).

## SELF-TEST (rule 4 — RUN AND PASS BEFORE trusting any (ii) number)
Variant (iii) ISA-L must read **~2× variant (i) scalar** on this single-thread clean
silesia chunk — the GUEST-MEASURED ratio, NOT the discredited 337/720 absolutes. The
prior campaign measured gzippy clean 92.7 ms/chunk vs rapidgzip(igzip) 39 ms/chunk = 2.38×
at the wall and ~2.1× single-thread (vendor bench: ISA-L is 2.1× the best PURE decoder s-t).
- PASS band: (iii)/(i) in roughly [1.7×, 2.6×]. (A wide band because absolutes are
  illegitimate guest targets; what must reproduce is "ISA-L is ~2× our scalar clean.")
- If (iii)/(i) reads ~1.0× (ISA-L no faster than our scalar) OR ≫3× (implausible),
  the bench is BROKEN — STOP, diagnose (is the chunk really clean? is variant (i) really
  the marker_inflate clean loop and not falling into the markered arm? is ISA-L getting
  the dict? are both decoding the same N bytes?). FIX before any (ii) number counts.
- Byte-exactness gate: all three variants' decoded output sha MUST match each other on
  every iteration. A rate win with wrong bytes is VOID (CLAUDE.md rule 6).

## PRE-REGISTERED PLATEAU FALSIFIER (the go/no-go for the multi-week build)
State NOW, before running (ii):

**FALSIFIER:** If variant (ii) — after E1 and whatever of E2/E3/E4 are tractable with
inline-ASM — plateaus NEAR the pure-decoder ceiling (i.e. (ii) lands at roughly variant
(i)'s class, materially below ISA-L, specifically **(ii)/(iii) ≤ ~0.65** meaning still
≥~1.5× slower than ISA-L) AND the residual gap to igzip-class clean rate EXCEEDS the
inter-run spread, THEN the engine front is **NOT PROVEN**:
  → report the achievable floor,
  → DO NOT integrate,
  → DO NOT start the multi-week production engine build,
  → re-confront placement/consumer-pace as the remaining lever,
  → surface to supervisor (this is a USER/supervisor-level finding: the 1.0× bar may not
    be reachable in pure-Rust without FFI).

**PASS:** variant (ii) reaches a clean rate that, fed through tier1-design-v2 §3, projects
a T8 wall ≤ rapidgzip + spread. Concretely: (ii) must reach a clean per-chunk rate
≤ ~39–45 ms-equivalent (i.e. (ii)/(iii) ≥ ~0.85, within ~15% of ISA-L) so the §3 coupled
model lands the decode-bound wall low enough that the wall re-binds on the shared pipeline
floor (~0.54s) rather than on decode. Anything between (the "NARROW MISS" band,
0.65 < (ii)/(iii) < 0.85) is an INCONCLUSIVE result → report the achievable floor and the
projected wall band, route to the disproof advisor, STOP for supervisor — do NOT
unilaterally start the build.

## DISPROOF discipline
- Δ < inter-run spread ⇒ TIE, full stop (no rate claim from a within-spread delta).
- best-of-N≥7 interleaved; report min/median/σ per variant; use MEDIAN as the robust stat
  (the BMI2 A/B showed best-of-N can crown a fast outlier; median is the verdict).
- Both/all variants byte-exact every iteration or the run is void.
- Frequency: run under the host freq-lock (no_turbo pinned); single-thread (taskset 1 core)
  — this is a per-chunk compute microbench, NOT a parallel wall.
- The bench bounds the COMPUTE ceiling only. It does NOT measure the T8 wall (the §3 model
  projects that). The riders (same-sink floor) capture the non-decode terms.

## RIDERS (run alongside — see plans/engine-bench-authorization.md §RIDERS)
1. Re-read `nearest_le_start` at cap=256 from the step-0 stall-residency probe.
2. Same-sink production-output floor: gzippy's floor with a REAL file sink (not /dev/null)
   AND confirm rapidgzip's 0.54s is the same-sink comparison.

DROP (not load-bearing): "placement dissolves once the engine is faster."
</content>
</invoke>
