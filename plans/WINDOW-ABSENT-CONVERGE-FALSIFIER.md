# WINDOW-ABSENT-CONVERGE — Pre-registered Falsifier

**Front:** AMD/Zen2 T>=2-vs-rapidgzip. Branch `kernel-converge-A` @ 8cad4f6b.
**Subject:** gz's window-absent (marker) decode — `marker_inflate.rs`
`read_internal_compressed_specialized::<true>` → `decode_marker_fast_loop` +
`decode_careful_tail::<true>`. Production caller: `chunk_decode.rs:2508`
`block.read(...)` (window-absent first-chunk-per-thread path).

## STALE-PREMISE RECONCILIATION (must resolve at Gate-2 before any lever)
The brief's GATED STARTING POINT says the marker loop "uses a slow NON-INLINED
`HuffmanCodingShortBitsCached::decode` interleaved with marker-emission." Reading
current HEAD, `decode_marker_fast_loop`:
- litlen decode = `self.asm.lut_litlen.decode` (fast `LutLitLenCode`, same as clean),
- dist decode = `DistTable` single-lookup (fast; `dist_hc` only in kill-switch arm).
So the slow `HuffmanCodingShortBitsCached::decode` (`dist_hc`) survives only in the
CAREFUL TAIL dist (`marker_inflate.rs:~2825`) + the fast-loop kill-switch arm — NOT in
the hot marker fast loop. Lever 1 (`#[inline(always)]`, marker-kernel cycle) already
landed on this branch. **Therefore the brief's mechanism is a HYPOTHESIS contradicted
by the source; the REAL current mechanism of the 11.7 cyc/B marker bucket must be
re-established by objdump+perf on the CURRENT HEAD binary before any convergence lever.**
The window-PRESENT clean path (4.7 cyc/B) uses a SEPARATE hand-tuned asm kernel
(`decode_clean_into_contig` → `asm_kernel::run_contig`, `chunk_decode.rs:1703`); the
marker path is a Rust LUT loop. The clean-vs-marker gap is therefore "asm kernel vs
Rust loop + u16 ring + distance_marker bookkeeping + marker-propagating backref",
NOT "fast vs slow huffman decode call".

## PRE-REGISTERED THRESHOLDS (per lever)
A lever = any byte-exact change intended to drop the marker decode cyc/B toward the
clean/rg ~4.7-7 band.

- **CONFIRMED** iff ALL hold:
  1. byte-exact: sha==zcat on silesia/monorepo/nasa T1+T4 both arches + flate2/
     libdeflate silesia differential @ multiple chunk sizes in the SAME commit;
  2. marker decode cyc/B (mfast_prof: MFAST_CYC/MFAST_EVENTS + careful) drops by
     > A/A spread, with objdump/perf NON-INERT proof the codegen actually changed;
  3. AMD-T4 gz/rg wall ratio drops toward parity (report actual Δ; prize bounded
     ~3-10% per locate — do not require 1.01, require a real Δ > A/A spread);
  4. NO regression (TIE-or-better) on Intel T>=2, AMD/Intel T1-native, T8, and the
     clean (window-present) path.
- **PARTIAL:** real gated cyc and/or wall drop that does not reach parity — report
  new marker cyc/B + AMD-T4 ratio + the remaining bucket.
- **FALSIFIED-per-lever:** cyc drop within A/A noise OR no wall move OR any regression
  elsewhere → revert that lever; record FALSIFY with mechanism + scope; try next.

## MEASUREMENT GATES (CLAUDE.md PROTOCOL)
- Gate-0: all arms sha==zcat; A/A << Δ; /dev/null both arms; objdump/perf proves the
  codegen change is non-inert.
- Gate-1: interleaved best-of-N>=7 (AMD), best-of-15 (Intel); Δ vs spread; label TIEs.
- Gate-2: cyc/B freq-pinned = mechanism; interleaved AB wall = verdict.
- Gate-3: AMD primary + Intel; T2/T4/T8 + T1-native no-regress.
- Gate-4: GZIPPY_DEBUG=1 → path=ParallelSM; confirm box binary built the tested sha.

## BOX HYGIENE
AMD solvency `root@10.0.2.240`: freeze gov=performance/boost=0 (NOT no_turbo),
idempotent + guaranteed restore (gov=ondemand, boost=1) + bounded watchdog; never
leave frozen; taskset cores 8,10,12,14 away from roaming `llama`; flag load-corrupted
runs (A/A spread blows up). Do NOT touch bench-lock.sh. Verify box-state at exit.
