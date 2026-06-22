# T1 INSTRUCTION-SURPLUS — PRE-REGISTERED FALSIFIER

Branch `t1-instruction-surplus` (off `kernel-converge-A` @036b835d).
Registered BEFORE any close code is written (governing law: no code from an
unconfirmed model; pre-register hypothesis + falsifier).

## STARTING POINT (gated cross-arch LAW, prior cycles)
After the resident-output-pool fault lever (now the gzippy-native T1 production
default, T1ResidentScope), gzippy-NATIVE T1 `prod`/igzip is ~1.17–1.30 across
corpora. The residual is an INSTRUCTION-COUNT surplus: native runs **+13–20%
MORE instructions** than igzip (monorepo 0.449 B vs igzip 0.374 B; silesia 2.94 B
vs 2.59 B; nasa 0.939 B vs 0.811 B). Native≈ISA-L KERNEL parity is prior LAW, so
the surplus is in the DRIVER / SHARED per-byte+per-chunk path, NOT a native-kernel
deficit.

Pre-located bucket (PROD-PATH-LOCATE): CRC second-touch — a SEPARATE per-byte pass
re-reads the decoded output to fold CRC (`chunk_decode.rs` native tail :1721-1729,
ISA-L tail :612-620) vs igzip's INLINE single-pass `update_checksum`. Pre-bounded
2–12% of wall (removal oracle `GZIPPY_ORACLE_CRC_OFF`).

## HYPOTHESIS (unvalidated until measured)
The +13–20% instruction surplus over igzip is decomposable by removal-oracle into
per-byte buckets {CRC second-touch, output copy, per-symbol emission} and per-chunk
buckets {window handoff, boundary record, chunk lifecycle/alloc, table build}.
Folding the CRC into the single decode pass (touch output ONCE, like igzip) removes
the CRC second-touch's instructions and memory traffic.

## METRICS (the only currency)
- Primary wall: gzippy-NATIVE `prod` (T1, `decompress_parallel(&[u8],…,1)`) / igzip
  (ISA-L monolith WITH CRC), `/dev/null` both arms, decode-only timed, interleaved
  best-of-N≥7 (N≥11 on AMD frozen, N=15 Intel taskset), Δ vs inter-run spread.
- Mechanism: AMD bare-metal `perf stat` retired-instructions, native vs igzip, per
  bucket via removal oracle (the difference-of-differences for CRC).
- Cells: {silesia, nasa, monorepo, squishy} × {Intel neurotic, AMD solvency} = 8.

## PRE-REGISTERED OUTCOMES
- **CONFIRMED** iff: gzippy-native `prod`/igzip drops to **≤1.10 on all 8 cells**
  (Intel+AMD), byte-exact (sha==zcat all arms), T4/T8 no regression vs the OLD
  binary (and not worse vs rapidgzip), AND the native instruction count drops toward
  igzip's (mechanism confirmed — the removed bucket's instructions actually leave).
- **PARTIAL**: a real gated drop (Δ≫spread, both arches, mechanism-confirmed) that
  does NOT reach 1.10 on all cells — report the new ratios, the remaining located
  residual, and the next lever. Do NOT narrate as "closed".
- **FALSIFIED-per-lever**: a candidate that does not cut instructions beyond noise,
  or cuts instructions but does not move the wall beyond spread, or regresses T>1 —
  report, drop it, keep the payers.

## GATES (every run)
- Gate-0: sha==zcat all arms; A/A |prod−prod2| ≪ Δ; `/dev/null` both arms; comparator
  self-test ≈1.0; each removal oracle proven NON-INERT (banner fired / counter > 0).
- Gate-1: interleaved best-of-N≥7; report Δ AND spread; Δ<spread ⇒ TIE.
- Gate-3: Intel AND AMD; T1 AND T4/T8.
- Gate-4: GZIPPY_DEBUG routing == thin-T1 / ParallelSM; native kernel build
  (`--features pure-rust-inflate` / `gzippy-native`); feature fingerprint; BINARY ON
  THE BOX VERIFIED to contain the change (grep the new symbol; confirm built sha)
  before any number is banked.
- Correctness: every close commit carries a flate2 + libdeflate differential over
  multiple chunk sizes + multi-member resume, in the SAME commit.

## SCOPE / NON-GOALS
- T1 ship-target is gzippy-NATIVE (no FFI, no ISA-L). KEEP gz's superior pieces (the
  resident pool, the pclmulqdq crc32fast algorithm — we remove the SECOND TOUCH, not
  the CRC algorithm). T>1 must not regress (changes T1-gated or proven safe at all T).
- Infinite funding / no phases: close EVERY located bucket this cycle, not the cheap
  ones; no ROI/cost language.
