# SURPASS PLAN — beat libdeflate+igzip (T1), pigz (all T), tie/beat rapidgzip (all T)

**Author:** technical-strategy advisor (Opus). **Date:** 2026-06-21. **Branch base:**
`kernel-converge-A @ 462b0549` (post-pinning-delete HEAD).

**Governing-policy stamp:** every recommendation below is a **HYPOTHESIS
(unvalidated)** until its paired Fulcrum/`standing.sh`/`kernel_gate.sh` measurement
runs, passes Gate-0 self-validation, and meets the Gate-1 significance bar on BOTH
arches. I tier each as `GATED` (already has a passing gated measurement supporting
the direction), `BUILD-AND-MEASURE` (designed, needs its first gate), or `NEAR-LIMIT`
(oracle-bounded small ceiling — honest about poor ROI). I cite gz `file:line` and
vendor blueprint `file:line` throughout. No claim here is a finding; the firing
A/B is the finding.

---

## 0. WHERE WE ACTUALLY ARE (the gated ground truth, from MATRIX-COMPLETE-2026-06-21)

| Target | Arch | Current gated status | Tier |
|---|---|---|---|
| vs **rapidgzip** T≥2, COMPRESSIBLE | Intel x86 | AT-PARITY→WINNING (silesia-T4 closed to 1.032 TIE by pin-delete; nasa/monorepo/bignasa win) | GATED |
| vs **rapidgzip** T≥2, STORED | Intel x86 | **LOSS** — pure_stored +60-79%, storedheavy/mix +14-33% (StoredParallel path) | GATED (the one real parallel loss) |
| vs **igzip** T1 single-core | Intel x86 | **LOSS ~24%** silesia (nasa is the exception: gz WINS −33%) | GATED + oracle-bounded |
| vs **libdeflate** T1 single-core | aarch64 (mac) | **LOSS ~21-24%** (gz/ld 1.21-1.24) | GATED + localized |
| vs **pigz** any T | any | **NOT MEASURED** — pigz absent as comparator | unmeasured |

**The whole surpass program is four fronts:** (1) the StoredParallel T≥2 loss vs rg
(localized, designable, byte-exact-tractable — DO FIRST); (2) pigz comparator (cheap,
likely already-won — measure to bank); (3) T1 vs igzip x86 (deeply studied,
near-limit, the scaffold is the one unexhausted surface); (4) T1 vs libdeflate aarch64
(localized to the clean decode CORE, shared with x86 → highest cross-arch leverage).

---

## TARGET 1 — FIX the StoredParallel T≥2 loss vs rapidgzip (highest goal-impact × tractability)

### The mechanism (localized, file:line)
`stored_split.rs:318` allocates a **monolithic zero-init output Vec** `vec![0u8; total]`
(100 MB for pure_stored) → first-touch page-fault storm; `:319-320` `fill_and_crc`
copies every stored run **into** that buffer; `:322-324`/`:375-396` `verify_and_write`
then `write_all`s the whole buffer **out** → a second full pass. That's **2× copy +
a 100 MB fault-in** where rapidgzip streams chunk-by-chunk through a small reused
buffer (`DecodedData.hpp` `vector<VectorView>` + `GzipChunkFetcher` writeAll loop).
This is exactly the +46-73% pure_stored T≥2 gap.

### KEY byte-exact observation that makes the fix clean
For a **pure-stored** stream the output is *verbatim input bytes*: `walk_stored_chain`
(`stored_split.rs:257-264`) records each `StoredRun { src_off, out_off, len }` where
`src_off` points directly into `gzip_data`. So `output[out_off..][..len] ==
gzip_data[src_off..][..len]` byte-for-byte — **no decode, no intermediate buffer is
logically required.**

### RECOMMENDED route (HYPOTHESIS — pair with the gate below)
Replace the `WalkEnd::Final` arm (`stored_split.rs:300-325`) with **chunked streaming
direct from input**, no monolithic buffer:
1. Partition the `runs` list into N ≈ `num_threads` contiguous byte-ranges of the
   output. Each worker CRC32s its disjoint range of input slices (hardware CLMUL /
   aarch64 HW CRC), producing a partial CRC + length.
2. Combine partials with `crc32_combine` (linear in N) → final CRC; compare to
   `expected_crc` BEFORE any write *for the pure-stored case* (we can: CRC is the only
   CPU work and the input is fully present, so the verify-before-write contract is
   **preserved** — no partial-output regression vs today).
3. Write the input slices to the sink via **`writev`/`write_vectored`** of the run
   slices (zero intermediate copy), or sequential `write_all` per run. Either way:
   no 100 MB zeroed Vec, no fault-in, ONE copy (the unavoidable sink write).

This is faithful to rg's chunked-reused-buffer streaming (`DecodedData.hpp:325-388`
narrow + the `GzipChunkFetcher` write loop) and to the data-plane viewlist port
already KEPT at `300e772b`.

**Byte-exact concern (call out explicitly):** the current path verifies CRC over the
*assembled* buffer before writing (`stored_split.rs:375-396`). The streaming variant
must compute the **same** CRC over the **same** byte order (out_off ascending) via
`crc32_combine` — add a unit test asserting `combine(parallel partials) ==
serial crc32(whole)` on a fixed fixture. Because we still CRC the whole logical output
before `writev`, the *no-partial-output-on-corruption* property is KEPT (do NOT stream
writes before the CRC check unless you accept ParallelSM's partial-output contract —
recommend keeping verify-before-write for stored since input is fully buffered).

### The gate that proves it (BUILD-AND-MEASURE)
- `scripts/bench/standing/standing.sh --box neurotic` on `pure_stored`, `storedheavy`,
  `storedmix` × T1/T2/T4/T8, interleaved N≥13, /dev/null all arms, sha==zcat Gate-0,
  A/A self-test ≤5%. Target: gz/rg ≤ 1.0 at T≥2 (currently 1.46-1.79).
- Gate-0 non-inert: a `STORED_STREAM_RUNS` counter fires; `GZIPPY_DEBUG=1` →
  `path=StoredParallel`; `df`/RSS shows the 100 MB alloc is gone (RSS drop is the
  mechanism witness).
- Cross-arch: replicate on mac (`standing_mac.sh`) and AMD when solvency returns.

**Tier: BUILD-AND-MEASURE, high confidence.** The mechanism is measured (page-fault +
double-copy), the fix is structural and byte-exact-tractable, and stored is
bandwidth-bound so removing 2 of 3 memory passes should move the wall hard.
**Realistic capturable: most of the +46-73% — likely flips pure_stored to TIE-or-win.**

---

## TARGET 2 — pigz at all T (cheap; confirm the likely-already-won verdict)

### The characterization (CONFIRM, do not assume)
pigz `-d` is **single-threaded decode** (zlib inflate + I/O helper threads only —
standard gzip can't be parallel-decoded without rapidgzip-style speculation, which
pigz does not implement). So:
- **T>1:** gzippy's existing ParallelSM/StoredParallel pipeline should beat pigz
  trivially (gz scales, pigz stays serial). No routing change needed.
- **T1:** gzippy must beat **zlib** (pigz's kernel). zlib is slower than igzip and
  libdeflate, so this is a *strictly easier* bar than Targets 3/4 — if gz is within
  24% of igzip it is almost certainly faster than zlib.

### The gate (BUILD-AND-MEASURE, cheap)
Add `pigz` as a comparator in `standing.sh`/`standing_mac.sh`:
- Gate-0: `pigz` present on the box + self-test (`pigz -dc vs zcat` byte-exact;
  pigz-vs-pigz A/A ≤ spread).
- Same `/dev/null` sink, interleaved N≥13, all T × all archive types, both boxes.
- Expected: gz wins every T>1 cell; gz wins T1 on compressible (gz beats zlib). The
  one to watch is T1 on near-incompressible `model`/stored where gz pipeline overhead
  could matter — measure it; if gz loses T1-model to pigz, that's a real find, route
  it like stored (StoredParallel already covers stored; model is the corner case).

**Tier: BUILD-AND-MEASURE, very high confidence it's already won at T>1.** No
implementation work expected beyond adding the comparator. **Capturable: banks an
existing win; only risk is a T1-corner cell.**

---

## TARGET 3 — T1 single-core vs igzip on x86 (the deep, near-limit gap)

### What is already gated (do NOT re-derive — this is the most-studied front)
- gz T1 = +24% silesia / +30% nasa vs igzip after 5 KEPT byte-exact wins
  (dist-preload, flag-bit, ratio-reserve, T1-depth-1, chunk-size-T1) + NIGHT36/40.
- **NIGHT35 Gate-2 injector (STRONG):** the `run_contig` hot loop IS on the production
  T1 wall — instruction-count reduction toward igzip `_04` DOES pay. But:
- **NIGHT18/23 removal-oracle (STRONG):** shedding the resumable/marker glue closes
  only **16-27% of the instr gap, ~3-6% of WALL**. A dedicated stateless T1 kernel is
  NOT a path to parity.
- **EMISSION-APPORTION-2026-06-21 (HYPOTHESIS, static objdump):** the hot literal
  emission loop is ALREADY converged with igzip `_04` (gz literal group **−1 instr**;
  packing/triple-LUT identical; MOVDQU copy at parity). Of the +3.22 instr/B STEP-0
  gap: ~+1.8 is the diffuse length-tail (dominated by NON-portable costs — the D-1
  anchor ~0.6 [NIGHT32 SLACK], parallel window/marker validation ~0.59 [no `_04`
  counterpart], resumable contract ~0.23); **~+1.4 instr/B (~44%) lives in the
  `decode_clean_into_contig` SCAFFOLD OUTSIDE `run_contig`** (`marker_inflate.rs:3572`
  calls `run_contig`; the scaffold wrapper is 5959 insns vs `_04`'s 331 monolithic).

### RECOMMENDED route ordering (honest tiers)

**(3a) APPORTION THE SCAFFOLD FIRST — the one unexhausted surface. BUILD-AND-MEASURE.**
Before any port: objdump-apportion `decode_clean_into_contig`'s *executed-per-byte*
body (or perf-annotate the clean path on a frozen box), splitting the ~+1.4 instr/B
scaffold into per-byte vs per-block. We have only its *static* size, not its dynamic
share. The blueprint is `_04`'s monolithic single-pass shape
(`igzip_decode_block_stateless.asm` `decode_huffman_code_block_stateless_04`,
331 insns, no fast-loop wrapper / table-ensure / per-call prologue). **This is the
next-attack surface and it is less heroic than deeper asm** — but its capturable %
is UNKNOWN until apportioned. Gate: `kernel_gate.sh` T1 production-wall A/B on a
frozen box, paired N≥15, instr/B (the only stable isolated number on the loaded LXC)
+ wall on frozen.

**(3b) The in-loop pre-copy refill cadence (element F on the backref arm) ~+0.23
instr/B.** The one genuinely `_04`-portable in-loop sub-bucket
(EMISSION-APPORTION §RANKED). Small, byte-exact, cheap. Port `_04`'s prefix-amortized
refill (`igzip_decode_block_stateless.asm:527-547`) to gz's `58:`/`51:` pre-copy
refill. Gate: same as 3a. Tier: BUILD-AND-MEASURE, small.

**(3c) A STATELESS clean-T1 kernel (sheds resumable/reclass). NEAR-LIMIT — be honest.**
The chunked decoder doesn't need the resumable contract at T1 single-shot, so in
principle a stateless kernel could shed the ~0.23 (resumable) + part of the validation.
But NIGHT18/23 already bounded this: **~3-6% of wall, HIGH byte-exact risk,
multi-session.** It is NOT a path to igzip parity. Recommend: do NOT lead with this;
it only becomes attractive if 3a's apportionment shows the scaffold *is* the resumable
glue (then 3a and 3c merge). Tier: NEAR-LIMIT.

### Routing-by-archive-type at T1 (the user authorized archive-type-detecting gates)
**HYPOTHESIS worth a measurement, not a build yet:** nasa already WINS T1 vs igzip
(−33%) while silesia LOSES (+24%) — the gap is corpus-dependent (tracks the clean
fraction / block structure). A per-archive-type T1 route is only justified if a
*different existing kernel/chunk-size wins on a detectable class*. We already have one
such gate that paid: `T1_TARGET_COMPRESSED_CHUNK_BYTES=1MiB` (single_member.rs:62),
nasa −5.32% cyc/B. **Before adding more gates, measure** whether engine-A-flat
(asm-off) beats the x86 asm `run_contig` at T1 per archive type — the cross-ISA-LAW
data (MEMORY: D3 RETIRE-ASM-LEANING) says engine A is +4.3% silesia / +2.6% nasa
*better on instructions* than the asm kernel asm-off, but the asm's value is IPC/cycles
which the unfrozen LXC can't resolve. Gate: frozen-box cyc/wall A/B engine-A vs asm
`run_contig` per corpus at T1. If engine A wins x86 T1 on cycles too → route x86 T1 to
engine A (and it becomes the sole portable kernel — big simplification, Target 5).

**Honest verdict for Target 3:** the literal igzip-T1-parity goal is **near the asm
limit** (~3-6% capturable by the bounded routes, high risk). The user FUNDED heroic
rewrites and forbade accept-until-parity, so the mandate stands — but the *highest-VoI
next step is 3a (apportion the scaffold)*, because it's the only surface with an
unknown (possibly larger) capturable % and it's less heroic than asm. **This
accept-vs-fund tension is an R3 for the user** (SOBER-STATE §4B): the oracle says
poor ROI; the goal says parity-or-bust. Surface it; don't resolve it in this doc.

---

## TARGET 4 — T1 single-core vs libdeflate on aarch64 (highest cross-arch leverage)

### What is already gated/localized (STRONG-attribution, deterministic instr on mac)
- gz aarch64 T1 = 19.37 instr/B / 5.75 cyc/B vs libdeflate 7.66 / 3.33 = **2.53× instr,
  1.73× cyc** (silesia, N=13, instr floor 0.008%). The biggest single gap in the
  project.
- **Localized (MEMORY, GZIPPY_THIN_T1_ORACLE removal + GZIPPY_TBUILD_MULT slope):**
  pipeline scaffold ~0% (marker path never fires at -p1), table-build ~5.4%,
  **CLEAN DECODE CORE 94.7% (11.09 instr/B excess)** = the Huffman inner loop +
  bit reader + CRC. NOT scaffold, NOT table-build.
- **Copy is already NEON** (`simd_copy.rs:108-128` neon mod, `consume_first_decode.rs`
  `copy_match_fast` aarch64 arms at :437/:507). So the lever is NOT a missing wide
  copy — **correct the prompt's "no NEON kernel" framing**: the NEON copy exists; the
  gap is the *Huffman decode core + bit reader + (SW) CRC*.

### RECOMMENDED route (BUILD-AND-MEASURE, the FULL-DESIGN target from MEMORY)
The production aarch64 clean kernel is engine A (`decode_huffman_libdeflate_style`,
`consume_first_decode.rs:632`, with flat `LitLenTable`/`DistTable`
`libdeflate_entry.rs:333/361`). Converge it toward libdeflate-aarch64
(`vendor/libdeflate/lib/decompress_template.h` — the engine-A blueprint) on the three
localized sub-components:
1. **aarch64 hardware CRC32** (`__crc32d`/`crc32x` via `std::arch::aarch64`) replacing
   the SW table CRC on the clean path. libdeflate uses HW CRC on arm64. Likely the
   cleanest single win (CRC is in the 94.7% core bucket). Gate: localize_mac.sh
   instr/B + cyc/B isolation of CRC sub-component (a CRC-off oracle vs HW-CRC arm).
2. **Huffman multi-symbol / table preload** in engine A's fastloop
   (`decode_huffman_fastloop_bounded`, `consume_first_decode.rs:1192`) mirroring
   `decompress_template.h`'s lookahead + single-refill discipline. Engine A already
   does up-to-8-literal lookahead + FASTLOOP_MARGIN=320 — measure whether
   libdeflate's refill cadence / preload-ahead-of-dependent-load is faithfully ported;
   port the gaps.
3. **Bit-reader width / refill** — confirm the u64 refill matches libdeflate's.

**Cross-arch leverage:** engine A is the SHARED pure-Rust clean kernel (cross-ISA-LAW
better than engine B, MEMORY D2). **Fixing it advances Intel asm-off AND aarch64.** So
Target 4's work is the same kernel that Target 3's 3a/3c touch — they converge on ONE
deliverable (the sole portable flat clean kernel, Target 5).

### The gate (BUILD-AND-MEASURE)
- `standing_mac.sh` + `localize_mac.sh`: `-p1`, /dev/null, interleaved best-of-N≥9,
  instr-floor determinism ≤0.25%, sha gz==libdeflate==gzip, path=ParallelSM. Target:
  gz/ld ≤ 1.0 at T1 (currently 1.21-1.24). Each sub-component (CRC, Huffman, bitreader)
  gated independently with a removal/slope oracle so the win is attributed causally,
  not by self-time.
- Cross-ISA replicate on Intel asm-OFF (the strongest LAW arm available without AMD).

**Tier: BUILD-AND-MEASURE.** The 2.53× is real and localized to a SHARED kernel.
**Realistic capturable: this is the largest single gap and the most leverage — HW CRC
+ refint/lookahead convergence plausibly closes a large fraction; the 94.7%-core
localization says there is no scaffold tax hiding the gain.** Honest: pure-Rust scalar
Huffman may not fully reach libdeflate's hand-NEON, but the FUNDED mandate is the full
NEON-specialized clean kernel — design it once, share it x86/aarch64.

---

## TARGET 5 — KEEP tying/beating rapidgzip + the sole-path end-state

### Maintain (GATED, do not regress)
- The **unpinned decode pool** (`chunk_fetcher.rs:766-778`, empty `ThreadPinning`) is
  the just-landed pin-delete that closed silesia-T4 to 1.032 TIE. Faithful to rg
  `BlockFetcher.hpp:185`. Any future affinity change must re-run `standing.sh` N=13
  frozen and prove no silesia-T4 regression.
- Compressible T≥2 is at-parity-to-winning; the only rg loss is StoredParallel
  (Target 1) + the corner cases (model, weights — incompressible/dense corners, not
  verdict-drivers).

### Sole-path collapse (BUILD-AND-MEASURE, mechanical/byte-exact — the goal criterion)
Per the master simplification plan (MEMORY, agent a0a03623) the 8 `DecodePath`
variants (`mod.rs:66-104`) collapse to ParallelSM + thin-T1 + framing shims:
- **DELETE-NOW (dead):** `bgzf::decompress_single_member_parallel` (ZERO callers);
  legacy `not(parallel_sm)` arms (StreamingSingle/LibdeflateSingle, `mod.rs:516-526`).
- **ABSORB then delete (parity-gated):** IsalSingleShot (`mod.rs:536-550`) ← thin-T1;
  MultiMemberSeq/Par + BGZF/"GZ" (libdeflate FFI) ← member/block framing shims over the
  shared pure-Rust kernel; StoredParallel ← keep as a thin stored-framing shim (after
  Target 1 it's a streaming-writev shim, not a separate buffer engine).
- **Hardest absorption = the aarch64 kernel parity (Target 4)** — must reach parity
  BEFORE deleting cross-arch paths, else aarch64 regresses.
- **MUST NOT DELETE:** the fuzz/differential oracles (flate2/libdeflater/zopfli behind
  dev features) — the C-FFI-as-test-oracle carve-out.
- Sequencing: deletion is LAST, gated on the sole path meeting-or-beating EACH removed
  path on perf AND correctness, both arches.

**Tier: BUILD-AND-MEASURE (byte-exact).** Doesn't move the wall but is the
done-criterion and lowers every future measurement's cost (SOBER-STATE §4D, user
directive feedback_stop_circling_converge_and_clean).

---

## ROUTING SYNTHESIS — the target route table (archive-type × threadcount × arch)

| archive type | T | x86_64 (Intel/AMD) | aarch64 | kernel | notes |
|---|---|---|---|---|---|
| compressed single-member | 1 | ParallelSM inline, 1MiB chunk | ParallelSM inline, 1MiB | x86: asm `run_contig` (or engine A if 3-route gate wins cyc); aarch64: engine A (HW-CRC+NEON, Target 4) | `mod.rs:222`, `single_member.rs:62` |
| compressed single-member | ≥2 | ParallelSM pipeline, 4MiB, UNPINNED | same | same kernel | `chunk_fetcher.rs:766` unpinned |
| stored-dominated single-member | any | StoredParallel → **chunked-streaming-writev** (Target 1) | same | none (verbatim copy + CRC) | `stored_split.rs` rewrite |
| mixed stored+Huffman | any | StoredParallel `decode_with_huffman_tail`, demote >50% tail → ParallelSM | same | shared kernel for tail | `stored_split.rs:330-368` (demotion gate KEPT) |
| multi-member | 1 | MultiMemberSeq → ABSORB to member-loop over shared kernel | same | shared kernel | currently libdeflate FFI (`mod.rs`) — absorb for sole-path |
| multi-member | ≥2 | MultiMemberPar → ABSORB | same | shared kernel | currently libdeflate FFI — absorb |
| BGZF / "GZ" subfield | any | GzippyParallel → block-framing shim (known boundaries, no block-finder) | same | shared kernel | currently libdeflate FFI — absorb |

**New archive-type gates authorized:** the stored-detection gate already exists
(`parallel_sm_unprofitable` + `first_block_is_stored`, `mod.rs:217-220`). Add a T1
per-class kernel route ONLY if Target 3's frozen-box A/B shows a *measured* per-class
winner (e.g. engine-A vs asm). Do not add speculative gates.

**Sole-path end-state:** 8 `DecodePath` → ~3 (ParallelSM-compressed, StoredParallel-shim,
framing shims for multi/BGZF) sharing ONE per-arch-dispatched flat clean kernel
(BMI2/asm on x86, HW-CRC+NEON engine A on aarch64), one thin-T1 + one T>1 orchestrator.

---

## THE MEASUREMENT MATRIX NEEDED (Gate-0 self-validating, both boxes, AMD owed)

| comparator | Intel (frozen) | mac (quiet aarch64) | AMD/Zen2 (owed) |
|---|---|---|---|
| rapidgzip | ✅ present | ✅ present (aarch64, no ISA-L — not SOTA) | owed |
| igzip (ISA-L) | ✅ present | weak/absent on arm64 | owed |
| **libdeflate** | **ADD** (top x86 T1 too) | ✅ present (SOTA aarch64) | **ADD** |
| **pigz** | **ADD** (Target 2) | **ADD** | **ADD** |

- Run via `standing.sh --box neurotic` / `standing_mac.sh`, extended with pigz+libdeflate
  arms. Gate-0 per arm: present + self-test to 1.0±spread (binary-vs-itself);
  byte-exact sha==zcat; same /dev/null sink; A/A ≤5% else cell UNTRUSTED; GHz/load
  reported. Gate-1: interleaved N≥13, Δ vs inter-run spread, TIE if Δ<spread.
- Axes: all T (1/2/4/8[/16]) × all archive types (silesia, monorepo, nasa, bignasa,
  squishy, weights, storedheavy, storedmix, pure_stored; model = corner, never a
  verdict-driver).
- **AMD/Zen2 is the standing LAW debt** — BMI2 `_pext/_pdep/_bzhi` are microcoded on
  Zen2, so every Intel result is NOT-YET-LAW until replayed there (`kernel_gate.sh
  --box solvency`, `AMD-STAGING.md` runbook).

---

## PRIORITIZED PLAN (goal-impact × tractability, highest first)

1. **Target 1 — StoredParallel chunked-streaming-writev.** Localized mechanism,
   byte-exact-tractable, the ONE real parallel rg loss, bandwidth-bound (big move
   expected). BUILD-AND-MEASURE. *Closes the last rg-loss front.*
2. **Target 2 — add pigz comparator + measure.** Cheap; banks a likely-existing win
   at all T>1 and the easy T1-vs-zlib bar. BUILD-AND-MEASURE, low effort.
3. **Target 4 — aarch64 clean-decode-core convergence (HW-CRC + Huffman/refill).**
   Biggest single gap (2.53× instr), localized to a SHARED kernel → advances x86 too.
   Highest cross-arch leverage. BUILD-AND-MEASURE.
4. **Target 3a — apportion the `decode_clean_into_contig` scaffold (~+1.4 instr/B).**
   The one unexhausted T1-vs-igzip surface; capturable % unknown (could be larger than
   the asm front), less heroic. BUILD-AND-MEASURE; sets up 3b/3c.
5. **Target 5 — sole-path collapse + dead-code delete (parity-gated).** Goal criterion;
   lowers future measurement cost. Mechanical/byte-exact.
6. **Convert the KEPT stack to LAW on AMD/Zen2** the moment solvency returns
   (`kernel_gate.sh`/`standing.sh --box solvency`) — one replay flips the whole banked
   stack NOT-YET-LAW → LAW.

**The R3 to surface to the user (do not resolve here):** Target 3 (T1-vs-igzip x86) is
oracle-bounded NEAR-LIMIT (~3-6% capturable by the bounded routes, high byte-exact
risk, multi-session) — the advisor-honest read is poor ROI, but the stated goal is
literal igzip parity and the user funded heroic rewrites. 3a (scaffold apportion) is
the cheapest way to *learn the real ceiling* before committing to the heroic asm/
stateless-kernel multi-session build. Put the accept-vs-fund fork to the user with
3a's apportionment number in hand.

---

## FLAGGED: near-irreducible vs clearly-achievable

- **Clearly achievable (mechanism localized, byte-exact route):** Target 1 (stored
  streaming), Target 2 (pigz — likely already won), Target 5 (cleanup).
- **Achievable with the funded build (localized to a shared kernel):** Target 4
  (aarch64 — HW CRC is a clean win; full libdeflate-NEON parity is the heroic stretch).
- **Near-limit (oracle-bounded, honest):** Target 3 T1-vs-igzip x86 — the hot emission
  loop is ALREADY converged (−1 instr vs `_04`); the residual is diffuse + dominated
  by non-portable parallel-machinery/resumable costs; the stateless-kernel removal
  oracle bounds it to ~3-6% wall. The scaffold (3a) is the only unmeasured surface and
  must be apportioned before pricing the heroic work.

**All stamps:** NOT-YET-LAW (Intel-frozen + quiet-mac); AMD/Zen2 owed for LAW; every
row above is a HYPOTHESIS until its paired gate fires and passes Gate-0/1 on both
arches.
