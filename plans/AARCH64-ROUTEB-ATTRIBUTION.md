# AARCH64 ROUTE-B — T1 attribution checkpoint (instrument-first, gated)

Branch `aarch64-routeb` (base `origin/kernel-converge-A` @c6152d17). Host:
aarch64-apple-darwin (NATIVE, M-series). macOS has no `perf` → wall-time only,
mitigated by interleaved best-of-N + A/A self-test + /dev/null both arms +
sha-verified. SINGLE-BOX, indicative → **NOT-YET-LAW** (a Linux aarch64 box is
owed for a law-grade number; AMD is x86). Governing law: only WRONG BYTES reverts.

## GATED MEASUREMENTS (Gate-0 self-test pass, Gate-1 Δ vs spread, sha-verified)

### 1. Production T1 gap (interleaved best-of-11, /dev/null, A/A≈1.000)
`scripts/bench/_aarch64_gap_local.sh`. gz = native `-p1` ParallelSM, bar = libdeflate-gunzip.

| corpus   | gz/libdeflate | Δ ms | spreads |
|----------|---------------|------|---------|
| silesia  | **1.194**     | ~48  | ≤30     |
| monorepo | **1.136**     | ~8   | ≤6      |
| nasa     | **1.166**     | ~19  | ≤11     |

### 2. Macro split — scaffold vs kernel (route-A thin oracle, this box)
`scripts/bench/_thin_oracle_local.sh` on silesia (bytes== all arms = non-inert):
- prod/libdeflate = 1.224, **thin/libdeflate = 1.095**, route-A capture (scaffold) = **10.6%**.
- ⇒ the +22% silesia gap ≈ **~10.6% parallel-pipeline SCAFFOLD** (sheddable by a thin-T1
  driver) + **~9.5% KERNEL residual**. (thin Δ≈20ms vs spread ~14-24 = borderline-signif;
  prod Δ≈46ms solid. Replicates the banked route-A finding on this box.)

### 3. The aarch64 kernel is ALREADY the libdeflate-template flat fastloop
`decode_huffman_fastloop_bounded` (consume_first_decode.rs:1302) — multi-literal
packed writes (≤8 lits → 1×u64 store), `copy_match_fast`, branchless u64 refill.
Wired into the aarch64 clean path at marker_inflate.rs:3221-3266 (cfg
`pure_inflate_decode && not(asm-kernel && x86_64)`). GZIPPY_DEBUG on silesia T1:
`flat_contig bytes=211738388` (= 99.9% of 211968000 output), `careful_calls=11`,
`clean_lut_builds=0` ⇒ the flat fastloop IS the kernel; careful tail + engine-B
negligible. (NOTE: this supersedes the prior `plans/AARCH64-WORKSTREAM.md` design,
which described an older literal-chain shape — the flat bounded fastloop was wired
in for aarch64 after that doc.)

### 4. GATE-2 sub-region injector — WHERE the per-symbol wall goes
`--features inject-probe`, knobs GZIPPY_INJ_{LIT,COPY,REFILL}=N inject a dependent
LCG chain into each emission sub-region (byte-transparent: sha==unset at every N;
non-inert: `inject_fired` = exact visit count). `scripts/bench/_inject_slope_local.sh`,
interleaved best-of-9. Marginal time per injected op = slopeΔ(N=16) / (visits×16):

| region  | silesia ns/op | nasa ns/op | silesia visits | nasa visits |
|---------|---------------|------------|----------------|-------------|
| LIT     | 1.01          | 0.99       | 10.31M         | 1.77M       |
| COPY    | 1.11          | 1.11       | 17.97M         | 6.51M       |
| REFILL  | 1.15          | 1.12       | 24.35M         | 7.63M       |

**VERDICT (Gate-2, replicated on 2 corpora with very different lit:match ratios):**
all three sub-regions are ON the T1 critical path — NO slack region. Marginal cost is
~UNIFORM (~1.0-1.15 ns/op); literal-store has marginally MORE ILP slack than
copy/refill (~12%, consistent both corpora). There is **NO single fat sub-region** a
NEON/asm kernel could target; the fastloop is uniformly throughput/latency-bound and
already template-faithful. (Mirrors x86 NIGHT35: kernel on the wall, surplus spread,
no single lever.)

## IMPLICATION for Route B (HYPOTHESIS — for supervisor/advisor, not yet acted on)
The brief's premise (aarch64 runs an OLD shape; Route B = hand-write a NEON kernel)
is partly falsified by §3-§4: the aarch64 kernel is already the converged
libdeflate-template fastloop and is uniformly bound (no slack sub-region for NEON to
exploit; NEON only helps long-match copy, whose marginal cost is not dominant). The
LARGER and cleaner aarch64 lever is the **thin-T1 driver** (sheds the ~10.6%
scaffold — already mandated by the END-STATE VISION), not a speculative NEON kernel
for a uniformly-bound +9.5% residual. Both are needed for ≥0.99× parity; sequencing
+ whether to fund the NEON kernel at all is a judgement call for the supervisor.

## RE-RUN
```
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features --features pure-rust-inflate            # production (injector elided)
bash scripts/bench/_aarch64_gap_local.sh /tmp/silesia.gz 11
BIN=./target/release/examples/streaming_thin bash scripts/bench/_thin_oracle_local.sh /tmp/silesia.gz 9
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features --features pure-rust-inflate,inject-probe # measurement
for k in GZIPPY_INJ_LIT GZIPPY_INJ_COPY GZIPPY_INJ_REFILL; do bash scripts/bench/_inject_slope_local.sh /tmp/silesia.gz $k 9 0 4 8 16; done
```

## 5. NIGHT25-STYLE DISASM DIFF — gz fastloop vs libdeflate fastloop (2026-06-21)
Gate-5 tier = DISASM-READ = **HYPOTHESIS** (not a wall verdict). But it DISCRIMINATES
the rival "codegen" hypotheses deterministically (these are facts readable from the asm,
not inferences about the wall). Artifacts: `plans/disasm/gz_fastloop_aarch64.s`
(gz `decode_huffman_fastloop_bounded`, 618 insns, prod build `pure-rust-inflate`,
target-cpu=native, codegen-units=1, INJECTOR ELIDED — verified 0 getenv/inject residue)
vs `plans/disasm/libdeflate_decompress_aarch64.s` (Homebrew libdeflate 1.25
`_libdeflate_deflate_decompress_ex`, the EXACT code `libdeflate-gunzip` = the §1 bar runs;
fastloop ≈ 0x508-0x84c). Both aarch64-apple-darwin.

### Bucketed per-iteration divergence (gz − libdeflate)
| bucket | finding | verdict |
|--------|---------|---------|
| **array bounds-checks** | gz: NONE. Table loads are raw `ldr w,[x4,x8,lsl#2]`; lit store raw `strb w,[x1,x2]`; copy raw ptrs. NO panic/brk landing pad, NO `bl` (fully inlined). get_unchecked/`*mut` already in effect. | **FALSIFIED** — no cheap get_unchecked win waiting |
| **register spills** | gz loop body (LBB1115_5..98): **0** sp-relative mem ops. Everything in x19-x28 + caller regs. | **FALSIFIED** — no spill to fix |
| **match-copy** | gz: NEON-128 `ldp/stp q0,q1` + explicit `prfm pldl1keep` (12 q-ops, 2 prefetch). libdeflate: scalar-64 `ldr x/str x` (0 NEON, 0 prefetch). | gz **MORE** aggressive — NOT a loss; a NEON kernel here is redundant |
| **refill / loop ORGANIZATION** | gz: top-of-loop `cmp bitsleft,#44; b.hs` refill-gate + nested 8-deep literal unroll (packs ≤8 lits → 1×u64 store via ORR-assembly) + returns to a per-iter budget check (LBB1115_98: 2 cmp + 2 b + 2 add). libdeflate: software-pipelined symbol-CHAIN — issues the NEXT litlen table-load EARLY (0x6ac in lit path, 0x5d8 mid-match-copy) to hide load-use latency, stores lits singly (`strb [post]`), and re-checks budget ONLY after a match (chains literals bounds-check-free). | **THE one material divergence** — a real libdeflate technique gz hasn't fully ported |

### DISCRIMINATED VERDICT (the brief's fork)
NEITHER of the brief's two clean outcomes; a refined THIRD:
- **NOT "gross/sloppy Rust codegen → cheap fix"**: the obvious Rust-level remedies
  (get_unchecked, de-spill, vectorize copy) are ALREADY APPLIED — falsified above. No
  cheap get_unchecked/restructure capture is sitting unclaimed.
- **NOT "NEON-kernel target"**: the divergence is in SCALAR decode scheduling, not the
  copy; the copy is already NEON-128+prefetch. Hand-NEON is aimed at the wrong region.
- **NOT yet "irreducible Rust-vs-C floor → escalate"**: that escalation is PREMATURE
  because exactly ONE concrete structural divergence remains — libdeflate's
  software-pipelined fastloop ORGANIZATION (early next-symbol table-load to hide
  load-use latency; bounds-check amortized per-match not per-iteration).

**Cheapest surviving candidate remedy** = a faithful port of libdeflate 1.25's
software-pipelined fastloop organization (a Rust LOOP-STRUCTURE refactor — moderate,
NOT multi-session hand-asm, NOT NEON). **Per Gate-2 whether it pays is UNPROVEN** and
MUST be settled by an interleaved AB (Gate-1), not assumed.
- PRIOR (explicit HYPOTHESIS, NOT banked): §4 showed the loop is uniformly
  throughput-bound with NO fat slack sub-region; latency-hiding pipelining helps
  LATENCY-bound loops, so the expected payoff is small → leans toward the
  irreducible-floor escalation. This prior sets up the AB; it is not a finding.

### RE-RUN (disasm diff)
```
RUSTFLAGS="-C target-cpu=native --emit asm -C codegen-units=1" cargo rustc --release --lib --no-default-features --features pure-rust-inflate
# gz: grep decode_huffman_fastloop_bounded in target/release/deps/gzippy-*.s
ar x /opt/homebrew/lib/libdeflate.a deflate_decompress.c.o && objdump -d --no-show-raw-insn deflate_decompress.c.o   # libdeflate bar
```

## OWED / CAVEATS
- NOT-YET-LAW: single box (macOS M-series, no perf), wall-only. Linux aarch64 box owed.
- PRE-EXISTING (not introduced here, confirmed by stash-and-test on pristine @c6152d17):
  2 lib-test failures on aarch64 — `marker_inflate::tests::dist_table_matches_dist_hc_differential`
  and `resumable::tests::dynamic_header_straddling_encoded_until_bits_errors_loudly`.
  Flag to supervisor (branch debt, x86/aarch64 divergence in dist_hc differential).
- The injector measures MARGINAL sensitivity (on-wall vs slack), not absolute share;
  it cannot rank existing per-region time (no removal-oracle possible — removing a
  literal-store/copy breaks bytes). The uniform marginal cost ⇒ existing share tracks
  existing work-per-region, not measurable here without perf.
