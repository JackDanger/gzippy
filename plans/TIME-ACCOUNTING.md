# TIME-ACCOUNTING.md — where every millisecond of the gzippy→rapidgzip x86_64 parity gap goes

Definitive per-T, per-stage accounting. Corpus = silesia-large (`/root/silesia.gz`,
68 MB gz → 212 MB raw, sha-pinned). Host = neurotic guest (trainer, 16-core x86_64),
frozen via `bench-lock` (Plex + 7 noisy LXCs paused, procs_running quiet-gate 1.0–1.75
on every run, no_turbo=1, governor=performance, even-core pin). Every wall is
interleaved best-of-N (N=11 parity / 9 oracle, warmup dropped) vs rapidgzip 0.16.0 on
the SAME pin, regular-file sink, sha==pin verified. Host restored + orphan-clean after
every session. Disproof: plans/TIME-ACCOUNTING-advisor-verdict.md.

> Method note: the campaign's prior "ocl_cf" (the `GZIPPY_ISAL_ENGINE_ORACLE` knob)
> was found to swap only **2 of 18 chunks** to ISA-L at T4/T8 (live counter), because
> production's clean tail routes through the `flip_to_clean` FOLD, not `finish_decode`.
> That broke the "74 ms engine" premise. The correct full-coverage ISA-L isolation is
> the **gzippy-isal build + the knob** (`ocl_REAL`: 16/17 chunks at T1, 14/18 at
> T4/T8). All splits below use `ocl_REAL`. (Bare gzippy-isal is NOT an engine swap —
> its clean tail is pure-Rust ResumableInflate2; gzip_chunk.rs:1957.)

## 1. The three walls (ms, min-of-N, tight, sha-verified)

| T  | native (production) | spread | ocl_REAL (gzippy pipe + REAL ISA-L) | spread | rapidgzip | spread |
|----|--------------------:|-------:|------------------------------------:|-------:|----------:|-------:|
| 1  | **1518** | 1% | **1015** | 1% | **919** | 1% |
| 4  | **667**  | 3% | **548**  | 3% | **496** | 2% |
| 8  | **414**  | 4% | **365**  | 6% | **357–363** | 3–6% |

## 2. Engine vs non-engine split (the corrected decomposition)

`engine = native − ocl_REAL` (cost of pure-Rust clean decode over REAL ISA-L).
`non-engine = ocl_REAL − rg` (gzippy pipeline cost over rg, engine removed).

| T  | gap (N−R) | ENGINE (N−ocl_REAL) | NON-ENGINE (ocl_REAL−R) | engine share |
|----|----------:|--------------------:|------------------------:|-------------:|
| 1  | 599 | **503** | **96** | 84% |
| 4  | 171 | **119** | **52** | 70% |
| 8  | ~54 | **49**  | **2**  | ~96% |

**Headline (inverts the charter's premise):** the gap is **ENGINE-dominated**, and the
non-engine residual **vanishes as T grows** — at T8, ocl_REAL (365) ≈ rg (363): with
ISA-L decode gzippy **ties rapidgzip**, so the entire T8 gap was the engine. The
charter's "53 ms non-engine @T4" is CONFIRMED (52 ms); its "74 ms engine" was
undercounted by the broken 2-chunk knob — the real engine term is 119 ms @T4.
(Coverage caveat: ocl_REAL still runs 4 small `finished_no_flip` chunks in pure-Rust at
T4/T8 ⇒ engine is a lower bound, non-engine an upper bound; see verdict ⚠.)

## 3. NON-ENGINE sub-decomposition — CAUSAL removal oracles (Δ = wall response to removing the stage)

Removing each stage and measuring the interleaved wall (not summing a trace):

| stage / knob | native T4 Δ | native T8 Δ | ocl_REAL T4 Δ | ocl_REAL T8 Δ | tag |
|--------------|------------:|------------:|--------------:|--------------:|-----|
| output writev (`SKIP_WRITEV_SYSCALL`) | **95** | **67** | **73** | **48** | SHARED w/ rg (both write 212 MB) — irreducible |
| marker/window bootstrap (`SEED_WINDOWS`) | ~0 (−5) | ~0 (1) | **0** | **2** | OFF critical path — NOT a lever |
| ring→data drain copy (`FOLD_NODRAIN`) | ~0 (−1) | 0 | n/a (isal path) | n/a | OFF critical path (copy-free-to-final already elides it) |
| per-byte CRC (`FOLD_NOCRC`) | ~0 (−3) | 0 | n/a | n/a | OFF critical path |

**The only non-zero non-engine critical-path stage is OUTPUT writev (48–95 ms)** — and
it is **SHARED**: rapidgzip writes the same 212 MB; matched rg --verbose shows
apply-window 0.089 s ≈ gzippy apply_window 0.086 s aggregate, CRC 0.020 s, alloc/copy
0.065 s. The non-engine RESIDUAL (ocl_REAL−rg: 96/52/2 ms) is therefore **NOT output**
(shared), **NOT bootstrap/drain/CRC** (Δ≈0); it is **low-T in-order pipeline scheduling
+ the 4 pure-Rust fallback chunks**, and it is **0 (parity) at T8**.

Corroborating fulcrum_total (native T8 trace, hypothesis-generator only): wall-critical
consumer = **97.7 % WAIT-on-workers** (the wall is set by worker decode), top self-time
`worker.block_body` **1.633 s** (the u16 marker+fold engine), `apply_window` 0.086 s,
`scan_candidate` (block finder/speculation) 0.141 s, writev self ≈ 0 (writev cost is the
causal 48–67 ms, not visible as trace self-time because it overlaps).

## 4. ENGINE sub-decomposition — why pure-Rust clean decode is ~1.6× ISA-L

Per-symbol rate from T1 (least parallel confound): decode_native ≈ 1348 ms vs
decode_ISA-L ≈ 845 ms ⇒ **~1.60× slower/symbol (0.62× ISA-L)**, consistent with the
campaign's cited 1.68×/0.594×. Source-verified attribution (plans/asm-kernel-
feasibility-report.md, vendor `isa-l/igzip/*.asm` first-hand; production loop
`resumable.rs:998-1001`, `gzip_chunk.rs` u16 ring):

| cause | igzip technique forfeited | gzippy state | share of the 1.6× | recoverable? |
|-------|---------------------------|--------------|------------------:|--------------|
| per-symbol LUT + **u16 ring 2-bytes/byte traffic** | #1 packed flat-u8 u32 table (≤3 symbols/load) | `output_ring: [u16; N]`, 1 symbol/iter | **dominant (~bulk of 87%)** | only via flat-u8 kernel rewrite |
| no speculative pipelined bulk loop | #2 branchless 8-byte multi-lit store + preload-before-type + SHLX 64-bit refill | per-symbol; SIMD multi-lit reverted @ca52389 | part of ~87% | needs flat-u8 |
| resumable yield-check tax | #4 slop-margin: NO per-symbol bounds/yield check | "every iteration a yield check" (resumable.rs:998) | part of ~87% | authorized to elide w/ FASTLOOP margin (unmeasured) |
| scalar back-ref copy | #3 MOVDQU overlap-doubling copy | u16 ring forbids flat-u8 MOVDQU | small | needs flat-u8 |
| **measured graft ceiling** | E2 AVX2 copy + E3 packed store + E4 wide refill ON the u16 ring | grafted, benched | **~13 % recoverable** | yes, but plateaus 0.41×→0.46× — 12σ short of tie |

**Crux (code-level fact, not preference):** all four igzip tricks require a flat-u8
output buffer where the back-ref source is already-final bytes. gzippy's faithful u16
marker ring (`m_window16` port) structurally forbids them ⇒ ~87 % of the engine gap is
**STRUCTURAL** (needs a flat-u8 igzip-class clean-tail kernel — either rapidgzip's
two-phase ISA-L handoff re-done as inline-Rust-asm, or a u8-direct post-flip ring),
and only ~13 % is recoverable by grafts on the current ring.

## 5. Per-stage verdict table (each tagged recoverable vs shared/irreducible)

| stage | T1 | T4 | T8 | tag |
|-------|---:|---:|---:|-----|
| **clean-decode engine** (pure-Rust vs ISA-L) | 503 | 119 | 49 | **gzippy-specific, structurally-hard** — THE lever; ~13 % cheap graft, ~87 % needs flat-u8 kernel |
| output writev | ~(in 96) | ~73* | ~48* | **shared/irreducible** (rg writes 212 MB too); *value is the removal Δ, mostly cancels vs rg |
| marker bootstrap / apply_window | small | ~0 | ~0 (2) | **off critical path** + shared (rg apply-window ≈ equal) |
| ring→data drain / CRC | — | ~0 | ~0 | **already optimal** (copy-free-to-final) |
| low-T pipeline scheduling residual | ~96 | ~52 | **~2 (parity)** | gzippy-specific, **self-resolves by T8** |

## 6. Bottom line
- At **T8 gzippy already ties rapidgzip once the clean decode is ISA-L-rate** (365 vs
  363). The non-engine pipeline is at parity at T8.
- The single lever across all T is the **clean-decode engine** (49/119/503 ms). It is
  not closeable by shaving pipeline stages (bootstrap/drain/CRC/window are all off the
  critical path or shared) — it requires a **flat-u8 igzip-class clean-tail kernel**;
  the u16 marker ring caps grafts at ~13 %.
- Output writev (~50–95 ms) is real and on the critical path but **shared with rg** —
  not a parity lever.

### Provenance
Branch `reimplement-isa-l` (worktree). native sha `2bd907efe1b366d3` (gzippy-native),
ocl/isal sha `b3d9435587186f27` (gzippy-isal), both RUSTFLAGS=`-C target-cpu=native`,
rapidgzip 0.16.0. Raw logs: /tmp/ta/*.log, traces /tmp/ta/ft/. Walls/oracles via
scripts/bench/{parity,oracle}.sh + bench-lock; ocl_REAL + removal deltas via the guest
runners in /tmp/ta/guest_*.sh (interleaved, sha-checked, regular-file sink).
