# RUNG (d) EVALUATION — extending the asm-kernel approach to the MARKER bootstrap loop

Date: 2026-06-11. Base `f3db1146` (post isal-decide bank). Branch `engine/asm-rung-d`,
worktree `/tmp/gz-asmd`. Written and committed BEFORE any implementation (charter
discipline, plans/asm-campaign.md §2/§7).

## 0. The target, precisely

The MARKER bootstrap loop is `Block::read_internal_compressed_specialized::<true>`
(`src/decompress/parallel/marker_inflate.rs:1491`): the window-absent u16 body
decode. Its hot path is the `'mfast` u16 speculative fast loop (`:1845-1985`,
ported 2026-05-31 from rg's `readInternalCompressedMultiCached`); its tail/edge
path is the shared careful loop (`:1987-2137`). It runs on BOTH builds
(`pure_inflate_decode` is set for gzippy-native AND gzippy-isal — build.rs:94-95,
`gzippy-isal = ["pure-rust-inflate", "isal-compression"]`): every chunk decodes
here from chunk start until the flip arms (`should_flip` per `read()`,
marker_inflate.rs:1133) or forever if it never arms.

The CLEAN contig loop (`decode_clean_into_contig`, `:2186`) — the rung-(c) asm
kernel's home — is its sibling. The two loops share the litlen LUT and the bit
reader but have DIVERGED in optimization level: the clean loop received P3.1
through P3.5 plus the asm kernel; the marker loop received NONE of those passes.

## 1. Per-symbol work: marker loop vs clean contig loop

Side-by-side of the `'mfast` loop (:1845) against the contig fast loop (:2533),
classified as INTRINSIC (the marker semantics demand it) vs NON-INTRINSIC
(missing optimization passes, no semantic reason):

### Intrinsic marker-mode extras (would survive a perfect port)
| # | extra | cost shape |
|---|-------|-----------|
| I1 | u16 store width (vendor `Window=uint16_t`): literal stores widened (3 shifts/ors per packet), backref copies move 2× bytes through `emit_backref_ring` (u16) | ~3 ALU ops/packet + 2× store/load traffic on copies |
| I2 | `distance_to_last_marker` bookkeeping: `+= lit_prefix` per packet (:1905); per backref the `>= distance` skip check, else O(length) backward scan (`emit_backref_ring:3530-3568`) | 1 add/packet; 1 cmp/backref in the armed-run common case; the scan only while clean run < distance |
| I3 | Ring modulus + wrap guard: `pos % RING_SIZE` (AND) + `dst_phys + FAST_OUT_SLOP <= RING_SIZE` per iteration; breaks to careful loop each 64 Ki-slot wrap | ~2 ALU + 1 predictable branch/iter |
| I4 | No clean-mode `distance > decoded` range check (vendor const-folds it OUT for markers — :1963) | NEGATIVE cost (marker mode is cheaper here) |
| I5 | Flip-arming check `should_flip` | per `read()` call, NOT per symbol — negligible |

Sum: low single-digit cycles per symbol plus 2× copy bandwidth. Vendor runs this
same shape (templated on Window) at acceptable rates — the banked rg trace shows
31.25% replaced-marker symbols with no marker-mode collapse — so intrinsic tax
does NOT explain a 2-4× rate gap.

### Non-intrinsic deficits (the un-run P3 passes), ranked by banked mechanism
| # | missing pass | clean-loop evidence | marker-loop site |
|---|--------------|---------------------|------------------|
| N1 | **DistTable single-lookup distance decode (P3.1/P3.4)**. Marker loop runs `dist_hc.decode` (12-bit `DistanceShortBitsCached` → symbol only) → bounds check → `DISTANCE_EXTRA` load → availability check + possible refill → peek/mask/consume → `DISTANCE_BASE` load (:1924-1956). The contig loop replaced this exact chain with ONE `DistEntry` load + in-register `consume_entry`/`decode_distance` (:2777-2808) | contig_prof measured the backref iteration at **84.8 vs 61.5 cyc — the dist chain IS the 23-cyc gap** (field doc, marker_inflate.rs:427-440); backref class = 62.6% of classed cycles | dist site fires once per backref |
| N2 | local-Bits register mirror (P3.1, Lever-B1 class). Marker loop threads `&mut Bits` — bitbuf/bitsleft/pos round-trip MEMORY each symbol (raw-pointer ring stores defeat aliasing analysis, the exact B1 finding) | P3.1 T1-recovery-class win on contig | every consume/refill/decode |
| N3 | Q3 lone-literal 1-byte store + P3.2 lit chain. Marker loop does the UNCONDITIONAL 8-byte speculative wide store even for `sym_count==1` (:1875-1879) — the shape the advisor REFUTED (Q3) on the clean loop — and has no runtime lit-chain (~1.96-2.57 lits/iter mechanism) | Q3 + P3.2 banked | every lone-literal packet (the dominant packet class) |
| N4 | P3.5 c1/c2/c4: no preload-before-copy, no fused litlen→dist spec load, bottom refill is an unconditional call + `decode` keeps its backstop (vs threshold-gated refill + `decode_prefilled`) | P3.5 +1.8% of T1 wall on contig | every iter |
| N5 | `record_backreference_for_sparsity` is a per-backref `&mut self` call (flag re-loaded from memory); contig hoists `track_backrefs` | hoisted in contig | per backref |

The rate data agrees with this classification: marker loop runs 51-83 MB/s
(pre-flip phases, fulcrum decide @ isal, banked 2026-06-11) to 166 MB/s
(bignasa-isal steady-state) vs the PRE-asm clean Rust loop at ~340 MB/s-class
(decode ceiling 642.6 ms for 211.6 MB at T1) and the asm clean loop at
~535 MB/s-class. A 2-4× gap on the same data shapes cannot be I1-I5 (a few
cycles + 2× copy traffic); the bulk is N1-N5.

## 2. What fraction of each failing cell's wall is marker-loop time

Labeled honestly: these are ATTRIBUTIONS from banked counters/traces, not causal
verdicts — except where a causal perturbation is on record.

| cell | marker-loop share | source + label |
|------|-------------------|----------------|
| bignasa-isal T8 (0.918-0.940) | **~100% of decoded bytes**: `flip_to_clean=0`, `isal_chunks=1/22` — ISA-L idle, every byte through the marker loop at 166 MB/s | fulcrum decide counters, banked (the 0-flip reading ruled SOLID after the counter-semantics audit). COUNTER-attribution; decode-criticality at T8 not separately perturbed in this cell |
| silesia-isal T4 (0.899) | pre-flip bootstrap phases at 51-83 MB/s; `body_bytes` = bootstrap-phase bytes ONLY, and silesia hands ~2/3 of volume to ISA-L post-flip ⇒ marker loop owns ~1/3 of bytes at the SLOWEST per-byte rate | decide counters + the banked counter-semantics caution. Attribution |
| isal model (0.857) / T16 (0.924-0.945) | same structure: pre-flip phase at 51-83 MB/s on every chunk | attribution |
| native silesia T4 (0.870) / T8 (0.966) | window-absent chunks decode marker-mode until arming; at T8 only ~136.8M of 211.9M bytes route through clean-contig (charter §9 — the asm kernel's T8 delta was -4.6% vs -22% at T1 BECAUSE marker-mode + pipeline own the rest) | charter §9 measured byte routing. Attribution |
| native model T8 (0.727-0.763) | the deepest cell; mixed marker + clean; the rung-(c) kernel already took -20% — the residual includes the marker share | attribution |
| CAUSAL anchor | the window-absent bootstrap was slow-injected (+50-200%, `GZIPPY_SLOW_MARKER_MODE` class) and the wall moved ~proportionally, frequency-neutral control survived | banked causal perturbation (CLAUDE.md header) — the marker loop IS on the critical path; what a SPEEDUP buys is bounded by §3, not by this slope |

## 3. Ceiling estimate

No marker-loop removal oracle exists yet (removal_oracle.rs covers the contig
clean loop only: NODECODE replay + NOSTORE). Building a marker NOSTORE oracle
means eliding u16 ring stores while keeping `pos`/`distance_marker` accounting —
moderate cost, NOT cheap (the marker scan READS the ring it just wrote, so a
NOSTORE arm changes `distance_marker` evolution unless the scan is also stubbed;
that makes the oracle semantics-bearing and audit-heavy). Per the charter's
"bound from rates" alternative:

**Rate-comparison bound.** The marker loop runs at 0.15-0.5× the clean-Rust
per-byte rate on comparable data. If the N1-N5 ports recover even HALF the gap
to the pre-asm clean Rust rate (i.e. marker 166 → ~250 MB/s on bignasa-class
data, 51-83 → ~80-120 MB/s on pre-flip phases):

- bignasa-isal T8: marker decode is ~100% of bytes; cell deficit 6-8%. A ≥30%
  marker-rate gain covers the full deficit ~2-4× over IF decode gates the wall
  there (the causal anchor says the marker loop is critical-path on
  window-absent workloads; T8 pipeline slack could absorb some).
- silesia-isal T4: marker owns ~1/3 of bytes at the slowest rate ⇒ marker time
  is plausibly ~half of decode time; deficit 9.5%. A 30-50% marker-rate gain
  is several × the deficit if pre-flip decode gates the T4 wall (bimodality
  noted in the decide run says scheduling also participates — honest caveat).
- native cells: smaller marker share (T1 nearly all clean after first chunks),
  T4/T8 partial. Expect smaller but nonzero deltas.

**Per-lever bound for increment 1 (N1, DistTable):** 23 cyc/backref measured on
the clean loop's identical chain. At silesia-class symbol mix (backref ~62.6% of
classed cycles, mean match ~6-7 bytes ⇒ roughly 1 backref per ~10-14 output
bytes against ~85-cyc backref iterations), removing ~23 cyc/backref is a
**~15-25% marker-loop rate gain by itself** — hypothesis to falsify on the
frozen A/B, not a promise. Backref-light corpora (high-literal) will show less.

KILL line for the rung: if the full N1-N5 port lands and the marker-heavy cells
(bignasa-isal first) do not move, the marker loop's criticality premise for
those cells is falsified at the SPEEDUP side (slow-down slope ≠ speed-up
ceiling) and the marker NOSTORE oracle must be built before any further spend.

## 4. The rung-(d) shape decision

Three candidates, with the recommendation and mechanism:

### (i) Extend the rung-(c) kernel with a marker mode (one kernel, mode flag) — REJECTED for now
Mechanism of rejection (rule 7: mechanism, not a narrow miss):
- The kernel's EXIT-STATE CONTRACT is contiguous-u8-specific in load-bearing
  ways: `out_lim`/`dst` are BYTE pointers with X3 "overshoot garbage above dst"
  semantics; marker mode needs u16-SLOT addressing on a WRAPPING ring (`%
  RING_SIZE` + wrap guard per iter), so every store, every backref source
  computation, and both top guards change shape — that is not a flag, it is a
  second body.
- Register pressure: KernCtx is at the 12-operand envelope (charter §5).
  Marker mode adds the ring base/mask, `distance_marker`, and the backward
  marker-scan loop (a data-dependent inner loop with a mid-arm bail — a NEW
  exit class). The P3.4-shape copy the kernel transliterated is u8
  `emit_backref_contig`; marker needs u16 `emit_backref_ring` incl. the scan.
- The c3 lesson cuts the other way here: rung (c) won BECAUSE the asm was
  built against the FINISHED (P3.1-P3.5) Rust loop as the register contract.
  The marker loop is NOT at its Rust plateau — asm written against today's
  marker loop would pin the dist_hc three-load chain into the contract and
  need a rewrite after the Rust passes land anyway.

### (ii) Sibling marker kernel — DEFERRED, entry condition defined
Plausible LATER, exactly like rung (c) was: build it only when (a) the Rust
marker loop has absorbed N1-N5 and frozen-measured its plateau, and (b) the
remaining marker-cell deficit still exceeds the ship bar, and (c) a marker
NOSTORE-class oracle (or an honest rate bound at the new plateau) shows the
remaining headroom fits the ≥2% bar. Cost honestly stated: a second
differential surface (the c2/c3-class gauntlet — 15k windowed trials, positive
controls, coverage counters) for a loop that is a MINORITY share on every cell
except bignasa-isal.

### (iii) Rust-level marker-loop optimization first — **RECOMMENDED**
Mechanism FOR:
- The marker loop is VIRGIN territory for five levers that are already
  proven-with-mechanism on the sibling loop (N1-N5), portable in pure Rust,
  byte-exact by the same equivalence arguments already differentialed
  (DistTable ≡ dist_hc: identical symbols ⇒ identical distance + bit
  consumption; raw==0 ⇔ `None`), and individually killable.
- The rung-(c) experience: the Rust passes (P3.1-P3.5) captured real wins
  BEFORE asm and DEFINED the asm contract. Same order here.
- Expected value: N1 alone is a measured-mechanism ~15-25% marker-rate
  lever; the whole Rust ladder plausibly covers the marker-heavy cells'
  deficits without writing a register contract for a wrapping u16 ring.
- It lifts BOTH builds at once (`pure_inflate_decode` on both; bignasa-isal's
  100%-marker bytes flow through exactly this loop).

**Increment 1 (this session): N1 — DistTable distance decode in the `'mfast`
marker fast loop.** Highest banked mechanism (the 84.8→61.5 cyc chain), the
smallest blast radius (dist site only; careful loop stays dist_hc verbatim),
and the cleanest byte-exactness argument (the P3.1 equivalence, already
tested). Requires sharing the P3.4-amortized `ensure dist table` build (fixed
static table / memcmp reuse / `dist_table_checked` latch) with the marker path
— the P3.1 "lazy: contig only" build gate moves up, the amortization semantics
stay. Falsifier pre-registered below.

Increments 2..n (later, one at a time, gauntleted): N2 local-Bits mirror, N3
Q3 store + lit chain, N4 c1/c2/c4 schedule, N5 hoist — then the (ii) entry
condition is evaluated.

## 5. Pre-registered falsifier for increment 1 (F-d1)

- Correctness gate: fmt/clippy clean; full suite BOTH feature sets
  (`pure-rust-inflate` and `pure-rust-inflate,gzippy-isal` compile + lib
  tests); new marker-path differential (window-absent decode w/ dictionary,
  DistTable arm vs dist_hc reference arm, byte + cursor equality) in the SAME
  commit; sha grid {silesia, model, bignasa, storedmix} × T{1,4,8} × BOTH
  builds vs base — all byte-identical, silesia == pinned corpus sha.
- Frozen masked interleaved A/B (bench-lock, no_turbo readback, RESTORE
  verified; same-sink law — both arms identical sink): **bignasa-isal T8**
  (the pure-marker cell) + **silesia T4 both builds**, n≥9, sha-verified arms.
- SHIP ≥2% on a marker-heavy cell; TIE ⇒ keep-if-byte-exact (rule 7a), bank
  the mechanism reading (backref density of the cell vs the 23-cyc bound) and
  proceed to N2 rather than abandoning the direction (rule 7: a TIE on one
  fix is not a refutation of the rung).
- Counter check: `flip_to_clean`/`isal_chunks` on bignasa-isal must be
  UNCHANGED by the lever (0 flips / 1 isal chunk) — the lever must not be a
  routing change in disguise.

---

## 6. F-d1 VERDICT (2026-06-11, same session): TIE-KEEP + the bignasa attribution CAUSALLY REFUTED

Increment 1 built, gauntleted, frozen-measured (commit `67431727`,
branch `engine/asm-rung-d`).

### Gauntlet (all green)
- fmt clean; clippy: default-features 0 errors; pure-rust-inflate 9 == 9 at
  base (pre-existing lint debt, count unchanged).
- Suites: local pure-rust-inflate **947/0** (incl. the new differential +
  corpus_silesia_if_available + resumable_decodes_real_silesia); local
  default-features **674/0**; guest gzippy-isal **939 pass / 2
  contention-flakes** (`diff_ratio_parallel_single_member_speedup` — the
  documented charter-§10 load-flake class — and `bench_production_inflate`,
  a perf-guard sibling; BOTH pass in isolation on the same box).
- Differential: `marker_fast_loop_dist_table_matches_dist_hc_and_payload` —
  5 payload classes × 2 resumable caps, ON/OFF arms byte+cursor+
  distance_marker equality, marker-resolved output == payload ground truth,
  engagement counter ON>0 / OFF==0.
- Sha grid: **72/72** byte-identical — {silesia, model, bignasa, storedmix}
  × T{1,4,8} × {nat-on, nat-off, nat-base, isal-on, isal-off, isal-base};
  silesia == the pinned corpus sha.
- Effect verification (`GZIPPY_MARKER_DIST_STATS=1`, production binaries):
  bignasa-isal T8 **lut_backrefs=11,591,490 / hc=0** (ON) ↔ **0 / 11,591,490**
  (kill-switch) — the lever is live and the kill-switch real; silesia T4
  4,352,108 on BOTH builds (deterministic, identical counts).

### Frozen interleaved A/B (bench-lock, n=9, file sink on /dev/shm both arms,
### sha-verified 81/81 + 27/27 reps, RESTORE verified after each bracket)
| cell | ON | OFF (same-bin kill) | base (cross-bin) | ON-vs-OFF | sign |
|------|----|----|----|-----------|------|
| bignasa-isal T8 | 1098 | 1086 | 1095 | +1.1% | 3/9 — TIE (spread 991-1109) |
| silesia-isal T4 | 571 | 574 | 570 | -0.5% | 5/9 — TIE |
| silesia-native T4 | 573 | 584 | 575 | **-1.9%** | **8/9** — consistent, below the 2% bar |
| bignasa-isal T1 (follow-up) | 1350 | 1348 | 1347 | +0.15% | 4/9 — TIE (tight ±5ms) — and VACUOUS, see below |

(Frozen walls are no_turbo-pinned — ~4x the turbo walls; arms within each
bracket are like-for-like.)

### The mechanism chain (causal probes, not attribution)
1. **T1 is vacuous for ANY marker-loop lever**: `GZIPPY_SLOW_HITS=1` +
   marker knob at T1 → **inject hits = 0**; no marker-dist stats line at T1.
   Sequential (T1) chunk decode gives every chunk its predecessor window —
   window-absent marker decode only exists under T>1 speculation. (The
   marker-knob comment in `read_internal_compressed_specialized` saying the
   `<true>` path "always uses the careful loop" is STALE since the mfast
   port; the knob now also de-selects the fast loop — perturbations include
   that shape change.)
2. **At T8 the marker loop has ≥2x slack under the wall on bignasa-isal**:
   inject site alive (36,171,210 hits), and the interleaved wall response is
   FLAT at f100 (≈100% per-event slow + careful-loop forcing): 426/419/454 →
   413/416/505 ms; only f400 moves it (798/828/818 ≈ 2x). A region whose 2x
   slowdown is wall-free cannot pay for a speedup (slow-down slope ≠ speed-up
   ceiling — here even the slope is flat at moderate scale).
3. Therefore the isal-decide "bignasa-isal = 100% marker bytes at 166 MB/s ⇒
   the marker loop owns the cell" read was the ANALYST-BIASED attribution
   class the measurement process warns about: the marker loop is huge in
   byte-share and WALL-NEUTRAL on that cell. The bignasa-isal deficit
   (0.918-0.940) lives in the SERIAL residue: T1 1350 → T8 1086 frozen is a
   1.24x speedup on 8 workers — drain/apply-window/writer, not decode.
4. silesia-native T4 (-1.9%, 8/9) is the one cell with a consistent
   direction: marker share ~1/3 of bytes, less pipeline slack at T4 than T8.
   Real but sub-bar.

### Disposition (rule 7a)
- **KEEP, default-ON**: byte-exact everywhere (72/72 + differential), TIE-or-
  slightly-better on every measured cell, kill-switch + effect counters are
  permanent A/B instrumentation, and it makes the marker loop structurally
  MORE like the contig loop (one dist-decode shape across the engine).
- **The N2-N5 ladder is DEPRIORITIZED, not refuted**: the same slack
  mechanism that nulls N1 on bignasa-isal T8 bounds every marker-loop
  micro-lever there; silesia-native T4 (the only responsive cell) caps the
  ladder's value at low single-digit %. Before ANY further marker-loop
  spend, the rung-(d) charter premise must be re-aimed: the next causal
  target on bignasa-isal is the SERIAL residue (the 1.24x scaling wall —
  drain/apply-window/markers-replacement/writer chain), which this session's
  probes localized for free.
- The marker NOSTORE oracle (eval §3) is NOT needed for this verdict — the
  slow-injection knee (flat at 2x, responsive at 4x+) already bounds the
  speed-up ceiling at ~zero for the T8 cell.
