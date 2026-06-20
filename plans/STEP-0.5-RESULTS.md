# STEP-0.5 RESULTS — engine A (flat) vs engine B (two-level) clean-decode A/B

Scope: **macOS-aarch64 (Apple M1 Pro), NOT-YET-LAW cross-arch.** Deterministic-
instruction primitive (`/usr/bin/time -l` instructions-retired + cycles-elapsed),
two-point (R, 2R) marginal-cost subtraction to isolate the kernel loop from all
fixed setup. Both arms PURE-RUST (FFI off), same in-RAM contiguous sink, same body
bit. Harness: `examples/kernel_ab_aarch64.rs` + `scripts/bench/standing/kernel_ab_aarch64.py`.
N=7, R=2000 (2R total output ≥256 MiB/corpus).

## The two engines compared
- **ENGINE A (flat)** — `decode_huffman_libdeflate_style` (consume_first_decode.rs:632):
  flat masked `LitLenTable` (TABLE_BITS=12), `saved_bitbuf`, up-to-8-literal
  lookahead, single refill/iteration, contiguous output, NEON copy. The EXISTING
  production decoder for bgzf / scan_inflate / multi-member.
- **ENGINE B (two-level)** — `Block::decode_clean_into_contig` (marker_inflate.rs:2970):
  ISA-L two-level packed table (`asm.lut_litlen`), the unpack branch-chain /
  P3.2 literal-chain, double-refill cadence. The PRODUCTION CLEAN T1 contig path
  the parallel-SM engine runs (chunk_decode.rs:1695). The CONTIG variant (not the
  ring `decode_clean_fast_loop`) is the CONSERVATIVE/fairest table-vs-cadence
  discriminator — it excludes the `% U8_RING_SIZE` ring-masking confound that
  would only bias further toward flat.

## Gate-0 (PASSED — else the numbers do not exist)
- byte-exact: engine A out == engine B out == flate2/gzip oracle, every corpus.
- cursor conservation: engine-A body_bit == engine-B body_bit, every corpus.
- non-inert (the aarch64 inert-knob hazard): `FLAT_DECODE_CALLS` advanced by
  EXACTLY `reps` after each engine-A timed loop ⇒ the flat kernel PROVABLY ran on
  aarch64 (not a silently-skipped arch-gated path).
- same sink: both arms write an in-RAM contiguous dst; the driver pipes the process
  to /dev/null identically for both.

## RESULTS (instr/B, cyc/B; per-rep marginal)
| corpus | B/rep | A instr/B | B instr/B | B/A | A cheaper by | A cyc/B | B cyc/B | cyc B/A |
|--------|------:|----------:|----------:|----:|-------------:|--------:|--------:|--------:|
| webster (real text)        | 81120 |  8.374 | 17.306 | **2.067×** |  −8.932 | 3.174 | 6.220 | 1.96× |
| mozilla (real binary)      | 62129 |  4.330 | 16.637 | **3.842×** | −12.307 | 2.018 | 4.716 | 2.34× |
| lit_extreme (per-symbol)   | 22973 | 13.910 | 32.916 | **2.366×** | −19.006 | 6.332 | 12.639 | 2.00× |
| backref_extreme (per-copy) | 92160 |  0.626 |  0.814 | **1.301×** |  −0.188 | 0.096 | 0.182 | 1.89× |

## VERDICT
**Engine A (flat) BEATS engine B (two-level) DECISIVELY on gzippy's OWN primitives —
on EVERY corpus, instr AND cyc.** 2.07–3.84× fewer instr/B on real corpora; 2.37×
on the per-symbol extreme; ~2× cyc/B everywhere.

**It closes the entire clean-core excess and then some.** Deliverable-1 measured the
production (engine-B) path at 35.97 instr/B on the per-symbol extreme vs libdeflate
16.47 (+19.50 excess), and silesia at 19.38 vs 7.69 (+11.69). On the isolated A/B:
- engine B per-symbol = 32.916 instr/B (consistent with the 35.97 production number
  minus CRC/ring/scaffold);
- engine A per-symbol = **13.910 instr/B — which BEATS libdeflate's 16.47**;
- engine A saves **−19.006 instr/B vs engine B**, ≈ the entire +19.50 clean-core excess.

So the +19.5/+11.69 aarch64 clean-core excess IS the flat-vs-two-level ENGINE
difference, and the flat engine ALREADY EXISTS and already wins.

## TABLE-WIDTH vs CADENCE/UNPACK (the design's rival hypotheses)
This A/B changes BOTH together (flat 12-bit table AND single-refill AND no-unpack-
chain AND multi-literal lookahead), so it proves the FLAT-ENGINE BUNDLE is the lever
but does not split the bundle internally. The cross-comparison gives a HYPOTHESIS-
tier read, NOT a gated split:
- libdeflate (flat, 11-bit) = 16.47 instr/B; engine A (flat, 12-bit) = 13.91; engine
  B (two-level, 12/13-bit) = 32.9. Two FLAT decoders of similar width sit at ~14–16
  instr/B; the two-level decoder is ~2× higher.
- ⇒ **HYPOTHESIS (unvalidated): the dominant lever is the engine STRUCTURE (two-level
  packed table + unpack branch-chain + double-refill cadence), NOT table WIDTH per se**
  (flat-11 libdeflate ≈ flat-12 engine A). An exact table-vs-cadence isolation needs
  a THIRD engine (flat table + engine-B cadence, or two-level + single refill) — a
  kernel modification, out of scope this turn. The design therefore converges the
  whole flat bundle (which engine A already is) rather than over-betting on the split.

## DESIGN CONSEQUENCE
The design premise — "BUILD a flat kernel from scratch" — is FALSIFIED as new work.
Engine A IS that kernel and already wins. Reframe to: **WIRE engine A into the
parallel-SM clean dispatch + the resumable seam** (see CLEAN-KERNEL-DESIGN.md §1/§6).

## RE-RUN
```
cargo build --release --no-default-features --features gzippy-native --example kernel_ab_aarch64
python3 scripts/bench/standing/kernel_ab_aarch64.py
```
Cross-ISA LAW (owed): build Intel asm-OFF and run the same A/B; engine A (pure-Rust)
must replicate the win there before any LAW claim.
