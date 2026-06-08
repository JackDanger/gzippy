# DISPROOF ADVISOR BRIEF — decode-compute vs store-bandwidth localization (contig clean tail)

## Context
gzippy → rapidgzip parity. native_fold ~0.77-0.79× rg. The ≤0.11× residual is LOCATED on
the T8 critical path = the gzippy-native FOLD contig clean Huffman decode loop
(`decode_clean_into_contig`, marker_inflate.rs). Prior turn: a per-WHOLE-LOOP-BODY slow knob
(`GZIPPY_SLOW_MODE`) proved the loop is on-path (monotone spin50<spin100, sleep preserves), but
the advisor flagged it CANNOT separate Huffman decode-COMPUTE (table lookup + bit extraction)
from literal-store / back-ref-copy STORE bandwidth. The packed multi-literal STORE *technique*
TIE'd (kept 7a). Owed: a DECODE-ONLY perturbation to localize WHICH sub-resource binds.

## What I built (byte-transparent instrument, OFF==reference)
Two SEPARATE env knobs in slow_knob.rs + wired into BOTH the fast (production VAR_V) and careful
loops of `decode_clean_into_contig`:
- `GZIPPY_SLOW_DECODE` — injects ONLY after each `lut_litlen.decode()` and each
  `dist_hc.decode()` (pure table-lookup + bit-extraction compute).
- `GZIPPY_SLOW_STORE` — injects ONLY after each literal-store and each `emit_backref_contig`
  (store/copy bandwidth).
Unlike the old `GZIPPY_SLOW_MODE`, neither knob forces the careful loop — the fast-loop gate stays
`slow_spin==0` only — so the perturbation lands on the PRODUCTION fast path. `GZIPPY_SLOW_KIND=sleep`
gives the freq-neutral control per knob. Byte-exact: OFF==DEC100==STORE100==DECsleep ==
028bd002…cb410f (silesia T8, x86_64 guest, pure-rust-inflate native).

## Measurements (locked guest 10.30.0.199, taskset 0,2,4,6,8,10,12,14, gov=perf, interleaved
## scripts/measure.sh, sha-OK every run; ratios are X×off = off_time/X_time)

PASS 1 (N=11, load 1.29):
  off 0.1649 | dec50 0.910 dec100 0.787 decsleep100 0.760 | store50 0.920 store100 0.819 storesleep100 0.804 | rg 1.255

PASS 2 (N=11, load 1.71):
  off 0.1659 | dec50 0.908 dec100 0.768 decsleep100 0.743 | store50 0.917 store100 0.812 storesleep100 0.809 | rg 1.244

PASS 3 tight (N=15, load 2.39):
  off 0.1674 | dec100 0.815 store100 0.827 | decsleep 0.756 storesleep 0.825 | rg 1.286

## My read (what I want disproved)
1. BOTH decode-compute AND store-bandwidth are on the contig clean T8 critical path — both are
   monotone (off > 50% > 100%) and neither sleep control collapses to off. Neither sub-resource
   is slack.
2. DECODE-COMPUTE is the more robustly-on-path binder: its freq-neutral SLEEP control PRESERVES or
   EXCEEDS its busy-spin delta in all 3 passes (0.760≤0.787, 0.743≤0.768, 0.756<0.815), the rule-2
   signature of real criticality. The STORE sleep control roughly EQUALS its spin (0.804≈0.819,
   0.809≈0.812, 0.825≈0.827) — no freq-neutral growth, so part of store's spin delta is a turbo
   artifact, i.e. store-bandwidth criticality is weaker.
3. This is consistent with the prior packed-multi-literal STORE technique TIE: the store-side
   headroom is already exploited; decode-compute is the residual binder.
4. Ceiling for any decode-side speed-up payoff = the standing engine-removed ocl_cf 0.925×
   (native_fold ~0.79× → 0.925× ≈ 0.13× headroom). NOT the slow-down slope (rule 3).
5. CONCLUSION: apply decode-side technique to the located binder — BMI2 PEXT/BZHI bit extraction,
   ISA-L-class packed-u32 multi-symbol LUT (one u32 load retires multiple symbols + bit-length),
   wider refill — on `read_internal_compressed_specialized::<false>` / `decode_clean_into_contig`,
   byte-exact, remove-and-measure on the T8 wall.

## Questions
- Q1: Is "both on-path, decode-compute the more robust binder" sound, or does the near-equal
  busy-spin (dec100≈store100) mean I can't separate them and should call it "decode AND store both
  bind"?
- Q2: Is the sleep-vs-spin discriminator (decode sleep preserves, store sleep flat) a valid
  rule-2 reading, or confounded (e.g. store's sleep batches differently because store events
  outnumber decode events, changing the ns-debt discharge cadence)?
- Q3: Is it legitimate to proceed to a decode-side technique now, bounded by ocl_cf 0.925×, or is
  a tighter clean-loop-only ISA-L oracle owed FIRST before any technique?
- Q4: Strongest disproof of "decode-compute is the binder"?

## ADDENDUM (PASS 2) — after the first verdict: BMI2-already-optimal + decode_free oracle CONTAMINATED ⇒ plateau/fork question

The first advisor (verdict file, this dir) REFUTED "decode-compute is the more-robust binder"
(serial-dependency-chain non-separability; sleep discriminator confounded by nanosleep
granularity + event-count cadence) and named two owed decisive probes: (a) a decode-compute-
removed oracle, (b) a BMI2 on/off A/B. I ran BOTH:

**(b) BMI2 A/B via disassembly (decisive, cheap):** the production binary is built with
`RUSTFLAGS=-C target-cpu=native` on the Haswell+ guest, so `target_feature=bmi2` is on. The
built binary already contains 433 BZHI/PEXT-family instructions; the hot decode masks in
`LutLitLenCode::decode` (`next_bits & ((1<<long_max_len)-1)`, `next_bits & ((1<<12)-1)`) are the
canonical BZHI idiom LLVM already lowers to `bzhi`. ⇒ a MANUAL BMI2 PEXT/BZHI bit-extraction
technique has NO headroom — the bit-extraction compute is already at the BMI2 ceiling. This
directly falsifies "bit-refill/extraction compute is the binder."

**(a) decode-compute-removed oracle (GZIPPY_DECODE_FREE, built + run) is CONTAMINATED, dead.**
I built a measurement-only knob that replaces the fast+careful clean-loop Huffman `decode()` with
a fixed synthetic single-literal (0x41, 9-bit consume) while leaving every store/copy running, to
bound decode-compute's share (symmetric of the accepted GZIPPY_FOLD_NODRAIN store-removed knob).
It ENTERED the path (eprintln confirmed, local_cap ~16.7M, 16.7M synthetic writes/call) but
produced BYTE-IDENTICAL correct output. Cause (verbose counters): the wrong synthetic bytes blow
out speculation — `flip_to_clean=874` (vs 12), `Speculation failure modes: other=857` — so 857
chunks RE-DECODE through the safety net with the correct engine, masking both the bytes and the
wall. A wrong-bytes decode oracle is fundamentally contaminated by the spec-failure re-decode net
(unlike NODRAIN, whose wrong bytes only surface at the terminal CRC, not at per-chunk window
seeding). I REVERTED it (instrument left = only the byte-exact decode/store knobs).

**Standing facts about the located region (decode_clean_into_contig fast loop):**
- BMI2 BZHI/PEXT bit-extraction: already emitted/optimal (433 occurrences).
- ISA-L-class packed-u32 multi-symbol LUT (sym_count 1..=3): ALREADY present (short_code_lookup).
- Packed multi-literal wide STORE technique: already ported, TIE'd, kept 7a.
- Fast loop is already software-pipelined (decode `pre` preloaded ahead of use → table-load
  latency hidden).
- Standing speed-up CEILING = engine-removed ocl_cf 0.925× (bounds compute+store TOGETHER);
  native_fold ~0.77-0.79× ⇒ ~0.13-0.15× total headroom for ALL clean-loop decode-side work.

## PASS 2 QUESTIONS (plateau/fork gate)
- P1: Given BMI2 is already optimal, the multi-symbol LUT already present, the store technique
  TIE'd, and the loop already pipelined — are the authorized decode-side techniques EXHAUSTED on
  this located region, i.e. is this the 1.0×-vs-no-FFI PLATEAU?
- P2: The campaign rule forbids declaring STOP/TIE without a VALIDATED oracle. The decode_free
  oracle is contaminated by the spec-failure net. Is there a non-contaminated way to bound
  decode-compute's share (e.g. a CORRECT-bytes oracle: decode the real stream once, cache the
  symbol/length/dist stream, replay it with decode() free — heavy but byte-exact), or does the
  BMI2-already-optimal disasm fact SUFFICE to bound the decode-compute lever to ~0 without it?
- P3: If a residual remains after BMI2/LUT/pipeline are all already in place, WHERE does rapidgzip
  (WITH_ISAL) get its remaining clean-tail speed that gzippy-native cannot match in pure Rust —
  is it the ISA-L hand-tuned asm inner loop (a structural FFI advantage = the fork), or a
  still-unported pure-Rust-reachable technique?
- P4: Is escalating the 1.0×-vs-no-FFI FORK now (with the bounded number native_fold ~0.79× vs
  ocl_cf 0.925× ceiling, BMI2/LUT/store/pipeline exhausted) the correct call, or is a specific
  named technique still owed first?
