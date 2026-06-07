# DISPROOF BRIEF — is the T8 binder the u16 marker path, or scheduling/serial?

You are an INDEPENDENT DISPROOF advisor. Try to BREAK the finding below. Read the
cited source first-hand. Verdict: UPHELD / UPHELD-WITH-CAVEATS / REFUTED, with the
strongest disproof attempt you could mount.

## Context
gzippy is porting rapidgzip's parallel single-member gzip decode to pure Rust, goal
= 1.0x wall TIE. T8 (8 pinned cores) gzippy ~0.221s vs rapidgzip ~0.130s = 1.70x.
The prior charter named the #1 T8 binder as "the u16 marker/post-flip path carrying
58.6% of decoded body bytes — the bulk of T>=2 bytes take the slow u16 path."

## FINDING THIS TURN (what I'm asking you to break)

### Premise correction (source-verified)
The "58.6% u16" came from counter `BOOTSTRAP_POST_FLIP_U16_BYTES`
(gzip_chunk.rs:97 decl, :1302 increment). Its increment fires when
`!block.contains_marker_bytes()` AFTER a block decode — i.e. it counts bytes in
blocks decoded in CLEAN (post-flip, `<false>`) mode. Since commit fc1c965b, the
`<false>` path decodes u8-DIRECT into the u8 view of output_ring
(marker_inflate.rs:1397-1401 `ring_modulus = U8_RING_SIZE`, :1685 `ring8.write`).
So the 57.5% measured (`post_flip_u16_bytes=96,898,692 / body_bytes=168,440,273`)
are bytes ALREADY on the fast u8 path, NOT the slow u16 path. The counter NAME and
its doc comment (gzip_chunk.rs:91-96) are STALE (pre-fc1c965b). The genuine
u16-marker (`<true>`) fraction is the inverse ≈ 42.5% (~71.5M bytes) — the pre-flip
prefix each speculative chunk must decode before 32KiB clean accumulates and it flips.

### Causal perturbation (this turn, byte-exact, frequency-neutral control)
Added `GZIPPY_SLOW_MARKER_MODE` — a u16-`<true>`-path twin of the existing clean-only
`GZIPPY_SLOW_MODE` knob (slow_knob.rs marker_spin_iters; wired marker_inflate.rs:1453).
Byte-transparent: knob OFF/marker100/clean100 ALL emit canonical silesia sha
028bd002...cb410f. Binary gzippy-mk == gzippy-varv on the wall (1.018x TIE).
Locked guest (10.30.0.199 double-ssh, 16c gov=performance, measure.sh interleaved
sha-verified, RAW=211968000, T8 CPUS=0,2,4,6,8,10,12,14, N=11). NOTE box load 3-5
(elevated; measure.sh's interleaved RELATIVE delta is load/turbo-robust).

T8 results (relative vs OFF):
- CLEAN +100% spin -> +27% wall; CLEAN +100% SLEEP control -> +27% wall (IDENTICAL =>
  NOT a turbo artifact). CLEAN +200% SLEEP -> +55%. => clean decode-compute genuinely
  gates ~27% of the T8 wall (freq-neutral confirmed, ~linear slope).
- MARKER +200% spin -> +21% wall; MARKER +200% SLEEP control -> +7% wall. The spin
  effect does NOT survive the freq-neutral control (21% -> 7%) => most of the marker
  spin's apparent effect is a turbo artifact; genuine u16-marker criticality ≈ +7% for
  +200% inject => marker decode-compute ≈ 3.5% of the T8 wall.
- T1 (CPUS=0): MARKER +100% -> +0% (TIE), +200% -> +4%. Near-flat = coverage check:
  at T1 almost all chunks are window-seeded clean (finished_no_flip=0), so the u16 path
  barely runs; the knob fires ∝ u16 bytes, so near-flat at T1 is expected & validates it.

### Conclusion I want broken
1. The u16 marker path is ~3.5% of the T8 wall (freq-neutral), NOT the dominant binder.
   The charter's "58.6% u16 = biggest prize" is FALSIFIED (misnamed counter + the
   marker knob's freq-neutral response is tiny).
2. Total decode-compute at T8 ≈ clean 27% + marker 3.5% ≈ ~30% of wall (~0.066s of
   0.221s). The remaining ~70% (~0.155s) is the scheduling/serial/overlap term —
   which alone EXCEEDS rapidgzip's entire 0.130s wall. So the dominant T8 binder is
   scheduling/serial/overlap (charter binder #2: pool fill 73-83%, ~0.06s serial
   in-order publish/drain/CRC, confirmed-offset prefetch head-of-line stalls), and
   rapidgzip ties DESPITE the same engine gap by overlapping decode under scheduling.
3. NEXT = attack the scheduling/serial term (decode-WAIT vs serial-WORK decomposition),
   NOT the u16 path. The clean kernel stays the confirmed T1 lever and a ~27% T8
   contributor (real but a minority; Rule-3 caveat: this is a slow-down SLOPE/floor,
   a faster kernel COULD unbind some scheduling cascade — the ceiling is not proven
   by the slope).

## Strongest disproof angles to attempt
- Is the marker knob actually firing on the u16 path (not const-folded away / dead)?
  It's wired into the careful loop (marker_inflate.rs:1644 inject) which the `<true>`
  path always uses; T1 near-flat could ALSO mean "knob is dead." Distinguish: the T8
  marker spin DID move +21% (so it fires); the question is only turbo-vs-critical.
- Is the spin-vs-sleep gap (21% vs 7%) a valid turbo-artifact reading, or is the SLEEP
  control UNDER-injecting (descheduling lets the consumer catch up, masking real
  criticality)? Could the truth be between 7% and 21%?
- Coverage: marker knob fires per u16 decode EVENT; 42.5% of BYTES are u16 but back-ref
  events emit many bytes per event, so EVENT coverage may differ from BYTE fraction —
  does that bias the +200%->+7% reading down?
- Rule-3: does a low marker SLOW-slope actually bound the marker SPEED-up ceiling, given
  a faster u16 path could let chunks flip sooner / consumer catch up (cascade unbind)?
- Does the ~70% scheduling residual hold, or is it an elimination-by-residual trap?

Cite file:line. Be adversarial.
