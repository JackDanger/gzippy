# Independent Disproof Verdict — inner-Huffman-kernel (VAR_V) PLATEAU

Role: adversarial read-only advisor. Method: first-hand source read of
`benches/engine_isolation.rs`, `src/decompress/parallel/marker_inflate.rs`
(`read_internal_compressed_specialized<false>` :1324-1609; `emit_backref_ring_u8`
:2695-2759), `src/decompress/parallel/lut_huffman.rs` (LUT builders + `decode`).
No builds run. I attacked the leader's PLATEAU (V/ISA-L = 0.554, pre-registered
PASS ≥ 0.85).

---

## 1. BYTE-EXACTNESS — **UPHELD** (gate is genuine, not vacuous)

`run_chunk` (engine_isolation.rs:743-746) sets
`exact[k] = o.len() >= n_actual && &o[..n_actual] == scalar && scalar_eq_isal`,
with `n_actual` taken from the VAR_I scalar probe and `scalar` anchored to ISA-L
(`scalar_eq_isal`, :739). This is a **full `n_actual`-byte** comparison, not a
prefix, and the anchor is the independent ISA-L oracle — so a variant cannot pass
by matching a short prefix. A VAR_V early-return (`out[base..out_pos.min(target_end)]`,
e.g. :465/508/514/533) yields `o.len() < n_actual` ⇒ `exact=false` ⇒ VOID, never a
silent pass; `SHA_ALL_EQUAL=yes` therefore means VAR_V matched all `n_actual` bytes.

I checked the two semantic risks directly. (a) **Packet unpack** (VAR_V fast loop
:478-499) reproduces production's `code <= 255 || sym_count > 1` literal rule
(marker_inflate.rs:1495): leading elements are literals, only a `remaining==1`
element may be a length/EOB code; the speculative 8-byte store writes the low-24
packed bytes unconditionally and advances by `lit_prefix` only, and the back-ref
copy overwrites any spurious trailing-code byte at `out_pos`. Correct. (b) The
**overlap discriminator** in `flat_backref_copy` (:343-372) is byte-for-byte the
same case split as production `emit_backref_ring_u8` (:2704-2758): `distance>=8`
word copy / `distance<8` exact copy for non-overlap, `distance==1` RLE memset,
`1<distance<length` sequential. The ONLY difference is production's `% U8_RING_SIZE`
+ wrap-straddle handling, which the flat buffer elides — see the optimism note below.

**Optimism (does the flat buffer hide a production cost?) — YES, and it CUTS FOR
PLATEAU.** VAR_V's flat linear buffer (window prepended, `out_pos-distance` direct)
pays no ring modulo and no wrap-straddle branch that production's clean path
(`emit_backref_ring_u8`) pays on every back-ref. VAR_V also pays no resumable-yield
tax, no marker bookkeeping (production `<false>` already const-folds that), no CRC.
So 0.555 is the **cost-stripped, optimistic** number. A real gzippy-ring integration
would be **≤ 0.555**, further from 0.85 — the optimism makes PLATEAU *more* robust,
not less. (It is, separately, a fair model of an igzip-style flat-output integration,
which is what the lever claims to port.)

## 2. SELF-TEST / DENOMINATOR FAIRNESS — **UPHELD**

`iii/i = 2.76 ∈ [2.5,3.6]` (engine_isolation.rs:891) passes. VAR_III
(`decompress_deflate_from_bit`) is a pure clean decode of the *same* slice, window,
and `n_actual`, byte-gated equal to scalar — a fair, charter-mandated "1.0" (the
goal IS ISA-L/rapidgzip parity). Both numerator and denominator are stripped of
production integration costs and both write to a **flat output buffer** (ISA-L's
`next_out` is linear), so the flat-buffer advantage is *shared*, not a VAR_V-only
artifact. Every "wouldn't survive production" cost in VAR_V (ring modulo, resumable
yield) would only *lower* the ratio. Denominator is fair; PLATEAU is not a
denominator artifact.

## 3. ISOLATION VALIDITY — **PARTIALLY OVERTURNED** (0.555 bounds THIS lever, not the pure-Rust ceiling)

The bench cleanly isolates trick #2: VAR_IV_E234 (best non-speculative u8 engine,
116 MB/s) → VAR_V (156) is a +34% honest delta for the speculative pipeline; VAR_V
vs VAR_I (u16-ring scalar) bundles too many axes to attribute. So far so good.

But 0.555 bounds **the packed-LUT + speculative-pipeline + flat-u8 + word-copy
stack** — NOT "the pure-Rust ceiling." CLAUDE.md explicitly authorizes, and VAR_V
does NOT contain: **BMI2 PEXT/BZHI runtime-dispatch bit reader** (VAR_V uses the
generic `Bits` peek/consume/refill), **inline-asm hot loop to match vendor codegen**,
and deeper SIMD in the decode itself. ISA-L's 1.80× edge is substantially a
hand-scheduled-asm + BMI2 edge that this Rust stack never attempts. Per governing
rule #7 ("a rejection needs a mechanism, not a narrow miss"), a TIE on *this* stack
does not reject those un-tried, charter-named levers. So the engine lever as a
**whole** is not closed by this bench (consistent with memory
`project_engine_plateau_pure_rust`: the faithful-u8 plateau is being measured on one
stack at a time).

## 4. §3 PROJECTION (the load-bearing call) — **OVERTURNED as framed**

The leader holds the pre-registered **ratio** (V/ISA-L ≥ 0.85) as governing and
calls PLATEAU even though VAR_V's projected `decode_wall ≈ 0.410s` lands **below the
~0.54 placement floor** (and below the 0.604 bar). These measure two different
things:

- **0.85 ratio** = "does pure-Rust decode *rate* equal ISA-L?" — a **producer-side
  attribution**.
- **decode_wall < floor** = "is decode still the *critical-path binder*?"

The charter's done-criterion is **WALL parity with rapidgzip**, not decode-rate
parity with ISA-L. The Measurement PROCESS preamble is explicit that busy-time /
latency-share / critical-path "blame" are all analyst-biasable and "NEVER conclude a
lever from attribution." The 0.85 decode-RATE ratio is exactly such an attribution.
If the projection is real, `decode_wall 0.410 < floor 0.54` means **decode has
stopped binding** — i.e. this lever *succeeded* at moving the binder onto placement,
which is a WIN by the wall-parity goal, mislabeled as a plateau-failure.

Caveat that keeps me from flipping all the way to "lever succeeded": the projection
is ITSELF an attribution (a microbench rate scaled into a wall), not a causal
perturbation. By rule #1 ("perturb, don't attribute") and rule #8 ("numbers from the
fullest Fulcrum test"), **neither** the 0.85 ratio **nor** the 0.410 projection is a
verdict. The honest state is INCONCLUSIVE-pending-integration, not PLATEAU.

Procedural credit: pre-registering the 0.85 falsifier and refusing to move it after
seeing 0.555 is correct discipline (rule #5). The fault is not goalpost-moving — it
is that the pre-registered goalpost was a **rate** target on a **wall**-parity
charter.

---

## OVERALL — PLATEAU is honest ONLY in a narrow sense; it is OVERSTATED as a verdict on the lever

- **Honest:** "The packed-LUT + speculative-pipeline (flat-u8) stack does not reach
  ISA-L decode-rate parity (0.555, ~16σ short of the pre-registered 0.85)." The
  bench, byte-gate, and denominator are sound, and the flat-buffer optimism only
  hardens this.
- **Overstated / premature** if read (as the orchestrator likely will) as "the
  pure-Rust engine lever is exhausted, decode is the residual binder":
  1. 0.555 bounds one stack, not the ceiling — **BMI2 bit-reader and inline-asm are
     charter-authorized and un-measured** (attack 3); rejecting them needs a
     mechanism, not this TIE.
  2. The leader's OWN projection says `decode_wall < placement floor` — i.e. decode
     **stopped binding** — which makes this lever a binder-move SUCCESS, not a
     plateau; the 0.85 ratio is a forbidden producer-side attribution as a verdict
     (attack 4).

**Recommended next action (the only thing that settles it):** stop adjudicating on
the rate ratio. INTEGRATE VAR_V into the production clean path (paying the real ring
modulo + resumable contract) and measure the **interleaved end-to-end wall** vs
rapidgzip (`scripts/measure.sh`, sha-verified) — the only verdict per rule #1. Either
the wall moves (lever was real; residual is placement) or it does not (then PLATEAU
is confirmed *on the wall*, and the remaining engine headroom is the un-tried
BMI2/asm levers, to be rejected only with a mechanism). Do not close the engine lever
on a 0.555 microbench ratio.
