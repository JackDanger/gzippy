# Disproof-advisor verdict — marker-loop port (commit f4c58fba)

Read-only, source-verified, no build/measure. Adversarial: goal was to BREAK,
not ratify. Every premise checked first-hand against cited `file:line`.

## TL;DR

| Claim | Verdict |
|-------|---------|
| C1 — byte-exact + faithful u16 deltas | **UPHELD** (structural; tests claimed, not re-run by me) |
| C2 — wall is a TIE | **UPHELD-WITH-CAVEATS** (honest "no regression," but instrument too noisy to be a disproof of anything ≤10–20%) |
| C3 — marker u16 path is ~2% of body ⇒ TIE expected | **REFUTED** — the `post_flip_u16_bytes` counter measures the CLEAN (flipped) sliver; the new marker loop's actual domain is the COMPLEMENT (~98% of bootstrap body). 2% is the wrong number for this claim. |

**Single most load-bearing correction (C3):** The commit/brief read the
`BOOTSTRAP_POST_FLIP_U16_BYTES` counter backwards. It increments **only when a
block ends CLEAN** (`flipped_clean = !block.contains_marker_bytes()`,
`gzip_chunk.rs:1489–1495`). The new fast loop runs **only in `<true>` (marker)
mode** — dispatched at `marker_inflate.rs:1252` when `contains_marker_bytes ==
true`, i.e. on the **pre-flip** bytes. Those pre-flip bytes are
`bs_b_bytes − post_flip ≈ 98%` of the bootstrap body, **not** the 2%
`post_flip` slice. So "the marker loop port has a ~2%-of-body ceiling" is the
inverse of what the counter shows. The TIE is therefore **not** explained by a
2% domain; it must be explained some other way (most likely: the loop pipelines
only the *literal* decode and leaves the marker-mode bottleneck — the O(length)
backward marker scan in `emit_backref_ring::<true>`,
`marker_inflate.rs:3006–3027` — completely unchanged).

---

## C1 — BYTE-EXACT + faithful (not a divergence) — UPHELD

Verified the three u16 deltas against both the clean fast loop it mirrors
(`marker_inflate.rs:1491–1646`) and the careful loop it falls through to
(`marker_inflate.rs:1841–1969`).

**Δ1 — widened u16 literal store. CORRECT.**
`widened = (p&0xFF) | ((p&0xFF00)<<8) | ((p&0xFF_0000)<<16)` (`:1701`):
- bits[0:8]=b0 ⇒ u16 slot0 = b0
- (p&0xFF00)<<8 ⇒ bits[16:24]=b1 ⇒ slot1 = b1
- (p&0xFF_0000)<<16 ⇒ bits[32:40]=b2 ⇒ slot2 = b2
- bits[48:64]=0 ⇒ slot3 = 0

Exactly the brief's "b0→slot0, b1→slot1, b2→slot2, slot3=0." Value-identical to
the careful loop's per-literal `ring_ptr.add(pos % RING_SIZE).write(code & 0xFF)`
(`:1868`): each committed slot holds a byte `<256`, and only `lit_prefix` slots
are committed (`pos += lit_prefix`, `:1740`) — trailing speculative slots are
overwritten by the next packet or the back-ref, same discipline as the clean
store (`:1527–1554`). The 3-byte packing cap is inherited from the clean loop
(`packed = sym0 & 0x00FF_FFFF`); both rely on the same LUT guarantee of ≤3
packed symbols, so no new constraint is introduced.

**No ring-wrap straddle. CONFIRMED.**
Top guard `dst_phys + FAST_OUT_SLOP <= RING_SIZE` with `FAST_OUT_SLOP = 8 +
MAX_RUN_LENGTH + 16 = 282` (`:1487`), `dst_phys = pos % RING_SIZE`, `RING_SIZE =
65536` slots (`:232`). The 8-byte store touches 4 slots: `dst_phys+4 ≤
dst_phys+282 ≤ RING_SIZE`. Worst-case back-ref extent after the literal advance:
`dst_phys + lit_prefix(≤3) + rounded_run(≤260)` = `dst_phys + 263 < dst_phys +
282 ≤ RING_SIZE` (rounded = `(258+3)&!3 = 260`, `emit_backref_ring:2909`). The
282-slot reservation, though originally sized in *bytes* for the u8 ring, is
*more* than sufficient as *slots* for the u16 run (258 slots, not 516). Safe.

**No bit-cursor desync on `break 'mfast`. CONFIRMED.**
`pre = self.lut_litlen.decode(bits)` is decoded but its bits are consumed only
inside the loop (`bits.consume(bit_count0)`, `:1714`); the bottom re-decodes a
fresh, un-consumed `pre` (`:1819`). On every `break 'mfast` the cursor sits
exactly before `pre`'s bits. The careful loop's first act is `bits.refill();
let (symbol,…) = self.lut_litlen_decode(bits)` (`:1840–1841`) — it re-decodes
from that position, discarding `pre`. `pos`/`emitted`/`distance_marker` are
shared locals carried forward. This is byte-for-byte the same fall-through
contract the **already-proven** clean fast loop uses (`:1641–1645`).

**Δ3 — dropped window-range check is faithful. CONFIRMED.**
Careful loop: `if !CONTAINS_MARKERS && distance > self.decoded_bytes + emitted`
(`:1948`) — the `!CONTAINS_MARKERS &&` const-folds the whole check away in
marker mode. The fast loop simply omits it. Identical behavior. (And it *must*
be omitted: valid marker back-refs legitimately reach past `decoded+emitted`
into the predecessor/marker window.)

**Δ2 — distance_marker maintenance is identical. CONFIRMED.**
Fast loop `distance_marker += lit_prefix` (`:1742`) == careful loop's
per-literal `distance_marker += 1` summed over the packet (`:1883`). Back-refs in
BOTH loops call the **same function** `emit_backref_ring::<CONTAINS_MARKERS>`
with `&mut distance_marker` (`:1802` vs `:1953`); its marker bookkeeping
(`:2990–3028`: fast skip `distance_marker >= distance`, else O(length) backward
scan) is therefore byte-identical regardless of which loop reached it. The
brief's worry ("maintained identically from fast or careful?") is moot — it is
literally one function.

**Caveat:** I did not build or run the 856 lib tests, the seam test, or the
silesia sha. The structural argument is sound and the loop is a faithful
transliteration of the proven clean loop + careful-loop primitives, but the
byte-exact *claim* rests on test runs I could not independently reproduce.

## C2 — WALL is a TIE — UPHELD-WITH-CAVEATS

+1.2% / +3.0% / +0.0% within a 10–38% inter-run spread is, per the charter's own
rule ("Δ < inter-run spread ⇒ TIE, full stop"), an honest TIE — **no regression
was measured.** Interleaved `measure.sh` (A/B/A/B…) makes the *delta*
frequency-neutral even if absolute times drift with turbo/thermal state, so the
freq-neutrality objection does not break the TIE.

**But the disproof cuts the other way:** a 10–38% spread cannot detect anything
smaller than ~10–20%. The measurement is nearly uninformative — it rules out a
*large* regression and a *large* win, nothing finer. So C2 is not evidence the
change is performance-neutral in mechanism; it is evidence only that the harness
is too noisy here to see whatever real (small) effect exists. Combined with the
C3 refutation below, a TIE is actually the *expected* outcome of pipelining the
wrong sub-step, not of a 2% domain.

## C3 — "marker u16 path is only ~2% of body" — REFUTED

Source-verified the counter increment site, denominator, and dispatch:

1. **Increment (`gzip_chunk.rs:1489–1495`):**
   ```
   let flipped_clean = !block.contains_marker_bytes();
   if flipped_clean { BOOTSTRAP_POST_FLIP_U16_BYTES += sink_len - before_len; }
   ```
   It accumulates a block's output **only when the block ends with NO marker
   bytes** — i.e. it counts CLEAN/flipped bytes, the explicit "Design-B1 prize"
   (`chunk_fetcher.rs:971`). It is **not** inverted in code (the prior naming-bug
   worry is unfounded) — but it counts the *clean* sliver, the opposite of "the
   marker path."

2. **Denominator (`chunk_fetcher.rs:948`):** `postflip_pct = post_flip /
   BOOTSTRAP_BODY_BYTES`. `BOOTSTRAP_BODY_BYTES` (`gzip_chunk.rs:1473`) sums every
   bootstrap block's output. With `post_flip = 1,425,448` and `pct = 2.0%`,
   `bs_b_bytes ≈ 71.3M ≈ the whole decompressed file` — so on the measured run the
   u16 bootstrap engine decoded ~all the bytes, and only **2% of them were in
   flipped-clean blocks.** That means ~**98%** of the engine's output was produced
   while `contains_marker_bytes == true`.

3. **Dispatch (`marker_inflate.rs:1252`):** the new fast loop runs under
   `read_internal_compressed_specialized::<true>`, taken iff `contains_marker_bytes
   == true` at read entry, and the loop body is gated `if CONTAINS_MARKERS`
   (`:1697`). Therefore the fast loop's domain = the **pre-flip** bytes =
   `bs_b_bytes − post_flip ≈ 98%` of the bootstrap body — **not** the 2%
   `post_flip` slice.

**Conclusion:** the claim "the marker loop port has a ~2%-of-body ceiling" is the
inverse of the data. 2% is the CLEAN portion the loop does **not** touch; the
loop touches the marker complement, which on the measured run is essentially the
whole decode body. The brief's own cross-check corroborates this: the charter's
"34.5% replaced markers" (a marker-heavy workload) is wildly inconsistent with a
2% marker-decode share but entirely consistent with a ~98% marker-decode domain.

**So why the TIE, really?** Not a tiny domain. The most likely mechanism (not
measured here, flag for the implementer): the fast loop pipelines only the
*literal* decode + speculative store; the marker-mode hot cost is the O(length)
**backward marker scan** in `emit_backref_ring::<true>`
(`marker_inflate.rs:3006–3027`), which the fast loop calls **unchanged**. The
`distance_marker >= distance` skip (`:3002`) already elides that scan once a chunk
has a window of clean history, and the RLE/word-copy arms were already shared
with the clean path — so the literal pipelining has little left to win. That is a
*mechanism* for the TIE (charter rule 7a/7b compliant), and it is the opposite of
"only 2% is reachable."

## Recommendation on disposition

The change is byte-exact and faithful (C1 UPHELD) and correctly KEPT per charter
rule 7a. But **strike the C3 rationale from the commit/charter record** — it
mis-attributes the TIE to a 2% domain that the counter does not support, and that
false floor could license abandoning the marker loop as "inherently 2%-bounded"
when in fact it owns ~98% of the bootstrap body. The honest open question is why
a 98%-domain pipelining change is wall-neutral; the answer to pursue is the
unchanged backward marker scan, not a phantom 2% ceiling.
