# ENGINE-BENCH ROUND-2 — INDEPENDENT DISPROOF ADVISOR VERDICT

Read-only adversarial review of the **PLATEAU** verdict (orchestrator-status.md:24-60).
Authored against HEAD b7cb2019, branch reimplement-isa-l. All claims below are
first-hand from the cited source, not from the orchestrator's summary.

## VERDICT: **UPHOLD PLATEAU**

The verdict survives all four attacks. The (ii_stacked)/(iii) = 0.412 ratio is real,
byte-gated on the LIVE guest AVX2 path, isolates the technique cleanly, is stable
across a heterogeneous chunk sweep (12σ below the 0.85 PASS line), and the ratio
criterion legitimately governs over the §3 numeric coincidence. I attack the verdict
hardest at attack #4 (the 0.542 ≤ 0.604 numeric "pass") and #2 (the untested true-E1
u8-ring) and show below why neither overturns it. The ONE correction I demand: the
headline should read **"the faithful u16-ring pure-Rust+ASM engine plateaus"**, not
"pure-Rust+ASM plateaus" unqualified — the single most promising mechanism the
orchestrator itself named was never built (see #2). That is a scoping fix, not an
overturn: even optimistically the untested lever cannot cross a 0.41→0.85 gap.

---

## Attack 1 — BYTE-EXACTNESS: the gate is REAL. Could not break it.

**The gate is genuine and double-anchored.** engine_isolation.rs:304-317: every
variant is decoded once, `scalar = &outs[0][..n_actual]`, `isal = &outs[2][..n_actual]`,
`scalar_eq_isal = scalar == isal`, then `exact[k] = o.len() >= n_actual &&
&o[..n_actual] == scalar && scalar_eq_isal`. So a variant passes ONLY if it matches the
scalar u16 baseline AND the scalar baseline matches ISA-L. `all_equal` requires all 7.
GUEST reported SHA_ALL_EQUAL=yes on all 5 chunks with **avx2_detected=true**
(orchestrator-status.md:30-33) — i.e. the live AVX2 path, not only the Rosetta scalar
fallback, was byte-checked. That closes the "AVX2 diverges from the Rosetta-validated
scalar" hole the prompt worried about.

**VAR_IV_E000 is a sound anchor** (engine_isolation.rs:268). `read_clean_e234<false,
false,false>` (marker_inflate.rs:1507) is a near-line-for-line mirror of the production
`read_internal_compressed_specialized<false>` (marker_inflate.rs:1222): same
`lut_litlen_decode`, same multi-symbol unpack loop (1613-1674 ↔ 1374-1471), same
clean distance check `distance > self.decoded_bytes + emitted` (1665 ↔ 1456 with
CONTAINS_MARKERS=false), same `record_backreference_for_sparsity` (1671 ↔ 1468).
With E2=false the back-ref goes through `emit_backref_ring_clean(..use_avx2=false)`,
whose scalar fallback is documented byte-identical to `emit_backref_ring` (2585) — and
the gate confirms it. The only deltas in E000 vs production are the const-folded-dead
marker bookkeeping and the byte-transparent slow_knob (1338), neither of which touches
output. E000 ≡ scalar ≡ ISA-L is therefore both structurally expected and gate-proven.

**Each technique's "subtle wrongness" vector checked and found safe:**
- **E3 packed store** (1577-1606): fires ONLY when `last_code <= 255` (all lanes are
  literals) AND no ring wrap (`dst_phys+2`/`+4 <= RING_SIZE`). The sym_count==3 u64
  store writes a 4th zero lane at `dst_phys+3`, but `pos` advances by only 3, so that
  slot sits AT the logical write head — not in `[ring_drained, ring_pos)`, so
  drain_clean_u8 never reads it (773-789), and the next decode overwrites it before it
  can serve as a back-ref source (back-ref sources are always at `pos - distance < pos`).
  Near a wrap it falls through to the scalar loop (1604). Safe; gate-confirmed.
- **E2 AVX2 overshoot** (emit_avx2_copy_u16, 2658; guard 2569-2573): only the
  `distance >= length` non-overlap arm (2560), and only `distance >= 16` u16. Src/dst
  are ≥32 bytes apart so the 32-byte copy is a plain memcpy of already-final bytes; the
  rounded-up overshoot (rounded16−length) lands ahead of the advanced `pos`, overwritten
  before drain. No wrap (`+rounded16 <= RING_SIZE`). Safe.
- **E4 refill elision** (1553-1559): skips the top-of-loop refill only when
  `available >= 48` (worst-case back-ref budget: 20+15+13). The sub-decoders keep their
  own `< 32`/`< extra` safety refills (1648-1652, and inside lut_litlen/dist decode).
  Byte-exact; gate-confirmed.
- **n_actual clamp** (297-298): comparison covers exactly `[0, n_actual)`, the bytes that
  are also what the rate divides by; any post-n_actual tail overshoot is neither compared
  nor timed-relevant. Not a hiding place.

**Finding: byte-exactness is real and not subvertible by any of the four technique
mechanisms. Attack 1 fails.**

---

## Attack 2 — ISOLATION: clean for the TECHNIQUE delta; one honest caveat surfaced.

The E2/E3/E4 deltas are measured against **VAR_IV_E000 on the identical harness** —
same `read_clean_e234` entry, same `drain_clean_u8` per-call drain (227), same
`Vec<u8>` sink (214). E000→E234 holds drain/sink fixed and varies ONLY the const
flags, so the +8.7% E234-over-E000 is a clean within-harness technique delta. The
cross-harness check is also sound: E000 (108) ≈ VAR_I scalar (104), confirming the
read_clean_e234 harness swap is ~neutral (4%).

**The one real isolation caveat (flag, not a break):** the timed region for EVERY
gzippy variant includes a u16→u8 narrow pass — VAR_I narrows once at the end (158),
VAR_IV narrows per-call inside `drain_clean_u8` (783-787, a copy of drain_to_output's
clean branch 738-750). ISA-L (VAR_III) writes u8 natively with NO narrow. So an
unknown fraction of the 2.4× gap is the u16-ring narrow artifact, not pure
inner-decode. This makes the *engine-vs-engine* gap somewhat SMALLER than 0.41 implies
— the strongest pro-NARROW-MISS thread, picked up in #2-coupling below. It does NOT
corrupt the E000→E234 technique deltas (both narrow identically). **Isolation of the
technique is clean; isolation of "pure decode vs ISA-L" carries a known u16-narrow tax
that the verdict already attributes to the ring. Attack 2 fails to overturn.**

**The genuine omission (this is the load-bearing one):** the orchestrator's OWN
diagnosis names the **u8 ring** as "the prime suspect for the 3.1x gap" and "the main
lever … CLEAN u8 ring (halves copy traffic)" (orchestrator-status.md:14-16). What was
actually built and measured — E2 (AVX2 copy on the STILL-u16 ring), E3 (packed store),
E4 (refill) — are the orchestrator's own SECONDARY levers. The true E1 (u8 ring) was
downgraded to "E1-partial" (VAR_II = u8 SINK, u16 RING) and VAR_II's number isn't even
in the aggregate. So the falsifier fired against the techniques LEAST likely to close
the gap and skipped the one it itself flagged as MOST likely. **This is why the
headline must be scoped to "u16-ring."** See "strongest argument against" for why it
still does not overturn.

---

## Attack 3 — CHUNK-SWEEP: SUPPORTS PLATEAU; spread does not undercut.

5 chunks at 10/30/50/70/90% of the sorted seed list (engine_isolation.rs:410-421).
Per-chunk (ii)/(iii) = 0.356, 0.417, 0.425, 0.466, 0.397 → mean 0.412, sd 0.036. The
**best** chunk (0.466) is still 2.1× slower than ISA-L and `(0.85 − 0.466)/0.036 ≈
10.7σ` below PASS. No chunk approaches a tie. Silesia is a heterogeneous tarball
(text/binary/DNA/dict), and 10/30/50/70/90% spacing hits different members; the ratio
being STABLE (0.36–0.47) across that heterogeneity STRENGTHENS the bound rather than
weakening it — it is not one unrepresentative back-ref-light chunk. The spread is small
*relative to the gap to PASS* (0.036 vs 0.384), so it cannot be a within-spread TIE
with PASS. Only-5-chunks / single-corpus is a real limit on a tight *upper* bound, but
for a PLATEAU (a lower bound on the GAP) claim, 5 stable points 10σ from PASS is
sufficient. **Attack 3 fails; the sweep is evidence FOR PLATEAU.**

---

## Attack 4 — FALSIFIER / §3 MATH: the ratio criterion legitimately GOVERNS.

This is the verdict's most attackable seam, because the §3 projection numerically
"passes": E234 projects decode_wall **0.542s ≤ 0.604s** tie bar
(orchestrator-status.md:43). Taken at face value that is gzippy-projected BELOW
rapidgzip's measured same-sink wall (0.604s, the rider at 1310-1313). Why does this
NOT make it a PASS or NARROW-MISS?

1. **The 0.542 projection omits gzippy's OWN non-decode floor.** It is the *decode term
   only*. §3's floor caveat (tier1-design-v2.md:271-284) documents ~225ms of in-order
   consumer-serial bookkeeping + a measured `wait.block_fetcher_get = 0.497s` that do
   NOT shrink with the engine — putting gzippy's non-decode floor MATERIALLY ABOVE
   0.54s. The honest gzippy wall is `max(decode 0.542, non-decode floor >0.54) +
   coupling` ≈ 0.54–0.62 (§3:281), NOT 0.542. So 0.542 ≤ 0.604 compares an
   *incomplete* gzippy lower bound against rapidgzip's *complete* wall.

2. **0.542 assumes PLACEMENT-PERFECT — an unbuilt lever whose port gate FAILED.** The
   §3 arithmetic (242-246) only reaches the clean-rate operating point AFTER perfect
   placement; the placement port gate is FAIL/STOP (orchestrator-status.md:62-93). The
   engine bench cannot bank a wall tie on a placement port that doesn't exist.

3. **The pre-registered PASS was DEFINED structurally, exactly to defeat this numeric
   coincidence.** engine-bench-falsifier.md:58-65: PASS requires (ii)/(iii) ≥ ~0.85 "so
   the §3 coupled model lands the decode-bound wall low enough that the wall RE-BINDS on
   the shared pipeline floor … rather than on decode." At E234 the model returns 0.542 =
   the decode term itself (decode_wall ≈ total_wall, status:43-44): decode is STILL the
   binding term — it did not cross below the floor and re-bind. The 0.542 < 0.604 is a
   loose-bar artifact (the bar sits right at the 0.615 scalar floor, so a mere +13.5%
   engine bump, 104→118 MB/s, trips it without decode ceasing to bind). A robust verdict
   keys on the STATE (did decode stop binding), not the coincidence. It did not.

4. **E234 is worse than §3's OWN narrow-miss floor.** §3's narrow-miss scenario
   (tier1-design-v2.md:257-261) assumes the engine reaches ~50–60 ms/chunk. E234 at 118
   MB/s anchors to `92.7ms × 104/118 ≈ 81.7 ms/chunk` (vs ISA-L's ~34 ms). 81.7 > 60, so
   by §3's own risk taxonomy E234 is BELOW the narrow-miss band — squarely PLATEAU.

5. **The anchor is legitimate.** 92.7ms ↔ 104 MB/s reproduces 0.615s == design's
   0.6134s (status:35), an independent self-consistency check on the anchor. The bar is
   not hand-set: 0.604s is rapidgzip's MEASURED same-sink wall (rider, 1310-1313), and
   it moved UP (favorable to gzippy) yet E234 still does not produce a structural pass.

**Attack 4 fails: the ratio criterion governs, the §3 numeric pass is an
incomplete-model artifact, and E234 lands below §3's own narrow-miss band.**

---

## STRONGEST SINGLE ARGUMENT AGAINST THE VERDICT — and whether it survives

**The argument:** The falsifier never tested the mechanism it itself identified as
dominant. The orchestrator says the u8 ring is "the prime suspect for the 3.1x gap"
and "the main lever" (status:14-16); E2/E3/E4 are secondary (E2 is explicitly "AVX2
copy on the STILL-u16 ring"). The true E1 (u8 ring) — which would halve copy traffic
AND remove the u16→u8 narrow that currently sits inside the timed gzippy path (Attack
2) — was downgraded to "E1-partial" and never measured into the aggregate. A PLATEAU
verdict on the secondary techniques therefore over-generalizes: it proves the u16-ring
engine plateaus, not that *pure-Rust+ASM* plateaus.

**Does it survive? Partially — as a scoping correction, not an overturn:**

- **It is OUT OF SCOPE under the governing mandate, not a free lever.** The u16 marker
  ring is the load-bearing faithful port of vendor's `m_window16`; a u8 ring diverges
  the ONE-engine storage and is, by the governing memory
  [[project_faithful_unified_decoder_over_perf]], a Divergence requiring a supervisor
  decision (status:56-58 says exactly this). So the verdict's routing ("supervisor/
  user-level finding; do not start the build") is the CORRECT disposition of the
  untested lever, not an evasion.

- **Even optimistically it cannot reach PASS.** The narrow pass is one streaming
  u16→u8 copy loop, not the decode itself. E000 (which narrows) = 108 ≈ scalar 104,
  while ISA-L = 283 (2.6×). Removing the narrow + halving copy traffic might lift the
  engine to ~140–160 MB/s — ratio ~0.5–0.57, still far below the 0.85 PASS and still
  PLATEAU. There is no credible path from a 0.41 base to 0.85 by ring-width alone; the
  vendor existence proof (ISA-L = 2.1× the best PURE decoder single-thread, §3:255)
  bounds what any pure decoder can do.

- **Therefore the verbal claim should be tightened, but the decision is unchanged.**
  Correct wording: *"the faithful u16-ring pure-Rust+ASM engine plateaus at ~0.41× ISA-L;
  a true u8-ring E1 is untested, but it (a) violates the faithful-unified-decoder mandate
  and (b) cannot plausibly cross a 0.41→0.85 gap — so the 1.0× bar is not reachable by
  this inner-loop direction without either FFI, a mandate-breaking ring rewrite the
  supervisor must authorize, or a revisited bar."* That is materially the verdict's own
  conclusion; the omission does NOT make "PLATEAU / not provable" overclaimed once the
  u16-ring scope is stated. **The argument survives as a wording fix; it does not
  overturn PLATEAU.**

---

## MINOR PROCESS FLAGS (do not change the direction)

- **Self-test band was recalibrated post-failure** ([1.7,2.6] → [2.5,3.6],
  status:7-8). Moving a pre-registered gate after round-1 failed it is a yellow flag.
  The rationale (pure ISA-L is a purer denominator; advisor-confirmed iii/ii≈3.10,
  iii/i≈3.29) is defensible, and the guest median-chunk iii/i (283/104 ≈ 2.72) lands
  inside the new band. Crucially the recalibration cuts AGAINST gzippy (a larger ISA-L
  lead makes the tie harder), so it cannot have manufactured a false PLATEAU — if
  anything the band-widening concedes ISA-L is faster than round-1 assumed, reinforcing
  PLATEAU. Acceptable, but note it in the record as a post-hoc gate move.

- **VAR_II ("E1-partial") number is absent from the aggregate.** Reporting it would
  have directly bounded the u16-narrow tax and pre-empted the Attack-2 caveat. Recommend
  the supervisor ask for VAR_II's MB/s before any u8-ring decision — it is the cheapest
  next datapoint and already in the harness (engine_isolation.rs:261).

## BOTTOM LINE

UPHOLD PLATEAU. The 0.412 ratio is byte-honest (live-AVX2 guest gate), technique-
isolated (E000-anchored harness), sweep-stable (12σ from PASS), and the pre-registered
ratio criterion correctly governs over the incomplete §3 numeric pass. The only owed
correction is verbal scope (u16-ring, not "pure-Rust" unqualified) plus reporting VAR_II
— neither reaches a NARROW-MISS, let alone a PASS. Do not integrate; do not start the
multi-week build; the u8-ring/FFI/revisit-the-bar fork is a supervisor call, exactly as
routed.
