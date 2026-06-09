# engine-W convergence gate — DIS-27 (E-core busy ⇒ (B) refuted ⇒ it's engine-W)

Independent disproof advisor, read-only, Opus. Source/ledger verified first-hand this turn:
PGR.hpp:575-628 (rg consumer in-order — confirmed), DIS-21 vendor re-derivation
(deflate.hpp:805/1319/1369-1390 — m_window16 IS a ring, confirmed by reading the ledger's
first-hand citations), DIS-18/DIS-19 instruction splits, DIS-20/21/24/25/26/27, DIS-22 single-shot.
No edits, no orphans, no box access (numbers taken from the ledger as the owner banked them).

## ONE-LINE VERDICT

The engine-W convergence is **BANKABLE with one FIX-NEEDED caveat and silesia-scope** — but it is
NOT primarily DIS-27 that earns it. The load-bearing kills happened in DIS-19 (rg marker-decodes
ALL streaming chunks, so the marker FRACTION is not a gz excess), **DIS-21** (the de-frag/flat-buffer
faithful lever is a PHANTOM — rg has the identical ring+drain+segments; the one real micro-convergence
TIES), and **DIS-25 P1** (gz partitions FEWER chunks than rg, killing over-partitioning). DIS-27 closes
the LAST faithful machinery candidate — E-core under-feeding (DIS-25 P3's own verdict) — by direct
worker-side measurement. With those, the faithful-convergence machinery levers are genuinely exhausted
and the residual is the marker-prefix Huffman SYMBOL RATE (asm-bounded, VAR_VIII 0.667× plateau,
user-gated). The caveat: DIS-27's "busy + high IPC ⇒ productive engine-W" does NOT separately rule out
fallback re-decode / speculative-discard as a (reducible, scheduling-side) slice of the 1.59× — high IPC
refutes STALL, not REDUNDANT-but-efficient work. Confirm isal_fallbacks + speculative-discard at
T16-Ephys (cheap) to make it airtight; DIS-6 + DIS-25-P1 make the likely outcome "confirms."

## Per-claim

### Q1 — "(B) refuted, it's engine-W"; busy≠productive → **SOUND on (B)-refutation; FIX-NEEDED on the busy=productive attribution**

- **(B) under-feeding REFUTED: SOUND.** A worker cannot be under-fed/idle and 72% busy at once. The
  E-cores being the BUSIEST resource (72% > gz's own P-cores at 67%) is a direct worker-side measurement
  that overturns DIS-25 P3's consumer-side roofline inference of (B). The instrument is Rule-4 validated
  (i7-13700T split cpu_core=P/cpu_atom=E PMUs; frozen-idle + T8-out-of-mask controls read ~idle). This is
  the right discriminator for the B-vs-b gate's owed Q1, and it answers it. SOUND.
- **"1.59× instructions = engine-W" is NOT airtight: FIX-NEEDED.** High IPC (2.34) refutes pipeline
  STALL / bandwidth bind. It does NOT refute REDUNDANT-but-efficient work: a chunk that is decoded,
  evicted, then COLD-RE-DECODED, or speculative u16-marker work later DISCARDED, executes real
  instructions at high IPC and inflates the 1.59× count without being irreducible first-pass engine-W.
  DIS-24 itself recorded isal_fallbacks 0→1→2 firing at high-T (each ~7.5× re-decode W, DIS-14). The
  gate must read isal_fallbacks AND the speculative-marker-discard fraction at the exact T16-Ephys cell.
  Mitigating evidence that it will confirm: DIS-25 P2's single-core per-byte RATE gap is only ~1.14×
  (flat across core types) — if gz were decoding 1.59× the BYTES it would show as a ~1.59× rate gap, so
  the count excess is dominated by heavier-per-byte marker work, not gross re-decode. But "dominated by"
  ≠ "zero," and a fb≥1 slice is a scheduling-side (faithful) term, not user-gated asm. **Read fb +
  discard at T16-Ephys before banking "the whole 1.59× is the asm engine."**
- Attribution of the per-byte half to the same DIS-18 marker engine: SOUND. DIS-18/DIS-19 located it
  on P-cores (T4); the E-cores run the same code, so the +49%/byte marker inner-loop is the same engine
  now executing on E-cores. No new mechanism needed for the per-byte part.

### Q2 — the 1.91× reconciliation; refuses a separate dispatch/feeding lever → **SOUND on the refusal; FIX-NEEDED on the amplifier's magnitude reconciliation**

- Refusing the **engine-W-under-BANDWIDTH-contention** sub-hypothesis (the B-vs-b gate's one surviving
  reconciliation reading): SOUND — IPC 2.34 (> rg 2.13) means not memory-stalled, so "heavier W ⇒ more
  traffic ⇒ worse scaling via bandwidth" is refuted. Good, clean.
- The "1.59× heavier W amplified through the in-order pipeline (slow heavy chunk head-of-line-stalls the
  consumer WHILE the worker is busy)" mechanism: internally plausible, BUT it is a CONSUMER-side stall,
  and **IPC measures the WORKER, not the consumer's wait** — the consumer can be head-of-line-blocked
  (idle) while the straggler worker runs at IPC 2.34. So IPC 2.34 does not by itself bound the
  amplifier. DIS-26 already SIZED exactly this consumer head-of-line/decode-wait amplification and
  bounded it at **≤10–23ms** (lean-consumer ceiling, grows with chunk count). The reconciliation is
  coherent ONLY if the claimed amplifier == that bounded ≤23ms term. The T16-Ephys loss is 311→333 ≈
  **22ms** (DIS-25 P3) — which is INSIDE the ≤23ms DIS-26 bound, so it reconciles. State that
  explicitly: the amplifier is the DIS-26-bounded consumer term driven by the heavier engine pole, not
  an unaccounted new quantity. (FIX-NEEDED = make this equality explicit; the numbers do close.)
- Refusing a SEPARATE dispatch/feeding lever: SOUND given the amplifier reduces to "lighter W shortens
  the pole" (Q3) and the feeding/partition alternatives are independently dead (DIS-25 P1, DIS-6).

### Q3 — the in-order amplifier; no faithful OOO/straggler-tolerance lever → **SOUND**

- rg's consumer is strictly in-order: **verified first-hand at ParallelGzipReader.hpp:575-628** — a
  `while ((nBytesDecoded<nBytesToRead)&&!eof())` loop calling `chunkFetcher().get(m_currentPosition)`,
  copying each chunk's bytes in order and advancing `m_currentPosition`. Out-of-order publish is
  UNFAITHFUL. SOUND.
- Faithful straggler-TOLERANCE rg has and gz lacks? Checked the obvious one: rg's BlockFetcher
  prefetch cache (2P) decodes chunks AHEAD on the pool, so a head-of-line straggler stalls EMISSION but
  not the workers (they prefetch ahead). gz has the byte-identical prefetch cache (chunk_fetcher.rs:520-529
  ↔ BlockFetcher.hpp:181-183, vendor-identical per DIS-25 / prefetch-horizon). And DIS-27's own datum —
  gz E-cores 72% busy, NOT idle — is the direct proof that the in-order emission stall is NOT idling gz's
  workers either. So there is no faithful straggler-tolerance gz is missing; in both tools the only way to
  shorten the pole is lighter per-chunk W. SOUND.
- Why rg extracts ~2× incremental from the same E-cores (DIS-25 P3) despite both being busy: rg's leaner
  W ⇒ a busy E-core finishes its chunk sooner ⇒ shorter pole ⇒ less emission stall. Coherent and reduces
  to the engine. SOUND.

### Q4 — full-curve convergence; only user-decisions + squishy remain → **SOUND-with-caveats (≈90% earned)**

What is genuinely converged (bankable), with the actual load-bearing kill cited:
- **de-frag / u16-output-fragmentation faithful lever: DEAD (DIS-21), mechanism-grade.** rg's m_window16
  is a 65536-u16 RING (deflate.hpp:805), resolveBackreference == emit_backref_ring (:1369-1390),
  appendToEquallySizedChunks == push_slice. gz is ALREADY a faithful structural port; "flat buffer" was
  a misread. The one real micro-divergence (word-copy vs memcpy) TIES (-3ms, cuts 1.5% instr, wall flat —
  textbook rule-3). This is the kill that retires DIS-19's "flatten/fuse the ring is the largest tractable
  sub-lever" — it was a phantom. Satisfies rule-7(a) (rejection has a mechanism: rg does it the same way).
- **marker FRACTION not a gz excess: DEAD (DIS-19).** rg marker-decodes EVERY streaming chunk via
  Block<false>::read + applyWindow; it does NOT u8-direct window-present chunks (that's only the
  random-access path). So "shrink gz's flip_to_clean fraction toward rg's bootstrap-only" (DIS-18
  convergence-target #1) is NOT a faithful convergence — rg has no smaller fraction. Correctly closed.
- **over-partitioning: DEAD (DIS-25 P1, direct).** gz partitions FEWER chunks than rg (34 vs 66 @ T24).
  Converging to rg's partitioning would mean MORE chunks, not fewer. The DIS-24 chunk-count-growth binder
  is refuted by measurement.
- **E-core under-feeding (B): REFUTED (DIS-27).** The last faithful work-distribution candidate.
- **lean-consumer / publish-off-consumer: BOUNDED ≤23ms (DIS-26).** Real, faithful, sub-dominant.
- **prefetch-depth (vendor-identical), out-of-order publish (rg in-order), output writev (shared floor,
  DIS-5): dead/unfaithful/shared.**
- ⇒ machinery is exhausted as a faithful lever across far more than "3×" — the residual is the
  **marker-prefix Huffman SYMBOL RATE** (~1.61e9, asm-bounded, VAR_VIII 0.667× plateau, user-gated). This
  is engine-W and it is the dominant single-core deficit (1.14×/byte, uniform across core types, DIS-25 P2).

Caveats on "only user decisions + squishy remain":
- **(a) The var8/per-chunk-T1 oracle is MOOT for gzippy-ISAL production only — NOT for native.** Production
  T1 routes to IsalSingleShot (DIS-22, 1.200× WIN), which bypasses ParallelSM, so the 124ms@T1 per-chunk
  term is off the ISAL production path. BUT gzippy-NATIVE has no ISA-L single-shot — it stays ParallelSM at
  T1 and loses 0.608× (STATE.md scorecard). So "moot" is build-specific. If the goal is "ParallelSM ties rg
  at every T" the T1-ParallelSM term is DODGED for isal, not closed, and is fully live for native. State it
  as moot-for-isal-production, not moot-globally.
- **(b) The fb/speculative-discard read at T16-Ephys (Q1) is the one cheap MEASUREMENT still owed** before
  the "1.59× = irreducible asm engine" attribution is airtight. It ranks ahead of squishy and ahead of the
  user decision. Likely confirms (DIS-6 closed offset-supply 3×; DIS-25 P1 shows gz under-partitions; the
  per-byte rate gap is only 1.14×), but it is not yet read at THIS cell.
- **(c) squishy corpus-generality** is owed for SCOPE (silesia is rg's tuning corpus — DIS-24/25 Q5). Tests
  generality, not the B-vs-b root; ranks after (b).
- **(d) USER decisions:** fund the full-kernel native clean-tail asm (VAR_VIII proven 0.667× ISA-L, FAILS
  its own 0.85 bar; even a perfect engine = real ISA-L still loses T4 0.90× per LEV-1, so asm is
  NECESSARY-not-SUFFICIENT) vs accept the loss vs move BAR-1. This is correctly the user's call.

Is there an un-refuted FAITHFUL lever? After this turn's verification: **no** — the candidate I went
hunting for (DIS-19's "flatten/fuse the u16 ring into rg's Block::read") is the DIS-21 phantom, and the
feeding/partition/fraction candidates are each independently dead. The only residual that is NOT a user
decision is the cheap fb/discard confirmation (b), which is a measurement, not a lever.

## Bottom line

- **Is the engine-W convergence BANKABLE?** YES, silesia-scoped, with ONE cheap measurement owed. It is
  earned — but credit it to the FULL chain (DIS-19 marker-fraction, DIS-21 de-frag-phantom, DIS-25-P1
  under-partition, DIS-26 ≤23ms bound, DIS-27 E-core-busy), not DIS-27 alone. DIS-27's specific job —
  refute (B) E-core under-feeding by direct worker measurement — is SOUND and it is the last faithful
  machinery candidate closed.
- **Is the machinery genuinely exhausted as a lever?** YES — every faithful work-distribution / pipeline /
  output candidate now has a mechanism-grade rejection (rule-7a), not a TIE. De-frag is the strongest
  (rg does it byte-structurally the same).
- **Are the remaining moves user-decisions + squishy?** ALMOST — ≈90%. One cheap measurement stands in
  front of "only user decisions remain": read **isal_fallbacks + speculative-marker-discard fraction at
  the T16-Ephys cell** to convert DIS-27's "busy ⇒ productive engine-W" from a strong inference into an
  airtight one (high IPC refutes stall, not redundant-efficient re-decode). Then squishy for scope. Then
  the asm/bar decision is the user's. And note the var8-T1 term is moot for ISAL-production but live for
  gzippy-native T1 (0.608×, can't single-shot).
