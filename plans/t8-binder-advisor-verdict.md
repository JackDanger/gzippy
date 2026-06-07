# DISPROOF VERDICT — is the T8 binder per-thread decode-compute, or the scheduling/serial term?

**VERDICT: UPHELD-WITH-CAVEATS.**

The charter's strong claim — *"constant ~1.70× = pure per-thread decode gap; the inner Huffman
kernel is the ONLY lever to 1.0× at T8"* — is **REFUTED at T8**. The brief's qualitative claim (T8
wall is dominated by a scheduling/serial/overlap term that overlaps decode; the kernel has a low T8
ceiling) is **UPHELD and independently corroborated**. But the brief's *precise* numeric ceiling
(≤18–22%) is **NOT robustly established** — two confounds push the true kernel ceiling higher
(plausibly ~30–45%), though still not the dominant binder. Hence WITH-CAVEATS, not clean UPHELD.

## Strongest disproof attempt (Angle 4, the one that bit)

**The slow_knob fires CLEAN-mode ONLY, but the charter itself says ~89% of chunks decode in
MARKER-mode at T≥2.** Verified first-hand: `marker_inflate.rs:1447` and `:2083` const-fold
`slow_spin`→0 when `CONTAINS_MARKERS`, so the injection (38.7M hits "∝ clean decoded bytes") never
touches the u16 marker path. At T1 the window is present so decode is ~all clean → +100% inject adds
+83% (knob covers ~all decode). At T8 the clean fraction SHRINKS (markers dominate until
apply_window resolves), so a clean-only injection covers a SMALLER slice of decode work. Part of the
4× T1:T8 attenuation is therefore **instrument-coverage shrinkage, not parallel-overlap**. If clean
is ~half of T8 decode bytes, a perfectly-critical decode could read as ~20% when it is really ~40%.
This is the strongest break: it inflates the apparent attenuation and makes the ≤20% ceiling
unsafe. It does **not**, however, rescue the charter — even a doubled ~40% ceiling leaves >50% of
the T8 wall as the scheduling/serial term.

**Second caveat (Angle 3, Rule 3):** the inference "low slow-down slope ⇒ low speed-up ceiling"
is exactly the move CLAUDE.md Rule 3 forbids. Slowing decode only worsens the consumer-lag
cascade monotonically; it can never reveal the regime where a *faster* decode lets the consumer keep
pace and collapses the cold-re-decode/eviction cascade nonlinearly (memory
`project_confirmed_offset_prefetch_gap`: consumer lags its own prefetcher ~318ms → eviction → 4/4
cold re-decodes = 73% of wall). A faster kernel could UNBIND that cascade and pay more than the
slope predicts. So the ceiling is a lower-slope-derived *floor on what's hidden*, not a proven cap.

## Why the qualitative claim survives anyway (the angles that did NOT break it)

- **Angle 1 (spare-core artifact): FAILS to break.** Process affinity (CPUS) pins to 8 cores;
  worker threads inherit it, so injected spin cannot escape to the 8 idle siblings. The `sleep`
  control descheduled the core and gave an EQUAL/LARGER rise (+20% ≥ spin +17%) — the opposite of a
  spare-core escape. The attenuation is genuine parallel overlap, not a turbo/idle-core artifact.
- **Angle 2 (VAR_V disable bias): FAILS to break, and cuts the RIGHT way.** Confirmed at
  `marker_inflate.rs:1478` (`!CONTAINS_MARKERS && slow_spin == 0 && !slow_yield`): any injection
  disables VAR_V and runs the careful loop, so spin100 conflates +inject with a VAR_V→careful
  penalty. That penalty is per-decode-compute and present at BOTH T, so it inflates the measured
  slope — making the true pure-inject slope EVEN LOWER, i.e. a LOWER ceiling. Strengthens the claim.
- **Angle 5 (spread/TIE): FAILS to break.** The 4× contrast is reproduced 3× with monotone
  OFF<spin100<spin200, spin200→+45% well outside spread, and T1 +83% far outside its ~6% spread.
  Moreover the wide T8 spread (18–58%) is itself a fingerprint of a SCHEDULING-dominated wall
  (run-to-run head-of-line/eviction jitter); a compute-bound wall would be tight like T1's 6%.
  This corroborates rather than undermines.
- **Independent corroboration (not from this instrument):** the first-hand stall decomposition in
  `project_confirmed_offset_prefetch_gap` (consumer-block 737ms = 73% of T8 wall, 4/4 waits FRESH
  COLD) and rapidgzip's own trace (11/12 waits join an IN-FLIGHT decode, ties the wall *despite the
  same 2.1× engine busy gap* via OVERLAP) prove the binder is scheduling/overlap, and that the wall
  can be tied WITHOUT a faster kernel. The `chunk_fetcher.rs` consumer loop confirms the structure:
  in-order `wait.block_fetcher_get` cold-get stall (`:1419-1469`) and the serial window-publish
  chain (`:1535-1740`) are on the critical path, with post-process overlapped onto the pool.
- **Reconciliation (Angle 4 narrative):** no instrument is impugned. The prior round concluded
  "compute binds at both T" by ELIMINATION (traffic/placement slack ⇒ residual = compute) — the
  attribution trap CLAUDE.md warns of. The slow_knob is a DIRECT perturbation of compute and
  outranks an elimination inference. The charter's own CURRENT STATE already concedes decode is
  "ABSORBED — the wall didn't move" and names production-overhead/marker-mode as the live binder; the
  stale headline ("pure decode gap, kernel only lever") is what's wrong, and it contradicts the
  charter's own body.

## Recommended next step — YES, re-point the campaign

1. **Demote the headline, not the kernel-as-T1-lever.** Strike "inner kernel is the ONLY lever to
   1.0× at T8" from the charter. Keep the kernel as a confirmed T1 lever (+83%). Align the headline
   with the charter's own CURRENT STATE.
2. **Settle the ceiling with an ORACLE, not the slope (Rule 3).** Before any kernel work-stretch,
   bound the T8 kernel ceiling by REMOVING decode (zero/near-zero-cost oracle) and measuring the
   interleaved wall — this resolves BOTH the Rule-3 cascade-unbinding risk and the clean-only
   coverage confound the slope cannot. A clean-only knob cannot bound a marker-heavy T8.
3. **Pursue the scheduling/overlap binder as primary at T8.** The pre-registered next measurement
   already owed in `project_confirmed_offset_prefetch_gap` — decompose consumer per-chunk time into
   SERIAL-WORK vs DECODE-WAIT — is the correct next experiment. Fix the cold-re-decode/eviction
   cascade so gzippy's consumer JOINS in-flight decodes (rapidgzip 11/12) instead of cold-starting
   them (gzippy 4/4), citing the rapidgzip file:line it makes gzippy match.
