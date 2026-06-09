# VAR_VIII full-kernel asm — independent Opus disproof gate (the owed advisor pass)

Read-only. Advisor = independent Opus. Branch `bench/var8-fullkernel`, worktree HEAD
`d56cb0f5`, bench built at pushed ancestor `7bf26096` (ancestry CONFIRMED first-hand:
`git merge-base --is-ancestor 7bf26096 d56cb0f5` → yes; `Bits` 0-line diff between them;
the standalone bench never calls the production `decode_clean_into_contig`, so the
`marker_inflate` delta is irrelevant). Verified first-hand this turn: the VAR_VIII kernel
(`engine_isolation.rs:1554-1919`), the D-copy asm (`:1774-1802`), `avx_backref_copy`
(`:412-456`), the byte-exact + interleaved harness (`:2001-2107`), `decode_var_iii`
(`:197-200`), constants (ITERS=11, NV=11, REQUESTED_N=4MiB). No code edited. No box access.

---

## CLAIM 1 — RESULT SOUND (0.667× / +14.6%-over-VAR_VI) → **SOUND**

- **Byte-exact: REAL.** The gate (`:2030`) is strict and triple-anchored: each variant must
  be `len >= n_actual` AND byte-equal to VAR_I scalar over `[0,n_actual)` AND scalar must
  itself equal VAR_III ISA-L. `SHA_ALL_EQUAL=yes` on all 5 ⇒ VAR_VIII == scalar == ISA-L FFI.
  Self-test iii/i=2.79 ∈ [2.5,3.6] ⇒ instrument valid (and 282/2.79≈101 reproduces the
  banked scalar anchor 104–118 — internally consistent).
- **asm_frac: REAL and honestly accounted.** `asm_bytes` is the net `out_pos` advance *inside*
  the asm (speculative store advances by `sym_count`, not the 8 physical bytes — `:1730`;
  back-ref by `length`); `tail_bytes` is the Rust resolution of long-dist / long-litlen /
  boundary exits. 0.938 median with 3.4–4.3K reentries vs VAR_VII's 372–446K (~100×) ⇒ the
  DIS-1 per-symbol-spill confound (asm carried 1–30 %, rate FELL with coverage) is decisively
  refuted; the kernel carries 93–99.9 % of bytes. The 2/5 chunks under 0.97 are long-DIST-heavy
  (long dist → Rust by design), not a careful-tail measurement artifact.
- **Rate measured cleanly.** Interleaved best-of-11, median-of-per-chunk-medians, 4 MiB chunks
  (per-call alloc + 32 KiB window-copy amortized to noise). **Critically, VAR_VI and VAR_VIII
  share byte-identical buffer setup** (same `vec![0u8; cap]`, same window prepend, same
  `out_ptr`), so the VAR_VI↔VAR_VIII delta is overhead-identical — the +14.6% is a pure
  engine delta, sign-stable on all 5 chunks (+14.6/16.9/26.0/23.6/25.7 %) beyond the 3–9 %
  spread.
- **The copy-confound — DISPROVED first-hand (the supervisor's key worry).** I read both copies.
  VAR_VIII's asm D-copy (`:1782-1791`) is the **same** AVX2 32-byte `vmovdqu` loop as VAR_VI's
  `avx_backref_copy` non-overlap path (`:424-435`), same slop-licensed overshoot. On the
  OVERLAP/RLE path VAR_VIII is **strictly SLOWER**: it byte-copies ALL overlaps incl. dist==1
  and dist≥16 (`:1795-1801`), whereas VAR_VI uses `write_bytes` RLE for dist==1 (`:418-420`)
  and SSE-16 for dist≥16 (`:440-449`). So the copy is **matched-or-conservative**, never faster
  → the +14.6% is NOT a copy artifact; it is attributable to exactly the isolated variables vs
  VAR_VI: F2 (dist gather in asm), D's back-edge staying *inside* the register-pinned loop, and
  F4 (no mid-block bit-state/pointer spill). The owner's "conservative — UNDER-states VAR_VIII"
  caveat is FAIR and verified; if anything the true clean-engine delta is larger.
- **One minor, non-material caveat:** VAR_III (ISA-L) is the FFI path
  (`decompress_deflate_from_bit`) with a different internal buffer model, so the absolute 0.667×
  carries a small per-call-overhead asymmetry vs the manual flat buffer. At 4 MiB chunks this is
  amortized to noise and is common-mode; it does not move the headline. The **+14.6%** (the
  load-bearing isolation result) is entirely overhead-symmetric and robust.

Verdict: byte-exact PASS, asm_frac purpose met overwhelmingly (median 0.938; literal 0.97 met
on 2/5, missed only on long-dist chunks where the miss is BY DESIGN), ratio 0.667 FAIL of the
0.85 bar, KILL ("≈ VAR_VI") REFUTED. **The result is SOUND.**

## CLAIM 2 — INTEGRATION DECISION (HOLD vs salvage the +14.6%) → **SOUND (hold), with one nuance**

HOLD on production integration is correct:

- **The change fails its own pre-registered ≥0.85 gate (0.667).** Respecting a pre-registered
  falsifier — not moving the bar post-hoc to bank a sub-incumbent win — is the discipline the
  whole campaign is built on. Integrating now would be a production change authorized by an
  isolation number alone, the exact decompose-loop failure mode CLAUDE.md rule 6/8 forbids.
- **Rule 7a does NOT mandate integrating this.** 7a is a *no-revert-on-TIE* rule for a byte-exact
  change ALREADY ON the production path that ties; it is not a mandate to stand up a brand-new
  production path that LOSES to its incumbent. The incumbent matters per build:
  - **gzippy-isal build:** clean chunks already run REAL ISA-L (≈282 MB/s, 1.0×). VAR_VIII at
    0.667× would be a **regression** there — never route isal chunks to it.
  - **gzippy-native (no-C-FFI) build:** the clean tail is VAR_VI-class pure Rust (~0.582×).
    Here VAR_VIII (~0.667×) is a genuine **+14.6% byte-exact engine improvement** that advances
    the user's explicit no-C-FFI / "steal ISA-L in pure Rust" sub-goal. So the win is real, but
    only for the native build.
- **Cost vs benefit of integrating into native:** cost is large and permanent — hand-written
  inline asm with callee-saved GPR (rbx/r12–r15) save/restore, `nostack` dropped, the §3.1
  register-pressure spill risk, x86_64-only, a permanent obligation to keep the asm byte-identical
  to the portable Rust loop (differential test), AND a real production divergence: moving the
  clean-tail dist decode off `DistanceShortBitsCached` onto the ISA-L small LUT (the §1 wrinkle).
  Benefit is a +14.6% on the **clean-tail RATE only** — a fraction of the wall, sub-ISA-L so it
  can never reach the LEV-1 ceiling, and it changes **no BAR-1 cell verdict** (native still fails
  ≥0.99 at every T). necessary-not-sufficient.

**Nuance (the salvage path must be explicit):** the +14.6% should be **bench-banked as the proven
in-process pure-Rust+asm clean-tail ceiling (~0.667× ISA-L)**, not integrated now. It is
salvageable ONLY through the verdict's own Stage-1/Stage-2 gate — i.e. wire it into the native
build's `decode_clean_into_contig` behind the byte-exact suite, then a **whole-system causal
perturbation** (does native T4 wall move toward the 0.900× LEV-1 ceiling, frozen box, interleaved,
sha-verified, `isal_chunks==0` asserted) decides KEEP/REVERT. Do not let "winning where we can win"
become a free layer-in ahead of that perturbation. Disposition: **HOLD now; salvage native-only,
Stage-1 gated.**

## CLAIM 3 — THE BAR-1-IS-STRUCTURAL REFRAME → **SOUND conclusion, FIX-NEEDED on one framing word**

The operational conclusion is correct and well-supported:

- **The load-bearing step does NOT depend on the 0.667× ceiling.** Even granting native a PERFECT
  engine (= ISA-L = gzippy-isal, the most generous possible), **gzippy-isal — which already runs
  real ISA-L on its clean chunks — itself fails BAR-1 at T1 (0.899×) and T4 (0.906× measured this
  campaign, frozen guest N=11, real-ISA-L binary isal_chunks=14, sha-OK; disproof-ledger OPEN-1).**
  And OPEN-1's falsifier showed rg-grade placement does NOT close T4 (placement TIE-to-−34ms). So
  engine-perfect AND placement-perfect still loses low-T ⇒ the low-T residual is neither the
  clean-tail engine nor schedulable placement. That chain is sound and disproof-survived.
- **Therefore VAR_VIII is correctly NOT the BAR-1 close, and native ≤ gzippy-isal at every T**
  (native is *slower* on clean chunks: 0.667× < 1.0×). So "gzippy-native pure-Rust no-C-FFI at
  ≥0.99× every T" requires closing TWO walls at once: (a) the clean-tail engine 0.667→1.0 gap
  (plateaus sub-ISA-L in-process), AND (b) the low-T structural residual that even gzippy-isal
  can't close. **"Likely UNREACHABLE" is a SOUND finding** — at the right strength (*likely*,
  not *proven*): neither wall yet has a removal-oracle proof of a hard ≥0.99-blocking floor.

**FIX-NEEDED — one framing word, "NOT an engine problem," is imprecise and risks a wrong next step.**
OPEN-1's own mechanism attributes the T4 residual to the **u16-marker bootstrap engine on the
window-absent prefix** ("rg's clean tail vs gzippy's u16 marker compute on the prefix … the
inner-loop asm question, NOT a schedulable placement gap"). Reconcile carefully:

- It is **not a CLEAN-TAIL-engine problem** — the thing VAR_VIII accelerates — which is exactly
  why VAR_VIII can't close it. ✔ (the reframe's true point).
- But it **is** partly an *engine* problem on a **different engine**: the window-absent u16
  marker-resolution prefix, whose compute is slower than rg's clean tail AND whose position
  (must run before windows arrive) is structural. So the binder is the **u16-marker bootstrap
  (structural position + its own slow compute) + the serial-output floor + chunk-0 bootstrap** —
  a hybrid, not a clean "structural not engine" dichotomy. Stating it as flatly "NOT an engine
  problem" could misdirect effort toward scheduling/placement (already falsified) instead of the
  marker-prefix compute or a structural prefix-elimination (rg's window-map/marker-resolve port).

**Next productive direction — endorsed, with the missing measurement named.** SIZE the isal low-T
structural residual with a **removal oracle**, not attribution: it is the campaign's own
self-described "one unsized low-T term." Concretely — an oracle that removes (i) the serial-output
floor, (ii) the chunk-0 bootstrap, and (iii) the window-absent u16-marker prefix compute,
each independently, measured interleaved + sha-verified on the frozen box vs rg at T1/T4. Only a
removal oracle can convert "likely unreachable" into "proven floor of magnitude X" (or refute it).
Until that oracle runs, "native ≥0.99 every T is unreachable" is the *leading hypothesis*, the
correct finding to surface to the user — but it is not yet a removal-proved floor, and it should be
reported as such.

---

## BOTTOM LINE

- **Claim 1 (result): SOUND.** Byte-exact real; asm_frac real and honest; +14.6% over VAR_VI is a
  clean, copy-confound-free, overhead-symmetric, sign-stable engine win; 0.667× ISA-L is sound
  (small non-material FFI-overhead asymmetry vs ISA-L only). KILL refuted; 0.85 bar genuinely FAILED.
- **Claim 2 (integration): HOLD is correct — SOUND, with the salvage path made explicit.**
  Bench-bank the +14.6% as the proven in-process pure-Rust+asm clean-tail ceiling (~0.667×). Do NOT
  integrate now (fails its own ≥0.85 gate; isolation number alone). It is a regression for the
  gzippy-isal build and a real-but-small, costly partial for the native build. Salvageable ONLY
  native-only, behind the Stage-1 byte-exact suite + a Stage-2 whole-system causal perturbation
  toward the 0.900× LEV-1 T4 ceiling; not a free rule-7a layer-in.
- **Claim 3 (BAR-1 reframe): conclusion SOUND, framing FIX-NEEDED.** "VAR_VIII is engine-only
  catch-up that plateaus sub-ISA-L; even a perfect engine loses low-T; native ≥0.99-every-T is
  likely unreachable; next = size the isal structural residual" — all correct. Drop the flat "NOT
  an engine problem": the low-T binder is the **u16-marker bootstrap (structural position + slow
  compute) + serial-output floor + chunk-0 bootstrap**, a hybrid. The unreachability is the leading
  hypothesis pending a **removal oracle** on those three terms — surface it to the user as such, not
  as a removal-proved floor.

Overall: the supervisor's HOLD + bench-bank + "attack the isal structural residual next" posture is
the right call. The single correction is to not let the reframe read as "engine work is done /
irrelevant" — the *marker-prefix* engine (distinct from the clean-tail engine VAR_VIII addressed)
is still inside the unsized residual, and the removal oracle is the owed measurement before the
"unreachable" finding is banked as proven.
