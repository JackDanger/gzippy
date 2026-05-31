# Closing a diffuse performance gap, provably — design (v2, adversary-hardened)

> v1 claimed `wall ≡ Σ(critical-path spans)` as an *identity* and built a closure proof on it.
> An adversary broke that proof (§0). This v2 keeps the parts that survive and replaces the proof
> with a **two-law lower bound** that is honest about shared-resource contention, plus an
> algorithm-neutral resource taxonomy, a contention-sound elasticity method, a monotone closure
> loop, and an *admissible* fundamentality proof. The headline claim is downgraded from
> "provably close the ENTIRE gap" to the thing that is actually provable (§7). That is on purpose.

---

## §0 — The one assumption that most threatens "provable", and the verdict

**Threatening assumption (v1, lines 14–16, 21–23):**
> `wall ≡ Σ(spans on the critical path)` — "an identity, not a model."

**It is NOT an identity for this program. It is fixable but only by adding a second law.**

`wall = length of the longest wait-for chain` is exact **only for a pure dependency DAG**: a
program whose every stall is an explicit edge "task A waits for task B." A 16-thread DEFLATE
decoder is not that program. Its threads contend for **shared resources with no edge in any
wait-for graph**:

- **DRAM bandwidth** (one memory controller; 16 streams of window/marker/output traffic).
- **LLC capacity** (16 working sets evict each other — *this project already hit it*: pre-sizing
  buffers improved hit-rate but **regressed wall** via resident-set/TLB growth — falsification
  ledger, `rpmalloc/Z-prewarm/segmented` row).
- **SMT execution ports / fill buffers** (2 HW threads per physical core share the backend; the
  ledger's own TopdownL1 says **Backend-bound 31% co-limited with Bad-Spec 26% / Retiring 28%** —
  the textbook signature of a backend-/memory-stall regime, NOT a clean dependency chain).

When thread *i* stalls 5ns waiting for a DRAM line that thread *j*'s traffic delayed, **there is no
`wait_for(i, j)` edge** — neither thread "waits for" the other in any semantic the tracer can emit.
The stall is real, it is on the wall, and it is **invisible to a wait-for DAG**. Therefore:

> **The conservation gate of v1 can SILENTLY PASS while being WRONG, or FAIL with no available fix.**
> Two failure modes:
> 1. Contention stalls get **mis-attributed** to whatever span happened to be executing when the
>    stall landed (the tracer times wall-clock span duration, which *includes* the stall). Then
>    `Σ(spans) = wall` passes — but `Δ_c` for that category is **inflated by contention that a port
>    of that category cannot remove**. You "close" the category to rapidgzip's instruction count and
>    the wall does not move, because the bytes the category was *waiting on* still saturate DRAM.
>    **This is the "decode TIE / copies wall-neutral" corpse, re-derived from first principles** —
>    and it means v1's method would have produced the same dead-ends it was built to escape.
> 2. If contention manifests as idle gaps **between** spans (cores stalled with no span open),
>    `Σ(spans) < wall`, the gate FAILS, and v1 says "add the missing edge" — but **there is no edge
>    to add.** The method dead-ends with no prescribed next step.

**Verdict: NOT fatal, but fatal *to the single-law proof*.** The fix is a **second conservation
law** — a resource roofline — combined with the critical path by `max`, and a taxonomy that
attributes time to **resources consumed**, not **code regions occupied**, so contention can't hide
inside a span's wall-clock duration. Built in §2–§3.

A subtle but decisive consequence, stated up front because it reshapes the whole method:

> **`Σ(critical-path) = wall` is the WRONG validity gate.** The correct relationship is
> **`wall = max( CP_dep , CP_res )`** (§2). Demanding `Σ(spans on a wait-for chain) = wall ± ε`
> *presumes the program is dependency-bound* — the very thing in question. When the program is
> resource-bound, that equality is **false by construction** and v1 would loop forever "adding
> edges." The gate must instead check **`wall = max(two laws) ± ε`** and, on the resource branch,
> the closure target is **traffic/occupancy**, not span length.

---

## §1 — What survives from v1 (kept, because it's correct)

These three are sound and stay verbatim in spirit:

1. **The diagnosis of why single-lever A/B rabbit-holes** (v1 §"Why…"): off-critical-path = free;
   the wait moves; diffuse = each lever ≈ gap/N below the noise floor. **All three are real and
   empirically confirmed** by the ledger (FastBootstrap: 1.72–1.89× faster decode, byte-identical,
   **wall TIE** — `a10edf55`). v1's core motivation is correct.
2. **"Decompose the whole gap ONCE, don't build N fixes to discover elasticity"** — correct and the
   most valuable idea in v1. We keep it; we only fix *how* elasticity is computed under contention (§4).
3. **The closure ledger as the artifact** (T4): every category gets `{ported→parity | fundamental+proof}`
   and the gap is closed iff every row is parity-or-proven. Kept (§6), with the fundamentality bar
   raised to an *admissible* proof (§5).

What is **replaced**: the single-law identity (§0), the `Σ=wall` gate (→ two-law gate, §3), the
code-region taxonomy (→ resource taxonomy, §2), and raw Coz elasticity (→ contention-sound
elasticity, §4).

---

## §2 — The model: TWO conservation laws, combined by `max`

Let the program be characterized, for a fixed input + thread count + host, by:

- **Law A — Dependency critical path `CP_dep`.** The true longest-path through the wait-for DAG
  (channels, condvars, the in-order consumer's window-resolution chain `resolve(N−1) → publish
  window(N) → process(N)`). This is what v1 had — but computed as a *real* longest-path DAG, not the
  `timeline_analyze.py` overlap-heuristic (which self-documents "a real critical path algorithm
  would build the DAG of wait_for edges and run longest-path"; T1 must replace it).

- **Law B — Resource roofline `CP_res`.** For each shared resource *r* with sustained capacity
  `B_r` (bytes/s, lines/s, or μops/s) and total demand `D_r` (bytes, lines, μops the program must
  move through *r*), the program **cannot finish faster than** `D_r / B_r`. Then
  `CP_res = max_r (D_r / B_r)`.

> **Two-law lower bound (the replacement for v1's identity).**
> **`wall ≥ max( CP_dep , CP_res )`.**
>
> **Proof.** (A) No execution is shorter than its longest dependency chain — each edge is a
> happens-before, and a chain of *k* happens-before of total latency *L* takes ≥ *L* wall (standard
> DAG-schedule bound; holds for any thread count). (B) No execution moves `D_r` units through a
> resource of throughput `B_r` in less than `D_r / B_r` (conservation of the resource itself;
> Little's-law / roofline). Both are lower bounds on the *same* wall, so the wall is ≥ their max. ∎

This is a **lower bound, not an identity** — and that is the honest upgrade. The real wall can
*exceed* `max(CP_dep, CP_res)` because of **second-order effects** (latency not fully hidden by MLP,
scheduler imperfection, partial-overlap of the two laws). So we define the **gap-to-bound**:

> `ε_model = wall − max(CP_dep, CP_res)`  ( ≥ 0 always ).

`ε_model` is the **residual the model cannot yet explain**. The validity gate (§3) is **not**
`Σ=wall`; it is **`ε_model ≤ τ`** for a pre-registered tolerance `τ` (start 10%, tighten as the
model improves). `ε_model` being large is *itself a finding*: it says "neither a dependency chain
nor a single resource roofline explains the wall — there's an un-modeled interaction (e.g. two
resources co-saturating, or latency-bound not bandwidth-bound)," and it tells you to add the missing
resource/edge. This is the **honest analogue** of v1's conservation gate: same spirit (loud failure
when the accounting is incomplete), but it can no longer *silently pass while wrong*, because a
contention stall that v1 would have buried inside a span's duration now shows up as a gap between
`CP_dep` and `wall` that `CP_res` must explain — or `ε_model` flags it.

**The differential becomes two-sided in both laws:**

- `Δ_dep,c = time_c(gzippy CP_dep) − time_c(rapidgzip CP_dep)` — per dependency category.
- `Δ_res,r = D_r(gzippy)/B_r − D_r(rapidgzip)/B_r` — per resource. Since `B_r` is a property of the
  **host, identical for both binaries**, this reduces to `Δ_res,r = (D_r(gzippy) − D_r(rapidgzip)) /
  B_r` — **a pure traffic/occupancy difference scaled by a measured hardware constant.** This is the
  algorithm-neutral comparison the v1 code-region taxonomy could not give (§4 of the attack brief).

---

## §3 — The validity gates (what makes each step checkable)

Three gates, each with a positive-control self-test (ledger methodology rule #2: no instrument is
trusted without one).

**GATE 1 — Dependency-path gate (replaces the longest-path heuristic).**
- Build the *real* wait-for DAG longest path (T1). 
- **Positive control:** on a synthetic *known-dependency-bound* workload (a hand-built trace with one
  serial chain and idle workers), the reconstructed `CP_dep` must equal the injected chain length
  within ε. If it doesn't, the DAG reconstruction is broken — fix before trusting any real trace.
  (`fulcrum-region-config.json` already encodes a `ground_truth` block — `cp_offpath_region`,
  `min_heavy_blockers` — extend it with this self-test.)

**GATE 2 — Roofline gate (NEW; the law v1 lacked).**
- For each resource *r*, measure `B_r` **on this host** with a saturating microbenchmark (STREAM for
  DRAM BW; a pointer-chase for LLC line-rate; `perf stat --topdown` for backend/port occupancy).
  `B_r` is a host constant, measured once.
- Measure `D_r` for each binary via **counters that the project does NOT yet collect** — the current
  harness is `cycles, instructions, cache-references, cache-misses, page-faults, branches,
  branch-misses` (`profile_single_member_decompression_x86_64.sh:230`). **This is insufficient for a
  roofline.** Add: `MEM_LOAD_RETIRED.L3_MISS` (→ DRAM line traffic), uncore IMC `cas_count.{read,
  write}` (→ DRAM bytes, the ground truth for BW), `OFFCORE_REQUESTS.*`, and `perf stat --topdown`
  / `--td-level 2` (→ Memory-Bound vs Core-Bound vs Ports). **Without these counters the roofline
  law is unmeasurable — naming them is part of the design.**
- **Positive control:** on a *known-bandwidth-bound* kernel (a `memcpy` larger than LLC at the same
  thread count), `CP_res(DRAM)` must equal its measured wall within ε, and `CP_dep` must be far
  *below* the wall. This proves the roofline arithmetic and proves the two laws disagree in the
  expected direction on a known-resource-bound case.

**GATE 3 — Model-completeness gate (replaces v1's `Σ = wall`).**
- Assert **`ε_model = wall − max(CP_dep, CP_res) ≤ τ`** (pre-registered τ, start 10%).
- **On PASS:** the wall is explained by one of the two laws; the differential of *that* law is the
  closeable decomposition.
- **On FAIL (`ε_model > τ`):** the wall is explained by NEITHER — there is an un-modeled interaction.
  Prescribed next step (this is where v1 dead-ended): (a) check whether **two** resources co-saturate
  (`CP_res` should be `max` but the wall is near their *sum* → partial serialization of resources →
  model them jointly); (b) check whether the regime is **latency-bound not bandwidth-bound** (DRAM BW
  well under `B_DRAM` but `MEM_LOAD_RETIRED` latency cycles high → the bound is `outstanding-misses ×
  miss-latency / MLP`, a different roofline); (c) if `CP_dep` and `CP_res` are *both* near the wall
  and overlap, the program is **co-limited** — see §5, this is an admissible *fundamentality* finding,
  not a failure.

**Critically:** GATE 3 can now distinguish the three regimes that v1 conflated:
| Regime | Signature | What "close the gap" means |
|---|---|---|
| Dependency-bound | `CP_dep ≈ wall ≫ CP_res` | Shorten the serial chain (port the consumer/window logic). |
| Resource-bound | `CP_res ≈ wall ≫ CP_dep` | Reduce **traffic** `D_r` (fewer bytes moved), NOT instruction count. |
| Co-limited | both ≈ wall, overlapping | Must reduce **both**; closing one alone moves the wall by < its Δ (this is the diffuse case, and §5 may prove a floor). |

The ledger's TopdownL1 "Backend 31% / Bad-Spec 26% / Retiring 28% co-limited" is a **prediction that
this gap is the co-limited regime** — which is *exactly* why every single lever measured TIE, and
exactly why a single-law model was doomed. The two-law model represents it; the single-law model
could not.

---

## §4 — Algorithm-neutral, resource-based taxonomy (replaces code-region categories)

v1's categories were **code regions** (`decode`, `marker-resolve`, `copy/drain`, …). Attack #4 is
correct that these are **not commensurable** between pure-Rust gzippy and ISA-L-C rapidgzip:
rapidgzip has no `deflate_block` bootstrap, no u16 marker ring, no `absorb_isal_tail`; its
"marker-resolve" is structurally a different shape. `Δ_c` between non-aligned regions is
apples-to-oranges. Two complementary taxonomies fix this:

**(a) Dependency taxonomy — by SEMANTIC PHASE, not code symbol.** Both decoders, regardless of
implementation, must: *find block boundaries → decode symbols → resolve back-references against a
window → publish that window to the next unit → write output in order*. These **phases are
algorithm-neutral** (they're forced by the DEFLATE format + in-order output), even though the *code*
differs. Map each tool's spans onto these phases (the trace patch must tag spans by phase, not by
function name). `Δ_dep,phase` is then commensurable: "gzippy's *window-publish phase* sits 40ms
longer on the critical path than rapidgzip's," independent of how each implements it.

**(b) Resource taxonomy — by RESOURCE CONSUMED.** This is the **fully** algorithm-neutral layer and
the answer to attack #4: categorize by **cycles, bytes-moved (DRAM), cache-lines-touched (LLC),
page-faults, μops-retired, branch-mispredicts** — physical quantities that mean the same thing for
any code. `Δ_res,r` (bytes gzippy moves − bytes rapidgzip moves, ÷ host BW) is **exactly comparable
by construction** — it's two numbers from the same counter on the same silicon. A category like
"gzippy moves 985MB extra memmove (2× output)" (memory: `project_sm_bootstrap_overshoot`) is a
*resource* Δ and is meaningful even though rapidgzip has no corresponding code region.

**The differential is reported in BOTH taxonomies and reconciled:** the dependency Δ tells you
*which phase* to port; the resource Δ tells you *whether the port can possibly help* (if the phase's
Δ is dominated by traffic that the port won't reduce, the dependency Δ is a mirage — back to the
contention-mis-attribution trap of §0). **A dependency-category is only "closeable by porting" if its
Δ_dep is NOT explained by an irreducible resource Δ.** That join is the heart of the rebuilt method.

---

## §5 — Contention-sound elasticity (fixes Coz under shared resources)

Attack #2 is correct and this is the most technically serious fix. **Coz virtual-speedup is unsound
in the resource-bound regime.** Mechanism: Coz simulates "category X is k× faster" by inserting
proportional delays into *all other* threads. In a **bandwidth-bound** program, delaying other
threads **reduces the number of threads contending for DRAM**, so the target category's apparent
throughput rises — *not because the category is elastic, but because you removed its competitors.*
Coz then reports `∂wall/∂speed > 0` (elastic) for a category whose real elasticity is ~0 (it's
DRAM-floored). **This is an artifact that would send the closure loop to port a category that cannot
move the wall** — the exact failure mode we're trying to escape, now dressed as a causal signal.

**Coz's own validity precondition** (from the Coz paper) is that virtual speedup is sound when the
program's progress is governed by the **scheduler-visible dependency structure** — i.e. the
dependency-bound regime. So:

> **Elasticity protocol (regime-gated):**
> - **Run GATE 2/3 FIRST.** Classify the regime.
> - **Dependency-bound regime → Coz is valid.** Use whole-program virtual-speedup (v1's T3) to get
>   `∂wall/∂speed` per dependency phase. Sound here because the wall *is* the chain Coz perturbs.
> - **Resource-bound / co-limited regime → Coz is NOT valid.** Substitute **measured roofline
>   headroom**: for resource *r*, elasticity is `∂wall/∂D_r ≈ 1/B_r` *iff* `CP_res(r) ≈ wall`
>   (you're on that roofline) and `0` otherwise (you're below it — moving less traffic through a
>   non-saturated resource doesn't move the wall). This is **directly measured**, not simulated:
>   reduce `D_r` synthetically (e.g. NT-stores to drop write traffic; a smaller window to drop read
>   traffic) and confirm the wall tracks `ΔD_r / B_r`. The "synthetic `D_r` reduction" is the
>   resource-world analogue of Coz's virtual speedup, and it does **not** perturb thread count, so it
>   does **not** corrupt the contention it measures.
> - **Coz sanity cross-check (catches the artifact):** in any regime, run Coz AND the roofline-headroom
>   method. If Coz reports a category elastic but its resource is below roofline (headroom method says
>   0), **Coz is lying via the thread-removal artifact — trust the roofline.** Disagreement between the
>   two IS the detector for attack #2.

This makes elasticity sound in *both* regimes and turns the v1 "one Coz campaign" into "one campaign
in the regime where Coz is valid, plus a measured-headroom campaign where it isn't, plus a
cross-check that detects the artifact."

---

## §6 — Monotone closure loop (fixes "removing a copy raises cache pressure elsewhere")

Attack #3 is correct: closing the top-Δ category is **not** guaranteed to reduce the wall, and **this
project has the receipt** — pre-sizing buffers improved hit-rate but *regressed* wall via
resident-set/TLB growth (ledger `rpmalloc/Z-prewarm/segmented` row), and removing a copy was
wall-neutral because the freed bandwidth wasn't the bottleneck. A naive "close top-Δ, repeat" loop is
**not monotone in wall** and can diverge.

> **Monotone closure loop (guarded):**
> For each candidate fix targeting category *c*:
> 1. **Predict** `Δwall_pred = Δ_c · elasticity_c` (§5, regime-correct elasticity).
> 2. **Cross-resource guard (the missing invariant):** before/after the fix, measure **ALL**
>    resource demands `{D_r}` and **both** critical paths, not just *c*'s. Accept the fix **iff**
>    `max(CP_dep, CP_res)` strictly decreases by ≥ `Δwall_pred − slack` **AND no other resource's
>    `D_r` rose enough to raise its roofline above the new bound.** I.e. accept iff the **new
>    `max(CP_dep, CP_res)` < old** — the bound itself, which already accounts for "the wait moved to
>    another resource."
> 3. **Reject + record** if the bound did not drop (the copy-removal/pre-size corpses are *rejected
>    by this guard automatically*: they didn't lower `max(CP_dep, CP_res)` because they raised a
>    different resource's roofline or freed a non-bottleneck resource → predicted ≈ 0 or negative).
> 4. **Monotonicity theorem:** the accepted sequence of fixes is monotone-decreasing in
>    `max(CP_dep, CP_res)` *by the acceptance rule itself* (we only accept a fix that strictly lowers
>    the bound). Since the bound is `≥ 0` and decreases by ≥ a fixed slack per accepted step, the loop
>    **terminates** (finitely many accepted fixes) at a point where **no available fix lowers the
>    bound** — which is either parity or a proven floor (§5 fundamentality). ∎
>
> The guard is what makes it monotone: v1's loop ranked by `Δ_c` and ported blindly; this loop
> **only commits a fix that is *verified* to lower the two-law bound**, so cache-pressure-elsewhere
> and wait-moved-to-another-resource can't cause a regression — they cause a **rejection**.

Note this loop **subsumes** wall-A/B (you still verify the bound dropped) but does it *with a
prediction first* and a *cross-resource guard*, so a TIE is now *informative* (it says "the bound
didn't move → you targeted a non-bottleneck → here's the resource that's actually binding") instead
of just another corpse.

---

## §5b — Admissible fundamentality proof (what "we proved a floor" must mean)

Attack #5 is correct: "provably fundamental" must be a **proof**, not fatigue. A category Δ_c is
admissibly **fundamental** (irreducible by any port) iff it meets ONE of these, each falsifiable:

1. **Resource-floor proof.** `Δ_res,r = (D_r(gzippy) − D_r(rapidgzip)) / B_r`, the resource *r* is at
   roofline (`CP_res(r) ≈ wall`, GATE 2), **and** the traffic difference `D_r(gzippy) − D_r(rapidgzip)`
   is **information-theoretically or format-required** — e.g. the in-order consumer must materialize
   the full window for marker resolution (a correctness requirement of DEFLATE back-references), so
   `D_r` cannot drop below the window-traffic floor without changing output. Admissible because it
   cites a *measured* roofline AND a *provable* lower bound on traffic. **Falsifiable:** exhibit a
   correct decoder that moves less traffic ⇒ proof void. (NB: the ledger's "bandwidth-bound DEAD" row
   was **RETRACTED** precisely by this falsification — two pure-Rust decoders moved 2–6× different
   traffic on identical data ⇒ NOT a floor. The bar works: it already killed one false "fundamental"
   claim.)
2. **Co-limit proof.** `CP_dep ≈ CP_res ≈ wall` and they **overlap** (the same spans that are on the
   dependency chain are the ones saturating the resource). Then closing either alone moves the wall by
   `< Δ` (the other binds), and closing both requires *simultaneous* algorithm+traffic reduction that
   no port of one category provides. Admissible because both laws are measured at the wall.
   **Falsifiable:** show a fix that lowers both ⇒ not co-limited.
3. **Microarch-floor proof.** The category's cost is a hardware constant a port cannot change at fixed
   algorithm — e.g. `Δ_res = (μops_gzippy − μops_rapidgzip) / port_throughput` where the μop count
   difference is the *minimum* for pure-Rust codegen of the same algorithm (verified by inspecting the
   compiled asm has no removable μops, `asm_compare.sh`). Admissible only with the asm shown.
   **Falsifiable:** exhibit asm with fewer μops, same output.

**Inadmissible** (these are "we gave up", banned from the ledger): "we tried N levers and they were
TIE"; "perf shows high cache-misses" (a symptom, not a floor — ledger rule #3); "it's probably
bandwidth" without the IMC counter and the traffic-lower-bound argument.

---

## §7 — The honest verdict: what is actually provable

**"Provably close the ENTIRE gap" is NOT achievable as stated, and claiming it would be the lie the
brief warns against.** Here is the rigorous truth, stated plainly:

> **What IS provable:** a **complete, two-law decomposition of the gap into**
> - a **closeable part** — categories where `Δ_dep` is on the dependency-bound critical path with
>   `Coz`-confirmed elasticity, OR a resource `Δ_res` you can reduce while *on* that resource's
>   roofline — **bounded above by `Σ(closeable Δ)` and verified shut, one guarded-monotone fix at a
>   time, each lowering the two-law bound** (§6); and
> - a **fundamental part** — categories with an *admissible* floor proof (§5b): resource-floor,
>   co-limit, or microarch-floor, **each falsifiable**, with the residual `ΣΔ_fundamental` reported
>   as the **proven irreducible gap at this thread count on this host**.
>
> The gap is "closed" in the only honest sense: **`gap = closeable + fundamental`, the closeable part
> is driven to ≤ ε, and the fundamental part carries a proof, not a shrug.**

This is **weaker than v1's headline and stronger than v1's actual guarantee** — because v1's "identity"
was false, so its "provably the entire gap" was provable only under an assumption (pure dependency-DAG)
that this program **violates** (the ledger's co-limited TopdownL1 is the proof it violates it). The
decompose-and-prove-the-floor framing is **the non-rabbit-holing tool the user wants**: it tells you,
in ONE measurement campaign, (a) exactly how much of the gap is closeable, (b) which fixes to build in
which order, with a prediction-first guard so TIEs become information, and (c) a *proof* of where to
STOP — so you never run the 15th single-lever A/B against a floor again.

**Brutally honest corollary specific to this gap:** the existing ledger already smells co-limited
(TopdownL1 co-limit; FastBootstrap faster-decode → wall-TIE; "both at hardware floors"). The two-law
model's **most likely verdict for THIS gap is that a large fraction is the *co-limit* fundamental case
(§5b.2)** — the in-order consumer's window-resolution chain (`CP_dep`) running in lockstep with the
window/output DRAM traffic (`CP_res`), each holding the other at the wall. If so, the provable result
is: **the closeable slice is small (the consumer-chain trim, P3), and the rest is a proven co-limit
floor** — which is a *legitimate, defensible "we are at the achievable parity"*, not a failure to find
a lever. **The method's value is that it would PROVE this in one campaign instead of discovering it
over another 14 corpses.**
>
> **Counter-evidence I must not bury (against my own lean):** the ledger ALSO reports gzippy moves
> **5.3× FEWER LLC-misses than rapidgzip** ("less memory-bound", NT-stores row) and **page-faults/sec
> are EQUAL across engines**. If gzippy is genuinely *less* DRAM-bound than the tool it's losing to,
> then `CP_res(DRAM)` may be BELOW the wall and the binding law is `CP_dep` (the consumer chain) or a
> **port-occupancy** roofline (the Backend-bound 31% is execution-ports/μops, not necessarily DRAM).
> That would make the gap **closeable** (port the chain / reduce μops), not a DRAM floor. **I cannot
> resolve this from the existing counters** — the project has never measured IMC bandwidth or a
> --topdown td-level-2 split, so "co-limited" vs "dependency-bound on the consumer chain" vs
> "port-bound" is currently *undetermined*. This is precisely the ambiguity GATE 2/3 exists to
> settle, and is the honest reason the roofline counters (T2) are the highest-value missing
> instrument: **the two-law model does not pre-judge the regime — it is the apparatus that decides
> it.** My "co-limit most likely" above is a *prior*, explicitly falsifiable by GATE 2, not a claim.

---

## §8 — The tool to build (revised; extends Fulcrum)

- **(T1) Real DAG longest-path + dependency-phase tagging** — replace `timeline_analyze.py`'s
  overlap-heuristic (self-admittedly "naive") with a true wait-for-DAG longest-path; tag spans by
  *semantic phase* (§4a), not function. Ship GATE 1's positive control.
- **(T2) Roofline collector (NEW, the law v1 lacked)** — add the missing counters (IMC
  `cas_count`, `MEM_LOAD_RETIRED.L3_MISS`, `OFFCORE_REQUESTS`, `--topdown td-level 2`); measure host
  `B_r` via STREAM/pointer-chase once; emit `D_r` per binary and `CP_res = max_r D_r/B_r`. Ship GATE
  2's positive control (memcpy-larger-than-LLC).
- **(T3) Two-law completeness check** — emit `CP_dep`, `CP_res`, `wall`, `ε_model`; FAIL loudly if
  `ε_model > τ` AND print the prescribed disambiguation (§3 GATE 3). This is the v1 "conservation
  gate" done right.
- **(T4) Two-taxonomy differential** — `Δ_dep,phase` table AND `Δ_res,r` table, with the join (§4):
  flag every dependency-Δ that is *explained away* by an irreducible resource-Δ (mirage detector).
- **(T5) Regime-gated elasticity** — Coz where valid + roofline-headroom where not + the
  disagreement cross-check that detects Coz's thread-removal artifact (§5).
- **(T6) Guarded-monotone closure ledger** — each row `{Δ_dep, Δ_res, elasticity, regime, verdict:
  ported→bound-dropped | fundamental + which-floor-proof}`; the loop only commits a fix that
  *measurably lowers* `max(CP_dep, CP_res)` (§6). Gap closed iff every row is bound-dropped-to-ε or
  carries an admissible floor proof.

---

## §9 — Self-attack on THIS rebuilt design (no sub-advisor was available; attacking in writing)

The brief asked for a sub-advisor to attack the rebuild; the `Agent`/`Task` subagent tool is **not
available** in this environment (verified: ToolSearch found no `Agent`/`Task` deferred tool). So I
attack it myself, hardest blows first, and concede where it bends.

**A1. "Two-law `max` is also a lower bound, not an identity — so `ε_model` can be large and you've
just relocated the hand-waving into a fudge tolerance τ."**
*Concede partially.* `ε_model` is real slack (imperfect overlap of the two laws, MLP not modeled).
The defense is honest: I **do not claim `ε_model = 0`**. I claim `ε_model` is *itself the finding* and
prescribe what to do when it's large (model two resources jointly, or switch to a latency-roofline).
The design is provable in the sense that **every quantity is measured and every gate is falsifiable** —
it is NOT provable in the sense of "wall reproduced to the bit by a closed-form." That's the correct,
honest standard for a hardware perf model; anyone claiming the latter is lying. **This is exactly the
v1→v2 downgrade I made on purpose.** If a reader needs "ENTIRE gap, closed-form," the answer is §7: no
such thing exists for a co-limited 16-thread decoder.

**A2. "Measuring `D_r` per binary at 16 threads via counters is itself contended/noisy — IMC counters
are socket-wide and capture *other tenants* on a shared box."**
*Serious, and it bites this project specifically* (the box is shared; ledger rule #1 is "absolutes are
noise"). Defense: (a) host-freeze the noisy neighbors (the project already does — frigate/plex cgroup
freeze) for the roofline campaign; (b) use **interleaved/relative** `D_r` the same way `measure.sh`
uses interleaved wall — measure gzippy and rapidgzip back-to-back so socket-wide contention is common
mode and the *difference* `D_r(gzippy) − D_r(rapidgzip)` is robust even if each absolute is noisy.
**`Δ_res,r` is a difference, so the common-mode socket noise cancels — same trick that makes
`measure.sh` trustworthy.** This is a genuine strength of the difference-based taxonomy.

**A3. "Coz-vs-roofline disagreement as an artifact-detector assumes one of them is right. What if
BOTH are wrong (e.g. an unmodeled resource)?"**
*Valid.* The cross-check detects the *known* thread-removal artifact; it does not certify correctness.
Defense: GATE 3's `ε_model` is the backstop — if both elasticity methods agree yet fixes don't drop
the bound (the §6 guard rejects them) and `ε_model` is large, the model is incomplete and T3 says so
loudly. The design **fails safe** (loud "model incomplete, here's the disambiguation") rather than
**fails silent** (v1's gate passing while wrong). Failing safe is the best achievable; I claim no more.

**A4. "The semantic-phase taxonomy (§4a) assumes gzippy and rapidgzip share phases, but rapidgzip's
parallelism granularity / window strategy may not decompose into the same phases — the join could be
ill-defined for the very category that matters."**
*Concede this is the weakest practical point.* If rapidgzip fuses boundary-scan into decode such that
no span boundary exists at the phase edge, `Δ_dep,phase` for that phase is unmeasurable on the
rapidgzip side. Mitigation: **fall back to the resource taxonomy (§4b) for any phase that doesn't
align** — `Δ_res,r` is *always* well-defined (it's counters, not regions), so the method degrades to
"resource-only differential" for unalignable phases rather than breaking. The dependency taxonomy is a
*bonus* (tells you which code to port); the resource taxonomy is the *floor* (always works). The proof
of decomposition (§2 two-law bound) rests only on the resource taxonomy + the real DAG longest-path,
**neither of which needs phase-alignment** — so A4 weakens the *guidance* (which port), not the
*provability* (the decomposition + floor).

**A5. "You still haven't proven the loop closes the gap — only that it monotonically lowers a *lower
bound*. The wall could sit above the bound forever (`ε_model`), so driving the bound to rapidgzip's
doesn't drive the *wall* to rapidgzip's."**
*The sharpest blow; concede the precise scope.* Correct: the loop provably drives `max(CP_dep, CP_res)`
down, and `wall ≥ that`. If `ε_model > 0` and *differs* between the two binaries, equalizing the bound
does not equalize the wall. **This is why §7 does not claim "close the wall to rapidgzip" — it claims
"decompose the gap; close the closeable; prove the floor."** The wall-A/B verdict (§6 step 2, the
production `measure.sh`) is still the *final* arbiter — the model **guides and bounds and proves the
floor**, but the closeable claim is only *ratified* when the guarded fix shows the **wall** (not just
the bound) dropped on interleaved production A/B. So the design's honest contract is: **the model is
the planner + the fundamentality prover; the interleaved wall-A/B remains the judge.** That is strictly
more than v1 (which had no floor proof and a false identity) and strictly less than "provably the whole
wall" (which is unattainable). Stated this way, A5 is not a defeat — it's the scope statement, and it's
in §7.

**Net of self-attack:** the design survives as **"provably decompose + provably bound the closeable +
provably-or-falsifiably floor the rest, with the interleaved wall-A/B as final judge."** It does **not**
survive as "provably close the entire wall in closed form" — and §7 says so. That asymmetry is the
honest deliverable.
