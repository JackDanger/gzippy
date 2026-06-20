# SILT4-PLAN — the confound-proof path for silesia-T4 (+16.5% vs rapidgzip)

**Author:** path-plotting advisor (Opus). **Date:** 2026-06-20. **Status:** a measurement
DESIGN, not a finding. Every number it asks for must pass Gate-0/1/2 (CLAUDE.md PROTOCOL)
before it counts. No conclusion in this file is a finding — they are HYPOTHESES with the
exact perturbation that would confirm/kill each.

---

## 0. STOP — three facts that change the brief before you run anything

1. **The brief's sha is not HEAD.** The standing rig gated silesia-T4=1.165 at
   `origin/kernel-converge-A @ 805c03c0`. **HEAD is `c91aa125`** (local `kernel-converge-A`,
   the night31/32 T1 work). `git merge-base --is-ancestor 805c03c0 HEAD` = **FALSE** — the
   rig commit is NOT in HEAD's history, and there is **no `scripts/bench/standing/` in the
   working tree at HEAD**. `standing.sh` was introduced at `c47a19c2`. So before any silT4
   number exists you must (a) pin a subject commit and (b) get the standing rig onto it
   (cherry-pick `c47a19c2`, or build the subject at the rig's commit). **Do not assume the
   1.165 reproduces at HEAD** — re-establish it first (S0).
2. **The view-list win IS present.** `git merge-base --is-ancestor 300e772b HEAD` = **TRUE**.
   So this is NOT the missing-merge case (FORK-B/L0 of T2-LOCATE-PLAN). The T≥2 data-plane
   convergence is in the binary; silT4 is a *genuine residual*, not a regression artifact.
3. **The cheap fix is doubly dead.** The prefetch-ordering A/B was already wall-falsified
   for silT4 (record). AND the obvious "shrink chunks to balance the tail" is *already
   measured bad*: `single_member.rs:56-58` documents that finer granularity at T>1
   **regressed silesia-T4 wall +20%** (block-finder + scheduling overhead + output-buffer
   cache residency co-vary). **Chunk-size is a confound, not a lever — never sweep it to
   localize silT4.**

**Therefore the honest job:** (A) re-establish silT4 at a pinned binary, (B) run ONE
discriminator that separates the only two mechanisms that can produce a corpus+T-specific
loss — *scheduling/tail-imbalance* (a fixable pipeline gap) vs *per-chunk decode-CPU
surplus amplified by the wave structure* (the kernel front, out of T≥2 scope) — then (C)
branch. The prior record's "tail load-imbalance" localization is a HYPOTHESIS to confirm
or kill at the current sha (bias #2: disproof/proof amnesia across restarts), not a fact to
build on.

---

## 1. WHAT THE CODE ACTUALLY DOES (deterministic, read at c91aa125)

These are mechanism facts (source-read tier = HYPOTHESIS for *causation*, but reliable for
*what the pipeline is*), needed to design the measurement:

- **Chunking is by COMPRESSED size.** T>1 uses `TARGET_COMPRESSED_CHUNK_BYTES = 4 MiB`
  (`single_member.rs:44, :221-225`); T1 uses 1 MiB. `adjusted_chunk_size_bytes` only shrinks
  below 4 MiB for *small* files (`default*2*threads > file_size`). silesia.gz (~68 MB
  compressed; raw output 211,968,000 B per `_ab_persym_t8.sh`) is large ⇒ **no adjustment ⇒
  ~16-17 chunks of 4 MiB compressed.**
- **The decode-time per chunk is HIGHLY variable for silesia.** Chunks are equal in
  *compressed* bytes but silesia is a heterogeneous tarball: a 4 MiB-compressed chunk of
  English text (`dickens`) expands ~3-4× and costs far more decode work than a 4 MiB chunk
  of already-incompressible data (the `x-ray`/`mr`/`sao` binaries, ~1×). monorepo is far
  more uniform. **This decode-time variance — not chunk count alone — is the straggler
  source if tail-imbalance is real.**
- **The consumer is in-order.** It blocks on `rx.recv()` for the next chunk in sequence —
  the `wait.future_recv` span (`chunk_fetcher.rs:2086, :4188`) and `wait.block_fetcher_get`
  (`:1792`). A late/slow chunk N stalls the writer even if N+1..N+k already finished. This
  is the structural mechanism a tail-imbalance loss would show as.
- **Prefetch DEPTH already matches rg.** gz: `cache_capacity = max(16, P)`,
  `prefetch_capacity = 2*P` (`chunk_fetcher.rs:705-706`); the strategy is asked for
  `prefetch(prefetch_cache.capacity())`. rg: `m_cache(max(16, P))`,
  `m_prefetchCache(2*P)` (`BlockFetcher.hpp:181-182`), `prefetch(m_prefetchCache.capacity())`
  (`:474`). **These are byte-identical sizings** → the BlockFetcher prefetch depth is NOT a
  divergence. (This is consistent with the prefetch A/B having been falsified.)
- **rg's `3 × hardware_concurrency` is a DIFFERENT knob:** it is the *block-finder*
  discovery look-ahead (`BlockFinder.hpp:212 m_prefetchCount`), governing how far ahead
  block *boundaries* are found, not how many chunks decode in flight. gz's counterpart is
  `async_block_finder` — this is the one structurally-open rg item, but it only matters if
  S2 shows the pipeline *under-fills* (workers idle waiting for boundaries), which the
  matching prefetch depth makes a low prior. Re-check, do not assume.

---

## 2. THE TWO RIVAL HYPOTHESES (pre-registered; the whole plan is to discriminate them)

A corpus-specific + T-specific loss has exactly two mechanism families. Pre-register both so
neither can be confirmed by narration:

- **H-TAIL (scheduling):** silesia's high per-chunk decode-time variance + ~17 chunks over 4
  cores ⇒ the last wave (and the slowest chunk in each wave) leaves cores idle; the in-order
  consumer blocks in `wait.future_recv` on a late straggler. **Falsifier:** if the wall is
  flat when you remove the imbalance (oracle: serialize-the-tail control, or force
  uniform-*decode-time* partitioning), H-TAIL is dead. **Confirmer:** effcores (avg busy
  cores = total_cyc/wall) is well below T (e.g. ~3.0-3.4 of 4) AND the idle is concentrated
  in `wait.future_recv` on the last 1-2 chunks AND removing the imbalance moves the wall
  proportionally.
- **H-KERNEL (per-chunk decode CPU):** gz decodes each silesia chunk slower than rg
  (the same instruction/cyc-per-byte surplus that gives T1=1.070), and the T4 wave structure
  multiplies it (4 cores each running the slower kernel ⇒ ~same ratio, but the tail wave
  exposes it more). **Falsifier:** if effcores ≈ T (cores NOT idle) and gz/rg per-chunk
  decode-cyc ratio ≈ the wall ratio, it is the kernel, not scheduling — and that is the
  **T1/igzip kernel front, OUT of T≥2 scope.** **Confirmer:** effcores ≈ T, gz cyc/byte on
  the same chunks ≫ rg, tail-removal oracle is FLAT.

**Why this matters:** the prior record's "tail load-imbalance, at-floor" verdict assumed
H-TAIL and then failed to fix it with prefetch ordering. That failure is *equally consistent
with H-KERNEL* (prefetch ordering can't fix a slower kernel). **No one has run the
discriminator that separates them at the wall.** That discriminator is this plan's S2 — the
single highest-VoI action.

---

## 3. THE TOOLING TO BUILD FIRST (user's rule: don't fix it while it's hard)

silT4 is hard to iterate on today because the per-chunk timeline exists but is not reduced
to the one metric the fork needs. Build these THREE small, byte-transparent tools *before*
any perturbation. Each is cheap and self-validating; together they make the fork a
one-command read.

### Tool 1 — `tail-imbalance` reducer over the existing timeline (PRIMARY)
- The data already exists: `GZIPPY_TIMELINE=/tmp/t.json` (instrument
  `instruments/trace_timeline.rs`) emits Chrome-trace spans incl. `worker.decode_chunk`
  (per-chunk decode duration, with worker id), `wait.future_recv` and
  `wait.block_fetcher_get` (the in-order consumer's blocking waits). **No production code
  change** — the instrument is env-gated and compiles transparent.
- **Build:** a `scripts/parallel_sm_tail_metric.py` (sibling to the existing
  `parallel_sm_log_summary.py`) that ingests the trace JSON and emits, deterministically:
  - per-chunk decode wall (sorted by completion order AND by sequence order),
  - **effcores = Σ(decode_durations) / total_drive_wall** (avg busy cores; the single
    scalar that separates H-TAIL from H-KERNEL: ≈T ⇒ kernel, ≪T ⇒ tail),
  - **tail-idle = Σ(`wait.future_recv` durations) / drive_wall** (fraction of wall the
    consumer spent blocked on a not-yet-ready in-order chunk),
  - the **last-wave profile**: decode time of the final `ceil(chunks/T)` chunks vs the mean,
  - **decode-time variance / max-min spread** across chunks (the silesia-vs-monorepo
    discriminator from §2).
- **Gate-0 self-validation (BLOCKING, the reducer asserts these or refuses to print):**
  - **conservation:** Σ(per-chunk decode) + Σ(consumer waits) + serial-tail reconciles to
    `drive` span wall within a stated tolerance (no double-count; the "62ms phantom" was a
    nested-span double-count — the reducer must dedupe nested spans).
  - **non-inert:** chunk-count from the trace == chunk-count from `--verbose` BlockFetcher
    stats == `ceil(compressed_len / chunk_size)`. If they disagree the trace is lying.
  - **same sink:** the traced run used `/dev/null` (record the sink in the trace header).

### Tool 2 — a `silt4` cell in the standing rig + an effcores column
- The standing rig (`standing.sh`, `c47a19c2`) already does interleaved N≥13, A/A
  self-test, sha==zcat, path=ParallelSM. **Add a single pinned cell** `silesia-T4` that
  *also* dumps effcores + tail-idle (Tool 1) alongside the gz/rg ratio, so every iteration
  prints "ratio 1.165 | effcores 3.1/4 | tail-idle 9%" in one line. This is what makes
  silT4 cheap to iterate: the fork-deciding numbers come out of the same command as the wall.

### Tool 3 — a `serialize-the-tail` / `uniform-decode-partition` oracle (for S3, build now)
- The clean Gate-2 perturbation for H-TAIL needs an oracle that **removes the imbalance
  without changing chunk size** (chunk size is the confound, §0.3). Two options, build the
  cheaper:
  - (a) **tail-serialize control:** force the last `(chunks mod T)` chunks to decode on a
    single worker back-to-back (no parallel tail). If silT4 is tail-bound, this *worsens*
    the wall proportionally to the tail (positive control that the tail is on the path); if
    flat, the tail is slack. Env-gated, byte-transparent, OFF==identity.
  - (b) **decode-time-balanced partition oracle:** partition by *decoded* (not compressed)
    size using a Pass-1-captured per-region expansion ratio, so every chunk costs ~equal
    decode time. If H-TAIL, the wall drops toward effcores≈T; if H-KERNEL, flat. This is the
    *removal oracle* (it CHEATS — knows expansion a priori — so it is a CEILING, never the
    prize; label it so).
- **Gate-0:** a HITS counter that fires (the oracle provably ran, not no-op'd) and a
  byte-exact sha. The repaired `seed_windows` pattern (counter `>0`, not a file-path no-op)
  is the template — do not ship an inert oracle (the campaign's most-repeated Gate-0 miss).

**Order:** Tool 1 → Tool 2 (so S0/S1 print the metric) → Tool 3 (needed only at S3). Tools 1
& 2 are pure analysis (zero production risk). Tool 3 is env-gated and OFF==identity.

---

## 4. THE DECISION TREE

```
S0  Re-establish silT4 at a pinned binary + Gate-0 the rig      ─┐
S1  Confirm the loss is real + corpus/T-shaped (the 4×{T1,T2,T4,T8} cells) ─┴─> FORK
        │
        ├─ silT4 ratio ≥ ~0.97 (loss gone at HEAD) ──> CLOSED-ON-ITS-OWN. Bank, AMD-debt, stop.
        │
        └─ silT4 ratio still ≥ ~1.10 (loss real) ──> S2 the discriminator
                 │
   S2  effcores + tail-idle (Tool 1) on silT4, gz AND rg side by side
                 ├─ effcores ≪ T  &  tail-idle large ──> H-TAIL ──> S3-TAIL
                 └─ effcores ≈ T  &  gz cyc/byte ≫ rg ──> H-KERNEL ──> S3-KERNEL
                 │
   S3-TAIL   causal: serialize-tail control (worsens?) + balanced-partition oracle (closes?)
                 ├─ both fire ──> H-TAIL CONFIRMED ──> S4-TAIL (mine rg blueprint, port the fix)
                 └─ oracle flat ──> H-TAIL DEAD ──> re-route to S3-KERNEL
   S3-KERNEL causal: per-chunk decode-cyc gz/rg on identical chunks; tail-removal oracle FLAT
                 └─ confirmed ──> silT4 is the kernel front (T1/igzip scope), NOT a T≥2 lever.
                                   Bank as "T≥2 schedule is at parity; residual is kernel." Defer to kernel campaign.
   S4-TAIL   port rg's tail mitigation faithfully; prize ≡ measured Δ of the byte-exact port, never gap-to-rg.
   AMD       every verdict above is Intel NOT-YET-LAW until solvency/Zen2 replicates (defer, don't block).
```

---

### S0 — re-establish silT4 + Gate-0 the rig (BLOCKING; no number exists until this passes)
- **Instrument:** `standing.sh` (cherry-pick `c47a19c2` onto the subject, or build the
  subject at that commit) on guest `REDACTED_IP` (`ssh -J neurotic root@REDACTED_IP`).
- **Gate-0 (ALL loud, else stop):**
  - (a) **rg present + self-tests** rg-vs-rg interleaved ≈ 1.0 ± spread on the silesia-T4
    cell; confirm native ELF `v0.16.0`, not a wheel.
  - (b) **both arms `/dev/null`** (SINK LAW); tmpfs/sink equivalence *measured* on this one
    cell, not asserted.
  - (c) `GZIPPY_DEBUG=1 → path=ParallelSM`, `GZIPPY_FORCE_PARALLEL_SM=1`, `sha==zcat`.
  - (d) **both binaries fresh, identical flags** (`--no-default-features --features
    pure-rust-inflate`, `RUSTFLAGS="-C target-cpu=native"`, fresh target dir — NOT a
    `/dev/shm` pin; binary-drift retired ~5% phantom instr once). **Log the subject sha.**
  - (e) `git merge-base --is-ancestor 300e772b <subject-sha>` logged (expect TRUE at HEAD).
- **Verdict:** rig trustworthy iff (a)-(e) pass. The *number* this step produces is the
  re-confirmed silesia-T4 ratio at the pinned sha — the brief's 1.165 is at a different
  commit and must be reproduced, not inherited.

### S1 — confirm the loss is real and corpus/T-shaped
- **Instrument:** `standing.sh` matrix, N≥13 interleaved, T1/T2/T4/T8 × silesia+monorepo
  (add nasa+squishy if cheap), `/dev/null` both arms, with Tool-2's effcores column.
- **Gate-1:** per cell Δ vs inter-run spread; `>1+spread` = real loss.
- **Branch:** silT4 < ~1.10 → loss gone/shrunk → bank + AMD-debt + stop (do NOT keep
  poking a closed cell — bias #4). silT4 ≥ ~1.10 AND monorepo-T4 ≈ TIE (the shape holds) →
  S2. *Confound:* if monorepo-T4 ALSO loses now, the "corpus-specific" framing is dead and
  this is a broader T4 regression → treat as T2-LOCATE-PLAN FORK-B, not silT4.

### S2 — THE DISCRIMINATOR (highest VoI; this is the step no prior session ran)
- **Instrument:** Tool 1 on the silT4 cell, **gz AND rg side by side** (rg via its own
  `--verbose`/`--analyze` chunk stats for per-chunk timing + chunk count; gz via
  `GZIPPY_TIMELINE`). Report effcores, tail-idle, last-wave profile, decode-time variance.
- **Gate-0:** Tool-1 conservation + non-inert chunk-count cross-check (§3); both runs
  `/dev/null`; rg chunk count and gz chunk count both reported (they should be ~equal if
  chunking matches — a divergence here is itself a finding).
- **Verdict (ATTRIBUTION-tier — routes, does not conclude):**
  - effcores ≪ T (e.g. < ~3.3/4) AND tail-idle concentrated in `wait.future_recv` on the
    last chunks ⇒ **route to S3-TAIL.**
  - effcores ≈ T AND gz per-chunk decode-cyc ≫ rg ⇒ **route to S3-KERNEL.**
- **Confound red-team:** effcores/tail-idle is attribution, NEVER the verdict (bias #7 —
  decompose-and-blame). The verdict is S3's perturbation. Also: compare gz effcores to **rg
  effcores** — if rg *also* runs at effcores ~3.2/4 on silesia-T4 (same tail shape) but a
  smaller absolute wall, that is H-KERNEL wearing H-TAIL's clothes (same idle, faster
  kernel) → S3-KERNEL.

### S3-TAIL — causal confirm of tail-imbalance (Gate-2, the only verdict)
- **Instrument:** Tool 3. Run BOTH:
  - **serialize-tail control:** must *worsen* the wall ∝ the tail (positive control the tail
    is on the path). Flat ⇒ tail is slack ⇒ H-TAIL dead → reroute S3-KERNEL.
  - **balanced-partition oracle (CEILING):** must *drop* the wall toward effcores≈T. The
    drop is the **upper bound** on any tail fix, never the prize.
- **Gate-0:** oracle HITS counter fired (non-inert), sha==zcat at dose-0/identity.
- **Gate-2:** interleaved wall response; **frequency-neutral** — the oracle changes
  scheduling not CPU load, so turbo-depression is not a risk here, but still run the A/A
  control. Δ survives ⇒ H-TAIL confirmed AND bounded.
- **Branch:** confirmed+bounded-worthwhile → S4-TAIL. Confirmed but ceiling < ~3-4% →
  record as "tail is on the path but the capturable prize is small"; weigh vs S3-KERNEL.

### S3-KERNEL — confirm the residual is per-chunk decode CPU
- **Instrument:** `perf stat` cyc/byte + instr/byte gz vs rg on the *same silesia chunks*
  (taskset P-cores; cyc/byte is frequency-invariant on the loaded LXC) + the `removal_oracle`
  (`GZIPPY_ORACLE_NODECODE`) tail-removal: if removing the tail's *decode* is flat at the
  wall, the tail is slack and the loss is whole-pipeline kernel CPU.
- **Gate-2:** gz cyc/byte ≫ rg with effcores≈T ⇒ silT4 = the kernel surplus, **OUT of T≥2
  scope** — it is the T1/igzip inner-kernel front (NIGHT40 lineage). Bank: "the T≥2
  *schedule* is at parity on silesia; the residual is the kernel, tracked by the kernel
  campaign." Do not build a scheduling fix for a kernel problem.

### S4-TAIL — port rg's tail mitigation faithfully (only if S3-TAIL confirmed worthwhile)
- **Mine the blueprint (§5).** The fix is a faithful port of whatever rg does to keep the
  tail filled, NOT an invention. Prize ≡ measured whole-program Δ of the byte-exact port on
  the standing rig, NEVER the gap-to-rg or the oracle ceiling (bias: ceiling-as-prize).

---

## 5. THE BLUEPRINT — what rg does about the tail (cite, don't guess)

- **Same chunking, same prefetch depth.** rg also uses 4 MiB-ish compressed chunks and
  `m_prefetchCache(2*P)` (`BlockFetcher.hpp:182`), asking `prefetch(prefetchCache.capacity())`
  (`:474`). So rg does **not** avoid the tail by deeper decode prefetch — gz already matches
  it. This is why the prefetch A/B was correctly falsified.
- **Subchunk splitting at block boundaries** (`ChunkData` `Subchunk`, `appendSubchunksToIndexes`,
  vendor `GzipChunkFetcher.hpp:380-396`). rg can resolve and emit *parts* of a chunk as
  block boundaries land, so a large late chunk does not block the writer as a single unit.
  gz has the subchunk machinery (`unsplit_blocks`, `append_subchunks_to_block_map`) but it is
  flagged as "scaffolding for the seekable-reader path" — **does gz's in-order consumer
  actually emit subchunks incrementally, or wait for the whole chunk?** If the latter, that
  is the structural divergence H-TAIL would point to. (Confirm by reading the consumer's
  write path, `chunk_fetcher.rs` consumer_loop, before claiming it.)
- **Block-finder look-ahead `3 × hardware_concurrency`** (`BlockFinder.hpp:212`). This keeps
  *boundary discovery* ahead of decode so workers never idle waiting for a boundary. gz's
  `async_block_finder` is the counterpart — verify its look-ahead depth. Only relevant if S2
  shows under-fill (low prior, given matched prefetch depth).
- **applyWindow / marker resolution** is ~0.1% in rg and gz already BEATS it (record) — NOT
  a silT4 target.

**Net:** the one structurally-open rg item that could fix a *confirmed* H-TAIL is
**incremental subchunk emission** (emit resolved sub-spans of a straggler chunk instead of
blocking the in-order writer on the whole chunk). That is the S4-TAIL port candidate — but
ONLY after S2+S3 confirm H-TAIL is the mechanism.

---

## 6. THE CONFOUND RED-TEAM (per step)

| Step | The trap | How this design rules it out |
|---|---|---|
| S0/S1 | Inheriting 1.165 from a different sha; stale `/dev/shm` pin; sink phantom. | Re-measure at a fresh, sha-logged, identically-flagged binary; /dev/null both arms; tmpfs≈null measured on one cell. |
| **chunk-size** | The graveyard: shrinking chunks to balance the tail co-varies output-buffer cache residency + block-finder overhead → already measured −20% at silT4. | **Never sweep chunk size.** The tail oracle (Tool 3) removes imbalance at *fixed* chunk size. |
| S2 effcores | Attribution-as-verdict (bias #7): "effcores low ⇒ tail is the lever." | effcores only ROUTES; the verdict is S3's perturbation. Also compare gz vs **rg** effcores (rg may share the tail shape). |
| S2 timeline | Span double-count / nested-span phantom (the 62ms ghost). | Tool-1 conservation gate dedupes nested spans + reconciles to drive wall, else refuses to print. |
| S3-TAIL oracle | Inert oracle silently measuring production (the SEED_WINDOWS `=1` no-op). | HITS counter must fire >0 and scale; sha==zcat at identity. |
| S3-TAIL ceiling | Ceiling-as-prize (the balanced-partition oracle CHEATS — knows expansion a priori). | Labeled CEILING; prize ≡ measured Δ of the byte-exact S4 port only. |
| S3-KERNEL | Mistaking a kernel surplus for a schedule bug (the prefetch A/B failure is consistent with BOTH). | The discriminator IS S2+S3: effcores≈T + cyc/byte≫rg + flat tail-removal ⇒ kernel, not schedule. |
| All | "finally / the lever is / at-floor" narration (bias #1/#3). | Every node is a HYPOTHESIS + its perturbation; advisor-before-banking on the FORK; re-confirm don't re-derive the prior "at-floor" prose. |

---

## 7. WHAT NEEDS AMD FOR LAW (defer, don't block)
- **Every silT4 verdict is Intel i7-13700T single-arch ⇒ NOT-YET-LAW** (Gate-3). solvency
  (`root@192.168.7.222`, Zen2) replicates. A scheduling/tail fix is unlikely to invert on
  Zen2 (no PEXT/PDEP in the consumer), but the cyc/byte arm of S3-KERNEL touches the marker
  path — **grep the hot path for `_pext_u64`/`_pdep_u64`/`bzhi` before trusting cross-arch
  portability** of any kernel conclusion. If solvency is offline, AMD is the ONE deferred
  step; it does not block S0-S4 on Intel.

---

## 8. THE FIRST STEP — leader brief

**Goal of the day:** build Tool 1 + Tool 2, pin a subject binary, and produce ONE
trustworthy silesia-T4 line: `gz/rg ratio (Δ vs spread) | effcores (gz, rg) | tail-idle`.
**No fix, no oracle, no S3 today.** That one line selects H-TAIL vs H-KERNEL and decides the
whole tree.

1. **Pin the subject.** Default: `kernel-converge-A @ c91aa125` (HEAD; contains the view-list
   win). Cherry-pick `c47a19c2` (standing.sh) onto it, or build the subject at the rig's
   commit. Log the sha; log `is-ancestor 300e772b` (expect TRUE).
2. **Build Tool 1** (`scripts/parallel_sm_tail_metric.py`) over `GZIPPY_TIMELINE` output:
   effcores, tail-idle, last-wave profile, decode-time variance — with the conservation +
   non-inert Gate-0 asserts. Validate it on one captured silT4 trace (chunk-count
   cross-check vs `--verbose`).
3. **Wire Tool 2** (effcores column into `standing.sh`'s silesia-T4 cell).
4. **Run S0 Gate-0** (rg self-test ≈1.0, /dev/null both arms, path=ParallelSM, sha==zcat,
   fresh identical builds) then the silesia-T4 cell N≥13 interleaved.

**Deliverable the leader returns:** the subject sha; the Gate-0 pass/fail log; the
silesia-T4 line (ratio+spread, gz/rg effcores, tail-idle); and which branch S2 selects
(H-TAIL → S3-TAIL, or H-KERNEL → S3-KERNEL). Nothing is banked as a finding without the
Gate-0 log attached.

---

## TL;DR
1. **Re-establish first.** silT4=1.165 is at `805c03c0`, not HEAD `c91aa125` (and the
   standing rig isn't even in HEAD's tree). The view-list win IS present (300e772b ancestor),
   so this is a real residual, not a missing merge.
2. **The cheap fixes are dead twice over:** prefetch-ordering wall-falsified; chunk-shrink
   measured −20%. Chunk size is a confound — never sweep it.
3. **The unrun discriminator is the work:** effcores + tail-idle (gz vs rg) separates
   H-TAIL (fixable schedule gap → port rg's incremental subchunk emission) from H-KERNEL
   (per-chunk decode surplus = the T1/igzip front, out of T≥2 scope). The prefetch A/B
   failure is consistent with BOTH — that is why no one has actually localized this.
4. **Build the tail-metric reducer + standing-rig column FIRST** so each iteration prints the
   fork-deciding number in one command. Then the tail oracle for the Gate-2 causal arm.
5. **Prize ≡ measured Δ of one byte-exact port, never the gap-to-rg or the oracle ceiling.**
   AMD/Zen2 owed for LAW on whatever Intel concludes.
