# gzippy encoder campaign — handoff

**Date:** 2026-08-17 (updated 2026-08-19)  
**Done when:** **0 / 1320** failing GATE promotion-board cells (size AND wall, per-label, T1 and T>1).  
**Not done when:** a session summary, a partial lever, or a hand measurement without `make lever`.

**Fresh board, commit `06b2e231`, this branch: 11 of 1320 failing** (was 13 at `86c19fc5`
earlier the same day, was a stale 36 on main). dd79_bin6 L3 T1 closed this session. Residual:
dd79_bin6 L3 T4 (tried, REGRESSES elsewhere — see below), photo.jpg L1/2/3 T4 (real fix found,
PARKED — see below, not merged), weights.safetensors L4/L7-9 (untouched, tiny).

---

## ⛔ TRIED AND DISCARDED, 2026-08-19 (never committed) — dd79_bin6 L3 T4 hash3-chain-repair
## does NOT safely generalize from T1 to `params_parallel`; confirmed pass→fail elsewhere

Applying the SAME fix that closed `gzip:dd79_bin6:L3:T1:size` (T1's hash3-chain-repair pick-min
arm, landed `804fdbb0`) directly to `params_parallel(3)` (the T>1 path, unconditional parameter
change, not a pick-min arm) looked genuinely promising going in: unlike the T4-seam fix, this is
an IN-PLACE matchfinder change inside the ONE parse T>1 already does, not a redundant second
encode — so no comparable wall-cost class was expected, and that held (dd79_bin6 L3 T4: 17%
slower than parent but still 1.27x FASTER than pigz, no wall regression; dickens L3 T4 exactly
unaffected). Closed the target cell cleanly: 4,452,034 → 4,426,383 B, now beats both gzip
(-10,039) and pigz (-15,531).

**But a full-corpus check (all 22 files, not just the target) found 5 files regress at L3 T4:**
dd79_text6 +5 B, engine.wasm +55 B, markup.xml +2 B, monorepo.tar +620 B,
**weights.safetensors +5,926 B**. Checked each against all three rivals for a pass→fail flip
(tie-guard does NOT catch this — it only checks EXACT libdeflate ties, and this was a close win
with a 2,252 B margin, not a tie): **`libdeflate:weights.safetensors:L3:T4:size` flips PASS
(83,080,006 ≤ 83,082,258) → FAIL (83,085,932 > 83,082,258).** A real, disqualifying regression
on a currently-passing cell — clause 3 is absolute regardless of the dd79_bin6 win.

**Tried a quick depth sweep (4/8/16/24 vs the shipped 32) hoping this was a magnitude problem,
not a direction problem — it is NOT rescued by any depth:** at `hash3_chain_depth=4`, dd79_bin6
still wins comfortably (4,426,504) but weights.safetensors is STILL worse than both libdeflate
and its own parent (83,086,176), barely different from depth 32's regression. This is CLAUDE.md's
own documented lesson recurring on a NEW mechanism (the `level.rs` engine.wasm L8 depth-scaling
receipt: "a deeper chain changes the PARSE, not merely the quality of one match... no multiplier
rescues it") — the regression is about LINKING the chain at all, not how deep it's walked.

**Discarded (never committed — this was an uncommitted local edit in a throwaway worktree, `git
checkout` cleanly reverted it, nothing to revert-commit).** Making this an ADDITIVE pick-min arm
(matching T1's safe pattern, instead of an unconditional parameter change) would restore the
"never worse" guarantee, but reintroduces the exact redundant-second-encode wall-cost class that
already sank the T4-seam fix above — not attempted, likely the same trap. Revival needs either a
cost-aware per-chunk decision that doesn't require a full trial encode, or accepting
weights.safetensors' loss is bounded and outside this specific promotion cell's scope (it is
not — L3 T4 vs libdeflate on that exact file is the flip). `dd79_bin6:L3:T4` stays open.

---

## ⛔ PARKED, NOT MERGED, 2026-08-19 — T4 seam fix (`agent/cursor-t4-seam` @ `ebd7f556`): real
## mechanism, real size win, CONFIRMED DISQUALIFYING wall regression against the actual rival

cursor-agent (driven as an implementation task, see the "LANDED" section below for the pattern)
correctly diagnosed the `photo.jpg`/`data.csv` T4 seam: T4's `params_parallel` path is missing
the gzip-shaped pick-min arm T1 already uses, so chunked encodes lose match discovery on
near-incompressible/low-ratio content. Built a per-chunk 2-way pick-min (`params_parallel` vs
`params_l{2,3}_gzip_deflate_fast`), then — after I measured a 1.63-2.02x wall regression against
its OWN PARENT and sent it back — added a size-ratio dead-zone gate (skip the extra arm when
the cheap arm's own output ratio is 20-50%, i.e. prose-shaped) that fixed `dickens`'s wall cost
back to noise-level.

**But `dickens`'s fix worked by giving up its size win entirely (independently confirmed: `wc -c`
shows dickens L2/L3 T4 output is BYTE-IDENTICAL to parent, delta=0 — the dead zone excludes it
from the extra arm, so it gets none of the benefit). The cells that DO keep their real size win
(`photo.jpg`, `data.csv` — both outside the 20-50% dead zone since one is ~99% and the other
~14%) still pay the full 2x-per-chunk cost, unresolved.**

**And critically — my own first wall check only compared the fix against its own PARENT (our
prior code), never against the actual RIVAL. Checked separately, directly:**

    L2 photo.jpg T4: parent 26.2ms BEATS pigz -p4 39.1ms (1.50x faster) -- PASSING today
                      fixed  53.7ms LOSES to pigz -p4 39.4ms (1.36x slower) -- WOULD FAIL
    L2 data.csv T4:  parent 30.5ms BEATS pigz -p4 37.7ms (1.23x faster) -- PASSING today
                      fixed  52.1ms LOSES to pigz -p4 37.7ms (1.38x slower) -- WOULD FAIL

**This is not "fails to help an already-failing wall cell" — it is a CONFIRMED pass→fail flip
on two cells that currently WIN against pigz at T4, exactly what CLAUDE.md's clause 3 treats as
absolute regardless of any size improvement elsewhere. NOT MERGED into the main lever branch.**
`agent/cursor-t4-seam` worktree/branch left as-is (not deleted, not pushed) for revival — per
CLAUDE.md "PARK monotone work; never DELETE it," the mechanism finding and the size-side numbers
are real and valuable even though this implementation can't ship. Revival needs a genuinely
cheaper way to decide "try the gzip arm" than running a full trial encode (already tried and
rejected by the agent: nested `thread::scope` inside T>1 — oversubscribes; gzip-first probe;
file-level classification) — or accepting the cost only where a wall census proves real T>1
slack absorbs it, which these two specific cells (already winning, thin margin) evidently do not.

**Process note:** the discipline of independently re-verifying a delegated agent's numbers (not
just its own report) caught this — the agent's OWN wall check was honest and correctly scoped
to what I'd asked for (before/after its own change), but I hadn't asked it to check against the
rival, which is the actual bar. My prompt's gap, not the agent's error — logged so the next
delegated wall-cost check asks for the rival comparison explicitly, not just parent-vs-after.

---

## ✅ LANDED, 2026-08-19, commit `804fdbb0` — dd79_bin6 L3 hash3-chain-repair, driven through
## cursor-agent (real implementation work, not review), independently re-verified before merge

Per the user's directive to drive expansive implementation work through `codex exec` and
`cursor-agent` (not just review), redirected the dd79_bin6 L3 residual investigation to
cursor-agent in an isolated worktree (codex itself hit its usage quota and did no real work —
see below). Result, in full:

**Real bug found, not a parameter tweak:** `HcMatchfinder::skip_bytes` overwrote `hash3_tab`
heads during match interiors WITHOUT linking `next3_tab` — leaving stale length-3 chain links
that `longest_match`'s own chain walk expected to be able to traverse. gzip links every inserted
position (vendor-precedented). Fixed as a new, purely-additive pick-min arm
(`params_l3_gzip_hash3_chain_repair`: `maintain_hash3_chain: true` + gzip's L3 chain depth 32),
gated by a new `LevelParams` field defaulting `false` everywhere else — cannot touch any other
level's or arm's output.

**Numbers (independently re-verified by me, not just trusted from the agent's report):**
`gzip:dd79_bin6:L3:T1` — 4,448,460 B (+0.271% vs gzip, FAILING) → 4,431,260 B (−0.116% vs gzip,
**PASSING**), −17,200 B. Corpus-wide L3 win rate (`pickmin_arm_audit`): 3/23 (wins dd79_bin6
specifically; `l3_native` still dominant at 17/23, `l3_libdeflate_greedy` still 0/23 dead
weight).

**Verification chain, every step re-run independently on the merged branch, not just taken from
the agent's report:** `cargo build --release` clean zero warnings; `cargo test --release` full
suite green, zero pin regeneration; direct `wc -c` size check matched the agent's number
exactly; `tie-guard.sh --levels 1,2,3,4,5,6,9` PASS (161 probed, 33 tied, 0 flipped) — run twice,
once in the agent's worktree, once again after cherry-picking onto the main lever branch.
Cherry-picked cleanly (`a3010bf6` → `804fdbb0`), pushed.

**Candidates correctly rejected, with reasoning, before landing on this one** (from `fulcrum
candidates gzip:dd79_bin6:L3:T1`, 24 applicable techniques read in full): [P14] early-exit on
non-improving candidates moves the WRONG direction for a match-COUNT deficit (can only find
equal-or-shorter matches, never more); insert-density/sparse `skip_bytes` was already falsified
earlier this session (see the retraction further down); [P10] `deflate_medium` overlap-fixup not
tried, lower priority since chain-repair already closed the named cell.

**Codex status: still exhausted (usage quota, resets ~Sep 17), did zero real work on its
assigned task before erroring out.** cursor-agent is now the sole practical driver for delegated
implementation work this session, per the user's explicit direction.

---

## ✅ L2-ONLY PARALLEL DISPATCH, 2026-08-19, commit `848b8924` — DUAL-ARCH VERIFIED before
## claiming anything, the discipline the section below was reverted for skipping

Built the narrower successor named in the retraction below: `deflate_one_shot_t1_l1_l2_parallel`
spawns L1's 2 arms + L2's 3 arms concurrently (5 threads), L3-L5 unchanged (original serial
fold). Byte-identity verified empirically (`cargo test --release`, zero pin/fingerprint
regeneration) before any wall measurement, same as always.

**Measured on BOTH architectures BEFORE writing this section, not after:**

    M1/aarch64 (this laptop), hyperfine paired:
      L2, 3 MiB incompressible: 96.5ms -> 52.2ms (1.85x)
      L2, dickens (12 MB):      185.8ms -> 103.9ms (1.79x)
      L1, 3 MiB (untouched code path, expect a tie): 51.8ms -> 52.0ms (1.00x, confirmed no
        regression from the unrelated code motion)

    x86_64 solvency (wall authority), hyperfine paired, run directly on the box against the
    exact arm binaries (not fulcrum's per-invocation timing, which has the jitter problem named
    below — hyperfine's own outlier detection handled the box's variable load fine, flagging
    "statistical outliers detected" on 2 of 4 runs but still reporting a clear, consistent
    signal each time):
      L2, dd79_bin6:  206.2ms -> 133.4ms (1.55x)
      L2, access.log: 250.5ms -> 146.8ms (1.71x)
      L2, dickens:    288.3ms -> 169.6ms (1.70x)
      L1, dd79_bin6 (untouched, expect a tie): 131.2ms -> 134.5ms (1.02x, within noise)

**Consistent, real win on both architectures, three different files each, L1 confirmed
unchanged on both.** This is the properly-scoped, properly-verified successor to the reverted
full-L1-L5 parallelization below — L2 alone, not extrapolated to L3-L5 without repeating this
exact dual-arch check first (that is precisely the corner the reverted version cut).

**Still not a `fulcrum try` adjudicated SHIP** — these are paired hyperfine numbers on two
machines, not the promotion-rule adjudicator. Given the box's own wall-census tooling has a
demonstrated jitter problem at short/fast T1 cells (documented below), getting an actual
`fulcrum try` wall verdict for this scoped version is the honest next step, not assumed from
these numbers alone — but the numbers are real, precise (tight enough for hyperfine's own
outlier detector to still call a clear winner), and now properly dual-arch, which the
FIRST version never was before being shipped.

---

## ⛔ RETRACTED AND REVERTED, 2026-08-19: the FULL parallel-dispatch "wall fix" was NOT a
## universal win — it is LEVEL-DEPENDENT, and net negative at 3 of 5 levels on the wall-authority
## box (superseded above by the properly-scoped, dual-arch-verified L2-only version)

Every section below celebrating commit `bc75535e` ("parallelize the T1 L1-L5 ratchet's arm
dispatch") as a 3.1-3.4x wall win is **WRONG AS STATED — read this section first, they are
left in place (quote-and-strike) as the record of what was believed and why it was wrong, not
as current fact.**

**What happened:** the 3.1-3.4x numbers were measured ONLY on this M1 laptop (aarch64). The
wall-census UNDECIDED investigation below (three attempts, all inconclusive from box noise)
prompted a DIFFERENT, more direct check: running `hyperfine` — a statistically rigorous tool,
NOT fulcrum's per-invocation timing — directly on the x86_64 solvency box (the project's
designated wall authority) against the exact two binaries `fulcrum try` had already built
there. That measurement is precise (tight std devs, no jitter, unlike fulcrum's VOIDs) and
reveals the fix is **LEVEL-DEPENDENT, not a universal win**, consistent across three different
files (dd79_bin6, access.log, dickens):

    L1: tie (1.00x)
    L2: BIG WIN, 1.59-1.83x FASTER, every file tested
    L3: LOSS, 1.19x SLOWER
    L5: LOSS, 1.04-1.08x SLOWER

This is net negative at 3 of 5 tested levels on the authoritative box — the opposite of the
"universal win" every earlier section claims. **REVERTED** (`git revert bc75535e`, commit
`0edecd3`) — CLAUDE.md: "a change that makes things worse gets reverted." Full test suite green
after the revert.

**Mechanistic read, not yet independently confirmed:** `User` time in every hyperfine run rose
sharply for the parallel form (e.g. dickens L5: 443ms serial vs 1845ms parallel, ~4.2x more
total CPU work) while wall time barely moved or regressed — the parallelism IS spreading work
across cores, but on this x86_64/Linux box the marginal thread-spawn/scheduling cost of L3's and
L5's ADDED arms (more concurrent OS threads) apparently outweighs their parallel savings, while
L2's smaller thread count (5 arms) stays a clear net win everywhere. Plausible but UNCONFIRMED:
whether this is pure thread-count overhead, cache/NUMA contention, or turbo-clock throttling
under higher concurrency — not measured, would need `fulcrum profile counters` on this box.

**What this does NOT retract:** the wall-cost MECHANISM this fix targeted is still real and
still open (the CRITICAL section further down: ~11 serial arm-encodes, 7.3x/8.7x slower than
libdeflate on the M1). The FOLD-SHAPE analysis (byte-identical equivalence proof for running
arms concurrently vs serially) is still correct and reusable. **The real, much narrower,
promising lead this session's correction surfaces: L2's parallel dispatch alone is a clear,
consistent win on the one architecture that matters for wall grading.** A future lever should
parallelize ONLY through L2's arm set (5 arms: L1's 2 + L2's 3) and keep L3-L5 serial as before
— NOT built here, this session ran out of room to design and re-measure it properly after
finding the problem. Any such lever needs BOTH architectures measured before being trusted,
which is exactly the discipline this correction is paying the cost of having skipped once.

**Process lesson, banked to memory:** a wall claim measured on only ONE architecture, however
carefully paired/locally rigorous, is not a wall claim — CLAUDE.md's own "SIZE IS ARCH-
INVARIANT; WALL IS ARCH-EXPOSED" was sitting right there and got skipped for the fast path
(believe the laptop, defer the box for adjudication) instead of the slow path (get a second
architecture's number before calling it a fix). The fulcrum UNDECIDED investigation below is
what indirectly forced the second-architecture check that caught this — a genuinely lucky save,
not a designed one.

---

## Wall census attempted on solvency, 2026-08-19: UNDECIDED (not NO-SHIP), box noise CONFIRMED
## by direct measurement, not inferred

`fulcrum try origin/lever/l3-gzip-deflate-fast-pickmin --threads 1 --levels 1-6` (`CAMPAIGN_
PROMOTE=1`, full 22-file corpus, x86_64 solvency box, commit `254a6dc1`). The box's `mpstat`
snapshot right before launch showed 99.78% system-wide idle (the vLLM tenant seen earlier via
`ps aux`'s cumulative-average CPU is GPU-bound with a bursty, not-continuous, CPU footprint) —
looked usable, so the run was launched rather than deferred again.

**Result: VERDICT UNDECIDED.** 528 wall cells: 281 VOID/VOID (both arms), 173 VOID(after)/OK
(base), 66 ABSENT/ABSENT (igzip — found via `/usr/bin/igzip`, a system package not the pinned
vendor build, `rival_bin()` doesn't reach it), only 6 OK/OK and 2 OK/VOID. The VOID reason on
inspected cells is `aa_bias` — an A-A self-consistency check (the SAME binary timed against
itself) failing, meaning the box's bursty tenant activity IS corrupting wall timing during the
run's actual multi-minute duration even though the single pre-launch snapshot looked clean.
**This directly confirms, by execution rather than inference, that this box is not currently
usable for a trustworthy wall verdict** — the earlier `ps aux`-based deferral this session was
right for the right reason, just under-evidenced; now it's evidenced. `fulcrum` self-protected
correctly (UNDECIDED + a re-run list, not a guessed SHIP or NO-SHIP) — the measurement
discipline did its job.

**The SIZE axis of this SAME run came back fully clean, independently confirming and EXTENDING
the earlier size-only verdict to a second architecture:** clause 3 "no pass->fail flips across
468 decidable cells," clause 4 "closed failing cell(s): gzip:photo.jpg:L3:T1:size," clause 6
"improvement 0.0263 vs harm 0.0000," **clause 7 "all required arch(s) covered: x86_64"** — the
earlier local size-only SHIP (commit `14fce138`) was aarch64-only (this laptop); this is the
first x86_64 confirmation of the same result. Artifact:
`/root/www/gzippy-bench/campaign/lever-origin-lever-l3-gzip-deflate-fast-pickmin/try.json`
(solvency box; not yet pulled to a synced location).

**Superseded — see the retraction at the top of this file.** ~~Next action for whoever
continues: re-run the SAME command once the box is verified quiet by a FULL-DURATION check (not
a single `mpstat` snapshot — this session's own miss), or once the vLLM tenant is gone. The
local hyperfine numbers (3.1-3.4x wall reduction, PLAN.md above) are still the best available
signal of DIRECTION and MAGNITUDE; they are not, and were never claimed to be, a substitute for
this adjudicator.~~ The M1-only numbers were WRONG (not just unadjudicated) — a direct x86_64
hyperfine check found the fix net negative at 3 of 5 levels, and `bc75535e` was reverted. The
UNDECIDED wall census below was for a commit that no longer exists on this branch; re-running it
against the reverted state would presumably resolve cleanly, but the branch has moved on from
needing that specific verdict.

**TWO FOLLOW-UP ATTEMPTS, same session, both converge on the same result — this is now
TRIANGULATED, not a single inconclusive run:**

1. Built a monitor polling the box's 5-min AND 15-min load averages every 90s (not another
   instantaneous snapshot — the smoothing this session's first attempt lacked), required 3
   consecutive reads under 4.0 (of 32 cores) before declaring quiet. Confirmed quiet
   (load5=1.80, load15=2.64) after ~5 minutes. **Re-ran the identical command: UNDECIDED
   again**, essentially unchanged (264 VOID/VOID, 188 VOID/OK, only 7 OK/OK of 528 wall
   cells — worse than attempt 1's 6 OK/OK, not better).
2. Hypothesized short/fast T1 cells need more statistical power against jitter, not just a
   quieter box: re-ran with `--scope 'levels=1,2,3,4,5' --n 45` (3x the default 15 samples,
   full measurement on the levels this change touches, L6 sentinel-sampled only). **UNDECIDED
   again, WORSE**: only 1 OK/OK of 452 wall cells (225 VOID/VOID, 175 VOID/OK) — MORE samples
   made it worse, not better, ruling out "insufficient n" as the fix too.

**Pattern across all three attempts: L6 (the one level this branch's change does NOT touch,
included as an unaffected control) mostly succeeds; L1-L5 (the exact coordinate the change
acts on, all shorter-duration T1 encodes at lower search depth) is almost universally VOID,
regardless of load-quiet or sample count.** This points at something more structural than
tenant contention — plausibly a timer-resolution or scheduling-jitter limitation of this
specific box/VM for SHORT operations specifically, which neither waiting nor resampling fixes.
This is itself a new operational finding worth carrying forward (not yet in
`reference_solvency_box_operational_lessons.md`): **this box may be fundamentally unable to
wall-adjudicate fast/short T1 cells with the current per-invocation timing method**, independent
of tenant load. A fix would need a different measurement strategy (e.g. many back-to-back
repeats of the same short op with an outer median, rather than per-invocation `--n`-sample
timing) — that is `fulcrum`-tool engineering, out of this branch's scope.

**Given three converging attempts, further retries on THIS box without a different measurement
strategy are not expected to resolve this.** The wall leg stays genuinely UNDECIDED. The size
leg is SHIP, confirmed on two architectures. Decision point for whoever continues, unchanged
in substance but now much better evidenced: merge size-clean with wall tracked as a
follow-up (needing either a quieter/different box or a fulcrum measurement-strategy fix), or
hold. This is a product/process call, not a measurement one — not resolved here.

---

## Fresh SIZE board, 2026-08-19, commit `86c19fc5`

`CAMPAIGN_LEVELS=1-9 make board-size-promote` (built igzip locally first — `make
vendor/isa-l/build/igzip`, missing on this laptop worktree until now — so all four rivals are
covered, not declared-absent): **13 of 1320 measured cells failing** (down from the stale
"36 failing" MEMORY.md board note). Artifact:
`/Users/jackdanger/www/gzippy-bench/campaign/size-all-86c19fc5/census.json`. Worst offenders:

    vs gzip: dd79_bin6 L3 T4 +0.35%, L3 T1 +0.27%, photo.jpg L1-3 T4 +0.04%, dd79_bin6 L2 T4 +0.03%,
             weights.safetensors L7-9 T1/T4 +0.00-0.02%
    vs pigz: dd79_bin6 L3 T4 +0.23%, L3 T1 +0.15%
    vs libdeflate: weights.safetensors L4 T4 +0.04%
    vs igzip: none (0 of 132)

**`dd79_bin6` L3 — insert-density hypothesis BUILT AND FALSIFIED, same session.** `fulcrum why
gzip:dd79_bin6:L3:T1` (structure layer only — lines/counters/params layers skipped, no
valgrind/Linux/anatomy-counters build on this host): gzip finds 1.85% MORE matches and 4.4%
FEWER literals than us. Memory (`project_len3_guard_dd79_mechanism.md`, PR #299, 2026-08-09)
named the L3 residual's cause as "unmeasured... plausibly insert-all vs skip_bytes." Read
`vendor/gzip/deflate.c`'s actual `deflate_fast` (confirmed the correct analog: gzip L1-3 all
dispatch to it, `deflate.c:686`): it is NOT dense — `max_insert_length` (`= max_lazy_match`,
`configuration_table[3].max_lazy = 6`) caps the interior-insert loop; matches longer than 6
bytes insert NOTHING for their interior, only resync the rolling hash. Our `HcMatchfinder::
skip_bytes` is unconditionally dense (a faithful libdeflate port) for every match length — the
OPPOSITE direction from every prior insert-density lever in this file (L1's `fast_hash_update_
inserts`/`fast_dense_interior_insert` family all went DENSER to match libdeflate, never sparser
to match gzip).

Built it exactly as scoped (new `LevelParams::hc_max_insert_length` field, default 0 = current
dense behavior everywhere; set to 6 ONLY on `params_l3_gzip_deflate_fast()`; new `HcMatchfinder::
resync_hashes` — `skip_bytes` minus its three table writes, same window-slide handling) —
**measured WORSE, not better**: `dd79_bin6` -3 standalone `gzip_deflate_fast` arm size rose from
4,448,442 B to 4,449,262 B (+820 B), and its corpus-wide win rate (`pickmin_arm_audit`) dropped
from 6/23 to 2/23. **REVERTED** (`git checkout`, back to `bc75535e`'s clean state — build/test
reconfirmed identical). This CLOSES the "insert-all vs skip_bytes, unmeasured" question from
memory with a definitive, scoped, measured NO: copying gzip's sparser insertion does not make
our matchfinder behave more like gzip's, because our OTHER heuristics (chain=32, nice=32,
Greedy strategy — all libdeflate-style, not gzip's) were apparently relying on the dense
insertion to find matches gzip's own compensating mechanism finds some other way. The
match-count gap's real cause is still open; this rules out ONE specific, previously-plausible
mechanism, not the whole class. `fulcrum candidates gzip:dd79_bin6:L3:T1` has 23 other
un-falsified techniques listed if this is picked up again — [P14] (zlib-ng's early-exit on
first non-improving candidate at levels <5) and [P10] (`deflate_medium`'s overlap-fixup search)
are the two that most directly touch match ACCEPTANCE rather than insertion and were not tried
this session.

**`photo.jpg` L1-3 T4 vs gzip (+0.04% each) — NAMED, not investigated: a real T4-only seam
residual, out of this branch's T1-scoped charter.** Direct check: our OWN T4 output is larger
than our OWN T1 output on this file by 3,604 B (L1), 11,257 B (L2), 10,229 B (L3) — and T1 is
already SMALLER than gzip on L2/L3 (the `#332`/`c8bbde67` closed cells), so the T4 seam cost is
large enough to erase that win and flip the T4 cell to a loss. MEMORY.md's board summary says
"the seam class is CLOSED (bit-splice merged in the Cursor wave)" — this file's T4 residual
contradicts that as stated, or names an exception the merged fix doesn't cover; not
reconciled. This is a T>1 mechanism (CLAUDE.md STEP 2, a separate code path from this branch's
T1 ratchet work, needing the causation/starvation tooling, not `fulcrum why`/`candidates`) —
named here as the next board-visible item, not pursued.

---

## 🛑 CRITICAL, MEASURED 2026-08-18 — the whole ratchet mechanism has a severe, likely-disqualifying wall cost, on BOTH routes

Discovered while responding to a fourth review's demand for real wall/RSS measurement of the
streaming fix (see "PROMOTION PAUSED" section below for that review's four findings — all
confirmed correct, but this is bigger than any of them). **This is not scoped to streaming.**
The mmap/file route from `b4b821c9` — reviewed clean by Fable, cursor-agent, and a prior
Codex pass — has the SAME problem, because it's the SAME mechanism (cumulative multi-arm
pick-min via `deflate_one_shot_t1_ratcheted`).

**Two independent measurements, both executed, neither noise:**

1. **Streaming route, directly measured (hyperfine, 15+ paired runs):** 3 MiB incompressible
   pipe input, `-5`. Branch: 295ms. Old code (main): 40ms. libdeflate: 34ms. gzip: 66ms. pigz:
   61ms. **The branch is 7.3x slower than our own prior code and 8.7x slower than libdeflate**
   — on an ordinary, common-shape request. 11x more instructions retired, 3x peak RSS.

2. **Mmap/file route, from the in-flight wall census** (`CAMPAIGN_OUT=/root/lever-b4b821c9-wall-3`,
   `--threads 1 --levels 1,2,3,4,5,6`, launched to validate the ORIGINAL ratchet from `b4b821c9`,
   not the streaming fix): **319 confirmed wall losses vs 75 confirmed wins** (81% loss rate)
   among `RESOLVED` (non-noise, non-VOID) cells so far, still running. Confirmed via the
   per-cell JSON artifacts (`a` = our gzippy, `b` = rival; `RESOLVED-a-slower` = we lost,
   re-verified against a `wall_class: WIN` example to be certain of the field semantics before
   concluding anything). Worst confirmed ratios: up to 29x slower than a rival. Multiple cells
   show the box's own 3x-n re-confirmation mechanism OVERRIDING an initial win into a confirmed
   loss (e.g. `gzip:weights.safetensors:L3:T1`: first pass 0.42x [win] → confirmed 2.43x
   [loss]) — this is exactly what CLAUDE.md's flip-confirmation exists to catch, and it is
   catching a real regression, not noise.

**Root cause (reasoned, not yet independently re-verified against a profiler — name this
before trusting it further):** `deflate_one_shot_t1_ratcheted` computes the CUMULATIVE union
of every level 1..N's own multi-arm pick-min to guarantee monotonicity. At L2 this is
~4-5 arms (L1's 2 + L2's 2-3); by L5 it's ~10-12 arms. Each arm is a FULL independent parse of
the input. On files small enough that a single rival pass is already fast (sub-100ms), running
10+ of our own passes to save a few hundred bytes is a wall-losing trade — the SIZE win the
ratchet buys is real and was correctly SHIP-verdicted, but nothing in this campaign's earlier
review chain (mine, Fable's, cursor-agent's, an earlier Codex pass) priced this cost before
now, because no one had run a real wall census against this exact ref until this session.

**What this means for the branch:** size-only promotion (the `--size-only` SHIP verdict from
earlier this session) is NOT sufficient on its own — CLAUDE.md's own bar is size AND wall, and
this looks headed for a wall NO-SHIP. Do not merge on the size verdict alone. Do not spend
further effort polishing streaming test coverage (the fourth review's findings #1/#3/#4) until
this is resolved — if the ratchet mechanism gets redesigned to cut arm count, those tests may
need to change again anyway.

**Options, not yet decided:**
1. **Cap the arm count / narrow the ratchet's cumulative scope** — e.g., each level compares
   against only level N-1's WINNING arm (not the full re-derivation this session's recursive
   and iterative implementations both do), which is a fundamentally different, cheaper
   construction than "compute everyone's own arms and fold." Needs its own design pass.
2. **Gate the ratchet on file size** — only pay the multi-arm cost above some threshold where
   the absolute wall cost stops mattering relative to rivals (small files are exactly where
   the RATIO looks worst even though the ABSOLUTE time is small). Same class of routing CLAUDE.md
   already permits (input-size, not content) but changes what "closes a size cell" means for
   small files.
3. **Revert to before `c8bbde67`** and re-approach the original L3 lever with wall priced in
   from the start, per the CLAUDE.md rule this session already learned the hard way once
   (run `cargo test` first) and is now learning again at a larger scale (measure wall before
   declaring a multi-arm mechanism safe, not after it's spread across two routes).

**DECIDED 2026-08-19, after Fable + cursor-agent's independent wall-cost design reviews (both
converged: no free bug, cost is intrinsic to "every level tries every arm and folds"; option 1
above as literally stated does not work — proven by counterexample that comparing only against
level N-1's own arm breaks transitivity). Two facts, BOTH independently verified by direct
execution before trusting them, reframe this away from "revert vs redesign":**

1. **`origin/main` (e888ac9f) — already merged, unrelated to this branch — ships a REAL ladder
   violation on the REAL corpus, not just synthetic fixtures.** `photo.jpg` T1: L2 = 6,462,189
   (the actual `#332` SHIP-closed cell), L3 = 6,472,401 — L3 is 10,212 B LARGER than L2, on the
   binary users actually get today. Caught and fixed a real trap while verifying this: my first
   check used a stale `~/www/gzippy` binary still reflecting an old build; `find src -newer
   target/release/gzippy` caught it, a forced rebuild changed the answer completely. **Always
   verify the binary**, this session's own rule, paid off again. `ladder_is_monotone_t1` only
   ever tested 1 MiB synthetic fixtures, never the GATE corpus, so this was invisible until now.
2. **PR #332's wall leg (the L2 pick-min, already merged) was never run** — this file's own
   "Where we are" table said so from before this session started ("Wall lever on #332: NOT
   RUN"). The wall-cost problem may predate `c8bbde67` and this whole branch; option 3 (revert)
   is not obviously sufficient to restore wall solvency on its own.

Given (1), paying 7.3x wall for an "unconditional" guarantee that main doesn't even uphold
beyond synthetic fixtures today is a bad trade. ~~**Adopted: Fable's "rung + nested bonus"
construction**~~ — **RETRACTED, same session, 2026-08-19: built, and the proof's own stated
premise (`rung(N) <= rung(N-1)`) was FALSE on the first real fixture checked — `binary` L2
(663,410) > L1 (662,577), reproduced 3x on a freshly-verified binary. See "IMPLEMENTED THEN
FALSIFIED AND REVERTED" below for the numbers and why there is no cheap patch inside this
fold shape. Left the description below intact — quote-and-strike, not delete — because it's
still the correct record of what was tried and why it looked sound going in.** — split each
level's arms into `rung(N)` (the level's own native, homogeneous
arm(s): Fast/Greedy/Lazy/zlib-depth family — the class that has never been observed to sag
below L6) and `bonus(N)` (the "gzip-shaped forced-min-match-3" vendor arms that are the ONLY
class ever observed to cause a sag — `gzip_primary`, `gzip_deflate_fast` at L2/L3,
`libdeflate_greedy` — nested so `bonus(N) ⊇ bonus(N-1)` always). `f(N) = min(rung(N), min(bonus(N)))`.

**Proof sketch (Fable):** `bonus(N) ⊇ bonus(N-1)` makes `min(bonus(N))` unconditionally
non-increasing in N, and `min` is monotone in both operands, so IF `rung(N) <= rung(N-1)`
holds (not proven, but true everywhere ever measured below L6, and now test-pinned by this
session's own RED-then-GREEN tests), THEN `f(N) <= f(N-1)` follows — the sag-causing class
(heterogeneous vendor arms) becomes structurally impossible to blame, and the residual risk
moves entirely to the native/rung class, which is empirically monotone and now actively tested,
not just hoped.

**Honest framing, stated explicitly per Fable's own flag:** this converts the guarantee from
"unconditional over all inputs" (a promise main doesn't even keep today) to "structural for the
one class that has ever broken it, test-pinned for the rest." That is a spec change, not merely
an optimization — recorded here as a deliberate choice, not a silent downgrade.

**Cost, conservative version being implemented** (keeps ALL current vendor arms in `bonus`,
does not yet drop `libdeflate_greedy` or unify chain8/chain32 — that thinning is Fable's
measured-not-guessed follow-up, `PICKMIN_ARM` win-rate audit under `anatomy-counters`, not done
here): L1 2 arms, L2 3, L3 5, L4 5, L5 5 — vs the current ratchet's 2/5/8/10/12. Real reduction,
not the full ~1.5x Fable projected with arm thinning, but zero risk of reopening a closed cell
since every arm that currently runs still runs, only the CROSS-LEVEL comparison scope narrows
(this level's own rung vs the cumulative bonus set, not every prior level's full arm set).
**This DOES change output bytes on some inputs relative to the current ratchet** (it no longer
compares against prior levels' native/rung arms, only their bonus arms) — tie-guard and a full
size-only lever re-run are required, not assumed safe by construction alone.

**Next action: get the actual `fulcrum try` verdict (SHIP/NO-SHIP/UNDECIDED with clause
numbers) once the census finishes** — the tallies above are this session's own log-reading,
useful for orientation but not the adjudicator. Do not decide among the three options above
without it.

**IMPLEMENTED THEN FALSIFIED AND REVERTED, same session, 2026-08-19.** `t1_rung_arms` +
`t1_bonus_arms` were built and wired in (kept the name `deflate_one_shot_t1_ratcheted`, no
call-site changes). Build was clean. Test suite immediately caught the problem before it went
anywhere near a box: `block_pins`/`fingerprint_suite` both failed on `binary:L2:T1` — expected
per the design's own "this DOES change output bytes" warning, so the pins were regenerated —
but the NEW value, `binary L2 T1 = 663,410`, is LARGER than `binary L1 T1 = 662,577`
(+833 B, +0.13%). **This is a ladder-monotonicity violation, not a pin-format artifact.**

Verified directly, not inferred from the pin diff: fresh `cargo build --release`, ran
`target/release/gzippy -1 -p1 -c` and `-2 -p1 -c` against the exact `fixtures::generate("binary")`
bytes (sha256 `5471319620839130044196dec14ef6c5ec49fefe57a5bbe69aec86f9f9e0d0e`), 3x each,
byte-identical every run: L1 = 662,577, L2 = 663,410. Deterministic, reproducible, on the
binary that was actually measured.

**Root cause: the proof's own stated premise — `rung(N) <= rung(N-1)`, which Fable's proof
sketch explicitly flagged as "not proven, but true everywhere ever measured below L6" — is
FALSE on this exact fixture at N=2.** `rung(2)` (`params(2)`'s own native parse) plus
`bonus(2)`'s two vendor arms all came in above `f(1) = 662,577`. This is CLAUDE.md's own
standing warning (`level.rs`'s engine.wasm L8 receipt: "a longer search can itself produce a
LARGER output") landing on the exact premise the whole construction rested on, on the very
first fixture checked. Not a coding bug — `cargo build` was clean, the arm lists matched the
design exactly — the DESIGN's mathematical premise was wrong, and "true everywhere ever
measured" turned out to mean "measured nowhere below L6," because nobody had actually run
`ladder_is_monotone_t1` against this construction before this check (the test suite runs
alphabetically by target name and stopped at `fingerprint_suite`, before reaching
`size_invariants.rs` — the pin-diff investigation caught this ahead of that test, not because
of it).

**SHARPER, 2026-08-19 (cursor-agent adversarial re-review of this exact section — codex
independently confirmed the audit methodology but hit its own usage limit before reaching a
verdict on this claim): the proof sketch was broken even INDEPENDENT of the falsified premise.**
Bonus nesting only gives `min(bonus(N)) <= min(bonus(N-1))` — it says nothing about `f(N-1)`
when `f(N-1)` was won by the RUNG side, not bonus, which is exactly what happened here:
`f(1) = 662,559` came from `rung(1)` (`l1_native`), while every `bonus(2)` arm sat at
663,392+ (`l2_gzip_primary`, the best bonus arm at L2, is barely below `l3_gzip_deflate_fast`
and still above `f(1)`). So even a hypothetically-true `rung(N) <= rung(N-1)` would not have
saved this case on its own — the full statement needed was `min(rung(N), min(bonus(N))) <=
min(rung(N-1), min(bonos(N-1)))`, and nesting alone never implies the RHS's rung term is
covered. Mechanistically: `rung(2)` (`params(2)`, `Strategy::Greedy`, depth 6, nice 10,
`far_len3_gate`, `greedy_len3_shadow`) parses the len-3-rich, ELF-record-like `binary` fixture
WORSE than `rung(1)` (`params(1)`, `Strategy::Fast`, depth 1, chainless one-pass) — confirming
CLAUDE.md's own standing warning in mechanistic, not just statistical, terms. Independently
reproduced per-arm sizes (unpadded, same fixture): `l1_native` 662,559, `l2_gzip_primary`
(L2's own best bonus arm) 663,392 — so `f(2) >= 663,392 > f(1) = 662,559` regardless of exactly
which bonus arms are nested in. **Same verdict (revert was correct, no cheap fix exists), a
strictly weaker and more defensible reason than "one inequality happened to be false" — the
construction's proof had a gap a correct premise couldn't have closed either.**

**Why there is no cheap fix inside this construction:** making `bonus(N)` also absorb
`rung(N-1)` (so a future level's bonus set structurally dominates a past level's rung choice
too) sounds like a small extension, but tracing it through shows `bonus(N)` would then have to
equal the full union of every arm — rung AND vendor — ever tried at any level `< N`. That IS
the old cumulative ratchet's exact cost shape (its per-level own-arm counts, 2/3/3/2/2 summed
level-by-level to reach L5 = 12 total encodes to compute `f(5)`), not a cheaper one. The
`t1_rung_arms`/`t1_bonus_arms` split's real (and only) saving was skipping that cumulative
walk — computing `f(N)` standalone from just that level's own rung+bonus, independent of any
other level. That standalone-ness is exactly what makes `rung(N) <= rung(N-1)` load-bearing,
and exactly what a falsified premise there breaks with no partial-credit fallback: any fix that
restores the guarantee by folding in predecessor arms un-does the saving that was the entire
point.

**Reverted to the known-correct cumulative ratchet** (`git checkout -- src/compress/deflate/mod.rs
src/compress/deflate/parse/mod.rs tests/fingerprints/ours_blocks.tsv`) — CLAUDE.md: "a change
that makes things worse gets reverted." Re-verified post-revert on the same fixture, same
binary-rebuild discipline: L1=662,577, L2=662,577, L3=661,068, L4=657,593, L5=657,593 — monotone
non-increasing, confirming the revert actually restored the safe baseline and this wasn't a
build-staleness illusion in either direction.

**Where this leaves the wall-cost problem: still open, but the fix has to live INSIDE the
proven-correct cumulative fold, not in a cheaper fold shape.** The one direction from the
original two-reviewer design brief that this finding does NOT invalidate is **1a — thin the
per-level ARM SET, not the fold structure**: the cumulative ratchet's cost is a sum over levels
of each level's own arm count (2/3/3/2/2 = 12 total by L5); cutting that per-level count (e.g.
dropping `libdeflate_greedy` or unifying the chain8/chain32 `gzip_deflate_fast` variants) cuts
the SAME total proportionally, with zero change to the correctness argument, because the fold
shape (`if cur.len() < best.len() { best = cur }` walking every level) is untouched — only
which arms feed `cur` changes. This needs the win-rate measurement Fable named
(`PICKMIN_ARM` attribution under `anatomy-counters`) before dropping anything, not a guess.
**Next action: run that attribution, on the GATE corpus, per level** — which arms ever win,
and how often — before proposing which to drop.

**DONE 2026-08-19: built `pickmin_arm_audit` (`src/compress/deflate/mod.rs`, `#[cfg(test)]`,
`#[ignore]`d — `cargo test --release --lib pickmin_arm_audit -- --ignored --nocapture`) and ran
it against the real 23-file GATE corpus, T1, levels 1-5.** Calls the exact same private encode
functions the shipped pick-min functions call (can't drift from what ships), so this is COUNTED
not inferred, per CLAUDE.md. Win-rate table (N/23 files where that arm produced the level's
minimum; ties count for both):

    L1  l1_native 21/23 (91.3%)          l1_gzip_primary 2/23 (8.7%)
    L2  l2_native 11/23 (47.8%)          l2_gzip_primary 2/23 (8.7%)   l2_gzip_deflate_fast 10/23 (43.5%)
    L3  l3_native 17/23 (73.9%)          l3_libdeflate_greedy 0/23 (0.0%)   l3_gzip_deflate_fast 6/23 (26.1%)
    L4  l4_native_greedy 1/23 (4.3%)     l4_lazy 22/23 (95.7%)
    L5  l5_baseline 3/23 (13.0%)         l5_zlib 21/23 (91.3%)

**One clean, zero-risk cut: `l3_libdeflate_greedy` — 0/23, never once the reason an L3 cell is
closed.** Removed from `deflate_one_shot_t1_l3_pick_min` (L3 now 2 arms, not 3; cumulative cost
to reach L5 now 2+3+2+2+2=11, was 12). Kept the function itself (`level.rs`, `#[allow(dead_code)]`,
still `pub`) so the audit keeps exercising it — a future nonzero count is the signal to
reconsider, not a hand-argument. `l3_libdeflate_greedy` having never won is itself informative:
it is L3's ONLY arm with no matching "native gzip-shaped forced-min-match-3" characterization —
`l3_gzip_deflate_fast` already covers that vendor-shape class at L3 with a cheaper/differently-
tuned config, and 0/23 says libdeflate's specific greedy-depth-12 shape adds nothing L3's other
two arms don't already reach on this corpus.

**Every other arm has a nonzero, sometimes-substantial win rate — none of the rest are safe to
drop outright on this data.** `l4_native_greedy` (1/23, `weights.safetensors` only) and
`l5_baseline` (3/23) are the next-thinnest margins but are NOT zero — dropping them needs either
a margin-size check (how many bytes would those cells actually lose, CLAUDE.md's margin-floor
precedent) or acceptance that this is a genuine size/wall tradeoff, not a free cut. Not pursued
further this session — the one unambiguous, zero-risk cut is banked; the rest is a real tradeoff
question, not a counting exercise, and deserves its own pass rather than being rushed here.

**Honest scale check: this alone does not solve the 7.3x/8.7x wall problem.** One arm off L3
cuts the cumulative-to-L5 total from 12 to 11 (~8%) — nowhere near enough. The wall-cost
question (this file's "CRITICAL" section at top) remains genuinely open; this is one small,
zero-risk piece of the "thin inside the correct fold" direction, not a resolution of it. Full
`cargo test --release` re-run: green, ZERO pins regenerated, confirming byte-identity as 0/23
predicts. `tie-guard.sh HEAD --levels 1,2,3,4,5,6,9`: **PASS** — 161 cells probed, 33 tied, 0
flipped.

**Adversarially reviewed 2026-08-19** (codex exec + cursor-agent, parallel, per standing
instruction — extended to codex this round at the user's request): codex hit its own usage
limit mid-run (no verdict, but confirmed the audit's L3 arm list matches the pre-removal
shipped function before cutting off). cursor-agent completed both claims as **VERIFIED WITH
CAVEAT**: independently reproduced the `binary` fixture numbers and the 0/23 count, confirmed
the audit is methodologically equivalent to the pre-removal shipped pick-min, confirmed full
suite + zero pin regen — and found a SHARPER root-cause for the falsified redesign than this
file's own first framing (see "IMPLEMENTED THEN FALSIFIED AND REVERTED" above, now corrected
in place). Honest limit named on Claim 2: 0/23 closes a counting question on 23 files, not a
universal theorem — accepted, the kept `#[allow(dead_code)]` audit arm is the sentinel for a
future corpus shift.

**`fulcrum try HEAD --size-only`, `CAMPAIGN_LEVELS=1-9`, full 22-file corpus, 594 decidable
cells (gzip+pigz+libdeflate; igzip declared absent — not built on this laptop worktree, and
this change is a proven byte-identical no-op so it cannot possibly show anything igzip-specific
new): VERDICT SHIP.** Every clause passes: clause 2 (arms differ), clause 1 (zero roundtrip
failures), clause 3 (**zero pass→fail flips across 594 cells**), clause 4 (**closes
`gzip:photo.jpg:L3:T1:size`** — one of the three cells this branch's earlier L3 work was
supposed to hold, confirmed still closed against the `origin/main` baseline), clause 5 (every
passing cell inside its erosion budget), clause 6 (improvement 0.0033 vs 0.0000 confirmed-real
harm), clause 7 (aarch64 covered), clause 8 (paired-method, n fixed before the run). Artifact:
`/Users/jackdanger/www/gzippy-bench/campaign/lever-HEAD/try.json`.

**This is a real, adjudicated, banked SIZE-leg result for the whole branch (`e888ac9f` ..
`217438d5`), not just today's thinning.** It is NOT a full promotion verdict — CLAUDE.md's bar
is size AND wall together, `--size-only` explicitly does not grade wall, and the wall leg (the
7.3x/8.7x regression named in this file's own CRITICAL section) remains genuinely unresolved:
today's one arm cut is ~8% of the cumulative arm cost, nowhere near enough on its own. **Per
CLAUDE.md ("land cleared work before starting a new lever"), the size leg being SHIP-clean is
itself worth banking now** (PR opened / updated with this verdict) rather than held hostage to
the still-open wall question — but merge-readiness needs the wall leg resolved or an explicit,
named decision to ship size-only with wall tracked as a follow-up, which is a call for the next
session or the user, not something to resolve by continuing to add levers unchecked.

**⚠ WALL LEG, 2026-08-19 (commit `bc75535e`) — SEE THE RETRACTION AT THE VERY TOP OF THIS FILE.
The "MECHANISM FOUND AND FIXED" framing below was PREMATURE: `bc75535e` was REVERTED (commit
`0edecd3`) after a second-architecture check found it net negative at 3 of 5 levels. Left
in place as the record of the reasoning, not as current fact.** Diffed what the ratchet's own
code was doing against what it NEEDED to do (the
`/goal` directive's own framing): every prior version computed each level's own arms then
folded the FINISHED result before starting the next level's arms — a purely SERIAL chain of
~11 independent O(n) parses to reach L5, on a 10-core laptop. Nothing about correctness ever
required that seriality (every arm is independent; none reads another's output) — it was an
artifact of the code's shape, not the algorithm. Rewrote `deflate_one_shot_t1_ratcheted` to
spawn every needed arm as its own thread via `std::thread::scope` and fold the results with the
IDENTICAL strict-`<` tie-break composition the serial form used (proof in the function's own
doc comment: three strict-`<`-ties-to-earlier-operand folds compose into "first element in a
fixed canonical order to hit the global minimum wins," and parallel execution changes WHEN each
arm is computed, never WHAT the fold compares).

**Verified, not assumed:** `cargo build --release` clean; `cargo test --release` full suite
green with **ZERO pin/fingerprint regeneration** — confirms the equivalence proof empirically.
`tie-guard.sh HEAD --levels 1,2,3,4,5,6,9`: PASS, 161 probed / 33 tied / 0 flipped.

**Measured locally (hyperfine, paired, this machine, before any box time — cheapest falsifier
first):**

    3 MiB incompressible, -5, T1: 259ms (serial) -> 76ms (parallel), 3.4x faster.
                                   libdeflate gap: 8.22x -> 2.42x.
    dickens (12 MB real text), -5, T1: 678ms -> 219ms, 3.1x faster.
                                   libdeflate gap: 5.25x -> 1.70x.
    4 KB incompressible, -5, T1: 3.5ms -> 3.0ms, no regression (noise-level either way).

Consistent, large win across synthetic incompressible, real compressible content, and small
files (no thread-spawn-overhead regression found at any size tested). **Does NOT fully close
the gap to libdeflate** — still 1.7-2.4x slower, down from 5.25-8.22x. This is a SCHEDULING fix,
not a reduction in total CPU work: `User` time in the 3 MiB case rose slightly (433ms -> 483ms,
thread overhead), while `Wall` time fell 3.4x. Peak memory rises from ~3 buffers to up to 11.

**BLOCKED on box adjudication, not on more work here:** the solvency box
(`root@10.0.2.240` via `-o ProxyJump=neurotic`) is reachable but NOT frozen right now — `ps aux`
shows a 4-worker vLLM inference job (`VLLM::Worker_TP0-3`) at ~400% combined CPU, load average
5.7-7.0. Per CLAUDE.md's own repeated wall-measurement discipline (paired, frozen box, "a
freshly rebooted/contended box is not a frozen box"), a wall census run against this contention
would produce numbers the campaign's own rules already know not to trust. **Next action for
whoever continues: check box load, and once quiet, run `fulcrum try HEAD --threads 1 --levels
1-6` for the adjudicated wall verdict** — the local numbers above are strong directional
evidence (3.1-3.4x wall reduction, reproduced on 3 different input shapes) but are NOT the
adjudicator CLAUDE.md requires for a final wall claim.

---

## ⛔ PROMOTION PAUSED 2026-08-18 — Codex pre-merge review found a real, confirmed gap

The `lever/l3-gzip-deflate-fast-pickmin` branch (now at `8b0a21b0`, worktree
`~/www/gzippy-lever-l3-pickmin` — the primary `~/www/gzippy` worktree is back on
`main`, see "LOCAL LAYOUT" note in memory `feedback_repo_discipline.md`) is a three-agent-reviewed
(Fable, cursor-agent x2, Codex) ratchet fix for a ladder-monotonicity regression the L3
pick-min lever introduced. Codex's review found what all three prior reviews missed:

**`deflate_one_shot_t1_ratcheted` only reaches the T1 whole-buffer/mmap route.** The
STREAMING route — what the CLI actually uses for stdin/pipe input
(`gzippy -N -c - < file`) and what every library caller of `compress_bytes`/
`compress_with_pipeline` gets regardless of input source — dispatches to entirely
separate code (`encode_gzip_reader_to_writer_sized`) never wired into the ratchet, and
is NOT ladder-monotone in production. **Confirmed by direct CLI execution** (not just
trusted): `binary` fixture, T1, `-p1`, real binary — L1=662,577 → L2=666,107 (up) →
L3=661,353 (down) → L4=663,583 (up again) → L5=657,593 (down).

**This is NOT a regression this branch introduced** — verified via a throwaway probe
that `origin/main` (before c8bbde67/b4b821c9 ever existed) shows the IDENTICAL sag set
on the streaming route today. It was simply never covered by `ladder_is_monotone_t1`
(which only ever tested `encode_gzip_bytes_to_vec`), so nothing caught it until now.

**User decision (2026-08-18): the ladder invariant is REQUIRED on the streaming route
too — not a vendor-fidelity nicety.** CLAUDE.md non-negotiable #5 applies. Fixing this
is now part of finishing this branch, not a follow-up.

**Done so far (commits `22f8042a`, `8b0a21b0`):**
1. Fixed a second, independent Codex finding: `deflate_one_shot_t1_ratcheted` was
   recursive, holding up to 5 near-input-sized buffers alive simultaneously at peak
   (an 83 MiB input could peak ~400+ MiB of ratchet-owned `Vec<u8>` alone). Rewritten
   as an ascending iterative fold — same computation, byte-identical output, only 2
   buffers alive at any moment.
2. Added two RED tests pinning the streaming gap as the fix's acceptance criteria
   (deliberately not silenced with an allowlist): `size_invariants.rs::streaming_t1_is_ladder_monotone`
   (in-process, `compress_bytes`, all fixtures) and
   `t1_mmap_route.rs::t1_pipe_stdin_is_ladder_monotone_l1_to_l5` (real CLI binary,
   piped stdin, L1-5). Both correctly exclude ONLY the two pre-existing `("text",7)`/
   `("text",8)` sags already shared with (and accepted for) the mmap route — confirmed
   present on `origin/main` too, unrelated to any pick-min/ratchet lever. Full measured
   streaming sag set: `("text",7)` +567 B, `("text",8)` +413 B (both excluded, shared),
   `("tabular",3)` +15,770 B, `("binary",1)` +3,530 B, `("binary",3)` +2,230 B,
   `("noise",1)` +5 B (these four are the real gap, currently failing).

**NOT yet done — this is the next work:** design and implement a fix that makes the
streaming route ladder-monotone WITHOUT losing its point (bounded memory for large
piped input — true single-pass streaming architecturally cannot re-read input to try
multiple candidate encodings and pick the smallest, so "just add pick-min to
streaming" does not work as a design; ANY hard-guarantee mechanism trades some of
that memory-boundedness away, at least past some size threshold). This is a real
architecture question, not a bug-fix — being routed through Fable and cursor-agent
for a design pass before implementation, per this session's established practice for
judgment calls of this weight.

**The remote wall census** (`--threads 1 --levels 1,2,3,4,5,6`, PID 975533 on solvency,
`CAMPAIGN_OUT=/root/lever-b4b821c9-wall-3`) was left running through this — it
measures the mmap-route ratchet's wall cost, which the iterative-vs-recursive rewrite
does not change (same computation). Its result remains informative for that question
but is NOT the final gating measurement — a fresh lever run against the truly final
commit (once streaming is fixed) is still required before promotion.

---

---

## Goal

Drop-in replacement for gzip, pigz, libdeflate-gzip, and igzip:

- Same CLI and observable behaviour
- Valid gzip (roundtrip through gzip, pigz, libdeflate — sha256, not `wc -c`)
- **Per-label** size AND wall wins at every level 0–9 on every corpus file
- T1 and T>1 both in scope

Decompression is done and frozen. This campaign is **encoder only**.

**Build order (do not regress earlier steps):**

1. **T=1** beats every rival on size and wall
2. **T>1** parallel path (valid gzip only; byte-identity to T1 or vendors is never a gate)
3. Exotic levels −10/−11/−12

---

## Where we are

| Milestone | Status |
|-----------|--------|
| Main | `e888ac9f` — merged PR #332 (L2 mmap pick-min) |
| GATE board | **14 / 1320** size failures (`gzippy-bench/campaign/size-all-e888ac9f/`) |
| In-flight branch | `lever/l3-gzip-deflate-fast-pickmin` @ `b6fa4f08` (code @ `c8bbde67`) — **unmerged, unpushed** |
| Size lever on L3 | **NOT RUN** |
| Wall lever on #332 | **NOT RUN** (prior attempt killed mid clause-8 confirm) |

---

## GATE board @ `e888ac9f` — 14 failing cells (by name)

Artifact: `~/www/gzippy-bench/campaign/size-all-e888ac9f/summary.txt`  
Corpus: 22 GATE files · levels 1–9 · threads 1 and 4 · 4 rivals.

### gzip (11)

| Cell | Ours | Rival | Margin |
|------|------|-------|--------|
| `gzip:dd79_bin6:L3:T1:size` | 4,452,246 | 4,436,422 | +0.36% |
| `gzip:dd79_bin6:L3:T4:size` | 4,452,034 | 4,436,422 | +0.35% |
| `gzip:photo.jpg:L3:T1:size` | 6,472,401 | 6,469,678 | +0.04% |
| `gzip:photo.jpg:L3:T4:size` | 6,472,369 | 6,469,678 | +0.04% |
| `gzip:photo.jpg:L2:T4:size` | 6,473,446 | 6,470,611 | +0.04% |
| `gzip:photo.jpg:L1:T4:size` | 6,474,687 | 6,471,864 | +0.04% |
| `gzip:dd79_bin6:L2:T4:size` | 4,460,801 | 4,459,419 | +0.03% |
| `gzip:weights.safetensors:L8:T1:size` | 83,117,467 | 83,099,633 | +0.02% |
| `gzip:weights.safetensors:L9:T1:size` | 83,117,467 | 83,099,652 | +0.02% |
| `gzip:weights.safetensors:L7:T1:size` | 83,113,840 | 83,112,303 | +0.00% |
| `gzip:weights.safetensors:L7:T4:size` | 83,112,978 | 83,112,303 | +0.00% |

### libdeflate (1)

| Cell | Ours | Rival | Margin |
|------|------|-------|--------|
| `libdeflate:weights.safetensors:L4:T4:size` | 83,112,979 | 83,082,171 | +0.04% |

### pigz (2)

| Cell | Ours | Rival | Margin |
|------|------|-------|--------|
| `pigz:dd79_bin6:L3:T1:size` | 4,452,246 | 4,441,970 | +0.23% |
| `pigz:dd79_bin6:L3:T4:size` | 4,452,034 | 4,441,914 | +0.23% |

**Class notes (do not re-derive):**

- **T4 failures** on `photo.jpg` / `dd79_bin6` are T>1 seam class — mmap pick-min does not touch them.
- **`weights.safetensors`** libdeflate L4–L9 thin-margin ties were mostly closed by #332; L4 T4 and gzip L7–L9 remain.
- **`dd79_bin6` L3** is the worst gzip margin and blocks pigz too.

---

## Landed — PR #332 @ `e888ac9f`

**Mechanism:** L2 T1 mmap pick-min (3 arms):

1. Shipped `params(2)` greedy baseline  
2. L1 gzip-primary hash arm  
3. Gzip `deflate_fast` arm (`params_l2_gzip_deflate_fast`: chain 8, nice 16, min-match 3, `hash3_chain_depth=8` gated to this arm only)

Pick-min threading: **2 parallel + 1 sequential** (3 parallel arms VOID'd wall pin-gate at cpu%=253).

**Cells closed (size lever SHIP, `CAMPAIGN_PROMOTE=1`):**

| Cell | Evidence |
|------|----------|
| `gzip:dd79_bin6:L2:T1:size` | 4,448,467 vs gzip 4,459,419 (−10,952 B) |
| `gzip:photo.jpg:L2:T1:size` | 6,462,189 vs gzip 6,470,611 (−8,422 B) |

Promotion record: `~/www/gzippy-bench/campaign/promotion-records/lever-332-size-SHIP-5ef737a9.json`  
Gates at merge: tie-guard **0/33** flips · clause 3 OK · `fingerprint_suite` + `block_pins` pass.

**Board impact:** 27 → **14** failures (`size-all-18238262` → `size-all-e888ac9f`).

---

## In-flight — L3 mmap pick-min @ `c8bbde67`

**Branch:** `lever/l3-gzip-deflate-fast-pickmin`  
**Commits:** `c8bbde67` (code) · `b6fa4f08` (this PLAN only)

### Mechanism (vendor-diff first)

`fulcrum why gzip:photo.jpg:L3:T1:size` and `fulcrum why gzip:dd79_bin6:L3:T1:size`:

- **gzip L3:** `deflate_fast` — chain 32, nice 32 (`vendor/gzip/deflate.c` `configuration_table[3]`)
- **libdeflate L3:** greedy, depth 12, nice 14 (we ship **Lazy** at same depth/nice)
- Our L3 gap on photo is algorithmic (−63% matches vs gzip), not Huffman

**Implementation (mirror L2):**

- `level_uses_t1_mmap_pick_min` → `1..=4`
- `deflate_one_shot_t1_l3_pick_min`: Lazy baseline ∥ libdeflate greedy ∥ gzip deflate_fast
- `params_l3_libdeflate_greedy()` — `Strategy::Greedy`, depth 12 / nice 14
- `params_l3_gzip_deflate_fast()` — Greedy, chain 32, nice 32, min-match 3, `hash3_chain_depth` in fast arm only
- Wired through all mmap pick-min match sites + `encode_gzip_unpadded_l3_pickmin`

**Files:**

- `src/compress/deflate/mod.rs`
- `src/compress/deflate/level.rs`

### Hand-measured on branch (NOT a verdict — run lever)

**Board invocation** (fulcrum / `board-size.sh`):

```bash
gzippy -{level} -p {threads} -c $CORPUS/file
```

This is **PureT1Mmap** (`encode_gzip_unpadded_slice_to_writer`).  
Stdin redirect (`-c - < file`) is **PureT1 streaming** — wrong path for these cells.

| Cell | main `e888ac9f` | branch `c8bbde67` | Rival | Expected |
|------|-----------------|-------------------|-------|----------|
| `gzip:photo.jpg:L3:T1:size` | 6,472,401 (+2,723) | **6,462,140 (−7,538)** | 6,469,678 | **CLOSES** |
| `gzip:dd79_bin6:L3:T1:size` | 4,452,246 (+15,824) | 4,448,460 (+12,038) | 4,436,422 | −3,786 B; **still FAIL** |
| `pigz:dd79_bin6:L3:T1:size` | +10,276 | +6,490 | 4,441,970 | **still FAIL** |
| libdeflate ties (photo L3, weights L3) | — | unchanged | — | OK |

**Expected post-merge board:** **13 / 1320** if only `gzip:photo.jpg:L3:T1:size` closes.  
T4 cells unchanged (mmap pick-min is T1-only).

### Gates on `c8bbde67`

| Gate | Result | Notes |
|------|--------|-------|
| `scripts/campaign/tie-guard.sh HEAD` | **0/33 flips** PASS | vs `origin/main`; re-run before merge |
| `cargo test --release --test fingerprint_suite --test block_pins` | **PASS** | no pin regen needed |
| `CAMPAIGN_PROMOTE=1 lever.sh c8bbde67 --size-only` | **NOT RUN** | **first command for next agent** |

---

## Next agent — do this in order

### 1. Size lever (required before PR)

```bash
cd ~/www/gzippy
git checkout lever/l3-gzip-deflate-fast-pickmin
cargo build --release
scripts/campaign/tie-guard.sh HEAD
cargo test --release --test fingerprint_suite --test block_pins
CAMPAIGN_PROMOTE=1 CAMPAIGN_OUT=lever-c8bbde67-size \
  scripts/campaign/lever.sh c8bbde67 --size-only
```

- If **SHIP** → open PR, merge, then re-board:

```bash
CAMPAIGN_PROMOTE=1 scripts/campaign/board-size.sh all
# artifact: ~/www/gzippy-bench/campaign/size-all-<sha>/
```

- Report cells closed **by name**. Zero is information.
- Fix any clause-3 regression before merge; never hand-roll measurements.

### 2. If L3 SHIPs — next size front: `dd79_bin6` L3

```bash
fulcrum why gzip:dd79_bin6:L3:T1:size --repo .
```

Pick-min exhausted the obvious vendor arms (gzip deflate_fast + libdeflate greedy). Remaining gap is parse/block-boundary class. Do **not** ship global `hash3_chain_depth` or strategy changes without tie-guard + L6/L9 spot check (receipt: global hash3 flipped libdeflate L6/L9 ties).

**Judgment call routed through Fable (2026-08-17), cross-check via cursor-agent in flight —
this is a RECOMMENDATION for the next lever session, not yet built or measured:**

`fulcrum why` on this cell: matches Δ1.81%, literals Δ4.59% — gzip covers 61,994 more
positions using only 26,350 more matches (≈2.35 B/match ⇒ predominantly short/len-3-class
matches we're missing). Algorithmic/parse gap, not Huffman.

Fable rejected the two obvious zlib-ng candidates ([P14] early-exit-on-non-improving —
wrong direction, can only produce equal-or-worse matches; [P10] deflate_medium — unmotivated
midpoint of two arms that both already lose here) and instead named two composable defects
inside the **existing** L3 gzip-deflate_fast arm:

1. **Depth mismatch.** gzip L3 walks its 3-byte-keyed chain to depth 32
   (`vendor/gzip/deflate.c` `configuration_table[3]`). Our `params_l3_gzip_deflate_fast()`
   (`level.rs:370-381`) sets `max_search_depth=32` for the **hash4** chain only — the hash3
   side-chain is pinned at `hash3_chain_depth = HC_HASH3_CHAIN_DEPTH = 8` (`hc.rs:91`), 4x
   shallower than gzip at L3. (At L2, gzip's own chain depth is 8, which is offered as why L2
   closed but L3 didn't — the existing depth-8 hash3 happened to be faithful there.)
2. **Stale chain maintenance.** `HcMatchfinder::skip_bytes` (`hc.rs:643-702`), called on every
   match interior, overwrites `hash3_tab[hash3]` heads but never writes `next3_tab` — interior
   inserts leave dead/stale hash3 chain links, unlike gzip's uniformly-linked chain. Same class
   of defect as the already-proven-causal L1 bucket-maintenance mechanism
   ([[project_l1_bucket_maintenance_mechanism]]).

Falsify-record check: `src/compress/ldx/ht_matchfinder.rs:21` cites a FALSIFY record at
`src/compress/deflate/parse/mod.rs:540` that **no longer exists at that line** (current line
540 is an unrelated struct field) — treated as non-binding/dangling per CLAUDE.md hard stop 2.
Doc comment needs updating (hygiene only, not a gate).

Proposed lever (NOT implemented): (A) one line — `hash3_chain_depth = 32` in
`params_l3_gzip_deflate_fast()`, arm-gated so L2's arm is untouched. (B) thread
`maintain_hash3_chain: bool` into `skip_bytes` to also write `next3_tab` when true, called
from `greedy.rs` gated on `params.hash3_chain_depth > 0` — **this changes L2's gzip-arm bytes
too** since `skip_bytes` is shared, so grading must cover L2+L3 together. (C) if A+B leave a
residual: separate arm-gated block-flush-cadence lever (gzip flushes far more often, trading
header bits for better local fit — rival spends 49,427 header bits vs our 16,760 on this
file) — measured and gated independently, composed only if needed.

**Do not start this lever until:** (1) the c8bbde67 size-only `make lever` verdict lands and,
if SHIP, is merged (land the win first); (2) cursor-agent's independent cross-check of this
recommendation completes — see next PLAN.md update.

**cursor-agent cross-check (2026-08-17), CONCUR WITH CHANGES:** independently opened
`ht_matchfinder.rs:21` (confirmed dangling — the cited line is a struct field, not a FALSIFY
record) and `hc.rs`'s `longest_match`/`skip_bytes` (confirmed defect 2 is real: `skip_bytes`
updates `hash3_tab` heads but never writes `next3_tab`, unlike `longest_match`). **Sequencing
correction to Fable's plan: lead with defect (B) — the `skip_bytes`/`next3_tab` chain repair —
not defect (A) — `hash3_chain_depth = 32`.** `hc.rs:76-90` already carries an in-file note that
depth gains on `dd79_bin6` L3 saturate at depth 4 (d=8 is a wash or worse) — that measurement
likely predates the chain-repair fix, so depth 32 alone is unlikely to close the gap and should
be re-swept AFTER (B), not shipped as the headline change. Cheap first experiment once B lands:
`hash3_chain_depth = max_search_depth` in the L3 gzip arm, then sweep. Also flagged: `lazy.rs`
calls `skip_bytes` too — any future arm enabling `hash3_chain_depth` there needs the same gate.
D3 (periodic cost-based flush, "front C" in Fable's plan) confirmed as the right independent
residual lever if B(+A) underdeliver — different mechanism, compose don't conflate.

**Revised proposed order for the next `dd79_bin6` L3 session: (B) chain-repair → re-sweep
depth on the L3 gzip arm → (A) if still short → (C) block-cadence, each gated and measured
separately, `tie-guard.sh` before every T1-output edit.**

---

## ⛔ BLOCKING — new regression found on `c8bbde67` (2026-08-17), confirmed by execution

`cargo test --release --test size_invariants ladder_is_monotone_t1`:

- **FAILS on this branch** (HEAD `2b5e860a`): `"text" level 4 produced a LARGER output than
  level 3 (333536 > 324803 bytes, +8733 B, +2.7%)`.
- **PASSES on `origin/main` (`e888ac9f`)** — verified directly in a side-by-side worktree build
  (`~/www/gzippy/lever-c8bbde67-size-full/arm-e888ac9fd7fe`), not inferred.

**This blocks merge regardless of the GATE promotion-board verdict** (a board SHIP with a
failing size-invariant test is still NO-MERGE — CLAUDE.md non-negotiable 5).

**Root mechanism (cursor-agent diagnosis, code-verified):** the branch widened L3's pick-min to
three arms including `params_l3_gzip_deflate_fast()` (Greedy, chain **32**/nice **32**, forced
len-3 acceptance) but left L4's pick-min untouched (two arms, Greedy/Lazy at libdeflate-table
**16**/**30**, no len-3 forcing). On the `text` synthetic fixture (1 MiB prose, aozora/dickens
class — not a toy edge case), L3's new deeper/more-aggressive arm now out-parses L4's capped
arms, so L3 leapfrogs L4. Arm diversity is asymmetric, not a depth/strategy abutment already
covered by `KNOWN_SAGS`.

**Fix recommendation: widen L4, do NOT add `("text", 3)` to `KNOWN_SAGS`.** The sag is not
coordinate-locked to a synthetic-only artifact (text models real prose shape) and `KNOWN_SAGS`
documents accepted defects, not ones with an available fix. Mirror the L2→L3 precedent:

1. **`level.rs`** — add `params_l4_gzip_deflate_fast()`: same envelope as
   `params_l3_gzip_deflate_fast()` (chain 32/nice 32/forced len-3) but based on `params(4)` so
   L4-specific fields stay intact. NOT a gzip-level-4 transliteration (gzip `-4` is actually
   lazy/16/16) — purely to give L4 a third arm that can reproduce L3's winning candidate class.
2. **`mod.rs`** — widen `deflate_one_shot_t1_l4_pick_min` to three arms (two parallel + one
   sequential, mirroring L3's shape), picking min against the new gzip-class arm.
3. Verify: `ladder_is_monotone_t1` passes (all fixtures, both directions); per-arm ablation on
   `text` L3/L4 confirms which arm wins; `tie-guard.sh` shows L1/L2 untouched (only L4 pick-min
   changes); re-run the branch's promotion measurement after the fix.

**Scope:** L1/L2/L3/T>1/streaming untouched by this fix — only the L4 T1 mmap pick-min path
widens. If 32/32 doesn't fully close the gap, escalate depth/nice above L3's values (L4 must at
minimum reproduce L3's winning candidate).

**Next agent: implement this fix BEFORE re-running any lever on `c8bbde67`.**

**UPDATE (2026-08-17, same session): the L4 fix above was implemented and built — it
correctly closed L3→L4 (L4 now ties L3 at 324,803 B) — but `ladder_is_monotone_t1` is
STILL RED: the violation moved one level up.**

```
thread 'ladder_is_monotone_t1' panicked: on fixture 'text', level 5 produced a LARGER
output than level 4 (333536 > 324803 bytes, +8733 bytes)
```

L5 uses a DIFFERENT code path (`deflate_one_shot_t1_zlib_pick_min`, `level_uses_t1_zlib_pick_min`
= `5..=7`, 2 arms: baseline Lazy vs zlib-shaped Lazy at deeper chain) with no gzip-`deflate_fast`
Greedy arm — same mechanism, next level up. **This is whack-a-mole, not a fix**, and both Fable
and cursor-agent (independent architecture reviews, full reports below) converged on the same
verdict before I could keep hand-mirroring arms: **stop mirroring arms level-by-level; the
structural fix is a ratchet — level N's output is capped at level (N-1)'s bytes when N's own
pick-min regresses.**

**Both reviewers' key findings (independently reached, both from static code reading):**
- Root cause: pick-min arm SETS are heterogeneous per level (different vendor-shaped candidates,
  not a monotonically-deepening single parser), so `params_inner` depth monotonicity does NOT
  imply pick-min output monotonicity. `level.rs`'s OWN existing receipt (engine.wasm L8: deeper
  search made output LARGER — "a longer match at position i displaces a better match at i+k")
  proves parameter reasoning alone can never certify monotonicity; only a byte comparison can.
- A ratchet only ever REPLACES output with something STRICTLY SMALLER, so **tie-guard risk is
  ~zero when the ratchet itself fires** (it's a no-op on any cell that's already monotone,
  which includes all current libdeflate ties). Real tie-guard exposure is on the underlying
  arm-mirroring edits (e.g. adding a new gzip-fast arm to a level), same as any T1 change —
  run tie-guard regardless.
- **Full L1–L9 ratchet is too expensive to ship unmeasured**: naive recursive re-derivation
  costs ~13-20x a single encode at L9 (sum of every inherited level's arm count). A **scoped
  ratchet over L1–L5** (or L1-L7, Fable's slightly larger scope) is cheap (single wrapper,
  wall cost dominated by the deepest arm already run in parallel) but the wall bill is
  UNMEASURED and must be gated by `fulcrum try` before landing, not assumed.
- Disagreement only on immediate tactic: cursor-agent says mirror one more gzip-fast arm onto
  L5 as a quick stopgap AND land the scoped ratchet as the structural follow-up; Fable says skip
  the mirror and go straight to a "cumulative arm registry" (mathematically the same thing as a
  ratchet, expressed as a static per-level superset of arms instead of a runtime byte-compare
  recursion) for L1–L7 directly. **Recommendation: take Fable's registry framing (it's
  statically testable — "each level's arm set is a superset of the level below's" — which is
  exactly non-negotiable #5's "test the INVARIANT, not the value") but cursor-agent's SCOPE
  (L1–L5 first, wall-measure before extending to L6/L7).**

**Implementation sketch for next session (do this instead of any more manual arm-mirroring):**

1. `level.rs`: add `LevelParams: PartialEq` (or an equivalent identity key) and
   `pub fn t1_pick_min_arms(level: u32) -> Vec<LevelParams>` built as
   `t1_pick_min_arms(level - 1)` extended with that level's OWN new arm(s), deduplicated. Delete
   the now-redundant `params_l4_gzip_deflate_fast()` (L4 inherits L3's arm for free once the
   registry exists) — do NOT add a `params_l5_gzip_deflate_fast()` by hand; let the registry
   inherit it.
2. `mod.rs`: replace the four separate `deflate_one_shot_t1_l{1..4}_pick_min` functions (and
   fold in `deflate_one_shot_t1_zlib_pick_min` for L5) with one `deflate_one_shot_t1_pick_min(data,
   level)` that runs the registry's arms (generalize `pick_min_two_vecs` to N arms — keep the
   `anatomy-counters` winner-attribution branch), folding with strict `<` in native-arms-first
   order (inherited arms must not swap in different bytes on an exact tie with the incumbent —
   incumbent wins ties). Collapse the current 4 near-duplicate level-dispatch `match` blocks
   (`encode_deflate_segment_to_sink`, `encode_deflate_slack_padded_to_sink`,
   `encode_gzip_bytes_to_vec`, `encode_gzip_unpadded_slice_to_writer`) onto one predicate —
   note `encode_gzip_bytes_to_vec` currently does NOT even route L5-L7 through the zlib pick-min
   at all (falls through to single-arm `params(level)`), a **pre-existing entry-point
   inconsistency** worth fixing in the same pass, flagged but not yet acted on.
3. New unit test: `t1_arm_sets_are_cumulative` — asserts inclusion, not values, per
   non-negotiable #5.
4. Delist any `KNOWN_SAGS` pairs the registry heals in the SAME commit (test forces this).
5. Adjudicate in cheapest-falsifier order: `ladder_is_monotone_t1` + `make holdout`/`make
   surface` (deterministic, free) → `scripts/campaign/tie-guard.sh <ref>` (T1 bytes can change,
   downward only, but arm-mirroring edits still need the cage) → `fulcrum try <ref> --threads
   1,4 --levels 1-9` (or at minimum `3,5,6,9` per cursor-agent) for the actual wall number. **If
   L5-L7 wall fails the budget, fall back to scoping the registry to L1–L5 only** and park L6/L7
   inclusion with the measured wall number as the receipt — a decision the adjudicator makes,
   not one pre-announced here.

**Scope limits to state in the eventual commit:** T1 whole-buffer/mmap route only. The stdin
streaming path (`encode_gzip_single_pass`, single-arm `params(level)`) and the T>1 path
(`params_parallel`) are untouched and have their own (untested, for streaming) ladders — do not
assume this fix reaches them.

**Status: branch NOT mergeable.** `ladder_is_monotone_t1` is still red on the working tree
(uncommitted L4 partial fix + unmodified L3 code). Do not add `("text", 4)` (or `("text", 5)`
once the violation moves again) to `KNOWN_SAGS` to silence this — CLAUDE.md's "re-writing the
rule to fit the result is not allowed," and the allowlist is explicitly for defects with no
available fix, not ones a lever just created.

---

## GATE promotion verdict on `c8bbde67`, full 1-9 level size census (2026-08-17)

The FIRST lever run (levels 2,6,9 only) returned a void NO-SHIP — confirmed by inspecting its
`try.json`: `"levels": [2, 6, 9]`, zero L3 cells, so it could not see this L3-scoped change at
all. Re-ran per the tool's own prescribed remedy:

```bash
CAMPAIGN_PROMOTE=1 CAMPAIGN_LEVELS=1-9 CAMPAIGN_OUT=lever-c8bbde67-size-full \
  scripts/campaign/lever.sh c8bbde67 --size-only
```

```
TRY — promotion evaluation over 792 cells (660 decidable on both arms; 132 not)
  clause 1 OK: verify — zero roundtrip failures
  clause 2 OK: arms differ
  clause 3 OK: no pass->fail flips across 660 decidable cells
  clause 4 OK: closed failing cell(s): gzip:photo.jpg:L3:T1:size
  clause 5 OK: every passing cell inside its erosion budget
  clause 6 OK: improvement 0.0033 vs residual harm 0.0000
  clause 7 OK: all required arch(s) covered: aarch64
VERDICT: SHIP
```

Matches the hand-measured expectation exactly: **closes `gzip:photo.jpg:L3:T1:size`** (the only
cell the L3 pick-min lever was built for), zero regressions anywhere else on the corpus.
Artifact: `~/www/gzippy/lever-c8bbde67-size-full/try.json`.

**This SHIP verdict is scoped to the GATE promotion-board rubric only — it does NOT know about
`ladder_is_monotone_t1`, a separate pinned invariant test outside the board's corpus.** A GATE
SHIP with a failing size-invariant test is still NO-MERGE (CLAUDE.md non-negotiable 5). The
commit `c8bbde67` ITSELF carries the ladder regression (confirmed: fails on `2b5e860a` which
is `c8bbde67`'s code + doc commits, passes on `origin/main` `e888ac9f`) — it was not introduced
by my later uncommitted L4 edit, the L4 edit only made the SYMPTOM move from L4 to L5.

**Net next-session task: land the L1–L5 arm registry (see implementation sketch above), confirm
`ladder_is_monotone_t1` passes clean, re-run `tie-guard.sh` + this same full-level `--size-only`
lever (now against the registry-patched ref) to confirm `gzip:photo.jpg:L3:T1:size` still closes
with zero new regressions, THEN merge. Do not merge `c8bbde67` as committed today.**

### 3. T4 cluster (after T1 L3 front moves)

Open T4 cells (not fixed by mmap pick-min):

- `gzip:photo.jpg:L1:T4:size`
- `gzip:photo.jpg:L2:T4:size`
- `gzip:photo.jpg:L3:T4:size`
- `gzip:dd79_bin6:L2:T4:size`
- `gzip:dd79_bin6:L3:T4:size`
- `pigz:dd79_bin6:L3:T4:size`

T>1 seam-shrinking is a **closed class** (CLAUDE.md census). Need T1 headroom or T>1-specific parse lever.

### 4. `weights.safetensors`

- `libdeflate:weights.safetensors:L4:T4:size` — thin margin, T4 only
- `gzip:weights.safetensors:L7:T1:size`, `L7:T4`, `L8:T1`, `L9:T1` — separate class

### 5. Wall leg for #332 (optional record; size already merged)

```bash
CAMPAIGN_PROMOTE=1 CAMPAIGN_OUT=lever-e888ac9f-wall \
  scripts/campaign/lever.sh e888ac9f --threads 1
```

Prior run killed ~3h in (after-grid-wall 264/264 done, clause-8 confirm pending). Pick-min 2+1 threading fix is in `e888ac9f`.

---

## Campaign rules (binding)

| Rule | Command / action |
|------|------------------|
| Vendor-diff before proposing a lever | `fulcrum why <cell> --repo .` |
| Tie cage before any T1-output change | `scripts/campaign/tie-guard.sh <ref>` (~2 min) |
| Never hand-roll size/wall | `make lever REF=<ref> ARGS="--size-only"` or `CAMPAIGN_PROMOTE=1 scripts/campaign/lever.sh` |
| GATE board (not tune board) | `CAMPAIGN_PROMOTE=1 scripts/campaign/board-size.sh all` |
| Fingerprint pins | `cargo test --release --test fingerprint_suite --test block_pins` |
| Land cleared PR before starting new work | — |
| Report progress | cells closed **by name**; zero is information |

**Hard stops:** do not generalise L2-only measurements to L5–L9; do not use stdin path when board uses `-c file`; 3-thread pick-min VOIDs wall pin-gate.

---

## Toolbox

| Question | Command |
|----------|---------|
| Why does this cell fail? | `fulcrum why <cell> --repo .` |
| Is this change good? | `CAMPAIGN_PROMOTE=1 scripts/campaign/lever.sh <ref> --size-only` |
| Where do we stand? | `CAMPAIGN_PROMOTE=1 scripts/campaign/board-size.sh all` |
| Tie pre-check | `scripts/campaign/tie-guard.sh <ref>` |
| Falsified / parked records | `make falsified` |

Corpus root: `~/www/gzippy-bench/corpus` (`CAMPAIGN_CORPUS_ROOT`).

---

## Disk / environment (M1 Mac)

**Freed this session (~2 GB):** `gzippy-bench/breadth_tmp/`, `gzippy/benchmark_data/`, `/tmp/gzippy-*` worktrees, stale git worktrees, large agent-tools logs.

**Kept:** `target/release/gzippy` (~1 GB), corpus, `size-all-e888ac9f/`, promotion records.

**Safe to delete if space needed:**

- `cargo clean` in gzippy (~1 GB; rebuild required)
- `~/www/gzippy-bench/{weights_diag,decision_diag,residuals_diag}/` (~390 MB diag scratch)

**Do not delete:** `~/www/gzippy-bench/corpus/`, `~/www/gzippy-bench/campaign/size-all-e888ac9f/`, promotion-record JSONs.

---

## Key code map

| Location | What |
|----------|------|
| `src/compress/deflate/mod.rs` | `deflate_one_shot_t1_l3_pick_min`, `encode_gzip_unpadded_l3_pickmin`, `level_uses_t1_mmap_pick_min` (`1..=4`) |
| `src/compress/deflate/level.rs` | `params_l3_libdeflate_greedy()`, `params_l3_gzip_deflate_fast()`, `params_l2_gzip_deflate_fast()`, `hash3_chain_depth` |
| `src/compress/deflate/matchfinder/hc.rs` | hash3 chain walk (gated via `LevelParams.hash3_chain_depth`) |
| `src/compress/io.rs` | T1 file > 128 KiB → mmap → `encode_gzip_unpadded_slice_to_writer` (**board path**) |
| `scripts/campaign/lever.sh` | Promotion adjudicator |
| `scripts/campaign/board-size.sh` | Size axis; `CAMPAIGN_PROMOTE=1` for GATE set |

---

## Cells closed this campaign (cumulative, by name)

| Cell | When | Evidence |
|------|------|----------|
| `gzip:dd79_bin6:L2:T1:size` | PR #332 | lever-332-size-SHIP-5ef737a9.json |
| `gzip:photo.jpg:L2:T1:size` | PR #332 | lever-332-size-SHIP-5ef737a9.json |
| `gzip:photo.jpg:L3:T1:size` | **pending** L3 lever @ `c8bbde67` | hand −7,538 B; **needs `make lever`** |

**Remaining after L3 (estimate): 13 / 1320** — see board table above minus `gzip:photo.jpg:L3:T1:size`.
