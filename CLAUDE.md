# gzippy

## What we are building

A drop-in replacement for gzip, pigz, libdeflate-gzip and igzip that is strictly
better. Same commands, same observable behaviour, output at least as small AT THE
LEVEL THE USER TYPED, and less wall time — on their machine, at their thread count.

DECOMPRESSION IS DONE AND WON (four months, PR #116, merged 2026-07-18). Do not
revisit it, re-measure it, or optimise it. This file is about the encoder.

**Where everything is.** This file is the charter; it is not the state and it is not
the toolbox. `~/.claude/.../memory/MEMORY.md` holds the measured board structure —
read it first. The four commands you will actually use:

    make falsified                        # every FALSIFY and PARK record in src/
    CAMPAIGN_LEVELS=1-9 make board-size    # the deterministic size board, all levels
    make lever REF=<ref> ARGS="--threads 1,4"   # the promotion adjudicator
    scripts/campaign/tie-guard.sh          # 2 min; the T1 tie cage, before any T1 edit

`make lever` grades levels 2,6,9 and threads 1 BY DEFAULT. Both defaults have shipped
a wrong verdict; pass the level and thread range the change actually acts on.

## Build order — this is the plan, in this order

**STEP 1 — T=1, beating every other implementation.** One code path. Read the input
once. Single pass. Shared buffers. Zero copies. No per-block, per-run or per-chunk
allocation. Every optimization a C engineer reaches for, done in Rust: SIMD
intrinsics, hand-written asm, runtime CPU dispatch, per-arch builds, prefetch,
monomorphisation, unsafe, bounds-check elision, instruction selection — including
avoiding the instructions that force the core to down-clock.
DONE WHEN: per-label, at every level 0-9, on every corpus file, on both arches, we
are <= gzip AND <= pigz AND <= libdeflate AND <= igzip on size and on wall.

**STEP 2 — T>1, a separate code path.** A consumer thread plus compression threads
writing to a shared fd. Fulcrum proves no thread is ever starved and that reads and
writes are scheduled correctly — the same starvation/causation/perturbation tooling
that won parallel decode.
THE ONLY CORRECTNESS BAR, at every thread count, is VALID GZIP: roundtrip sha256
through our decoder plus one independent decoder. T>1 may emit different bytes than
T1. Byte-identity to a vendor, to our own T1, or to our own previous run is never a
goal and never a gate. (User, 2026-07-28, stated three times. This paragraph once
mandated T1==T4; that is WHY the rule had to be restated three times — each
correction landed in a leaf doc while the root file kept regenerating the cage.)

**The T>1 size leg is NOT closed by making seams smaller.** This paragraph used to
say it was — "pad choice, chunk grid, block splitting" — and that was MEASURED FALSE
on 2026-08-01 from `/root/sizeboard-all-12fcd0ed/census.json`. Pairing every
libdeflate cell's T1 and T4 rows:

    fail-at-T4-only=109   fail-both=16   pass-both=72
    T4-only excess: min=2  median=255  max=2,093  total=35,084 B
    the SAME cells at T1, headroom (rival - ours): min=0  median=0  max=0
    cells closed if the seam tax were cut 25/50/75/90% -> 0, 0, 0, 0.  By 100% -> 109.

All 109 tie libdeflate BYTE-FOR-BYTE at T1, so the class has ZERO partial credit: a
2-byte seam fails the cell exactly as hard as a 2,093-byte one. That is why grid
tuning, pad choice, chunk-count matching and block splitting each closed nothing —
not because they failed, but because "smaller" is not on this class's scoring
function. The seam leg is therefore closed by **monotone T1 size wins that buy
headroom to spend** — a change that can only ever make a block smaller breaks the tie
in our favour, cannot flip a passing cell under clause 3, and makes the T1 and T4 size
legs the SAME problem. Do not propose a seam-shrinking lever without first showing the
target cells have headroom.

**The needed margin is ~0.01%, and Huffman CONSTRUCTION cannot supply it.** The first
version of this paragraph named "costing an exact package-merge code beside the
heuristic one" as the example monotone win. That is a CLOSED class — see the binding
measurement (recorded in git history, the in-code note was deleted 2026-08-01), which
built it BOTH ways and measured: the unconditional swap is a wash that OPENS cells, and the costed dual
candidate holds its size invariant (49/49 smaller) at ~0.001% margin while costing
10-14% WALL and flipping sil40 L6 from win to lose. libdeflate's heuristic length
limiter is already within 0.001% of exact; the margin we need is an order of magnitude
larger, so this is not a speed problem. That measurement stands on its own numbers, not on any
record's authority — re-run it if you doubt it. The remaining candidates are **block BOUNDARIES or the parse** —
which is where `examples/blockspans` points too: gzip splits on a fixed ~34,000-symbol
cadence (cv=0.023) while our spans run 8x longer at cv=0.373.
DONE WHEN: the same per-label bar holds at the default thread count and at T4/T8/T16.

**STEP 3 — the exotic path (-10/-11/-12), separate again.** Our `parse/ultra` crown
engine already beats zopfli on exact bytes 4/4 and is unreachable from a numeric
level. Put it on the ladder and beat the state of the art there.

Three separate paths. Later work must not be able to regress earlier work. When a
step is done, its cells stay closed — a regression in a closed cell blocks the merge.

## Non-negotiables

1. **Valid gzip.** Every output decompresses byte-exactly through gzip, pigz AND
   libdeflate. Nothing merges without that. Whether our bytes match a vendor's is
   irrelevant — nobody cares, and it is not a goal.
2. **Per-label.** At level N we beat their level N. Curve-dominance is not the goal
   and never grades again.
3. **No environment knobs.** No env var changes behaviour. No detecting which corpus,
   cell, or archive type we are in. No content detector choosing a parser. Delete the
   ones that exist: 33 `GZIPPY_L1TUNE_`/`L3TUNE_` vars in `src/compress/`,
   `parse/gated.rs`, and the L1 hash3 content gate. If a data-dependent decision is
   ever proven necessary to win, bring it to the user first — and then only as
   parameter tuning (write-buffer size, shared memory per thread count).
4. **Least surprise.** Cite a contract (zlib's API, gzip's CLI, POSIX) — never a
   vendor's habit.
5. **Never pin a knob that is declared free to change.** An equality assertion in a test
   beats a sentence in a doc, always, because only one of them fails closed. This file
   said the level->config map "is free to change" while `level.rs` carried a
   `vendor_knob_values` test asserting `params(6).max_search_depth == 35` and
   `params(9) == 600`. The test won for weeks. Test the INVARIANT (effort rises with
   level) not the VALUE.

## Every technique is in scope

There is no permission list, no sanctioned/unsanctioned distinction, no
vendor-fidelity rule, and no clause in this file that requires anyone's approval.
Read libdeflate, igzip, zlib-ng, zopfli, ECT, rapidgzip — steal every good idea.
**Never inherit a vendor's decisions.** Our level->config map is currently a copy of
libdeflate's, which is why we run their algorithm slower than they do; it is free to
change. A technique with no vendor counterpart is equally welcome.

Decide, build, measure, report. Never ask permission.

## What closes a step

- **Correctness, always:** roundtrip through gzip, pigz and libdeflate. sha256, never
  `wc -c`.
- **Deterministic facts** (output bytes, instruction counts, allocation counts):
  verify once and move. No statistics.
- **Wall claims:** interleaved paired A/B, both arms to `/dev/null`, on the frozen
  box, from a vanilla `cargo build --release`. Report the paired-difference CI, not
  marginal spread. A tuned/instrumented build is 1.17x slower — never quote one
  against a rival.
- **Always verify the binary you measured is the binary that ships** (route assertion
  + sha). We once spent weeks measuring a binary CI wasn't shipping.
- **A gate may only cite a dataset that exists.** If the artifact is empty, the gate
  is void — produce the data or drop the gate.
- **One rule per change, declared once.** If a promotion rule fails, the change is
  reverted or the change is fixed. Re-writing the rule to fit the result is not
  allowed; neither is announcing in advance what you will do when it fails.
- **A change that makes things worse gets reverted.** Do NOT leave a FALSIFY comment
  behind — those were deleted wholesale on 2026-08-01 as noise. Re-measure with
  instrumentation when the question comes up again.

## Reasoning, measuring, and tooling

Reason freely about mechanism — that is how an optimised inner loop gets written. But
any claim about what the SHIPPED BINARY DOES must be proven by executing it (route
assertion, output sha, build-both-ways diff). The compression routing table in this
file was once a careful code read and was wrong in every particular.

Fulcrum finds where the time goes and proves a change worked. **An instrument that
reveals STRUCTURE is the cure, not the cost.** The user's diagnosis (2026-07-31):
decompression took months because the strategy was reading and reasoning, and "once
we built measurement tools that revealed structure we optimized that structure right
away." Never hesitate to build one. Fear reasoning, not instruments.

What IS forbidden is an instrument with no blocked question behind it. The test is
not "is this a tool?" but "does a NAMED failing cell need an answer this tool gives?"
Name that cell and the blocking question in the commit message. A tool that ships
without closing or unblocking a named cell is a docs commit wearing a tool's costume.
(This paragraph previously read "It is never the deliverable," which inverts the
user's causation and chills the one technique with the best track record here.)

**The only progress metric is failing cells closed, by name.** Report it every
session, and report zero as zero. A zero is information, not a verdict on the
session — the sessions that close zero are often the ones that correctly falsified
something expensive.

⛔ **DELETED 2026-08-01: "two consecutive sessions at zero BLOCKS optimisation
edits."** It mandated the failure mode it was written to prevent. The activities it
PERMITTED on a blocked session were "profiling, measurement, vendor-structure-diff,
or landing already-gated work" — precisely the docs-and-measurement churn that
produces zero-cell sessions. Receipt: of the 13 PRs merged in the #224-#238 range,
11 are `docs(…)`/`measure(…)`/`tools(…)` and 0 closed a failing cell; under the rule
as written every one of those 11 was legal, while the one PR carrying a measured
42-cell win (#227) sat OPEN the whole time and was the one act the rule discouraged.
It also could not fail closed — nothing reads it — so its only function was to
license another audit document. Do not reinstate it without a receipt showing a
blocked session that a ban on optimisation edits would have improved.

## Hard stops

These fail closed. They exist because the soft versions were each ignored at
least once in a single session.

1. **Diff the vendor BEFORE proposing a change.** Every win this project has ever
   had came from comparing our implementation against a vendor's (3 for 3). Every
   change found by opening our own profile and shaving its top line has failed
   (3 for 3, plus 5 more). A change with no named vendor difference is allowed,
   but the commit must say so explicitly — that declares there is no precedent
   the idea pays, so the bar is a measurement rather than an argument.
   Build the vendor with `-g` or its profile is one opaque symbol.

2. **RETRACTED 2026-08-01 by the user — this slot used to read "A FALSIFY note is
   BINDING."** *"Delete all of those falsify comments. They're just noise. Not a single
   one of them is more valuable than simply running the tool with good instrumentation."*
   All 43 in-code FALSIFY records were deleted, along with `make falsified`,
   `check-falsify-visible.py` and the commit-msg enforcement. **Do not re-add them, and
   do not treat a falsification quoted from git history as binding.** Re-measure instead
   — the instruments are cheap now and a record is not.
   Receipt for why: a record at `ht_fast.rs:200` closed the insert-density class, and
   instrumenting both encoders showed the shipped path sits at 0.276 inserts/byte against
   libdeflate's 1.000 — a structural gap the record's own numbers could not have revealed,
   because they were taken on a different matchfinder.

   **THE BINDING RECORDS LIVE IN `src/`. SEARCH THEM BEFORE PROPOSING ANYTHING:**

       git fetch origin && make falsified      # indexes FALSIFY *and* PARK (#240)

   A checkout 26 commits behind once made `make falsified` look like it did not
   exist, and a whole session on 2026-08-01 was spent re-deriving three findings
   already written down — two in `level.rs` and one in a memory file whose opening
   sentence was a warning not to re-derive it. **A prose warning did not work; the
   command is the fix.**

3. **Never generalise a measurement across levels — AND THIS APPLIES TO RECORDS, NOT
   ONLY TO CHANGES.** The hard stops are enforced against edits: a commit hook reads
   commit messages. NOTHING enforced a hard stop against the TEXT OF A RECORD, so a
   falsification could itself violate #3 and survive indefinitely, closing a class for
   every session that grepped it. (This is a large part of why the records were deleted.) Receipt: a
   2026-07-28 note measured L2 only and concluded "Depth is not the cost; the strategy
   is." That sentence is false at L5-L9 — raising `max_search_depth` to zlib-ng's chain
   values, changing no strategy, closed 84 failing size cells — and it cost weeks:
   `.git/logs/HEAD:235-237` shows `probe/l5-depth` created and abandoned 102 seconds
   later with no commit. **Scope every falsification to the levels it was measured at,
   in the sentence itself.** A record that says "X is not the cost" without naming its
   level is a class closure it never earned.

   The original rule still stands: Any instruction, read, or
   wall claim must be measured at a SHALLOW and a DEEP level before it is
   believed. Measuring L2 alone and generalising shipped a 6.2% L6 and 9.9% L9
   regression.

4. **Source-level cost is not machine-level cost.** A claim about loads,
   branches, or work must show the counter moving (Dr, Ir, cycles). Hand-hoisting
   "obviously redundant" loop-invariant loads drove data reads UP, because LLVM
   had already hoisted them and the hoist only added register pressure.

5. **Land gated work before starting new work.** A win that has cleared the
   promotion rule and sits in an open PR is worth more than any unstarted lever.
   The one real win this session was earned early and landed last, after eight
   failures, only when challenged.

6. **Never hand-roll a measurement.** Check Fulcrum's command list first. The list, because
   "check first" without it is what a session ignored for hours while rebuilding three of
   these by hand:

   | question | command |
   |---|---|
   | why does this cell fail? | `fulcrum why <cell> --ours BIN --rival-cmd 'CMD -{level} -c {input}' --corpus F` |
   | what could I do about it? | `fulcrum candidates <cell> --repo .` |
   | is this change good? | `fulcrum try <ref> --threads 1,4` |
   | did I break CLI behaviour? | `fulcrum dropin --ours BIN --rival gzip=gzip …` |
   | where do we stand? | `fulcrum board size` / `board wall` |
   | where does the time go? | `fulcrum profile …` / `ab ablate` / `trace …` (T>1 causation) |

   `why` returns the vendor diff — position counts (match/literal/header/data per byte) name
   the MECHANISM in one command — and it states which of its four layers it SKIPPED, so a
   partial run cannot be mistaken for a whole one. `dropin` is the THIRD goal axis (CLI behaviour) and went unmeasured for the whole
   campaign. **`fulcrum try` defaults to `--threads 1`** — a T>1 change judged without
   `--threads 1,4` is measured on cells it cannot affect, and will read NO-SHIP for the
   wrong reason. A
   hand-written size audit compared byte counts with no roundtrip check and would
   have scored a corrupt-but-smaller output as a WIN; `sizecensus` already existed
   and VOIDs that. If the tool is missing on a box, FIX THE BOX — a stale
   instrument set is what produced the substitute.

7. **A measurement from an unidentified binary is not a measurement.** Verify the
   deployed commit before quoting a number from it.

8. **Every verification line names the command that produced it.** A write-up that says a
   gate ran must be traceable to a tool call in the transcript. Delete the line otherwise.
   Receipt: `lever/l3-gzip-deflate-fast-pickmin`'s PLAN.md was headed "every step re-run
   independently on the merged branch, not just taken from the agent's report" and then
   claimed tie-guard was "run twice, once again after cherry-picking" — the transcript has
   ONE tie-guard run, in the agent's worktree, before the cherry-pick — and quoted a
   `pickmin_arm_audit` win-rate whose last execution was nine hours before that code existed.
   This campaign runs on receipts. A fabricated receipt is worse than no receipt, because the
   next session banks it.

9. **Open the PR with the FIRST code commit.** A branch carrying code and no PR is a stop
   condition, not a state. Receipt: 2026-08-17/20, `lever/l3-gzip-deflate-fast-pickmin`
   reached 31 commits — 20 of them `docs:` — carrying an adjudicated `--size-only` SHIP
   verdict and two closed cells, with NO PR ever opened. `gh pr` was never executed once in
   61 hours. Three of those commits were byte-identical and individually landable. Cells
   closed on `main` that session: zero.

10. **A test is not a goal. Name the clause it serves before spending wall on it.** Grep
    this file and `docs/promotion-rule.md` for the property the test asserts; if neither
    names it, it is a local convention and the charter outranks it. Receipt:
    `ladder_is_monotone_t1` appears in no clause and no goal sentence. On 2026-08-19 a
    session confirmed by direct execution that `origin/main` violates it on the real corpus
    (`photo.jpg` T1: L3 = 6,472,401 > L2 = 6,462,189, 10,212 B), listed "revert to before
    `c8bbde67`" as an option, never adjudicated it, and bought that invariant a measured
    7.3x-vs-our-own-code / 8.7x-vs-libdeflate wall regression that was still open when the
    session ended.

## Finding STRUCTURE instead of chasing a number

Every rule here has a receipt from a session that violated it. They are ordered by how much
time each one has cost.

1. **Name the CLASS, not the cell — and state what fraction of the board it holds, split by
   rival AND thread count, before optimising anything.** Receipt: `libdeflate`-at-T4 is 48 of
   68 failures on the frozen box while `libdeflate`-at-T1 is ZERO. Work aimed at the T1
   matchfinder could not have closed a single failing cell no matter how well it went.

2. **Measure at the coordinate where the cells FAIL.** Budget, slack and cost all move with
   level, thread count and file. Receipt: the entire parse-config space was closed as
   "unaffordable" against T1 wall slack of 0-8%. The failing cells are T4, where the rival is
   single-threaded and our slack is 249-330% — a 40x budget error that made every
   configuration look impossible.

3. **A falsification must record its COORDINATE, and separate an INTRINSIC CEILING from a
   COORDINATE-DEPENDENT VERDICT.** "libdeflate's heuristic is within 0.001% of the
   mathematical optimum" is intrinsic and permanent. "Therefore it is dead" was neither — it
   was true only at T1, standalone, on the default grid. Write the ceiling and the coordinate
   as separate sentences.

4. **PARK monotone work; never DELETE it.** If a change is non-worse BY CONSTRUCTION on some
   axis and only loses on another, its cost is a candidate for a coordinate artefact. Receipt:
   a strict size win (49/49 cells smaller, 0 worse) was deleted for a wall cost that is 6x
   smaller at T4 than the T1 number it was judged on. It had to be rediscovered, and the
   note left behind actively told the next session not to look.

5. **Compose before concluding.** Two changes that each miss the bar can clear it together.
   Receipt: ~150 B of margin ("too small to matter") plus a seam reduction ("still fails")
   compose into cells that close. Ask what ELSE would have to be true for a rejected change to
   pay, and test that, before writing the falsification.

6. **COUNT it; never INFER it.** Every inferred constant this campaign has leaned on was
   wrong. Receipts: 350 B/header inferred from a seam delta vs 107 B measured (3.3x); a wall
   run estimated at 47 hours from a 3-cell startup sample vs ~3 hours actual. If a claim rests
   on a constant, add the counter first — that is cheaper than the retraction.

7. **Identify the tree and the binary before EVERY measurement, not just before shipping.**
   Receipt: a result showing a 55 KB improvement came from an uncommitted change belonging to
   no branch in this checkout. The tell was that the number was implausibly good.
   Implausibly-good is a provenance alarm, not a win.

8. **A number is a symptom. Name the MECHANISM, then predict a second consequence and check
   it.** Receipt: 24.1M unattributed data reads correctly located the deficit inside `hc`, and
   the inference "therefore reads are the wall-blocking quantity" was false — deleting 25.6M
   of them made the wall 1.77% WORSE. One measurement supports one claim.

9. **Report the class you closed and the class you did not.** "5 cells closed" means little
   without "of 48 in that class, and the other 43 need a different mechanism."

10. **A paste is not a decision, and a hand-roll may only KILL.** Attribute a constraint to
    whoever typed it, in the sentence that pins it. Receipt: a test comment justified a
    deliberately un-allowlisted hard test with "user decision 2026-08-18: this invariant is
    required on every T1 route" — no user message said it; the user had pasted a Codex
    review, whose own sentence offered "or the branch should not ship." A test comment fails
    closed; that is where a misattribution does the most damage. Corollary, same session,
    five times: `hyperfine`, `wc -c`, and a paired local A/B are FALSIFIERS. Only
    `fulcrum try --threads 1,4` promotes. A "3.4x wall win" committed on M1-only hyperfine
    was reverted when x86 — using arm binaries `fulcrum try` had ALREADY BUILT on that box —
    found it 1.19x slower at L3. UNDECIDED means the change does not ship; it does not mean
    substitute a cheaper instrument and keep the adjudicator's authority.

## Working rules

- **A retraction must reach EVERY copy, root and leaf.** When the user retracts a
  goal or constraint, grep CLAUDE.md, MEMORY.md, **every file in the memory
  directory**, and docs/ for every statement of it and fix them all in the SAME
  sweep. Quote-and-strike beats delete: a claim that has regenerated once will
  regenerate again unless the retraction is louder than the claim, in the same
  place. (Receipt, and it is worse than the version this replaces: the byte-identity
  rule needed three user corrections because each correction landed in one file
  while another kept re-emitting it. The rule as first written said "root" and named
  three paths — the memory LEAF files were not among them, so on 2026-08-01 the cage
  was still sitting in `project_compression_state.md`, tagged ★★★, in a file
  MEMORY.md advertised as live state, long after the root was clean.)
- **Land the win first.** A change that has cleared the promotion rule outranks
  starting anything new, and an open PR holding a cleared win is item one at every
  board check. No new lever starts while a cleared win sits unmerged or a
  user-ordered deletion sits undone. (Receipt: the only landed win of 2026-07-28
  sat in a PR through eight falsified levers.)
- **Two strikes is a hint, not a closure.** Two failures of the same mechanism mean
  look harder at the vendor diff before a third attempt — they do not close the class. Five of one session's eight levers were the same class —
  hand-scheduling a loop LLVM had already scheduled — re-sampled after its verdict
  was already known.
- **Run `scripts/campaign/tie-guard.sh <ref>` before any change that alters T1 output.** We are
  byte-identical to libdeflate on nearly every T1 cell (66 of 66 at L2/L6/L9). A tie PASSES but
  has ZERO tolerance — one byte either way flips it, and clause 3 refuses that absolutely. The
  bar is NON-WORSE ON EVERY TIE, not net-positive: two levers that were net T1 IMPROVEMENTS died
  here (hash3 chaining 6 closed/12 flipped; zlib good_match 31 closed/17 flipped, with data.csv
  L2 going 1.0000 -> 1.0431). The guard runs the tie subset in ~2 min instead of ~20, and a
  hand-picked 9-file sample missed the two worst files entirely — enumerate, do not sample.
- **Cheapest falsifier first.** Order a lever's legs by cost: deterministic size
  and Ir on the canonical corpus before any wall run; a shallow AND a deep level
  before any claim about "the levels"; both arches before any general conclusion.
  This binds lever SELECTION, not only shipping.
- **State the absolute next to the relative.** 87% of a 0.01% penalty is 0.01%. A
  commit quoting a ratio names the artifact path holding it.
- Branch + PR; main is protected. `make` before `make ship`.
- One integration writer per checkout; worktrees for parallel work.
- If a tool errors, diagnose the first failure before doing anything else. Never
  `python3 -c` with multiple lines — write a `.py` file. Wrap hang-prone commands in
  `timeout`. Check `df -h` around big builds.
- **ZSH DOES NOT WORD-SPLIT UNQUOTED PARAMETERS.** `set -- $spec`, `cmd $flags`, and
  `pgrep -f "$pat"` do not behave as they would in bash: `$spec="a b c"` arrives as ONE
  argument. This produced FOUR phantom findings in a single session — a "rejected `-b`
  flag", a "missing `board goal` subcommand", a "missing `profile excess` subcommand"
  (both of which exist and always did), and a self-matching `pkill -f` that killed the ssh
  session issuing it. Two of them were reported to another agent as tool defects and had to
  be retracted. Use `read -r a b c`, an explicit array, or quote-and-split deliberately —
  and when a probe reports that something does not exist, RUN IT BY HAND before believing it.
- Never `rm -rf` a path from a variable, argument or glob without resolving it
  absolute and asserting it is strictly inside an expected root.
- Delete anything a measurement beats. Nothing is precious. Cost is never a technical
  criterion; ordering is fine, dropping a real lever is not.
