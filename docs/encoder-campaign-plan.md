# The encoder campaign plan

Synthesised 2026-07-29 from three parallel mining passes over the DECODE campaign's
968 MB transcript archive (what won, what stuck), plus an architecture audit of the
encoder against `encoder-architecture.md`, `compressor-architecture.md`, and five
vendored implementations.

This file exists because the decode campaign's own forensics found **533
conclusion-sentences against 1,127 reversal-sentences — reversals outnumbering
conclusions 2:1, across 74 self-refuted "the lever is X" claims.** A session on
2026-07-29 reproduced that fingerprint exactly: 9 levers attempted, 9 falsified,
board unchanged at 165 failing cells. The plan's job is to not do that again.

---

## 1. What actually wins, counted

Decode's discovery-method tally across four winning months:

| method | wins | falsifications |
|---|---|---|
| vendor structural diff / convergence / port | ~9 | — |
| **causal perturbation** (removal oracle, ablation, blocked-on decomposition) | **~8** | — |
| self-profile top-line shaving | **0** | **>=8** |

The encoder's own record is the same shape: 3/3 wins from vendor diffs, 0/8 from
shaving our own profile's top line.

**Two consequences.**

1. A lever selected from our own profile is not a candidate. It is a coin flip with
   a known-bad prior.
2. **We have never built the second winning family.** Every decode win that was not
   a vendor port came from a causal instrument — stub or delete a region, watch the
   number respond. *"The oracle is what separated 'recoverable lever' from
   'cycle-cheap slack' every time."* The encoder has no such instrument. That gap,
   not any missing insight, is the largest single hole in this campaign.

## 2. The ordering lesson, which is the reason the last session failed

Decode's kernel-converge series **halved** the igzip T1 gap by *faithful structural
transliteration* of igzip's loop shape. Only after that convergence did single-op
deletions (`cursor2` single-refill, B2 pre-copy refill) become visible — and each was
then worth 5-10%.

The encoder has been attempting the deletions **without the convergence**. We
"run libdeflate's algorithm slower than they do", which is precisely the
pre-convergence state decode was in on 2026-06-18. The answer then was not knob
search on our own shape; it was converging on the vendor's structure with the
vendor's profile as the control.

Decode's sequencing verdict, stated by the campaign's own retrospective: *"every
attempt to optimize ahead of a trusted instrument produced the June circles; every
post-`fulcrum score` week produced closed cells."* Decode drifted for weeks until one
command generated the loss map, then went 30/40 -> 110/120 -> zero-loss in ten days.

## 3. The board, as measured

165 failing per-label SIZE cells (roundtrip-verified census, canonical 17-file
corpus x L1-9 x 4 rivals x T1/T4; 1020 measured, 0 VOID).

| rival | T1 | T4 |
|---|---|---|
| libdeflate | 11 | **113** |
| gzip | 11 | 12 |
| pigz | 8 | 8 |
| igzip | 1 | 1 |

Failures are **uniformly distributed** across all 17 files (9-17 each, no outlier),
which rules out content-shaped causes and points at a per-chunk fixed cost.

**Measured per-chunk constant** (silesia 212 MB, 405 chunks): **+18.7 B/chunk at L2,
+32.1 B/chunk at L6.** Seams account for ~5.4 B. The dominant term is **extra
dynamic-header mass from restarting the block grid inside every chunk** — 2-3x the
seam cost, growing with level.

That explains both banked T>1 reverts: the pigz 10-bit pad attacked only the sub-byte
alignment sliver (0.0007% size, 0.7-0.9% wall), and the level-scaled chunk grid
divided the constant rather than removing it (0.01% size, 2-4% wall).

**The structural conclusion:** the T4 size cells are unclosable while both (a) chunks
are independently coded and (b) T1 ships libdeflate's exact sizes (zero slack at
L2/L4-L9). One of (a) or (b) must change. (a) is the complete fix.

## 4. The plan

### Phase 0 — trust the instrument before optimising anything

Decode's most expensive mistake was **measuring an artifact that was not the
product** — weeks of real, gated, cross-arch wins on a build flavour CI never
shipped. Its most-recurred class: the stale routing table, the instrumented-build
tax quoted against rivals, T10 measured as T1.

0.1 **Land the gated work.** PRs #179 (rules + docs) and #180 (L3 win, env-channel
    deletion, commit-msg hook). `docs/vendor-structure-comparison.md` and
    `vendor-technique-index.md` are cited by CLAUDE.md as in-repo but exist only on
    unmerged branches — one dead branch from losing the campaign's key artifacts.

0.2 **Build the encoder removal oracle.** The missing win family. Region-stub arms
    (lazy peek, a hash-insert tier, the block-split search, the prefilter, the
    seq-store) each compiled out behind a non-shipping feature, then measured for
    size AND wall response per level. Monotonic response => on the critical path;
    flat => slack. This is what tells us which regions can pay before we spend a
    lever on one.

0.3 **Preflight on every measurement, refusing rather than warning:** route
    assertion (`encode-path=`), binary sha matched to source commit, explicit
    `-p1`/`-pN`, `/dev/null` both arms, same-run A/A bracketing 1.0.

### Phase 1 — T>1: buy slack and cut chunk count (NOT a coding-locus rework)

**The "parse in parallel, code serially / closes 103 cells by construction" plan was
FALSIFIED on paper by adversarial review before any code was written.** Two reasons,
both structural:

* Histograms are built in the PARSE, not the emit (`parse/mod.rs:121-131`), and the
  block grid is a per-symbol observation stream with per-block state
  (`block_split.rs:14-61`). A writer that re-blocks across seams must re-run those
  observations serially over every literal byte — nowhere near 4-6% of work.
* If workers keep their own grid instead, the grid still restarts per segment, which
  IS the dominant term. Deleting the 5.4 B seam while keeping the 13-27 B header term
  closes almost nothing.

Exactness is required, not approximation: T1 is byte-identical to libdeflate at
L2/L4-L9, so +1 byte fails the cell. A seam-parsed token stream cannot reproduce T1's
tokens exactly (pending lazy deferral, a match spanning the boundary, grid phase), and
the divergence is sign-indefinite.

**What the arithmetic actually says.** Per-chunk cost is 18.7-32.1 B on 512 KiB
chunks = 0.0036-0.0061% of input. Census of all 133 failing T4 cells, gap-to-rival
divided by (chunks x per-chunk constant):

| gap / chunk-overhead | cells | reading |
|---|---|---|
| <= 1 | 38 (29%) | chunk overhead alone explains the failure |
| <= 2 | 30 (23%) | reachable with slack + grid together |
| <= 5 | 32 (24%) | |
| > 5 | 33 (25%) | REAL ratio deficit — almost all L1 |

So roughly **half the T4 board is chunk overhead** and is closable two cheap ways that
need no pipeline rework:

1. **Buy ~0.01% of T1 slack.** Block sizing (zlib-ng's symbol budget is 3x smaller
   than the number we inherited untested), TOO_FAR, insert policy. Any of these worth
   0.01% puts T4-with-independent-chunks under the rival at every affected label —
   and closes T1 size cells at the same time.
2. **Thread-aware chunk grid.** Now legal: the T-invariance "HARD INVARIANT" in
   `pipelined.rs:77-90` was retracted 2026-07-28. Chunks of `input/(k*T)` cut chunk
   count 25-100x, shrinking residual overhead to a few hundred bytes per file. Note
   the prior level-scaled-grid revert divided the constant WITHOUT slack and paid
   2-4% wall from imbalance; a T-aware grid with a split tail is a different shape and
   is unmeasured.

**The remaining 33 cells are a different front — the L1 size class (#25).** Gaps of
60 K-636 K bytes (access.log 911x the chunk overhead, monorepo 515x, data.csv 421x).
That is our `Fast` parser against libdeflate's `deflate_compress_fastest`, and no
seam, grid, or coding-locus change touches it.

**If both cheap routes fail**, the coding-locus rework returns as the fallback, and
its gate is then the seam-token diff: parse dict-seeded segments, concatenate token
streams, diff against T1's token stream. Deterministic, no pipeline code. NOT an
instruction-count falsifier — "instruction counts LOCATE, they never predict the
wall" is banked law and the earlier Ir-based gate violated it.

### Phase 2 — T1: converge on libdeflate's flat loop, then delete

Not more hoisting. **Transliterate the structure**, then let the removal oracle find
the deletions — decode's exact winning sequence.

The target: our hot loop threads state by reference (`in_base: &mut usize`,
`next_hashes: &mut [u32; 2]`) through `run_block` into an `#[inline(always)]`
`longest_match`. LLVM must prove non-aliasing among `buf`, `sink` and those `&mut`s
to keep `cutoff`/`nice_len`/`next_hashes[1]` in registers, and the operation diff
shows it does not: 1.7-2.5M memory reads each where libdeflate pays zero, plus 27%
of our matchfinder reads in unattributed spill/reload. libdeflate's equivalent is one
flat loop body owning its scalars.

Measured context: at L2 we execute 1.12x libdeflate's instructions (555.1M vs 496.0M) at better
IPC, with fewer frontend stalls and a lower L1D miss rate. 61% of the excess is LOAD
instructions. Our emit path is already 0.90x theirs (57.4M vs 63.8M). **The entire debt is
parse+matchfinder.**

Falsifier: flatten ONE parser (greedy/L2) so it owns every scalar, diff Ir/Dr against
the banked baseline (555.1M/104.0M vs libdeflate's 496.0M/83.8M). If the ~20M excess
Dr does not substantially close, stop.

Free oracle: output is byte-identical, so this is a pure refactor — byte-identity's
one legitimate use.

### Phase 3 — own the level table

The map is still libdeflate's verbatim, which is why zero slack exists at L2/L4-L9
and why every tiny T>1 cost fails a cell. Fix L4 monotonicity (gzip and zlib-ng both
go lazy at L4; we inherited libdeflate's outlier choice), sweep block sizing
(zlib-ng's symbol budget is 3x smaller than the number we inherited untested), insert
policy, TOO_FAR.

**Blocked on a law decision:** promotion rule clause 5 caps erosion on a passing cell
at 0.5%. Lazy-at-L4 costs 17.7% wall against passing gzip/pigz cells. Either the rule
gets a pre-registered carve-out for deliberate level-semantics fixes — landed
separately and first, per the rule's own discipline — or P4 stays broken. This needs
an explicit decision, not silent drift.

## 5. The command surface

Four verbs carry the campaign:

- **`fulcrum board`** — where do we stand. Failing cells only, ranked, stale-flagged
  against the subject commit, denominator stated. The loss map is GENERATED, never
  narrated.
- **`fulcrum why <cell>`** — the automated vendor diff. Both binaries at the same
  build shape (vendor with `-g` or it is one opaque symbol), per-line Ir+Dr, position
  counts, matched-thread counters, declared-parameter diff.
- **`fulcrum oracle <region>`** — the missing family. Stub a region, report size and
  wall response per level. Monotonic => critical path; flat => slack.
- **`fulcrum try <ref>`** — the gate. Both arms from git refs, NO-OP and stale-control
  refusal, verify, size+wall censuses at a shallow AND a deep level, promotion rule
  clause by clause, verdict SHIP / NO-SHIP(clause+numbers) / UNDECIDED(what to re-run).

## 6. Standing hazards, with the refusal for each

1. **A lever from our own profile is not a candidate.** Name the vendor difference or
   declare "no counterpart, bar is the measurement".
2. **"Floor" / "closed" / "no lever" is a bias firing, not a finding.** Five declared
   floors in decode, five refuted; one "CAMPAIGN CLOSED" survived 17 hours before a
   writev-gather flipped it. Banned unless a removal oracle has run AND the vendor is
   shown to pay the same cost. "Finally" in our own prose is the documented tripwire.
3. **Verify the artifact before the number.** No measurement counts without route
   assertion, binary sha, explicit thread count, `/dev/null` both arms.
4. **A win that cannot move a failing cell's axis, or that sits unmerged, is zero.**
   Name the cell and the axis in writing first. `gh pr list` before any new lever.
5. **Trust no instrument that has not been made to lie on purpose.** Decode's
   governing scoreboard was wrong-SIGN (best-of-N), its sink hid losses (file write),
   its classifier inverted verdicts on the exact case at issue while a 40-check
   selftest passed. Every verdict needs a same-run A/A bracketing 1.0 and a selftest
   containing the case being decided.

## 7. What the architecture docs still need

- The T>1 **coding-locus** decision. `encoder-architecture.md` says parallel
  "segments the input, calls the T1 kernel per segment, orders output" — satisfiable
  by a design that structurally cannot pass the board. Record the per-chunk constant,
  the zero-slack interaction, and the token-handoff target.
- Replace the stale "structural suspects" list (indirection, prefetch, bounds checks)
  with the operation-level verdict: loads and register pressure at the
  parse<->matchfinder interface.
- `encoder-architecture.md:17-23` still calls byte-identity with libdeflate "a
  contract". That is the cage codified in the target doc.
- `pipelined.rs:77-90` declares T-invariant bytes a "HARD INVARIANT". It was retracted
  as a requirement on 2026-07-28. Keep it as a product choice or drop it, but it is
  not a correctness law and it is what forbids a thread-aware grid.
- An emit-throughput budget per level (Ir/byte). Phase 1's scaling ceiling depends on
  it and it exists nowhere.
- RSS as a stated axis with a number.
