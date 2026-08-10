# The promotion rule

This is the rule a change must clear to ship. It is written down so it cannot be adjusted
after a measurement comes back inconvenient.

## Why it is being rewritten (2026-07-28)

The previous rule was an **ablation gate**: across four entropy classes and a set of levels,
no cell may regress beyond 1.02 and the geometric mean must be <= 1.0.

It rejected a change that **closed two per-label cells and flipped none**. Investigating
that disagreement showed the gate was mis-specified rather than unlucky: it scores a 2%
slowdown at a level where we already beat libdeflate by 12% exactly the same as a 2%
slowdown at a level where we are behind. The contract does not.

**The gate was NOT changed in the branch that failed it.** The failing change was reverted
first, this rule landed separately, and only then is that change re-evaluated. Changing a
rule in the same breath as the result that embarrassed it is the one move this project
forbids outright, and "my proxy was wrong" is exactly what it would sound like from the
inside either way.

## The rule

A change may ship when **all** of the following hold.

**1. Correctness — absolute.** Compress, decompress with our own decoder at every thread
count, sha256 against the original, plus every independent decoder present (gzip, pigz,
libdeflate). Zero failures. (`fulcrum verify`)

**2. It must actually change the generated code.** If both arms compile to identical
binaries the change is a no-op and no timing result about it is meaningful. Checked from
binary hashes before any measurement. (`fulcrum ablate`)

**3. No pass -> fail flips.** Not one. Any previously-passing per-label cell that fails
after the change blocks the ship, regardless of what else improved.

**4. Progress.** At least one failing cell closes, **or** the total fail-gap
(`sum of max(0, ratio - 1)` over failing cells) decreases by at least 1%.

**5. Margin-floor erosion rule on passing cells** *(redesigned 2026-08-10; implemented in
[fulcrum#25](https://github.com/JackDanger/fulcrum/pull/25) under the owner's delegated
redesign).*

> ~~A passing cell may degrade only by the smaller of a quarter of its margin and 0.5%:
> `new_ratio - old_ratio <= min(0.25 * (1.0 - old_ratio), 0.005)`~~
>
> **Struck 2026-08-10.** The flat budget convicted on single-layout lottery rolls and
> priced margin at zero. It survives below only for thin-margin cells.

The rule now has two parts, both applying to **wall** cells only — size cells are exact
integers and unchanged: a size erosion or size flip convicts directly, no confirmation
involved.

*Convictions require cross-layout confirmation.* A beyond-budget wall erosion suspect —
and a wall pass -> fail flip that survives clause 3's existing 3x-n re-measure — only
convicts after the cross-layout confirm machinery (`fulcrum layout confirm`, auto-run by
`fulcrum try`) says CONFIRMED-REAL. One confirm covers each suspect (corpus, level,
threads) coordinate, capped at ~12 coordinates per run; the cap and any overflow are
stated in the output, and overflow suspects stay UNDECIDED — never convicted by default.
LAYOUT-ARTIFACT acquits, with the confirm numbers printed; confirm-UNDECIDED leaves the
suspect UNDECIDED. A coordinate the floors file does not cover is UNDECIDED with
`layout calibrate` named — a floor is never borrowed from another coordinate.

*The budget is the margin floor.* A CONFIRMED-real erosion on a **winning** wall cell
(pre-lever ratio <= 0.80) is acceptable iff the confirmed post-lever ratio still clears
the floor:

```
post_ratio  <=  min(0.80, 1 - 3 * layout_floor(cell))
```

Thin-margin cells (pre-lever ratio > 0.80) keep the old flat 0.005 budget exactly as
written above. **Clause 3 remains absolute**: a CONFIRMED-real pass -> fail flip blocks
the ship regardless of margin.

This still stops death by a thousand cuts — the floor is a hard line no accumulation of
"harmless" degradations may cross — while pricing margin as capital: a cell won 4-5x may
spend some of that win to close cells elsewhere, which is the point of a rival-anchored
Pareto goal.

*Receipts for the redesign.* The #295/#296/#310 adjudications were vetoed on erosion
lists dominated by proven layout artifacts — cross-layout confirms went 2/2
LAYOUT-ARTIFACT on the tested drivers, including cells whose code was byte-identical
between arms — while the genuinely real erosions were 2-9% on cells we win by 4-5x. The
campaign goal is rival-anchored Pareto dominance per label, not preservation of every
interior number; margin is capital, and a rule that forbids spending it converts wins
into cages.

**6. Net improvement.** Total improvement on failing cells must exceed total harm on
passing cells by at least 2x. This stops closing one easy cell by worsening many hard ones.

**7. Cross-architecture.** Rules 3-6 hold on every architecture we measure (aarch64, Intel,
AMD Zen2). A win on one architecture and a loss on another is not a win. Wall verdicts come
from the frozen box.

**8. The statistical method is fixed before the run** — paired interleaved, per-pair median
ratio, stated n. Note the measured A/A noise floor of this rig is ~1.5% on the worst cell at
n=15, so a single cell moving less than that is not evidence of anything; raise n on close
calls rather than reading the tea leaves.

## Scope

The board is per-label — our level N against their level N — on **both** size and wall,
against all four rivals, over the corpus. A change evaluated on a narrower slice has not
been evaluated; say so rather than generalising from it.

## How to tell a proxy correction from an excuse

Before changing this rule again, all of these must pass:

* **Symmetry.** Would I propose the same change if the patch had failed the *new* rule and
  passed the old one?
* **Timing.** Rule changes land separately from, and before, the change they would affect.
  Never in the same branch.
* **Backtest.** Apply the old and new rules to past accepted and rejected changes. The new
  rule should predict long-run board progress better, not merely bless the present patch.
* **Counterfactual, written first.** State what result would cause a revert before running
  the measurement.
