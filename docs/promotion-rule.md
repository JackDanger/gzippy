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

**5. Erosion budget on passing cells.** A passing cell may degrade only by the smaller of
a quarter of its margin and 0.5%:

```
new_ratio - old_ratio  <=  min(0.25 * (1.0 - old_ratio), 0.005)
```

This is what stops death by a thousand cuts — many individually "harmless" degradations
that never flip a cell on their own until one finally does and nothing can be attributed.

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
