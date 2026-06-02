# Dead-Ends Ledger — gzippy parallel-SM campaign

## Campaign Verdict

**gzippy's decode engine is competitive-or-faster than rapidgzip on a per-instruction
basis.** The 3-bucket+TMA instrument (2026-06-02) established:

- **Instructions: EQUAL** (10.90B vs 10.84B = 1.01× at T8-noSMT). There is no
  algorithmic or compute gap.
- **Cycles: 1.42×** (7.17B vs 5.06B) — the gap is entirely STALLS.
- **arm64: gzippy is FASTEST** — beats rapidgzip 1.2–2.8× T1–T8 at every thread
  count since 2026-05-30.

The **~1.18× T8-noSMT / ~1.54× T16 x86 gap** is:
- Fundamental in-order-consumer coordination (the serial window-resolution publish
  chain; adding workers shrinks the parallel decode term but not the serial chain).
- A memory-traffic CORRELATE of rapidgzip's leaner 128 KiB segmented buffer
  structure — the correlate is real but removing it (6 independent oracles)
  consistently yields 0% wall improvement.
- **NOT** inner-loop decode speed (gzippy window-absent is 0.64× — i.e., FASTER
  than rapidgzip's; clean decode TIEs at 1.04×).
- **NOT** recoverable by any single lever that survived causal perturbation testing.
  Every located lever has been perturbed and falsified.

The T16 negative-scaling is **substantially SMT** — rapidgzip regresses identically
T8→T16 (+22.7% vs gz +20.8%). It is not a gzippy structural defect.

**Status: EARNED STOP** (2026-06-02, advisor-confirmed) — falsification not fatigue.
Five independent causal refutations, cross-tool comparison, and a thorough
3-bucket+TMA+static decomposition all converge on the same conclusion: the residual
is diffuse in-order-consumer coordination with no single recoverable lever. The only
structurally-named un-refuted candidate (eager successor-window resolution) is
murky/mostly-refuted.

---

## Per-Lever Index

| File | Lever | Verdict |
|------|-------|---------|
| [data-plane-2touch.md](data-plane-2touch.md) | Granule + in-place-resolve + writev/vmsplice (aggregate port PR#123) | REFUTED — file sink REGRESSED +6–12%; vmsplice-pipe sign-unstable |
| [footprint-align-segmented.md](footprint-align-segmented.md) | Faithful SegmentedU8 DecodedData port (commit 2b8bfae) | HELD — footprint −29% but T16 +5.5% from A3-removal; re-entry = segment-native A3 |
| [footprint-bandwidth.md](footprint-bandwidth.md) | Resident footprint / DRAM-bandwidth as wall lever | REFUTED — 6 independent oracles; footprint is overlapped slack |
| [coherence-contention.md](coherence-contention.md) | Cache-coherence / lock-contention | REFUTED — HITM 0.05–0.15%, gzippy≈rapidgzip at T16, all kernel-side |
| [page-faults.md](page-faults.md) | Page-fault / TLB-walk removal | REFUTED — 3 removal oracles all TIE; amplify-binds/remove-doesn't = overlapped slack |
| [copies-wall-neutral.md](copies-wall-neutral.md) | Copy-elimination (absorb_isal_tail 212 ms, consumer publish copy) | REFUTED — 0.0% wall; copies are worker-overlapped, off critical path |
| [placement-resolve-ahead.md](placement-resolve-ahead.md) | Frontier-placement / prefetch-decode-ahead port | CAUSALLY REFUTED — oracle TIE; gzippy already prefetches frontier at 94% |
| [eager-resolve-phantom.md](eager-resolve-phantom.md) | Eager post-process / resolve-ahead during consumer stall | REFUTED — 0 ready tasks (dependency-blocked); consumer-side variant +195 ms net loss |
| [fill-lever.md](fill-lever.md) | Consumer-wait 167 ms vs rapidgzip 18 ms "fill lever" | DEAD — guard-rejects≈0; priority dispatch already shipped + oracle-TIE'd |
| [marker-over-marking-phantom.md](marker-over-marking-phantom.md) | Over-marking / excess window-absent fraction | NOT A LEVER — static fraction matches vendor (31.97% vs 31.25%); runtime 95% is structural speculation depth |

---

## Open Candidates

See [`docs/open-candidates.md`](../open-candidates.md) for the two not-yet-refuted
directions with their ship-gates:

1. **vmsplice-on-pipe** — ~4–5% pipe-gap closure; sign-unstable; needs N≥21
   build-(b)-only test on a real pipe.
2. **segment-native-A3** — re-land `feat/footprint-align`'s SegmentedU8 WITH A3
   prefilled into segment 0; predicted footprint −29% + T16 TIE-or-better.

---

## Methodology Notes

The campaign established several durable measurement rules (encoded in `CLAUDE.md`):

- **Slow-down slope ≠ speed-up ceiling** (Rule 3): amplifying a region proves
  touchability, NOT that speeding it pays. Always run the REMOVAL oracle.
- **CPU-sums lie**: a region can be large in total worker CPU and 0% of wall when
  it is overlapped across threads. The 212 ms copy was the load-bearing example.
- **Causal perturbation disposes; attribution proposes**: `fulcrum critpath`,
  `decompose`, and source-reading are hypothesis generators. Only a frozen
  interleaved removal oracle is the verdict.
- **Instrument before optimizing**: three instruments were silently broken during
  this campaign (clean-window oracle emitting empty output; `combine_crc` 62 ms
  double-count phantom; `non_marker_count` misnamed field). Validate
  positive/negative controls before trusting any number.
