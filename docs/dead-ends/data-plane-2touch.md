# Dead End: data-plane-2touch (granule + in-place-resolve + writev/vmsplice)

## Hypothesis

Port rapidgzip's full data-plane output path: 128 KiB granule `DecodedData`
(SegmentedU8 + segmented u16 marker buffer), in-place-resolve (narrow u16→u8 in the
same allocation, eliminating the separate `narrowed` buffer), and writev/vmsplice for
scatter-gather output to pipe/file sinks. Together these would close the 1.47×
T8-noSMT wall gap by matching rapidgzip's leaner buffer lifecycle and eliminating
extra copies.

## How Measured

Measured on the clean frozen-clock neurotic harness (`scripts/bench/clean_bench.sh`,
frozen-clock, N≥7 interleaved, sha-verified, T4/T8/T16).

- **Aggregate port PR#123** (`feat/dataplane-2touch`): landed granule + in-place + writev
  simultaneously and measured the WHOLE-SYSTEM wall vs rapidgzip.
- **writev-on-pipe sub-component** measured separately vs monolithic buffer output,
  N=41 interleaved.
- **vmsplice-PIPE path** identified as the one sub-component that cannot be ported
  to a file-write sink (Linux-specific, pipe-only, no cross-platform equivalent in
  the file-write arm).

## Verdict: REFUTED (file sink) / SIGN-UNSTABLE (pipe sink)

**File output (the production sink for most users):**
- PR#123 aggregate removal REGRESSED wall +6–12% across T4/T8/T16 and increased
  dtlb store-walks +22%. The port made things measurably worse.
- The regression mechanism: the faithful 128 KiB granule + in-place-resolve + writev
  together removed the A3 prefill (a measured +4.2% T16 production win from `gzip_chunk.rs:178`).
  Removing A3 re-exposes T16 starvation and adds overhead.
- **Key finding:** the writev/in-place direction is INERT for file output — the
  vmsplice win is pipe-only (zero-copy page-flip to the reader). For the file-write
  sink, in-place-resolve + writev is structurally equivalent to the existing path
  once copies are confirmed wall-neutral (`project_copies_wall_neutral_2026_05_29`).

**Pipe output (vmsplice path):**
- Closes ~4–5% of the pipe-output gap but sign-FLIPPED in one N=41 run vs a prior run.
- Ship-gate NOT met: result is sign-unstable under the required N≥21 interleaved
  frozen test over monolithic buffers with writev ON/OFF as the single variable.

**Durable correctness fixes on this branch (kept for any revival):**
- vmsplice UAF (use-after-free on early-death of the pipe reader)
- vmsplice early-death SIGPIPE handling
- vmsplice SIGPIPE recovery

## Code Location

Branch: `feat/dataplane-2touch`

The correctness fixes for vmsplice (UAF, early-death, SIGPIPE) live on this branch
and should be cherry-picked to any future vmsplice revival before the performance
experiment.

## Open Candidate Status

The vmsplice-on-pipe sub-path remains an open candidate (not yet cleanly refuted).
See `docs/open-candidates.md` for the ship-gate required before landing it.

## Related Entries

- `docs/dead-ends/footprint-bandwidth.md` — the broader theory that DRAM/footprint
  drives the gap (6 refutations including this aggregate port)
- `docs/dead-ends/copies-wall-neutral.md` — absorb_isal_tail copy (212 ms CPU) = 0%
  wall, establishing that copy-elimination is off the critical path
- `project_t8_gap_fully_mapped_2026_06_02` memory — EARNED STOP: footprint levers
  fully refuted including this aggregate port
