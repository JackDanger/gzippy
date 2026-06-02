# Open Candidates — not-yet-refuted levers with ship-gates

These are directions that have NOT been causally refuted and remain buildable, with
explicit gates that must be passed before landing.

---

## 1. vmsplice-on-pipe

**What it is:** Use `vmsplice(2)` for zero-copy page-flip delivery to a pipe sink
(stdout pipe, not file output). Closes ~4–5% of the pipe-output wall gap by
eliminating the write syscall's kernel copy.

**Why not yet refuted:** A prior run showed ~4–5% improvement on a pipe sink.
However, a later N=41 interleaved run sign-FLIPPED (slight regression). The
measurement is sign-unstable and does not meet the N≥21 frozen-clock ship-gate.

The file-write arm of the data-plane port (writev on file) was REFUTED (PR#123
REGRESSED on file output). This candidate is PIPE-ONLY — do not conflate with
the file-write direction.

**Correctness fixes on feat/dataplane-2touch (must cherry-pick for any revival):**
- vmsplice UAF (use-after-free on early-death of the pipe reader)
- vmsplice early-death SIGPIPE handling
- vmsplice SIGPIPE recovery path

**Ship-gate:**
1. Build a `(b)`-only path (vmsplice on pipe, zero changes to the file-write arm)
   over MONOLITHIC buffers (not the segmented port) — one variable.
2. writev ON/OFF perturbation, N≥21 interleaved, frozen-clock neurotic harness
   (`scripts/bench/clean_bench.sh`), real pipe sink (not /dev/null), sha-verified.
3. Sign-stable win at T4/T8/T16 AND T8-noSMT ratio improves by more than the 1–2%
   spread. A TIE at this gate is a TIE, not a rejection — can re-attempt with a
   mechanism.

**Branch:** `feat/dataplane-2touch` (writev/vmsplice path is there alongside the
in-place-resolve code)

---

## 2. segment-native-A3 (SegmentedU8 + A3 in segment 0)

**What it is:** Re-land the faithful `DecodedData` SegmentedU8 port from
`feat/footprint-align` (commit `2b8bfae`) WITH the A3 prefill restored as a
prefault of segment 0 only (not the full monolith).

**Why not yet refuted:** The `feat/footprint-align` port achieved:
- Footprint: –29% RSS (1040 → 738 MB), moving toward rapidgzip's ~350 MB
- T4 / T8 wall: TIE (within frozen-clock spread)

It was HELD (not landed) because removing A3 caused T16 +5.5% regression. A3 is
a measured +4.2% T16 production win (`gzip_chunk.rs:178`) — prefaulting the chunk's
output buffer at decode start. The segmented port removed it because the first
128 KiB segment can be prefaulted instead.

**Predicted outcome:** Footprint −29% (confirmed in `feat/footprint-align`) +
T16 TIE-or-better (restoring A3 semantics via segment-0 prefault). This is the
ONLY footprint lever with a clear fix path and a non-regressing segmentation history.
It is also the highest-faithfulness change: gzippy's `data_with_markers` is still a
glibc monolith — a future O2 pass (segment data_with_markers + glibc→rpmalloc) can
layer on top.

**Ship-gate:**
1. Base: `feat/footprint-align` commit `2b8bfae`.
2. Add A3-equivalent prefault: at `SegmentedU8` construction time (or decode task
   start), prefault segment 0 of the new segmented buffer.
3. Measure T4/T8/T16 with spread on frozen-clock neurotic harness (N≥9 interleaved,
   sha-verified). Gate: T16 ≤ baseline +2% (within spread). A TIE at T8 is
   acceptable (footprint win is independently valuable per the memory-optimization
   mandate).
4. Advisor sign-off before landing.

**Branch:** `feat/footprint-align` (commit `2b8bfae` is the reference segmented port)

**Note on residency hypothesis:** The page-fault removal oracles proved that faults
themselves are overlapped slack (`docs/dead-ends/page-faults.md`). Those oracles
warmed the MONOLITHIC buffer — they did NOT test the 128 KiB segment's TLB residency
benefit (a reused 128 KiB segment = 32 hot pages in dTLB+L2). The residency
hypothesis is untested for segmentation. The ship-gate measurement will determine
whether it pays; if it TIEs at T4/T8, the footprint win (–29% RSS, independently
valuable for memory-constrained deployments) may still justify landing.

---

## Lever Board Summary

| Candidate | Branch | Predicted gain | Gate status |
|-----------|--------|----------------|-------------|
| vmsplice-on-pipe | `feat/dataplane-2touch` | ~4–5% pipe-sink wall | sign-unstable; N≥21 needed |
| segment-native-A3 | `feat/footprint-align` @ 2b8bfae | footprint −29% + T16 TIE | A3-in-seg0 implementation needed |
