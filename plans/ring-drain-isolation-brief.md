# Ring→data drain isolation — brief for disproof advisor

## Question
The landed copy-free FOLD drain left a residual `native_fold 0.737× → ocl_cf 0.925×`
= 0.188× UPPER BOUND. The prior advisor (fold-contig-landed) said this confounds THREE
terms: (1) intrinsic symbol rate, (2) the engine ring-write, (3) the ring→`chunk.data`
drain memcpy that `ocl_cf` (ISA-L decodes straight into the final buffer) does NOT pay.
The advisor asked for a same-engine pure-Rust ring-based copy-free-to-final oracle to
strip term (3) from the residual. This turn built and ran that isolation.

## The isolation instrument (measurement-only, OFF==identity, WRONG BYTES when ON)
Two env knobs in `gzip_chunk.rs` `ContigFoldSink::push_clean_u8`:
- `GZIPPY_FOLD_NODRAIN=1` — skip the ring→data drain memcpy (`extend_from_slice`),
  advance `chunk.data` length via `writable_tail_reserve`+`commit` over uninitialized
  space so all downstream accounting (subchunk decoded_size, writev iovecs,
  window-publish) stays panic-free. The DECODE itself (engine `block_body`, ring write,
  back-ref resolution from `output_ring`) is UNCHANGED ⇒ wall delta isolates term (3).
- `GZIPPY_FOLD_NOCRC=1` — additionally skip the per-clean-byte CRC `update` (CRC IS
  paid by `ocl_cf`, so this is the symmetric control: how much of the second-touch is
  CRC vs the copy).

### Self-test (Rule 4)
- OFF (both knobs unset): sha 028bd002…cb410f == rg == production ⇒ byte-exact identity.
- NODRAIN=1: sha fc7be034… (different) ⇒ the knob FIRES. Decode runs fully; the run
  exits 1 ONLY at the terminal gzip-trailer CRC32 check (after the full decode + output
  stream) — diagnosed: `Decompression error: parallel SM: CRC32 mismatch`. So the
  measured wall IS the full decode wall; the failure is a post-decode trailer check, not
  an early abort.

## The numbers (locked guest REDACTED_IP, taskset 0,2,4,6,8,10,12,14, gov=perf,
## interleaved measure.sh N=11, RAW=211968000, native build target-cpu=native, HEAD
## fc7336c3 + the 2 isolation knobs)
4 interleaved passes, best-of-11 each (the jitter floor):

| pass | native_fold | nodrain | nodrain_nocrc | rg     | nodrain/native |
|------|-------------|---------|---------------|--------|----------------|
| 1    | 0.1824      | 0.1645  | 0.1667        | 0.1358 | 1.109×         |
| 2    | 0.1807      | 0.1669  | 0.1691        | 0.1378 | 1.083×         |
| 3    | 0.1823      | 0.1684  | 0.1707        | 0.1464 | 1.083×         |
| 4    | 0.1831      | 0.1675  | 0.1673        | 0.1381 | 1.093×         |

- native_fold best ∈ [0.1807, 0.1831]; nodrain best ∈ [0.1645, 0.1684]. The two ranges
  NEVER OVERLAP across 4 independent passes — sign-stable, monotonic.
- Within-pass spread is high (28–76%, this guest's inherent interleaved tail spread), so
  measure.sh's spread-rule prints TIE; but the best-of-N (the charter's metric) cleanly
  separates.
- nodrain_nocrc ≈ nodrain (Δ < 0.3%) ⇒ the CRC second-touch is NEGLIGIBLE; the
  recoverable component is the ring→data drain MEMCPY specifically.

## Conclusion (claims to disprove)
- C1: removing the ring→data drain memcpy moves native_fold ~0.745× → ~0.812× rg =
  **~+0.067× recoverable** (sign-stable best-of-N, 4 passes, non-overlapping).
- C2: the 0.188× residual to ocl_cf 0.925× therefore SPLITS into ~0.067× ring-drain-copy
  (free to pure-Rust, the next tooth) + ~0.11× UPPER BOUND remaining = intrinsic symbol
  rate + the engine ring-WRITE (which nodrain does NOT remove). The true inner-Huffman
  symbol-rate gap is ≤ 0.11×, smaller than the prior 0.188× bound.
- C3: CRC is not the lever (nodrain_nocrc ≈ nodrain).
- To BANK C1 needs the byte-EXACT copy-free-to-final refactor (engine writes clean
  output directly into chunk.data's reserved tail, eliminating the ring for the clean
  phase so the drain copy disappears WITH correct bytes). The no-drain knob only BOUNDS it.

## Frequency-neutral evidence (angle 1, pre-empted)
The nodrain/native ratio is LOAD-INVARIANT: load 2.33 → 1.109×, load 1.97 → 1.083×/1.083×,
load 1.21 → 1.093×. A turbo artifact would SHRINK the delta as load rises (less turbo
headroom under contention); a flat ratio across load 1.2→2.3 is the frequency-neutral
signature (interleaved = both contenders see identical per-trial contention). The ~+0.067×
survives the load sweep.

## Disproof angles requested
1. Is the best-of-N non-overlap a valid signal given the high within-pass spread, or is
   the ~15ms a turbo/frequency artifact (need a freq-neutral control)?
2. Does skipping `extend_from_slice` but committing uninitialized space genuinely leave
   the DECODE timing unchanged, or does writing garbage to chunk.data change cache
   behavior of subsequent reads (window-publish, marker-resolve gather over chunk.data)
   in a way that under- or over-states the drain cost?
3. Is ~0.11× a sound UPPER BOUND on the remaining intrinsic+ring-write gap, or does the
   nodrain run ALSO accidentally remove some legitimate work (e.g. the 2 naturally-seeded
   chunks' clean continuation, or window-publish reading short/garbage data)?
4. Is the byte-exact copy-free-to-final refactor the right faithful next step, and does
   rapidgzip's DecodedData actually decode the clean tail into one contiguous buffer with
   no ring (the existence proof for ~+0.067×)?
