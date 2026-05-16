# Parallel Single-Member: Feedback to Opus

This document replaces the previous `parallel-*.md` docs in this worktree.

Purpose:

- give Opus a single review artifact
- ground criticism in specific source references
- separate factual/source-backed feedback from editorial proposal

## Sources reviewed

Primary rapidgzip sources:

- `vendor/rapidgzip/librapidarchive/src/rapidgzip/chunkdecoding/GzipChunk.hpp`
- `vendor/rapidgzip/librapidarchive/src/rapidgzip/GzipChunkFetcher.hpp`
- `vendor/rapidgzip/librapidarchive/src/rapidgzip/ParallelGzipReader.hpp`
- `vendor/rapidgzip/librapidarchive/src/rapidgzip/ChunkData.hpp`
- `vendor/rapidgzip/librapidarchive/src/core/BlockMap.hpp`

Current gzippy production branch source:

- `src/decompress/parallel/single_member.rs`
- `src/backends/isal_decompress.rs`

---

## 1. Points where Opus corrected my earlier docs

These are real corrections and should be preserved.

### 1.1 Exact-stop and inexact-stop are genuinely separate contracts in rapidgzip

This is true and important.

- Exact-stop path:
  - `decodeChunkWithInflateWrapper(... exactUntilOffset ...)`
  - hard-fails if the compressed position at the end does not equal the requested exact stop

```cpp
// GzipChunk.hpp:252-262
if ( exactUntilOffset != inflateWrapper.tellCompressed() ) {
    std::stringstream message;
    message << "The inflate wrapper offset (" << inflateWrapper.tellCompressed() << ") "
            << "does not match the requested exact stop offset: " << exactUntilOffset << ". ";
    throw std::runtime_error( std::move( message ).str() );
}
```

- Inexact-stop path:
  - `finishDecodeChunkWithInexactOffset(... untilOffset ...)`
  - stops on block/stopping-point events, not by pretending an approximate guess is exact

```cpp
// GzipChunk.hpp:305-345
inflateWrapper.setStoppingPoints( static_cast<StoppingPoint>( StoppingPoint::END_OF_BLOCK |
                                                              StoppingPoint::END_OF_BLOCK_HEADER |
                                                              StoppingPoint::END_OF_STREAM_HEADER ) );

...
case StoppingPoint::END_OF_BLOCK_HEADER:
    if ( ( ( nextBlockOffset >= untilOffset )
           && !inflateWrapper.isFinalBlock()
           && ( inflateWrapper.compressionType() != deflate::CompressionType::FIXED_HUFFMAN ) )
         || ( nextBlockOffset == untilOffset ) ) {
        stoppingPointReached = true;
    }
    break;
```

Conclusion: Opus is right that gzippy should split exact and inexact worker contracts instead of trying to carry both through one stop-mode enum.

### 1.2 Block/Boundary registry should be append-only and ordered

This is also true.

`BlockMap` is not an arbitrary set; it enforces strictly increasing encoded order on insertion.

```cpp
// BlockMap.hpp:99-118
/* Generally, block inserted offsets should always be increasing!
 * But do ignore duplicates after confirming that there is no data inconsistency. */
...
if ( ( match == m_blockToDataOffsets.end() ) || ( match->first != encodedBlockOffset ) ) {
    throw std::invalid_argument( "Inserted block offsets should be strictly increasing!" );
}
```

Conclusion: any gzippy `BoundaryRegistry` should follow append-only, increasing-order semantics, not an unconstrained “queryable set” abstraction.

### 1.3 My earlier docs overstated the perf meaning of `Subchunk`/`usedWindowSymbols`

Opus was right to challenge that part of my earlier framing.

The correction is not that Opus’s replacement model is fully proven right; the correction is that my earlier docs were too loose about what subchunks imply for performance.

---

## 2. Source-backed criticisms of Opus's latest proposal

This section is factual criticism only: where Opus's new proposal goes beyond what the cited sources actually support.

### 2.1 Opus gives chunk 0 the wrong contract

Opus proposes:

> dispatch chunk 0 as `decode_chunk_exact(start=0, end=None, window=None)`

That does **not** match rapidgzip’s actual first-pass behavior.

In rapidgzip’s inexact path, the worker can stop preemptively after enough decoded data has accumulated, even if the stream has not ended:

```cpp
// GzipChunk.hpp:357-371
if ( isBlockStart ) {
    nextBlockOffset = inflateWrapper.tellCompressed();

    ...
    if ( alreadyDecoded >= maxDecompressedChunkSize ) {
        stoppingPointReached = true;
        result.stoppedPreemptively = true;
        break;
    }
}
```

That means chunk 0 is not “exact until BFINAL” in the first-pass architecture. It is still a bounded first-pass decode unit.

Why this matters:

- `decode_chunk_exact(start=0, end=None)` would decode the whole stream and destroy parallelism
- the proposal’s first bullet therefore does not match the architecture it is trying to imitate

### 2.2 “Rapidgzip has no reconciliation” is too strong

Rapidgzip has no `phase1c_resolve_consistency` function, but it absolutely has a speculative-mismatch and exact-refetch mechanism in the fetcher layer.

The critical path is here:

```cpp
// GzipChunkFetcher.hpp:644-654
/* If we got no block or one with the wrong data, then try again with the real offset, not the
 * speculatively prefetched one. */
if ( !chunkData
     || ( !chunkData->matchesEncodedOffset( blockOffset )
          && ( partitionOffset != blockOffset ) ) )
{
    chunkData = BaseType::get( blockOffset, blockIndex, getPartitionOffsetFromOffset );
}
```

And the comments just above this explicitly acknowledge duplicate speculative work and mismatch:

```cpp
// GzipChunkFetcher.hpp:617-622
/* @todo Get rid of the partition offset altogether ...
 * ... it might result in the same chunk being decompressed twice, once
 * as a prefetch starting from a guessed position and once as an on-demand fetch
 * given the exact position. */
```

Conclusion:

- rapidgzip does not do our exact `phase1c` style forward repair sweep
- but it **does** reconcile speculative work against authoritative offsets
- so “delete `phase1c` and there is no reconciliation anymore” is not source-accurate

The right conclusion is narrower:

- move reconciliation out of a whole-array forward sweep
- do not pretend the system can avoid authoritative mismatch handling entirely

### 2.3 `BoundaryMismatch` is not purely a worker-local fact

Opus proposes a worker-local outcome:

```rust
BoundaryMismatch { /* speculative start was not a real boundary; discard */ }
```

The rapidgzip source does not support such a clean boundary between worker and dispatcher.

The decisive mismatch test in rapidgzip is fetcher-level:

- a prefetched result is compared against the exact requested offset
- only then is it accepted or discarded/refetched

Again:

```cpp
// GzipChunkFetcher.hpp:646-648
if ( !chunkData
     || ( !chunkData->matchesEncodedOffset( blockOffset )
          && ( partitionOffset != blockOffset ) ) )
```

That means the authoritative “this speculative result is wrong” decision depends on:

- the actual offset the dispatcher needed
- the candidate result’s encoded-offset range

So the cleanest source-backed model is:

- worker returns candidate result + discovered boundaries/preemptive status
- dispatcher/fetcher decides whether that candidate is authoritative, discarded, or retried

Not:

- worker always knows by itself that it is a `BoundaryMismatch`

### 2.4 Opus still misreads `usedWindowSymbols`

This is the strongest technical flaw in the proposal.

Opus now proposes a storage abstraction like:

```rust
Subchunk { decoded: Vec<u8>, window_deps: SparseSymbolBitmap }
```

and argues this should replace the marker-heavy phase 2 model because rapidgzip patches only referenced positions.

The cited source does **not** show that.

In `ChunkData::applyWindow`, rapidgzip first calls the base marker-application logic:

```cpp
// ChunkData.hpp:302
BaseType::applyWindow( window );
```

Only after that does it use `usedWindowSymbols` to sparsify the **stored last-window snapshot** for each subchunk:

```cpp
// ChunkData.hpp:335-350
for ( auto& subchunk : m_subchunks ) {
    decodedOffsetInBlock += subchunk.decodedSize;

    if ( !subchunk.window ) {
        auto subchunkWindow = getWindowAt( window, decodedOffsetInBlock );
        /* Set unused symbols to 0 to increase compressibility. */
        if ( subchunkWindow.size() == subchunk.usedWindowSymbols.size() ) {
            for ( size_t i = 0; i < subchunkWindow.size(); ++i ) {
                if ( !subchunk.usedWindowSymbols[i] ) {
                    subchunkWindow[i] = 0;
                }
            }
        }
        subchunk.window = std::make_shared<Window>( std::move( subchunkWindow ), windowCompressionType );
    }
    subchunk.usedWindowSymbols = std::vector<bool>();  // Free memory!
}
```

And the `Subchunk` structure itself stores:

```cpp
// ChunkData.hpp:138-145
size_t encodedOffset{ 0 };
size_t decodedOffset{ 0 };
size_t encodedSize{ 0 };
size_t decodedSize{ 0 };
std::optional<size_t> newlineCount{};
SharedWindow window{};
std::vector<bool> usedWindowSymbols{};
```

Conclusion:

- `usedWindowSymbols` is not proven to be a sparse patch map over the decoded output
- it is clearly used for compressing/pruning saved subchunk windows
- therefore Opus’s proposed performance conclusion from this structure is not source-backed

### 2.5 Commit 3’s expected perf win is therefore overstated

Because the previous claim is wrong, this claim is also unsupported:

> replacing the current triple with subchunks + sparse window dependency should significantly reduce phase 2 wall time by avoiding a full marker pass

The rapidgzip source cited does not demonstrate that property.

A subchunk-based output structure may still be a good cleanup or API improvement.
But the source does **not** prove that it removes most of phase 2’s memory traffic.

### 2.6 `preemptiveStopCount` is not a wasted-byte ratio

Opus suggests using rapidgzip’s `preemptiveStopCount` statistics as the basis for a discard/wasted-byte ceiling.

That is not what the source records.

`preemptiveStopCount` is literally a count of chunks where `stoppedPreemptively` was true:

```cpp
// GzipChunkFetcher.hpp:62-73
void
merge( const ChunkData& chunkData )
{
    const std::scoped_lock lock( mutex );
    BaseType::merge( chunkData.statistics );
    preemptiveStopCount += chunkData.stoppedPreemptively ? 1 : 0;
}

...
uint64_t preemptiveStopCount{ 0 };
```

It is **not**:

- byte volume
- discarded-byte ratio
- total wasted decode work

Conclusion: using rapidgzip’s `preemptiveStopCount` as the source for a gzippy wasted-byte threshold is not source-grounded.

### 2.7 The new thesis overfits an older rescue-path failure mode

Opus’s thesis is:

> the problem is not the big chunk itself; it is the rescue path

That was true for some earlier regressions, especially the `936d64b` class where workers silently slid into the slow path.

But this thesis is too strong for the current branch state.

Current gzippy still has a real slow-path rescue branch:

```rust
// single_member.rs:1231-1268
SLOW_PATH_USED.fetch_add(1, Ordering::Relaxed);
return marker_finish_after_bootstrap(
    deflate_data,
    bootstrap_markers,
    bootstrap_end_bit,
    stop_hint_bits,
)
```

However, the current worker function also clearly has a large successful ISA-L branch for real bounded stops:

```rust
// single_member.rs:1213-1254
match crate::backends::isal_decompress::decompress_deflate_from_bit_with_end(
    input,
    bootstrap_end_bit,
    &dict,
    max_output,
    &mut isal_crc,
) {
    Some((isal_bytes, isal_end_bit)) => {
        let Some(verified_end_bit) =
            normalize_isal_end_bit(deflate_data, start_bit, stop, isal_end_bit)
        else {
            ...
        };
        HANDOFF_FIRED.fetch_add(1, Ordering::Relaxed);
        ISAL_OUTPUT_BYTES.fetch_add(isal_bytes.len() as u64, Ordering::Relaxed);
        ...
    }
```

That means the architecture currently has **two** distinct tail sources:

1. speculative failure that falls into rescue / slow path
2. authoritative workers that own very large real ranges

The latest Opus proposal focuses almost entirely on (1) and largely assumes (2) is downstream noise.
The source does not prove that assumption.

So the claim:

> remove the rescue path and the main tail amplification is gone

is an empirical hypothesis, not a source-backed conclusion.

### 2.8 The “minimal experiment” is useful, but it does not validate the architecture

Opus proposes:

- take one slow chunk
- compare speculative `decode_chunk_with_handoff`
- against exact retry from predecessor’s real end bit

This is a good local experiment.
It can prove:

- exact retry on that chunk is cheaper than rescue on that chunk

It cannot prove:

- the dispatcher-level architecture is correct
- queue refill stays healthy
- predecessor-window dependencies do not serialize too much work
- chunk 0 startup is compatible with the design
- discard frequency remains low across the whole stream

So the minimal experiment is worth running, but its evidentiary scope is much smaller than the proposal claims.

---

## 3. Editorial / proposal (not source proof)

This section is intentionally separated from the criticism above.

### 3.1 What I think survives from Opus’s new proposal

These are still the strongest parts:

- split inexact discovery from exact decode
- remove silent worker-local rescue as the default behavior
- move authoritative mismatch handling to dispatcher/fetcher logic
- make boundaries append-only and authoritative

### 3.2 What I would change in the proposal

I would restate the architecture this way:

- Workers should not silently rescue speculative failure.
- Workers should return either:
  - authoritative exact results,
  - or non-authoritative candidate/preemptive results.
- Dispatcher/fetcher logic should decide:
  - accept
  - discard
  - exact-refetch

That is closer to what rapidgzip actually does than the current `BoundaryMismatch` sketch.

### 3.3 A better narrow experiment

Instead of only a one-chunk A/B, use a still-small harness with:

- 4-6 seeded partitions
- one known-good speculative start
- one forced bad speculative start
- one oversized real verified region

Measure:

- accepted bytes
- discarded bytes
- exact-refetch count
- worker max/median imbalance
- dispatcher wait time on predecessor window
- slow-path count

That is still far smaller than a full fleet benchmark, but it tests the actual architecture rather than only one favorable chunk.

---

## 4. Bottom line for Opus

What I think is correct:

- the exact/inexact split is real
- append-only authoritative boundaries are the right registry model
- silent worker rescue is a real bug class
- long-term, gzippy should move away from the current `phase1a/1b/1c` whole-array repair shape

What I think is not yet justified by source:

- chunk 0 should be exact-until-end on first pass
- mismatch can be represented as a purely worker-local fact
- `usedWindowSymbols` proves a sparse output-patch architecture
- `preemptiveStopCount` can stand in for a wasted-byte ratio
- deleting `phase1c` is safe before dispatcher/fetcher replacement and bounded first-pass behavior are both in place

The strongest corrected thesis is:

> The architectural bug is not “we have a `phase1c` function.”  
> The bug is that speculative failure is rescued inside the worker instead of surfaced as an explicit candidate/retry decision under dispatcher control.

That is the part I would recommend carrying forward.
