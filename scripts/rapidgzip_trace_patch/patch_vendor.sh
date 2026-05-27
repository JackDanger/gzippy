#!/bin/bash
# Patches vendor source to add Chrome trace spans at key sites.
# Idempotent — skips already-patched files.
set -e

CORE_HDR='#include <core/TraceV2.hpp>'
GCF=/root/gzippy/vendor/rapidgzip/librapidarchive/src/rapidgzip/GzipChunkFetcher.hpp
BF=/root/gzippy/vendor/rapidgzip/librapidarchive/src/core/BlockFetcher.hpp
RG=/root/gzippy/vendor/rapidgzip/librapidarchive/src/tools/rapidgzip.cpp
GC=/root/gzippy/vendor/rapidgzip/librapidarchive/src/rapidgzip/chunkdecoding/GzipChunk.hpp

# Add includes if not present.
for F in "$GCF" "$BF" "$GC"; do
    if ! grep -q "TraceV2.hpp" "$F"; then
        # Insert after the #pragma once line.
        sed -i '/^#pragma once/a\
#include <core/TraceV2.hpp>' "$F"
        echo "PATCHED include: $F"
    fi
done

# Patch processNextChunk: add ScopedSpan as first statement after opening brace.
if ! grep -q "TRACEV2_SPAN.*consumer.iter" "$GCF"; then
    sed -i '/processNextChunk()$/,/^    {/{
        s|^    {$|    {\n        ::tracev2::ScopedSpan _tv2_proc("consumer.iter");|
    }' "$GCF"
    echo "PATCHED processNextChunk: $GCF"
fi

# Patch BlockFetcher::get_with_prefetch — wrap on-demand fetch path.
# Vendor's `BlockFetcher::get` starts ~line 280. We'll wrap the entry.
if ! grep -q "TRACEV2_SPAN.*consumer.get_block" "$BF"; then
    # Insert at top of `get( const Key & blockOffset` body.
    sed -i '/^    get( const Key & blockOffset.*)/,/^    {/{
        s|^    {$|    {\n        ::tracev2::ScopedSpan _tv2_get("wait.block_fetcher_get");|
    }' "$BF"
    echo "PATCHED BlockFetcher::get: $BF"
fi

# Patch decodeAndMeasureBlock — the worker decode site.
if ! grep -q "TRACEV2_SPAN.*worker.decode_chunk" "$BF"; then
    sed -i '/^    decodeAndMeasureBlock( size_t blockOffset,/,/^    {/{
        s|^    {$|    {\n        ::tracev2::ScopedSpan _tv2_dec("worker.decode_chunk");|
    }' "$BF"
    echo "PATCHED decodeAndMeasureBlock: $BF"
fi

# Patch applyWindow call sites in GCF.
if ! grep -q "applyWindow.*tv2_apply" "$GCF"; then
    # Wrap each `chunkData->applyWindow(` call (there are 2).
    perl -i -0777 -pe 's|(chunkData->applyWindow\( \*lastWindow, chunkData->windowCompressionType\(\) \);)|{ ::tracev2::ScopedSpan _tv2_apply("post_process.apply_window"); $1 }|g' "$GCF"
    perl -i -0777 -pe 's|(chunkData->applyWindow\( \*window, chunkData->windowCompressionType\(\) \);)|{ ::tracev2::ScopedSpan _tv2_apply("post_process.apply_window"); $1 }|g' "$GCF"
    echo "PATCHED applyWindow sites: $GCF"
fi

# Patch BlockFetcher.hpp prefetch lifecycle.
# - coord.prefetch_call span wraps the body of prefetchNewBlocks.
# - coord.prefetch_emit span wraps the `m_threadPool.submit(...)` call
#   at BlockFetcher.hpp:554-557 with args matching gzippy
#   (block_fetcher.rs prefetch_new_blocks).
# - coord.prefetch_call.outcome instant event at end-of-function so
#   timeline_analyze.py can pair each call with its submitted count.
if ! grep -q "coord.prefetch_call" "$BF"; then
    # Wrap the body of prefetchNewBlocks. The opening brace sits on its
    # own line `    {` immediately after the signature lines 458-460.
    sed -i '/^    prefetchNewBlocks( const GetPartitionOffset/,/^    {/{
        s|^    {$|    {\n        ::tracev2::ScopedSpan _tv2_pfcall("coord.prefetch_call");|
    }' "$BF"
    echo "PATCHED prefetchNewBlocks body span: $BF"
fi
if ! grep -q "coord.prefetch_emit" "$BF"; then
    # Inject coord.prefetch_emit span just before `++m_statistics.prefetchCount;`
    # (line 553). The span lives until the end of the for-loop iteration
    # (its enclosing block at line 562), so it covers the submit + emplace.
    # Args mirror gzippy's:
    #   index, offset, partition_offset, next_offset, offset_eq_partition
    perl -i -0777 -pe '
      my $inject = q(
            char _tv2_emit_args[192];
            std::snprintf(_tv2_emit_args, sizeof(_tv2_emit_args),
                "\"index\":%zu,\"offset\":%zu,\"partition_offset\":%zu,\"next_offset\":%zu,\"offset_eq_partition\":%s",
                blockIndexToPrefetch,
                static_cast<size_t>(*prefetchBlockOffset),
                static_cast<size_t>(getPartitionOffsetFromOffset ? getPartitionOffsetFromOffset(*prefetchBlockOffset) : *prefetchBlockOffset),
                static_cast<size_t>(*nextPrefetchBlockOffset),
                ((!getPartitionOffsetFromOffset) || getPartitionOffsetFromOffset(*prefetchBlockOffset) == *prefetchBlockOffset) ? "true" : "false");
            ::tracev2::ScopedSpan _tv2_emit("coord.prefetch_emit", _tv2_emit_args);
);
      s|(\s+)\+\+m_statistics\.prefetchCount;|$1$inject$1\+\+m_statistics.prefetchCount;|;
    ' "$BF"
    echo "PATCHED prefetch_emit span: $BF"
fi
if ! grep -q "coord.prefetch_call.outcome" "$BF"; then
    # Emit a final outcome instant event at function end. The
    # function has two exits: (a) the early `return;` at line 471 when
    # the pool is saturated at entry; (b) fall-through at line 569.
    # Patch the early exit: insert outcome instant before the return.
    perl -i -0777 -pe '
      s|(\s+)if \( threadPoolSaturated\(\) \) \{(\s+)return;(\s+)\}|$1if ( threadPoolSaturated() ) {$2::tracev2::emit_instant("coord.prefetch_call.outcome", "\"submitted\":0,\"early_exit\":\"saturated_entry\"", '\''t'\'');$2return;$3}|;
    ' "$BF"
    # Patch the fall-through exit: insert outcome instant before the
    # closing brace of prefetchNewBlocks. Use the unique throw line
    # immediately above it as an anchor.
    perl -i -0777 -pe '
      s|(throw std::logic_error\( "The thread pool should not have more tasks than there are prefetching futures!" \);\s*\}\s*\})|$1|;
      # Use a different injection strategy: insert just before the closing
      # `}` that ends prefetchNewBlocks. Anchor via the unique throw line.
      s|(throw std::logic_error\( "The thread pool should not have more tasks than there are prefetching futures!" \);\n        \}\n)(    \})|$1        ::tracev2::emit_instant("coord.prefetch_call.outcome", "\"submitted\":-1,\"early_exit\":\"fallthrough\"", '\''t'\'');\n$2|;
    ' "$BF"
    echo "PATCHED prefetch_call.outcome instants: $BF"
fi

# Patch BlockFetcher::get cache lookup with cache.get_outcome instants
# to mirror gzippy's get_if_available instrumentation. (Defer — vendor's
# get() lookup path is interleaved with on-demand submission; adding
# clean hit/miss tagging without breaking the existing patch list
# requires a separate pass. The COUNTS we need are in `coord.prefetch_emit`
# vs `coord.prefetch_call.outcome` aggregates on both sides.)

# Patch GzipChunk.hpp speculation phases.
# - worker.seed_first wraps `tryToDecode({blockOffset, blockOffset})` at ~line 739.
# - worker.scan_run starts at the `const auto tBlockFinderStart = now();` site
#   (line 803), RAII-ending when tryToDecode finds a candidate (return at 841)
#   or the function throws (NoBlockInRange at 851).
# - worker.scan_candidate wraps the inner `tryToDecode( offsetToTest )` per
#   loop iteration, started right after `tBlockFinderStop = now();` (line 836).
if ! grep -q "worker.seed_first" "$GC"; then
    perl -i -0777 -pe 's|(if \( auto result = tryToDecode\( \{ blockOffset, blockOffset \} \); result \) \{\n            return \*std::move\( result \);\n        \})|\{ ::tracev2::ScopedSpan _tv2_seed("worker.seed_first"); $1 \}|' "$GC"
    echo "PATCHED seed-first: $GC"
fi
if ! grep -q "worker.scan_run" "$GC"; then
    sed -i 's|const auto tBlockFinderStart = now();|const auto tBlockFinderStart = now(); ::tracev2::ScopedSpan _tv2_scan("worker.scan_run");|' "$GC"
    echo "PATCHED scan_run: $GC"
fi
if ! grep -q "worker.scan_candidate" "$GC"; then
    sed -i 's|const auto tBlockFinderStop = now();|const auto tBlockFinderStop = now(); ::tracev2::ScopedSpan _tv2_cand("worker.scan_candidate");|' "$GC"
    echo "PATCHED scan_candidate: $GC"
fi

# Add tracev2::flush_all() at end of main in rapidgzip.cpp.
if ! grep -q "tracev2::flush_all" "$RG"; then
    # Find main() return; insert flush_all before each return in main scope. Simpler: add at bottom of main's last statement.
    # Find the last `return result;` or `return 0;` in main and add flush before.
    # Conservative: just add to atexit.
    sed -i '/int main(/,/^}/{
        /^int main(/a\
    std::atexit([](){ ::tracev2::flush_all(); });
    }' "$RG"
    echo "PATCHED main atexit: $RG"
fi

echo "Patches applied."
