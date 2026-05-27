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
