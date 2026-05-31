#!/bin/bash
set -e

TP=/root/gzippy/vendor/rapidgzip/librapidarchive/src/core/ThreadPool.hpp
GCF=/root/gzippy/vendor/rapidgzip/librapidarchive/src/rapidgzip/GzipChunkFetcher.hpp
BF=/root/gzippy/vendor/rapidgzip/librapidarchive/src/core/BlockFetcher.hpp
PGR=/root/gzippy/vendor/rapidgzip/librapidarchive/src/rapidgzip/ParallelGzipReader.hpp

# Instrument the REAL in-order consumer self-work in ParallelGzipReader::read.
# WHY: without this, rapidgzip's consumer-side output copy + CRC are invisible,
# so a cross-tool report shows rapidgzip "consumer-write = 0ms wall-critical"
# purely because gzippy instruments its write and rapidgzip did not — an
# INSTRUMENTATION ARTIFACT that would falsely crown consumer-write the lever.
# The consumer loop body (ParallelGzipReader.hpp ~613-625) does:
#   processCRC32(...)            -> the consumer's CRC self-work
#   writeFunctor(...)            -> the consumer's output copy
# Wrap each so they are comparable to gzippy's consumer.combine_crc /
# consumer.write_data.
if ! grep -q "TraceV2.hpp" "$PGR"; then
    sed -i '/^#pragma once/a\
#include <core/TraceV2.hpp>' "$PGR"
    echo "PATCHED include: $PGR"
fi
if ! grep -q 'consumer.combine_crc' "$PGR"; then
    # Wrap the processCRC32 call in its OWN brace scope so the span closes
    # before the write span opens (no overlap, clean partition).
    perl -i -0777 -pe '
        s|(processCRC32\( chunkData, offsetInBlock, nBytesToDecode \);)|\{ ::tracev2::ScopedSpan _tv2_crc("consumer.combine_crc"); $1 \}|;
    ' "$PGR"
    # Wrap the writeFunctor call in its own brace scope.
    perl -i -0777 -pe '
        s|(writeFunctor\( chunkData, offsetInBlock, nBytesToDecode \);)|\{ ::tracev2::ScopedSpan _tv2_wr("consumer.write_data"); $1 \}|;
    ' "$PGR"
    if grep -q "consumer.write_data" "$PGR"; then echo "PATCHED consumer write/crc: $PGR"; else echo "WARN: consumer write/crc patch did not match"; fi
fi

# ThreadPool.hpp references ::tracev2 below; ensure the include is present
# (patch_vendor.sh adds it to GCF/BF/GC but NOT to ThreadPool.hpp).
if ! grep -q "TraceV2.hpp" "$TP"; then
    sed -i '/^#pragma once/a\
#include <core/TraceV2.hpp>' "$TP"
    echo "PATCHED include: $TP"
fi

# Patch workerMain — pool.pick around condvar wait + dequeue, pool.run_task around task() call.
if ! grep -q "tv2_pick" "$TP"; then
    perl -i -0777 -pe '
        s|(\s+while \( m_threadPoolRunning \)\s*\{)\n(\s+std::unique_lock<std::mutex> tasksLock\( m_mutex \);)|$1\n            ::tracev2::ScopedSpan _tv2_pick("pool.pick");\n$2|;
        s|(\s+tasksLock\.unlock\(\);)\n(\s+task\(\);)|$1\n                _tv2_pick.~ScopedSpan(); /* end pick before task */\n                ::tracev2::ScopedSpan _tv2_run("pool.run_task");\n$2|;
    ' "$TP"
    if grep -q "tv2_pick" "$TP"; then echo "PATCHED workerMain: $TP"; else echo "WARN: workerMain patch did not match"; fi
fi

# Add a flush_all INSIDE workerMain, right before its closing brace, so worker
# thread events get out. workerMain ends as:
#         }            <- closes the `while ( m_threadPoolRunning )` loop
#     }                <- closes workerMain
#
#     void
#     spawnThread()
# Insert the flush between the while-loop close and the function close so it is
# INSIDE workerMain (a prior version inserted it OUTSIDE, breaking the class).
if ! grep -q "tv2_worker_flush" "$TP"; then
    perl -i -0777 -pe '
        s|(        \}\n)(    \}\n\n    void\n    spawnThread\(\))|$1        ::tracev2::flush_all(); /* tv2_worker_flush */\n$2|;
    ' "$TP"
    if grep -q "tv2_worker_flush" "$TP"; then echo "PATCHED workerMain flush: $TP"; else echo "WARN: workerMain flush patch did not match"; fi
fi

# wait.future_recv — the IN-ORDER CONSUMER STALL, the analog of gzippy's
# ttp.rx_recv_block / future.recv.
#
# WHY the old `wait.block_fetcher_get` was NOT the consumer wait: it wraps the
# WHOLE BlockFetcher::get() body, which on the common path returns a CACHED
# result with zero blocking (lines ~302-309), and also runs prefetchNewBlocks
# dispatch work. So it conflates cache-hit fast-returns + dispatch with the
# real future-block, and critpath saw rapidgzip consumer-wait ≈ 0 — making the
# cross-tool wall-critical comparison useless.
#
# The TRUE in-order consumer stall in BlockFetcher::get is the future-wait
# region: the `while ( queuedResult.wait_for(1ms) == timeout ) {...}` loop plus
# `queuedResult.get()` (vendor BlockFetcher.hpp, anchored on tFutureGetStart).
# That is where the consumer thread sits blocked until the next in-order chunk's
# future resolves. Wrap exactly that with a ScopedSpan named `wait.future_recv`
# (fulcrum's Span::is_wait() recognizes `wait.*`), ending it right after the
# future is taken (explicit dtor, like the pool.pick patch) so the span covers
# ONLY the block, not insertIntoCache.
# Wrap the wait region in its OWN brace scope so the ScopedSpan destructs
# EXACTLY ONCE at block-end (no explicit-dtor double-emit): declare `result`
# before the block, open `{` + the span right after tFutureGetStart, and close
# `}` immediately after `queuedResult.get()` assigns into result. insertIntoCache
# and the return happen after the block, outside the wait span.
if ! grep -q "wait.future_recv" "$BF"; then
    perl -i -0777 -pe '
        # Open the scope + the span at tFutureGetStart.
        s|(\[\[maybe_unused\]\] const auto tFutureGetStart = now\(\);)|std::shared_ptr<BlockData> result;\n        $1\n        {\n        ::tracev2::ScopedSpan _tv2_futrecv("wait.future_recv");|;
        # The original line declared `auto result = ...`; turn it into an
        # assignment to the pre-declared result, then close the brace scope.
        s|auto result = std::make_shared<BlockData>\( queuedResult\.get\(\) \);|result = std::make_shared<BlockData>( queuedResult.get() );\n        } /* end wait.future_recv (consumer in-order stall) */|;
    ' "$BF"
    # The header must include TraceV2.hpp.
    if ! grep -q "TraceV2.hpp" "$BF"; then
        sed -i '/^#pragma once/a\
#include <core/TraceV2.hpp>' "$BF"
    fi
    if grep -q "wait.future_recv" "$BF"; then echo "PATCHED wait.future_recv (consumer in-order stall): $BF"; else echo "WARN: wait.future_recv patch did not match"; fi
fi

echo "Done."
