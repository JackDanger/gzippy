#!/bin/bash
set -e

TP=/root/gzippy/vendor/rapidgzip/librapidarchive/src/core/ThreadPool.hpp
GCF=/root/gzippy/vendor/rapidgzip/librapidarchive/src/rapidgzip/GzipChunkFetcher.hpp

# Patch workerMain — pool.pick around condvar wait + dequeue, pool.run_task around task() call.
if ! grep -q "tv2_pick" "$TP"; then
    perl -i -0777 -pe '
        s|(\s+while \( m_threadPoolRunning \)\s*\{)\n(\s+std::unique_lock<std::mutex> tasksLock\( m_mutex \);)|$1\n            ::tracev2::ScopedSpan _tv2_pick("pool.pick");\n$2|;
        s|(\s+tasksLock\.unlock\(\);)\n(\s+task\(\);)|$1\n                _tv2_pick.~ScopedSpan(); /* end pick before task */\n                ::tracev2::ScopedSpan _tv2_run("pool.run_task");\n$2|;
    ' "$TP"
    if grep -q "tv2_pick" "$TP"; then echo "PATCHED workerMain: $TP"; else echo "WARN: workerMain patch did not match"; fi
fi

# Add a flush_all to workerMain end so worker thread events get out.
if ! grep -q "tv2_worker_flush" "$TP"; then
    perl -i -0777 -pe '
        s|(\s+void\s+workerMain\( size_t threadIndex \)\s*\{\s+if \( const auto pinning =[^}]+\}\s*\n\n\s+while \( m_threadPoolRunning \))|$1|;
        # add flush_all at end of workerMain (before closing brace)
        s|(\s+\}\s*\n\s+\}\s*\n)(\s+void\s+spawnThread\(\))|$1        ::tracev2::flush_all(); /* tv2_worker_flush */\n    }\n\n$2|;
    ' "$TP"
    if grep -q "tv2_worker_flush" "$TP"; then echo "PATCHED workerMain flush: $TP"; else echo "WARN: workerMain flush patch did not match"; fi
fi

# wait.future_recv around std::future::get on chunk futures in GzipChunkFetcher.
# Find chunkData = m_chunkFetcher->get(...) is BaseType::get which we already span as wait.block_fetcher_get,
# so consumer's wait IS captured. Skip duplicate.

echo "Done."
