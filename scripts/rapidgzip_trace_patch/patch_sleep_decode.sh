#!/bin/bash
# FIXED-SLEEP coordination-isolation patch for the rapidgzip vendor.
#
# Mirrors gzippy's GZIPPY_SLEEP_DECODE_NS: when RAPIDGZIP_SLEEP_DECODE_NS=<ns>
# is set, every worker decode SLEEPS <ns> nanoseconds. This equalizes the
# per-chunk "decode" cost to a fixed constant identical to gzippy's sleep so
# the gzippy-vs-rapidgzip wall delta is PURE coordination/scheduling
# structure.
#
# DESIGN NOTE (asymmetry, deliberate): rapidgzip's `decodeBlock` produces a
# fully-valid ChunkData whose downstream consumer (applyWindow, marker
# resolution, window map, write) requires internally-consistent offsets and
# windows. Fabricating that in C++ is fragile, so the sleep is added AFTER
# `decodeBlock` (it still runs, producing a valid chunk). The per-chunk wall
# is therefore (real_decode + sleep). gzippy's sleep_replay returns a zeroed
# chunk with ~0 decode, so its per-chunk wall is (~0 + sleep).
#
# The real-decode term is a per-tool FIXED BASELINE captured at ns=0. The
# DECISIVE readout — does the gzippy-rapidgzip wall DELTA grow with the sleep
# (latency-overlap defect) or stay a fixed offset (per-chunk coordination
# tax) — depends on d(delta)/d(ns), in which both tools' ns=0 baselines
# cancel. The sleep is injected identically (same ns, same per-chunk
# granularity, same number of worker tasks ≈ same chunk count) in both, so
# the SLOPE comparison is clean. Absolute deltas at each ns are reported
# alongside the ns=0 baselines so the decode-baseline offset is explicit.
#
# Idempotent — skips if already patched. Run on the box, then rebuild the
# build-trace binary.
set -e

BF=/root/gzippy/vendor/rapidgzip/librapidarchive/src/core/BlockFetcher.hpp

if grep -q "RAPIDGZIP_SLEEP_DECODE_NS" "$BF"; then
    echo "SLEEP patch already present in $BF"
    exit 0
fi

# Ensure <cstdlib> (getenv) and <cstdint> are available.
if ! grep -q "#include <cstdlib>" "$BF"; then
    sed -i '/^#include <chrono>/a\
#include <cstdlib>' "$BF"
    echo "ADDED include <cstdlib>: $BF"
fi

# Inject the fixed sleep immediately after `decodeBlock` returns inside
# decodeAndMeasureBlock. A function-static reads the env once (atomic across
# threads via the C++11 static-init guarantee). When >0, every worker sleeps
# that many nanoseconds — identical to gzippy's per-chunk std::thread::sleep.
perl -i -0777 -pe '
  my $inject = q{
        {
            static const long long _sleep_ns = []() -> long long {
                const char* e = std::getenv("RAPIDGZIP_SLEEP_DECODE_NS");
                return e ? std::atoll(e) : 0;
            }();
            if ( _sleep_ns > 0 ) {
                std::this_thread::sleep_for( std::chrono::nanoseconds( _sleep_ns ) );
            }
        }};
  s|(\s+auto blockData = decodeBlock\( blockOffset, nextBlockOffset \);)|$1$inject|;
' "$BF"
echo "PATCHED fixed-sleep into decodeAndMeasureBlock: $BF"
echo "Done. Rebuild build-trace: cmake --build /root/gzippy/vendor/rapidgzip/librapidarchive/build-trace --target rapidgzip -j"
