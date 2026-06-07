#!/usr/bin/env bash
#
# rss_vs_t.sh — D2 GUARD for the gzippy-native cache mandate.
#
# Measures process RSS as thread-count T rises. RSS is a COARSE T-scaling guard
# (the PRIMARY mandate instrument is the in-process byte accounting behind
# GZIPPY_MEM_STATS=1 — see src/decompress/inflate/mem_stats.rs). RSS only catches
# UNEXPECTED large T-scaling (e.g. a per-thread copy of a shared table).
#
# For each T in TLIST, N interleaved trials:
#   * PEAK RSS via /usr/bin/time -l "maximum resident set size" (BYTES on macOS).
#   * PLATEAU RSS: a concurrent `ps -o rss=` poller @ ~50ms; median of the
#     plateau (first/last 20% dropped). ps RSS is KiB -> converted to BYTES.
#     NEVER cross-compared with peak as equal.
#   * decoded sha256 must == EXPECT_SHA (a wrong-sha trial is VOID).
#   * GZIPPY_DEBUG stderr must show path=ParallelSM (native engine, not a fallback).
# Output streams to /dev/null so output buffering is T-invariant.
#
# Env (positive control) GZIPPY_MEM_BALLAST_MIB=N is inherited by the child
# gzippy processes automatically — set it before invoking this script and the
# regression slope should recover ~N MiB/thread.
#
# Args (positional or env): BIN GZ TLIST N
#   BIN   default ./target/release/gzippy
#   GZ    default /tmp/silesia.gz
#   TLIST default "1 2 4 8 16"
#   N     default 7
#   EXPECT_SHA env, default the canonical silesia sha.
#
# Usage:
#   scripts/bench/rss_vs_t.sh
#   GZIPPY_MEM_BALLAST_MIB=8 scripts/bench/rss_vs_t.sh   # positive control
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-${1:-./target/release/gzippy}}"
GZ="${GZ:-${2:-/tmp/silesia.gz}}"
TLIST="${TLIST:-${3:-1 2 4 8 16}}"
N="${N:-${4:-7}}"
EXPECT_SHA="${EXPECT_SHA:-028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f}"
OUT="${OUT:-/tmp/rss_vs_t.dat}"
TIME_BIN="/usr/bin/time"

if [ ! -x "$BIN" ]; then echo "BIN not executable: $BIN" >&2; exit 2; fi
if [ ! -f "$GZ" ]; then echo "GZ not found: $GZ" >&2; exit 2; fi

echo "BIN=$BIN GZ=$GZ TLIST='$TLIST' N=$N BALLAST=${GZIPPY_MEM_BALLAST_MIB:-0}MiB POOL_CAP=${GZIPPY_STAGING_POOL_CAP:-4}"
: > "$OUT"   # columns: T peak_bytes plateau_bytes sha_ok path_ok

# Sample the RSS plateau of one decode (separate run from the peak run).
sample_plateau() {
    local T="$1"
    local samp; samp="$(mktemp)"
    GZIPPY_DEBUG=1 "$BIN" -d -c -p "$T" "$GZ" >/dev/null 2>/dev/null &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        ps -o rss= -p "$pid" 2>/dev/null >> "$samp"
        sleep 0.05
    done
    wait "$pid" 2>/dev/null
    python3 "$HERE/_plateau_median.py" "$samp"
    rm -f "$samp"
}

for trial in $(seq 1 "$N"); do
    for T in $TLIST; do
        terr="$(mktemp)"
        # PEAK + sha + path assert run.
        sha="$(GZIPPY_DEBUG=1 "$TIME_BIN" -l "$BIN" -d -c -p "$T" "$GZ" 2>"$terr" | shasum -a 256 | awk '{print $1}')"
        peak="$(grep -i 'maximum resident set size' "$terr" | awk '{print $1}')"
        [ -z "$peak" ] && peak=0
        pathok="$(grep -c 'path=ParallelSM' "$terr")"
        shaok=0; [ "$sha" = "$EXPECT_SHA" ] && shaok=1
        rm -f "$terr"
        # PLATEAU run.
        plateau="$(sample_plateau "$T")"
        [ -z "$plateau" ] && plateau=0
        echo "$T $peak $plateau $shaok $pathok" >> "$OUT"
        printf 'trial %d T=%-2d peak=%s plateau=%s sha_ok=%d path_ok=%s\n' \
            "$trial" "$T" "$peak" "$plateau" "$shaok" "$pathok"
    done
done

python3 "$HERE/_rss_regression.py" "$OUT"
