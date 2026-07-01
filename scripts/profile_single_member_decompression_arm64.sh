#!/usr/bin/env bash
# profile_single_member_decompression_arm64.sh
#
# Local arm64 (Apple Silicon) single-member decompression gate + samply profile.
# Run via `make profile-single-member-decompression-arm64` on this Mac.
#
# Exercises production single-member routing on arm64:
#   classify → LibdeflateSingle; decode via libdeflate one-shot (no parallel SM)
#
# Phases (local):
#   1. release build with debuginfo
#   2. gzip(1) -9 single-member fixture
#   3. decode sweep T={1,2,4,8} → md5 vs raw source
#   4. on PASS: samply record (if installed)
#   5. stdout guide with artifact paths under target/tooling/
#
# Config (env):
#   GZIPPY_FIXTURE_MB         raw MiB (default: 64)
#   GZIPPY_THREAD_SWEEP       default: "1 2 4 8"
#   GZIPPY_PROFILE_TRIALS     default: 10
#   GZIPPY_PROFILE_T          samply T (default: 1 — primary arm64 hot path)
#   GZIPPY_PROFILE_ITERATIONS samply loop count (default: 10)
#   GZIPPY_PROFILE_ALWAYS=1   profile even on correctness FAIL
#   GZIPPY_SKIP_PROFILE=1     correctness only
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

FIXTURE_MB="${GZIPPY_FIXTURE_MB:-64}"
THREAD_SWEEP="${GZIPPY_THREAD_SWEEP:-1 2 4 8}"
PROFILE_TRIALS="${GZIPPY_PROFILE_TRIALS:-10}"
PROFILE_T="${GZIPPY_PROFILE_T:-1}"
PROFILE_ITERATIONS="${GZIPPY_PROFILE_ITERATIONS:-10}"
PROFILE_ALWAYS="${GZIPPY_PROFILE_ALWAYS:-0}"
SKIP_PROFILE="${GZIPPY_SKIP_PROFILE:-0}"

GZIPPY_BIN="$REPO_ROOT/target/release/gzippy"
BYTES=$((FIXTURE_MB * 1024 * 1024))

md5_file() {
    if command -v md5 >/dev/null 2>&1; then
        md5 -q "$1"
    else
        md5sum "$1" | awk '{print $1}'
    fi
}

file_size() {
    if stat -f%z "$1" >/dev/null 2>&1; then
        stat -f%z "$1"
    else
        stat -c%s "$1"
    fi
}

prepare_raw_fixture() {
    local dest="$1"
    if [ -f benchmark_data/silesia-large.bin ]; then
        head -c "$BYTES" benchmark_data/silesia-large.bin >"$dest"
        echo "source: benchmark_data/silesia-large.bin (first ${FIXTURE_MB} MiB)"
        return 0
    fi
    if [ -f benchmark_data/silesia.tar ]; then
        head -c "$BYTES" benchmark_data/silesia.tar >"$dest"
        echo "source: benchmark_data/silesia.tar (first ${FIXTURE_MB} MiB)"
        echo "WARN: not silesia-large.bin — md5 will differ from homelab x86 reference"
        return 0
    fi
    local seed=test_data/text-1MB.txt
    if [ ! -f "$seed" ]; then
        echo "MISSING fixture: need benchmark_data/silesia-large.bin or $seed"
        echo "  Run: ./scripts/prepare_benchmark_data.sh silesia-large"
        exit 1
    fi
    : >"$dest"
    while [ "$(file_size "$dest")" -lt "$BYTES" ]; do
        cat "$seed" >>"$dest"
    done
    head -c "$BYTES" "$dest" >"${dest}.tmp" && mv "${dest}.tmp" "$dest"
    echo "source: repeated $seed (${FIXTURE_MB} MiB synthetic)"
    echo "WARN: md5 differs from homelab silesia-head reference"
}

print_local_profile_guide() {
    local profile_dir="$1"
    local skipped_reason="$2"

    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo " PROFILING — single-member arm64 (local artifacts)"
    echo "══════════════════════════════════════════════════════════════════════"

    if [ -z "$profile_dir" ]; then
        echo ""
        echo " Profiling was not captured: ${skipped_reason:-unknown}"
        echo " Fix correctness, then: make profile-single-member-decompression-arm64"
        echo " Or: GZIPPY_PROFILE_ALWAYS=1 make profile-single-member-decompression-arm64"
        echo "══════════════════════════════════════════════════════════════════════"
        return
    fi

    echo ""
    echo " Local directory:"
    echo "   ${profile_dir}"
    echo ""
    echo " CPU — open in Firefox Profiler (samply):"
    echo "   samply load ${profile_dir}/gzippy.profile.json"
    echo "   # or drag gzippy.profile.json into https://profiler.firefox.com"
    echo ""
    if [ -f scripts/profile_diff.py ]; then
        echo " Band diff (needs rapidgzip arm64 build + second samply capture):"
        echo "   scripts/profile_capture.sh --help"
    fi
    echo " Manifest: cat ${profile_dir}/README.txt"
    echo ""
    echo " Re-profile: GZIPPY_PROFILE_T=4 make profile-single-member-decompression-arm64"
    echo "══════════════════════════════════════════════════════════════════════"
}

if [ "$(uname -m)" != "arm64" ]; then
    echo "ERROR: this target is for arm64 macOS (got $(uname -m))"
    exit 1
fi

SHORT_HEAD=$(git rev-parse --short HEAD)

echo "=== profile-single-member-decompression-arm64 ==="
echo "machine:   $(uname -m) $(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
echo "commit:    $SHORT_HEAD"
echo "path:      single-member LibdeflateSingle (libdeflate one-shot)"
echo "fixture:   ${FIXTURE_MB} MiB raw → gzip -9"
echo "sweep:     T={${THREAD_SWEEP}} (${PROFILE_TRIALS} trials/T)"
echo "profiling: T=${PROFILE_T} (${PROFILE_ITERATIONS} iterations) unless GZIPPY_SKIP_PROFILE=1"
echo

echo "--- build (debuginfo for samply stacks) ---"
BUILD_LOG=/tmp/prof-arm64-build.log
RUSTFLAGS='-C debuginfo=1 -C strip=none -C force-frame-pointers=yes' \
    cargo build --release >"$BUILD_LOG" 2>&1 || {
    echo "BUILD FAILED"
    grep -E '^error' "$BUILD_LOG" || tail -20 "$BUILD_LOG"
    exit 1
}
grep -E 'Compiling gzippy |Finished' "$BUILD_LOG" || true
test -x "$GZIPPY_BIN" || { echo "BUILD FAILED: no binary"; exit 1; }
echo

echo "--- fixture (gzip -9 single-member) ---"
SRC=$(mktemp /tmp/prof-arm64-src.XXXXXX.bin)
FIX=$(mktemp /tmp/prof-arm64-fix.XXXXXX.gz)
trap 'rm -f "$SRC" "$FIX"' EXIT
prepare_raw_fixture "$SRC"
gzip -9 -c "$SRC" >"$FIX"
REF=$(md5_file "$SRC")
echo "raw $(file_size "$SRC") bytes  →  gz $(file_size "$FIX") bytes"
echo "reference md5: $REF"
echo

echo "--- routing (production CLI: decompress_single_member → libdeflate on arm64) ---"
GZIPPY_DEBUG=1 "$GZIPPY_BIN" -d -c -p "$PROFILE_T" "$FIX" >/dev/null 2>/tmp/prof-arm64-route.err || true
grep -E '\[gzippy\]|\[parallel_sm\]' /tmp/prof-arm64-route.err | head -5 || true
if grep -q '\[parallel_sm\]' /tmp/prof-arm64-route.err 2>/dev/null; then
    echo "WARN: parallel_sm on arm64 — unexpected (ISA-L parallel SM is x86-only)"
fi
echo

echo "--- decode sweep (${PROFILE_TRIALS} trials per T) ---"
printf '%-5s %-8s %-34s %s\n' "T" "trial" "md5" "result"
FAIL=0
ERR_SEEN=""
for T in $THREAD_SWEEP; do
    T_FAIL=0
    for trial in $(seq 1 "$PROFILE_TRIALS"); do
        OUT=$(mktemp /tmp/prof-arm64-out.XXXXXX)
        rc=0
        "$GZIPPY_BIN" -d -c -p "$T" "$FIX" >"$OUT" 2>/tmp/prof-arm64-err-"$T"-"$trial" || rc=$?
        if [ "$rc" -ne 0 ]; then
            printf '%-5s %-8s %-34s %s\n' "$T" "$trial" "(decode error rc=$rc)" "ERROR"
            T_FAIL=1
            ERR_SEEN="$ERR_SEEN $T"
            rm -f "$OUT"
            continue
        fi
        H=$(md5_file "$OUT")
        rm -f "$OUT"
        if [ "$H" = "$REF" ]; then
            printf '%-5s %-8s %-34s %s\n' "$T" "$trial" "$H" "ok"
        else
            printf '%-5s %-8s %-34s %s\n' "$T" "$trial" "$H" "WRONG"
            T_FAIL=1
            ERR_SEEN="$ERR_SEEN $T"
        fi
    done
    if [ "$T_FAIL" -ne 0 ]; then
        FAIL=1
        printf '%-5s %-8s %-34s %s\n' "$T" "—" "—" "FAIL (${PROFILE_TRIALS} trials)"
    else
        printf '%-5s %-8s %-34s %s\n' "$T" "—" "—" "PASS (${PROFILE_TRIALS}/${PROFILE_TRIALS})"
    fi
done
echo

if [ -n "$ERR_SEEN" ]; then
    FIRST_ERR=$(echo "$ERR_SEEN" | awk '{print $1}')
    echo "--- stderr (T=$FIRST_ERR, trial=1) ---"
    cat /tmp/prof-arm64-err-"$FIRST_ERR"-1 || true
    echo
fi

PROFILE_DIR=""
PROFILE_SKIP_REASON=""
RUN_PROFILE=0
if [ "$SKIP_PROFILE" = "1" ]; then
    PROFILE_SKIP_REASON="GZIPPY_SKIP_PROFILE=1"
elif [ "$FAIL" -eq 0 ]; then
    RUN_PROFILE=1
elif [ "$PROFILE_ALWAYS" = "1" ]; then
    RUN_PROFILE=1
else
    PROFILE_SKIP_REASON="correctness FAIL (set GZIPPY_PROFILE_ALWAYS=1 to profile anyway)"
fi

if [ "$RUN_PROFILE" -eq 1 ]; then
    if ! command -v samply >/dev/null 2>&1; then
        PROFILE_SKIP_REASON="samply not installed (brew install samply)"
        RUN_PROFILE=0
    fi
fi

if [ "$RUN_PROFILE" -eq 1 ]; then
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    PROFILE_DIR="$REPO_ROOT/target/tooling/profile-single-member-decompression-arm64/${SHORT_HEAD}-${STAMP}"
    mkdir -p "$PROFILE_DIR"
    cp "$FIX" "$PROFILE_DIR/fixture.gz"

    echo "--- samply capture (T=$PROFILE_T, $PROFILE_ITERATIONS decodes) ---"
    echo "output: $PROFILE_DIR"
    samply record --save-only -o "$PROFILE_DIR/gzippy.profile.json" -- \
        bash -c "for i in \$(seq $PROFILE_ITERATIONS); do \
            \"$GZIPPY_BIN\" -d -c -p $PROFILE_T \"$FIX\" >/dev/null; done" \
        2>"$PROFILE_DIR/samply.stderr" || true

    cat >"$PROFILE_DIR/README.txt" <<EOF
profile-single-member-decompression-arm64 capture
commit: $SHORT_HEAD
threads: $PROFILE_T
fixture: copied as fixture.gz (gzip -9 single-member)
binary: $GZIPPY_BIN (release + debuginfo)

gzippy.profile.json   samply profile — samply load or Firefox Profiler
samply.stderr         sampler stderr
EOF
    echo "PROFILE_DIR=$PROFILE_DIR"
elif [ -n "$PROFILE_SKIP_REASON" ]; then
    echo "PROFILE_SKIP_REASON=$PROFILE_SKIP_REASON"
fi

if [ "$FAIL" -eq 0 ]; then
    echo ""
    echo "RESULT: PASS — every thread count decodes byte-perfect"
else
    echo ""
    echo "RESULT: FAIL — see ERROR / WRONG rows above"
fi

print_local_profile_guide "$PROFILE_DIR" "$PROFILE_SKIP_REASON"

exit "$FAIL"
