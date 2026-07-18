#!/usr/bin/env bash
# profile_single_member_decompression_x86_64.sh
#
# Homelab x86_64 single-member decompression gate: correctness sweep + CPU/alloc
# profiling. Run via `make profile-single-member-decompression-x86_64`.
#
# Exercises production single-member routing on x86_64:
#   T=1  → ISA-L sequential (IsalSingle)
#   T≥2, compressed >10 MiB → parallel single-member (ISA-L inflatePrime)
#
# Phases (remote, one SSH session):
#   1. sync + release build with debuginfo (perf/heaptrack need symbols)
#   2. gzip(1) -9 fixture from silesia head (reproduces production layout)
#   3. decode sweep across thread counts → md5 vs raw source
#   4. on PASS (or GZIPPY_PROFILE_ALWAYS=1): perf stat + perf record +
#      folded stacks + flamegraph SVG; optional heaptrack for allocations
#   5. stdout ends with RESULT + local profiling guide (paths, scp, next steps)
#
# Config (override via env):
#   GZIPPY_REMOTE_SSH         ssh target (default: root@$YOUR_BENCH_HOST)
#   GZIPPY_REMOTE_DIR         repo on remote (default: gzippy)
#   GZIPPY_FIXTURE_MB         raw MiB (default: 64)
#   GZIPPY_THREAD_SWEEP       thread counts (default: "1 2 4 8 9 12 16")
#   GZIPPY_PROFILE_TRIALS     trials per T (default: 10)
#   GZIPPY_PROFILE_T          T for perf/heaptrack capture (default: 8)
#   GZIPPY_PROFILE_ITERATIONS perf record loop count (default: 5)
#   GZIPPY_PROFILE_ALWAYS=1   capture profiles even if correctness FAIL
#   GZIPPY_SKIP_PROFILE=1     correctness only, no perf capture
set -euo pipefail

REMOTE_SSH="${GZIPPY_REMOTE_SSH:-root@$YOUR_BENCH_HOST}"
REMOTE_DIR="${GZIPPY_REMOTE_DIR:-gzippy}"
FIXTURE_MB="${GZIPPY_FIXTURE_MB:-64}"
THREAD_SWEEP="${GZIPPY_THREAD_SWEEP:-1 2 4 8 9 12 16}"
PROFILE_TRIALS="${GZIPPY_PROFILE_TRIALS:-10}"
PROFILE_T="${GZIPPY_PROFILE_T:-8}"
PROFILE_ITERATIONS="${GZIPPY_PROFILE_ITERATIONS:-5}"
PROFILE_ALWAYS="${GZIPPY_PROFILE_ALWAYS:-0}"
SKIP_PROFILE="${GZIPPY_SKIP_PROFILE:-0}"

BRANCH=$(git rev-parse --abbrev-ref HEAD)

print_local_profile_guide() {
    local profile_dir="$1"
    local skipped_reason="$2"
    local local_dest="./target/tooling/profile-single-member-decompression-latest"

    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo " PROFILING — single-member x86_64 (homelab artifacts)"
    echo "══════════════════════════════════════════════════════════════════════"

    if [ -z "$profile_dir" ]; then
        echo ""
        echo " Profiling was not captured: ${skipped_reason:-unknown}"
        echo " Fix correctness first, then re-run:"
        echo "   make profile-single-member-decompression-x86_64"
        echo " Or: GZIPPY_PROFILE_ALWAYS=1 make profile-single-member-decompression-x86_64"
        echo "══════════════════════════════════════════════════════════════════════"
        return
    fi

    echo ""
    echo " Remote directory (all artifacts on homelab):"
    echo "   ${profile_dir}"
    echo ""
    echo " Copy to this machine:"
    echo "   mkdir -p ./target/tooling"
    echo "   scp -r '${REMOTE_SSH}:${profile_dir}/' '${local_dest}/'"
    echo ""
    echo " CPU — flamegraph + perf report + band diff vs rapidgzip:"
    echo "   open ${local_dest}/gzippy.flamegraph.svg"
    echo "   perf report -i ${profile_dir}/gzippy.perf.data"
    echo "   python3 scripts/profile_diff.py \\"
    echo "     --gzippy-folded ${local_dest}/gzippy.folded \\"
    echo "     --rapidgzip-folded <rapidgzip.folded> \\"
    echo "     --out ${local_dest}/diff.json --out-md ${local_dest}/diff.md"
    echo ""
    echo " Allocations: heaptrack_gui ${local_dest}/gzippy.heaptrack*.gz"
    echo " Counters:    less ${local_dest}/gzippy.perfstat.txt"
    echo " Manifest:    cat ${profile_dir}/README.txt"
    echo ""
    echo " Re-profile another T:"
    echo "   GZIPPY_PROFILE_T=16 GZIPPY_PROFILE_ALWAYS=1 make profile-single-member-decompression-x86_64"
    echo "══════════════════════════════════════════════════════════════════════"
}

echo "=== profile-single-member-decompression-x86_64 ==="
echo "branch:    $BRANCH"
echo "remote:    ssh ${REMOTE_SSH}"
echo "path:      single-member (T=1 ISA-L; T≥2 parallel SM when >10 MiB gz)"
echo "fixture:   ${FIXTURE_MB} MiB raw → gzip -9 → decode T={${THREAD_SWEEP}} (${PROFILE_TRIALS} trials/T)"
echo "profiling: T=${PROFILE_T} (${PROFILE_ITERATIONS} perf iterations) unless GZIPPY_SKIP_PROFILE=1"
echo

echo "--- push $BRANCH so the remote builds exactly local HEAD ---"
if git push origin "$BRANCH" 2>&1 | tail -1; then
    :
else
    echo "WARN: push failed; remote will build whatever origin/$BRANCH already is"
fi
echo

REMOTE_LOG=$(mktemp)
trap 'rm -f "$REMOTE_LOG"' EXIT

set +e
ssh ${REMOTE_SSH} \
    "BRANCH='${BRANCH}' REMOTE_DIR='${REMOTE_DIR}' FIXTURE_MB='${FIXTURE_MB}' THREAD_SWEEP='${THREAD_SWEEP}' PROFILE_TRIALS='${PROFILE_TRIALS}' PROFILE_T='${PROFILE_T}' PROFILE_ITERATIONS='${PROFILE_ITERATIONS}' PROFILE_ALWAYS='${PROFILE_ALWAYS}' SKIP_PROFILE='${SKIP_PROFILE}' bash -s" \
    <<'REMOTE' | tee "$REMOTE_LOG"
set -euo pipefail
cd "$REMOTE_DIR"

GZIPPY_BIN="$PWD/target/release/gzippy"
SHORT_HEAD=$(git rev-parse --short HEAD)

echo "--- sync + build (debuginfo for profiling) ---"
git fetch origin "$BRANCH" --quiet
git reset --hard "origin/$BRANCH" --quiet
echo "HEAD: $SHORT_HEAD  $(git log -1 --format=%s)"
BUILD_LOG=/tmp/profx86-build.log
RUSTFLAGS='-C debuginfo=1 -C strip=none -C force-frame-pointers=yes' \
    cargo build --release --features isal-compression >"$BUILD_LOG" 2>&1 || {
    echo "BUILD FAILED"
    grep -E '^error' "$BUILD_LOG" || tail -20 "$BUILD_LOG"
    exit 1
}
grep -E 'Compiling gzippy |Finished' "$BUILD_LOG" || true
test -x "$GZIPPY_BIN" || { echo "BUILD FAILED: no binary"; exit 1; }
echo

echo "--- fixture (gzip(1) CLI output, single-member) ---"
SRC=/tmp/profx86-src.bin
FIX=/tmp/profx86-fixture.gz
if [ ! -f benchmark_data/silesia-large.bin ]; then
    echo "MISSING benchmark_data/silesia-large.bin — run: ./scripts/prepare_benchmark_data.sh silesia-large"
    exit 1
fi
head -c "$(( FIXTURE_MB * 1024 * 1024 ))" benchmark_data/silesia-large.bin > "$SRC"
gzip -9 -c "$SRC" > "$FIX"
REF=$(md5sum < "$SRC" | cut -d' ' -f1)
echo "raw $(stat -c%s "$SRC") bytes  →  gz $(stat -c%s "$FIX") bytes"
echo "reference md5: $REF"
echo

echo "--- routing (GZIPPY_DEBUG=1, T=$PROFILE_T, file CLI → decompress_single_member) ---"
GZIPPY_DEBUG=1 "$GZIPPY_BIN" -d -c -p "$PROFILE_T" "$FIX" >/dev/null 2>/tmp/profx86-route.err || true
grep -E '\[gzippy\]|\[parallel_sm\]' /tmp/profx86-route.err | head -8 || true
if [ "$PROFILE_T" -gt 1 ] && ! grep -q '\[parallel_sm\]' /tmp/profx86-route.err 2>/dev/null; then
    echo "WARN: T=$PROFILE_T but no [parallel_sm] — compressed size may be below 10 MiB gate"
fi
echo

echo "--- decode sweep (${PROFILE_TRIALS} trials per T) ---"
printf '%-5s %-8s %-34s %s\n' "T" "trial" "md5" "result"
FAIL=0
ERR_SEEN=""
for T in $THREAD_SWEEP; do
    T_FAIL=0
    for trial in $(seq 1 "$PROFILE_TRIALS"); do
        rc=0
        "$GZIPPY_BIN" -d -c -p "$T" "$FIX" \
            > /tmp/profx86-out 2>/tmp/profx86-err-"$T"-"$trial" || rc=$?
        if [ "$rc" -ne 0 ]; then
            printf '%-5s %-8s %-34s %s\n' "$T" "$trial" "(decode error rc=$rc)" "ERROR"
            T_FAIL=1
            ERR_SEEN="$ERR_SEEN $T"
            continue
        fi
        H=$(md5sum < /tmp/profx86-out | cut -d' ' -f1)
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
    echo "--- stderr from first ERROR trial (T=$FIRST_ERR, trial=1) ---"
    cat /tmp/profx86-err-"$FIRST_ERR"-1 || true
    echo
    echo "--- diagnostic decode (GZIPPY_DEBUG=1, T=$FIRST_ERR) ---"
    GZIPPY_DEBUG=1 "$GZIPPY_BIN" -d -c -p "$FIRST_ERR" "$FIX" \
        > /dev/null 2>/tmp/profx86-diag.err || true
    cat /tmp/profx86-diag.err || true
    echo
fi

PROFILE_DIR=""
PROFILE_SKIP_REASON=""
HEAPTRACK_MISSING=0
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
    if ! command -v perf >/dev/null 2>&1; then
        PROFILE_SKIP_REASON="perf not installed on remote (apt install linux-perf)"
        RUN_PROFILE=0
    fi
fi

if [ "$RUN_PROFILE" -eq 1 ]; then
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    PROFILE_DIR="$PWD/target/tooling/profile-single-member-decompression-x86_64/${SHORT_HEAD}-${STAMP}"
    mkdir -p "$PROFILE_DIR"

    echo "--- profile capture (T=$PROFILE_T, $PROFILE_ITERATIONS decode iterations) ---"
    echo "output directory: $PROFILE_DIR"
    echo

    PERF_EVENTS="cycles:u,instructions:u,cache-references:u,cache-misses:u,page-faults,major-faults,branches:u,branch-misses:u"

    echo "[perf stat] one decode"
    perf stat -e "$PERF_EVENTS" -- "$GZIPPY_BIN" -d -c -p "$PROFILE_T" "$FIX" \
        > /dev/null 2>"$PROFILE_DIR/gzippy.perfstat.txt" || true

    echo "[perf record] $PROFILE_ITERATIONS decodes (call-graph=dwarf, needs debuginfo build)"
    perf record -F 999 --call-graph=dwarf -o "$PROFILE_DIR/gzippy.perf.data" -- \
        bash -c "for i in \$(seq $PROFILE_ITERATIONS); do \"$GZIPPY_BIN\" -d -c -p $PROFILE_T \"$FIX\" >/dev/null; done" \
        2>"$PROFILE_DIR/perf.record.stderr" || true

    FLAMEGRAPH_TOOL=""
    if command -v inferno-collapse-perf >/dev/null 2>&1 && command -v inferno-flamegraph >/dev/null 2>&1; then
        FLAMEGRAPH_TOOL=inferno
    elif command -v stackcollapse-perf.pl >/dev/null 2>&1 && command -v flamegraph.pl >/dev/null 2>&1; then
        FLAMEGRAPH_TOOL=perl
    fi

    if [ -f "$PROFILE_DIR/gzippy.perf.data" ]; then
        if [ "$FLAMEGRAPH_TOOL" = inferno ]; then
            perf script -i "$PROFILE_DIR/gzippy.perf.data" 2>>"$PROFILE_DIR/perf.script.stderr" \
                | inferno-collapse-perf > "$PROFILE_DIR/gzippy.folded"
            inferno-flamegraph --title "gzippy T=$PROFILE_T single-member" \
                < "$PROFILE_DIR/gzippy.folded" > "$PROFILE_DIR/gzippy.flamegraph.svg"
            echo "wrote $PROFILE_DIR/gzippy.flamegraph.svg (inferno)"
        elif [ "$FLAMEGRAPH_TOOL" = perl ]; then
            perf script -i "$PROFILE_DIR/gzippy.perf.data" 2>>"$PROFILE_DIR/perf.script.stderr" \
                | stackcollapse-perf.pl > "$PROFILE_DIR/gzippy.folded"
            flamegraph.pl --title "gzippy T=$PROFILE_T single-member" \
                "$PROFILE_DIR/gzippy.folded" > "$PROFILE_DIR/gzippy.flamegraph.svg"
            echo "wrote $PROFILE_DIR/gzippy.flamegraph.svg (FlameGraph.pl)"
        else
            echo "WARN: no inferno or FlameGraph.pl — only gzippy.perf.data (cargo install inferno)" >&2
        fi
    fi

    if command -v heaptrack >/dev/null 2>&1; then
        echo "[heaptrack] one decode (allocation call paths)"
        heaptrack -o "$PROFILE_DIR/gzippy" \
            "$GZIPPY_BIN" -d -c -p "$PROFILE_T" "$FIX" > /dev/null 2>"$PROFILE_DIR/heaptrack.stderr" \
            || true
    else
        HEAPTRACK_MISSING=1
        echo "WARN: heaptrack not installed — alloc detail via perf stacks only" >&2
    fi

    cat >"$PROFILE_DIR/README.txt" <<EOF
profile-single-member-decompression-x86_64 capture
commit: $SHORT_HEAD
threads: $PROFILE_T
fixture: $FIX (gzip -9 single-member)
binary: $GZIPPY_BIN (release + debuginfo, isal-compression)

gzippy.perf.data         perf samples — perf report -i
gzippy.perfstat.txt      counters (IPC, cache, page faults)
gzippy.folded            collapsed stacks for scripts/profile_diff.py
gzippy.flamegraph.svg    CPU flamegraph (open in browser)
gzippy.heaptrack.*.gz    allocation trace (if heaptrack ran)
EOF

    echo "PROFILE_DIR=$PROFILE_DIR"
    if [ "$HEAPTRACK_MISSING" -eq 1 ]; then
        echo "PROFILE_HEAPTRACK=missing"
    fi
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

exit "$FAIL"
REMOTE
REMOTE_EXIT=${PIPESTATUS[0]}
set -e

PROFILE_DIR=$(grep -E '^PROFILE_DIR=' "$REMOTE_LOG" | tail -1 | cut -d= -f2- || true)
PROFILE_SKIP_REASON=$(grep -E '^PROFILE_SKIP_REASON=' "$REMOTE_LOG" | tail -1 | cut -d= -f2- || true)
HEAPTRACK_FLAG=$(grep -E '^PROFILE_HEAPTRACK=missing' "$REMOTE_LOG" | tail -1 || true)
SKIP_ARG="$PROFILE_SKIP_REASON"
if [ -n "$HEAPTRACK_FLAG" ]; then
    SKIP_ARG="heaptrack_missing"
fi

print_local_profile_guide "$PROFILE_DIR" "$SKIP_ARG"

exit "$REMOTE_EXIT"
