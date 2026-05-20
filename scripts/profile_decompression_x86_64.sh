#!/usr/bin/env bash
# profile_decompression_x86_64.sh
#
# Fastest x86_64 decompression CORRECTNESS check. Run by
# `make profile-decompression-x86_64`.
#
# Why x86_64-only: the parallel single-member path (and its known
# corruption bug at T>=9) only runs on x86_64 + ISA-L. arm64 dev
# machines route single-member through libdeflate and never exercise
# the buggy path — so this check MUST run on the homelab x86_64 box.
#
# What it does (and nothing more — this is the FAST check, not the
# full matrix/profile run):
#   1. push the current branch, sync + build it on the remote
#   2. make ONE gzip(1)-CLI fixture (gzip-CLI output, not `gzippy -c`,
#      because only gzip(1) layout reproduces the production bug)
#   3. decode that fixture at a sweep of thread counts
#   4. compare every decode's md5 against the source — FAIL if any
#      thread count disagrees
#
# Total runtime ~35s (build dominates). Output is a plain table on
# stdout ending in `RESULT: PASS` or `RESULT: FAIL`. On ERROR, reruns
# one decode with GZIPPY_DEBUG=1 so `[parallel_sm] driver error: …`
# shows the inner `FetchError` (e.g. `InflateFailed(InvalidBlock)`).
# The CLI also prints `chunk decode failed: …` once that detail is wired
# through `ParallelError::DecodeFailed`.
#
# Config (override via env):
#   GZIPPY_REMOTE_SSH    ssh target      default: -J neurotic root@10.30.0.199
#   GZIPPY_REMOTE_DIR    repo dir there  default: gzippy
#   GZIPPY_FIXTURE_MB    raw fixture MiB default: 64
#   GZIPPY_THREAD_SWEEP  thread counts   default: "1 2 4 8 9 12 16"
#   GZIPPY_PROFILE_TRIALS decode runs per T (default 10; catches flaky races)
set -euo pipefail

REMOTE_SSH="${GZIPPY_REMOTE_SSH:--J neurotic root@10.30.0.199}"
REMOTE_DIR="${GZIPPY_REMOTE_DIR:-gzippy}"
FIXTURE_MB="${GZIPPY_FIXTURE_MB:-64}"
THREAD_SWEEP="${GZIPPY_THREAD_SWEEP:-1 2 4 8 9 12 16}"
PROFILE_TRIALS="${GZIPPY_PROFILE_TRIALS:-10}"

BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "=== profile-decompression-x86_64 (fast correctness check) ==="
echo "branch:   $BRANCH"
echo "remote:   ssh ${REMOTE_SSH}"
echo "fixture:  ${FIXTURE_MB} MiB raw  →  gzip -9  →  decode at T={${THREAD_SWEEP}} (${PROFILE_TRIALS} trials/T)"
echo

echo "--- push $BRANCH so the remote builds exactly local HEAD ---"
if git push origin "$BRANCH" 2>&1 | tail -1; then
    :
else
    echo "WARN: push failed; remote will build whatever origin/$BRANCH already is"
fi
echo

# The remote half. Single-quoted heredoc — nothing expands locally;
# the three values it needs arrive as exported env vars on the ssh line.
ssh ${REMOTE_SSH} \
    "BRANCH='${BRANCH}' REMOTE_DIR='${REMOTE_DIR}' FIXTURE_MB='${FIXTURE_MB}' THREAD_SWEEP='${THREAD_SWEEP}' PROFILE_TRIALS='${PROFILE_TRIALS}' bash -s" \
<<'REMOTE'
set -euo pipefail
cd "$REMOTE_DIR"

echo "--- sync + build ---"
git fetch origin "$BRANCH" --quiet
git reset --hard "origin/$BRANCH" --quiet
echo "HEAD: $(git rev-parse --short HEAD)  $(git log -1 --format=%s)"
BUILD_LOG=/tmp/profx86-build.log
RUSTFLAGS='-C debuginfo=1 -C strip=none -C force-frame-pointers=yes' \
    cargo build --release --features isal-compression >"$BUILD_LOG" 2>&1 || {
    echo "BUILD FAILED"
    grep -E '^error' "$BUILD_LOG" || tail -20 "$BUILD_LOG"
    exit 1
}
grep -E 'Compiling gzippy |Finished' "$BUILD_LOG" || true
test -x target/release/gzippy || { echo "BUILD FAILED: no binary"; exit 1; }
echo

echo "--- fixture (gzip(1) CLI output) ---"
SRC=/tmp/profx86-src.bin
FIX=/tmp/profx86-fixture.gz
head -c "$(( FIXTURE_MB * 1024 * 1024 ))" benchmark_data/silesia-large.bin > "$SRC"
gzip -9 -c "$SRC" > "$FIX"
REF=$(md5sum < "$SRC" | cut -d' ' -f1)
echo "raw $(stat -c%s "$SRC") bytes  →  gz $(stat -c%s "$FIX") bytes"
echo "reference md5: $REF"
echo

echo "--- decode sweep (${PROFILE_TRIALS} trials per T) ---"
printf '%-5s %-8s %-34s %s\n' "T" "trial" "md5" "result"
FAIL=0
ERR_SEEN=""
for T in $THREAD_SWEEP; do
    T_FAIL=0
    for trial in $(seq 1 "$PROFILE_TRIALS"); do
        rc=0
        ./target/release/gzippy -d -c -p "$T" "$FIX" \
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

# On failure: plain stderr from the first failing trial, then a dedicated
# GZIPPY_DEBUG=1 decode (driver logs FetchError detail before CLI maps it).
if [ -n "$ERR_SEEN" ]; then
    FIRST_ERR=$(echo "$ERR_SEEN" | awk '{print $1}')
    echo "--- stderr from first ERROR trial (T=$FIRST_ERR, trial=1, no GZIPPY_DEBUG) ---"
    cat /tmp/profx86-err-"$FIRST_ERR"-1 || true
    echo
    echo "--- diagnostic decode (GZIPPY_DEBUG=1, T=$FIRST_ERR, one shot) ---"
    GZIPPY_DEBUG=1 ./target/release/gzippy -d -c -p "$FIRST_ERR" "$FIX" \
        > /dev/null 2>/tmp/profx86-diag.err || true
    cat /tmp/profx86-diag.err || true
    echo
fi

if [ "$FAIL" -eq 0 ]; then
    echo "RESULT: PASS — every thread count decodes byte-perfect"
    exit 0
else
    echo "RESULT: FAIL — see ERROR / WRONG rows above (diagnostic stderr if ERROR)"
    exit 1
fi
REMOTE
