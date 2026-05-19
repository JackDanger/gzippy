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
# stdout ending in `RESULT: PASS` or `RESULT: FAIL`.
#
# Config (override via env):
#   GZIPPY_REMOTE_SSH    ssh target      default: -J neurotic root@10.30.0.199
#   GZIPPY_REMOTE_DIR    repo dir there  default: gzippy
#   GZIPPY_FIXTURE_MB    raw fixture MiB default: 64
#   GZIPPY_THREAD_SWEEP  thread counts   default: "1 2 4 8 9 12 16"
set -euo pipefail

REMOTE_SSH="${GZIPPY_REMOTE_SSH:--J neurotic root@10.30.0.199}"
REMOTE_DIR="${GZIPPY_REMOTE_DIR:-gzippy}"
FIXTURE_MB="${GZIPPY_FIXTURE_MB:-64}"
THREAD_SWEEP="${GZIPPY_THREAD_SWEEP:-1 2 4 8 9 12 16}"

BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "=== profile-decompression-x86_64 (fast correctness check) ==="
echo "branch:   $BRANCH"
echo "remote:   ssh ${REMOTE_SSH}"
echo "fixture:  ${FIXTURE_MB} MiB raw  →  gzip -9  →  decode at T={${THREAD_SWEEP}}"
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
    "BRANCH='${BRANCH}' REMOTE_DIR='${REMOTE_DIR}' FIXTURE_MB='${FIXTURE_MB}' THREAD_SWEEP='${THREAD_SWEEP}' bash -s" \
<<'REMOTE'
set -euo pipefail
cd "$REMOTE_DIR"

echo "--- sync + build ---"
git fetch origin "$BRANCH" --quiet
git reset --hard "origin/$BRANCH" --quiet
echo "HEAD: $(git rev-parse --short HEAD)  $(git log -1 --format=%s)"
RUSTFLAGS='-C debuginfo=1 -C strip=none -C force-frame-pointers=yes' \
    cargo build --release --features isal-compression 2>&1 \
    | grep -E 'Compiling gzippy |Finished|^error' || true
test -x target/release/gzippy || { echo "BUILD FAILED"; exit 1; }
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

echo "--- decode sweep ---"
printf '%-5s %-34s %s\n' "T" "md5" "result"
FAIL=0
for T in $THREAD_SWEEP; do
    H=$(./target/release/gzippy -d -c -p "$T" "$FIX" 2>/dev/null | md5sum | cut -d' ' -f1)
    if [ "$H" = "$REF" ]; then
        printf '%-5s %-34s %s\n' "$T" "$H" "ok"
    else
        printf '%-5s %-34s %s\n' "$T" "$H" "WRONG"
        FAIL=1
    fi
done
echo

if [ "$FAIL" -eq 0 ]; then
    echo "RESULT: PASS — every thread count decodes byte-perfect"
    exit 0
else
    echo "RESULT: FAIL — parallel-SM corruption present at the WRONG rows above"
    exit 1
fi
REMOTE
