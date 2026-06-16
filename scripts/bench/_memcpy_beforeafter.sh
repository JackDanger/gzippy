#!/usr/bin/env bash
# Before/after instruction-count for the SegmentedU8 memcpy-append fix.
# Baseline = pinned b22e1b14 (the 2 changed files git-checked-out) built into a
# separate target dir; after = the working tree. N=7, perf instructions, sha-gate.
set -u
SRC=/root/gz-lever
EV="cpu_core/instructions/u"
RF="-C target-cpu=native"
B_TGT=/dev/shm/gzl-base
A_TGT=/dev/shm/gzl-tgt   # already built (working tree)
cd "$SRC" || exit 5

F1=src/decompress/parallel/chunk_data.rs
F2=src/decompress/parallel/segmented_buffer.rs
cp "$F1" /tmp/F1.after; cp "$F2" /tmp/F2.after

echo "=== build BASELINE (pinned b22e1b14 of the 2 files) -> $B_TGT ==="
git checkout -- "$F1" "$F2" 2>&1 | tail -1
CARGO_TARGET_DIR="$B_TGT" RUSTFLAGS="$RF" cargo build --release --no-default-features \
  --features gzippy-native --bin gzippy >/tmp/base_build.log 2>&1
rc=$?; [ "$rc" -ne 0 ] && { echo "BASE BUILD FAIL"; grep error /tmp/base_build.log|head; exit 8; }
echo "   baseline built"
# restore the after files (working tree) — A_TGT binary already built from them
cp /tmp/F1.after "$F1"; cp /tmp/F2.after "$F2"

BGZ="$B_TGT/release/gzippy"
AGZ="$A_TGT/release/gzippy"

measure() { # bin corpus t  -> prints instr
  local bin="$1" f="$2" t="$3"
  perf stat -r 7 -e "$EV" -o /tmp/m.stat -- \
    env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 0,2,4,6 "$bin" -d -c -p "$t" "$f" >/dev/null 2>/dev/null
  grep "$EV" /tmp/m.stat | awk '{gsub(/,/,"",$1); print $1}'
}

printf "%-10s %-3s %15s %15s %8s\n" corpus T baseline after pct
for c in silesia monorepo nasa; do
  f=/root/$c.gz
  ref=$(gzip -dc "$f"|sha256sum|cut -d' ' -f1)
  for t in 1 4; do
    sa=$(GZIPPY_FORCE_PARALLEL_SM=1 "$AGZ" -d -c -p "$t" "$f" 2>/dev/null|sha256sum|cut -d' ' -f1)
    [ "$sa" != "$ref" ] && { echo "SHA FAIL after $c T$t"; continue; }
    b=$(measure "$BGZ" "$f" "$t")
    a=$(measure "$AGZ" "$f" "$t")
    pct=$(awk -v b="$b" -v a="$a" 'BEGIN{printf "%+.1f%%", (a-b)/b*100}')
    printf "%-10s T%-2s %15s %15s %8s\n" "$c" "$t" "$b" "$a" "$pct"
  done
done
echo "BEFOREAFTER_DONE"
