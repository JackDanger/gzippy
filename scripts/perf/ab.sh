#!/bin/bash
# ab.sh — one-command byte-identity + wall A/B against HEAD for a working-tree change.
#
# The whole lever loop an agent needs before a board run:
#   1. builds two release binaries: HEAD (a fresh git worktree) and the
#      working tree, each in its OWN --target-dir (the user's global
#      `~/.cargo/config.toml target-dir` otherwise collides every build
#      into one binary — this bit the first two A/B runs ever run);
#   2. re-links the two vendored paths a fresh worktree lacks (git worktrees
#      do not materialize submodules, and isal-sys's patch path +
#      ISAL_SOURCE break without them);
#   3. byte-diffs every corpus member at every requested level/thread pair;
#   4. interleaves wall timing A/B with medians (round-robin, never one arm
#      after the other — interleaving is the only noise defense on a shared box).
#
# Output: a BA-ratio table and a byte-delta verdict. Local numbers are
# HYPOTHESIS-tier; board law still needs the frozen authority box
# (CLAUDE.md). Byte identity is the DONE bar for "wall-only" claims; any
# byte delta is printed member-by-member, never summarized into silence.
#
# Usage:
#   scripts/perf/ab.sh                 # L2,L4,L9 @ -p1, byte-diff + wall
#   scripts/perf/ab.sh -l 3,5 -p 4     # other cells
#   scripts/perf/ab.sh --keep          # keep .review-tmp/ab-* for re-runs
#   scripts/perf/ab.sh --byte-only     # skip the wall phase
set -euo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
ROOT=$(pwd)
SCRATCH="$ROOT/.review-tmp/ab"

LEVELS="2,4,9"
THREADS="1"
REPS=5
BYTE_ONLY=0
KEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        -l) LEVELS="$2"; shift 2 ;;
        -p) THREADS="$2"; shift 2 ;;
        -r) REPS="$2"; shift 2 ;;
        --byte-only) BYTE_ONLY=1; shift ;;
        --keep) KEEP=1; shift ;;
        *) echo "usage: ab.sh [-l levels] [-p threads] [-r reps] [--byte-only] [--keep]"; exit 1 ;;
    esac
done

# 0. One shared input big enough for stable timing (silesia blocks x2 + text x2).
mkdir -p "$SCRATCH"
if [ ! -s "$SCRATCH/in16m.bin" ]; then
    cat corpus/silesia_blocks/*.bin corpus/silesia_blocks/*.bin \
        test_data/text-1MB.txt test_data/text-1MB.txt > "$SCRATCH/in16m.bin" 2>/dev/null
fi
IN="$SCRATCH/in16m.bin"

# 1. Fresh HEAD worktree. IFS-correct removal, ours only.
if [ -e "$SCRATCH/ldx-baseline" ]; then
    git worktree remove --force "$SCRATCH/ldx-baseline" 2>/dev/null || rm -rf "$SCRATCH/ldx-baseline"
fi
git worktree add --quiet "$SCRATCH/ldx-baseline" HEAD

# 2. The two vendored paths the build needs that worktrees don't carry.
mkdir -p "$SCRATCH/ldx-baseline/vendor/isal-rs"
ln -sfn "$ROOT/vendor/isal-rs/isal-sys" "$SCRATCH/ldx-baseline/vendor/isal-rs/isal-sys"
ln -sfn "$ROOT/vendor/isa-l" "$SCRATCH/ldx-baseline/vendor/isa-l"

# 3. Two separate target dirs — NEVER one shared (global target-dir trap).
echo "[ab] building baseline (HEAD)..." >&2
(cd "$SCRATCH/ldx-baseline" && cargo build --release --target-dir "$SCRATCH/target-A" 2>&1 | tail -1) >&2
cp "$SCRATCH/target-A/release/gzippy" "$SCRATCH/gzA"
echo "[ab] building candidate (working tree)..." >&2
cargo build --release --target-dir "$SCRATCH/target-B" 2>&1 | tail -1 >&2
cp "$SCRATCH/target-B/release/gzippy" "$SCRATCH/gzB"
A="$SCRATCH/gzA"
B="$SCRATCH/gzB"

# 4. Byte identity over the corpus, every level x thread pair requested.
echo "[ab] byte identity:" >&2
byte_mismatch=0
byte_files=0
for lvl in $(echo "$LEVELS" | tr ',' ' '); do
    for f in corpus/silesia_blocks/*.bin test_data/*.txt; do
        n=$(basename "$f")
        for th in $(echo "$THREADS" | tr ',' ' '); do
            byte_files=$((byte_files + 1))
            "$A" -n -$lvl -p"$th" -c "$f" > "$SCRATCH/a.gz" 2>/dev/null
            "$B" -n -$lvl -p"$th" -c "$f" > "$SCRATCH/b.gz" 2>/dev/null
            if ! cmp -s "$SCRATCH/a.gz" "$SCRATCH/b.gz"; then
                echo "  BYTE-DIFF L$lvl p$th $n ($(wc -c < "$SCRATCH/a.gz") vs $(wc -c < "$SCRATCH/b.gz"))"
                byte_mismatch=$((byte_mismatch + 1))
            fi
        done
    done
done
rm -f "$SCRATCH/a.gz" "$SCRATCH/b.gz"
if [ "$byte_mismatch" -eq 0 ]; then
    echo "[ab] byte identity: OK ($byte_files outputs, levels $LEVELS, -p$THREADS)"
else
    echo "[ab] byte identity: $byte_mismatch/$byte_files outputs DIFFER (not a wall-only change)"
fi

# 5. Interleaved wall timing on the shared corpus, medians.
if [ "$BYTE_ONLY" -eq 0 ]; then
python3 - "$A" "$B" "$IN" "$LEVELS" "$REPS" >>"$SCRATCH/wall.log" <<'PY'
import subprocess, sys, time
A, B, inp, levels, reps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(','), int(sys.argv[5])
def run(bin, lvl, th):
    t0 = time.perf_counter()
    subprocess.run([bin, '-n', f'-{lvl}', f'-p{th}', '-c', inp], stdout=subprocess.DEVNULL, check=True)
    return time.perf_counter() - t0
print("--- A/B wall (median of interleaved reps, in16m corpus) ---")
for lvl in levels:
    run(A, lvl, 1); run(B, lvl, 1)  # warm both
    ta, tb = [], []
    for _ in range(reps):
        ta.append(run(A, lvl, 1)); tb.append(run(B, lvl, 1))
    a = sorted(ta)[len(ta) // 2]
    b = sorted(tb)[len(tb) // 2]
    print(f"  L{lvl}: HEAD {a*1000:8.1f} ms | candidate {b*1000:8.1f} ms | B/A = {b/a:.4f}")
PY
tail -3 "$SCRATCH/wall.log"
fi

# 6. Cleanup unless asked to keep.
if [ "$KEEP" -eq 0 ]; then
    git worktree remove --force "$SCRATCH/ldx-baseline" 2>/dev/null || true
    rm -rf "$SCRATCH/target-A" "$SCRATCH/target-B" "$SCRATCH/gzA" "$SCRATCH/gzB"
fi
echo "[ab] scratch: $SCRATCH (in16m.bin + wall.log kept) — --keep retains the binaries too"
