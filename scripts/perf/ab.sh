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
#   3. byte-diffs every corpus member at every requested level/thread pair,
#      and refuses to print broken output as "identical": nonzero exits,
#      empty streams, and streams that do not decode back through an
#      INDEPENDENT decoder are BYTE-INVALID (wave-4 stress review) — two
#      arms failing identically used to launder a false "byte identity: OK";
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
# Env:
#   AB_SCRATCH=<dir>   reuse a fixed scratch (makes the --keep "for re-runs"
#                      promise actually land somewhere); default stays
#                      per-PID so concurrent runs never share paths
#   AB_B_ENV='V=.. V2=..'  env vars routed to the CANDIDATE ONLY — the loop
#                      for env-gated levers (GZIPPY_LADDER): the change under
#                      test is the environment, not the source, and the
#                      receipts must say so. No spaces inside values.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
ROOT=$(pwd)
# Unique scratch per invocation: concurrent sessions share this repo, and a
# fixed `.review-tmp/ab` path recreates the 2026-11-07 wrong-file-under-the-
# same-name race between two batteries (one rebuilds in16m.bin while the
# other is timing against it). AB_SCRATCH opts back into a fixed path.
SCRATCH="${AB_SCRATCH:-$ROOT/.review-tmp/ab-$$}"

LEVELS="2,4,9"
THREADS="1"
REPS=5
BYTE_ONLY=0
KEEP=0
FEATURES=""
while [ $# -gt 0 ]; do
    case "$1" in
        -l) LEVELS="$2"; shift 2 ;;
        -p) THREADS="$2"; shift 2 ;;
        -r) REPS="$2"; shift 2 ;;
        -f) FEATURES="$2"; shift 2 ;;
        --byte-only) BYTE_ONLY=1; shift ;;
        --keep) KEEP=1; shift ;;
        *) echo "usage: ab.sh [-l levels] [-p threads] [-r reps] [-f cargo-features] [--byte-only] [--keep]"; exit 1 ;;
    esac
done
feat_args=""
[ -n "$FEATURES" ] && feat_args="--features $FEATURES"
# Vacuous-input guard (wave-4): an empty level/thread list must NOT reach the
# byte phase as zero iterations and print "byte identity: OK (0 outputs)"; an
# empty/non-numeric rep count must not die later with a cryptic "integer
# expression expected" inside `test`. Guard BEFORE `[ "$REPS" -lt 1 ]`:
# that coercion would otherwise crash first on -r ''.
levels_list=$(echo "$LEVELS" | tr ',' ' ')
threads_list=$(echo "$THREADS" | tr ',' ' ')
if [ -z "$(echo "$LEVELS" | tr -d ' ,')" ] || [ -z "$(echo "$THREADS" | tr -d ' ,')" ] || [ -z "${REPS// /}" ]; then
    echo "ab.sh: empty -l/-p/-r argument (levels='$LEVELS' threads='$THREADS' reps='$REPS')" >&2
    exit 1
fi
case "$LEVELS" in *[!0-9,]*) echo "ab.sh: -l must be comma-separated integers (got '$LEVELS');" >&2; exit 1 ;; esac
case "$THREADS" in *[!0-9,]*) echo "ab.sh: -p must be comma-separated integers (got '$THREADS');" >&2; exit 1 ;; esac
case "$REPS" in *[!0-9]*) echo "ab.sh: -r must be a positive integer (got '$REPS');" >&2; exit 1 ;; esac
[ "$REPS" -lt 1 ] && REPS=1

SCRATCH_ARG="$SCRATCH"
KEEP_ARG="$KEEP"
# Cleanup on ANY way out (EXIT/INT/TERM/HUP): the wall phase used to leak the
# HEAD worktree and target dirs whenever the run died mid-wall, and a caller
# killing the script directly never reached a plain EXIT trap.
ab_cleanup() {
    st=$(( ${1:-$?} ))
    # Stop the EXIT re-entry, then IGNORE further signals while the cleanup
    # itself is running (clearing handlers would let a second kill nuke the
    # trap's own `git worktree remove` mid-flight — the kill-a run proved that
    # leak empirically). Ignoring is inherited by the cleanup's children too.
    trap - 0
    trap '' 1 2 3 15
    git worktree remove --force "$SCRATCH_ARG/ldx-baseline" 2>/dev/null || true
    if [ "$KEEP_ARG" -eq 0 ]; then
        rm -rf "$SCRATCH_ARG/target-A" "$SCRATCH_ARG/target-B" "$SCRATCH_ARG/gzA" "$SCRATCH_ARG/gzB"
    fi
    exit "$st"
}
trap 'ab_cleanup' EXIT
trap 'ab_cleanup 130' INT
trap 'ab_cleanup 143' TERM
trap 'ab_cleanup 129' HUP

# 0. One shared input big enough for stable timing (silesia blocks x2 + text x2).
# CORPUS VERIFY-STAMP (the 2026-11-07 lesson: an ad-hoc probe script rebuilt this
# corpus at 9.5 MB — one `cat` missing the x2 — and two probe batteries then
# measured a different file under the same name, poisoning three claims before
# detection). The stamp is the recipe's byte count; any drift rebuilds.
# macOS `wc -c <` prints leading whitespace — strip it or the comparison can
# never pass (the wave-1 review found this kiling the tool on its own platform).
# SHA half of the stamp (wave-4 stress-review): the size half cannot see
# same-size-different-bytes content, and a silent rebuild under another
# battery's timer is exactly that poison class. On a REUSED scratch (AB_SCRATCH)
# a sha mismatch against the stamp is a hard fail; a fresh per-PID scratch keeps
# the documented build-on-drift (= first build) semantics.
mkdir -p "$SCRATCH"
stampsz() { wc -c < "$1" | tr -d ' '; }
cansha() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
    else sha256sum "$1" | awk '{print $1}'; fi
}
STAMP="$SCRATCH/in16m.sha256"
IN="$SCRATCH/in16m.bin"
if [ -s "$IN" ] && [ -f "$STAMP" ] && [ "$(<"$STAMP")" != "$(cansha "$IN")" ]; then
    echo "ab.sh: corpus sha DRIFT vs stamp at $IN (same path, different bytes) — refusing to rebuild under a reused scratch; move it aside yourself or pick a fresh AB_SCRATCH" >&2
    exit 2
fi
if [ -s "$IN" ] && [ ! -f "$STAMP" ]; then
    echo "ab.sh: $IN has no sha stamp (pre-hardening scratch?) — identity unverifiable, refusing to measure or rebuild it; delete the file or use a fresh AB_SCRATCH" >&2
    exit 2
fi
if [ ! -s "$IN" ] || [ "$(stampsz "$IN")" != "16085866" ]; then
    cat corpus/silesia_blocks/*.bin corpus/silesia_blocks/*.bin \
        test_data/text-1MB.txt test_data/text-1MB.txt > "$IN" 2>/dev/null
    if [ "$(stampsz "$IN")" != "16085866" ]; then
        echo "ab.sh: corpus verify-stamp FAILED (built $(stampsz "$IN") bytes, expected 16085866)" >&2
        exit 2
    fi
fi
printf '%s\n' "$(cansha "$IN")" > "$STAMP"

# The independent roundtrip decoder for the byte phase (charter bar: every
# stream a claim rests on must decode back through a decoder we do not own).
# Ladder: zcat/gzcat (invoked PATH first, then the absolute /usr/bin paths) ->
# python3's gzip module -> nothing (hard fail). DANGER this box demonstrated:
# an installed gzippy ships argv0 ALIASES named zcat and gzcat, which would
# then validate gzippy with gzippy — so every shell candidate is demoted when
# its --version names gzippy. All decode via STDIN: BSD zcat has a
# compress-(.Z)-style FILE-argument quirk that stdin bypasses.
ab_oracle=""
ab_oracle_bin=""
for cand in zcat /usr/bin/zcat gzcat /usr/bin/gzcat; do
    case "$cand" in
        */*) [ -x "$cand" ] || continue; p="$cand" ;;
        *) p=$(command -v "$cand" 2>/dev/null || true); [ -n "$p" ] || continue ;;
    esac
    if "$p" --version 2>&1 | grep -qi 'gzippy'; then continue; fi
    ab_oracle_bin="$p"
    break
done
if [ -n "$ab_oracle_bin" ]; then
    ab_oracle="$ab_oracle_bin"
    ab_roundtrip() { "$ab_oracle_bin" < "$1"; }
elif command -v python3 >/dev/null 2>&1; then
    ab_oracle="python3 ($(command -v python3))"
    ab_roundtrip() { python3 -c 'import gzip, sys; sys.stdout.buffer.write(gzip.decompress(sys.stdin.buffer.read()))' < "$1"; }
else
    echo "ab.sh: no independent decoder available (zcat/gzcat/python3 all missing or gzippy-shimmed) — byte phase cannot run" >&2
    exit 2
fi

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
(cd "$SCRATCH/ldx-baseline" && cargo build --release $feat_args --target-dir "$SCRATCH/target-A" 2>&1 | tail -1) >&2
cp "$SCRATCH/target-A/release/gzippy" "$SCRATCH/gzA"
echo "[ab] building candidate (working tree)..." >&2
cargo build --release $feat_args --target-dir "$SCRATCH/target-B" 2>&1 | tail -1 >&2
cp "$SCRATCH/target-B/release/gzippy" "$SCRATCH/gzB"
A="$SCRATCH/gzA"
B="$SCRATCH/gzB"
# Provenance stamps: every wall receipt needs corpus + binary identity in its
# log or the cross-session numbers are forensically useless (wave-1 review);
# the candidate's env identity too, or two GZIPPY_LADDER settings leave
# indistinguishable receipts; and same-code runs are flagged as calibration —
# equal binary shas held 4/4 in wave-4 stress runs, so within one invocation
# sha equality is the reliable same-artifact signal (`git status` would ALWAYS
# be non-empty for a real lever, since the lever IS the working tree change).
ab_mode="A/B"
if [ "$(cansha "$A")" = "$(cansha "$B")" ]; then
    ab_mode="A/A-identity (one binary both arms; calibration — unless b_env is set, then the env IS the lever being measured)"
fi
{
    echo "--- canonical: corpus=$(cansha "$IN") A=$(cansha "$A") B=$(cansha "$B") head=$(git rev-parse HEAD) levels=$LEVELS threads=$THREADS features='$FEATURES' b_env='${AB_B_ENV:-}' oracle=$ab_oracle mode=$ab_mode ---"
} >> "$SCRATCH/wall.log"

# 4. Byte identity over the corpus, every level x thread pair requested.
# Wave-4: a binary that errors (or emits nothing) must NOT launder truncated
# output through `cmp` — two arms failing identically used to yield a FALSE
# byte-identity OK (`2>/dev/null` hid the failure; sizes were only stamped
# for the corpus, never the outputs). Guard: nonzero exit / empty stream is
# invalid, and every emitted member stream must roundtrip through the
# independent decoder back to its exact source bytes — the same one
# independent decoder the charter's bar already demands (pipefail makes a
# decoder's nonzero exit poison the pipeline even when `cmp` sees a matching
# prefix, so identical-truncation on both arms can no longer read as OK).
echo "[ab] byte identity (roundtrip oracle: $ab_oracle):" >&2
byte_mismatch=0
byte_invalid=0
byte_files=0
for lvl in $levels_list; do
    for f in corpus/silesia_blocks/*.bin test_data/*.txt; do
        n=$(basename "$f")
        for th in $threads_list; do
            byte_files=$((byte_files + 1))
            rm -f "$SCRATCH/a.gz" "$SCRATCH/b.gz"
            if "$A" -n -$lvl -p"$th" -c "$f" > "$SCRATCH/a.gz" 2>>"$SCRATCH/a-byteerr.log"; then rcA=0; else rcA=$?; fi
            if [ -n "${AB_B_ENV:-}" ]; then
                # AB_B_ENV must be ROUTED THROUGH env, not ${AB_B_ENV} assignment
                # prefixes — assignment-prefix words are parsed before
                # expansion, so an expanded assignment can't reintroduce itself
                # as a command prefix; `env` re-parses it as arguments.
                if env ${AB_B_ENV} "$B" -n -$lvl -p"$th" -c "$f" > "$SCRATCH/b.gz" 2>>"$SCRATCH/b-byteerr.log"; then rcB=0; else rcB=$?; fi
            else
                if "$B" -n -$lvl -p"$th" -c "$f" > "$SCRATCH/b.gz" 2>>"$SCRATCH/b-byteerr.log"; then rcB=0; else rcB=$?; fi
            fi
            if [ "$rcA" -ne 0 ] || [ "$rcB" -ne 0 ] || [ ! -s "$SCRATCH/a.gz" ] || [ ! -s "$SCRATCH/b.gz" ]; then
                echo "  BYTE-INVALID L$lvl p$th $n (rcA=$rcA rcB=$rcB $(wc -c < "$SCRATCH/a.gz" | tr -d ' ') vs $(wc -c < "$SCRATCH/b.gz" | tr -d ' ')); stderr captured in a-/b-byteerr.log"
                byte_invalid=$((byte_invalid + 1))
            elif ! ab_roundtrip "$SCRATCH/a.gz" | cmp -s - "$f" || ! ab_roundtrip "$SCRATCH/b.gz" | cmp -s - "$f"; then
                echo "  BYTE-INVALID L$lvl p$th $n (roundtrip FAILS: an arm's stream does not decode back to $n — truncated/corrupt member output, see byteerr logs)"
                byte_invalid=$((byte_invalid + 1))
            elif ! cmp -s "$SCRATCH/a.gz" "$SCRATCH/b.gz"; then
                echo "  BYTE-DIFF L$lvl p$th $n ($(wc -c < "$SCRATCH/a.gz" | tr -d ' ') vs $(wc -c < "$SCRATCH/b.gz" | tr -d ' '))"
                byte_mismatch=$((byte_mismatch + 1))
            fi
        done
    done
done
if [ "$byte_invalid" -gt 0 ]; then
    echo "[ab] byte identity: FAILED — $byte_invalid/$byte_files outputs BROKEN (nonzero rc, empty stream, or roundtrip decode failure; receipts untrustworthy)" >&2
    exit 3
fi
if [ "$byte_mismatch" -eq 0 ]; then
    echo "[ab] byte identity: OK ($byte_files outputs, levels $LEVELS, -p$THREADS)"
else
    echo "[ab] byte identity: $byte_mismatch/$byte_files outputs DIFFER (not a wall-only change)"
fi

# 5. Interleaved wall timing on the shared corpus, medians. The wall phase
# honors -p (the wave-1 review caught it measuring -p1 while advertising the
# requested thread cells), alternates arm order per rep (an unalternated fixed
# order lets thermal/cache ordering favor one arm every pair), and ties a
# local wall-delta slack note to every receipt.
if [ "$BYTE_ONLY" -eq 0 ]; then
BP_ENV="${AB_B_ENV:-}" python3 - "$A" "$B" "$IN" "$LEVELS" "$REPS" "$THREADS" >>"$SCRATCH/wall.log" <<'PY'
import os, subprocess, sys, time
A, B, inp, levels, reps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(','), int(sys.argv[5])
threads = [int(x) for x in sys.argv[6].split(',')]
# .split() (not .split(' ')) so a stray trailing space in AB_B_ENV can't yield
# an empty key; values with spaces are rejected at the shell layer anyway.
b_env = dict(kv.split('=', 1) for kv in os.environ['BP_ENV'].split()) if os.environ['BP_ENV'] else None
def run(bin, lvl, th):
    env = dict(os.environ)
    if b_env and bin == B:
        env.update(b_env)
    t0 = time.perf_counter()
    subprocess.run([bin, '-n', f'-{lvl}', f'-p{th}', '-c', inp], stdout=subprocess.DEVNULL, check=True, env=env)
    return time.perf_counter() - t0
print("--- A/B wall (median of interleaved reps, in16m corpus) ---", flush=True)
for lvl in levels:
    for th in threads:
        run(A, lvl, th); run(B, lvl, th)  # warm both
        ta, tb = [], []
        for i in range(reps):
            if i % 2 == 0:
                ta.append(run(A, lvl, th)); tb.append(run(B, lvl, th))
            else:
                tb.append(run(B, lvl, th)); ta.append(run(A, lvl, th))
        a = sorted(ta)[len(ta) // 2]
        b = sorted(tb)[len(tb) // 2]
        # Raw reps retained in the receipt so medians are auditable (wave-1:
        # a receipt without its samples is a claim, not evidence), flushed per
        # row so a mid-wall kill keeps the samples already taken.
        print(f"  [raw] L{lvl} p{th} A={[round(x, 4) for x in ta]} B={[round(x, 4) for x in tb]}", flush=True)
        print(f"  L{lvl} -p{th}: HEAD {a*1000:8.1f} ms | candidate {b*1000:8.1f} ms | B/A = {b/a:.4f}"
              f"{'  (B carries b_env: ' + ' '.join(sorted(b_env)) + ')' if b_env else ''}"
              f" [~5% of a wall delta is NOT admissible locally — hypothesis-tier; the frozen box adjudicates]", flush=True)
PY
tail -14 "$SCRATCH/wall.log"
fi

echo "[ab] scratch: $SCRATCH (in16m.bin + in16m.sha256 + wall.log kept) — --keep retains the binaries too"
