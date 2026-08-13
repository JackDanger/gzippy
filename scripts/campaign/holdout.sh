#!/usr/bin/env bash
# holdout.sh — THE OVERFIT ALARM. Grade size on a NEVER-TUNE corpus and compare
# the win-rate to the same grading on the tuning board.
#
# WHY THIS EXISTS. Every number this campaign has ever quoted comes from one
# 22-file set. TUNE/GATE splits that set in two, which protects against fitting
# a parameter to the file blocking a gate — but BOTH halves were chosen from the
# same population, so neither can answer the question a user actually asks:
# "will this be smaller on MY archive, which is not in your corpus?"
#
# The holdout is a third population. Its members are archive TYPES absent from
# the tuning corpus (tar-of-source, JSONL logs, protobuf TLV, wide CSV, VM image
# mix, CJK prose, XML feed, FASTA, MIME/base64, Apache log, pointer heap,
# Markdown), generated from seeds by `examples/holdout_gen.rs` — never stored in
# repo as data, byte-identical on every box, and definitionally never tuned on.
#
# THE ALARM: a holdout win-rate MATERIALLY BELOW the board win-rate at the same
# levels, threads and rivals means the board is measuring fit, not compression.
# The comparison is run with THE SAME grading code on both corpora in the same
# invocation, so a difference cannot be a methodology difference.
#
# THE ONE RULE: no parameter, threshold or level-map value may EVER be fitted
# against these files. Reading a holdout number to choose a knob converts the
# alarm into another tuning set and destroys the only unbiased estimate we have.
#
#   scripts/campaign/holdout.sh                  # holdout + board comparison
#   scripts/campaign/holdout.sh --holdout-only   # skip the board leg
#   CAMPAIGN_LEVELS=1,6,9 scripts/campaign/holdout.sh   # cheaper sweep
#
# env: CAMPAIGN_LEVELS (default 1-9), CAMPAIGN_THREADS (default 1,4),
#      CAMPAIGN_CORPUS_ROOT (board leg), CAMPAIGN_OUT, GZIPPY_BIN
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

HOLDOUT_ONLY=0
[ "${1:-}" = "--holdout-only" ] && HOLDOUT_ONLY=1

LEVELS_SPEC="${CAMPAIGN_LEVELS:-1-9}"
THREADS_SPEC="${CAMPAIGN_THREADS:-1,4}"
# zsh does not word-split unquoted parameters and this file may be sourced by a
# reader in either shell; expand ranges explicitly rather than relying on it.
expand_levels() {
  local spec="$1" out=""
  local IFS=','
  for part in $spec; do
    case "$part" in
      *-*) local lo="${part%-*}" hi="${part#*-}"; for ((i = lo; i <= hi; i++)); do out="$out $i"; done;;
      *) out="$out $part";;
    esac
  done
  printf '%s' "${out# }"
}
LEVELS="$(expand_levels "$LEVELS_SPEC")"
THREADS="${THREADS_SPEC//,/ }"

GZ="${GZIPPY_BIN:-$REPO/target/release/gzippy}"
[ -x "$GZ" ] || { echo "holdout: no gzippy at $GZ — cargo build --release" >&2; exit 2; }
GZ_SHA="$(shasum -a 256 "$GZ" | cut -d' ' -f1)"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
DIRTY=0; git -C "$REPO" diff --quiet HEAD -- src Cargo.toml Cargo.lock 2>/dev/null || DIRTY=1

# Rivals. igzip is not packaged on every box; its absence is RECORDED in the
# report rather than passed over in silence (lib.sh G2's rule, same reason).
LD="$(command -v libdeflate-gzip || true)"
PIGZ="$(command -v pigz || true)"
RIVALS="gzip"
[ -n "$PIGZ" ] && RIVALS="$RIVALS pigz"
[ -n "$LD" ] && RIVALS="$RIVALS libdeflate"
IGZIP="$REPO/vendor/isa-l/build/igzip"; [ -x "$IGZIP" ] || IGZIP="$(command -v igzip || true)"
[ -n "$IGZIP" ] && RIVALS="$RIVALS igzip"
ABSENT=""
for r in gzip pigz libdeflate igzip; do case " $RIVALS " in *" $r "*) ;; *) ABSENT="$ABSENT $r";; esac; done

OUT="${CAMPAIGN_OUT:-$HOME/www/gzippy-bench/campaign/holdout-$(git -C "$REPO" rev-parse --short HEAD)}"
mkdir -p "$OUT" || exit 2
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "HOLDOUT  commit=${COMMIT:0:12} dirty=$DIRTY bin=${GZ_SHA:0:16}"
echo "  levels=$LEVELS  threads=$THREADS  rivals=$RIVALS${ABSENT:+  ABSENT:$ABSENT}"
echo "  out=$OUT"

# --- materialize the holdout from seeds; the generator verifies its own pins ---
HOLD="$WORK/holdout"
GEN="$REPO/target/release/examples/holdout_gen"
# Always rebuild: a STALE generator binary is exactly the failure mode the pins
# exist to catch, and it would catch it by refusing a perfectly good run.
( cd "$REPO" && cargo build --release --example holdout_gen >/dev/null 2>&1 ) \
  || { echo "holdout: cannot build holdout_gen" >&2; exit 2; }
[ -x "$GEN" ] || { echo "holdout: no holdout_gen at $GEN" >&2; exit 2; }
"$GEN" "$HOLD" > "$OUT/holdout-manifest.tsv" || {
  echo "holdout: generator REFUSED (pin mismatch) — see stderr above. The holdout" >&2
  echo "  bytes changed, so no win-rate from this run is comparable to an earlier one." >&2
  exit 3; }
echo "  materialized $(wc -l < "$OUT/holdout-manifest.tsv" | tr -d ' ') members, pins verified"

# --- grade a list of files into a TSV ----------------------------------------
# Columns: corpus_set file level threads rival ours rival_bytes win
# `win` is 1 when ours <= rival (CLAUDE.md: "output at least as small AT THE
# LEVEL THE USER TYPED"). A cell whose roundtrip fails is not scored — it ABORTS
# the run, because a corrupt-but-smaller output must never be able to score.
#
# Files are passed as REAL PATHS, never staged as symlinks: gzip's CLI (and ours,
# correctly) refuses a symbolic link without -f, so a staged-symlink corpus dir
# aborted the board leg on its first file. Cite the contract, not a workaround.
grade() { # <set-name> <tsv> <file>...
  local set_name="$1" tsv="$2"; shift 2
  : > "$tsv"
  local f base L T r ours rb sha_in sha_rt
  for f in "$@"; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    sha_in="$(shasum -a 256 "$f" | cut -d' ' -f1)"
    for L in $LEVELS; do
      # rival sizes that do not depend on thread count: measure once, reuse.
      local gz_b ld_b
      gz_b="$(gzip -"$L" -c "$f" </dev/null 2>/dev/null | wc -c | tr -d ' ')"
      ld_b=""; [ -n "$LD" ] && ld_b="$("$LD" -"$L" -c "$f" </dev/null 2>/dev/null | wc -c | tr -d ' ')"
      for T in $THREADS; do
        ours="$WORK/out.gz"
        "$GZ" -"$L" -p "$T" -c "$f" </dev/null > "$ours" 2>/dev/null || {
          echo "holdout: gzippy FAILED on $base L$L T$T" >&2; exit 4; }
        sha_rt="$(gzip -dc "$ours" 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"
        [ "$sha_rt" = "$sha_in" ] || {
          echo "holdout: ROUNDTRIP FAILED on $base L$L T$T — the run is VOID" >&2; exit 5; }
        local ob; ob="$(wc -c < "$ours" | tr -d ' ')"
        for r in $RIVALS; do
          case "$r" in
            gzip) rb="$gz_b";;
            libdeflate) rb="$ld_b";;
            pigz) rb="$("$PIGZ" -"$L" -p "$T" -c "$f" </dev/null 2>/dev/null | wc -c | tr -d ' ')";;
            igzip) # igzip's ladder is 0-4; a level above its max is not a cell.
                   [ "$L" -le 4 ] || continue
                   rb="$("$IGZIP" -"$L" -T "$T" -c "$f" </dev/null 2>/dev/null | wc -c | tr -d ' ')";;
          esac
          [ -n "$rb" ] && [ "$rb" -gt 0 ] || continue
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$set_name" "$base" "$L" "$T" "$r" "$ob" "$rb" \
            "$([ "$ob" -le "$rb" ] && echo 1 || echo 0)" >> "$tsv"
        done
      done
    done
    printf '  %-14s %s\n' "$set_name" "$base"
  done
}

grade holdout "$OUT/holdout.tsv" "$HOLD"/*

BOARD_TSV=""
if [ "$HOLDOUT_ONLY" = 0 ]; then
  CORPUS="${CAMPAIGN_CORPUS_ROOT:-$HOME/www/gzippy-bench/corpus}"
  SPLIT="$REPO/corpus_split.json"
  if [ -d "$CORPUS" ] && [ -f "$SPLIT" ]; then
    # The comparison leg is the TUNE set: the files parameters were actually
    # fitted on. Comparing against GATE would understate the overfit signal,
    # since GATE is itself (partly) held out.
    TUNE_FILES=()
    while IFS= read -r m; do
      [ -f "$CORPUS/$m" ] && TUNE_FILES+=("$CORPUS/$m")
    done < <(python3 -c 'import json,sys;print("\n".join(json.load(open(sys.argv[1]))["tune"]["files"]))' "$SPLIT")
    [ "${#TUNE_FILES[@]}" -gt 0 ] || { echo "holdout: TUNE set resolved to zero files under $CORPUS" >&2; exit 2; }
    grade board "$OUT/board.tsv" "${TUNE_FILES[@]}"
    BOARD_TSV="$OUT/board.tsv"
  else
    echo "  board leg SKIPPED — no corpus at $CORPUS (set CAMPAIGN_CORPUS_ROOT)"
  fi
fi

python3 "$HERE/holdout_report.py" \
  --holdout "$OUT/holdout.tsv" ${BOARD_TSV:+--board "$BOARD_TSV"} \
  --commit "$COMMIT" --binary-sha "$GZ_SHA" --dirty "$DIRTY" \
  --absent-rivals "${ABSENT# }" | tee "$OUT/report.txt"
status=${PIPESTATUS[0]}
echo "  report=$OUT/report.txt"
exit "$status"
