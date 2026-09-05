#!/usr/bin/env bash
# PARITY CENSUS — is every T>1 output byte-identical to T1 on the real corpus?
#
# One question. The decoder answered it in four months; the encoder's parallel
# path is being re-pointed at the 55-commit "one encoder" stack, and the
# thread-parity test the tree currently ships covers only four generated
# fixtures — at those sizes the chunk grid is trivially single-chunk, so the
# test cannot see a real chunk seam (docs/plan-2026-09-one-encoder.md §3).
#
# What it does: gzippy -{level} -c FILE at each -p, sha256 every output, and
# require every thread count T>1 to reproduce the -p1 stream EXACTLY.
# --roundtrip additionally un-gzips each T>1 output and sha256-checks it
# against the original (non-negotiable #1: a smaller corrupt output must VOID,
# never win).
#
# Route assertion: every T>1 encode is invoked with -v and its stderr is
# required to name the requested thread count ("... N threads)"). Census
# receipt (board-size.sh:36-37): without an explicit -p, gzippy silently used
# T=num_cpus — an all-equal matrix is also what a silent fallback to the
# single-route produces, so an unproven parallel route is a FAIL here, not a
# pass. This is the one wrong answer the instrument must never give.
#
# Usage:
#   scripts/campaign/parity-census.sh [tune|gate|all] [--threads 1,2,4,8,16] [--roundtrip]
#   --threads must contain 1 (the T1 baseline) and at least one count > 1.
#
# Output: parity.tsv + parity.md under the campaign artifacts; exit 0 only if
# every compared cell is IDENTICAL (and, with --roundtrip, roundtrips).
#
# Guards are the campaign lib's, same as board-size.sh: declared corpus only
# (G1), the GATE contract (G3), and an identified subject binary
# (repo sha + dirty flag from campaign_preflight; "a measurement from an
# unidentified binary is not a measurement").
set -uo pipefail

CAMPAIGN_REPO="${CAMPAIGN_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
# shellcheck source=lib.sh
source "$CAMPAIGN_REPO/scripts/campaign/lib.sh"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else return 1; fi
}
# A hash tool is not optional: a missing one otherwise makes sha256_of yield
# the empty string everywhere, and "" == "" passes every cell.
command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 \
  || die "no sha256sum or shasum on PATH" "an unhashable census would compare empty strings and pass"

THREADS="1,2,4,8,16"
SET_NAME="tune"
ROUNDTRIP=0
while [ $# -gt 0 ]; do
  case "$1" in
    tune|gate|all) SET_NAME="$1"; shift;;
    --threads) THREADS="$2"; shift 2;;
    --roundtrip) ROUNDTRIP=1; shift;;
    *) die "unknown argument '$1'" "usage: parity-census.sh [tune|gate|all] [--threads 1,2,4,8,16] [--roundtrip]";;
  esac
done

OUR_BIN="${LOADING_DOCK:-$CAMPAIGN_REPO/target/release/gzippy}"
[ -x "$OUR_BIN" ] || die "no gzippy at $OUR_BIN" \
  "build it: cargo build --release --no-default-features --features pure-rust-inflate" \
  "or set   LOADING_DOCK=/path/to/gzippy"
BIN_SHA=$(sha256_of "$OUR_BIN") || die "no sha256 tool on this box" \
  "install sha256sum (coreutils) or shasum — an unidentified-binary census is void"

# The question this instrument exists for is "T>1 == T1": a thread list
# without 1 has no baseline, and a list of only 1 compares nothing. Either
# way the run would print a verdict with zero comparisons, which is the
# false pass this script exists to make impossible.
IFS=',' read -r -a THREAD_ARR <<< "$THREADS"
has_t1=0; has_gt1=0
for t in "${THREAD_ARR[@]}"; do
  [ "$t" = "1" ] && has_t1=1
  [ "$t" -gt 1 ] 2>/dev/null && has_gt1=1
done
[ "$has_t1" = 1 ] || die "--threads must include 1 (the T1 baseline)" "got: $THREADS"
[ "$has_gt1" = 1 ] || die "thread list has no T>1 member — nothing to compare" "got: $THREADS"

note "subject" "sha=${BIN_SHA:0:12} binary=$OUR_BIN"

# gzippy splices GZIP/PIGZ env vars ahead of argv, so whatever the box exports
# silently retargets every encode here (worst case: GZIP=-d errors every cell;
# quiet case: a -b or format flag measures a different encoder). The census
# must measure the binary it built.
unset GZIP PIGZ

campaign_guard_gate "$SET_NAME"
campaign_preflight
campaign_corpus_args "$SET_NAME"

OUT_DIR=$(campaign_outdir "parity-${BIN_SHA:0:12}-$(date +%m%d%H%M)")
OUT_TSV="$OUT_DIR/parity.tsv"
OUT_MD="$OUT_DIR/parity.md"
: > "$OUT_TSV"
printf '# parity census — subject binary sha %s (repo %s%s)\n\n' \
  "$BIN_SHA" "${CAMPAIGN_GZIPPY_SHA:0:12}" \
  "$( [ "${CAMPAIGN_GZIPPY_DIRTY:-0}" = 0 ] && echo ', clean' || echo ', DIRTY TREE' )" > "$OUT_MD"
printf 'set=%s threads=%s roundtrip=%s date=%s\n\n' "$SET_NAME" "$THREADS" "$ROUNDTRIP" "$(date -u +%FT%TZ)" >> "$OUT_MD"
printf '| file | level | threads | verdict | note |\n|---|---|---|---|---|\n' >> "$OUT_MD"

# Corrupt output must FAIL the cell, never pass by being identical:
# identical-to-baseline AND non-roundtripping both count as divergences here.
# Prefer the SYSTEM gzip as the independent decoder: target/release/ungzippy
# is a symlink to gzippy itself (Makefile: "Create ungzippy symlink"), so it
# is not independent — a shared encode/decode bug would pass both legs.
GZ_DECODER=""
if [ "$ROUNDTRIP" = 1 ]; then
  if command -v gzip >/dev/null 2>&1; then
    GZ_DECODER=gzip
  elif [ -x "$CAMPAIGN_REPO/target/release/ungzippy" ]; then
    UNG_SHA=$(sha256_of "$CAMPAIGN_REPO/target/release/ungzippy") || exit 2
    if [ "$UNG_SHA" = "$BIN_SHA" ]; then
      die "--roundtrip: ungzippy is gzippy itself (symlink) and gzip is absent" \
        "install gzip, or provide an independent decoder on PATH"
    fi
    GZ_DECODER="$CAMPAIGN_REPO/target/release/ungzippy"
  else
    die "--roundtrip requested but no gzip-class decoder found"
  fi
  printf 'roundtrip decoder: %s\n\n' "$GZ_DECODER" >> "$OUT_MD"
fi

# CAMPAIGN_CORPUS_ARGS is a --corpus <path> sequence from campaign_corpus_args.
CORPUS_PATHS=()
i=0
while [ $i -lt ${#CAMPAIGN_CORPUS_ARGS[@]} ]; do
  if [ "${CAMPAIGN_CORPUS_ARGS[$i]}" = "--corpus" ]; then
    CORPUS_PATHS+=("${CAMPAIGN_CORPUS_ARGS[$((i+1))]}")
    i=$((i+2))
  else
    i=$((i+1))
  fi
done
[ ${#CORPUS_PATHS[@]} -gt 0 ] || die "corpus resolved to zero paths"

IFS=',' read -r -a THREAD_ARR <<< "$THREADS"

# A cell that cannot PROVE it ran the parallel route for its requested thread
# count is a divergence-class failure (see the header receipt).
route_proven() {  # route_proven <errfile> <threads>
  grep -q "($2 threads)" "$1"
}

FAILS=0
CELLS=0
COMPARED=0
for file in "${CORPUS_PATHS[@]}"; do
  fname=$(basename "$file")
  for level in 0 1 2 3 4 5 6 7 8 9; do
    base_sha=""
    for t in "${THREAD_ARR[@]}"; do
      out="$OUT_DIR/.f_${fname}_L${level}_T${t}.gz"
      err="$OUT_DIR/.e_${fname}_L${level}_T${t}.err"
      CELLS=$((CELLS+1))
      if ! "$OUR_BIN" -"$level" -p "$t" -v -c "$file" > "$out" 2>"$err"; then
        FAILS=$((FAILS+1))
        printf '%s\t%s\t%s\tERROR\tencode failed\n' "$fname" "$level" "$t" >> "$OUT_TSV"
        printf '| %s | %s | %s | ERROR | encode failed |\n' "$fname" "$level" "$t" >> "$OUT_MD"
        rm -f "$out" "$err"
        continue
      fi
      # Verify the output is a thing the instrument means to compare: nonempty
      # and gzip magic. An empty stdout (e.g. output silently went elsewhere)
      # otherwise hashes "" for every cell and all-empty compares PASS.
      if [ ! -s "$out" ] || [ "$(head -c 2 "$out" | od -An -tx1 | tr -d ' \n')" != "1f8b" ]; then
        FAILS=$((FAILS+1))
        printf '%s\t%s\t%s\tNOT-GZIP\toutput empty or missing 1f8b magic\n' "$fname" "$level" "$t" >> "$OUT_TSV"
        printf '| %s | %s | %s | NOT-GZIP | empty / no magic |\n' "$fname" "$level" "$t" >> "$OUT_MD"
        rm -f "$out" "$err"
        continue
      fi
      # Route assertion: every T>1 encode must have proved its thread count
      # on stderr. A silent single-route fallback (or a box that lost the -p)
      # makes every sha in the matrix equal WITHOUT answering the question.
      if [ "$t" -gt 1 ] && ! route_proven "$err" "$t"; then
        FAILS=$((FAILS+1))
        printf '%s\t%s\t%s\tROUTE-UNCONFIRMED\t-v stderr lacks "($t threads)"\n' "$fname" "$level" "$t" >> "$OUT_TSV"
        printf '| %s | %s | %s | ROUTE-UNCONFIRMED | no parallel-route proof |\n' "$fname" "$level" "$t" >> "$OUT_MD"
        rm -f "$out" "$err"
        continue
      fi
      sha=$(sha256_of "$out") || exit 2
      if [ "$t" = "1" ]; then
        base_sha="$sha"
        printf '%s\t%s\t%s\tT1-baseline\t-\n' "$fname" "$level" "$t" >> "$OUT_TSV"
      elif [ -z "$base_sha" ]; then
        # Baseline leg failed to produce a sha; fail closed instead of
        # hashing against "".
        FAILS=$((FAILS+1))
        printf '%s\t%s\t%s\tNO-BASELINE\tT1 leg did not produce a baseline\n' "$fname" "$level" "$t" >> "$OUT_TSV"
        printf '| %s | %s | %s | NO-BASELINE | T1 missing |\n' "$fname" "$level" "$t" >> "$OUT_MD"
      elif [ "$sha" = "$base_sha" ]; then
        COMPARED=$((COMPARED+1))
        verdict="IDENTICAL"
        note_txt="-"
        if [ "$ROUNDTRIP" = 1 ]; then
          rt=$(mktemp)
          if ! "$GZ_DECODER" -dc < "$out" > "$rt" 2>/dev/null \
             || [ "$(sha256_of "$rt")" != "$(sha256_of "$file")" ]; then
            verdict="IDENTICAL-BUT-RT-FAIL"
            note_txt="roundtrip did not reproduce the original"
            FAILS=$((FAILS+1))
          fi
          rm -f "$rt"
        fi
        printf '%s\t%s\t%s\t%s\t%s\n' "$fname" "$level" "$t" "$verdict" "$note_txt" >> "$OUT_TSV"
        printf '| %s | %s | %s | %s | %s |\n' "$fname" "$level" "$t" "$verdict" "$note_txt" >> "$OUT_MD"
      else
        COMPARED=$((COMPARED+1))
        FAILS=$((FAILS+1))
        printf '%s\t%s\t%s\tDIVERGES\tT1=%s T%s=%s\n' "$fname" "$level" "$t" "${base_sha:0:12}" "$t" "${sha:0:12}" >> "$OUT_TSV"
        printf '| %s | %s | %s | **DIVERGES** | T1=%s T%s=%s |\n' "$fname" "$level" "$t" "${base_sha:0:12}" "$t" "${sha:0:12}" >> "$OUT_MD"
      fi
      rm -f "$out" "$err"
    done
  done
done

printf '\n**cells=%d compared=%d fails=%d**\n' "$CELLS" "$COMPARED" "$FAILS" >> "$OUT_MD"
note "result" "cells=$CELLS compared=$COMPARED fails=$FAILS"
note "artifact" "$OUT_MD"
if [ "$COMPARED" -eq 0 ]; then
  note "FAIL" "zero cells compared — a census that compares nothing cannot pass"
  exit 1
fi
[ "$FAILS" -eq 0 ] || { note "FAIL" "divergences found — table: $OUT_TSV"; exit 1; }
note "PASS" "every T>1 output byte-identical to T1 (compared=$COMPARED)"
exit 0
