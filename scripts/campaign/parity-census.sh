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
# require every thread count to reproduce the -p1 stream exactly. --roundtrip
# additionally un-gzips each T>1 output and sha256-checks it against the
# original (non-negotiable #1: a smaller corrupt output must VOID, never win).
#
# Usage:
#   scripts/campaign/parity-census.sh [tune|gate|all] [--threads 1,2,4,8,16] [--roundtrip]
#
# Output: parity.tsv + parity.md under the campaign artifacts; exit 0 only if
# every measured cell is IDENTICAL (and, with --roundtrip, roundtrips).
#
# Guards are the campaign lib's, same as board-size.sh: declared corpus only
# (G1), the GATE contract (G3), and an identified subject binary (sha + mtime;
# "a measurement from an unidentified binary is not a measurement").
set -uo pipefail

CAMPAIGN_REPO="${CAMPAIGN_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
# shellcheck source=lib.sh
source "$CAMPAIGN_REPO/scripts/campaign/lib.sh"

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
BIN_SHA=$(sha256sum "$OUR_BIN" | cut -d' ' -f1)
note "subject" "sha=${BIN_SHA:0:12} binary=$OUR_BIN"

campaign_guard_gate "$SET_NAME"
campaign_preflight
campaign_corpus_args "$SET_NAME"

OUT_DIR=$(campaign_outdir "parity-${BIN_SHA:0:12}-$(date +%m%d%H%M)")
OUT_TSV="$OUT_DIR/parity.tsv"
OUT_MD="$OUT_DIR/parity.md"
: > "$OUT_TSV"
printf '# parity census — subject %s\n\n' "$BIN_SHA" > "$OUT_MD"
printf 'set=%s threads=%s roundtrip=%s date=%s\n\n' "$SET_NAME" "$THREADS" "$ROUNDTRIP" "$(date -u +%FT%TZ)" >> "$OUT_MD"
printf '| file | level | threads | verdict | note |\n|---|---|---|---|---|\n' >> "$OUT_MD"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# Corrupt output must FAIL the cell, never pass by being identical:
# identical-to-baseline AND non-roundtripping both count as divergences here.
GZ_DECODER=""
if [ "$ROUNDTRIP" = 1 ]; then
  if [ -x "$CAMPAIGN_REPO/target/release/ungzippy" ]; then
    GZ_DECODER="$CAMPAIGN_REPO/target/release/ungzippy"
  elif command -v gzip >/dev/null 2>&1; then
    GZ_DECODER=gzip
  else
    die "--roundtrip requested but no gzip-class decoder found"
  fi
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
BASE_T="${THREAD_ARR[0]}"

FAILS=0
CELLS=0
for file in "${CORPUS_PATHS[@]}"; do
  fname=$(basename "$file")
  for level in 0 1 2 3 4 5 6 7 8 9; do
    base_sha=""
    for t in "${THREAD_ARR[@]}"; do
      out="$OUT_DIR/.f_${fname}_L${level}_T${t}.gz"
      CELLS=$((CELLS+1))
      if ! "$OUR_BIN" -"$level" -p "$t" -c "$file" > "$out" 2>"$out.err"; then
        FAILS=$((FAILS+1))
        printf '%s\t%s\t%s\tERROR\tencode failed\n' "$fname" "$level" "$t" >> "$OUT_TSV"
        printf '| %s | %s | %s | ERROR | encode failed |\n' "$fname" "$level" "$t" >> "$OUT_MD"
        rm -f "$out" "$out.err"
        continue
      fi
      sha=$(sha256_of "$out")
      if [ "$t" = "$BASE_T" ]; then
        base_sha="$sha"
        printf '%s\t%s\t%s\tbaseline\t-\n' "$fname" "$level" "$t" >> "$OUT_TSV"
      elif [ "$sha" = "$base_sha" ]; then
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
        FAILS=$((FAILS+1))
        printf '%s\t%s\t%s\tDIVERGES\tT1=%s T%s=%s\n' "$fname" "$level" "$t" "${base_sha:0:12}" "$t" "${sha:0:12}" >> "$OUT_TSV"
        printf '| %s | %s | %s | **DIVERGES** | T1=%s T%s=%s |\n' "$fname" "$level" "$t" "${base_sha:0:12}" "$t" "${sha:0:12}" >> "$OUT_MD"
      fi
      rm -f "$out" "$out.err"
    done
  done
done

printf '\n**cells=%d fails=%d**\n' "$CELLS" "$FAILS" >> "$OUT_MD"
note "result" "cells=$CELLS fails=$FAILS"
note "artifact" "$OUT_MD"
[ "$FAILS" -eq 0 ] || { note "FAIL" "divergences found — table: $OUT_TSV"; exit 1; }
note "PASS" "every T>1 output byte-identical to T1"
exit 0
