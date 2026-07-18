#!/usr/bin/env bash
# localize_mac.sh — DELIVERABLE 2: localize the aarch64 T1 gap (macOS-local).
#
# Splits gzippy-native's -p1 instr/B (and cyc/B) gap vs libdeflate into
# pipeline-scaffold / table-build / clean-decode-core using ONLY portable,
# BYTE-EXACT, deterministic-instr perturbations (no x86-only knob):
#
#   pipeline scaffold  = normal -p1  -  GZIPPY_THIN_T1_ORACLE   (byte-exact removal-oracle)
#   table-build litlen = GZIPPY_TBUILD_MULT slope (mult 1..4)    (byte-transparent slope)
#   clean decode core  = thin  -  table-build
#
# Gate-0 (LOUD-FAIL else no number):
#   byte-exact   : every gzippy arm sha == libdeflate sha == gzip -d sha.
#   THIN non-inert : stderr prints "THIN_T1_ORACLE ... marker_chunks=0".
#   TBUILD non-inert : instr/B strictly rises with mult (slope > 0).
#   same sink /dev/null both arms ; -p1 (NOT GZIPPY_THREADS).
#
# SCOPE: macOS-aarch64, NOT-YET-LAW cross-arch. Attribution = HYPOTHESIS-tier,
# STRONG because instr counts are deterministic. Pair with Intel(asm-off).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

CORPORA="silesia big"
N=11
BINARY="$ROOT/target/release/gzippy"
MULTS="1 2 3 4"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --corpora) CORPORA="$2"; shift;; --corpora=*) CORPORA="${1#*=}";;
    -N) N="$2"; shift;; -N*) N="${1#-N}";;
    --binary) BINARY="$2"; shift;; --binary=*) BINARY="${1#*=}";;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "localize_mac.sh: unknown arg '$1'" >&2; exit 2;;
  esac
  shift
done

fail() { echo "### LOCALIZE_MAC GATE FAILED: $* ###" >&2; exit 1; }
command -v libdeflate-gunzip >/dev/null || fail "libdeflate-gunzip not on PATH"
[ -x "$BINARY" ] || fail "gzippy binary not at $BINARY (build: cargo build --release --no-default-features --features gzippy-native)"
GZSHA="$(shasum -a 256 "$BINARY" | cut -d' ' -f1)"

DBG="$(GZIPPY_DEBUG=1 "$BINARY" -dc -p1 /dev/null 2>&1 || true)"
"$BINARY" --version >/dev/null 2>&1 || true

resolve_corpus() {
  local name="$1"
  [ -f "$name" ] && { echo "$name"; return; }
  for c in "/tmp/$name.gz" "$ROOT/benchmark_data/$name.gz" "$ROOT/benchmark_data/$name.tar.gz"; do
    [ -f "$c" ] && { echo "$c"; return; }
  done
  return 1
}

WORK="$(mktemp -d /tmp/localize_mac.XXXXXX)"; TL="$WORK/tl.txt"; CSV="$WORK/samples.csv"
echo "corpus,bytes,arm,mult,rep,instr,cyc" > "$CSV"
trap 'rm -rf "$WORK"' EXIT

parse_tl() { local f="$1" i c; i="$(awk '/instructions retired/{print $1; exit}' "$f")"; c="$(awk '/cycles elapsed/{print $1; exit}' "$f")"; [ -n "$i" ] && [ -n "$c" ] && echo "$i $c"; }

declare -A CP CB
for corp in $CORPORA; do
  p="$(resolve_corpus "$corp")" || fail "corpus '$corp' not found"
  CP[$corp]="$p"
  echo "== GATE-0 byte-exact + non-inert: $corp ($p) =="
  ref="$(gzip -dc "$p" | shasum -a 256 | cut -d' ' -f1)"
  CB[$corp]="$(gzip -dc "$p" | wc -c | tr -d ' ')"
  [ "$ref" = "$(libdeflate-gunzip -dc "$p" | shasum -a 256 | cut -d' ' -f1)" ] || fail "$corp libdeflate sha mismatch"
  [ "$ref" = "$("$BINARY" -dc -p1 "$p" | shasum -a 256 | cut -d' ' -f1)" ] || fail "$corp gzippy normal sha mismatch"
  # THIN byte-exact AND non-inert (marker_chunks=0 banner)
  thin_err="$WORK/thin_err.$corp"
  thin_sha="$(GZIPPY_THIN_T1_ORACLE=1 "$BINARY" -dc -p1 "$p" 2>"$thin_err" | shasum -a 256 | cut -d' ' -f1)"
  [ "$ref" = "$thin_sha" ] || fail "$corp THIN sha mismatch (not byte-exact)"
  grep -q "THIN_T1_ORACLE" "$thin_err" || fail "$corp THIN oracle did NOT fire (no banner — inert)"
  grep -q "marker_chunks=0" "$thin_err" || fail "$corp THIN banner missing marker_chunks=0"
  # TBUILD byte-exact (transparency) at mult=4
  [ "$ref" = "$(GZIPPY_TBUILD_MULT=4 "$BINARY" -dc -p1 "$p" | shasum -a 256 | cut -d' ' -f1)" ] || fail "$corp TBUILD_MULT=4 not byte-transparent"
  echo "   PASS  sha=$ref  bytes=${CB[$corp]}  (THIN byte-exact+non-inert, TBUILD byte-transparent)"
done

# warmup
for corp in $CORPORA; do p="${CP[$corp]}"; libdeflate-gunzip -dc "$p" >/dev/null 2>&1 || true; "$BINARY" -dc -p1 "$p" >/dev/null 2>&1 || true; done

echo "== measure: interleaved best-of-N=$N  (libdeflate, normal, thin, tbuild mult{$MULTS}) =="
for rep in $(seq 1 "$N"); do
  for corp in $CORPORA; do
    p="${CP[$corp]}"; b="${CB[$corp]}"
    /usr/bin/time -l libdeflate-gunzip -dc "$p" >/dev/null 2>"$TL"; read -r I C < <(parse_tl "$TL"); echo "$corp,$b,libdeflate,0,$rep,$I,$C" >> "$CSV"
    /usr/bin/time -l "$BINARY" -dc -p1 "$p" >/dev/null 2>"$TL"; read -r I C < <(parse_tl "$TL"); echo "$corp,$b,normal,0,$rep,$I,$C" >> "$CSV"
    /usr/bin/time -l env GZIPPY_THIN_T1_ORACLE=1 "$BINARY" -dc -p1 "$p" >/dev/null 2>"$TL"; read -r I C < <(parse_tl "$TL"); echo "$corp,$b,thin,0,$rep,$I,$C" >> "$CSV"
    for m in $MULTS; do
      /usr/bin/time -l env GZIPPY_TBUILD_MULT=$m "$BINARY" -dc -p1 "$p" >/dev/null 2>"$TL"; read -r I C < <(parse_tl "$TL"); echo "$corp,$b,tbuild,$m,$rep,$I,$C" >> "$CSV"
    done
  done
done

python3 "$HERE/localize_mac_report.py" "$CSV"; RC=$?
echo
echo "  subject: gzippy-native sha=$GZSHA (FFI off)  | scope: macOS-aarch64 NOT-YET-LAW cross-arch"
exit $RC
