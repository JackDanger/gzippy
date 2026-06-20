#!/usr/bin/env bash
# standing_mac.sh — THE MACOS-LOCAL GROUND-TRUTH RIG (aarch64 / Apple Silicon).
#
#   "Where does gzippy-native stand vs the state of the art on aarch64?" -> one
#   local invocation. No guest, no ssh. Mirrors standing.sh's contract for the
#   Mac.
#
# Measurement primitive: `/usr/bin/time -l` exposes the Apple-Silicon PMU per
# process — "instructions retired" (deterministic to ~0.04% warm) and "cycles
# elapsed" (~0.5% on P-cores). Instruction count is therefore a DETERMINISTIC
# instrument; we divide both by the DECOMPRESSED byte count for instr/B + cyc/B
# and report gzippy-native(-pN) vs libdeflate-gunzip ratios.
#
# Usage:
#   scripts/bench/standing/standing_mac.sh                          # silesia+big, -p1, N=11
#   scripts/bench/standing/standing_mac.sh --corpora "silesia big" -N 13
#   scripts/bench/standing/standing_mac.sh --threads "1 4 8"
#   scripts/bench/standing/standing_mac.sh --binary /path/to/gzippy --no-build
#
# Gate-0 self-validation it ENFORCES (a run that fails any prints FAIL, no number):
#   (a) byte-exact   : gzippy sha == libdeflate sha == gzip -d sha, per corpus.
#   (b) non-inert    : GZIPPY_DEBUG shows build-flavor=parallel-sm+pure AND
#                      path=ParallelSM (FFI-off pure-Rust engine, production path).
#   (c) A/A self-test: within-arm instr spread <= 0.10% (gz-vs-gz instr ratio ~1.0);
#                      within-arm cyc spread > 3% marks cyc UNTRUSTED (instr stays).
#   (d) same sink    : /dev/null for BOTH arms.
#   (e) -p1 not GZIPPY_THREADS (the latter is INERT and once caused a phantom 2.2x);
#       cold first run is a warmup (untimed); E-core/throttle outliers dropped.
#
# SCOPE STAMP: results are macOS-aarch64 (pure-Rust ParallelSM kernel
# run_contig_ref_biased). NOT-YET-LAW cross-arch — pair with Intel(asm-off) for
# cross-ISA LAW on the pure-Rust portion. Time-stamped to the built binary sha.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

CORPORA="silesia big"
THREADS="1"
N=11
BUILD=1
BINARY="$ROOT/target/release/gzippy"
KEEP=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --corpora) CORPORA="$2"; shift;;
    --corpora=*) CORPORA="${1#*=}";;
    --threads) THREADS="$2"; shift;;
    --threads=*) THREADS="${1#*=}";;
    -N) N="$2"; shift;;
    -N*) N="${1#-N}";;
    --binary) BINARY="$2"; shift;;
    --binary=*) BINARY="${1#*=}";;
    --no-build) BUILD=0;;
    --keep) KEEP=1;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "standing_mac.sh: unknown arg '$1'" >&2; exit 2;;
  esac
  shift
done

fail() { echo "### STANDING_MAC GATE FAILED: $* ###" >&2; exit 1; }

command -v libdeflate-gunzip >/dev/null || fail "libdeflate-gunzip not on PATH (brew install libdeflate)"
command -v gzip >/dev/null || fail "gzip not on PATH"
[ -x /usr/bin/time ] || fail "/usr/bin/time missing"

# ---- build the FFI-off pure-Rust native binary (unless --no-build) ----
if [ "$BUILD" = "1" ]; then
  echo "== building gzippy-native (FFI off): cargo build --release --no-default-features --features gzippy-native =="
  ( cd "$ROOT" && cargo build --release --no-default-features --features gzippy-native >/dev/null 2>&1 ) \
    || fail "cargo build failed"
  BINARY="$ROOT/target/release/gzippy"
fi
[ -x "$BINARY" ] || fail "gzippy binary not executable at $BINARY"
GZSHA="$(shasum -a 256 "$BINARY" | cut -d' ' -f1)"

# ---- resolve a corpus name to a .gz path ----
resolve_corpus() {
  local name="$1"
  if [ -f "$name" ]; then echo "$name"; return; fi
  for cand in "/tmp/$name.gz" "$ROOT/benchmark_data/$name.gz" "$ROOT/benchmark_data/$name.tar.gz"; do
    [ -f "$cand" ] && { echo "$cand"; return; }
  done
  return 1
}

# ---- GATE-0(b): non-inert / production path ----
echo "== GATE-0(b) build-flavor + path assertion =="
FIRST_CORP="$(echo "$CORPORA" | awk '{print $1}')"
FCPATH="$(resolve_corpus "$FIRST_CORP")" || fail "corpus '$FIRST_CORP' not found"
DBG="$(GZIPPY_DEBUG=1 "$BINARY" -dc -p1 "$FCPATH" 2>&1 >/dev/null || true)"
echo "$DBG" | grep -q "build-flavor=parallel-sm+pure" || fail "build-flavor != parallel-sm+pure (got: $(echo "$DBG" | grep build-flavor))"
echo "$DBG" | grep -q "path=ParallelSM"               || fail "path != ParallelSM (got: $(echo "$DBG" | grep path=))"
echo "   PASS  build-flavor=parallel-sm+pure  path=ParallelSM  sha=$GZSHA"

WORK="$(mktemp -d /tmp/standing_mac.XXXXXX)"
TL="$WORK/tl.txt"
CSV="$WORK/samples.csv"
echo "corpus,bytes,arm,threads,rep,instr,cyc,real" > "$CSV"
cleanup() { [ "$KEEP" = "1" ] || rm -rf "$WORK"; }
trap cleanup EXIT

# parse one /usr/bin/time -l capture -> "instr cyc real"
parse_tl() {
  local f="$1" instr cyc real
  instr="$(awk '/instructions retired/{print $1; exit}' "$f")"
  cyc="$(awk '/cycles elapsed/{print $1; exit}' "$f")"
  real="$(awk 'NR==1{print $1; exit}' "$f")"
  [ -n "$instr" ] && [ -n "$cyc" ] || return 1
  echo "$instr $cyc $real"
}

# ---- per-corpus: GATE-0(a) sha gate + decompressed byte count ----
declare -A CORP_PATH CORP_BYTES
for corp in $CORPORA; do
  p="$(resolve_corpus "$corp")" || fail "corpus '$corp' not found (tried path, /tmp/$corp.gz, benchmark_data)"
  CORP_PATH[$corp]="$p"
  echo "== GATE-0(a) sha + size: $corp ($p) =="
  ref="$(gzip -dc "$p" | shasum -a 256 | cut -d' ' -f1)"
  nbytes="$(gzip -dc "$p" | wc -c | tr -d ' ')"
  ld="$(libdeflate-gunzip -dc "$p" | shasum -a 256 | cut -d' ' -f1)"
  gz="$("$BINARY" -dc -p1 "$p" | shasum -a 256 | cut -d' ' -f1)"
  [ "$ref" = "$ld" ] || fail "$corp: libdeflate sha != gzip -d sha"
  [ "$ref" = "$gz" ] || fail "$corp: gzippy sha != gzip -d sha (BYTE MISMATCH)"
  CORP_BYTES[$corp]="$nbytes"
  echo "   PASS  sha=$ref  bytes=$nbytes"
done

# ---- measurement: interleaved best-of-N, /dev/null sink BOTH arms ----
echo "== measure: interleaved best-of-N=$N  (warmup dropped; arms: libdeflate + gzippy-p{$THREADS}) =="
# warmup (cold first run, untimed) per corpus per arm
for corp in $CORPORA; do
  p="${CORP_PATH[$corp]}"
  libdeflate-gunzip -dc "$p" >/dev/null 2>&1 || true
  for t in $THREADS; do "$BINARY" -dc -p"$t" "$p" >/dev/null 2>&1 || true; done
done

for rep in $(seq 1 "$N"); do
  for corp in $CORPORA; do
    p="${CORP_PATH[$corp]}"; b="${CORP_BYTES[$corp]}"
    # arm: libdeflate (1-thread)
    /usr/bin/time -l libdeflate-gunzip -dc "$p" >/dev/null 2>"$TL"
    read -r I C R < <(parse_tl "$TL") || fail "parse libdeflate time -l ($corp rep $rep)"
    echo "$corp,$b,libdeflate,1,$rep,$I,$C,$R" >> "$CSV"
    # arm(s): gzippy -pN
    for t in $THREADS; do
      /usr/bin/time -l "$BINARY" -dc -p"$t" "$p" >/dev/null 2>"$TL"
      read -r I C R < <(parse_tl "$TL") || fail "parse gzippy time -l ($corp p$t rep $rep)"
      echo "$corp,$b,gzippy,$t,$rep,$I,$C,$R" >> "$CSV"
    done
  done
done

[ "$KEEP" = "1" ] && cp "$CSV" "$ROOT/artifacts/standing_mac_samples.csv" 2>/dev/null || true

# ---- report + gates ----
python3 "$HERE/standing_mac_report.py" "$CSV"
RC=$?
echo
echo "  subject: gzippy-native sha=$GZSHA  (FFI off, build-flavor=parallel-sm+pure)"
echo "  scope:   macOS-aarch64 (Apple Silicon) — NOT-YET-LAW cross-arch; pair with Intel(asm-off) for pure-Rust LAW"
[ "$KEEP" = "1" ] && echo "  raw samples kept: $WORK/samples.csv"
exit $RC
