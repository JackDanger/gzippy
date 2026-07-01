#!/usr/bin/env bash
# mac_pipeline_components.sh — DECOMPOSE gzippy's parallel PIPELINE TAX by
# component on the QUIET mac (Apple Silicon), deterministic instructions.
#
# Context: mac_rg_gap.sh established that gz's parallel pipeline inflates
# instructions T1->T by 1.8-4.1x vs rapidgzip's 1.2-1.8x = a 1.5-2.3x PIPELINE-TAX
# DIFFERENTIAL (arch-independent CONVERGENCE gap). THIS rig localizes WHICH gz
# pipeline component carries that differential, via byte-exact / correct-output
# removal oracles measured with /usr/bin/time -l (instructions retired =
# deterministic ~0.04% warm). Each arm's instruction delta vs base4 = that
# component's causal share of the tax.
#
# ARMS (all byte-EXACT vs gzip -d unless noted; sha-gated):
#   base1        : -p1  baseline (clean in-order decode, NO markers) = the no-tax floor
#   base4        : -p4  baseline (full speculative marker pipeline)   = tax present
#   noprefetch4  : -p4  GZIPPY_NO_PREFETCH=1 (no speculation -> on-demand CLEAN
#                  decode, no markers). base4 - noprefetch4 = CAUSAL marker-machinery
#                  tax (speculation forces window-absent u16-marker decode +
#                  apply_window + replace_markers). parallelism is ~instruction-
#                  neutral so the delta is the marker work, not the thread split.
#   thin1        : -p4  GZIPPY_THIN_T1_ORACLE=1 (clean SERIAL one-pass)  cross-check of base1.
#   nocrc4       : -p4  GZIPPY_FOLD_NOCRC=1 (skip per-byte CRC update). Shows CRC is
#                  NOT a tax component (runs at T1 too). NOT a tax arm; diagnostic only.
#
# GATE-0 (enforced): byte-exact gz==gzip per corpus & arm (nocrc4 exempt);
# path=ParallelSM build-flavor=parallel-sm+pure at p1/p4; same /dev/null sink;
# interleaved best-of-N (default 7); A/A within-arm instr spread reported, >0.5%
# marks the cell UNTRUSTED; warmup dropped.
#
# SCOPE: macOS-aarch64. The PIPELINE tax is coordination/marker-bound (~arch-
# independent) so the COMPONENT RANKING transfers; absolute magnitudes are
# aarch64. AMD/Zen2 + x86 owed for LAW.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
CORPORA="${CORPORA:-silesia monorepo nasa squishy}"
N="${N:-7}"
BUILD="${BUILD:-1}"
B="$ROOT/target/release/gzippy"
WORK="$(mktemp -d /tmp/mac_pipe_comp.XXXXXX)"
CSV="$WORK/samples.csv"
echo "corpus,arm,rep,instr,cyc,sha_ok" > "$CSV"

fail(){ echo "### COMPONENTS GATE FAILED: $* ###" >&2; exit 1; }
resolve(){ for c in "$1" "/tmp/$1.gz" "$ROOT/benchmark_data/$1.gz"; do [ -f "$c" ] && { echo "$c"; return; }; done; return 1; }

if [ "$BUILD" = 1 ]; then
  echo "== build gzippy-native (FFI off) =="
  ( cd "$ROOT" && cargo build --release --no-default-features --features gzippy-native >/dev/null 2>&1 ) || fail "build"
fi
[ -x "$B" ] || fail "no binary $B"
GIT="$(cd "$ROOT" && git rev-parse --short HEAD)"; SHA="$(shasum -a 256 "$B" | cut -d' ' -f1)"

# GATE-0(b) path assertion
for t in 1 4; do
  D="$(GZIPPY_DEBUG=1 "$B" -dc -p"$t" "$(resolve "$(echo "$CORPORA"|awk '{print $1}')")" 2>&1 >/dev/null || true)"
  echo "$D" | grep -q "build-flavor=parallel-sm+pure" || fail "build-flavor p$t"
  echo "$D" | grep -q "path=ParallelSM" || fail "path p$t"
done
echo "   PASS path=ParallelSM build=parallel-sm+pure  git=$GIT sha=${SHA:0:12}"

declare -A REF
for corp in $CORPORA; do
  p="$(resolve "$corp")" || fail "corpus $corp"
  REF[$corp]="$(gzip -dc "$p" | shasum -a 256 | cut -d' ' -f1)"
done

# arm runner: prints "instr cyc sha"
arm_cmd(){  # $1 corpus-path ; sets ENVV/ARGS via globals
  local p="$1"
  # nocrc4 exits nonzero by design (CRC verify fails) — tolerate it; time -l
  # still wrote the PMU counters we parse.
  env $ENVV /usr/bin/time -l "$B" -dc $ARGS "$p" >/dev/null 2>"$WORK/t.txt" || true
  awk '/instructions retired/{i=$1} /cycles elapsed/{c=$1} END{print i, c}' "$WORK/t.txt"
}
arm_sha(){ local p="$1"; env $ENVV "$B" -dc $ARGS "$p" 2>/dev/null | shasum -a 256 | cut -d' ' -f1; }

set_arm(){
  case "$1" in
    base1)       ENVV="";                          ARGS="-p1";;
    base4)       ENVV="";                          ARGS="-p4";;
    noprefetch4) ENVV="GZIPPY_NO_PREFETCH=1";      ARGS="-p4";;
    thin1)       ENVV="GZIPPY_THIN_T1_ORACLE=1";   ARGS="-p4";;
    nocrc4)      ENVV="GZIPPY_FOLD_NOCRC=1";       ARGS="-p4";;
  esac
}
ARMS="base1 base4 noprefetch4 thin1 nocrc4"

# warmup + sha gate
echo "== warmup + sha gate =="
for corp in $CORPORA; do
  p="$(resolve "$corp")"
  for a in $ARMS; do
    set_arm "$a"
    env $ENVV "$B" -dc $ARGS "$p" >/dev/null 2>&1 || true
    # nocrc4 exits nonzero by design (CRC verify fails; bytes still correct),
    # so tolerate a nonzero sha pipeline under set -e/pipefail.
    s="$(arm_sha "$p" || true)"; ok=1
    if [ "$a" != "nocrc4" ]; then [ "$s" = "${REF[$corp]}" ] || { echo "  BYTE MISMATCH $corp/$a"; ok=0; }; fi
    echo "$corp,$a,0,0,0,$ok" >> "$CSV"
    [ "$ok" = 1 ] || fail "byte mismatch $corp/$a"
  done
done
echo "   PASS all arms byte-exact (nocrc4 exempt)"

# interleaved measurement
echo "== measure interleaved best-of-N=$N (instructions retired; /dev/null sink) =="
for rep in $(seq 1 "$N"); do
  for corp in $CORPORA; do
    p="$(resolve "$corp")"
    for a in $ARMS; do
      set_arm "$a"
      read -r I C < <(arm_cmd "$p")
      echo "$corp,$a,$rep,$I,$C,1" >> "$CSV"
    done
  done
done
cp "$CSV" "$ROOT/artifacts/mac_pipeline_components_samples.csv" 2>/dev/null || true
python3 "$HERE/mac_pipeline_components_report.py" "$CSV"
echo "  subject gzippy-native git=$GIT sha=${SHA:0:12} ; scope macOS-aarch64 NOT-YET-LAW (x86/AMD owed)"
echo "  raw: $CSV"
