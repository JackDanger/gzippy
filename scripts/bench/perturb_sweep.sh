#!/usr/bin/env bash
# perturb_sweep.sh — guest-side fulcrum-perturb sweep producer (rival-discrimination).
# Produces a fulcrum `perturb` sweep-dir for ONE (corpus, T, knob):
#   baseline.txt, baseline_recheck.txt, spin/t{10,20,30}.txt, sleep/t{10,20,30}.txt,
#   meta.txt (region_self_ms is a PLACEHOLDER -> patched locally from the T1 denom).
# All runs: GZIPPY_FORCE_PARALLEL_SM=1, taskset MASK, regular-file sink, sha-verified.
# Injection levels are mapped so the busy deltas are proportional 1:2:3 (matches
# fulcrum INJECT_LEVELS=[10,20,30]):
#   KNOB=A (RIVAL A, clean inner Huffman, GZIPPY_SLOW_MODE):
#       baseline forces the CAREFUL loop at ~0 injection (=5 -> spin 1) so the
#       fast->careful switch cost is constant across all arms and cancels.
#       t10=50(spin11) t20=100(spin22) t30=150(spin33).
#   KNOB=B (RIVAL B, u16 marker resolution, GZIPPY_SLOW_MARKER_MODE):
#       marker path is always careful; baseline = knob OFF.
#       t10=50 t20=100 t30=150.
set -u
fail(){ echo "SWEEP_FAIL=$1"; exit "${2:-1}"; }

CORPUS="${CORPUS:?}"; T="${T:?}"; MASK="${MASK:?}"; KNOB="${KNOB:?}"
OUTDIR="${OUTDIR:?}"; N="${N:-15}"; GZIPPY_BIN="${GZIPPY_BIN:?}"
REF_SHA="${REF_SHA:?}"; MODE="${MODE:-full}"   # full | denom(baseline+spin/t30)

[ -x "$GZIPPY_BIN" ] || fail "no-binary:$GZIPPY_BIN" 5
mkdir -p "$OUTDIR/spin" "$OUTDIR/sleep" || fail "mkdir" 5
SINK="$OUTDIR/sink.bin"; : > "$SINK" || fail "sink" 5

case "$KNOB" in
  A) BASE_ENV="GZIPPY_SLOW_MODE=5"; KVAR="GZIPPY_SLOW_MODE";;
  B) BASE_ENV="";                   KVAR="GZIPPY_SLOW_MARKER_MODE";;
  *) fail "bad-knob:$KNOB" 2;;
esac
L1=50; L2=100; L3=150   # -> spin 11/22/33, proportional 1:2:3

timed(){ # $1=extra-env-string ; echo "secs sha"
  local ev="$1" s e secs sha
  s=$(date +%s.%N)
  # shellcheck disable=SC2086
  env GZIPPY_FORCE_PARALLEL_SM=1 $ev taskset -c "$MASK" "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS" >"$SINK" 2>/dev/null
  e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}')
  sha=$(sha256sum "$SINK" | cut -d' ' -f1)
  echo "$secs $sha"
}

run_into(){ # $1=file $2=env  ; one sample, sha-checked
  local f="$1" ev="$2" out secs sha
  out=$(timed "$ev"); secs=${out%% *}; sha=${out##* }
  [ "$sha" = "$REF_SHA" ] || { echo "!!SHA $f got=$sha"; echo "SHA_DIVERGE" >> "$OUTDIR/diverge.txt"; }
  echo "$secs" >> "$f"
}

# Fresh files
: > "$OUTDIR/baseline.txt"; : > "$OUTDIR/baseline_recheck.txt"
for L in 10 20 30; do : > "$OUTDIR/spin/t$L.txt"; : > "$OUTDIR/sleep/t$L.txt"; done
: > "$OUTDIR/diverge.txt"

# baseline block (start of bracket)
for ((i=0;i<N;i++)); do run_into "$OUTDIR/baseline.txt" "$BASE_ENV"; done

if [ "$MODE" = denom ]; then
  # T1 denominator: baseline + top spin level only (interleaved), for region_self.
  for ((i=0;i<N;i++)); do
    run_into "$OUTDIR/baseline.txt" "$BASE_ENV"
    run_into "$OUTDIR/spin/t30.txt" "$BASE_ENV $KVAR=$L3"
  done
  rm -f "$SINK"
  DIV=$(grep -c SHA_DIVERGE "$OUTDIR/diverge.txt" 2>/dev/null); DIV=${DIV:-0}
  echo "DENOM_DONE knob=$KNOB cell=$(basename "$CORPUS" .gz):T$T diverge=$DIV outdir=$OUTDIR"
  exit 0
fi

# interleaved sweep: spin t10/20/30 then sleep t10/20/30, round-robin
for ((i=0;i<N;i++)); do
  run_into "$OUTDIR/spin/t10.txt"  "$BASE_ENV $KVAR=$L1"
  run_into "$OUTDIR/spin/t20.txt"  "$BASE_ENV $KVAR=$L2"
  run_into "$OUTDIR/spin/t30.txt"  "$BASE_ENV $KVAR=$L3"
  run_into "$OUTDIR/sleep/t10.txt" "$BASE_ENV $KVAR=$L1 GZIPPY_SLOW_KIND=sleep"
  run_into "$OUTDIR/sleep/t20.txt" "$BASE_ENV $KVAR=$L2 GZIPPY_SLOW_KIND=sleep"
  run_into "$OUTDIR/sleep/t30.txt" "$BASE_ENV $KVAR=$L3 GZIPPY_SLOW_KIND=sleep"
done

# baseline recheck block (end of bracket -> drift detector)
for ((i=0;i<N;i++)); do run_into "$OUTDIR/baseline_recheck.txt" "$BASE_ENV"; done
rm -f "$SINK"

DIV=$(grep -c SHA_DIVERGE "$OUTDIR/diverge.txt" 2>/dev/null); DIV=${DIV:-0}
# meta.txt — region_self_ms is a PLACEHOLDER (patched locally from T1 denom).
cat > "$OUTDIR/meta.txt" <<EOF
region=$( [ "$KNOB" = A ] && echo clean_inner_huffman_loop || echo u16_marker_resolution )
region_self_ms=1.0
perturb_cmd=$KVAR=${L1}/${L2}/${L3} (spin11/22/33) FORCE_PARALLEL_SM corpus=$(basename "$CORPUS") T=$T mask=$MASK
sha_ok=$( [ "$DIV" -eq 0 ] && echo 1 || echo 0 )
cell_id=$(basename "$CORPUS" .gz):T$T:RIVAL_$KNOB
freeze_state=frozen
quiet_state=quiet
EOF
echo "SWEEP_DONE knob=$KNOB cell=$(basename "$CORPUS" .gz):T$T diverge=$DIV outdir=$OUTDIR"
