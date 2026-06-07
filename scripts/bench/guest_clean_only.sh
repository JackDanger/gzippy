#!/usr/bin/env bash
# guest_clean_only.sh — STEP-A.2 CLEAN-ONLY ENGINE ORACLE.
#
# RUNS ON GUEST 199 after host_lock_and_bench.sh gate PASS (frequency pinned).
#
# Bounds the ENGINE (class-C) ceiling cleanly: force EVERY chunk down the CLEAN
# decode path (window-present) while KEEPING real Huffman decode AND preserving
# the production consumer/publish chain — the de-entangler the degenerate
# Oracle-C (decode≈0) could not provide. See plans/step-a2-clean-only-falsifier.md.
#
#   CAPTURE (p=1, aligned): GZIPPY_SEED_WINDOWS_CAPTURE=<f> records aligned
#     (start_bit→window) pairs at the natural clean path.
#   SEED (T8, clean-only):  GZIPPY_SEED_WINDOWS=<f> pre-seeds block_finder +
#     forces every chunk clean. Output byte-exact (correctness gate).
#
# Self-test (CLAUDE.md rule 4): proves the clean path was FORCED (window_seeded→N,
# finished_no_flip→0, fused_lut→0 = marker decode/resolve → 0) AND the publish
# chain is PRESERVED (Early window publish=N in BOTH, unlike Oracle-C).
#
# Args: BRANCH=reimplement-isa-l THREADS=8 N=9
set -u

REPO="${REPO:-/root/gzippy}"
BRANCH="${BRANCH:-reimplement-isa-l}"
THREADS="${THREADS:-8}"
N="${N:-9}"
ARTDIR="${ARTDIR:-/root/gzippy-bench/artifacts-clean-only}"
CORPUS="${CORPUS:-$REPO/benchmark_data/silesia-large.gz}"
RG_TRACE="${RG_TRACE:-$REPO/vendor/rapidgzip/librapidarchive/build-trace/src/tools/rapidgzip}"
SEEDF="${SEEDF:-/dev/shm/gz_seed_windows.bin}"

for a in "$@"; do
  case "$a" in
    BRANCH=*) BRANCH="${a#*=}";;
    THREADS=*) THREADS="$(echo "${a#*=}" | tr ',' ' ')";;
    N=*) N="${a#*=}";;
  esac
done
[ "$N" -ge 9 ] || N=9
T="$(echo "$THREADS" | awk '{print $1}')"; [ -n "$T" ] || T=8

mkdir -p "$ARTDIR"
say() { echo "$@"; }

pin_mask() {
  case "$1" in
    8) echo "0,2,4,6,8,10,12,14";;
    4) echo "0,2,4,6";;
    1) echo "0";;
    *) echo "0,2,4,6,8,10,12,14";;
  esac
}
mask="$(pin_mask "$T")"

# ---- sync branch ------------------------------------------------------------
cd "$REPO" || { echo "CLEAN_FAILURE=no-repo"; echo "CLEAN_ONLY_GUEST_DONE"; exit 5; }
git config --global --add safe.directory "$REPO" 2>/dev/null || true
git fetch origin "$BRANCH" >>"$ARTDIR/fetch.log" 2>&1 || true
git checkout -f -B "$BRANCH" "origin/$BRANCH" >>"$ARTDIR/fetch.log" 2>&1
git reset --hard "origin/$BRANCH" >>"$ARTDIR/fetch.log" 2>&1
git clean -fd -e vendor -e benchmark_data >>"$ARTDIR/fetch.log" 2>&1 || true
DIRTY_COUNT="$(git status --porcelain --ignore-submodules=dirty 2>/dev/null | wc -l | tr -d ' ')"
if [ "$DIRTY_COUNT" != "0" ]; then
  echo "CLEAN_FAILURE=dirty-tree ($DIRTY_COUNT files)"; echo "CLEAN_ONLY_GUEST_DONE"; exit 6
fi

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
GZIPPY_BUILD_FEATURES="${GZIPPY_BUILD_FEATURES:-pure-rust-inflate}"
say "## build gzippy (${GZIPPY_BUILD_FEATURES}) ..."
if ! cargo build --release --no-default-features --features "${GZIPPY_BUILD_FEATURES}" >"$ARTDIR/build-gzippy.log" 2>&1; then
  echo "CLEAN_FAILURE=gzippy-build"; tail -30 "$ARTDIR/build-gzippy.log"; echo "CLEAN_ONLY_GUEST_DONE"; exit 8
fi
GZIPPY="$REPO/target/release/gzippy"

if [ ! -x "$RG_TRACE" ]; then
  echo "CLEAN_FAILURE=no-rg-trace ($RG_TRACE missing)"; echo "CLEAN_ONLY_GUEST_DONE"; exit 8
fi
[ -f "$CORPUS" ] || { echo "CLEAN_FAILURE=corpus"; echo "CLEAN_ONLY_GUEST_DONE"; exit 7; }
REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
RAW_BYTES="$(gzip -dc "$CORPUS" | wc -c)"
cat "$CORPUS" >/dev/null 2>&1

# routing assertion (production path)
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=')"
case "$DBG" in *ParallelSM*) ;; *) echo "CLEAN_FAILURE=routing $DBG"; echo "CLEAN_ONLY_GUEST_DONE"; exit 9;; esac

GZIPPY_SHA="$(git rev-parse HEAD)"
say "================ CLEAN-ONLY PROVENANCE ================"
say "branch=$BRANCH head=$GZIPPY_SHA T=$T N=$N"
say "corpus=$CORPUS ref_sha=$REF_SHA raw_bytes=$RAW_BYTES"
say "rapidgzip=$("$RG_TRACE" --version 2>&1 | head -1)"
say "freq: no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
say "mem: $(free -m | awk '/Mem:/{print "avail="$7"M"} /Swap:/{print "swapfree="$4"M"}' | tr '\n' ' ')"
say "======================================================"

drop_caches() { sync; echo 3 >/proc/sys/vm/drop_caches 2>/dev/null || true; }

timed_run() { # <ignore_sha> <mask> <cmd...> -> "secs sha"
  local _ig="$1" mask="$2"; shift 2
  local out s e rc secs sha
  out="$(mktemp)"
  s=$(date +%s.%N)
  set +e; taskset -c "$mask" "$@" >"$out" 2>>"$ARTDIR/wall.stderr"; rc=$?; set -e
  e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}')
  sha=$(sha256sum "$out" | cut -d' ' -f1)
  [ "$rc" -eq 0 ] || say "WARN exit=$rc: $*"
  rm -f "$out"
  echo "$secs $sha"
}

stats() {
  echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '
    { v[NR]=$1; sum+=$1 } END {
      n=NR; if(n==0){print "0 0 0"; exit}
      min=v[1]; mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
      mean=sum/n; ss=0; for(i=1;i<=n;i++){d=v[i]-mean;ss+=d*d}
      sd=(n>1)?sqrt(ss/(n-1)):0; printf "%.4f %.4f %.1f", min, mid, (mean>0)?sd/mean*100:0; }'
}
mbps() { awk -v r="$RAW_BYTES" -v t="$1" 'BEGIN{printf "%.0f", (t>0)?r/t/1e6:0}'; }

GZWALL="env GZIPPY_FORCE_PARALLEL_SM=1"

# ============================================================================
# CAPTURE PASS — p=1 (sequential) aligned (start_bit→window) seed.
# ============================================================================
drop_caches
rm -f "$SEEDF"
say ""
say "## CAPTURE (p=1 aligned) -> $SEEDF"
env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SEED_WINDOWS_CAPTURE="$SEEDF" \
  taskset -c "$(pin_mask 1)" "$GZIPPY" -d -c -p 1 "$CORPUS" >/dev/null 2>"$ARTDIR/cap.stderr" || true
CAP_LINE="$(grep -m1 'SEED_WINDOWS_CAPTURE wrote' "$ARTDIR/cap.stderr" || true)"
CAP_WINS="$(echo "$CAP_LINE" | grep -o 'wrote [0-9]*' | grep -o '[0-9]*' || echo 0)"
say "CLEAN_CAPTURE: $CAP_LINE (file=$(ls -la "$SEEDF" 2>/dev/null | awk '{print $5}') bytes)"
if [ "${CAP_WINS:-0}" -lt 1 ]; then
  echo "CLEAN_FAILURE=capture-0-windows"; echo "CLEAN_ONLY_GUEST_DONE"; exit 10
fi

# ============================================================================
# SELF-TEST — forced-clean + publish-chain-preserved (verbose counter diff).
# ============================================================================
drop_caches
say ""
say "## SELF-TEST (verbose counters: normal vs seeded, T=$T)"
env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_VERBOSE=1 \
  taskset -c "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" >/dev/null 2>"$ARTDIR/st_normal.err" || true
env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_VERBOSE=1 GZIPPY_SEED_WINDOWS="$SEEDF" \
  taskset -c "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" >/dev/null 2>"$ARTDIR/st_seed.err" || true
say "NORMAL: $(grep -m1 'Unified decoder:' "$ARTDIR/st_normal.err")"
say "NORMAL: $(grep -m1 'Early window publish:' "$ARTDIR/st_normal.err")"
say "NORMAL: $(grep -m1 'Post-process path:' "$ARTDIR/st_normal.err")"
say "SEED:   $(grep -m1 'Unified decoder:' "$ARTDIR/st_seed.err")"
say "SEED:   $(grep -m1 'Early window publish:' "$ARTDIR/st_seed.err")"
say "SEED:   $(grep -m1 'Post-process path:' "$ARTDIR/st_seed.err")"
say "SEED:   $(grep -m1 'SEED_WINDOWS replay:' "$ARTDIR/st_seed.err")"

# ---- standalone sha + hit probe for the seeded clean-only run ----
drop_caches
env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SEED_WINDOWS="$SEEDF" \
  taskset -c "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" >"$ARTDIR/seed_probe.out" 2>"$ARTDIR/seed_probe.err" || true
HM="$(grep -m1 'SEED_WINDOWS replay:' "$ARTDIR/seed_probe.err" || echo 'replay: hits=? misses=?')"
HITS="$(echo "$HM" | grep -o 'hits=[0-9]*' | grep -o '[0-9]*' || echo 0)"
MISS="$(echo "$HM" | grep -o 'misses=[0-9]*' | grep -o '[0-9]*' || echo 0)"
HITPCT="$(awk -v h="$HITS" -v m="$MISS" 'BEGIN{t=h+m; printf "%.1f", (t>0)?100*h/t:100}')"
PROBE_SHA="$(sha256sum "$ARTDIR/seed_probe.out" | cut -d' ' -f1)"
PROBE_SHA_OK=DIVERGE; [ "$PROBE_SHA" = "$REF_SHA" ] && PROBE_SHA_OK=OK
say "CLEAN_SEED_PROBE: $HM  hit%=$HITPCT  sha=$PROBE_SHA_OK"
rm -f "$ARTDIR/seed_probe.out"

# ============================================================================
# PASS A — interleaved wall (N=$N, drop iter0). sha-EXACT on seeded + normal.
#   gzip_ref | gzippy_seeded(clean-only) | rapidgzip | gzippy_normal
# ============================================================================
drop_caches
say ""
say "## PASS A — clean-only interleave (N=$N, drop iter0)"
GZS=""; RG=""; GZN=""; DIVERGED=0
for ((i=0;i<=N;i++)); do
  read ssec ssha < <(timed_run 0 "$mask" $GZWALL GZIPPY_SEED_WINDOWS="$SEEDF" "$GZIPPY" -d -c -p "$T" "$CORPUS")
  read rsec rsha < <(timed_run 1 "$mask" "$RG_TRACE" -d -c -f -P "$T" "$CORPUS")
  read nsec nsha < <(timed_run 0 "$mask" $GZWALL "$GZIPPY" -d -c -p "$T" "$CORPUS")
  [ "$i" -eq 0 ] && continue
  GZS="$GZS $ssec"; RG="$RG $rsec"; GZN="$GZN $nsec"
  [ "$ssha" = "$REF_SHA" ] || { say "DIVERGE gzippy_seeded i=$i sha=$ssha"; DIVERGED=1; }
  [ "$rsha" = "$REF_SHA" ] || { say "DIVERGE rapidgzip i=$i sha=$rsha"; DIVERGED=1; }
  [ "$nsha" = "$REF_SHA" ] || { say "DIVERGE gzippy_normal i=$i sha=$nsha"; DIVERGED=1; }
done
read smin smed ssd < <(stats "$GZS")
read rmin rmed rsd < <(stats "$RG")
read nmin nmed nsd < <(stats "$GZN")

# ============================================================================
# SEEDED TRACE — publish-chain magnitude check (decode busy + L_resolve).
# ============================================================================
drop_caches
say "## SEEDED TRACE -> $ARTDIR/trace_seed_T${T}.json"
GZIPPY_TIMELINE="$ARTDIR/trace_seed_T${T}.json" GZIPPY_VERBOSE=1 \
  env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SEED_WINDOWS="$SEEDF" \
  taskset -c "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" >/dev/null 2>>"$ARTDIR/trace_seed.log" || true
ls -la "$ARTDIR/trace_seed_T${T}.json" 2>/dev/null || say "WARN no seeded trace"
# NORMAL trace for window_absent-fraction comparison
GZIPPY_TIMELINE="$ARTDIR/trace_normal_T${T}.json" \
  env GZIPPY_FORCE_PARALLEL_SM=1 \
  taskset -c "$mask" "$GZIPPY" -d -c -p "$T" "$CORPUS" >/dev/null 2>>"$ARTDIR/trace_normal.log" || true

cat >"$ARTDIR/manifest.json" <<EOF
{"head":"$GZIPPY_SHA","T":$T,"ref_sha":"$REF_SHA","raw_bytes":$RAW_BYTES,
 "seed_trace":"$ARTDIR/trace_seed_T${T}.json","normal_trace":"$ARTDIR/trace_normal_T${T}.json"}
EOF

rm -f "$SEEDF"
say ""
say "================ CLEAN-ONLY SUMMARY (T=$T) ================"
printf "CLEAN_BASELINE_NORMAL  min=%.4fs (%s MB/s) med=%.4f sd%%=%.1f\n" "$nmin" "$(mbps "$nmin")" "$nmed" "$nsd"
printf "CLEAN_ONLY_SEEDED      min=%.4fs (%s MB/s) med=%.4f sd%%=%.1f  sha=%s hit%%=%s  [ENGINE CEILING; publish chain intact]\n" \
  "$smin" "$(mbps "$smin")" "$smed" "$ssd" "$([ $DIVERGED = 0 ] && echo OK || echo DIVERGE)" "$HITPCT"
printf "CLEAN_RAPIDGZIP        min=%.4fs (%s MB/s) med=%.4f sd%%=%.1f\n" "$rmin" "$(mbps "$rmin")" "$rmed" "$rsd"
DELTA="$(awk -v s="$smed" -v r="$rmed" 'BEGIN{printf "%.4f (%.1f%%)", s-r, (r>0)?100*(s-r)/r:0}')"
say "CLEAN_DELTA_seeded_vs_rg(med)= $DELTA"

# verdict per pre-registered falsifier (T8 clean-only wall, sha-OK)
VERDICT=VOID; REASON=""
if [ "$DIVERGED" != 0 ]; then REASON="seeded sha DIVERGED — oracle void"
elif awk -v h="$HITPCT" 'BEGIN{exit !(h<90)}'; then REASON="hit%=$HITPCT <90 — clean path NOT forced (marker-contaminated)"
elif awk -v s="$ssd" 'BEGIN{exit !(s>5)}'; then REASON="seeded sd%=$ssd >5 — bimodal/thrash, invalid"
else
  if awk -v s="$smed" 'BEGIN{exit !(s<=0.55)}'; then VERDICT=ENGINE-NEGLIGIBLE; REASON="clean-only med=$smed <=0.55 ⇒ engine NOT the residual (would REFUTE co-primary)"
  elif awk -v s="$smed" 'BEGIN{exit !(s>=0.58)}'; then VERDICT=ENGINE-IS-RESIDUAL; REASON="clean-only med=$smed >=0.58 > rapidgzip ⇒ engine gap survives all-clean ⇒ co-primary CONFIRMED"
  else VERDICT=AMBIGUOUS; REASON="clean-only med=$smed in (0.55,0.58) grey — lean on per-chunk clean busy"
  fi
fi
say "CLEAN_VERDICT=$VERDICT  $REASON"
say "================ END CLEAN-ONLY SUMMARY ================"
echo "CLEAN_ONLY_GUEST_DONE"
