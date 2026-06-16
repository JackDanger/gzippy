#!/usr/bin/env bash
# _optgate_guest.sh — GUEST-SIDE CAPTURE harness for `fulcrum optgate`.
#
# Produces the cyc/byte A/B artifact the optgate ANALYZER renders a verdict from
# (fulcrum src/optgate.rs::OptGateInput). Fulcrum does NOT launch binaries; THIS
# script is the project's measurement policy: it runs the interleaved, frozen-box
# A/B (base vs after vs rapidgzip + a clean-path T1 pair), captures per-run
# `perf stat` cycles+instructions, the bytes decompressed, the box run-queue, and
# the output sha — then writes the per-arm sample files + the ENV-INVARIANT
# manifest. The mac-side wrapper assembles optgate.json and renders the verdict.
#
# OUTPUTS under $ARTDIR/$RUNID/:
#   samples_base.txt  samples_after.txt  samples_rg.txt
#   samples_clean_base.txt  samples_clean_after.txt
#       per line: "<cycles> <instructions> <bytes> <procs_running> <sha>"
#   meta.env          REFERENCE_SHA / BYTES / K / CLEAN_K / ARCH / commits
#   manifest.txt      ENV-INVARIANT block (devnull pre/post, bin/corpus shas vs
#                     pin, freeze readback) consumed by `fulcrum provenance`
#
# Discipline reused VERBATIM from the parity spine via lib_decide_guest.sh:
# allowlist env scrub, freeze readback (CONCRETE-WRONG never overridable),
# instantaneous procs_running quiet gate, regular-file sinks, sha-verify each run.
#
# Inputs (env): BASE_BIN AFTER_BIN RG CORPUS T CLEAN_T N ARTDIR RUNID
#   GOV NO_TURBO HOST_FROZEN ALLOW_LOAD CORPUS_RAW_SHA256
#   BASE_SHA_PINNED AFTER_SHA_PINNED RG_SHA_PINNED ARCH CROSS_ARCH
#   BASE_COMMIT AFTER_COMMIT CLAIMS_ABSOLUTE
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "$HERE/lib_decide_guest.sh"

: "${BASE_BIN:?}"; : "${AFTER_BIN:?}"; : "${CORPUS:?}"
: "${ARTDIR:?}"; : "${RUNID:?}"
T="${T:-1}"; CLEAN_T="${CLEAN_T:-1}"; N="${N:-13}"
RG="${RG:-rapidgzip}"
GOV="${GOV:-performance}"; NO_TURBO="${NO_TURBO:-1}"
CORPUS_RAW_SHA256="${CORPUS_RAW_SHA256:-}"
BASE_SHA_PINNED="${BASE_SHA_PINNED:-}"; AFTER_SHA_PINNED="${AFTER_SHA_PINNED:-}"
RG_SHA_PINNED="${RG_SHA_PINNED:-}"
ARCH="${ARCH:-unknown}"; CROSS_ARCH="${CROSS_ARCH:-0}"
BASE_COMMIT="${BASE_COMMIT:-}"; AFTER_COMMIT="${AFTER_COMMIT:-}"
CLAIMS_ABSOLUTE="${CLAIMS_ABSOLUTE:-1}"

OUT="$ARTDIR/$RUNID"
mkdir -p "$OUT"
export ARTDIR   # lib primitives write run.stderr under $ARTDIR
MANIFEST="$OUT/manifest.txt"; : > "$MANIFEST"
mf() { echo "$1" >> "$MANIFEST"; }

# ---- contamination guard + box state (spine discipline) ----------------------
scrub_gzippy_env ""
freeze_readback               # ACT_GOV/ACT_TURBO/FREEZE_STATE or die
quiet_gate                    # RUN_AVG/QUIET_STATE or die (ALLOW_LOAD=1 to proceed loaded)

[ -x "$BASE_BIN" ]  || decide_fail "no-base-bin:$BASE_BIN" 5
[ -x "$AFTER_BIN" ] || decide_fail "no-after-bin:$AFTER_BIN" 5
RG_CMD=""
if command -v "$RG" >/dev/null 2>&1; then RG_CMD="$(command -v "$RG")"
elif [ -x "$RG" ]; then RG_CMD="$RG"
else decide_fail "no-rapidgzip:$RG" 12; fi
command -v perf >/dev/null 2>&1 || decide_fail "no-perf (cyc/byte needs perf stat)" 15

[ -f "$CORPUS" ] || decide_fail "no-corpus:$CORPUS" 7
BYTES="$(gzip -dc "$CORPUS" | wc -c)"
REFSHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
[ "$BYTES" -gt 0 ] || decide_fail "corpus-zero-bytes:$CORPUS" 7
if [ -n "$CORPUS_RAW_SHA256" ] && [ "$REFSHA" != "$CORPUS_RAW_SHA256" ]; then
  decide_fail "corpus-sha-drift got=$REFSHA pin=$CORPUS_RAW_SHA256" 7
fi

MASK_T="$(pin_mask "$T")";        [ -n "$MASK_T" ]        || decide_fail "bad-T:$T" 8
MASK_CLEAN="$(pin_mask "$CLEAN_T")"; [ -n "$MASK_CLEAN" ] || decide_fail "bad-CLEAN_T:$CLEAN_T" 8

SINK="$OUT/sink.bin"; assert_regular_sink "$SINK"

# procs_running RIGHT NOW (the per-sample run-queue the optgate quiet gate reads).
procs_running_now() { awk '/^procs_running/{print $2}' /proc/stat 2>/dev/null || echo 0; }

# perf_run <mask> <cmd...> -> echoes "<cycles> <instructions> <procs_running> <sha>"
# Captures cpu_core PMU explicitly (Intel hybrid P-cores; the workload is pinned
# to P-cores so cpu_atom would read <not counted>). Default (non-CSV) perf output
# so fulcrum's cycles parser (`<count> <event>`) consumes it directly.
perf_run() {
  local mask="$1"; shift
  local pr pstat cyc ins sha rc
  pr="$(procs_running_now)"
  pstat="$(mktemp /tmp/.optgate_perf_XXXXXX)"
  assert_regular_sink "$SINK"
  set +e
  perf stat -e cpu_core/cycles/,cpu_core/instructions/ -o "$pstat" \
    -- taskset -c "$mask" "$@" >"$SINK" 2>>"$ARTDIR/run.stderr"
  rc=$?
  set -e 2>/dev/null || true
  sha="$(sha256sum "$SINK" | cut -d' ' -f1)"
  # parse the two cpu_core counts (strip thousands separators).
  cyc="$(awk '/cpu_core\/cycles\//{gsub(/,/,"",$1); print $1; exit}' "$pstat")"
  ins="$(awk '/cpu_core\/instructions\//{gsub(/,/,"",$1); print $1; exit}' "$pstat")"
  rm -f "$pstat"
  [ "$rc" -eq 0 ] || echo "## WARN exit=$rc: $*" >&2
  [ -n "$cyc" ] && [ -n "$ins" ] || { echo "## WARN: perf produced no cpu_core counts for: $*" >&2; cyc=0; ins=0; }
  echo "$cyc $ins $pr $sha"
}

# argv builders
gz_argv()  { GZARGV=("$1" -d -c -p "$2" "$CORPUS"); }       # <bin> <t>
rg_argv()  { RGARGV=("$RG_CMD" -d -c -f -P "$1" "$CORPUS"); } # <t>

S_BASE="$OUT/samples_base.txt";          : > "$S_BASE"
S_AFTER="$OUT/samples_after.txt";        : > "$S_AFTER"
S_RG="$OUT/samples_rg.txt";              : > "$S_RG"
S_CBASE="$OUT/samples_clean_base.txt";   : > "$S_CBASE"
S_CAFTER="$OUT/samples_clean_after.txt"; : > "$S_CAFTER"

DIVERGED=0
emit() { # <samplefile> <arm-is-gz:0|1> <cyc ins pr sha...>
  local file="$1" isgz="$2" cyc="$3" ins="$4" pr="$5" sha="$6"
  echo "$cyc $ins $BYTES $pr $sha" >> "$file"
  if [ "$isgz" = 1 ] && [ "$sha" != "$REFSHA" ]; then
    echo "!! SHA DIVERGENCE arm=$file sha=$sha"; DIVERGED=1
  fi
}

echo "## optgate capture: target T=$T (mask=$MASK_T) clean T=$CLEAN_T (mask=$MASK_CLEAN) N=$N (drop warmup iter0)"
echo "## bytes=$BYTES ref=$REFSHA freeze=$FREEZE_STATE quiet=$QUIET_STATE runnable_avg=${RUN_AVG:-NA}"
for ((i=0;i<=N;i++)); do
  gz_argv "$BASE_BIN"  "$T";     read -r c1 n1 p1 s1 < <(perf_run "$MASK_T" "${GZARGV[@]}")
  gz_argv "$AFTER_BIN" "$T";     read -r c2 n2 p2 s2 < <(perf_run "$MASK_T" "${GZARGV[@]}")
  rg_argv "$T";                  read -r c3 n3 p3 s3 < <(perf_run "$MASK_T" "${RGARGV[@]}")
  gz_argv "$BASE_BIN"  "$CLEAN_T"; read -r c4 n4 p4 s4 < <(perf_run "$MASK_CLEAN" "${GZARGV[@]}")
  gz_argv "$AFTER_BIN" "$CLEAN_T"; read -r c5 n5 p5 s5 < <(perf_run "$MASK_CLEAN" "${GZARGV[@]}")
  [ "$i" -eq 0 ] && continue   # warmup
  emit "$S_BASE"   1 "$c1" "$n1" "$p1" "$s1"
  emit "$S_AFTER"  1 "$c2" "$n2" "$p2" "$s2"
  emit "$S_RG"     0 "$c3" "$n3" "$p3" "$s3"
  emit "$S_CBASE"  1 "$c4" "$n4" "$p4" "$s4"
  emit "$S_CAFTER" 1 "$c5" "$n5" "$p5" "$s5"
done
[ "$DIVERGED" -eq 0 ] || decide_fail "sha-mismatch (wrong-bytes — number VOID)" 11
rm -f "$SINK"

# ---- meta for the artifact assembler -----------------------------------------
{
  echo "REFERENCE_SHA=$REFSHA"
  echo "BYTES=$BYTES"
  echo "K=$T"
  echo "CLEAN_K=$CLEAN_T"
  echo "ARCH=$ARCH"
  echo "CROSS_ARCH=$CROSS_ARCH"
  echo "BASE_COMMIT=$BASE_COMMIT"
  echo "AFTER_COMMIT=$AFTER_COMMIT"
  echo "AFTER_SHA=$(sha256sum "$AFTER_BIN" | cut -d' ' -f1)"
  echo "BASE_SHA=$(sha256sum "$BASE_BIN" | cut -d' ' -f1)"
} > "$OUT/meta.env"

# ---- ENV-INVARIANT manifest (consumed by `fulcrum provenance <art-dir>`) ------
# EVERY value OBSERVED at runtime: sha RE-READ from the actual file, /dev/null
# stat'd live. *_pinned/_expected are the campaign-declared expectations a drift
# is measured against; absent => that aspect degrades to INCOMPLETE, never VOID.
devnull_is_char() { [ -c /dev/null ] && echo 1 || echo 0; }
RG_BIN_SHA="$(sha256sum "$RG_CMD" 2>/dev/null | cut -d' ' -f1)"
BOOST_FILE=/sys/devices/system/cpu/cpufreq/boost
mf "runid=$RUNID"
mf "protocol=fulcrum-v3"
mf "comparator_path=$RG_CMD"
mf "comparator_present=1"
mf "comparator_sink=regular-file"
mf "freeze_state=$FREEZE_STATE"
mf "quiet_state=$QUIET_STATE"
mf "runnable_avg=${RUN_AVG:-NA}"
mf "host_frozen_ack=$HOST_FROZEN"
# the two campaign-corrupting bug classes' guards:
mf "devnull_is_char_pre=$(devnull_is_char)"
mf "gzippy_bin_sha256=$(sha256sum "$AFTER_BIN" | cut -d' ' -f1)"   # RE-READ the AFTER subject
[ -n "$AFTER_SHA_PINNED" ] && mf "gzippy_bin_sha256_pinned=$AFTER_SHA_PINNED"
[ -n "$RG_BIN_SHA" ]      && mf "rapidgzip_bin_sha256=$RG_BIN_SHA"
[ -n "$RG_SHA_PINNED" ]   && mf "rapidgzip_bin_sha256_pinned=$RG_SHA_PINNED"
mf "corpus_sha256=$REFSHA"
[ -n "$CORPUS_RAW_SHA256" ] && mf "corpus_sha256_expected=$CORPUS_RAW_SHA256"
mf "freeze_no_turbo=$ACT_TURBO"
mf "freeze_governor=$ACT_GOV"
[ -r "$BOOST_FILE" ] && mf "freeze_boost=$(cat "$BOOST_FILE")"
mf "claims_absolute=$CLAIMS_ABSOLUTE"
mf "env_host_id=$( (cat /etc/machine-id 2>/dev/null || hostname) | sha256sum | cut -c1-12 )"
mf "devnull_is_char_post=$(devnull_is_char)"
mf "finished=$(date -u '+%FT%TZ')"

echo "OPTGATE_ARTIFACTS=$OUT"
echo "OPTGATE_GUEST_DONE"
