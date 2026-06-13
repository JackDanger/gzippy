#!/usr/bin/env bash
# _oracle_guest.sh — guest-side runner for oracle.sh (P5). The oracle/perturbation
# sibling of _parity_guest.sh.
#
# Unlike _parity_guest.sh (which ABORTS on any seeding/oracle env because it
# measures PRODUCTION), this runner DELIBERATELY sets ONE oracle/perturbation knob
# (per --kind) — but it stays honest, and as of the TOOLING-AUDIT hardening it
# matches the parity spine's contamination bar so a CEILING number (ocl_cf) is as
# trustworthy as the production number:
#   - ENV-SCRUB ALLOWLIST (parity.sh A1): every GZIPPY_* the operator did not
#     explicitly pass is UNSET before measuring, so a leaked seeding/oracle var
#     cannot ride in alongside the one --kind knob (the audit's A2 "NO env-scrub").
#   - CONTENT-FINGERPRINT STALE-BINARY guard (parity.sh A1): the binary must match
#     the synced sources by a sha of ALL build inputs, not mtime — the audit's A2
#     "NO stale-binary fingerprint guard" / "stale-binary measurement" class.
#   - HOST-FREEZE HARD-FAIL (parity.sh A1): a READABLE thawed governor/no_turbo is
#     a hard ABORT (was WARN-only — the audit's A2 "host-freeze is WARN-only"),
#     HOST_FROZEN=1 may rescue ONLY an unreadable (NA) readback.
#   - LOADAVG readback (audit A1 defect 2): a Plex-loaded frozen box prints a
#     high-spread number; we read loadavg back and HARD-FAIL above a threshold so a
#     loaded-box ABSOLUTE number cannot be banked (override with ALLOW_LOAD=1).
#   - IN-SCRIPT FALLBACK==0 ASSERT for engine-isolation (audit A2): the ISA-L
#     engine-oracle ceiling is VOID if any clean chunk fell back to pure-Rust
#     (ISAL_ENGINE_ORACLE_FALLBACKS>0) — that would blend pure-Rust into the
#     "ISA-L ceiling". Read from the GZIPPY_VERBOSE sidecar IN-SCRIPT, not by hand.
#   - it LABELS every non-production run loudly (NOT-PRODUCTION / output-may-be-garbage);
#   - sha-verify is still ENFORCED for kinds that must stay byte-exact (ceiling/
#     same-sink/engine-isolation/perturb); kinds that knowingly produce garbage
#     (clean-only seeded) are marked SHA-NOT-CHECKED so a garbage run can never
#     masquerade as a parity number;
#   - it reuses the SAME host-lock readback, regular-file sink, interleaved best-of-N.
#
# Inputs (env, from oracle.sh): the _parity_guest.sh set PLUS:
#   KIND          clean-only|engine-isolation|same-sink|ceiling|perturb
#   SLOW_KNOB     e.g. GZIPPY_SLOW_DECODE   (for KIND=perturb)
#   SLOW_PCT      e.g. 50                    (for KIND=perturb)
#   HOST_FROZEN   1 to acknowledge an UNREADABLE (NA) host-freeze readback
#   ALLOW_LOAD    1 to acknowledge a loaded box (skips the loadavg hard-fail)
#   MAX_LOADAVG   loadavg-1min threshold (default 2.0) above which we hard-fail
set -u

fail() { echo "ORACLE_FAIL=$1"; echo "ORACLE_GUEST_DONE"; exit "${2:-1}"; }

: "${GUEST_SRC:?}"; : "${GZIPPY_BIN:?}"; : "${CORPUS:?}"; : "${CORPUS_RAW_SHA256:?}"
: "${T:?}"; : "${N:?}"; : "${MASK:?}"; : "${KIND:?}"
RG="${RG:-rapidgzip}"; RG_TRACE="${RG_TRACE:-}"
ARTDIR="${ARTDIR:-/dev/shm/gzippy-oracle-art}"
GOV="${GOV:-performance}"; NO_TURBO="${NO_TURBO:-1}"
SLOW_KNOB="${SLOW_KNOB:-}"; SLOW_PCT="${SLOW_PCT:-}"
HOST_FROZEN="${HOST_FROZEN:-0}"
ALLOW_LOAD="${ALLOW_LOAD:-0}"
MAX_LOADAVG="${MAX_LOADAVG:-2.0}"
CAP="${CAP:-}"

mkdir -p "$ARTDIR"
cd "$GUEST_SRC" || fail "no-guest-src:$GUEST_SRC" 5

# ---- 1. CONTAMINATION GUARD: ALLOWLIST scrub of GZIPPY_* env ------------------
# Mirror _parity_guest.sh: ALLOWLIST every GZIPPY_* this runner sets ITSELF (the
# per-KIND oracle knob is set inside the measured `env` invocation below, NOT
# inherited — so it must NOT be pre-set in the environment either). Anything else
# is UNSET before measuring, so a leaked second oracle/seeding var cannot ride in
# alongside the one --kind knob. This closes the audit's A2 "NO env-scrub" defect.
# The KIND knob names themselves (e.g. GZIPPY_ISAL_ENGINE_ORACLE) must NOT already
# be in the env — oracle.sh passes KIND/SLOW_* as plain config, the knob is applied
# per-command; an inherited engine-oracle var would be a setup error, so we scrub
# (and below, refuse) it like any other.
SCRUBBED=""
for v in $(env | sed -n 's/^\(GZIPPY_[A-Z_0-9]*\)=.*/\1/p'); do
  case "$v" in
    # GZIPPY_BIN is the oracle.sh CONFIG path (where the binary lives), not a
    # gzippy runtime knob; scrubbing it would break the runner's own deref.
    GZIPPY_BIN) ;;
    *) SCRUBBED="$SCRUBBED $v=${!v}"; unset "$v";;
  esac
done
if [ -n "$SCRUBBED" ]; then
  echo "## SCRUBBED inherited GZIPPY_* env before measuring (the --kind knob is applied per-command, NOT inherited):$SCRUBBED"
  # An inherited seeding/oracle/perturbation var is a SETUP ERROR for an oracle run
  # — it means two oracle knobs would be active at once, contaminating the ceiling.
  case "$SCRUBBED" in
    *GZIPPY_SEED*|*GZIPPY_*ORACLE*|*GZIPPY_BYPASS*|*GZIPPY_SLEEP_DECODE*|*GZIPPY_SLOW*)
      fail "contaminated-env (a second seeding/oracle/perturbation var was inherited:$SCRUBBED — an oracle run may carry ONLY the one --kind knob)" 2;;
  esac
fi

# ---- 2. binary present + CONTENT-FINGERPRINT STALE-BINARY guard --------------
# The audit's A2 "NO stale-binary fingerprint guard" fix. oracle.sh does NOT build
# (it measures whatever is at $GZIPPY_BIN, usually built by a prior parity.sh
# --build). parity.sh --build stamps "$GZIPPY_BIN.inputs.sha" with a CONTENT
# fingerprint of ALL build inputs; we re-compute it here and require a match, so a
# stale binary from a different source state cannot be measured. Clock-independent.
[ -x "$GZIPPY_BIN" ] || fail "no-binary:$GZIPPY_BIN (oracle.sh --build, or parity.sh --build first)" 5
FPRINT="$GZIPPY_BIN.inputs.sha"
input_fingerprint() {
  { find src crates examples build.rs Cargo.toml Cargo.lock vendor benches -type f 2>/dev/null \
      | LC_ALL=C sort | xargs sha256sum 2>/dev/null; } | sha256sum | cut -d' ' -f1
}
if [ ! -f "$FPRINT" ]; then
  fail "no-build-fingerprint:$FPRINT absent — cannot prove $GZIPPY_BIN matches the synced sources; build via parity.sh --build (which stamps it)" 6
fi
CUR_FP="$(input_fingerprint)"
STAMP_FP="$(cat "$FPRINT" 2>/dev/null)"
if [ "$CUR_FP" != "$STAMP_FP" ]; then
  fail "stale-binary:$GZIPPY_BIN built from different inputs (stamp=$STAMP_FP cur=$CUR_FP) — rebuild via parity.sh --build" 6
fi

# ---- 3. corpus + correctness oracle -----------------------------------------
[ -f "$CORPUS" ] || fail "no-corpus:$CORPUS" 7
REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
[ "$REF_SHA" = "$CORPUS_RAW_SHA256" ] || fail "corpus-sha-drift got=$REF_SHA pin=$CORPUS_RAW_SHA256" 7
RAW_BYTES="$(gzip -dc "$CORPUS" | wc -c)"

# ---- 4. production-path assert (the engine must still be ParallelSM) ---------
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=' || true)"
case "$DBG" in *ParallelSM*) ;; *) fail "routing not ParallelSM: ${DBG:-none}" 9;; esac

# ---- 5. host-lock READBACK (HARD-FAIL on a readable thaw — parity.sh A1) -----
# Was WARN-only (the audit's A2 "host-freeze is WARN-only" — why ocl_cf drifted
# 0.945×↔0.989× across host states). Now a READABLE thawed governor/no_turbo is a
# hard ABORT; HOST_FROZEN=1 may rescue ONLY an unreadable (NA) readback.
ACT_GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
ACT_TURBO="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA)"
ACT_AFFIN="$(taskset -pc $$ 2>/dev/null | sed 's/.*: //' || echo NA)"
gov_state=MATCH; [ "$ACT_GOV" = "$GOV" ] || gov_state=$([ "$ACT_GOV" = NA ] && echo NA || echo WRONG)
trb_state=MATCH; [ "$ACT_TURBO" = "$NO_TURBO" ] || trb_state=$([ "$ACT_TURBO" = NA ] && echo NA || echo WRONG)
case "$gov_state/$trb_state" in
  MATCH/MATCH) ;;
  *WRONG*)
    fail "host-not-frozen governor=$ACT_GOV no_turbo=$ACT_TURBO (expected $GOV/$NO_TURBO) — a READABLE thawed value cannot be overridden. Freeze the box." 13;;
  *)
    if [ "$HOST_FROZEN" = 1 ]; then
      echo "## WARN: host freeze unreadable (governor=$ACT_GOV no_turbo=$ACT_TURBO) but HOST_FROZEN=1 acknowledged."
    else
      fail "host-freeze-unreadable governor=$ACT_GOV no_turbo=$ACT_TURBO (LXC sysfs hidden?). Pass HOST_FROZEN=1 to acknowledge out-of-band freeze." 13
    fi;;
esac

# ---- 5b. QUIET-BOX readback (instantaneous runnable; audit A1 defect 2) ------
# A frozen box can still be loaded if a neighbor escaped the host freeze (audit:
# Plex 600% CPU manufactured the 36/21 artifact). A loaded box inflates the
# ABSOLUTE wall + spread (31%/22% vs single-digit quiet); the RATIO survives but
# the ABSOLUTE ocl_cf/split numbers do NOT. We gate on INSTANTANEOUS procs_running
# (averaged briefly), NOT the 1-min loadavg: loadavg is a ~60s EMA that carries
# pre-freeze neighbor load for a full minute after the freeze (a FALSE not-quiet
# signal). procs_running (/proc/stat) is the live runnable count (~1-2 quiet).
# Hard-fail above the threshold; ALLOW_LOAD=1 acknowledges a ratio-only run.
QUIET_MAX_RUNNABLE="${QUIET_MAX_RUNNABLE:-2.0}"; QUIET_SAMPLES="${QUIET_SAMPLES:-4}"
LOAD1="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo NA)"
RUN_SUM=0; RUN_CNT=0
for _qs in $(seq 1 "$QUIET_SAMPLES"); do
  _r="$(awk '/^procs_running/{print $2}' /proc/stat 2>/dev/null || echo NA)"
  [ "$_r" = NA ] && break
  RUN_SUM=$((RUN_SUM + _r)); RUN_CNT=$((RUN_CNT + 1))
  [ "$_qs" -lt "$QUIET_SAMPLES" ] && sleep 1
done
if [ "$RUN_CNT" -gt 0 ]; then
  RUN_AVG="$(awk -v s="$RUN_SUM" -v c="$RUN_CNT" 'BEGIN{printf "%.2f", s/c}')"
  run_hot="$(awk -v a="$RUN_AVG" -v m="$QUIET_MAX_RUNNABLE" 'BEGIN{print (a+0>m+0)?1:0}')"
  echo "## quiet-gate: runnable_avg=$RUN_AVG (max=$QUIET_MAX_RUNNABLE) loadavg1=$LOAD1(EMA,context)"
  if [ "$run_hot" = 1 ]; then
    if [ "$ALLOW_LOAD" = 1 ]; then
      echo "## WARN: runnable_avg=$RUN_AVG > $QUIET_MAX_RUNNABLE but ALLOW_LOAD=1 — ABSOLUTE numbers contention-inflated, trust RATIO only."
    else
      fail "host-loaded runnable_avg=$RUN_AVG > $QUIET_MAX_RUNNABLE — a neighbor escaped the freeze; a loaded-box ABSOLUTE number is contention-inflated. Run with --lock (freezes neighbors), wait for quiet, or pass ALLOW_LOAD=1 for ratio-only." 13
    fi
  fi
else
  RUN_AVG=NA
fi

RG_CMD=""
if command -v "$RG" >/dev/null 2>&1; then RG_CMD="$RG"
elif [ -n "$RG_TRACE" ] && [ -x "$RG_TRACE" ]; then RG_CMD="$RG_TRACE"; fi

# ---- per-KIND knob set + verification policy --------------------------------
# GZ_ENV  : extra env prepended to the gzippy command.
# CHECK_SHA: 1 = byte-exact required (abort on mismatch); 0 = garbage-by-design.
# WITH_RG : 1 = also run rapidgzip interleaved (for a relative number).
# ASSERT_NO_FALLBACK: 1 = abort unless ISAL_ENGINE_ORACLE_FALLBACKS==0 (ceiling).
GZ_ENV=""; CHECK_SHA=1; WITH_RG=1; LABEL="$KIND"; PRODUCTION=0; ASSERT_NO_FALLBACK=0
case "$KIND" in
  ceiling)
    : "${CAP:?ceiling needs CAP=<bypass-capture-file>}"
    GZ_ENV="GZIPPY_BYPASS_DECODE=$CAP"; CHECK_SHA=1
    LABEL="ceiling(decode-removed FLOOR; NOT-PRODUCTION but byte-exact)";;
  clean-only)
    GZ_ENV="GZIPPY_SEED_WINDOWS=1"; CHECK_SHA=0
    LABEL="clean-only(SEEDED; masks-binder; SHA-NOT-CHECKED; NOT-PRODUCTION)";;
  engine-isolation)
    # ISA-L engine oracle — C-FFI engine, off the production decode graph. The
    # ceiling is VOID if any clean chunk fell back to pure-Rust (that would blend
    # pure-Rust into the "ISA-L ceiling"); assert fallbacks==0 in-script.
    GZ_ENV="GZIPPY_ISAL_ENGINE_ORACLE=1"; CHECK_SHA=1; ASSERT_NO_FALLBACK=1
    LABEL="engine-isolation(ISA-L engine; NOT-PRODUCTION; byte-exact; fallbacks==0 asserted)";;
  same-sink)
    GZ_ENV=""; CHECK_SHA=1; PRODUCTION=1
    LABEL="same-sink(production knobs; byte-exact control)";;
  perturb)
    : "${SLOW_KNOB:?perturb needs SLOW_KNOB=GZIPPY_SLOW_*}"; : "${SLOW_PCT:?perturb needs SLOW_PCT=N}"
    GZ_ENV="GZIPPY_SLOW_MODE=$SLOW_PCT $SLOW_KNOB=1 GZIPPY_SLOW_HITS=1"; CHECK_SHA=1
    LABEL="perturb($SLOW_KNOB +$SLOW_PCT%; byte-exact; slope NOT a ceiling)";;
  *) fail "unknown KIND=$KIND" 2;;
esac

BIN_SHA="$(sha256sum "$GZIPPY_BIN" | cut -c1-16)"
echo "================ ORACLE PROVENANCE ================"
echo "kind=$KIND  $LABEL"
echo "guest_src=$GUEST_SRC head=$(git rev-parse --short HEAD 2>/dev/null || echo NA) T=$T N=$N mask=$MASK"
echo "binary=$GZIPPY_BIN bin_sha=$BIN_SHA inputs_fp=$CUR_FP(==stamp)  <- stale-binary guard PASSED"
echo "corpus=$CORPUS ref_sha=$REF_SHA raw_bytes=$RAW_BYTES"
echo "gz_env='$GZ_ENV'  check_sha=$CHECK_SHA production=$PRODUCTION assert_no_fallback=$ASSERT_NO_FALLBACK"
echo "governor=$ACT_GOV no_turbo=$ACT_TURBO affinity=$ACT_AFFIN runnable_avg=${RUN_AVG:-NA} loadavg1=$LOAD1 host_frozen=$HOST_FROZEN"
[ "$PRODUCTION" = 1 ] || echo "## NOTE: this is an ORACLE/PERTURBATION run — NOT a production parity number."
echo "=================================================="

# ---- in-script fallback==0 coverage assert (engine-isolation; audit A2) ------
# Run ONE GZIPPY_VERBOSE pass with the oracle knob to read the coverage counters
# IN-SCRIPT (was hand-checked out-of-band). A clean tail that silently fell back to
# pure-Rust would sha-pass while contaminating the ISA-L ceiling — refuse it here.
if [ "$ASSERT_NO_FALLBACK" = 1 ]; then
  COVLOG="$ARTDIR/coverage.txt"
  # shellcheck disable=SC2086
  env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_VERBOSE=1 $GZ_ENV \
    taskset -c "$MASK" "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS" >/dev/null 2>"$COVLOG" || true
  # The binary's GZIPPY_VERBOSE line is "... isal_chunks=N isal_fallbacks=M"
  # (chunk_fetcher.rs:870-874). Match that exact label (the old grep looked for
  # isal_oracle_chunks=, a label that never existed in the output -> false
  # coverage-unreadable abort).
  COVLINE="$(grep -m1 'isal_chunks=' "$COVLOG" || true)"
  OC_CHUNKS="$(printf '%s' "$COVLINE" | sed -n 's/.*isal_chunks=\([0-9]*\).*/\1/p')"
  OC_FALLBACKS="$(printf '%s' "$COVLINE" | sed -n 's/.*isal_fallbacks=\([0-9]*\).*/\1/p')"
  echo "## coverage: isal_chunks=${OC_CHUNKS:-NA} isal_fallbacks=${OC_FALLBACKS:-NA}  (from $COVLINE)"
  [ -n "$OC_CHUNKS" ] || fail "coverage-unreadable: no isal_chunks= line in GZIPPY_VERBOSE — cannot certify the ceiling (engine oracle did not run?)" 15
  [ "${OC_CHUNKS:-0}" -gt 0 ] || fail "coverage-zero: isal_oracle_chunks=0 — the ISA-L engine never ran; ceiling MEANINGLESS" 15
  [ "${OC_FALLBACKS:-1}" -eq 0 ] || fail "coverage-contaminated: isal_oracle_fallbacks=$OC_FALLBACKS>0 — pure-Rust blended into the ISA-L ceiling (Rule 4); ceiling VOID. Widen the oracle reserve bound or fix the uncovered chunk." 15
fi

SINK_GZ="$ARTDIR/sink_gz.bin"; SINK_RG="$ARTDIR/sink_rg.bin"
# Defend the sink path (same pipe-phantom risk as parity.sh): drop any planted
# FIFO/symlink, assert a plain regular file.
for s in "$SINK_GZ" "$SINK_RG"; do
  rm -f "$s" 2>/dev/null || true; : > "$s" || fail "cannot-create-sink:$s" 14
  [ -f "$s" ] && [ ! -L "$s" ] && [ ! -p "$s" ] || fail "sink-not-regular-file:$s (symlink/FIFO)" 14
done
timed() { local sink="$1"; shift; local s e secs sha rc
  s=$(date +%s.%N); set +e; taskset -c "$MASK" "$@" >"$sink" 2>>"$ARTDIR/run.stderr"; rc=$?; set -e 2>/dev/null||true; e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}'); sha=$(sha256sum "$sink" | cut -d' ' -f1)
  [ "$rc" -eq 0 ] || echo "## WARN exit=$rc: $*" >&2; echo "$secs $sha"; }

GZT=""; RGT=""; DIVERGED=0
for ((i=0;i<=N;i++)); do
  # shellcheck disable=SC2086
  read gsec gsha < <(timed "$SINK_GZ" env GZIPPY_FORCE_PARALLEL_SM=1 $GZ_ENV "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS")
  if [ "$WITH_RG" = 1 ] && [ -n "$RG_CMD" ]; then
    read rsec _ < <(timed "$SINK_RG" "$RG_CMD" -d -c -f -P "$T" "$CORPUS")
  else rsec=""; fi
  [ "$i" -eq 0 ] && continue
  GZT="$GZT $gsec"; [ -n "$rsec" ] && RGT="$RGT $rsec"
  if [ "$CHECK_SHA" = 1 ] && [ "$gsha" != "$REF_SHA" ]; then echo "!! SHA DIVERGENCE gzippy i=$i sha=$gsha"; DIVERGED=1; fi
done
rm -f "$SINK_GZ" "$SINK_RG"

if [ "$CHECK_SHA" = 1 ] && [ "$DIVERGED" -ne 0 ]; then
  fail "sha-mismatch on a byte-exact KIND ($KIND) — oracle VOID (Rule 4)" 11
fi

stats() { echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '
  { v[NR]=$1 } END { n=NR; if(n==0){print "0 0 0"; exit}
    min=v[1]; max=v[n]; mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
    printf "%.4f %.4f %.0f", min, mid, (min>0)?(max-min)/min*100:0 }'; }
mbps() { awk -v r="$RAW_BYTES" -v t="$1" 'BEGIN{printf "%.0f", (t>0)?r/t/1e6:0}'; }

read gmin gmed gsp < <(stats "$GZT")
echo ""
echo "================ ORACLE SUMMARY (kind=$KIND T=$T) ================"
printf "gzippy[%s]  min=%.4fs (%s MB/s) med=%.4f spread=%s%%  sha=%s\n" \
  "$KIND" "$gmin" "$(mbps "$gmin")" "$gmed" "$gsp" \
  "$([ "$CHECK_SHA" = 1 ] && echo OK || echo NOT-CHECKED)"
if [ -n "$RGT" ]; then
  read rmin rmed rsp < <(stats "$RGT")
  RATIO="$(awk -v g="$gmin" -v r="$rmin" 'BEGIN{printf "%.3f", (g>0)?r/g:0}')"
  printf "rapidgzip    min=%.4fs (%s MB/s) med=%.4f spread=%s%%\n" "$rmin" "$(mbps "$rmin")" "$rmed" "$rsp"
  printf "gzippy[%s]=%sms  rg=%sms  ratio=%s\n" "$KIND" \
    "$(awk -v t="$gmin" 'BEGIN{printf "%.0f", t*1000}')" \
    "$(awk -v t="$rmin" 'BEGIN{printf "%.0f", t*1000}')" "$RATIO"
fi
[ "$PRODUCTION" = 1 ] || echo "## REMINDER: oracle/perturbation number — do NOT bank as parity. See CLAUDE.md Measurement PROCESS."
echo "================ END ORACLE SUMMARY ================"
echo "ORACLE_GUEST_DONE"
