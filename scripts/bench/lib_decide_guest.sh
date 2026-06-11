# lib_decide_guest.sh — guest-side primitives for _decide_guest.sh.
#
# POLICY (plans/fulcrum2-charter.md "extend, don't fork"): every function here is
# copied VERBATIM (modulo function-parameterization noted inline) from the
# hash-pinned parity spine `_parity_guest.sh`, with a provenance marker. The spine
# itself is NOT modified (its instrument-registry row stays VALIDATED). Any future
# change to the spine's primitives MUST be mirrored here or the decide row in
# plans/instrument-registry.md reverts to UNVALIDATED.
#
# Sourced with: GOV, NO_TURBO, HOST_FROZEN, ALLOW_LOAD, ARTDIR set by the caller.

# VERBATIM-FROM _parity_guest.sh:24 (fail token renamed for the decide runner)
decide_fail() { echo "DECIDE_FAIL=$1"; echo "DECIDE_GUEST_DONE"; exit "${2:-1}"; }

# VERBATIM-FROM parity.sh:94-101 (pin_mask — the canonical mask convention;
# free-placement numbers lie, so every run goes through this).
pin_mask() {
  case "$1" in
    1) echo "0";; 4) echo "0,2,4,6";;
    8) echo "0,2,4,6,8,10,12,14";;
    16) echo "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15";;
    *) echo "";;
  esac
}

# VERBATIM-FROM _parity_guest.sh:44-71 (allowlist env scrub), parameterized:
# $1 = space-separated EXTRA allowlisted vars (the knob under test + capture vars).
# The hard-fail cases (seed/oracle/bypass/slow vars present in the inherited env)
# are IDENTICAL to the spine's.
scrub_gzippy_env() {
  local extra_allow="$1" SCRUBBED="" v keep
  for v in $(env | sed -n 's/^\(GZIPPY_[A-Z_0-9]*\)=.*/\1/p'); do
    keep=0
    case "$v" in
      GZIPPY_FORCE_PARALLEL_SM|GZIPPY_DEBUG|GZIPPY_BIN|GZIPPY_BIN2) keep=1;;
    esac
    case " $extra_allow " in *" $v "*) keep=1;; esac
    if [ "$keep" = 0 ]; then SCRUBBED="$SCRUBBED $v=${!v}"; unset "$v"; fi
  done
  if [ -n "$SCRUBBED" ]; then
    echo "## SCRUBBED non-production GZIPPY_* env before measuring:$SCRUBBED"
    case "$SCRUBBED" in
      *GZIPPY_SEED*|*GZIPPY_*ORACLE*|*GZIPPY_BYPASS*|*GZIPPY_SLEEP_DECODE*|*GZIPPY_SLOW*)
        decide_fail "contaminated-env (seeding/oracle var present:$SCRUBBED — not a production run)" 2;;
    esac
  fi
}

# VERBATIM-FROM _parity_guest.sh:276-281 (regular-file sink defense — a FIFO with
# a draining reader is the writev phantom).
assert_regular_sink() { # <path>
  local p="$1"
  rm -f "$p" 2>/dev/null || true
  : > "$p" || decide_fail "cannot-create-sink:$p" 14
  [ -f "$p" ] && [ ! -L "$p" ] && [ ! -p "$p" ] || decide_fail "sink-not-regular-file:$p (symlink/FIFO — pipe-phantom risk)" 14
}

# EXTENDED-FROM _parity_guest.sh:289-301 (timed run -> "secs sha rss_mb").
# Extension over the spine: wraps the command in /usr/bin/time -f '%M' -o <tmp>
# to capture peak RSS (kilobytes, converted to MB) WITHOUT mixing it into the
# command's own stderr (the -o flag writes the time stats to a file, not stderr).
# CALLERS MUST read three fields (`read -r sec sha rss`): bash `read`'s last
# variable slurps the REMAINDER of the line, so a two-var read corrupts sha
# with the appended rss (caught live: false SHA DIVERGENCE on a correct pin).
# Stale claim removed: it is NOT safe to ignore the
# third field. Knob callers read the third field as rss_mb for meta.txt rendering.
timed_masked() { # <mask> <sink> <cmd...> -> echoes "secs sha rss_mb"
  local mask="$1" sink="$2"; shift 2
  local s e secs sha rc rss_mb=0 _tfile
  _tfile=$(mktemp /tmp/.decide_rss_XXXXXX)
  s=$(date +%s.%N)
  set +e
  /usr/bin/time -f '%M' -o "$_tfile" taskset -c "$mask" "$@" >"$sink" 2>>"$ARTDIR/run.stderr"
  rc=$?
  set -e 2>/dev/null || true
  e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}')
  sha=$(sha256sum "$sink" | cut -d' ' -f1)
  rss_mb=$(awk 'NR==1 && $1~/^[0-9]+$/{printf "%.0f", $1/1024}' "$_tfile" 2>/dev/null || echo 0)
  rm -f "$_tfile"
  [ "$rc" -eq 0 ] || echo "## WARN exit=$rc: $*" >&2
  echo "$secs $sha $rss_mb"
}

# VERBATIM-FROM _parity_guest.sh:326-332 (min/med/spread% over a sample string).
stats() { # echoes "min med spreadpct"
  echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '
    { v[NR]=$1 } END {
      n=NR; if(n==0){print "0 0 0"; exit}
      min=v[1]; max=v[n]; mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
      printf "%.4f %.4f %.0f", min, mid, (min>0)?(max-min)/min*100:0 }'
}

# VERBATIM-FROM _parity_guest.sh:180-203 (host-freeze readback; CONCRETE-WRONG can
# never be overridden, NA only with HOST_FROZEN=1 ack). Sets ACT_GOV/ACT_TURBO/
# ACT_AFFIN globals for the manifest.
freeze_readback() {
  HOST_FROZEN="${HOST_FROZEN:-0}"
  ACT_GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
  ACT_TURBO="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA)"
  ACT_AFFIN="$(taskset -pc $$ 2>/dev/null | sed 's/.*: //' || echo NA)"
  local gov_state trb_state
  gov_state=MATCH; [ "$ACT_GOV" = "$GOV" ] || gov_state=$([ "$ACT_GOV" = NA ] && echo NA || echo WRONG)
  trb_state=MATCH; [ "$ACT_TURBO" = "$NO_TURBO" ] || trb_state=$([ "$ACT_TURBO" = NA ] && echo NA || echo WRONG)
  case "$gov_state/$trb_state" in
    MATCH/MATCH) FREEZE_STATE=frozen;;
    *WRONG*)
      decide_fail "host-not-frozen governor=$ACT_GOV no_turbo=$ACT_TURBO (expected $GOV/$NO_TURBO) — a READABLE thawed value cannot be overridden. Freeze the box." 13;;
    *)
      if [ "$HOST_FROZEN" = 1 ]; then
        echo "## WARN: host freeze unreadable (governor=$ACT_GOV no_turbo=$ACT_TURBO) but HOST_FROZEN=1 acknowledged — proceeding (auditable in provenance)."
        FREEZE_STATE=acknowledged
      else
        decide_fail "host-freeze-unreadable governor=$ACT_GOV no_turbo=$ACT_TURBO (LXC sysfs hidden?). Pass HOST_FROZEN=1 to acknowledge the box is frozen out-of-band. A thawed-host number is contaminated." 13
      fi;;
  esac
}

# VERBATIM-FROM _parity_guest.sh:214-238 (instantaneous procs_running quiet gate;
# loadavg is a ~60s EMA that lies for a minute after the freeze). Sets RUN_AVG.
quiet_gate() {
  ALLOW_LOAD="${ALLOW_LOAD:-0}"
  QUIET_MAX_RUNNABLE="${QUIET_MAX_RUNNABLE:-2.0}"; QUIET_SAMPLES="${QUIET_SAMPLES:-4}"
  LOAD1="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo NA)"
  local RUN_SUM=0 RUN_CNT=0 _qs _r run_hot
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
        echo "## WARN: runnable_avg=$RUN_AVG > $QUIET_MAX_RUNNABLE but ALLOW_LOAD=1 — ABSOLUTE numbers contention-inflated; trust RATIO only."
        QUIET_STATE=loaded-acked
      else
        decide_fail "host-loaded runnable_avg=$RUN_AVG > $QUIET_MAX_RUNNABLE — a neighbor escaped the freeze; a loaded-box ABSOLUTE number is contention-inflated." 13
      fi
    else
      QUIET_STATE=quiet
    fi
  else
    RUN_AVG=NA; QUIET_STATE=unreadable
    echo "## quiet-gate: procs_running unreadable; falling back to loadavg1=$LOAD1 (context only)"
  fi
}
