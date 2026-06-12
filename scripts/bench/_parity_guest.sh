#!/usr/bin/env bash
# _parity_guest.sh — the GUEST-SIDE half of parity.sh (P0). Not run directly by
# the owner; parity.sh ships it to $GUEST_SRC and executes it over the double-hop.
#
# It is self-contained and correctness-critical. It enforces, MECHANICALLY:
#   - build in $GUEST_SRC with the pinned RUSTFLAGS + selected feature (Rule 6);
#   - production-path assertion (GZIPPY_DEBUG -> path=ParallelSM) (Rule 6);
#   - host-lock READBACK warn (governor/no_turbo/taskset) (Rule 6);
#   - REGULAR-FILE sink, NEVER a pipe (a pipe backpressure-inflated writev into a
#     phantom — the contamination this whole wrapper exists to prevent);
#   - WINDOW-ABSENT-PRESERVING: it sets NO GZIPPY_SEED_* and NO engine oracle, and
#     it ABORTS if any seeding env leaked in (a seeded run routes to the clean
#     engine and masks the binder — it is NOT production);
#   - interleaved best-of-N gzippy vs rapidgzip (Rule 6);
#   - MANDATORY sha-verify of EVERY run against the decompressed-corpus pin, with
#     a loud ABORT on any mismatch (Rule 4 — a fast wrong-bytes win is a loss).
#
# Inputs (env, all required, passed by parity.sh from guest.env + flags):
#   GUEST_SRC GZIPPY_BIN CORPUS CORPUS_RAW_SHA256 RG RG_TRACE
#   FEATURE T N MASK GOV NO_TURBO RUSTFLAGS_PIN DO_BUILD DO_FULCRUM ARTDIR
#   CARGO_LOCK (path to scripts/cargo-lock.sh, relative to GUEST_SRC)
set -u

fail() { echo "PARITY_FAIL=$1"; echo "PARITY_GUEST_DONE"; exit "${2:-1}"; }

: "${GUEST_SRC:?}"; : "${GZIPPY_BIN:?}"; : "${CORPUS:?}"; : "${CORPUS_RAW_SHA256:?}"
: "${FEATURE:?}"; : "${T:?}"; : "${N:?}"; : "${MASK:?}"
# Optional second binary for three-way interleave (rg vs gz1 vs gz2).
# When unset/empty, behavior is byte-identical to the two-way path.
GZIPPY_BIN2="${GZIPPY_BIN2:-}"
FEATURE2="${FEATURE2:-}"
RG="${RG:-rapidgzip}"
RG_TRACE="${RG_TRACE:-}"
RUSTFLAGS_PIN="${RUSTFLAGS_PIN:--C target-cpu=native}"
DO_BUILD="${DO_BUILD:-0}"
DO_FULCRUM="${DO_FULCRUM:-0}"
ARTDIR="${ARTDIR:-/dev/shm/gzippy-parity-art}"
GOV="${GOV:-performance}"
NO_TURBO="${NO_TURBO:-1}"

mkdir -p "$ARTDIR"
cd "$GUEST_SRC" || fail "no-guest-src:$GUEST_SRC" 5

# ---- 1. CONTAMINATION GUARD: ALLOWLIST scrub of GZIPPY_* env ------------------
# A denylist of known seeding/oracle vars can be defeated by a renamed/new oracle
# or anything inherited via the shell profile / ssh SendEnv-AcceptEnv. So we
# ALLOWLIST: any GZIPPY_* the operator did not explicitly intend is UNSET before
# we measure. Only GZIPPY_FORCE_PARALLEL_SM and GZIPPY_DEBUG (set per-command
# below) survive into the measured process. This makes a seeded/binder-masked run
# structurally impossible, not merely deny-checked. We still LOG what we scrub so
# a leaked var is visible, then we report any seeding/oracle leak as a hard fail
# (it should never have been in the environment of a production parity run).
SCRUBBED=""
for v in $(env | sed -n 's/^\(GZIPPY_[A-Z_0-9]*\)=.*/\1/p'); do
  case "$v" in
    # The two runtime knobs we set ourselves, PLUS GZIPPY_BIN / GZIPPY_BIN2 — the
    # parity.sh CONFIG paths (where the binaries live), NOT gzippy runtime knobs.
    # Scrubbing them would unset variables the runner itself dereferences (set -u).
    GZIPPY_FORCE_PARALLEL_SM|GZIPPY_DEBUG|GZIPPY_BIN|GZIPPY_BIN2) ;;
    *) SCRUBBED="$SCRUBBED $v=${!v}"; unset "$v";;
  esac
done
if [ -n "$SCRUBBED" ]; then
  echo "## SCRUBBED non-production GZIPPY_* env before measuring:$SCRUBBED"
  # If any scrubbed var is a known binder-masker, refuse outright: its mere
  # presence means this invocation was set up as an oracle, not production.
  case "$SCRUBBED" in
    *GZIPPY_SEED*|*GZIPPY_*ORACLE*|*GZIPPY_BYPASS*|*GZIPPY_SLEEP_DECODE*|*GZIPPY_SLOW*)
      fail "contaminated-env (seeding/oracle var present:$SCRUBBED — not a production parity run)" 2;;
  esac
fi

# ---- 2. build (under cargo-lock) + content fingerprint -----------------------
# The stale-binary guard uses a CONTENT fingerprint, not mtime: mtime comparison
# was both incomplete (missed vendor/benches) and unsound across hosts (the binary
# carries the guest clock, sources the owner clock — rsync -a preserves owner
# mtimes, so a clock-skew false-negative could measure a stale binary). A sha of
# ALL build inputs is clock-independent and complete.
FPRINT="$GZIPPY_BIN.inputs.sha"
# The complete set of build inputs (everything rsync ships that can change the bin).
input_fingerprint() {
  # Deterministic: sorted file list, sha each, sha the digest stream. Quiet on
  # missing optional dirs.
  { find src crates examples build.rs Cargo.toml Cargo.lock vendor benches -type f 2>/dev/null \
      | LC_ALL=C sort | xargs sha256sum 2>/dev/null; } | sha256sum | cut -d' ' -f1
}

CARGO_LOCK="${CARGO_LOCK:-scripts/cargo-lock.sh}"
if [ "$DO_BUILD" = 1 ]; then
  echo "## build feature=$FEATURE RUSTFLAGS='$RUSTFLAGS_PIN' (cargo-lock serialized)"
  if [ -x "$CARGO_LOCK" ]; then BUILDER=(sh "$CARGO_LOCK"); else BUILDER=(); fi
  set +e
  RUSTFLAGS="$RUSTFLAGS_PIN" "${BUILDER[@]}" \
    cargo build --release --no-default-features --features "$FEATURE" \
    >"$ARTDIR/build.log" 2>&1
  brc=$?
  set -e 2>/dev/null || true
  if [ "$brc" -ne 0 ]; then
    grep -E 'error' "$ARTDIR/build.log" | head -30
    fail "build rc=$brc (see $ARTDIR/build.log)" 8
  fi
  grep -E 'Compiling gzippy|Finished' "$ARTDIR/build.log" | tail -3 || true
  # Stamp the fingerprint of the inputs THIS binary was built from.
  input_fingerprint > "$FPRINT" 2>/dev/null || true
fi
[ -x "$GZIPPY_BIN" ] || fail "no-binary:$GZIPPY_BIN (run with --build)" 5

# ---- 2b. STALE-BINARY guard (only matters when we did NOT just build) --------
# Without --build we measure whatever binary is at $GZIPPY_BIN. Compare the CURRENT
# build-input fingerprint to the one stamped at build time. Mismatch ⇒ the synced
# sources differ from what the binary was built from ⇒ stale ⇒ abort. Absent stamp
# ⇒ we cannot prove freshness ⇒ abort (require --build). Clock-independent; covers
# src/build.rs/Cargo.*/vendor/benches.
if [ "$DO_BUILD" != 1 ]; then
  if [ ! -f "$FPRINT" ]; then
    fail "no-build-fingerprint:$FPRINT absent — cannot prove $GZIPPY_BIN matches the synced sources; re-run with --build" 6
  fi
  CUR_FP="$(input_fingerprint)"
  STAMP_FP="$(cat "$FPRINT" 2>/dev/null)"
  if [ "$CUR_FP" != "$STAMP_FP" ]; then
    fail "stale-binary:$GZIPPY_BIN built from different inputs (stamp=$STAMP_FP cur=$CUR_FP) — re-run with --build" 6
  fi
fi

# ---- 2c. GZIPPY_BIN2 existence + stale-binary guard (only when BIN2 set) ----
if [ -n "$GZIPPY_BIN2" ]; then
  [ -x "$GZIPPY_BIN2" ] || fail "no-binary-2:$GZIPPY_BIN2 (not executable — ensure --bin2 points to a built binary)" 5
  FPRINT2="$GZIPPY_BIN2.inputs.sha"
  if [ "$DO_BUILD" != 1 ]; then
    if [ ! -f "$FPRINT2" ]; then
      fail "no-build-fingerprint-2:$FPRINT2 absent — cannot prove $GZIPPY_BIN2 matches the synced sources; create the stamp or re-run with --build" 6
    fi
    CUR_FP2="$(input_fingerprint)"
    STAMP_FP2="$(cat "$FPRINT2" 2>/dev/null)"
    if [ "$CUR_FP2" != "$STAMP_FP2" ]; then
      fail "stale-binary-2:$GZIPPY_BIN2 built from different inputs (stamp=$STAMP_FP2 cur=$CUR_FP2) — re-run with --build" 6
    fi
  fi
fi

# ---- 3. corpus + correctness oracle -----------------------------------------
[ -f "$CORPUS" ] || fail "no-corpus:$CORPUS" 7
REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
if [ "$REF_SHA" != "$CORPUS_RAW_SHA256" ]; then
  fail "corpus-sha-drift got=$REF_SHA pin=$CORPUS_RAW_SHA256 (corpus differs from banked)" 7
fi
RAW_BYTES="$(gzip -dc "$CORPUS" | wc -c)"

# ---- 4. production-path assertion (Rule 6) -----------------------------------
# EXPECT_PATH (default ParallelSM) is the production DecodePath this cell must
# take. On the gzippy-isal build, single-threaded single-member decode routes to
# IsalSingleShot (one ISA-L call — DIS-15), NOT ParallelSM, so the T1-isal cell
# is measured with EXPECT_PATH=IsalSingleShot. NOTE: GZIPPY_FORCE_PARALLEL_SM is
# a DEAD no-op in the binary (zero source refs; morning-brief-confirmed), so it
# neither forces nor masks routing — the measured run below follows production
# classify_gzip exactly. We keep setting it only for provenance continuity.
EXPECT_PATH="${EXPECT_PATH:-ParallelSM}"
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=' || true)"
case "$DBG" in
  *path=$EXPECT_PATH*) ;;
  *) fail "routing not $EXPECT_PATH (got: ${DBG:-<no path= line>})" 9;;
esac

# ---- 4b. production-path assertion for GZIPPY_BIN2 (only when set) ----------
if [ -n "$GZIPPY_BIN2" ]; then
  DBG2="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN2" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=' || true)"
  case "$DBG2" in
    *path=$EXPECT_PATH*) ;;
    *) fail "routing-2 not $EXPECT_PATH (got: ${DBG2:-<no path= line>})" 9;;
  esac
fi

# ---- 4c. BUILD_FLAVOR guard — hard fail before timing if binary reports wrong flavor ----
# Derive the expected flavor from $FEATURE (sane default) unless overridden by
# $EXPECTED_FLAVOR. This catches the "measured the wrong binary" footgun where a
# gzippy-native binary is measured with --feature gzippy-isal (or vice versa).
# Flavors (from build.rs):
#   gzippy-isal        → "parallel-sm+isal"
#   gzippy-native / pure-rust-inflate → "parallel-sm+pure"
#   default / other    → "legacy-serial"
if [ -z "${EXPECTED_FLAVOR:-}" ]; then
  case "$FEATURE" in
    gzippy-isal)                      EXPECTED_FLAVOR="parallel-sm+isal";;
    gzippy-native|pure-rust-inflate)  EXPECTED_FLAVOR="parallel-sm+pure";;
    *)                                EXPECTED_FLAVOR="legacy-serial";;
  esac
fi
ACTUAL_FLAVOR="$("$GZIPPY_BIN" --version 2>/dev/null | grep -oE '\([^)]+\)' | tr -d '()' || true)"
case "$ACTUAL_FLAVOR" in
  "$EXPECTED_FLAVOR") ;;
  *) fail "build-flavor mismatch: expected '$EXPECTED_FLAVOR' from feature=$FEATURE, binary reports '${ACTUAL_FLAVOR:-<no flavor>}' — rebuild with correct --features or set EXPECTED_FLAVOR" 10;;
esac
echo "## build-flavor=$ACTUAL_FLAVOR (matches expected=$EXPECTED_FLAVOR)"

# ---- 5. host-lock READBACK (ABORT on thaw — Rule 6 frozen host) --------------
# A thawed host must NOT silently print a number. Mismatch (or an NA readback on an
# LXC where the sysfs is hidden) is a HARD FAIL unless the operator explicitly
# acknowledges with HOST_FROZEN=1 (e.g. the host-lock harness on the jump host
# froze the box but the guest cannot read it back). With HOST_FROZEN=1 we still
# print what we read so the acknowledgement is auditable. We do NOT change the
# governor here (that is a privileged write owned by the host-lock harness).
HOST_FROZEN="${HOST_FROZEN:-0}"
ACT_GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
ACT_TURBO="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA)"
# Read back effective affinity too (offline cores would already sha-miss, but an
# unhonored mask is worth surfacing).
ACT_AFFIN="$(taskset -pc $$ 2>/dev/null | sed 's/.*: //' || echo NA)"
# A readback is one of: MATCH (frozen), NA (unreadable — e.g. LXC sysfs hidden), or
# a CONCRETE-WRONG value (genuinely thawed, readable). HOST_FROZEN=1 may ONLY
# rescue the NA case (nothing to verify against) — it must NOT be able to override a
# CONCRETE-WRONG readback (a box we can see is thawed). That closes the "operator
# asserts frozen on an actually-thawed readable host" residual.
gov_state=MATCH; [ "$ACT_GOV" = "$GOV" ] || gov_state=$([ "$ACT_GOV" = NA ] && echo NA || echo WRONG)
trb_state=MATCH; [ "$ACT_TURBO" = "$NO_TURBO" ] || trb_state=$([ "$ACT_TURBO" = NA ] && echo NA || echo WRONG)
case "$gov_state/$trb_state" in
  MATCH/MATCH) ;;  # frozen, proceed
  *WRONG*)
    fail "host-not-frozen governor=$ACT_GOV no_turbo=$ACT_TURBO (expected $GOV/$NO_TURBO) — a READABLE thawed value cannot be overridden. Freeze the box." 13;;
  *) # one or both are NA (unreadable) and the rest MATCH — allow ONLY with ack.
    if [ "$HOST_FROZEN" = 1 ]; then
      echo "## WARN: host freeze unreadable (governor=$ACT_GOV no_turbo=$ACT_TURBO) but HOST_FROZEN=1 acknowledged — proceeding (auditable in provenance)."
    else
      fail "host-freeze-unreadable governor=$ACT_GOV no_turbo=$ACT_TURBO (LXC sysfs hidden?). Pass HOST_FROZEN=1 to acknowledge the box is frozen out-of-band. A thawed-host number is contaminated." 13
    fi;;
esac

# ---- 5b. QUIET-BOX readback (instantaneous runnable; mirrors _oracle_guest.sh) -
# A freq-frozen box can STILL be loaded if a noisy neighbor escaped the host
# freeze (the failure this consolidation fixes: Plex/transmission ran free and
# bounced loadavg). We gate on INSTANTANEOUS procs_running (averaged briefly), NOT
# the 1-min loadavg — loadavg is a ~60s EMA that carries pre-freeze neighbor load
# for a full minute after the freeze, a FALSE not-quiet signal. procs_running
# (/proc/stat, visible to the LXC) is the live runnable count: ~1-2 on a quiet
# frozen box. Hard-fail above the threshold so a loaded-box absolute is never
# banked. ALLOW_LOAD=1 acknowledges a deliberately-loaded ratio-only run.
ALLOW_LOAD="${ALLOW_LOAD:-0}"; MAX_LOADAVG="${MAX_LOADAVG:-2.0}"
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
      echo "## WARN: runnable_avg=$RUN_AVG > $QUIET_MAX_RUNNABLE but ALLOW_LOAD=1 — ABSOLUTE numbers contention-inflated; trust RATIO only."
    else
      fail "host-loaded runnable_avg=$RUN_AVG > $QUIET_MAX_RUNNABLE — a neighbor escaped the freeze; a loaded-box ABSOLUTE number is contention-inflated. Run with --lock (freezes neighbors), wait for quiet, or pass ALLOW_LOAD=1 for ratio-only." 13
    fi
  fi
else
  RUN_AVG=NA
  echo "## quiet-gate: procs_running unreadable; falling back to loadavg1=$LOAD1 (context only)"
fi

# rapidgzip presence (the comparison target).
RG_CMD=""
if command -v "$RG" >/dev/null 2>&1; then RG_CMD="$RG"
elif [ -n "$RG_TRACE" ] && [ -x "$RG_TRACE" ]; then RG_CMD="$RG_TRACE"
else fail "no-rapidgzip (not in PATH and RG_TRACE absent) — cannot measure parity" 12; fi

# NOTE: parity.sh rsyncs with --exclude '.git/', so a guest-side `git rev-parse`
# would read a STALE clone hash, not the synced working tree — it does NOT describe
# the measured bytes. We therefore report the BINARY's mtime+sha as the load-bearing
# provenance, and label the git line as unreliable.
GIT_HEAD="$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
BIN_SHA="$(sha256sum "$GZIPPY_BIN" | cut -c1-16)"
echo "================ PARITY PROVENANCE ================"
echo "guest_src=$GUEST_SRC git_head=$GIT_HEAD(UNRELIABLE: .git excluded from sync) feature=$FEATURE T=$T N=$N mask=$MASK"
echo "binary=$GZIPPY_BIN bin_sha=$BIN_SHA mtime=$(date -r "$GZIPPY_BIN" '+%F %T' 2>/dev/null || echo NA)  <- the load-bearing build identity"
if [ -n "$GZIPPY_BIN2" ]; then
  BIN2_SHA="$(sha256sum "$GZIPPY_BIN2" | cut -c1-16)"
  echo "binary2=$GZIPPY_BIN2 feature2=$FEATURE2 bin2_sha=$BIN2_SHA mtime=$(date -r "$GZIPPY_BIN2" '+%F %T' 2>/dev/null || echo NA)  <- second binary identity"
fi
echo "corpus=$CORPUS ref_sha=$REF_SHA raw_bytes=$RAW_BYTES"
echo "rapidgzip=$("$RG_CMD" --version 2>&1 | head -1)"
echo "governor=$ACT_GOV no_turbo=$ACT_TURBO affinity=$ACT_AFFIN runnable_avg=${RUN_AVG:-NA} loadavg1=$LOAD1 host_frozen=$HOST_FROZEN"
echo "=================================================="

# ---- 6. interleaved best-of-N, REGULAR-FILE sink, sha-verify EVERY run -------
# CRITICAL: output goes to a regular file on /dev/shm (NEVER a pipe). A pipe sink
# backpressure-inflated writev into a phantom — the exact contamination class this
# wrapper exists to kill.
SINK_GZ="$ARTDIR/sink_gzippy.bin"
SINK_RG="$ARTDIR/sink_rapidgzip.bin"
SINK_GZ2=""   # populated below only when GZIPPY_BIN2 is set

# Defend the sink path: remove any pre-existing node (a planted FIFO/symlink from a
# prior or hostile run would otherwise survive into the first iterations), then
# assert each sink is a plain regular file — never a symlink or FIFO. A FIFO with a
# draining reader is exactly the writev phantom this wrapper exists to prevent.
assert_regular_sink() { # <path>
  local p="$1"
  rm -f "$p" 2>/dev/null || true
  : > "$p" || fail "cannot-create-sink:$p" 14
  [ -f "$p" ] && [ ! -L "$p" ] && [ ! -p "$p" ] || fail "sink-not-regular-file:$p (symlink/FIFO — pipe-phantom risk)" 14
}
assert_regular_sink "$SINK_GZ"
assert_regular_sink "$SINK_RG"
if [ -n "$GZIPPY_BIN2" ]; then
  SINK_GZ2="$ARTDIR/sink_gz2.bin"
  assert_regular_sink "$SINK_GZ2"
fi

timed() { # <sink> <cmd...> -> echoes "secs sha"
  local sink="$1"; shift
  local s e secs sha rc
  s=$(date +%s.%N)
  set +e
  taskset -c "$MASK" "$@" >"$sink" 2>>"$ARTDIR/run.stderr"; rc=$?
  set -e 2>/dev/null || true
  e=$(date +%s.%N)
  secs=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f", b-a}')
  sha=$(sha256sum "$sink" | cut -d' ' -f1)
  [ "$rc" -eq 0 ] || echo "## WARN exit=$rc: $*" >&2
  echo "$secs $sha"
}

GZT=""; GZ2T=""; RGT=""; DIVERGED=0
echo "## interleave (N=$N, drop warmup iter0)"
for ((i=0;i<=N;i++)); do
  read gsec gsha < <(timed "$SINK_GZ" env GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN" -d -c -p "$T" "$CORPUS")
  read rsec rsha < <(timed "$SINK_RG" "$RG_CMD" -d -c -f -P "$T" "$CORPUS")
  if [ -n "$GZIPPY_BIN2" ]; then
    read g2sec g2sha < <(timed "$SINK_GZ2" env GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN2" -d -c -p "$T" "$CORPUS")
  fi
  if [ "$i" -eq 0 ]; then continue; fi
  GZT="$GZT $gsec"; RGT="$RGT $rsec"
  if [ -n "$GZIPPY_BIN2" ]; then GZ2T="$GZ2T $g2sec"; fi
  if [ "$gsha" != "$REF_SHA" ]; then echo "!! SHA DIVERGENCE gzippy i=$i sha=$gsha"; DIVERGED=1; fi
  if [ "$rsha" != "$REF_SHA" ]; then echo "!! SHA DIVERGENCE rapidgzip i=$i sha=$rsha"; DIVERGED=1; fi
  if [ -n "$GZIPPY_BIN2" ] && [ "$g2sha" != "$REF_SHA" ]; then echo "!! SHA DIVERGENCE gzippy2 i=$i sha=$g2sha"; DIVERGED=1; fi
done
rm -f "$SINK_GZ" "$SINK_RG"
[ -n "$SINK_GZ2" ] && rm -f "$SINK_GZ2"

# ABORT on any wrong bytes (Rule 4) — the number is VOID.
if [ "$DIVERGED" -ne 0 ]; then
  fail "sha-mismatch (a wrong-bytes win is a loss — Rule 4); number VOID" 11
fi

stats() { # echoes "min med spreadpct"
  echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '
    { v[NR]=$1 } END {
      n=NR; if(n==0){print "0 0 0"; exit}
      min=v[1]; max=v[n]; mid=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
      printf "%.4f %.4f %.0f", min, mid, (min>0)?(max-min)/min*100:0 }'
}
mbps() { awk -v r="$RAW_BYTES" -v t="$1" 'BEGIN{printf "%.0f", (t>0)?r/t/1e6:0}'; }

read gmin gmed gsp < <(stats "$GZT")
read rmin rmed rsp < <(stats "$RGT")

# RELATIVE signal (jitter-immune): ratio = rg_time / gzippy_time = gzippy_tput/rg_tput
RATIO="$(awk -v g="$gmin" -v r="$rmin" 'BEGIN{printf "%.3f", (g>0)?r/g:0}')"
MARGIN="$(awk -v a="$gsp" -v b="$rsp" 'BEGIN{m=(a>b)?a:b; print m/100.0}')"
VERDICT="$(awk -v x="$RATIO" -v m="$MARGIN" 'BEGIN{d=x-1; if(d>m)print "WIN(gzippy)"; else if(d<-m)print "LOSS"; else print "TIE"}')"

echo ""
echo "================ PARITY SUMMARY (T=$T) ================"
printf "gzippy    min=%.4fs (%s MB/s) med=%.4f spread=%s%%\n" "$gmin" "$(mbps "$gmin")" "$gmed" "$gsp"
printf "rapidgzip min=%.4fs (%s MB/s) med=%.4f spread=%s%%\n" "$rmin" "$(mbps "$rmin")" "$rmed" "$rsp"
echo "RELATIVE: gzippy is ${RATIO}x rapidgzip's throughput => $VERDICT  (TIE margin=${MARGIN})"
# The one canonical summary line the spec asks for:
printf "gzippy=%sms  rg=%sms  ratio=%s  sha=OK  verdict=%s\n" \
  "$(awk -v t="$gmin" 'BEGIN{printf "%.0f", t*1000}')" \
  "$(awk -v t="$rmin" 'BEGIN{printf "%.0f", t*1000}')" \
  "$RATIO" "$VERDICT"

# ---- three-way summary (only when BIN2 set) -----------------------------------
if [ -n "$GZIPPY_BIN2" ]; then
  read g2min g2med g2sp < <(stats "$GZ2T")
  RATIO_RG_GZ2="$(awk -v g="$g2min" -v r="$rmin" 'BEGIN{printf "%.3f", (g>0)?r/g:0}')"
  RATIO_GZ1_GZ2="$(awk -v g="$g2min" -v g1="$gmin" 'BEGIN{printf "%.3f", (g>0)?g1/g:0}')"
  MARGIN2="$(awk -v a="$g2sp" -v b="$rsp" 'BEGIN{m=(a>b)?a:b; print m/100.0}')"
  VERDICT2="$(awk -v x="$RATIO_RG_GZ2" -v m="$MARGIN2" 'BEGIN{d=x-1; if(d>m)print "WIN(gz2)"; else if(d<-m)print "LOSS"; else print "TIE"}')"
  echo ""
  echo "================ THREE-WAY SUMMARY (T=$T) ================"
  printf "gz1 (%s)  min=%.4fs (%s MB/s) med=%.4f spread=%s%%\n" "$FEATURE"  "$gmin"  "$(mbps "$gmin")"  "$gmed"  "$gsp"
  printf "gz2 (%s)  min=%.4fs (%s MB/s) med=%.4f spread=%s%%\n" "$FEATURE2" "$g2min" "$(mbps "$g2min")" "$g2med" "$g2sp"
  printf "rg            min=%.4fs (%s MB/s) med=%.4f spread=%s%%\n" "$rmin" "$(mbps "$rmin")" "$rmed" "$rsp"
  echo "RATIO rg/gz1=${RATIO}x  rg/gz2=${RATIO_RG_GZ2}x  gz1_wall/gz2_wall=${RATIO_GZ1_GZ2}x (>1 means gz2 faster)"
  echo "verdict gz2-vs-rg: $VERDICT2"
  printf "gz1=%sms  gz2=%sms  rg=%sms  ratio_rg_gz1=%s  ratio_rg_gz2=%s  ratio_gz1_gz2=%s  sha=OK\n" \
    "$(awk -v t="$gmin"  'BEGIN{printf "%.0f", t*1000}')" \
    "$(awk -v t="$g2min" 'BEGIN{printf "%.0f", t*1000}')" \
    "$(awk -v t="$rmin"  'BEGIN{printf "%.0f", t*1000}')" \
    "$RATIO" "$RATIO_RG_GZ2" "$RATIO_GZ1_GZ2"
  echo "========================================================="
fi

# ---- 7. optional --fulcrum decompose ----------------------------------------
if [ "$DO_FULCRUM" = 1 ]; then
  echo ""
  echo "## --fulcrum: capturing a window-absent trace for fulcrum_total decompose"
  if [ -x scripts/bench/fulcrum_total_capture.sh ]; then
    bash scripts/bench/fulcrum_total_capture.sh \
      LABEL="parity_T${T}" T="$T" CORPUS="$CORPUS" ARTDIR="$ARTDIR" GZIPPY="$GZIPPY_BIN" \
      || echo "## WARN: fulcrum capture failed (non-fatal; parity number above stands)"
    echo "## analyze on any host:  python3 scripts/fulcrum_total.py $ARTDIR/trace_parity_T${T}.json"
  else
    echo "## WARN: scripts/bench/fulcrum_total_capture.sh absent — skipping decompose"
  fi
fi

echo "================ END PARITY SUMMARY ================"
echo "PARITY_GUEST_DONE"
