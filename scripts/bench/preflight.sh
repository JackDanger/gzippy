#!/usr/bin/env bash
# scripts/bench/preflight.sh — MANDATORY pre-measurement enforcement gate.
#
# WHY THIS EXISTS (built 2026-07-10 from the branch bias post-mortem)
# ------------------------------------------------------------------
# Prose reminders about measurement hygiene have failed REPEATEDLY on this
# campaign — the same handful of blindspots recurred across sessions and across
# context-compaction because "remember to use /dev/null" / "remember which box"
# lived only in prose that fell out of context. The governing law is: build the
# TOOL that enforces the gate, not diligence. This script is that tool. A number
# does not count as a measurement until `preflight.sh` prints `PREFLIGHT=OK`.
#
# It mechanically refuses to green-light a measurement that repeats any of the
# receipted branch blindspots:
#   (a) SINK LAW      — file sink dilutes the rg/gz ratio toward 1.0 and HID the
#                       storedheavy loss (0.96 file vs 0.78 /dev/null). Both arms
#                       MUST sink to /dev/null; /dev/null must be an intact device.
#   (b) BOX IDENTITY  — solvency(AMD,10.0.2.240) vs trainer/lxc199(Intel,
#                       10.30.0.199) were confused for whole rounds. The intended
#                       box is asserted against the box actually under our feet.
#   (c) GATE-4 PATH   — GZIPPY_FORCE_PARALLEL_SM is inert at HEAD; storedheavy
#                       routes StoredParallel(->demote->ParallelSM). We measured
#                       the wrong/forced path for many rounds. Assert the SHIPPED
#                       binary's build-flavor, the OBSERVED DecodePath, and that
#                       decoded bytes == gunzip (sha).
#   (d) FREEZE        — the freeze is OUR OWN script (bench-lock on neurotic /
#                       boost=0 + kill-STOP llama on solvency), reproducible, NOT
#                       external. "the box can't be frozen" was a wrong-knob error.
#                       Freeze must be engaged AND self-test-reproduce a known
#                       authority cell within tolerance, else the box state differs
#                       from every banked number and nothing is comparable.
#   (e) SIGNIFICANCE  — best-of-N is UNUSABLE at the ~30-42ms /dev/null walls this
#                       campaign lives at (false-NOISY). Under the wall floor the
#                       method MUST be paired-diff (per-sample Δ CI, N>=51).
#   (f) FREQ-NEUTRAL CONTROL — a busy-SPIN perturbation depresses turbo and
#                       confounds co-located regions (the marker_fast_loop
#                       "1:1 slope 0.988" was the DISCARDED spin arm; the SLEEP
#                       control said ~0.23/overlapped). Controls must be sleep.
#
# It emits ONE machine-checkable line: `PREFLIGHT=OK` or `PREFLIGHT=FAIL ...`.
# Wire it in: a measurement driver must `preflight.sh || exit 1` before it runs.
#
# ---------------------------------------------------------------------------
# USAGE
#   # configure via env, then run on the BENCH BOX (not the mac):
#   EXPECT_BOX=solvency \
#   GZIPPY_BIN=/dev/shm/standing-target/release/gzippy \
#   RG_BIN=/root/oracle_c/rapidgzip-native \
#   CORPUS=/root/storedheavy.gz \
#   SINK=/dev/null \
#   METHOD=paired-diff \
#   A_CMD='gzippy -d -c -p 4 /root/storedheavy.gz' \
#   B_CMD='rapidgzip -d -c -f -P 4 /root/storedheavy.gz' \
#   AUTHORITY_CELL=storedheavy-T4 AUTHORITY_RATIO=0.78 AUTHORITY_TOL=0.06 \
#     scripts/bench/preflight.sh
#
#   scripts/bench/preflight.sh --emit-template   # print a copy-paste config block
#   scripts/bench/preflight.sh --print-rig       # print the rig manifest it enforces
#
# EXIT: 0 iff PREFLIGHT=OK. Non-zero otherwise. Never mutates box state.
# ---------------------------------------------------------------------------
set -u

WALL_FLOOR_MS="${WALL_FLOOR_MS:-60}"   # below this, best-of-N is forbidden (paired-diff only)
FAILS=()
WARNS=()
fail() { FAILS+=("$1"); echo "GATE $1"; }
warn() { WARNS+=("$1"); echo "WARN $1"; }
ok()   { echo "GATE $1"; }

# ---- box identity table (the durable fact that kept getting re-derived wrong) --
# name | expect-hostname-substr | expect-arch | expect-primary-ip-substr | freeze-kind
box_row() {
  case "$1" in
    solvency)      echo "solvency x86_64 10.0.2.240 amd-boost0-llamastop" ;;
    trainer)       echo "trainer  x86_64 10.30.0.199 neurotic-benchlock"  ;;
    lxc199)        echo "trainer  x86_64 10.30.0.199 neurotic-benchlock"  ;;
    neurotic)      echo "neurotic x86_64 10.30 neurotic-benchlock"        ;;
    local|m1|mac)  echo "$(hostname -s 2>/dev/null) arm64 127 none-local" ;;
    *)             echo "UNKNOWN UNKNOWN UNKNOWN UNKNOWN" ;;
  esac
}

if [ "${1:-}" = "--print-rig" ]; then
  cat <<'RIG'
RIG MANIFEST (enforced by preflight.sh) — see memory reference_rig_manifest.md
  solvency  = AMD EPYC 7282 Zen2, ssh jackdanger@10.0.2.240 (physical host).
              AUTHORITY box for storedheavy (loss is AMD-Zen2-specific).
              FREEZE: echo 0 >/sys/devices/system/cpu/cpufreq/boost ; kill -STOP <llama>
                      ; governor=performance ; taskset. RESTORE: boost=1 ; kill -CONT llama
                      ; verify llama STAT != T ; no orphans (feedback_llama_pause_no_orphan).
  trainer   = Intel i7-13700T LXC 199 on neurotic, ssh -J neurotic root@10.30.0.199.
              cpufreq/boost = "not implemented" (fixed-freq). FREEZE: bench-lock.sh
              acquire (freezes noisy LXCs + no_turbo=1 + governor lock + verify-quiet).
  SINK LAW  = both arms to /dev/null. A FILE sink dilutes rg/gz toward 1.0 and HIDES
              losses. dfa67ccc "authority" 0.94/0.96 scorelines were FILE-SINK => CONTAMINATED.
  SINK BIN  = /root/fulcrum-score2way (times to /dev/null + separate piped SHA), NOT
              the old file-sink /root/fulcrum.
  SIGNIF    = walls collapse to ~30-42ms at /dev/null => best-of-N UNUSABLE. Use paired
              per-sample Δ(native-rg) 95% CI, N>=51.
  AUTHORITY = storedheavy T4 0.78 / T8 (pre-reaper) / T16 0.93 on /dev/null frozen solvency
              (base a0769138). Reaper c49855a3: T4 0.87, T8 1.15, T16 0.89.
RIG
  exit 0
fi

if [ "${1:-}" = "--emit-template" ]; then
  cat <<'TMPL'
# ---- preflight config block (fill in, then: preflight.sh) ----
export EXPECT_BOX=solvency            # solvency | trainer | lxc199 | neurotic
export GZIPPY_BIN=/dev/shm/standing-target/release/gzippy
export RG_BIN=/root/oracle_c/rapidgzip-native
export CORPUS=/root/storedheavy.gz
export SINK=/dev/null                 # MUST be /dev/null
export METHOD=paired-diff             # paired-diff (walls<60ms) | best-of-N (walls>=60ms)
export A_CMD='gzippy -d -c -p 4 /root/storedheavy.gz'
export B_CMD='rapidgzip -d -c -f -P 4 /root/storedheavy.gz'
export AUTHORITY_CELL=storedheavy-T4  # optional freeze self-test
export AUTHORITY_RATIO=0.78
export AUTHORITY_TOL=0.06
export CONTROL_CMD=''                 # if a perturbation control: MUST be sleep-based, not spin
TMPL
  exit 0
fi

echo "=== preflight.sh $(date -u +%FT%TZ) ==="

# ===========================================================================
# GATE b — BOX IDENTITY (run first; several later gates are box-specific)
# ===========================================================================
EXPECT_BOX="${EXPECT_BOX:-}"
if [ -z "$EXPECT_BOX" ]; then
  fail "box: FAIL EXPECT_BOX unset — you must declare the intended box (solvency|trainer|lxc199|neurotic|local)"
  FREEZE_KIND="unknown"
else
  read -r exp_host exp_arch exp_ip FREEZE_KIND <<<"$(box_row "$EXPECT_BOX")"
  act_host="$(hostname -s 2>/dev/null || hostname 2>/dev/null)"
  act_arch="$(uname -m 2>/dev/null)"
  act_ip="$( { hostname -I 2>/dev/null || ifconfig 2>/dev/null | awk '/inet /{print $2}'; } | tr ' ' '\n' | grep -v '^127' | head -3 | tr '\n' ',' )"
  echo "  box: EXPECT=$EXPECT_BOX (host~$exp_host arch=$exp_arch ip~$exp_ip) ACTUAL(host=$act_host arch=$act_arch ip=[$act_ip])"
  bad=0
  [ "$exp_arch" = "UNKNOWN" ] && { fail "box: FAIL unknown EXPECT_BOX='$EXPECT_BOX'"; bad=1; }
  if [ "$exp_arch" != "UNKNOWN" ] && [ "$act_arch" != "$exp_arch" ]; then
    fail "box: FAIL arch mismatch expected $exp_arch got $act_arch (WRONG BOX — solvency=x86_64 AMD, local mac=arm64)"; bad=1
  fi
  if [ "$exp_ip" != "UNKNOWN" ] && [ "$exp_ip" != "none-local" ] && [ "$exp_ip" != "127" ]; then
    echo "$act_ip" | grep -q "$exp_ip" || warn "box: primary IP does not contain '$exp_ip' — confirm this is really $EXPECT_BOX and not a jump host"
  fi
  [ "$bad" = 0 ] && ok "box: PASS $EXPECT_BOX identity consistent (freeze-kind=$FREEZE_KIND)"
  if [ "$act_arch" = "arm64" ] || [ "$(uname -s)" = "Darwin" ]; then
    if [ "${ALLOW_LOCAL:-0}" != 1 ]; then
      fail "box: FAIL running on local mac (arm64/Darwin) — NOT a certified gate box. Set ALLOW_LOCAL=1 only for smoke, never for a banked number."
    else
      warn "box: local mac SMOKE mode (ALLOW_LOCAL=1) — result is NOT bankable."
    fi
  fi
fi

# ===========================================================================
# GATE a — SINK LAW (/dev/null both arms; device intact)
# ===========================================================================
SINK="${SINK:-}"
if [ "$SINK" != "/dev/null" ]; then
  fail "sink: FAIL SINK='$SINK' — must be /dev/null (a file sink dilutes rg/gz toward 1.0 and HID the storedheavy loss: 0.96 file vs 0.78 devnull)"
else
  if [ -c /dev/null ]; then ok "sink: PASS /dev/null is an intact char device, both arms sink here"
  else fail "sink: FAIL /dev/null is NOT a char device (rm-of-/dev/null trap) — every wall is garbage until restored (mknod -m666 /dev/null c 1 3)"; fi
fi
# Scan the actual A/B commands for a non-/dev/null file redirect (the sink asymmetry generator).
for pair in "A:${A_CMD:-}" "B:${B_CMD:-}"; do
  arm="${pair%%:*}"; cmd="${pair#*:}"
  [ -z "$cmd" ] && continue
  # a ">FILE" or "-o FILE" that is not /dev/null
  redir="$(printf '%s' "$cmd" | grep -oE '(>|-o )[[:space:]]*[^ ]+' | grep -vE '/dev/null|>&|>>?[[:space:]]*$' || true)"
  if [ -n "$redir" ]; then
    warn "sink: arm $arm writes to a non-/dev/null target ($redir) — if that is a regular file this is a phantom-regression generator; both arms must sink to /dev/null"
  fi
done

# ===========================================================================
# GATE c — GATE-4 PRODUCTION-PATH FINGERPRINT on the SHIPPED binary
# ===========================================================================
GZIPPY_BIN="${GZIPPY_BIN:-}"
CORPUS="${CORPUS:-}"
EXPECT_FLAVOR="${EXPECT_FLAVOR:-native}"   # substring expected in build-flavor (native/pure/parallel-sm)
if [ -z "$GZIPPY_BIN" ] || [ ! -x "$GZIPPY_BIN" ]; then
  fail "gate4: FAIL GZIPPY_BIN='$GZIPPY_BIN' missing/not-executable — cannot fingerprint the shipped binary"
elif [ -z "$CORPUS" ] || [ ! -r "$CORPUS" ]; then
  fail "gate4: FAIL CORPUS='$CORPUS' missing — cannot exercise the production path"
else
  dbg="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM="${GZIPPY_FORCE_PARALLEL_SM:-}" "$GZIPPY_BIN" -d -c ${GZ_THREADS:+-p $GZ_THREADS} "$CORPUS" 2>&1 >/dev/null | head -40 || true)"
  flavor="$(printf '%s\n' "$dbg" | grep -oE 'build-flavor=[^ ]+' | head -1)"
  path="$(printf '%s\n' "$dbg" | grep -oE 'path=[A-Za-z]+' | head -1)"
  echo "  gate4: $flavor $path (GZIPPY_FORCE_PARALLEL_SM='${GZIPPY_FORCE_PARALLEL_SM:-<unset>}')"
  # flavor must match intended; must NOT be a legacy libdeflate-decode flavor
  if [ -z "$flavor" ]; then
    fail "gate4: FAIL no build-flavor printed — binary too old or GZIPPY_DEBUG not honored; cannot verify this is the pure-Rust ship target"
  elif ! printf '%s' "$flavor" | grep -qiE "$EXPECT_FLAVOR"; then
    fail "gate4: FAIL build-flavor '$flavor' does not match EXPECT_FLAVOR='$EXPECT_FLAVOR' (buildflavor-disconnect: release/CI once shipped libdeflate-decode not pure-rust)"
  else
    ok "gate4: PASS build-flavor '$flavor' matches"
  fi
  # observed decode path — StoredParallel demotes to ParallelSM; accept either but PRINT which
  case "$path" in
    path=ParallelSM)     ok "gate4: PASS observed $path" ;;
    path=StoredParallel) warn "gate4: observed $path — this DEMOTES to ParallelSM internally; confirm you are measuring the path you intend (storedheavy routed StoredParallel for many rounds while agents thought they forced ParallelSM)" ;;
    path=Isal*|path=Libdeflate*|path=*Streaming*) fail "gate4: FAIL observed $path is NOT the pure-Rust parallel path — wrong routing/flavor for a native ship gate" ;;
    "") fail "gate4: FAIL no path= line printed — routing did not run the single-member parallel pipeline (check corpus is single-member and >0 bytes)" ;;
    *)  warn "gate4: observed $path (unexpected) — verify against the routing table" ;;
  esac
  # inert-force trap
  if [ -n "${GZIPPY_FORCE_PARALLEL_SM:-}" ] && [ "$path" != "path=ParallelSM" ] && [ "$path" != "path=StoredParallel" ]; then
    fail "gate4: FAIL GZIPPY_FORCE_PARALLEL_SM is SET but path=$path — the knob is INERT here; you are measuring the default route, not the forced one"
  fi
  # sha correctness vs gunzip reference (a fast win with wrong bytes is a loss)
  if command -v gunzip >/dev/null 2>&1 && command -v sha256sum >/dev/null 2>&1; then
    ref_sha="$(gunzip -c "$CORPUS" 2>/dev/null | sha256sum | cut -d' ' -f1)"
    got_sha="$("$GZIPPY_BIN" -d -c ${GZ_THREADS:+-p $GZ_THREADS} "$CORPUS" 2>/dev/null | sha256sum | cut -d' ' -f1)"
    if [ -n "$ref_sha" ] && [ "$ref_sha" = "$got_sha" ]; then ok "gate4: PASS decoded sha == gunzip ($got_sha)"
    else fail "gate4: FAIL decoded sha mismatch gzippy=$got_sha gunzip=$ref_sha — output is WRONG, any speed number is void"; fi
  else
    warn "gate4: gunzip/sha256sum unavailable — could not sha-verify output"
  fi
fi

# ===========================================================================
# GATE d — FREEZE ENGAGED + AUTHORITY SELF-TEST
# ===========================================================================
case "$FREEZE_KIND" in
  amd-boost0-llamastop)
    boost="$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo NA)"
    gov="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
    llama_st="$(ps -eo stat,comm 2>/dev/null | awk 'tolower($2) ~ /llama/{print $1; exit}')"
    echo "  freeze(solvency): boost=$boost governor=$gov llama_stat=${llama_st:-<none>}"
    [ "$boost" = 0 ] || fail "freeze: FAIL cpufreq/boost=$boost (want 0) — CPB not disabled; this is the 'box can't be frozen' wrong-knob error. echo 0 >/sys/devices/system/cpu/cpufreq/boost"
    [ "$gov" = performance ] || warn "freeze: governor=$gov (want performance)"
    if [ -n "$llama_st" ]; then
      case "$llama_st" in T*) ok "freeze: PASS llama is STOPPED (STAT=$llama_st)";; *) fail "freeze: FAIL llama STAT=$llama_st is RUNNING — kill -STOP it for the quiet window (and CONT + verify STAT!=T after, no orphans)";; esac
    else
      warn "freeze: no llama process seen — if it should be running for the no-orphan protocol, confirm"
    fi
    ;;
  neurotic-benchlock)
    if [ -x /root/bench-lock.sh ]; then
      bl="$(/root/bench-lock.sh verify 2>&1 | head -2)"
      echo "  freeze(neurotic/trainer): $bl"
      printf '%s' "$bl" | grep -q 'BENCH_LOCK=quiet' && ok "freeze: PASS bench-lock reports quiet" || fail "freeze: FAIL bench-lock not quiet ($bl) — a neighbor escaped the freeze; do NOT bank an absolute"
      nt="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA)"
      [ "$nt" = 1 ] || warn "freeze: intel_pstate/no_turbo=$nt (want 1); on fixed-freq trainer this may read NA — ok"
    else
      warn "freeze: /root/bench-lock.sh not found on this box — cannot assert the freeze; run bench-lock acquire on neurotic first"
    fi
    ;;
  none-local) warn "freeze: local mac — no freeze gate (not bankable)";;
  *) [ -n "$EXPECT_BOX" ] && warn "freeze: no freeze convention for box '$EXPECT_BOX'";;
esac

# AUTHORITY self-test: prove the frozen box reproduces a KNOWN cell within tolerance,
# else its state differs from every banked number and nothing is comparable.
if [ -n "${AUTHORITY_CELL:-}" ] && [ -n "${AUTHORITY_RATIO:-}" ] && [ -n "${A_CMD:-}" ] && [ -n "${B_CMD:-}" ]; then
  tol="${AUTHORITY_TOL:-0.06}"; n="${AUTHORITY_N:-9}"
  echo "  authority: self-testing $AUTHORITY_CELL (target rg/gz=$AUTHORITY_RATIO +/- $tol, N=$n, interleaved, /dev/null)"
  suma=0; sumb=0; okrun=1
  for ((i=1;i<=n;i++)); do
    s=$(date +%s.%N); eval "$A_CMD" >/dev/null 2>&1 || okrun=0; e=$(date +%s.%N)
    ta=$(awk -v a="$s" -v b="$e" 'BEGIN{print b-a}')
    s=$(date +%s.%N); eval "$B_CMD" >/dev/null 2>&1 || okrun=0; e=$(date +%s.%N)
    tb=$(awk -v a="$s" -v b="$e" 'BEGIN{print b-a}')
    suma=$(awk -v x="$suma" -v y="$ta" 'BEGIN{print x+y}')
    sumb=$(awk -v x="$sumb" -v y="$tb" 'BEGIN{print x+y}')
  done
  # ratio = B/A = rg/gz  (A_CMD is gzippy, B_CMD is rg; >1 means gz faster)
  ratio=$(awk -v a="$suma" -v b="$sumb" 'BEGIN{ if(a>0) printf "%.3f", b/a; else print "NA"}')
  dev=$(awk -v r="$ratio" -v t="$AUTHORITY_RATIO" 'BEGIN{d=r-t; if(d<0)d=-d; printf "%.3f", d}')
  echo "  authority: measured rg/gz=$ratio (target $AUTHORITY_RATIO, dev=$dev, tol=$tol)"
  [ "$okrun" = 1 ] || fail "authority: FAIL an A/B run errored during self-test"
  if [ "$ratio" = NA ]; then fail "authority: FAIL could not compute ratio"
  elif awk -v d="$dev" -v t="$tol" 'BEGIN{exit !(d<=t)}'; then ok "authority: PASS box reproduces $AUTHORITY_CELL within tolerance — state matches the banked regime"
  else fail "authority: FAIL box does NOT reproduce $AUTHORITY_CELL ($ratio vs $AUTHORITY_RATIO) — box state differs from banked numbers; freeze/sink/binary is off, do not compare"; fi
else
  warn "authority: no AUTHORITY_CELL self-test configured — STRONGLY recommended (proves the frozen box matches the banked regime before you trust a delta)"
fi

# ===========================================================================
# GATE e — SIGNIFICANCE METHOD auto-selection (paired-diff at small walls)
# ===========================================================================
METHOD="${METHOD:-}"
wall_ms=""
if [ -n "${A_CMD:-}" ]; then
  s=$(date +%s.%N); eval "$A_CMD" >/dev/null 2>&1 || true; e=$(date +%s.%N)
  wall_ms=$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.0f", (b-a)*1000}')
  echo "  significance: subject wall ~= ${wall_ms}ms (floor=${WALL_FLOOR_MS}ms)"
fi
if [ -z "$METHOD" ]; then
  fail "significance: FAIL METHOD unset — declare paired-diff or best-of-N"
elif [ -n "$wall_ms" ] && awk -v w="$wall_ms" -v f="$WALL_FLOOR_MS" 'BEGIN{exit !(w<f)}'; then
  if [ "$METHOD" = "paired-diff" ]; then ok "significance: PASS wall ${wall_ms}ms < ${WALL_FLOOR_MS}ms and METHOD=paired-diff (per-sample Δ CI, N>=51)"
  else fail "significance: FAIL wall ${wall_ms}ms < ${WALL_FLOOR_MS}ms but METHOD='$METHOD' — best-of-N is UNUSABLE at small walls (false-NOISY). Use paired-diff (per-sample Δ(gz-rg) 95% CI, N>=51)."; fi
else
  [ "$METHOD" = "paired-diff" ] && ok "significance: PASS METHOD=paired-diff (always valid)" || ok "significance: PASS wall >= floor, METHOD='$METHOD' acceptable (paired-diff still preferred)"
fi

# ===========================================================================
# GATE f — CONTROL MUST BE FREQ-NEUTRAL (sleep), never a busy spin
# ===========================================================================
if [ -n "${CONTROL_CMD:-}" ]; then
  if printf '%s' "$CONTROL_CMD" | grep -qiE 'while[[:space:]]+(true|:)|spin|:\(\)|yes[[:space:]]*>|busy|for *\(\( *; *; *\)\)|dd .*if=/dev/zero'; then
    fail "control: FAIL CONTROL_CMD looks like a BUSY-SPIN ($CONTROL_CMD) — spin depresses turbo AND steals ALU from co-located regions (the marker_fast_loop 1:1 slope 0.988 was the DISCARDED spin arm). Use a sleep that yields the core."
  elif printf '%s' "$CONTROL_CMD" | grep -qiE 'sleep|nanosleep|usleep'; then
    ok "control: PASS CONTROL_CMD is sleep-based (freq-neutral)"
  else
    warn "control: CONTROL_CMD='$CONTROL_CMD' — cannot confirm it is sleep-based; a freq-neutral control MUST yield the core (sleep), not busy-spin"
  fi
fi

# ===========================================================================
# VERDICT
# ===========================================================================
echo "---"
if [ "${#WARNS[@]}" -gt 0 ]; then echo "PREFLIGHT_WARNINGS=${#WARNS[@]}"; fi
if [ "${#FAILS[@]}" -eq 0 ]; then
  echo "PREFLIGHT=OK box=$EXPECT_BOX sink=$SINK method=$METHOD wall_ms=${wall_ms:-NA}"
  exit 0
else
  echo "PREFLIGHT=FAIL failed_gates=${#FAILS[@]} — a number produced now does NOT count as a measurement. Fix the FAILs above and re-run."
  exit 1
fi
