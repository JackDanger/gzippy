#!/usr/bin/env bash
# _settle_guest.sh — SETTLEMENT run: gz vs rapidgzip(NATIVE) interleaved, BOTH
# sinks (/dev/null AND regular-file) per cell, to resolve the parity-vs-loss
# question and the sink-inversion paradox. Reuses the certified VERBATIM
# primitives from lib_decide_guest.sh (timed_masked, freeze_readback,
# quiet_gate, pin_mask, stats). NO product code touched.
#
# Per cell, per sink: interleaved rg,gz,rg,gz... N pairs (drop warmup iter0).
#  - regular-file arm: sha-verify EVERY run (Rule 4).
#  - /dev/null arm: byte-exactness PRE-VERIFIED once per tool to a regfile,
#    then timed to /dev/null (cannot sha a discarded stream). NEVER rm /dev/null.
# Champions (igzip/libdeflate/pigz) timed on the regfile arm as a bonus.
#
# Inputs (env): BIN RG CELLS N ARTDIR RUNID GOV NO_TURBO HOST_FROZEN ALLOW_LOAD
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "$HERE/lib_decide_guest.sh"

: "${BIN:?}"; : "${RG:?}"; : "${CELLS:?}"; : "${ARTDIR:?}"; : "${RUNID:?}"
N="${N:-12}"
GOV="${GOV:-performance}"; NO_TURBO="${NO_TURBO:-1}"
COMP_IGZIP="${COMP_IGZIP:-/usr/bin/igzip}"
COMP_LIBDEFLATE="${COMP_LIBDEFLATE:-/usr/bin/libdeflate-gunzip}"
COMP_PIGZ="${COMP_PIGZ:-/usr/bin/pigz}"

OUT="$ARTDIR/$RUNID"; mkdir -p "$OUT"
export ARTDIR="$OUT"            # timed_masked appends to $ARTDIR/run.stderr
MANIFEST="$OUT/manifest.txt"; : > "$MANIFEST"
mf() { echo "$1" >> "$MANIFEST"; }

settle_fail() { echo "SETTLE_FAIL=$1"; echo "SETTLE_GUEST_DONE"; exit "${2:-1}"; }

# ---- box-valid gates (certified spine) --------------------------------------
scrub_gzippy_env ""
freeze_readback                # dies on a readable-thawed host
quiet_gate                     # dies on a loaded box (unless ALLOW_LOAD=1)

[ -x "$BIN" ] || settle_fail "no-binary:$BIN" 5
[ -x "$RG" ]  || settle_fail "no-rg-native:$RG" 12
BIN_SHA="$(sha256sum "$BIN" | cut -d' ' -f1)"
RG_SHA="$(sha256sum "$RG" | cut -d' ' -f1)"

mf "runid=$RUNID"
mf "bin=$BIN"
mf "bin_sha=$BIN_SHA"
mf "rg=$RG"
mf "rg_sha=$RG_SHA"
mf "rg_version=$("$RG" --version 2>&1 | head -1 | tr -d '\n')"
mf "protocol=settle-bothsinks-v1"
mf "host_cpu_model=$(awk -F': *' '/^model name/{print $2; exit}' /proc/cpuinfo 2>/dev/null || true)"
mf "host_kernel=$(uname -r 2>/dev/null || true)"
mf "host_id=$( (cat /etc/machine-id 2>/dev/null || hostname) | sha256sum | cut -c1-12 )"
mf "freeze_state=$FREEZE_STATE"
mf "quiet_state=$QUIET_STATE"
mf "governor=$ACT_GOV"
mf "no_turbo=$ACT_TURBO"
mf "runnable_avg=${RUN_AVG:-NA}"
mf "n=$N"
mf "cells=$CELLS"
mf "devnull_class=$(sink_class_of /dev/null)"
mf "started=$(date -u '+%FT%TZ')"

# champions present?
declare -a CHAMPS=()
declare -A CHAMP_BIN=()
for pair in "igzip:$COMP_IGZIP" "libdeflate:$COMP_LIBDEFLATE" "pigz:$COMP_PIGZ"; do
  nm="${pair%%:*}"; bp="${pair#*:}"
  if [ -x "$bp" ] || command -v "$bp" >/dev/null 2>&1; then
    CHAMPS+=("$nm"); CHAMP_BIN[$nm]="$bp"; mf "champ_present=$nm:$bp"
  else mf "champ_absent=$nm:$bp"; fi
done

champ_argv() { # <tool> <bin> <t> <f> -> CHAMP_ARGV
  case "$1" in
    igzip)      CHAMP_ARGV=("$2" -d -c "$4");;
    libdeflate) CHAMP_ARGV=("$2" -c "$4");;
    pigz)       CHAMP_ARGV=("$2" -d -c -p "$3" "$4");;
    *)          CHAMP_ARGV=("$2" -d -c "$4");;
  esac
}

declare -A REFSHA
corpus_path() { echo "/root/$1.gz"; }
ensure_ref() {
  local c="$1" f
  [ -n "${REFSHA[$c]:-}" ] && return 0
  f="$(corpus_path "$c")"
  [ -f "$f" ] || settle_fail "no-corpus:$f" 7
  REFSHA[$c]="$(gzip -dc "$f" | sha256sum | cut -d' ' -f1)"
  mf "corpus_${c}_sha=${REFSHA[$c]}"
}

# regular-file sink (one shared regfile, same for both arms within a sink)
REGSINK="$OUT/sink_reg.bin"
PREVERIFY="$OUT/sink_preverify.bin"

run_cell() { # <corpus> <T>
  local c="$1" t="$2" mask f cdir i
  mask="$(pin_mask "$t")"
  [ -n "$mask" ] || settle_fail "bad-T:$t" 8
  ensure_ref "$c"
  f="$(corpus_path "$c")"
  cdir="$OUT/cell_${c}_T${t}"; mkdir -p "$cdir"

  # routing assert (production path must be ParallelSM on the native build)
  local dbg
  dbg="$(GZIPPY_DEBUG=1 "$BIN" -d -c -p "$t" "$f" 2>&1 >/dev/null | grep -m1 'path=' || true)"
  case "$dbg" in
    *path=ParallelSM*) ;;
    *) settle_fail "routing $c:T$t not ParallelSM (got: ${dbg:-none})" 9;;
  esac

  # ---- PRE-VERIFY byte-exactness (once each) so the /dev/null arm is honest --
  assert_regular_sink "$PREVERIFY"
  taskset -c "$mask" "$BIN" -d -c -p "$t" "$f" > "$PREVERIFY" 2>>"$ARTDIR/run.stderr"
  [ "$(sha256sum "$PREVERIFY" | cut -d' ' -f1)" = "${REFSHA[$c]}" ] \
    || settle_fail "gz-bytes-wrong $c:T$t" 11
  assert_regular_sink "$PREVERIFY"
  taskset -c "$mask" "$RG" -d -c -f -P "$t" "$f" > "$PREVERIFY" 2>>"$ARTDIR/run.stderr"
  [ "$(sha256sum "$PREVERIFY" | cut -d' ' -f1)" = "${REFSHA[$c]}" ] \
    || settle_fail "rg-bytes-wrong $c:T$t" 11

  # ---- SINK = /dev/null : interleaved rg,gz (timing only; bytes pre-verified)
  local GZ_DN="" RG_DN="" gsec rsec _sha _rss
  echo "## cell $c:T$t mask=$mask sink=/dev/null  interleave N=$N (drop iter0)"
  for ((i=0;i<=N;i++)); do
    read -r rsec _sha _rss < <(timed_masked "$mask" /dev/null "$RG"  -d -c -f -P "$t" "$f")
    read -r gsec _sha _rss < <(timed_masked "$mask" /dev/null "$BIN" -d -c    -p "$t" "$f")
    [ "$i" -eq 0 ] && continue
    RG_DN="$RG_DN $rsec"; GZ_DN="$GZ_DN $gsec"
  done
  echo "$GZ_DN" | tr ' ' '\n' | grep -v '^$' > "$cdir/devnull_gz.txt"
  echo "$RG_DN" | tr ' ' '\n' | grep -v '^$' > "$cdir/devnull_rg.txt"

  # ---- SINK = regular-file : interleaved rg,gz (sha-verify EVERY run) --------
  local GZ_RF="" RG_RF="" gsha rsha DIV=0
  assert_regular_sink "$REGSINK"
  echo "## cell $c:T$t mask=$mask sink=regular-file  interleave N=$N (drop iter0)"
  for ((i=0;i<=N;i++)); do
    read -r rsec rsha _rss < <(timed_masked "$mask" "$REGSINK" "$RG"  -d -c -f -P "$t" "$f")
    read -r gsec gsha _rss < <(timed_masked "$mask" "$REGSINK" "$BIN" -d -c    -p "$t" "$f")
    [ "$i" -eq 0 ] && continue
    RG_RF="$RG_RF $rsec"; GZ_RF="$GZ_RF $gsec"
    [ "$gsha" = "${REFSHA[$c]}" ] || { echo "!! SHA gz $c:T$t i=$i"; DIV=1; }
    [ "$rsha" = "${REFSHA[$c]}" ] || { echo "!! SHA rg $c:T$t i=$i"; DIV=1; }
  done
  [ "$DIV" -eq 0 ] || settle_fail "sha-mismatch-regfile $c:T$t" 11
  echo "$GZ_RF" | tr ' ' '\n' | grep -v '^$' > "$cdir/regfile_gz.txt"
  echo "$RG_RF" | tr ' ' '\n' | grep -v '^$' > "$cdir/regfile_rg.txt"

  # ---- champions (bonus) on the regular-file sink ---------------------------
  local tool
  for tool in "${CHAMPS[@]:-}"; do
    [ -n "$tool" ] || continue
    champ_argv "$tool" "${CHAMP_BIN[$tool]}" "$t" "$f"
    local CS="" cdiv=0 csec csha
    assert_regular_sink "$REGSINK"
    for ((i=0;i<=N;i++)); do
      read -r csec csha _rss < <(timed_masked "$mask" "$REGSINK" "${CHAMP_ARGV[@]}")
      [ "$i" -eq 0 ] && continue
      CS="$CS $csec"
      [ "$csha" = "${REFSHA[$c]}" ] || cdiv=1
    done
    if [ "$cdiv" -ne 0 ]; then echo "champ_sha_fail=$c:T$t:$tool" >> "$cdir/champ_meta.txt"; continue; fi
    echo "$CS" | tr ' ' '\n' | grep -v '^$' > "$cdir/regfile_${tool}.txt"
  done

  mf "cell_done=$c:T$t:mask=$mask"
}

IFS=',' read -ra CELL_ARR <<< "$CELLS"
for cell in "${CELL_ARR[@]}"; do
  corpus="${cell%%:*}"; tt="${cell##*:}"
  run_cell "$corpus" "$tt"
done

rm -f "$REGSINK" "$PREVERIFY"
mf "finished=$(date -u '+%FT%TZ')"
echo "SETTLE_ARTIFACTS=$OUT"
echo "SETTLE_GUEST_DONE"
