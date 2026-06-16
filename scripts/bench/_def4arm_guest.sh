#!/usr/bin/env bash
# _def4arm_guest.sh — DEFINITIVE 4-arm cycles/byte + wall-ratio experiment.
# Pinned binary only (no rebuild). Arms: gz(CRC), rg(CRC), rg(noCRC).
# gz(noCRC) is derived from an isolated crc32fast bench (separate step).
# cycles/byte via perf stat (freq-robust); wall ratio via interleaved best-of-N.
set -u
BIN="${BIN:?}"; CORPUS="${CORPUS:?}"; T="${T:?}"; REPS="${REPS:-15}"
MASK="${MASK:-0,2,4,6,8,10,12,14}"; RG="${RG:-rapidgzip}"
ART="${ART:-/dev/shm/def4arm}"; mkdir -p "$ART"
SINK="/dev/null"
VFILE="/mnt/internal/def4arm_verify.bin"

[ -x "$BIN" ] || { echo "FAIL no-bin:$BIN"; exit 5; }
[ -f "$CORPUS" ] || { echo "FAIL no-corpus:$CORPUS"; exit 6; }
command -v "$RG" >/dev/null || { echo "FAIL no-rg"; exit 7; }

BIN_SHA="$(sha256sum "$BIN" | cut -c1-16)"
REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
RAW_BYTES="$(gzip -dc "$CORPUS" | wc -c)"

# routing assert
DBG="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$BIN" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -m1 path=)"
case "$DBG" in *ParallelSM*) ;; *) echo "FAIL routing:$DBG"; exit 9;; esac

# freeze readback
GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null||echo NA)"
TRB="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null||echo NA)"
RUN="$(awk '/^procs_running/{print $2}' /proc/stat)"

# perf event pick
EVC="cycles"; EVI="instructions"
if perf stat -e cpu_core/cycles/ -- true >/dev/null 2>&1; then EVC="cpu_core/cycles/"; EVI="cpu_core/instructions/"; fi

echo "=============== DEF-4ARM corpus=$(basename "$CORPUS") T=$T REPS=$REPS ==============="
echo "bin_sha=$BIN_SHA raw_bytes=$RAW_BYTES ref_sha=${REF_SHA:0:16} gov=$GOV no_turbo=$TRB runnable=$RUN routing=ok event=$EVC"

# ---- byte-exactness (single run each, real file) ----------------------------
verify(){ "$@" >"$VFILE" 2>/dev/null; local s; s="$(sha256sum "$VFILE"|cut -d' ' -f1)"; [ "$s" = "$REF_SHA" ] && echo OK || echo "DIFF($s)"; }
echo "byte-exact: gz=$(verify env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$BIN" -d -c -p "$T" "$CORPUS") rg_crc=$(verify taskset -c "$MASK" "$RG" -d -c -f -P "$T" --verify "$CORPUS") rg_nocrc=$(verify taskset -c "$MASK" "$RG" -d -c -f -P "$T" --no-verify "$CORPUS")"
rm -f "$VFILE"

# ---- perf stat per arm: capture cycles + instructions (mean over REPS) -------
perf_arm() {  # name  cmd...
  local name="$1"; shift
  perf stat -r "$REPS" -e "$EVC,$EVI" -o "$ART/$name.stat" -- "$@" >"$SINK" 2>>"$ART/run.err"
  local cyc instr el
  cyc="$(grep -E "$EVC" "$ART/$name.stat" | head -1 | tr -d ',' | awk '{print $1}')"
  instr="$(grep -E "$EVI" "$ART/$name.stat" | head -1 | tr -d ',' | awk '{print $1}')"
  el="$(grep -E 'seconds time elapsed' "$ART/$name.stat" | awk '{print $1}')"
  local cpb ipb
  cpb="$(awk -v c="$cyc" -v b="$RAW_BYTES" 'BEGIN{printf "%.3f", (b>0)?c/b:0}')"
  ipb="$(awk -v c="$instr" -v b="$RAW_BYTES" 'BEGIN{printf "%.3f", (b>0)?c/b:0}')"
  printf "%-12s cyc/byte=%-7s instr/byte=%-7s elapsed=%-8s\n" "$name" "$cpb" "$ipb" "$el"
  echo "$name cyc=$cyc instr=$instr cpb=$cpb ipb=$ipb el=$el" >>"$ART/summary.txt"
}

: >"$ART/summary.txt"
echo "--- perf stat (cpu_core, REPS=$REPS) ---"
perf_arm gz_crc    env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$BIN" -d -c -p "$T" "$CORPUS"
perf_arm rg_crc    taskset -c "$MASK" "$RG" -d -c -f -P "$T" --verify "$CORPUS"
perf_arm rg_nocrc  taskset -c "$MASK" "$RG" -d -c -f -P "$T" --no-verify "$CORPUS"

# ---- interleaved best-of-N wall (ratio) -------------------------------------
echo "--- interleaved best-of-N wall (N=$REPS) ---"
GZW=""; RGCW=""; RGNW=""
timed(){ local s e; s=$(date +%s.%N); "$@" >"$SINK" 2>/dev/null; e=$(date +%s.%N); awk -v a="$s" -v b="$e" 'BEGIN{printf "%.4f",b-a}'; }
for ((i=0;i<=REPS;i++)); do
  g=$(timed env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$BIN" -d -c -p "$T" "$CORPUS")
  rc=$(timed taskset -c "$MASK" "$RG" -d -c -f -P "$T" --verify "$CORPUS")
  rn=$(timed taskset -c "$MASK" "$RG" -d -c -f -P "$T" --no-verify "$CORPUS")
  [ "$i" -eq 0 ] && continue
  GZW="$GZW $g"; RGCW="$RGCW $rc"; RGNW="$RGNW $rn"
done
mn(){ echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | head -1; }
sp(){ echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{v[NR]=$1}END{printf "%.0f",(v[1]>0)?(v[NR]-v[1])/v[1]*100:0}'; }
gmin=$(mn "$GZW"); rcmin=$(mn "$RGCW"); rnmin=$(mn "$RGNW")
printf "gz_crc min=%ss spread=%s%%\n" "$gmin" "$(sp "$GZW")"
printf "rg_crc min=%ss spread=%s%%\n" "$rcmin" "$(sp "$RGCW")"
printf "rg_nocrc min=%ss spread=%s%%\n" "$rnmin" "$(sp "$RGNW")"
printf "WALL RATIO gz/rg_crc = %s   (rg_crc/gz = %s)\n" \
  "$(awk -v g="$gmin" -v r="$rcmin" 'BEGIN{printf "%.3f",g/r}')" \
  "$(awk -v g="$gmin" -v r="$rcmin" 'BEGIN{printf "%.3f",r/g}')"
echo "DEF4ARM_DONE corpus=$(basename "$CORPUS") T=$T"
