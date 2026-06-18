#!/usr/bin/env bash
# _distpreload_cycbyte_guest.sh — single-core, FREQUENCY-PINNED, best-of-N
# cycles-per-byte A/B for the asm_kernel dist-entry preload.
#
#   arm A = 8383a2eb (baseline: 16-byte MOVDQU back-ref copy)
#   arm B = 2b10aa48 (dist-preload: speculative dist-entry preload)
#
# WHY THIS IS A SOUND GATED SIGNAL ON A LOADED BOX (two decorrelated reviews):
# cyc/byte is a FREQUENCY-INDEPENDENT COUNT (cycles is a counter, not wall time),
# so an interleaved best-of-N cyc/byte A/B on a single isolated core is a valid
# signal for an INNER-KERNEL IPC/branch change even under box load. HARD SCOPE
# LIMIT: this settles the T1 inner-kernel question ONLY — NOT the T4/T8 wall
# (bus/coherence/lock contention dominate there). The two arms differ ONLY in
# src/decompress/parallel/asm_kernel.rs (run_contig); every other phase
# (block-find, marker resolve, apply_window, CRC) is byte-identical, so the
# whole-program Δcycles is attributable to run_contig.
#
# BOX REALITY (measured 2026-06-18): the bench box is an LXC CONTAINER — no_turbo,
# cpufreq governor, and cpuN/online are HOST-MANAGED and READ-ONLY from inside, so
# hardware frequency-pinning / SMT-sibling-offlining are IMPOSSIBLE here (and the
# user forbids freezing the host: Plex runs on a neighbor LXC). The SOUND
# SUBSTITUTE, used here: taskset-isolate to one core + INTERLEAVE arms (A,B see the
# same freq regime) + best-of-N (min-cycles ≈ least-interrupted run) + a MEASURED
# GHz-STABILITY GATE (reject if achieved-GHz spread is large) + the self-test gate.
# Because cyc/byte is a count, we VERIFY frequency was stable rather than FORCE it.
# This script makes NO host mutations, so there is nothing to restore.
#
# GATE-0 SELF-VALIDATION (printed FIRST; a run failing any of these is VOID):
#   (a) CONSUMER CONFIRMED  — run_contig actually runs: KERN_ENTRIES>0 each arm
#       (needs GZIPPY_VERBOSE=1 + GZIPPY_ASM_STATS=1 — the dump is VERBOSE-gated).
#   (b) COMPARATOR SELF-TEST — baseline-vs-baseline (A2/A1) ratio == 1.000 ± spread.
#   (c) NON-INERT          — sha(BIN_A) != sha(BIN_B) (arms are different binaries).
#   (d) FREQ STABLE        — measured achieved-GHz spread small (gate in analyzer).
#   (e) NOT MEMORY-BOUND   — LLC-miss rate reported (single-core isolation confounder).
#
# Usage: PIN=4 REPS=15 CORPORA="silesia monorepo" _distpreload_cycbyte_guest.sh
set -u

BIN_A=${BIN_A:-/root/gz-baseline/target/release/gzippy}     # 8383a2eb baseline
BIN_B=${BIN_B:-/root/gz-fullrewrite/target/release/gzippy}  # 2b10aa48 dist-preload
PIN=${PIN:-4}            # isolated core to pin the kernel on (P-core, prod-relevant)
REPS=${REPS:-15}
CORPORA=${CORPORA:-"silesia monorepo"}
OUT=/tmp/distpreload
ANALYZE="$(dirname "$0")/_distpreload_cycbyte_analyze.py"
rm -rf "$OUT"; mkdir -p "$OUT"

echo "================ DIST-PRELOAD cyc/byte A/B ================"
echo "BIN_A(8383a2eb)=$BIN_A  sha=$(sha256sum $BIN_A 2>/dev/null|cut -c1-16)"
echo "BIN_B(2b10aa48)=$BIN_B  sha=$(sha256sum $BIN_B 2>/dev/null|cut -c1-16)"
echo "pin=cpu$PIN reps=$REPS corpora='$CORPORA'"
echo "load:$(cat /proc/loadavg)"

# ---------- GATE-0 (c): non-inert (different binaries) ----------
SHA_A=$(sha256sum "$BIN_A" | cut -c1-16); SHA_B=$(sha256sum "$BIN_B" | cut -c1-16)
echo "--- GATE0(c) NON-INERT: sha_A=$SHA_A sha_B=$SHA_B  $([ "$SHA_A" != "$SHA_B" ] && echo PASS || echo FAIL-SAME-BINARY)"

# ---------- GATE-0 (d): box freq state (informational; gate is measured-GHz) ----------
echo "--- GATE0(d) FREQ STATE (LXC: host-managed/read-only; gate = measured-GHz spread, see analyzer):"
echo "    no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)  governor cpu$PIN=$(cat /sys/devices/system/cpu/cpu$PIN/cpufreq/scaling_governor 2>/dev/null)"
echo "    online mask=$(cat /sys/devices/system/cpu/online)"

# ---------- GATE-0 (a): kernel engaged (KERN_ENTRIES>0), both arms ----------
echo "--- GATE0(a) CONSUMER (run_contig engaged):"
FIRST=$(echo $CORPORA | awk '{print $1}'); FF=/root/$FIRST.gz
for tag in A B; do
  B_=$([ $tag = A ] && echo "$BIN_A" || echo "$BIN_B")
  GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1 GZIPPY_FORCE_PARALLEL_SM=1 \
    taskset -c $PIN "$B_" -d -c -p1 "$FF" >/dev/null 2>"$OUT/g0a.$tag"
  L=$(grep "asm-kernel:c" "$OUT/g0a.$tag" | tail -1)
  ENTR=$(echo "$L" | sed -n 's/.*entries=\([0-9]*\).*/\1/p')
  ASMB=$(echo "$L" | sed -n 's/.*asm_bytes=\([0-9]*\).*/\1/p')
  PATH_=$(grep -oE "path=[A-Za-z]+" "$OUT/g0a.$tag" | head -1)
  echo "    arm$tag $PATH_ entries=${ENTR:-0} asm_bytes=${ASMB:-0}  $([ "${ENTR:-0}" -gt 0 ] && echo KERN_OK || echo KERN_ZERO)"
done

# ---------- GATE-0 sha-correctness (Rule 4) ----------
REF=$(zcat "$FF" | sha256sum | cut -c1-16)
OA=$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$BIN_A" -d -c -p1 "$FF" 2>/dev/null | sha256sum | cut -c1-16)
OB=$(GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$BIN_B" -d -c -p1 "$FF" 2>/dev/null | sha256sum | cut -c1-16)
echo "--- GATE0 SHA: ref=$REF A=$OA B=$OB  $([ "$REF" = "$OA" ] && [ "$REF" = "$OB" ] && echo SHA_OK || echo SHA_MISMATCH)"

# ---------- MEASUREMENT: interleaved best-of-N, single-run perf stat CSV ----------
EVENTS="cpu_core/instructions/,cpu_core/cycles/,cpu_core/branches/,cpu_core/branch-misses/,cpu_core/cache-references/,cpu_core/cache-misses/,task-clock"
echo "--- measuring (interleaved A1,A2,B per rep; A2=self-test twin of A) ---"
: > "$OUT/bytes.txt"
for corp in $CORPORA; do
  F=/root/$corp.gz
  [ -f "$F" ] || { echo "  SKIP $corp (no $F)"; continue; }
  BYTES=$(zcat "$F" | wc -c)
  echo "$corp $BYTES" >> "$OUT/bytes.txt"
  echo "  $corp raw_bytes=$BYTES"
  for r in $(seq 1 $REPS); do
    for arm in A1 A2 B; do
      case $arm in A1|A2) BIN=$BIN_A;; B) BIN=$BIN_B;; esac
      taskset -c $PIN perf stat -x, -e "$EVENTS" \
        -- env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c $PIN "$BIN" -d -c -p1 "$F" \
        >/dev/null 2>"$OUT/$corp.$arm.$r.csv"
    done
  done
done

echo "--- ANALYSIS ---"
python3 "$ANALYZE" "$OUT" $CORPORA
echo "DONE_DISTPRELOAD_CYCBYTE"
