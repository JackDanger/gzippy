#!/usr/bin/env sh
# _perf_attr_guest.sh — STEP 1 of the LOCATE-+40% charter: ATTRIBUTE gzippy-isal's
# +40% instruction excess (DIS-17: 7.28e9 vs rg 5.18e9 at T4) to specific gzippy
# PIPELINE functions, via a SYMBOLED build + perf record -e instructions.
#
# The shipped bench binary is STRIPPED (Cargo.toml profile.release strip=true), so
# it cannot be sliced by symbol. Here we rebuild gzippy-isal with PRODUCTION codegen
# (fat LTO, codegen-units=1, opt-level 3 — instruction count UNCHANGED) but RETAIN
# debug symbols (RUSTFLAGS -C strip=none -C debuginfo=2). Then:
#   - perf stat -e instructions (gz vs rg) to confirm the +40% reproduces SYMBOLED;
#   - perf record -e instructions (user) for per-SYMBOL self-instruction share;
#   - dso split (isolate the ISA-L nasm kernel vs the rust pipeline);
#   - inline-aware caller tree (DWARF) to NAME the inlined pipeline regions;
#   - perf annotate (source lines) of the top gzippy symbols to attribute within
#     the LTO blob.
# Read-only on the corpus; regular-file sink in /dev/shm; sha-verified.
#
# Env (exported by the local wrapper perf_attr.sh): GUEST_SRC CORPUS REFSHA RG
#   MASK T REPS PERIOD ART CARGO_LOCK
set -u
: "${GUEST_SRC:?}"; : "${CORPUS:?}"; : "${MASK:?}"; : "${T:?}"
REPS="${REPS:-6}"; PERIOD="${PERIOD:-300000}"; ART="${ART:-/dev/shm/perfattr}"
REFSHA="${REFSHA:-}"; RG="${RG:-rapidgzip}"; CARGO_LOCK="${CARGO_LOCK:-scripts/cargo-lock.sh}"
RUSTFLAGS_SYM="-C target-cpu=native -C strip=none -C debuginfo=2"
mkdir -p "$ART"; SINK="$ART/sink.bin"; rm -f "$SINK"
cd "$GUEST_SRC" || { echo "PERFATTR_FAIL no-src:$GUEST_SRC"; exit 5; }

echo "################ PERF-ATTR T=$T MASK=$MASK REPS=$REPS PERIOD=$PERIOD ################"

# ---- 1. BUILD symboled gzippy-isal (production codegen + symbols) ------------
echo "=== df before build ==="; df -h / | tail -1
echo "=== BUILD gzippy-isal symboled (RUSTFLAGS='$RUSTFLAGS_SYM') ==="
if [ -x "$CARGO_LOCK" ]; then BUILDER="sh $CARGO_LOCK"; else BUILDER=""; fi
# shellcheck disable=SC2086
RUSTFLAGS="$RUSTFLAGS_SYM" $BUILDER cargo build --release --no-default-features \
  --features gzippy-isal > "$ART/build.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then echo "PERFATTR_FAIL build rc=$rc"; grep -E 'error' "$ART/build.log" | head -25; exit 8; fi
grep -E 'Finished|Compiling gzippy ' "$ART/build.log" | tail -2 | sed 's/^/   /'
echo "=== df after build ==="; df -h / | tail -1
GZ="$GUEST_SRC/target/release/gzippy"
cp -f "$GZ" "$ART/gzippy-isal-sym" 2>/dev/null || true
file "$GZ" 2>/dev/null | sed 's/^/   /'
echo "   symbols=$(nm "$GZ" 2>/dev/null | wc -l)"

# ---- 2. ASSERT production path + isal_chunks>=14 -----------------------------
echo "=== ASSERT path=ParallelSM + isal_chunks>=14 ==="
DBG="$(GZIPPY_DEBUG=1 GZIPPY_VERBOSE=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null)"
echo "$DBG" | grep -E 'path=|isal_chunks' | sed 's/^/   DBG /'
echo "$DBG" | grep -q 'ParallelSM' || { echo "PERFATTR_FAIL routing-not-ParallelSM"; exit 9; }
ICH="$(echo "$DBG" | sed -n 's/.*isal_chunks=\([0-9]*\).*/\1/p' | head -1)"
IFB="$(echo "$DBG" | sed -n 's/.*isal_fallbacks=\([0-9]*\).*/\1/p' | head -1)"
echo "   ASSERT isal_chunks=${ICH:-?} isal_fallbacks=${IFB:-?}"
[ "${ICH:-0}" -ge 14 ] || { echo "PERFATTR_FAIL isal_chunks=${ICH:-0}<14"; exit 10; }
[ "${IFB:-1}" = "0" ] || echo "   WARN isal_fallbacks=${IFB} (!=0 — a clean tail fell back)"

# ---- pick a perf instructions event that works on this (hybrid) PMU ----------
EV="instructions:u"
if perf stat -e cpu_core/instructions/ -- true >/dev/null 2>&1; then EV="cpu_core/instructions/u"; fi
echo "=== perf event = $EV ==="

# ---- 3. perf stat instruction count: confirm +40% reproduces symboled --------
echo "=== perf stat instructions (REPS=$REPS): gz vs rg ==="
# shellcheck disable=SC2086
perf stat -r "$REPS" -e "$EV",cpu_core/cycles/ -o "$ART/gz.stat" -- \
  env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p "$T" "$CORPUS" >"$SINK" 2>>"$ART/run.err"
gzsha="$(sha256sum "$SINK" | cut -d' ' -f1)"
[ -n "$REFSHA" ] && [ "$gzsha" != "$REFSHA" ] && echo "   !! GZ SHA MISMATCH gz=$gzsha ref=$REFSHA"
# shellcheck disable=SC2086
perf stat -r "$REPS" -e "$EV",cpu_core/cycles/ -o "$ART/rg.stat" -- \
  taskset -c "$MASK" "$RG" -d -c -f -P "$T" "$CORPUS" >"$SINK" 2>>"$ART/run.err"
rgsha="$(sha256sum "$SINK" | cut -d' ' -f1)"
[ -n "$REFSHA" ] && [ "$rgsha" != "$REFSHA" ] && echo "   !! RG SHA MISMATCH rg=$rgsha ref=$REFSHA"
echo "--- gz.stat ---"; grep -E 'instructions|cycles|elapsed' "$ART/gz.stat" | sed 's/^/   /'
echo "--- rg.stat ---"; grep -E 'instructions|cycles|elapsed' "$ART/rg.stat" | sed 's/^/   /'

# ---- 4. perf record (no callgraph): clean per-symbol SELF attribution --------
echo "=== perf record $EV -c $PERIOD (self-attribution) ==="
# shellcheck disable=SC2086
perf record -e "$EV" -c "$PERIOD" -o "$ART/gz.self.data" -- \
  env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p "$T" "$CORPUS" >"$SINK" 2>>"$ART/run.err"
rsha="$(sha256sum "$SINK" | cut -d' ' -f1)"
[ -n "$REFSHA" ] && [ "$rsha" != "$REFSHA" ] && echo "   !! REC SHA MISMATCH $rsha"
echo "--- event count ---"; perf report -i "$ART/gz.self.data" --stdio 2>/dev/null | grep -iE 'event count|samples' | head -3 | sed 's/^/   /'
echo "=== DSO split (isolate ISA-L nasm kernel vs rust pipeline vs libc/kernel) ==="
perf report -i "$ART/gz.self.data" --stdio -n --no-children --sort dso 2>/dev/null | grep -vE '^#|^$' | head -15
echo "=== PER-SYMBOL self instruction share (top 45) ==="
perf report -i "$ART/gz.self.data" --stdio -n --no-children --sort symbol 2>/dev/null | grep -vE '^#|^$' | head -45

# ---- 5. perf record (DWARF callgraph): inline-aware NAMING of pipeline regions
echo "=== perf record $EV -c $((PERIOD*3)) --call-graph dwarf (inline naming) ==="
# shellcheck disable=SC2086
perf record -e "$EV" -c "$((PERIOD*3))" --call-graph dwarf,16384 -o "$ART/gz.dwarf.data" -- \
  env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c "$MASK" "$GZ" -d -c -p "$T" "$CORPUS" >"$SINK" 2>>"$ART/run.err"
echo "=== INLINE-AWARE self share (--inline, top 60) — names inlined fns inside LTO blobs ==="
perf report -i "$ART/gz.dwarf.data" --stdio -n --no-children --inline --sort symbol 2>/dev/null | grep -vE '^#|^$' | head -60

# ---- 6. annotate the top gzippy (non-ISA-L) symbols by SOURCE LINE -----------
echo "=== SOURCE-LINE annotate of top gzippy symbols ==="
TOPSYMS="$(perf report -i "$ART/gz.self.data" --stdio -n --no-children --sort symbol 2>/dev/null \
  | grep -vE '^#|^$' \
  | grep -viE 'decode_huffman_code_block_stateless|\[unknown\]|libc|\bmemcpy\b|memmove|memset|__memmove|kallsyms' \
  | awk '{ for(i=1;i<=NF;i++) if($i=="[.]"){ s=""; for(j=i+1;j<=NF;j++) s=s (j>i+1?" ":"") $j; print s; break } }' \
  | head -4)"
echo "   top gzippy symbols to annotate:"; echo "$TOPSYMS" | sed 's/^/     /'
OLDIFS="$IFS"; IFS='
'
for sym in $TOPSYMS; do
  [ -z "$sym" ] && continue
  echo "----- annotate: $sym -----"
  perf annotate -i "$ART/gz.self.data" --stdio --no-source --symbol="$sym" 2>/dev/null \
    | grep -E '^\s+[0-9]+\.[0-9]+ :|\.rs:[0-9]' | sort -rn | head -18 | sed 's/^/   /'
done
IFS="$OLDIFS"

rm -f "$SINK"
echo "PERFATTR_DONE"
