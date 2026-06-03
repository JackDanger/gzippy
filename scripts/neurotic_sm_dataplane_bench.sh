#!/usr/bin/env bash
# neurotic_sm_dataplane_bench.sh — wall + trace + fulcrum on guest 199 (via neurotic jump).
#
# Run from the laptop after pushing reimplement-isa-l:
#   scripts/neurotic_sm_dataplane_bench.sh
#
# Exercises production parallel-SM (pure-rust-inflate, GZIPPY_FORCE_PARALLEL_SM=1)
# on silesia-large.gz at T8 with interleaved sha-verified wall times, optional
# writev A/B (GZIPPY_DISABLE_WRITEV), and fulcrum vs rapidgzip when fulcrum is
# on PATH and traces were captured (GZIPPY_TIMELINE).
set -euo pipefail

# shellcheck disable=SC2128
NEUROTIC_SSH=(ssh -o ConnectTimeout=15 -J neurotic root@10.30.0.199)
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
ROUNDS="${ROUNDS:-7}"
# Pinned P-cores for T8 (TESTING.md headline cell). Not passed via ssh argv — commas/spaces split.
PIN_MASK="${PIN_MASK:-0,2,4,6,8,10,12,14}"

if ! git rev-parse "origin/${BRANCH}" >/dev/null 2>&1 \
  || [ -n "$(git log "origin/${BRANCH}"..HEAD 2>/dev/null || true)" ]; then
  echo "Pushing ${BRANCH} to origin..."
  git push -u origin "${BRANCH}"
fi

echo "=== neurotic SM dataplane bench (branch ${BRANCH}) ==="
"${NEUROTIC_SSH[@]}" bash -s "${BRANCH}" "${ROUNDS}" "${PIN_MASK}" <<'REMOTE'
set -euo pipefail
BRANCH="$1"
ROUNDS="$2"
PIN_MASK="$3"
PIN=(taskset -c "$PIN_MASK")
cd /root/gzippy
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
echo "Building gzippy (pure-rust-inflate)..."
cargo build --release --no-default-features --features pure-rust-inflate 2>&1 \
  | grep -E 'Compiling gzippy |Finished|error' || true

RG=vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip
if [ ! -x "$RG" ]; then
  echo "Building rapidgzip..."
  (cd vendor/rapidgzip/librapidarchive/build && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j"$(nproc)" rapidgzip)
fi

BD=benchmark_data
SL="$BD/silesia-large.bin"
SLG="$BD/silesia-large.gz"
[ -s "$SL" ] || { cat "$BD/silesia.tar" "$BD/silesia.tar" > "$SL"; head -c $((76*1024*1024)) "$BD/silesia.tar" >> "$SL"; }
[ -s "$SLG" ] || gzip -9 -c "$SL" > "$SLG"

GZ=target/release/gzippy
export GZIPPY_FORCE_PARALLEL_SM=1
export GZIPPY_DEBUG=1

echo "Routing check (must show path=IsalParallelSM):"
GZIPPY_DEBUG=1 "${PIN[@]}" "$GZ" -d -c -p 8 "$SLG" >/dev/null 2>&1 | grep -E 'path=|IsalParallelSM' || true

bench_wall() {
  local label="$1"
  shift
  local -a extra=("$@")
  echo ""
  echo "── WALL $label (interleaved best-of-${ROUNDS}, taskset -c ${PIN_MASK}) ──"
  ref=$(gzip -dc "$SLG" | sha256sum | awk '{print $1}')
  echo "reference sha256: $ref"
  gb=99; rb=99
  for _t in $(seq 1 "$ROUNDS"); do
    s=$(date +%s.%N)
    GZIPPY_DEBUG=1 "${extra[@]}" "${PIN[@]}" "$GZ" -d -c -p 8 "$SLG" >/dev/null
    e=$(date +%s.%N); g=$(awk "BEGIN{print $e-$s}")
    s=$(date +%s.%N)
    "${PIN[@]}" "$RG" -d -c -f -P 8 "$SLG" >/dev/null
    e=$(date +%s.%N); r=$(awk "BEGIN{print $e-$s}")
    gb=$(awk -v a="$gb" -v b="$g" 'BEGIN{print (b<a)?b:a}')
    rb=$(awk -v a="$rb" -v b="$r" 'BEGIN{print (b<a)?b:a}')
    out_sha=$("${PIN[@]}" "${extra[@]}" "$GZ" -d -c -p 8 "$SLG" 2>/dev/null | sha256sum | awk '{print $1}')
    [ "$out_sha" = "$ref" ] || { echo "SHA MISMATCH $label"; exit 1; }
  done
  out=$(gzip -dc "$SLG" | wc -c)
  awk -v g="$gb" -v r="$rb" -v out="$out" \
    'BEGIN{printf "  gzippy best=%.3fs (%.0f MB/s)  rapidgzip best=%.3fs (%.0f MB/s)  ratio=%.3fx\n",
      g,out/g/1e6,r,out/r/1e6,g/r}'
}

bench_wall "default (/dev/null sink)"
bench_wall "writev-off" GZIPPY_DISABLE_WRITEV=1

echo ""
echo "── GZIPPY_TIMELINE capture (T8, one run each) ──"
rm -f /tmp/gz-timeline-T8.json /tmp/rg-timeline-T8.json
GZIPPY_TIMELINE=/tmp/gz-timeline-T8.json GZIPPY_FORCE_PARALLEL_SM=1 \
  "${PIN[@]}" "$GZ" -d -c -p 8 "$SLG" >/dev/null 2>/dev/null || true
RG_TRACE=vendor/rapidgzip/librapidarchive/build-trace/src/tools/rapidgzip
if [ -x "$RG_TRACE" ]; then
  GZIPPY_TIMELINE=/tmp/rg-timeline-T8.json "${PIN[@]}" "$RG_TRACE" -d -c -f -P 8 "$SLG" >/dev/null 2>/dev/null || true
else
  echo "  (no build-trace rapidgzip — skip rg timeline; run scripts/rapidgzip_trace_patch on guest)"
fi
ls -la /tmp/gz-timeline-T8.json /tmp/rg-timeline-T8.json 2>/dev/null || true
if command -v fulcrum >/dev/null 2>&1 && [ -s /tmp/gz-timeline-T8.json ] && [ -s /tmp/rg-timeline-T8.json ]; then
  echo ""
  echo "── fulcrum vs (guest) ──"
  fulcrum vs /tmp/gz-timeline-T8.json /tmp/rg-timeline-T8.json --labels gzippy,rapidgzip 2>/dev/null | head -40 || true
  fulcrum flow /tmp/gz-timeline-T8.json 2>/dev/null | head -25 || true
elif command -v fulcrum >/dev/null 2>&1 && [ -s /tmp/gz-timeline-T8.json ]; then
  fulcrum vs /tmp/gz-timeline-T8.json /tmp/gz-timeline-T8.json --labels gzippy,gzippy-self 2>/dev/null | head -20 || true
fi

echo ""
echo "Done. commit=$(git rev-parse --short HEAD)"
REMOTE

TRACE_DIR="${TRACE_DIR:-/tmp/gzippy-neurotic-traces}"
mkdir -p "$TRACE_DIR"
echo ""
echo "── Fetching timelines to ${TRACE_DIR} ──"
"${NEUROTIC_SSH[@]}" "cat /tmp/gz-timeline-T8.json 2>/dev/null" >"${TRACE_DIR}/gz-timeline-T8.json" || true
"${NEUROTIC_SSH[@]}" "cat /tmp/rg-timeline-T8.json 2>/dev/null" >"${TRACE_DIR}/rg-timeline-T8.json" || true

if command -v fulcrum >/dev/null 2>&1; then
  if [ -s "${TRACE_DIR}/gz-timeline-T8.json" ] && [ -s "${TRACE_DIR}/rg-timeline-T8.json" ]; then
    echo ""
    echo "── fulcrum vs (laptop) ──"
    fulcrum vs "${TRACE_DIR}/gz-timeline-T8.json" "${TRACE_DIR}/rg-timeline-T8.json" \
      --labels gzippy,rapidgzip 2>/dev/null | head -50 || true
    echo ""
    fulcrum flow "${TRACE_DIR}/gz-timeline-T8.json" 2>/dev/null | head -30 || true
  elif [ -s "${TRACE_DIR}/gz-timeline-T8.json" ]; then
    echo "fulcrum: only gzippy timeline (no build-trace rapidgzip on guest)"
    fulcrum flow "${TRACE_DIR}/gz-timeline-T8.json" 2>/dev/null | head -30 || true
  fi
else
  echo "fulcrum not on laptop PATH — traces in ${TRACE_DIR}"
fi
