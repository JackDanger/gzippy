#!/usr/bin/env bash
# guest_engine_isolation.sh — §2.3 ENGINE ISOLATION BENCH on guest 199.
#
# RUNS ON GUEST 199 after host_lock_and_bench.sh gate PASS (freq pinned:
# no_turbo=1, governor=performance). Builds the standalone engine_isolation
# microbench (variants i scalar-u16 / ii E1-partial-u8 / iii ISA-L oracle),
# captures a guest seed (one clean known-window silesia chunk), runs it.
#
# The bench files (benches/engine_isolation.rs + the Cargo.toml/src/lib.rs
# deltas) are UNCOMMITTED on the leader's branch, so the host driver stages
# them under $STAGE and this script overlays them after syncing the repo.
set -u

REPO="${REPO:-/root/gzippy}"
BRANCH="${BRANCH:-reimplement-isa-l}"
STAGE="${STAGE:-/root/gzippy-bench/engine-bench-stage}"
ARTDIR="${ARTDIR:-/root/gzippy-bench/artifacts-engine-isolation}"
CORPUS="${CORPUS:-$REPO/benchmark_data/silesia-gzip.tar.gz}"
SEEDF="${SEEDF:-/dev/shm/engine.seed}"
FEATURES="pure-rust-inflate,isal-compression"

mkdir -p "$ARTDIR"
say() { echo "$@"; }

cd "$REPO" || { echo "ENGINE_FAILURE=no-repo"; echo "ENGINE_GUEST_DONE"; exit 5; }
git config --global --add safe.directory "$REPO" 2>/dev/null || true
git fetch origin "$BRANCH" >>"$ARTDIR/fetch.log" 2>&1 || true
git checkout -f -B "$BRANCH" "origin/$BRANCH" >>"$ARTDIR/fetch.log" 2>&1
git reset --hard "origin/$BRANCH" >>"$ARTDIR/fetch.log" 2>&1
git clean -fd -e vendor -e benchmark_data >>"$ARTDIR/fetch.log" 2>&1 || true

# Overlay the uncommitted bench files staged by the host driver.
[ -f "$STAGE/engine_isolation.rs" ] || { echo "ENGINE_FAILURE=no-staged-bench"; echo "ENGINE_GUEST_DONE"; exit 4; }
cp "$STAGE/engine_isolation.rs" "$REPO/benches/engine_isolation.rs"
cp "$STAGE/Cargo.toml" "$REPO/Cargo.toml"
cp "$STAGE/lib.rs" "$REPO/src/lib.rs"
say "overlaid bench files from $STAGE"

[ -f "$CORPUS" ] || { echo "ENGINE_FAILURE=corpus"; echo "ENGINE_GUEST_DONE"; exit 7; }

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
say "## freq: no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo NA) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
say "## rustc: $(rustc --version 2>&1) head=$(git rev-parse HEAD)"

# --- build the gzippy bin (for seed capture) + the bench, both native x86_64 ---
say "## build gzippy bin (${FEATURES}) ..."
if ! cargo build --release --no-default-features --features "${FEATURES}" --bin gzippy >"$ARTDIR/build-bin.log" 2>&1; then
  echo "ENGINE_FAILURE=bin-build"; tail -40 "$ARTDIR/build-bin.log"; echo "ENGINE_GUEST_DONE"; exit 8
fi
GZIPPY="$REPO/target/release/gzippy"

say "## build engine_isolation bench (${FEATURES}) ..."
if ! cargo build --release --no-default-features --features "${FEATURES}" --bench engine_isolation >"$ARTDIR/build-bench.log" 2>&1; then
  echo "ENGINE_FAILURE=bench-build"; tail -40 "$ARTDIR/build-bench.log"; echo "ENGINE_GUEST_DONE"; exit 8
fi
BENCH_BIN="$(ls -t "$REPO"/target/release/deps/engine_isolation-* 2>/dev/null | grep -v '\.d$' | head -1)"
[ -x "$BENCH_BIN" ] || { echo "ENGINE_FAILURE=no-bench-bin"; echo "ENGINE_GUEST_DONE"; exit 8; }
say "## bench bin: $BENCH_BIN"

# --- capture a guest seed (one p=1 decode records clean (start_bit -> window)) ---
sync; echo 3 >/proc/sys/vm/drop_caches 2>/dev/null || true
rm -f "$SEEDF"
say "## capture seed -> $SEEDF"
env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SEED_WINDOWS_CAPTURE="$SEEDF" \
  taskset -c 0 "$GZIPPY" -d -c -p 1 "$CORPUS" >/dev/null 2>"$ARTDIR/cap.err" || true
say "## $(grep -m1 'SEED_WINDOWS_CAPTURE wrote' "$ARTDIR/cap.err" || echo 'capture: no line')"
[ -f "$SEEDF" ] || { echo "ENGINE_FAILURE=no-seed"; tail -10 "$ARTDIR/cap.err"; echo "ENGINE_GUEST_DONE"; exit 10; }

# --- run the bench (single core, pinned), seed path overridden via env if supported ---
# The bench hardcodes /tmp/engine.seed; symlink the shm seed there for it.
ln -sf "$SEEDF" /tmp/engine.seed
sync; echo 3 >/proc/sys/vm/drop_caches 2>/dev/null || true
say ""
say "## ENGINE BENCH RUN (guest native x86_64, taskset core 0) ##"
( cd "$REPO" && taskset -c 0 "$BENCH_BIN" ) 2>"$ARTDIR/bench.err" | tee "$ARTDIR/bench.out"
say "## bench.err (selftest note etc):"
cat "$ARTDIR/bench.err" 2>/dev/null | tail -5

echo "ENGINE_GUEST_DONE"
