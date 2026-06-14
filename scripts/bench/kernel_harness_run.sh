#!/usr/bin/env bash
# kernel_harness_run.sh — build + seed-capture + run the perf-counter engine
# isolation bench on a Linux x86_64 box. Self-contained: clones/fetches the
# perf/kernel-harness branch into a scratch repo, builds the gzippy bin (for seed
# capture) + the engine_isolation bench, captures a clean-window seed, then runs
# the bench pinned to one P-core with GZIPPY_KERNEL_PERF=1.
#
# Env:
#   SRC_REPO  = an existing local gzippy checkout to clone from (for vendor/corpus)
#   BRANCH    = perf/kernel-harness
#   PCORE     = physical core to pin (default 0). On the hybrid Intel chip this
#               MUST be a P-core or the cpu_core PMU reads 0.
set -u
BRANCH="${BRANCH:-perf/kernel-harness}"
REMOTE="${REMOTE:-origin}"
SRC_REPO="${SRC_REPO:-/root/gzippy}"
WORK="${WORK:-/root/gzippy-kernel-harness}"
PCORE="${PCORE:-0}"
FEATURES="pure-rust-inflate,isal-compression"
CARGO="${CARGO:-$HOME/.cargo/bin/cargo}"
[ -x "$CARGO" ] || CARGO="cargo"

say(){ echo "$@"; }
fail(){ echo "KH_FAILURE=$1"; echo "KERNEL_HARNESS_DONE"; exit 1; }

[ -d "$SRC_REPO" ] || fail "no-src-repo:$SRC_REPO"

# Use the SRC_REPO directly (it has vendor submodules + corpus); just sync branch.
cd "$SRC_REPO" || fail "cd-src"
git config --global --add safe.directory "$SRC_REPO" 2>/dev/null || true
git fetch "$REMOTE" "$BRANCH" >/tmp/kh_fetch.log 2>&1 || fail "fetch"
git checkout -f -B "$BRANCH" "$REMOTE/$BRANCH" >>/tmp/kh_fetch.log 2>&1 || fail "checkout"
git submodule update --init --recursive vendor >>/tmp/kh_fetch.log 2>&1 || true

CORPUS="$SRC_REPO/benchmark_data/silesia-gzip.tar.gz"
[ -f "$CORPUS" ] || fail "no-corpus:$CORPUS"

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
say "## host=$(hostname) arch=$(uname -m) core=$PCORE"
say "## rustc=$($CARGO --version 2>&1) head=$(git rev-parse --short HEAD)"
say "## perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo NA)"

say "## build gzippy bin ($FEATURES)..."
$CARGO build --release --no-default-features --features "$FEATURES" --bin gzippy >/tmp/kh_bin.log 2>&1 \
  || { tail -30 /tmp/kh_bin.log; fail "bin-build"; }
GZIPPY="$SRC_REPO/target/release/gzippy"

say "## build engine_isolation bench ($FEATURES)..."
$CARGO build --release --no-default-features --features "$FEATURES" --bench engine_isolation >/tmp/kh_bench.log 2>&1 \
  || { tail -40 /tmp/kh_bench.log; fail "bench-build"; }
BENCH_BIN="$(ls -t "$SRC_REPO"/target/release/deps/engine_isolation-* 2>/dev/null | grep -v '\.d$' | head -1)"
[ -x "$BENCH_BIN" ] || fail "no-bench-bin"
say "## bench bin: $BENCH_BIN"

# Capture a clean-window seed (one p=1 decode records start_bit -> window).
SEEDF=/dev/shm/engine.seed
rm -f "$SEEDF"
say "## capture seed -> $SEEDF"
env GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SEED_WINDOWS_CAPTURE="$SEEDF" \
  taskset -c "$PCORE" "$GZIPPY" -d -c -p 1 "$CORPUS" >/dev/null 2>/tmp/kh_cap.err || true
grep -m1 'SEED_WINDOWS_CAPTURE wrote' /tmp/kh_cap.err 2>/dev/null || say "## (no capture line)"
[ -f "$SEEDF" ] || { tail -10 /tmp/kh_cap.err; fail "no-seed"; }
ln -sf "$SEEDF" /tmp/engine.seed

say ""
say "## ============ KERNEL HARNESS RUN (GZIPPY_KERNEL_PERF=1) ============"
( cd "$SRC_REPO" && env GZIPPY_KERNEL_PERF=1 taskset -c "$PCORE" "$BENCH_BIN" ) 2>/tmp/kh_run.err
say "## ---- stderr (selftest notes etc) ----"
tail -20 /tmp/kh_run.err 2>/dev/null
echo "KERNEL_HARNESS_DONE"
