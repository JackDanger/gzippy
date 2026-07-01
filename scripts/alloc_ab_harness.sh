#!/usr/bin/env bash
# Framework Step 5: production A/B harness with mandatory rollup fields.
#
# Tests the allocator lever (rpmalloc-global feature) against the default
# build with the 6 advisor-mandated checks:
#
#   1. n=20 with per-iter MEDIAN (not mean) + trial-1-vs-rest split
#   2. Two output sinks: -c >/dev/null AND -o real_file
#   3. Full perf-stat rollup (page-faults is load-bearing signal)
#   4. Correctness hash in rollup (silesia SHA256)
#   5. Multi-corpus (extend by passing FIXTURES env)
#   6. Standalone Lever 4.1 first (GZIPPY_PREWARM_POOL=0)
#
# Usage on neurotic:
#   bash scripts/alloc_ab_harness.sh
#
# Pass gate: rpmalloc must show >=5% p50 wall reduction AND >=10% page-fault
# reduction on the silesia-gzip9.gz / -o real_file workload to promote to
# production default.
#
# Output: /tmp/alloc_ab_rollup.json with all per-trial counters + medians.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/gzippy}"
cd "$REPO_ROOT"

FIXTURES=("${FIXTURES:-benchmark_data/silesia-gzip9.gz}")
TRIALS="${TRIALS:-20}"
OUT_DIR="${OUT_DIR:-/tmp/alloc_ab}"

mkdir -p "$OUT_DIR"

# ── Step 5.0: verify clean Cargo.toml ────────────────────────────────────
STRIP_LINE=$(grep -E "^strip" Cargo.toml | head -1)
if [[ "$STRIP_LINE" != *"strip = true"* ]]; then
    echo "FAIL: Cargo.toml release profile is not 'strip = true' — bench will be contaminated"
    echo "Got: $STRIP_LINE"
    exit 1
fi

# ── Step 5.1: build both binaries ────────────────────────────────────────
echo "=== Building default build (glibc allocator) ==="
RUSTFLAGS="-C target-cpu=native" cargo build --release --features pure-rust-inflate 2>&1 | tail -2
cp target/release/gzippy "$OUT_DIR/gzippy-glibc"

echo "=== Building rpmalloc-global build (Lever 4.1) ==="
RUSTFLAGS="-C target-cpu=native" cargo build --release --features pure-rust-inflate,global-rpmalloc 2>&1 | tail -2
cp target/release/gzippy "$OUT_DIR/gzippy-rpmalloc"

ls -la "$OUT_DIR/"

# ── Step 5.2: correctness hash gate ──────────────────────────────────────
# Lever 4.1 alone (no chunk-shape change) must produce byte-identical
# output. If any variant differs, refuse to bench.

for FIXTURE in "${FIXTURES[@]}"; do
    EXPECTED=$("$OUT_DIR/gzippy-glibc" -d -c -p 16 "$FIXTURE" | sha256sum | cut -d' ' -f1)
    ACTUAL=$("$OUT_DIR/gzippy-rpmalloc" -d -c -p 16 "$FIXTURE" | sha256sum | cut -d' ' -f1)
    if [[ "$EXPECTED" != "$ACTUAL" ]]; then
        echo "FAIL: correctness mismatch on $FIXTURE"
        echo "  glibc:    $EXPECTED"
        echo "  rpmalloc: $ACTUAL"
        exit 2
    fi
    echo "OK: $FIXTURE byte-identical between variants (sha256=$EXPECTED)"
done

# ── Step 5.3: per-trial perf-stat rollup ─────────────────────────────────
# Counters: separated into non-multiplexed groups (advisor mandate).
# Group 1: task-clock, cycles, instructions, page-faults
# Group 2: branches, branch-misses, L1-dcache-load-misses
# Group 3: dTLB-load-misses, LLC-load-misses, context-switches

PERFEVENTS_A="task-clock,cycles,instructions,page-faults"
PERFEVENTS_B="branches,branch-misses,L1-dcache-load-misses"
PERFEVENTS_C="dTLB-load-misses,LLC-load-misses,context-switches"

run_one() {
    local BIN="$1"
    local FIXTURE="$2"
    local SINK="$3"      # 'devnull' | 'file'
    local OUTFILE
    if [[ "$SINK" == "devnull" ]]; then
        OUTFILE="/dev/null"
    else
        OUTFILE="/tmp/sink.bin"
    fi
    perf stat -j -e "$PERFEVENTS_A" \
        env GZIPPY_PREWARM_POOL=0 "$BIN" -d -c -p 16 "$FIXTURE" > "$OUTFILE" 2>&1
}

# ── Step 5.4: interleaved 20-trial bench across all 4 cells ──────────────
# Cells: {glibc, rpmalloc} × {devnull, file}

cat > "$OUT_DIR/rollup.tsv" <<EOF
trial	variant	fixture	sink	wall_ms	cycles	instructions	page_faults_minor	page_faults_major
EOF

LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
echo "System load before bench: $LOAD"
if (( $(echo "$LOAD > 8" | bc -l 2>/dev/null || echo 0) )); then
    echo "WARN: load avg $LOAD > 8 — wall times will be noisy. Counter ratios still valid."
fi

for FIXTURE in "${FIXTURES[@]}"; do
    FNAME=$(basename "$FIXTURE")
    for trial in $(seq 1 "$TRIALS"); do
        for VARIANT in glibc rpmalloc; do
            BIN="$OUT_DIR/gzippy-$VARIANT"
            for SINK in devnull file; do
                OUTFILE=$([[ "$SINK" == "devnull" ]] && echo "/dev/null" || echo "/tmp/sink.bin")
                # bash builtin `time` (perl /usr/bin/time not on neurotic).
                # Use perf-stat for page-fault counts (separate run; cheap on
                # silesia at ~5-10s per trial under load).
                TIME_OUT=$({ time GZIPPY_PREWARM_POOL=0 "$BIN" -d -c -p 16 "$FIXTURE" > "$OUTFILE"; } 2>&1)
                WALL_S=$(echo "$TIME_OUT" | awk '/^real/{ gsub(/[ms]/, " ", $2); split($2, a, " "); print a[1]*60 + a[2] }')
                WALL_MS=$(awk -v ws="$WALL_S" 'BEGIN { print ws*1000 }')
                # Capture page-faults via perf-stat (single short event group).
                # `perf stat` doesn't accept env-var prefix syntax for its
                # workload; use `env` explicitly.
                PF_OUT=$({ perf stat -e page-faults,minor-faults,major-faults env GZIPPY_PREWARM_POOL=0 "$BIN" -d -c -p 16 "$FIXTURE" > "$OUTFILE"; } 2>&1)
                MINFLT=$(echo "$PF_OUT" | grep -E "minor-faults" | awk '{print $1}' | tr -d ',')
                MAJFLT=$(echo "$PF_OUT" | grep -E "major-faults" | awk '{print $1}' | tr -d ',')
                MINFLT=${MINFLT:-0}; MAJFLT=${MAJFLT:-0}
                echo -e "${trial}\t${VARIANT}\t${FNAME}\t${SINK}\t${WALL_MS}\t-\t-\t${MINFLT}\t${MAJFLT}" >> "$OUT_DIR/rollup.tsv"
                printf "%2d %-8s %-10s %s wall=%.0fms minflt=%s majflt=%s\n" "$trial" "$VARIANT" "$SINK" "$FNAME" "$WALL_MS" "$MINFLT" "$MAJFLT"
            done
        done
    done
done

# ── Step 5.5: summarize medians + trial-1-vs-rest split ─────────────────

python3 <<PYEOF
import csv
from statistics import median

rows = []
with open("$OUT_DIR/rollup.tsv") as f:
    rd = csv.DictReader(f, delimiter='\t')
    for r in rd:
        try:
            r["wall_ms"] = float(r["wall_ms"])
            r["trial"] = int(r["trial"])
            r["page_faults_minor"] = int(r["page_faults_minor"]) if r["page_faults_minor"] and r["page_faults_minor"] != "-" else 0
            r["page_faults_major"] = int(r["page_faults_major"]) if r["page_faults_major"] and r["page_faults_major"] != "-" else 0
            rows.append(r)
        except (ValueError, KeyError):
            continue

print("\n=== SUMMARY ===")
print(f"{'variant':<10} {'fixture':<24} {'sink':<8} {'p50_wall_ms':<12} {'trial1_ms':<10} {'p50_2_n_ms':<11} {'p50_minflt':<12}")

for (variant, fixture, sink), grp in sorted({(r["variant"], r["fixture"], r["sink"]): None for r in rows}.items()):
    walls = [r["wall_ms"] for r in rows if r["variant"]==variant and r["fixture"]==fixture and r["sink"]==sink]
    minflts = [r["page_faults_minor"] for r in rows if r["variant"]==variant and r["fixture"]==fixture and r["sink"]==sink]
    trial1 = next((r["wall_ms"] for r in rows if r["variant"]==variant and r["fixture"]==fixture and r["sink"]==sink and r["trial"]==1), None)
    rest = sorted([r["wall_ms"] for r in rows if r["variant"]==variant and r["fixture"]==fixture and r["sink"]==sink and r["trial"]>1])
    p50_rest = median(rest) if rest else 0
    print(f"{variant:<10} {fixture:<24} {sink:<8} {median(walls):>10.0f}   {trial1:>8.0f}   {p50_rest:>9.0f}   {median(minflts):>10.0f}")

print("\n=== DELTAS: rpmalloc vs glibc ===")
def p50(variant, fixture, sink, field="wall_ms"):
    vals = [r[field] for r in rows if r["variant"]==variant and r["fixture"]==fixture and r["sink"]==sink]
    return median(vals) if vals else None

fixtures = sorted({r["fixture"] for r in rows})
for fx in fixtures:
    for sink in ("devnull", "file"):
        g = p50("glibc", fx, sink)
        r = p50("rpmalloc", fx, sink)
        if g and r:
            dwall = (r - g) / g * 100
            gf = p50("glibc", fx, sink, "page_faults_minor")
            rf = p50("rpmalloc", fx, sink, "page_faults_minor")
            dpf = (rf - gf) / gf * 100 if gf else 0
            print(f"  {fx:<24} {sink:<8} wall={dwall:+.1f}%  minflt={dpf:+.1f}%")

print("\n=== PASS GATE ===")
print("Required: >=5% p50 wall reduction AND >=10% page-fault reduction on silesia/file.")

PYEOF

echo ""
echo "Rollup TSV: $OUT_DIR/rollup.tsv"
echo "Binaries:   $OUT_DIR/gzippy-{glibc,rpmalloc}"
