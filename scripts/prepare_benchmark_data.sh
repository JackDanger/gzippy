#!/bin/bash
set -e

# =============================================================================
# Benchmark Data Preparation
# =============================================================================
# Single source of truth for preparing benchmark data used by:
#   - ./bench-decompress.sh
#   - CI benchmarks workflow
#   - cargo test bench_*
#
# Benchmark datasets:
#   1. silesia (~212MB) - Standard compression benchmark corpus (mixed binary/text)
#      Downloaded from https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip
#
#   2. silesia-large (~500MB) - Silesia repeated ~2.5x for parallel benchmark scaling
#
#   3. software (~211MB) - Synthetic source code patterns (dense LZ77 matches)
#
#   4. logs (~211MB) - Synthetic log data (highly repetitive)
#
# The compressed versions (.gz) are checked into git for caching.
# Run this script to regenerate uncompressed versions from the cached .gz files.
# =============================================================================

BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark_data}"
SILESIA_URL="https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip"

# File paths
SILESIA_ZIP="$BENCHMARK_DIR/silesia.zip"
SILESIA_TAR="$BENCHMARK_DIR/silesia.tar"
SILESIA_XZ="$BENCHMARK_DIR/silesia.tar.xz"
SILESIA_GZ="$BENCHMARK_DIR/silesia-gzip.tar.gz"
SILESIA_LARGE="$BENCHMARK_DIR/silesia-large.bin"
SILESIA_LARGE_GZ="$BENCHMARK_DIR/silesia-large.gz"
SOFTWARE_ARCHIVE="$BENCHMARK_DIR/software.archive"
SOFTWARE_GZ="$BENCHMARK_DIR/software.archive.gz"
LOGS_TXT="$BENCHMARK_DIR/logs.txt"
LOGS_GZ="$BENCHMARK_DIR/logs.txt.gz"

# Create directory
mkdir -p "$BENCHMARK_DIR"

# -----------------------------------------------------------------------------
# Helper: Download silesia corpus
# -----------------------------------------------------------------------------
download_silesia() {
    if [[ -f "$SILESIA_TAR" ]]; then
        echo "[OK] silesia.tar exists"
        return 0
    fi

    # Try to decompress from cached .xz first (47MB, checked into git)
    if [[ -f "$SILESIA_XZ" ]]; then
        echo "[CACHE] Decompressing silesia.tar from cached silesia.tar.xz..."
        xz -d -k -c "$SILESIA_XZ" > "$SILESIA_TAR"
        return 0
    fi

    echo "[DOWNLOAD] Downloading silesia corpus..."

    if [[ ! -f "$SILESIA_ZIP" ]]; then
        curl -L -o "$SILESIA_ZIP" "$SILESIA_URL"
    fi

    echo "[EXTRACT] Extracting silesia.zip..."
    EXTRACT_DIR="$BENCHMARK_DIR/silesia_extract"
    mkdir -p "$EXTRACT_DIR"
    unzip -o "$SILESIA_ZIP" -d "$EXTRACT_DIR"

    echo "[TAR] Creating silesia.tar..."
    tar -cf "$SILESIA_TAR" -C "$EXTRACT_DIR" .

    # Cleanup
    rm -rf "$EXTRACT_DIR"

    # Create xz compressed version for git cache (47MB vs 65MB gzip)
    if [[ ! -f "$SILESIA_XZ" ]]; then
        echo "[COMPRESS] Creating silesia.tar.xz for git cache..."
        xz -9 -k -c "$SILESIA_TAR" > "$SILESIA_XZ"
    fi

    # Create gzip version for decompression benchmarks
    if [[ ! -f "$SILESIA_GZ" ]]; then
        echo "[COMPRESS] Creating silesia-gzip.tar.gz for benchmarks..."
        gzip -9 -c "$SILESIA_TAR" > "$SILESIA_GZ"
    fi
}

# -----------------------------------------------------------------------------
# Helper: Create ~500MB silesia-large for parallel benchmarks
# -----------------------------------------------------------------------------
create_silesia_large() {
    if [[ -f "$SILESIA_LARGE" ]]; then
        echo "[OK] silesia-large.bin exists"
        return 0
    fi

    # Try to decompress from cached .gz first
    if [[ -f "$SILESIA_LARGE_GZ" ]]; then
        echo "[CACHE] Decompressing silesia-large.bin from cached silesia-large.gz..."
        gzip -d -c "$SILESIA_LARGE_GZ" > "$SILESIA_LARGE"
        return 0
    fi

    # Need silesia.tar first
    download_silesia

    echo "[CREATE] Creating silesia-large.bin (~500MB)..."
    # silesia.tar is ~212MB, repeat 2x + partial to reach ~500MB
    cat "$SILESIA_TAR" "$SILESIA_TAR" > "$SILESIA_LARGE"
    head -c $((76 * 1024 * 1024)) "$SILESIA_TAR" >> "$SILESIA_LARGE"

    echo "[COMPRESS] Creating silesia-large.gz for git cache..."
    gzip -9 -c "$SILESIA_LARGE" > "$SILESIA_LARGE_GZ"
}

# -----------------------------------------------------------------------------
# Helper: Generate software.archive (source code patterns)
# -----------------------------------------------------------------------------
generate_software() {
    if [[ -f "$SOFTWARE_ARCHIVE" ]]; then
        echo "[OK] software.archive exists"
        return 0
    fi

    # Try to decompress from cached .gz first
    if [[ -f "$SOFTWARE_GZ" ]]; then
        echo "[CACHE] Decompressing software.archive from cached .gz..."
        gzip -d -c "$SOFTWARE_GZ" > "$SOFTWARE_ARCHIVE"
        return 0
    fi

    echo "[GENERATE] Creating software.archive (~211MB)..."
    python3 -c "
import os
patterns = [
    b'    pub fn new() -> Self {\n        Self { data: Vec::new(), count: 0 }\n    }\n',
    b'    #[inline(always)]\n    pub fn get(&self) -> usize { self.count }\n',
    b'// TODO: Optimize this loop\nfor i in 0..data.len() { sum += data[i]; }\n',
    b'fn main() {\n    let args: Vec<String> = std::env::args().collect();\n    println!(\"{:?}\", args);\n}\n',
    b'impl Iterator for MyIter {\n    type Item = u32;\n    fn next(&mut self) -> Option<Self::Item> { self.inner.next() }\n}\n',
    b'#[derive(Debug, Clone, PartialEq)]\npub struct Config {\n    pub name: String,\n    pub value: i64,\n}\n',
]
target = 211 * 1024 * 1024
with open('$SOFTWARE_ARCHIVE', 'wb') as f:
    written = 0
    i = 0
    while written < target:
        header = f'// Line {i % 10000}\n'.encode()
        p = patterns[i % len(patterns)]
        f.write(header + p)
        written += len(header) + len(p)
        i += 1
print(f'Generated software.archive: {os.path.getsize(\"$SOFTWARE_ARCHIVE\") / 1024 / 1024:.1f} MB')
"

    # Compress with high compression for dense LZ77
    echo "[COMPRESS] Creating software.archive.gz..."
    gzip -9 -c "$SOFTWARE_ARCHIVE" > "$SOFTWARE_GZ"
}

# -----------------------------------------------------------------------------
# Helper: Generate logs.txt (repetitive log patterns)
# -----------------------------------------------------------------------------
generate_logs() {
    if [[ -f "$LOGS_TXT" ]]; then
        echo "[OK] logs.txt exists"
        return 0
    fi

    # Try to decompress from cached .gz first
    if [[ -f "$LOGS_GZ" ]]; then
        echo "[CACHE] Decompressing logs.txt from cached .gz..."
        gzip -d -c "$LOGS_GZ" > "$LOGS_TXT"
        return 0
    fi

    echo "[GENERATE] Creating logs.txt (~211MB)..."
    python3 -c "
import os
patterns = [
    '2026-01-20 14:30:{:02d} INFO [gzippy.core] Processed block {} in {}ms\n',
    '2026-01-20 14:30:{:02d} DEBUG [gzippy.sched] Lane {} claimed chunk at 0x{:x}\n',
    '2026-01-20 14:30:{:02d} WARN [gzippy.mem] Memory usage at {}%\n',
    '2026-01-20 14:30:{:02d} ERROR [gzippy.io] Write failed: attempt {}\n',
    '2026-01-20 14:30:{:02d} TRACE [gzippy.net] Packet {} received, {} bytes\n',
]
target = 211 * 1024 * 1024
with open('$LOGS_TXT', 'wb') as f:
    written = 0
    i = 0
    while written < target:
        p = patterns[i % len(patterns)]
        entry = p.format(i % 60, i % 10000, (i * 7) % 100, i % 8, i * 4096, 70 + i % 25)
        f.write(entry.encode())
        written += len(entry)
        i += 1
print(f'Generated logs.txt: {os.path.getsize(\"$LOGS_TXT\") / 1024 / 1024:.1f} MB')
"

    # Compress with fast compression (matches real log compression)
    echo "[COMPRESS] Creating logs.txt.gz..."
    gzip -1 -c "$LOGS_TXT" > "$LOGS_GZ"
}

# -----------------------------------------------------------------------------
# Main: Prepare all benchmark data
# -----------------------------------------------------------------------------
prepare_all() {
    echo "=== Preparing Benchmark Data ==="
    echo "Directory: $BENCHMARK_DIR"
    echo ""

    download_silesia
    create_silesia_large
    generate_software
    generate_logs

    echo ""
    echo "=== Benchmark Data Ready ==="
    ls -lh "$BENCHMARK_DIR"/*.tar "$BENCHMARK_DIR"/*.gz "$BENCHMARK_DIR"/*.bin \
           "$BENCHMARK_DIR"/*.archive "$BENCHMARK_DIR"/*.txt 2>/dev/null || true
}

# Parse arguments
case "${1:-all}" in
    all)
        prepare_all
        ;;
    silesia)
        download_silesia
        ;;
    silesia-large)
        create_silesia_large
        ;;
    software)
        generate_software
        ;;
    logs)
        generate_logs
        ;;
    --help|-h)
        echo "Usage: $0 [all|silesia|silesia-large|software|logs]"
        echo ""
        echo "Prepares benchmark data for gzippy performance testing."
        echo "Compressed versions are cached in git; uncompressed are regenerated on demand."
        ;;
    *)
        echo "Unknown option: $1"
        echo "Run '$0 --help' for usage"
        exit 1
        ;;
esac
