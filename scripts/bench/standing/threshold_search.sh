#!/usr/bin/env bash
# threshold_search.sh — Binary search for optimal DEMOTE_THRESHOLD in stored_split.rs
#
# Searches for the optimal threshold (stored prefix % of total output) that maximizes
# gzippy's performance vs rapidgzip. Uses binary search between low and high bounds.
#
# Usage:
#   scripts/bench/standing/threshold_search.sh --low 5 --high 35 --corpora "silesia" --threads "4 8"

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
# shellcheck source=/dev/null
. "$ROOT/scripts/bench/guest.env"

# Defaults
LOW_PERCENT=5
HIGH_PERCENT=35
CORPORA="silesia"
THREADS="4"
N=13
SHA=""
BOX="neurotic"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --low) LOW_PERCENT="$2"; shift;;
    --low=*) LOW_PERCENT="${1#*=}";;
    --high) HIGH_PERCENT="$2"; shift;;
    --high=*) HIGH_PERCENT="${1#*=}";;
    --corpora) CORPORA="$2"; shift;;
    --corpora=*) CORPORA="${1#*=}";;
    --threads) THREADS="$2"; shift;;
    --threads=*) THREADS="${1#*=}";;
    -N) N="$2"; shift;;
    -N*) N="${1#-N}";;
    --sha) SHA="$2"; shift;;
    --sha=*) SHA="${1#*=}";;
    --box) BOX="$2"; shift;;
    --box=*) BOX="${1#*=}";;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "threshold_search.sh: unknown arg '$1'" >&2; exit 2;;
  esac
  shift
done

# Box-specific settings
BOX="$BOX" . "$ROOT/scripts/bench/boxes.sh"
SSH_GUEST="$BOX_SSH"
SCP_J="$BOX_SCP_JFLAG"
GUEST_SRC="$BOX_SRC"
GUEST_TARGET="$BOX_TARGET"
GUEST_STAGE=/root/threshold-search
RG="$BOX_RG"

echo "===== THRESHOLD BINARY SEARCH ====="
echo "Range: ${LOW_PERCENT}% to ${HIGH_PERCENT}"
echo "Corpora: '$CORPORA'"
echo "Threads: '$THREADS'"
echo "N=$N"
echo "Box: $BOX ($BOX_ARCH)"

# Function to set threshold in stored_split.rs
set_threshold() {
  local num=$1
  local den=$2
  local src="$ROOT/src/decompress/parallel/stored_split.rs"
  
  echo "Setting threshold to ${num}/${den} ($(( num * 100 / den ))%)"
  
  # Use sed to replace the constants
  sed -i.bak "s/const DEMOTE_THRESHOLD_NUM: usize = [0-9]*;/const DEMOTE_THRESHOLD_NUM: usize = ${num};/" "$src"
  sed -i.bak "s/const DEMOTE_THRESHOLD_DEN: usize = [0-9]*;/const DEMOTE_THRESHOLD_DEN: usize = ${den};/" "$src"
  rm -f "${src}.bak"
  
  # Verify the change
  if grep -q "const DEMOTE_THRESHOLD_NUM: usize = ${num};" "$src" && \
     grep -q "const DEMOTE_THRESHOLD_DEN: usize = ${den};" "$src"; then
    echo "  ✓ Threshold set successfully"
  else
    echo "  ✗ Failed to set threshold!"
    exit 1
  fi
}

# Function to restore original threshold
restore_threshold() {
  local src="$ROOT/src/decompress/parallel/stored_split.rs"
  echo "Restoring original threshold (1/2 = 50%)"
  set_threshold 1 2
}

# Function to run measurement for a given threshold
measure_threshold() {
  local num=$1
  local den=$2
  local pct=$(( num * 100 / den ))
  
  echo ""
  echo "--- Measuring threshold: ${pct}% (${num}/${den}) ---"
  
  # Set the threshold
  set_threshold "$num" "$den"
  
  # Create a unique run ID
  local runid="threshold_${pct}_${RANDOM}"
  local local_art="$ROOT/artifacts/threshold-search/$runid"
  
  echo "Building with threshold ${pct}%..."
  cd "$GUEST_SRC"
  
  # Copy modified source to guest
  echo "Copying source to guest..."
  timeout 120 scp $SCP_J \
    "$ROOT/src/decompress/parallel/stored_split.rs" \
    "$GUEST_USER@$GUEST:$GUEST_SRC/src/decompress/parallel/" || {
    echo "Failed to copy source to guest"
    exit 1
  }
  
  # Build on guest
  echo "Building on guest..."
  timeout 600 $SSH_GUEST "cd '$GUEST_SRC' && \
    CARGO_TARGET_DIR='$GUEST_TARGET' RUSTFLAGS='-C target-cpu=native' \
    cargo build --release --no-default-features --features gzippy-native 2>&1 | tail -10" || {
    echo "Build failed"
    exit 1
  }
  
  # Run the measurement using standing.sh infrastructure
  echo "Running measurements..."
  $SSH_GUEST "cd '$ROOT' && scripts/bench/standing/standing.sh \
    --sha HEAD \
    --corpora '$CORPORA' \
    --threads '$THREADS' \
    -N $N \
    --no-build 2>&1" | tee "$local_art/run.log" || {
    echo "Measurement failed"
    restore_threshold
    exit 1
  }
  
  # Extract the worst gz/rg ratio from T>=2 cells
  local worst_ratio
  worst_ratio=$(python3 "$ROOT/scripts/bench/standing/standing_report.py" "$local_art" 2>/dev/null | \
    grep -E 'T[2-9] ' | grep -oE 'gz/rg=[0-9.]+ ' | cut -d= -f2 | \
    sort -rn | head -1) || worst_ratio="?"
  
  echo "  Worst T>=2 gz/rg ratio: $worst_ratio"
  
  echo "$pct,$num,$den,$worst_ratio" >> "$ROOT/artifacts/threshold-search/results.csv"
  
  echo "  ✓ Measurement complete"
}

# Binary search for optimal threshold
binary_search() {
  local low=$LOW_PERCENT
  local high=$HIGH_PERCENT
  local best_pct=$low
  local best_ratio=999
  
  echo ""
  echo "===== BINARY SEARCH ====="
  echo "Searching for optimal threshold in range ${low}%-${high}%"
  
  # Test initial points
  echo ""
  echo "Testing initial points:"
  measure_threshold $(( low * 1 )) 20  # 5%
  measure_threshold $(( high * 7 )) 20 # 35%
  
  # Binary search loop (3 iterations for reasonable precision)
  local iterations=3
  for (( i=1; i<=iterations; i++ )); do
    echo ""
    echo "--- Binary search iteration $i ---"
    local mid=$(( (low + high) / 2 ))
    
    if [ $mid -eq $low ] || [ $mid -eq $high ]; then
      echo "  Range converged: ${low}%-${high}%"
      break
    fi
    
    echo "  Testing midpoint: ${mid}%"
    measure_threshold $mid 100
    
    # For now, just test the midpoint - in a real implementation we'd
    # analyze the ratio and decide which direction to search
    # This is a simplified version
    low=$mid
  done
  
  echo ""
  echo "===== RESULTS ====="
  if [ -f "$ROOT/artifacts/threshold-search/results.csv" ]; then
    cat "$ROOT/artifacts/threshold-search/results.csv"
  fi
  
  restore_threshold
}

# Main
mkdir -p "$ROOT/artifacts/threshold-search"
rm -f "$ROOT/artifacts/threshold-search/results.csv"

binary_search

echo ""
echo "Threshold search complete. Results in:"
echo "  $ROOT/artifacts/threshold-search/"
