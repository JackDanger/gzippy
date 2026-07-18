#!/usr/bin/env bash
# mac_rg_gap.sh — THE QUIET-BOX gz-vs-rapidgzip GAP MAP (macOS / Apple Silicon).
#
# The chronic #2 blocker was that rapidgzip only existed on the heavily-contended
# Intel LXC (load ~15 from other tenants -> noisy). This rig moves the gz-vs-rg
# structural comparison onto the QUIET deterministic mac. rapidgzip 0.16.0 (pip
# wheel, aarch64) is the comparator arm here; gzippy-native (FFI-off pure-Rust
# ParallelSM) is the subject.
#
# Measurement primitive: `/usr/bin/time -l` exposes the Apple-Silicon per-process
# PMU. "instructions retired" is DETERMINISTIC to ~0.04% warm and is the PRIMARY
# metric (gz/rg total-work ratio). "cycles elapsed" (~0.5% P-core) is SECONDARY.
# "real" wall is COARSE (~10ms clock) -> tertiary; we make inputs large so the
# wall dwarfs the clock granularity, but instr is the verdict driver.
#
# CAVEAT (scope): mac rapidgzip runs its aarch64 path (NOT ISA-L, x86-only); so
# this is gz-aarch64 vs rg-aarch64. VALID for the PIPELINE/tax structural compare
# (~arch-independent) but does NOT test the x86 ISA-L inner-kernel gap (Intel
# silesia-T4 +16% = gz-vs-ISA-L) — that cell stays OWED to a quiet Intel window.
#
# Usage:
#   scripts/bench/standing/mac_rg_gap.sh
#   scripts/bench/standing/mac_rg_gap.sh --corpora "silesia monorepo nasa squishy" \
#         --threads "1 2 4 8" -N 11
#   scripts/bench/standing/mac_rg_gap.sh --rg /tmp/rgvenv/bin/rapidgzip --no-build
#
# Gate-0 self-validation ENFORCED (a cell failing any prints FAIL, no number):
#   (a) byte-exact   : gzippy sha == rapidgzip sha == gzip -d sha, per corpus.
#   (b) non-inert    : GZIPPY_DEBUG shows build-flavor=parallel-sm+pure AND
#                      path=ParallelSM (FFI-off pure-Rust engine, production path).
#   (c) A/A self-test: within-arm instr spread is reported; >0.5% marks instr
#                      UNTRUSTED for that cell; cyc spread >3% marks cyc UNTRUSTED.
#   (d) same sink    : /dev/null for BOTH arms.
#   (e) -pN (gz) / -PN (rg), the real production flags (NOT inert env knobs);
#       cold first run is a warmup (untimed); E-core/throttle outliers dropped.
#
# SCOPE STAMP: macOS-aarch64 (Apple Silicon). NOT-YET-LAW cross-arch. The x86
# ISA-L kernel cells are owed to a quiet Intel window; AMD/Zen2 owed for LAW.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

CORPORA="silesia monorepo nasa squishy"
THREADS="1 2 4 8"
N=11
BUILD=1
BINARY="$ROOT/target/release/gzippy"
RG="${RG_BIN:-/tmp/rgvenv/bin/rapidgzip}"
KEEP=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --corpora) CORPORA="$2"; shift;;
    --corpora=*) CORPORA="${1#*=}";;
    --threads) THREADS="$2"; shift;;
    --threads=*) THREADS="${1#*=}";;
    -N) N="$2"; shift;;
    -N*) N="${1#-N}";;
    --binary) BINARY="$2"; shift;;
    --binary=*) BINARY="${1#*=}";;
    --rg) RG="$2"; shift;;
    --rg=*) RG="${1#*=}";;
    --no-build) BUILD=0;;
    --keep) KEEP=1;;
    -h|--help) sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "mac_rg_gap.sh: unknown arg '$1'" >&2; exit 2;;
  esac
  shift
done

fail() { echo "### MAC_RG_GAP GATE FAILED: $* ###" >&2; exit 1; }

command -v gzip >/dev/null || fail "gzip not on PATH"
[ -x /usr/bin/time ] || fail "/usr/bin/time missing"
[ -x "$RG" ] || fail "rapidgzip not executable at '$RG' (pip install rapidgzip in a venv; pass --rg PATH)"

RGVER="$($RG --version 2>&1 | head -1)"

# ---- build the FFI-off pure-Rust native binary (unless --no-build) ----
if [ "$BUILD" = "1" ]; then
  echo "== building gzippy-native (FFI off): cargo build --release --no-default-features --features gzippy-native =="
  ( cd "$ROOT" && cargo build --release --no-default-features --features gzippy-native >/dev/null 2>&1 ) \
    || fail "cargo build failed"
  BINARY="$ROOT/target/release/gzippy"
fi
[ -x "$BINARY" ] || fail "gzippy binary not executable at $BINARY"
GZSHA="$(shasum -a 256 "$BINARY" | cut -d' ' -f1)"
GITSHA="$(cd "$ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"

resolve_corpus() {
  local name="$1"
  if [ -f "$name" ]; then echo "$name"; return; fi
  for cand in "/tmp/$name.gz" "$ROOT/benchmark_data/$name.gz"; do
    [ -f "$cand" ] && { echo "$cand"; return; }
  done
  return 1
}

# ---- GATE-0(b): non-inert / production path ----
echo "== GATE-0(b) build-flavor + path assertion (gz) =="
FIRST_CORP="$(echo "$CORPORA" | awk '{print $1}')"
FCPATH="$(resolve_corpus "$FIRST_CORP")" || fail "corpus '$FIRST_CORP' not found"
for t in $THREADS; do
  DBG="$(GZIPPY_DEBUG=1 "$BINARY" -dc -p"$t" "$FCPATH" 2>&1 >/dev/null || true)"
  echo "$DBG" | grep -q "build-flavor=parallel-sm+pure" || fail "build-flavor != parallel-sm+pure at p$t"
  echo "$DBG" | grep -q "path=ParallelSM"               || fail "path != ParallelSM at p$t"
done
echo "   PASS  build-flavor=parallel-sm+pure  path=ParallelSM (all T)  sha=$GZSHA  git=$GITSHA"
echo "   rapidgzip: $RGVER  ($RG)"

WORK="$(mktemp -d /tmp/mac_rg_gap.XXXXXX)"
TL="$WORK/tl.txt"
CSV="$WORK/samples.csv"
echo "corpus,bytes,arm,threads,rep,instr,cyc,real" > "$CSV"
cleanup() { [ "$KEEP" = "1" ] || rm -rf "$WORK"; }
trap cleanup EXIT

parse_tl() {
  local f="$1" instr cyc real
  instr="$(awk '/instructions retired/{print $1; exit}' "$f")"
  cyc="$(awk '/cycles elapsed/{print $1; exit}' "$f")"
  real="$(awk 'NR==1{print $1; exit}' "$f")"
  [ -n "$instr" ] && [ -n "$cyc" ] || return 1
  echo "$instr $cyc $real"
}

# ---- per-corpus: GATE-0(a) sha gate (gz==rg==gzip) + decompressed byte count ----
declare -A CORP_PATH CORP_BYTES
for corp in $CORPORA; do
  p="$(resolve_corpus "$corp")" || fail "corpus '$corp' not found (tried path, /tmp/$corp.gz, benchmark_data)"
  CORP_PATH[$corp]="$p"
  echo "== GATE-0(a) sha + size: $corp ($p) =="
  ref="$(gzip -dc "$p" | shasum -a 256 | cut -d' ' -f1)"
  nbytes="$(gzip -dc "$p" | wc -c | tr -d ' ')"
  rgs="$("$RG" -dc -P1 "$p" 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"
  gz="$("$BINARY" -dc -p1 "$p" | shasum -a 256 | cut -d' ' -f1)"
  [ "$ref" = "$rgs" ] || fail "$corp: rapidgzip sha != gzip -d sha"
  [ "$ref" = "$gz" ]  || fail "$corp: gzippy sha != gzip -d sha (BYTE MISMATCH)"
  CORP_BYTES[$corp]="$nbytes"
  echo "   PASS  sha=$ref  bytes=$nbytes"
done

# ---- warmup (cold first run, untimed) per corpus per arm per T ----
echo "== warmup (untimed) =="
for corp in $CORPORA; do
  p="${CORP_PATH[$corp]}"
  for t in $THREADS; do
    "$BINARY" -dc -p"$t" "$p" >/dev/null 2>&1 || true
    "$RG"     -dc -P"$t" "$p" >/dev/null 2>&1 || true
  done
done

# ---- measurement: interleaved best-of-N, /dev/null sink BOTH arms ----
echo "== measure: interleaved best-of-N=$N  (gz -pT vs rg -PT, both -> /dev/null) =="
for rep in $(seq 1 "$N"); do
  for corp in $CORPORA; do
    p="${CORP_PATH[$corp]}"; b="${CORP_BYTES[$corp]}"
    for t in $THREADS; do
      # arm: gzippy -pT
      /usr/bin/time -l "$BINARY" -dc -p"$t" "$p" >/dev/null 2>"$TL"
      read -r I C R < <(parse_tl "$TL") || fail "parse gzippy time -l ($corp p$t rep $rep)"
      echo "$corp,$b,gzippy,$t,$rep,$I,$C,$R" >> "$CSV"
      # arm: rapidgzip -PT
      /usr/bin/time -l "$RG" -dc -P"$t" "$p" >/dev/null 2>"$TL"
      read -r I C R < <(parse_tl "$TL") || fail "parse rapidgzip time -l ($corp P$t rep $rep)"
      echo "$corp,$b,rapidgzip,$t,$rep,$I,$C,$R" >> "$CSV"
    done
  done
done

if [ "$KEEP" = "1" ]; then
  mkdir -p "$ROOT/artifacts" 2>/dev/null || true
  cp "$CSV" "$ROOT/artifacts/mac_rg_gap_samples.csv" 2>/dev/null || true
fi

python3 "$HERE/mac_rg_gap_report.py" "$CSV"
RC=$?
echo
echo "  subject:    gzippy-native sha=$GZSHA  git=$GITSHA  (FFI off, build-flavor=parallel-sm+pure)"
echo "  comparator: $RGVER"
echo "  scope:      macOS-aarch64 (Apple Silicon) — gz-aarch64 vs rg-aarch64."
echo "              PIPELINE/tax compare transfers cross-arch; x86 ISA-L inner-kernel cells OWED to quiet Intel."
echo "  stamp:      NOT-YET-LAW (single-arch aarch64; AMD/Zen2 owed for LAW)"
[ "$KEEP" = "1" ] && echo "  raw samples kept: $WORK/samples.csv  (and artifacts/mac_rg_gap_samples.csv)"
exit $RC
