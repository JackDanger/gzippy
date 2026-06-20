#!/usr/bin/env bash
# run_contig_objdump.sh — disassemble run_contig from a fresh guest build so a
# reviewer can eyeball the hot loop (instruction count / a lengthened loop-carried
# dependency chain — the N32 trap = a register live-range extended across the
# refill) BEFORE committing or building the full kernel_gate A/B.
#
# Builds gzippy-native (STRIP=false so run_contig keeps its symbol; #[inline(never)]
# guarantees a standalone body) at one sha on the chosen box, then emits:
#   - the FULL run_contig disassembly  -> artifacts/objdump/<runid>/run_contig.full.s
#   - the HOT LOOP body (the `2:`->`jb 2b` back-edge region; best-effort heuristic
#     extraction)                       -> .../run_contig.hotloop.s
#   - an instruction-count summary (total + per-mnemonic top-20)
#   - the address-normalised disasm sha (same signature kernel_gate's .o-diff uses)
#
# Two shas can be diffed by running twice and `diff`-ing the .hotloop.s files (or
# comparing the printed sigs) — a fast pre-check before paying for the A/B.
#
# Usage:
#   scripts/bench/kernel-ab/run_contig_objdump.sh                 # HEAD of branch
#   scripts/bench/kernel-ab/run_contig_objdump.sh --sha <sha> [--box neurotic]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
# shellcheck source=/dev/null
. "$ROOT/scripts/bench/guest.env"

BOX="${BOX:-neurotic}"
SHA=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --box) BOX="$2"; shift;; --box=*) BOX="${1#*=}";;
    --sha) SHA="$2"; shift;; --sha=*) SHA="${1#*=}";;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "run_contig_objdump.sh: unknown arg '$1'" >&2; exit 2;;
  esac
  shift
done
# shellcheck source=/dev/null
BOX="$BOX" . "$ROOT/scripts/bench/boxes.sh"

if [ -z "$SHA" ]; then SHA="$(git ls-remote origin "$BOX_BRANCH" | cut -f1)"; fi
SHA="$(git rev-parse "$SHA" 2>/dev/null || echo "$SHA")"
RUNID="objdump_$(date -u '+%Y%m%dT%H%M%SZ')_${SHA:0:8}"
LOCAL_ART="$ROOT/artifacts/objdump/$RUNID"
mkdir -p "$LOCAL_ART"

echo "== run_contig_objdump  box=$BOX_NAME  sha=$SHA =="

REMOTE=$(cat <<REMOTE_EOF
set -e
cd '$BOX_SRC'
git fetch origin '$BOX_BRANCH' --quiet 2>&1 | tail -1
git reset --hard '$SHA' 2>&1 | tail -1
CARGO_TARGET_DIR='$BOX_TARGET' RUSTFLAGS='-C target-cpu=native' CARGO_PROFILE_RELEASE_STRIP=false \
  timeout 600 cargo build --release --no-default-features --features '$BOX_FEATURES' 2>&1 | tail -3
BIN='$BOX_TARGET/release/gzippy'
OUT=/dev/shm/run_contig_objdump; rm -rf \$OUT; mkdir -p \$OUT
objdump -d -C --no-show-raw-insn "\$BIN" \
  | awk '/::run_contig>:/{f=1} f&&/^\$/{f=0} f{print}' > \$OUT/run_contig.full.s
NLINES=\$(wc -l < \$OUT/run_contig.full.s)
if [ "\$NLINES" -lt 5 ]; then echo "ERROR: run_contig symbol not found (built stripped?)"; exit 3; fi
# HOT LOOP heuristic: the densest back-edge region. Emit the whole body annotated;
# mark jump targets that are jumped-to from BELOW (loop back-edges).
SIG=\$(sed -E 's/^[[:space:]]*[0-9a-f]+:[[:space:]]*//; s/0x[0-9a-f]+/0xADDR/g; s/<[^>]*>//g' \$OUT/run_contig.full.s | grep -vE '^[[:space:]]*\$' | sha256sum | cut -c1-16)
# instruction histogram
awk '{for(i=1;i<=NF;i++) if(\$i ~ /^[a-z][a-z0-9.]+\$/ && \$(i-1) ~ /:\$/){print \$i; break}}' \$OUT/run_contig.full.s 2>/dev/null > \$OUT/mnem.txt || true
# simpler mnemonic extraction: 2nd field after the address+colon
sed -E 's/^[[:space:]]*[0-9a-f]+:[[:space:]]*//' \$OUT/run_contig.full.s | awk 'NF{print \$1}' | grep -E '^[a-z]' > \$OUT/mnem.txt || true
TOTAL=\$(grep -cE ':' \$OUT/run_contig.full.s || echo 0)
echo "=== run_contig @ $SHA ==="
echo "disasm-lines: \$NLINES   addr-normalised-sig: \$SIG"
echo "--- instruction count (insns with an address) ---"
grep -cE '^[[:space:]]*[0-9a-f]+:' \$OUT/run_contig.full.s
echo "--- top-20 mnemonics ---"
sort \$OUT/mnem.txt | uniq -c | sort -rn | head -20
echo "--- back-edge labels (loop heads; jumped-to from later addrs) ---"
grep -oE '<[^>]*\+0x[0-9a-f]+>' \$OUT/run_contig.full.s | sort | uniq -c | sort -rn | head -10 || true
REMOTE_EOF
)

$BOX_SSH "$REMOTE" 2>&1 | tee "$LOCAL_ART/summary.txt"

echo "=== pull disasm -> $LOCAL_ART ==="
# shellcheck disable=SC2086
scp -o ConnectTimeout=15 $BOX_SCP_JFLAG \
  "$BOX_GUEST:/dev/shm/run_contig_objdump/run_contig.full.s" "$LOCAL_ART/run_contig.full.s" 2>/dev/null || true

echo
echo "full disasm: $LOCAL_ART/run_contig.full.s"
echo "(diff two shas: run this twice with --sha, then 'diff' the run_contig.full.s files)"
