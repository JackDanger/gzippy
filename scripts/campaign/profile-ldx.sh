#!/usr/bin/env bash
# profile-ldx.sh — self-time profile of the ldx encoder, by function, on macOS/ARM.
#
# NAMED BLOCKED QUESTION: the port is 1.12-1.33x libdeflate at L1/L2/L6 and the
# banked codegen attribution (x86 callgrind Ir) does NOT transfer — applying its
# top-ranked fix bought 1.08x at L1 and noise elsewhere. Without a profile on
# THIS arch the next step is a guess.
#
# ⚠ THE TRAP THIS EXISTS TO AVOID: `[profile.release]` sets `strip = true`, so a
# release build has ONE text symbol and every profiler reports `???`. The repo
# already carries the fix — `[profile.release-syms]`, a symbol-visible twin whose
# __text bytes are verified byte-identical by tests/symbol_canary.rs. Profile
# THAT. (`/usr/bin/sample`, `atos` and `nm` all fail silently on the stripped
# binary; they do not tell you it is stripped.)
#
#   scripts/campaign/profile-ldx.sh <corpus-file> <level> [iters]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2
F="${1:?usage: profile-ldx.sh <corpus-file> <level> [iters]}"; L="${2:?}"; N="${3:-60}"
command -v samply >/dev/null || { echo "profile-ldx: samply not on PATH (cargo install samply)" >&2; exit 2; }
cargo build --profile release-syms --example ldxloop 2>&1 | grep -E "^error" && exit 2
OUT="${TMPDIR:-/tmp}/ldxprof-L$L.json"
samply record --save-only -o "$OUT" -r 999 -- \
  target/release-syms/examples/ldxloop "$F" "$L" "$N" >/dev/null 2>&1
python3 scripts/campaign/profattr.py "$OUT" target/release-syms/examples/ldxloop
