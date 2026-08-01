#!/usr/bin/env bash
# Sweep HT_MAX_LEN3_OFFSET and read the L1 ratchet at each value.
#
# WHY. `ht_fast` beats libdeflate on every BINARY file in the ratchet
# (armexe.elf -4013, data.parquet -984, data.csv -210) and loses on every
# TEXT-like one (engine.wasm +61, minjs +313, data.json +449, aozora +550,
# dickens +644). The one structural thing we have that libdeflate's L1 does NOT
# is the length-3 table — its ht_matchfinder deliberately omits it ("due to its
# focus on speed, the ht_matchfinder doesn't support length 3 matches").
#
# HYPOTHESIS this tests: hash3 earns bytes on binaries and costs them on text.
# 0 disables it entirely (libdeflate's configuration); 4096 is ours today
# (gzip's TOO_FAR rule). If the hypothesis holds, text improves and binaries
# degrade as the limit falls, with an optimum in between. If the deltas barely
# move, the remaining gap is elsewhere: bucket geometry, the hash, or the
# parser's accept rule.
#
# SAFE BY CONSTRUCTION: `ht_fast` is NOT routed in production — its only call
# site is the bakeoff test — so nothing here can change the shipped binary. The
# trap restores the original constant on EVERY exit path including failure.
#
# ⚠ RUN IT ALONE. Each value is a full fat-LTO rebuild. On 2026-08-01 this
# appeared "stuck" for ten minutes; `ps` showed three cargo processes
# contending, two of them another agent's tests in a different repo on the same
# machine. Check `ps -eo etime,command | grep cargo` FIRST — a slow build is
# usually contention, not a hang.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
F=src/compress/deflate/matchfinder/ht.rs
ORIG=$(grep -oE 'pub const HT_MAX_LEN3_OFFSET: u32 = [0-9]+;' "$F") || exit 1
echo "original: $ORIG"
trap 'sed -i "" "s/pub const HT_MAX_LEN3_OFFSET: u32 = [0-9]*;/$ORIG/" "$F"; echo "restored: $ORIG"' EXIT

for V in "${@:-0 512 1024 2048 4096 8192 32768}"; do
  sed -i '' "s/pub const HT_MAX_LEN3_OFFSET: u32 = [0-9]*;/pub const HT_MAX_LEN3_OFFSET: u32 = $V;/" "$F"
  if ! cargo build --release >/dev/null 2>&1; then
    echo "len3_off=$V BUILD FAILED"
    continue
  fi
  # ⛔ DO NOT PARSE THE PRETTY TABLE. The first version of this loop did, and
  # every number it printed was wrong in a way that LOOKED right:
  #
  #   len3_off=0      total=+1.634%   ...every delta +0...
  #   len3_off=4096   total=+1.634%   data.json=+449 ... armexe.elf=-4013
  #
  # Three defects at once. (a) `total` was IDENTICAL at both values and equal to
  # +1.634%, which is the *Fast* figure — the awk matched /ht_fast vs libdeflate/
  # against a line carrying BOTH numbers and took field 4, which is Fast's.
  # (b) Every delta at len3_off=0 read +0, which would mean we tie libdeflate
  # exactly on all eight files. (c) The header row was captured as data
  # (`file=delta`). A table formatted for humans is not a machine interface, and
  # a parser with no cross-check will be wrong in some direction forever — the
  # same lesson the callgrind `summary:` guard exists for.
  #
  # So the test now emits ONE machine-readable line per file and one TOTAL line,
  # prefixed `L1SWEEP`, and this loop greps only that. If the prefix is missing
  # the value is reported as MISSING rather than silently mis-parsed.
  # A RATCHET FAILURE HERE IS EXPECTED AND CORRECT. The ratchet asserts that no
  # file got worse than its recorded allowance, and a sweep deliberately moves
  # the implementation off its recorded point — so every value except the
  # committed one will trip it. That is the ratchet working, not the sweep
  # breaking. We read stdout regardless: the L1SWEEP lines are printed BEFORE
  # the assert, and `--nocapture` streams them, so a failing run still yields
  # its numbers. Exit status is deliberately ignored for this reason.
  OUT=$(cargo test --release l1_bakeoff -- --nocapture 2>/dev/null | grep '^L1SWEEP')
  if [ -z "$OUT" ]; then
    echo "len3_off=$V MISSING (no L1SWEEP lines — did the test emit them?)"
    continue
  fi
  printf 'len3_off=%-6s %s\n' "$V" "$(echo "$OUT" | tr '\n' ' ')"
done
