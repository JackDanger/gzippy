#!/usr/bin/env bash
# stage-corpus.sh — materialise the DERIVED corpus members from sha-pinned sources.
#
#   scripts/campaign/stage-corpus.sh [--verify-only]
#
# Most squishy members are files you either have or don't. `sil40` is different: it is
# DERIVED from silesia, and until 2026-07-30 its definition existed nowhere — not a sha,
# not a size, not a generator, in either repo. That is worse than a missing file, because
# `sil40` is a `goal::MIN_CORPORA` member, so the mandated minimum promotion surface could
# not be reconstructed by anyone. `make board-size-promote` was blocked on it.
#
# THE CORPUS, for the avoidance of another undefined member:
#   * SQUISHY (https://squishy.jackdanger.com/) is the canonical corpus and what the
#     per-label board is graded on. It is the user's curated set; prefer it always.
#   * SILESIA is a secondary quick test — useful for a fast read, not for proving.
#     `sil40` is the 40 MB slice of it that this repo already refers to twice
#     ("+143,807 B on silesia 40 MB"; "a 40 MB silesia slice" in `pipelined.rs`).
#
# The source is sha-pinned and checked BEFORE slicing, so a different silesia.tar cannot
# silently produce a different `sil40` and shift the promotion surface underneath a
# banked number.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$HERE/lib.sh"

VERIFY_ONLY=0
[ "${1:-}" != "--verify-only" ] || VERIFY_ONLY=1

# --- sil40 ---------------------------------------------------------------------------
# Source: silesia.tar, pinned by /root/archive/corpora/MANIFEST.sha256 on the rig.
SIL_SHA=028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f
SIL_BYTES=211968000
# 40 MB decimal, matching this repo's naming convention elsewhere (`text-1MB`).
SIL40_BYTES=40000000
SIL40_SHA=e5b7e686bf31e0157ffd36a2c050a7e6057833c81558a0b48cce9374b5e01dda

find_silesia() {
  for c in "$CAMPAIGN_CORPUS_ROOT/../silesia.tar" "$CAMPAIGN_REPO/benchmark_data/silesia.tar" \
           /root/archive/corpora/silesia.tar "$HOME/www/gzippy/benchmark_data/silesia.tar"; do
    [ -f "$c" ] && { printf '%s' "$c"; return 0; }
  done
  return 1
}

sha_of() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1 || sha256sum "$1" | cut -d' ' -f1; }

out="$CAMPAIGN_CORPUS_ROOT/sil40"
if [ -f "$out" ]; then
  got="$(sha_of "$out")"; sz=$(wc -c < "$out" | tr -d ' ')
  note "sil40" "present bytes=$sz sha256=${got:0:16}"
  [ "$sz" = "$SIL40_BYTES" ] || die "sil40 is $sz bytes, expected $SIL40_BYTES" \
    "A derived member of the wrong size silently changes the promotion surface. Delete and re-stage."
  exit 0
fi
[ "$VERIFY_ONLY" = 0 ] || die "sil40 absent and --verify-only was given"

src="$(find_silesia)" || die "no silesia.tar found" \
  "Looked beside the corpus, in the repo's benchmark_data/, and at /root/archive/corpora/." \
  "silesia is the SOURCE for sil40; squishy members are staged directly."
note "source" "$src"
got="$(sha_of "$src")"; sz=$(wc -c < "$src" | tr -d ' ')
[ "$got" = "$SIL_SHA" ] || die "silesia.tar sha mismatch" \
  "  expected $SIL_SHA" "  got      $got" \
  "Slicing a different silesia would produce a different sil40 and shift the promotion" \
  "surface underneath every banked number. Refusing."
[ "$sz" = "$SIL_BYTES" ] || die "silesia.tar is $sz bytes, expected $SIL_BYTES"

head -c "$SIL40_BYTES" "$src" > "$out" || die "could not write $out"
newsha="$(sha_of "$out")"
note "sil40" "STAGED bytes=$(wc -c < "$out" | tr -d ' ') sha256=$newsha"
if [ -n "$SIL40_SHA" ] && [ "$newsha" != "$SIL40_SHA" ]; then
  die "sil40 sha mismatch: expected $SIL40_SHA got $newsha"
fi
note "ok" "sil40 matches the pinned sha" 
