#!/usr/bin/env bash
# ensure-corpus.sh — guarantee the pinned corpus is present and IDENTICAL to the
# one banked numbers were taken against (P3 of OWNER-HELP-SUGGESTIONS).
#
# Runs ON THE GUEST (parity.sh invokes it over the double-hop before measuring).
# It is also safe to run locally if $CORPUS happens to exist locally.
#
# Identity is checked on the DECOMPRESSED payload (CORPUS_RAW_SHA256 in
# guest.env) — that is the load-bearing sha: it is the correctness oracle every
# measured run is verified against (Rule 4). The compressed-file sha is an
# additional cheap check when pinned.
#
# If $CORPUS is absent or fails its sha, regenerate from $CORPUS_SRC_URL (the
# canonical squishy corpus) so ANY guest is reproducible.
#
# Usage:
#   scripts/bench/ensure-corpus.sh                 # verify; regenerate if needed
#   scripts/bench/ensure-corpus.sh --verify-only   # never fetch; nonzero if wrong
#   scripts/bench/ensure-corpus.sh --print-sha     # print the decompressed sha & exit
#   scripts/bench/ensure-corpus.sh --help
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() { sed -n '2,24p' "${BASH_SOURCE[0]}"; }

VERIFY_ONLY=0
case "${1:-}" in
  -h|--help) usage; exit 0;;
  --verify-only) VERIFY_ONLY=1;;
  --print-sha) PRINT_SHA=1;;
esac

# shellcheck source=/dev/null
. "$HERE/guest.env"

: "${CORPUS:?guest.env did not define CORPUS}"
: "${CORPUS_RAW_SHA256:?guest.env did not define CORPUS_RAW_SHA256}"

raw_sha() { gzip -dc "$1" 2>/dev/null | sha256sum | cut -d' ' -f1; }

if [ "${PRINT_SHA:-0}" = 1 ]; then
  [ -f "$CORPUS" ] || { echo "corpus absent: $CORPUS" >&2; exit 1; }
  echo "decompressed_sha256=$(raw_sha "$CORPUS")"
  echo "pinned_sha256=$CORPUS_RAW_SHA256"
  exit 0
fi

corpus_ok() {
  [ -f "$CORPUS" ] || return 1
  local got; got="$(raw_sha "$CORPUS")"
  if [ "$got" != "$CORPUS_RAW_SHA256" ]; then
    echo "ensure-corpus: $CORPUS decompressed sha MISMATCH" >&2
    echo "  got=$got" >&2
    echo "  pin=$CORPUS_RAW_SHA256" >&2
    return 1
  fi
  # Optional compressed-file identity check when pinned.
  if [ -n "${CORPUS_GZ_SHA256:-}" ]; then
    local gzsha; gzsha="$(sha256sum "$CORPUS" | cut -d' ' -f1)"
    if [ "$gzsha" != "$CORPUS_GZ_SHA256" ]; then
      echo "ensure-corpus: $CORPUS compressed sha MISMATCH (got=$gzsha pin=$CORPUS_GZ_SHA256)" >&2
      return 1
    fi
  fi
  return 0
}

if corpus_ok; then
  echo "ensure-corpus: OK  $CORPUS  (decompressed sha == pin)"
  exit 0
fi

if [ "$VERIFY_ONLY" = 1 ]; then
  echo "ensure-corpus: VERIFY-ONLY and corpus is absent/wrong — refusing to fetch." >&2
  exit 1
fi

# ---- regenerate from the canonical source -----------------------------------
: "${CORPUS_SRC_URL:?guest.env did not define CORPUS_SRC_URL — cannot regenerate}"
echo "ensure-corpus: regenerating $CORPUS from $CORPUS_SRC_URL ..."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

# The squishy source is a tar of the silesia files; build the single-member gzip
# the corpus pin expects (concatenated payload, then gzip -9). We download, untar,
# concatenate in a STABLE order, and gzip. If the URL yields an already-.gz, use
# it directly.
ARC="$TMP/src.bin"
if ! timeout 600 curl -fsSL "$CORPUS_SRC_URL" -o "$ARC"; then
  echo "ensure-corpus: download FAILED from $CORPUS_SRC_URL" >&2
  exit 4
fi

case "$CORPUS_SRC_URL" in
  *.gz)
    cp "$ARC" "$CORPUS"
    ;;
  *.tar|*.tar.*)
    mkdir -p "$TMP/x"
    tar xf "$ARC" -C "$TMP/x"
    # Concatenate member files in sorted (stable) order, then gzip -9.
    find "$TMP/x" -type f | LC_ALL=C sort | xargs cat | gzip -9 > "$CORPUS"
    ;;
  *)
    # Unknown: assume raw payload; gzip it.
    gzip -9 < "$ARC" > "$CORPUS"
    ;;
esac

if corpus_ok; then
  echo "ensure-corpus: regenerated OK  $CORPUS"
  exit 0
fi

echo "ensure-corpus: regenerated corpus STILL fails the sha pin — the source URL or" >&2
echo "  the concatenation order does not match the banked corpus. Investigate before" >&2
echo "  trusting any number (a drifted corpus invalidates the comparison)." >&2
exit 5
