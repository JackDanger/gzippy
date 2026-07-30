#!/usr/bin/env bash
# verify.sh — the ENCODER CORRECTNESS ORACLE. Run this before any size or wall claim.
#
#   scripts/campaign/verify.sh [tune|gate|all] [extra fulcrum verify args]
#
# Thin guarded invocation of `fulcrum verify`; NO measurement logic lives here, per
# "all measurement tooling goes into fulcrum". This script's only jobs are the guards in
# lib.sh (declared corpus, identified instrument) and pointing the oracle at OUR decoder.
#
# WHY OUR OWN DECODER IS THE ORACLE. gzippy's decompressor is finished and is the fastest
# in the world, which makes it both the most faithful and the cheapest oracle available:
# compress, decompress with it at EVERY thread count, sha256 against the original. Vendor
# decoders are kept in the loop as a CROSS-CHECK (`--cross`) so that a shared
# misunderstanding of the format cannot pass — not as the primary oracle.
#
# RECEIPT FOR THIS FILE EXISTING, from the session that added it: a hand-rolled
# shell roundtrip loop over the same corpus reported ALL 90 cells FAILING on a build that
# was in fact correct — every cell failed identically, the signature of a harness bug (an
# unset variable), not an encoder bug. Ten minutes went into "debugging" a working
# matchfinder. `fulcrum verify` additionally asserts things the hand-rolled loop did not
# even attempt: P4 (size monotonic across levels) and P8 (thread count does not change the
# answer). CLAUDE.md hard stop #6 already said never hand-roll a measurement; this is the
# encoder-correctness half of making that easy to obey.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$HERE/lib.sh"

SET_NAME="${1:-tune}"
[ $# -eq 0 ] || shift
LEVELS="${CAMPAIGN_LEVELS:-0-9}"
CTHREADS="${CAMPAIGN_THREADS:-1,4}"

campaign_guard_gate "$SET_NAME"
campaign_preflight
campaign_corpus_args "$SET_NAME"

GZ="$CAMPAIGN_REPO/target/release/gzippy"
[ -x "$GZ" ] || die "no gzippy at $GZ" "build it: (cd $CAMPAIGN_REPO && cargo build --release)"
note "binary" "gzippy sha256=$(shasum -a 256 "$GZ" | cut -c1-16) levels=$LEVELS threads=$CTHREADS"

OUT="${CAMPAIGN_OUT:-$(campaign_outdir "verify-$(git -C "$CAMPAIGN_REPO" rev-parse --short HEAD)")}/report.json"

set -x
exec "$CAMPAIGN_FULCRUM" verify \
  --ours "$GZ -{level} -p {threads} -c {input}" \
  --decoder "$GZ -d -p {threads} -c {input}" \
  "${CAMPAIGN_CORPUS_ARGS[@]}" \
  --levels "$LEVELS" \
  --compress-threads "$CTHREADS" \
  --decode-threads "$CTHREADS" \
  --cross 'gzip -dc' \
  --cross 'pigz -dc' \
  --cross 'libdeflate-gzip -dc' \
  --out "$OUT" \
  "$@"
