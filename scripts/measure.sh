#!/usr/bin/env bash
# DEPRECATED (2026-05-31) — superseded by the locked+gated clean-bench harness.
#
# measure.sh ran interleaved sha-verified deltas but on an UNLOCKED, UNGATED,
# UNVERIFIED-PROVENANCE box: no frequency lock, no aperf/mperf gate, no
# RUN_TRUSTWORTHY verdict, no restore guarantee. Those are exactly the gaps the
# campaign post-mortem demanded we close.
#
# USE INSTEAD:  scripts/bench/clean_bench.sh   (see scripts/bench/README.md)
# (referenced in memory; kept as a stub rather than deleted.)
echo "DEPRECATED — use scripts/bench/clean_bench.sh (see scripts/bench/README.md)" >&2
exit 1
