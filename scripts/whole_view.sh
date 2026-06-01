#!/usr/bin/env bash
# DEPRECATED (2026-05-31) — superseded by the locked+gated clean-bench harness.
#
# whole_view.sh produced a full-system decode view but on an UNLOCKED, UNGATED
# box and auto-emitted fulcrum component attributions (including the telescoping
# +0.0% identity the design explicitly bans from standard output). Trustworthy
# wall numbers now come ONLY from the gated harness; timeline traces it captures
# are archived as artifacts for MANUAL fulcrum inspection.
#
# USE INSTEAD:  scripts/bench/clean_bench.sh   (see scripts/bench/README.md)
# (referenced in memory; kept as a stub rather than deleted.)
echo "DEPRECATED — use scripts/bench/clean_bench.sh (see scripts/bench/README.md)" >&2
exit 1
