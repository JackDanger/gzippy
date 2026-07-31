#!/usr/bin/env bash
# board-wall.sh — the WALL axis of the per-label board, in one command.
#
# WHY THIS EXISTS. `board-size.sh` has existed since the campaign's first hour and every
# size number quoted has passed its four guards. The WALL board — the other half of the
# goal, and the half that decides every optimisation lever — was run RAW, by hand, with
# the flags retyped each time. The size axis was guarded and the wall axis was not, purely
# because size came first. Every wall guard below cites the incident that earned it.
#
# The wall axis needs THREE guards that size does not, because size is deterministic and
# wall is not:
#
#   W1 BOX FREEZE. Size is immune to box load; wall is not. `fulcrum freeze run` pauses
#      the named processes and SIGCONTs them on every exit path. Refuse to measure a wall
#      number on a loaded box rather than emit one that cannot be reproduced.
#
#   W2 VANILLA BUILD. CLAUDE.md: "a tuned/instrumented build is 1.17x slower — never quote
#      one against a rival." Instruction-count work in this campaign routinely leaves an
#      instrumented binary in target/release. Quoting one against libdeflate would invent a
#      17% deficit that is not in the shipped code.
#
#   W3 IDENTIFIED BINARY. Hard stop #7: "a measurement from an unidentified binary is not
#      a measurement." Recorded with the run, not asserted in prose.
#
# Plus the four shared guards from lib.sh: G1 undeclared corpus member, G2 missing rival,
# G3 GATE needs CAMPAIGN_PROMOTE=1, G4 dirty/unidentified fulcrum.
#
#   scripts/campaign/board-wall.sh                     # tune set, L1-9, T1+T4
#   CAMPAIGN_LEVELS=6 CAMPAIGN_THREADS=1 scripts/campaign/board-wall.sh
#
# Args: [tune|gate|all]
# env: CAMPAIGN_LEVELS, CAMPAIGN_THREADS, CAMPAIGN_OUT, CAMPAIGN_N, CAMPAIGN_NO_FREEZE=1
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$HERE/lib.sh"

SET_NAME="${1:-tune}"
LEVELS="${CAMPAIGN_LEVELS:-1-9}"
THREADS="${CAMPAIGN_THREADS:-1,4}"
N="${CAMPAIGN_N:-9}"

campaign_guard_gate "$SET_NAME"
campaign_preflight
campaign_corpus_args "$SET_NAME"
campaign_rival_args

GZ="$CAMPAIGN_REPO/target/release/gzippy"
[ -x "$GZ" ] || die "no gzippy at $GZ" "build it: (cd $CAMPAIGN_REPO && cargo build --release)"
GZ_SHA="$(shasum -a 256 "$GZ" | cut -d' ' -f1)"

# --- W2: VANILLA BUILD ------------------------------------------------------------
# Refuse to quote a wall number from an instrumented binary. The anatomy/tune features
# compile in counters and are ~1.17x slower; a wall ratio from one is not about the
# shipped code. Checked by asking the BINARY, not by trusting the last build command.
#
# The FIRST version of this guard ran `gzippy --build-flavor` — a flag that does not
# exist. gzippy printed "Unknown option" AND EXITED 0, so the grep matched nothing and an
# instrumented binary sailed straight through. The guard was written by eye and never
# handed the input it existed to catch; it is the same defect as `fulcrum try`'s empty
# roundtrip command and `wallcensus`'s single-threaded rivals. Hence: ask a surface that
# EXISTS (`--version` has carried BUILD_FLAVOR since the build-flavor disconnect of
# 2026-06-24), assert the probe itself worked, and test the guard against a real
# instrumented build before trusting it.
FLAVOR="$("$GZ" --version 2>&1)" || die "gzippy --version failed" "$FLAVOR"
case "$FLAVOR" in
  *parallel-sm*|*legacy-serial*) : ;;   # probe reached the flavor string
  *) die "cannot read a build flavor out of 'gzippy --version'" \
         "got: $FLAVOR" \
         "This guard must never pass by failing to find a marker (see the note above)." ;;
esac
case "$FLAVOR" in
  *INSTRUMENTED*)
    die "target/release/gzippy is an INSTRUMENTED build" \
        "$FLAVOR" \
        "A tuned/instrumented build is ~1.17x slower and must never be quoted against a rival." \
        "Rebuild vanilla:  (cd $CAMPAIGN_REPO && cargo build --release)" ;;
esac

OUT="${CAMPAIGN_OUT:-$(campaign_outdir "wall-${SET_NAME}-$(git -C "$CAMPAIGN_REPO" rev-parse --short HEAD)")}"

# --- W3: IDENTIFIED BINARY --------------------------------------------------------
note "binary" "gzippy sha256=${GZ_SHA:0:16} commit=$CAMPAIGN_GZIPPY_SHA (vanilla: verified)"
note "method" "levels=$LEVELS threads=$THREADS n=$N sink=/dev/null (BOTH arms)"
note "out" "$OUT"

# --- W1: BOX FREEZE ---------------------------------------------------------------
# `freeze run` pauses the named processes and SIGCONTs them on EVERY exit path,
# including signals — see feedback_llama_pause_no_orphan. Opt out only for a smoke run.
FREEZE=()
if [ "${CAMPAIGN_NO_FREEZE:-0}" != "1" ]; then
  FREEZE=("$CAMPAIGN_FULCRUM" freeze run --ttl-s 7200 --procs "llama-swap,llama-server" --)
  note "freeze" "llama-swap,llama-server paused for the run (SIGCONT on every exit path)"
else
  # `note` uses a %s conversion, so an escape written here would print literally.
  printf '\033[1;33mfreeze DISABLED\033[0m (CAMPAIGN_NO_FREEZE=1) — smoke only, NOT quotable\n'
fi

# -p {threads} is passed EXPLICITLY at every cell, same receipt as board-size.sh: without
# an explicit -pN gzippy uses num_cpus, and an agent once measured T10 believing it was T1.
set -x
exec "${FREEZE[@]}" "$CAMPAIGN_FULCRUM" board wall \
  --ours "$GZ -{level} -p {threads} -c {input}" \
  "${CAMPAIGN_RIVAL_ARGS[@]}" \
  --levels "$LEVELS" \
  --threads "$THREADS" \
  "${CAMPAIGN_CORPUS_ARGS[@]}" \
  --n "$N" \
  --sink /dev/null \
  --roundtrip-cmd 'gzip -dc' \
  --ours-commit "$CAMPAIGN_GZIPPY_SHA" \
  --out "$OUT"
