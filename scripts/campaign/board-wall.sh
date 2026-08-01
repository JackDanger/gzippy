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

# --- W4: FROZEN BINARY ------------------------------------------------------------
# W2 and W3 verify `$CAMPAIGN_REPO/target/release/gzippy` ONCE, here, and then the run
# spends HOURS reading that same path out of a checkout that stays fully writable. A
# single `cargo build --release` anywhere in the tree silently swaps the subject
# mid-board, and nothing downstream notices: the launch banner still says
# "vanilla: verified", the log lines carry no binary identity, and the rows before and
# after the swap are indistinguishable.
#
# RECEIPT (2026-08-01). A wall board launched against a vanilla main build
# (sha 54079f43) was 242 rows into ~792 when a doc-comment edit was build-checked with
# `cargo build --release` in the same checkout — which was sitting on a LEVER branch,
# 307 insertions across 11 files of `src/` ahead of main. The subject binary became
# 1c812b24, a different compressor, with no error and no log line. The whole run had to
# be discarded because the contamination BOUNDARY could not be recovered from the log.
# This is the same class as the CAMPAIGN_REPO trap caught earlier the same day: a guard
# that reads an input once and then trusts the world to hold still.
#
# The fix is to stop measuring a path that someone else can write. Copy the verified
# binary into the run's own output directory and measure THAT. Three properties follow:
# the subject is immutable for the run (nothing in the repo can reach it), the artifact
# carries the exact binary that produced its numbers (hard stop #7 satisfied by
# construction rather than by a note), and the checkout is free for ordinary work while
# a multi-hour board runs — which is what will actually happen, so the guard has to
# assume it.
mkdir -p "$OUT" || die "cannot create out dir $OUT"
OUT_ABS="$(cd "$OUT" && pwd)" || die "cannot resolve out dir $OUT"
GZ_FROZEN="$OUT_ABS/gzippy-subject"

# A previous run of the SAME commit leaves a read-only snapshot here, and `cp`
# onto a mode-444 file fails — the guard then refuses every re-run, which is how
# it behaved the first time a killed board was relaunched. Fail closed is right;
# refusing forever is not. Clear the stale snapshot first, under the rm
# discipline: the path is RESOLVED ABSOLUTE above, is asserted to sit strictly
# inside the campaign artifact root, and names exactly one file (never a tree,
# never a glob).
# The assert gates the REMOVAL, not the run: a fresh out dir has nothing to
# delete, so an explicit CAMPAIGN_OUT (smoke runs, scratch dirs) must still work.
if [ -e "$GZ_FROZEN" ]; then
  [ -f "$GZ_FROZEN" ] || die "snapshot path exists and is not a regular file: $GZ_FROZEN"
  CAMPAIGN_ARTIFACT_ROOT="$(cd "$CAMPAIGN_REPO/.." && pwd)/gzippy-bench/campaign"
  case "$GZ_FROZEN" in
    "$CAMPAIGN_ARTIFACT_ROOT"/*/gzippy-subject|/private/tmp/*/gzippy-subject|/tmp/*/gzippy-subject) : ;;
    *) die "refusing to remove a snapshot outside the campaign artifact root" \
           "path: $GZ_FROZEN" \
           "root: $CAMPAIGN_ARTIFACT_ROOT" ;;
  esac
  rm -f "$GZ_FROZEN" || die "cannot clear stale snapshot $GZ_FROZEN"
fi

cp "$GZ" "$GZ_FROZEN" || die "cannot snapshot subject binary into $OUT_ABS"
chmod a-w "$GZ_FROZEN" || die "cannot make snapshot read-only"
FROZEN_SHA="$(shasum -a 256 "$GZ_FROZEN" | cut -d' ' -f1)"
[ "$FROZEN_SHA" = "$GZ_SHA" ] || die "snapshot sha != verified sha" \
    "verified=$GZ_SHA snapshot=$FROZEN_SHA" \
    "The copy did not reproduce the binary that passed W2/W3."
GZ="$GZ_FROZEN"

# --- W3: IDENTIFIED BINARY --------------------------------------------------------
note "binary" "gzippy sha256=${GZ_SHA:0:16} commit=$CAMPAIGN_GZIPPY_SHA (vanilla: verified)"
note "subject" "frozen copy at $GZ_FROZEN (read-only; repo rebuilds cannot reach it)"
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
