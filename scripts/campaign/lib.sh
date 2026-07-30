#!/usr/bin/env bash
# lib.sh — the ONE place the encoder campaign's measurement surface is defined.
#
# WHY THIS FILE EXISTS. Before it, every census was an uncommitted shell script in
# ~/www/gzippy-bench (not a git repo — eight such scripts, zero version control), and each
# one encoded its own corpus and its own rival set. The consequences were not hypothetical:
#
#   * `run_sizecensus_threads.sh` declared THREE rivals. igzip was simply absent, with no
#     note, so "igzip: 1 failing cell" in the campaign plan was never measured by it.
#   * The two FALSIFY records that currently gate the block-END work (campaign plan §4
#     "block budget 300K -> 900K", and commit 7fdb742b) were measured on `logs.txt`,
#     `text-1MB` and `shortmatch-4M`. NONE of those three is a declared member of the
#     corpus split. One of them is not on any box. A FALSIFY note is BINDING under
#     CLAUDE.md hard stop #2, so undeclared-corpus evidence is currently blocking work.
#   * A census that silently covered "17 of the 20 canonical members" set the headline
#     165-cell figure. The missing members were never named.
#
# So the guards below are mechanical, not remembered. Every one cites the incident it
# prevents. A guard that fires prints what to do about it and exits non-zero; none of them
# can be satisfied by trying again.
#
# Source it, then call `campaign_preflight`.
set -uo pipefail

CAMPAIGN_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_REPO="$(cd "$CAMPAIGN_LIB_DIR/../.." && pwd)"

# --- knobs that are LOCATIONS, never behaviour -------------------------------------
# CLAUDE.md non-negotiable #3 forbids env vars that change what the compressor does.
# These change only where files are found and which declared set is measured; none of
# them reaches the encoder. Keep it that way.
CAMPAIGN_CORPUS_ROOT="${CAMPAIGN_CORPUS_ROOT:-$HOME/www/gzippy-bench/corpus}"
CAMPAIGN_FULCRUM="${CAMPAIGN_FULCRUM:-$HOME/www/fulcrum/target/release/fulcrum}"
CAMPAIGN_SPLIT="${CAMPAIGN_SPLIT:-$CAMPAIGN_REPO/corpus_split.json}"
CAMPAIGN_OUT_ROOT="${CAMPAIGN_OUT_ROOT:-$HOME/www/gzippy-bench/campaign}"

die() { printf '\n\033[1;31mCAMPAIGN REFUSES\033[0m %s\n' "$1" >&2; shift; for l in "$@"; do printf '  %s\n' "$l" >&2; done; echo >&2; exit 2; }
note() { printf '\033[1;34m%s\033[0m %s\n' "$1" "$2"; }

# --- G1: the corpus set is DECLARED, and membership is checked ----------------------
# Receipt: two binding FALSIFY records rest on logs.txt / text-1MB / shortmatch-4M, none
# of which is a declared member. This function is the only way to get --corpus args, so
# an undeclared file cannot enter a census by accident again.
#
# campaign_corpus_args <tune|gate|all>  -> sets CAMPAIGN_CORPUS_ARGS, CAMPAIGN_CORPUS_N
campaign_corpus_args() {
  local set_name="$1"
  case "$set_name" in tune|gate|all) ;; *) die "unknown corpus set '$set_name'" "declared sets: tune, gate, all";; esac
  [ -f "$CAMPAIGN_SPLIT" ] || die "no corpus split at $CAMPAIGN_SPLIT" \
    "The TUNE/GATE contract governs whether a promotion is even valid; it must live" \
    "beside the code it grades. Restore it from the fulcrum repo."
  [ -d "$CAMPAIGN_CORPUS_ROOT" ] || die "no corpus at $CAMPAIGN_CORPUS_ROOT" "set CAMPAIGN_CORPUS_ROOT"

  local listing; listing="$(python3 "$CAMPAIGN_LIB_DIR/split.py" "$CAMPAIGN_SPLIT" "$set_name" "$CAMPAIGN_CORPUS_ROOT")" || exit 2
  # split.py exits non-zero and names the missing members itself; it never shrinks a set.
  CAMPAIGN_CORPUS_ARGS=(); CAMPAIGN_CORPUS_N=0
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    CAMPAIGN_CORPUS_ARGS+=(--corpus "$CAMPAIGN_CORPUS_ROOT/$f")
    CAMPAIGN_CORPUS_N=$((CAMPAIGN_CORPUS_N + 1))
  done <<< "$listing"
  [ "$CAMPAIGN_CORPUS_N" -gt 0 ] || die "corpus set '$set_name' resolved to zero files"
  note "corpus" "set=$set_name files=$CAMPAIGN_CORPUS_N root=$CAMPAIGN_CORPUS_ROOT"
}

# --- G2: all four rivals, or an EXPLICIT declared absence ---------------------------
# Receipt: the shipped size census measured three rivals and said nothing about the
# fourth. CLAUDE.md grades against gzip AND pigz AND libdeflate AND igzip; a census
# missing one has not evaluated the goal, so silence is the failure mode to kill.
#
# To proceed without a rival you must say so and say why:
#   CAMPAIGN_ALLOW_MISSING_RIVAL='igzip=not packaged for aarch64 darwin'
campaign_rival_args() {
  CAMPAIGN_RIVAL_ARGS=(); local missing=()
  local igzip_local="$CAMPAIGN_REPO/vendor/isa-l/build/igzip"

  _rival() { # name, binary, template
    local name="$1" bin="$2" tmpl="$3"
    if command -v "$bin" >/dev/null 2>&1 || [ -x "$bin" ]; then
      CAMPAIGN_RIVAL_ARGS+=(--rival "$name=$tmpl")
    else
      missing+=("$name")
    fi
  }
  _rival gzip       gzip            'gzip -{level} -c {input}'
  _rival pigz       pigz            'pigz -{level} -p {threads} -c {input}'
  _rival libdeflate libdeflate-gzip 'libdeflate-gzip -{level} -c {input}'
  if [ -x "$igzip_local" ]; then
    CAMPAIGN_RIVAL_ARGS+=(--rival "igzip=$igzip_local -{level} -T {threads} -c {input}")
  elif command -v igzip >/dev/null 2>&1; then
    CAMPAIGN_RIVAL_ARGS+=(--rival 'igzip=igzip -{level} -T {threads} -c {input}')
  else
    missing+=(igzip)
  fi

  if [ "${#missing[@]}" -gt 0 ]; then
    local declared="${CAMPAIGN_ALLOW_MISSING_RIVAL:-}"
    for m in "${missing[@]}"; do
      case "$declared" in
        *"$m="*) note "rival" "DECLARED ABSENT: $m — reason recorded in the artifact";;
        *) die "rival '$m' is not on this box" \
             "CLAUDE.md grades per-label against gzip, pigz, libdeflate AND igzip." \
             "A census missing a rival has not evaluated the goal." \
             "" \
             "Build it:   make -C $CAMPAIGN_REPO vendor/isa-l/build/igzip   (for igzip)" \
             "Or declare the absence WITH a reason, which is stamped into the artifact:" \
             "  CAMPAIGN_ALLOW_MISSING_RIVAL='$m=why it cannot be here' \$0 ...";;
      esac
    done
  fi
  CAMPAIGN_RIVAL_N=$(( ${#CAMPAIGN_RIVAL_ARGS[@]} / 2 ))
  note "rivals" "$CAMPAIGN_RIVAL_N of 4 declared${missing[0]+ (absent: ${missing[*]})}"
}

# --- G3/G4/G5: the instrument and the subject are both identified -------------------
# Receipt (G4): this session opened with fulcrum refusing to measure — a DIRTY build one
# commit behind origin/main. A refusing instrument is what makes hand-rolled tables the
# path of least resistance, which is how the undeclared-corpus falsifications happened.
# Receipt (G5): "we once spent weeks measuring a binary CI wasn't shipping."
campaign_preflight() {
  [ -x "$CAMPAIGN_FULCRUM" ] || die "no fulcrum at $CAMPAIGN_FULCRUM" \
    "build it: (cd ~/www/fulcrum && cargo build --release)"
  local fv; fv="$("$CAMPAIGN_FULCRUM" version --json 2>/dev/null)" || die "fulcrum version failed"
  python3 "$CAMPAIGN_LIB_DIR/instrument.py" <<<"$fv" || exit 2
  CAMPAIGN_FULCRUM_SHA="$(python3 -c 'import json,sys;print(json.load(sys.stdin)["commit"])' <<<"$fv")"

  CAMPAIGN_GZIPPY_SHA="$(git -C "$CAMPAIGN_REPO" rev-parse HEAD)"
  CAMPAIGN_GZIPPY_DIRTY=0
  git -C "$CAMPAIGN_REPO" diff --quiet HEAD -- src Cargo.toml Cargo.lock 2>/dev/null || CAMPAIGN_GZIPPY_DIRTY=1
  note "instrument" "fulcrum=${CAMPAIGN_FULCRUM_SHA:0:12} clean"
  note "subject" "gzippy=${CAMPAIGN_GZIPPY_SHA:0:12} dirty=$CAMPAIGN_GZIPPY_DIRTY"
  [ "$CAMPAIGN_GZIPPY_DIRTY" = 0 ] || note "WARNING" "src is dirty — the artifact records a sha that does not describe the binary"
}

# --- G3: GATE is promotion-only, and that is enforced, not remembered ---------------
# corpus_split.json's own contract: "A promotion is judged on GATE. If a change was
# fitted on GATE, the promotion is void regardless of the numbers." Exploration must
# therefore not reach for GATE casually. Note the campaign plan currently names
# access.log and monorepo.tar — both GATE members — as Front B's headline targets.
campaign_guard_gate() {
  local set_name="$1"
  if [ "$set_name" != tune ] && [ "${CAMPAIGN_PROMOTE:-0}" != 1 ]; then
    die "corpus set '$set_name' includes GATE members and CAMPAIGN_PROMOTE is not 1" \
      "corpus_split.json: GATE files are run ONLY at promotion time and NEVER inspected" \
      "while choosing a parameter. Fitting on GATE voids the promotion." \
      "" \
      "Exploring?  use the tune set." \
      "Promoting?  CAMPAIGN_PROMOTE=1 \$0 ...  (stamped into the artifact)"
  fi
}

campaign_outdir() { # <name>
  local d="$CAMPAIGN_OUT_ROOT/$1"
  mkdir -p "$d" || die "cannot create $d"
  printf '%s' "$d"
}
