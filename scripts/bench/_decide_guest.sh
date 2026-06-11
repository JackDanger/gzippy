#!/usr/bin/env bash
# _decide_guest.sh — GUEST-SIDE half of `fulcrum decide` (plans/fulcrum2-charter.md).
#
# ONE pass over the requested cells producing, per cell, under $ARTDIR/$RUNID/:
#   wall_gz.txt / wall_rg.txt        interleaved wall samples (canonical mask,
#                                    regular-file sink, EVERY run sha-verified)
#   trace.json + verbose.txt         GZIPPY_TIMELINE capture + counter sidecar
#   prof.txt                         GZIPPY_CONTIG_PROF capture (engine classes)
#   knob_<name>/{base.txt,knob.txt,  same-binary kill-switch A/B samples +
#                effect_*.txt}       per-arm effect captures (counter predicates)
#   manifest.txt                     provenance: bin sha, freeze/quiet readbacks,
#                                    masks, routing asserts, corpus pins
#
# Discipline is the parity spine's, via lib_decide_guest.sh (VERBATIM-copied,
# provenance-labeled primitives): allowlist env scrub (hard-fail on seed/oracle/
# slow vars), freeze readback (CONCRETE-WRONG never overridable), instantaneous
# procs_running quiet gate, regular-file sinks, sha-verify every measured run.
#
# Inputs (env): GUEST_SRC BIN FEATURE CELLS N KNOB_N DO_KNOBS KNOB_CELLS
#   ARTDIR RUNID RG RG_TRACE GOV NO_TURBO HOST_FROZEN ALLOW_LOAD CORPUS_RAW_SHA256
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "$HERE/lib_decide_guest.sh"

: "${GUEST_SRC:?}"; : "${BIN:?}"; : "${FEATURE:?}"; : "${CELLS:?}"
: "${ARTDIR:?}"; : "${RUNID:?}"
N="${N:-9}"; KNOB_N="${KNOB_N:-7}"
DO_KNOBS="${DO_KNOBS:-1}"; KNOB_CELLS="${KNOB_CELLS:-silesia:1,silesia:16}"
KNOB_FILTER="${KNOB_FILTER:-}"   # comma list of knob names; empty = all
RG="${RG:-rapidgzip}"; RG_TRACE="${RG_TRACE:-}"
GOV="${GOV:-performance}"; NO_TURBO="${NO_TURBO:-1}"
CORPUS_RAW_SHA256="${CORPUS_RAW_SHA256:-}"

OUT="$ARTDIR/$RUNID"
mkdir -p "$OUT"
MANIFEST="$OUT/manifest.txt"
: > "$MANIFEST"
mf() { echo "$1" >> "$MANIFEST"; }

cd "$GUEST_SRC" 2>/dev/null || true   # not load-bearing; binary is pre-staged

# ---- contamination guard + box state (spine discipline) ----------------------
scrub_gzippy_env ""           # nothing extra allowlisted for the baseline state
freeze_readback               # sets ACT_GOV/ACT_TURBO/ACT_AFFIN/FREEZE_STATE or dies
quiet_gate                    # sets RUN_AVG/QUIET_STATE or dies

[ -x "$BIN" ] || decide_fail "no-binary:$BIN" 5
BIN_SHA="$(sha256sum "$BIN" | cut -d' ' -f1)"
RG_CMD=""
if command -v "$RG" >/dev/null 2>&1; then RG_CMD="$RG"
elif [ -n "$RG_TRACE" ] && [ -x "$RG_TRACE" ]; then RG_CMD="$RG_TRACE"
else decide_fail "no-rapidgzip" 12; fi

mf "runid=$RUNID"
mf "bin=$BIN"
mf "bin_sha=$BIN_SHA"
mf "feature=$FEATURE"
mf "rg_version=$("$RG_CMD" --version 2>&1 | head -1 | tr -d '\n')"
mf "freeze_state=$FREEZE_STATE"
mf "quiet_state=$QUIET_STATE"
mf "governor=$ACT_GOV"
mf "no_turbo=$ACT_TURBO"
mf "runnable_avg=${RUN_AVG:-NA}"
mf "host_frozen_ack=$HOST_FROZEN"
mf "n=$N"
mf "knob_n=$KNOB_N"
mf "do_knobs=$DO_KNOBS"
mf "knob_cells=$KNOB_CELLS"
mf "cells=$CELLS"
mf "started=$(date -u '+%FT%TZ')"

# ---- knob registry (plans/fulcrum2-charter.md section A) ----------------------
# name|ENV=val(the non-default arm)|effect-predicate-id
# NOTE: GZIPPY_PIN_WORKERS from the directive does NOT exist in src (grep-verified
# at 93a19384); GZIPPY_EAGER_POSTPROC (in-tree, A/B-able) is carried instead.
KNOB_REGISTRY="
dist_amort|GZIPPY_DIST_AMORT=0|prof_dist
stored_flip|GZIPPY_NO_STORED_FLIP=1|none
seeded_block|GZIPPY_SEEDED_BLOCK=0|verbose_seeded
exact_block|GZIPPY_EXACT_BLOCK=0|verbose_exact
hit_drive|GZIPPY_NO_HIT_DRIVE=1|none
slab_alloc|GZIPPY_SLAB_ALLOC=1|rpmalloc_stats
slab_off|GZIPPY_SLAB_ALLOC=0|rpmalloc_stats_off
eager_postproc|GZIPPY_EAGER_POSTPROC=1|none
"

# ---- corpus reference oracles -------------------------------------------------
declare -A REFSHA RAWBYTES
corpus_path() { echo "/root/$1.gz"; }
ensure_ref() { # <corpus>
  local c="$1" f
  [ -n "${REFSHA[$c]:-}" ] && return 0
  f="$(corpus_path "$c")"
  [ -f "$f" ] || decide_fail "no-corpus:$f" 7
  REFSHA[$c]="$(gzip -dc "$f" | sha256sum | cut -d' ' -f1)"
  RAWBYTES[$c]="$(gzip -dc "$f" | wc -c)"
  if [ "$c" = silesia ] && [ -n "$CORPUS_RAW_SHA256" ] \
     && [ "${REFSHA[$c]}" != "$CORPUS_RAW_SHA256" ]; then
    decide_fail "corpus-sha-drift silesia got=${REFSHA[$c]} pin=$CORPUS_RAW_SHA256" 7
  fi
  mf "corpus_${c}_sha=${REFSHA[$c]}"
  mf "corpus_${c}_raw_bytes=${RAWBYTES[$c]}"
}

# ---- helpers -------------------------------------------------------------------
SINK_A="$OUT/sink_a.bin"; SINK_B="$OUT/sink_b.bin"
assert_regular_sink "$SINK_A"; assert_regular_sink "$SINK_B"

expect_path_for() { # <T> -> the production DecodePath this cell must take
  if [ "$FEATURE" = "gzippy-isal" ] && [ "$1" = 1 ]; then echo IsalSingleShot
  else echo ParallelSM; fi
}

routing_assert() { # <corpus> <T>
  local f t="$2" want dbg
  f="$(corpus_path "$1")"; want="$(expect_path_for "$t")"
  dbg="$(GZIPPY_DEBUG=1 "$BIN" -d -c -p "$t" "$f" 2>&1 >/dev/null | grep -m1 'path=' || true)"
  case "$dbg" in
    *path=$want*) ;;
    *) decide_fail "routing $1:T$t not $want (got: ${dbg:-<no path= line>})" 9;;
  esac
}

# ---- per-cell measurement -------------------------------------------------------
run_cell() { # <corpus> <T>
  local c="$1" t="$2" mask cdir f i gsec gsha rsec rsha GZT="" RGT="" DIVERGED=0
  mask="$(pin_mask "$t")"
  [ -n "$mask" ] || decide_fail "bad-T:$t (use 1,4,8,16)" 8
  ensure_ref "$c"
  f="$(corpus_path "$c")"
  cdir="$OUT/cell_${c}_T${t}"; mkdir -p "$cdir"
  routing_assert "$c" "$t"
  echo "## cell $c:T$t mask=$mask — wall interleave N=$N (drop warmup iter0)"
  for ((i=0;i<=N;i++)); do
    read -r gsec gsha _grss < <(timed_masked "$mask" "$SINK_A" "$BIN" -d -c -p "$t" "$f")
    read -r rsec rsha _rrss < <(timed_masked "$mask" "$SINK_B" "$RG_CMD" -d -c -f -P "$t" "$f")
    [ "$i" -eq 0 ] && continue
    GZT="$GZT $gsec"; RGT="$RGT $rsec"
    [ "$gsha" = "${REFSHA[$c]}" ] || { echo "!! SHA DIVERGENCE gz $c:T$t i=$i sha=$gsha"; DIVERGED=1; }
    [ "$rsha" = "${REFSHA[$c]}" ] || { echo "!! SHA DIVERGENCE rg $c:T$t i=$i sha=$rsha"; DIVERGED=1; }
  done
  [ "$DIVERGED" -eq 0 ] || decide_fail "sha-mismatch $c:T$t (wrong-bytes — number VOID)" 11
  echo "$GZT" | tr ' ' '\n' | grep -v '^$' > "$cdir/wall_gz.txt"
  echo "$RGT" | tr ' ' '\n' | grep -v '^$' > "$cdir/wall_rg.txt"
  mf "cell_done=$c:$t:mask=$mask:sha_ok=1"

  # -- trace capture (1 run; counters/trace are NOT wall numbers -> labeled).
  #    REGULAR-FILE sink, never a pipe (the writev-phantom class): the spine rule
  #    applies to captures too — backpressure would distort the trace itself. ----
  echo "## cell $c:T$t — trace+counter capture (labeled unfrozen-counters)"
  local tjson="$cdir/trace.json" verb="$cdir/verbose.txt" tsha
  rm -f "$tjson"
  assert_regular_sink "$SINK_A"
  GZIPPY_TIMELINE="$tjson" GZIPPY_VERBOSE=1 \
    taskset -c "$mask" "$BIN" -d -c -p "$t" "$f" >"$SINK_A" 2>"$verb"
  tsha="$(sha256sum "$SINK_A" | cut -d' ' -f1)"
  [ "$tsha" = "${REFSHA[$c]}" ] || decide_fail "sha-mismatch trace-capture $c:T$t" 11
  [ -s "$tjson" ] || echo "## WARN: empty trace $c:T$t (analyzer will refuse that row)"

  # -- contig_prof capture (1 run; rdtsc shares; perturbs ~25cyc/iter — never a wall) --
  echo "## cell $c:T$t — contig_prof capture"
  local prof="$cdir/prof.txt" psha
  assert_regular_sink "$SINK_A"
  GZIPPY_CONTIG_PROF=1 GZIPPY_VERBOSE=1 \
    taskset -c "$mask" "$BIN" -d -c -p "$t" "$f" >"$SINK_A" 2>"$prof"
  psha="$(sha256sum "$SINK_A" | cut -d' ' -f1)"
  [ "$psha" = "${REFSHA[$c]}" ] || decide_fail "sha-mismatch prof-capture $c:T$t" 11
}

# ---- knob A/B on one cell --------------------------------------------------------
run_knobs_for_cell() { # <corpus> <T>
  local c="$1" t="$2" mask f cdir line name envkv pred
  mask="$(pin_mask "$t")"; f="$(corpus_path "$c")"
  cdir="$OUT/cell_${c}_T${t}"; mkdir -p "$cdir"
  ensure_ref "$c"
  while IFS='|' read -r name envkv pred; do
    [ -n "$name" ] || continue
    if [ -n "$KNOB_FILTER" ]; then
      case ",$KNOB_FILTER," in *",$name,"*) ;; *) continue;; esac
    fi
    local kdir="$cdir/knob_${name}" i bsec bsha brss ksec ksha krss BS="" KS="" DIV=0
    local RSS_BASE=0 RSS_KNOB=0 RSS_N=0
    mkdir -p "$kdir"
    local kvar="${envkv%%=*}" kval="${envkv#*=}"
    echo "## knob $name ($envkv) on $c:T$t — same-binary A/B, KNOB_N=$KNOB_N pairs"
    for ((i=0;i<=KNOB_N;i++)); do
      read -r bsec bsha brss < <(timed_masked "$mask" "$SINK_A" "$BIN" -d -c -p "$t" "$f")
      read -r ksec ksha krss < <(timed_masked "$mask" "$SINK_B" env "$kvar=$kval" "$BIN" -d -c -p "$t" "$f")
      [ "$i" -eq 0 ] && continue
      BS="$BS $bsec"; KS="$KS $ksec"
      [ "$bsha" = "${REFSHA[$c]}" ] || { echo "!! SHA base $name i=$i"; DIV=1; }
      [ "$ksha" = "${REFSHA[$c]}" ] || { echo "!! SHA knob $name i=$i sha=$ksha"; DIV=1; }
      # Accumulate RSS (last value wins — peak RSS is stable across iterations).
      [ "${brss:-0}" -gt 0 ] && RSS_BASE="$brss"
      [ "${krss:-0}" -gt 0 ] && RSS_KNOB="$krss"
    done
    if [ "$DIV" -ne 0 ]; then
      # A knob arm with wrong bytes is its own finding (the switch is NOT
      # byte-transparent) — record, don't rank.
      echo "knob_sha_fail=$name" >> "$kdir/meta.txt"
      mf "knob_sha_fail=$c:$t:$name"
      continue
    fi
    echo "$BS" | tr ' ' '\n' | grep -v '^$' > "$kdir/base.txt"
    echo "$KS" | tr ' ' '\n' | grep -v '^$' > "$kdir/knob.txt"
    {
      echo "knob=$name"; echo "env=$envkv"; echo "pred=$pred"
      echo "cell=$c:$t"; echo "mask=$mask"; echo "sha_ok=1"
      echo "rss_base_mb=$RSS_BASE"; echo "rss_knob_mb=$RSS_KNOB"
    } > "$kdir/meta.txt"
    mf "knob_done=$c:$t:$name"
  done <<< "$KNOB_REGISTRY"
}

# ---- knob effect captures (once, on the FIRST knob cell; predicates -> analyzer) --
run_knob_effects() { # <corpus> <T>
  local c="$1" t="$2" mask f line name envkv pred
  mask="$(pin_mask "$t")"; f="$(corpus_path "$c")"
  ensure_ref "$c"
  local edir="$OUT/knob_effects_${c}_T${t}"; mkdir -p "$edir"
  while IFS='|' read -r name envkv pred; do
    [ -n "$name" ] || continue
    if [ -n "$KNOB_FILTER" ]; then
      case ",$KNOB_FILTER," in *",$name,"*) ;; *) continue;; esac
    fi
    local kvar="${envkv%%=*}" kval="${envkv#*=}" sha
    echo "## knob-effect capture $name ($envkv) on $c:T$t"
    assert_regular_sink "$SINK_A"
    # GZIPPY_RPMALLOC_STATS=1 is always set so the rpmalloc_stats predicate can
    # verify slab_alloc engagement (rpmalloc_alloc.rs:523 prints stats only when
    # this env var is set; it is a no-op for non-rpmalloc knobs).
    GZIPPY_VERBOSE=1 GZIPPY_CONTIG_PROF=1 GZIPPY_RPMALLOC_STATS=1 taskset -c "$mask" \
      "$BIN" -d -c -p "$t" "$f" >"$SINK_A" 2>"$edir/effect_base_${name}.txt"
    sha="$(sha256sum "$SINK_A" | cut -d' ' -f1)"
    [ "$sha" = "${REFSHA[$c]}" ] || echo "effect_base_sha_fail=$name" >> "$edir/fails.txt"
    assert_regular_sink "$SINK_A"
    GZIPPY_VERBOSE=1 GZIPPY_CONTIG_PROF=1 GZIPPY_RPMALLOC_STATS=1 env "$kvar=$kval" \
      taskset -c "$mask" \
      "$BIN" -d -c -p "$t" "$f" >"$SINK_A" 2>"$edir/effect_knob_${name}.txt"
    sha="$(sha256sum "$SINK_A" | cut -d' ' -f1)"
    [ "$sha" = "${REFSHA[$c]}" ] || echo "effect_knob_sha_fail=$name" >> "$edir/fails.txt"
  done <<< "$KNOB_REGISTRY"
  mf "knob_effects=$c:$t"
}

# ---- main loop -------------------------------------------------------------------
IFS=',' read -ra CELL_ARR <<< "$CELLS"
for cell in "${CELL_ARR[@]}"; do
  corpus="${cell%%:*}"; tt="${cell##*:}"
  run_cell "$corpus" "$tt"
done

if [ "$DO_KNOBS" = 1 ]; then
  first_knob_cell=1
  IFS=',' read -ra KCELL_ARR <<< "$KNOB_CELLS"
  for cell in "${KCELL_ARR[@]}"; do
    corpus="${cell%%:*}"; tt="${cell##*:}"
    routing_assert "$corpus" "$tt"
    run_knobs_for_cell "$corpus" "$tt"
    if [ "$first_knob_cell" = 1 ]; then
      run_knob_effects "$corpus" "$tt"
      first_knob_cell=0
    fi
  done
fi

rm -f "$SINK_A" "$SINK_B"
mf "finished=$(date -u '+%FT%TZ')"
echo "DECIDE_ARTIFACTS=$OUT"
echo "DECIDE_GUEST_DONE"
