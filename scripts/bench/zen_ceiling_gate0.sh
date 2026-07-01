#!/usr/bin/env bash
# ZEN2 STEP-1 ceiling Gate-0 self-validation (UNPAUSED — correctness is load-independent,
# so we validate before pausing llama to keep the freeze window minimal).
# Validates: flavor=parallel-sm+pure, path=ParallelSM, base+rg sha==zcat, and NON-INERT
# for ALL THREE ceiling arms (u8 / u16-consumer / u16-worker): counter>0 AND sha differs.
set -u
: "${GZ:?}"; : "${RG:?}"; : "${CORPUS_DIR:=/root}"; : "${OUT:=/dev/shm/zen-ceil-out}"
: "${CELLS:?}"
mkdir -p "$OUT"; LOG="$OUT/gate0.log"; exec > "$LOG" 2>&1
fail(){ echo "GATE0_FAIL=$*"; echo "FAIL $*" > "$OUT/GATE0"; exit 2; }

echo "== ZEN ceiling GATE0 $(date -u +%FT%TZ) =="
echo "host: $(uname -srm) cores=$(nproc)  load: $(cat /proc/loadavg)"
[ -x "$GZ" ] || fail "no gz $GZ"; [ -x "$RG" ] || fail "no rg $RG"
SIL="$CORPUS_DIR/silesia.gz"; [ -f "$SIL" ] || fail "no silesia"

FLAVOR="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c "$SIL" >/dev/null 2>/tmp/flav.$$; grep -m1 'build-flavor=' /tmp/flav.$$ | sed -n 's/.*build-flavor=\([a-z+-]*\).*/\1/p')"
echo "build-flavor: '$FLAVOR' (expect parallel-sm+pure)"
[ "$FLAVOR" = "parallel-sm+pure" ] || fail "flavor='$FLAVOR'"
echo "gz_sha256: $(sha256sum "$GZ"|cut -c1-16)  rg: $("$RG" --version 2>&1|head -1)"

PATHL="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p4 "$SIL" >/dev/null 2>/tmp/p.$$; grep -m1 'path=' /tmp/p.$$)"
echo "silesia path: $PATHL"
echo "$PATHL" | grep -qE 'path=(ParallelSM|StoredParallel)' || fail "routing '$PATHL'"

SEEN=""
for cell in $CELLS; do
  corp="${cell%%:*}"; case " $SEEN " in *" $corp "*) continue;; esac; SEEN="$SEEN $corp"
  F="$CORPUS_DIR/$corp.gz"; [ -f "$F" ] || fail "missing $F"
  REF="$(zcat "$F"|sha256sum|cut -c1-16)"
  GB="$(GZIPPY_FORCE_PARALLEL_SM=1 "$GZ" -d -c -p4 "$F" 2>/dev/null|sha256sum|cut -c1-16)"
  RS="$("$RG" -d -c -P4 "$F" 2>/dev/null|sha256sum|cut -c1-16)"
  echo "$corp: ref=$REF gz_base=$([ "$GB" = "$REF" ]&&echo OK||echo BAD) rg=$([ "$RS" = "$REF" ]&&echo OK||echo BAD)"
  [ "$GB" = "$REF" ] || fail "$corp gz_base sha (production must be correct)"
  [ "$RS" = "$REF" ] || fail "$corp rg sha"

  # u8 ceiling
  H8="$(GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_MARKER_CEILING=1 GZIPPY_SLOW_HITS=1 "$GZ" -d -c -p4 "$F" 2>/tmp/c8.$$ >/tmp/o8.$$; grep -m1 'marker-CEILING oracle hits' /tmp/c8.$$ | grep -oE '[0-9]+$')"
  S8="$(cat /tmp/o8.$$|sha256sum|cut -c1-16)"
  echo "  u8   : hits=${H8:-0} sha=$S8 ($([ "$S8" != "$REF" ]&&echo DIFFERS-good||echo SAME-INERT))"
  [ "${H8:-0}" -gt 0 ] 2>/dev/null || fail "$corp u8 ceiling INERT (hits=${H8:-0})"
  [ "$S8" != "$REF" ] || fail "$corp u8 ceiling output == baseline (inert)"

  # u16 consumer-serial arm
  GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_MARKER_CEILING_U16=1 "$GZ" -d -c -p4 "$F" 2>/tmp/cu.$$ >/tmp/ou.$$
  HU="$(sed -n 's/.*consumer-serial) hits = \([0-9]*\).*/\1/p' /tmp/cu.$$ | head -1)"
  RBU="$(sed -n 's/.*consumer-serial).*resolve_bytes = \([0-9]*\).*/\1/p' /tmp/cu.$$ | head -1)"
  SU="$(cat /tmp/ou.$$|sha256sum|cut -c1-16)"
  echo "  u16c : hits=${HU:-0} resolve_bytes=${RBU:-0} sha=$SU ($([ "$SU" != "$REF" ]&&echo DIFFERS-good||echo SAME-INERT))"
  [ "${HU:-0}" -gt 0 ] 2>/dev/null || fail "$corp u16-consumer INERT (hits=${HU:-0})"
  [ "${RBU:-0}" -gt 0 ] 2>/dev/null || fail "$corp u16-consumer resolve_bytes=0 (resolve did not run)"
  [ "$SU" != "$REF" ] || fail "$corp u16-consumer output == baseline (inert)"

  # u16 worker-parallel arm
  GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_MARKER_CEILING_U16W=1 "$GZ" -d -c -p4 "$F" 2>/tmp/cw.$$ >/tmp/ow.$$
  HW="$(sed -n 's/.*worker-parallel) hits = \([0-9]*\).*/\1/p' /tmp/cw.$$ | head -1)"
  RBW="$(sed -n 's/.*worker-parallel).*resolve_bytes = \([0-9]*\).*/\1/p' /tmp/cw.$$ | head -1)"
  SW="$(cat /tmp/ow.$$|sha256sum|cut -c1-16)"
  echo "  u16w : hits=${HW:-0} resolve_bytes=${RBW:-0} sha=$SW ($([ "$SW" != "$REF" ]&&echo DIFFERS-good||echo SAME-INERT))"
  [ "${HW:-0}" -gt 0 ] 2>/dev/null || fail "$corp u16-worker INERT (hits=${HW:-0})"
  [ "${RBW:-0}" -gt 0 ] 2>/dev/null || fail "$corp u16-worker resolve_bytes=0 (resolve did not run)"
  [ "$SW" != "$REF" ] || fail "$corp u16-worker output == baseline (inert)"
done
echo "GATE0 PASS — all arms non-inert, base+rg correct, path+flavor OK"
echo PASS > "$OUT/GATE0"
