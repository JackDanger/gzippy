#!/usr/bin/env bash
# bmi2_ceiling_ab.sh — BMI2 ceiling-bound A/B (runs ON the build guest 199).
#
# Builds gzippy-native TWICE from origin/$BRANCH:
#   BMI2-ON  : RUSTFLAGS="-C target-cpu=native"            (production default)
#   BMI2-OFF : RUSTFLAGS="-C target-cpu=native -C target-feature=-bmi2"
# Then runs interleaved best-of-N wall on silesia, sha-verifying every trial.
# The ON-vs-OFF wall delta IS the BMI2 ceiling (byte-exact perturbation, no
# extrapolation). Routing asserted path=ParallelSM each binary.
set -euo pipefail

BRANCH="${BRANCH:-reimplement-isa-l}"
REPO="${REPO:-/root/gzippy}"
CORPUS="${CORPUS:-/root/gzippy/benchmark_data/silesia-large.gz}"
T="${T:-8}"
N="${N:-9}"
MASK="${MASK:-0-7}"

[ -f "$CORPUS" ] || { echo "FAIL: no corpus $CORPUS"; exit 7; }
REF_SHA="$(gzip -dc "$CORPUS" | sha256sum | cut -d' ' -f1)"
echo "ref_sha=$REF_SHA  corpus=$CORPUS  T=$T N=$N mask=$MASK"

cd "$REPO"
git fetch -q origin "$BRANCH"
git checkout -f -B "$BRANCH" "origin/$BRANCH" >/dev/null 2>&1
git reset --hard "origin/$BRANCH" >/dev/null 2>&1
echo "head=$(git rev-parse --short HEAD)"

build() { # $1=label  $2=rustflags
  echo "## build $1 ($2)"
  RUSTFLAGS="$2" cargo build --release --no-default-features \
    --features gzippy-native >/tmp/build-$1.log 2>&1 \
    || { echo "FAIL build $1"; tail -20 /tmp/build-$1.log; exit 8; }
  cp target/release/gzippy "/tmp/gzippy-$1"
}

build on  "-C target-cpu=native"
build off "-C target-cpu=native -C target-feature=-bmi2"

# confirm OFF actually lacks bmi2 in codegen: it must still be byte-exact + ParallelSM
for b in on off; do
  dbg="$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 "/tmp/gzippy-$b" -d -c -p "$T" "$CORPUS" 2>&1 >/dev/null | grep -m1 'path=' || true)"
  case "$dbg" in *ParallelSM*) ;; *) echo "FAIL routing $b: $dbg"; exit 9;; esac
  sha="$("/tmp/gzippy-$b" -d -c -p "$T" "$CORPUS" 2>/dev/null | sha256sum | cut -d' ' -f1)"
  [ "$sha" = "$REF_SHA" ] || { echo "FAIL sha $b: $sha != $REF_SHA"; exit 5; }
  echo "routing+sha OK: $b ($dbg)"
done

run1() { # $1=binary -> prints seconds
  local s e
  s=$(date +%s.%N)
  taskset -c "$MASK" "/tmp/gzippy-$1" -d -c -p "$T" "$CORPUS" >/dev/null 2>/dev/null
  e=$(date +%s.%N)
  awk "BEGIN{printf \"%.4f\", $e-$s}"
}

echo "## interleaved best-of-$N (ON vs OFF)"
on_best=99; off_best=99
on_list=""; off_list=""
i=0
while [ "$i" -lt "$N" ]; do
  o=$(run1 on);  on_list="$on_list $o";  awk "BEGIN{exit !($o<$on_best)}"  && on_best=$o
  f=$(run1 off); off_list="$off_list $f"; awk "BEGIN{exit !($f<$off_best)}" && off_best=$f
  i=$((i+1))
done
echo "ON  walls:$on_list  best=$on_best"
echo "OFF walls:$off_list best=$off_best"
awk "BEGIN{d=($off_best-$on_best); printf \"DELTA(off-on)=%.4fs  bmi2_speedup_on=%.2f%%\n\", d, 100*d/$off_best}"
