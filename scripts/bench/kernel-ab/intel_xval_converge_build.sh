#!/usr/bin/env bash
# intel_xval_converge_build.sh — build the x86 binaries for the SOLE-PATH
# clean-convergence cross-validation (kernel-converge-A @cffa61ee):
#   gz-asmoff   : HEAD (cffa61ee) pure-rust-inflate ON, asm-kernel OFF
#                 -> CONVERGED engine A (flat fastloop + NEW decode_clean_careful_flat
#                    resumable tail; engine-B clean loop + lut_litlen double-build RETIRED)
#   gz-asmon    : HEAD (cffa61ee) unmodified -> x86 BMI2 run_contig (engine A cfg'd OUT)
#   gz-hybrid   : c4c3cc97 (pre-convergence PARENT) pure-rust-inflate ON, asm-kernel OFF
#                 -> HYBRID: engine A bulk fastloop + engine B's two-level careful TAIL
#                    + its per-block engine-B lut_litlen build (the double-build, D2 target)
# Gate-0: build-flavor=parallel-sm+pure all; sha pin; engine-A wire-in symbol present
# only in asm-off binaries; run_contig present only in asm-on.
set -euo pipefail
HEAD_SHA=${HEAD_SHA:-cffa61ee}
HYBRID_SHA=${HYBRID_SHA:-c4c3cc97}
SRC=/mnt/internal/gz-head
TGT=/dev/shm/ixv-target
OUT=/dev/shm/ixv
export RUSTFLAGS="-C target-cpu=native"
export CARGO_TARGET_DIR="$TGT"
mkdir -p "$OUT"
cd "$SRC"

build_asmoff() { # $1=sha $2=outname
  echo "== [$2] fetch+reset to $1 (asm-OFF) =="
  git fetch origin kernel-converge-A >/dev/null 2>&1
  git reset --hard "$1" >/dev/null 2>&1
  git submodule update --init vendor/isal-rs >/dev/null 2>&1 || true
  echo "  BUILT_SHA=$(git rev-parse --short HEAD)"
  git checkout Cargo.toml >/dev/null 2>&1 || true
  sed -i 's/^pure-rust-inflate = \["rpmalloc-caches", "asm-kernel"\]$/pure-rust-inflate = ["rpmalloc-caches"]/' Cargo.toml
  grep -n '^pure-rust-inflate' Cargo.toml
  cargo build --release --no-default-features --features gzippy-native 2>&1 | tail -2
  cp "$TGT/release/gzippy" "$OUT/$2"
  git checkout Cargo.toml >/dev/null 2>&1
}

build_asmon() { # $1=sha $2=outname
  echo "== [$2] fetch+reset to $1 (asm-ON, unmodified) =="
  git fetch origin kernel-converge-A >/dev/null 2>&1
  git reset --hard "$1" >/dev/null 2>&1
  git submodule update --init vendor/isal-rs >/dev/null 2>&1 || true
  echo "  BUILT_SHA=$(git rev-parse --short HEAD)"
  cargo build --release --no-default-features --features gzippy-native 2>&1 | tail -2
  cp "$TGT/release/gzippy" "$OUT/$2"
}

build_asmon  "$HEAD_SHA"   gz-asmon
build_asmoff "$HEAD_SHA"   gz-asmoff
build_asmoff "$HYBRID_SHA" gz-hybrid
# leave the checkout back at HEAD
git reset --hard "$HEAD_SHA" >/dev/null 2>&1

echo "== Gate-0: build flavor =="
for b in gz-asmon gz-asmoff gz-hybrid; do
  echo -n "  $b: "; GZIPPY_DEBUG=1 "$OUT/$b" --version 2>&1 | head -1 || true
done
echo "== Gate-0: engine-A wire-in symbol presence (objdump) =="
for b in gz-asmoff gz-hybrid gz-asmon; do
  echo -n "  $b fastloop_bounded: "; nm "$OUT/$b" 2>/dev/null | grep -c fastloop_bounded || true
done
echo -n "  gz-asmoff careful_flat: "; nm "$OUT/gz-asmoff" 2>/dev/null | grep -c careful_flat || true
echo -n "  gz-hybrid careful_flat: "; nm "$OUT/gz-hybrid" 2>/dev/null | grep -c careful_flat || true
echo "IXV_CONV_BUILD_DONE"
