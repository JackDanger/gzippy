#!/bin/bash
# pw_frozen.sh — FROZEN 3-way for the per-worker retention lever (task step 5).
# Runs ON THE GUEST under the host bench-lock. Cells: model T8, silesia T4,
# silesia T8, bignasa T8, silesia T16. Tools: rg=rapidgzip,
# gz1=/root/bin-head-isal (HEAD), gz2=/root/bin-pw (this lever, default ON).
# Interleaved best-of-9; EVERY run decodes to /dev/shm and is sha-verified
# (sha outside the timed window). A mismatch prints SHA-MISMATCH and aborts.
set -u
OUT=/dev/shm/pw_out.bin
declare -A PIN=( [model]=80521b40281d6ce7 [bignasa]=255c34ef2e0fdefe [silesia]=028bd002c89c9a90 )

mask() { case "$1" in 4) echo 0,2,4,6;; 8) echo 0,2,4,6,8,10,12,14;; 16) echo 0-15;; esac; }

run1() { # tool corpus T -> ms (and sha-verifies)
  local tool=$1 c=$2 T=$3 m s e ms got
  m=$(mask "$T")
  s=$(date +%s%N)
  case "$tool" in
    rg)  taskset -c "$m" rapidgzip -d -c -P "$T" "/root/$c.gz" > $OUT 2>/dev/null ;;
    gz1) taskset -c "$m" /root/bin-head-isal -d -c -p "$T" "/root/$c.gz" > $OUT 2>/dev/null ;;
    gz2) taskset -c "$m" /root/bin-pw -d -c -p "$T" "/root/$c.gz" > $OUT 2>/dev/null ;;
  esac
  e=$(date +%s%N); ms=$(( (e-s)/1000000 ))
  got=$(sha256sum $OUT | cut -c1-16)
  if [ "$got" != "${PIN[$c]}" ]; then
    echo "SHA-MISMATCH tool=$tool corpus=$c T=$T got=$got want=${PIN[$c]} — STOP"
    exit 9
  fi
  echo "$ms"
}

cell() { # corpus T
  local c=$1 T=$2 i a b d
  echo "=== cell $c T$T (ms per trial: rg gz1 gz2) ==="
  local -a RG=() G1=() G2=()
  for i in 1 2 3 4 5 6 7 8 9; do
    a=$(run1 rg  "$c" "$T") || exit 9
    b=$(run1 gz1 "$c" "$T") || exit 9
    d=$(run1 gz2 "$c" "$T") || exit 9
    RG+=("$a"); G1+=("$b"); G2+=("$d")
    echo "trial$i: $a $b $d"
  done
  best() { printf '%s\n' "$@" | sort -n | head -1; }
  med()  { printf '%s\n' "$@" | sort -n | sed -n 5p; }
  echo "BEST $c T$T rg=$(best "${RG[@]}") gz1=$(best "${G1[@]}") gz2=$(best "${G2[@]}")"
  echo "MED  $c T$T rg=$(med "${RG[@]}") gz1=$(med "${G1[@]}") gz2=$(med "${G2[@]}")"
}

date
cell model 8
cell silesia 4
cell silesia 8
cell bignasa 8
cell silesia 16
rm -f $OUT
echo FROZEN-DONE
