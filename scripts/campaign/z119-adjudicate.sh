#!/usr/bin/env bash
# z119 adjudication leg: rivals on the priced cells, the len-3 fp axis on
# access.log L5, and the interleaved wall min-of-5. Requires the sizes sweep
# to have finished (nothing here may compete with it for CPU).
set -uo pipefail
CORPUS="$HOME/www/gzippy-bench/corpus"
BASE=/tmp/z119-wt-branch/target/release/gzippy
TUNE="$HOME/www/gzippy-bench/gzippy-tune-z119"
RIVAL_DIR=/Users/jackdanger/www/gzippy/rig
OUT="$HOME/www/gzippy-bench/z119-results"
mkdir -p "$OUT"
PIGZ=/Users/jackdanger/www/gzippy/pigz/pigz
GZIP=/usr/bin/gzip

echo "=== rivals on the three priced cells (gzip -5, pigz -5 T1) ==="
for m in access.log data.sqlite minjs.min.js; do
  g=$($GZIP -c -5 "$CORPUS/$m" | wc -c)
  p=$($PIGZ -5 -p 1 -c "$CORPUS/$m" 2>/dev/null | wc -c)
  echo "$m	gzip5=$g	pigz5=$p" >> "$OUT/rivals.tsv"
done
cat "$OUT/rivals.tsv"

echo "=== len3 fp: access.log L5 T1 ==="
TOOL_GZ="$OUT/access.log.gzip5.gz"; $GZIP -c -5 "$CORPUS/access.log" > "$TOOL_GZ"
for arm in base A B; do
  case "$arm" in base) BIN=$BASE; env="";; A) BIN=$TUNE; env="GZIPPY_LDX_MIN3=1";; B) BIN=$TUNE; env="GZIPPY_LDX_MIN3=1 GZIPPY_LDX_LEN_GRADE=1";; esac
  env $env $BIN -c -p 1 -5 "$CORPUS/access.log" > "$OUT/access.log.L5.$arm.gz" 2>/dev/null
done
for arm in gzip base A B; do
  case "$arm" in gzip) f="$TOOL_GZ";; *) f="$OUT/access.log.L5.$arm.gz";; esac
  echo "== $arm ==" >> "$OUT/fp-access-log.tsv"
  cargo run --release --quiet --manifest-path /Users/jackdanger/www/gzippy/Cargo.toml --example fingerprint_tool -- fp "$f" 2>/dev/null | grep -E "^  (len3|len4_7|len8_15|file_bytes|blocks)" >> "$OUT/fp-access-log.tsv"
done
cat "$OUT/fp-access-log.tsv"

echo "=== wall min-of-5: access.log L5 T1 (interleaved base/A/B; base=default binary, arms=ladder-tune binary) ==="
python3 - "$BASE" "$TUNE" "$CORPUS/access.log" "$OUT" <<'EOF'
import subprocess, time, sys, os
base, tune, corpus, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
arms = [('base', base, {}), ('A', tune, {'GZIPPY_LDX_MIN3': '1'}), ('B', tune, {'GZIPPY_LDX_MIN3': '1', 'GZIPPY_LDX_LEN_GRADE': '1'})]
times = {k: [] for k, _, _ in arms}
for i in range(5):
    for name, binpath, env_extras in arms:
        env = dict(os.environ); env.update(env_extras)
        env.update({k: v for k, v in os.environ.items() if k.startswith('GZIPPY_') and k not in env_extras})
        t0 = time.perf_counter()
        subprocess.run([binpath, '-c', '-p', '1', '-5', corpus], cwd=None, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, check=True)
        times[name].append(time.perf_counter() - t0)
lines = [f"{k}\tmin={min(v):.4f}\truns={' '.join(f'{t:.4f}' for t in v)}" for k, v in times.items()]
open(os.path.join(out, 'wall.tsv'), 'w').write('\n'.join(lines) + '\n')
print(open(os.path.join(out, 'wall.tsv')).read())
EOF
echo "adjudication leg done"
