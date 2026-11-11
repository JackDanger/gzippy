#!/bin/bash
# COUNTERS — the settle/pass counter deltas for the final-tree flag census
# ("record if cheap"). Instrument builds (anatomy-counters rides; stdout bytes
# are NOT compared here — output to /dev/null; instrument builds are ~1.17x
# slower and never quoted for wall).
#   CNTR   = anatomy-counters only                     (serial reference)
#   SKIPC  = anatomy-counters,nearopt-fixedpoint-skip  (pass_entries delta pair)
#   FLUSHC = anatomy-counters,flush,flipsafe           (settle witness arm)
# Expectations (the recorded reconciliation, #465/#467 records):
#   SKIPC.nearopt_pass_entries - CNTR.nearopt_pass_entries
#        == CNTR.nearopt_finite_skip_would_save   (mid-loop fixedpoint skips)
#   FLUSHC.near_opt_only_literals_blocks == CNTR.near_opt_only_literals_blocks
#        (the SAFE decision space IS the serial space — exactness corroboration)
#        and FLUSHC.near_opt_flip_safe_settles > 0 on flip-landing cells.
set -euo pipefail
R=/Users/jackdanger/www/gzippy
RUN=$R/.review-tmp/wC-final/run
TGT=$R/.review-tmp/wC-final
CNTR=$TGT/tgt-cntr/release/gzippy
SKIPC=$TGT/tgt-skipc/release/gzippy
FLUSHC=$TGT/tgt-flushc/release/gzippy
BENCH=/Users/jackdanger/www/gzippy-bench/corpus
OUT=$RUN/counters
mkdir -p "$OUT"
unset GZIP PIGZ || true
for b in "$CNTR" "$SKIPC" "$FLUSHC"; do [ -x "$b" ] || { echo "missing $b" >&2; exit 1; }; done

CELLS=$RUN/counters_cells.tsv
: > "$CELLS"

cell() { # arm-tag bin member level p
    local tag=$1 bin=$2 name=$3 lev=$4 p=$5
    local path
    case "$name" in
        in16m) path=$R/.review-tmp/in16m.bin ;;
        logs.part-aa) path=$R/.review-tmp/wC-bytes/run/corpus/logs.part-aa ;;
        weights.safetensors|sil40) path=$BENCH/$name ;;
        *) path=$R/.review-tmp/wC-bytes/run/corpus/silesia/${name#silesia-} ;;
    esac
    [ -e "$path" ] || { echo "missing member $name" >&2; exit 1; }
    printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$bin" "$name" "$lev" "$p" >> "$CELLS"
    "$bin" --level "$lev" -p "$p" -c < "$path" > /dev/null 2> "$OUT/$tag.stderr"
}

# --- CNTR serial reference cells (one run each; feeds BOTH reconciliations) ---
cell c_in16m_L9p4    "$CNTR" in16m               9 4
cell c_in16m_L11p4   "$CNTR" in16m              11 4
cell c_in16m_L12p4   "$CNTR" in16m              12 4
cell c_logsaa_L12p4  "$CNTR" logs.part-aa       12 4
cell c_sil40_L11p4   "$CNTR" sil40              11 4
cell c_weights_L9p4  "$CNTR" weights.safetensors 9 4
cell c_weights_L12p4 "$CNTR" weights.safetensors 12 4
cell c_mozilla_L9p4  "$CNTR" silesia-mozilla     9 4

# --- SKIPC arm (nearopt-fixedpoint-skip ON): pass_entries deltas at L11/L12 ---
cell s_in16m_L11p4   "$SKIPC" in16m              11 4
cell s_in16m_L12p4   "$SKIPC" in16m              12 4
cell s_logsaa_L12p4  "$SKIPC" logs.part-aa       12 4
cell s_sil40_L11p4   "$SKIPC" sil40              11 4
cell s_weights_L12p4 "$SKIPC" weights.safetensors 12 4

# --- FLUSHC arm (flush+flipsafe ON): settle witness + only-lits equality ---
cell f_in16m_L9p4    "$FLUSHC" in16m              9 4
cell f_in16m_L11p4   "$FLUSHC" in16m             11 4
cell f_sil40_L11p4   "$FLUSHC" sil40             11 4
cell f_weights_L9p4  "$FLUSHC" weights.safetensors 9 4
cell f_mozilla_L9p4  "$FLUSHC" silesia-mozilla    9 4

python3 "$RUN/parse_count.py" "$RUN/counters_cells.tsv" "$OUT" |
    tee "$RUN/counters.tsv"
