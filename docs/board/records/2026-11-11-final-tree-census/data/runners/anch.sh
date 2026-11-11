#!/bin/bash
# ANCHORS — final-tree census section 1.
# Five anchor streams on in16m.bin (16,085,866 B, sha 1010fa2f787d0a92...),
# arm A (main b1502a07) and arm B (06859145) each; every stream is checked
# against the recorded expectation (size + sha256 from the wave-B byte-census
# record) and roundtripped through THREE independent decoders:
#   /usr/bin/gzip (Apple gzip 479), vendor/pigz/pigz 2.8, python3 zlib
#   member-chain (decode_sha.py).
# NOTE: /opt/homebrew/bin/gzip is a gzippy argv0 alias (verified: prints
# gzippy 0.1.7) — DEMOTED, never used as a decoder on this box.
set -euo pipefail
R=/Users/jackdanger/www/gzippy
RUN=$R/.review-tmp/wC-final/run
A=$R/.review-tmp/wC-final/tgt-a/release/gzippy
B=$R/.review-tmp/wC-final/tgt-b/release/gzippy
IN=$R/.review-tmp/in16m.bin
PYDEC=$RUN/decode_sha.py
unset GZIP PIGZ || true

cat > "$RUN/anchors.expected.tsv" <<'EOF'
-9	2	3816930	edc698cc16c8b91f50e59b17fd06a66b7e141bdee455be83017b3d9213b99e5a
-9	4	3740910	327b083588efca2178d97f977cbdca37e9a3815b7440aea5ccbe099e566c5475
-8	1	3821170	2a0fb084316ae7f6e5dd8034c729672a3ab520a57fe344a48f6e996e9c35025d
-8	4	3819910	b7090a22573814e2520aec366da9d89b5c7ec713f8034c026373cc016020299d
-6	2	3848412	7997e3ace401da72542d80f41fdf4e6f0ea19e7ca9cf1fc0098ec8cdd7834190
-6	4	3848412	7997e3ace401da72542d80f41fdf4e6f0ea19e7ca9cf1fc0098ec8cdd7834190
EOF

: > "$RUN/anchors.tsv"
in_sha=$(shasum -a 256 "$IN" | awk '{print $1}')
[ "$in_sha" = 1010fa2f787d0a928fd10823085c62a9ee080fcff96b2608c5d216b2b8a1463c ] || { echo "FATAL input sha changed" >&2; exit 1; }

while IFS=$'\t' read -r lev p exp_size exp_sha; do
    tag="L${lev#-}p$p"
    for arm in A B; do
        bin=$([ "$arm" = A ] && echo "$A" || echo "$B")
        stream="$RUN/tmp/anchor_${arm}_${tag}.gz"
        "$bin" "$lev" -p"$p" -c < "$IN" > "$stream"
        sha=$(shasum -a 256 "$stream" | awk '{print $1}')
        size=$(stat -f%z "$stream" | tr -d ' ')
        # three decoder legs; each leg's decompressed bytes must sha to the input
        l1=$(/usr/bin/gzip -dc < "$stream" | shasum -a 256 | awk '{print $1}')
        l2=$("$R/vendor/pigz/pigz" -dc < "$stream" | shasum -a 256 | awk '{print $1}')
        if python3 "$PYDEC" "$stream" "$in_sha" > "$RUN/tmp/pydec.log" 2>&1; then l3=OK; else l3=FAIL; cat "$RUN/tmp/pydec.log" >&2; fi
        v=PASS
        [ "$sha" = "$exp_sha" ] || v=FAIL
        [ "$size" = "$exp_size" ] || v=FAIL
        [ "$l1" = "$in_sha" ] || v=FAIL
        [ "$l2" = "$in_sha" ] || v=FAIL
        [ "$l3" = "OK" ] || v=FAIL
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$arm" "$tag" "$lev -p$p" "$size" "$sha" "$l1" "$l2" "$l3" "$v" >> "$RUN/anchors.tsv"
        rm -f "$stream"
    done
done < "$RUN/anchors.expected.tsv"

column -t -s$'\t' "$RUN/anchors.tsv"
echo "FAIL rows: $(awk -F'\t' '$9=="FAIL"' "$RUN/anchors.tsv" | wc -l | tr -d ' ')"
