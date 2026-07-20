#!/bin/bash
# Build 3 distinct-character corpora from silesia tar. Run on solvency.
set -e
TAR="${1:-/root/silesia-ref.tar}"
D=/root/frontier-corpora
rm -rf "$D"
mkdir -p "$D/ex"
cd "$D/ex"
tar xf "$TAR"
cd "$D"
# sil = balanced mix: source-tar (samba) + medical-index (nci) + xml
cat ex/silesia/samba ex/silesia/nci ex/silesia/xml > sil
# text = text-dominant: dickens + webster + reymont
cat ex/silesia/dickens ex/silesia/webster ex/silesia/reymont > text
# bin = low-compressibility mixed binary: x-ray + sao + mr + osdb
cat ex/silesia/x-ray ex/silesia/sao ex/silesia/mr ex/silesia/osdb > bin
rm -rf ex
echo "=== sizes ==="
ls -l sil text bin
echo "=== sha256 ==="
sha256sum sil text bin
