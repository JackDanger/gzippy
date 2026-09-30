# gzippy

A parallel gzip compressor and decompressor in Rust. One binary answers to
`gzip`, `gunzip`, `gzcat`, and `zcat`, reads and writes standard RFC 1952
streams, and uses every core you have.

## What it does

**Compression.** A single pure-Rust DEFLATE engine serves every level and
thread count, and every input is encoded exactly once — no trial encodes,
no keep-the-smaller-of-two. There is no C compressor on the production
routing graph; the vendored C encoders remain in-tree only as differential
test oracles.

Levels 0–9 follow libdeflate's level ladder, and most of them are a
decision-for-decision transliteration of
`vendor/libdeflate/lib/deflate_compress.c` (`src/compress/ldx/`).
Levels 1 and 6 stay on
gzippy's own arm; that is measured, not accidental: our level 1 beats pigz -1
on text where the transliteration does not, and level 6 is the one place the
transliteration costs more wall than it saves. Levels 10–12 switch to a
near-optimal parse — close to
[Zopfli](https://github.com/google/zopfli)'s ratio at a fraction of its
runtime. Every level and thread count emits a standard single-member gzip
stream that any tool reads; no gzippy-specific output format is produced
outside test oracles.

**Decompression.** One pure-Rust engine handles every gzip stream in the
wild: plain single-member files, pigz-style multi-member concatenations
(`cat a.gz b.gz`), tabix/HTS [BGZF](https://samtools.github.io/hts-specs/SAMv1.pdf)
blocks, and gzippy's own "GZ" multi-block format. Parallel single-member
files are read by a parallel marker pipeline — a structural port of
[rapidgzip](https://github.com/mxmlnkn/rapidgzip) — that finds block
boundaries in the compressed bytes and hands chunks to workers, so one big
file's inflation spreads across every core. Every path verifies the trailing
CRC32 and ISIZE before it claims success. There is no C in the decode graph
at all; Deflate64 (ZIP method 9) is a separate library entry point.

### Engine highlights

- **One-encoder routing.** Which engine serves which level is one small
  function (`level_uses_ldx`); `tests/one_encode_only.rs` counts encoder
  entries so an encode-per-input regression is caught, not asserted.
- **Parse budget 48.** The near-optimal parser's binary-tree descent stops
  after 48 candidate probes per descent. Measured on a frozen cloud census:
  1.9% faster at the `-9 -p4` route and 1,251 B smaller than the uncapped
  shape it replaced, and the headline parallel wall comparison against pigz
  moved from 1.084× to 1.055×.
- **The `good_match` knob pair.** zlib's lazy-match tuning (chain depth,
  nice length, good match) is carried explicitly by the port's hash-chain
  matchfinder, which is what lets it match the old arms byte-for-byte at
  levels 6–7 and retire them.
- **Length-3 matches at level 3.** Level 3 runs the lazy parser with the
  far-Length-3 gate: a long-distance 3-byte match is taken only when it
  beats the three literals it replaces (`src/compress/ldx/far_len3.rs`).
- **NEON / SSE match extension.** The kernel that extends a candidate match
  compares 16 bytes at a time (NEON with exact-mismatch locate on
  aarch64, SSE on x86) instead of byte-walking.
- **Software-pipelined chain walk.** The hash-chain walk loads the next
  chain node while still testing the current candidate, one node prefetched
  ahead, in the `ldx` matchfinder.
- **Pure-Rust parallel decode.** The parallel single-member pipeline
  (`ParallelSM`) is the sole single-member decode path on every
  architecture — the decode engine builds without any C FFI.

## Performance

We re-census sizes on frozen hardware and bank the receipts rather than
quote numbers that rot. The banked board (AMD Zen2 cloud box, the canonical
21-file corpus — 19 Silesia members and two other streams — levels 1–9 × 1
and 4 threads, matched levels against gzip, pigz, and libdeflate):

| axis | result |
|---|---|
| size, all 1,006 comparable cells | 12 fail (1.2%) — every one single-threaded, worst margin +1.07% |
| size at 4 threads | zero failures |
| byte-equal ties | 105 cells are byte-identical to a rival's own bytes |
| wall, `silesia.tar -9 -p4` vs pigz | 1.055× (was 1.08× before the parse-budget change) |

A cell here is one (file × level × thread count × rival) comparison; full
definitions and per-cell receipts are in
[docs/board/sprint-2026-09-25.md](docs/board/sprint-2026-09-25.md).

For a feel rather than a census: on an M4 laptop, 211 MB of logs compress in
0.07 s across 14 threads (0.53 s single-threaded, versus GNU gzip's 0.58 s).
Decompression runs at 300–2000 MB/s depending on input and thread count.

Run your own files through it. `man gzippy-tuning` covers the knobs, and
[docs/board/](docs/board/) holds the campaign records behind everything
claimed above — the frozen-box receipts, the per-cell definitions, and the
residual loss list.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/JackDanger/gzippy/main/scripts/install.sh | bash
```

<details>
<summary>Per-platform</summary>

**macOS (Homebrew)**

```bash
brew tap jackdanger/gzippy https://github.com/JackDanger/gzippy
brew install jackdanger/gzippy/gzippy
```

**Debian / Ubuntu**

```bash
curl -fsSL https://jackdanger.github.io/gzippy/gzippy-signing-key.asc \
    | gpg --dearmor | sudo tee /etc/apt/keyrings/gzippy.gpg >/dev/null
echo "deb [signed-by=/etc/apt/keyrings/gzippy.gpg] https://jackdanger.github.io/gzippy stable main" \
    | sudo tee /etc/apt/sources.list.d/gzippy.list >/dev/null
sudo apt-get update && sudo apt-get install gzippy
```

**Arch (AUR)**: `gzippy-bin`

**From source**

```bash
git clone --recursive https://github.com/JackDanger/gzippy
cd gzippy && cargo build --release
```

</details>

## Usage

Same flags as gzip, plus parallelism and a few extras. `gunzip file.gz`,
`zcat file.gz`, and `gzippy -9 file` all work; the extra names are one binary
checking `argv[0]`. Existing scripts and pipelines run unchanged.

```bash
gzippy -9 big.tar                      # keep-it-simple, every core
gzippy -dc /var/log/prod-2026-04-22.gz | grep -i error
find /var/log -name '*.log' -mtime +30 -print0 | xargs -0 gzippy -k -6
tar c big-dir | gzippy --ultra > big-dir.tgz   # near-zopfli ratio, sane speed
gzippy -p4 -R backup.sql               # rsync-friendly, four threads
pg_dump mydb | gzippy -6 | aws s3 cp - s3://bucket/dump.gz
gzippy --max precious.tar              # densest level (12)
GZIPPY_DEBUG=1 gzippy -d mystery.gz    # show which decode route ran
```

For output, zlib (`-z`, `.zz`) and single-entry ZIP (`-K`) containers are
also supported. Manual: `man gzippy`, wire format: `man gzippy-format`,
tuning: `man gzippy-tuning`.

## Analyze

`gzippy --analyze FILE` prints a compression fingerprint of any file:
entropy, LZ77 coverage, a color canvas of the bytes, match histograms,
and a verdict.

```
$ gzippy --analyze Cargo.lock
  entropy    [██████▄   ]   5.22/8   MEDIUM
  LZ77 cover [█████████ ]   89.7%    EXTREME
  matches    4.42K  avg length 8.8 B  avg back-distance 8.1 KB
  est. gzip  [█▅        ]  ~16% of raw
```

## Library

```toml
gzippy = "0.8"
```

```rust
let compressed = gzippy::compress(&data, 6)?;
let restored = gzippy::decompress(&compressed)?;

// Explicit worker count (parallel; buffers the whole input):
let out = gzippy::compress_with_threads(&data, 6, 4)?;

// ⚠ Buffers the entire input — like every compression entry point in this
// crate today (the whole-buffer encoder is the shipped design; a true
// streaming path is open work). Peak memory is `input + output`.
let n1 = gzippy::compress_to_writer(reader, writer, 6)?;

// Deflate64 (ZIP method 9), the same engine as the gz/zip container paths:
let unzipped = gzippy::decompress_deflate64(&member)?;
```

`decompress_to_writer` streams the other direction.
`gzippy::decompress` accepts gzip data only and checks the magic up front;
full API: `cargo doc --open`.

## Credits

Built on ideas and code from [pigz](https://zlib.net/pigz/) (Mark Adler),
[libdeflate](https://github.com/ebiggers/libdeflate) (Eric Biggers),
[Zopfli](https://github.com/google/zopfli) (Google),
[zlib-ng](https://github.com/zlib-ng/zlib-ng),
[rapidgzip](https://github.com/mxmlnkn/rapidgzip) (Maximilian Knespel),
[ISA-L](https://github.com/intel/isa-l) (Intel), and
[ECT](https://github.com/fhanau/Efficient-Compression-Tool) (Felix Hanau).
Portions of this codebase are ports of those projects; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the per-project
licenses, copyright notices, and the list of derived files.

[zlib license](LICENSE) for gzippy's own code, with third-party-derived
files under their upstream licenses (MIT, Apache-2.0, BSD-3-Clause, zlib —
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
By [Jack Danger](https://github.com/jackdanger).
