# gzippy

A parallel gzip.
In Rust.

One binary.
Four names: `gzippy`, `gunzip`, `gzcat`, `zcat`.

Standard gzip streams. RFC 1952.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/JackDanger/gzippy/main/scripts/install.sh | bash
```

<details>
<summary>Other platforms</summary>

macOS:

```bash
brew tap jackdanger/gzippy https://github.com/JackDanger/gzippy
brew install jackdanger/gzippy/gzippy
```

Debian / Ubuntu:

```bash
curl -fsSL https://jackdanger.github.io/gzippy/gzippy-signing-key.asc \
    | gpg --dearmor | sudo tee /etc/apt/keyrings/gzippy.gpg >/dev/null
echo "deb [signed-by=/etc/apt/keyrings/gzippy.gpg] https://jackdanger.github.io/gzippy stable main" \
    | sudo tee /etc/apt/sources.list.d/gzippy.list >/dev/null
sudo apt-get update && sudo apt-get install gzippy
```

Arch: `gzippy-bin` (AUR).

From source:

```bash
git clone --recursive https://github.com/JackDanger/gzippy
cd gzippy
cargo build --release
```

</details>

## Compress

```bash
gzippy -9 big.tar              # level 9, every core
gzippy --ultra big.tar         # level 11, true zopfli parse
gzippy --max precious.tar      # level 12, densest
gzippy -p 4 big.tar            # four threads
gzippy -z big.bin              # zlib wrapper (RFC 1950), .zz suffix
```

Rejected loudly, not silently wrong (issue #302):
`-R` rsyncable, `-K` zip.

## Decompress

Standard streams only, in and out.

```bash
gunzip big.tar.gz
zcat file.gz | grep error
gzippy -dc prod-2026-04-22.gz | grep -i error
GZIPPY_DEBUG=1 gzippy -d mystery.gz    # print which decode route ran
```

Handles every gzip file we know of:

- plain files
- multi-member files (`cat a.gz b.gz`)
- BGZF blocks (tabix/HTS)
- our own multi-block GZ format

Parallel by default on the first:

- finds block boundaries in compressed bytes
- spreads them across every core
- port of rapidgzip's structure

CRC32 and ISIZE verified every time.

## The one-encode rule

One encode per input.

No trial runs. No keep-the-smaller-of-two.

Which engine runs at which level:

```text
level_uses_ldx: 0, 2-5, 7-9   -> the libdeflate port
level 1, 6                    -> our own arm (measured)
level 10-12                   -> near-optimal parse
```

`tests/one_encode_only.rs` counts encoder entries.

## How fast

Frozen-box censuses, not benchmarks that rot.

21-file corpus, 9 levels, 1 and 4 threads, matched levels vs gzip, pigz, libdeflate:

| axis | result |
|---|---|
| size, 1006 cells | 12 fail. All single-threaded. Worst +1.07% |
| at 4 threads | zero failures |
| byte-identical to a rival | 105 cells |
| silesia.tar -9 -p4 vs pigz | 1.055x wall |

Full receipts: [docs/board/sprint-2026-09-25.md](docs/board/sprint-2026-09-25.md).

A single laptop number for feel: 211 MB of logs on an M4, 0.07 s on
14 threads, 0.53 s single-threaded (GNU gzip: 0.58 s). Decodes at
300-2000 MB/s.

Run your own files.
`man gzippy-tuning` has the knobs.

## Analyze

```bash
gzippy --analyze FILE
```

Entropy. LZ77 coverage. Match histograms. A byte canvas. A verdict.

```
$ gzippy --analyze Cargo.lock
  entropy    [██████▄   ]   5.22/8   MEDIUM
  LZ77 cover [█████████ ]   89.7%    EXTREME
  matches    4.42K  avg length 8.8 B  avg back-distance 8.1 KB
  est. gzip  [█▅        ]  ~16% of raw
```

(One block, abridged — real output adds a banner, canvas, legends and
histograms.)

## As a library

```toml
gzippy = "1.0"
```

```rust
let compressed = gzippy::compress(&data, 6)?;
let restored = gzippy::decompress(&compressed)?;

// Explicit thread count (parallel; buffers the whole input):
let out = gzippy::compress_with_threads(&data, 6, 4)?;

// Streaming wrapper (also buffers the input today):
let n = gzippy::compress_to_writer(reader, writer, 6)?;

// Streams to a writer on the decode side:
let n = gzippy::decompress_to_writer(&compressed, &mut out)?;

// Deflate64 (ZIP method 9):
let out = gzippy::decompress_deflate64(&member)?;
```

`decompress_to_writer` streams the decode direction.
Full API: `cargo doc --open`.

## Docs

- `man gzippy`
- `man gzippy-format`
- `man gzippy-tuning`
- [docs/board/](docs/board/) — the measurement records behind this file

## Credits

Ports and ideas from the giants. Per-file licenses:

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- [pigz](https://zlib.net/pigz/) — Mark Adler
- [libdeflate](https://github.com/ebiggers/libdeflate) — Eric Biggers
- [Zopfli](https://github.com/google/zopfli)
- [zlib-ng](https://github.com/zlib-ng/zlib-ng)
- [rapidgzip](https://github.com/mxmlnkn/rapidgzip) — Maximilian Knespel
- [ISA-L](https://github.com/intel/isa-l) — Intel
- [ECT](https://github.com/fhanau/Efficient-Compression-Tool) — Felix Hanau

gzippy: the zlib license, [LICENSE](LICENSE).

By [Jack Danger](https://github.com/jackdanger).
