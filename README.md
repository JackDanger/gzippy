# gzippy

The fastest gzip. Parallel, written in Rust.

```bash
gzippy file.txt           # Compress → file.txt.gz
gzippy -d file.txt.gz     # Decompress → file.txt
cat data | gzippy > out   # Works with pipes too
```

## Install

**macOS / Linux — Homebrew**

```bash
brew tap jackdanger/gzippy https://github.com/JackDanger/gzippy
brew install jackdanger/gzippy/gzippy
```

Installs `gzippy`, `gzip`, `gunzip`, `ungzippy`, and `zcat` — all the same binary.

> If Homebrew's `gzip` package is installed, remove it first: `brew uninstall gzip`  
> The macOS system `/usr/bin/gzip` is untouched.

**Cargo (all platforms)**

```bash
cargo install gzippy
```

**Build from source**

```bash
git clone --recursive https://github.com/JackDanger/gzippy
cd gzippy && cargo build --release
```

## How fast?

Compressing 211 MB of logs on an M4 MacBook Pro (14 cores):

| Tool | Speed | Time |
|------|-------|------|
| gzippy (14 threads) | ~3000 MB/s | 0.07s |
| gzippy (1 thread) | ~400 MB/s | 0.53s |
| GNU gzip | ~360 MB/s | 0.58s |
| Apple gzip (NEON) | ~315 MB/s | 0.67s |

Decompression: 300–2000 MB/s depending on file type and thread count.

## Options

Works like gzip: `-1` to `-9`, `-c` (stdout), `-d` (decompress), `-k` (keep original), `-f` (force), `-v` (verbose).

Extra options:
- `-p4` — use 4 threads (default: all cores)
- `--level 11` / `--ultra` — smaller output, slower
- `--level 12` / `--max` — smallest output

## Requirements

- 64-bit Linux or macOS
- Rust 1.70+ (for `cargo install` or build from source only)

## Standing on shoulders

- [**pigz**](https://zlib.net/pigz/) by Mark Adler — how to parallelize gzip
- [**libdeflate**](https://github.com/ebiggers/libdeflate) by Eric Biggers — fast deflate
- [**zlib-ng**](https://github.com/zlib-ng/zlib-ng) — zlib with SIMD
- [**rapidgzip**](https://github.com/mxmlnkn/rapidgzip) — parallel decompression
- [**ISA-L**](https://github.com/intel/isa-l) by Intel — SIMD assembly

## License

[zlib license](LICENSE) — same as zlib and pigz.

## About

Made by [Jack Danger](https://github.com/jackdanger).
