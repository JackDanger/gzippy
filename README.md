# gzippy

The fastest gzip on any hardware. Drop-in for `gzip`, `gunzip`, `gzcat`,
`zcat`, and `ungzippy` — same RFC 1952 output, every decompressor on Earth
still reads your files.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/JackDanger/gzippy/main/scripts/install.sh | bash
```

One line. Detects macOS, Debian, Ubuntu, and most Linux, then uses the
right package manager.

<details>
<summary>Per-platform commands</summary>

**macOS — Homebrew**

```bash
brew tap jackdanger/gzippy https://github.com/JackDanger/gzippy
brew install jackdanger/gzippy/gzippy
```

**Debian / Ubuntu — apt**

```bash
curl -fsSL https://jackdanger.github.io/gzippy/gzippy-signing-key.asc \
    | gpg --dearmor \
    | sudo tee /etc/apt/keyrings/gzippy.gpg >/dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/gzippy.gpg] \
    https://jackdanger.github.io/gzippy stable main" \
    | sudo tee /etc/apt/sources.list.d/gzippy.list >/dev/null
sudo apt-get update
sudo apt-get install gzippy
```

Replace system gzip (via dpkg-divert):

```bash
sudo apt-get install gzippy-replace-gzip
```

**Build from source**

```bash
git clone --recursive https://github.com/JackDanger/gzippy
cd gzippy && cargo build --release
```

</details>

## How fast?

Measured on the [Silesia compression corpus](http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia)
(202 MB of mixed text, source code, images, and database dumps), Apple
M4 Pro with 14 cores, macOS 15. Level 6. Median of 15 runs per tool.

### Compression

| Tool        |    Throughput  |   Time  |
|-------------|---------------:|--------:|
| **gzippy**  | **~700 MB/s**  | **0.29 s** |
| pigz        |    ~150 MB/s   |  1.30 s |
| Apple gzip  |     ~40 MB/s   |  5.00 s |

### Decompression

| Tool        | Throughput |
|-------------|-----------:|
| **gzippy**  | **~930 MB/s** |
| Apple gzip  |    ~900 MB/s |
| pigz        |    ~790 MB/s |

Reproduce with [`scripts/readme_benchmark.py`](scripts/readme_benchmark.py)
after `cargo build --release` and `(cd pigz && make)`.

## One binary, many names

```
gzip    gunzip    gzcat    zcat    ungzippy    gzippy
```

All six commands are the same Rust binary. `gunzip file.gz` and
`gzippy -d file.gz` take identical code paths at identical speed.
Installers put gzippy ahead of the system gzip in `$PATH`, so the
takeover is silent — and `/usr/bin/gzip` stays untouched.

## Beyond gzip

```
$ gzippy --analyze Cargo.lock
gzippy --analyze  Cargo.lock  (22.1 KB)
──────────────────────────────────────────────────────────────────────────────
  entropy    [██████▄   ]   5.22/8   MEDIUM      (mixed text and binary, or a data file)
  LZ77 cover [████████▄ ]   85.4%    EXTREME     (most bytes come for free — this file is very squishy)
  matches    2.05K  avg length 9.4 B  avg back-distance 4.8 KB
  est. gzip  [█▅        ]  ~16% of raw

  match-length histogram — how long is each reused sequence?
     3 -   4 B  ████████████████         73.0%  literal repeats; common everywhere
     9 -  16 B  █▇                        8.3%  words, variable names, small keys
    17 -  32 B  █▆                        7.7%  lines of code, struct layouts

  (canvas, colour legend, distance histogram, verdict — see `man gzippy`)
```

`gzippy --analyze FILE` prints a compression fingerprint: entropy, LZ77
coverage, an 80×20 colour canvas of the bytes, match-length and
distance histograms, and a one-line verdict. No other gzip does this.

The full manual lives in `man gzippy`. The `"GZ"` parallel-block wire
format and the tuning guide have their own pages: `man gzippy-format`,
`man gzippy-tuning`.

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
