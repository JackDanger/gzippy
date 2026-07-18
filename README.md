# gzippy

A parallel gzip replacement in Rust. One binary answers to `gzip`, `gunzip`,
`gzcat`, and `zcat`, produces standard RFC 1952 output, and uses every core
you have.

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

Same flags as gzip. `gunzip file.gz`, `zcat file.gz`, and `gzippy -9 file`
all work; the extra names are one binary checking `argv[0]`. Existing scripts
and pipelines run unchanged. Manual: `man gzippy`.

## Performance

Decompression matches or beats rapidgzip, the fastest parallel gzip
decompressor we know of, across a 12-corpus by thread-count matrix on AMD,
Intel, and Apple Silicon, with byte-exact output. Compression runs on all
cores and is well ahead of pigz and system gzip on multicore machines.

No benchmark table here because numbers rot. Run your own files through it;
`man gzippy-tuning` covers the knobs.

## Analyze

`gzippy --analyze FILE` prints a compression fingerprint of any file:
entropy, LZ77 coverage, a color canvas of the bytes, match histograms,
and a verdict. No other gzip tool does this.

```
$ gzippy --analyze Cargo.lock
  entropy    [██████▄   ]   5.22/8   MEDIUM
  LZ77 cover [████████▄ ]   85.4%    EXTREME
  matches    2.05K  avg length 9.4 B  avg back-distance 4.8 KB
  est. gzip  [█▅        ]  ~16% of raw
```

## Library

```toml
gzippy = "0.8"
```

```rust
let compressed = gzippy::compress(&data, 6)?;
let restored = gzippy::decompress(&compressed)?;

// Explicit threads, or streaming without buffering the whole input:
let out = gzippy::compress_with_threads(&data, 6, 4)?;
let n = gzippy::compress_to_writer(reader, writer, 6)?;
```

`decompress_to_writer` streams the other direction. Full API: `cargo doc --open`.

## One caveat

With `threads > 1` at levels 0-5, gzippy emits its own "GZ" multi-block
format for maximum parallel speed. Only gzippy reads it. Use one thread or
levels 6-9 when the output must be plain gzip. Details in `man gzippy-format`.

## Credits

Built on ideas and code from [pigz](https://zlib.net/pigz/) (Mark Adler),
[libdeflate](https://github.com/ebiggers/libdeflate) (Eric Biggers),
[zlib-ng](https://github.com/zlib-ng/zlib-ng),
[rapidgzip](https://github.com/mxmlnkn/rapidgzip), and
[ISA-L](https://github.com/intel/isa-l).

[zlib license](LICENSE). By [Jack Danger](https://github.com/jackdanger).
