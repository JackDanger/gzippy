# isal-sys (gzippy fork)

Fork of [milesgranger/isal-rs](https://github.com/milesgranger/isal-rs) `isal-sys` with
stopping-point inflate bindings for gzippy parallel single-member decode.

## ISA-L C source

This crate does **not** bundle ISA-L. It resolves the C tree at build time:

| Context | How |
|---------|-----|
| **gzippy checkout** | `.cargo/config.toml` sets `ISAL_SOURCE=vendor/isa-l` |
| **Standalone clone** | `export ISAL_SOURCE=/path/to/JackDanger/isa-l` (branch `gzippy-stopping-points`) |

C patches live in [JackDanger/isa-l](https://github.com/JackDanger/isa-l) branch
`gzippy-stopping-points`. See gzippy `packaging/isal-patches/`.

## Rust deltas vs crates.io 0.5.3+496255c

- `build.rs` — `ISAL_SOURCE` / gzippy vendor path resolution
- `wrapper.h` — exported Huffman table builder declarations
- `src/lib.rs` — `isal_internals` module
- `src/igzip_lib.rs` — stopping-point fields on `inflate_state` (87384 bytes)
