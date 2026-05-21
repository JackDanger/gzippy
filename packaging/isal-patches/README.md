# ISA-L patches for gzippy parallel single-member decode

C patches live on `vendor/isa-l` branch `gzippy-stopping-points` (JackDanger/isa-l).
Rust bindings live on `vendor/isal-rs` branch `gzippy-stopping-points` (JackDanger/isal-rs),
crate `isal-sys`, patched via `[patch.crates-io]` in the root `Cargo.toml`.

| Patch | Files |
|-------|-------|
| `igzip_lib.h-stopping-points.patch` | `include/igzip_lib.h` — stopping-point API |
| `igzip_inflate.c-stopping-points.patch` | `igzip/igzip_inflate.c` — implementation + exported Huffman helpers |

Diff base for the `.patch` files: crates.io `isal-sys` 0.5.3+496255c bundled tree.
The same hunks apply cleanly onto current `vendor/isa-l` (apple-arm64-fix base).

Rust deltas (in `vendor/isal-rs/isal-sys/`): `build.rs`, `wrapper.h`, `src/lib.rs`
(`isal_internals`), `src/igzip_lib.rs` (stopping-point fields; `inflate_state` size
87384 bytes, verified via C `sizeof`).
