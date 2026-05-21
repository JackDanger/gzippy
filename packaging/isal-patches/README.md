# ISA-L patches for gzippy parallel single-member decode

These patches are applied on `vendor/isa-l` branch `gzippy-stopping-points`
(JackDanger/isa-l). They are derived from rapidgzip's stopping-point extension
to ISA-L inflate.

| Patch | Files |
|-------|-------|
| `igzip_lib.h-stopping-points.patch` | `include/igzip_lib.h` — stopping-point API |
| `igzip_inflate.c-stopping-points.patch` | `igzip/igzip_inflate.c` — implementation + exported Huffman helpers |

Diff base for the `.patch` files: crates.io `isal-sys` 0.5.3+496255c bundled tree.
The same hunks apply cleanly onto current `vendor/isa-l` (apple-arm64-fix base).

Rust bindings (`crates/isal-sys-patched/src/igzip_lib.rs`) must match the patched
header; `inflate_state` size is 87384 bytes (verified via C `sizeof`).
