# ISA-L patches for gzippy parallel single-member decode

## Layout (three repos, one C source)

```
gzippy/
├── vendor/isa-l          → JackDanger/isa-l @ gzippy-stopping-points   (C patches)
├── vendor/isal-rs/       → JackDanger/isal-rs @ gzippy-stopping-points (Rust bindings)
│   └── isal-sys/         patched crate; [patch.crates-io] target
├── .cargo/config.toml    ISAL_SOURCE=vendor/isa-l
└── packaging/isal-patches/   human-readable diffs (reference only)
```

**Single source of truth for C:** `vendor/isa-l`. Also used by CI igzip builds and
benchmark scripts. `isal-sys` copies from `ISAL_SOURCE` at compile time — it does not
embed its own isa-l tree.

**Clone:**
```bash
git submodule update --init vendor/isa-l vendor/isal-rs
```

**Before pushing gzippy** after bumping either submodule pointer, push the fork
branch first (`vendor/isa-l` → JackDanger/isa-l, `vendor/isal-rs` → JackDanger/isal-rs).
`scripts/pre-push` runs `scripts/verify_submodule_remotes.sh` to catch missing commits.

Crates.io releases strip `[patch.crates-io]`; release binaries must be built from a
full git checkout with both submodules initialized.

## C patches (JackDanger/isa-l)

| Patch | Files |
|-------|-------|
| `igzip_lib.h-stopping-points.patch` | `include/igzip_lib.h` — stopping-point API |
| `igzip_inflate.c-stopping-points.patch` | `igzip/igzip_inflate.c` — implementation + exported Huffman helpers |

Diff base: intel/isa-l commit `496255c` (bundled in crates.io `isal-sys` 0.5.3+496255c).
Regenerate from `vendor/isa-l`:

```bash
cd vendor/isa-l
git diff 496255c..HEAD -- include/igzip_lib.h \
  > ../packaging/isal-patches/igzip_lib.h-stopping-points.patch
git diff 496255c..HEAD -- igzip/igzip_inflate.c \
  > ../packaging/isal-patches/igzip_inflate.c-stopping-points.patch
```

## Rust deltas (JackDanger/isal-rs / isal-sys)

See `vendor/isal-rs/isal-sys/README.md`. Wired via root `Cargo.toml`:

```toml
[patch.crates-io]
isal-sys = { path = "vendor/isal-rs/isal-sys" }
```
