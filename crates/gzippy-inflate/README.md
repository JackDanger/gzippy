# gzippy-inflate

Pure-Rust DEFLATE / gzip inflate primitive. The decode-side counterpart
of the [gzippy](https://github.com/jackdanger/gzippy) parent crate,
intended to be separately publishable to crates.io.

## Status

**v0.1.0 — API scaffold.** The public surface is stable; the
implementation delegates to `libdeflater` until v0.2.0 migrates the
pure-Rust inflate path in from the parent crate.

```rust
use gzippy_inflate::Inflate;

let decoded = Inflate::decode_gzip(&compressed)?;
```

See `plans/unified-decoder.md` §3.11 for the roadmap.

## Features

- `std` (default) — std-only surface; pure-`no_std` builds use the
  `alloc` crate via `extern crate alloc`.
- `async` — enables `gzippy_inflate::async::AsyncInflate` for
  tokio-compatible async input.

## License

Zlib (matches parent crate).
