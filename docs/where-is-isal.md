# Where is ISA-L? — which decoder each gzippy run actually uses

This doc makes it **structurally unambiguous** which inner DEFLATE decoder a
given gzippy run exercised: the pure-Rust engine (the production goal) or the
ISA-L C FFI (legacy/oracle comparator). Every performance number is
uninterpretable without this — a memory-model measurement taken on the wrong
build can *invert* the sign of the effect, which is exactly the fiasco this doc
+ the FULCRUM provenance witness exist to prevent.

## The feature → decoder mapping (confirmed against source)

The `parallel_sm` chunk pipeline (block finder, prefetcher, consumer,
window-map, marker resolution) compiles under **either** feature. Only the
**inner windowed decode** differs:

| Build | Inner windowed decode | `isal_inflate` dynsyms | Role |
|-------|----------------------|------------------------|------|
| `--no-default-features --features pure-rust-inflate` | **PURE RUST** — `IsalInflateWrapper` wraps `Inflate<Clean,Generic,Streaming>` / `ResumableInflate2`; no real ISA-L FFI in the decode graph | **0** | **canonical production / measurement path** (CLAUDE.md GOAL + Measurement Rule 6) |
| `--features isal-compression` | **REAL ISA-L C FFI** — `IsalInflateWrapper` → `isal_raw::inflate_state` | **>0** | legacy / oracle comparator, NOT the optimization target |

Notes:
- The window-absent marker **bootstrap** (`deflate_block.rs`) is pure-Rust in
  **both** builds (it emits the u16 markers).
- Enabling **both** features ⇒ pure-Rust wins the `not(pure-rust-inflate)`
  guard, so the binary is pure-Rust.
- x86 *shipping* artifacts have historically still linked ISA-L; pure-rust is
  the **directed goal + campaign baseline** measured here.

## The witness: `isal_inflate` symbol count in the binary

The load-bearing, build-flag-independent fact is the count of `isal_inflate`
C-FFI symbols **in the actual binary that ran**:

- **0** ⇒ no ISA-L inflate FFI linked ⇒ inner decode is **PURE RUST**.
- **>0** ⇒ ISA-L inflate FFI present ⇒ inner decode is (or may be) **ISA-L C**.

Read it directly:

```bash
nm target/release/gzippy | awk '{print $NF}' | sed 's/^_*//' \
  | grep -E '^isal_inflate' | grep -vE '^ZN|^R' | wc -l
# 0 = pure-Rust ; >0 = ISA-L
```

The `grep -vE '^ZN|^R'` rejects Rust-mangled names that merely *mention*
`isal_inflate` (e.g. a helper called `count_isal_inflate_symbols`); only
unmangled C entry points (`isal_inflate`, `isal_inflate_init`,
`isal_inflate_stateless`, …) count.

## Structural enforcement: the FULCRUM provenance witness

Every FULCRUM bundle/report now self-labels the decoder so no run is
interpretable without it (`fulcrum/src/provenance.rs`):

```bash
fulcrum provenance target/release/gzippy \
  --features "pure-rust-inflate" \
  --routing "$(GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 \
               ./target/release/gzippy -d -c -p8 corpus.gz 2>&1 >/dev/null | grep path=)" \
  --rev "$(git describe --always --dirty)"
```

It prints a one-glance header (decoder + symbol count + features + routing +
rev), folds the witness into the bundle `meta`, and **fails closed** when the
declared features contradict the binary's symbol witness (e.g. a
`pure-rust-inflate` build that still links `isal_inflate`) or when the symbol
table can't be read. The bench harness (`gzippy-bench/c1c2_guest_bench.sh`)
runs this first and aborts if `isal_inflate` dynsyms != 0, so a pure-rust
campaign run can never be silently measured on the ISA-L build.

## Routing witness

`GZIPPY_DEBUG=1` prints the path taken. For the parallel single-member engine
(the measurement target) you want:

```
[gzippy] decompress_single_member path=IsalParallelSM threads=8 ...
```

`GZIPPY_FORCE_PARALLEL_SM=1` drops the thread floor to 0 so the engine runs at
every thread count (measurement aid; regresses T1–3, not the production
default). The path name says `IsalParallelSM` for **both** decoders — the name
is historical; the `isal_inflate` dynsym witness is what tells the two apart.
