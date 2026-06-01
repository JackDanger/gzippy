# Where is ISA-L? — the definitive codec map

> **Status:** standing reference. Read-only audit, every claim cited `file:line`
> against the tree at branch `docs/codec-map` (forked from `origin/main`
> `07e628f`). If the code drifts from this doc, the code wins — re-derive and
> update here.
>
> This document exists so no agent ever has to re-answer "are we supposed to be
> using pure-Rust or ISA-L?" from scratch again.

---

## TL;DR for a confused agent (read this, then stop being confused)

1. **gzippy is NOT pure-Rust today.** Decode uses C/FFI codecs (ISA-L, libdeflate,
   zlib-ng) on every shipped artifact. The pure-Rust DEFLATE engine is the *goal*,
   not the current default.
2. **What ships:** x86_64-linux ships `--features isal-compression` (C ISA-L on the
   single-member + parallel-SM path). macOS (both arches) and arm64-linux ship the
   **default** build = **no features** → libdeflate one-shot for single-member; the
   whole parallel-SM pipeline is **compiled out**. `pure-rust-inflate` ships in **no**
   artifact. (`.github/workflows/release.yml:116-132`)
3. **The pure-Rust inner decoder only exists when you build it:** it is reachable
   only under `--features pure-rust-inflate` (CI tests it at `ci.yml:111`; no release
   target builds it). Default `cargo build --release` does **not** include it.
4. **For perf measurement, the production decode engine = build with
   `--no-default-features --features pure-rust-inflate`** (CLAUDE.md Rule 6;
   `Makefile` `bench-sm-pure-rust`). That is the C-FFI-free engine the campaign is
   optimizing. Force it on at all thread counts with `GZIPPY_FORCE_PARALLEL_SM=1`.
5. **The combined-decoder answer:** in the parallel-SM worker, the *windowed* decode
   calls ISA-L C **only** under `isal-compression && !pure-rust-inflate && x86_64`;
   otherwise it calls the pure-Rust unified `Inflate` decoder. The *window-absent
   bootstrap* (`deflate_block`) is **always pure-Rust** regardless of feature. ISA-L
   writes its output **into gzippy's `chunk.data` buffer** (not its own).

---

## 1. Feature → cfg truth table

The two compile-time predicates that gate the parallel pipeline are emitted by
`build.rs::emit_parallel_sm_cfgs` (`build.rs:61-86`):

```
pure_inflate_decode = (x86_64 || aarch64) && feature(pure-rust-inflate)        // build.rs:76
parallel_sm         = (x86_64 && (feature(isal-compression)||feature(pure-rust-inflate)))
                      || (aarch64 && feature(pure-rust-inflate))               // build.rs:77-78
```

Re-exported as `sm_cfg::PARALLEL_SM = cfg!(parallel_sm)` (`sm_cfg.rs:10`) and the
ISA-L-C predicate `sm_cfg::USE_ISAL_INFLATE = isal-compression && !pure-rust-inflate
&& x86_64` (`sm_cfg.rs:14-18`).

Feature names (`Cargo.toml:40-86`): `default = []` (line 41); `isal-compression`
pulls in `isal-rs` + arena allocator (line 43); `pure-rust-inflate` pulls in arena
allocator (line 60); `isal` is a marker feature set by build.rs when the static lib
builds (line 42).

| arch | feature-set | `parallel_sm` | `pure_inflate_decode` | `USE_ISAL_INFLATE` (`isal-compression && !pure-rust-inflate && x86_64`) | `isal_decompress::is_available()` (`isal-compression && x86_64`) |
|------|-------------|:---:|:---:|:---:|:---:|
| **x86_64** | `default` (no features) | ✗ | ✗ | ✗ | ✗ |
| x86_64 | `isal-compression` | ✓ | ✗ | ✓ | ✓ |
| x86_64 | `pure-rust-inflate` | ✓ | ✓ | ✗ | ✗ |
| x86_64 | `isal-compression,pure-rust-inflate` | ✓ | ✓ | ✗ (pure wins) | ✓ |
| **aarch64** | `default` (no features) | ✗ | ✗ | ✗ | ✗ |
| aarch64 | `isal-compression` | ✗ (¹) | ✗ | ✗ (²) | ✗ (²) |
| aarch64 | `pure-rust-inflate` | ✓ | ✓ | ✗ | ✗ |
| aarch64 | `isal-compression,pure-rust-inflate` | ✓ | ✓ | ✗ | ✗ |

(¹) `parallel_sm` on aarch64 requires `pure-rust-inflate`; `isal-compression` alone
does not enable it (`build.rs:77-78`).
(²) ISA-L's C library is x86-only; every ISA-L call site is gated on
`target_arch = "x86_64"` so it is unreachable on aarch64 even with the feature on
(`isal_decompress.rs:14`, `isal_decompress.rs:20`; the wrapper at
`inflate_wrapper.rs:157-161`).

`is_available()` source: decode = `cfg!(all(feature="isal-compression",
target_arch="x86_64"))` (`isal_decompress.rs:13-15`); compress = `cfg!(feature =
"isal-compression")` but every body is also arch-gated (`isal_compress.rs:10-16`).

---

## 2. Decode routing tree — codec named per leaf

Entry: `decompress::mod::decompress_gzip_libdeflate` (`decompress/mod.rs:238`)
classifies via `classify_gzip` (`decompress/mod.rs:159-197`) into a `DecodePath`
(`decompress/mod.rs:69-84`), then dispatches (`decompress/mod.rs:258-285` and the
single-member sub-dispatcher `decompress_single_member_for` at
`decompress/mod.rs:374-443`).

Classification order (`classify_gzip`, `decompress/mod.rs:160-196`):

```
has_bgzf_markers ("GZ" subfield)            → GzippyParallel        (mod.rs:160-162)
is_likely_multi_member && T>1               → MultiMemberPar        (mod.rs:163-168)
is_likely_multi_member && T==1              → MultiMemberSeq        (mod.rs:163-168)
PARALLEL_SM && T>=4(*) && len>10MiB && compressible
                                            → IsalParallelSM        (mod.rs:170-177)
   ...same gate, but stored-dominated & first_block_is_stored
                                            → StoredParallel        (mod.rs:186-188)
isal_decompress::is_available()             → IsalSingle            (mod.rs:190-192)
len > 1 GiB                                 → StreamingSingle       (mod.rs:193-195)
otherwise                                   → LibdeflateSingle      (mod.rs:196)
```

(*) `parallel_sm_min_threads()` = 4 in production, 0 when `GZIPPY_FORCE_PARALLEL_SM`
is set (`decompress/mod.rs:109-119`). The `>10MiB` constant is
`MIN_PARALLEL_COMPRESSED` (`decompress/mod.rs:92`). "compressible" = NOT
`parallel_sm_unprofitable` (ISIZE/len ratio ≥ 1.15) (`decompress/mod.rs:133-152`).

| `DecodePath` | actual codec called | source of the call | active when |
|---|---|---|---|
| `GzippyParallel` | **libdeflate FFI** (per-block) | `bgzf::decompress_bgzf_parallel` (`mod.rs:259-263`) | all arches/features |
| `MultiMemberPar` | **libdeflate FFI** (parallel members); on error falls back to sequential libdeflate | `bgzf::decompress_multi_member_parallel` → `decompress_multi_member_sequential` (`mod.rs:264-275`) | all |
| `MultiMemberSeq` | **libdeflate FFI** (`gzip_decompress_ex`, member-by-member) | `decompress_multi_member_sequential` (`mod.rs:276`, body `mod.rs:547-612`) | all |
| `IsalParallelSM` | **ISA-L C** *or* **pure-Rust unified Inflate** for windowed worker decode; **pure-Rust bootstrap always** — see §3 | `parallel::single_member::decompress_parallel` (`mod.rs:386-393`) | only when `PARALLEL_SM` (x86 isal/pure, arm64 pure) |
| `StoredParallel` | **no inner codec** — copies explicit stored-block `LEN` runs in parallel; mis-fire → one-shot (ISA-L if avail else libdeflate) | `stored_split::decompress_stored_parallel` (`mod.rs:395-415`); one-shot at `mod.rs:450-467` | only when `PARALLEL_SM` |
| `IsalSingle` | **ISA-L C** raw stateful `isal_inflate`, one-shot stream | `isal_decompress::decompress_gzip_stream` (`mod.rs:416-425`; body `isal_decompress.rs:20-42`) | only `isal-compression && x86_64` |
| `StreamingSingle` | **zlib-ng** via `flate2::read::GzDecoder`, 1 MiB buffer | `decompress_single_member_streaming` (`mod.rs:426-431`; body `mod.rs:471-489`) | any arch, input > 1 GiB, no ISA-L |
| `LibdeflateSingle` | **libdeflate FFI** `gzip_decompress_ex` one-shot (grows buffer; chains trailing members) | `decompress_single_member_libdeflate` (`mod.rs:432-436`; body `mod.rs:494-542`) | default fallback (the only single-member path on macOS / arm64-linux default build) |

Other entry points: `decompress_gzip_to_vec` (`mod.rs:289`) mirrors the same
routing; `decompress_zlib_turbo` (`mod.rs:615-634`) → `bgzf::inflate_into_pub`;
`decompress_raw_bytes` (`mod.rs:641-669`) → libdeflate then zlib-ng for >1 GiB.

**No silent backend fallback** inside the single-member dispatcher: each arm is
terminal (`mod.rs:373-443`). The only "fallbacks" are explicit router corrections
(MultiMemberPar→Seq on scan failure; StoredParallel→one-shot on
`NotStoredDominated`).

---

## 3. The parallel-SM inner decode (the combined-decoder question)

The `IsalParallelSM` leaf runs `parallel::single_member::decompress_parallel`. Inside
each worker, a chunk is decoded in up to two phases. The decoder selection is **purely
by cfg**, resolved at compile time. There are three `IsalInflateWrapper`
definitions in `inflate_wrapper.rs`, mutually exclusive:

| cfg gate | wrapper backing | `inflate_wrapper.rs` |
|---|---|---|
| `all(isal-compression, not(pure-rust-inflate), x86_64)` | patched **ISA-L C** `inflate_state` + 128 KiB inline staging buffer | struct `:157-188`; impl/`read_stream` calls `isal_inflate` (via `gzip_chunk.rs:601`) |
| `pure_inflate_decode` (= `(x86_64\|aarch64) && pure-rust-inflate`) | **pure-Rust** `inflate::unified::Inflate<Clean, Generic, Streaming>` | struct `:654-662`; `read_stream` → `self.inner.read_stream` `:789-800` |
| `not(parallel_sm)` | unusable stub (`UnsupportedPlatform`) | `:829-846` |

So the windowed worker decode:
- **ISA-L build** (`isal-compression`, no `pure-rust-inflate`, x86_64): ISA-L C.
- **pure-rust build** (`pure-rust-inflate`, any of x86_64/aarch64): pure-Rust
  `Inflate` (resumable unified decoder).
- If BOTH features are on, `cfg(pure_inflate_decode)` wins the wrapper (the
  `isal-compression && not(pure-rust-inflate)` gate excludes it). `USE_ISAL_INFLATE`
  is correspondingly false (`sm_cfg.rs:14-18`).

### Window-absent bootstrap — ALWAYS pure-Rust

The speculative prefetch of a chunk with no predecessor window runs
`decode_chunk_marker_bootstrap_then_isal` (`gzip_chunk.rs:836`), whose phase-1 is
`bootstrap_with_deflate_block` (`gzip_chunk.rs:873-880`, `:1218`). That uses
`deflate_block::Block` (`gzip_chunk.rs:1330`), a literal pure-Rust port of
rapidgzip's `deflate::Block`. `deflate_block.rs` is gated **only** on
`#![cfg(parallel_sm)]` (`deflate_block.rs:1`) — there is **no** isal vs pure-rust
distinction. The bootstrap is pure-Rust on every build that has the pipeline,
including the ISA-L build. (This matches MEMORY `project_sm_bootstrap_overshoot`:
the slow pure-Rust window-absent bootstrap runs even when ISA-L is the windowed
backend.) After the bootstrap arms a clean window, the windowed decoder (ISA-L *or*
pure per the table above) takes over for the bulk.

### Where ISA-L writes — into gzippy's `chunk.data`, not its own buffer

In the per-chunk decode loop the wrapper is handed a mutable slice **of
`chunk.data`** (`gzip_chunk.rs:365-389`):

- pure-rust A3 path: `read_stream_starting_at(output_slice, ...)` where
  `output_slice` is built from `chunk.data.as_mut_ptr()` (`gzip_chunk.rs:366-370`).
- non-pure-rust (ISA-L) path: `read_stream(spare)` where `spare` is
  `chunk.data.as_mut_ptr().add(prev_data_len + n_bytes_read)`
  (`gzip_chunk.rs:380-389`).

The 128 KiB `buffer` inside the ISA-L `IsalInflateWrapper` (`inflate_wrapper.rs:187`)
is the **input** refill staging (`next_in`/`avail_in`, vendor `m_buffer`), **not** the
output — ISA-L's `next_out` points at gzippy's `chunk.data`. So ISA-L decodes
directly into gzippy's chunk buffer; there is no separate ISA-L output buffer to copy
back.

---

## 4. Compression routing — codec named per cell

Entry: `compress::mod::compress_with_pipeline` (`compress/mod.rs:30`). Verified
against source:

| condition | actual codec | source |
|---|---|---|
| L11 / `-F` / `-I` / `-J` (`args.use_zopfli()`) | **zopfli** (pure-Rust `ZopfliGzEncoder`) | `compress/mod.rs:38-53` |
| T1, L0–L3, not huffman/rle, ISA-L available | **ISA-L C** streaming `compress_gzip_stream_direct` | `compress/mod.rs:56-71` |
| T1, L1–L5, not huffman/rle | **libdeflate FFI** one-shot if probe ratio ≥ 0.10, else **zlib-ng** (`flate2`) streaming | probe `compress/mod.rs:74-91`; libdeflate via `compress::parallel::compress_single_member` `:93-108`; flate2 branch `:110-133` |
| T1, L6–L9 (or huffman/rle) | **zlib-ng** (`flate2`) streaming | `compress/mod.rs:135-160` |
| T>1 (all levels) | `SimpleOptimizer::compress` → dispatches `ParallelGzEncoder` (L0–5, "GZ" multi-block) or `PipelinedGzEncoder` (L6–9, single-member) | `compress/mod.rs:163-171` |

Notes:
- The T>1 split between `ParallelGzEncoder` and `PipelinedGzEncoder` lives inside
  `SimpleOptimizer`, not in `compress_with_pipeline` (`compress/mod.rs:170-171`); the
  L0–5 vs L6–9 boundary is per the CLAUDE.md table and is decided there.
- ISA-L availability for compression = `cfg!(feature="isal-compression")` with
  arch-gated bodies (`isal_compress.rs:10-16`), so the L0–L3 ISA-L cell only fires on
  the x86_64-linux release artifact (the only one built with the feature).
- Library API `compress_raw_bytes`/`compress_bytes` (`compress/mod.rs:183-246`)
  mirror the same backends (ISA-L L0–3 then libdeflate).

---

## 5. What ships vs the goal — the reconciliation (and the real gap)

### The GOAL (CLAUDE.md)

> "pure-Rust DEFLATE engine as the SOLE production decode path and C-FFI removed
> from the decode graph (kept behind a dev/oracle feature as fuzz oracles)" … "done
> when … the pure-Rust decoder is the sole decode path with C-FFI off the decode
> graph."

### What `cargo build --release` (DEFAULT, no features) actually produces

- `default = []` (`Cargo.toml:41`) ⇒ `isal-compression` OFF, `pure-rust-inflate`
  OFF ⇒ `parallel_sm` FALSE, `pure_inflate_decode` FALSE, `is_available()` FALSE
  (`build.rs:76-78`, `isal_decompress.rs:13-15`).
- **x86_64 default:** parallel-SM compiled out; ISA-L unavailable; single-member →
  `LibdeflateSingle` (**libdeflate FFI**). Multi-member/bgzf → libdeflate FFI.
- **aarch64 default:** identical — **libdeflate one-shot** for single-member.
- Pure-Rust inner decoder is **absent** from the binary (its wrapper is the
  `not(parallel_sm)` stub).

### What the SHIPPED artifacts build with (`release.yml:116-132`)

| release target | features flag | resulting decode for single-member | parallel-SM? | pure-Rust inner decode? |
|---|---|---|---|---|
| `x86_64-unknown-linux-gnu` | `--features isal-compression` | ISA-L (one-shot or parallel-SM ≥10 MiB,T≥4) | **yes (ISA-L C)** | no (ISA-L wrapper) |
| `aarch64-apple-darwin` | `""` (default) | **libdeflate one-shot** | no (compiled out) | no |
| `x86_64-apple-darwin` | `""` (default) | **libdeflate one-shot** | no (compiled out) | no |
| `aarch64-unknown-linux-gnu` | `""` (default) | **libdeflate one-shot** | no (compiled out) | no |

CI builds confirm the same default split: x86 uses `--features isal-compression`,
non-x86 uses bare `cargo build --release` (`ci.yml:83-93`, and every release/clippy
job at `ci.yml:169,262,401,506,646,804,958`). The only place `pure-rust-inflate` is
exercised is a **test** run: `cargo test --release --no-default-features --features
pure-rust-inflate` (`ci.yml:109-111`). No release target compiles it.

### Is pure-rust-inflate reachable in a shipped artifact?

**No.** It ships in zero release targets. It is reachable only via an explicit
`--features pure-rust-inflate` build — CI tests, local `make bench-sm-pure-rust`
(`Makefile:333`), benches gated `required-features = ["pure-rust-inflate"]`
(`Cargo.toml:164-179`), and the `coz_bench` example (`Cargo.toml:153-155`).

### ⚠️ CONTRADICTION TO NAME (code vs goal)

The shipped reality **contradicts** the CLAUDE.md goal in two ways, and matches
MEMORY `project_pure_rust_inflate_not_shipped_2026_05_29`:

1. **C-FFI is on the decode graph of every shipped artifact.** x86-linux ships C
   ISA-L; macOS/arm64-linux ship libdeflate FFI. The goal's "C-FFI removed from the
   decode graph / kept behind a dev/oracle feature" is **not met** in any release
   build today.
2. **The pure-Rust engine is the goal's "sole decode path" but ships nowhere.** It
   is test/feature-gated only. The arm64 parallel-SM enablement
   (commits `155a2eb`/`1a526b2`/`089e1d3`, MEMORY `project_arm64_fastest`) makes the
   pure-Rust path *buildable and fastest* on arm64 **under `--features
   pure-rust-inflate`**, but the arm64 release targets still build with **no
   features** (`release.yml:125-128`), so that win is not in a shipped binary yet.
   Flipping arm64 release to `--features pure-rust-inflate` is the unshipped lever.

This is a real, intentional gap (the campaign is mid-flight), not a bug to "fix"
silently — but any claim that "gzippy is pure-Rust" or "C-FFI is off the decode
graph" is **false** for shipped artifacts as of `07e628f`.

---

## 6. The measurement implication — which build is "production" for perf

**For perf work the production decode engine is the pure-Rust parallel-SM pipeline,
built with `--no-default-features --features pure-rust-inflate`.** This is the
C-FFI-free engine the campaign optimizes toward "sole decode path," and it is what
CLAUDE.md Rule 6 mandates:

> "build `--no-default-features --features pure-rust-inflate`;
> `GZIPPY_FORCE_PARALLEL_SM=1` to exercise the engine at every T" — assert the path
> with `GZIPPY_DEBUG=1` → `path=IsalParallelSM`.

Mechanics:
- Build: `cargo build --release --no-default-features --features pure-rust-inflate`
  (the harness in `Makefile:333` `bench-sm-pure-rust` does exactly this and A/Bs it
  vs the `isal-compression` build).
- Force the engine on at every thread count: `GZIPPY_FORCE_PARALLEL_SM=1` drops the
  T≥4 floor to 0 (`decompress/mod.rs:109-119`) so T1–T3 also exercise the engine
  instead of the libdeflate one-shot confound.
- Assert routing: `GZIPPY_DEBUG=1` must print `path=IsalParallelSM`
  (`decompress/mod.rs:249-256`).
- Measure interleaved + sha-verified via `scripts/measure.sh` (CLAUDE.md Rule 6),
  never an internal slice.

The `--features isal-compression` build is the **competitive reference / what x86
ships**, useful as the ISA-L baseline in an A/B, but it is **not** the engine being
optimized toward the goal. Do not report a pure-Rust perf number from a default or
ISA-L build.

---

## Citation index (quick lookup)

- Feature defs: `Cargo.toml:40-86`
- cfg emission: `build.rs:61-86`
- cfg re-export + USE_ISAL_INFLATE: `src/decompress/parallel/sm_cfg.rs:10,14-18`
- Decode routing classifier: `src/decompress/mod.rs:159-197`
- Decode dispatch: `src/decompress/mod.rs:258-285`, `:374-443`
- Single-member backends: `:419` (ISA-L), `:435/494` (libdeflate), `:430/471` (zlib-ng)
- Multi-member libdeflate: `:547-612`
- Wrapper variants: `src/decompress/parallel/inflate_wrapper.rs:157-188` (ISA-L),
  `:654-826` (pure-Rust), `:829-846` (stub)
- ISA-L `isal_inflate` call: `src/decompress/parallel/gzip_chunk.rs:601`
- ISA-L writes into chunk.data: `src/decompress/parallel/gzip_chunk.rs:365-389`
- Pure-Rust bootstrap: `src/decompress/parallel/gzip_chunk.rs:836,873-880,1218,1330`;
  `src/decompress/parallel/deflate_block.rs:1` (parallel_sm-only gate)
- ISA-L availability: `src/backends/isal_decompress.rs:13-15`;
  `src/backends/isal_compress.rs:10-16`
- Compression routing: `src/compress/mod.rs:30-171`
- Release feature matrix: `.github/workflows/release.yml:116-132,190`
- CI feature split: `.github/workflows/ci.yml:83-93,109-111`
- target-cpu=native for all local builds: `.cargo/config.toml`
</content>
</invoke>
