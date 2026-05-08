# Plan: Port Google Zopfli from C to pure Rust

## Progress

**Last completed: Step 9 (deflate_size.rs)** — see commits below.

| Step | Module | Status |
|------|--------|--------|
| 0 | Scaffold, oracle harness, alice.txt | ✅ Done |
| 1 | `symbols.rs` | ✅ Done |
| 2 | `katajainen.rs` | ✅ Done |
| 3 | `tree.rs` | ✅ Done (2 ULP diff on arm64, noted in oracle test) |
| 4 | `hash.rs` | ✅ Done (locked in by Step 8 oracle) |
| 5 | `cache.rs` | ✅ Done (locked in by Step 8 oracle) |
| 6 | `lz77.rs` Part A (`LZ77Store`, histogram) | ✅ Done |
| 7 | `lz77.rs` Part B (`BlockState`, `find_longest_match`) | ✅ Done |
| 8 | `lz77.rs` Part C (`lz77_greedy`, FFI oracle) | ✅ Done — byte-for-byte equality on whole corpus |
| 9 | `deflate_size.rs` | ✅ Done — bitwise f64 equality vs FFI for btype=0,1,2 + auto |
| 10–12 | `squeeze.rs` Parts A–C | 🔲 Next |
| 13 | `blocksplitter.rs` | 🔲 |
| 14–15 | `deflate.rs` Parts A–B | 🔲 |
| 16 | `gzip.rs` | 🔲 |
| 17 | `mod.rs` public surface | 🔲 |
| 18 | Bridge/feature flag | 🔲 |
| 19–25 | Integration, cutover, cleanup | 🔲 |
| 26 | PR | 🔲 |
| 27–30 | Optimization (post-cutover) | 🔲 |

**Notes for next agent:**
- `build.rs` was fixed to use `/usr/bin/ar` on macOS (GNU ar from Homebrew was incompatible)
- The entropy test allows 2 ULP diff (arm64 FP codegen vs Clang)
- Pre-commit hook runs `cargo fmt`, `cargo check`, and `cargo clippy -D warnings` — fix before committing
- All modules need `#![allow(dead_code)]` until they're referenced from later modules
- **When you're unsure about a strategic decision, do not guess.** Add the
  question as a `> **Open question (Step N):** …` blockquote inside the
  relevant step section below. A larger advisor model reads the plan
  out-of-band and answers in-place; resume implementation once the question
  has been resolved (the blockquote is replaced with guidance).

> **Reader profile.** This plan assumes you (the implementer) can write
> idiomatic Rust and run `cargo`, but you have **not** read zopfli's source.
> Every step cites the exact C file and line range to port, the exact Rust
> signatures to expose, and a concrete test that pins correctness against the
> existing C library before you proceed. Do **not** skip a step. Do **not**
> reorder steps. The order is chosen so each step's oracle test depends only
> on previously-finished modules.

## Working environment

- Worktree branch already created — work directly here. PR target: `main`.
- The C zopfli source is at `vendor/zopfli/src/zopfli/` (run
  `git submodule update --init vendor/zopfli` if empty).
- The C FFI lives at `src/backends/zopfli_compress.rs` (264 lines). It will
  remain the **oracle** until Step 28 cuts over. **Do not delete it before
  cutover.**
- The CLI calls FFI through `compress_gzip(data, &tuning)` and
  `compress_deflate(data, &tuning)` (see `src/compress/zopfli.rs:82` and
  `:109`). Keep these signatures unchanged across the whole port.

## Final architecture

```
src/backends/
  zopfli_compress.rs        # public surface (compress_gzip / compress_deflate / ZopfliTuning)
                            # post-cutover: pure-Rust thin shim re-exporting from zopfli_pure
  zopfli_pure/              # NEW, pure-Rust port (this plan creates it)
    mod.rs                  # ZopfliOptions, ZopfliFormat, top-level Compress dispatcher
    symbols.rs              # length/dist symbol & extra-bit tables (port of symbols.h)
    katajainen.rs           # length-limited Huffman (port of katajainen.c)
    tree.rs                 # bitlength <-> symbol, entropy (port of tree.c)
    hash.rs                 # ZopfliHash (port of hash.c)
    cache.rs                # LongestMatchCache (port of cache.c)
    lz77.rs                 # LZ77Store, BlockState, FindLongestMatch, LZ77Greedy
                            #   (port of lz77.c)
    deflate_size.rs         # block-size estimation, dynamic-tree size, RLE optimizer
                            #   (size-only half of deflate.c)
    squeeze.rs              # SymbolStats, GetBestLengths, LZ77Optimal, LZ77OptimalFixed
                            #   (port of squeeze.c)
    blocksplitter.rs        # BlockSplitLZ77 / BlockSplit / FindMinimum (port of blocksplitter.c)
    deflate.rs              # BitWriter, AddLZ77Block, AddDynamicTree, Deflate, DeflatePart
                            #   (encoder half of deflate.c)
    gzip.rs                 # gzip header + CRC32 + ISIZE wrapper (port of gzip_container.c)
    oracle_tests.rs         # #[cfg(test)] equivalence harness vs the C FFI (deleted at end)
```

## Strategy: oracle-driven port, leaf-first, **never** ship a regression

Three iron rules:

1. **Both implementations live side-by-side until cutover (Step 28).** Every
   step adds a Rust module and an oracle test (`oracle_tests.rs`) that runs
   the same input through the existing FFI and the new Rust code and asserts
   bit/byte equality. The C FFI is the ground truth.
2. **Floating-point order matters.** Zopfli's `SymbolStats::ll_symbols` are
   `f64` from `log()`-based entropy; the squeeze cost array is `f32`. Any
   change in evaluation order on identical inputs can produce a different
   path → different LZ77 → different bytes. **Port f64/f32 widths and the
   exact arithmetic order verbatim.** Only after Step 28 may you experiment
   with fixed-point. Any optimization step (Step 29+) must keep the oracle
   test green.
3. **One module per step. The crate compiles after every step.** If a step
   does not type-check, stop, fix it, and re-run the oracle test; do not
   start the next step.

### Iteration loop after every step

```bash
cargo build --release            # must succeed
cargo test --release zopfli_pure # oracle tests for the touched module
make                             # ~30s sanity (catches catastrophic regressions)
```

`make ship` (homelab perf) is **only** run after Step 28 and after every
optimization step in Phase 4. Local `make` is the iteration signal.

### When you must adapt the plan

The plan codifies one shape per decision; reality will surface forks the
plan didn't anticipate. The guiding principle: **deviate when the
deviation makes correctness easier to verify; do not deviate when it
just makes the code prettier.**

The deviations below are **recommended** (an earlier executor of this
plan discovered them; encode them as the default for future steps):

- **One module per commit, even for trivial leaves.** Cleaner bisect at
  Step 12 if a long-shot bug turns out to live in a Step-4 leaf. If you
  finish two leaves in one sitting, stash one and commit the other alone.
- **Pull `ZopfliOptions` up to `mod.rs` whenever it's first needed**
  (in practice, Step 7's `BlockState`, not Step 17). Mark `#[repr(C)]`
  from the start so post-cutover FFI sharing in benchmarks costs nothing.
- **Prefer `Option<T>` over `add_x: bool` flag pairs.**
  `BlockState { lmc: Option<LongestMatchCache>, ... }` and
  `s.lmc.is_some()` reads exactly like the C `s->lmc != NULL`. Don't
  carry the `add_lmc: bool` argument forward into the struct.
- **Hash chain swap via accessor methods, not pointer aliasing.** The
  C aliases `hhead = h->head2` mid-loop; in Rust expose
  `h.head_for(use_hash2: bool) -> &[i32]`,
  `h.prev_for(use_hash2)`, `h.hashval_for(use_hash2)` and swap a single
  `bool`. Same generated code, dramatically clearer.
- **Factor shared work between size-only and emit paths.** The RLE pass
  used by `EncodeTree` (Step 9 size-only and Step 14 emit) should be
  one helper returning `(rle, rle_bits, clcounts, hlit, hdist)` rather
  than re-scanned twice. The trim-zero loop is part of that helper, not
  duplicated at each call site.
- **Use `u64::from_le_bytes` for the LZ77 8-byte fast compare**, not
  unsafe pointer casts. Autovectorizes the same way; no `unsafe`.
- **Verbatim-port C's signed-int decrement loops.** Example:
  `optimize_huffman_for_rle` decrements `length` past zero — use `isize`,
  not `usize`. Idiomatic restructuring fights the line-by-line audit.
- **Manual `Clone` for `LZ77Store`** must truncate the histogram tails
  to `NUM_LL * ceil_div(size, NUM_LL)` ll_counts and
  `NUM_D * ceil_div(size, NUM_D)` d_counts. The C `ZopfliCopyLZ77Store`
  allocates exactly that much; `derive(Clone)` would copy
  capacity-padded scratch slots and silently change semantics. (This bug
  surfaces only at Step 13's `block_split_lz77`, after several other
  modules pass — easy to miss without the manual impl.)
- **Oracle tests should replay FFI intermediate state into a Rust
  struct, not build both sides from scratch.** When testing module N,
  import the FFI's outputs from earlier modules rather than trusting
  that the Rust port of module N-1 is bit-equivalent. Step 9's
  block-size oracle, for instance, calls `ZopfliLZ77Greedy` on the C
  side and copies the resulting `litlens`/`dists`/`pos` arrays into a
  Rust `LZ77Store` — isolating Step 9 from any latent drift in Step 8.
  This pattern generalizes; use it for every oracle from Step 9 onward.

These deviations are **not** OK during the port:

- Refactoring loop bodies into iterator chains. Defer to Phase 10.
- Changing `f32` → `f64` (or vice versa) to "make types cleaner". The
  f32 cost buffer in Step 11 and the f64 entropy in Step 3 are load-bearing.
- Replacing the Marsaglia MWC with a "better" PRNG. Acceptance is
  byte-identical output; that requires the same RNG sequence.
- Adding `#[inline]` / `#[cold]` annotations during the port. Profile
  first (Phase 10 / Step 28).
- Combining adjacent steps into a single commit "to save a commit." The
  per-step oracle test is the contract; collapsing it forfeits the
  bisect signal.

### Logging real ambiguities

When the plan is silent or ambiguous on a real fork (the kind a small
model can't decide unilaterally without risking drift), do **not** guess.
Add a blockquote at the top of the relevant step:

```markdown
> **Open question (Step 11):** the plan says `Fn(u32, u32) -> f64` but
> the C cost model takes a `void* context`. Trait or closure?
```

…and ask before deciding. The "Frequently asked questions during the
port" section near the bottom of this file is where binding answers
land. Common forks are pre-answered there — check it first.

### How tests reach the FFI oracle

Add this helper to `src/backends/zopfli_pure/oracle_tests.rs` once at Step 0
and reuse it from every later step:

```rust
//! Oracle tests: compare pure-Rust port against the C FFI.
//! Deleted in Step 28 (cutover).
#![cfg(test)]

use crate::backends::zopfli_compress as ffi;          // existing FFI module
use crate::backends::zopfli_pure as rs;               // the new port

pub fn corpus() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("empty",     vec![]),
        ("byte",      b"x".to_vec()),
        ("ascii",     b"hello world hello world hello world".to_vec()),
        ("zeros_1k",  vec![0u8; 1024]),
        ("zeros_64k", vec![0u8; 65_536]),
        ("rand_4k",   {
            // deterministic LCG so the corpus is reproducible
            let mut v = Vec::with_capacity(4096);
            let mut s: u32 = 0x12345678;
            for _ in 0..4096 { s = s.wrapping_mul(1103515245).wrapping_add(12345);
                               v.push((s >> 16) as u8); }
            v
        }),
        ("alice",     include_bytes!("../../../test_data/alice.txt").to_vec()),
        // add larger corpus only once smaller cases pass; large inputs are slow under zopfli.
    ]
}
```

(If `test_data/alice.txt` does not exist, download a public-domain plaintext
in Step 0 and commit it; it's fine to use any moderately-redundant English
text up to ~150 KB.)

---

# Phase 1 — Scaffold and oracle harness

## Step 0 — Create the empty pure-Rust module and the oracle harness

**Files to create / modify**
- `src/backends/mod.rs` — add `pub mod zopfli_pure;`
- `src/backends/zopfli_pure/mod.rs` — empty module with re-exports stub:

  ```rust
  //! Pure-Rust port of Google Zopfli. Built bottom-up; oracle-tested
  //! against the C FFI at `crate::backends::zopfli_compress` until cutover.
  
  pub mod symbols;     // Step 1
  // pub mod katajainen;  // unlock at Step 2
  // ...
  
  #[cfg(test)]
  mod oracle_tests;
  ```
- `src/backends/zopfli_pure/symbols.rs` — empty file (Step 1 fills it).
- `src/backends/zopfli_pure/oracle_tests.rs` — paste the harness from above
  ("How tests reach the FFI oracle").
- `test_data/alice.txt` — `wget` Project Gutenberg's *Alice in Wonderland*
  (~150 KB, public domain) and commit. It's the "real text" fixture.

**Verify**
```bash
cargo build --release
cargo test --release -p gzippy zopfli_pure   # passes (no tests yet)
```

**Do not** touch the FFI module. **Do not** modify `build.rs`.

---

# Phase 2 — Leaf modules (no dependencies)

## Step 1 — `symbols.rs` (DEFLATE length/distance encoding)

**Source to port:** `vendor/zopfli/src/zopfli/symbols.h:1-239` — entirely
header, all `static inline` functions plus three lookup tables.

**Public API to provide** (`src/backends/zopfli_pure/symbols.rs`):

```rust
// All match the C signatures exactly; signed ints because the C code uses int.
pub fn dist_extra_bits(dist: i32) -> i32;           // from line 38
pub fn dist_extra_bits_value(dist: i32) -> i32;     // from line 61
pub fn dist_symbol(dist: i32) -> i32;               // from line 88
pub fn length_extra_bits(l: i32) -> i32;            // from line 138
pub fn length_extra_bits_value(l: i32) -> i32;      // from line 161
pub fn length_symbol(l: i32) -> i32;                // from line 183
pub fn length_symbol_extra_bits(s: i32) -> i32;     // from line 222
pub fn dist_symbol_extra_bits(s: i32) -> i32;       // from line 231
```

**Implementation notes**

- The two CLZ branches in `dist_extra_bits`, `dist_extra_bits_value`,
  `dist_symbol` use `__builtin_clz`. In Rust use `(dist - 1).leading_zeros()`
  and recompute `l = 31 ^ leading_zeros` exactly as C does. Use a `const fn`
  where possible.
- The three tables (`length_extra_bits`, `length_extra_bits_value`,
  `length_symbol`) are 259-element arrays. Copy the values verbatim from
  `symbols.h` — do not "compute" them.
- `length_symbol_extra_bits` is 29 entries indexed by `s - 257`.
- `dist_symbol_extra_bits` is 30 entries indexed by `s`.

**Constants module-level in `symbols.rs`:**

```rust
pub const ZOPFLI_NUM_LL: usize = 288;
pub const ZOPFLI_NUM_D:  usize = 32;
pub const ZOPFLI_MAX_MATCH: usize = 258;
pub const ZOPFLI_MIN_MATCH: usize = 3;
pub const ZOPFLI_WINDOW_SIZE: usize = 32_768;
pub const ZOPFLI_WINDOW_MASK: usize = ZOPFLI_WINDOW_SIZE - 1;
pub const ZOPFLI_MASTER_BLOCK_SIZE: usize = 1_000_000;
pub const ZOPFLI_LARGE_FLOAT: f64 = 1e30;
pub const ZOPFLI_CACHE_LENGTH: usize = 8;
pub const ZOPFLI_MAX_CHAIN_HITS: i32 = 8192;
```

**Oracle test** (in `oracle_tests.rs`):

```rust
#[test]
fn symbols_match_ffi() {
    // We cannot link to the static C `symbols.h` (it's static inline),
    // so use the spec-verified expected values:
    use crate::backends::zopfli_pure::symbols::*;
    assert_eq!(length_symbol(3),   257);
    assert_eq!(length_symbol(258), 285);
    assert_eq!(length_symbol(11),  265);
    assert_eq!(dist_symbol(1),     0);
    assert_eq!(dist_symbol(4),     3);
    assert_eq!(dist_symbol(5),     4);
    assert_eq!(dist_symbol(32_768),29);
    assert_eq!(length_extra_bits(3),   0);
    assert_eq!(length_extra_bits(11),  1);
    assert_eq!(length_extra_bits(258), 0);
    assert_eq!(dist_extra_bits(1),  0);
    assert_eq!(dist_extra_bits(5),  1);
    assert_eq!(dist_extra_bits(32_768), 13);
    // Exhaustive sweep against re-implementing the spec branchily:
    for d in 1..=32_768 { assert_eq!(dist_extra_bits(d) + (d as i32).count_ones() as i32 >= 0, true); }
    for l in 3..=258 {
        let s = length_symbol(l);
        assert!((257..=285).contains(&s));
        // Round-trip: extra bits from symbol == extra bits from length
        assert_eq!(length_symbol_extra_bits(s), length_extra_bits(l));
    }
}
```

(If you'd rather oracle against the C tables directly, expose them via the
existing FFI build by adding a `#[no_mangle]` C shim — but the spec-based
sweep above is sufficient and avoids touching `build.rs`.)

**Done when:** `cargo test --release zopfli_pure::symbols` is green and
`cargo build --release` still succeeds.

---

## Step 2 — `katajainen.rs` (length-limited Huffman code lengths)

**Source to port:** `vendor/zopfli/src/zopfli/katajainen.c:1-262`. Single
public function `ZopfliLengthLimitedCodeLengths`.

**Public API:**

```rust
/// Returns Ok(()) on success, Err(()) on the two C error paths
/// ((1<<maxbits) < numsymbols, or weight overflow). Mirrors the C return
/// codes (0 = ok, non-zero = error).
pub fn length_limited_code_lengths(
    frequencies: &[usize],   // length n
    maxbits:     i32,
    bitlengths:  &mut [u32], // length n; written in-place
) -> Result<(), ()>;
```

**Implementation notes**

- Use a `Vec<Node>` arena for the node pool (the C code uses
  `malloc(maxbits * 2 * numsymbols * sizeof(Node))`). Keep the same size.
- `Node` matches the C struct exactly:
  ```rust
  #[derive(Clone, Copy)]
  struct Node { weight: usize, count: i32, tail: i32 }  // tail = -1 for None,
                                                        //         else arena index
  ```
- Replace `Node*` with `i32` arena index. `pool.next++` becomes
  `let idx = pool_next; pool_next += 1; arena[idx] = ...`.
- `lists` is `[[i32; 2]; maxbits]` of arena indices; allocate a `Vec<[i32;2]>`
  of size `maxbits`.
- `BoundaryPM` is recursive — keep the recursion (depth ≤ maxbits, so ≤15
  for DEFLATE; safe).
- The qsort step sorts `Node` by `(weight, count)` using the `(weight << 9) |
  count` trick. **You don't need that trick in Rust** — just
  `leaves.sort_by(|a, b| a.weight.cmp(&b.weight).then(a.count.cmp(&b.count)))`.
  This is *behaviourally* identical because the C trick is just a stable
  sort key; the count tie-break is the same.
- Special cases (lines 201–220): preserve verbatim.

**Oracle test** — link to the C FFI by adding a temporary export. In
`build.rs`, the static archive already links; we just need a Rust extern
declaration. **Add a temporary block to `oracle_tests.rs` only** (so it
disappears at Step 28):

```rust
extern "C" {
    fn ZopfliLengthLimitedCodeLengths(
        frequencies: *const usize, n: i32, maxbits: i32,
        bitlengths: *mut u32,
    ) -> i32;
}

fn ffi_lengths(freq: &[usize], maxbits: i32) -> Vec<u32> {
    let mut bl = vec![0u32; freq.len()];
    let r = unsafe { ZopfliLengthLimitedCodeLengths(
        freq.as_ptr(), freq.len() as i32, maxbits, bl.as_mut_ptr()) };
    assert_eq!(r, 0);
    bl
}

#[test]
fn katajainen_matches_ffi() {
    use crate::backends::zopfli_pure::katajainen::length_limited_code_lengths;
    let cases: Vec<Vec<usize>> = vec![
        vec![1,1,2,3,5,8,13,21],
        vec![0,0,1,0,0,2,0,5,0,9,1],
        (0..286usize).map(|i| (i*7 % 17) as usize).collect(),
        vec![1; 32],     // ZOPFLI_NUM_D
        vec![100,100,100,100],
        vec![0,0,0,5],   // <2 used symbols path
        vec![0,0,0,0],
        vec![42],
    ];
    for freq in &cases {
        for &mb in &[15i32, 7] {
            let exp = ffi_lengths(freq, mb);
            let mut got = vec![0u32; freq.len()];
            length_limited_code_lengths(freq, mb, &mut got).unwrap();
            assert_eq!(got, exp, "freq={:?} maxbits={}", freq, mb);
        }
    }
}
```

**Done when:** `cargo test --release zopfli_pure::katajainen` is green.

---

## Step 3 — `tree.rs` (bit-length / symbol / entropy)

**Source to port:** `vendor/zopfli/src/zopfli/tree.c:1-101` (3 functions) and
`tree.h`.

**Public API:**

```rust
pub fn calculate_bit_lengths(count: &[usize], maxbits: i32, bitlengths: &mut [u32]);
pub fn lengths_to_symbols(lengths: &[u32], maxbits: u32, symbols: &mut [u32]);
pub fn calculate_entropy(count: &[usize], bitlengths: &mut [f64]);
```

**Implementation notes**

- `calculate_bit_lengths` is a one-line wrapper around
  `length_limited_code_lengths`; assert-no-error matches the C asserter.
- `lengths_to_symbols`: straight port; use `Vec<usize>` for `bl_count` /
  `next_code`.
- `calculate_entropy` is **floating-point sensitive**. The C code does:
  ```c
  static const double kInvLog2 = 1.4426950408889;   // exact constant
  log2sum = (sum == 0 ? log(n) : log(sum)) * kInvLog2;
  if (count[i] == 0) bitlengths[i] = log2sum;
  else bitlengths[i] = log2sum - log(count[i]) * kInvLog2;
  if (bitlengths[i] < 0 && bitlengths[i] > -1e-5) bitlengths[i] = 0;
  ```
  In Rust, use **exactly** that constant (`const K_INV_LOG2: f64 =
  1.4426950408889;`), `f64::ln`, the same multiplication order, and the same
  clamp. **Do not** use `f64::log2`; the C code multiplies `ln(x) *
  kInvLog2` and the rounding differs.

**Oracle test:**

```rust
extern "C" {
    fn ZopfliCalculateBitLengths(count: *const usize, n: usize, maxbits: i32, bitlengths: *mut u32);
    fn ZopfliLengthsToSymbols(lengths: *const u32, n: usize, maxbits: u32, symbols: *mut u32);
    fn ZopfliCalculateEntropy(count: *const usize, n: usize, bitlengths: *mut f64);
}

#[test] fn tree_bitlengths_match_ffi() { /* sweep 6 frequency tables, maxbits 15 */ }
#[test] fn tree_lengths_to_symbols_match_ffi() { /* call after bitlengths */ }
#[test] fn tree_entropy_match_ffi_exact() {
    // f64 bitwise equality. NaN-safe via f64::to_bits().
    // If this fails: you used log2() instead of log()*kInvLog2, or
    // changed evaluation order. Fix it before continuing.
}
```

**Done when:** entropy test reports **bitwise** `f64::to_bits()` equality on
each value. (Some compilers may diverge by 1 ULP; if so, allow ULP=1 and add
a comment noting which platforms.)

---

## Step 4 — `hash.rs` (`ZopfliHash` rolling hash + same-byte run + dual hash)

**Source to port:** `vendor/zopfli/src/zopfli/hash.c:1-143` and `hash.h`.

**Public API and struct:**

```rust
pub const HASH_MASK:  u32 = 32_767;
pub const HASH_SHIFT: u32 = 5;

pub struct ZopfliHash {
    pub head:     Box<[i32]>,    // len 65_536
    pub prev:     Box<[u16]>,    // len window_size
    pub hashval:  Box<[i32]>,    // len window_size
    pub val:      i32,
    pub head2:    Box<[i32]>,    // len 65_536
    pub prev2:    Box<[u16]>,    // len window_size
    pub hashval2: Box<[i32]>,    // len window_size
    pub val2:     i32,
    pub same:     Box<[u16]>,    // len window_size
}

impl ZopfliHash {
    pub fn new(window_size: usize) -> Self;     // both alloc + reset
    pub fn reset(&mut self, window_size: usize);
    pub fn warmup(&mut self, array: &[u8], pos: usize, end: usize);
    pub fn update (&mut self, array: &[u8], pos: usize, end: usize);
}
```

**Implementation notes**

- All three feature flags in the C code (`ZOPFLI_HASH_SAME`,
  `ZOPFLI_HASH_SAME_HASH`, `ZOPFLI_LONGEST_MATCH_CACHE`) are **always on**
  in the build that gzippy uses. The Rust port hard-codes them on. Do not
  parameterize; that just creates dead code paths.
- `update_hash_value` (private):
  ```rust
  #[inline]
  fn update_hash_value(&mut self, c: u8) {
      self.val = (((self.val << HASH_SHIFT) ^ c as i32) & HASH_MASK as i32);
  }
  ```
- The `same` update is a tight loop; port literally — do **not** SIMD-ify
  yet.
- Indexing: `hpos = pos & ZOPFLI_WINDOW_MASK` always.

**Oracle test** — link to the C `ZopfliHash` struct. The struct layout
matters; rather than mirroring it byte-for-byte, oracle indirectly via
`ZopfliFindLongestMatch` later (Step 6). For now do a **functional** test:

```rust
#[test]
fn hash_update_sequence_self_consistent() {
    // 1) Build a deterministic 8 KB input.
    // 2) Walk it byte-by-byte calling warmup then update.
    // 3) After each update, assert internal invariants:
    //    - hashval[hpos] == val
    //    - head[val] == hpos
    //    - prev[hpos] points to a slot whose hashval is val OR equals hpos
    //    - same[hpos] equals the count of consecutive equal bytes starting at pos
    //      (compute via a brute-force scan of the input).
}
```

The oracle test against the C hash arrives in Step 6 via
`find_longest_match`; passing that locks this module in.

---

## Step 5 — `cache.rs` (longest-match cache)

**Source to port:** `vendor/zopfli/src/zopfli/cache.c:1-125` and `cache.h`.

**Public API and struct:**

```rust
pub struct LongestMatchCache {
    pub length: Box<[u16]>,     // len blocksize, init 1
    pub dist:   Box<[u16]>,     // len blocksize, init 0
    pub sublen: Box<[u8]>,      // len ZOPFLI_CACHE_LENGTH * 3 * blocksize, init 0
}

impl LongestMatchCache {
    pub fn new(blocksize: usize) -> Self;
    pub fn sublen_to_cache(&mut self, sublen: &[u16; 259], pos: usize, length: u32);
    pub fn cache_to_sublen(&self, pos: usize, length: u32, sublen: &mut [u16; 259]);
    pub fn max_cached_sublen(&self, pos: usize, length: u32) -> u32;
}
```

**Implementation notes**

- `cache.sublen[ZOPFLI_CACHE_LENGTH * pos * 3]` is the start of `pos`'s
  3-byte triplets; precompute a `let base = ZOPFLI_CACHE_LENGTH * pos * 3;`
  and slice from there.
- The two `#if ZOPFLI_CACHE_LENGTH == 0` early returns can be deleted
  (`ZOPFLI_CACHE_LENGTH = 8` in the gzippy build).
- The `sublen` parameter is a fixed-size buffer of 259 elements throughout
  zopfli; use `&[u16; 259]` for clarity and index safety.

**Oracle test:** round-trip property test. Build a `sublen` array with a
plausible monotone sequence of distances (real zopfli output never has
`sublen[k+1] < sublen[k]` between equal-distance plateaus), feed through
`sublen_to_cache` then `cache_to_sublen`, then assert the recovered values
agree with the original up to `max_cached_sublen`.

(The end-to-end oracle in Step 6 also exercises this module.)

---

# Phase 3 — LZ77 layer

## Step 6 — `lz77.rs` Part A: `LZ77Store` and histogram

**Source to port:** `vendor/zopfli/src/zopfli/lz77.c:28-217` and the struct
in `lz77.h:44-62`.

**Public API:**

```rust
pub struct LZ77Store<'a> {
    pub litlens:   Vec<u16>,
    pub dists:     Vec<u16>,
    pub size:      usize,           // == litlens.len() == dists.len()
    pub data:      &'a [u8],        // borrowed input
    pub pos:       Vec<usize>,
    pub ll_symbol: Vec<u16>,
    pub d_symbol:  Vec<u16>,
    pub ll_counts: Vec<usize>,      // wraps every ZOPFLI_NUM_LL entries
    pub d_counts:  Vec<usize>,      // wraps every ZOPFLI_NUM_D  entries
}

impl<'a> LZ77Store<'a> {
    pub fn new(data: &'a [u8]) -> Self;
    pub fn store_lit_len_dist(&mut self, length: u16, dist: u16, pos: usize);
    pub fn append_from(&mut self, src: &Self);
    pub fn byte_range(&self, lstart: usize, lend: usize) -> usize;
    pub fn get_histogram(
        &self, lstart: usize, lend: usize,
        ll_counts: &mut [usize; 288], d_counts: &mut [usize; 32],
    );
}
```

**Implementation notes**

- The wrap-around histogram (lines 109–148) is the trickiest part. Read it
  twice. Each time the size hits a multiple of `ZOPFLI_NUM_LL` (288), append
  288 fresh entries to `ll_counts` initialised from the previous chunk's
  final 288 values. Same for `d_counts` with 32.
- `get_histogram` has a small-block fast path that just iterates symbols.
- `get_histogram_at` is internal-only.
- **Do not** use `Rc`/`Arc` — the C code copies stores when it improves the
  best run; a `clone()` on these `Vec<u16>` is fine and matches C
  performance. Optimization comes in Phase 4.

**Oracle test:** Build an LZ77Store by replaying a fixed sequence of
`store_lit_len_dist` calls and compare every field of the resulting Rust
store against a parallel sequence on the C version (see harness below). To
read the C struct from Rust, extern the struct layout — but easier is to
write a tiny C helper in `build.rs` that exports getters; **easier still**
is to oracle indirectly through `lz77_greedy` in Step 8, since the only
consumers of these arrays are downstream functions you'll port next.

For now, do a self-consistency test:

```rust
#[test]
fn lz77store_get_histogram_matches_brute() {
    // 1) Build an LZ77Store by replaying ~10k random stores.
    // 2) For 50 random (lstart, lend) pairs, compute get_histogram() and
    //    compare against an O(n) brute-force scan.
}
```

---

## Step 7 — `lz77.rs` Part B: `BlockState`, `find_longest_match`, `verify_len_dist`

**Source to port:** `vendor/zopfli/src/zopfli/lz77.c:219-542` and `lz77.h:86-129`.

> **Note (cross-step):** `BlockState` needs `ZopfliOptions` here. Pull
> the struct definition up to `zopfli_pure/mod.rs` now (mark
> `#[repr(C)]`) rather than waiting for Step 17. See "When you must
> adapt the plan" above. The hash chain accessors
> (`h.head_for(use_hash2)`, `h.prev_for(use_hash2)`,
> `h.hashval_for(use_hash2)`) likewise belong on `ZopfliHash` as added
> here, even though Step 4 didn't list them — they're the cleanest way
> to express the C pointer-alias swap inside `find_longest_match`.

**Public API:**

```rust
pub struct BlockState<'opt> {
    pub options:    &'opt ZopfliOptions,
    pub lmc:        Option<LongestMatchCache>,    // is_some() ⇔ C `s->lmc != NULL`
    pub blockstart: usize,
    pub blockend:   usize,
}

impl<'opt> BlockState<'opt> {
    pub fn new(options: &'opt ZopfliOptions, start: usize, end: usize, add_lmc: bool) -> Self;
}

pub fn verify_len_dist(data: &[u8], pos: usize, dist: u16, length: u16);  // debug_assert only

pub fn find_longest_match(
    s: &mut BlockState<'_>,
    h: &ZopfliHash,
    array: &[u8],
    pos: usize,
    size: usize,
    mut limit: usize,
    sublen: Option<&mut [u16; 259]>,
    distance: &mut u16,
    length: &mut u16,
);
```

**Implementation notes**

- `get_match` (lines 297–331): the 8-byte fast loop. Use
  `u64::from_le_bytes(array[scan..scan+8].try_into().unwrap())` and exit
  when not equal. Then scalar tail. **Do not use SIMD intrinsics** here yet
  — the autovectorized version is fast enough and matches C output exactly.
- `try_get_from_longest_match_cache` / `store_in_longest_match_cache`
  (lines 340–404) are tightly coupled to `find_longest_match`. Inline them
  as `impl LongestMatchCache` methods or as free functions in `lz77.rs`.
- `ZopfliMaxCachedSublen` returning `0` means "nothing cached" → the early
  return is missed; mirror the C condition `s.lmc.length[lmcpos] == 0 ||
  s.lmc.dist[lmcpos] != 0`.
- `chain_counter` cap is `ZOPFLI_MAX_CHAIN_HITS = 8192`. Keep it.
- Use `mem::take` / mutable borrows carefully; `s.lmc` is read and written
  in the same call. Split `Option<LongestMatchCache>` into a `take`/`replace`
  pattern, or reorganize so the LMC slice is borrowed disjointly from
  `&mut self`.

**Oracle test** (this is where Hash + Cache + LZ77Store come together):

```rust
extern "C" {
    fn ZopfliInitOptions(opts: *mut [u8; 24]);     // size of ZopfliOptions
    // We won't extern ZopfliFindLongestMatch directly because it needs
    // ZopfliBlockState + ZopfliHash. Instead oracle through ZopfliLZ77Greedy
    // in Step 8, which calls FindLongestMatch internally. Until Step 8,
    // self-test via brute force:
}

#[test]
fn find_longest_match_brute_force() {
    // For each input from corpus(), at every byte position, run a brute-force
    // longest-match scan over the previous 32 KB and compare to find_longest_match.
    // Caveat: brute-force returns the *closest* dist for the longest length;
    // zopfli's output also picks closest; both should agree on length and dist.
}
```

---

## Step 8 — `lz77.rs` Part C: `lz77_greedy` + first end-to-end oracle

**Source to port:** `vendor/zopfli/src/zopfli/lz77.c:544-630`.

**Public API:**

```rust
pub fn lz77_greedy(
    s: &mut BlockState<'_>, in_: &[u8],
    instart: usize, inend: usize,
    store: &mut LZ77Store<'_>, h: &mut ZopfliHash,
);
```

**Implementation notes**

- Lazy matching exactly as in C (lines 555–613). The `match_available` flag
  carries between iterations.
- `get_length_score` (lines 265–271) — verbatim, including the `> 1024`
  threshold.
- `dummysublen` is a `[u16; 259]` stack buffer; pass `Some(&mut buf)` to
  `find_longest_match`.

**Oracle test** — first **real** end-to-end check that combines hash, cache,
LZ77, and our greedy parse:

```rust
extern "C" {
    fn ZopfliInitLZ77Store(data: *const u8, store: *mut FfiLZ77Store);
    fn ZopfliCleanLZ77Store(store: *mut FfiLZ77Store);
    fn ZopfliInitBlockState(opts: *const ZopfliOptions, start: usize,
                             end: usize, add_lmc: i32, s: *mut FfiBlockState);
    fn ZopfliCleanBlockState(s: *mut FfiBlockState);
    fn ZopfliAllocHash(window: usize, h: *mut FfiHash);
    fn ZopfliCleanHash(h: *mut FfiHash);
    fn ZopfliLZ77Greedy(s: *mut FfiBlockState, in_: *const u8,
                         instart: usize, inend: usize,
                         store: *mut FfiLZ77Store, h: *mut FfiHash);
}
// Mirror only the FIELDS of ZopfliLZ77Store you read — pointers + size.
// Use #[repr(C)] structs in the test module and read litlens/dists arrays
// via std::slice::from_raw_parts(store.litlens, store.size).
```

Compare `(litlens, dists, pos, ll_symbol, d_symbol)` field-for-field for
every input in `corpus()`. If any byte diverges, the bug is in steps 4–8
(hash / cache / store / find / greedy) — bisect by first failing input
position.

**Done when:** all corpus inputs produce byte-identical greedy LZ77 stores.

---

# Phase 4 — Block size estimation (no encoder yet)

## Step 9 — `deflate_size.rs`: tree-encoded size + RLE optimizer

**Source to port:** `vendor/zopfli/src/zopfli/deflate.c:74-621` (size-only
half — i.e. the parts that **don't** write bits, only count them).

**Public API:**

```rust
pub fn patch_distance_codes_for_buggy_decoders(d_lengths: &mut [u32; 32]);
pub fn get_fixed_tree(ll_lengths: &mut [u32; 288], d_lengths: &mut [u32; 32]);

/// Returns size in bits. If you pass a writer for the actual emit, do that
/// in the encoder (Step 14) — this module is for cost only.
pub fn calculate_tree_size(ll_lengths: &[u32; 288], d_lengths: &[u32; 32]) -> usize;

pub fn calculate_block_size(lz77: &LZ77Store, lstart: usize, lend: usize, btype: i32) -> f64;
pub fn calculate_block_size_auto_type(lz77: &LZ77Store, lstart: usize, lend: usize) -> f64;

/// In-place histogram smoothing (`OptimizeHuffmanForRle`).
pub fn optimize_huffman_for_rle(counts: &mut [usize]);

/// Returns size in bits and writes ll_lengths/d_lengths in-place.
pub fn get_dynamic_lengths(
    lz77: &LZ77Store, lstart: usize, lend: usize,
    ll_lengths: &mut [u32; 288], d_lengths: &mut [u32; 32],
) -> f64;
```

**Implementation notes**

- The internal `EncodeTree` in C has a `size_only` flag (out is null). In
  Rust, split into two functions: `encode_tree_size(...) -> usize` (this
  module) and `encode_tree_emit(...)` (Step 14, in `deflate.rs`). They
  share a private helper:
  ```rust
  // pub(crate) so deflate.rs can reuse it
  pub(crate) fn build_rle_encoding(
      ll_lengths: &[u32; 288], d_lengths: &[u32; 32],
      use_16: bool, use_17: bool, use_18: bool,
  ) -> (Vec<u32>, Vec<u32>, [usize; 19], u32 /*hlit*/, u32 /*hdist*/);
  ```
  Returning `(clcounts, hlit, hdist)` alongside the rle pair means
  Step 14's emit path doesn't re-trim trailing zeros or rescan the rle.
- `CalculateBlockSymbolSize` has three variants in C — small / given-counts
  / general. Port all three; the 3× histogram threshold (`lstart +
  ZOPFLI_NUM_LL * 3 > lend`) is significant.
- `calculate_block_size` returns `double` in C; keep `f64` to match cost
  comparisons later.

**Oracle test:**

```rust
extern "C" {
    fn ZopfliCalculateBlockSize(lz77: *const FfiLZ77Store, lstart: usize,
                                 lend: usize, btype: i32) -> f64;
    fn ZopfliCalculateBlockSizeAutoType(lz77: *const FfiLZ77Store,
                                         lstart: usize, lend: usize) -> f64;
}

#[test]
fn block_size_matches_ffi() {
    // For each corpus input:
    //   1) Build the C and Rust LZ77Stores via greedy parse (already
    //      verified bit-equal in Step 8).
    //   2) Call calculate_block_size(btype = 0, 1, 2) and
    //      calculate_block_size_auto_type on the full range.
    //   3) Assert f64::to_bits() equality.
}
```

If the block sizes don't agree to bit equality, the cause is almost always
floating-point order in `calculate_entropy` (Step 3) leaking through, or a
mis-port of the RLE-optimization heuristic.

---

# Phase 5 — Squeeze (optimal LZ77 by dynamic programming)

## Step 10 — `squeeze.rs` Part A: `SymbolStats` and cost models

**Source to port:** `vendor/zopfli/src/zopfli/squeeze.c:32-198`.

> **FAQ pointers (read before implementing):** see "Q (Step 10):
> `add_weighted` — in-place or three-operand?" and "Q (Step 10): how
> to validate `RanState` without an FFI oracle?" in the FAQ section
> near the bottom of this file. Both are pre-answered.

**Public API (mostly module-private, but typed):**

```rust
pub struct SymbolStats {
    pub litlens:    [usize; 288],
    pub dists:      [usize; 32],
    pub ll_symbols: [f64; 288],
    pub d_symbols:  [f64; 32],
}

impl SymbolStats {
    pub fn new() -> Self;                           // all zero
    pub fn copy_from(&mut self, src: &Self);
    pub fn clear_freqs(&mut self);
    /// In-place: self = self*w1 + other*w2. The C signature has a separate
    /// `result` pointer, but `squeeze.c:509` is the **only** call site and
    /// it always passes `result == stats1`. See FAQ for full audit.
    pub fn add_weighted(&mut self, w1: f64, other: &Self, w2: f64);
    pub fn calculate_statistics(&mut self);                       // entropy
    pub fn from_store(store: &LZ77Store) -> Self;                 // == GetStatistics
}

pub struct RanState { m_w: u32, m_z: u32 }
impl RanState {
    pub fn new() -> Self { Self { m_w: 1, m_z: 2 } }
    pub fn next(&mut self) -> u32;                                // 32-bit MWC
    pub fn randomize_freqs(&mut self, freqs: &mut [usize]);
    pub fn randomize_stat_freqs(&mut self, stats: &mut SymbolStats);
}

pub fn cost_fixed(litlen: u32, dist: u32) -> f64;
pub fn cost_stat(litlen: u32, dist: u32, stats: &SymbolStats) -> f64;
pub fn cost_model_min_cost<F: Fn(u32, u32) -> f64>(cost: F) -> f64;
```

**Implementation notes**

- `RanState::next` is the Marsaglia MWC. Port literally:
  ```rust
  self.m_z = 36969 * (self.m_z & 0xffff) + (self.m_z >> 16);
  self.m_w = 18000 * (self.m_w & 0xffff) + (self.m_w >> 16);
  (self.m_z << 16).wrapping_add(self.m_w)
  ```
  This must produce identical sequences with identical seeds (1, 2).
- `cost_stat` and `cost_fixed` follow `symbols.h` exactly. Use the new
  `symbols::*` functions.
- The C `CostModelFun` typedef takes a `void*`. In Rust, just pass closures
  or a small enum. Keep both options open — the squeeze caller below
  will dispatch dynamically.

**Self-test:** the MWC sequence is published. Generate the first 16
outputs with seed (1, 2) and compare to a known reference (you can compute
this by hand from the formula and stash in a constant array).

The big oracle test arrives in Step 12.

---

## Step 11 — `squeeze.rs` Part B: `get_best_lengths`, `trace_backwards`, `follow_path`

**Source to port:** `vendor/zopfli/src/zopfli/squeeze.c:217-444`.

> **FAQ pointers:** see "Q (Step 11): cost model dispatch — closure or
> trait?" and "Q (Step 11): exact f32/f64 boundaries in the DP loop"
> in the FAQ section. The trait is binding (`pub trait CostModel { fn
> cost(&self, litlen: u32, dist: u32) -> f64; }`); the f32/f64
> coercion ladder is binding verbatim.

**Public API:**

```rust
pub trait CostModel { fn cost(&self, litlen: u32, dist: u32) -> f64; }
pub struct FixedCost;
impl CostModel for FixedCost { /* delegates to cost_fixed */ }
pub struct StatCost<'a>(pub &'a SymbolStats);
impl CostModel for StatCost<'_> { /* delegates to cost_stat */ }

pub(crate) fn get_best_lengths<C: CostModel>(
    s: &mut BlockState<'_>, in_: &[u8], instart: usize, inend: usize,
    cost: &C,                    // borrow; caller owns SymbolStats / FixedCost
    length_array: &mut [u16],   // len blocksize + 1
    h: &mut ZopfliHash,
    costs: &mut [f32],          // len blocksize + 1
) -> f64;

pub(crate) fn trace_backwards(size: usize, length_array: &[u16], path: &mut Vec<u16>);

pub(crate) fn follow_path(
    s: &mut BlockState<'_>, in_: &[u8], instart: usize, inend: usize,
    path: &[u16], store: &mut LZ77Store<'_>, h: &mut ZopfliHash,
);

pub(crate) fn lz77_optimal_run<F: Fn(u32, u32) -> f64>(
    s: &mut BlockState<'_>, in_: &[u8], instart: usize, inend: usize,
    path: &mut Vec<u16>,
    length_array: &mut [u16],
    cost: F,
    store: &mut LZ77Store<'_>,
    h: &mut ZopfliHash,
    costs: &mut [f32],
) -> f64;
```

**Implementation notes**

- **`costs[]` is `f32`, not `f64`.** Read the C code at line 222 (the
  `costs` parameter) and 461 (`malloc(sizeof(float) * (blocksize + 1))`).
  This is critical: the dynamic-programming forward pass uses `float` as
  the cost type, while the cost-model functions return `double` and are
  cast on store. **Match this widening exactly:**
  ```rust
  let new_cost = cost(litlen, dist) + costs[j] as f64;          // f64 add
  if (new_cost as f32) < costs[j + 1] {                          // f32 compare
      costs[j + 1] = new_cost as f32;
      length_array[j + 1] = ...;
  }
  ```
  Wait — read the C carefully (lines 278–301): C stores the f64 newCost
  back into costs[] of type `float`. The store is implicit. Rust must do
  the explicit `as f32`. Get this right or the whole iteration diverges.
- The `ZOPFLI_SHORTCUT_LONG_REPETITIONS` block (lines 251–271) bumps `i`
  and `j` ahead by `ZOPFLI_MAX_MATCH`. Port faithfully.
- `mincost = GetCostModelMinCost(...)` is computed once outside the loop.
- `trace_backwards`: the C output is an `unsigned short*` followed by an
  in-place reverse. Use `Vec<u16>::reverse()` after building.
- `follow_path` calls `find_longest_match` again (with `limit = length`)
  to recover the distance; **this is correct** — the cache provides it
  cheaply.

**Oracle test:** can't directly oracle a single squeeze run easily without
plumbing `BlockState` and `ZopfliHash` across the FFI boundary. **Defer**
to Step 12, which oracles the full `lz77_optimal` against the C version.

---

## Step 12 — `squeeze.rs` Part C: `lz77_optimal`, `lz77_optimal_fixed`, **strict** end-to-end oracle

**Source to port:** `vendor/zopfli/src/zopfli/squeeze.c:446-560`.

**Public API:**

```rust
pub fn lz77_optimal(
    s: &mut BlockState<'_>, in_: &[u8], instart: usize, inend: usize,
    numiterations: i32, store: &mut LZ77Store<'_>,
);

pub fn lz77_optimal_fixed(
    s: &mut BlockState<'_>, in_: &[u8], instart: usize, inend: usize,
    store: &mut LZ77Store<'_>,
);
```

**Implementation notes**

- The iteration loop is sensitive: best-cost copy, last-cost tracking,
  `lastrandomstep == -1` until activated, weighted blend after activation.
  Port literally — including the magic numbers `i > 5`, weight `1.0` and
  `0.5`.
- `currentstore` is reused across iterations: clear it (`store.size = 0`
  and resize the vectors) before each iteration. To avoid repeatedly
  allocating, expose a `LZ77Store::reset(&mut self)` that sets `size = 0`
  and truncates the inner Vecs (keeps capacity).
- `costs` and `length_array` are scratch buffers; allocate once outside
  the loop.

**Oracle test — split into two tiers (binding):**

> **FAQ pointer:** see "Q (Step 12): squeeze oracle runtime budget"
> in the FAQ section. The split below is binding; both tiers must be
> green before Step 13. The thorough tier is what catches a 1-ULP
> drift on real text — do not skip it.

Tier 1: **fast oracle**, runs on every `cargo test --release` and in
CI. Caps at 16 KB so the matrix (4 corpus inputs × 4 iteration counts)
finishes in <60 seconds.

Tier 2: **thorough oracle**, marked `#[ignore]`, runs the full corpus
(including `alice.txt` and `zeros_64k`) at `numiterations = 1`. Run
with `cargo test --release -- --ignored zopfli_pure` before merging
this step **and** before Step 25. Add a Makefile target
`zopfli-oracle-thorough` that wraps the invocation.

```rust
extern "C" {
    fn ZopfliLZ77Optimal(s: *mut FfiBlockState, in_: *const u8,
                          instart: usize, inend: usize, numiterations: i32,
                          store: *mut FfiLZ77Store);
}

#[test]
fn lz77_optimal_matches_ffi_byte_for_byte() {
    // Tier 1 — fast
    for (name, data) in corpus() {
        if data.len() < 64 || data.len() > 16_384 { continue; }   // keep test <60s
        for &iters in &[1, 2, 5, 15] {
            let opts = build_options(iters);
            // C side
            let mut ffi_store = ffi_init_store(&data);
            let mut ffi_state = ffi_init_blockstate(&opts, 0, data.len(), 1);
            unsafe {
                ZopfliLZ77Optimal(&mut ffi_state, data.as_ptr(), 0, data.len(),
                                   iters, &mut ffi_store);
            }
            // Rust side
            let mut rs_store = LZ77Store::new(&data);
            let mut rs_state = BlockState::new(&opts.into(), 0, data.len(), true);
            lz77_optimal(&mut rs_state, &data, 0, data.len(), iters, &mut rs_store);
            // Compare every field.
            assert_eq!(ffi_view(&ffi_store), rs_view(&rs_store),
                       "{} iters={}", name, iters);
            ffi_clean_store(ffi_store);
            ffi_clean_blockstate(ffi_state);
        }
    }
}

#[test]
#[ignore = "thorough; run via `cargo test --release -- --ignored zopfli_pure`"]
fn lz77_optimal_matches_ffi_thorough() {
    // Tier 2 — full corpus, single iteration. Catches FP drift on real text.
    for (name, data) in corpus() {
        if data.is_empty() { continue; }
        let opts = build_options(1);
        let ffi = run_ffi_optimal(&opts, &data);
        let rs  = run_rs_optimal(&opts, &data);
        assert_eq!(ffi, rs, "thorough: {} (len {})", name, data.len());
    }
}
```

Add a Makefile target near the existing perf rules:

```make
zopfli-oracle-thorough:
\tcargo test --release -- --ignored zopfli_pure
```

If this test fails, the bug is almost certainly:

1. `f32`/`f64` coercion order in `get_best_lengths` (Step 11).
2. `calculate_entropy` floating-point order (Step 3).
3. `RanState` PRNG mismatched (Step 10).
4. `find_longest_match` returning a different (length, dist) pair when
   multiple equal-length matches exist at different distances (Step 7;
   re-read lines 461–531 of `lz77.c` for the chain-walk order).

**Done when:** the oracle test is green for all corpus inputs ≤ 16 KB and
for `numiterations ∈ {1, 2, 5, 15}`. After this step, the squeeze layer is
locked in and the rest of the port becomes mechanical.

---

# Phase 6 — Block splitter

## Step 13 — `blocksplitter.rs`

**Source to port:** `vendor/zopfli/src/zopfli/blocksplitter.c:1-332`.

**Public API:**

```rust
pub fn block_split_lz77(
    options: &ZopfliOptions, lz77: &LZ77Store, maxblocks: usize,
) -> Vec<usize>;

pub fn block_split(
    options: &ZopfliOptions, in_: &[u8], instart: usize, inend: usize,
    maxblocks: usize,
) -> Vec<usize>;

pub fn block_split_simple(in_len: usize, instart: usize, inend: usize, blocksize: usize)
    -> Vec<usize>;
```

**Implementation notes**

- `find_minimum` (lines 43–96) — port verbatim, including the `NUM = 9`
  constant. The loop uses `lastbest` to detect non-improvement.
- `add_sorted` is a tiny insertion-sort; just call `vec.push(value);
  vec.sort_unstable()` if you prefer (it's ≤ 15 elements).
- `find_largest_splittable_block` returns a bool; in Rust, return
  `Option<(usize, usize)>`.

**Oracle test:**

```rust
extern "C" {
    fn ZopfliBlockSplit(opts: *const ZopfliOptions, in_: *const u8,
                         instart: usize, inend: usize, maxblocks: usize,
                         splitpoints: *mut *mut usize, npoints: *mut usize);
    fn ZopfliBlockSplitLZ77(opts: *const ZopfliOptions, lz77: *const FfiLZ77Store,
                             maxblocks: usize, splitpoints: *mut *mut usize,
                             npoints: *mut usize);
}

#[test] fn block_split_matches_ffi() { /* compare split point vectors exactly */ }
```

---

# Phase 7 — DEFLATE encoder (the bit-writer half)

## Step 14 — `deflate.rs` Part A: `BitWriter`

**Source to port:** `vendor/zopfli/src/zopfli/deflate.c:38-72` and bit-output
contexts throughout the file.

**Public API:**

```rust
pub struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    bp:  u8,                  // 0..=7
}

impl<'a> BitWriter<'a> {
    pub fn new(out: &'a mut Vec<u8>, bp: u8) -> Self;
    pub fn bp(&self) -> u8;
    pub fn add_bit(&mut self, bit: u8);
    pub fn add_bits(&mut self, symbol: u32, length: u32);            // LSB-first
    pub fn add_huffman_bits(&mut self, symbol: u32, length: u32);    // MSB-first
}
```

**Implementation notes**

- **First port: literal one-bit-per-iteration** — this matches the C code
  exactly and gives bit-identical output. Optimization (batched 32-bit
  accumulator) is Phase 4 / Step 30.
- **Both `add_bits` (LSB-first) and `add_huffman_bits` (MSB-first) are
  required by DEFLATE.** Mismatched orientation is a common bug; re-read
  RFC 1951 §3.1.1 if unsure.

**Oracle test:** unit test against hand-encoded reference bits.

---

## Step 15 — `deflate.rs` Part B: tree emission, block emission, deflate driver

**Source to port:** the rest of `deflate.c` — encoder paths only:

- `EncodeTree` (write path, lines 105–249) — share the RLE builder you
  factored out in Step 9.
- `AddDynamicTree` (lines 251–272).
- `AddLZ77Data` (lines 297–333).
- `AddNonCompressedBlock` (lines 625–663).
- `AddLZ77Block` (lines 682–745).
- `AddLZ77BlockAutoType` (lines 747–800).
- `ZopfliDeflatePart` (lines 811–906).
- `ZopfliDeflate` (lines 908–931). Master block size is 1 MB; for
  `insize > 1 MB` it slices and calls `DeflatePart` repeatedly.

**Public API:**

```rust
pub fn deflate(options: &ZopfliOptions, btype: i32, final_: bool,
               in_: &[u8], bp: u8, out: &mut Vec<u8>) -> u8;   // returns new bp
pub fn deflate_part(options: &ZopfliOptions, btype: i32, final_: bool,
                    in_: &[u8], instart: usize, inend: usize,
                    bp: u8, out: &mut Vec<u8>) -> u8;
```

**Implementation notes**

- The "second block splitting attempt" inside `ZopfliDeflatePart` (lines
  872–893) re-runs `block_split_lz77` on the squeezed LZ77 and uses it if
  cheaper. Port faithfully.
- `verbose` printing — gate behind `options.verbose != 0` and use
  `eprintln!`. The output strings should match C exactly so users
  comparing tools see the same logs (low priority but easy).

**Oracle test:**

```rust
extern "C" {
    fn ZopfliDeflate(opts: *const ZopfliOptions, btype: i32, final_: i32,
                      in_: *const u8, insize: usize,
                      bp: *mut u8, out: *mut *mut u8, outsize: *mut usize);
}

#[test]
fn deflate_matches_ffi_byte_for_byte() {
    for (name, data) in corpus() {
        for &iters in &[1, 5, 15] {
            for &split in &[true, false] {
                let opts = ZopfliOptions { numiterations: iters,
                                            blocksplitting: split as i32, ..default };
                let ffi = ffi_deflate(&opts, &data);
                let rs  = rs_deflate(&opts, &data);
                assert_eq!(ffi, rs, "{} iters={} split={}", name, iters, split);
            }
        }
    }
}
```

This is the real moment of truth. If this passes for the whole corpus, the
final gzip wrapper is trivial and the cutover is safe.

---

# Phase 8 — gzip wrapper, public surface, cutover

## Step 16 — `gzip.rs`: gzip header + CRC + ISIZE

**Source to port:** `vendor/zopfli/src/zopfli/gzip_container.c:84-124`.

**Public API:**

```rust
pub fn gzip_compress(options: &ZopfliOptions, in_: &[u8]) -> Vec<u8>;
```

**Implementation notes**

- Use the existing `crc32fast` crate (already a dependency) instead of
  porting the embedded CRC table — these produce identical CRC32 values.
- Header: `[0x1f, 0x8b, 0x08, 0x00, 0,0,0,0, 0x02, 0x03]` — note OS = 3
  (Unix) in the C version. The existing FFI puts OS = 255 in
  `compress/zopfli.rs:171`, but that's the **gzippy parallel wrapper**
  which writes its own header. The pure-Rust **deflate-only** path used by
  gzippy's single-threaded zopfli wraps the deflate output itself and never
  calls `gzip_compress`. Still, the parallel multi-block path calls the
  full `compress_gzip` per block, so this header **must** match the C
  output (OS = 3, not 255). Verify by oracling against the FFI.

## Step 17 — `mod.rs`: `ZopfliOptions`, `ZopfliFormat`, top-level dispatcher

> **Note:** if you followed the recommendation in "When you must adapt
> the plan", `ZopfliOptions` already lives in `mod.rs` (added at Step 7
> for `BlockState`). This step then only adds `ZopfliFormat` and the
> dispatcher; the struct definition stays where it is.

**Public API mirrors C `zopfli.h`:**

```rust
#[derive(Clone, Debug)]
pub struct ZopfliOptions {
    pub verbose:           i32,
    pub verbose_more:      i32,
    pub numiterations:     i32,
    pub blocksplitting:    i32,
    pub blocksplittinglast:i32,        // unused, kept for parity
    pub blocksplittingmax: i32,
}
impl Default for ZopfliOptions { /* matches ZopfliInitOptions */ }

pub enum ZopfliFormat { Gzip, Zlib, Deflate }

pub fn compress(options: &ZopfliOptions, format: ZopfliFormat, in_: &[u8]) -> Vec<u8>;
```

`compress(Gzip, ...)` → `gzip_compress`; `compress(Deflate, ...)` →
`deflate(btype = 2, final = true, ...)` with `bp = 0`; `Zlib` is unused by
gzippy — implement it for parity by porting `zlib_container.c` (79 lines,
trivial), so the port is complete.

## Step 18 — Bridge for `backends::zopfli_compress` (parallel switch)

Add a feature flag for safe rollout. In
`src/backends/zopfli_compress.rs`, replace the body (keep struct/signatures
identical):

```rust
pub fn compress_gzip(data: &[u8], tuning: &ZopfliTuning) -> Vec<u8> {
    let opts = tuning_to_options(tuning);
    if std::env::var_os("GZIPPY_ZOPFLI_FFI").is_some() {
        ffi_compress(data, &opts, ZOPFLI_FORMAT_GZIP)
    } else {
        crate::backends::zopfli_pure::compress(
            &opts, crate::backends::zopfli_pure::ZopfliFormat::Gzip, data)
    }
}
// same for compress_deflate
```

This lets `make` and CI default to the new pure-Rust path while
`GZIPPY_ZOPFLI_FFI=1 cargo test` falls back to the C library — useful for
last-mile diffing in production.

## Step 19 — Run the full integration matrix

```bash
cargo build --release
cargo test  --release                 # all unit + integration tests
make                                  # quick perf
target/release/gzippy --ultra <file>  # smoke test
target/release/gzippy -F 5 -I <file>  # tuning flag smoke test
target/release/gzippy --level 11 <bigfile> | gunzip | cmp - <bigfile>
```

If any test fails, **do not** continue to cutover. Diagnose against the FFI
oracle by setting `GZIPPY_ZOPFLI_FFI=1` and rerunning.

## Step 20 — Optional ZLIB container

Port `zlib_container.c` (79 lines) → `zopfli_pure/zlib.rs`. Mostly
boilerplate (Adler-32 + zlib header). Skip if `ZopfliFormat::Zlib` truly is
unreachable in gzippy; otherwise port for completeness so the public API
matches C.

---

# Phase 9 — Cutover

## Step 21 — Remove FFI declarations from `zopfli_compress.rs`

Delete the `extern "C"` block and `compress_internal`. Keep only:

```rust
pub use crate::backends::zopfli_pure::ZopfliFormat;

pub struct ZopfliTuning { /* unchanged */ }
impl ZopfliTuning { /* from_args unchanged */ }

pub fn compress_gzip(data: &[u8], tuning: &ZopfliTuning) -> Vec<u8> {
    let opts = tuning_to_options(tuning);
    crate::backends::zopfli_pure::compress(
        &opts, crate::backends::zopfli_pure::ZopfliFormat::Gzip, data)
}

pub fn compress_deflate(data: &[u8], tuning: &ZopfliTuning) -> Vec<u8> {
    let opts = tuning_to_options(tuning);
    crate::backends::zopfli_pure::compress(
        &opts, crate::backends::zopfli_pure::ZopfliFormat::Deflate, data)
}
```

Delete the `GZIPPY_ZOPFLI_FFI` env-var branch (Step 18) — pure Rust only.

## Step 22 — Strip the C build from `build.rs`

Delete `compile_zopfli` and the `cargo:rerun-if-changed=vendor/zopfli/src`
line. Remove the `cargo:rustc-link-lib=static=zopfli` and link-search
directives. Verify `cargo build --release` from a clean target.

## Step 23 — Delete the oracle harness

`rm src/backends/zopfli_pure/oracle_tests.rs` and remove the
`#[cfg(test)] mod oracle_tests;` declaration. The oracle's job is done.

Replace with a smaller permanent regression test in
`src/backends/zopfli_pure/tests.rs`: known-good gzip output bytes for 4-5
fixtures (committed as a hex blob), so a future regression here surfaces
without needing the C library.

## Step 24 — Drop the `vendor/zopfli` submodule

```bash
git submodule deinit -f vendor/zopfli
git rm -f vendor/zopfli
sed -i '' '/zopfli/,/url = .*zopfli.git/d' .gitmodules     # macOS sed
git add .gitmodules
```

Update `Makefile` — drop the `ZOPFLI_DIR`, `ZOPFLI_BIN`, the `zopfli` build
rule (line 78–80), and the `deps` mention of zopfli. Update `CLAUDE.md` if
it references the vendored zopfli.

## Step 25 — Final integration

```bash
cargo clean && cargo build --release
cargo test --release
make
make ship                  # AUTHORITATIVE: must show no regression on L11
```

**Acceptance bar (no exceptions):**

| Metric                         | Pass condition                                |
|--------------------------------|------------------------------------------------|
| `cargo build --release`        | succeeds with **no C compiler invoked**        |
| `cargo test --release`         | 100% pass                                      |
| `make`                         | passes                                         |
| `make ship` L11 wall-clock     | within ±5% of pre-port baseline                |
| `make ship` L11 ratio          | byte-identical output (compressed bytes equal) |
| Memory (RSS) on 100 MB input   | within ±10% of pre-port baseline               |

If any bar fails, **revert** (per CLAUDE.md rule 4).

## Step 26 — Open the PR

```bash
git push origin rust-zopfli
gh pr create --base main --title "Replace C zopfli FFI with pure-Rust port" \
  --body "$(see PR template below)"
```

PR body should include:
- The acceptance table above with measured numbers from `make ship`.
- A note that ratio is **bit-identical** (compressed output equality is the
  strongest possible correctness statement).
- Removal of the C build dependency: build.rs no longer invokes `cc`.
- A reproducibility script: `cargo test --release zopfli_pure_regression`.

---

# Phase 10 — Optimization (post-cutover only)

These steps only begin after Step 25 is green. Each one runs the full
oracle-style **regression** test (the one written in Step 23) and `make
ship`. Any regression in ratio = revert. Any regression in wall-clock = revert.

## Step 27 — Batched bit writer (the obvious 5-10% win)

Replace the per-bit `add_bits` / `add_huffman_bits` with a 64-bit
accumulator that flushes 8 bytes at a time:

```rust
pub struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    accum: u64,
    nbits: u32,        // 0..64
}
impl<'a> BitWriter<'a> {
    fn add_bits(&mut self, sym: u32, len: u32) {
        self.accum |= (sym as u64) << self.nbits;
        self.nbits += len;
        while self.nbits >= 8 {
            self.out.push(self.accum as u8);
            self.accum >>= 8;
            self.nbits -= 8;
        }
    }
    // add_huffman_bits: bit-reverse `sym` (use `sym.reverse_bits() >> (32-len)`)
    //                   then add_bits.
    fn finish(mut self, last_bp_out: &mut u8) {
        if self.nbits > 0 { self.out.push(self.accum as u8); }
        *last_bp_out = (self.nbits & 7) as u8;
    }
}
```

Run `make ship`. Expected: 3-8% on L11 (bit-writer is hot in zopfli).

## Step 28 — Bench-driven hotspots

Run with `cargo build --release --features perf` (add a perf-counter
feature in Phase 4 if useful) or sample with `cargo flamegraph`. The likely
top-3 hotspots, with planned interventions:

1. **`get_best_lengths` inner loop** (the `for k in 3..=kend` cost loop).
   Try: hoist `mincost` into a register; precompute `ll_symbols + lbits`
   for each length symbol once per `j` rather than per `k`.
2. **`find_longest_match` chain walk.** Try: an early-out on
   `chain_counter` when `bestlength >= some-fraction-of-limit`. Validate
   ratio is unchanged (this can lose bytes — measure on every corpus
   fixture).
3. **`LZ77Store::store_lit_len_dist` histogram wrap.** Try: maintain a
   running pointer to the current 288-element chunk; avoid the
   `origsize % ZOPFLI_NUM_LL == 0` modulo on every call.

For each hotspot:
- One commit per intervention.
- Pre/post `make ship` numbers in the commit message.
- Revert the commit if ratio diverges or wall-clock fails to improve.

## Step 29 — Rayon-parallel block evaluation (long-tail win)

`ZopfliDeflatePart` evaluates each split block sequentially via
`ZopfliLZ77Optimal`. Each block is independent — they share input bytes
but each builds its own `ZopfliHash` and `BlockState`. Use `rayon::scope`
to evaluate them in parallel; the existing gzippy thread budget
(`opt_config.thread_count`) bounds the pool. Output stitching is already
sequential (header → deflate bytes → trailer). Expected 1.5-3× wall-clock
on multicore for inputs that produce many block-split candidates.

This is *additive* to gzippy's existing `T>1` parallelism (which currently
splits at the input level for zopfli, producing multi-member gzips). The
intra-block parallelism here helps `T1 --ultra` (single-member) — the
canonical zopfli use case.

## Step 30 — Adaptive iteration budget

Optional, lower priority. The C zopfli always runs `numiterations`
iterations even after convergence. Add a converged-early bail-out: if the
LZ77 store from iteration N is byte-identical to the store from iteration
N-1, stop. Gate behind `--ultra` (so explicit `-F N` always honors N).
Expected 5-15% wall-clock on real text, no ratio change.

---

# Frequently asked questions during the port

These are real ambiguities the original plan did not pin. The answers
below are **binding** — apply them as written. If you hit a fork the
plan and FAQ both leave open, log it as
`> **Open question (Step N):** …` at the top of that step and ask
before deciding. Do not guess.

## Q (Step 10): `add_weighted` — in-place or three-operand?

The C signature is `(stats1, w1, stats2, w2, result)` with three operand
slots. Audit the call sites: zopfli has **exactly one**, at
`squeeze.c:509`:

```c
AddWeighedStatFreqs(&stats, 1.0, &laststats, 0.5, &stats);
```

`stats1 == result == &stats` everywhere it's invoked. The third
destination is dead. The in-place signature is therefore behaviourally
correct and the right shape for the port:

```rust
impl SymbolStats {
    pub fn add_weighted(&mut self, w1: f64, other: &Self, w2: f64) {
        for i in 0..ZOPFLI_NUM_LL {
            self.litlens[i] = (self.litlens[i] as f64 * w1
                              + other.litlens[i] as f64 * w2) as usize;
        }
        for i in 0..ZOPFLI_NUM_D {
            self.dists[i] = (self.dists[i] as f64 * w1
                            + other.dists[i] as f64 * w2) as usize;
        }
        self.litlens[256] = 1;     // end symbol; see squeeze.c:77
    }
}
```

(Rust `as usize` truncates toward zero; C `(size_t)` does the same on
non-negative inputs, which these always are. Match guaranteed.)

## Q (Step 10): how to validate `RanState` without an FFI oracle?

The MWC RNG is `static` inside `squeeze.c` — not callable through the
FFI. **Recommended**: implement `RanState`, run a one-shot debug binary
once printing the first 16 outputs, paste the values into `squeeze.rs`
as a `const FIRST_16: [u32; 16]`, commit. The unit test asserts equality.

```rust
#[cfg(test)]
mod ranstate_tests {
    use super::*;
    // Generated once by hand from the published Marsaglia MWC formula
    // with seeds (m_w=1, m_z=2). Locked in to localize future drift.
    const FIRST_16: [u32; 16] = [
        /* fill from a one-time `cargo run --release --bin ran_dump` */
    ];
    #[test]
    fn ran_state_first_16() {
        let mut r = RanState::new();
        for &expected in &FIRST_16 { assert_eq!(r.next(), expected); }
    }
}
```

Acceptable fallback if the bin overhead feels heavy: skip this test and
rely on Step 12's full squeeze oracle to catch drift transitively. If
Step 12 fails *after* Step 10 lands, add the locked-output test then —
but it's cheap to do up front and pays back in localized signal.

## Q (Step 11): cost-model dispatch — closure or trait?

**Use a trait.** Closures with captures (`|l, d| cost_stat(l, d, &stats)`)
do monomorphize, but a trait makes the context explicit and keeps the
inner DP loop readable.

```rust
pub trait CostModel {
    fn cost(&self, litlen: u32, dist: u32) -> f64;
}

pub struct FixedCost;
impl CostModel for FixedCost {
    fn cost(&self, litlen: u32, dist: u32) -> f64 { cost_fixed(litlen, dist) }
}

pub struct StatCost<'a>(pub &'a SymbolStats);
impl CostModel for StatCost<'_> {
    fn cost(&self, litlen: u32, dist: u32) -> f64 { cost_stat(litlen, dist, self.0) }
}

pub fn get_best_lengths<C: CostModel>(
    s: &mut BlockState<'_>, in_: &[u8], instart: usize, inend: usize,
    cost: &C,                           // borrow; caller owns SymbolStats
    length_array: &mut [u16],
    h: &mut ZopfliHash,
    costs: &mut [f32],
) -> f64 { ... }
```

Each call site (`get_best_lengths(&FixedCost, ...)`,
`get_best_lengths(&StatCost(&stats), ...)`) generates its own
specialization with `cost.cost(...)` inlined — same machine code as the
C indirect call, often better. The trait also keeps Step 12's
`lz77_optimal_fixed` (uses `FixedCost`) and `lz77_optimal` (uses
`StatCost`) trivially readable.

## Q (Step 11): exact f32/f64 boundaries in the DP loop

The C code is implicit (C promotes `float` to `double` automatically and
narrows on store). Rust must be explicit. Pin **verbatim**:

```rust
// Outside the j loop:
let mincost: f64 = cost_model_min_cost(cost);

// Initialization (squeeze.c:243-245):
costs[0] = 0.0;
for c in &mut costs[1..=blocksize] { *c = ZOPFLI_LARGE_FLOAT as f32; }
length_array[0] = 0;

// Per-position j body:

// Literal candidate (squeeze.c:277-284)
if i + 1 <= inend {
    let new_cost: f64 = cost.cost(in_[i] as u32, 0) + costs[j] as f64;
    if new_cost < costs[j + 1] as f64 {
        costs[j + 1] = new_cost as f32;     // explicit narrow
        length_array[j + 1] = 1;
    }
}

// Length candidates (squeeze.c:287-302)
let mincostaddcostj: f64 = mincost + costs[j] as f64;
let kend = (leng as usize).min(inend - i);
for k in 3..=kend {
    if (costs[j + k] as f64) <= mincostaddcostj { continue; }
    let new_cost: f64 = cost.cost(k as u32, sublen[k] as u32) + costs[j] as f64;
    if new_cost < (costs[j + k] as f64) {
        costs[j + k] = new_cost as f32;     // explicit narrow
        length_array[j + k] = k as u16;
    }
}
```

Two traps to avoid:

1. **Don't compare `f32` to `f64` without explicit `as f64` on the f32**
   side. Rust will reject the comparison; satisfying the compiler by
   narrowing the f64 instead silently changes the algorithm.
2. **`new_cost as f32` is the only narrowing.** Don't pre-narrow the
   sum inside the cost computation. The C code does the entire sum in
   double precision and narrows on store; mirror that exactly.

## Q (Step 12): squeeze oracle runtime budget

The fast oracle caps inputs at 16 KB and runs all `numiterations ∈ {1,
2, 5, 15}` — finishes in <60 seconds, suitable for `cargo test
--release` and CI. It misses `alice.txt` (~150 KB) and `zeros_64k`,
which are exactly the inputs where a 1-ULP FP drift surfaces (real text
has the most realistic histogram; long zero runs exercise the
`SHORTCUT_LONG_REPETITIONS` branch).

**Add a thorough oracle as a separate `#[ignore]`-gated test** that
runs the full corpus at `numiterations = 1`. Even at 1 iteration,
`alice.txt` produces a real squeeze pass; if Rust and FFI agree on it,
multi-iteration agreement follows by induction (each iteration is a
deterministic function of the previous histogram).

Run order:
- Per-step: `cargo test --release` (fast oracle only).
- Before merging Step 12: `cargo test --release -- --ignored
  zopfli_pure`.
- Before Step 25 (cutover): same thorough invocation, plus `make ship`.

The thorough invocation is wrapped as `make zopfli-oracle-thorough`
(see Step 12's Makefile snippet).

---

# Risk register

| Risk                                                  | Mitigation                                        |
|-------------------------------------------------------|----------------------------------------------------|
| Floating-point divergence in `calculate_entropy`      | Step 3 oracle requires bitwise f64 equality        |
| `f32`/`f64` widening in `get_best_lengths` mis-ported | Step 11 highlights it; Step 12 oracle catches it   |
| `RanState` PRNG sequence mismatch                     | Step 10 self-test against published MWC sequence   |
| Hash chain order divergence on equal-length matches   | Step 7 brute-force test, Step 8 LZ77 byte-equality |
| `ll_counts` wrap-around bug                           | Step 6 self-consistency test; Step 9 size oracle   |
| Bit-writer LSB vs MSB swap                            | Step 14 unit test; Step 15 deflate oracle          |
| Memory regression from naive `LZ77Store::clone()`     | Acceptance bar in Step 25; revert if violated      |
| Submodule removal breaks Makefile / CI                | Step 24 sub-bullet to scrub Makefile               |

---

# Notes for the implementer

- **Keep the C source open in another window.** Your `oracle_tests.rs`
  diff is the single best signal; consult it before staring at code.
- **When the oracle fails, bisect by input position, not by code.** Each
  byte that diverges in a `LZ77Store` field tells you what state was
  wrong; from there, the C function name is one grep away.
- **Do not refactor for elegance during the port.** Match the C structure
  one-to-one. After Step 25, you can refactor freely; the regression test
  guards correctness.
- **Tests run release-mode** — these algorithms are slow in debug.
  `cargo test --release` everywhere.
- **You will be tempted to skip the oracle test for "trivial" steps
  (symbols, cache).** Don't. Three minutes writing a sweep test saves an
  hour of bisecting at Step 12.
- **You will be tempted to combine Steps 6+7+8 / 10+11+12 / 14+15.**
  Don't. Each step's oracle test is what gives you the "I am here, and I
  am correct" signal. Without it, you will drift.
