# Plan: gzippy zopfli — performance roadmap

> **Status (May 8 2026): the C → Rust port is complete and outperforms
> the reference C zopfli on the original acceptance bar.** This file is
> now a perf roadmap, not a porting manual. The original step-by-step
> port plan (Phases 1–10, 1881 lines) is preserved verbatim under
> "Appendix A — Original port plan" at the end of this file.

## Where we are

Head-to-head against vendor C zopfli on this box (Apple silicon,
release build, warm caches), measured 2026-05-08:

| Input | Tool | Wall-clock | Compressed bytes |
|-------|------|-----------:|----------------:|
| alice.txt (151 KB) | C zopfli (vendor) | 0.275 s | 50,706 |
| alice.txt (151 KB) | gzippy `--ultra -p1` | **0.205 s** (0.74× C) | 50,716¹ |
| text-1MB.txt | C zopfli (vendor) | 1.412 s | 383,918 |
| text-1MB.txt | gzippy `--ultra -p1` | **1.332 s** (0.94× C) | 383,931¹ |
| text-1MB.txt | gzippy `--ultra -p8` | **0.292 s** (0.21× C) | 392,847² |

¹ Deflate payload **bit-identical** to C zopfli. The byte-count delta
is exactly the FNAME field in gzippy's gzip header (`alice.txt\0` =
10 bytes; `text-1MB.txt\0` = 13 bytes). The gzippy header writer also
sets MTIME, XFL, OS=255 differently from C zopfli's `0,0,0,0,XFL=2,OS=3`.
This is intentional gzippy behavior to preserve user metadata, not a
port regression. To produce a strict byte-equal-to-C output, route
through `zopfli_pure::compress(..., ZopfliFormat::Gzip, ...)` directly
— which the regression fixtures in `tests.rs` do.

² Multi-member overhead (each thread emits an independent gzip member
with its own Huffman tree). Structural to gzippy's pre-existing
multi-thread zopfli scheme; not a port issue. See "Phase 14 (stretch)"
below for what closing this gap would take.

## Alignment to the prime directive

CLAUDE.md says **"gzippy aims to be the fastest gzip implementation
ever created."** For the L11 / `--ultra` path specifically:

- **T=1 wall-clock vs C reference:** 6–25% faster depending on input
  size. The original plan's acceptance bar was "≤ 1.1× C version
  wall-clock"; we delivered ≤ 0.94× across the test inputs.
- **T=1 ratio vs C reference:** bit-identical deflate payload.
- **T=8 wall-clock:** 4.8× faster than C, at a 2.3% ratio cost.
- **No C compiler on the build path:** `cargo clean && cargo build`
  invokes zero C for zopfli (`build.rs:1-29`).
- **Test surface:** 673 tests green. The pure-Rust zopfli ships with
  6 pinned hex regression fixtures + flate2 roundtrips (committed
  when the FFI oracle was still live; they are byte-equal to C zopfli
  output at the time of the cutover).

The prime directive is satisfied for L11. **There is still real
runtime on the table** — see "Open work" — but the headline goal is
delivered.

## Open work

Three categories. Items inside each are ordered by ROI.

### A. Ship what we have

- [ ] **`make ship`** — homelab L11 wall-clock gate. Plan rule 3
  ("benchmark everything") makes this the only AUTHORITATIVE signal,
  and rule 4 ("revert regressions") gives us the right to back out
  anything it flags. Has never been run on this branch. **Prerequisite
  for the PR.**
- [ ] **Open the PR** to `main`. Branch is `rust-zopfli`, PR is #83
  (currently the plan-only landing). After `make ship` passes, push
  the implementation commits and update the PR body with the head-to-head
  numbers above.
- [ ] **CI green**: pre-push hook runs `cargo fmt`, `cargo check`,
  `cargo clippy -D warnings`. Already passing locally; will run again
  on push.

### B. Close the T=1 hot-path gap (Phase 11)

The profiling note in the historical plan says, on a 1 MB `--ultra`
T=1 run, `deflate_part` time breaks down as **squeeze 64%, initial
split 22%, resplit 11%, emit 0.5%.** Squeeze dominates by ~6×, so
that's where the ROI is. Concrete plan in "Phase 11" below. Realistic
target: **0.7× C wall-clock on 1 MB T=1** (we're at 0.94× now).

### C. Stretch — close the T=8 ratio gap (Phase 14)

Today's `--ultra -p8` produces a multi-member gzip — each thread emits
an independent block with its own Huffman tree. That's where the 2.3%
ratio cost comes from. A single-member parallel zopfli (parallelize
the squeeze pass across input chunks while emitting one gzip member)
would close it. Big project; sketch in "Phase 14" below. Not blocking
the PR.

---

# Phase 11 — Hot-path squeeze opts (Step 28-redux)

> **Background:** plan Step 28 ("bench-driven hotspots") was deferred
> "pending flamegraph access" and the explicit fast-path attempts that
> followed (Step 27 batched bit writer; Step 29 parallel block eval;
> Step 28a parallel `find_minimum`) addressed every hot zone *except*
> the actual #1 hotspot — the `get_best_lengths` inner loop. The wins
> below stay inside that loop.

**Validation rule for every commit in this phase:** the regression
fixtures in `src/backends/zopfli_pure/tests.rs` must remain byte-equal,
and the head-to-head wall-clock numbers above must improve (or revert).
There is **no fixture regeneration** allowed in this phase — these
optimizations preserve the algorithm exactly; they only move work
between iterations of the hot loop. If a commit flips a hex blob,
revert it.

## 11.1 — Precompute `len_cost[k]` outside the DP loop

**Cost-stat inner loop today** (`squeeze.rs:170-180`):

```rust
pub fn cost_stat(litlen: u32, dist: u32, stats: &SymbolStats) -> f64 {
    if dist == 0 {
        stats.ll_symbols[litlen as usize]
    } else {
        let lsym  = length_symbol(litlen as i32) as usize;     // table lookup
        let lbits = length_extra_bits(litlen as i32) as f64;   // table lookup
        let dsym  = dist_symbol(dist as i32) as usize;         // CLZ + arith
        let dbits = dist_extra_bits(dist as i32) as f64;       // CLZ + arith
        lbits + dbits + stats.ll_symbols[lsym] + stats.d_symbols[dsym]
    }
}
```

Inside `get_best_lengths`, the inner `for k in 3..=kend` loop calls
this with `(litlen=k, dist=sublen[k])`. The `length_symbol(k)` and
`length_extra_bits(k)` table lookups depend **only on `k`**, never on
the cost model's state — so we can hoist them.

**Change:** add a private cache on `SymbolStats`:

```rust
pub struct SymbolStats {
    // existing fields ...
    /// `len_cost[k - 3]` = `ll_symbols[length_symbol(k)] + length_extra_bits(k)`
    /// for k in 3..=258. Refreshed in `calculate_statistics()` after
    /// `ll_symbols[]` is updated.
    pub len_cost: [f64; ZOPFLI_MAX_MATCH - ZOPFLI_MIN_MATCH + 1],
}

impl SymbolStats {
    pub fn calculate_statistics(&mut self) {
        calculate_entropy(&self.litlens, &mut self.ll_symbols);
        calculate_entropy(&self.dists, &mut self.d_symbols);
        for k in ZOPFLI_MIN_MATCH..=ZOPFLI_MAX_MATCH {
            let lsym  = length_symbol(k as i32) as usize;
            let lbits = length_extra_bits(k as i32) as f64;
            self.len_cost[k - ZOPFLI_MIN_MATCH] = lbits + self.ll_symbols[lsym];
        }
    }
}
```

Inner loop becomes `dbits + stats.d_symbols[dsym] + stats.len_cost[k - 3]`
— **one table lookup, two adds, no symbol arithmetic.** `cost_stat` and
`StatCost::cost` keep the same signature; no public API change.

`FixedCost` does not need this — `cost_fixed` is already cheap
(integer math, no table). Leave it alone.

**Expected impact:** -10% to -20% T=1 wall-clock. The `for k` loop runs
~256 times per `j` and `j` runs `blocksize` times; even saving 2
function calls per inner iteration is a lot of work.

**Risk:** none if `len_cost[]` is recomputed wherever `ll_symbols[]`
is overwritten. Audit: `calculate_statistics()` is the only writer of
`ll_symbols[]`, called from `from_store()` and from `lz77_optimal`'s
post-`add_weighted` block. The fixture tests will catch any miss.

## 11.2 — Specialize the literal-cost fast path

In the same loop, the literal candidate (`squeeze.rs:319-327`) goes
through the trait dispatch:

```rust
let new_cost: f64 = model.cost(in_[i] as u32, 0) + costs[j] as f64;
```

Under `StatCost`, this is just `stats.ll_symbols[in_[i] as usize]` —
one array load. The trait call inlines, but adding a dedicated
`fn literal_cost(&self, byte: u8) -> f64` to `CostModel` (default impl
forwards to `cost(byte as u32, 0)`) and overriding it in `StatCost`
removes the `if dist == 0` branch from the per-byte hot path.

**Expected impact:** -2% to -5% T=1.

## 11.3 — Skip `find_longest_match` when costs are already saturated

The DP loop already short-circuits on `costs[j+k] <= mincostaddcostj`
(line 333), but `find_longest_match` is called *before* that guard
even runs (line 307). On low-entropy inputs (long zero runs etc.) the
SHORTCUT_LONG_REPETITIONS branch handles it; on real text it runs
unconditionally.

**Don't optimize this without profiling first.** The `mincost`-based
guard saves cost-model calls but not match-finding work; structural
changes here risk diverging from the C algorithm. Pin this as
profile-driven only:

```bash
cargo install flamegraph
sudo cargo flamegraph --release --bin gzippy -- --ultra -p1 -c \
    test_data/text-1MB.txt > /dev/null
```

If the flamegraph shows `find_longest_match` is a meaningful fraction
of squeeze time, evaluate. Otherwise skip.

## 11.4 — `LZ77Store::store_lit_len_dist` histogram-wrap branch

`lz77.rs:67-113` does a `% ZOPFLI_NUM_LL == 0` test on every store.
Replacing it with a "next-wrap" boundary tracked as a field is purely
mechanical and shaves a divide. Likely <1% impact. Do **only** if 11.1
and 11.2 have shipped and there's still T=1 budget left.

---

# Phase 12 — Adaptive iteration budget (Step 30)

The C zopfli runs `numiterations` (default 15) iterations of the squeeze
loop unconditionally. After convergence — when iteration N produces a
byte-identical LZ77 store to iteration N-1 — every further iteration
is wasted CPU.

**Implementation:** in `lz77_optimal` (squeeze.rs:498), keep a hash of
the previous iteration's store (e.g. `(litlens.len(), crc32fast::hash(litlens
bytes), crc32fast::hash(dists bytes))` is sufficient for inequality
detection). Bail out of the iteration loop when consecutive hashes
match.

**Gate:** only when CLI did not pass `-F N` explicitly. The `--ultra`
default of 15 should adapt; user-specified `-F` honours the request.

**Expected impact:** 5-15% on real text where convergence happens
before iteration 15. Zero impact on inputs that need all 15 (pessimal
but already paying full price).

**Validation:** must keep the regression fixtures byte-identical. The
fixtures use `numiterations=15` and `numiterations=1`; convergence
detection cannot change the *winning* iteration's output, only when
the loop exits.

---

# Phase 13 — Ship and PR

1. Run `make ship` (the homelab gate). If wall-clock regresses on any
   benchmark vs the pre-port baseline (`main` HEAD), revert the
   offending commit per CLAUDE.md rule 4.
2. Push `rust-zopfli` to origin (it's already pushed; just refresh).
3. Update PR #83's body with the head-to-head numbers from
   "Where we are" above.
4. Add an "Acceptance" table to the PR body:

   | Bar | Result |
   |-----|--------|
   | `cargo build` invokes no C compiler | ✅ |
   | `cargo test --release` 100% green | ✅ (673/673) |
   | T=1 ratio bit-identical to C | ✅ (deflate payload) |
   | T=1 wall-clock ≤ 1.1× C | ✅ (0.94× on 1 MB; 0.74× on alice) |
   | Memory ≤ ±10% C | (pending `make ship`'s RSS measurement) |
   | All tuning flags work (`-F`, `-I`, `-J`) | ✅ |
   | `vendor/zopfli` not built | ✅ |

5. Request review.

---

# Phase 14 (stretch) — Single-member parallel zopfli

**Out of scope for the PR** but worth tracking. The 2.3% ratio cost
of `--ultra -p8` is structural: today, gzippy's parallel zopfli scheme
splits the input N ways and compresses each chunk as an independent
gzip member. Each member carries its own Huffman tree (~30-100 bytes
of overhead) and breaks cross-chunk back-references. That's where the
2.3% goes.

**What single-member parallel zopfli would look like:**

1. Split input into `T` chunks (with `ZOPFLI_WINDOW_SIZE = 32 KB`
   prefix overlap so back-references work across the seam).
2. Run `lz77_optimal` on each chunk **concurrently** — each thread has
   its own `BlockState`/`LongestMatchCache`/`ZopfliHash`. The greedy
   parse for block-split discovery already runs concurrently inside
   `deflate_part` (Step 29).
3. Concatenate the per-chunk LZ77 stores into one global store
   (existing `LZ77Store::append_from`).
4. Run `block_split_lz77` over the global store (already concurrency-
   parallelized via Step 28a-redux's `find_minimum`).
5. Emit one DEFLATE stream with one block per split-point — a single
   gzip member.

**Gotcha:** zopfli's block-split decisions affect the squeeze pass
indirectly (the cost model converges differently per chunk). The
current pipeline does block-split *after* squeeze; a clean
single-member impl might want to flip this — split first, then squeeze
each block. That's how the C reference's `ZopfliDeflatePart` already
works for input ≤ master block size. The structural change is
manageable; the engineering surface is wide.

**Estimated effort:** 1-2 weeks of focused work. Worth it only if the
T=8 ratio gap is explicitly identified as a customer pain point.

---

# Doctrine (still binding for any zopfli-pure edits)

These rules drove the port. They remain binding for any future surgery
on `src/backends/zopfli_pure/` — they're how we kept the cutover
risk-free and they're why the regression fixtures still hold years
into the future.

1. **Ratio is sacred.** `compress_gzip(data, &tuning)` output bytes
   under fixed options must remain stable. The 6 hex fixtures in
   `tests.rs` are the contract. Changing them requires a deliberate
   `GZIPPY_REGEN_FIXTURES=1` regeneration with PR-level review.
2. **Floating-point widths are load-bearing.** The squeeze cost array
   is `f32`; cost models return `f64`; comparisons promote f32→f64
   explicitly; stores narrow f64→f32 explicitly. See `squeeze.rs:323`
   and FAQ "Q (Step 11)" below.
3. **Never compromise ratio for clippy/style/readability.** The C
   port style — verbatim signed-int decrement loops, table-driven
   symbol math, `Box<[T]>` over slick arena types — exists because
   it's algorithmically correct. Refactor only after demonstrating
   no ratio drift on the full fixture set.
4. **Run `make` before committing, `make ship` before merging.**
   Local `make` (~30s) catches catastrophic regressions; `make ship`
   on the homelab is the authoritative wall-clock signal. The
   pre-push hook enforces `cargo fmt + check + clippy -D warnings`
   locally.
5. **One module per commit.** Cleaner bisect when a long-shot bug
   crosses module boundaries.
6. **When unsure, log the question.** Add
   `> **Open question (file/section):** …` as a blockquote in this
   plan or in a relevant module's doc-comment. Do not guess.

---

# FAQ (binding)

These were the strategic forks the original port surfaced; their
answers stayed binding through cutover and remain binding for any
future zopfli-pure work.

## Q: `SymbolStats::add_weighted` — in-place or three-operand?

**In-place.** The C signature has a third `result` slot, but
`squeeze.c:509` is the **only** call site and `result == stats1` there.
Current impl: `squeeze.rs:62-70`. Forces `litlens[256] = 1` (end
symbol) per C tail behavior. Phase 11.1 adds a `len_cost[]` rebuild
to `calculate_statistics()`; `add_weighted` does not need to
recompute `len_cost[]` itself because every call to it is followed
by a `calculate_statistics()` call (audit: `squeeze.rs:532-533`).

## Q: Cost-model dispatch — closure or trait?

**Trait.** `pub trait CostModel { fn cost(&self, litlen: u32, dist: u32) -> f64; }`
with `FixedCost` and `StatCost<'a>(&'a SymbolStats)` impls
(`squeeze.rs:185-203`). Each call site monomorphizes; the trait method
inlines under `#[inline]`. Same machine code as a direct call, more
self-documenting than a captured closure. Phase 11.2 will add an
optional `fn literal_cost(&self, byte: u8) -> f64` with default impl
forwarding to `self.cost(byte as u32, 0)`.

## Q: Exact `f32`/`f64` boundaries in the DP loop?

Pinned verbatim in `squeeze.rs:266-349`:

```rust
let mincost: f64 = cost_model_min_cost(model);
// ...
let mincostaddcostj: f64 = mincost + costs[j] as f64;   // f32 → f64
for k in ZOPFLI_MIN_MATCH..=kend {
    if (costs[j + k] as f64) <= mincostaddcostj { continue; }
    let new_cost: f64 = model.cost(...) + costs[j] as f64;
    if new_cost < costs[j + k] as f64 {
        costs[j + k] = new_cost as f32;                  // f64 → f32 (only narrow)
        // ...
    }
}
```

Two traps: (1) never compare f32 to f64 implicitly — the explicit `as
f64` on the f32 side is required; (2) the *only* narrowing is on the
store. Don't pre-narrow the sum. The C does the entire arithmetic in
double and narrows on store; we mirror that.

## Q: `RanState` Marsaglia MWC — how to validate?

Independent 64-bit reference in `squeeze.rs:587-613` recomputes the
MWC formula in full-width math then truncates; the unit test asserts
the first 64 outputs of `RanState::next()` match it. Different code
paths, same formula — divergence pinpoints which step.

## Q: Squeeze oracle runtime budget?

**Historical** (the FFI oracle is gone; its job is done). For any
*new* squeeze surgery: regenerate the regression fixtures only after
all six byte-equal vs the previous build, run `make ship`, then PR.
There is no thorough-vs-fast tier any more — the port is locked.

---

# Risk register

Updated for the post-cutover state. Original port risks are archived
in Appendix A.

| Risk | Mitigation |
|------|------------|
| Phase 11 changes drift the encoder output | 6 hex fixtures + flate2 roundtrip; bit-identical or revert |
| Phase 12 convergence detector fires too early | Hash-based comparison, not byte sampling; if iter N == iter N-1 byte-for-byte the rest of the loop *cannot* improve (deterministic) |
| `make ship` regresses despite local `make` passing | Revert the offending commit; CLAUDE.md rule 4 |
| Phase 14 single-member parallel zopfli changes block-split decisions | Pin a Phase-14-specific fixture set; fixture diffs reviewed by hand |
| arm64 vs x86_64 FP entropy diff (~2 ULP) | `tree.rs::tests` allows the diff explicitly; the squeeze layer absorbs it via fixed-point post-narrowing |
| Future contributor disables `vendor/zopfli` submodule and loses reference | Submodule kept registered but not built; checkout-on-demand via `git submodule update --init vendor/zopfli` |

---

# Notes for the next agent

- **Where the port started:** see git log `git log --grep='Step.*zopfli-pure'`
  for the 23-commit port arc. Each commit message explains its step's
  scope and validation. The original detailed plan (Phase 1–10) is in
  Appendix A below.
- **Where the port lives:** `src/backends/zopfli_pure/*.rs` (~4400 LOC,
  no FFI). Public surface: `compress(opts, ZopfliFormat, &[u8]) -> Vec<u8>`.
  Tuning bridge: `src/backends/zopfli_compress.rs` adapts the
  `ZopfliTuning` CLI struct to `ZopfliOptions`. CLI integration:
  `src/compress/mod.rs:38-53` (zopfli is the early-intercept path for
  level 11 / `-F` / `-I` / `-J`).
- **The C source is not built but is checked out.** `vendor/zopfli` is
  a registered submodule; `git submodule update --init vendor/zopfli`
  pulls the C source for side-by-side reading. The `vendor/zopfli/zopfli`
  binary can be built with `make -C vendor/zopfli zopfli` if you need
  head-to-head wall-clock numbers.
- **Pre-commit hook** runs `cargo fmt + check + clippy -D warnings`.
  Fix locally before committing.
- **Profiling on macOS:** `cargo install flamegraph` then `sudo cargo
  flamegraph --release --bin gzippy -- --ultra -p1 -c
  test_data/text-1MB.txt > /dev/null`. The dtrace-backed flamegraph
  is the right tool for Phase 11 hotspot validation.

---

# Appendix A — Original port plan (historical)

The original 30-step port plan that drove the work commits in
`b416bdb` (Step 10) through `4eac60f` (post-cutover cleanup) lived
here. It is preserved in the git history at commit `c7e0b41` (the
post-cutover plan revision). To read it:

```bash
git show c7e0b41:plan.md > /tmp/plan-original.md
$EDITOR /tmp/plan-original.md
```

Or browse directly on GitHub at the same SHA. The detailed step bodies
contain:

- Phase 1 (Step 0) — module scaffold and oracle harness
- Phase 2 (Steps 1–5) — leaf modules: symbols, katajainen, tree, hash, cache
- Phase 3 (Steps 6–8) — LZ77 store, longest-match finder, greedy parse
- Phase 4 (Step 9) — block-size estimation
- Phase 5 (Steps 10–12) — squeeze cost models, forward DP, multi-iteration
- Phase 6 (Step 13) — block splitter
- Phase 7 (Steps 14–15) — DEFLATE encoder
- Phase 8 (Steps 16–20) — gzip wrapper, public surface, FFI bridge
- Phase 9 (Steps 21–26) — cutover, FFI removal, Makefile cleanup, PR
- Phase 10 (Steps 27–30) — post-cutover optimization (Step 27 reverted;
  29 + 29b + 28a-redux landed; 28 deferred to Phase 11 here; 30
  deferred to Phase 12 here)

The "When you must adapt the plan" doctrine, the FAQ, and the
oracle-driven leaf-first methodology developed during the port are
worth re-reading before any surgery on `zopfli_pure`. They are
preserved in this file's "Doctrine" and "FAQ" sections above.
