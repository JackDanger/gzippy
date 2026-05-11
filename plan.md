# Plan: gzippy zopfli — correctness-first roadmap

> **Ratio is sacred.** The user's directive: *"we don't need to compress
> better, but we absolutely can't be worse"* than zopfli. This plan
> applies that as a hard guarantee: any path that produces more
> compressed bytes than C zopfli on the same input under the same tuning
> is a P0 regression. A corpus audit on 2026-05-08 found one such
> regression (`--ultra -pN > 1`) and several unvalidated paths. Phase
> 11 below is the unblock — perf work in Phase 12+ is gated on it.

## Status (corpus audit, 2026-05-08)

Head-to-head against the vendor C zopfli binary on this box (Apple
silicon, release build, warm caches), measured today:

| Input | Tool | Wall-clock | Compressed bytes | Δ vs C |
|-------|------|-----------:|-----------------:|--------|
| alice.txt (151 KB) | C zopfli | 0.275 s | 50,706 | — |
| alice.txt | gzippy `--ultra -p1` | **0.205 s** (0.74×) | 50,716 | +10 (FNAME hdr; deflate identical) |
| alice.txt | gzippy `--ultra -p8` | **0.060 s** (0.22×) | **51,752** | **+1,036 (+2.04% — P0)** |
| text-1MB.txt | C zopfli | 1.412 s | 383,918 | — |
| text-1MB.txt | gzippy `--ultra -p1` | **1.332 s** (0.94×) | 383,931 | +13 (FNAME hdr; deflate identical) |
| text-1MB.txt | gzippy `--ultra -p8` | **0.292 s** (0.21×) | **392,847** | **+8,929 (+2.33% — P0)** |

**Acceptance bar from the original plan was wall-clock-only; the new bar
is wall-clock AND ratio. Under the new bar we ship the T=1 path today;
the T>1 path violates the constraint.**

## Ratio audit results

### ✅ Confirmed bit-identical to C zopfli (deflate payload, T=1)

The 6-input corpus below produces **bit-identical DEFLATE bytes** to
the vendor C zopfli binary. (Wrapping gzip headers differ — gzippy
writes FNAME/MTIME/OS preserving caller metadata; C zopfli writes
zeroes. The compression itself is faithful.)

| Input | Size | C deflate | Rust deflate | Match |
|-------|-----:|----------:|-------------:|:-----:|
| `random_1k.bin` | 1,024 | 1,029 | 1,029 | ✅ |
| `random_64k.bin` | 65,536 | 65,542 | 65,542 | ✅ |
| `yesabc_8k.txt` | 8,192 | 29 | 29 | ✅ |
| `mixed_repeat.bin` | 900 | 19 | 19 | ✅ |
| `byte_runs.bin` | 8,192 | 429 | 429 | ✅ |
| `alice.txt` | 151,191 | 50,688 | 50,688 | ✅ |
| `big1.5M.txt` (master-block boundary) | 1,199,767 | 434,025 | 434,025 | ✅ |
| `alice.txt -F 1` | — | — | — | ✅ |
| `alice.txt -F 5` | — | — | — | ✅ |
| `alice.txt -F 30` | — | — | — | ✅ |

The 4 hex regression fixtures in `src/backends/zopfli_pure/fixtures/`
also re-verify byte-equal to the **current** C zopfli output (re-checked
2026-05-08 against `vendor/zopfli/zopfli` rebuilt from source).

All ratio-critical functions are present and pass through Step
12/13/15 byte-equality oracles at port time:
`optimize_huffman_for_rle`, `try_optimize_huffman_for_rle`,
`patch_distance_codes_for_buggy_decoders`, `lz77_optimal_fixed`,
`add_lz77_block_auto_type`, `calculate_block_size_auto_type`,
`encode_tree`. The `expensivefixed` re-squeeze branch in
`add_lz77_block_auto_type` (deflate.rs:381–398) matches C
deflate.c:760–800 line-for-line. `find_minimum`'s parallel branch is
function-deterministic (`f` is `EstimateCost` which is pure on `lz77`)
and has its own `find_minimum_serial_and_parallel_agree` test.

### ❌ Confirmed regression — `--ultra -pN > 1` on input > `block_size`

**`gzippy --ultra -pN > 1` produces multi-member gzip output, one
member per CPU thread, each with its own Huffman tree. The sum is
2.0–2.3% bigger than C zopfli's single-member output.**

Trigger: `compress/zopfli.rs:70`:
```rust
if self.thread_count == 1 || data.len() <= self.block_size {
    self.compress_single(data, writer)        // single-member, intra-block parallel
} else {
    self.compress_parallel(data, writer)      // ❌ multi-member: ratio loss
}
```

Default `block_size = 128 KB` (cli.rs:55). Any L11 input over 128 KB
with `-p > 1` falls off the cliff. Inputs ≤ 128 KB are unaffected
(they always go through `compress_single`).

Cause is structural to gzippy's pre-existing parallel zopfli scheme,
not the port. The squeeze itself is faithful — each per-thread block
*is* zopfli-optimal *for that block*. The loss comes from emitting
N independent Huffman trees instead of one optimal tree across all
chunks, plus the loss of cross-chunk back-references at member
boundaries.

Fix: see Phase 11.1 below. Immediate (option A) and long-term
(Phase 15) approaches are both staged.

### ⚠ Unvalidated paths (no continuous oracle)

These passed Step-time oracles when the FFI was live but have **no
continuous guard** today. Under the strict reading they are risks:

- **arm64 vs x86_64 entropy: 2-ULP `f64` drift in `calculate_entropy`.**
  Documented in `tree.rs::tests` with an explicit ULP allowance. Was
  claimed to be "absorbed by the squeeze layer's f64→f32 narrowing,"
  but this absorption was never measured against C output on arm64
  (the FFI oracle ran on x86_64 in CI). It is plausible — though not
  observed — that an arm64 build of gzippy produces deflate bytes
  that differ from the C zopfli binary built on the same arm64
  hardware. **Phase 11.2 closes this with a corpus oracle test that
  runs on every CI architecture.**
- **`--no-block-split` (`-I`) mode.** The C zopfli **CLI** doesn't
  expose `blocksplitting=0`; only the library API does. The Step 13
  port-time oracle linked the C library directly with
  `blocksplitting=0` and covered this path on the corpus, byte-equal.
  After Step 23 (oracle removed) there is no continuous guard. **Phase
  11.2's corpus oracle should call the C library, not the binary, so
  this path stays covered.**
- **`-J 0` (unlimited block split count).** Same status as `-I`.

These are gaps in *test coverage*, not known regressions. Phase 11.2
turns the gaps into guards.

---

# Phase 11 — Lock the ratio (must complete before any perf work)

Phase 12+ (perf) must not begin until 11.1 lands and 11.2 passes on
both architectures. Each commit in this phase has a hard validation
rule: the corpus oracle (11.2) must stay green and the 4 hex fixtures
must stay byte-equal.

## 11.1 — Eliminate the `--ultra -pN > 1` ratio gap

Two staged fixes. Land 11.1.A first (zero algorithmic risk), then
11.1.B (the proper structural answer) when capacity allows. **11.1.A
is the gate for the PR**; 11.1.B can land in a follow-up.

### 11.1.A — Force `--ultra` to single-member output (P0, this PR)

Make `compress_buffer` always route through `compress_single` when the
zopfli path is active. Loses input-level multi-thread scaling for L11
users; preserves bit-identical ratio to C zopfli. Intra-block
parallelism (Step 29's `std::thread::scope` in `deflate_part`) still
fires under `thread_budget = 0`, so users on multicore boxes still
get *some* concurrency — just bounded by the number of zopfli block
splits per master block (typically 5–15) instead of by `-pN`.

**Patch shape** (compress/zopfli.rs):

```rust
fn compress_buffer<W: Write + Send>(&self, data: &[u8], writer: W) -> io::Result<()> {
    if data.is_empty() {
        return self.write_empty_gzip(writer);
    }
    // Zopfli path is always single-member: emitting one gzip member per
    // CPU thread (the old `compress_parallel` path) costs +2.0–2.3%
    // compression vs C zopfli on input > block_size. Under our "ratio
    // is sacred" rule that's a P0 regression. Intra-block parallelism
    // inside `deflate_part` (gated on thread_budget=0) still uses cores.
    self.compress_single(data, writer)
}
```

`compress_parallel` becomes dead code; remove it once `compress_buffer`
no longer references it. Update `ZopfliGzEncoder::new`'s docstring to
say `thread_count` is advisory only on the L11 path.

**Wall-clock impact** on this box (measured for the plan):

| Workload | Before | After | Δ |
|----------|-------:|------:|---|
| 1 MB text `--ultra -p8` | 0.292 s | ~1.33 s | **+356% wall-clock** |
| alice.txt `--ultra -p8` | 0.060 s | ~0.21 s | **+250% wall-clock** |

Yes, that's a real wall-clock loss for L11 multi-thread users. It is
the *necessary* price under the ratio constraint until 11.1.B lands.
The user is opting into "near-zopfli" with `--ultra`; this realigns the
flag with what zopfli actually means.

**Documentation** (the only other change):

- `--help` text for `--ultra`: change "near-zopfli" → "true zopfli
  ratio (single-member; `-p` controls intra-block parallelism only)".
- README zopfli section: note that `--ultra -p` does not split
  the input. To get input-level parallelism at zopfli ratio, see
  the `--ultra-fast` flag (added in Phase 15) once it ships.

**Test additions:**

- A new test in `src/backends/zopfli_pure/tests.rs` (or
  `compress/tests.rs`) compresses `alice.txt` at `-p1` and `-p8`
  under `--ultra` and asserts `output == output_p1`. This pins the
  invariant: at L11, thread count must not change output bytes.
- The corpus oracle (11.2) catches the same invariant from the other
  side.

### 11.1.B — Single-member parallel zopfli (was Phase 14; now mandatory follow-up)

Recover the wall-clock that 11.1.A gives up, without sacrificing ratio.
Sketch:

1. Slice input into `T` chunks with `ZOPFLI_WINDOW_SIZE = 32 KB` prefix
   overlap so back-references work across the seam.
2. Run `lz77_optimal` on each chunk concurrently (each thread owns
   `BlockState` / `LongestMatchCache` / `ZopfliHash`). The greedy parse
   for block-split discovery already runs concurrently inside
   `deflate_part` (Step 29) — same machinery scales to per-input
   chunks.
3. Concatenate per-chunk LZ77 stores into one global store
   (`LZ77Store::append_from`).
4. Run `block_split_lz77` on the global store (parallelized via
   Step 28a-redux's `find_minimum`).
5. Emit one DEFLATE stream with one block per split point — single
   gzip member.

Validation: must emit deflate bytes that are bit-identical to
`--ultra -p1` on the same input across the entire corpus. Add this as
a fixture-style test under a new `--ultra-fast` flag (or whatever
name the merge ends up using); the flag is opt-in until parity is
proven across the corpus, including arm64.

**Estimated effort:** 1–2 weeks of focused work. Out of scope for the
unblocking PR; track as a follow-up issue.

## 11.2 — Continuous corpus oracle vs C zopfli

Build a permanent ratio guard. The Step 13/15 FFI oracle was deleted
post-cutover (Step 23) on the assumption that the 6 hex fixtures were
sufficient. They are not — they cover only 6 short inputs and would
not have caught the `-pN > 1` regression that 11.1 fixes today.

**New test** — gated `#[ignore]` so plain `cargo test` stays fast,
runs in CI on a dedicated job and on `make oracle-vs-c`:

```rust
// src/backends/zopfli_pure/tests.rs (or a new tests/oracle.rs)
#[test]
#[ignore = "requires vendor/zopfli/zopfli built; run via `make oracle-vs-c`"]
fn corpus_deflate_payloads_match_c_zopfli() {
    // 1. For each input in a fixed corpus (alice, random_1k,
    //    random_64k, yesabc_8k, mixed_repeat, byte_runs, big1.5M):
    // 2. Run `vendor/zopfli/zopfli --i15 -c <file>` capturing stdout.
    // 3. Run our `compress(opts, ZopfliFormat::Gzip, &input)` with
    //    matching options.
    // 4. Strip both outputs' gzip headers + trailers (header is
    //    variable-length: skip 10 bytes + optional FEXTRA, FNAME,
    //    FCOMMENT, FHCRC fields).
    // 5. Assert deflate payloads are bytewise equal.
    // 6. Repeat with options { numiterations ∈ [1, 5, 30],
    //    blocksplitting ∈ [0, 1] }.
}
```

Wire `vendor/zopfli` lazily: a `build.rs` step (or just a Make rule)
runs `make -C vendor/zopfli zopfli` only when the oracle test is
invoked. The submodule is already registered (kept post-cutover for
exactly this reason); the C build is one `make` away.

**For `blocksplitting=0` and `maxblocks=0` paths,** the C binary's
CLI doesn't expose them, so the oracle either:
- Builds a tiny C harness (`tests/oracle/zopfli_lib_oracle.c`) that
  links against `vendor/zopfli`'s `libzopfli.a` and exposes
  `compress_with_options(in, opts) -> bytes`. A `cc::Build` invocation
  in `build.rs` gated on a `oracle` cargo feature is sufficient.
- Or skips those tuning combinations from the binary-based oracle and
  pins them to known-good fixtures captured at port time.

**Recommended:** the harness path. The C is 4,800 lines and already
builds in 5 seconds; the oracle becomes a real continuous guard
without depending on whichever flags the binary happens to expose this
year.

**Add `Makefile` target:**

```make
oracle-vs-c: $(GZIPPY_BIN)
\t$(MAKE) -C vendor/zopfli zopfli
\tcargo test --release --features oracle -- --ignored corpus_deflate_payloads
```

Run on every CI architecture. Land before merging the PR — no perf
work in Phase 12 may begin until this is green on both arm64 and
x86_64.

## 11.3 — arm64 ULP audit

With 11.2 in place, the arm64 question becomes mechanical: **does the
corpus oracle pass on arm64?**

If yes: the 2-ULP entropy diff is genuinely absorbed by squeeze; close
the risk-register entry.

If no: identify the divergent input, bisect the squeeze iteration
where the path forks, and either (a) match C's exact rounding by
forcing `f64::ln` semantics across architectures (likely impossible —
ARM and x86 libm differ), or (b) replace `calculate_entropy`'s
`ln(x) * kInvLog2` with a fixed-point or table-driven equivalent that
is bit-identical across architectures. Option (b) is what C zopfli's
own ports to embedded targets use; it has prior art.

This is a research task, not a fixed-cost step. Open it as a separate
issue if 11.2 surfaces a real divergence.

---

# Phase 12 — Hot-path squeeze opts (was Phase 11 — ratio-gated)

Each commit in this phase **must** keep:
1. The 4 hex fixtures byte-equal (no `GZIPPY_REGEN_FIXTURES`).
2. The 11.2 corpus oracle green on both architectures.

If either fails, revert.

## 12.1 — Precompute `len_cost[k]` outside the DP loop

Today `cost_stat` (squeeze.rs:170) does 4 table lookups per
`(j, k)` inner-loop iteration. Two of them — `length_symbol(k)` and
`length_extra_bits(k)` — are pure functions of `k` and can be hoisted.

**Audit (already done for the plan):** `ll_symbols[]` is written in
exactly two places: `calculate_statistics()` (squeeze.rs:73, via
`calculate_entropy`) and `copy_from()` (squeeze.rs:49). Both must
also rebuild `len_cost[]`. `add_weighted` writes `litlens[]` only —
it's always followed by a `calculate_statistics()` call (line 533 /
538), so no extra wiring needed.

**Patch shape:**

```rust
pub struct SymbolStats {
    pub litlens:    [usize; ZOPFLI_NUM_LL],
    pub dists:      [usize; ZOPFLI_NUM_D],
    pub ll_symbols: [f64; ZOPFLI_NUM_LL],
    pub d_symbols:  [f64; ZOPFLI_NUM_D],
    /// `len_cost[k - ZOPFLI_MIN_MATCH]` = `ll_symbols[length_symbol(k)] +
    /// length_extra_bits(k)` for k in 3..=258. Refreshed by
    /// `calculate_statistics()` and `copy_from()`. **Must** be rebuilt
    /// wherever `ll_symbols[]` is written; the audit above is binding.
    pub len_cost: [f64; ZOPFLI_MAX_MATCH - ZOPFLI_MIN_MATCH + 1],
}

impl SymbolStats {
    pub fn calculate_statistics(&mut self) {
        calculate_entropy(&self.litlens, &mut self.ll_symbols);
        calculate_entropy(&self.dists, &mut self.d_symbols);
        self.rebuild_len_cost();
    }

    pub fn copy_from(&mut self, src: &Self) {
        self.litlens    = src.litlens;
        self.dists      = src.dists;
        self.ll_symbols = src.ll_symbols;
        self.d_symbols  = src.d_symbols;
        self.len_cost   = src.len_cost;       // mandatory; do not skip
    }

    fn rebuild_len_cost(&mut self) {
        for k in ZOPFLI_MIN_MATCH..=ZOPFLI_MAX_MATCH {
            let lsym  = length_symbol(k as i32) as usize;
            let lbits = length_extra_bits(k as i32) as f64;
            self.len_cost[k - ZOPFLI_MIN_MATCH] = lbits + self.ll_symbols[lsym];
        }
    }
}
```

`cost_stat` becomes:

```rust
pub fn cost_stat(litlen: u32, dist: u32, stats: &SymbolStats) -> f64 {
    if dist == 0 {
        stats.ll_symbols[litlen as usize]
    } else {
        let dsym  = dist_symbol(dist as i32) as usize;
        let dbits = dist_extra_bits(dist as i32) as f64;
        dbits + stats.d_symbols[dsym] + stats.len_cost[(litlen - ZOPFLI_MIN_MATCH as u32) as usize]
    }
}
```

The arithmetic is algebraically identical to today's expression. The
sum order changes (was `lbits + dbits + ll_symbols + d_symbols`; new
is `dbits + d_symbols + (lbits + ll_symbols)`). f64 addition is
commutative but not associative; reordering can shift by 1 ULP. **This
is the failure mode 11.2 exists to catch.** Land the change, run the
oracle, accept if green or revert if red. If red, force the original
sum order: `let len_part = stats.ll_symbols[lsym] + lbits; let
dist_part = stats.d_symbols[dsym] + dbits; len_part + dist_part` —
that's the C order and is what 11.1.B would expect anyway.

**FixedCost does not need len_cost** — `cost_fixed` is integer math.
Leave it alone.

**Expected impact:** -10% to -20% T=1 wall-clock. The inner loop runs
~256 times per `j` and `j` runs `blocksize` times.

## 12.2 — Specialize the literal-cost fast path

Inside the DP loop (`squeeze.rs:319-327`), the literal candidate calls
`model.cost(in_[i] as u32, 0)` which under `StatCost` is just
`stats.ll_symbols[in_[i] as usize]`. The trait dispatch inlines, but a
dedicated `literal_cost(byte)` method on `CostModel` (default impl
forwards to `cost(byte as u32, 0)`) lets the optimizer skip the
`if dist == 0` branch in the per-byte path.

**Ratio-safe** — algebraically identical. Land + oracle.

**Expected impact:** -2% to -5% T=1.

## 12.3 — Skip `find_longest_match` when costs already saturated (PROFILE-DRIVEN)

**Do not implement without flamegraph evidence.** Today
`find_longest_match` runs unconditionally before the
`mincostaddcostj` guard. Skipping it requires proving the optimum
never benefits from a match at saturated positions. The C reference
runs it unconditionally, so any algorithmic change here is a ratio
regression risk and **must** be oracle-validated against the full
corpus before merge. Flag as PROFILE-DRIVEN ONLY in commit messages
and require a flamegraph in the PR description.

## 12.4 — `LZ77Store::store_lit_len_dist` histogram-wrap branch

Sub-1% nicety; replace the `% ZOPFLI_NUM_LL == 0` check with a
"next-wrap" boundary tracked as a field. Algorithm-neutral.

---

# Phase 13 — Adaptive iteration budget

The C zopfli runs `numiterations` (default 15) unconditionally. After
convergence — when iteration N produces a byte-identical LZ77 store
to N-1 — every further iteration is wasted CPU. Detection:
hash the store between iterations, bail when stable.

**Ratio guarantee:** **does not change the *winning* iteration's
output**, only when the loop exits. The fixture tests use
`numiterations ∈ {15, 1}`; convergence detection cannot affect them.
Add an explicit fixture for an input that converges before iter 15
(verified that the bail fires, and the output is the same as today).

**Gate:** only when CLI did not pass `-F N` explicitly. The `--ultra`
default of 15 should adapt; user-specified `-F` honours the request.

**Expected impact:** 5–15% on real text where convergence happens
before iteration 15.

---

# Phase 14 — Ship and PR

1. Run `make ship` (homelab gate) **after Phase 11.1.A and 11.2 are in,
   before Phase 12 starts**. If wall-clock regresses on any benchmark
   vs the pre-port baseline (`main` HEAD), revert the offending commit
   per CLAUDE.md rule 4. Wall-clock loss on the L11 multi-thread path
   from 11.1.A is *expected* — annotate it explicitly in the PR body
   as a deliberate ratio-correctness trade.
2. Push `rust-zopfli`. Update PR #83's body with:
   - The Phase 11 corpus audit table.
   - The `--ultra -pN > 1` regression and 11.1.A's fix.
   - The 11.2 oracle + Makefile target.
   - The 11.1.B follow-up issue link.

3. Acceptance table for the PR body:

   | Bar | Result |
   |-----|--------|
   | `cargo build` invokes no C compiler for production binary | ✅ |
   | `cargo test --release` 100% green | ✅ (673/673) |
   | `make oracle-vs-c` green on x86_64 | ✅ (Linux + macOS x86_64 in CI matrix; commit 382656d) |
   | `make oracle-vs-c` green on arm64 | ✅ macOS arm64 local + Linux arm64 in CI; closes Phase 11.3 absent a CI surprise |
   | T=1 ratio bit-identical to C zopfli on full corpus | ✅ |
   | T>1 ratio bit-identical to C zopfli on full corpus | ✅ (Phase 11.1.A — `-pN > 1` now byte-equal to `-p1`, commit ef3b96f) |
   | T=1 wall-clock ≤ 1.1× C | ✅ (0.94× on 1 MB; 0.74× on alice) |
   | T>1 wall-clock ≤ 1.1× C | ⚠ regresses post-11.1.A (1× C); recovered by Phase 15 |
   | All tuning flags work (`-F`, `-I`, `-J`) | ✅ |
   | `vendor/zopfli` not built for production | ✅ (built only under `--features oracle`) |

---

# Phase 15 — Single-member parallel zopfli (was Phase 14 stretch; now mandatory follow-up)

This is Phase 11.1.B promoted to its own phase. Recovers the
wall-clock 11.1.A gives up, without ratio loss. Out of scope for the
unblocking PR; tracked as a separate issue.

See "11.1.B" above for the sketch. Validation: deflate bytes
bit-identical to `--ultra -p1` on the entire corpus across both
architectures, before the new flag flips on by default.

---

# Doctrine (still binding for any zopfli-pure edits)

1. **Ratio is sacred.** Any commit that changes a deflate byte vs the
   pre-commit C-zopfli output is a P0 regression. The 4 hex fixtures
   plus the Phase 11.2 corpus oracle are the contract.
2. **Floating-point widths are load-bearing.** The squeeze cost array
   is `f32`; cost models return `f64`; comparisons promote f32→f64
   explicitly; stores narrow f64→f32 explicitly. See `squeeze.rs:323`
   and the FAQ below. Any change to the f32/f64 ladder requires a
   corpus-oracle pass.
3. **f64 sum order matters.** Reordering an associative-looking sum can
   shift by 1 ULP and propagate into the squeeze decision, changing
   the output. When in doubt, preserve the C arithmetic order
   verbatim. The corpus oracle catches the failure but reverting is
   cheap; pre-oracle runs should mirror C's order by default.
4. **Never compromise ratio for clippy/style/readability.** The C
   port style — verbatim signed-int decrement loops, table-driven
   symbol math, `Box<[T]>` over slick arena types — exists because
   it's algorithmically correct.
5. **Run `make` before committing, `make oracle-vs-c` before merging
   any zopfli-pure change, `make ship` before flipping defaults.**
6. **One module per commit.** Cleaner bisect when long-shot bugs
   cross module boundaries.
7. **When unsure, log the question.** Add
   `> **Open question (file/section):** …` as a blockquote in this
   plan or in a relevant module's doc-comment. Do not guess.

---

# FAQ (binding)

## Q: `SymbolStats::add_weighted` — in-place or three-operand?

**In-place.** `squeeze.c:509` is the only call site and `result ==
stats1` there. Current impl: `squeeze.rs:62-70`. After Phase 12.1
lands, `add_weighted` does *not* need to rebuild `len_cost[]` itself
because every call to it is followed by a `calculate_statistics()`
call (audit: `squeeze.rs:532-533`).

## Q: Cost-model dispatch — closure or trait?

**Trait.** `CostModel` with `FixedCost` and `StatCost<'a>(&'a SymbolStats)`
impls (`squeeze.rs:185-203`). Each call site monomorphizes; the trait
method inlines under `#[inline]`. Phase 12.2 adds an optional
`literal_cost(&self, byte: u8) -> f64` with a default impl forwarding
to `self.cost(byte as u32, 0)`.

## Q: Exact `f32`/`f64` boundaries in the DP loop?

Pinned verbatim in `squeeze.rs:266-349`:

```rust
let mincost: f64 = cost_model_min_cost(model);
// ...
let mincostaddcostj: f64 = mincost + costs[j] as f64;     // f32 → f64
for k in ZOPFLI_MIN_MATCH..=kend {
    if (costs[j + k] as f64) <= mincostaddcostj { continue; }
    let new_cost: f64 = model.cost(...) + costs[j] as f64;
    if new_cost < costs[j + k] as f64 {
        costs[j + k] = new_cost as f32;                    // f64 → f32 (only narrow)
    }
}
```

Two traps: (1) never compare f32 to f64 implicitly — explicit `as f64`
on the f32 side; (2) the *only* narrowing is on the store. Don't
pre-narrow the sum. The C does the entire arithmetic in double and
narrows on store; we mirror that. **Sum order also matters** — see
Doctrine #3 and Phase 12.1's caveat about `cost_stat`'s reordering.

## Q: `RanState` Marsaglia MWC — how to validate?

Independent 64-bit reference in `squeeze.rs:587-613` recomputes the
MWC formula in full-width math then truncates; the unit test asserts
the first 64 outputs match.

## Q: `--ultra -pN > 1` ratio regression — why is it not just the gzip header?

The header explanation only accounts for the FNAME field (≤ 13 bytes
on the test corpus). The +1,036 byte alice and +8,929 byte 1 MB
deltas are the **deflate payload itself**: each per-thread block
emits its own dynamic Huffman tree (~30–100 bytes per tree) and
loses every back-reference that would have crossed a member boundary.
At default 128 KB block_size, alice (151 KB) is sliced into roughly 2
members; 1 MB into ~8. The cost is non-trivial and not header-related.
Phase 11.1 is the fix.

---

# Risk register

| Risk | Mitigation |
|------|------------|
| Phase 12 changes drift the encoder output | 4 hex fixtures + Phase 11.2 corpus oracle + bit-identical or revert |
| Phase 12.1 sum-order shift causes 1-ULP entropy drift | If 11.2 fails after 12.1, force C's sum order: `(ll_symbols + lbits) + (d_symbols + dbits)` |
| Phase 13 convergence detector fires too early | Hash-based comparison, not byte sampling; if iter N == iter N-1 byte-for-byte the rest of the loop *cannot* improve (deterministic) |
| `make ship` regresses despite local `make` passing | Revert; CLAUDE.md rule 4 |
| Phase 11.1.A wall-clock loss on T>1 sticks indefinitely | Phase 15 (single-member parallel zopfli) is the planned recovery; track as a follow-up issue |
| Phase 15 changes block-split decisions | Phase-15-specific fixture set; corpus oracle must remain bit-identical to `--ultra -p1` |
| arm64 vs x86_64 entropy 2-ULP drift | Phase 11.2 corpus oracle runs on both; Phase 11.3 documents the response if a real divergence surfaces |
| Future contributor disables `vendor/zopfli` submodule and oracle goes silent | `make oracle-vs-c` runs `git submodule update --init vendor/zopfli` first; CI fails-closed if oracle skips |

---

# Notes for the next agent

- **The corpus audit lives in commit `49e817e`'s test scratchpad** —
  the inputs are reproducible (deterministic LCG / `head -c`); see
  the bash blocks in this commit's PR comment thread for the exact
  recipes. Phase 11.2 codifies them.
- **Where the port lives:** `src/backends/zopfli_pure/*.rs` (~4400 LOC,
  no FFI). Public surface: `compress(opts, ZopfliFormat, &[u8]) -> Vec<u8>`.
  Tuning bridge: `src/backends/zopfli_compress.rs`. CLI integration:
  `src/compress/mod.rs:38-53`.
- **The C source is checked out but not built for production.**
  `vendor/zopfli` is registered; `git submodule update --init
  vendor/zopfli` pulls the C source. `make -C vendor/zopfli zopfli`
  builds the binary. **Phase 11.2 makes this build mandatory at oracle
  time** (and only at oracle time — production cargo build remains
  C-free).
- **Pre-commit hook** runs `cargo fmt + check + clippy -D warnings`.
- **Profiling on macOS for Phase 12.3:** `cargo install flamegraph;
  sudo cargo flamegraph --release --bin gzippy -- --ultra -p1 -c
  test_data/text-1MB.txt > /dev/null`. Required evidence in any 12.3
  PR.

---

# Appendix A — Original port plan (historical)

The 30-step C → Rust port plan that drove commits `b416bdb` (Step 10)
through `4eac60f` (post-cutover cleanup) lives at commit `c7e0b41`
in `plan.md`. Browse via:

```bash
git show c7e0b41:plan.md > /tmp/plan-original.md
```

Phases covered there:

- 1 (Step 0) — module scaffold and oracle harness
- 2 (1–5) — leaves: symbols, katajainen, tree, hash, cache
- 3 (6–8) — LZ77 store, longest-match, greedy
- 4 (9) — block-size estimation
- 5 (10–12) — squeeze cost models, forward DP, multi-iteration
- 6 (13) — block splitter
- 7 (14–15) — DEFLATE encoder
- 8 (16–20) — gzip wrapper, public surface, FFI bridge
- 9 (21–26) — cutover, FFI removal, Makefile cleanup
- 10 (27–30) — post-cutover optimization

The "When you must adapt the plan" doctrine, the FAQ, and the
oracle-driven leaf-first methodology developed during the port are
preserved in this file's "Doctrine" and "FAQ" sections above.

---

# Appendix B — `compress_parallel` is dead code post-11.1.A

After Phase 11.1.A lands, `ZopfliGzEncoder::compress_parallel`
(compress/zopfli.rs:103-123) has zero callers. Delete it in the same
commit. Its sibling helpers (`write_gzip_header`, `write_empty_gzip`)
stay.

The `block_size` field on `ZopfliGzEncoder` no longer affects routing
(was the trigger for multi-member splitting). It still serves as the
zopfli "master block" size hint, but `ZopfliDeflate` already slices
at `ZOPFLI_MASTER_BLOCK_SIZE = 1 MB` internally — the field is
effectively decorative on the L11 path. Document or remove.
