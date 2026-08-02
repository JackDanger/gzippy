# Where we actually stand on: arch-specific work, the level curve, cache/memory

The user's definition of done (2026-08-01): *"You're not done until you show me that
you've used every CPU architecture specific approach, that we found exactly the right
parameters to make this a smooth and awesome curve of compression levels, that we've
tuned this for cache levels and for memory pressure."*

**All three legs are measured below, and all three are open.** Nothing here is a wall
claim; all of it is deterministic (byte counts, peak RSS, static code audit).

---

## LEG 1 — CPU-architecture-specific approaches: the encoder has ONE, and it is a hint

Audited by grep across `src/`, then by tracing the call path of every hit.

| | files with any arch gate |
|---|---|
| `src/decompress/` | **18** |
| `src/compress/` | **3** |

And of those three encoder files, only one is on the shipping L0-L9 path:

| file | what it actually is | on the graded ladder? |
|---|---|---|
| `matchfinder/common.rs:144-160` | `_mm_prefetch` (x86) / `prfm` via `asm!` (aarch64) | **yes — and it is a PREFETCH HINT, not compute** |
| `parse/ultra/squeeze.rs` | real AVX + NEON kernels | **no** — `ultra` is reached only via explicit `--zopfli-*` flags (`src/compress/mod.rs:110-133`), never from `-1`..`-9` |
| `parse/mod.rs:1223,1241` | the strings "avx2"/"aarch64" **inside doc comments** | n/a — no code |

⚠ **CORRECTION — I first wrote "the encoder does zero SIMD compute" here. That is
FALSE, and I proved it false by disassembling the shipped binary.** The correct claim is
narrower: the encoder has almost no *explicit* arch-specific code. **Autovectorisation
supplies real SIMD anyway.**

**Open question O-8 — "we never measured whether rustc actually vectorizes [our
rebase]" — is now ANSWERED: YES.** The shipped binary contains **132 `smax.8h` + 132
`orr.8h`**, in exactly our idiom, 4-way unrolled at 32 `i16` lanes per iteration:

```
ldp   q1, q2, [x16, #-0x20]      ldp   q3, q4, [x16]
smax.8h v1,v1,v0   smax.8h v2,v2,v0   smax.8h v3,v3,v0   smax.8h v4,v4,v0
orr.8h  v1,#0x80,lsl #8   (x4)
stp   q1, q2, [x16, #-0x20]      stp   q3, q4, [x16], #0x40
```

`smax` against zero then `orr 0x8000` is precisely `(0x8000) | (v & !(v >> 15))`. So
**[G3] SIMD slide_hash is effectively ACHIEVED on aarch64 via autovectorisation** — we
match zlib-ng's technique without writing an intrinsic.

**A METHOD WARNING WORTH MORE THAN THE RESULT:** `cargo rustc --emit asm` produced a
`.s` with **zero** `smax.8h`, and I nearly reported "the shipped binary does not
vectorise". That `.s` is emitted **pre-LTO** and does not reflect the shipped artifact.
The release profile is `lto = "fat"`. **Disassemble the BINARY (`otool -tv`), never
trust `--emit asm`, when the claim is about what ships.**

The decoder — done and won — has 18 files of *explicit* SIMD: `asm_kernel.rs`,
`bmi2.rs`, `simd_copy.rs`, `simd_huffman.rs`, `lut_huffman.rs`, `crc32.rs`.

**So the real leg-1 question is sharper than "add SIMD":** which encoder hot loops
autovectorise and which do not? That is measurable the same way — disassemble and look.

### First case examined: the histogram — and it is ARCHITECTURE, not a missing intrinsic

igzip hand-wrote `isal_update_histogram` in assembly, which means they judged that
autovectorisation could not reach it. Ours (`parse/mod.rs:378-383`, `:404-413`):

```rust
*self.litlen_freqs.get_unchecked_mut(lit as usize) += 1;   // once per literal
```

**This is a scatter-increment FUSED INTO THE PARSE LOOP**, executed one literal at a
time as tokens are produced. igzip's is a **separate bulk pass over a buffer**. A fused
scatter cannot vectorise by construction: consecutive literals may target the same
bucket, so the increments carry a possible loop-carried dependency the compiler must
assume. No intrinsic fixes that — the loop shape forbids it.

The vendor technique is therefore not "SIMD the increment" but **"make the histogram a
separate bulk pass so it CAN be vectorised (typically with N partial histograms summed
at the end to break the dependency)"**. That is a real approach to steal, and it has a
real cost: a second pass over the literals, i.e. more memory traffic — which is
presumably why libdeflate does not do it either.

**Not yet measured, and it must be before this is attempted:** the histogram's share of
encoder time. The tree already has the instrument — `anatomy_count!(histogram_updates)`
and a `bucket-oracle-no-histogram` Cargo feature that ABLATES it. **Use the existing
ablation; do not hand-roll a substitute.** The share needs an Ir or wall run (trainer or
solvency), not this box.

The other two vendor edges autovectorisation cannot reach are `vpmaddwd` hashing [H4]
and CRC32-instruction hashing [H2]. **Both CHANGE THE HASH, hence which candidates are
found, hence output bytes** — so neither is output-neutral and both need the full
promotion gate, not just a wall number.

### What the vendors do that we do not

From `docs/vendor-technique-index.md` (94 techniques; these are the arch ones marked
`Ours: NO`):

* **[G7] whole-kernel assembly — igzip, on x86 AND aarch64.** `body`, `finish`,
  `icf_body`, `encode`, `map` are all asm (`vendor/isa-l/igzip/aarch64/`). Ours: two
  prefetch hints.
* **[G5] SIMD histogram — igzip** `isal_update_histogram` asm
  (`igzip_update_histogram.asm:257`). Ours: scalar frequency counting in the Sink.
* **[H4] SIMD multiply-add hash (`vpmaddwd`) — igzip** L3 `compute_hash_mad`, two
  rounds over 16 lanes (`igzip_gen_icf_map_lh1_06.asm:206-207`). Ours: no vectorised
  hashing at all.
* **[H2] CRC32-instruction hashing — igzip** `_mm_crc32_u32` on SSE4.2, `crc32cw` on
  aarch64. Ours: multiplicative only.
* **[G3] SIMD `slide_hash` — zlib-ng**, `_mm*_subs_epu16` over head+prev, many arches
  (`slide_hash_avx2.c:20-46`, `functable.c:252-415`). Ours: the libdeflate branchless
  scalar rebase — auto-vectorisable in principle, **never verified to actually
  vectorise** (open question O-8).

**Verdict: leg 1 is not partially done. It is not started.**

---

## LEG 2 — the level curve is NOT smooth: 18 of 22 files sag

`scripts/campaign/curve.sh`, main `120bfa9c`, T1, all 22 corpus files, levels 1-9. A
"sag" is `-N` producing a LARGER file than `-(N-1)`.

```
files with a sag : 18 of 22        clean : 4  (access.log, data.csv, ecoli.fastq, monorepo.tar)
total sag events : 26
by level         : L2=2  L3=1  L4=17  L5=1  L6=1  L7=2  L8=1  L9=1
```

Worst offenders:

| file | at | bytes worse | % |
|---|---|---|---|
| **data.sqlite** | **L4** | **+2,125,650** | **16.77%** |
| **data.sqlite** | **L2** | **+1,664,177** | **12.61%** |
| tool.bin | L4 | +229,186 | 1.08% |
| sil40 | L4 | +143,807 | 0.92% |
| data.parquet | L4 | +79,424 | 0.56% |
| dickens | L4 | +63,565 | 1.38% |

`data.sqlite` at `-2` is **12.6% worse than `-1`**, and at `-4` is **16.8% worse than
`-3`**. A user asking for more compression gets materially less.

**This is larger than the record says.** `level.rs:267-278` records the sag as "10 of 11
TUNE files at L4". On the full 22-file corpus it is **17 files at L4 plus 9 further sag
events at L2/L3/L5/L6/L7/L8/L9** that the TUNE-only view never saw. Per-label grading is
structurally blind to all of it — every one of these files can pass every cell while the
ladder is visibly broken.

*(Label check: the sag levels were computed independently twice — once in the shell
script's `SAGS AT` column, once in the Python analysis — and cross-checked equal. A
first pass had them all off by one.)*

---

## LEG 3 — cache and memory: O(input), and no cacheline work at all

Measured with `/usr/bin/time -l` peak RSS; full detail in `memory-is-o-n-not-o-1.md`.

```
90,868,376-byte input       -p1        -p4       -p16
  L1                       1.95x      2.44x      2.58x
  L6                       0.15x      2.44x      2.57x     <-- flat at T1, O(n) at T>1
```

* At **T1**, L1 is O(input) (1.95x) while L2-L9 are O(1) (a flat 13.7 MB).
* At **T>1, EVERY level is O(input)** — L6 is exactly as bad as L1. 90 MB in, 234 MB
  held at `-p16`.
* Mechanism: `read_to_end` the whole input plus a reserve of `input/2` = 1.5x, which the
  fitted input coefficient (1.498) corroborates.

### Cache-level tuning: the vendor precedent we have not taken

* **[C2] single-allocation state carving with cacheline layout — zlib-ng.** Window,
  prev, head, pending_buf and state are carved from ONE allocation with 64-byte-aligned
  sub-buffers (`deflate.c:165-227`), and the state struct is `ALIGNED_(64)` with fields
  grouped by cacheline (`deflate.h:138-314`). **Ours: PARTIALLY — per-object boxing with
  thread-local pooling, and NO deliberate cacheline grouping of hot loop state.** That is
  exactly the register/spill finding in `vendor-structure-comparison.md` §4.
* **[H5] hash-mask shrink for tiny inputs — igzip.** `if (hash_mask > 2*avail_in)
  hash_mask = (1 << bsr(avail_in)) - 1` (`igzip.c:1402-1403`): a 100-byte input touches a
  128-entry table, not 8K. **Ours: NO — full-size tables regardless of input size**, and
  the reset cost scales with table size (`parse/fast.rs:2776-2795`).
* **[C3] user-supplied graded working memory — igzip** (`level_buf`, MIN/SMALL/MEDIUM/
  LARGE/XL). **Ours: NO.** ⚠ A related record exists at `pipelined.rs:147` — scaling this
  *with LEVEL* buys size and LOSES wall. Scaling it with *input size or thread count* is
  a different axis and is not what that record measured.

---

## Honest summary

| leg | state |
|---|---|
| arch-specific approaches | **no DELIBERATE arch work** in the encoder (one prefetch hint) — but autovectorisation is real and verified: the rebase ships as 132 `smax.8h`+`orr.8h`. **O-8 answered YES.** The open question is now *which loops fail to vectorise* |
| smooth level curve | **broken** — 18/22 files sag, 26 events, worst is −16.8% |
| cache / memory tuning | **not started** — O(input) memory, no cacheline grouping, no table sizing |

None of these three is a cell on the board, which is why a board-driven campaign has
walked past all of them. They are also, jointly, the difference between "ties libdeflate"
and the stated destination: massively parallel, low memory, beating vendors on speed and
compression.
