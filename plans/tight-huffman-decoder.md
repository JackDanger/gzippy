# Tight Huffman decoder for `decode_huffman_body_resumable`

**Goal.** Implement every libdeflate/ISA-L technique inside the
resumable Huffman decode loop and get an Opus advisor to agree the
result is vendor-competitive.

**Anchor commit.** `8651f38` on branch `reimplement-isa-l`. The six
structural levers (B1–B6) from the prior seven-advisor-pass session
are in place. This plan layers vendor-competitive techniques on top.

**Authorization.** `CLAUDE.md` update 2026-05-27 explicitly authorizes
full re-implementation of the inner Huffman loop, including techniques
previously falsified at the structural-port stage. The
`feedback_no_innovation` memory is amended: architecture still ports
faithfully; inner loop is in scope.

---

## 1. Baseline (arm64, commit `8651f38`, post-harness-fix)

`cargo bench --no-default-features --features pure-rust-inflate --bench
tight_huffman_baseline` (best-of-30 after 5 warmup; decoder construction
hoisted out of the timed region per advisor item E1):

| case | raw | deflate | compress% | rust ns/B | flate2 ns/B | ratio |
|---|---:|---:|---:|---:|---:|---:|
| `text_words` | 512 KiB | 174 KiB | 33.1% | 1.06 | 1.40 | **0.76×** |
| `text_words_8MiB` | 8 MiB | 2.7 MiB | 33.0% | 1.04 | 1.34 | **0.78×** |
| `text_fixed_L1` | 512 KiB | 256 KiB | 48.9% | 1.33 | 1.55 | 0.86× |
| `mixed` | 512 KiB | 300 KiB | 57.3% | 2.73 | 2.51 | 1.09× |
| `mixed_8MiB` | 8 MiB | 4.8 MiB | 57.2% | 2.75 | 2.56 | 1.08× |
| `repetitive` (match-dominated) | 512 KiB | 3.4 KiB | 0.7% | 0.05 | 0.05 | 1.13× |
| `random` (stored-dominated) | 512 KiB | 553 KiB | 105.5% | 2.76 | 2.90 | 0.95× |

**What this says.** Post B1-B6, on arm64:
- Pure-Rust is ~1.3× faster than flate2/zlib-ng on text-heavy workloads.
- 8MiB variants show no cache-miss falloff (decode is not memory-bound).
- `mixed` is slightly slower than flate2 — the gap closes if T3
  multi-literal pays back.
- `text_fixed_L1` shows fixed-Huffman blocks decode 14% faster than
  flate2, but the specialization (T2) hasn't been applied yet.

**Initial harness was buggy.** Before the advisor pre-flight critique
fix, `time_flate2` was constructing the decoder INSIDE the timed region
while `time_resumable` constructed outside. That made `text_words` look
like 0.44× when the honest number is 0.76×. The previous 524 KiB output
buffer (`deflate.len() * 32` = 16 MiB) was page-faulting inside the
timed region for `random`. Both fixed.

**What this does NOT say.** Nothing about x86_64 ISA-L parity. ISA-L on
neurotic is ~1.5-2× faster than zlib-ng on text; matching that needs
`text_words` ratio ≤ 0.5× — about 1.5× faster than today. The neurotic
bench needs the user's homelab. The arm64 number is the *floor* — we
must non-regress here.

**Implication for plan.** x86_64 techniques (T1 BMI2 dispatch, T3
packed writes via x86 intrinsics) land behind `cfg(target_arch =
"x86_64")` + non-regression on arm64. Their wins are claimed only after
the neurotic A/B confirms.

## 2. Inventory of techniques to land

Reordered after advisor critique (T4 before T3; new T0; sequencing
adjusted so each lever lands in a state where the next can pay back).
Each has a vendor counterpart cited; each ships with an A/B
microbench run.

### T0 — Raw-pointer literal writes

**Current.** Literal emit at `resumable.rs:993, :1038` writes via
slice index: `output[out_pos] = entry.literal_value()`. Even with
`#[inline(always)]`, Rust emits an `unwrap_or(panic)` bounds check
on every literal byte unless LLVM proves it elides — and on a
hot loop with a yielding control flow, LLVM cannot always prove it.

**Change.** Lift `out_ptr = output.as_mut_ptr()` and `out_end =
output.as_mut_ptr().add(output.len())` to function locals (similar to
B1's `in_data_ptr`). Replace literal writes with `unsafe {
out_ptr.add(out_pos).write(entry.literal_value()); }`. The yield
check at top of loop still bounds the writes safely.

**Vendor counterpart.** `consume_first_decode.rs:738-739` lifts
`out_ptr` and `out_end` to function locals; all literal writes use
raw pointer arithmetic (`:1021-1024, :1057-1058`).

**Risk.** Mishandled bounds → UB. Mitigate: keep the `if out_pos >=
output.len()` yield check at top of loop unchanged; the unsafe writes
land between yield checks where `out_pos < output.len()` is invariant.

**Acceptance.** All 635 tests still green. `text_words` ns/B improves
≥ 5% on arm64; `random` (no literal writes) unchanged within noise.

**Why T0 first.** T3's multi-literal packed write
(`write_unaligned::<u32>` writing 4 bytes at once) cannot beat the
per-byte path until the per-byte path itself uses raw pointers —
otherwise the comparison is "raw pointer × 4" vs "bounds-checked × 4"
and we measure the wrong thing.

### T4 — Yield-elide FASTLOOP when both buffers have margin

**Current.** Loop top checks (a) `out_pos >= output.len()` and (b)
`bit_position_of(bitsleft, in_pos) >= encoded_until_bits` every
iteration. (b) is a multi-op compute (`in_pos*8 − unread`); (a) is
single instruction but near-perfectly predicted (not-taken until end).

**Change.** When `output.len() - out_pos >= FASTLOOP_MARGIN = 320` AND
`in_data_len - in_pos >= 32`, enter a tighter inner loop that omits
both checks. Re-enter the safe loop at boundary. Vendor pattern:
`consume_first_decode.rs:752` (`in_fastloop_end = in_data.len() -
32`), `:858-859` (`while in_pos < in_fastloop_end && out_pos +
FASTLOOP_MARGIN <= out_end`).

**Resumable contract.** The fastloop is only entered when both buffers
have margin. The yield (`output filled`) physically cannot fire inside
the fastloop; the `encoded_until_bits` cap is converted to
`in_fastloop_end` so the loop also can't read past the cap. On exit
from the fastloop, fall through to the safe loop which handles the
tail symbols.

**Vendor counterpart.** `consume_first_decode.rs:858-1147` IS the
fastloop. libdeflate uses identical pattern.

**Why T4 BEFORE T3.** Per advisor item D3: multi-literal lives INSIDE
the fastloop because the per-iteration overhead T3 amortizes is
exactly what T4 removes. Implementing T3 in the safe loop would
measure smaller than its true potential and be revert-prone.

**Acceptance.** `text_words` ns/B improves ≥ 5%; `text_words_8MiB`
also improves (rules out small-input artifact); `repetitive`
unchanged within noise.

### T3 — Multi-literal lookahead (up to 8 literals per iteration)

**Current.** One literal per loop iteration. With B2's PRELOAD
pattern, `entry` is preloaded at end of each branch.

**Change (vendor `consume_first_decode.rs:870-1030`).** Inside the
T4 fastloop, after the first literal classify, check the preloaded
next entry. If it's also a literal AND `bitsleft >= 2 *
LitLenTable::TABLE_BITS`, emit both with one packed
`write_unaligned::<u16>`. Lookup third; if literal AND `bitsleft >=
36`, write packed `u32` (3 bytes) and continue. Up to 4 packed in u32.
Vendor goes to 8 — extend to 8 per advisor item C1.

**Bit budget (advisor item B2):** post-refill bitsleft ∈ [56, 63].
After 4 literals: bitsleft ∈ [8, 15] — must refill before next.
After 2 literals: bitsleft ∈ [32, 39] — safe for next lookup
without refill. The check is `bitsleft >= remaining_literals_in_batch
* TABLE_BITS`, not a fixed threshold.

**Falsified previously?** Yes — commit `ca52389` regressed -15%.
NON-binding now: that measurement was pre-B1 (no register-local lift)
and pre-B2 (no PRELOAD). The dep chain causing the regression was:
literal classify → next lookup → next classify (sequential). With
PRELOAD + register locals, the next lookup overlaps with the byte
emit. Per CLAUDE.md update, re-attempt with fresh measurement.

**Vendor counterpart.** `consume_first_decode.rs:870-1030`. Vendor
unconditionally takes this path inside the fastloop.

**Acceptance.** `text_words` improves ≥ 15% on arm64
post-T4. `text_words_8MiB` shows the same delta (rules out artifact).
If it regresses, revert and document the new falsification with
post-T0+T4+T1 dep-chain analysis.

### T1 — BMI2 dispatch hoisted out of the inner loop

**Current.** `bmi2::decode_extra_bits` uses compile-time
`#[cfg(target_feature = "bmi2")]`. Stock release builds don't set
the flag → BZHI path is dead.

**Change (advisor item F1: hoist out of loop).** Function-pointer
selection at the TOP of `decode_huffman_body_resumable` based on
`is_x86_feature_detected!("bmi2")`. The body then calls through the
function pointer (or via a generic over a trait), which the compiler
inlines into two variants of the body. NOT per-call dispatch — that
would cost more than BZHI saves (advisor item B1).

Two viable implementations:
- **Const-generic variant**: `fn decode_huffman_body_resumable<const
  HAS_BMI2: bool>(...)`. Top-level callers pick the variant. Compiler
  monomorphises into two bodies; LLVM inlines BZHI calls.
- **Indirect call hoisted to top**: select `decode_extra_bits` impl
  via fn pointer at top of body, pass into the inner loop. One indirect
  call per loop entry, not per-symbol.

Pick const-generic (no per-call indirection at all).

**Vendor counterpart (CORRECTED per advisor item A1).** The IN-TREE
port at `consume_first_decode.rs:168-182` uses compile-time
`target_feature` gating, SAME as gzippy today. UPSTREAM libdeflate
uses multi-version dispatch (different code path entirely). T1's
authority is "match upstream's dispatch shape because compile-time
gating leaves the fast path dead on portable binaries."

**Acceptance (advisor item F7).** On arm64: no codegen change (BMI2
path unreachable). On x86_64-BMI2: BZHI emitted in the BMI2 variant.
**Final accept gate is neurotic A/B**, not local — arm64 cannot
measure x86_64 wins.

### T2 — Fixed-Huffman static-table specialization

**Current.** `resume_decode_fixed_resumable` reuses
`decode_huffman_body_resumable` with the static fixed-Huffman tables.
The dispatch costs nothing, but the body decoder treats fixed and
dynamic identically — full subtable branch, full PRELOAD pattern.

**Observation (advisor item A3: requires assertion, not assertion).**
Fixed-Huffman tables per RFC 1951 §3.2.6: litlen codes are 7, 8, or 9
bits; dist codes are 5 bits. **None exceeds `LitLenTable::TABLE_BITS
= 12` or `DistTable::TABLE_BITS = 9`.** Therefore no subtable entry
SHOULD be produced. But the `LitLenTable` builder might still emit
subtable pointers for technical reasons (e.g. padding) — must be
verified, not asserted.

**Step 0 of T2 (new, per advisor item D2).** Add a unit test
`fixed_huffman_table_has_no_subtable_entries` that builds the static
fixed-Huffman tables via `libdeflate_decode::get_fixed_tables()` and
asserts every entry of the main table has `!is_subtable_ptr()`. If
the assertion FAILS, T2 cannot proceed as written — must fall back
to a different specialization (e.g. an inline subtable check in the
fixed body that errors loudly because it can't hit).

**Step 0.5 of T2.** Add `text_fixed_L1` to the bench (already added
in the harness update above) so T2's acceptance has a measurement.

**Change.** New function `decode_huffman_body_fixed_resumable` that
inlines consume + classify with no `is_subtable_ptr()` check.
Smaller code → better I-cache; same LUT (advisor item A2 corrects
the earlier "smaller LUT" claim — the static table is the same
type; the win is cache locality + I-cache reduction, not D-cache).

**Vendor counterpart.** libdeflate has separate `decompress_template`
instantiations for static / dynamic. ISA-L's
`igzip_decode_huffman_codes` switches based on block type.

**Acceptance.** `text_fixed_L1` ns/B improves ≥ 10%; `text_words`
(L6 dynamic) unchanged.

### T3 — Multi-literal lookahead (2-/3-literal packed write)

**Current.** One literal per loop iteration. With the B2 PRELOAD
pattern in place, `entry` is preloaded at the end of each branch.

**Change (vendor `consume_first_decode.rs:870-1030`).** After the first
literal classify, check the preloaded next entry too. If it's also a
literal AND `bitsleft >= 2 * TABLE_BITS`, emit both with one packed
`write_unaligned::<u16>`. Then preload the third. If that's a literal
too, emit four with `write_unaligned::<u32>`. Cap at 4 (vendor goes to
8; the marginal value drops fast).

**Falsified previously?** Yes — commit `ca52389` regressed -15%. But
that was pre-B1 (no register-local lift) and pre-B2 (no PRELOAD).
The dep chain that caused the regression was: literal classify → next
lookup → next classify. With PRELOAD, the next lookup is already in
flight when the first literal is emitted, so the OoO engine has more
work to overlap.

**Vendor counterpart.** Vendor unconditionally takes this path. The
condition `bitsleft >= REFILL_THRESHOLD` at top-of-loop gates the
batched emit.

**Acceptance.** `text_words` ns/B improves ≥ 10% on arm64. If it
regresses, revert and document the new falsification with the
post-B1+B2+B3 dep-chain analysis.

### T4 — Yield-elide FASTLOOP when output has margin

**Current.** Loop top checks `out_pos >= output.len()` and
`bit_position >= encoded_until_bits` every iteration. Per-iteration
overhead: two conditional jumps.

**Change.** When `output.len() - out_pos >= FASTLOOP_MARGIN = 320` AND
`in_data_len - in_pos >= 32`, enter a tighter inner loop that omits the
two checks. Re-enter the safe loop at the boundary. Vendor pattern:
`consume_first_decode.rs:858-859` (`while in_pos < in_fastloop_end &&
out_pos + FASTLOOP_MARGIN <= out_end`).

**Resumable contract.** The fastloop is only entered when both buffers
have margin. The yield (`output filled`) physically cannot fire inside
the fastloop, so removing the check is safe. On exit from the
fastloop, fall through to the safe loop which handles the tail.

**Vendor counterpart.** `consume_first_decode.rs:858-1147` IS the
fastloop. libdeflate calls this pattern FASTLOOP.

**Acceptance.** `text_words` ns/B improves ≥ 5%; `repetitive` (where
the fastloop fires for most of the decode) improves ≥ 10%; small
fixtures don't regress.

### T5 — Table prefetch ahead of dependent load

**Current.** Each iteration's first action is `entry =
litlen.lookup(bitbuf)` — an L1 load on the table.

**Change.** At the END of each iteration's branch (where we currently
preload the next entry), ALSO issue `_mm_prefetch::<_MM_HINT_T0>` on
the +1 entry past the just-loaded one. Vendor pattern is
`__builtin_prefetch` on the next dist-table entry while the litlen
decode is in flight.

**Acceptance.** Measurable on the longest text fixture (where the
working set exceeds L1). May not help on small inputs.

### T6 — Inline subtable resolve for litlen (post-T2)

**Current (post-B3).** Subtable resolve is in the EXCEPTIONAL branch
with re-classification of literal / EOB / length. Vendor does the same
but with the dist lookup already speculatively in flight.

**Change.** In the EXCEPTIONAL-subtable-LENGTH fall-through, hoist
`dist.lookup(bitbuf)` BEFORE the length-extra extraction so the dist
table load overlaps with the BMI2 BZHI. Tiny change; vendor pattern at
`consume_first_decode.rs:1075-1076`.

### T7 — Specialized fast-distance path

**Current.** `decode_distance` always uses BMI2 BZHI for extra-bits.
For distance codes < 4 (dist 1-4, no extra bits), the BZHI is wasted —
distance is just `dist_entry.distance_base()`.

**Change.** Branch on `extra_bits == 0` at the top of decode_distance.
The most common short matches (RLE-like, dist=1) take the no-extra
path. Already partially handled by `decode_extra_bits` returning 0
when extra_bits=0, but the BZHI instruction still runs.

**Vendor counterpart.** None — vendor unconditionally runs the
extract. This is a minor win at best.

**Status.** Probably skip; document as falsified-by-vendor-pattern if
arm64 bench shows no improvement.

## 3. Process

For each technique T1..T6:

1. Implement (branch off `reimplement-isa-l` or stay on same branch).
2. Run `cargo test --release --no-default-features --features
   pure-rust-inflate --lib` to confirm correctness (635 lib tests + 27
   resumable + 4 bmi2 + corpus differential).
3. Run `cargo bench --no-default-features --features
   pure-rust-inflate --bench tight_huffman_baseline` — capture
   text_words, mixed, repetitive, random ns/B.
4. Compare to the previous landing's numbers. Wins land; regressions
   revert with documented reason.
5. After each landing, re-run the Opus advisor for sign-off review.

## 4. Done-when

User's controlling goal (set 2026-05-27): *"get an Opus advisor to
agree we've implemented a vendor-competitive tight Huffman decoder in
Rust with all optimizations."*

Done when:

1. T1–T6 each landed or formally documented as falsified (commit
   `cfg(off)` + comment with measured A/B).
2. `tight_huffman_baseline` shows `text_words` ratio ≤ 0.6× (i.e.
   pure-Rust ≥ 1.66× faster than flate2/zlib-ng on arm64).
3. All 635+ tests still green.
4. Opus advisor concedes — same loop as the prior seven-pass session:
   refuse → name lever → land → re-ask, until concede.

## 5. Risks

1. **PRELOAD-changes-dep-chain hypothesis for T3 multi-literal might
   not hold.** If it still regresses, falsify with the post-B1+B2 dep
   analysis and move on.
2. **T4 FASTLOOP could break resumable contract** if the entry/exit
   bookkeeping for the inner loop is wrong. Risk: silent over-decode
   past `encoded_until_bits`. Mitigation: the corpus differential
   harness (D1) catches this.
3. **T5 prefetch could regress on small inputs** by polluting cache.
   Mitigation: gate on input-size threshold.
4. **arm64 baseline may not predict x86_64 behavior.** Mitigation:
   structural arguments + non-regression on arm64 + final neurotic bench
   before claiming production-ready.
