# JOB-2 Advisor Gate — `writable_tail_reserve` under-sizing fix (commit b50c0d23)

Read-only Opus disproof gate. Owner could not run its own advisor (Agent tool
absent); this is the owed gate. I read the diff and all cited source first-hand;
I did NOT re-run the ISA-L suite (read-only, no Rosetta/x86 FFI in this env), so
measured A/B and test-count claims are verified at the LOGIC level, not re-executed.

## Scope of the actual change (verified)
- Production change is exactly ONE line: `segmented_buffer.rs:247`
  `self.buf.reserve(min_spare - (self.buf.capacity()-len))` →
  `self.buf.reserve(min_spare)` (+ a 10-line explanatory comment).
- `gzip_chunk.rs` diff is a SINGLE hunk at line 2639, entirely inside
  `mod isal_tail_parity` (the `synthetic_dynamic_gz` corpus-absent fallback) —
  TEST-ONLY, no production logic touched.
- `isal_stored_fixed_probe.rs` is a new test file, every fn `#[test]` /
  `#[cfg(all(parallel_sm, target_arch="x86_64"))]`.
- `debug_assert!(spare >= min_spare)` at :250 is UNCHANGED — the fix makes the
  assert PASS, it does not delete the safety net.

## CLAIM 1 — BYTE-TRANSPARENT → SOUND
`reserve` mutates Vec CAPACITY only, never `len` or contents. The committed
output in the production caller (`gzip_chunk.rs:274`) is bounded by
`keep_len = min(keep_len, written)` (`gzip_chunk.rs:335`) and `commit(keep_len)`
(`:340`); `written` is what ISA-L actually wrote into the (now correctly-sized)
slice. No uninitialized/poison byte is ever committed: `poison_reserved_tail`
paints `[len, len+spare)`, but only `[len, len+keep_len)` (all ISA-L-written) is
committed and CRC'd (`decoded_range`, `:346`). Downstream length accounting
(`decoded_size`, writev iovecs, window-publish, CRC) all key off committed `len`,
which is unchanged. The slice pointer is fetched AFTER the `reserve`
(`segmented_buffer.rs:253-254`) and the caller re-fetches `out` each call
(`:274`), so the (possibly NEW) realloc-move introduces no stale-pointer hazard —
consistent with the `contig_decode_window` H4 invalidation contract.
For the off-by-default `fold_nodrain` caller (`:1017`) the bytes are already
deliberately WRONG ("never a production path", `:1015`); there the fix actually
makes `commit(n)` SAFER (old under-reserve could push `len > capacity`, the very
thing `commit`'s `debug_assert` at :313 guards). Either way: output bytes
identical. SOUND.

## CLAIM 2 — NATIVE UNAFFECTED → SOUND
Production caller `:274` lives in a fn gated
`#[cfg(all(parallel_sm, feature="isal-compression", target_arch="x86_64"))]`
(`gzip_chunk.rs:205`). The native build compiles the STUB at `:391`
(`#[cfg(not(all(...isal-compression...)))]`) which `Ok(false)` and never calls
`writable_tail_reserve`. The only other caller, `:1017`, sits behind the runtime
`if fold_nodrain_enabled()` (env `GZIPPY_FOLD_NODRAIN=1`, off by default,
`:182-185`); native production takes the `else` `extend_from_slice` branch
(`:1020`). ⇒ gzippy-native NEVER reaches `writable_tail_reserve` and is
byte-unchanged. `:274` cannot compile or run on native. SOUND.

## CLAIM 3 — THE BUG REALITY → SOUND (with a flagged narrative imprecision)
The bug is REAL and I derived its window independently. `ensure_buf`
(`segmented_buffer.rs:104-109`) is a NO-OP whenever `capacity != 0`, so on a
REUSED buffer it does not guarantee capacity and the `reserve` branch is live.
Old code: inside the guard `spare < min_spare`, it requested
`additional = min_spare - spare`. `Vec::reserve` no-ops when `capacity (= len+spare)
>= len + additional`, i.e. when `2*spare >= min_spare`. So for any
`min_spare/2 <= spare < min_spare` the reserve no-ops and the post-condition
`spare >= min_spare` FAILS → debug_assert fires (debug/test: worker panic → the
chunk result is never delivered → consumer awaits forever → pipeline HANG,
matching the project's known parallel-hang signature); release: assert compiled
out, `decompress_..._into` gets a short slice, returns `None`
(`gzip_chunk.rs:282`) → safe byte-exact pure-Rust decline. The fix
`reserve(min_spare)` makes `Vec::reserve`'s own guarantee `capacity >= len +
min_spare` hold UNCONDITIONALLY → post-condition always satisfied. Correct and
complete; not masking a deeper invariant (the invariant IS the sizing contract,
and the assert is retained).
FLAG (not a refutation): the commit message's worked example
(len=49KiB, cap=16.6MiB, min_spare=16.5MiB ⇒ "reserve(704KiB) no-ops") does NOT
reconcile precisely — with those figures `cap-len ≈ 16.55MiB > 16.5MiB`, so the
guard wouldn't even be entered, and `additional` rounds to ~345KiB not 704KiB.
The numbers are loose/rounded illustration; the MECHANISM (the half-window above)
is sound and the empirical A/B (below) is the real proof. Recommend the owner not
lean on that specific arithmetic.
The surviving declines are genuine, not residual bugs: (a) non-full-window
bootstrap chunks fall back at `:223`; (b) the `until_exact` EXACT-bit case
(`:290-301`) requires `b.bit_offset == stop_hint_bits` and declines to the
bit-precise pure-Rust engine otherwise — the faithful "land-exactly-or-fail"
vendor case. The reserve fix does not touch this logic, so T8 staying at 37
declines is expected, not a leak.

## CLAIM 4 — NO REGRESSION → SOUND-ON-LOGIC (measured counts not re-run)
The production surface is one capacity-only line; the new/changed test code is
cfg/test-gated and touches no shared production state, so it cannot plausibly
introduce flakes. The "7 pre-existing flakes reproduced on a stash baseline"
claim is the right disproof shape (baseline-diff) and is credible, but I did NOT
re-execute the suites (read-only, no x86 ISA-L FFI here). The A/B (T8 44→37,
T4 17→15, isal_chunks now >0 on tiny blocks) is consistent with "more chunks now
pass the correctly-sized slice instead of declining" and is the actual evidence
the bug was real. Verified at logic level; not independently re-run.

## BOTTOM LINE
KEEP / MERGE. The change is byte-exact and safe: capacity-only growth, output
bounded to ISA-L-written committed bytes, native fully unaffected (cfg + off-by-
default knob), the retained debug_assert now genuinely holds, and surviving
declines are the faithful exact-bit/bootstrap cases. No hidden hazard found.
One documentation nit (Claim 3): the worked-example arithmetic in the commit
message is imprecise — the mechanism and empirical A/B are sound, so this is a
note, not a blocker. Per-claim: 1 SOUND, 2 SOUND, 3 SOUND (narrative-nit),
4 SOUND-on-logic (measured counts owner-run, not re-executed here).
