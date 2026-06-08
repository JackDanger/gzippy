# Copy-free clean-tail WALL oracle — brief for disproof advisor

HEAD 7aae6c4a + this turn's copy-free oracle overlay. Branch reimplement-isa-l.
Guest 10.30.0.199 (locked), gov=performance, turbo on, taskset 0,2,4,6,8,10,12,14.

## What changed this turn (the fix the prior advisor OWED)
The prior `GZIPPY_ISAL_ENGINE_ORACLE` decoded the clean tail with REAL ISA-L into a
fresh per-chunk 64 MiB `Vec` (`isal_decompress::decompress_deflate_from_bit_with_boundaries`),
then `copy_from_slice`'d the kept region into `chunk.data.writable_tail()`
(gzip_chunk.rs:203,247-256). The prior advisor (clean-tail-wall-oracle-advisor-verdict.md)
ruled CONCLUSION 1 INCONCLUSIVE because that per-chunk alloc+copy costs ~0.17× of the wall
— LARGER than both the prod→ocl gap and the gap to the 0.85× threshold — so the
clean-engine speed-up ceiling was UNREADABLE. Predicted: copy-free ocl ≈ 0.84-0.87×.

THIS TURN made it copy-free:
- NEW `isal_decompress::decompress_deflate_from_bit_into(data, bit, dict, out: &mut [u8])`
  — decodes ISA-L DIRECTLY into a caller buffer, NO internal Vec, NO CRC inside, returns
  `(written, end_bit, boundaries)`; returns None (fall back) if caller under-reserved.
- NEW `SegmentedU8::writable_tail_reserve(min_spare)` — one contiguous spare region of the
  full chunk size (not capped at 128 KiB), in the chunk's OWN pooled buffer.
- Oracle now: `out = chunk.data.writable_tail_reserve(64 MiB-reserve)` → ISA-L decodes into
  it → `commit(keep_len)` (ZERO copies) → CRC over `decoded_range(decode_start, keep_len)`
  (zero-copy view). Boundary offsets rebased to `decode_start` (matches production's
  `decode_base = chunk.decoded_size()`).
- 64 MiB is RESERVED (capacity) in the recycled pool buffer, NOT a fresh alloc+memset
  Vec; ISA-L stops at a block boundary long before exhausting it.

## SELF-TEST (Rule 4 — PASSED)
PROD (pure-Rust clean tail) == ORACLE (copy-free ISA-L) == rg sha
028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f (byte-exact, OFF==identity).
isal_oracle_chunks=14 isal_oracle_fallbacks=0 (ALL 14 clean tails took copy-free ISA-L,
ZERO under-reserve fallbacks). Routing flip_to_clean=12 finished_no_flip=4 window_seeded=2
= IDENTICAL to prod unseeded ⇒ the 89% window-absent marker bootstrap is PRESERVED (charter
OSCILLATION rule honored; did NOT seed).

## RESULT — THE WALL NUMBER (3 interleaved N=11 passes, sha-OK every run, measure.sh)
| variant | pass1 | pass2 | pass3 |
| rg | 1.000 | 1.000 | 1.000 |
| prod (pure-Rust clean tail) | 0.755× | 0.746× | 0.733× |
| ocl (COPY-FREE ISA-L clean tail, unseeded) | 0.895× | 0.892× | 0.870× |

The copy-free ISA-L clean tail BEATS production by ~0.14-0.16× of the wall ratio and lands
AT/ABOVE the 0.85× TIE threshold — EXACTLY as the prior advisor predicted (S≈C≈0.17×).
The prior contaminated oracle showed ocl 0.70× ≈ prod 0.75× (copy masked the engine win);
removing the copy moved ocl 0.70→0.89×.

Verbose (copy-free, unseeded): decodeBlock SUM 0.645s (prod was ~0.83s last turn),
Real Decode 0.101s, Fill 79.5%, isal_oracle_chunks=14 fallbacks=0.

Sign-stable across load 1.27→2.01 (prod and ocl both drift with load but ocl stays
~0.14× ahead) ⇒ freq-neutral (interleaved harness runs rg/prod/ocl back-to-back in the
same turbo state).

## CLAIMS to disprove
- C1: with the copy confound removed, the ISA-L clean-engine RATE is a genuine WALL lever
  worth ~0.14-0.16× of the gzippy→rg gap (the clean-engine ceiling is now READABLE and the
  Rule-3 speed-up ceiling FIRED: a copy-free clean engine reaches the TIE zone). This
  REVERSES the prior INCONCLUSIVE: the clean engine is NOT slack; it is a real lever.
- C2: the residual gap above ocl (0.89× vs rg 1.0×) is the window-absent STRUCTURE
  (marker bootstrap + spec-fail re-decodes + resolve pass), consistent with the prior
  reconciliation (seedfull ocl 0.86× vs the structure removed).
- C3: the faithful fix is to COLLAPSE gzippy's two engines (marker_inflate u16 + resumable
  u8 clean) into ONE width-templated primitive mirroring rg's readInternalCompressedMultiCached
  — but the COPY-FREE oracle shows the pure-Rust clean engine (resumable.rs) is ~0.14× of
  wall slower than ISA-L at the clean tail, so a pure-Rust 1.0× requires the unified primitive
  to match ISA-L's clean rate (the VAR_VI 0.6× standalone plateau is the open risk).

## Disproof angles requested
1. Is the copy-free oracle a VALID instrument now (C→0)? Any residual confound (e.g. the
   64 MiB reserve still touches pages; boundary rebase to decode_start correct)?
2. Does C1 (engine rate IS a wall lever) survive — or is 0.89× still inside spread/TIE such
   that "engine is the lever" overreaches? Is the prod→ocl 0.14× delta real (Δ > spread)?
3. C3 faithfulness: does the copy-free result change the no-FFI 1.0× picture — is pure-Rust
   1.0× reachable, or does the ISA-L-clean-beats-pure-Rust-by-0.14× number imply the unified
   pure-Rust primitive must close a ~0.14× wall gap that the VAR_VI 0.6× plateau says it can't?
4. Anything that makes ocl look artificially GOOD (the symmetric of the prior copy confound)?
