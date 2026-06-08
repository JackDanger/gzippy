# DoS termination guard for `deflate_stream_byte_len` — fix + disproof verdict

Branch: `fix/pure-rust-multimember` (stacked on f7970c99 + e9d1aba3)
Date: 2026-06-08

## The bug (confirmed empirically)

`deflate_stream_byte_len` in `src/decompress/scan_inflate.rs` is the boundary
walker the multi-member resume calls on the FAILURE path over UNTRUSTED trailing
bytes (call sites: `parallel/single_member.rs:147` `trailing_member_after_first`,
and `parallel/sm_driver.rs:247` `read_parallel_sm_resume_multi`). On adversarial
trailing bytes it INFINITE-LOOPED (DoS).

Mechanism: `0xaa` → bfinal=0 / btype=01 (fixed). The shared decoder consumes the
bytes, refill past EOF yields zeros, the fixed block resolves its 7-bit EOB from
those zeros and returns Ok, then a zero-length stored block synthesized at EOF
(`consume_first_decode.rs:1409-1410`) returns Ok WITHOUT setting bfinal and
without advancing the bit cursor. The walk loop (exits only on bfinal/Err) spun
forever. The deleted ISA-L FFI errored cleanly here; the pure-Rust port did not.

Empirical proof: with the guard reverted, the watchdog unit test fired its 5s
timeout and FAILED with "deflate_stream_byte_len HUNG on adversarial 0xaa
garbage" (test finished in 5.01s). With the guard, the same call returns a
terminal Err in ~0ms.

## The fix (src/decompress/scan_inflate.rs, in `deflate_stream_byte_len`)

Three bounds added to the walk loop (the doc comment claimed bounded memory but
there was NO iteration bound):

1. Non-advancement: after each non-final block, if `bit_pos_now <= last_bit_pos`
   the cursor is stuck (refill-past-EOF zero synthesis) → terminal Err.
2. EOF-without-BFINAL: if `bit_pos_now.div_ceil(8) >= deflate_data.len()` we have
   consumed all real input without a BFINAL → truncated member → terminal Err.
3. Hard iteration ceiling `max_iters = len + 16` — a backstop in case the
   underflow-wrapped bit cursor oscillates rather than monotonically stalling.

A genuine multi-member input still walks each intermediate block (each advances
bit_position strictly) to the real BFINAL and resumes correctly. The guard only
fires on stuck/truncated/garbage input.

## Tests added (src/tests/correctness.rs)

- `test_deflate_stream_byte_len_terminates_on_adversarial_garbage` — calls
  `deflate_stream_byte_len(&[0xaa; 4096])` inside a 5s watchdog thread; a hang
  regression PANICS ("HUNG") rather than wedging CI. Config-independent (runs
  under both default and pure-rust-inflate features).
- `test_big_member_plus_gzip_magic_garbage_is_terminal_not_hang` — end-to-end:
  valid 17 MiB member + gzip magic + 0xaa*4096 garbage → `decompress_single_member`
  must return terminal Err (not hang, not silent truncation), inside a 30s
  watchdog. `#[cfg(parallel_sm)]`-gated because the trailing-member RESUME path
  (where the hang lived) only exists on the pure-Rust parallel-SM path; under the
  non-parallel-SM routing this input takes a different backend that decodes
  member 1 and ignores non-member trailing bytes (no resume, no hang). Matches
  gzip(1), which exits 1 "data stream error" on magic+garbage.

## Validation

- Hang repro now returns terminal Err — proven both by the unit watchdog (pre-fix
  HUNG@5.01s / post-fix ok@0ms) and the e2e (terminal Err @0.35s under parallel_sm).
- 3 existing multi-member tests STILL pass:
  test_concatenated_members_large_first_member_no_truncation,
  test_concatenated_members_past_window_multiple_trailing,
  test_corrupt_single_member_is_terminal_error_not_resumed.
- Full bin suite under `--no-default-features --features pure-rust-inflate`
  (the production path): 898 passed, 0 failed, 12 ignored.
- Default-features correctness+routing: 158 passed, 0 failed.
- Byte-exact round-trips local: 25 MB random (path=StoredParallel) and 30 MB
  compressible across T1/T4/T8 (path=ParallelSM, GZIPPY_FORCE_PARALLEL_SM=1) —
  all output SHA256 == input SHA256. The guard only affects the
  adversarial/truncated path → terminal Err; happy path untouched. (The corpus
  dual-sha 028bd002…cb410f is the silesia decompressed SHA on neurotic; not run
  locally — corpus is host-side. The trace_parity_* and resumable real-silesia
  tests, all green, cover byte-exact single/multi/bgzf decode.)

## Advisor disproof (claude -p WORKED — synchronous)

Verdict: **SOUND**. The advisor tried four adversarial classes (0xaa garbage,
truncated final block, trailing non-member garbage, dense tiny valid blocks) and
broke none:
- Termination: `bit_position()` is bounded above by `len*8` (pos saturates at
  data.len at EOF), so it cannot strictly increase forever; a fixed point trips
  the `<=` check; `max_iters` is the guaranteed cap regardless of oscillation.
- No false-Err: smallest valid non-final block = 10 bits (3 header + 7-bit EOB),
  so max block count = 0.8*len < len+16 → ceiling unreachable on valid streams.
  The `<=` check never fires on a valid intermediate block (each consumes >=10
  bits, strictly increasing). The EOF check sits AFTER `if bfinal break`, so a
  member's real final block never reaches it.
- No overflow/panic (div_ceil/saturating ops).
- Non-blocking out-of-scope note: the guard bounds only the OUTER walk; it relies
  on decode_{stored,fixed,dynamic} terminating on garbage (they do).

## Ready for supervisor

Fix committed to `fix/pure-rust-multimember`. Worktree:
`.claude/worktrees/multimember-fix` (detached at branch tip + this commit).
