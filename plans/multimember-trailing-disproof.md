# Disproof verdict — pure-Rust trailing-member decode (task #8 step 2)

Branch: `fix/pure-rust-multimember`  Commit: `f7970c99`
Base: `reimplement-isa-l` @ `e3747a4c`

## The gap (confirmed by observation, not inference)

Running the previously-`#[ignore]`d test on the base produced a HARD ERROR, not
silent truncation:

```
called `Result::unwrap()` on an Err value:
Decompression("parallel SM: chunk decode failed:
  Decode(ExactStopMissed { requested: 142650408, actual: 142649896 })")
```

`requested`/`actual` are BIT offsets ≈ 17.83 MiB — i.e. member 1's true deflate
end. The single-stream SM engine slices `[header..len-8]` as ONE deflate body and
its block finder cannot cross member 1's gzip footer into member 2's header, so it
errors near the boundary (it never reaches the buffer-end trailer it sliced off).
ISA-L FFI used to consume-and-loop this; deleting it exposed the hole.

## The three disproof questions

### Q1 — does it handle the MISSED-SCAN-WINDOW boundary, not just the fixture?

YES. The fix is keyed on the *engine failure*, not on the fixture shape:
- `decompress_parallel` decodes single-member first. On a decode/size/CRC failure
  (the misroute signature) AND a confirmed real trailing member, it resumes.
- `trailing_member_after_first` walks member 1's deflate to its byte-aligned
  BFINAL end (`scan_inflate::deflate_stream_byte_len`) and checks for a gzip magic
  at `header + deflate_len + 8` — this is independent of WHERE the boundary falls
  (16 MiB or 160 MiB), so it covers any past-window member, not the fixture only.
- End-to-end CLI proof on a NON-fixture input: `cat big.gz small.gz` with
  big = 18 MiB random (> 16 MiB window) → byte-exact AND `cmp`-equal to `gunzip -c`.
- `test_concatenated_members_past_window_multiple_trailing` adds a THREE-member
  stream (17 MiB + text + 700 KiB incompressible) decoded through both the Vec and
  the writer/out_fd sinks → resumes MORE THAN ONE trailing member.

### Q2 — is each member's CRC32 + ISIZE verified?

YES. The resume loop slices exactly one member and hands it to the unmodified
`read_parallel_sm`, which reads THAT member's own 8-byte trailer and verifies
both `total_size == ISIZE` and `total_crc == CRC32` (sm_driver.rs:65-78). A wrong
byte in any member ⇒ terminal `SizeMismatch`/`CrcMismatch`. The deflate-boundary
walk is byte-aligned at BFINAL so each member's trailer is located exactly.

### Q3 — does it regress single-member (or bgzf / gzippy-parallel)?

NO.
- Single-member HOT PATH is untouched on success: the resume only runs on the
  FAILURE path. A true single member succeeds on the first `read_parallel_sm` and
  returns — no walk, no second decode. The only added work on the success path is
  passing an `&mut usize` to `drive_capturing` (a single store of `total_size`).
- `drive_impl`'s new `bytes_written_out` is `None` for every existing caller
  (`drive`, `drive_clean_window_oracle`, all tests) ⇒ provably zero behavior change.
- A CORRUPT single member fails, finds no trailing member
  (`trailing_member_after_first` ⇒ false), and surfaces the ORIGINAL terminal
  error — no silent truncation, no infinite loop
  (`test_corrupt_single_member_is_terminal_error_not_resumed`).
- bgzf / multi-member-parallel / gzippy-parallel routing is unchanged (the fix is
  inside `decompress_parallel`, reached only on the single-member route).

## Byte-exact evidence
- Full lib suite: 889 passed / 0 failed / 12 ignored.
- Full lib suite under `GZIPPY_POISON_RESERVE=1` (single-threaded): 889 / 0 / 12.
  (Two `bgzf::bench_*` corpus tests flake ONLY under parallel poison-reserve —
  a pre-existing corpus-prep race in code my diff does not touch; both PASS in
  isolation and on the base.)
- routing tests: 43 passed.
- CLI single-member roundtrip on a 25 MiB (> 16 MiB) payload: byte-exact.
- CLI multi-member misroute (`cat big.gz small.gz`, big=18 MiB): byte-exact AND
  equal to `gunzip -c`.
- Dual-sha single-member (`028bd002…cb410f`, silesia) is a neurotic-only run;
  the single-member decode path bytes are unchanged by construction (success path
  untouched), and the local large single-member roundtrip is byte-exact.

## Residual notes
- The resume path streams through the `Write` trait (not the zero-copy `out_fd`
  writev) — correctness over speed on this rare path.
- `SkipWriter` drops exactly the validated, in-order prefix already streamed
  (chunks are written only after they validate), so member 1's prefix is never
  duplicated when the whole stream is re-decoded member-by-member.

## Disproof self-review (7 angles, no confirmed bugs)

- out_fd / writev interleaving: attempt 1 (out_fd=Some) writes member-1 chunks via
  writev directly to the fd; with out_fd set the BufWriter is never touched
  (consumer takes the fd branch, skips the buffered branch). The resume forces
  out_fd=None and writes the TAIL through the SkipWriter→BufWriter→same fd, after
  attempt 1's synchronous writev completed ⇒ correct order. `total_size` (the
  `already_written` count) is incremented on BOTH the fd-writev path
  (chunk_fetcher.rs:3940) and the buffered path (3973), so the skip count is right
  regardless of sink. VERIFIED in production: `gzippy -d mm.gz` (file output, real
  writev) on `cat b.gz s.gz` (b=18 MiB) → byte-exact, debug trace shows
  "resuming members past 17833949 streamed bytes" then full 18000051-byte output.
- SkipWriter byte-position skip is independent of the member boundary: the resume
  re-derives the identical full byte stream; dropping the first `already_written`
  bytes == exactly attempt 1's validated in-order prefix.
- `total_crc: 0` from the resume is unused (caller reads only size; per-member CRC
  is verified inside each `read_parallel_sm`).
- `deflate_stream_byte_len` cannot infinite-loop: every block makes forward
  progress or returns Err; the loop exits on BFINAL.
- No removed guard: the original error→`ParallelError` mappings are preserved in
  the non-resumable else branch.

VERDICT: handles the general past-window boundary (not just the fixture), verifies
every member's CRC/ISIZE, decodes correctly through both the Vec and the
fd/writev production sinks, and does not regress single-member or the other routes.
