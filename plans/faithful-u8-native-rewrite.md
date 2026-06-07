# FAITHFUL u8 NATIVE REWRITE — leader charter (supervisor, user-corrected 2026-06-07)

This SUPERSEDES the u16-ring + asm-graft direction and the "asm-port/FFI/revisit-bar fork."
Two advisors + the user established: the +13% engine PLATEAU was measured on a u16-ring clean
arch **rapidgzip never uses for the bulk**; the faithful u8 native path was never built in
production. Build it correctly and re-measure.

## USER DEFINITIONS (authoritative — override any plan/memory text)
- **gzippy-native**: does LITERALLY what rapidgzip does, ENTIRELY in Rust, **u8 whenever
  rapidgzip uses u8 — FULL STOP**, no C-FFI. = faithful port of rapidgzip WITHOUT_ISAL: one
  deflate::Block that flips the SAME buffer u16→**u8 WIDTH** in place at 32 KiB and decodes the
  clean bulk **u8-direct** (deflate.hpp:1282-1291; u8 memcpy "~400MB/s→~6GB/s" at :1244).
- **gzippy-faithful (isal)**: ALSO u8 wherever rapidgzip uses u8, FULL STOP, **+ ISA-L via
  C-FFI** (= rapidgzip WITH_ISAL: deflate::Block ≤32KiB markered prefix → ISA-L u8 stream bulk).
Both u8 in the clean tail; differ ONLY in whether the u8 decoder is gzippy-Rust or ISA-L-C.

## USER DIRECTIVE — NO SHORTCUTS (verbatim intent, 2026-06-07)
"The development cost on our end is entirely dominated by us taking shortcuts to save time and
avoid high development cost rather than rewriting correctly." ⇒ when you hit a hard rewrite
(the ring backing → u8, setInitialWindow's full-65536 rotate+value-downcast, the post-flip
max-distance seam back-ref), REWRITE IT CORRECTLY. Do NOT graft/narrow/flag-flip to dodge the
real change. Keeping the ring u16 + narrowing at drain was exactly the shortcut that caused the
plateau and the deviation.

## THE JOB (you, the leader, IMPLEMENT this and DELEGATE the design/review/measure GATES)
1. **Establish the ONE production clean path** (CLAUDE.md "ONE PRODUCTION PATH"; there has been
   recurring which-impl-is-production confusion). The 2026-06-07 advisor found production
   marker_inflate.rs decodes the clean bulk u16 (stores + a u16→u8 narrow at drain,
   marker_inflate.rs:42,743-750). A prior u8-direct port (commit 5256075) may be on a
   different/dead path. Confirm exactly which function the CLI's native parallel-SM path calls
   for the clean tail. (Delegate the source map to a read-only subagent, SYNCHRONOUS.)
2. **Rewrite that production clean path to faithful u8-direct flip-in-place**, mirroring vendor:
   the ring backing = one 128KB store viewed as [u16;65536] pre-flip / [u8;131072] post-flip;
   port setInitialWindow's full-65536 rotate + value-downcast (explicit (x&0xFF) as u8, NOT a
   bit-reinterpret); post-flip readInternal writes u8 DIRECTLY (one store), no narrow-at-drain.
   ONE engine, SAME cursor — this honors no-FFI AND one-engine AND u8 (they are the same thing
   when done faithfully). The faithful-unified memory's "REMAINING MEMORY-MODEL DEVIATION" +
   "FAITHFUL PLAN" sections are the correct recipe; the seam TRAPS (A adversarial 32768-distance
   back-ref across the repositioned seam; B reinterpret-vs-downcast; C window-seed tail-zero)
   are MANDATORY tests.
3. **Byte-exact every step** (dual-sha 028bd002…cb410f BOTH features; all lib tests green;
   the adversarial seam test). A faster path with wrong bytes is void.
4. **Measure on the locked guest** (you run it / hold the ssh; idle-before, restore-after,
   N≥9 interleaved, sha-verified): does faithful u8-direct close the gap vs rapidgzip's 0.604s
   same-sink wall? The advisor's load-bearing prediction: once flat-u8 linear, decode may STOP
   binding above ~120 MB/s and the tie comes from the shared ~0.54-0.60s floor — i.e. the asm
   kernel port may be UNNECESSARY. Report the measured wall + whether decode still binds.

## CHECKPOINT (STOP)
Report: which function is the production native clean path; the u8 rewrite landed byte-exact?;
the measured same-sink wall vs 0.604s + whether decode still binds. Route through an independent
disproof advisor (verdict to plans/faithful-u8-native-advisor-verdict.md). Then STOP for
supervisor gate. (gzippy-faithful's u8+ISA-L-FFI path and any asm-kernel port are LATER/
contingent — native u8 first.)

## DISCIPLINES (enforced — yields + orphans hit EVERY round)
- RUN SUBAGENTS SYNCHRONOUSLY (block with timeout, collect in-turn). Do NOT background-and-yield
  for a Monitor/notification — there is NO auto-reinvoke; TWO leaders died this way. Run the
  measurement YOURSELF via Bash holding the ssh; only delegate the advisor (synchronously).
- NO detached sleep sentinel. Before finishing, pgrep MUST show none of your claude -p subagents
  and no orphaned timeout `sleep` procs — kill them (the supervisor has cleaned orphans every
  round).
- SOURCE-VERIFY every premise first-hand (a wrong premise wasted a whole turn).
- Serialize builds via cargo-lock.sh (df -h around builds); don't run multi-line python via Bash
  (write a .py file); diagnose the FIRST error before retrying. Update plans/orchestrator-status.md.
