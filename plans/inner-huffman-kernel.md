# INNER HUFFMAN LUT-DECODE KERNEL — leader charter (supervisor, 2026-06-07)

The faithful u8 native rewrite LANDED byte-exact (fa9fd73c) and is a wall TIE: the gap to
rapidgzip is a constant ~1.70× at BOTH T1 and T8. CAUSALLY established (advisor-upheld):
clean-path TRAFFIC, ring width, and PLACEMENT are all SLACK; the binder is **per-symbol
Huffman LUT-decode COMPUTE** — the inner kernel. Now on the FAITHFUL flat-u8 path, this is the
real and only remaining lever to the 1.0× tie. Inner Huffman loop = OPEN territory (full
reimplementation + inline ASM authorized); pure-Rust, NO C-FFI on the native decode graph.

## THE LEVER (igzip's actual fast path, from plans/asm-kernel-feasibility-report.md Q1)
NOT the E2/E3/E4 grafts already tried (+13%, on u16, no table redesign). The binder is the
DECODE TABLE + pipeline:
1. **Packed flat short-code table** (igzip: one u32 load retires up to 3 literals + the
   bit-length; igzip_decode_block_stateless asm:57-68, igzip_inflate.c:46-57). This directly
   attacks "per-symbol LUT-decode compute."
2. **Speculative software-pipelined loop** (store 8 bytes unconditionally; preload the next
   lit/len AND dist symbol before knowing the current type; asm:507-627), with a slop-margin
   headroom guard (asm:48,488-512). These now apply because the path is flat-u8.

## THE JOB (you, the leader, IMPLEMENT + delegate the design/review/measure GATES)
1. **Isolation-bench the table+pipeline lever FIRST** (the tiered gate — prove the ceiling
   BEFORE full production integration): extend the existing standalone engine bench
   (benches/engine_isolation.rs) with a variant that decodes the clean flat-u8 chunk via the
   packed-flat-short-code table + speculative pipeline (pure-Rust + inline-asm where Rust codegen
   lags). Compare vs (i) the current u8 production loop and (iii) the ISA-L oracle (positive
   control, guest ratio). Pre-register the falsifier: PASS = the variant's clean MB/s, via §3
   (same-sink 0.604s bar), projects a T8 wall ≤ 0.604s + spread (≈ closes the 1.70×). PLATEAU =
   stays ~pure class with residual > spread.
2. **Byte-exact** the bench variant (SHA-equal vs scalar + ISA-L on the swept chunks). An oracle
   variant with wrong bytes is void.
3. If the isolation bench PASSES, **then** (gated by this checkpoint) integrate into the
   production flat-u8 clean path, byte-exact (dual-sha 028bd002…cb410f both features; all tests),
   and re-measure the production wall on the locked guest. If it PLATEAUS, report the achievable
   ceiling — that is the supervisor/user finding (is igzip-class reachable in pure-Rust+asm at
   all, or does even the full table+pipeline fall short → the FFI/bar fork re-opens).

## NO SHORTCUTS (user directive, standing): our cost is dominated by shortcuts, not correct
rewrites. Build the table + pipeline CORRECTLY (the real packed-table redesign + the speculative
loop), not a partial graft that dodges the table change. The +13% graft was the shortcut.

## CHECKPOINT (STOP)
Report: the isolation-bench variant's clean MB/s vs the ISA-L oracle + the §3 wall projection +
the settled PASS/PLATEAU verdict; if integrated, the production wall delta. Route through an
independent disproof advisor (verdict to plans/inner-huffman-kernel-advisor-verdict.md). Then
STOP for supervisor gate.

## DISCIPLINES (enforced — yields + orphans hit EVERY round)
- RUN SUBAGENTS SYNCHRONOUSLY (block with timeout, collect in-turn). Do NOT background-and-yield
  for a Monitor/notification — there is NO auto-reinvoke; multiple leaders died this way. Run the
  measurement YOURSELF via Bash holding the ssh; only the advisor is a delegated SYNCHRONOUS call.
- NO detached sleep sentinel. Before finishing, pgrep MUST show none of your claude -p subagents
  and no orphaned timeout `sleep` procs — kill them explicitly.
- SOURCE-VERIFY every premise first-hand. Serialize builds via cargo-lock.sh (df -h around
  builds); don't run multi-line python via Bash (write a .py file); wrap hang-prone commands in
  timeout; diagnose the FIRST error before retrying; numbers only from the locked guest. Update
  plans/orchestrator-status.md.
