# ISA-L EQUIVALENCE + ASSEMBLY-ANALYSIS TOOLING — lead-auditor charter (2026-06-09)

## GOVERNING PRINCIPLE (user, 2026-06-09 — your north star)
"Anytime our performance is different, it's a sign that there is a difference in the COMPILED
CODE. It's important to know what that difference IS." Your entire job is to find the SPECIFIC
compiled-code differences. A performance gap with no identified machine-instruction difference
is an UNFINISHED investigation. Drill to the disassembly; name the exact divergent instructions.

## THE HUNCH (user): our compiled gzippy-isal may NOT look like the vendor (rapidgzip) one.
The campaign has ASSUMED "gzippy-isal's ISA-L clean-tail decode == rapidgzip's ISA-L." That
assumption is UNAUDITED. Build REUSABLE analytical tooling and render a verdict at THREE levels.

## LEVEL 1 — SAME PATCHED SOURCE?
rapidgzip and gzippy BOTH patch ISA-L (boundary-recording / inexact-offset support). They must
match for the decode to be equivalent.
- LOCATE both ISA-L sources: gzippy's `vendor/isa-l` + `vendor/isal-rs` (the Rust binding);
  rapidgzip's ISA-L (in `vendor/rapidgzip/` — find how rapidgzip vendors/patches ISA-L, e.g.
  its CMake fetch + .patch files, or its inlined igzip).
- DIFF: the ISA-L VERSION (git tag/commit), the exact PATCH (the boundary-recording changes to
  igzip_inflate / igzip_decode_block_stateless), and the build/compile flags (nasm, AVX2,
  target-features, -O level, the `isal-rs` build.rs flags vs rapidgzip's CMake). Report every
  material difference. Are we even on the same igzip version?

## LEVEL 2 — SAME COMPILED ASSEMBLY?
Extract + compare the COMPILED MACHINE CODE of the ISA-L inflate hot loop from BOTH binaries.
- gzippy-isal binary: build at HEAD (cargo, feature gzippy-isal, x86_64, target-cpu per the
  bench). rapidgzip binary: rapidgzip 0.16.0 (on the guest /root, or build from vendor/).
- Build TOOLING (reusable, scripts/analysis/): disassemble (objdump -d / llvm-objdump) the
  igzip inflate symbols from each, normalize (strip addresses/symbol-name mangling), and produce
  an instruction-level diff + summary (AVX2/BMI2 instruction histogram, hot-loop instruction
  count, whether the same igzip kernel is emitted, inlining differences, whether gzippy's ISA-L
  is the .asm hand-written kernel or a C fallback). If symbols are stripped, use the ISA-L
  source symbol names / pattern-match the known igzip_decode_block_stateless prologue.
- VERDICT: is the SAME igzip kernel running in both, instruction-for-instruction? Or does
  gzippy link a different/older/C-fallback ISA-L?

## LEVEL 3 — SAME CALL CONTEXT?
How each CALLS ISA-L: rapidgzip `IsalInflateWrapper` (isal.hpp) vs gzippy
`decompress_deflate_from_bit_into` (backends/isal_decompress.rs) -> the FFI -> the patched
inflate. Compare the per-call setup (buffer alloc, bit-align, state init, the inexact-offset/
boundary handoff), the calling convention, and the per-call OVERHEAD (this connects to the
FFI-handoff term — measured negligible at T4 but verify the SETUP isn't heavier than rg's).

## DELIVERABLE
A reusable tooling suite under scripts/analysis/ (source-differ, disasm-extractor+differ,
build-flag-auditor) + plans/isal-equivalence-verdict.md: the verdict at each level with the
specific diffs. If you need parallel sub-agents (you likely CANNOT spawn them in this env —
verify), report the specific fan-out to the SUPERVISOR and I will launch them.

## DISCIPLINES
git WORKTREE. This is SOURCE/BINARY/DISASM analysis — do NOT freeze the box / do NOT run wall
measurements (a residual-sizing oracle holds the box). Serialize builds via cargo-lock.sh, df -h
around builds. SOURCE-VERIFY first-hand; do NOT trust prior prose. No multi-line python via Bash
(write .py files — this is tooling, so .py/.sh scripts are the deliverable). Wrap hang-prone cmds
in timeout. Diagnose the FIRST error before retrying. NO orphan processes. Hand ME raw findings +
the tooling for my Opus gate; do NOT claim "advisor-vetted." Update plans/orchestrator-status.md.
STOP at the checkpoint and report the three-level verdict + the tooling built.
