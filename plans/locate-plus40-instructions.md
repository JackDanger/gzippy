# LOCATE THE +40% PIPELINE INSTRUCTIONS (the BAR-1 lever) — owner charter 2026-06-09

## GOVERNING PRINCIPLE (user): a performance difference IS a compiled-code difference; know what it IS.
Frozen-box-measured (DIS-17): gzippy-isal executes +40% more instructions than rapidgzip at T4
(7.28e9 vs 5.18e9 = ~2.1e9 extra) wrapping the BYTE-IDENTICAL ISA-L `_04` kernel. So the extra
~2.1e9 instructions are in gzippy's PIPELINE (NOT the engine — same nasm). This is the BAR-1
lever and the per-chunk ParallelSM pipeline doing more work/byte than rg's leaner consumer
(confirmed independently: single-shot T1 beats rg 1.197x by removing chunking). LOCATE which
functions emit the extra instructions; then faithfully converge them to rg's leaner pipeline.

## STEP 0 — proof-of-binary: gzippy-isal at HEAD, isal_chunks>=14, env-unset, path=ParallelSM.

## STEP 1 — ATTRIBUTE the +40% to specific gzippy functions (perf record -e instructions)
perf record -e instructions (or cycles+annotate) on gzippy-isal T4 decode of silesia (symboled
build — build with debug symbols, NOT stripped). Get the per-function instruction share. The
extra ~2.1e9 are OUTSIDE the ISA-L kernel (decode_huffman_code_block_stateless_04) — they are in
the gzippy pipeline. ATTRIBUTE them to the candidate gzippy operations (cite the actual hot
functions): u16 MARKER decode + RESOLVE/replace_markers (the 2-pass marker handling), the
ring write + DRAIN to chunk.data (the FOLD), the u16->u8 NARROWING, per-chunk CRC, window-map
lookup/publish, per-chunk init+set_dict, the consumer handoff. Which functions own the +40%?
Cross-map to rg's equivalents (ParallelGzipReader.hpp / GzipChunk.hpp / DecodedData.hpp — the
pipeline-fidelity-verdict.md already mapped the structure) — where does gzippy do MORE
instructions than rg's counterpart, and WHY (extra pass? extra copy? per-symbol overhead)?

## STEP 2 — name the faithful convergence + (if clear + byte-transparent) attempt + measure
For the top instruction-owner, name the faithful reduction (match rg's leaner approach, vendor
file:line). Candidate the campaign has flagged: gzippy decodes the clean tail into a SEPARATE
buffer + drains/narrows vs rg's IN-PLACE narrowing (DecodedData.hpp:344-388 reinterpret_cast +
reusedDataBuffers) — extra copies/passes = extra instructions. If a faithful byte-transparent
reduction is clear, attempt it (OFF==identity, dual-sha, full suite) and MEASURE the T4 wall +
the instruction count (does it drop toward rg's 5.18e9?). NO WORK-DISPLACEMENT. Else report the
located attribution + the ranked convergence targets for a supervised turn.

PRE-REGISTER falsifier: the located function(s) must sum to ~the 2.1e9 excess; a convergence that
cuts instructions but not the wall (or displaces work) is REVERTED.

## GATES + DISCIPLINES
git WORKTREE; box is FREE — bench-lock freeze, release clean. Build SYMBOLED for perf attribution
(separate from the stripped bench binary). Numbers ONLY from the bench-locked quiet guest;
matched same-sink; interleaved N>=11; sha-verified; path=ParallelSM + isal_chunks>=14 asserted.
The Agent/advisor tool is UNAVAILABLE to you — run rigorous self-disproof and hand ME (supervisor)
RAW perf attribution + numbers + isal_chunks readback for my Opus gate; do NOT claim
"advisor-vetted." Run measurements YOURSELF holding the ssh. SOURCE-VERIFY first-hand. Serialize
builds via cargo-lock.sh, df -h around builds. No multi-line python via Bash (write a .py). Wrap
hang-prone cmds in timeout. Diagnose the FIRST error before retrying. NO orphan processes / sleep
sentinels — pgrep clean on local + guest + neurotic before finishing. Update
plans/orchestrator-status.md + the disproof-ledger. STOP at the checkpoint and report: the
per-function +40% attribution (perf evidence), the ranked faithful convergence targets, any
attempted fix's wall+instruction delta, and raw numbers for my gate.
