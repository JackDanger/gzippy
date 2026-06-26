# COPY-FLOOR C — gated verdict (2026-06-25)

Mandate: close the residual of the COPY-FLOOR — gzippy-native loses to igzip on
logs T1 by ~+10-14% cyc (after/igzip ~1.138 Intel / ~1.105 AMD, where "after" =
HEAD + the unmerged per-block-build win `feat/perblock-build-speedup`). The prior
leader called the residual "the broad pure-Rust-vs-ISA-L codegen floor, not
surgically capturable" — REJECTED as an un-gated prose floor. Find the lever or
produce a PROPER (non-rep-movsb) gated removal-oracle ceiling for a SPECIFIC region.

Base = `reimplement-isa-l` HEAD `38f44528`. Binaries built FRESH at HEAD,
`RUSTFLAGS=-C target-cpu=native`, `--no-default-features --features
pure-rust-inflate` (gzippy-native, parallel-sm+pure), path=ParallelSM, sha-verified
vs `gzip -dc`. base sha `7603c180…`. Corpus `/root/floor/logs.gz` (pigz, ~3749
blocks, 15.2× ratio). Box law: trainer Intel i7-13700T (LOADED, run-queue 4-24 ⇒
cyc/B VOID-QUIET; instr/B load-immune; contention-invariant paired sign-test is the
verdict). AMD box NOT touched this session (runs the user's llama).

Tools: `fulcrum chainlat` (llvm-mca critical-recurrence analyzer — built fresh on the
trainer, the box copy lacked it), `fulcrum abmeasure` (load-immune gate), perf
record/annotate at true T1 (`-p1`, `taskset -c 3`).

## STAGE 1 — LOCATE (perf T1, deterministic-ish)

**The "memmem is 31%" reading is a MULTI-THREAD SAMPLING ARTIFACT — DISCARDED.**
A first profile at default threads=16 showed memchr memmem (the trailing-member
scan, `format.rs:72`) at 31% vs run_contig 14.7%. That is an artifact: run_contig is
spread across 16 threads (diluted), the serial memmem scan is not. At TRUE T1
(`-p1`, single core), the honest split is:

  run_contig                         63.58%   ← the decode kernel (copy + litlen)
  LutLitLenCode::rebuild_from        13.73%   ← per-block table build (separate pass)
  finish_decode_chunk_contig_native  11.36%   ← per-chunk CRC/window finish
  memmem (trailing scan)              2.14%   ← NOT a T1 lever
  Block::read_header                  1.66%

igzip on logs at T1: `decode_huffman_code_block_stateless_04` 74.72% +
`crc32_gzip_refl_by8_02` 12.73%. gz's (run_contig 63.6 + rebuild_from 13.7 = 77.3%)
maps to igzip's combined decode (74.7%); gz finish 11.4% ↔ igzip crc 12.7%.

perf annotate of run_contig: the single hottest cluster is the **MOVDQU large-copy
loop `71:`** (back-edge jmp 6.20%, load 4.43%, store 2.95%) for LONG matches
(len 49..240, dist≥16) + the 3× MOVDQU burst (len≤48). **The dist<16 scalar
period-growth path (`55:`/`56:`) is COLD** — not in the top-40 hot instructions.

## STAGE 2 — STRUCTURAL DIFF (igzip vs gz copy)

igzip's hottest cluster on logs is ALSO its copy loop (vmovdqu store 7.43% / load
5.43% / jmp 5.39% ≈ 23%). logs is copy-heavy via LONG matches for BOTH tools (not
short distances — the brief's "15.2× ⇒ short-dist-dominated" premise is wrong for log
data: log redundancy is long repeated templates at moderate distance).

Disassembly diff of the hot large-copy loop:

  gz   c4ea1: movdqu %xmm0,(%r13,%r14,1) / sub $16,%r15 / jle / add $16,%r13 / movdqu (%r13),%xmm0 / jmp
  igz  38e41: vmovdqu %xmm1,(%rdx,%rdi)  / sub $16,%r8  / jle / add $16,%rdx / vmovdqu (%rdx),%xmm1  / jmp

**Instruction-IDENTICAL** (same 6 ops, same loop-carried `src+=16 → load → store`
recurrence). The ONLY difference: gz emits legacy SSE `movdqu`; igzip emits VEX
`vmovdqu`. gz already faithfully ported igzip's large/small-byte-copy (COPY_SIZE=16).

## STAGE 3 — PROPER ORACLE (NOT rep-movsb)

### 3a. chainlat (llvm-mca) on the copy loop — deterministic parity, cross-arch
  raptorlake: gz 1.210 cyc/iter == igzip 1.210  → **Δ = 0.000** (RECURRENCE-bound)
  znver3:     gz 1.110          == igzip 1.110  → **Δ = 0.000**
  znver4:     gz 1.110          == igzip 1.110  → **Δ = 0.000**
Both bound by the identical `add src,16 → load → store` loop-carried recurrence.
This is the PROPER non-rep-movsb bound: the copy loop is at an exact parity ceiling
on every modeled arch.

### 3b. VEX faithful-encoding fix (the one encoding difference) — abmeasure TIE
Converted all 11 back-ref copy `movdqu`→`vmovdqu` (byte-exact: identical low-128
semantics, ymm0 upper is dead scratch; matches igzip; avoids any latent AVX-SSE
dirty-upper false-dependency penalty — a real effect llvm-mca CANNOT model).
after sha `85c8c66d…` (11 vmovdqu verified in run_contig), output byte-exact vs gzip
on logs + silesia, path=ParallelSM.

abmeasure base `7603c180` vs after `85c8c66d`, logs.gz, Intel, N=15:
  instr/B  3.8322 → 3.8321  (Δ +0.0001 — byte-exact, as expected)
  cyc/B    1.1834 → 1.1782  (Δ +0.4%, VOID-QUIET) — after faster 10/15, slower 5/15
  paired sign-test p = 0.302  → **NOT significant (TIE)**
  contention-invariant ratio 0.9982 (TRENDED, not certified)

The VEX fix is a TIE — **no significant wall win**. This DYNAMICALLY confirms the
chainlat static Δ=0.000: there is no exploitable AVX-SSE penalty here; the copy is
at a genuine parity ceiling, confirmed both statically (llvm-mca) AND dynamically
(abmeasure). The change is byte-exact + igzip-faithful + trends slightly positive,
so KEPT per CLAUDE.md ("a correct byte-identical change may be KEPT on a TIE"), but
it is NOT claimed as a win.

### 3c. Corroboration
- Prior rep-movsb/ERMS oracle (byte-exact; ERMS handles forward-overlap = LZ77
  semantics correctly): cutting copy instructions made the wall WORSE (+11%) ⇒ gz's
  copy already ≥ hardware-best ERMS. The brief's critique (rep-movsb biased-low for
  short overlapping copies) is moot: the short-dist path is COLD on logs (3a/Stage-1).
- Literal-decode recurrence (chainlat, raptorlake): gz 11.11 cyc/iter (1 sym/iter) vs
  igzip 30.07 cyc/iter (2 sym/iter ≈ 15.0 cyc/sym). **gz's per-symbol litlen-decode
  chain is SHORTER than igzip's** — so the litlen decode is not the gz disadvantage
  either (matches the prior "gz recurrence is tighter, 2 vs 3 serial loads").

## VERDICT

**COPY REGION (the brief's named target) = GATED PARITY CEILING, cross-arch.** Not a
prose floor — a deterministic ceiling:
  - hot copy loop is instruction-IDENTICAL to igzip (objdump);
  - chainlat cycles/iter Δ = 0.000 on raptorlake + znver3 + znver4;
  - the only encoding difference (SSE→VEX) is an abmeasure TIE (p=0.302), so even the
    latent AVX-SSE penalty is null here;
  - rep-movsb/ERMS oracle: gz copy ≥ hardware-best;
  - the brief's short-dist-overlap sub-hypothesis is FALSIFIED (that path is cold on
    logs; the hot copy is the long-match large-copy, identical to igzip).
There is NO copy-region lever. Closing further needs no copy change.

**The residual is NOT in the copy and NOT in the literal-decode recurrence** (both
gz-favorable-or-equal). By elimination + the T1 perf split it lives in:
  (i)  the per-block table build `rebuild_from` (13.7% at HEAD; gz separate pass vs
       igzip's inline build) — PARTIALLY captured by the unmerged
       `feat/perblock-build-speedup`; and/or
  (ii) the BACK-REF iteration's loop-carried chain (litlen + dist decode consume the
       bitbuf twice per back-ref; logs is copy-heavy so this is the dominant
       iteration — NOT chainlat'd this session: assembling the non-contiguous
       back-ref body is the next step); and/or
  (iii) `finish_decode_chunk_contig_native` (11.4%; per-chunk CRC/window).
These are OPEN HYPOTHESES (unvalidated) for the next leader — the copy is ruled out.

## ARTIFACTS / RE-RUN
- branch `feat/copyfloor-c` (UNMERGED, UNPUSHED): the byte-exact VEX TIE change.
- trainer: base `/root/gz-base-7603c180`, after `/root/gz-after-vex`,
  fulcrum+chainlat `/root/fulcrum-chainlat/target/release/fulcrum`.
- re-run copy chainlat: `fulcrum chainlat --bin <gz> --symbol run_contig
  --start 0xc4ea1 --stop 0xc4eba --cmp-bin <libisal> --cmp-symbol
  decode_huffman_code_block_stateless_04 --cmp-start 0x38e41 --cmp-stop 0x38e5a
  --assert-loop --mcpu raptorlake`.
- re-run gate: `fulcrum abmeasure --base-bin /root/gz-base-7603c180 --after-bin
  /root/gz-after-vex --corpus /root/floor/logs.gz --n 15 --core 3`.
