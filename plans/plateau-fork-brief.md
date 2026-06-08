# PLATEAU / FORK gate — the engine clean-rate is bounded; is it UN-CLOSEABLE in pure-Rust? (owner 2026-06-08)

## The validated picture (matched comparator, frozen host, advisor-vetted twice)
| run | wall | ratio vs rg | provenance |
|---|---|---|---|
| rapidgzip 0.16.0 (WITH_ISAL) | 376-383 ms | 1.000× | target |
| ocl_cf (gzippy + real ISA-L clean engine) | 404 ms | **0.945×** | CEILING, coverage 14/0, drain-split ✓ |
| production native (pure-Rust) | 434-440 ms | 0.857-0.870× | current |

- ocl_cf is the validated speed-UP ceiling (removal, not slope): copy-free, coverage
  14/0 verified, matched /dev/shm sink, window-absent-preserving, byte-exact.
- COVERAGE SYMMETRY ✓: native routing flip_to_clean=12 finished_no_flip=4
  window_seeded=2 == ocl_cf's 14 covered chunks. The 36ms gap is on the SAME chunks,
  pure symbol rate (no bootstrap-savings confound).
- DRAIN SPLIT ✓ (advisor-owed, RAN): GZIPPY_FOLD_NODRAIN/NOCRC ⇒ drain+CRC
  second-touch = ~0-1ms (N=21 best-of, sign-unstable, below noise). The bulk clean
  tail is u8-DIRECT (decode_clean_into_contig, no ring/drain — banked commit 0f5bc85b).
  ⇒ the 36ms is ~97% pure-Rust-vs-ISA-L SYMBOL RATE on decode_clean_into_contig.
- Two advisor passes UPHELD 404ms as a valid STEP-2 ceiling; C2 (21ms non-engine
  residual = bootstrap/placement) stands as a SEPARATE bar.

## STEP-2 techniques — every one done or structurally inapplicable (source+disasm verified)
1. **single-level L1-resident table** — DONE. lut_huffman.rs: short_code_lookup =
   [u32; 4096] = 16 KiB (ISAL_DECODE_LONG_BITS=12), the ISA-L igzip DECODE_LOOKUP
   geometry, single 12-bit root lookup + long_code_lookup fallback. Already L1-resident.
2. **BMI2 PEXT/BZHI** — MAXED. Native binary disasm: 1163 bzhi/pext/shrx/shlx idioms.
   (prior turn measured 433; the native-cpu build lowers the mask idioms to BMI2.)
   No manual-PEXT headroom.
3. **static-Huffman specialization** — NO HEADROOM. silesia-large is L9 = ~all
   dynamic-Huffman blocks; a precomputed fixed-Huffman const table targets ~zero blocks.
4. **FASTLOOP yield-elision / margin** — DONE. decode_clean_into_contig has the VAR_V
   software-pipelined fast loop (pre-decode-ahead, sym_count==1 fast byte, speculative
   8-byte packed store, direct base[*pos-d] back-ref), gated only off the perturbation
   knob; the resumable yield-check is amortized into the fast loop.
5. **table _mm_prefetch ahead of the dependent load** — STRUCTURALLY IMPOSSIBLE for a
   serial Huffman decode. The table index is `short_code_lookup[bits.peek() & 0xFFF]`,
   and `bits.peek()` is data-dependent on the PREVIOUS symbol's bit consumption (only
   known after this iter's decode+backref). There is no "ahead of the load" point — the
   index IS the dependent value. ISA-L's asm doesn't prefetch the table for the same
   reason; it pipelines the DECODE (which gzippy already does via `pre`). Disasm shows
   39 prefetch instrs already emitted elsewhere.
6. **store side** — TIE'd + KEPT 7a (prior turn): packed multi-literal fast loop covers
   ~69% of clean events; store is not the binder.
7. **bounds checks** — NONE in the hot loop (disasm: 0 panic/bounds-check strings; the
   unsafe contig path elides them).

## THE PLATEAU CLAIM
The pure-Rust clean engine (decode_clean_into_contig) is a FAITHFUL ISA-L igzip
algorithm port, already at the LLVM codegen ceiling: L1 single-level table, BMI2 maxed,
no bounds checks, software-pipelined fast loop, store TIE'd, prefetch structurally
inapplicable. The residual 36ms (0.870×→0.945×) is the intrinsic
**hand-scheduled-asm (ISA-L) vs LLVM-codegen (pure-Rust)** gap on an identical
algorithm. The ONLY remaining pure-Rust lever to match igzip's asm is to TRANSLITERATE
igzip's hot loop into inline assembly (CLAUDE.md-authorized) — a large, high-risk
undertaking NOT in the STEP-2 technique list, and exactly the technique-grind the
charter warns against past a validated bound.

## THE FORK
- **goal #1 (gzippy-native, no-FFI, 1.0× bar)**: bounded at **0.870× rg** today; even a
  PERFECT pure-Rust clean engine (= ISA-L rate, the ocl_cf ceiling) reaches only
  **0.945× rg** because of the SEPARATE 21ms non-engine residual (C2). So no-FFI 1.0× is
  HARD-BOUNDED at 0.945× by the engine ceiling ALONE, and at <0.945× in practice unless
  BOTH (a) the asm-vs-LLVM 36ms is closed via inline-asm transliteration AND (b) the 21ms
  non-engine residual is closed.
- **goal #2 (gzippy-isal, WITH-FFI)**: ocl_cf shows the ISA-L clean engine reaches
  0.945× rg — itself still 21ms short of rg, i.e. even the FFI path does not tie rg on
  this matched comparator (rg's edge over ocl_cf is the 21ms non-engine residual:
  marker-region pure-Rust decode + bootstrap structure + scheduling).

## QUESTIONS (try to BREAK the plateau)
Q1. Is the plateau claim valid — i.e. are all 5 STEP-2 techniques genuinely done/
    inapplicable, especially #5 (is table-prefetch REALLY structurally impossible, or is
    there a decode-2-ahead pipeline that would let it work)?
Q2. Is the 36ms genuinely the asm-vs-LLVM intrinsic gap, or is there an un-tried
    ALGORITHMIC pure-Rust technique (not in the list) with real headroom before
    inline-asm? (e.g. wider root table, 2-level→1-level for dist, decode-2-symbols-ahead.)
Q3. Is this a REAL, HARD-BOUNDED fork to escalate (no-FFI 1.0× requires inline-asm
    transliteration of igzip's hot loop AND closing the 21ms non-engine residual), or is
    there a cheaper unblocked path I'm missing?
Q4. Frozen host is 1.4 GHz pinned. Does measuring the plateau at low freq mis-state it
    (engine gap larger at turbo)? Should the fork be escalated with a turbo-freq number?

Be decisive. End with: ESCALATE-FORK / KEEP-GRINDING(name the technique) / NEED-MORE-DATA(name it).
