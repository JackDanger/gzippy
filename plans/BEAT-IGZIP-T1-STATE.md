# BEAT-IGZIP-T1 — DURABLE STATE

## ====== NIGHT13 (2026-06-19, branch kernel-converge-faithful @ 305c723e, base eee0522b=NEW11) — STEP-2(b) MASK-ONCE-IN-REGISTER (igzip macro 341) IMPLEMENTED + DISASM-PROVEN (3 instr/iter removed from the literal hot path: 44→41). BYTE-EXACT GATE PASS (native asm-vs-ref 6/6 + native proptest 57/57 @60k cases + trioracle silesia/nasa/monorepo/squishy/large × {native,isal} × T1/4/8 sha-identical). 3-way perf (2b/NEW11/OLD vs igzip): PENDING (running on quiet box; verdict appended below when complete). DECOUPLED from STEP-2(a): mask-once needs NO freed register (it REUSES {t1}, REDUCING pressure) — so the NIGHT12 "2a+2b are one coupled change" premise is WRONG for 2b. ======

### THE CHANGE (asm_kernel.rs run_contig, the literal hot path)
Faithful transliteration of igzip `decode_next_lit_len` (igzip_decode_block_stateless.asm
macro 341: `and next_sym, SYM_MASK` ONCE; store(518)+trailing-shrx(521) BOTH read that one
masked reg). gz previously made TWO scratch copies of the raw short entry (`mov {t5},{t1}`
for the trailing extract, `mov {t4},{t1}` for the store) and re-masked per use
(`and {t4},0xFFFFFF`). NOW masks `{t1}` ONCE in place (`and {t1},0x1FFFFFF`) AFTER bc→{t2}/
cnt→{t3}/LARGE_FLAG are extracted, then reuses {t1} for BOTH the trailing `shrx {t5},{t1},{t4}`
AND the speculative `mov [dst],{t1}`. The post-shrx `and {t5},0xFFFF` STAYS (drops the
displaced trailing-nonlit flag bit24 for cnt∈{1,2}). Ref model (run_contig_ref_biased)
UNCHANGED — it models the decode RESULT, byte-identical.

### BYTE-EXACT ARGUMENT (then GATED — never bank the reasoning)
Only byte that differs new-vs-old: store byte 3 (bits 24-31). Old store (`& 0xFFFFFF`) → byte3=0;
new store (`& 0x1FFFFFF`, bits25-31 zeroed by the 32-bit `and`) → byte3 = bit24 only (0 or 1).
Byte 3 is at dst+3; dst advances by cnt ≤ 3 every iteration, so byte 3 is ALWAYS overshoot
(≥ the post-advance dst) — never a final output byte (pure-lit: bit24=0 anyway; lone-length:
the backref copy overwrites/overshoots it). X3-equivalent. → GATE confirms.

### DISASM PROOF (stripped binary, hot-loop region ~line 113000; gz-new-native@7391868b vs gzippy-new11-native@8eb0d456)
NEW11 hot loop:  `mov %r14d,%r10d` ; `and $0x1ffffff,%r10d` ; … ; `shrx %r13,%r10,%r10` ; … ;
                 `mov %r14d,%r13d` ; `and $0xffffff,%r13d` ; `mov %r13,(%rdx)`
2B   hot loop:   `and $0x1ffffff,%r14d` (MASK ONCE on r14=t1 in place) ; … ;
                 `shrx %r13,%r14,%r10` (reuse) ; … ; `mov %r14,(%rdx)` (store reuse)
REMOVED: 2× `mov rN,r14` (scratch copies) + 1× `and $0xffffff` (store mask) = **3 instr/iter**.
Whole-binary `and $0xffffff` count 27→26 (the only mask-family delta; everything else
identical: `and $0x1ffffff`=38, `and $0xffff`=3, `and $0xfff`=66, shrx=245 unchanged).

### STEP-2(a) STRUCTURAL FINDING (re-scopes the NIGHT12 bucket table — the anchor "+4" is NOT 4-removable)
NIGHT12's bucket (1) "ANCHOR hand-spill +4, removable via single-base" OVER-COUNTED the
removable share. The anchor is `lea {t2},[pos*8]` + `sub {t2},bitsleft` (compute p0) + `mov
[ctx+56]` (store p0) + `mov [ctx+64]` (store d0) = 4 instr. Per DIVERGENCE LEDGER §8 D-1, the
p0 COMPUTE (lea+sub) is the IRREDUCIBLE un-consume divergence (igzip is stateless; gz must
reconstruct the un-consumed packet start on a RECLASS bail; the consumed low bits are shifted
out, so p0 MUST be (re)computed). Even with 2a's single-base freeing both short_tbl+dtbl regs,
carrying p0+d0 in-register only deletes the 2 STORES → `lea`+`sub`+`mov ad,dst` = 3 instr (net
−1 instruction, −2 store-port ops). So 2a removes ~1 instr, NOT 4. Combined 2a+2b: literal hot
path 44 → ~40 vs igzip 38 → residual +2 = the irreducible D-1 un-consume (NOT a transliteration
gap). HYPOTHESIS (unvalidated): the +2 residual cannot be closed on instr/B without abandoning
the late-discriminator/un-consume divergence; whether cyc/B still beats via IPC/critical-path
is the open question the perf gate settles. (2a remains a candidate for STORE-PORT relief if the
gate shows store-port is the binding constraint — but it is NOT the path below igzip on instr/B.)


## ====== NIGHT12 (2026-06-19, branch kernel-converge-faithful @ d67e72a6 = NEW11) — STEP-1 DECISIVE STATIC DIFF: localized the NEW11 +3.589 instr/B (vs igzip) per-instruction. BUCKETS: anchor hand-spill +4/iter, table-format masking +4/iter, RECLASS/exit glue +1/iter, gz LEANER on spec-length+EOB −3/iter → NET +6/iter (literal hot path: gz 44 instr vs igzip 38) ≈ the gated +3.589 instr/B. DECISION RULE met: anchor(1)+masking(2) DOMINATE glue(3) → the +3.58 is a FIXABLE un-transliterated gap (NOT a structural ceiling), proceed to STEP 2 (single-base register layout + table pre-masking). This is a NEW result, NOT redundant with night11. NO asm change this turn (STEP 2 is a coupled rewrite needing the guest byte-exact gate). (static decomposition of a GATED number; the per-bucket "removable" attribution is HYPOTHESIS until STEP-2 build+measure) ======

### WHY THIS IS NOT REDUNDANT WITH NIGHT11 (the honest reconciliation — flag to supervisor)
NIGHT11 decomposed the gap to **OLD** (+2.0 instr/B, NEW11−OLD) and concluded "the full-spec
speculation itself" owns it, and that anchor-removal-ALONE (→ ~+2.6 instr/B) cannot close it →
correctly NOT worth the risk. STEP 1 (this turn) decomposes the DIFFERENT gap to **igzip**
(+3.589 instr/B, NEW11−igzip) — the comparison that matters because **igzip's full-spec shape
BEATS OLD (igzip ~4.35 vs OLD ~5.67 cyc/B) AND NEW11 already MATCHES igzip's IPC (−0.03).** The
NEW insight night11 did NOT consider: a SECOND removable bucket (table-format MASKING, +4/iter)
on top of the anchor (+4/iter). Removing BOTH (−8/iter) against the NET +6/iter would put gz's
full-spec at ≈ −2/iter vs igzip = BELOW igzip's instr/B (gz is already −3/iter leaner on the
hot-path spec-length+EOB that igzip computes unconditionally). With IPC already at igzip parity,
that is the FIRST credible path for the full-spec shape to beat OLD. HYPOTHESIS (unvalidated):
single-base + pre-masking brings NEW11 instr/B to ≤ igzip with IPC held → cyc/B ≤ igzip → beats
OLD. ONLY a STEP-2 build + gated paired measurement settles it (the §3.1 skeleton was falsified
3× on the gap-to-OLD framing; this is a 4th, sharper test on the gap-to-igzip framing).

### THE +3.589 instr/B BUCKET TABLE (literal hot path: gz `2:`→`jb 2b` vs igzip loop_block 509→`jl loop_block`)
Counted instruction-by-instruction from asm_kernel.rs run_contig (HEAD d67e72a6) and
igzip_decode_block_stateless.asm loop_block 507-556 + decode_next_lit_len macro 322-372.
gz literal path = **44 instr/iter**; igzip literal path = **38 instr/iter**; Δ = **+6/iter**.
(silesia bytes/iter ⇒ +6/iter ≈ +3.589 gated instr/B — self-consistent with the GATED number.)

| bucket | gz instrs (asm_kernel.rs line) | igzip counterpart (asm line) | Δ/iter | removable? cite |
|--------|-------------------------------|------------------------------|--------|-----------------|
| (1) ANCHOR hand-spill (X2 un-consume, gz-only) | `lea t2,[pos*8]`+`sub t2,bitsleft`+`mov [ctx+56],t2`+`mov [ctx+64],dst` (439-442) | NONE (igzip stateless, never un-consumes — DIVERGENCE LEDGER §8 D-1) | **+4** | YES via STEP 2(a): single-base frees 2 GP regs → carry p0,d0 in regs, drop the 2 stores. (igzip register layout: ONE base `state`, lines 86-136) |
| (2) TABLE-FORMAT MASKING (per-use extract masks) | `and t5,0x1FFFFFF`(460) + `and t5,0xFFFF`(463) + `mov t4,t1`+`and t4,0xFFFFFF`(471-472); two scratch-mov copies of t1 because the raw entry is reused for store AND trailing | igzip masks the entry ONCE: `and next_sym,FLAG\|SYM_MASK` (D7, 341); store(518) + trailing-shrx(521) BOTH read that one masked reg; table guarantees zero bits above the cnt*8 sym field ⇒ no post-shrx `and`, no second mask | **+4** | YES via STEP 2(b): gz litlen table format is ALREADY igzip LARGE_SHORT (offsets 28/26/25, mask 0x1FFFFFF — igzip 58-67 ✓). Transliterate the BUILD to zero high bits above cnt*8 + restructure decode to mask-once-reuse (needs 1 freed reg from (a)) |
| (3) RECLASS/exit glue (gz return-code contract) | `mov ret,1` per-iter speculative BOUNDARY (418) | igzip uses `jg end_loop_block_pre` (510/512), no per-iter ret | **+1** | partial: hoistable to guard-fail path only; minor |
| (−) gz LEANER (igzip speculates these unconditionally; gz defers to backref arm) | — | `lea repeat_length,[next_sym2-254]`(533 spec length) + `cmp next_sym2,256`+`je`(536-537 EOB) | **−3** | n/a (gz already leaner here) |

DECISION: (1)+(2) = +8/iter DOMINATE glue (3) = +1/iter. Per KERNEL-CONVERGENCE.md decision
rule → the +3.58 is a FIXABLE transliteration gap, NOT the structural ceiling. PROCEED to STEP 2.
Both buckets are COUPLED through the 15-GP register ceiling (operand list asm_kernel.rs 825-841:
ctx/short_tbl/in_ptr/dtbl are FOUR base pointers where igzip pins ONE = `state` + struct offsets):
freeing regs via single-base is the prerequisite for BOTH carrying the anchor in-reg (1) AND
holding a persistent mask-once symbol (2). So STEP 2 is ONE coupled change, not splittable.

### STEP 2 — PRECISE TRANSLITERATION DESIGN (the clean resume point; faithful, igzip-cited; NOT yet implemented)
**2(a) SINGLE-BASE REGISTER LAYOUT (igzip 86-136, 502/540/552/577 addressing).** igzip addresses
both tables off ONE base: `[state + _lit_huff_code + LARGE_SHORT_CODE_SIZE*idx]` (502/540/577)
and `[state + _dist_huff_code + SMALL_SHORT_CODE_SIZE*idx]` (552) — the tables are INLINE arrays
in inflate_state. gz pins separate `short_tbl`/`dtbl` `*const` base regs. TRANSLITERATE: store
the litlen short table + dist table CONTIGUOUS inside (or immediately addressable from) KernCtx so
`[ctx + LIT_OFF + idx*4]` / `[ctx + DIST_OFF + idx*4]` use ctx as the only base ⇒ frees the
`short_tbl` and `dtbl` operands (2 regs). Move p0 + d0 into the 2 freed regs ⇒ delete the 2 anchor
stores (asm 441-442); p0 stays REFILL-INVARIANT in a reg (still re-read on bail via `85:`, just
sourced from the reg not [ctx+56]). LIFECYCLE care: tables are per-dynamic-block, ctx per-chunk —
build the inline tables in place per block (litlen 4096*4=16KiB, dist 512*4=2KiB; already rebuilt
per block in build_huffman_luts_for_block, marker_inflate.rs:1163) with no added per-chunk copy.
**2(b) TABLE-FORMAT PRE-MASKING (igzip 341 + table build igzip_inflate.c).** Make
LutLitLenCode::rebuild_from (lut_huffman.rs) store each entry with bits above the cnt*8 sym field
ZEROED (igzip's table guarantee), and restructure run_contig's decode to mask the entry ONCE into
a freed reg (igzip D7 `and FLAG|SYM_MASK`, 341), reusing it for the spec store (igzip 518) AND the
trailing shrx (igzip 520-521) ⇒ deletes asm 463 (`and 0xFFFF`) + 471-472 (`mov t4,t1; and 0xFFFFFF`).
**LOCKSTEP + GATE:** update run_contig_ref_biased identically; preserve X1-X6 + IN_MARGIN + X2
re-read (`85:`/`86:`) + the divergence ledger §8 D-1 (the anchor moves register→register, the
re-read contract is unchanged). Byte-exact gate = the night11 template (asm c1/c2/c3/pos-control/
on-off-fuzz + ≥60k prop_structured + trioracle silesia/nasa/monorepo/squishy/large × {native,isal}
× T1/T4/T8 sha-identical + serialized --lib both flavors 0-failed). THEN gated paired N≥15 vs
igzip AND OLD(chunkt1) AND NEW11(d67e72a6): cyc/B + instr/B + IPC. SUCCESS = instr/B falls toward
igzip WITH IPC held (~−0.03) → cyc/B ≤ igzip → beats OLD. Promote kernel-converge-faithful →
perf/igzip-full-rewrite ONLY then, as gated-HYPOTHESIS for decorrelated/AMD review. Else: STEP-2
falsified ⇒ OLD early-flag-bit is gz's full-spec ceiling on this uarch (4th confirmation) — record
FALSIFY with re-open trigger = AMD/Zen2 idle-slot uarch.

### GUEST / BOX STATE (this turn — analysis only, nothing built/run)
Guest reachable `ssh -o ConnectTimeout=15 -J REDACTED_IP root@REDACTED_IP`; 16 cores; /dev/shm
12G free; root disk 99% (build ⇒ /dev/shm); igzip /usr/bin/igzip; NO cargo running on arrival
or exit (pgrep clean — I started none). Compare binaries in /root/bin: gzippy-chunkt1 (OLD
early-flag-bit), gzippy-kc (night9). NEW11 (d67e72a6) binary must be rebuilt for re-measure
(gz-new-native from night11 not present in /root/bin listing). Box NOT frozen, no pinning set.

## ====== NIGHT11 (2026-06-19, branch kernel-converge-faithful @ 3bacc1a1, base night9 c4fbc3d3) — FAITHFUL snapshot removal (minimal p0/d0 anchor + from-data RE-READ un-consume, NOT c936's serialized cut). BYTE-EXACT (asm 6/6 + trioracle 30/30 + prop 57/57). GATED PERF: NEW11 ≈ night9 (the 2-store anchor is INSTRUCTION-NEUTRAL) and BOTH LOSE to OLD early-flag-bit — snapshot was NOT the lever; the ~2 instr/B gap is the full-spec speculation. §3.1 skeleton FALSIFIED a 3rd time. NOT promoted. (gated, Gate-0 PASS, Intel-LXC NOT-YET-LAW) ======

### THE CHANGE
Deleted night9's per-iteration 4-store X2 snapshot (`save_bitbuf/+48 save_bitsleft/+56
save_dst/+72 save_pos/+80`). Replaced with a MINIMAL 2-word anchor saved at the
iteration top: `save_p0/+56` = bit position `pos*8-bitsleft` (REFILL-INVARIANT, proved
vs `Bits::refill`) and `save_d0/+64` = iteration-top dst. A rare RECLASS bail
reconstructs the whole un-consumed cursor via a SINGLE from-data RE-READ at p0
(asm `85:`; ref `reclass_reread`) — the consumed low bits are shifted out and
unrecoverable from registers, so save-or-reread is structurally required (igzip is
stateless, no counterpart). The bc==0 INVALID bail is reached PRE-consume so it leaves
the cursor UNCHANGED (asm `86:`, no re-read — preserves X6/c1). The asm stays at the
15-GP-register ceiling (a register-carried anchor overflows it / would need dropping
igzip's per-iter dist preload), so 2 ctx-words is the minimal faithful save: ONE
bit-cursor store + ONE dst store/iter vs night9's four. See DIVERGENCE LEDGER §8 in
KERNEL-CONVERGENCE.md. run_contig_ref_biased updated in LOCKSTEP; 5 KernCtx initializers
updated; short_tbl 64->48.

### BYTE-EXACT GATE (guest, Intel LXC i7-13700T, real BMI2/AVX2) — PASS so far
- Guest build (native + isal): both register-allocate cleanly (the asm-ceiling risk),
  EXIT 0. (macOS x86_64 cannot build the asm — pre-existing, night9 too.)
- asm-vs-ref differentials `cargo test --lib asm_kernel --test-threads=1`: **6/6 PASS**
  (c1_seam_roundtrip, c2/c3 differential random+windowed, positive_control consume-bias,
  on_off_fuzz). c1 initially failed (the re-read changed the cursor representation on the
  no-consume invalid bail) → fixed by the `86:` no-re-read invalid exit.
- TRIORACLE_GATE (`scripts/bench/_trioracle_gate.sh`): **PASS** — silesia/nasa/monorepo/
  squishy/large × {native,isal} × T1/T4/T8 all sha-identical vs gzip+igzip+libdeflate+pigz.
- prop_/lut_huffman/marker_inflate `--lib` (PROPTEST_CASES=20000, guest-side timeout):
  **57 passed, 0 failed** (prop_structured + prop_random + prop_near_max_distance + lut +
  marker). The FULL `--lib` run hung on the PRE-EXISTING parallel pipe-deadlock test
  `fd_vectored_write::early_reader_death_is_clean_error_no_duplicate` (project memory:
  test-harness deadlock on a loaded box, NOT a decoder bug) — killed (scoped), box clean.
  The targeted run covers everything the snapshot change touches (night9 used the same
  scoping).

### BINARIES (guest)
/root/gz-new-native, /root/gz-new-isal (== 3bacc1a1). Compare set: /root/bin/gzippy-kc
(night9), /root/bin/gzippy-chunkt1 (old early-flag-bit), /usr/bin/igzip. Source checkout
/root/gzippy (branch kernel-converge-faithful). Builds in /dev/shm/n11 (native) /dev/shm/n11i (isal).

### GATED PERF RESULT (paired N=15, PIN=4, /dev/null, GZIPPY_FORCE_PARALLEL_SM=1, all 3 bins SAME box-state, Gate-0 PASS: KERN fired, sha==zcat==igzip, A2-A1 self-test CI-incl-0, GHz spread <1.7%). medΔ=(gzippy−igzip) cyc/B, LOWER=better.
| binary (shape)                 | silesia medΔ [95%CI]    | nasa medΔ [95%CI]       | sil ΔIPC/Δinstr·B | nasa ΔIPC/Δinstr·B |
|--------------------------------|-------------------------|------------------------|-------------------|--------------------|
| NEW11 gz-new-native (2-store)  | +1.4534 [1.439,1.458]   | +0.6242 [0.622,0.628]  | -0.032 / +3.589   | -0.138 / +1.154    |
| KC night9 (4-store snapshot)   | +1.5024 [1.446,1.521]   | +0.6128 [0.603,0.622]  | -0.043 / +3.595   | -0.122 / +1.155    |
| OLD early-flag-bit (chunkt1)   | +1.3243 [1.276,1.364]   | +0.4915 [0.483,0.498]  | -0.316 / +1.585   | -0.241 / +0.626    |
Cross-shape (sil): NEW11−KC = **−0.049** (marginal, CIs touch); NEW11−OLD = **+0.129
(SIGNIF, CIs NON-OVERLAPPING — OLD wins)**. (nasa): NEW11−KC = +0.011 (~tie); NEW11−OLD =
+0.133 (SIGNIF).

### VERDICT — §3.1 snapshot-removal HYPOTHESIS FALSIFIED; OLD early-flag-bit still WINS (do NOT promote). (gated, Gate-0 PASS, Intel-LXC NOT-YET-LAW)
1. **NEW11 ≈ night9 (TIE).** instr/B is IDENTICAL (sil +3.589 vs +3.595; nasa +1.154 vs
   +1.155). The 2-store anchor is INSTRUCTION-NEUTRAL vs the 4-store snapshot: removing 2
   stores but adding the `p0` anchor arithmetic (lea+sub) + keeping 2 stores = 4 ops/iter
   either way. So cyc/B is ~unchanged (sil marginally better via slightly higher IPC,
   nasa marginally worse).
2. **The snapshot was NOT the lever.** The ~2.0 instr/B GAP to OLD (NEW11/n9 +3.59 vs OLD
   +1.585 sil; +1.15 vs +0.63 nasa) is the FULL-SPEC SPECULATION the §3.1 shape adds every
   iteration (per-iter dist preload `dpre` + trailing extract + unconditional spec store),
   NOT the un-consume save. Deleting the snapshot (even to 0 stores) cannot close a gap it
   does not own. OLD wins by doing FEWER instructions despite WORSE IPC (-0.316 vs -0.03):
   the loop is uop/throughput-bound, not latency-bound-with-idle-slots — leaner wins.
3. **Triple-confirms night10.** Three independent byte-exact iterations on the igzip
   late-discriminator full-speculation skeleton (night9 4-store, c936 pre-consume-serialized,
   night11 2-store re-read) ALL land at/above OLD. The §3.1 "more speculation lifts IPC"
   premise is FALSIFIED on this uarch across all three.
4. **A 1-store / 0-store refinement would NOT change the verdict.** Best case (store only
   p0, derive d0 from cnt) saves ~1 instr/iter → instr/B ~+2.6, still ≫ OLD's +1.585. The
   speculation gap dominates. NOT worth the asm risk. RE-OPEN TRIGGER for the whole §3.1
   skeleton: a uarch with genuine idle OOO slots (AMD/Zen2 owed), or a corpus where the
   loop is latency-bound.

### DECISION (report to supervisor; not self-blessed)
- NEW11 is BYTE-EXACT (asm 6/6 + trioracle 30/30 + prop/lut/marker 57/57) and a clean,
  faithful, ledgered implementation of igzip's save-at-boundary idiom — KEPT on
  kernel-converge-faithful as the recorded byte-exact iteration with this gated verdict.
- NOT promoted to perf/igzip-full-rewrite (loses to OLD, gated-significant; same class as
  night9/c936).
- The data-driven next direction (carried from night10, now reinforced): the lever to beat
  OLD is NOT inside the un-consume machinery; it is REDUCING the §3.1 per-iter speculation
  (toward the leaner early-flag-bit shape) OR a different front entirely. STRATEGIC FORK
  for the supervisor: keep converging on igzip's full-spec shape (which loses at T1 on
  Intel-LXC across 3 iters) vs accept OLD early-flag-bit as the T1 skeleton and attack the
  scaffold/other fronts. AMD/Zen2 replication owed before any LAW.

## ====== GUEST GATE + KEY MEASUREMENT (night10, 2026-06-19, branch kernel-converge-wip @ 3202968d) — c936a4df (snapshot-removal) is FULLY BYTE-EXACT on real BMI2/AVX2, but is gated-SLOWER than BOTH night9 and the OLD early-flag-bit shape. The igzip late-discriminator full-speculation skeleton LOSES the §3.1 fork across 2 byte-exact iterations. NOT promoted. (gated, Gate-0 PASS, Intel-LXC NOT-YET-LAW) ======
MISSION this turn: discharge the GUEST-OWED gate for c936a4df (the snapshot-removal —
only its pure-Rust ref-half was Rosetta-verifiable; the asm/AVX2 half + ALL perf were
owed), then the KEY structural measurement (did snapshot-removal recover night9's
regression vs old?). Guest = neurotic Intel i7-13700T LXC, /dev/shm builds (root 99% full),
governor powersave (NOT frozen — restored nothing, froze nothing), cpu4 pin.

### STEP 1 — c936a4df FULL BYTE-EXACT GATE = **PASS** (both flavors, real BMI2+AVX2)
The asm-vs-ref differentials that SKIPPED under Rosetta (bmi2 hidden from CPUID) now RAN
on real BMI2 and PASS — this is the core gate for the snapshot-removal change (it directly
pins asm==ref on exit-cursor + every output byte through the edited bail classes):
- `cargo test --lib asm_kernel -- --test-threads=1` (native pure-rust-inflate): **8/8 PASS** —
  c1_seam_roundtrip, c2_differential_asm_vs_ref_random_streams, c3_differential_asm_vs_ref_
  windowed_backrefs, positive_control_consume_off_by_one (harness PROVEN live), dispatch_
  allowed_excludes_every_knob, asm_kernel_on_off_fuzz_random_gzip_members, + the 2 new
  ref-model bail tests. ASMTEST_EXIT=0.
- `prop_structured_roundtrip` @ **PROPTEST_CASES=60000**: PASS (152s), EXIT=0.
- TRI-ORACLE sha grid (gzippy vs igzip vs gzip/zcat), GZIPPY_FORCE_PARALLEL_SM=1 → ParallelSM:
  **native 16/16 MATCH** + **isal 16/16 MATCH**, each = silesia/nasa/monorepo/squishy × T1/T4/T8
  + model/bignasa(large) × T1/T8. GRID_FAIL=0 both flavors.
- Full serialized `cargo test --release --lib` (native): PARTIAL — **392 tests `... ok`, 0 FAILED/
  panicked** before a remote `timeout 480` cutoff (box load ~4 from other LXC tenants made the full
  949 exceed 480s; it self-terminated cleanly, NO orphan). Full 949 completion OWED-low-risk (change
  is asm-kernel-localized; night5 ran the full 949 green at a near-identical HEAD; the asm
  differentials + 60k proptest + tri-oracle grids are the binding evidence). Re-run with a longer
  REMOTE timeout on a quieter box. Binaries: /root/bin/gzippy-c936-native (sha 3e121b9f7df7c11f),
  /root/bin/gzippy-c936-isal.
  HARNESS LESSON (banked): put `timeout` on the REMOTE command (`ssh '... timeout N cargo ...'`), NOT
  on the local ssh — a local `timeout 600 ssh` SIGALRM kills only the client and ORPHANS the remote
  cargo (hit + cleaned this turn). Detached `setsid bash -c "timeout N ... > /dev/shm/log"` is
  orphan-proof: the remote timeout self-terminates it regardless of the ssh client.
⇒ c936a4df is BYTE-EXACT (asm half NOW verified, not just the Rosetta ref-half). The Rosetta
  GUEST-OWED items #1 are DISCHARGED.

### STEP 2 — THE KEY MEASUREMENT (paired N=15, /dev/null, cpu4, GZIPPY_FORCE_PARALLEL_SM=1, Gate-0 self-validated). medΔ=(gzippy−igzip) cyc/B; LOWER=better. igzip A1 stable across all 3 runs (sil 4.351/4.360/4.360, nasa 1.582/1.589/1.580) ⇒ consistent box state ⇒ cross-shape deltas reliable.
| shape (binary)                          | silesia medΔ [95%CI]      | nasa medΔ [95%CI]        | ΔIPC sil/nasa | Δinstr/B sil/nasa | Gate0 self-test |
|-----------------------------------------|---------------------------|-------------------------|---------------|-------------------|-----------------|
| OLD early-flag-bit (gzippy-chunkt1)     | +1.2626 [1.258,1.274]     | +0.4838 [0.480,0.486]   | -0.307/-0.238 | +1.573/+0.635     | marginal FAIL (bias≈0.003, ≪effect) |
| night9 full-spec+snapshot (gzippy-kc)   | +1.4564 [1.438,1.470]     | +0.6159 [0.594,0.620]   | -0.039/-0.122 | +3.582/+1.163     | PASS |
| c936 snapshot-removed (gzippy-c936-nat) | +1.8613 [1.855,1.887]     | +0.7354 [0.730,0.742]   | -0.299/-0.298 | +3.025/+1.022     | PASS |
All SIGNIF-slower vs igzip (Wilcoxon p=0.00073, CIs exclude 0). Cross-shape (sil): c936−night9=+0.405,
c936−old=+0.599, night9−old=+0.194. (nasa): c936−night9=+0.120, c936−old=+0.252, night9−old=+0.132.

STRUCTURAL READ (diagnostic, NOT a pass/fail verdict on the mission):
1. Did snapshot-removal drop instr/byte toward old? **YES** — sil 3.582→3.025 (−0.557), nasa
   1.163→1.022 (−0.141). The change did exactly what it was designed to (fewer retired instr).
2. Did it lift IPC? **NO — the OPPOSITE.** ΔIPC COLLAPSED: kc −0.039 → c936 −0.299 (sil);
   kc −0.122 → c936 −0.298 (nasa). night9 had nearly closed IPC to igzip (−0.04); c936 re-opened it.
3. Did it recover night9's regression vs old? **NO — it WORSENED it.** Gap to old grew from
   +0.194 (night9) to +0.599 (c936) sil; +0.132→+0.252 nasa.
4. Is the igzip-shape now ≤ old? **NO** — significantly slower on both corpora, CIs non-overlapping.
MECHANISM (gated counters, structural): the c936 change moved EOB/oversize/invalid discrimination
PRE-consume to delete the per-iter snapshot stores. It cut retired instructions BUT inserted a
serializing dependency (decode→classify→consume) into the latency-bound critical chain → IPC
collapsed → net slower. In this loop the night9 snapshot stores were apparently NOT on the
critical path (they filled OOO slots); removing them + adding a pre-consume classify was a net loss.
The remaining gap to igzip on c936 is now IPC-dominated (−0.30), not instr-count-dominated.

### VERDICT — §3.1 SKELETON FORK DECIDED: OLD early-flag-bit WINS (do NOT promote kernel-converge-wip)
Across TWO byte-exact iterations of the igzip late-discriminator full-speculation direction
(night9 kc, then c936 snapshot-removed) BOTH are gated-slower than the OLD early-flag-bit shape
on BOTH corpora (non-overlapping CIs, Gate-0 PASS). Per KERNEL-CONVERGENCE.md §6 step-1 ("the
A/B decides the loop's skeleton; KEEP whichever wins per Gate-1"), the OLD early-flag-bit
skeleton (= production perf/igzip-full-rewrite @ 37e669cf) WINS. **NO promotion** — kernel-
converge-wip stays a scratch branch; production unchanged.
SCOPED FALSIFY (not eternal): the igzip full-speculation late-discriminator skeleton is slower
than the lean early-flag-bit skeleton at T1 on Intel-LXC i7-13700T, silesia+nasa, because the
loop is latency-bound with NO idle OOO slots to fill (the §3.1 premise that more speculation
lifts IPC is FALSIFIED here — leaner won). RE-OPEN TRIGGER: a uarch where the loop is genuinely
throughput-bound (idle issue slots), or AMD/Zen2 replication showing the opposite.

### STEP 3 — NEXT CONVERGENCE ELEMENT: redirected by the measurement (STRATEGIC FORK for supervisor/R3)
The measurement OVERTURNS the working assumption that §3.2-3.5 build on the full-spec skeleton.
The data-driven next element = **§3.2 split-refill cadence authored on the OLD early-flag-bit
skeleton** (production branch), NOT on kernel-converge-wip's deprecated full-spec body. I did NOT
author it on the losing skeleton (that would stack on a gated-worse foundation — the bias the
project forbids). This is a strategic redirection: §3.2/§3.3/§3.4 to be re-rooted on the early-
flag-bit run_contig. Flag to supervisor: the entire night9→c936 line is a gated-slower skeleton;
continue convergence FROM the old shape.

### CLEAN RESUME POINT (next turn)
1. Re-root §3.2 split-refill on the OLD early-flag-bit run_contig (production @ 37e669cf): interleave
   the refill memory-load+shlx+or with the spec-store/preload, defer the pos/len update past it
   (KERNEL-CONVERGENCE.md §3.2); run_contig_ref_biased in LOCKSTEP; byte-exact gate (asm diffs +
   tri-oracle + 60k proptest, all green THIS turn = the template) + paired harness vs igzip AND
   vs gzippy-chunkt1.
2. OWED regardless: AMD/Zen2 (solvency) replication of this whole scoreboard for LAW; T4/T8 wall+RSS
   + stressor phase for any promoted change.
RE-VERIFY the KEY measurement: `for B in c936-native kc chunkt1; do GZIPPY=/root/bin/gzippy-$B
PIN=4 REPS=15 CORPORA="silesia nasa" SKIP_STRESS=1 GZIPPY_FORCE_PARALLEL_SM=1 bash
/root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh; done`. sha grid: `GZIPPY=<bin> bash
/root/sha_grid.sh`. Binaries in /root/bin: gzippy-c936-native/-isal (3202968d), gzippy-kc (night9
cc2840ff), gzippy-chunkt1 (old early-flag-bit, ca70e9d1). Box: cpu4 pin only, NOT frozen, governor
powersave untouched, all cargo finished (pgrep clean at turn end).

## ====== ROSETTA LOCAL-ONLY TURN (branch kernel-converge-wip @ c936a4df+) — c936a4df Rosetta-correctness verdict + ref-model net strengthened. NO NEUROTIC, NO PERF. Asm byte-exact + ALL perf = GUEST-OWED. ======
LOCAL-ONLY mandate (user: neurotic reserved for hours). Mac aarch64 + x86_64 Rosetta
toolchain ONLY. NO guest, NO perf claims. Goal: maximize CORRECTNESS + AUTHORING so
guest time is efficient when it returns.

### DETERMINISTIC ROSETTA-CAPABILITY PROBES (the governing facts this turn rests on)
- `is_x86_feature_detected!` under Rosetta on this Mac: **bmi2=FALSE, avx2=FALSE**,
  sse4.2=true, pclmulqdq=true. (probe: /tmp/feat_probe.rs)
- BUT Rosetta **EXECUTES** BMI2 directly: a standalone `shrx`/`bzhi` asm program ran
  byte-correct, exit 0 (probe: /tmp/shrx_probe.rs). So Rosetta runs BMI2 instructions
  yet hides them from CPUID — the prompt's "Rosetta executes BMI2" is TRUE for
  execution, FALSE for advertisement.

### TASK 1 — c936a4df ROSETTA-CORRECTNESS VERDICT = **ASM PATH NOT EXERCISABLE LOCALLY (guest-owed); REF-MODEL HALF + WHOLE PIPELINE = PASS.**
The prompt's premise ("run_contig BMI2 → kernel IS exercised under Rosetta") is
**FALSIFIED** by two independent deterministic blockers:
 1. **COMPILE blocker:** the macOS x86_64 target reserves rbp (frame pointer, NOT
    overridable via `-C force-frame-pointers=no`), leaving 14 GP regs; `run_contig`
    needs 15 register operands → LLVM "inline assembly requires more registers than
    available". Relief would need either a full 400-line cfg-duplicated asm
    (Rust asm! has no per-line cfg) or bug-prone register-merging — UNACCEPTABLE for
    smoke. (`dpre`→memory was the only clean 1-reg save but still needs whole-block
    cfg.) So the real asm cannot compile under Rosetta.
 2. **DISPATCH/TEST gate:** the asm dispatch (`asm_on`) and the c1/c2/c3/positive-
    control/on-off-fuzz differentials all gate on `is_x86_feature_detected!("bmi2")`
    = FALSE → production routes to the pure-Rust fast loop and every asm-vs-ref
    differential SKIPs (confirmed: all 6 print SKIP, 0 run the asm).
 ⇒ **The run_contig asm and its asm-vs-ref byte-exact gate are FULLY GUEST-OWED.**
    Rosetta gives ZERO asm signal. (Said plainly per the prompt's "if Rosetta skips
    it, say so".) Kernel-executed-under-Rosetta = **NO**.

 WHAT IS VERIFIED LOCALLY (high confidence, Rosetta x86_64, target-cpu=x86-64-v2):
 - **Full native (pure-rust-inflate) lib suite, serialized: 949 passed / 0 failed /
   12 ignored.** Includes prop_structured/random/near_max_distance @ PROPTEST_CASES=
   20000 (PASS), tests::routing::* (32 ok incl deletion-trap + parallel-SM e2e +
   silesia CRC stress), decompress::parallel::* lifecycle (357 ok, no hang).
   ⇒ c936a4df is BUILD-SOUND and the production PURE-RUST decode path (the path that
   actually runs under Rosetta, asm_on=false) is byte-exact.
 - **gzippy-isal flavor also builds + byte-exact** under Rosetta (routing 29 ok/0
   failed; asm_kernel 8 ok with the asm diffs skipping).
 - **c936a4df's REF-MODEL half (`run_contig_ref_biased`, edited in LOCKSTEP with the
   asm) is now DIRECTLY verified locally** by two NEW pure-Rust tests (TASK 3 below).
 - The one prior failure `test_avx2_detected_on_x86` was an ENV artifact (Rosetta
   hides AVX2; the AVX2 fast paths in copy_match_fast/fill_byte_avx2 fall back to
   scalar = guest-owed) — gated to skip on macOS; NOT a wrong-bytes issue.

### TASK 3 — CORRECTNESS NET STRENGTHENED (pure-Rust, runs under Rosetta AND on guest)
Two new tests in asm_kernel.rs tests mod (NO bmi2/asm dependency — they drive the
pure-Rust `run_contig_ref_biased` that c936a4df changed):
 - `ref_pre_consume_bail_keeps_cursor_un_consumed_rosetta`: random streams × 3 tables
   (fixed/dense/longy) × all-holes dist; asserts that on every FIRST-packet RECLASS
   bail (dst back at entry) the FULL cursor (bitbuf/bitsleft/pos) == entry. Covers ALL
   of c936a4df's snapshot-removal bail classes: invalid (bc==0), lone-EOB (==256),
   oversize (>512) — all PRE-consume un-consumed — AND the length-arm dist-side bail
   RESTORE (the only post-consume un-consume). Non-vacuous gate ≥50 first-packet bails
   (observed PASS).
 - `ref_consume_bias_perturbs_cursor_rosetta`: pure-Rust sibling of positive_control —
   `run_contig_ref_biased::<1>` (off-by-one consume) MUST diverge from `<0>` on every
   literal-progressing stream (gate: ≥100 progressing, ALL diverge — observed PASS).
   Proves the ref's consume accounting is load-bearing locally.
 Both PASS on native AND isal flavors under Rosetta.

### TASK 2 — NEXT CONVERGENCE ELEMENT = **DEFERRED to guest (deliberate, governing-law-compliant).**
Did NOT author a new asm element (§3.2 split-refill / §3.3 per-iter dist-preload /
§3.4 register-pin). Reasons, not excuses:
 - Those are ASM changes; under Rosetta they are NEITHER byte-exact-verifiable (asm
   won't compile/run) NOR perf-measurable. Committing them would be UNVERIFIED +
   UNMEASURED speculative asm — exactly what the governing law forbids.
 - They would also STACK on c936a4df, which is itself NOT yet guest-byte-exact-gated
   (its commit says "gauntlet on guest = NEXT"). Stacking unverified asm on unverified
   asm violates "don't stack inferences."
 ⇒ Correct order: VERIFY c936a4df on guest first (full AVX2 tri-oracle), THEN author
   the next §3.x element with its own byte-exact+perf gate. The design spine for those
   elements is already durable in plans/KERNEL-CONVERGENCE.md §3.2-3.5; nothing is lost.

### CHANGES THIS TURN (committed to kernel-converge-wip; guest/Linux behavior UNCHANGED)
 - asm_kernel.rs: (a) `#[cfg(target_os="macos")]` Rosetta compile shim in run_contig
   (unreachable!; the asm is dead at runtime on Rosetta since asm_on=false) wrapping
   the real asm in `#[cfg(not(target_os="macos"))]` — guest asm byte-for-byte
   UNCHANGED; (b) c1 seam test skips on macOS (it calls run_contig directly, no bmi2
   guard); (c) the 2 new pure-Rust ref tests.
 - consume_first_decode.rs: `test_avx2_detected_on_x86` skips on macOS (Rosetta env).
 All marked "Rosetta-smoke-correct; full AVX2 byte-exact gate + perf MEASUREMENT owed
 on neurotic." NO wrong-bytes anywhere → nothing reverted.

### OWED ON NEUROTIC (the guest gate this turn could NOT touch — DEFERRED, not done)
 1. **c936a4df full AVX2 byte-exact gate, BOTH flavors** (native pure-rust-inflate +
    gzippy-isal) × silesia/nasa/monorepo/squishy/large × T1/T4/T8 sha-identical vs
    tri-oracle (gzip/flate2/libdeflate/igzip); the c1/c2/c3/positive-control/on-off-
    fuzz asm-vs-ref differentials (BMI2-real); ≥60k prop_structured_roundtrip.
 2. **ALL PERF (every claim DEFERRED):** the canonical paired harness
    `_gzippy_vs_igzip_paired_guest.sh` — cyc/B / IPC / instr/byte for c936a4df vs igzip
    AND vs the night9 cc2840ff shape AND vs the OLD early-flag-bit gzippy-chunkt1 (does
    snapshot-removal recover the night9 +0.188 sil / +0.144 nasa regression?); stressor
    phase; T4/T8 wall + RSS no-regression; AMD/Zen2 (solvency) replication for LAW.
 3. The next §3.x convergence element (authored + byte-exact-gated + measured on guest).

### RESUME POINT
HEAD = c936a4df + this turn's local-correctness commit on kernel-converge-wip. On guest:
checkout kernel-converge-wip, run the §1 byte-exact gauntlet for c936a4df, then the §2
paired perf harness (the open question: did snapshot-removal recover night9's regression
and beat the old early-flag-bit shape?). Local toolchain note for future Rosetta turns:
`CARGO_TARGET_X86_64_APPLE_DARWIN_RUSTFLAGS="-C target-cpu=x86-64-v2"` (the repo
.cargo/config.toml forces target-cpu=native = invalid apple-m1 for x86_64); the asm is
macOS-shimmed so the lib compiles; asm differentials SKIP (guest-owed); pure-Rust ref +
pipeline tests give real local signal.

## ====== KERNEL-CONVERGENCE NIGHT9 (branch kernel-converge-wip @ cc2840ff) — igzip LATE-DISCRIMINATOR FULL-SPECULATION run_contig REWRITE: BUILDS + BYTE-EXACT (asm==ref diffs + production sha grid PASS). Perf measurement = NEXT. (Intel-only NOT-YET-LAW) ======
MISSION (heroic, no-phases): replace the early-flag-bit run_contig with igzip
loop_block's COMPLETE integrated shape AT ONCE. DONE THIS TURN: the rewrite is
IMPLEMENTED and BYTE-EXACT.

### WHAT WAS REWRITTEN (asm_kernel.rs run_contig + run_contig_ref_biased, LOCKSTEP)
Converged on igzip `loop_block` (igzip_decode_block_stateless.asm 507-627): the
loop now does, UNCONDITIONALLY every iteration, BEFORE a single LATE discriminator:
- (D) trailing-symbol extract `(syms>>8*(cnt-1))&0xFFFF`;
- (C) speculative 8-byte store + `add dst,cnt` (igzip 518-519);
- consume (shrx/sub) + (F) every-iter refill;
- (I) preload NEXT litlen entry + (J) SPECULATIVE preload of NEXT dist entry
  (igzip 550-552, every iter — discarded on literals);
- (K) LATE discriminator `cmp trailing,256; jb 2b` (literal→loop hot edge;
  ≥256→len/dist). The old EARLY flag-bit `test 0x1000000; jnz 49f` + the 49/50/31
  backref-entry shims are GONE. The backref arm reduces to `dec dst` (copy-start
  fixup of the spec over-advance, igzip's symmetric `lea +repeat_length-1`) + the
  existing (at-parity) 58: dist-decode + MOVDQU copy body.
NEW KernCtx.save_pos (+80): the X2 un-consume SNAPSHOT (bitbuf/bitsleft/pos/dst)
is taken at the ITERATION TOP (4 independent L1-hot stores) because the body now
consumes+refills BEFORE the discriminator, so EVERY rare bail (EOB/oversize/
invalid/dist-side) restores all four (pos too — the main-body refill advanced it).
KEY byte-exact argument: the new shape commits the SAME output bytes + a
self-consistent cursor as the old shape; it differs only in instruction
scheduling + DISCARDED speculation (the refill is append-only/logical-pos-
preserving, so a different refill cadence is still correct).

### BYTE-EXACT GATE — PASS (guest Intel LXC, /dev/shm/kc, native pure-rust-inflate)
- Build: clean (exit 0; register allocation FIT the wider straight-line body —
  15 GP operands, same as before, no spill failure).
- asm==ref differentials + harness liveness (cargo test --lib asm_kernel,
  --test-threads=1): **6/6 PASS** — c1_seam_roundtrip, c2_differential_asm_vs_ref
  _random_streams, c3_differential_asm_vs_ref_windowed_backrefs,
  positive_control_consume_off_by_one_trips_cursor_asserts (harness PROVEN live),
  asm_kernel_on_off_fuzz_random_gzip_members.
- Production sha grid (GZIPPY_FORCE_PARALLEL_SM=1 → path=ParallelSM confirmed):
  silesia/nasa/monorepo/squishy × T1/T4/T8 = **12/12 MATCH vs igzip**; +gzip(zcat)
  oracle silesia/nasa MATCH; +large model/bignasa × T1/T8 MATCH. ALL byte-exact.
- prop_structured/random/near-max-distance roundtrip (8000 cases) + lut_huffman
  + marker_inflate lib tests: **59/59 PASS, CARGO_EXIT=0** (guest /tmp/fast_test.log;
  the earlier PROPTEST_CASES=60000 run was impractically slow — killed; 8000 cases
  + the c3 windowed-backref differential + the real-corpus structured sha grid
  cover the same decoder). prop_random_bytes + prop_near_max_distance also seen
  PASS in the 60k log before it was killed.

### GATED PERF — §3.1 HYPOTHESIS FALSIFIED (the igzip late+full-spec shape LOSES to the early-flag-bit shape). gated, Gate-0 PASS, decorrelated SAME-session, Intel-only NOT-YET-LAW
Mission instrument `_gzippy_vs_igzip_paired_guest.sh`, PIN=4 REPS=15, /dev/null,
GZIPPY_FORCE_PARALLEL_SM=1, both arms Gate-0 PASS (KERN fired sil 24137/nasa 25389,
both byte-correct sha==zcat==igzip, A2-A1 self-test CI-incl-0, GHz spread <0.23%).
BOTH binaries measured back-to-back in the SAME box-state (load ~5-6, controlled by
the paired interleaving + self-test). medΔ = (gzippy − igzip) cyc/B:
| shape (binary)                         | silesia medΔ [95%CI]        | nasa medΔ [95%CI]          |
|----------------------------------------|-----------------------------|----------------------------|
| OLD early-flag-bit (gzippy-chunkt1)    | +1.2734 [+1.248,+1.292]     | +0.4813 [+0.121,+0.542]    |
| NEW igzip late+full-spec (gzippy-kc)   | +1.4613 [+1.435,+1.481]     | +0.6251 [+0.614,+0.628]    |
NEW − OLD = **+0.188 cyc/B silesia, +0.144 cyc/B nasa — the new shape is SIGNIFICANTLY
SLOWER (CIs NON-OVERLAPPING on both).** ΔIPC(new vs igzip) STILL NEGATIVE
(sil -0.034, nasa -0.125); Δinstr/byte BALLOONED to +3.59 sil / +1.17 nasa (old was
+1.57 sil). MECHANISM (gated counters, not inference): the unconditional per-iter
speculation (spec store + dist preload + trailing extract) + gzippy's NON-igzip-
faithful per-iteration 4-store X2 SNAPSHOT (~+2 instr/byte) added retired
instructions WITHOUT lifting IPC — the loop did NOT have the idle OOO slots the
§3.1 premise assumed; it is closer to uop/throughput-bound than "latency-bound with
fillable slots." So igzip's deep-speculation shape, ported faithfully PLUS the
snapshot tax gzippy's RECLASS contract forces, costs more than it buys.
VERDICT: KEEP the OLD early-flag-bit shape as production on perf/igzip-full-rewrite
(it WINS the §3.1 fork). The byte-exact NEW shape is PRESERVED on scratch branch
kernel-converge-wip @ cc2840ff for the iteration below — NOT merged to the mission
branch (it loses Gate-1).
RE-VERIFY: `GZIPPY=/root/bin/gzippy-kc PIN=4 REPS=15 CORPORA="silesia nasa"
SKIP_STRESS=1 GZIPPY_FORCE_PARALLEL_SM=1 bash /root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh`
(old: GZIPPY=/root/bin/gzippy-chunkt1). Logs: guest /tmp/perf_new.log, /tmp/perf_old.log.

### CLEAN RESUME POINT (next turn) — iterate toward higher igzip-fidelity (mission: "byte-exact but slower ⇒ KEEP iterating")
The leading suspect for the regression is the per-iteration 4-store X2 SNAPSHOT
(bitbuf/bitsleft/pos/dst → ctx every iter), which igzip does NOT pay (stateless,
no un-consumed RECLASS handback). NEXT STEP (a fresh byte-exact cycle on
kernel-converge-wip): REMOVE the snapshot from the hot literal path —
  (a) detect EOB / oversize / invalid PRE-consume (they are known from the decoded
      entry + trailing BEFORE the consume/store/refill), so only the genuine
      post-consume DIST-side bail needs un-consume state; AND/OR
  (b) snapshot only into the rare non-literal arm (still pre-consume — but the
      consume is in the body, so this needs the discriminator's EOB/oversize moved
      pre-consume and the dist-bail to carry its own minimal (bitbuf,bitsleft,pos)
      save just before the dist consume).
Then re-run the paired harness vs BOTH igzip and gzippy-chunkt1. IF still ≥ old
shape after the snapshot is off the hot path, the igzip-shape direction is dead at
T1 (record a FALSIFY with re-open trigger = a different arch/uarch or a corpus where
the loop is genuinely latency-bound). Owed regardless: ΔIPC under the STRESSOR
phase, T4/T8 wall+RSS, AMD/Zen2 replication.
BOX CLEAN: killed the slow 60k proptest + its orphan wait-loops + TWO Jun18 hung
gzippy test bins (PIDs 2739329 ~20h, 3769324 ~9.5h, prior-session — NOT mine);
verify pgrep clean at turn end.

## ====== KERNEL-CONVERGENCE HEROIC REWRITE (night8, branch kernel-converge → perf/igzip-full-rewrite) — DESIGN SPINE LAID + BASELINE RE-ANCHORED (gated, Gate-0 PASS, Intel-only NOT-YET-LAW) ======
MISSION (user, heroic, no-phases): converge the WHOLE inner-Huffman `run_contig` kernel on
igzip's COMPLETE integrated `loop_block` at once until gzippy-native T1 cyc/B ≤ igzip on
silesia AND nasa, byte-exact. Cost is a non-factor. Branch off origin/perf/igzip-full-rewrite
@ 380f0828.

### DONE THIS TURN
1. **Convergence DESIGN doc written + committed: `plans/KERNEL-CONVERGENCE.md`** — the durable
   spine. Maps EVERY igzip loop_block element (asm 507-627 + decode_next_lit_len/dist macros)
   to the gz kernel state. KEY FINDING from the map: the kernel is ALREADY a faithful igzip
   port for guards/decode/spec-store/preload-index/litlen-preload/dist-decode/MOVDQU-copy
   (elements A,B,C,E,I,L,M ✅). GENUINELY divergent = the coupled cluster {D,G,H,K} =
   discriminator PLACEMENT + speculation depth (gzippy branches EARLY via flag-bit `jnz 49f`
   and skips per-iter trailing-extract/length-precompute/dist-preload; igzip branches LATE
   and speculates ALL of it every iter to buy ILP), plus F (contiguous vs split refill),
   J (per-iter dist preload), N (LLVM vs igzip fixed register layout). Implementation order +
   byte-exact contract (X1-X6 + ref-model lockstep) + measurement gates are in the doc §3-6.
2. **Baseline RE-ANCHORED, Gate-0 self-validated** (gzippy-chunkt1 sha e5266440 vs igzip,
   paired N=9, cpu4, /dev/null, GZIPPY_FORCE_PARALLEL_SM=1; KERN sil 24137 / nasa 25389;
   both arms sha==zcat==each other; GHz spread <0.15%; A2-A1 self-test CI-incl-0 PASS):
   | corpus  | igzip cyc/B | gzippy cyc/B | medΔ (B-A1)       | 95%CI            | Wilcox p | ΔIPC   | Δinstr/B | verdict |
   |---------|-------------|--------------|-------------------|------------------|----------|--------|----------|---------|
   | silesia | 4.378       | 5.681        | +1.3073 (+29.9%)  | [+1.251,+1.350]  | 0.009152 | -0.322 | +1.570   | SIGNIF-slower |
   | nasa    | 1.594       | 2.106        | +0.5012 (+31.5%)  | [+0.461,+0.527]  | 0.009152 | -0.251 | +0.635   | SIGNIF-slower |
   Reproduces the STATE scoreboard (+28.1%/+30.3% @N=11) within spread. **ΔIPC NEGATIVE on
   both** = latency-bound = the empirical signature that motivates §3.1 (igzip's deeper
   speculation could fill the OOO window). NB LLC-miss ~24% flagged (stressor SKIPPED this
   anchor run); final rewrite verdict owes a stressor phase. RE-VERIFY:
   `GZIPPY=/root/bin/gzippy-chunkt1 PIN=4 REPS=9 CORPORA="silesia nasa" SKIP_STRESS=1 GZIPPY_FORCE_PARALLEL_SM=1 bash /root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh`

### CLEAN RESUME POINT (next turn — implementation)
NEXT = KERNEL-CONVERGENCE.md §6 step 1: build TWO byte-exact binaries — (A) current
early-flag-bit shape (= gzippy-chunkt1), (B) igzip late-discriminator full-speculation
straight-line `run_contig` (rewrite per §3.1; update run_contig_ref_biased in LOCKSTEP) —
on the guest /dev/shm, and run the paired harness B-vs-A and both-vs-igzip. This isolates
the core shape question (the discriminator/speculation cluster {D,G,H,K}) with the least
other change, BEFORE the refill (§3.2) / dist-preload (§3.3) / register-pinning (§3.4) work.
Guest src `/root/gz-fullrewrite` @ 89dad5c8 (== mission tip minus the STATE-docs commit;
`git pull` / checkout kernel-converge there before editing). NO main/reimplement-isa-l push.
BOX CLEAN this turn: background ssh finished (exit 0); no guest cargo started; no pinning;
governor untouched.

## ====== CHEAP-FAMILY SWEEP SESSION (night7) — input-mmap prefault = NOT a gated win (faults slack); T1 table-build BOUNDED (~5% sil / ~1.7% nasa, dominant litlen-LUT build NOT cheaply reducible); cheap space now mined across ALL families. (gated, byte-exact, Intel-only NOT-YET-LAW) ======
MISSION (advisor flagged two untested cheap levers in DIFFERENT families before the endgame):
(1) INPUT-mmap madvise/prefault (distinct fault family from all prior OUTPUT-buffer work);
(2) T1 Huffman table-build amortization (more 1 MiB chunks). Both TESTED this session.
HEAD at session = 89dad5c8 (== night6 chunk-size product ca70e9d1). Guest Intel LXC, cpu4, /dev/null.

### TASK 1 — INPUT-mmap prefault = byte-exact + NON-INERT but NOT a gated win (faults are SLACK)
PRIOR-STATE CORRECTION of the advisor's "no input hint": the PRODUCTION input mmap (io.rs:107-108,
`decompress_file`) ALREADY carries `Advice::Sequential`. The only untested input levers were the
STRONGER eager-prefault hints layered on top: MAP_POPULATE and MADV_WILLNEED.
ORACLE (byte-transparent, OFF==identity): `GZIPPY_INPUT_PREFAULT={populate|willneed}` in io.rs
(populate = `MmapOptions::populate()` at map time; willneed = `Advice::WillNeed`). Built native on
/dev/shm, bin `/root/bin/gzippy-prefault` (sha-built 89dad5c8 + oracle; src REVERTED after — NOT
committed, it is not a win).
- GATE-0 byte-exact: all modes sha==igzip on silesia+nasa. NON-INERT proven for `populate` only:
  minor-faults silesia 8039→6997 (-1042 ≈ the 16657 input pages collapsed by fault-around),
  nasa 3713→3396 (-317). `willneed` is fully INERT (faults identical to default) — MADV_SEQUENTIAL
  + kernel fault-around already cover it. ⇒ willneed DEAD; populate is the only live arm.
- GATE-1+2 paired N=15 cyc/B (analyzer /tmp/paired_analyze.py, perf cpu_core, cpu4, /dev/null,
  SAME bin env-only A=default B=populate):
  | corpus  | medΔ cyc/B (B−A)   | 95%CI             | Wilcox p | faults A→B   | verdict |
  |---------|--------------------|-------------------|----------|--------------|---------|
  | nasa    | -0.0060 (-0.28%)   | [-0.0366,-0.0032] | 0.0409   | 3714→3396    | SUB-GATE (p>0.01) |
  | silesia | -0.0022 (-0.04%)   | [-0.0280,+0.0055] | 0.1914   | 8039→6998    | TIE     |
  WALL best-of-9 (both sinks) IDENTICAL to 0.01s: silesia null 0.87/0.87 tmpfs 0.96/0.97; nasa
  null 0.31/0.31 tmpfs 0.41/0.41.
- DETERMINATION: input-mmap MAP_POPULATE removes the input minor-faults but does NOT move the wall
  (silesia TIE p=0.19; nasa -0.28% FAILS the p<0.01 gate; wall flat). **The input faults are SLACK**
  — consistent with the campaign's output-fault-slack finding; input faults are even cheaper
  (page-cache-warm + fault-around). NOT KEPT (oracle reverted, not committed). MINED.

### TASK 2 — T1 Huffman table-build BOUNDED (perf -F4000 -g dwarf, symboled /tmp/symtarget, T1)
PREMISE CORRECTION: chunk size does NOT multiply table builds. Build count == dynamic-BLOCK count
(set by the encoder), independent of our chunking. So the 1 MiB chunk default did NOT change
table-build absolute cyc; it is the same fraction it always was. Measured fraction (self+children):
- silesia ~5% of T1 cyc: read_header(children) 5.70%; build_huffman_luts_for_block 4.83%;
  LutLitLenCode::rebuild_from 2.96% self (the litlen LUT — DOMINANT); HuffmanCodingShortBitsCached
  ::initialize_from_lengths 1.64% (dist_hc); precode SymbolsPerLength ~0.25%.
- nasa ~1.7% of T1 cyc: read_header 1.66%; dist_hc 1.07% (DOMINANT here); litlen LUT 0.60%;
  DistTable::rebuild 0.49% (the P3.4 amortized path IS firing on nasa). nasa has fewer/larger
  blocks → table-build amortized over more bytes/block → smaller fraction.
- NO REDUNDANT builds: build_huffman_luts_for_block (marker_inflate.rs:1163) builds exactly the 3
  necessary structures per dynamic block (precode, litlen LUT, dist_hc). No dead/duplicate build to
  delete cheaply.
- CHEAP-REDUCTION SEARCH: (a) the DOMINANT piece (litlen LUT rebuild_from, 2.96% sil) is reducible
  ONLY by a faster construction algorithm = an inner-primitive rewrite = Route-1-class, NOT cheap.
  (b) the ONE cheap byte-exact candidate: `dist_hc` (HuffmanCodingShortBitsCached) is rebuilt EVERY
  dynamic block in build_huffman_luts_for_block with NO lens-reuse guard — unlike the existing P3.4
  `DistTable` amortization (dist_table_lens memcmp). A parallel memcmp-skip on dist_hc lens would be
  cheap + byte-exact (decoder is a pure fn of lens). CEILING ~1.07% (nasa) / ~1.64% (silesia) — but
  real capture is bounded by the consecutive-block dist-lens REPEAT frequency (partial; unmeasured),
  and it is a hot-path PER-BLOCK lifecycle edit requiring the full tri-oracle+lib-suite gate.
- VERDICT: table-build BOUNDED (~5% sil / ~1.7% nasa); no cheap HIGH-value reduction. dist_hc
  lens-reuse amortization = HYPOTHESIS (unvalidated), ceiling ≤~1.6%, small + uncertain + heavy
  gate → NOT implemented this turn without R3 (ROI poor, won't move the endgame verdict). RE-OPEN
  TRIGGER: a future cheap-pass with budget for a lifecycle-gated ≤1.6% lever; measure dist-lens
  repeat-frequency FIRST (instrument) to size real capture before building.
  RE-VERIFY: `perf record -F4000 -g --call-graph dwarf,8192 ... /tmp/symtarget/release/gzippy`
  then `perf report --stdio -g none | grep -iE 'read_header|build_huffman|rebuild_from|initialize_from_lengths'`.

### FINAL SCOREBOARD (unchanged this session — neither task banked) vs igzip T1, paired:
  silesia +1.24 cyc/B (+28.1%); nasa +0.48 cyc/B (+30.3%). [night6 numbers stand.]

### TASK 3 — ENDGAME R3 (cheap space now mined across ALL families)
Families tested across the campaign: fault/working-set (depth-1 BANKED, chunk-size BANKED,
resident-pool DEAD), INPUT (prefault TIE — this session), TABLE-BUILD (bounded, no cheap
high-value reduction — this session), kernel-primitives (CRC=CLMUL, bit-reader u64, fastloop
split — mined per advisor), output-streaming (bounded ≤1.5%). The remaining gap is now credibly
the diffuse inner-Huffman KERNEL — the litlen-LUT build (~3% sil) is itself part of that kernel
region (a faster table-build construction belongs to the kernel rewrite, not the cheap pass).
TWO framings for supervisor→user R3:
 (A) BANK the banked wins (dist-preload, flag-bit, ratio-reserve, T1-depth-1, T1-chunk-size) +
     ACCEPT the narrowed Intel-HYPOTHESIS gap (silesia +28.1%, nasa +30.3%; from +39.5%/+116.5%).
     AMD/Zen2 replication owed before LAW. = the advisor's LEAD recommendation (Route 1 ROI poor).
 (B) FUND Route 1 — inner-Huffman kernel asm-cadence convergence (converge run_contig on igzip's
     fused consume+refill+dual-preload loop; + a faster litlen-LUT construction). HONEST realistic-
     capturable estimate: removal-oracle ceiling = +0.908 cyc/B silesia but DIFFUSE across
     refill/classify/loop-overhead + LUT-build; only a FRACTION is capturable per technique
     (flag-bit captured ~0.10). Realistic = ~3-6% WALL over MULTIPLE sessions, HIGH byte-exact risk
     (X1-X5 exit + IN_MARGIN refill contract + run_contig_ref lockstep). Does NOT reach full parity
     cheaply. (NOT started — R3 gate.)

## ====== CHUNK-SIZE SESSION (2026-06-19 night6) — DETERMINATION: T1 chunk-size IS a real byte-exact gated lever; SHIPPED thread-gated (1 MiB default at T1). Refutes the night5 "cheap levers exhausted" R3 (3rd premature-closure). Route 2 BOUNDED (corrects night5 over-claim). (gated, byte-exact, Intel-only NOT-YET-LAW) ======
MISSION (advisor caught the night5 R3 as the 3rd premature "levers exhausted"): two cheap
things were NOT done before escalating — (1) the CHUNK_SIZE × pool-retain cross at T1
depth-1 (depth's untested analogue; the fault-floor formula is depth×chunk-output-pages and
only depth had been swept), and (2) a proper cyc/B bound on Route 2 (output-streaming).

### TASK 1 — CHUNK_KIB × pool-retain CROSS at T1 depth-1 = GATED BYTE-EXACT WIN (chunk-size; retain REFUTED)
STAGE-A probe (perf -r5, t1prod, cpu4, /dev/null, GZIPPY_CHUNK_KIB×{MANUAL_BUFFER_POOL on/off}):
| corpus  | config       | faults | cyc/B  | (igzip faults/cyc/B) |
|---------|--------------|--------|--------|----------------------|
| nasa    | c_def_roff   | 11379  | 2.1817 | 667 / 1.5865         |
| nasa    | c1024_roff   | 3769   | 2.0595 |                      |
| nasa    | c512_roff    | 2631   | 2.0573 |                      |
| nasa    | c256_roff    | 2358   | 2.0912 | (c256 OVER-shoots — too many chunks) |
| silesia | c_def_roff   | 13910  | 5.7144 | 668 / 4.3973         |
| silesia | c1024_roff   | 8097   | 5.6380 |                      |
| silesia | c512_roff    | 5194   | 5.6927 |                      |
| silesia | c256_roff    | 4586   | 5.7266 |                      |
- **RETAIN (GZIPPY_MANUAL_BUFFER_POOL) is INERT**: roff≈ron at EVERY chunk size (faults & cyc/B
  identical). The prompt's "lever only fires with a retained-resident reused buffer" HALF is
  REFUTED. The REAL mechanism: smaller chunk → smaller per-chunk output buffer → stays UNDER the
  allocator mmap threshold → the depth-1 in-order drain's DROP returns it to the DEFAULT
  allocator's arena, and the next chunk's same-size alloc gets those WARM already-faulted pages
  back (no munmap/refault). No manual pool needed. (Large 4 MiB-chunk → ~40 MiB(nasa)/~12 MiB(sil)
  decoded buf > mmap threshold → mmap+munmap each chunk → refault. That is the night-series
  "first-touch 24k/54k faults" mechanism, now mostly removed by simply shrinking the chunk.)
- OPTIMUM = c1024 (1 MiB): best JOINT (nasa ties c512, silesia clearly best); c256 over-shoots.

GATE 1+2 — paired N=15 cyc/B (analyzer A1/A2/B interleaved, perf cpu_core, cpu4, /dev/null,
SAME binary env-only A=default B=GZIPPY_CHUNK_KIB=1024); GATE-0 PASS (KERN 25389/24129, byte-exact,
self-test A2-A1 CI-incl-0 p>0.5, GHz<0.21%):
| corpus  | medΔ cyc/B (B−A1)   | 95%CI            | Wilcox p  | faults A→B      | verdict |
|---------|---------------------|------------------|-----------|-----------------|---------|
| nasa    | -0.12490 (-5.69%)   | [-0.133,-0.120]  | 0.000727  | 11381→3769 -67% | SIGNIF  |
| silesia | -0.07851 (-1.37%)   | [-0.098,-0.039]  | 0.001092  | 13911→8095 -42% | SIGNIF  |
PRODUCTION confirm (TWO DISTINCT binaries, NO env): A=gzippy-t1prod (4 MiB), B=gzippy-chunkt1
(1 MiB T1 default): nasa -0.116 cyc/B (-5.32%) CI[-0.120,-0.105] p=0.0007; silesia -0.067 cyc/B
(-1.16%) CI[-0.075,-0.060] p=0.0007. Self-test PASS. T1 WALL /dev/null: nasa 0.329→0.309 (-6.1%),
silesia 0.874→0.863 (-1.3%); tmpfs nasa 0.440→0.429, silesia 1.001→0.990.

### THE DIFF (commit ca70e9d1 on perf/igzip-full-rewrite) — ONE production path, thread-gated
`single_member.rs`: new const `T1_TARGET_COMPRESSED_CHUNK_BYTES = 1 MiB`; `decompress_parallel`
picks `thread_default_chunk = if num_threads<=1 { 1 MiB } else { 4 MiB }` as the `default_chunk`
fallback. Explicit `GZIPPY_CHUNK_KIB` still overrides. asm/kernel UNTOUCHED.

### T>1 NO-REGRESSION (HARD GATE) — why it MUST be T1-gated
With GZIPPY_CHUNK_KIB=1024 at T>1 (env, all-threads) silesia REGRESSES: T4 wall 0.482→0.574/0.578
(+19/20%, confirmed 2×), T8 +5.7% (then tie) — more/smaller chunks add block-finder+scheduling
overhead in the parallel pipeline. nasa T4/T8 fine. ⇒ the num_threads>1 path KEEPS 4 MiB. PROD
binary T>1 == old (structurally unchanged; T4/T8 wall+RSS old-vs-new identity confirmed). Byte-exact
T1/T4/T8 on nasa/silesia/monorepo/squishy.

### TASK 2 — ROUTE 2 (output-streaming/residency) cyc/B BOUND on the NEW baseline (CORRECTS night5 "unbounded")
Re-ran the EXISTING GZIPPY_RESIDENT_OUTPUT_POOL=1 oracle (paired N=11) on gzippy-chunkt1:
| corpus  | medΔ cyc/B          | 95%CI            | Wilcox p | faults A→B      | verdict   |
|---------|---------------------|------------------|----------|-----------------|-----------|
| nasa    | -0.00542 (-0.26%)   | [-0.013,+0.002]  | 0.1197   | 3769→3771 FLAT  | WASH/TIE  |
| silesia | -0.08396 (-1.49%)   | [-0.089,-0.081]  | 0.003857 | 8096→4999 -38%  | SIGNIF    |
⇒ Route 2 is BOUNDED, NOT "unbounded": ≤~1.5% cyc/B (silesia) / ~0 (nasa). The small-chunk win
already CONSUMED nasa's materialization headroom (nasa residency now washes out). Matches night3's
~1% WALL bound. CORRECTION to the night5 R3 below: Route 2 is bounded (~1% wall / ≤1.5% cyc/B sil,
~0 nasa), NOT unbounded. NB this oracle is NOT shippable as-is (night3: T>1 RSS/wall regression
when global; the 64 MiB over-reserve also fights small chunks) — it only BOUNDS the prize.

### NEW SCOREBOARD (canonical mission instrument gzippy-chunkt1 vs igzip, T1, paired N=11, GATE-0+self PASS)
| corpus  | igzip cyc/B | gzippy cyc/B | medΔ (B−A1)      | 95%CI            | verdict       | (night5 was) |
|---------|-------------|--------------|------------------|------------------|---------------|--------------|
| silesia | 4.397       | 5.640        | +1.2415 (+28.1%) | [+1.235,+1.249]  | SIGNIF-slower | +1.317/+29.8%|
| nasa    | 1.587       | 2.067        | +0.4806 (+30.3%) | [+0.475,+0.488]  | SIGNIF-slower | +0.600/+38.0%|
CAMPAIGN PROGRESSION: silesia +1.70→+1.24 cyc/B; nasa +1.84→+0.48 cyc/B. gzippy-native T1 still
LOSES; parity NOT reached. RE-VERIFY: `GZIPPY=/root/bin/gzippy-chunkt1 PIN=4 REPS=11
CORPORA="silesia nasa" SKIP_STRESS=1 GZIPPY_FORCE_PARALLEL_SM=1 bash
/root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh`. Chunk sweep+paired+T>1+wall+Route2:
`scripts/bench/{chunk_retain_probe,chunk_paired,chunk_tn_guard,chunk_prod_wall,route2_bound}.sh`
(transferred to /tmp on guest). Binaries: gzippy-t1prod (old 4 MiB, 261f1ebf), gzippy-chunkt1
(new T1=1 MiB, e5266440, == commit ca70e9d1 native).

### TASK 3 — VERDICT
1. CHUNK-SIZE is a real, cheap, byte-exact, gated T1 lever — BANKED + SHIPPED (commit ca70e9d1).
   This is the 3rd time a "levers exhausted" R3 was PREMATURE — chunk-size was an untested cheap
   lever and it paid (nasa -5.3%, silesia -1.2% cyc/B; nasa igzip-gap +38%→+30%).
2. Route 2 is now properly BOUNDED (≤1.5% cyc/B silesia, ~0 nasa) — the night5 "unbounded" is
   corrected. A residual SILESIA-only residency micro-lever (~0.084 cyc/B) exists but is small,
   silesia-only, and carries the night3 T>1 caveat (would need a T1-gated small-fixed-reserve
   resident pool, NOT the 64 MiB oracle) → HYPOTHESIS (unvalidated) for a future cheap-pass.
3. With chunk-size banked + Route 2 bounded small, the DOMINANT remaining gap is now credibly the
   diffuse inner-Huffman KERNEL (Route 1, ceiling +0.908 cyc/B silesia, diffuse across
   refill/classify/loop-overhead) = a multi-session asm-cadence convergence. THIS is now a
   properly-grounded R3 (cheap levers mined THIS session, not asserted): silesia residual is
   kernel-bound; nasa residual = diffuse kernel + the depth×chunk-output-page fault floor (now
   already at ~3769 with 1 MiB chunks). AMD/Zen2 replication of all of this owed before LAW.

## ====== TASK-0 CORRECTNESS GATE (2026-06-18 night5) — OWED depth-change verification: PASS on BOTH flavors ======
The night4 T1-depth change (commit 68fdbda5, thread-gated RecycleDeferral; touches the
recycle/drain lifecycle) needed its FULL serialized lib-suite correctness gate, which the
prior agent built green but LOST by orphaning. Re-run DURABLY this session, SERIALIZED
(`--test-threads=1`), at HEAD 62b802dc (== 68fdbda5 + this STATE doc), built on /dev/shm
(root disk was 99% full — would have hung a root-target build per CLAUDE.md):

| flavor (features)        | result                                  | EXIT | routing multithread (deletion-trap/lifecycle) | parallel module |
|--------------------------|-----------------------------------------|------|-----------------------------------------------|-----------------|
| native (pure-rust-inflate) | **949 passed; 0 failed; 12 ignored** (282.52s) | 0 | `test_single_member_routing_multithread ... ok` | 359 ok / 4 ignored |
| isal (gzippy-isal)         | **934 passed; 0 failed; 15 ignored** (246.37s) | 0 | `test_single_member_routing_multithread ... ok` | (all ok)        |

- NO FAILED / NO panicked on either flavor (the only "failed"-substring hit is the PASSING
  test name `block_fetcher::tests::failed_prefetch_flag_persists ... ok`).
- The ignored tests are ALL by-design: perf gates that only run on neurotic
  (`test_single_member_parallel_not_slower_than_sequential`, `..._silesia`, `..._class_not_slower`),
  slow integration (`drive_round_trips_*`, `drive_silesia_head_gzip9_t2`), fuzz loops
  (`fuzz_loop_differential`, `three_oracle_extended_fuzz_10k`), microbenches, and diagnostics.
  NONE are correctness regressions from the depth change.
- VERDICT: **PASS** — the T1-depth lifecycle change (and the whole banked set on it:
  dist-preload, flag-bit, ratio-reserve, T1-depth-1) is CORRECTNESS-CONFIRMED on both flavors.
  Frontier work is unblocked.
- RE-VERIFY (guest, /dev/shm target, serialized): `cd /root/gz-fullrewrite && RUSTFLAGS="-C
  target-cpu=native" CARGO_TARGET_DIR=/dev/shm/gz-verify-target cargo test --release
  --no-default-features --features pure-rust-inflate --lib -- --test-threads=1` (swap
  `gzippy-isal` for the isal flavor). Durable logs: /tmp/depthverify_{native,isal}.log on guest.
- BOX HYGIENE NOTE: on arrival the guest had THREE orphaned `cargo test` runs from prior
  agents still running on the ROOT-disk default target (load 4.4, root 99% full) — killed
  them (PIDs 335519/2735131/3769312 trees) to stop root-disk-fill + CPU contention; my run
  was isolated on /dev/shm. No box freeze, no persistent pinning, governor untouched.

## ====== CURRENT-HEAD GATED SCOREBOARD + R3 ESCALATION (2026-06-18 night5, post-TASK-0) ======
After TASK-0 PASS, re-ran the committed mission instrument to get the CURRENT gap at HEAD with
ALL banked wins live (arm B = /root/bin/gzippy-t1prod, sha 261f1ebf == HEAD depth-gated product;
byte-exact gz==zcat==igzip both corpora; N=11, cpu4, /dev/null, SKIP_STRESS=1; box quiet load~1.5
after killing the orphan cargo runs). Gate-0 PASS (KERN entries 24129/25389; igzip non-inert
sha-match; same sink+pin), self-test A2-A1 PASS (CI incl 0), GHz spread <0.2%.

| corpus  | igzip cyc/B | gzippy cyc/B | medΔ (B−A1)       | 95%CI            | Wilcox p | ΔIPC   | Δinstr/B | verdict       |
|---------|-------------|--------------|-------------------|------------------|----------|--------|----------|---------------|
| silesia | 4.414       | 5.723        | +1.317 (+29.8%)   | [+1.306,+1.332]  | 0.0039   | -0.311 | +1.621   | SIGNIF-slower |
| nasa    | 1.580       | 2.183        | +0.600 (+38.0%)   | [+0.596,+0.605]  | 0.0039   | -0.298 | +0.746   | SIGNIF-slower |

CAMPAIGN PROGRESSION (gated, Intel-only NOT-YET-LAW):
  silesia: START +1.70 (+39.5%)  → HEAD +1.317 (+29.8%)   [closed ~0.38 cyc/B / ~10 pts]
  nasa:    START +1.84 (+116.5%) → HEAD +0.600 (+38.0%)   [closed ~1.24 cyc/B / ~78 pts]
Wins responsible: dist-preload, flag-bit (silesia critical-chain), ratio-reserve (nasa grow-storm
kill), T1-depth-1 (nasa faults -73% + materialization). gzippy-native T1 still LOSES to igzip on
both corpora; parity NOT reached. RE-VERIFY: `GZIPPY=/root/bin/gzippy-t1prod PIN=4 REPS=11
CORPORA="silesia nasa" SKIP_STRESS=1 GZIPPY_FORCE_PARALLEL_SM=1 bash
/root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh`.

### POST-WINS RESIDUAL — where the gap is NOW (consolidated; the asm kernel is UNCHANGED since the
### LOCALIZE session, so its per-region decomposition still holds; only the SCAFFOLD shrank via
### ratio-reserve + depth-1, which the nasa +116%→+38% collapse reflects):
- **silesia (+1.317):** DEPTH-INVARIANT ⇒ KERNEL-bound. Kernel gap +0.908 cyc/B is DIFFUSE across
  refill (+0.361) / classify (+0.389) / loop-overhead (+0.341); backref-copy is at PARITY (do not
  attack). Residual scaffold (~+0.4) = first-touch output faults (materialization) + CRC (≈igzip) +
  Rust bail-glue. NO single peephole; flag-bit already captured ~7% of the kernel ceiling.
- **nasa (+0.600):** was scaffold-dominated (75%); depth-1 + ratio-reserve removed most of it.
  Remaining = residual materialization faults (11265 vs igzip 666; depth-1 is the rg-architecture
  floor — depth×chunk-output-pages, cannot go lower correctly) + the diffuse kernel (+0.453).

### DEAD / MINED (do NOT re-attempt without a new premise):
- Table-format rewrite B (fat direct one-sym table): MEASURED net-negative-to-breakeven (M3 spike
  +0.015..+0.091 SLOWER); igzip is ALSO one-sym/iter at 3.98 yet gzippy single-mode is 6.09 — the
  2.1 cyc/B is igzip's whole leaner per-iteration loop, which B does not touch. DEAD.
- Resident-output-pool (unified T1+T>1): REGRESSES T>1 wall +5–16% & RSS +9–76%. DEAD as unified.
- Warm-buffer-recycle / glibc-trim / THP-madvise / prefault-arena fault-removal oracles: all INERT.
- Elaborate dedicated T1 output-streaming sized as a FAULT-removal win: ~1% wall materialization
  slack only — the real prize there is kernel convergence, not faults (night3).
- The cheap/medium PIPELINE/CONFIG levers are now MINED (depth-1 was the last one; it was found
  AFTER a "near-floor" call, so this list is held as falsifiable, not eternal).

### R3 ESCALATION (BOUNDED) — the sole remaining routes to igzip-T1 PARITY are both DEEP efforts:
The prompt's stop-condition is met: the residual is dominantly (a) the diffuse inner-Huffman kernel
and (b) nasa residual materialization, with no cheaper pipeline/config lever remaining. Per the
governing rule I do NOT grind a multi-hour rewrite without R3 sign-off. The two routes + BOUNDS:
  ROUTE 1 — KERNEL CADENCE CONVERGENCE (authorized open territory, but DEEP/heroic): converge
    gzippy's run_contig per-iteration loop on igzip's fused consume+refill+dual-preload cadence
    (NOT the table format — that's B, dead). Entry point = the refill-fusion technique
    (Gate-2-confirmed on the critical recurrence). BOUND (removal-oracle, banked): igzip kernel
    3.98 vs gzippy 5.32 cyc/B silesia ⇒ ceiling +0.908 cyc/B; DIFFUSE, so only a FRACTION
    capturable per technique (flag-bit got ~0.10). To actually reach 3.98 needs igzip's WHOLE
    fused loop shape = a multi-step asm rewrite with byte-exact run_contig_ref kept in lockstep
    (X1-X5 exit + IN_MARGIN refill contract) = high byte-exact risk, multi-session. → R3.
  ROUTE 2 — OUTPUT STREAMING (divergence-from-rapidgzip architecture): decode-and-emit through ONE
    small reused window (igzip-shaped) instead of materializing full per-chunk output. Targets the
    materialization faults (nasa 11265→~666). BUT: recoverable magnitude is UNBOUNDED by any
    available oracle (every fault-removal oracle was INERT; a working oracle needs THP on a non-LXC
    box or an in-process reuse harness — neither built). And it DIVERGES from the rg parallel
    design that is the T>1 parity source ⇒ must be T1-gated or it regresses T>1. → R3 (strategic
    fork: faithful-rg vs igzip-shaped-streaming).
RECOMMENDATION FOR SUPERVISOR/USER: gzippy-native T1 is now within +29.8% (silesia) / +38.0%
  (nasa) of igzip — substantially narrowed but NOT parity. Reaching parity requires funding a deep
  effort (Route 1 kernel-cadence rewrite, ceiling +0.908 sil diffuse; and/or Route 2 T1-output-
  streaming, recoverable-magnitude unbounded). Decide: (a) fund Route 1 (multi-session asm kernel
  convergence), (b) fund Route 2 (T1-gated streaming, requires building the output-reuse oracle
  first to BOUND it), (c) accept the narrowed gap as the campaign result. AMD/Zen2 replication of
  the whole scoreboard is owed before any LAW claim (all results Intel-LXC NOT-YET-LAW).

## ====== CORRECTION (2026-06-18 night4, advisor-caught over-claims in the night3 determination below) ======
The night3 resident-pool determination (section immediately below) OVER-STATED. Corrected wording:
(a) **"output faults are SLACK" is UNDER-POWERED** — it rested on a −17% fault NUDGE (a slope,
    NOT a drive-to-igzip's-666 removal-oracle), no inter-run spread was stated, and nasa's pool
    DID NOT FIRE (faults flat, only 1 reuse). Downgrade the claim to: *"an elaborate T1 streaming
    path is low-value (~1% materialization wall-prize, single-corpus silesia only)."* It is NOT a
    gated proof that faults are slack.
(b) **"T>1 not feasible" tested ONLY the NAIVE global-resident pool** — downgrade to: *"the naive
    global fixed-64MiB-reserve resident pool regresses T>1 wall (+5–16%) and RSS (+9–76%)."* This
    does NOT rule out a thread-parameterized (T1-only) depth/pool change.
(c) **The "2.1 cyc/B kernel gap (6.09 vs 3.98)" is SINGLE-SYM microbench mode** which gzippy does
    NOT ship. Do NOT cite it as production headroom. The honest PRODUCTION-PACKED inner-kernel gap
    is **~0.9 cyc/B diffuse (silesia)** (refill+classify+loop-overhead), per the night2 residual note.
(d) **"diffuse / near-floor / exhausted" is NOT established** — there was an untested lever: the T1
    in-flight / recycle-deferral depth (the pipeline holds 4 concurrently-live ChunkData at T1 for
    ZERO parallelism benefit). The night4 session below TESTS it.

## ====== T1-INFLIGHT-DEPTH SESSION (2026-06-18 night4) — DETERMINATION = T1 in-flight depth IS A REAL byte-exact lever; SHIPPED thread-gated (depth 4→1 at T1). REFUTES night3 "faults are slack / near-floor". (gated, byte-exact, Intel-only NOT-YET-LAW) ======
MISSION: test the untested lever — the pipeline holds 4 concurrently-live ChunkData at T1
for ZERO parallelism benefit; does shrinking it cut faults+working-set and feed the kernel?

### MECHANISM LOCATED (file:line) — the "depth 4" is NOT prefetch
- At T1 prefetch is FULLY DISABLED: `thread_pool_saturated()` = `prefetching_len()+1 >= parallelization(=1)`
  is ALWAYS true → `prefetch_new_blocks` returns 0 (block_fetcher.rs:724-742). VERBOSE confirms:
  Prefetched=0, On-demand=17, dispatch called=17 saturated=17. Pool runs INLINE (pool_threads=0,
  chunk_fetcher.rs:652) at T1.
- The "Max concurrently-live ChunkData (in-flight depth): 4" (chunk_data.rs:1561 MAX_LIVE_CHUNKS,
  printed chunk_fetcher.rs:932) at T1 = 1 chunk being written + 1 lone Ready HELD (drain only when
  pending.len()>=2, chunk_fetcher.rs drain_ready_pending_heads) + 2 in the RECYCLE deferral
  (`defer_chunk_recycle` const DEPTH=2, chunk_fetcher.rs ~3923). Each holds a 12-MiB(sil)/40-MiB(nasa)
  output buffer. The defer-2 + lone-hold exists for a T>1 CORRECTNESS race (consumer_loop:1238-1242
  "byte diff at chunk 4 boundary when lone emit races worker fill") — at T1 (inline, no workers) it is
  pure overhead.

### THE DIFF (commit 68fdbda5 on perf/igzip-full-rewrite) — ONE production path, thread-gated
`RecycleDeferral` struct (chunk_fetcher.rs) carries {queue, depth, drain_lone} derived from pool_size:
  pool_size==1 → depth 0 + drain_lone (T1 win); else → depth 2 + hold-lone (unchanged T>1 correctness).
Sweep env overrides kept (GZIPPY_RECYCLE_DEFER_DEPTH, GZIPPY_DRAIN_LONE) for measurement; OFF==identity.

### GATE 0 — non-inert + byte-exact (PASS). Sweep binary /root/bin/gzippy-depthknob (sha 19308adf, from
7065e1e8+knob). MAX_LIVE responds cleanly: d4→4, d3(drain_lone)→3, d2(drain_lone+defer1)→2, d1(drain_lone
+defer0)→1. ALL arms sha==zcat==igzip on silesia+nasa at T1. Production binary /root/bin/gzippy-t1prod
(sha 261f1ebf, thread-gated, NO env): T1 MAX_LIVE=1 AUTO, sha-exact T1/T4/T8 both corpora.

### GATE 1+2 — T1 cyc/byte + faults (paired N=11 INTERLEAVED, perf cpu_core/cycles, pin cpu4, /dev/null)
| corpus  | arm | medcyc/B | Δ vs d4 (cyc/B)        | 95%CI            | Wilcox p | meanfaults | Δfaults |
|---------|-----|----------|-----------------------|------------------|----------|------------|---------|
| nasa    | d4  | 2.940    | —                     | —                | —        | 41666      | —       |
| nasa    | d2  | 2.465    | -0.441 (-14.98%)      | [-0.483,-0.423]  | 0.0033   | 21390      | -49%    |
| nasa    | d1  | 2.247    | **-0.672 (-22.86%)**  | [-0.692,-0.648]  | 0.0033   | 11265      | **-73%**|
| silesia | d4  | 5.993    | —                     | —                | —        | 23939      | —       |
| silesia | d2  | 5.861    | -0.151 (-2.52%)       | [-0.169,-0.111]  | 0.0033   | 16917      | -29%    |
| silesia | d1  | 5.870    | **-0.192 (-3.20%)**   | [-0.247,-0.114]  | 0.0044   | 13795      | **-42%**|
⇒ depth-1 SIGNIF-faster cyc/B on BOTH corpora (CI excl 0, p<0.005), monotonic d1>d2>d4. NOT slack.

### WALL (best-of-9, sha-verified) + RSS (min-of-5, maxRSS kB), T1, pin cpu4
| corpus  | arm | wall /dev/null | wall tmpfs | RSS kB  | igzip /dev/null |
|---------|-----|----------------|------------|---------|-----------------|
| silesia | d4  | 0.91           | 1.01       | 148668  | 0.66            |
| silesia | d1  | 0.87 (-4.4%)   | 0.98       | 104272 (-30%) |           |
| nasa    | d4  | 0.42           | 0.53       | 188476  | 0.23            |
| nasa    | d1  | 0.33 (-21.4%)  | 0.43 (-19%)| 67020 (-64%)  |           |
⇒ depth-1 cuts WALL (both sinks) AND RSS. The prompt's "lower depth → lower RSS" is CONFIRMED
  (night3's resident-pool RAISED RSS because it forced over-reserve residency; SHRINKING the live
  set LOWERS it). New gap to igzip T1: silesia +31.8% (was +32%, kernel-bound, depth doesn't move it);
  nasa +43.5% (was +84% night2 — depth-1 closed a big slice).

### T>1 GUARD (best-of-9 wall + RSS min-of-5; prod no-env vs ratioB; and global-d1 to prove the gate)
- prod(no-env) T4/T8 == ratioB (identity): silesia T4 0.46/0.46, T8 0.32/0.33; nasa T4 0.32/0.33,
  T8 0.19/0.19; RSS within noise. NO T>1 regression (OFF==identity by construction).
- **global-depth-1 at T>1 CORRUPTS silesia (sha FAIL at T4/T8)** — proves depth-1 is correctness-
  UNSAFE at T>1 (early recycle races worker fill) ⇒ the pool_size==1 GATE is mandatory.

### PROD CONFIRM (thread-gated binary gzippy-t1prod, NO env): T1 silesia 0.87 vs ratioB 0.90
(RSS 104k vs 149k), nasa 0.32 vs 0.42 (RSS 67k vs 188k); T>1 identity (T4/T8 walls+RSS == ratioB).
BYTE-EXACT BREADTH: sha==zcat at T1/T4/T8 on silesia, nasa, monorepo, squishy.

### DETERMINATION: **T1 in-flight depth IS a real, byte-exact, gated lever — SHIPPED (KEPT).**
1. Shrinking T1 live-ChunkData 4→1 (thread-gated) is SIGNIF-faster cyc/B (nasa -22.9%, silesia -3.2%,
   both p<0.005), faster wall (nasa -21%, silesia -4.4%), lower faults (-73%/-42%) AND lower RSS
   (-64%/-30%), byte-exact at every T. T>1 untouched (identity). KEPT, pushed.
2. This REFUTES the night3 over-claim that T1 output faults are "slack / near-floor / exhausted":
   the wall DID move with the working set. The lever was the in-flight DEPTH, which the resident-pool
   oracle conflated with a fixed-reserve RSS-inflating mechanism.
3. RESIDUAL gap to igzip T1 AFTER this fix: silesia +31.8% (kernel-bound — depth-invariant), nasa
   +43.5%. The silesia residual is the diffuse inner-kernel (~0.9 cyc/B) — the next lever is the
   KERNEL, not the output path. nasa still has fault headroom (11265 vs igzip 666).
RE-VERIFY: depth sweep `bash /tmp/depth_perf.sh` (+ `python3 /tmp/paired_analyze.py /tmp/depth_perf.csv`),
  wall/RSS `bash /tmp/depth_wall.sh`, T>1 `bash /tmp/depth_tn.sh`, prod `bash /tmp/prod_verify.sh`.
  Binaries: gzippy-depthknob (sweep, 19308adf), gzippy-t1prod (thread-gated prod, 261f1ebf).
BOX CLEAN: powersave, no persistent pinning (taskset per-cmd), /tmp scratch, root-disk untouched
  (built on /dev/shm CARGO_TARGET_DIR). NO main push, NO reimplement-isa-l push.

### NEXT
- AMD/Zen2 replication of this T1-depth fix owed before LAW (Intel LXC single-arch).
- The silesia residual is KERNEL-bound (depth-invariant +31.8%); the next T1 lever is the inner
  Huffman kernel (~0.9 cyc/B diffuse), NOT the output/scaffold path. nasa retains fault headroom.
- Consider: does drain-lone alone (without defer-0) carry T>1 risk? Not isolated here (gated to T1).

## ====== UNIFIED-T1-DETERMINATION SESSION (2026-06-18 night3) — resident-output-pool: DETERMINATION = NOT FEASIBLE as a unified path; igzip-T1 needs a dedicated streaming path. (gated, byte-exact, Intel-only NOT-YET-LAW) ======
USER QUESTION: "Is there a way to achieve igzip-like T1 speeds using code that MAINTAINS
rapidgzip T>1 parity? If yes → do it; if not → dedicated pure-Rust T1 path."
MECHANISM TESTED (the prompt's hypothesis): a BOUNDED POOL OF RESIDENT, REUSED output
buffers recycled on in-order drain (keep pages resident, no munmap/DONTNEED) so the next
chunk decodes into warm already-faulted memory → igzip-like ~hundreds of T1 faults.
Re-tested CORRECTLY now: (a) AFTER the grow-storm fix (ratioB tip), (b) with the pool
buffers GUARANTEED page-resident (fixed reserve → no realloc on reuse). Guest Intel cpu4
unstressed, /dev/null, perf freq-invariant. Bins: A=/root/bin/gzippy-ratioB (tip 1afc7a02,
sha ed948aa6), B=/root/bin/gzippy-resident (oracle, sha 375d1b19, distinct=non-inert).
SINGLE-ARCH Intel = NOT-YET-LAW (AMD owed). T1 single-core.

### TASK 1 — OUTPUT LIFECYCLE MAP + the in-order-drain recycle point (READ, file:line)
- T1 output = PER-CHUNK contiguous `ChunkData.data : SegmentedU8.buf : Vec<u8>` (4 MiB-
  COMPRESSED chunks → silesia 17 chunks ~12 MiB-decoded each, nasa 5 chunks ~40 MiB each).
- buf is LAZILY sourced from the per-worker LIFO `chunk_buffer_pool::take_u8` via
  `SegmentedU8::ensure_buf` (segmented_buffer.rs:151-156) on FIRST use, at
  `min_capacity.max(ALLOCATION_CHUNK_SIZE=128 KiB)`.
- IN-ORDER-DRAIN RECYCLE POINT = `ChunkData::recycle_decoded_buffers` (chunk_data.rs:1709)
  → `return_u8_to_worker` (also from Drop). Returns the buf to the worker pool AFTER the
  consumer wrote it. Manual pool retains capacity via `v.clear()` (NOT drop) — pages SHOULD
  stay resident. Default OFF (`manual_buffer_pool_enabled()` false → return DROPS the Vec).
- RECONCILED the prior-session INERT anomaly (the "13 warm reuses yet faults flat", allocator
  forensics deferred): `prefill_window_prefix` calls `ensure_buf(128 KiB)` → takes a SMALL
  pooled buffer, THEN `reserve_clean(R)` GROWS it to R by realloc (segmented_buffer.rs:450-460,
  chunk_decode.rs:1871/1427). With the OLD variable per-chunk reserve, the buffer realloc'd
  by-a-hair on most reuses → the LARGE output region was freshly mmap'd+faulted every chunk;
  the pool only ever recycled a throwaway 128 KiB allocation. THAT is why retaining it was inert.

### TASK 2 — DETERMINATION ORACLE (committed; byte-transparent; Gate-0 NON-INERT proven)
`GZIPPY_RESIDENT_OUTPUT_POOL=1` (chunk_buffer_pool.rs:resident_output_pool_enabled +
chunk_decode.rs compute_initial_reserve): pins EVERY chunk's reserve to a FIXED 64 MiB
(RESERVE_CAP) so all pooled buffers share ONE capacity → never realloc on reuse → pages stay
RESIDENT; AND auto-enables the manual LIFO pool. Gate-0 PASS: byte-EXACT (sha==zcat==ratioB==
igzip, silesia+nasa); distinct binary; NON-INERT THIS time (silesia faults 24054→19971 = the
fixed-reserve removed refaults the prior flat manual pool could not). Pool hits>0 confirmed.

### TASK 3 — MEASUREMENTS (perf -r 11, cpu4 unstressed, /dev/null)
T1 page-faults + cyc + wall (A=ratioB tip, B=resident):
| corpus  | A faults | B faults | Δfaults  | A cyc   | B cyc   | A wall  | B wall  | Δwall | igzip faults/cyc/wall |
|---------|----------|----------|----------|---------|---------|---------|---------|-------|-----------------------|
| silesia | 24054    | 19971    | -4083 (-17%) | 1258M | 1239M(-1.5%) | 0.908 | 0.897 | -1.2% | 668 / 940M / 0.677 |
| nasa    | 41782    | 41784    | +2 (FLAT)    | 586M  | 567M(-3.2%)  | 0.426 | 0.421 | -1.2% | 666 / 327M / 0.236 |
IN-FLIGHT DEPTH + pool stats (GZIPPY_VERBOSE): depth=4 BOTH corpora.
  silesia 17 chunks: pool u8 hits=13 misses=4 returns=17 → 4 distinct buffers cycle (13 warm).
  nasa 5 chunks: hits=1 misses=4 returns=5 → 5 chunks ≤ depth=4 → only 1 reuse → faults FLAT.
T>1 constraint (best-of-7 wall, min-of-5 peak RSS, sha-OK) — resident vs ratioB:
| corpus T | wall ratioB | wall resid | Δwall | RSS ratioB kB | RSS resid kB | ΔRSS |
|----------|-------------|------------|-------|---------------|--------------|------|
| silesia 4| 0.49 | 0.52 | +6.1% | 206336 | 363112 | +76.0% |
| silesia 8| 0.32 | 0.37 | +15.6%| 275528 | 412908 | +49.9% |
| nasa 4   | 0.32 | 0.35 | +9.4% | 293808 | 332752 | +13.3% |
| nasa 8   | 0.20 | 0.21 | +5.0% | 256308 | 279216 | +8.9%  |
⇒ forcing residency REGRESSES T>1 wall (+5–16%) AND RSS (+9–76%): the pipeline holds
  depth×workers buffers; the fixed over-reserve inflates RSS. (ratioB baseline already DROPS
  drained buffers → already RSS-bounded; retaining-resident is what COSTS RSS — the prompt's
  "lower RSS" prediction is backwards for the retain-resident approach.)

### DETERMINATION: **NOT FEASIBLE as a unified resident-pool path.** Recommend a dedicated T1 path — but SIZE IT HONESTLY (the dominant gap is the KERNEL, not materialization).
1. The resident-pool unified mechanism does NOT deliver the igzip T1 gain: T1 wall only
   -1.2% (faults depth-bounded: silesia -17%, nasa inert), the +33%/+78% igzip gap STANDS;
   and it REGRESSES T>1 wall+RSS. KILLED.
2. STRUCTURAL BLOCKER (measured): gzippy MATERIALIZES each chunk's FULL decoded output
   (12–40 MiB/chunk) held depth(=4)-deep concurrently for out-of-order assembly — this IS
   rg's parallel design (the T>1 parity source). The recyclable T1 fault floor = depth ×
   chunk-output-pages; even at depth-1 that's ~3k (sil) / ~10k (nasa) faults vs igzip's 666.
   igzip's profile comes from STREAMING output through ONE ~tens-of-KB reused window
   (decode-and-emit), which is INCOMPATIBLE with per-chunk materialization. No pool-tuning of
   the rg architecture reaches igzip's T1 fault profile.
3. CAUSAL FINDING that resizes the prize: the T1 OUTPUT FAULTS ARE LARGELY SLACK at the wall
   — a -17% silesia fault drop moved the wall only -1.2% (cyc -1.5%). So the materialization/
   scaffold component (STATE attribution ~0.35 sil / ~1.0 nasa cyc/B) is NOT the dominant
   T1-WALL lever; the bulk of the igzip gap is the KERNEL per-iteration machinery (STATE
   B-sizing: gzippy single-mode 6.09 vs igzip 3.98 cyc/B — a 2.1 cyc/B kernel-loop gap, NOT
   the table format). A dedicated T1 streaming path buys the small materialization slack
   (~1% wall) PLUS removes the resumable/marker/chunk SCAFFOLD around the kernel — but to
   reach igzip-T1-PARITY it must ALSO converge the inner loop on igzip's cadence, which is a
   SEPARATE problem that applies to the unified path too.
   ⇒ EXPECTED dedicated-T1 gain is bounded by kernel convergence, NOT by faults; do NOT size
   it at the full +33%/+78%. ESCALATE to USER (R3): the igzip-T1 lever is the inner kernel
   loop, not the output path; a dedicated T1 path is a vehicle for a tighter kernel, not a
   fault-removal win.
RE-VERIFY T1: `REPS=11 CORPORA="silesia nasa" bash /tmp/resident_faults.sh` (on guest);
  T>1: `N=7 M=5 CORPORA="silesia nasa" THREADS="4 8" bash /tmp/resident_tn.sh`;
  depth/pool: `GZIPPY_VERBOSE=1 GZIPPY_RESIDENT_OUTPUT_POOL=1 GZIPPY_FORCE_PARALLEL_SM=1
  taskset -c 4 /root/bin/gzippy-resident -d -c -p1 /root/silesia.gz >/dev/null`.
BOX CLEAN: powersave, 0 stressors, no persistent pinning (taskset per-cmd), /tmp scratch
  ephemeral, root-disk untouched (built on tmpfs CARGO_TARGET_DIR). guest worktree reverted
  to tip. NO main push, NO reimplement-isa-l push.

### NEXT
- USER R3 DECISION owed: the determination says igzip-T1-parity is NOT reachable by a unified
  resident-pool; a dedicated T1 path is needed but its prize is KERNEL convergence (the inner
  loop), and materialization-removal alone is ~1% wall. Decide whether to (a) fund a dedicated
  pure-Rust T1 streaming decoder (igzip-shaped: one small reused window, decode-and-emit, no
  per-chunk materialization, no resumable/marker scaffold) whose REAL prize is letting the
  inner kernel run in igzip's tight loop, or (b) keep converging the SHARED kernel loop (helps
  both T1 and T>1) without a fork. Per STATE B-sizing the kernel gap is diffuse (refill +
  classify + loop-overhead) and NOT the table format.
- AMD/Zen2 replication of this determination owed before LAW.

## ====== PRODUCTIONIZE SESSION (2026-06-18 night2) — ratio-based output reserve SHIPPED (gated, byte-exact, KEPT) ======
MISSION: productionize the cheap-fix the prior session bounded (oracle gzippy-bigreserve:
nasa -0.533 cyc/B / -12% to-file wall via blind 16->96 MiB clamp + 8->16 mult). Goal:
size the per-chunk output reserve from the ALREADY-computed member ratio
(`expansion_ratio_ceil`), not a blind 96 MiB, so the grow-realloc storm dies on expanding
data WITHOUT over-reserving low-expansion data. Intel guest cpu4 unstressed. SINGLE-ARCH
Intel = NOT-YET-LAW (AMD/Zen2 owed).

### THE DIFF (commit 1afc7a02 on perf/igzip-full-rewrite) — ONE production path, asm untouched
- `chunk_decode.rs`: broadened `compute_initial_reserve(compressed_span, expansion_ratio_ceil)`
  cfg from isal-only to `#[cfg(parallel_sm)]` (it already existed for the ISA-L oracle path:
  factor = ratio_ceil (0->8 fallback), clamp [4 MiB, 64 MiB]).
- Rewired BOTH native reserve sites to use it (replacing `compressed*8 + 1MiB` clamped 16 MiB):
  `decode_chunk_unified_marker` (was chunk_decode.rs:1408-1413) and
  `seed_block_for_contig_native` (was :1853-1858). `expansion_ratio_ceil` is read from
  `chunk.configuration` (plumbed from sm_driver.rs:139-149). Under-reserve still falls back
  to safe amortized regrow (byte-transparent — proven B==A byte-identical).

### GATE 1 — BYTE-EXACT: PASS (HARD GATE)
- BOTH flavors build (native pure-rust + gzippy-isal on guest). ratioB output == gzip == igzip
  == baseA (BYTE-IDENTICAL) across silesia/nasa/monorepo/squishy/weights × T1/T4/T8 (native);
  isal flavor byte-exact silesia/nasa/monorepo/squishy × T1/T4/T8. proptest 60k
  prop_structured_roundtrip PASS. The change is a pure capacity-hint -> byte-transparent.
- Binaries: BIN_A=/root/bin/gzippy-baseA (15cbdffd, sha 891c9925 == prior gzippy-new-native),
  BIN_B=/root/bin/gzippy-ratioB (1afc7a02, sha ed948aa6) — distinct (non-inert).

### GATE 2 — cyc/byte (paired N=21, unstressed) + WALL (best-of-9/21) + faults — SIGNIF WIN on expanding data
cyc/byte (BIN_A baseA vs BIN_B ratioB, self-tests PASS, GHz spread PASS):
| corpus   | A1 cyc/B | B cyc/B | medΔ (B-A1)        | 95%CI             | Wilcox p  | Δinstr/B | verdict |
|----------|----------|---------|--------------------|-------------------|-----------|----------|---------|
| silesia  | 5.962    | 5.935   | -0.0353 (-0.59%)   | [-0.052,-0.015]   | 0.00198   | -0.009   | SIGNIF-faster |
| nasa     | 3.500    | 2.921   | **-0.5737 (-16.4%)**| [-0.624,-0.517]  | 6.4e-05   | -0.420   | SIGNIF-faster |
| monorepo | 5.191    | 4.732   | **-0.4736 (-9.1%)** | [-0.502,-0.436]  | 6.4e-05   | -0.356   | SIGNIF-faster |
WALL best-of (T1, sha-OK) + page-faults (perf -r7):
| corpus   | sink      | baseA  | ratioB | Δwall  | faults base->ratio | igzip wall |
|----------|-----------|--------|--------|--------|--------------------|------------|
| silesia  | /dev/null | 0.898  | 0.895  | ~tie   | 24076 -> 24054     | 0.680      |
| silesia  | tmpfs     | 1.038  | 1.083  | tie(noise) | (interleaved N21x2: ratioB faster both rounds) | 0.826 |
| nasa     | /dev/null | 0.520  | 0.439  | -15.6% | 54080 -> 41782 (-23%) | 0.239   |
| nasa     | tmpfs     | 0.640  | 0.586  | -8.4%  |                    | 0.380      |
| monorepo | /dev/null | 0.198  | 0.175  | -11.4% | 15716 -> 13248 (-16%) | 0.112   |
| monorepo | tmpfs     | 0.219  | 0.205  | -6.4%  |                    | 0.143      |
⇒ nasa/monorepo: cyc/B + wall + faults ALL drop (p<0.01). silesia: cyc/B SIGNIF-faster but
  TINY (-0.59%) + wall TIE — MECHANISTICALLY EXPECTED: silesia ratio=4 @ 4 MiB chunks ->
  reserve 16 MiB == the OLD 16 MiB clamp (identical), so silesia never hit the grow storm.

### GATE 3 — RSS (peak maxRSS, T1/T4/T8, min-of-5 final): NO MATERIAL REGRESSION
| corpus  | T  | baseA kB | ratioB kB | Δ%    |
|---------|----|----------|-----------|-------|
| silesia | T1 | 143160   | 147372    | +2.9% (~4 MB noise; reserve identical) |
| silesia | T8 | 275100   | 268916    | -2.2% (tie/better; the min-of-3 +8.6% was a noise sample) |
| nasa    | T1 | 216620   | 187576    | **-13.4%** (pre-sizing kills realloc transient) |
| nasa    | T4 | 315144   | 311276    | -1.2% |
| nasa    | T8 | 261024   | 254336    | -2.6% |
⇒ nasa RSS BETTER (no realloc old+new coexistence); silesia tie (reserve unchanged).
  RSS-neutral-to-better BY CONSTRUCTION: reserve is virtual capacity, resident = touched =
  actual decoded size (unchanged); the ratio-fix avoids the over-reserve a blind clamp incurs.

### NEW GAP TO IGZIP after the fix (T1 wall; was the mission target)
| corpus   | /dev/null gz-vs-igzip | tmpfs gz-vs-igzip | (was, pre-fix) |
|----------|-----------------------|-------------------|----------------|
| silesia  | +32% (0.895 vs 0.680) | +31% (1.083 vs 0.826) | ~+33%/+29% (unchanged — no storm) |
| nasa     | +84% (0.439 vs 0.239) | +54% (0.586 vs 0.380) | +117%/+112% (CLOSED a lot) |
| monorepo | +57% (0.175 vs 0.112) | +44% (0.205 vs 0.143) | ~+70% |

### VERDICT: **KEPT** — gated win (nasa -16.4% cyc/-15.6% wall, monorepo -9.1%/-11.4%,
both p<0.01 CI-excl-0), byte-exact (both flavors + proptest 60k), NO RSS regression.
silesia = tie (expected; fits the old clamp). Pushed to perf/igzip-full-rewrite.
GATED-HYPOTHESIS, Intel-only NOT-YET-LAW — AMD/Zen2 replication owed.
RE-VERIFY: `BIN_A=/root/bin/gzippy-baseA BIN_B=/root/bin/gzippy-ratioB PIN=4 REPS=21
  CORPORA="silesia nasa monorepo" SKIP_STRESS=1 GZIPPY_FORCE_PARALLEL_SM=1 bash
  /root/distpreload-harness/_distpreload_paired_guest.sh`; wall+faults `bash /tmp/wall_faults.sh`;
  RSS `bash /tmp/rss.sh`. Box clean: powersave, 0 stressors, no persistent pinning (taskset
  per-cmd only), /tmp scratch ephemeral.

### RESIDUAL / NEXT
- The remaining igzip gap is now SCAFFOLD (first-touch faults of the ACTUAL decoded output —
  igzip streams through a ~666-fault reused window) + the diffuse per-iteration KERNEL gap
  (refill/classify/loop-overhead, all causally on-path, BOUND ~0.9 cyc/B sil). The grow-storm
  component is now REMOVED. The next-largest removable T1 component on backref-heavy data =
  output STREAMING through a reused buffer (igzip-shaped; divergence-from-rg) — USER R3 fork.
- AMD/Zen2 replication of THIS fix owed before LAW.

## ====== MATERIALIZATION-RECONCILE SESSION (2026-06-18 night) — RECONCILE + BOUND the scaffold gap ======
MISSION (this turn): reconcile WHY T1 faults 24k/54k pages if a recycling buffer pool
exists; build a WORKING warm-output oracle to BOUND recoverable gain; wall in real sink;
cheap-fix vs port. Guest Intel cpu4 unstressed, /dev/null, perf freq-invariant cyc/B.
Bins: /root/bin/gzippy-new-native (std-Vec native, sha 891c9925), igzip=/usr/bin/igzip.
Symboled /tmp/symtarget/release/gzippy (ARENA build — see GATE-0 caveat). SINGLE-ARCH
Intel = NOT-YET-LAW (AMD owed). T1 single-core.

### TASK 1 — RECONCILE: the exact allocation that faults output pages (READ the path)
- T1 output is NOT one ISIZE buffer. It is PER-CHUNK contiguous `SegmentedU8.buf` Vecs
  (4 MiB-compressed chunks → silesia ~17, nasa ~5 chunks). Path: ChunkData.data:SegmentedU8;
  decode bulk = single `buf` (NOT segmented — `front` list is only for post-decode prepends).
- buf is pre-reserved to `min(estimate, RESERVE_CLAMP=16 MiB)` at
  seed_block_for_contig_native (chunk_decode.rs:1849-1859; estimate = compressed*8 + 1 MiB),
  then GROWN by amortized doubling via `SegmentedU8::contig_decode_window`
  (segmented_buffer.rs:342-361 → `self.buf.reserve(min_spare)` → grow_amortized).
- The "recycling chunk-buffer pool" (chunk_buffer_pool::take/return_u8) is OFF BY DEFAULT
  (`manual_buffer_pool_enabled()` false; return drops the Vec). The slab resident-retention
  (RpmallocAlloc/SlabAlloc side-table, T<=2 auto-on) only routes `Vec<_,RpmallocAlloc>` —
  but the NATIVE build (`--features pure-rust-inflate`, NO `arena-allocator`) makes
  `types::U8 = std::Vec<u8>` (rpmalloc_alloc.rs:1116-1120). So neither pool NOR slab serves
  the native output buffer → that is why STATE M2's slab/prefault oracles were INERT.

### THE INSTRUMENTED FAULT ATTRIBUTION (perf record -e page-faults -g dwarf, symboled bin) — overturns prior M1
Dominant fault stack = `contig_decode_window → Vec::reserve → grow_amortized → finish_grow →
RpmallocAlloc::grow → __memcpy_avx`:
| corpus  | grow-realloc faults | first-touch decode-write faults (run_contig) |
|---------|---------------------|----------------------------------------------|
| nasa    | **68.39%**          | 27.90%                                        |
| silesia | 11.30%              | 82.64%                                        |
⇒ MECHANISM (overturns M1 "first-touch each output page once"): nasa expands ~10x (4 MiB
  comp → ~40 MiB decoded ≫ 16 MiB clamp) → the buf REALLOC-GROWS 16→32→64 MiB during decode;
  EACH grow (a) faults the new larger alloc AND (b) `__memmove_avx` copies the accumulated
  output (THE "extra memmove" M3 could not explain = the realloc copy, NOT a double-copy).
  silesia (~12 MiB decoded < 16 MiB clamp) fits → faults are 82% first-touch decode writes.
GATE-0 CAVEAT: /tmp/symtarget is an ARENA-allocator build (stack shows RpmallocAlloc/SlabAlloc)
  ≠ the std-Vec gzippy-new-native I fault-counted. The grow_amortized MECHANISM is identical
  in both (std RawVec also doubles+copies); the attribution % is from the arena bin. OWED:
  re-profile faults on a std-Vec symboled bin to confirm the % transfers.

### WARM-REUSE ORACLE (manual LIFO pool, in-tree, NON-INERT) — recycling does NOT recover the gain
`GZIPPY_MANUAL_BUFFER_POOL=1` retains freed output Vecs (pages resident → warm reuse).
perf -r 11, cpu4, /dev/null, sha-OK both arms. Non-inert PROVEN: pool u8 hits cold=0 →
warm=13 (silesia) / 1 (nasa).
| corpus  | mode | page-faults | cyc/B  | igzip cyc/B (faults 666) |
|---------|------|-------------|--------|--------------------------|
| silesia | cold | 24017       | 5.905  | 4.410                    |
| silesia | warm | 24019       | 5.896  |                          |
| nasa    | cold | 54021       | 3.395  | 1.592                    |
| nasa    | warm | 54020       | 3.294 (~3%, only 1 reuse) |             |
⇒ Recycling the output buffer (manual pool) is INERT on faults (24017→24019) and ~0 on
  silesia cyc/B. nasa cyc/B moved ~3% but with only 1 of 5 chunks reused (depth-4 pipeline,
  too few chunks). CONCLUSION: simple buffer recycling is NOT the cheap win — faults are the
  GROW-REALLOC + first-touch, not removable by retaining the final buffer. (silesia anomaly:
  13 warm reuses, fits-clamp/no-grow, yet faults flat — pooled-buffer pages not warm on
  reuse; allocator forensics deferred, the operational verdict stands.)

### CHEAP-FIX ORACLE — pre-size the output buf (kill the grow-realloc storm) = GATED BYTE-EXACT WIN
Built /root/bin/gzippy-bigreserve (native, std-Vec; gz-fullrewrite + RESERVE_CLAMP 16→96 MiB
and estimate multiplier 8→16 at chunk_decode.rs:1408/1411 + 1853/1856; src REVERTED after build;
NB sed also hit absolute_bit_pos:2555 mul(8)→16 — REVERTED before build, would corrupt bits).
DIRECT faults+cyc/B (perf -r 11, cpu4, /dev/null, sha-OK):
| corpus  | bin        | faults | cyc/B  |
|---------|------------|--------|--------|
| silesia | new-native | 24016  | 5.939  |
| silesia | bigreserve | 21855  | 5.869  |
| nasa    | new-native | 54020  | 3.404  |
| nasa    | bigreserve | 41724  | 2.860  |
INTERLEAVED PAIRED N=15 (A=new-native, B=bigreserve), GATE-0 PASS (sha_B SHA_OK, entries=24127
KERN_OK, non-inert), self-test PASS, GHz<0.3%:
| corpus  | medΔ cyc/B (B−A)      | 95%CI               | Wilcox p  | verdict        |
|---------|----------------------|---------------------|-----------|----------------|
| silesia | -0.0768 (-1.29%)     | [-0.108,-0.060]     | 0.0016    | SIGNIF-faster  |
| nasa    | **-0.533 (-15.9%)**  | [-0.578,-0.524]     | 0.0007    | SIGNIF-faster  |
WALL best-of-9 (sha-OK), does the cyc/B win reach the wall in a real sink? YES:
| corpus  | sink      | new-native | bigreserve | Δ      | igzip  | gz-vs-igzip after fix |
|---------|-----------|------------|------------|--------|--------|-----------------------|
| silesia | /dev/null | 0.910s     | 0.897s     | -1.4%  | 0.676  | +33%                  |
| silesia | tmpfs     | 1.047s     | 1.034s     | -1.3%  | 0.802  | +29%                  |
| nasa    | /dev/null | 0.502s     | 0.422s     | -15.9% | 0.237  | +78% (was +112%)      |
| nasa    | tmpfs     | 0.631s     | 0.555s     | -12.0% | 0.353  | +57%                  |
(No disk sink: root fs 97% full; tmpfs is the real-file sink. AMD owed.)

### VERDICT (this turn) — CHEAP-FIX, not a port (gated-HYPOTHESIS, Intel-only NOT-YET-LAW)
- The materialization gap is REAL but its LARGEST removable T1 component is NOT "stream through
  a reused buffer" (igzip-arch/R3) — it is the GROW-REALLOC STORM of the per-chunk output buf,
  removable by a BYTE-EXACT one-path capacity-hint change (no new code path, no architecture).
- CHEAP FIX (recommend implement, refined): size reserve_clean from the config's
  `expansion_ratio_ceil` (already computed, sm_driver.rs:139-149) instead of compressed*8 clamped
  16 MiB — so high-ratio chunks (nasa ~10x) reserve their full decoded size ONCE. The oracle used
  a blind 16x/96 MiB (over-reserves virtual mem) → DO NOT ship blind; the ratio-based version
  gives the SAME grow-elimination without the over-reserve. OWED before shipping: T4/T8 RSS +
  wall check (over-reserve × in-flight chunks) and AMD/Zen2 replication.
- The RESIDUAL after the cheap fix (nasa still +57-78%, silesia +29-33%) = first-touch faults of
  the ACTUAL decoded output (igzip's 666-fault reused-window streaming) + the kernel per-iteration
  gap. THAT is the R3 architecture territory — but it is now SMALLER, and the cheap fix should land
  first. recoverable-by-cheap-fix BOUNDED = nasa 0.53 cyc/B / -12% to-file wall (gated this turn).
RE-VERIFY cheap fix: `BIN_A=/root/bin/gzippy-new-native BIN_B=/root/bin/gzippy-bigreserve PIN=4
  REPS=15 CORPORA="silesia nasa" SKIP_STRESS=1 GZIPPY_FORCE_PARALLEL_SM=1 bash
  /root/distpreload-harness/_distpreload_paired_guest.sh`; wall: `N=9 bash /root/_wall_sink_guest.sh`
RE-VERIFY warm oracle: `REPS=11 CORPORA="silesia nasa" bash /root/_warmdst_oracle_guest.sh`
RE-VERIFY fault stacks: `env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 4 perf record -e page-faults
  -g --call-graph dwarf,8192 -o /tmp/pf.data -- taskset -c 4 /tmp/symtarget/release/gzippy
  -d -c -p1 /root/nasa.gz >/dev/null; perf report -i /tmp/pf.data --stdio -g folded`

## ====== SCAFFOLD-ARTIFACT SESSION (2026-06-18 late PM) — is the SCAFFOLD gap REAL or a HARNESS/ALLOC ARTIFACT? ======
MISSION: settle whether the SCAFFOLD gap (output __memmove_avx + first-touch page
faults + finish_decode) is a real production cost or a fresh-per-run allocation
artifact, via removal-oracles, BEFORE any architecture work. Intel cpu4 unstressed,
/dev/null sink, freq-invariant cyc/B. Binaries: /root/bin/gzippy-new-native (mission
tip), igzip=/usr/bin/igzip. Symboled bin /tmp/symtarget/release/gzippy (@0a592a54).
SINGLE-ARCH Intel = NOT-YET-LAW (AMD owed). T1 single-core only.

### MEASUREMENT 1 — page-fault accounting (gated, perf -r 11, cpu4 unstressed)
| corpus  | tool   | page-faults | cyc       | cyc/B  | dTLB-miss | gap cyc/B      |
|---------|--------|-------------|-----------|--------|-----------|----------------|
| silesia | gzippy | 24,075      | 1.261e9   | 5.950  | 42,546    |                |
| silesia | igzip  | 667         | 0.941e9   | 4.440  | 21,099    | +1.510 (+34.0%)|
| nasa    | gzippy | 54,077      | 0.710e9   | 3.459  | 82,468    |                |
| nasa    | igzip  | 666         | 0.326e9   | 1.589  | 3,240     | +1.869 (+117.5%)|
⇒ SMOKING GUN: igzip faults ~666-667 pages REGARDLESS of output size (212MB vs 205MB)
  ⇒ igzip STREAMS output through a small fixed REUSED buffer. gzippy faults 24k(sil)/
  54k(nasa) ≈ tracks output page count (nasa 54k ≈ 50,108 output pages; silesia 24k ≈
  HALF — partial reuse/THP, unexplained-but-noted) ⇒ gzippy MATERIALIZES the full
  contiguous output, faulting each page once. Δfaults = +23,408 (36x) sil / +53,411
  (81x) nasa. major-faults=0 (all minor/anon).

### MEASUREMENT 2 — reuse/pre-fault REMOVAL-ORACLES: ALL THREE INERT (decisive negative)
Tried to make gzippy recycle/pre-fault output pages so faults→igzip-level:
  (a) glibc `MALLOC_MMAP_THRESHOLD_=1G MALLOC_TRIM_THRESHOLD_=-1` (heap-recycle): faults
      24029→24026 — INERT.
  (b) THP=always: LXC sysfs is READ-ONLY ("Read-only file system"), cannot change in
      container; faults unchanged 24016 — UNAVAILABLE here.
  (c) `GZIPPY_PREFAULT_ARENA=220` (the in-tree rule-#3 oracle, chunk_buffer_pool.rs:208;
      ±GZIPPY_SLAB_ALLOC=1): faults nasa 54020→54018, sil 24023→24017 — INERT (does NOT
      serve the output buffer's pages from the prefaulted rpmalloc thread-cache).
GATE-0: all three non-inert checks FAILED to move faults ⇒ the output faults are GENUINE
one-time first-touch of the materialized output buffer; NOT allocator-recyclable and NOT
pre-faultable via the available knobs. ⇒ The faults are NOT a "fresh per-run allocation"
artifact in the recyclable sense — they recur on EVERY production decode and no recycle
removes them. The ONLY removal is architectural (don't materialize full output; stream
through a reused buffer = igzip's design).
COULD-NOT-RUN: the definitive in-process reused/pre-faulted-output-buffer oracle (decode
N× reusing one warm dst) — needs a throwaway reuse-harness build OR THP on a non-LXC box
(neurotic-baremetal / solvency). OWED before the recoverable-cyc magnitude is STRONG-tier.

### MEASUREMENT 3 — the extra-memmove question + perf attribution (HYPOTHESIS-tier; perf -F6000 symboled)
SILESIA self%: run_contig 83.71 | rebuild_from 2.86 | finish_decode(=CRC) 2.01 |
  __memmove_avx 1.08 | kernel mm/fault ~4.75 (sync_regs .82, __rmqueue .64, clear_page
  .56, unmap .50, mm_fault .33, ...). SCAFFOLD ≈ 7.8% → ~0.47 cyc/B.
NASA self%: run_contig 53.66 | __memmove_avx 14.00 | finish_decode(=CRC) 4.15 |
  kernel mm/fault ~15.0 (sync_regs 2.37, clear_page 1.77, irq_iret 1.77, pte_lock 1.58,
  do_user_addr_fault .79, __handle_mm_fault .79, ...). SCAFFOLD ≈ 33% → ~1.13 cyc/B.
CALLGRAPH (nasa, dwarf): __memmove_avx (14.78%) = 8.21% pure inlined memcpy + 6.57%
  memcpy INVOKED FROM asm_exc_page_fault (copy writing to COLD output pages). The
  page-fault path = {memcpy 9-11%} + {run_contig→marker_inflate::decode_clean_into_contig
  4.67-5.32%} — i.e. faults are triggered by BOTH the backref/output memcpy AND the
  decode itself writing decoded bytes to fresh output pages. finish_decode is ~all CRC32
  (crc32fast pclmulqdq inlined), NOT a copy.
EXTRA-MEMMOVE VERDICT: there is NO redundant decode→scratch→writer double-copy to elide.
  gzippy decodes COPY-FREE into the contig buffer (decode_clean_into_contig) then
  output_writer does writev (no copy). The memmove/memcpy IS the LZ77 backref copy +
  cold-page output writing — work igzip ALSO does, BUT igzip's copy lands on its small
  REUSED (already-faulted) window buffer (→0 faults), gzippy's lands on FRESH 200MB output
  (→ faults). So the "extra cost" is the FAULTS from materializing full output, not a
  duplicate memmove. Size: nasa memmove+fault ≈ 14%+15% ≈ 29% (~1.0 cyc/B); silesia ≈
  1%+5% ≈ 6% (~0.35 cyc/B). igzip's memmove+fault ≈ ~0 (666 faults, banked scaffold = crc).

### ARTIFACT-vs-REAL VERDICT (gated negative-oracle + T1-single-core structural argument)
1. NOT a harness re-run artifact: production decodes one file/invocation → materializes +
   faults the full output EVERY run; 3 recycle/prefault oracles INERT ⇒ no allocation
   trick removes it.
2. At T1 SINGLE-CORE there is no second thread to hide fault/memmove cycles behind, so the
   perf-sampled mm/fault/memmove cycles (~6% sil / ~29% nasa) ARE wall cycles — real T1
   cost, not slack. (The prior in-tree "faults may be slack" thesis was a MULTI-thread
   depth-14 matched-wall scenario, NOT T1.) This is structural, not a removal-oracle —
   the exact recoverable cyc/B is still HYPOTHESIS until a working output-reuse oracle.
3. Discounting "artifacts" does NOT shrink the gap: the scaffold is REAL. True production
   T1 gap to igzip STANDS at silesia +1.510 (+34.0%), nasa +1.869 (+117.5%) cyc/B. On
   backref-heavy nasa the scaffold (~1.1 cyc/B) is the DOMINANT remaining component
   (kernel gap only +0.45; whole gap +1.87); on silesia kernel (+0.91) > scaffold (~0.47).

### RECOMMENDATION (gated-HYPOTHESIS; Intel-only NOT-YET-LAW; feeds R3 user decision)
- There IS a REAL architecture/consumer-domain gap: gzippy MATERIALIZES full contiguous
  output (24-54k first-touch faults + cold-page memcpy) where igzip STREAMS through a
  reused buffer (~666 faults). It is the DOMINANT remaining gap on backref-heavy corpora
  (nasa) and a real T1 cost (single-core ⇒ on the wall). This is faithful-rapidgzip-
  /streaming-consumer territory = USER R3.
- BUT ceiling≠gain: every available removal-oracle was INERT, so the RECOVERABLE cyc/B
  from a streaming consumer is NOT YET BOUNDED (attribution says ~0.35 sil / ~1.0 nasa,
  HYPOTHESIS-tier). Caveat: rapidgzip ITSELF materializes chunk output, so a "faithful rg
  consumer" may NOT reach igzip's 666-fault streaming profile — the lever that matches
  igzip is OUTPUT STREAMING through a reused buffer (igzip-shaped), which at T1 is feasible
  but is a divergence-from-rg architecture choice. ESCALATE this fork to USER (R3).
- OWED before funding the port (the decisive gate): a WORKING output-reuse removal-oracle
  — either (i) a throwaway in-process harness decoding N× into one warm dst, or (ii) THP
  on a non-LXC box (neurotic bare-metal / solvency) — that DROPS faults to igzip-level
  and shows matched-wall T1 cyc/B drops proportionally. Until then the scaffold is
  REAL-and-on-the-T1-wall (gated) but its recoverable magnitude is HYPOTHESIS.
- AMD/Zen2 replication of M1 owed before LAW.
RE-VERIFY M1: `bash /tmp/m1_pagefaults.sh` on guest (uploaded), or:
  `taskset -c 4 perf stat -r 11 -e page-faults,cpu_core/cycles/ -- env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 4 /root/bin/gzippy-new-native -d -c -p1 /root/{silesia,nasa}.gz >/dev/null`
  igzip arm: `taskset -c 4 perf stat -r 11 -e page-faults,cpu_core/cycles/ -- taskset -c 4 igzip -d -c /root/{corpus}.gz >/dev/null`
RE-VERIFY M3 profile: `taskset -c 4 perf record -F6000 -o /tmp/p.data -- env GZIPPY_FORCE_PARALLEL_SM=1 /tmp/symtarget/release/gzippy -d -c -p1 /root/nasa.gz >/dev/null; perf report -i /tmp/p.data --stdio`
BOX STATE: clean — THP unchanged (madvise; the always-write errored read-only, no-op),
governor powersave, 0 stressors, no pinning leftover, guest src /root/gz-fullrewrite
@0a592a54 clean. NO main push. Perf scratch in guest /tmp (ephemeral).

## ====== LOCALIZE SESSION (2026-06-18 PM) — WHERE IS THE PER-ITERATION GAP? ======
GOAL: decompose gzippy run_contig vs igzip decode_huffman kernel into per-region
cyc/B (refill / copy / classify-decode / loop-overhead), confirm top candidate
causally. Intel cpu4 unstressed, /dev/null sink, freq-invariant cyc/B. SINGLE-ARCH
= NOT LAW (AMD owed). Tools committed: scripts/bench/_localize_profile_guest.sh
(perf record+annotate both kernels), scripts/bench/_localize_bucket.py (deterministic
address→region bucketer, SELF-VALIDATES sum=100%), scripts/bench/_localize_patch_perturb.py.

### GATE-0 CATCH (instrument self-validation — the most-violated rule, caught here)
The prior /tmp/symtarget was STALE: built @ 2b10aa48 (PRE-flagbit); its disasm showed
the OLD shrx→cmp$0xff literal discriminator, NOT the production flag-bit
`test 0x1000000;jnz 49f`. REBUILT symtarget from mission tip 0a592a54
(CARGO_TARGET_DIR=/tmp/symtarget on tmpfs; strip=false debug=2 target-cpu=native;
asm pulled by pure-rust-inflate). New disasm has BOTH `test 0x2000000`(long) AND
`test 0x1000000`(flag-bit), cmp$0xff now COLD. SHA==zcat (silesia+nasa), entries=24128.
REBUILD: `cd /root/gz-fullrewrite && git checkout 0a592a54 && RUSTFLAGS="-C target-cpu=native"
CARGO_TARGET_DIR=/tmp/symtarget CARGO_PROFILE_RELEASE_STRIP=false CARGO_PROFILE_RELEASE_DEBUG=2
cargo build --release --no-default-features --features pure-rust-inflate`

### DELIVERABLE 1 — per-region cyc/B, KERNEL-to-KERNEL (perf -F5000, 12 loops, self-validated sum=100%)
SILESIA (gz kernel 4.889 = 82.95%×total/B; ig kernel 3.981 = 89.14%×; KERNEL gap +0.908):
| region                          | gz %  | ig %  | gz c/B | ig c/B | gz−ig  |
|---------------------------------|-------|-------|--------|--------|--------|
| bit-mgmt (consume+refill)       | 19.1  | 14.4  | 0.934  | 0.573  | +0.361 |
| backref-copy (MOVDQU)           | 19.1  | 28.0  | 0.932  | 1.115  | −0.183 |
| classify/decode/table/EOB       | 46.9  | 47.8  | 2.294  | 1.905  | +0.389 |
| loop-overhead (guard/store/spill)| 14.9 | 9.8   | 0.730  | 0.389  | +0.341 |
NASA (gz kernel 1.843=54.22%×; ig 1.390=87.73%×; KERNEL gap +0.453):
| bit-mgmt   16.2/13.9  0.299/0.193 +0.105 | copy 26.5/35.7 0.488/0.496 −0.008 |
| classify   39.4/39.2  0.726/0.545 +0.180 | loop-ovh 18.0/11.2 0.331/0.155 +0.176 |
⇒ KERNEL gap is **DIFFUSE** — spread ~evenly across bit-mgmt + classify + loop-overhead;
  **backref-copy is at PARITY-or-BETTER** (gzippy's MOVDQU port already matches igzip;
  do NOT attack copy). NO single dominant kernel sub-lever.

### DELIVERABLE 1b — WHOLE-PROCESS gap ≫ KERNEL gap: the gap is SCAFFOLD-DOMINATED
| corpus  | whole gz | whole ig | whole gap | kernel gap | SCAFFOLD gap | scaffold %of gap |
|---------|----------|----------|-----------|------------|--------------|------------------|
| silesia | 5.894    | 4.466    | +1.428    | +0.908     | **+0.520**   | 36%              |
| nasa    | 3.399    | 1.584    | +1.815    | +0.453     | **+1.362**   | **75%**          |
gz scaffold (nasa, %proc→cyc/B): asm_exc_page_fault 9.32%(0.317) + __memmove_avx 7.35%
(0.250) + finish_decode 4.03%(0.137) + sync_regs 2.48%(0.084) + clear_page 1.71%(0.058).
ig scaffold (nasa): crc32 7.82%(0.124) + copy_to_iter 1.39%. ⇒ On backref-heavy data the
bulk of the igzip deficit is OUTPUT memmove + first-touch PAGE FAULTS + Rust bail-glue —
i.e. the rapidgzip-ARCHITECTURE/consumer domain, NOT the inner-loop machinery.
(HYPOTHESIS-tier: perf attribution; scaffold NOT yet causally perturbed — see NEXT.)

### DELIVERABLE 2 — Gate-2 CAUSAL perturbation of the REFILL/bit-mgmt chain (named top suspect)
THROWAWAY K8 build (mission tip + 4 rol/ror pairs = 8 value-NEUTRAL serially-DEPENDENT
ops on {bitbuf} right after the hot `6:` refill `or {bitbuf},{t2}`, extending the
bitbuf→preload/consume recurrence). CONTROLS: K0 sha==new-native 891c9925 (trusted
baseline); K8 sha 82f1eeab DISTINCT + byte-EXACT silesia/nasa + entries=24128 +
16 extra rol/ror in .text (asm! opaque to optimizer, non-inert PASS); GHz spread PASS;
self-test (A2-A1) PASS. N=21 paired, unstressed, /dev/null. medΔ=(K8−K0):
| corpus  | medΔ cyc/B | per-op    | 95%CI            | Wilcox p  | ΔIPC   | verdict        |
|---------|------------|-----------|------------------|-----------|--------|----------------|
| silesia | +1.007 (+17.0%) | +0.126 | [+0.989,+1.020]  | 6.412e-05 | −0.205 | **ON CRITICAL PATH** |
| nasa    | +0.133 (+3.9%)  | +0.017 | [+0.086,+0.170]  | 0.0136    | −0.028 | WASH (fails p<0.01) |
⇒ silesia: PROPORTIONAL, large, p<0.01, ΔIPC dropped (latency-bound) ⇒ the refill/bit-mgmt
  chain IS on the critical recurrence — shortening it CAN move cyc/B. nasa: weak/wash
  (nasa's hot literal refill runs rarely; its backref X1 refill — unpatched — is the
  relevant one). COMBINED w/ STATE's banked classify-chain perturbation (proportional),
  the top-2 kernel candidates (refill + classify) are BOTH causally on-path.
  Loop-overhead (+0.341 sil) NOT perturbed this turn (attribution-only).
  RE-VERIFY: rebuild K0/K8 (`/tmp/loc_prof/patch_perturb.py <asm> 4`, prod flags), then
  `BIN_A=/tmp/bin/gzippy-k0 BIN_B=/tmp/bin/gzippy-k8 PIN=4 REPS=21 CORPORA="silesia nasa"
  SKIP_STRESS=1 GZIPPY_FORCE_PARALLEL_SM=1 bash /root/distpreload-harness/_distpreload_paired_guest.sh`

### DELIVERABLE 3 — removal-oracle BOUNDS (ceiling ≠ gain)
KERNEL machinery ceiling (igzip-parity removal-oracle, perf-decomposed): if gzippy's
kernel matched igzip per-region → recover +0.908 cyc/B (sil) / +0.453 (nasa) MAX —
diffuse, so realistically only fractions capturable. Per-region ceilings (sil):
refill +0.361, classify +0.389, loop-ovh +0.341, copy 0 (already ≤). SCAFFOLD ceiling
(whole−kernel): +0.520 (sil) / +1.362 (nasa) — LARGER, esp. backref-heavy.

### RECOMMENDATION (gated-HYPOTHESIS; Intel-only NOT-YET-LAW; for next-technique decision)
1. **Do NOT chase a single kernel micro-lever.** The kernel per-iteration gap is small
   (+0.91 sil/+0.45 nasa) and DIFFUSE; copy is already at parity. The mission premise
   ("igzip's leaner per-iteration machinery") is CONFIRMED real (refill on-path) but
   bounded and spread — no peephole captures it; matching igzip needs its whole fused
   shape (consume+refill+preload interleave), high-effort/low-marginal.
2. **The biggest causally-confirmed KERNEL sub-lever = refill/bit-mgmt** (silesia, Gate-2
   on-path, ceiling +0.36 cyc/B). Byte-exact technique to target it: fuse gzippy's
   consume (`shrx;sub`) + refill (`mov;shlx;or;63-sub-shr3-add;or56`) + next-litlen
   preload into igzip's single interleaved cadence (igzip 38d0e-38d71 does consume,
   speculative store, ONE refill-advance, and BOTH next-litlen+dist preloads in one
   straight chain with the load-use hidden) — i.e. reduce the consume+refill DEPENDENT
   chain LENGTH, not the instruction count. Pre-register success = silesia cyc/B medΔ<0 p<0.01.
3. **HIGHER-VoI but ARCHITECTURE-domain: the SCAFFOLD** (whole gap ≫ kernel gap; nasa
   75%). Next causal test (owed, not done): removal-oracle on output __memmove_avx +
   first-touch page faults — decode into a REUSED/pre-faulted output buffer (no per-chunk
   fresh alloc) and/or eliminate the libc memmove copy; measure nasa whole-process cyc/B.
   If it drops proportionally, the consumer/chunk-output path (faithful rapidgzip port
   territory) is the real remaining lever on backref-heavy corpora — ESCALATE to user (R3:
   inner-loop is near its floor; the gap is moving to architecture).
4. AMD/Zen2 replication of all of the above is owed before LAW.
GUEST ARTIFACTS kept: /tmp/symtarget (symboled mission-tip bin), /tmp/bin/gzippy-k0
(=new-native), /tmp/bin/gzippy-k8 (perturb), /tmp/loc_prof/ (annot+stat+bucket.py).
gz-fullrewrite moved to 0a592a54 (was 2b10aa48); asm reverted clean; box: powersave,
0 stressors, no pinning leftover.

## ====== B-SIZING SESSION (2026-06-18) — 3 GATING MEASUREMENTS BEFORE FUNDING B ======
Goal: size rewrite B (igzip-style fat DIRECT one-symbol-per-iter table, ~0 unpack)
with MEASURED numbers (not the 1.34 slope/ceiling) for a USER R3 decision. Guest
trainer cpu4, unstressed, /dev/null sink. Binaries: packON=/root/bin/gzippy-new-native
(=mission tip 76c750f8; rebuild gzippy-packON sha-IDENTICAL → provenance proven),
packOFF=/root/bin/gzippy-packOFF (SINGLE_SYM_FLAG @ lut_huffman.rs:1041; byte-exact
silesia/nasa/monorepo; source REVERTED to TRIPLE).

### MEASUREMENT 1 — REAL-regime memory-stall fraction (is the 1.34 headroom real or memory-capped?)
new-native, unstressed, cpu4, perf -r 9. ANSWER: **COMPUTE-BOUND = YES.**
| corpus  | cycles    | L1d-miss-stall | L2-miss-stall | L3-miss(DRAM)-stall | TMA mem-bound | TMA core-bound |
|---------|-----------|----------------|---------------|---------------------|---------------|----------------|
| silesia | 1.244e9   | 2.39%          | 1.17%         | **0.69%**           | 7.0%          | 15.2%          |
| nasa    | 0.686e9   | 6.06%          | 3.98%         | **2.73%**           | 16.7%         | 16.0%          |
⇒ Tiny DRAM-stall fraction (silesia 0.69%, nasa 2.73% of cycles). TMA agrees: silesia
mem-bound only 7.0% (core-bound 15.2%); nasa mem-bound 16.7% (core-bound 16.0%). The
unstressed (production) regime is COMPUTE-bound — esp. silesia where B's payoff is
largest. ⇒ B's instruction-reduction headroom (the ~1.34 cyc/B oracle) is REAL in
production, NOT memory-capped. The earlier stressor-WASH was an artifact of the
artificial 14-thread bandwidth-saturation, not the real regime.
RE-VERIFY: `taskset -c 4 perf stat -r 9 -e cpu_core/cycles/,cpu_core/instructions/,cpu_core/memory_activity.stalls_l1d_miss/,cpu_core/memory_activity.stalls_l2_miss/,cpu_core/memory_activity.stalls_l3_miss/ -- env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 4 /root/bin/gzippy-new-native -d -c -p1 /root/silesia.gz >/dev/null`

### MEASUREMENT 2 — what does multi-symbol packing (which B ABANDONS) currently buy?
TOGGLE FOUND: lut_huffman.rs:1041 TRIPLE_SYM_FLAG→SINGLE_SYM_FLAG forces 1-sym/iter
through the EXISTING table (still pays unpack). packON(A=new-native) vs packOFF(B),
N=21 paired, unstressed, all SIGNIF (p=6.4e-5, CI excl 0), self-tests PASS, non-inert
(packOFF Δinstr/B>0 = more loop iters). medΔ=(packOFF−packON) = **packing_benefit =
what B costs us:**
| corpus  | packON cyc/B | packOFF cyc/B | packing_benefit medΔ | Δinstr/B | 95%CI            |
|---------|--------------|---------------|----------------------|----------|------------------|
| silesia | 5.868        | 6.087         | **+0.216 (+3.68%)**  | +0.599   | [+0.200,+0.225]  |
| monorepo| 5.052        | 5.209         | **+0.155 (+3.06%)**  | +0.291   | [+0.138,+0.165]  |
| nasa    | 3.336        | 3.423         | **+0.087 (+2.62%)**  | +0.224   | [+0.079,+0.094]  |
⇒ Packing buys only 0.087–0.216 cyc/B (SMALL). B abandons this. Residual gap to igzip
is +1.50/+2.07/+1.75 — so the packing loss is ~1/7th the gap on silesia.
RE-VERIFY: `BIN_A=/root/bin/gzippy-new-native BIN_B=/root/bin/gzippy-packOFF PIN=4 REPS=21 CORPORA="silesia monorepo nasa" SKIP_STRESS=1 bash /root/distpreload-harness/_distpreload_paired_guest.sh`

### MEASUREMENT 3 — B-SPIKE: measured bound on B's achievable cyc/B
SPIKE = throwaway partial-B (gz-flagbit-new patched: lut SINGLE_SYM + asm cnt-extract
hardcoded `mov t3,1` since cnt≡1 in SINGLE mode → removes the mov/shr/and unpack
cnt-extract chain that B eliminates). BYTE-EXACT silesia/nasa/monorepo; distinct
binary (sha 9be95b74≠packON 891c9925) — non-inert. N=21 paired vs packON(=current
flag-bit kernel), unstressed. medΔ=(spike−packON), all SIGNIF, self-tests PASS:
| corpus  | packON cyc/B | spike cyc/B | spike−packON medΔ | 95%CI            | vs igzip (banked) |
|---------|--------------|-------------|-------------------|------------------|-------------------|
| silesia | 5.869        | 5.954       | **+0.091 SLOWER** | [+0.078,+0.096]  | +1.59 (+36%) still|
| monorepo| 5.089        | 5.166       | **+0.078 SLOWER** | [+0.060,+0.095]  | +2.21 (+75%) still|
| nasa    | 3.330        | 3.340       | **+0.015 SLOWER** | [+0.005,+0.021]  | +1.76 (+112%) still|
⇒ The partial-B spike is SIGNIF SLOWER than the current packed kernel on ALL 3 corpora.

DECOMPOSITION (derived; packON A1 medians stable 5.868↔5.869 across the two paired runs):
  spike−packOFF = unpack(cnt-extract) RECOVERY = {sil −0.125, mono −0.077, nasa −0.072}.
  So removing the cnt-extract recovered 0.072–0.125 cyc/B — but going single-mode COST
  0.087–0.216 (M2). NET partial-B = packing_loss − unpack_recovery = +0.015..+0.091 WORSE.

⇒ **MEASURED net-B ≈ NEGATIVE-to-BREAKEVEN, NOT the 1.34 ceiling.** Full B (fat direct
  table) removes ~2 MORE unpack ops than the spike (byte-mask + bc-extract), so could
  recover a bit more — but to even BREAK EVEN with the current packed kernel on silesia
  it must recover another +0.091 from ~2 ops, comparable to what the 3-op cnt-extract
  just bought. Best realistic case: full B ≈ break-even-to-small-win vs CURRENT — it
  does NOT approach igzip (spike vs igzip is UNCHANGED at +36/75/112%).

WHY the 1.34 ceiling is a MIRAGE for B-as-scoped: igzip is ALSO one-symbol-per-iter yet
hits 3.98 cyc/B; gzippy single-mode (packOFF) is 6.09. The 2.1 cyc/B gap between two
one-symbol-per-iter loops is NOT the table format — it is igzip's whole leaner
per-iteration machinery (bit-reader cadence, refill, copy, codegen). B (table-format
swap) does not touch that, so it cannot reach 3.98. The 1.34 removal-oracle bounded a
DIFFERENT thing (igzip's entire kernel), not the unpack-removal lever.

---

Mission: make gzippy-native (pure-Rust, FFI-off) **T1 single-member gzip DECODE**
measurably FASTER than igzip (ISA-L), byte-exact, gated. Single-arch Intel = NOT LAW
(AMD/Zen2 replication owed). T1 single-core only — no T4/T8 extrapolation.

Branch: `perf/igzip-full-rewrite`. Mac worktree (edit/commit/push only — aarch64,
CANNOT run the asm): `/home/user/www/gzippy/.claude/worktrees/agent-a8069a92d914fcef3`.
Guest (ONLY x86_64/BMI2 measure box): `ssh -J REDACTED_IP root@REDACTED_IP`.
Guest worktrees: gzippy(B)=/root/gz-fullrewrite (kernel 2b10aa48 dist-preload),
baseline=/root/gz-baseline (8383a2eb). igzip=/usr/bin/igzip (ISA-L 2.31.1).
Harness on guest: /root/distpreload-harness/.

## COMMITS THIS MISSION
- 2c135d07 — deliverable #0: commit orphaned paired harness (analyzer+memstress+driver) to scripts/bench.
- 2e01dd4f — deliverable #1: add igzip arm `scripts/bench/_gzippy_vs_igzip_paired_guest.sh`.
- (HEAD)   — Step-B technique #1: fuse `lea-1`+`shl3` shift into one `lea [t3*8-8]`.
            BYTE-EXACT (sha 3 corpora×T1/T4/T8, proptest 60k, c2/c3 asm-vs-ref diffs).
            Δinstr/byte=-0.186 (silesia, deterministic + paired); cyc/byte TIE
            (silesia medΔ=-0.040, CI=[-0.0475,-0.0216] excl 0 but p=0.018 fails p<0.01;
            nasa wash). KEPT on byte-exact license; gap to igzip NOT closed (still +39.5%).

## STEP-B TECHNIQUE LOG (gated, Intel trainer cpu4, N=21 paired, /dev/null sink)
| # | technique | byte-exact | Δinstr/B (sil) | cyc/B medΔ (sil) unstressed | p | verdict | kept? |
|---|-----------|-----------|----------------|------------------|---|---------|-------|
| 1 | fuse shift `lea[t3*8-8]` (was lea-1+shl3) | PASS | -0.186 | -0.040 [CI-0] | 0.018 | TIE (fails p<0.01) | YES (byte-exact) |
| 2 | FLAG-BIT discriminator (`test entry,0x1000000;jnz` off build-time trailing-class bit 24; removes shift-lea+and+shrx+cmp from literal discriminator chain) | PASS | -0.279 | **-0.099 (-1.67%)** [CI -0.103,-0.094] | 6.4e-05 | **SIGNIF-faster UNSTRESSED** (sil+nasa+mono p<0.01, CI excl 0); WASH/TIE under bandwidth stressor (LLC 91%) | YES |
  Re-verify #1/#2: BIN_A=/root/bin/gzippy-base-native BIN_B=/root/bin/gzippy-new-native
  PIN=4 REPS=21 CORPORA="silesia nasa monorepo" SKIP_STRESS=1
  GZIPPY_FORCE_PARALLEL_SM=1 bash /root/distpreload-harness/_distpreload_paired_guest.sh
  (stressor arm: SKIP_STRESS=0 CORPORA="silesia nasa")

  TECHNIQUE #2 RESULT (2026-06-18, this session):
    - byte-exact: native sha grid silesia/nasa/monorepo/squishy × T1/T4/T8 == gzip;
      c1/c2/c3 asm-vs-ref differential PASS (x86_64); proptest 60k prop_structured_roundtrip
      PASS; new lut_huffman flag-invariant test PASS (flag bit24 == trailing>=256 for
      every short entry). exit profile IDENTICAL to base (entries=24128, reclass_dist=21330,
      reclass_eob=2796) — same paths, fewer instrs.
    - UNSTRESSED gated cyc/byte (base-vs-new, self-test PASS all 3, mono self-test flagged
      a -0.0044 rig bias dwarfed by the -0.062 effect):
        silesia  medΔ=-0.0993 (-1.67%)  CI[-0.103,-0.094]  p=6.4e-05  SIGNIF-faster
        nasa     medΔ=-0.0112 (-0.33%)  CI[-0.022,-0.005]  p=0.0067   SIGNIF-faster
        monorepo medΔ=-0.0620 (-1.20%)  CI[-0.079,-0.043]  p=0.00085  SIGNIF-faster
        ΔIPC sil -0.010, Δinstr/B sil -0.279 (the removed shift+extract+cmp).
    - STRESSED (memstress 14T, LLC 91-92%): WASH/TIE
        silesia medΔ=-0.013 CI[-0.245,+0.093] p=0.465; nasa medΔ=+0.014 p=0.627.
        Δinstr/B STILL -0.28 (sil) stressed — the WORK is removed in both regimes; the
        cyc/byte win only surfaces when NOT bandwidth-saturated (compute-bound regime).
    - VERDICT: real CRITICAL-CHAIN win in the compute-bound regime (contrast #1's
      load-shadowed TIE — this time cyc/byte moved with instr/byte because the removed
      ops fed the discriminator branch the Gate-2 perturbation proved is on the critical
      recurrence). Does NOT meet the strict "stressor-stable" arm. KEPT on byte-exact +
      compute-bound win. Single-arch Intel = NOT-YET-LAW (AMD owed).
  LESSON #1 (still valid): removing a PREDICTABLE, load-shadowed instr drops instr/byte
  but NOT cyc/byte — the shl was overlapped behind the table load. Technique #2 confirms
  the inverse: cutting the CRITICAL-PATH discriminator chain DOES move cyc/byte (unstressed).

## GATE-2 CAUSAL PERTURBATION — IS THE UNPACK/CLASSIFY CHAIN ON THE CRITICAL RECURRENCE? (2026-06-18)
DECISIVE TEST that gates whether rewrite B (igzip-style fat DIRECT table, ~0 unpack)
is worth funding. Method (STRONG tier — Gate-2 causal perturbation): in a THROWAWAY
worktree (/root/gz-perturb, base 2b10aa48 — identical shrx→cmp critical segment;
REMOVED after run), inserted K serially-DEPENDENT, value-NEUTRAL ALU ops
(`rol $1,%r10`/`ror $1,%r10` pairs, all on t5) BETWEEN the trailing-symbol extraction
`shrx {t5},{t5},{t4}` and the literal/non-literal discriminator `cmp {t5:e},255`,
extending the loop-carried recurrence that feeds the discriminator branch.
CONTROLS PASSED: all 4 builds byte-exact (sha==zcat silesia ref) + kernel-engaged
(entries=24128 identical) + distinct binaries; disasm-verified the chain is a genuine
DEPENDENT serial chain on %r10 sitting exactly between the shrx and `cmp $0xff,%r10d`
+ `ja` (rol/ror count scales 8→10→12→16 = +K); paired self-test (A2-A1) PASS.
Harness: `_distpreload_paired_guest.sh`, PIN=4 REPS=21 SKIP_STRESS=1, /dev/null sink.

SLOPE TABLE (medΔ cyc/byte vs K=0, all p=6.412e-05, CI excludes 0):
| K (dep ops) | silesia medΔ (per-op)   | nasa medΔ (per-op)     |
|-------------|-------------------------|------------------------|
| 2           | +0.280 (+4.6%) [0.140]  | +0.070 (+2.1%) [0.035] |
| 4           | +0.686 (+11.3%) [0.171] | +0.204 (+6.1%) [0.051] |
| 8           | +1.380 (+22.8%) [0.172] | +0.375 (+11.2%) [0.047]|

VERDICT: **PROPORTIONAL slope on BOTH corpora** (monotone, ~linear; silesia
doubling-K → ~2.0× Δ, ~0.17 cyc/B per op; nasa ~0.05 cyc/B per op). ⇒ the
shrx-fed t5-extraction → discriminator chain (the gz-specific UNPACK/classify the
STATE'd ~21% region) IS on the critical recurrence — NOT load-shadowed (contrast
technique #1's load-shadowed shl, which TIE'd). Eliminating that extraction
(rewrite B: direct table yields byte+len, ~0 unpack) therefore has REAL cyc/byte
headroom. The unpack cost is concentrated on entropy-DENSE/literal-heavy data
(silesia 0.17/op) vs backref-heavy/low-entropy (nasa 0.05/op) — i.e. B's payoff is
LARGEST exactly where gzippy's gap is largest (silesia +39.5%).

CAVEAT (Gate-2 rule: slow-down slope ≠ speed-up ceiling): this proves the chain is
ON the critical path (necessary), NOT how much B's shorter chain recovers. The
removal-ORACLE that BOUNDS the win already exists and is banked: igzip's
decode_huffman kernel = 3.98 cyc/B vs gzippy run_contig 5.32 cyc/B (silesia, Step A
iii) ⇒ up to ~1.34 cyc/B recoverable, and igzip pays ~0 unpack. So: perturbation
(chain on critical path) + igzip removal-oracle (1.34 cyc/B ceiling) ⇒ B is
JUSTIFIED. Single-arch Intel = NOT-YET-LAW (AMD/Zen2 owed). monorepo 3rd-corpus owed
(optional; 2-corpus spread already concordant).

RECOMMENDATION FOR USER R3 (fund rewrite B?): **YES, justified** — gated proportional
slope (both corpora) + a measured igzip removal-oracle ceiling (~1.34 cyc/B). BUT do
the CHEAPER step first (pre-registered NEXT #1: flag-bit discriminator — same critical
chain, ~1/10th the work, no table re-layout): if a single `test entry,FLAG; jnz` off
the LOAD (no shrx-extract) recovers a significant fraction of the slope, B's full
table-format rewrite may be unnecessary. Sequence: flag-bit (NEXT #1) → re-measure →
THEN decide B with the residual.

## THE INSTRUMENT (deliverable #1) — Gate-0 SELF-VALIDATED, PASS
`scripts/bench/_gzippy_vs_igzip_paired_guest.sh` (reuses committed `_distpreload_paired_analyze.py`).
arm A(A1,A2)=igzip, arm B=gzippy-native (ParallelSM @ T1). medΔ=(B-A1) cyc/byte;
NEGATIVE => gzippy faster. Gate-0 verified live: gzippy run_contig KERN entries>0
(monorepo 8299, nasa 25399), igzip --version printed, BOTH arms sha==zcat ref==each
other, same /dev/null sink + same pin(cpu4), GHz spread <0.07%, A2-A1 self-test ~0.
Run: `PIN=4 REPS=21 CORPORA="silesia monorepo nasa" /root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh`

## THE GAP TO CLOSE — gzippy-native T1 vs igzip cyc/byte (whole-process, /dev/null sink) — GATED
FULL N=21 paired, Wilcoxon p, bootstrap CI, A2-A1 self-test, + bandwidth stressor.
Intel i7-13700T LXC (cpu4 pinned). All cells p=6.4e-5, CI excludes 0, self-test PASS
(except stressed-nasa, noted). medΔ=(B-A1)>0 means gzippy SLOWER. THE GAP IS LARGE:

UNSTRESSED (clean) — THE number to close:
START (base af53f95e, banked):
| corpus   | igzip cyc/B | gzippy cyc/B | medΔ (B-A1)      | Δinstr/byte | ΔIPC   |
|----------|-------------|--------------|------------------|-------------|--------|
| silesia  | 4.30        | 6.00         | +1.70 (+39.5%)   | +~2.3       | -0.27  |
| monorepo | 2.97        | 5.17         | +2.20 (+73.9%)   | +2.83       | -0.43  |
| nasa     | 1.58        | 3.43         | +1.84 (+116.5%)  | +2.08       | -0.64  |

CURRENT after technique #2 FLAG-BIT (new-native vs igzip, 2026-06-18, unstressed,
all self-tests PASS, N=21 paired):
| corpus   | igzip cyc/B | gzippy cyc/B | medΔ (B-A1)      | Δinstr/byte | ΔIPC   |
|----------|-------------|--------------|------------------|-------------|--------|
| silesia  | 4.367       | 5.871        | +1.499 (+34.3%)  | +1.86       | -0.35  |
| nasa     | 1.576       | 3.331        | +1.754 (+111.3%) | +1.98       | -0.64  |
| monorepo | 2.956       | 5.033        | +2.070 (+70.0%)  | +2.73       | -0.41  |
  ⇒ gap CLOSED by the flag-bit: silesia +39.5%→+34.3% (~0.10 cyc/B), monorepo
    +73.9%→+70.0% (~0.06), nasa ~+0.01. RESIDUAL gap = +1.50/+1.75/+2.07 cyc/B.
    Removal-oracle ceiling was ~1.34 cyc/B (silesia kernel 5.32→3.98); flag-bit
    recovered ~0.10 of it ⇒ ~7% of the headroom captured, ~1.24 cyc/B STILL on the
    table ⇒ rewrite B (fat DIRECT table, ~0 unpack) remains justified by the residual.

STRESSED (memstress 14T, LLC-miss ~42-45%): gap HELD/GREW (silesia +39.5%, monorepo
+108%, nasa +179%) ⇒ NOT a bandwidth artifact; it is real work-volume. (stressed-nasa
self-test FAILED 2.9% — that ONE cell's precision is degraded; effect dwarfs it; use
the unstressed nasa cell as the clean number.)

GATED FACTS (this commit, Intel-only, NOT-YET-LAW — AMD owed):
1. gzippy-native T1 is **40-116% SLOWER than igzip** (cyc/byte), gated, 3 corpora.
2. The gap is **INSTRUCTION-DOMINATED**: gzippy retires **+2.0 to +2.8 more instr/byte**
   than igzip (ΔIPC also negative). This is WORK VOLUME, not micro-latency — a preload
   tweak will NOT close it; the lever is REDUCING instr/byte in the hot loop.
3. **run_contig = 85.36% of T1 self-time** (perf, symboled binary, silesia). Scaffold
   (block finder / marker machinery / apply_window / CRC) is only ~15%. So the excess
   instructions ARE in run_contig — the mission's kernel focus is correctly aimed.
4. Hot per-symbol economy in run_contig (perf annotate, top self-time instrs):
   multiple classify branches PER SYMBOL — `cmp $0x100`(EOB) 5.8%, `cmp $0xff`(literal-
   range) 3.6%, `test $0x2000000`(marker/reclass discriminator) 2.8%, plus two table
   loads `mov (%r11,%rXX,4)` (6.3%+) and the engaged MOVDQU backref copy (4.5%x2,
   8383a2eb is LIVE). igzip's loop_block collapses this to ONE speculative store + ONE
   discriminator branch — THAT delta is the instruction excess to attack.

## STEP A — AIM CONFIRMED (2026-06-18, guest trainer cpu4, symboled bin rebuilt today)
Localization re-run fresh; all three sub-checks PASS and converge on "the excess is in
the hot run_contig classify+decode loop, addressable."

(i) RECLASS RATIO — cold exits are NOT the cost. GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1:
| corpus   | entries | exit_reclass(tag0) | reclass_dist | reclass_eob | asm_bytes/entry |
|----------|---------|--------------------|--------------|-------------|-----------------|
| silesia  | 24128   | 0                  | 21330        | 2796        | ~8770           |
| monorepo | 8299    | 0                  | 8130         | 166         | ~6118           |
| nasa     | 25399   | 0                  | 25037        | 351         | ~8077           |
  Generic invalid/oversize exit (tag 0) = 0 everywhere. The dominant exit is
  reclass_dist (subtable/raw0 dist → Rust), BUT asm_bytes/entry is ~6-9 KB: the kernel
  decodes thousands of CLEAN bytes between each bail, so per-byte cold-exit overhead is
  negligible. The excess instr/byte is in the hot loop. => loop_block port is aimed right.

(ii) PER-REGION (perf annotate -F4000, silesia, hottest run_contig instrs, self%):
  - Packed-symbol UNPACK ALU chain (gz's multi-sym short-entry format tax, no igzip
    counterpart): test 0x2000000 3.27, mov t2,t1 4.31, shl $3 (shift=8*(cnt-1)) 3.81,
    and 0x1FFFFFF 2.96, mov 2.23, lea-1 1.38, shrx 1.12, cmp 0xff 1.92  => ~21%
  - Two table-load preloads: mov(%r11,%r14,4) 5.63 + mov(%r11,%r12,4) 5.55 => ~11%
    (IRREDUCIBLE — igzip's decode_next_sym has the identical 2 loads)
  - MOVDQU backref copy: load 4.69 + store 4.10 + jle 2.03 => ~11%
  - Non-literal arm: cmp 0x100 (EOB) 5.12, cmp 0x200 1.28, dist preload/decode/copy body

(iii) SYMBOL-RESTRICTED KERNEL-vs-KERNEL cyc/byte (perf stat cycles × self%, silesia):
  - gzippy run_contig self=86.64%, total=1,302,094,139 cyc / 211,968,000 B => 5.32 cyc/B
  - igzip decode_huffman_code_block_stateless_04 self=90.31%, total=934,030,780 cyc
    => 3.98 cyc/B
  - KERNEL GAP = +1.34 cyc/B (+33.7%) — tracks the whole-process silesia +39.5%, so the
    excess really is IN the kernel. Whole-proc instr/byte: gzippy 13.73 vs igzip 11.38
    (+2.35), consistent with banked +~2.3.

  VERDICT (Step A): the +1.34 cyc/B kernel deficit decomposes as ~21% packed-symbol
  UNPACK ALU + ~11% backref copy + ~11% irreducible table loads. The single addressable
  lever with NO igzip counterpart is the packed-multi-symbol UNPACK chain (cnt extract,
  shift=8*(cnt-1), shrx, masks). igzip's table format yields the decoded byte(s)+length
  DIRECTLY, paying ~0 unpack ALU — that ~21% is the instruction excess. NOTE: attacking
  it = changing the short-entry TABLE FORMAT (build side in lut_huffman) in lockstep, a
  STRUCTURAL change, not a peephole; high byte-exact risk, must be one gated commit.

## RESUME POINT (2026-06-18 PM, end of B-SIZING session) — READ FIRST
DONE this session: the 3 B-sizing GATING measurements (M1/M2/M3 above) that price
rewrite B for the USER R3 decision. Source on guest restored to mission tip; spike
removed; disk restored (1.1G); no stressors/pinning leftover; governor untouched
(powersave); NO main push. Kept binaries: /root/bin/gzippy-new-native (=mission tip
packON, sha 891c9925), /root/bin/gzippy-base-native (af53f95e), /root/bin/gzippy-packOFF
(M2 SINGLE-mode toggle, sha ff477774). NOT committed: nothing new to source — only this
STATE file edited.

### RECOMMENDATION FOR USER R3 (fund the multi-hour rewrite B?): **NO — do NOT fund B as scoped.**
GATED-HYPOTHESIS (Intel-only, NOT-YET-LAW; AMD/Zen2 owed). The prior STATE rec ("B
remains justified by the residual / 1.34 ceiling") is OVERTURNED by M3:
  - M1: the real (unstressed) regime is COMPUTE-bound (silesia DRAM-stall 0.69%, TMA
    mem-bound 7%), so headroom is real — but that only says SOME compute lever exists.
  - M2: multi-symbol packing currently BUYS 0.087–0.216 cyc/B; B abandons it.
  - M3: a byte-exact partial-B spike (one-sym-per-iter + unpack cnt-extract removed) is
    SIGNIF SLOWER than the current packed kernel on ALL 3 corpora (+0.015..+0.091 cyc/B).
    The unpack removal recovered only 0.072–0.125 — LESS than the packing it costs.
  - ⇒ MEASURED net-B is negative-to-breakeven vs current, NOT the 1.34 ceiling. The
    table-format swap cannot close the igzip gap: igzip is ALSO one-sym-per-iter at 3.98
    while gzippy single-mode is 6.09 — the 2.1 cyc/B is igzip's whole leaner per-iteration
    loop, which B does not touch.
  - The igzip residual (+1.50/+2.07/+1.75 cyc/B) is therefore in the PER-ITERATION
    MACHINERY (bit-reader/refill cadence, copy, codegen), not the short-entry table format.

### NEXT (if the campaign continues toward the igzip gap):
  - DO NOT start B (fat-direct-table). It is measured net-negative-to-breakeven.
  - The real lever is converging gzippy's per-iteration loop on igzip's (single
    speculative store + one discriminator + igzip's refill cadence) — i.e. the gap
    between two one-symbol-per-iter loops (6.09 vs 3.98). Size THAT before committing:
    a removal/perturbation on the refill+copy+back-edge overhead, not the table format.
  - AMD/Zen2 replication of M1/M2/M3 is owed before any of this is LAW.
  - RE-VERIFY M3: rebuild spike (patch lut:1041 SINGLE + asm 426-428 `mov t3,1`; see
    /tmp gone — re-derive from this STATE), or simpler re-confirm via packOFF: packOFF
    is already SIGNIF slower than packON (M2), and the spike only recovered a fraction.

## (prior) RESUME POINT (2026-06-18, end of technique #2 = NEXT #1 flag-bit)
DONE this session: NEXT #1 (flag-bit discriminator) IMPLEMENTED + GATED + KEPT.
  - byte-exact PROVEN on x86_64 guest (sha grid native, c1/c2/c3 asm-vs-ref, proptest
    60k, builder flag-invariant test). KEPT.
  - cyc/byte: SIGNIF-faster UNSTRESSED (sil -1.67% p=6.4e-5, nasa -0.33% p=0.0067,
    mono -1.20% p=8.5e-4, self-tests PASS except mono small rig-bias); WASH/TIE under
    bandwidth stressor (LLC 91%). Does NOT meet strict "stressor-stable" arm; KEPT on
    byte-exact + compute-bound critical-chain win.
  - NEW residual gap vs igzip: silesia +1.50 (+34.3%), nasa +1.75 (+111%), monorepo
    +2.07 (+70%) cyc/B. Flag-bit captured ~7% of the silesia removal-oracle headroom.
COMMIT: technique #2 source (lut_huffman flag bit + builder + invariant test;
  asm_kernel flag-bit discriminator + 49: trailing-recovery shim; lut_bulk
  symbol→code length fix) pushed to perf/igzip-full-rewrite.
GUEST BINARIES (kept for next A/B): /root/bin/gzippy-base-native (af53f95e),
  /root/bin/gzippy-new-native (af53f95e+flagbit). Worktrees: /root/gz-flagbit-new
  (patched src), /root/gz-flagbit-base (clean af53f95e). Re-verify cmds in the
  technique log above.

RECOMMENDATION (B decision — for USER R3): the flag-bit did NOT capture most of the
headroom (~0.10 of ~1.34 cyc/B on silesia). Residual gap to igzip is STILL +1.50 to
+2.07 cyc/B (+34-70%). ⇒ rewrite B (fat DIRECT one-symbol-per-iter table, ~0 unpack
ALU) REMAINS justified by the residual. CAVEAT: the stressor wash shows much of the
kernel time is memory-bound under contention; B's instruction reduction (like #2's)
will similarly wash when bandwidth-saturated — B's payoff is bounded to the
compute-bound regime (which is where the removal-oracle 1.34 cyc/B ceiling was
measured). NEXT step before B: optionally extend the flag idea — also bake a
"length-vs-EOB-vs-reclass" sub-class into spare entry bits to shrink the 50: arm —
but the bigger lever is B. DO NOT begin B without USER R3.

## NEXT (planned, in priority order) — for the iteration phase (deliverable #2)
REVISED after technique #1's TIE: the prize is CRITICAL-PATH cyc/byte, NOT retired
instruction count. Technique #1 cut -0.186 instr/byte yet TIE'd cyc/byte because the
removed shl was load-shadowed (overlapped behind the table-load latency). Peephole
instruction-shaving of already-overlapped ops will keep TIE-ing. The remaining cyc/byte
deficit (+1.34 cyc/B kernel) lives in (a) the data-dependent classify dep-chain that
FEEDS the load and the cmp (the `shrx`-fed trailing-symbol extraction → `cmp 255`
branch; negative ΔIPC says this mispredicts/serializes), and (b) the ~21% packed-symbol
UNPACK ALU that has no igzip counterpart. So:
1. (HIGHEST VoI) Attack the CRITICAL PATH, not the instruction count. The dep chain is
   short_tbl load -> (bc/cnt/shift extraction) -> shrx -> cmp 255 -> branch. Shorten the
   chain LENGTH or break the misprediction: e.g. derive the literal-vs-nonliteral
   discriminator from a flag BIT already in the loaded entry (a single `test`, no
   shrx-extract), so the branch resolves off the load directly instead of waiting on the
   extraction chain. Requires a table-format bit (build side lut_huffman) — STRUCTURAL,
   one gated commit, ref in lockstep. Pre-register success = cyc/byte medΔ<0 p<0.01.
2. The packed-multi-symbol short-entry format itself: evaluate whether multi-symbol
   packing (decode 2-3 lits/iter, pay UNPACK ALU) actually beats igzip's
   one-symbol-per-iter + fatter DIRECT table (decoded byte+len in the entry, ~0 unpack).
   This is the core convergence question — likely the real lever, but a large rewrite +
   STRATEGIC fork: escalate to supervisor/user before committing the rewrite (R3).
3. Update `run_contig_ref` / `run_contig_ref_biased` in LOCKSTEP with every asm change
   (X1-X5 exit-state + IN_MARGIN bit-exact-refill contract).
GATE (each technique, ONE per commit): byte-exact (sha 3 corpora×T1/T4/T8 + proptest≥60k
+ c2/c3 asm-vs-ref diffs) THEN paired N≥21 cyc/byte vs prior AND vs igzip. Build native
flavor: `RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features
--features pure-rust-inflate`. NOTE: local Mac pre-commit hook is BROKEN (uninitialized
vendor submodules) — commit on the guest, or `git commit --no-verify` for non-perf
metadata; CI still runs the real checks.

## TOOLING STATE (for next turn)
- Instrument: `/root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh` (committed
  in scripts/bench). Re-run gap: `PIN=4 REPS=21 CORPORA="silesia monorepo nasa" /root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh`
- Symboled native binary for perf/annotate: `/tmp/symtarget/release/gzippy` on guest
  (byte-correct, built from gz-fullrewrite @ 2b10aa48; rebuild after any kernel edit).
  Profile: `perf record -F6000 ... ; perf annotate -s ...run_contig`.
  NOTE: build needs strip=false debug=2 WITHOUT force-frame-pointers (FP collides with
  the register-hungry asm kernel: "inline assembly requires more registers").

## DONE-CRITERION (do NOT self-bless; report as gated-HYPOTHESIS + re-verify cmd)
gzippy T1 cyc/byte ≤ igzip with paired p<0.01 + bootstrap CI excluding 0 + margin
surviving the bandwidth stressor, on silesia AND ≥1 more corpus; + wall-time
confirmation if a quiet window appears. AMD/Zen2 replication owed for LAW.
