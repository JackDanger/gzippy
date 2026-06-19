# KERNEL-CONVERGENCE — igzip `loop_block` ↔ gzippy `run_contig` full convergence design

Mission: converge gzippy-native's ENTIRE inner-Huffman decode kernel (`run_contig` in
`src/decompress/parallel/asm_kernel.rs`) on igzip's COMPLETE integrated hand-asm loop
(`vendor/isa-l/igzip/igzip_decode_block_stateless.asm`, `loop_block` 507-627 + the
`decode_next_lit_len` / `decode_next_dist` macros) until gzippy-native T1 cyc/byte ≤ igzip
on silesia AND nasa, byte-exact. This doc is the DURABLE SPINE: it maps every igzip
loop_block element to the gz kernel state and the convergence change, so the multi-turn
effort survives turn boundaries. It is honest about what is ALREADY ported (most of the
shape) and what genuinely diverges (the remaining diffuse gap).

Status legend: ✅ already faithfully ported · 🔶 partial / divergent · ❌ not present.

---

## 0. STARTING STATE (gated, Intel LXC i7-13700T, NOT-YET-LAW; AMD owed)
From `plans/BEAT-IGZIP-T1-STATE.md` scoreboard (binary gzippy-chunkt1 == commit ca70e9d1,
paired N≥11 vs igzip, /dev/null, cpu4, GATE-0 self-validated):
- silesia: igzip 4.397 cyc/B, gzippy 5.640 → **+1.2415 (+28.1%)**
- nasa:    igzip 1.587 cyc/B, gzippy 2.067 → **+0.4806 (+30.3%)**

KERNEL-only gap (perf-decomposed, prior LOCALIZE session): silesia +0.908, nasa +0.453 —
DIFFUSE across bit-mgmt + classify + loop-overhead; backref-copy already at parity-or-better.
WHOLE-process gap is scaffold-dominated on nasa (75% = first-touch page faults + memmove);
that scaffold is the SEPARATE consumer/output-streaming front (out of this kernel's scope,
tracked in STATE). THIS doc targets the kernel cyc/B gap.

Reference numbers to BEAT (from STATE): igzip silesia 3.98 cyc/B, nasa 1.59 cyc/B (kernel).

---

## 1. igzip `loop_block` — the COMPLETE integrated body (the blueprint)
Single straight-line basic block; literal and length/dist share the ENTIRE prefix and
diverge only at the FINAL discriminator (line 555-556). The per-iteration chain (igzip
asm line → action):

```
508-512  guards: cmp next_in,end_in jg out ; cmp next_out,end_out jg out
515      decode_next_lit_len: consume CURRENT preloaded next_sym (SHRX read_in; sub len)
518-519  SPECULATIVE STORE: mov [next_out],next_sym ; add next_out,next_sym_num   (up to 3 packed lits)
520-521  extract trailing: lea next_sym2,[8*next_sym_num-8] ; SHRX next_sym2,next_sym,next_sym2
524-525  preload INDEX for next litlen: mov tmp3,mask ; and tmp3,read_in
528-530  REFILL part A (early OR): mov tmp1,[next_in] ; SHLX tmp1,tmp1,read_in_length ; or read_in,tmp1
533      speculative length: lea repeat_length,[next_sym2-254]
536-537  EOB test: cmp next_sym2,256 ; je end_symbol_pre
540      preload NEXT litlen entry: mov next_sym,[lit_huff_code + LARGE*tmp3]
543-547  REFILL part B (ptr/len): mov tmp1,64 ; sub tmp1,read_in_length ; shr tmp1,3 ; add next_in,tmp1 ; lea read_in_length,[+8*tmp1]
550-552  preload NEXT dist entry (SPECULATIVE, EVERY iter): mov next_bits2,mask ; and next_bits2,read_in ; movzx next_sym3,[dist_huff_code + SMALL*next_bits2]
555-556  DISCRIMINATOR (at the END): cmp next_sym2,256 ; jl loop_block     (literal/length<256 → loop)
558-627  decode_len_dist: dist decode (preloaded next_sym3) → look_back → MOVDQU large/small_byte_copy → loop_block
```

Macro `decode_next_lit_len` (322-372): reads `next_sym` (preloaded short entry), gets
`rcx = code_len` (LARGE_SHORT_CODE_LEN_OFFSET=28), `next_sym_num` = sym_count (1/2/3),
tests LARGE_FLAG; if long-code, resolves via long table; `%%end:` does `SHRX read_in,rcx ; sub len,rcx`.

KEY igzip property: the discriminator is LATE and igzip does the trailing-extract (520-521),
BOTH preloads (litlen 540 + dist 550-552), the refill (split 528-530 / 543-547), and the
length precompute (533) ALL speculatively EVERY iteration — even on a pure literal where the
dist preload + length precompute are thrown away. This maximizes the independent-op window
(ILP) feeding the OOO engine; the load-use latencies of the two preloads are hidden behind
the store/advance/refill-arithmetic. The literal path pays extra retired instructions to BUY
IPC. (igzip whole-proc IPC > gzippy; gzippy's measured ΔIPC is NEGATIVE → latency-bound.)

Register layout (igzip, fixed; lines 86-136): state=r11, read_in=r9, read_in_length=r10,
next_out=r12(saved), end_out=r13(saved), next_sym=r14(saved), rfc_lookup=r15(saved),
next_in=rbx(saved), end_in=rbp(saved), repeat_length=r8; scratch tmp1=rdi tmp2=rsi tmp3=rax
tmp4=rdx, rcx = shift count. next_sym lives in a CALLEE-SAVED reg across the whole loop.

---

## 2. gzippy `run_contig` CURRENT state vs each igzip element

| # | igzip element (asm line) | gzippy state | run_contig label | Convergence action |
|---|--------------------------|--------------|------------------|--------------------|
| A | guards (508-512) | ✅ identical (`dst<out_lim`,`pos<in_lim`) | `2:` | none |
| B | decode current preloaded sym (515) | ✅ entry preloaded in `t1`; bc/cnt extract | `2:`..424-428 | none |
| C | speculative store + advance (518-519) | ✅ 8-byte store `entry&0xFFFFFF`, add cnt | 475-476 | none (timing differs, see G) |
| D | extract trailing sym (520-521) | 🔶 AVOIDED on literal path via flag-bit `test 0x1000000;jnz 49f`; recovered only in cold `49:` | 441-442 / 526-531 | **DECISION POINT (§3.1)** |
| E | preload index for next litlen (524-525) | ✅ folded into the preload load (`and 0xFFF`) | 471-473 | none |
| F | REFILL split A/B (528-530, 543-547) | ✅ DONE (NIGHT19 @c4ac5acc): `6:` reordered to igzip's exact 524-552 — index PRE-OR into {t4}, OR, litlen-entry load, DEFERRED ptr/len advance, dist preload. GATED win vs pre-reorder (silesia −0.25, nasa −0.04, monorepo −0.05 cyc/B; all IPC, 0 instr change) | `6:` 537-552 | **DONE §3.2** |
| G | speculative length precompute (533) | ❌ not on literal path (computed in backref arm `31:`) | 571 | tied to §3.1 |
| H | EOB test (536-537) | 🔶 in the cold non-literal arm `50:` (`cmp 256;je 82f`) | 539-540 | tied to §3.1 |
| I | preload NEXT litlen entry (540) | ✅ issued early, store hides its latency | 471-473 (lit), 489-491 (long) | none |
| J | preload NEXT dist entry EVERY iter (550-552) | 🔶 only INSIDE the backref arm post-consume (`dpre`) | 560-562 | **§3.3 spec dist preload** |
| K | discriminator (late, 555-556) | 🔶 EARLY (flag-bit `jnz 49f` BEFORE store) | 441-442 | **§3.1 (core question)** |
| L | dist decode (558-578) | ✅ libdeflate DistTable in-reg decode (bzhi/shrx/base) | `58:` | none |
| M | MOVDQU large/small_byte_copy (603-627) | ✅ faithful 16-byte port, COPY_SIZE=16 | `71:`-`74:` | none (at parity — STATE) |
| N | register layout (fixed callee-saved) | 🔶 LLVM-allocated via `asm!` operands | — | **§3.4 register pinning** |

Already ported: A,B,C,E,I,L,M (the bulk). Genuinely divergent / convergence targets:
D+G+H+K (the discriminator placement & speculation depth — ONE coupled question), F (refill
cadence), J (per-iter dist preload), N (register layout).

---

## 3. THE CONVERGENCE CHANGES (the integrated rewrite — all coupled)

### 3.1 Discriminator placement + speculation depth — THE CORE QUESTION (D,G,H,K)
gzippy's prior session ADDED the early flag-bit discriminator (`jnz 49f`) which captured
~0.10 cyc/B silesia by removing the trailing-extract + `cmp 255` from the literal critical
chain. igzip does the OPPOSITE: late discriminator + full per-iter speculation (extract,
length precompute, dist preload) to maximize ILP. The measured gzippy ΔIPC is NEGATIVE
(latency-bound), which is the SIGNATURE that more independent speculative work (igzip's shape)
could fill the OOO window and win — but it is NOT proven; it could also lose (the flag-bit
win shows leaner can pay too).

HYPOTHESIS (unvalidated, to be settled by gated cyc/B): converging to igzip's late-
discriminator full-speculation straight-line body lifts IPC enough to beat the early-flag-bit
form. The rewrite implements igzip's exact shape:
- decode current sym → speculative store+advance → trailing extract → BOTH preloads (litlen
  + dist) issued → refill (split) → length precompute → EOB test → late discriminator.
- Literal path falls through to `loop_block`; length path falls to `decode_len_dist`.

Falsifier / fork: build BOTH forms (the rewrite vs current flag-bit) as distinct byte-exact
binaries and run the paired harness. KEEP whichever wins per Gate-1. If the straight-line
form loses on the literal-heavy corpus (silesia) but the diffuse gap persists, the lever is
elsewhere (§3.2/§3.4) — do not over-fit to one shape. This is the first thing to MEASURE in
the implementation turn (cheaper than the full register rewrite, isolates the shape question).

### 3.2 Refill cadence — split OR early / ptr-len update late (F)
igzip splits the refill: the `or read_in,tmp1` (528-530) executes early; the next_in/len
pointer-arithmetic (543-547) is DEFERRED past the litlen preload, EOB test. gzippy's `6:`
does the whole refill contiguously, creating a tight `pos`-carried chain (load[in_ptr+pos]
→shlx→or→63-sub-shr3-add→pos; pos feeds the NEXT load address). Converge by interleaving:
issue the memory load + shlx + or, then do the spec store / preload, then the
63-sub-shr3-add pos/len update — so the loop-carried `pos` update overlaps independent work.
Byte-exact: refill is append-only (adds high bits; low bits used by preload unaffected by
order — igzip relies on the same property at 524-525 vs 528-530). The `or bitsleft,56`
unconditional form is already in place (no `<48` branch). Re-derive the IN_MARGIN proof for
the new instruction order (still ≤1 refill/iter, ≤7 pos advance, 8-byte read).

### 3.3 Speculative per-iteration dist preload (J)
igzip preloads the dist short entry EVERY iteration (550-552) before knowing if it is a
length. gzippy only preloads inside the backref arm (`dpre`). Converge by issuing the dist
short-entry gather speculatively in the main body (post-refill, from the low 9 bits of the
refilled bitbuf) so the backref arm finds it already in flight. On literal-heavy data this is
a WASTED load per iter (cost), bought back only if the dist load-use latency on the length
path dominates. Couple to §3.1: only meaningful with the late discriminator (otherwise the
literal path branches away before the preload pays). MEASURE the net (silesia literal-heavy =
cost, nasa backref-heavy = benefit) — likely corpus-split; pre-register both.

### 3.4 Register layout / pressure (N)
igzip pins next_sym, next_out, end_out, next_in, end_in, rfc_lookup in callee-saved regs;
the loop body never spills. gzippy's `asm!` exposes bitbuf/bitsleft/pos/dst as inout and the
rest as `out(reg) _`, letting LLVM allocate — under the wider straight-line body (§3.1) the
operand count may exceed the GP register file (the STATE noted "inline assembly requires more
registers" with frame pointers). Convergence: keep cold invariants in `KernCtx` memory
operands (already done — `[ctx+N]`), minimize live scratch in the hot body, and if LLVM still
spills the hot chain, pin the loop-carried values (bitbuf, bitsleft, pos, dst, the two
preloaded entries) to specific registers via `in("rXX")`/`lateout` to match igzip's no-spill
body. Verify with `perf annotate` that the hot loop has zero stack traffic (igzip parity).

### 3.5 litlen-LUT build (`LutLitLenCode::rebuild_from`, ~3% silesia self-time)
In scope per mission. After the loop convergence is byte-exact + measured, evaluate an
igzip/libdeflate-style faster table construction (igzip builds the LARGE table with the
sym_count packing in `igzip_inflate.c`). Lower priority than the loop (the loop is the
dominant gap); a separate byte-exact commit. Bounded ~3% sil / ~1.7% nasa (STATE TASK 2).

---

## 4. BYTE-EXACT CONTRACT (SACRED — re-derive in lockstep)
The asm region and `run_contig_ref` / `run_contig_ref_biased` (asm_kernel.rs 846-1005) are
bound by the X1-X6 exit-state contract (module doc 46-101). EVERY asm change must update the
ref model in LOCKSTEP. The differential (`c2_differential_asm_vs_ref_random_streams`,
asm_kernel.rs ~1088) pins asm==ref on (exit code, full bit cursor pos/bitbuf/bitsleft, dst
advance, every output byte < dst). Contracts to preserve through the rewrite:
- X1: `(bitbuf,bitsleft,pos)` + dst at every exit == what the Rust loop holds — REFILL
  PLACEMENT must stay bit-identical (the ref must mirror §3.2's new refill order exactly).
- X2: cursor sits BEFORE a fresh un-consumed packet; rare/exceptional packets handed to Rust
  UN-consumed (spill/restore bitbuf/bitsleft/dst before any RECLASS bail).
- X3: bytes in [entry dst, exit dst) final; bytes ≥ dst may be speculative-store overshoot.
- X5: `bitsleft >= 48` at exit OR cursor unchanged (caller re-derives via `decode_prefilled`).
- IN_MARGIN=40 proof (no slow refill in asm): re-verify for the new instruction order.

Gauntlet (every byte-exact commit, on the guest — Mac is aarch64, cannot run the asm):
tri-oracle gzip+flate2+libdeflate+igzip, both flavors (native pure-rust-inflate + gzippy-isal)
× silesia/nasa/monorepo/squishy/large × T1/T4/T8 sha-identical; ≥60k prop_structured_roundtrip;
c2/c3 asm-vs-ref differential; serialized `cargo test --release --lib` (both flavors) 0-failed.
Commit ONLY byte-exact states.

---

## 5. MEASUREMENT GATES (Gate-0..5; the only currency)
- Instrument: `/root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh` (committed in
  scripts/bench). Self-validates Gate-0: KERN entries>0, distinct binaries, igzip arm
  non-inert sha==zcat, same /dev/null sink + pin, GHz spread, A2-A1 self-test ~0.
- Run: `GZIPPY=<bin> PIN=4 REPS=21 CORPORA="silesia nasa" SKIP_STRESS=1 GZIPPY_FORCE_PARALLEL_SM=1 bash /root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh`
- SUCCESS = gated cyc/B ≤ igzip on silesia AND nasa (paired N≥9, Wilcoxon p<0.01 + bootstrap
  CI excluding 0). prize ≡ measured Δ.
- T>1 no-regression: run_contig is the SHARED kernel → measure T4/T8 wall + RSS; rapidgzip
  parity must hold (should improve).
- Build (guest, /dev/shm target — root disk 99% full): `RUSTFLAGS="-C target-cpu=native"
  CARGO_TARGET_DIR=/dev/shm/<t> cargo build --release --no-default-features --features pure-rust-inflate`.
  Symboled profiling build: add `CARGO_PROFILE_RELEASE_STRIP=false CARGO_PROFILE_RELEASE_DEBUG=2`
  (NO force-frame-pointers — collides with the register-hungry asm).
- Single-arch Intel → all results gated-HYPOTHESIS NOT-YET-LAW; AMD/Zen2 (solvency) owed for LAW.

---

## 6. IMPLEMENTATION ORDER (cheapest-isolating first; each its own byte-exact gate)
1. **§3.1 shape A/B** ✅ DONE — the kernel is the late-discriminator full-speculation shape
   (bb2cff5d: Stage-A subtable-inline + mask-once + full per-iter speculation).
2. **§3.2 split-refill cadence** ✅ DONE (NIGHT19 @c4ac5acc) — gated win vs pre-reorder on all
   3 corpora (silesia −0.25 cyc/B etc.), all via IPC (0 instr change). silesia now at IPC
   parity with igzip (ΔIPC −0.017). The residual gap to igzip is now INSTRUCTION-bound
   (Δinstr/B +3.03) → NEXT FRONT is the register-discipline/table-format instruction reduction
   (steps 4+5 below, now PRIORITY).
3. **§3.3 per-iter dist preload** — coupled to §3.1; corpus-split expected.
4. **§3.4 register pinning** — last; verify zero hot-loop stack traffic via perf annotate.
5. **§3.5 litlen-LUT build** — separate, after the loop is at/under igzip.
Maintain `plans/BEAT-IGZIP-T1-STATE.md` after each step (what implemented, byte-exact status,
current cyc/B vs igzip). Commit only byte-exact states to perf/igzip-full-rewrite.

---

## 7. RESUME POINTERS
- Kernel asm: `src/decompress/parallel/asm_kernel.rs` `run_contig` (385-791) + ref model
  `run_contig_ref_biased` (897-1005). LUT build: `src/decompress/parallel/lut_huffman.rs`
  `LutLitLenCode::rebuild_from`. Bit reader: `src/decompress/parallel/bit_reader.rs`.
- igzip blueprint: `vendor/isa-l/igzip/igzip_decode_block_stateless.asm` loop_block 507-627,
  macros decode_next_lit_len 322-372 / decode_next_dist 396-440 / *_with_load.
- Guest (ONLY box that can build/run the asm): `ssh -o ConnectTimeout=15 -J REDACTED_IP root@REDACTED_IP`.
  Product binary: `/root/bin/gzippy-chunkt1` (== ca70e9d1). Git checkout `/root/gzippy`
  (fetch+checkout the branch there; build to `/dev/shm/<t>` — root disk 99% full).
  igzip `/usr/bin/igzip`. macOS x86_64 CANNOT build the asm (register-alloc fails —
  pre-existing, night9 too): use `cargo check --target x86_64-apple-darwin` for ref-model
  typeck only; the guest (linux x86_64) is the sole asm build/run gate.

---

## 8. DIVERGENCE LEDGER (gzippy state igzip lacks — kept MINIMAL & faithful)

Each row is a place gzippy's `run_contig` must keep state that igzip's stateless
`loop_block` does NOT, why igzip lacks it, and why the gz form is the minimal faithful
choice. (Mission rule: a true divergence is legal when byte-exact + ledgered.)

### RESOLVED-1 — bit-24 TRAILING_NONLIT_FLAG REMOVED (NIGHT20 @85e98c9f, CONVERGED)
- **Was.** gz set a build-time `LARGE_TRAILING_NONLIT_FLAG` (bit 24) in each
  litlen short entry whose trailing packed symbol was non-literal (cnt∈{1,2}).
  A leftover from the EARLY-flag-bit discriminator shape (`test entry,bit24;jnz`).
- **igzip has no counterpart.** `make_inflate_huff_code_lit_len` (igzip_inflate.c
  387-599) sets no class flag; the decoder classifies at RUNTIME (loop_block
  555-556 `cmp next_sym2,256`). The flag was DEAD in gz too (the kernel uses the
  late `cmp {t5},256` at run_contig:565).
- **Removed** to match igzip's build exactly (deleted the 2 per-entry conditional
  ORs + const + offset + doc). Byte-exact (bit 24 only ever read inside the 25-bit
  packed field; decoders drop it via the cnt-shift + `& 0xFFFF`; triple sym3≥256
  sets it as natural DATA). GATED: trioracle 5-corpus × both flavors × T1/4/8
  sha-identical + c2/c3 differential. NOT a divergence any longer — a convergence.

### D-1 — X2 un-consume cursor (NIGHT11, the snapshot replacement)
- **What diverges.** On a rare RECLASS bail (EOB / oversize / invalid / dist-subtable /
  invalid-dist) gzippy must hand the un-decoded packet BACK to the Rust careful loop
  UN-CONSUMED (it re-decodes it). The late-discriminator body has already consumed +
  refilled + speculatively stored by the time a bail is known.
- **Why igzip lacks it.** igzip's decoder is STATELESS: EOB ends the block and errors
  abort; it never re-decodes a consumed packet, so its handlers (`end_loop_block`
  640-642) merely PERSIST the current (post-consume) `read_in/read_in_length/next_out`
  — they never un-consume. There is no igzip counterpart to roll back.
- **Why the consumed state is unrecoverable from registers.** `consume` is
  `bitbuf >>= bc` — the low `bc` bits are SHIFTED OUT and lost; `refill` then ORs new
  high bits and advances `pos`. So the pre-consume `bitbuf` cannot be rebuilt from the
  post-consume registers. A SAVE (per-iter snapshot) or a FROM-DATA RE-READ is
  structurally required. This is the irreducible divergence.
- **night9's form (rejected by night10's gate).** A per-iteration 4-store snapshot of
  `(bitbuf, bitsleft, pos, dst)` → ctx (`save_bitbuf/+48 save_bitsleft/+56 save_dst/+72
  save_pos/+80`). Gated-slower than the old early-flag-bit shape (+0.194 cyc/B silesia);
  4 stores/iter on the hot literal path.
- **c936's form (DEAD — do not reuse).** Moved EOB/oversize/invalid PRE-consume to delete
  the snapshot → serialized the classify into the critical chain → IPC collapsed
  (ΔIPC −0.04→−0.30), net slower than night9.
- **NIGHT11 minimal faithful form.** Keep the late discriminator + consume-forward; on a
  bail RECONSTRUCT via a single FROM-DATA RE-READ at the iteration-top BIT POSITION
  `p0 = pos*8 - bitsleft`. `p0` is REFILL-INVARIANT (proved vs `Bits::refill`: the `|56`
  + pos-advance preserve `pos*8 - bitsleft`), so it equals the un-consumed packet start
  regardless of the body's consume/refill. Re-read: `byte=p0>>3; skip=p0&7;
  bitbuf=load_u64(in_ptr+byte)>>skip; bitsleft=64-skip (∈[57,64]⇒X5); pos=byte+8`.
  In-range by IN_MARGIN. The re-read yields a DIFFERENT but EQUIVALENT cursor
  REPRESENTATION (the caller only re-runs `decode_prefilled(&lb)`, never inspecting the
  cursor shape; the ref model `reclass_reread` uses the identical re-read so the c2/c3
  differential pins asm==ref field-for-field).
- **Why still TWO ctx words (`save_p0/+56`, `save_d0/+64`), not zero.** The asm is at the
  15-GP-operand register ceiling (night9: "register allocation FIT … 15 GP operands");
  carrying `p0`+`d0` in registers overflows it (the macOS x86_64 build already fails
  register-alloc on the unmodified night9). Freeing a register would require dropping an
  igzip speculation feature (the per-iter dist preload `dpre`, igzip 550-552) — a
  divergence AWAY from igzip. So the anchor is a MINIMAL 2-word ctx save: one bit-cursor
  store + one dst store per iteration (50% fewer than night9's four), uniform across all
  bails. (A 1-store form — store `p0` only, compute `d0` from the live `cnt` in the rare
  arms — is a possible follow-up if the gate shows store-port is the binding constraint.)
- **NIGHT13 re-scope of the STEP-2(a) "free a register" idea (single-base table co-location).**
  An ALTERNATE way to free regs (vs dropping `dpre`): inline the litlen+dist tables into ctx
  and address them `[ctx+disp+idx*scale]` (igzip's one-base `[state+_lit_huff_code+…]`), freeing
  `short_tbl`+`dtbl` (2 regs). BUT even with both freed, carrying `p0`+`d0` in-register deletes
  ONLY the 2 STORES → the anchor becomes `lea p0,[pos*8]`+`sub p0,bitsleft`+`mov d0,dst` = 3 instr
  (net −1 instruction, −2 store-port ops). The `lea`+`sub` p0-COMPUTE is irreducible (it IS this
  D-1 divergence — the un-consumed packet start must be (re)computed each iteration). So 2a
  removes ~1 instr, NOT the "+4" the NIGHT12 bucket table implied. 2a is therefore a STORE-PORT
  relief lever (worth it ONLY if the gate shows store-port binding), not the path to instr/B ≤
  igzip. The path below igzip on instr/B does not exist while D-1 stands (residual +2/iter on the
  literal path after 2a+2b vs igzip's stateless 0).
- **STEP-2(b) is CONVERGENCE, not a divergence (no D-row).** mask-once-in-register (igzip macro
  341) DELETED gz-specific scratch (two `mov rN,t1` copies + a per-use store mask), moving gz
  TOWARD igzip's single-masked-reg reuse. It needs NO freed register (reuses `{t1}`), so it is
  fully decoupled from 2a — falsifying NIGHT12's "2a+2b are one coupled change" premise.
- **Gate.** Byte-exact via c1/c2/c3 asm-vs-ref differentials + prop roundtrip + tri-oracle
  sha grid; perf via the paired harness vs igzip / night9(kc) / old(chunkt1). GUEST-OWED
  (status tracked in BEAT-IGZIP-T1-STATE.md).

### D-cadence — cold long-literal bottom `64:` kept CONTIGUOUS (NIGHT19, minor)
- **What diverges.** NIGHT19 converged the HOT literal bottom (`6:`) to igzip's split-refill
  schedule (524-552), but left the COLD long-literal bottom `64:` (reached only via the rare
  long-Huffman-code path at `20:`) as a contiguous refill+preload. So the cold path's cadence
  still diverges from igzip's.
- **Why minimal/faithful.** `64:` is off the hot critical path (long codes are rare — silesia
  long-litlen ≈ a few % of symbols); converging it would not move the wall and only adds risk.
  Byte-exact either way (same refill, different schedule). Re-open trigger: if a later
  decomposition shows the long-code path is a measurable cyc/B share, apply the same 524-552
  reorder to `64:`.
