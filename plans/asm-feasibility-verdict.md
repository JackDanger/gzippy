# FULL-KERNEL inline-asm clean-tail port — FEASIBILITY VERDICT

Read-only design + verdict for the supervisor's delegated GO/NO-GO gate. Branch
`reimplement-isa-l`, HEAD `d56cb0f5`. NO production code touched, NO box access, NO build run.
Every vendor `file:line` and every gzippy `file:line` below was opened and read first-hand this
turn (the campaign has mis-cited before — see STATE §7). Where I rely on a banked number rather
than a fresh measurement, it is flagged GATED.

**Scope under decision:** a SINGLE `core::arch::asm!` hot loop owning the WHOLE clean-tail decode
— bit-buffer, in/out pointers, lit/len table base pinned in registers ACROSS the back-edge,
exiting to Rust ONLY at block-boundary / long-code / EOB / output-margin-exhausted (NOT per
symbol, NOT per back-ref). This is the construct the prior NO-GO (DIS-1 / VAR_VII) never built.

---

## 0. BOTTOM LINE

**VERDICT: CONDITIONAL-GO.**

- **GO** to build the isolation prototype (`VAR_VIII` in `benches/engine_isolation.rs`) — it is
  OWED (phase2 advisor), cheap (bench-only, reuses VAR_VII's asm scaffold + VAR_VI's
  `avx_backref_copy` + the already-ported ISA-L LUT), and it is the ONLY way to retire the open
  empirical question "can a register-pinned full kernel beat LLVM's best (VAR_VI)?" **DIS-1 does
  NOT bind it** — DIS-1 measured a per-back-ref asm↔Rust re-entry that spilled 4 regs to `bits`
  ×300–460K/chunk AND did the dist-decode + copy in Rust; the full kernel removes BOTH costs by
  construction.
- **CONDITIONAL** on a pre-registered isolation gate: production integration (Stage 1+) is
  authorized ONLY if `VAR_VIII/VAR_III(isal) ≥ 0.85` AND `VAR_VIII` materially exceeds
  `VAR_VI/VAR_III` (the LLVM ceiling, currently ~0.55–0.60×). If `VAR_VIII ≈ VAR_VI`, the PLATEAU
  is EARNED, the DIS-1 NO-GO is correctly re-triggered, and the route dies at the bench — no
  multi-session production cost incurred.
- **HARD CAVEAT the supervisor MUST carry to the user (necessary-NOT-sufficient):** even a PERFECT
  kernel = ISA-L's own instruction rate brings gzippy-native UP to gzippy-ISAL parity — and
  gzippy-ISAL itself passes BAR-1 (≥0.99× at EVERY T) **only at T8** (0.990×, soft 9% spread),
  and LOSES T1 (0.899×) / T4 (0.900×). The T1/T4 deficit is largely **NON-engine** (placement
  co-primary, LEV-2/LEV-5, still OPEN). So the asm kernel is exactly on-target for the user's
  *engine-parity / "steal ISA-L's techniques in pure Rust, no C-FFI"* sub-goal, but it will **NOT
  clear BAR-1 by itself.** Authorize it for the engine co-primary; do not let it be sold as the
  BAR-1 close. Fund placement in the same plan (re-confirms the prior asm-kernel report's
  recommendation).

**What changed since the 2026-06-07 report (`asm-kernel-feasibility-report.md`) — materially
STRENGTHENS feasibility:** that report's whole fork (Arm A flat-u8 two-phase handoff vs Arm B u16
ring) is now **OBSOLETE**. Two of its "hard 80% integration" items have since LANDED in production:
1. The clean tail is **already a flat-u8 LINEAR buffer** — `decode_clean_into_contig`
   (`marker_inflate.rs:2071`), wired at `gzip_chunk.rs:1303` with `n_max_to_decode = usize::MAX`.
   Window is the DICTIONARY PREFIX at `base[0..window_len)`, output at `base[window_len..]`,
   back-ref source = `base[*pos − distance]` directly. **NO u16 ring, NO `% RING_SIZE`, NO
   wrap-straddle** (the function's own comment, `:2144-2147`). This is igzip's exact host-buffer
   model — the precondition the prior report said was missing is MET, in production, inside ONE
   engine (the post-flip `Block`). The Arm A/Arm B faithfulness tension DISSOLVED: the contig path
   is simultaneously flat-u8-linear (Arm A's speed precondition) AND one-engine (Arm B / the
   governing-memory invariant).
2. The **ISA-L packed LUT is already ported** — `lut_huffman.rs`: `short_code_lookup: [u32; 1<<12]`
   (`ISAL_DECODE_LONG_BITS=12`), `long_code_lookup: [u16; 1264]`, up-to-3-symbol packing, exact
   igzip field layout (`:61-65`, `:242-244`), and `LutLitLenCode::decode` (`:1030-1072`) is a
   line-faithful transliteration of `decode_next_lit_len`. The prior report's "new tables, new
   build, new tests = the hard 80%" is DONE and byte-exact-gated in production.

So the remaining work is genuinely **just the kernel** — the part the prior report called "the easy
~20%" — plus the byte-exact x86_64 specialization seam. The integration mountain it feared is
already climbed.

---

## 1. SOURCE-MAP — igzip ISA-L kernel ↔ gzippy production clean loop (instruction-for-instruction)

igzip vendor: `vendor/isa-l/igzip/igzip_decode_block_stateless.asm` (`asm`), `igzip_inflate.c` (`C`).
gzippy production clean tail: `decode_clean_into_contig` (`marker_inflate.rs`, `mi`), its fast loop
`mi:2166-2312`, careful tail `mi:2314-2400`, back-ref `emit_backref_contig` (`mi:3546`),
symbol decode `LutLitLenCode::decode` (`lut_huffman.rs`, `lut:1030`), bit engine
`Bits` (`consume_first_decode.rs`, `bits:197-313`).

| igzip technique (vendor file:line) | gzippy production today (file:line) | Status — does LLVM emit igzip's instructions? |
|---|---|---|
| **Packed flat short table**, one `u32` load retires ≤3 literals + count + len (`asm:57-68,322-372`; build `C:448-487,594-596`) | `LutLitLenCode.short_code_lookup:[u32;1<<12]`, `decode` pulls `sym`/`sym_count`/`bit_count` from one load (`lut:1038-1050`) | **PORTED, parity.** Same table, same single-load decode. ✓ |
| **F1 one-iteration-ahead lit/len gather** across the back-edge (`asm:524-525,540` preload feeds next iter) | `pre = self.lut_litlen.decode(bits)` preloaded before the loop (`mi:2169`) and at the bottom (`mi:2305`); every break leaves a fresh un-consumed `pre` (`mi:2308-2311`) | **PRESENT in Rust.** LLVM keeps `pre` live; faithful pipeline. ~parity. |
| **F2 speculative DIST gather BEFORE the lit-vs-len branch** (`asm:550-552` `next_sym3`) | **ABSENT.** Dist is decoded only AFTER the trailing-code test resolves to a length, via `self.dist_hc.decode(bits)` (`mi:2260`) — a separate method call, data-dependent, not speculative | **GAP.** LLVM cannot hoist a dist gather above the type branch without the asm's manual schedule. |
| **F3 unconditional branchless refill** via `SHLX`/`or`, IN_BUFFER_SLOP (`asm:528-530,543-547`) | `bits.refill()` (`bits:245-280`) is branchless-ish but the LUT decode gates it `if available()<32 { refill() }` (`lut:1034-1036`); dist-extra path also re-checks+refills (`mi:2270-2275`) | **PARTIAL.** Production has a per-decode availability branch igzip elides; refill is not unconditional. |
| **F4 loop state pinned in callee-saved GPRs** (`asm` uses a stack frame + pinned hot regs) | State lives in the `Bits` struct fields (`bitbuf`/`bitsleft`/`pos`) + `self.decoded_bytes` + `out_pos`/`pre` locals | **THE CRUX GAP.** LLVM register-allocates within the loop, but `self.dist_hc.decode` and `self.record_backreference_for_sparsity` (`mi:2297`) are call/optimization barriers that can force the bit-state + pointers to memory. This is the "LLVM blind spot" the advisor flagged as intrinsic. |
| **C speculative 8-byte packed-literal store**, advance by ACTUAL count (`asm:518-519`) | `(base.add(*pos) as *mut u64).write_unaligned(packed)` then advance by leading-literal count (`mi:2220-2245`); lone-literal fast-paths a 1-byte write (`mi:2206-2214`) | **PORTED, parity.** ✓ (advisor-confirmed lean single-byte for sym_count==1). |
| **D MOVDQU overlap-doubling back-ref copy** (xmm, NOT ymm): non-overlap `MOVDQU` 16B strides (`asm:603-612`); overlap doubles `look_back_dist` (`asm:614-627`); back-edge `jle/jmp loop_block` (`asm:602,627`) | `emit_backref_contig` (`mi:3546`): **SCALAR** 8-byte word copy (`:3560-3572`), RLE `slice::fill` for dist==1 (`:3580-3584`), per-byte for general overlap (`:3585-3590`). No SIMD; returns to Rust loop, not an asm back-edge | **GAP (two-fold).** (a) scalar 8B not MOVDQU 16B; (b) the copy is a Rust call, so the back-edge crosses an asm/LLVM boundary. VAR_VI's `avx_backref_copy` closes (a) at the LLVM level; only the kernel closes (b). |
| **F4-slop fastloop, no per-symbol bounds checks** within `end_out−(16+258)` / `end_in−8` (`asm:48,488-489,509-512`); careful `rep movsb` tail at the boundary (`asm:637-703`) | `FAST_OUT_SLOP=8`/`FAST_IN_SLOP=8` gate (`mi:2164-2178`), fast loop unchecked within slop, bails to the bounds-checked careful loop `mi:2314-2400` at the boundary | **PORTED, parity.** ✓ Same slop-margin structure. |
| **EOB detect** `cmp next_sym,256` (`asm:536-537`) | `code == END_OF_BLOCK_SYMBOL` on the trailing code (`mi:2251`) | **PORTED.** ✓ |

**Net source-map finding:** gzippy production already faithfully reproduces tricks #1, #C(store),
#F1(litlen preload), #F4(slop) — that is exactly why VAR_V/VAR_VI reach ~0.55–0.60× ISA-L (not the
old 0.41×). The THREE residual divergences are precisely igzip's hand-asm-only moves that LLVM does
not emit: **F2** (speculative cross-branch dist gather), **F3** (unconditional flag-free refill), and
**D + F4-register-pinning** (SIMD overlap copy whose back-edge stays inside the register-pinned
loop). These three are the entire gap between LLVM's best and ISA-L — and they are co-dependent: F2/D
only pay if the bit-state and pointers never spill across the back-edge (F4). That co-dependence is
why a per-symbol or per-back-ref asm (VAR_VII) cannot capture them and a single back-edge-in-asm
kernel (VAR_VIII) is the only construct that can. **This is the load-bearing GO rationale.**

**One wrinkle the integration must reconcile (flagged, not fatal):** production's dist decode uses
`self.dist_hc: DistanceShortBitsCached` (`mi:368`, a `HuffmanCodingReversedBitsCached` structure),
NOT the ISA-L `LutDistCode` small table. The isolation bench VAR_VI/VII use `LutDistCode` (the ISA-L
`u16[1<<10]` small table). For VAR_VIII's F2 dist-gather-in-asm to be BOTH bench-valid AND
production-faithful, the production clean tail's dist path should be moved onto the ISA-L small LUT
(`InflateHuffCodeSmall`, `lut:259-268`, already defined) so the asm gathers a vendor-shaped dist
table. That is a bounded, byte-exact-gated change, but it is real work and a real divergence point —
account for it in Stage 1.

---

## 2. PRIOR NO-GO (DIS-1 / VAR_VII) — why it does NOT bind the full kernel

Source-verified in `benches/engine_isolation.rs:1211-1550` and `plans/phase2-inline-asm-advisor-verdict.md`:

- VAR_VII's asm covers ONLY the literal run + refill + litlen gather (`bench:1316-1408`). On **every
  length code** it takes `ja 8f` → exit_code 0 → falls OUT to Rust, which does the dist decode +
  back-ref copy in `emit_one_backref` (`bench:1456-1468`), then `continue 'asm_reentry` (`:1473`).
- So on silesia (~31% back-refs) the asm↔Rust boundary is crossed ~once per ~3 symbols, and on each
  crossing the loop-carried bit-state is written back to memory and reloaded:
  `bits.bitbuf = read_in; bits.bitsleft = …; bits.pos = …; out_pos = …` (`bench:1411-1414`). That is
  the **4-reg spill ×300–460K/chunk** DIS-1 measured (78 MB/s, 0.276× ISA-L, rate FALLS as asm
  coverage rises). The asm carried a negligible byte share; 155 MB/s was essentially the careful
  tail's rate.
- The full kernel **removes the crossing**: F2 (dist gather) and D (MOVDQU copy) run INSIDE the asm,
  the copy's back-edge is `jmp 2b` (igzip `asm:627`), and the bit-state/pointers are never written
  back mid-block. Rust is entered ONLY at block-boundary / long-code / EOB — i.e. ~once per block,
  not ~once per 3 symbols. The phase2 advisor explicitly OWED this VAR_VIII and ruled the NO-GO
  "not earned" until it is built. This verdict honors that.

---

## 3. DESIGN — the full-kernel asm (`VAR_VIII`, then the production x86_64 specialization)

### 3.1 Register allocation (the F4 pin)
Hot, pinned for the whole block (must stay in registers across the back-edge):
- `read_in` (u64 bit buffer = `Bits::bitbuf`) — GPR
- `read_in_length` (i64, gzippy's `bitsleft` convention: low byte = real count, `|56` high marker —
  must mirror `bits:245-313`, NOT igzip's accounting; VAR_VII already proved this mirror, `bench:1323-1341`)
- `next_in_pos` (u64 input byte index), `in_ptr` (const base)
- `next_out` (u64 output index), `out_ptr` (mut base = `chunk.data` reserved tail)
- `litlen_short` (`short_code_lookup.as_ptr()`, u32 table)
- `dist_short` (the ISA-L small dist table ptr — see §1 wrinkle)
- scratch: `tmp`, `cnt`, `sym`, `dist_sym`

That is ~10 pinned + ~4 scratch ≈ 14 live — over the 13 freely-usable GPR budget once you exclude
rsp. **Faithful resolution (igzip's own):** igzip does NOT fit all in registers — it uses a stack
frame with named spill slots for the COLD values (`start_out_mem_offset`, `end_out`, the long-table
base) and pins only the hot ones. So VAR_VIII must **drop `options(nostack)`** (VAR_VII used it) and
let the asm spill `out_limit`/`in_limit`/`dist_short`/long-table-base to stack, keeping
`read_in`/`read_in_length`/`next_in_pos`/`next_out`/`litlen_short`/`out_ptr` + 2 scratch pinned. The
limits are loop-invariant and read once per top-guard, so a stack slot for them is free. **This is the
single biggest asm-authoring risk: clobber/save discipline for callee-saved GPRs (rbx, r12–r15) that
Rust inline-asm requires you to preserve.** It is bounded and well-trodden, but it is the maintenance
cost.

### 3.2 The packed-flat short table
Already built (`lut_huffman.rs`). The asm indexes `litlen_short[read_in & 0xFFF]`, tests
`LARGE_FLAG_BIT (1<<25)` for long-code → exit-to-Rust, reads `bit_count = sym>>28`, `count =
(sym>>26)&3`, packed literals = `sym & 0x1FFFFFF`. Identical to VAR_VII's gather (`bench:1342-1374`),
which is already byte-exact-gated.

### 3.3 The back-edge that stays in asm (the new part vs VAR_VII)
On a trailing length code, INSTEAD of `ja 8f`→Rust:
1. emit leading literals via the speculative 8-byte store (`asm:518` / `mi:2220`).
2. **F2 in asm:** gather `dist_short[read_in & 0x3FF]`, resolve dist-extra via `SHRX`/`BZHI`
   (BMI2), compute `distance = DISTANCE_BASE[dsym] + extra`. (`DISTANCE_BASE`/`DISTANCE_EXTRA`
   passed as table ptrs.)
3. range check `distance ∈ [1, *pos]` (clean-mode, `mi:2283-2290`); on failure `jmp` to a
   Rust-error exit.
4. **D in asm:** the MOVDQU overlap copy (`asm:591-627`) — non-overlap 16B strides, overlap
   self-doubling — writing `base[*pos..]` from `base[*pos−dist..]`, advance `next_out += length`.
5. `jmp 2b` — back to the top guard, state still pinned. NO write-back.

### 3.4 Exit-to-Rust seams (byte-exact, the only crossings)
- **long-code / invalid** (`LARGE_FLAG_BIT` set, or `bit_count==0`): exit; Rust decodes exactly that
  one symbol via the validated `LutLitLenCode::decode` long path (`lut:1052-1071`) — long codes are
  rare — then re-enters. (VAR_VII's exit_code 1 path, `bench:1482-1509`, already correct.)
- **EOB**: set `at_end_of_block`, commit, return.
- **output-margin / input-slop exhausted** (top guard `jae 9f`): hand the block tail to the EXISTING
  Rust careful loop (`mi:2314-2400`) — it owns the resumable boundary and the bounds-checked edge.
- On EVERY exit the bit cursor sits before a FRESH un-consumed symbol (VAR_VII's invariant,
  `bench:1308-1311`), so the Rust careful loop re-decodes from the same position — no desync, no
  carried state. This is the property that makes the seam byte-exact.

### 3.5 Resumable contract + flip seam (LOW risk — already handled)
Production calls `decode_clean_into_contig` once per chunk with `n_max_to_decode = usize::MAX`
(`gzip_chunk.rs:1303`) — chunk-at-a-time, so NO mid-clean-tail resumability is needed (the prior
report's Arm A finding). The window prefix is installed by the existing flip (`setInitialWindow`
port, the 32 KiB predecessor at `base[0..window_len)`); the asm reads back-refs straight from it.
The flip seam already has its adversarial byte-exact test (`contig_clean_matches_ring_clean_on_*`,
referenced `mi:2038-2040`). The asm specialization must keep passing it.

### 3.6 Portability
x86_64-only. The Rust `decode_clean_into_contig` STAYS as the portable / non-asm path (it is already
`#[cfg]`-gated for `x86_64 + isal-compression`/`pure_inflate_decode`, `mi:2063-2070`). arm64 never
takes the parallel-SM path at all (CLAUDE.md). So portability is HANDLED by carrying both — at the
cost of keeping the asm and the Rust loop byte-identical (the differential test does this).

---

## 4. ISOLATION-BENCH GATE (the pre-registered falsifier — design)

In `benches/engine_isolation.rs`, add `VAR_VIII_fullkernel` alongside the existing table
(`bench:1577-1607`). Reuse: VAR_VII's refill/gather/store asm (`bench:1316-1378`), VAR_VI's
`avx_backref_copy` (`bench:380+`) transliterated INTO the asm, the already-swept clean silesia
chunks, the interleaved best-of-N timing, and the existing self-test
(`(iii)/(i) ∈ [2.5,3.6]` guest, `bench:34-37`).

**PASS criteria (pre-registered, ALL required):**
1. **Byte-exact:** `VAR_VIII` SHA-equal to `VAR_I` (scalar) AND `VAR_III` (ISA-L FFI oracle) on
   every swept clean chunk. A win with wrong bytes is VOID (CLAUDE.md rule 4).
2. **Coverage:** a `VIII_COVERAGE` counter (mirror `VII_COVERAGE`, `bench:1539-1546`) shows
   `asm_frac ≥ ~0.97` — the asm carried the bulk (proves we are NOT re-measuring the careful tail,
   the exact instrument failure that confounded DIS-1).
3. **Rate:** `VAR_VIII / VAR_III(isal) ≥ 0.85` (the standing pre-registered bar, `bench:1593-1605`)
   AND `VAR_VIII` materially exceeds `VAR_VI / VAR_III` (currently ~0.55–0.60×, GATED from the
   phase2 verdict). The delta over VAR_VI is the whole point: it isolates exactly the F2+D+F4
   register-pinning that LLVM cannot emit.

**KILL (PLATEAU earned → NO-GO):** if `VAR_VIII ≈ VAR_VI` (within isolation spread), the
register-pinned full kernel does NOT beat LLVM's best → the clean-rate gap is intrinsic to codegen we
cannot reach in-process → the DIS-1 NO-GO is correctly re-triggered and the route dies at the bench.
This is the cheap, honest falsifier the phase2 advisor demanded ("PLATEAU only if VAR_VIII ≈ VAR_VI").

**Whole-system projection bar (Stage-1 gate, distinct from isolation):** PASS at isolation does NOT
imply a wall move. The isolation ratio (~0.55×) and the whole-system ratio (ocl_cf ≈0.945× @T8) differ
because the engine is **slack-masked at T8** (Phase-0: seeded pure-Rust ties at T8). The causally-
confirmed engine payoff is at **T4**: LEV-1 removal oracle = native 0.761× → ISA-L 0.900×, +0.139–
0.159× (sign-stable, 5× spread). So the Stage-1 whole-system gate is: **does native's T4 wall move
toward ~0.900× (the LEV-1 ceiling) on the frozen box, interleaved + sha-verified?** Not "does it tie
rg" — even ISA-L doesn't tie rg at T4.

---

## 5. BUILD PLAN (if GO past the gate)

- **Stage 0 — `VAR_VIII` isolation prototype (bench only, no production touch).** Build the
  full-kernel asm per §3.3-3.4; reuse VAR_VII scaffold + VAR_VI copy. Gate on §4 (1)+(2)+(3).
  **This is the whole decision.** If KILL → STOP, NO-GO banked, zero production cost.
- **Stage 1 — x86_64 specialization of `decode_clean_into_contig`** (only if Stage 0 PASSES). Wire
  the kernel as a `#[cfg(target_arch="x86_64")]` fast path inside the existing function, Rust loop as
  fallback. Reconcile the dist path onto the ISA-L small LUT (§1 wrinkle). Byte-exact gate: all 850+
  lib tests + silesia differential + the flip-seam adversarial test (`contig_clean_matches_ring_*`) +
  proptest/fuzz oracle. KEEP only on byte-exact.
- **Stage 2 — whole-system measurement** vs the LEV-1 T4 0.900× ceiling and the ocl_cf T8 number,
  frozen box, interleaved N≥7, sha-verified, `path=ParallelSM` asserted, `isal_chunks==0` confirmed
  (native fingerprint). KEEP on a sign-stable move toward the ceiling; revert-but-layer on a TIE
  (rule 7a — a byte-exact change is kept on a TIE).
- **Each stage byte-exact + isolation-gated before the next.** No stage advances on attribution.

---

## 6. WHAT FALSIFIES GO MID-BUILD

1. **`VAR_VIII ≈ VAR_VI`** at isolation → PLATEAU earned, register-pinning doesn't beat LLVM →
   NO-GO (the primary kill).
2. **`asm_frac < ~0.97`** → the kernel isn't carrying the load; the measured rate is the tail again
   (the DIS-1 confound) → instrument-invalid, fix the kernel scope before trusting any number.
3. **Byte-exactness fails** at the flip seam or the dist-LUT reconciliation and can only be fixed by
   re-introducing per-symbol asm↔Rust re-entry → that revives the DIS-1 spill → NO-GO.
4. **Stage 0 PASSES but Stage 1 T4 wall does NOT move toward 0.900×** → the isolation gain doesn't
   translate (engine slack-masked even at T4) → engine isn't the binder at the cell it was thought to
   bind → STOP and re-localize (placement co-primary owns the residual).
5. **Register pressure forces a hot-value spill** that can't be relieved without re-entry → if the
   pinned set can't be held across the back-edge, F4 fails and F2/D stop paying → revert to VAR_VI.

---

## 7. THE NECESSARY-NOT-SUFFICIENT CEILING (must reach the user)

The asm kernel's whole-system ceiling = bring gzippy-native to gzippy-ISAL parity. From the GATED
BAR-2 scorecard (STATE §2): ISAL = 0.899/0.900/0.990× @ T1/T4/T8; native = 0.608/0.761/0.915×.

- **T1:** even ISA-L is 0.899× and rg ALSO uses ISA-L at T1 (engine MATCHED) yet still loses — T1 is
  structurally NON-engine-closable (per-chunk FFI handoff + serial-output floor + chunk-0 bootstrap,
  STATE §3). The asm kernel changes ~nothing at T1. **Still FAIL 0.99.**
- **T4:** native 0.761× → at best ~0.900× (LEV-1 ceiling, causally confirmed). **Still FAIL 0.99.**
- **T8:** native 0.915× → at best ~0.945–0.990× (ocl_cf/ISAL). **Could PASS at the 0.99 threshold,
  soft.**

So the kernel is a legitimate, on-target close of the **engine co-primary** and of the user's explicit
*no-C-FFI / "steal ISA-L in pure Rust"* sub-goal — but BAR-1 (≥0.99× at EVERY T) also needs the
**placement co-primary** (LEV-5: placement-perfect alone = 0.56–0.66s = +7–26%, still loses; engine
survives perfect placement and vice-versa — they are CO-primary, neither alone ties). Authorize the
asm kernel **as part of a plan that also funds placement**, not as a standalone BAR-1 close. This
re-confirms the 2026-06-07 report's standing recommendation, now on firmer ground because the host-
buffer integration it feared is already done.

---

## 8. DISCIPLINE / PROVENANCE

- Vendor kernel map: spot-verified first-hand this turn against `igzip_decode_block_stateless.asm`
  (F1/F2/F3/D/F4 line ranges) cross-checked with the prior report's already-advisor-verified map.
- gzippy production map: `decode_clean_into_contig` (`marker_inflate.rs:2071-2403`),
  `emit_backref_contig` (`:3546`), `LutLitLenCode::decode` (`lut_huffman.rs:1030`), `Bits`
  (`consume_first_decode.rs:197-313`), wiring (`gzip_chunk.rs:1295,1303`) — all read first-hand.
- Prior-attempt failure mode: `benches/engine_isolation.rs` VAR_VI (`:757`) / VAR_VII (`:1211`) read
  in full; DIS-1 / phase2-inline-asm advisor verdict read in full.
- Rate anchors (GATED, not re-measured): ISA-L 283 / scalar 104–118 / VAR_V 156 / VAR_V÷ISA-L 0.554
  / VAR_VI ~0.55–0.60× (`project_engine_plateau`, phase2 verdict); LEV-1 T4 0.761→0.900 removal
  oracle; BAR-2 scorecard (STATE §2); LEV-4/5 co-primary (`project_pregate`).
- NO build run, NO box access, NO production code edited. Self-disproof only (the Agent/advisor tool
  was unavailable). This file is the deliverable for the supervisor's delegated gate.
