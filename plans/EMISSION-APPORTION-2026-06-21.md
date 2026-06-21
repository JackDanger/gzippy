# EMISSION APPORTIONMENT — where does STEP-0's +3.22 instr/B (gz vs ISA-L `_04`) live?

**Date:** 2026-06-21 **Branch:** kernel-converge-A **Pinned sha:** 59b3123e
**Method:** STATIC objdump (load-immune, deterministic) of gz `run_contig` and ISA-L
`decode_huffman_code_block_stateless_04`, both x86_64, on the Intel guest
(i7-13700T LXC). **No code change.** **Stamp:** NOT-YET-LAW (single-arch Intel; AMD
owed) and **HYPOTHESIS-tier** (objdump AIMS the port; the wall verdict is the later
`kernel_gate` T4 causal test). This apportionment exists to AIM the port; it does not
move a wall.

## Provenance (Gate-0)
- gz binary built ON the guest at 59b3123e, `RUSTFLAGS=-C target-cpu=native`,
  `--no-default-features --features pure-rust-inflate`, `STRIP=false DEBUG=2`.
  bin sha256[:16] = `23f7d6865d5ebe26`, `git rev-parse HEAD = 59b3123e28cd`. ✓ pinned.
- gz disasm: `gzippy::…::asm_kernel::imp::run_contig` (327 insns total, all cold paths).
- `_04` disasm: `/usr/lib/x86_64-linux-gnu/libisal.so.2.0.31` @ 0x38be0 (331 insns total).
- STEP-0 measured ARM A = `Block::decode_clean_into_contig` (the WHOLE clean per-symbol
  path) vs ARM B = `_04` called directly. `decode_clean_into_contig` calls `run_contig`
  at marker_inflate.rs:3572 — so `run_contig` is the hot loop of the measured path, BUT
  the measured path also includes the Rust SCAFFOLD that `run_contig` does not (this is
  the crux below). `decode_clean_into_contig` disassembles to **5959 insns** (fast-loop
  wrapper + careful tail inlined); `_04` is **331 insns** monolithic.

## What the disasm MEASURED (load-immune, deterministic — the trustworthy layer)

### Hot LITERAL group (one pass through the shared prefix that ends `<256` → loops)
| | gz `run_contig` (`2:`→`jb 2b`) | `_04` (`38c8a`→`jl`) |
|---|---|---|
| executed insns / group | **37** | **38** |

gz is **−1 instruction per literal group** vs `_04`. Aligned element-by-element:
- guards 4 == 4; bc/cnt/flag extract+mask 9 == 9; trailing extract 2 == 2; spec store
  + advance 2 == 2; consume 2 == 2; refill A 3 == 3; litlen index 2 == 2; litlen preload
  1 == 1; refill B 5 == 5; dist preload 3 == 3; discriminator 2 == 2 → **all PARITY**.
- gz **adds** the D-1 anchor `lea p0,[pos*8]; sub p0,bitsleft` = **+2** (`c47c9/c47d1`).
- gz **omits** the EOB test `cmp 256; je` = **−2** (`_04` 38d44/38d4b; gz defers it to
  the length arm) and the speculative length precompute `lea −254` = **−1** (`_04`
  38d3d; gz does it only on the length arm). Net **−1**.

**Packing is IDENTICAL (both pack up to 3 literals/group).** gz `and $0x3,%r12d` (cnt
bits 26-27) == `_04` `and $0xc000000; shr $0x1a` (cnt bits 26-27); same LARGE_SHORT
1/2/3 scheme. gz's LUT build (`make_inflate_huff_code_lit_len`, lut_huffman.rs:458, the
singleton/pair/triple three-loop pass) is a faithful port of igzip_inflate.c:386-599 and
is called with `TRIPLE_SYM_FLAG` (lut_huffman.rs:1045) → triples enabled, same as igzip.
⇒ **symbols-per-iteration is at PARITY; "gz decodes fewer symbols/iteration" is
FALSIFIED by the disasm.**

### Length / backref group (prefix falls through `>=256` → dist decode + copy)
| | gz | `_04` |
|---|---|---|
| prefix (shared) | 37 | 38 |
| length TAIL (1× 16B copy, short dist, bitsleft≥48) | **~53** | **~35** |
| **total length group** | **~90** | **~73** (Δ **+17**) |

Length-tail Δ (+18) decomposition (gz − `_04`), all MEASURED:
- dist-CORE decode (consume + base + extra): gz 17 ≈ `_04` 17 → **PARITY**.
- **MOVDQU back-ref copy: gz 10 ≈ `_04` 9 → PARITY (+1). ELEMENT M HOLDS — NOT overturned.**
- **distance range + window-base/marker validation** (`je dist==0`, `cmp 0x8000;ja`,
  `cmp out_base;jb`) = **+5** — the PARALLEL window-absent machinery; `_04` is a single
  stateless stream and has no counterpart.
- D-1 `d0` capture on the length arm (`mov d0,dst; sub d0,cnt`) = **+2** (NON-ADDRESSABLE).
- pre-copy refill check (`cmp 48; jae`) = **+2** (`_04` amortises refill in the prefix).
- resumable boundary restore (`mov ret,1`) + carried→t1 + loop tail = **+2** (resumable
  contract; `_04` exits with a bare `jle`).
- EOB/oversize re-dispatch on the length arm = **+3** (the flip side of the literal-path
  −2 EOB saving — net favourable across paths).
- `dec dst` / misc = +1.

## Per-byte conversion (INFERRED — Gate-0: needs the group/byte ratio)
groups/B is anchored by the NIGHT32 D-1 measurement: D-1 = `lea+sub` (2 insns) on every
group ⇒ 0.607 instr/B ⇒ **groups/B ≈ 0.30** (independent of any ratio×ratio synthesis;
it back-solves a MEASURED instr/B against a counted 2-insn/group cost). Taking silesia
≈ 39% length-groups of all groups → **len-groups/B ≈ 0.117, lit-groups/B ≈ 0.183**.

Hot-loop Δ instr/B = (−1)(0.183) + (+17)(0.117) = **+1.81 instr/B**.

| bucket | per-group | instr/B (INFERRED) | class |
|---|---|---|---|
| D-1 anchor (lea/sub all groups + d0 on length) | +2 / +2 | **+0.6 – 0.83** | **NON-ADDRESSABLE** (NIGHT32 SLACK; confirms ~0.607) |
| length-tail distance/window-base validation | +5 (len) | **+0.59** | architectural (parallel window-absent) |
| length-tail pre-copy refill cadence | +2 (len) | +0.23 | ADDRESSABLE (element F, backref arm) |
| length-tail resumable boundary + loop tail | +2 (len) | +0.23 | resumable contract (semi-addr.) |
| EOB/len-precompute (net across lit+len) | −3 lit /+3 len | **−0.20** | gz FAVOURABLE (late discriminator) |
| copy / dec / misc | +2 (len) | +0.23 | near-parity |
| **HOT-LOOP SUBTOTAL** | | **≈ +1.8** | |
| **SCAFFOLD residual (decode_clean_into_contig − run_contig)** | | **≈ +1.4** | NOT in run_contig; no loop_block counterpart |
| **TOTAL (STEP-0 measured)** | | **+3.22** | |

## CONSERVATION CHECK — does NOT reconcile inside `run_contig` (REPORTED, not forced)
The buckets visible in `run_contig`'s disasm sum to **≈ +1.8 instr/B**, i.e. only ~56% of
the measured **+3.22**. **~+1.4 instr/B (≈ 44%) is NOT in `run_contig` at all.** It lives
in the `decode_clean_into_contig` SCAFFOLD (a 5959-insn Rust function — fast-loop
wrapper, table-ensure, careful tail, per-call prologue/epilogue) which ARM A includes and
which has **no `_04` / loop_block counterpart** (`_04` is 331 insns, fully monolithic).
This residual is sized (5959 vs 331 static) but its per-byte EXECUTED share was NOT
measured here (objdump gives static size, not dynamic per-byte) — that is the explicit
OWED follow-on (objdump-apportion `decode_clean_into_contig`'s executed-per-byte body,
or a perf-annotate of the clean path).

## VERDICT — the premise is REVISED, no loop_block emission element is the #1 target
The advisor's leading candidates are **FALSIFIED by the disasm**:
- "gz decodes fewer symbols/iteration than `_04`'s multi-symbol loop" → FALSE; packing is
  identical (both triple, same LUT pass).
- "per-symbol bookkeeping" on the literal path → FALSE; gz literal group is **−1 insn**
  below `_04`. gz's per-symbol literal EMISSION already matches-or-beats `_04`.

What the disasm shows instead:
1. `run_contig`'s hot LITERAL emission is **already converged (−1 vs `_04`)**; the
   MOVDQU copy (element M) and the dist-core are at **parity**.
2. The only hot-loop divergence is the **length/backref TAIL (≈+1.8 instr/B gross)**, and
   it is **DIFFUSE + dominated by NON-portable costs**: the NON-ADDRESSABLE D-1 anchor
   (~0.6, NIGHT32 SLACK) + the architectural parallel **window-base/marker validation**
   (~0.59, no `_04` counterpart) + the resumable contract (~0.23). The single genuinely
   `_04`-portable sub-bucket inside the loop is the **pre-copy refill cadence** on the
   backref arm (~+0.23 instr/B — apply element-F's split-refill to the `58:`/`51:`
   pre-copy refill so it amortises like `_04`'s prefix). Small.
3. The **LARGEST addressable surplus (~+1.4 instr/B) is the SCAFFOLD OUTSIDE
   `run_contig`**, which has no loop_block element to transliterate. The port target is
   therefore **NOT** a loop_block emission technique; it is converging
   `decode_clean_into_contig`'s per-byte scaffold toward `_04`'s monolithic single-pass
   shape.

### RANKED addressable buckets (largest first)
1. **SCAFFOLD residual ≈ +1.4 instr/B** — outside `run_contig`. Largest. NEXT ACTION:
   apportion it (objdump `decode_clean_into_contig`'s executed-per-byte path / perf
   annotate the clean loop) BEFORE any port — we do not yet know which scaffold sub-part
   is per-byte vs per-block.
2. **Length-tail pre-copy refill cadence ≈ +0.23 instr/B** — the one in-loop, `_04`-
   portable element (element F on the backref arm). Small but byte-exact and cheap.
3. **Length-tail resumable boundary/loop-tail ≈ +0.23 instr/B** — resumable contract;
   semi-addressable.

### NON-ADDRESSABLE subtractions (confirmed)
- **D-1 anchor**: confirmed **≈ 0.6–0.83 instr/B** (lea/sub on all groups + d0 on length),
  consistent with NIGHT32's measured **0.607** for the lea/sub core. NIGHT32 SLACK
  (deleting it raises cyc/B). EXCLUDED from the port.
- **Wide-copy (element M)**: gz 10 ≈ `_04` 9 insns/copy → **at parity. Element M HOLDS;
  NOT overturned.** EXCLUDED from the port.

**Stamp:** NOT-YET-LAW (Intel-only; AMD/Zen2 owed). HYPOTHESIS-tier (objdump aims; the
`kernel_gate` T4 wall A/B is the causal verdict). The conservation gap (~+1.4 instr/B
outside `run_contig`) is the headline: it redirects the port from the (already-converged)
hot loop to the clean-path scaffold, and is itself the first thing to apportion next.
