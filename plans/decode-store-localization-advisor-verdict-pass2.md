# DISPROOF ADVISOR VERDICT — PASS 2 (plateau / fork gate)

**Headline: the emerging "BMI2/LUT/store/pipeline are exhausted ⇒ this located decode region
is the 1.0×-vs-no-FFI PLATEAU/FORK" conclusion does NOT survive disproof. REFUTED on P1, P2,
P4; P3 returns UNDETERMINED (cannot yet attribute the residual to an ISA-L asm fork).**

The conclusion fails for three independent reasons, any one of which is sufficient to block the
fork:

1. **Category error in the BMI2 bound.** "BMI2 bit-extraction is optimal" bounds ONE micro-
   resource (the bit math). It says *nothing* about the resource that actually dominates a serial
   Huffman decode loop: the **dependent table LOAD latency**. The whole PASS-2 case treats
   "decode-compute" as ≡ "bit-extraction" and never touches table-load. So the disasm fact does
   not bound the decode-compute lever — it bounds the wrong sub-resource.

2. **The owner's OWN standing oracle says this region cannot reach parity.** `ocl_cf` (whole
   clean-engine removed) = **0.925×rg**, on the same ×rg axis where `native_fold` = 0.79×rg and
   `rg` = 1.0. Removing decode **and** store **entirely** still leaves gzippy at 92.5% of
   rapidgzip — **sub-parity**. Therefore NO amount of decode-side work on this loop can close the
   gap to 1.0×rg; ≥0.075×rg of the deficit lives **outside** the clean engine. Forking "at the
   engine" is forking at a region whose removal oracle proves it is not the whole binder. (See
   `[[project_pregate_placement_is_dominant_lever]]` — PLACEMENT + ENGINE are co-primary; this
   oracle is the quantitative confirmation.)

3. **A non-contaminated bound is buildable and was not built.** The contamination argument kills
   only the *wrong-bytes* decode_free oracle. The *correct-bytes replay* oracle the owner names
   in P2 produces correct window seeds ⇒ no spec-failure net ⇒ clean wall. Campaign rule: no
   STOP/TIE/FORK without a validated removal bound. That bound is owed and obtainable.

---

## P1 — Are the authorized decode-side techniques EXHAUSTED on this region (the plateau)?

**VERDICT: REFUTED.** The exhaustion claim is overstated on two fronts.

**(a) The BMI2 evidence is insufficient as stated.** "433 BZHI/PEXT in the binary" is a
*whole-binary* count plus an *assertion* about how LLVM lowers `next_bits & ((1<<n)-1)`. It is
not a disassembly of the specific hot function (`LutLitLenCode::decode` / the VAR_V fast loop)
confirming (i) those masks actually became `bzhi` on the hot path and (ii) there is no scalar
shift/and chain that a single `PEXT` could collapse for *parallel multi-field* extraction
(symbol + length + dist subfields in one PEXT). The 433-count is consistent with BMI2 appearing
anywhere (CRC, table build, cold paths). To bound the bit-extraction lever you must disassemble
the hot function specifically. As stated, the bound is asserted, not demonstrated.

**(b) Named, authorized techniques are NOT confirmed present.** The CLAUDE.md authorization list
includes items the brief either omits or conflates:
- **Table `_mm_prefetch` ahead of the dependent load** — the brief's "software-pipelined preload"
  hides the *bit-refill* latency (preloading next bits). That is a DIFFERENT resource from
  prefetching the *decode-table cache line*. The table load depends on the just-decoded index, so
  bit-preload does not hide it. Not confirmed addressed.
- **Fixed/static-Huffman specialization** — ISA-L and libdeflate carry a separate, branch-lean
  decode path for static blocks. Not mentioned as present in gzippy's clean loop.
- **FASTLOOP_OUTPUT_MARGIN yield-check elision** — the libdeflate technique of checking output
  margin ONCE then running a tight per-symbol-branchless inner loop. The brief says "fast loop"
  but does not confirm the per-iteration resumable/margin branch is fully elided. An extra
  unpredictable branch per symbol in a load→decode→store serial chain is exactly the kind of tax
  ISA-L's asm omits.
- **Decode-table geometry** — "multi-symbol LUT sym_count 1..=3 present" does not establish the
  table is the same first-level *width* / L1-residency as ISA-L's `DECODE_LOOKUP`. A deeper
  single-level table that retires the common literal in one L1-resident load (no second-level
  walk) is a distinct, unconfirmed lever that directly attacks table-load latency.

Conclusion: this is **not** demonstrated to be the plateau. At least four authorized techniques —
and the dominant resource (table-load latency) — are unexamined.

---

## P2 — Does BMI2-already-optimal SUFFICE to bound the decode-compute lever to ~0, or is a non-contaminated oracle owed?

**VERDICT: REFUTED — disasm does NOT suffice; the correct-bytes replay oracle is owed.**

- BMI2-optimal bounds **bit-extraction**, not **decode-compute's wall share**. Decode-compute in
  this loop = bit-refill (preloaded) + bit-extraction (BMI2-maxed) + **the dependent table
  load** (untouched). The third term is the classic Huffman binder and is *invisible* to the
  BMI2 argument. So the disasm fact cannot drive the decode-compute lever to ~0 — it leaves the
  largest sub-term unbounded.
- The contamination finding is correct **only for the wrong-bytes oracle**. Wrong synthetic bytes
  corrupt the per-chunk window seed ⇒ `flip_to_clean=874`, `other=857` re-decodes ⇒ masked wall.
  The asymmetry the owner noticed is real and instructive: **store** removal (NODRAIN) keeps the
  decoded symbols that seed the window, so its wrong bytes only surface at the terminal CRC;
  **decode** removal corrupts the seed itself. That is precisely why the bound must come from a
  **correct-bytes replay** oracle: decode the real stream once, cache the
  (symbol, length, dist) stream, then on the measured run replay it with `decode()` made free
  while every store/copy runs. Correct bytes ⇒ correct seeds ⇒ zero spec-failure net ⇒ clean
  `off → decode_free` wall delta = decode-compute's removable share. Heavy, but byte-exact and
  non-contaminated. The owner already sketched it; the contamination of the *other* oracle is not
  a license to skip it.
- Note subtraction is NOT a substitute: in a serial dependent loop `off ≠ t_decode + t_store`
  (overlap), so `ocl_cf − NODRAIN` does not cleanly yield the decode share. The replay oracle is
  the real instrument.

**Decision value is real, not ceremonial:** ocl_cf already caps ALL clean-loop decode-side work at
~0.135×rg (0.79→0.925). The replay oracle splits that 0.135 into decode-share vs store-share. If
decode-share is large, a table-load-latency technique is worth it; if ~0, the engine-side fork is
*justified* (for the ≤0.135 it can ever buy). Either way the fork verdict is currently
**unvalidated**.

---

## P3 — Is the residual an ISA-L asm structural FFI advantage (the fork), or a still-unported pure-Rust technique?

**VERDICT: UNDETERMINED — cannot be attributed to an FFI fork yet.**

"rapidgzip uses ISA-L, ISA-L is hand-tuned asm, therefore the residual is a structural FFI
advantage" is **not a located mechanism** — it is the hand-wave the campaign's reject-rule
(rule 7) explicitly forbids. To bank the fork you must point at the *specific* ISA-L asm
advantage and show it is unreachable in pure Rust. Until the P1 techniques are tried, the residual
has at least four *pure-Rust-reachable* candidate homes:

1. **Table-load latency** — attacked by a deeper L1-resident single-level table and/or genuine
   table-line prefetch. This is the prime suspect once bit-extraction is BMI2-maxed.
2. **Per-symbol margin/yield branch** (FASTLOOP elision) — pure-Rust reachable.
3. **Static-Huffman fast path** — pure-Rust reachable.
4. **The non-engine 0.925→1.0×rg deficit** — by ocl_cf, *most* of the gap to parity is NOT in the
   engine at all; it is scheduler/placement. ISA-L asm cannot be the explanation for a deficit
   that survives full engine removal.

So the residual is **provisionally pure-Rust-reachable and partly non-engine**; the FFI-fork
attribution is unproven.

---

## P4 — Escalate the 1.0×-vs-no-FFI FORK now, or is a named technique still owed?

**VERDICT: REFUTED — do NOT escalate the fork now. A named technique is owed, and the fork as
framed targets the wrong region.**

Two blocking facts:
- **The region cannot reach parity even fully removed** (ocl_cf 0.925×rg < 1.0). Forking "at the
  clean engine" mislocates the gap; ≥0.075×rg is outside it. The correct fork target, if any, is
  the FULL deficit (engine ≤0.135 + placement ≥0.075), not "the decode loop is where FFI wins."
- **No validated decode-compute bound exists** (P2). Per campaign rule, no STOP/TIE/FORK without a
  removal/oracle bound.

**Strongest single still-owed pure-Rust decode-side technique: attack TABLE-LOAD LATENCY, not bit
math.** Concretely, in priority order:
1. **Disassemble the hot function** and confirm whether the per-iteration margin/resumable branch
   is present; if so, implement the **FASTLOOP_OUTPUT_MARGIN** single-check tight inner loop
   (highest-leverage, cheapest, directly mirrors ISA-L/libdeflate).
2. **Widen the first-level decode table to a single L1-resident lookup** that retires the common
   literal with no second-level walk (ISA-L `DECODE_LOOKUP` geometry) — the direct attack on the
   dependent-load latency that BMI2 cannot touch.
3. Build the **correct-bytes replay decode-free oracle** to put a validated number on
   decode-compute's share before deciding #1/#2 are exhausted.

If #1 and #2 land (or are confirmed already present at ISA-L geometry) **and** the replay oracle
shows decode-compute's removable share ≈ 0, *then* the engine-side fork is validated for the
≤0.135×rg it can buy — but the parity verdict still routes through the non-engine 0.075+ deficit,
which is the larger and co-primary lever per `[[project_pregate_placement_is_dominant_lever]]`.

---

## Bottom line for the campaign log

- **KEEP:** the clean FOLD loop is on the T8 critical path (re-confirmed, not in dispute).
- **DO NOT BANK:** "BMI2/LUT/store/pipeline exhausted ⇒ decode-compute lever ≈ 0 ⇒ plateau/fork."
  BMI2-optimal bounds bit-extraction, not table-load latency; the exhaustion list omits
  table-prefetch, static-Huffman specialization, FASTLOOP yield-elision, and table geometry.
- **DECISIVE NEW FINDING:** ocl_cf 0.925×rg < 1.0 ⇒ the clean engine, *fully removed*, is still
  sub-parity. The fork-at-the-engine framing is refuted by the owner's own oracle; most of the
  parity gap is non-engine (placement).
- **OWED before any fork:** (1) disasm the hot function (settles FASTLOOP-branch + confirms hot-path
  bzhi); (2) try the table-load-latency techniques (FASTLOOP elision, single-level L1 table);
  (3) the correct-bytes replay oracle for a validated decode-compute share. The wrong-bytes
  oracle's contamination excuses none of these — the correct-bytes variant is buildable and clean.
