# ENGINE ISOLATION BENCH — independent disproof advisor verdict

Reviewer: independent Opus advisor (read-only). Repo HEAD 249f25b5, branch reimplement-isa-l.
Method: source-verified every premise first-hand (bench, production clean loop, ISA-L oracle,
build cfg). I did NOT build or run; I attacked the bench's *construction* and the leader's
*conclusions* against the code.

## ONE-LINE BOTTOM LINE
The bench is a **TRUSTWORTHY instrument for what it actually measures** (gzippy's current clean
inner loop vs a *pure* ISA-L clean decode on one chunk), and that gap is real at **~3.1×**. But
two things must be corrected before it gates a multi-week build: (1) the headline ceiling number
should be quoted as **iii/ii ≈ 3.10×, not iii/i = 3.29×** (variant (i) carries ~6% sink overhead
production does not pay); (2) the self-test "FAIL" is a **mis-calibrated band, not a broken
instrument** — but the bench has **not yet run the variants the falsifier is defined over** (E2–E4),
so the go/no-go is **genuinely un-evaluated, neither proven nor refuted.**

---

## (a) BENCH VALIDITY — CORROBORATE-WITH-CAVEATS

### Variant (i) IS the production clean loop (better than the falsifier doc claims)
The falsifier doc says variant (i) drives `read_internal_compressed_canonical`. **It does not** —
and that is *good news*. Trace: `Block::read` (`marker_inflate.rs:968`) → for Fixed/Dynamic →
`read_internal_compressed` (`:981`) → the cfg-selected `read_internal_compressed` at `:1114`
(compiled because this build sets `pure_inflate_decode`; see below) → `read_internal_compressed_specialized::<false>` (`:1191`), which is the **ISA-L-LUT multi-cached / TRIPLE_SYM packed-literal
loop** (`:1206-1245`), with marker-maintenance dead-stripped (`CONTAINS_MARKERS=false`). That is
exactly the production post-FlipToClean hot path, *not* the ~50 MB/s canonical fallback. So the
worry "is variant (i) the slow canonical arm / missing the multi-literal cached loop?" is
**refuted by source**: it is the fast LUT loop.

- **cfg confirmation:** `build.rs:93-107` — `has_pure_rust_inflate ⇒ parallel_sm ⇒ pure_inflate_decode`.
  Built with `--features pure-rust-inflate,isal-compression` on x86_64 ⇒ `pure_inflate_decode` is
  set ⇒ the `:1114` LUT dispatch compiles (not the canonical `:1462`). ✓
- **clean-arm assertion is real:** `set_initial_window_impl` sets `contains_marker_bytes=false`
  (`:714`) for a full 32 KiB window; the bench asserts `mid.window.len()==MAX_WINDOW_SIZE` (`bench:215-220`)
  and `!contains_marker_bytes()` (`bench:132,160`). Genuine clean decode. ✓
- **LUT warmth:** `block_huffman_luts_ready` resets per `read_header` (`:790`) and rebuilds per
  block (`:1198-1201`) — *identical to production's per-block rebuild*. Plus a discarded warmup
  iteration (`bench:262-264`) and best-of-11 median. Caches warm; not an un-warmed-LUT artifact. ✓

### Variant (ii) is production-representative; (i) carries ~6% extra. Key finding.
Both variants share `drain_to_output` (`:731`), which on the clean branch (`:738-750`) **already
narrows the u16 ring to a u8 `u8buf`** and calls `push_clean_u8`. The ONLY difference:
- (ii) `U8Sink::push_clean_u8` → `extend_from_slice` into `Vec<u8>` (`bench:120-122`). This mirrors
  **production's clean-tail sink** `gzip_chunk.rs:842` (`chunk.append_clean(bytes)`, u8-direct, no
  narrow). So **(ii) ≈ production** (production additionally folds CRC/subchunk accounting in
  `append_clean`, a fixed streaming cost orthogonal to decode-loop speed).
- (i) uses a `Vec<u16>` sink whose `push_clean_u8` is the **default trait impl** (`:42-50`,
  Vec<u16> does *not* override it, confirmed `:107-120`): it **re-widens** the already-narrowed
  u8buf back to u16 via a 256-stack buffer, stores 2 bytes/elem, then **narrows a second time**
  at `:150`. So (i) does u16→u8→u16→u8 — overhead production never pays.

Measured cost of that overhead: **(ii)/(i)=1.059, i.e. ~6%.** This is decisive for question (2)
below: the u16/narrow tax does *not* massively inflate (i); the inner loop dominates.

### ISA-L oracle (iii) does the SAME work — and the bench is biased AGAINST it (conservative)
`decompress_deflate_from_bit` (`isal_decompress.rs:307`): raw deflate (`crc_flag=ISAL_DEFLATE`,
`:325`, no gzip reparse, no CRC compute), 32 KiB dict via `isal_inflate_set_dict` (`:351-362`),
`inflatePrime` for the sub-byte start (`:327-339`) — same start_bit, same dict, same raw stream as
(i)/(ii). `cap = max_output.clamp(...)` = n_actual (`:371`); loop breaks at `remaining==0` (`:377`)
⇒ decodes **exactly n_actual bytes**, same N. No short-circuit: the premature-BFINAL guard
(`:396-407`) only fires if BFINAL lands early, in which case n_actual (set by (i)) is already short
and all three stop together — CRCs then match over the full n_actual.
- **Bias direction:** (iii)'s timed region includes `vec![0u8; cap]` which *zeroes* 4 MiB every
  iteration (`:372`) — a cost gzippy's `Vec::with_capacity` (no zero) does *not* pay. So ISA-L is
  carrying ~0.4 ms of extra setup; the true decode gap is if anything **larger** than reported.
  The 3.1× is a **conservative** floor on ISA-L's advantage, not an inflated one.

### Byte-exactness gate — STRONG (agree)
Not just CRC: the bench does a full `a == b` / `a == c` memcmp over n_actual (`bench:245-247`); CRC
is printed only for diagnostics. A full 4 MiB memcmp + matching CRC is a strong gate; a CRC32
collision on equal-length naturally-decoded output is negligible. **Agree the gate is strong.**

### CAVEAT — single chunk does NOT bound the ceiling
N=4 MiB from the **single** median-`start_bit` seed entry (`bench:213-221`). ISA-L's relative
advantage is workload-dependent: its TRIPLE_SYM packed-literal LUT pays most on literal-heavy text;
SIMD back-ref copy pays most on match-heavy data; dynamic-vs-fixed Huffman mix shifts header cost.
silesia is a tar of heterogeneous members — one 4 MiB mid-stream slice can be atypical either way.
**One chunk is suggestive, not a bound.** The bench should sweep ≥3–5 entries (low/median/high
start_bit, spanning a text region and a binary region) and report the *range* before this number
gates a multi-week commitment.

---

## (b) THE 3.29× CEILING NUMBER — REFUTE the headline, CORROBORATE the corrected ~3.1×

**The headline iii/i = 3.287 over-states the production gap.** Variant (i) is *not* the production
clean baseline — it carries the +6% u16/narrow sink tax (production uses the u8-direct clean-tail
sink, `gzip_chunk.rs:842`, mirrored by variant (ii)). The production-representative ceiling is:

  **iii/ii = 0.033563 / 0.010814 = 3.10×.**

So the leader should quote **~3.1× (range 3.10–3.29× bracketing sink uncertainty)**, not 3.29×.

**Is ~3.1× a plausible ISA-L-AVX2-vs-Rust-scalar-u16 gap, or a broken-instrument red flag?**
**Plausible, not a red flag.** ISA-L inflate is hand-AVX2/BMI2 C with packed multi-literal stores,
wide refill, and SIMD copy; gzippy's current loop is scalar with a **u16 ring** (`output_ring:
Box<[u16; RING_SIZE]>`, `:290`) — every back-ref copy moves 2 bytes/elem, single-literal stores,
scalar refill. libdeflate/ISA-L routinely beat naive scalar decoders by 2–4×. ~3.1× sits squarely
in that band.

**Why is the isolated 3.1× LARGER than the prior step-a2 2.38×?** Because the *denominators differ*,
not because the instrument is wrong. The 2.38× was gzippy-clean **vs rapidgzip(igzip)'s per-chunk
decode at the wall** — a denominator that itself carries marker-resolution + window-apply +
pipeline. This bench's (iii) is a **pure** ISA-L clean decode with *zero* marker machinery and no
CRC. A purer/faster denominator ⇒ a larger ratio. **The 3.1× and the 2.38× are not the same
measurement and must not be conflated** — 3.1× = "gzippy-current-clean-loop vs pure-ISA-L-clean,"
which is the right quantity for bounding the *inner-loop* ceiling, but it is *not* the
gzippy-vs-rapidgzip-system gap.

**CRUX answered:** No, (i)'s sink overhead does NOT materially over-state the inner-loop gap — it
accounts for only 6%. Stripping it (→ (ii)) still leaves ISA-L 3.1× ahead. The gap lives in the
**inner loop** (u16 ring copies, scalar refill, single-literal store), which is exactly what E2–E4
target and exactly what this bench has **not yet measured**.

---

## (c) THE PLATEAU / "NOT YET PROVEN" CONCLUSION — CORROBORATE (with a sharpening)

The leader's read — "E1 plateaus (+6% output traffic only); engine front NOT YET PROVEN because
E2–E4 (the dominant inner-loop gap) are unmeasured" — is **correct and disciplined**:

- **Cannot REJECT now (rule 3 + rule 7).** E1 was never the lever that closes 3×; it is the
  smallest sub-lever. Its +6% plateau says nothing about E2 (SIMD u8 back-ref copy — the u16 ring
  is *still u16* in variant (ii), `bench:19-21`, ring type `:290`), E3 (packed multi-literal store),
  or E4 (wide refill). Per rule 7, rejecting the engine front would require a *mechanism* + how
  rapidgzip does it differently — and rapidgzip/ISA-L's existence proof (388 MB/s pure) says the
  inner-loop headroom is *real and large*. So "reject now / re-confront placement" is **not
  licensed** by this data.
- **Cannot declare PROVEN.** The pre-registered FALSIFIER (`engine-bench-falsifier.md:43-65`) is
  defined over variant (ii) *after E2/E3/E4*. Only E1-partial exists. The go/no-go is **un-evaluated.**

**Sharpening (the gap in the current bench vs its charter):** the charter intended variant (ii) to
*be* the technique stack ("START with E1, add E2/E3/E4 as tractable, inline-asm where Rust lags").
The bench as run only implements E1-partial (output traffic; ring still u16). So the honest status
is: **instrument validated and ceiling bounded (pure-ISA-L = 388 MB/s; current clean = 125 MB/s),
but the variants the falsifier adjudicates were not built.** The correct next step is **in-bench**:
extend variant (ii) into a real E2–E4 prototype (u8 ring + SIMD copy + packed store + wide refill,
inline-asm where needed) and re-run — that is *bench* work, not the production integration the
charter gates. Only then is the falsifier evaluable. The leader should not be read as "stuck"; the
bench did its first job (bound the ceiling, validate the oracle) and now needs its second
(prototype the techniques in the (ii) slot).

---

## (d) RIDER 2 — SAME-SINK FLOOR CORRECTION — CORROBORATE-WITH-CAVEATS

`/dev/shm` (tmpfs) is a **legitimate same-sink proxy** for the writev syscall + buffered-write CPU
cost — *provided rapidgzip's 0.604s was measured to the identical sink.* It omits real-disk
writeback/metadata, but neither gzippy nor rapidgzip fsyncs (same as gzip(1)), so writeback is off
both critical paths; the omission is shared and the **ratio is robust**. Caveat: the *absolute*
0.604s is mildly optimistic vs a pressured on-disk sink — fine for a same-sink comparison, do not
quote it as an on-disk floor.

**Does the correction change the §3 verdict? No.** The bar moves from 0.54s → 0.604s (a ~12%
relaxation, favorable). But gzippy's same-sink T8 wall is **1.211s ≈ 2.0× rapidgzip's 0.604s.**
gzippy must roughly *halve* its wall to tie; a 12% bar relaxation does not move that qualitative
verdict. The ~2× same-sink wall gap is consistent with the ~3.1× isolated inner-loop gap (partly
hidden by parallel overlap), which **corroborates** the engine — not placement — as the dominant
remaining lever, *contingent* on E2–E4 actually landing near ISA-L (still unproven).

---

## IS THIS A TRUSTWORTHY INSTRUMENT TO GATE THE MULTI-WEEK BUILD?

**Yes — as far as it goes, and it does not yet go far enough.** It is internally valid: correct
production hot-path dispatch (verified to source, `:1114→:1191→:1206`), genuine clean decode, same
N / same dict / same raw stream for the oracle, full-memcmp byte gate, sub-1% sigma on (i)/(ii),
and a *conservative* bias against ISA-L. The self-test "FAIL" is a **mis-calibrated band** (set from
the 2.1–2.38× system numbers; the bench measures a purer, larger ~3.1× gap), **not** a third broken
instrument — *but* per rule 4 the band must be **recalibrated and re-passed** (e.g. to ~[2.5×, 3.6×]
with the denominator-difference documented) before it can serve as the guard its charter requires;
do not leave a standing red "SELFTEST=FAIL" as the gate.

**Before this number authorizes the multi-week build, three things are owed:**
1. **Re-quote the ceiling as iii/ii ≈ 3.10×** (production-representative), not iii/i = 3.29×.
2. **Sweep ≥3–5 chunks** (text + binary regions) and report the range — one chunk is not a bound.
3. **Build the E2–E4 prototype in variant (ii)** and re-run; the pre-registered falsifier is
   defined over that and is currently un-evaluated. Neither "proven" nor "refuted" yet.

**Net verdict:** (a) bench validity **CORROBORATE-WITH-CAVEATS**; (b) the 3.29× headline **REFUTE**
(correct to ~3.1×, which is then a real and plausible gap); (c) plateau/not-proven conclusion
**CORROBORATE** (sharpen: build E2–E4 in-bench next); (d) rider-2 floor correction **CORROBORATE-
WITH-CAVEATS** (legit same-sink ratio; does not change the ~2× §3 verdict). The bench is **not
broken** and is **not misleading once the headline is corrected** — but it is **incomplete**: it
has bounded the ceiling and validated the oracle; it has not yet measured the variants that decide
the go/no-go.
