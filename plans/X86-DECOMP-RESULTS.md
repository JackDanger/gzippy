# x86 T1 decode decomposition — cheap-header igzipbare oracle (recovery run)

Branch `kernel-converge-A` @2d36c500. Oracle: `scripts/bench/_igzipbare_oracle_guest.sh`
driving `examples/streaming_thin.rs` (gzippy-isal example, RUSTFLAGS=-C target-cpu=native,
built in /dev/shm). Re-run of the LOST prior-cycle decomposition; banking every cell.

Arms (all from the SAME isal binary; gz kernel codegen byte-identical across arms):
- `thin`      = gz contig driver + gz run_contig kernel + gz table-build (NO CRC)
- `igzipbare` = gz contig driver + igzip read_header (via isal_inflate stop) + igzip `_04` (NO CRC)
- `cheap`     = gz contig driver + LEAN `gzippy_read_header_export` + igzip `_04` (NO CRC)
- `igzip`     = full ISA-L monolithic streaming decode (WITH CRC) — the x86 bar

Decomposition reported per corpus:
- HEADER-ARTIFACT removed = (igzipbare − cheap)/igzip   [non-inert proof: must exceed A/A spread]
- CLEAN SCAFFOLD          = (cheap − igzip)/igzip         [gz contig driver vs igzip monolith; gz skips CRC]
- INNER-DECODE KERNEL ceiling = (thin − cheap)/thin       [gz kernel+tables vs igzip read_header+`_04`]

Ratios = arm/igzip (>1 = slower than igzip). Best-of-15 interleaved, randomized arm order,
/dev/null both arms, pinned cpu4.

## CHEAP-HEADER NON-INERT STATUS (honest finding)
The cheap path is a genuinely distinct code path (source-verified: `igzip_bare_cheap`
calls `gzippy_read_header_export` at streaming_thin.rs:469, NOT `isal_inflate` as bare
does at :310) and produces byte-exact output (GATE0 bytes==zcat PASS for cheap+cheap2).
HOWEVER, EMPIRICALLY the per-block isal_inflate header overhead it removes is ~0:
`(igzipbare − cheap)/igzip` ≤ 0.3% on EVERY corpus on BOTH arches, below A/A spread.
⇒ The "contamination" the cheap variant was built to remove is empirically NEGLIGIBLE;
`cheap ≈ bare` everywhere. CONSEQUENCE: the legacy "contaminated" (bare−igzip) scaffold
numbers were already clean; the cheap variant CONFIRMS them rather than correcting them.
The decomposition below is robust whether computed from `cheap` or `bare`.

## INTEL — neurotic (i7-13700T LXC), pin cpu4, best-of-15, load ~1.5–2.3 [BANKED]
Gate-0 PASS: bytes==zcat ALL arms; A/A self-tests 0.07–4.25ms ≪ inter-arm Δ.

| corpus   | thin/igzip | bare/igzip | cheap/igzip | HEADER-ARTIFACT (bare−cheap)/igzip | CLEAN SCAFFOLD (cheap−igzip)/igzip | KERNEL ceiling (thin−cheap)/thin |
|----------|-----------|-----------|-------------|-----------------------------------|------------------------------------|----------------------------------|
| silesia  | 1.169     | 1.077     | 1.078       | −0.0%                             | **7.8%**                           | **7.8%**                         |
| nasa     | 1.188     | 1.019     | 1.016       | 0.3%                             | 1.6% (TIE: Δ3.6 < spread8.4)       | **14.5%**                        |
| monorepo | 1.229     | 1.059     | 1.060       | −0.0%                            | 6.0% (Δ6.3 ≈ spread5.9, borderline)| **13.8%**                        |
| squishy  | 1.136     | 1.109     | 1.110       | −0.2%                            | **11.0%**                          | 2.3% (TIE: Δ33 < thin-spread43)  |

Cross-check vs lost prior partial (kernel-ceiling): silesia 7.6/nasa 14.5/monorepo 14.4/
squishy 2.4 — REPLICATES (mine 7.8/14.5/13.8/2.3, within noise). ✓

## AMD — solvency (EPYC 7282 Zen2), FROZEN gov=performance boost=0, pin cpu4, best-of-15, load ~1–2 [BANKED]
Zen2 PEXT/PDEP microcoded (~18cyc): both gz run_contig and igzip `_04` use BMI2 there
(kernel-edge mechanism differs from Intel). Frozen via /root/bench-env.sh; thawed+VERIFIED
after (gov=ondemand, boost=1, paranoid=4). Gate-0 PASS: bytes==zcat ALL arms; A/A
self-tests 0.05–1.46ms ≪ inter-arm Δ.

| corpus   | thin/igzip | bare/igzip | cheap/igzip | HEADER-ARTIFACT (bare−cheap)/igzip | CLEAN SCAFFOLD (cheap−igzip)/igzip | KERNEL ceiling (thin−cheap)/thin |
|----------|-----------|-----------|-------------|-----------------------------------|------------------------------------|----------------------------------|
| silesia  | 1.094     | 1.111     | 1.108       | 0.3%                              | **10.8%**                          | −1.3% (TIE: Δ5 < spread12–16)    |
| nasa     | 1.100     | 1.004     | 1.003       | 0.1%                              | 0.3% (TIE: Δ0.4 < spread)          | 8.8% (Δ12.8 ≈ cheap-spread12, borderline) |
| monorepo | 1.161     | 1.097     | 1.092       | 0.5%                              | **9.2%**                           | **5.9%**                         |
| squishy  | 1.073     | 1.156     | 1.156       | 0.0%                              | **15.6%**                          | **−7.8%** (gz kernel BEATS igzip `_04`; Δ60 ≫ spread) |

Cross-check vs lost prior partial (kernel-ceiling): silesia −0.1(TIE)/monorepo +6.9/
squishy −6.1 — REPLICATES (mine silesia −1.3 TIE / monorepo +5.9 / squishy −7.8). ✓
nasa (+8.8%) is the NEW AMD cell the prior cycle never finished.

## VERDICT (the complete 8-cell picture)
The decomposition is now complete for BOTH levers × 4 corpora × 2 arches.

KERNEL CEILING (thin−cheap)/thin — inner-decode swap (gz kernel → igzip read_header+`_04`):
| corpus   | Intel  | AMD/Zen2 |
|----------|--------|----------|
| silesia  | 7.8%   | −1.3% (TIE) |
| nasa     | 14.5%  | 8.8% (borderline) |
| monorepo | 13.8%  | 5.9%     |
| squishy  | 2.3% (TIE) | −7.8% (gz BEATS igzip) |

CLEAN SCAFFOLD (cheap−igzip)/igzip — gz contig driver vs igzip monolith (gz skips CRC):
| corpus   | Intel  | AMD/Zen2 |
|----------|--------|----------|
| silesia  | 7.8%   | 10.8%    |
| nasa     | 1.6% (TIE) | 0.3% (TIE) |
| monorepo | 6.0%   | 9.2%     |
| squishy  | 11.0%  | 15.6%    |

- KERNEL is INTEL-specific & corpus-dependent — large on Intel nasa/monorepo (~14%),
  small/TIE on Intel silesia(7.8)/squishy(TIE); on AMD it is null (silesia TIE),
  modest (monorepo/nasa), or NEGATIVE (squishy −7.8%: gz's own run_contig kernel is
  FASTER than igzip `_04` on Zen2). The arches DISAGREE on the kernel ⇒ kernel-lever
  is NOT cross-arch LAW (re-confirms the prior cycle's de-confounded oracle).
  Mechanism-hypothesis (not gated): Zen2 microcoded PEXT/PDEP (~18cyc) erases igzip
  `_04`'s Intel BMI2 edge.
- CLEAN SCAFFOLD (gz contig driver vs igzip monolith) is the cross-arch-CONSISTENT
  residual: present on BOTH arches, LARGER on AMD, and DOMINANT where the kernel is
  null/negative (silesia, squishy on both; squishy 11–16%). nasa scaffold is ~0 (TIE)
  on both — nasa's entire gap is kernel+CRC. NOTE: scaffold INCLUDES gz's CRC-skip
  advantage subtracted out (igzip computes CRC, cheap does not) ⇒ the TRUE gz-driver
  structural overhead is even larger than these numbers (CRC would widen, not narrow).

CHEAP-HEADER NON-INERT: HEADER-ARTIFACT (bare−cheap)/igzip ≤ 0.5% on ALL 8 cells,
below A/A spread ⇒ the per-block isal_inflate header overhead is empirically negligible;
the cheap variant CONFIRMS (does not correct) the legacy bare numbers on both arches.
