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

## AMD — solvency (EPYC 7282 Zen2), FROZEN gov=performance boost=0, pin cpu4, best-of-15 [PENDING]
Zen2 PEXT/PDEP microcoded (~18cyc): both gz run_contig and igzip `_04` use BMI2 there
(kernel-edge mechanism differs from Intel — factor in).

(cells land below as measured)
