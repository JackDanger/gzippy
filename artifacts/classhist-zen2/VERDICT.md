# Zen2 decode-work instruction-surplus: per-class concentration check

Date: 2026-06-29  Arch: AMD EPYC 7282 (Zen2)  Box: root@10.0.2.240
gz commit: reimplement-isa-l @ 10efea80 (--no-default-features --features pure-rust-inflate, RUSTFLAGS=-C target-cpu=native)
gz sha (shipped/stripped):   ce3450043a9e24ba6478c677e330d82a8fb7c34829b2b0420de3a45127677ea0
gz sha (unstripped/subject): 8f35e6aebe0b065055e30f750325878f1a77ecfab12c830ca2395852a0d81b6a
  (unstripped is code-identical; strip removes only .symtab — used so perf/objdump can attribute)
Tool: fulcrum classhist (Linux/x86-64 port), fulcrum branch classhist-linux-port @ 179c102
Comparators: igzip 2.31.0 (/usr/bin/igzip, libisal.so.2.0.31); rapidgzip 0.16.0 native ELF (/root/rgbuild/src/tools/rapidgzip)
Gate-4: GZIPPY_DEBUG=1 -> path=ParallelSM; per-arm sha == gzip -dc (Gate-0(a) PASS both runs).

## instr/B surplus magnitude (fulcrum abmeasure, load-immune, T1)
  logs    vs igzip:      gz 3.5034 instr/B  vs igzip 2.9865  => ratio 1.173 (+17.3%)
  silesia vs rapidgzip:  gz 14.73  instr/B  vs rg    14.17   => ratio 1.039 (+3.9%); gz WINS wall 0.834x cyc/B

## per-class surplus (classhist; Gate-5 WEAK lens = HYPOTHESIS, licenses cut-and-remeasure)
delta(pp) = gz share - comparator share:
  logs vs igzip:     load +5.2  store +1.2  mov +1.5  branch +2.7 | simd -0.8  shift -3.5  logic -2.8  arith -1.9
  silesia vs rg:     load +8.0  store +7.5  mov +5.5  branch +0.9 | simd -7.5  arith -4.3  logic -4.2  shift -3.8

absolute surplus attribution (logs, ratio 1.173): load +43.6%  branch +33.1%  mov +22.3%  store +16.4%  ...

## VERDICT: CONCENTRATED (Zen2 differs from M1 distributed)
The decode-work instruction surplus is CONCENTRATED in the MEMORY-MOVEMENT cluster
(load/store/mov) with a matching SIMD/compute DEFICIT vs ISA-L. 'load' is the single
largest class (44% of the logs surplus; top class on silesia too). Cross-corpus +
cross-comparator consistent on Zen2. Signature: gz moves bytes SCALAR where
ISA-L/rapidgzip move them WIDE (SIMD). Consistent with the prior COPY-FLOOR hypothesis.

LICENSES (as a HYPOTHESIS to cut-and-remeasure, NOT a finding): widen gz's back-ref
copy + table/state loads to SIMD wide loads/stores to match ISA-L; verify via
fulcrum abmeasure/optgate A/B (wall-win-or-no-win). This is a Gate-5 WEAK shape lens,
not a per-class retire count; the A/B is the verdict.
