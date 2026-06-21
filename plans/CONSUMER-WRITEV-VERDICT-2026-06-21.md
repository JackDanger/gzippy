# CONSUMER-OUTPUT WRITEV VERDICT — 2026-06-21 (FROZEN, Intel/neurotic, NOT-YET-LAW)

**VERDICT: STOP.** Consumer-output `writev` is **SLACK** at the silesia-T4 loss cell.
The removal ceiling (arm B) is FLAT; the byte-exact fix (arm C) is FLAT. The
H-KERNEL attribution stands — the front is the pure-Rust kernel, not the consumer
output path. The consumer-output writev direction is **FALSIFIED at HEAD**.

## What ran (the only verdict)
- 3 arms vs rapidgzip-native v0.16.0 (`/root/oracle_c/rapidgzip-native`), silesia
  T4 (real loss cell) AND T8 (the forgotten +27% claim's cell), N=15 interleaved,
  `/dev/null` BOTH arms, frozen guest (no_turbo=1, gov=performance, procs_running=1).
- Subject: gzippy-native @ **c488c448** (kernel-converge-A HEAD), built ON guest,
  build-flavor=parallel-sm+pure, path=ParallelSM. Binary sha 7aa1fc132806.
- Arms: A = HEAD baseline · B = `GZIPPY_SKIP_WRITEV_SYSCALL=1` (removal ceiling) ·
  C = `GZIPPY_OVERLAP_WRITER=1` (real byte-exact fix, dedicated in-order writer thread).
- Rig: `scripts/bench/standing/_consumer_writev_guest.sh` +
  `consumer_writev_report.py`.

## GATE-0 (all PASS, non-inert witnesses fired)
- build-flavor=parallel-sm+pure; built sha == c488c448; path=ParallelSM.
- A byte-exact: sha 028bd002c89c9a90 == zcat ref. ✔
- C byte-exact: sha 028bd002c89c9a90 == zcat ref. ✔
- B NON-INERT: rc=0 (trailer CRC verified → decode RAN) AND output bytes=0 vs
  ref 211968000 (sha e3b0c442… = empty) → writev was FULLY skipped, not inert. ✔
- C NON-INERT: `gzippy-out-writ` writer thread observed engaged (seen=1); baseline
  A control did NOT spawn it (seen=0). ✔
- Self-tests license the ratios: A/A best=0.996 med=0.997; RG/RG best=1.018 med=1.001.

## GATE-1 numbers (best-of-N / median; sd = inter-run spread)
silesia **T4** (the real loss cell):
| arm | best ms | med ms | sd | cyc/B | gz/rg best | gz/rg med |
|-----|--------:|-------:|---:|------:|-----------:|----------:|
| A   | 579.45  | 584.25 |11.1| 9.621 | 1.1438 | 1.1170 |
| B   | 579.66  | 589.65 | 9.6| 9.631 | 1.1442 | 1.1273 |
| C   | 575.95  | 584.30 |10.7| 9.615 | 1.1369 | 1.1171 |
| RG  | 506.61  | 523.05 |15.6| 9.348 | 1.0000 | 1.0000 |

- gap A vs rg = **+72.8ms (+14.4%) best / +61.2ms (+11.7%) med**.
- **B ceiling: A−B = −0.22ms best / −5.40ms med — FLAT** (|Δ| << spread 20.7ms).
  Removing the ENTIRE 211 MB output writev closes **~0%** of the +14.4% gap.
- C fix: A−C = +3.50ms best / −0.05ms med — **FLAT** (closes ~0–5%, within noise).

silesia **T8** (the forgotten +27% cell):
| arm | best ms | med ms | sd | gz/rg best | gz/rg med |
|-----|--------:|-------:|---:|-----------:|----------:|
| A   | 296.04  | 319.94 |16.3| 1.0329 | 1.0664 |
| B   | 298.72  | 313.75 |14.1| 1.0422 | 1.0458 |
| C   | 292.07  | 315.02 |17.1| 1.0190 | 1.0501 |
| RG  | 286.61  | 300.00 | 8.0| 1.0000 | 1.0000 |

- gap A vs rg = **+9.4ms (+3.3%) best / +19.9ms (+6.6%) med** — the **+27% claim is
  NOT reproduced frozen at HEAD; DISCARD it.**
- B ceiling: A−B = −2.68ms best / +6.19ms med — **FLAT** (|Δ| << spread 30.4ms).
- C fix: A−C = +3.97ms best / +4.91ms med — **FLAT** (|Δ| << spread 33.3ms).

**T4 and T8 AGREE: B (removal ceiling) is FLAT at both. No disagreement.**

## FALSIFY entry
- **Premise (falsified):** "the silesia-T4 (and T8) loss vs rapidgzip is the serial
  consumer-output `writev` on the critical path; removing/overlapping it closes a
  meaningful share of the gap."
- **Disproof:** the FULL removal oracle (arm B, 0 bytes written, decode still ran)
  is FLAT at T4 (Δ −0.22ms best / −5.40ms med vs spread 20.7ms) and at T8 — the
  upper bound on the payoff of ANY writev fix is ~0% of the +14.4% T4 gap. The
  byte-exact OVERLAP_WRITER fix (C) is correspondingly FLAT.
- **Scope:** frozen Intel i7-13700T (neurotic LXC), sha c488c448, silesia, T4+T8,
  `/dev/null` sink (the canonical loss-cell sink, same as rg). Single-arch →
  **NOT-YET-LAW; AMD/Zen2 replication owed.**
- **Re-open trigger:** if the TARGET METRIC changes from `/dev/null` throughput to
  **real-file / tmpfs output** (where the 211 MB page-cache copy is non-slack and
  not discarded), re-run this 3-arm rig with a file sink — OVERLAP_WRITER may pay
  THERE. It does not pay for the /dev/null-measured loss cell. Also stale: the
  `output_writer.rs` doc claim "removal moved T8 0.79x→0.98x" is contradicted here
  (HEAD T8 baseline is already ~1.03x best and B is flat) — likely a pre-HEAD
  measurement; do not cite.

## Consequence
The front is the **pure-Rust kernel** (cyc/B 9.62 gz vs 9.35 rg at T4 = ~2.9%
inner-codegen surplus, engine-bound), not the consumer output path. No
OVERLAP_WRITER default-flip is warranted from the loss cell. Step 2 (flip
default-on + full no-regress matrix) is NOT recommended.
