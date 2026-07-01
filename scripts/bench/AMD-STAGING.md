# AMD / cross-arch staging — run this the moment solvency returns

The kernel front is gated on ONE remaining real verdict: the **AMD/Zen2 STOP-gate**.
BMI2 (`PEXT`/`PDEP` and the `SHRX`/`SHLX`/`BZHI` the kernel leans on) is **microcoded
on Zen2**, so the Intel IPC-shadow that currently absorbs gzippy-native's +3.40 instr/B
over igzip may VANISH on AMD. Until a kernel win replicates on AMD it is **NOT-YET-LAW**
(CLAUDE.md GATE-3). solvency is offline as of 2026-06-20; this file is the staged
one-command runbook so there is zero re-derivation when it returns.

## The box abstraction (single source of truth)

`scripts/bench/boxes.sh` defines every box. A rig is run on a box with `--box <name>`
(or `BOX=<name>`):

| box | arch | ssh | freeze convention |
|-----|------|-----|-------------------|
| `neurotic` | Intel i7-13700T (LXC) | `ssh -J neurotic root@10.30.0.199` | `intel_pstate/no_turbo=1` + `governor=performance` (host-set; LXC read-only) |
| `solvency` | AMD EPYC 7282 Zen2 (bare metal) | `ssh jackdanger@10.0.2.240` | `governor=performance` + cpufreq `boost=0` (**NOT** intel no_turbo) |

All three rigs read it:
- `scripts/bench/standing/standing.sh --box <name>` — ground-truth matrix vs rg+igzip
- `scripts/bench/kernel-ab/kernel_gate.sh --box <name>` — the kernel KEEP/TIE/REVERT gate
- `scripts/bench/kernel-ab/run_contig_objdump.sh --box <name>` — hot-loop disasm

## STEP 0 — when solvency comes back, CONFIRM the paths (one read-only ssh)

The solvency `BOX_*` paths in `boxes.sh` are **best-known from memory
(`reference_solvency_bench_box`) and unverified** (the box was offline when staged).
Do NOT bank an AMD number until these are confirmed live:

```bash
ssh jackdanger@10.0.2.240 '
  uname -srm; nproc
  ls -la *.gz                                    # corpora present? need silesia/monorepo/nasa
  ls -la /home/jackdanger/gz-head/.git 2>/dev/null || echo "NO gz checkout — clone it"
  which igzip objdump nm cargo
  ls -la <rapidgzip-native ELF path?>            # find the native rg ELF
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
  cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo "no cpufreq/boost knob"
'
```

Then edit `boxes.sh` solvency case (or export the `SOLVENCY_*` overrides it honours):
`SOLVENCY_SRC` (git checkout), `SOLVENCY_CORPUS_DIR`, `SOLVENCY_RG` (native rg ELF),
`SOLVENCY_IGZIP`, `SOLVENCY_PINBASE` (a single physical core, no SMT sibling).
If there is no gzippy checkout, clone it first:
`git clone git@github.com:JackDanger/gzippy <SOLVENCY_SRC>` then it auto fetch+resets.

## STEP 1 — FREEZE the AMD box (operator step; rigs do NOT mutate the box)

AMD freeze is **different** from Intel — there is no `intel_pstate/no_turbo`:

```bash
ssh jackdanger@10.0.2.240 '
  sudo cpupower frequency-set -g performance        # or: per-cpu scaling_governor=performance
  echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost   # disable boost (NOT no_turbo)
  # readback to confirm:
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # -> performance
  cat /sys/devices/system/cpu/cpufreq/boost                   # -> 0
'
```

Zen2 boost-off can take a moment to settle; the rigs' GHz-spread gate (GATE0d) will
WARN/untrust a cell if the box did not hold still — re-run if it warns.

## STEP 2 — run the cross-arch LAW matrix (one command each)

```bash
# (a) the kernel STOP-gate: does the latest kernel KEEP on AMD too?
scripts/bench/kernel-ab/kernel_gate.sh --box solvency \
    --sha <candidate> --base <baseline> --corpora "silesia monorepo nasa" --threads 1 -N 21

# (b) where does gzippy-native stand vs rg+igzip on AMD across T?
scripts/bench/standing/standing.sh --box solvency \
    --corpora "silesia monorepo nasa" --threads "1 2 4 8" -N 13
```

A kernel change is **LAW** only when `kernel_gate` returns the SAME verdict
(KEEP on both corpora) on **both** `--box neurotic` AND `--box solvency`. A KEEP on
Intel that becomes TIE/REVERT on AMD is the BMI2-microcode shadow predicted above —
record it as a FALSIFY of the Intel-only KEEP, not a LAW.

## What is ready right now (Intel, NOT-YET-LAW)

- `kernel_gate.sh` self-validated on neurotic (its Gate-0 + the N40 self-test).
- `boxes.sh` solvency case staged (paths flagged for STEP-0 confirmation).
- `nasa` added as the third corpus across standing + kernel_gate + the paired harness.
- AMD owes: STEP-0 path confirm → STEP-1 freeze → STEP-2 matrix. No AMD numbers exist
  yet (the box was offline; none were fabricated).
