# scripts/bench — the rock-solid clean-bench harness

The ONE trustworthy way to compare gzippy vs rapidgzip wall time for the
campaign. It locks the bench host's frequency, freezes noisy neighbours, proves
the lock held with a hardware gate, runs an interleaved sha-verified benchmark
with full provenance, and **guarantees the host is restored to baseline even on
crash / network drop / SIGKILL**.

## How to run

From the laptop:

```bash
scripts/bench/clean_bench.sh "8"            # T8 only (default)
scripts/bench/clean_bench.sh "4 8 16"       # sweep T4, T8, T16
scripts/bench/clean_bench.sh "8" --lever    # + opt-in causal GZIPPY_SLOW_BOOTSTRAP sweep
scripts/bench/clean_bench.sh "8" --allow-dirty   # measure a dirty tree (stamps DIRTY, RUN_TRUSTWORTHY=false)
```

`clean_bench.sh` ships the harness to the host (`neurotic`), then hands the
entire lifecycle to `host_lock_and_bench.sh` on the host and streams its log.
The laptop side is a dumb viewer — **the restore guarantee does not live on the
laptop.**

Env knobs: `HOST=neurotic`, `HOST_DIR=/root/gzippy-bench`, `WATCHDOG_TTL=2700`.

## Architecture (single host-process lifecycle)

| script | runs on | role |
|---|---|---|
| `clean_bench.sh` | laptop | ship harness, launch host lifecycle, stream log, independent post-run baseline check |
| `host_lock_and_bench.sh` | host | the trap-protected lock owner: resolve guests → write state → **arm watchdog before first mutation** → apply lock → gate → run guest bench → restore + disarm |
| `gzippy_bench_restore.sh` | host | idempotent restore from state file (also the watchdog target) |
| `guest_bench.sh` | guest 199 | provenance gate + interleaved sha-verified benchmark + results table |
| `lib_state.sh` | host | atomic state read/write + baked-in safe defaults |
| `lib_gate.sh` | host | aperf/mperf frequency-stability gate |
| `cstate_hold.c` | host | holds `/dev/cpu_dma_latency=0` (C-state floor) for the run |

## What the gate guarantees

Before ANY timing row is emitted, the gate proves the cores are actually running
at a fixed, locked frequency:

1. The lock applies (all **volatile**, so a host reboot is itself a full
   restore): freeze noisy LXCs, `no_turbo=1`, uncore MSR `0x620=0x1e1e`, C-state
   floor `/dev/cpu_dma_latency=0`, governor=performance, `min==max` per core.
2. `stress-ng --cpu 8` pinned to the P-cores provides a steady load.
3. Per P-core, the gate samples `Δaperf/Δmperf` (MSR 0xE8/0xE7) over ~1s. That
   ratio is the achieved-vs-base frequency multiplier. The base clock comes from
   MSR `0xCE` (PLATFORM_INFO bits 15:8 × 100 MHz = 1400 MHz on this i7-13700T).
4. **Every** P-core must read ratio ∈ [0.985, 1.015] **and** derived MHz within
   ±1.5% of base. A measured PASS looks like all cores at ~1396 MHz (−0.27%).
5. **Gate FAIL ⇒ the box is restored and ONLY a failed-gate record is emitted —
   never timing rows.** Untrustworthy frequency state is never silently turned
   into "results".

## What RUN_TRUSTWORTHY means

`RUN_TRUSTWORTHY=true` is emitted by `guest_bench.sh` only if ALL hold:

- **Provenance clean** — working tree not dirty (gzippy's own tracked files +
  submodule *pointers*; uncommitted edits *inside* a vendor submodule are
  ignored, and the exact submodule commit is recorded). `--allow-dirty`
  overrides the abort but forces `RUN_TRUSTWORTHY=false`.
- **Correctness** — every single run's stdout sha256 matches the `gzip(1)`
  reference decompression of the corpus. Iteration 0 is dropped (warmup); output
  goes to `/dev/null`; these are non-overridable constants, not flags.
- **Stability** — no cell has sd > 5% (a cell over 5% is tagged `FAILED`).
- The gate passed on the host (else no timing rows reach this script at all).

The reported signal is the **interleaved relative ratio** (rapidgzip_time /
gzippy_time): both tools run back-to-back each trial so they see identical
contention, making the ratio jitter-immune. A delta inside the trial spread is a
TIE.

## The restore guarantee (why it is bulletproof)

- **State is written atomically (tmp+fsync+rename) BEFORE any mutation.**
- **The systemd watchdog is armed BEFORE the first mutation** via
  `systemd-run --on-active=<TTL> --unit=gzippy-bench-restore`. This is the
  PRIMARY guarantee: if the laptop ssh dies or the host script is `kill -9`'d
  (so the EXIT trap cannot fire), the watchdog still fires at the TTL and runs
  the idempotent restore. Proven: a `kill -9` of the lock owner left the box
  locked, then the watchdog restored `no_turbo=0`, uncore `0x82b`, all guests
  thawed.
- The `trap … EXIT INT TERM` restore is a fast-path nicety, not the guarantee.
- Restore is **idempotent** (running twice is a no-op) and **verifies
  readbacks**, logging any mismatch.
- Everything mutated is volatile sysfs / MSR / cgroup.freeze — a host reboot is a
  clean slate. Nothing persistent is written.

The keep-allowlist (never frozen): `199` (bench target), `152`/`153` (pihole
DNS), `115`/`116` (wireguard route), `109` (lab-proxy). It is an **allowlist**:
any new guest is frozen by default. A self-access guard aborts if a required
keep guest is not running. VMs (e.g. `102 ha`) are left running (not frozen);
the gate verifies core stability regardless, and a VM-induced gate fail aborts
cleanly with no bad numbers.

## Why fulcrum model / consumer is NOT in standard output

`fulcrum model` / `fulcrum consumer` produce a **telescoping +0.0% identity**
(`L_resolve ≡ publish_span/N` predicts the wall by construction) — a tautology,
not a prediction. They are kept OUT of standard output. Standard output is the
wall table plus the opt-in causal lever test. `GZIPPY_TIMELINE` traces are
archived as artifacts under `/root/gzippy-bench/artifacts/` for **manual**
fulcrum inspection only — never auto-interpreted, and this harness never prints
"+0.0% CONFIRMED" anywhere.

## Self-tests (the proof it is solid)

- `_selftest_readonly.sh` — exercises `base_mhz_from_msr`, state write/read.
  Mutates nothing (writes only `/tmp`).
- `_selftest_watchdog.sh` — applies the REAL lock, `kill -9`'s the owner, proves
  the watchdog restores baseline within TTL. (Use a short `TTL=90` for the test.)

These ship to the host but are not part of a normal bench run.
