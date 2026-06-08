# BENCH.md — the ONE measurement system (entry point + spine)

The gzippy→rapidgzip measurement system is ONE coherent thing with a single
documented entry point. This file is the map; the companions are
`plans/GUEST.md` (guest access) and `plans/KNOBS.md` (the env-knob/oracle index).

## TL;DR — the two commands

```sh
# 1. PARITY (the production number, gzippy-native vs rapidgzip):
scripts/bench/parity.sh --build -T 8 -N 11            # build + measure, host frozen
scripts/bench/parity.sh --no-sync -T 8 -N 9           # reuse the built binary
scripts/bench/parity.sh --build --fulcrum -T 8        # + window-absent trace for decompose

# 2. ORACLE / PERTURBATION (ceilings & criticality, ONE driver, many --kind):
scripts/bench/oracle.sh --kind engine-isolation -T 4  # "ocl_cf" ISA-L-engine ceiling
scripts/bench/oracle.sh --kind ceiling      -T 8      # decode-removed FLOOR
scripts/bench/oracle.sh --kind same-sink    -T 8      # production byte-exact floor
scripts/bench/oracle.sh --kind clean-only   -T 8      # seeded clean-engine ceiling
scripts/bench/oracle.sh --kind perturb --slow GZIPPY_SLOW_DECODE=50 -T 8
```

Both ALWAYS: freeze the host (pause Plex + noisy neighbors, verify QUIET) →
sync+build in the ONE pinned `$GUEST_SRC` → assert `path=ParallelSM` → host-lock
+ quiet readback → interleaved best-of-N to a REGULAR-FILE sink → sha-verify
EVERY run → thaw the host. A contaminated/loaded/wrong-bytes number is
structurally refused, not merely discouraged.

## The spine (what calls what)

```
parity.sh / oracle.sh           (laptop entry points; arg parse + sync)
  └─ lib_hostlock.sh            (laptop: bracket the run with the host freeze)
       └─ neurotic:/root/bench-lock.sh   (HOST: freeze noisy LXCs, verify quiet,
            │                              arm watchdog, restore — see below)
            └─ reuses /root/gzippy-bench/{lib_state.sh, gzippy_bench_restore.sh,
                                          lib_gate.sh}  (proven lock/restore/gate)
  └─ guest.env                  (THE single source of truth: paths, pins, ssh)
  └─ ensure-corpus.sh           (verify/regenerate the pinned corpus)
  └─ _parity_guest.sh / _oracle_guest.sh   (GUEST: the correctness-critical runner)
  └─ fulcrum_total_capture.sh   (GUEST: trace + counter sidecar, window-absent)
       └─ scripts/fulcrum_total.py        (laptop: the trustworthy decompose; 25/25 selftest)
```

ONE source of truth for each concern:
- **guest access** → `guest.env` (`$SSH_GUEST`, `$GUEST_SRC`, `$CORPUS`, pins)
- **host freeze** → `scripts/bench/host/bench-lock.sh` (deployed to `/root/bench-lock.sh`)
- **sha-verify / regular-file sink / interleaved best-of-N** → the `_*_guest.sh` runners
- **matched comparator** → rapidgzip, interleaved per trial, same pin
- **fulcrum decompose** → `fulcrum_total.py` (+ `fulcrum_total_capture.sh`)
- **knob/oracle index** → `plans/KNOBS.md`

Retired one-off drivers live in `scripts/bench/deprecated/` with a replacement
map — no capability was lost (see that README).

## THE FREEZE (the core fix: a genuinely quiet box)

`scripts/bench/host/bench-lock.sh` (host `/root/bench-lock.sh`) is the single,
standalone host freeze lifecycle. It REPLACES the four overlapping host scripts
(`/root/clock_freeze.sh`, `host_freeze_watchdog.sh`, `freeze_wrapper.sh`,
`freeze_profile.sh`), each of which froze ONLY LXC 105(plex)+111(frigate) and
left transmission(170)/sabnzbd(166)/llama(211)/immich(212) running free — the
loadavg-1.6↔11 bounce that blocked every absolute number.

`bench-lock.sh acquire`:
1. compute freeze set = **running LXCs − KEEP_ALLOWLIST** (allowlist, not
   blocklist — a NEW noisy LXC is frozen by default). Keep = `199 152 153 115
   116 109` (bench-target + DNS + VPN/route + proxy).
2. write baseline state atomically → **arm a systemd watchdog** (force-restore
   after TTL even if the caller dies) → freeze the noisy LXCs → no_turbo=1 →
   uncore lock → governor=performance + min==max.
3. **verify QUIET** via *instantaneous* `procs_running` (/proc/stat) averaged over
   a few samples — NOT the 1-min loadavg, which is a ~60s EMA that still carries
   the pre-freeze neighbor load for a full minute (a FALSE not-quiet signal). A
   quiet frozen box reads runnable≈1; loaded reads higher. One re-check absorbs a
   periodic-daemon (pvestatd) blip.

`bench-lock.sh release` thaws ALL + disarms the watchdog (idempotent). Restore is
guaranteed three ways: explicit release, the systemd watchdog timer, and the fact
that every mutation is volatile sysfs/MSR/cgroup (a reboot is a full restore).
**Plex is never left paused.** `lib_hostlock.sh` runs release from an EXIT trap so
even an aborted measure restores the box.

The guest runners ALSO gate on `procs_running` (HARD-FAIL above the threshold,
`ALLOW_LOAD=1` for a deliberate ratio-only run) — a belt to the host freeze's
suspenders, so a loaded-box ABSOLUTE number can never be banked from either side.

### Knobs (all have safe defaults)

| knob | default | meaning |
|------|---------|---------|
| `--lock` / `--no-lock` | lock ON | bracket with the host freeze (off = external manager owns it) |
| `HOSTLOCK_TTL` | 1800 | watchdog seconds |
| `QUIET_MAX_RUNNABLE` | 2.0 | avg runnable tasks the box must be at/under |
| `QUIET_SAMPLES` | 4 | 1 Hz samples to average |
| `ALLOW_LOAD=1` | off | acknowledge a loaded box (RATIO-only; absolute void) |

## Discipline (Measurement PROCESS, mechanized)

- Rule 4 sha-verify: EVERY run vs the decompressed-corpus pin; any mismatch ABORTS.
- Rule 6 frozen host: freeze + readback (governor/no_turbo) HARD-FAIL on a readable
  thaw; interleaved best-of-N; `path=ParallelSM` asserted before measuring.
- REGULAR-FILE sink, never a pipe (a pipe backpressure-inflated writev → phantom).
- WINDOW-ABSENT-PRESERVING: `parity.sh` refuses any seeding/oracle env (production);
  `oracle.sh` carries exactly ONE `--kind` knob and labels non-production runs.
- The verdict is a CAUSAL PERTURBATION, never attribution — fulcrum is a hypothesis
  generator. `oracle.sh --kind perturb` is the perturbation dial.
