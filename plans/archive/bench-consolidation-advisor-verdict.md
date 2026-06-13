# Disproof-advisor verdict — unified bench freeze + consolidation

Role: independent disproof. Question 1: does the unified freeze ACTUALLY
guarantee a trustworthy quiet-box number (Plex really paused + restored, no
artifact)? Question 2: is any capability lost in consolidation?

## Q1 — trustworthy quiet box: VERDICT = YES (survived disproof)

Three falsification attempts, all run live on neurotic:

1. **Is Plex really paused?** During `bench-lock acquire`, read back
   `lxc/105(plex)/cgroup.freeze = 1` AND its PID-1 systemd in state `Ss`
   (cgroup-suspended, cannot run). transmission(170) also `=1`. NOT a no-op
   freeze — the noisy neighbors are genuinely stopped. The OLD failure (only
   105+111 frozen, transmission/sabnzbd/llama/immich free) is fixed: the freeze
   set is `running − KEEP_ALLOWLIST = 105 108 111 166 170 211 212 509`.

2. **Is the box genuinely quiet (not just freq-pinned)?** The original symptom
   was loadavg 1.6↔11. Disproof of the *gate*: a frozen box reads `procs_running
   = 1` (live runnable) while `loadavg1` still shows ~2.0-3.5 — i.e. loadavg is a
   FALSE not-quiet signal (a 60s EMA carrying pre-freeze load). The gate was
   corrected to average `procs_running`; it then read `quiet runnable_avg=1.00`.
   The validation measurements landed TIGHT (native-T8 spread 3%, T4 6%, ocl_cf-T4
   4%, rg 1-2%) vs the loaded-box 31%/22% — the artifact is gone.

3. **Is restore GUARANTEED even if the caller dies?** Strongest disproof:
   `acquire 12` then NEVER call release (simulate caller death). At t=18 the
   systemd watchdog had auto-restored: `plex.freeze=0 no_turbo=0 frozen=[] wd=
   inactive`. Plex is never left paused. Three independent restore paths exist
   (explicit release, systemd watchdog timer, volatile-state ⇒ reboot restores)
   and `lib_hostlock` release runs from an EXIT/INT/TERM trap.

Residual (HONEST): the *acquire-time* quiet readback can momentarily read
`loaded` (e.g. runnable=2.25/5.0) from a transient — the freeze-application
kworkers, pvestatd's periodic poll, or the agent's own concurrent ssh/scp. A
single re-check (added) resolves it, and the AUTHORITATIVE gate is the guest-side
`procs_running` read at actual benchmark time (which read 1.25-1.5 in every
validation run). So a transient blip warns but does not bank a bad number, and a
genuinely loaded box is HARD-FAILED on the guest side. The one remaining unfrozen
contender is kvm VM 102 (we do not cgroup-freeze VMs — same convention as
host_lock_and_bench.sh); it idled at ~8% and the aperf/mperf gate would catch any
VM-induced freq instability. Acceptable; documented.

## Q2 — capability loss: VERDICT = NONE (with one noted gap, preserved)

Every retired driver maps to a live replacement (scripts/bench/deprecated/
README.md): ceiling/clean-only/engine-isolation(ocl_cf)/same-sink/perturb are all
`oracle.sh --kind`; fulcrum capture is `parity.sh --fulcrum`. The ONE not-yet-a-
kind is `guest_step0` (STALL_RESIDENCY_PROBE + consumer-block decompose) — its
knobs still exist and its driver+analyzer are preserved in deprecated/, flagged
to re-add as `--kind step0` if revisited. No oracle deleted; fulcrum_total
selftest still 25/25.

## Brittleness fixed / noted

- FIXED: `oracle.sh --no-sync` did NOT ship its guest runner ⇒ a stale
  `_oracle_guest.sh` ran (observed: the pre-quiet-gate version aborted on the old
  loadavg text). Now it scp's the runner on `--no-sync` like parity.sh.
- NOTED (not blocking): `orphan-check.sh` reported `GUEST_OFFENDERS` for PIDs that
  were its OWN ephemeral ssh-sweep shells (already gone on re-check; 0 gzippy/rg
  orphans confirmed). A self-detection race — worth excluding the sweep's own
  shell, but it does not affect measurement trust.

## Bottom line

The unified freeze delivers a trustworthy quiet box: Plex (and every noisy
neighbor) is genuinely paused, quiet is verified by the right signal
(procs_running, not lagging loadavg), the number reproduces tight where it was
high-spread, and restore is guaranteed three ways (Plex never left paused). No
capability lost. APPROVED to merge.
