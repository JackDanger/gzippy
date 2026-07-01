How testing/profiling runs on the neurotic x86_64 box

1. The machine topology

There are three hosts in the loop, two SSH hops:

MacBook Pro (arm64, where Claude runs)
   │  ssh neurotic
   ▼
neurotic  ── bare-metal Proxmox HOST, root shell
   │         Intel i7-13700T, base clock 1400 MHz, kernel governs all guests
   │  ssh root@10.30.0.199   (one hop, from the host)
   ▼
guest 199 "trainer"  ── LXC container, /root/gzippy lives here

- ssh -J neurotic root@10.30.0.199 = laptop → guest in one command (ProxyJump through the host).
- Why two hosts matter: the LXC guest shares the host kernel. So CPU frequency, uncore ratio, and C-states are controlled from the host (neurotic), while the build/run/bench happens inside the guest (199). The host locks the clocks; the guest
measures on those locked cores.
- The gzippy checkout is /root/gzippy on the guest. Corpus is /root/gzippy/benchmark_data/silesia-large.gz. The clean-bench scripts live in /root/gzippy-bench/ on the host.

I verified live: the guest hostname is trainer, /root/gzippy is present (currently detached at fc5d5d3, a footprint branch).

2. The "clean bench" harness — the one trustworthy A/B

This is a 7-file lifecycle (committed at scripts/bench/ in git history, deployed to /root/gzippy-bench/ on the host). The design rule it encodes: never report a bare absolute MB/s (they swing ~2× with box load and have flipped this project's
conclusions from the same binary), and never emit a timing row unless the clock lock is hardware-proven and every output sha matches.

┌─────────────────────────┬───────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│         Script          │  Runs on  │                                                 Role                                                  │
├─────────────────────────┼───────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ clean_bench.sh          │ laptop    │ dumb viewer: ships harness, launches host lifecycle, streams log, independent post-run baseline check │
├─────────────────────────┼───────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ host_lock_and_bench.sh  │ host      │ trap-protected lock owner                                                                             │
├─────────────────────────┼───────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ lib_state.sh            │ host      │ atomic state capture + safe defaults                                                                  │
├─────────────────────────┼───────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ lib_gate.sh             │ host      │ aperf/mperf frequency-stability gate                                                                  │
├─────────────────────────┼───────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ gzippy_bench_restore.sh │ host      │ idempotent restore (also the watchdog target)                                                         │
├─────────────────────────┼───────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ cstate_hold.c           │ host      │ holds /dev/cpu_dma_latency=0                                                                          │
├─────────────────────────┼───────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ guest_bench.sh          │ guest 199 │ provenance + interleaved sha-verified benchmark                                                       │
└─────────────────────────┴───────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────┘

Invocation (from the laptop)

scripts/bench/clean_bench.sh "8"          # T8 only
scripts/bench/clean_bench.sh "4 8 16"     # sweep
scripts/bench/clean_bench.sh "8" --lever  # + causal GZIPPY_SLOW_BOOTSTRAP sweep
clean_bench.sh scps the 6 host scripts to neurotic:/root/gzippy-bench/, stages guest_bench.sh onward to the guest, then runs ssh neurotic 'bash host_lock_and_bench.sh ...' and streams output. The restore guarantee deliberately does not live on
the laptop — if the laptop's ssh dies, the host watchdog still restores the box.

The host lock lifecycle (host_lock_and_bench.sh), in strict order

1. Resolve guests dynamically — pct list, compute freeze = running_LXCs − KEEP_ALLOWLIST. The allowlist (199 152 153 115 116 109) protects the bench target + DNS + VPN/route, so it never freezes the container it's measuring or its own access
path. Self-access guard aborts if a required guest isn't running. VMs (qm list) are left running.
2. Write the state file BEFORE any mutation (/run/gzippy-bench/state) — captures no_turbo, THP, uncore MSR 0x620, and per-CPU governor/min/max so restore is exact.
3. Arm a systemd-run watchdog timer (gzippy-bench-restore, TTL 2700s) before the first mutation. This is the primary restore guarantee — fires even if the host script is kill -9'd.
4. Apply the lock (all volatile):
  - freeze noisy LXCs via cgroup.freeze
  - no_turbo=1 (intel_pstate)
  - uncore lock: wrmsr -a 0x620 0x1e1e (fixed ratio, all CPUs)
  - C-state floor: compile + launch cstate_hold (writes 0 to /dev/cpu_dma_latency, holds the fd open — a one-shot python -c won't work because the fd must stay open)
  - governor=performance, scaling_min=scaling_max per core (with turbo off, max = base 1.4 GHz)
5. GATE (see below). Fail ⇒ restore + emit only a failed-gate record, never timing rows.
6. Run guest_bench.sh on 199 over one ssh hop.
7. Teardown via trap: restore + disarm watchdog + verify readbacks.

The frequency gate (lib_gate.sh) — why numbers are trustworthy

Before any timing row, it proves the cores are pinned at a fixed clock using the aperf/mperf invariant:
- Pin a steady load: stress-ng --cpu 8 on the P-core mask.
- Per P-core, sample Δaperf/Δmperf (MSR 0xE8/0xE7) over ~1s → the achieved-vs-base frequency multiplier.
- Base clock from MSR 0xCE (PLATFORM_INFO bits 15:8 × 100 MHz = 1400 MHz).
- Every P-core must read ratio ∈ [0.985, 1.015] AND derived MHz within ±1.5% of base (a PASS looks like all cores ~1396 MHz, −0.27%). Any core out of band ⇒ gate FAIL ⇒ box restored, only a failed-gate record emitted.

Restore + watchdog (gzippy_bench_restore.sh, lib_state.sh)

Idempotent restore from the state file (falls back to baked-in i7-13700T defaults: no_turbo=0, uncore=0x82b, governor=performance, min=800000/max=4800000, THP=madvise). Order: thaw guests first → no_turbo → uncore → per-cpu governor/min/max →
THP → kill the C-state holder → verify readbacks and log any mismatch. Two callers: the host script's trap (fast path) and the systemd watchdog (the guarantee). Every locked setting is volatile (sysfs/MSR/cgroup), so a host reboot is itself a
full restore.

The recurring "leave the box CLEAN" discipline in the conversation logs comes from agents dying mid-run and leaving the box locked; the manual force-restore is: stop gzippy-bench-restore.timer, echo 0 to each frozen cgroup.freeze, no_turbo=0,
restore scaling min/max, wrmsr -a 0x620 0x82b, pkill -x cstate_hold, then re-verify no_turbo=0 uncore=0x82b no frozen guests.

3. The benchmark itself (guest_bench.sh, on the guest)

Runs after the host has locked + gated. Enforces provenance-or-no-numbers:

1. Provenance gate — aborts if the working tree is dirty (gzippy's own tracked files + submodule pointers; edits inside vendor submodules like the rapidgzip trace patch are ignored via --ignore-submodules=dirty, but the exact submodule commit
is recorded). --allow-dirty overrides but forces RUN_TRUSTWORTHY=false.
2. Build the production binary: cargo build --release --no-default-features --features pure-rust-inflate (the pure-Rust inflate as the sole decode path).
3. Assert routing: GZIPPY_DEBUG=1 GZIPPY_FORCE_PARALLEL_SM=1 gzippy -d -c -p 8 ... must print path=ParallelSM, else abort. (GZIPPY_FORCE_PARALLEL_SM=1 forces the parallel-SM engine at every thread count.)
4. Reference sha: gzip -dc corpus | sha256sum is the correctness oracle; corpus prewarmed into page cache.
5. Interleaved engine: per thread count, each iteration runs gzippy-baseline, gzippy-ORACLE, then rapidgzip back-to-back (so all three see identical per-trial contention; the ratio is jitter-immune). taskset pins to physical P-cores:
T4=0,2,4,6, T8=0,2,4,6,8,10,12,14 (1 thread/P-core, the "headline cell"), T16=0-15 (SMT/oversubscribed). Iteration 0 is dropped (warmup), output to /dev/null, every run's sha must match the gzip(1) reference — one divergence makes the whole
run untrustworthy.
6. Stats: min/median/sd% per cell; sd > 5% tags the cell FAILED. Verdict WIN/LOSS/TIE is computed against the larger of the two spreads (Δ inside spread = TIE).
7. RUN_TRUSTWORTHY=true only if: provenance clean AND every sha matched AND no cell sd>5% AND the gate passed.
8. rapidgzip is the rapidgzip binary, invoked rapidgzip -d -c -f -P <T>.

Comparisons are always gzippy vs rapidgzip, interleaved, ratio-only. The --lever flag adds a causal GZIPPY_SLOW_BOOTSTRAP sweep (spin vs sleep at 0/50/100% — the frequency-neutral control that distinguishes a real critical-path region from a
turbo-depression artifact).

4. Unit tests / correctness

Run inside the guest over the jump host:
ssh -J neurotic root@10.30.0.199 'cd /root/gzippy && git fetch -q && git reset --hard -q origin/<branch> && cargo test --release ...'
Patterns seen in the logs: full cargo test --release, feature-scoped (--features pure-rust-inflate --lib corpus_silesia), and crate-scoped (cargo test --release -p gzippy-inflate --features route-c-dynasm). Code is synced to the guest either
by git fetch/reset --hard to a pushed branch, or by rsync/scp of individual files for fast iteration. Long-running runs use nohup bash -c '...' and commands are wrapped in timeout (a hard-won rule: a hung multi-line python-via-bash wedged the
whole tool channel once).

5. Fulcrum runs (the hypothesis generator)

Fulcrum is a separate repo (~/www/fulcrum, public JackDanger/fulcrum) — a cross-tool trace analyzer. The flow:

- gzippy side: emits a Chrome-trace JSON via GZIPPY_TIMELINE=/tmp/out.json gzippy -d -c ... (and GZIPPY_LOG_FILE= for the parallel-SM span log). guest_bench.sh auto-saves timeline_base_T{N}.json / timeline_oracle_T{N}.json artifacts (saved,
not auto-interpreted).
- rapidgzip side: patched to emit the same trace vocabulary. scripts/rapidgzip_trace_patch/ (patch_vendor.sh + TraceV2.hpp) instruments the vendor and builds a build-trace variant at vendor/rapidgzip/librapidarchive/build-trace.
- Analysis (on the laptop, offline, re-runnable): fulcrum vs A B (span-by-span busy + wall-critical diff), fulcrum flow (per-stage slack/serial/starved), fulcrum critpath, fulcrum causal, fulcrum decompose/memlife, fulcrum sweep capture/mine.

Governing discipline (from CLAUDE.md and memory): Fulcrum is a hypothesis generator, never the verdict. The verdict is always a causal perturbation (slow-injection or an oracle removal) measured through the clean-bench wall — never
producer-side attribution, never a CPU-sum.

6. Profiling (perf / PMU)

PMU profiling runs inside the guest but depends on the host setting perf_event_paranoid (save+restore). Patterns:
- IPC: taskset -c <mask> perf stat -x, -e instructions,cycles -- <cmd>, median of 5, interleaved gzippy vs rapidgzip (see guest_ceiling_bench.sh's ipc()).
- Page faults / RSS: /usr/bin/time -v parsing "Maximum resident set size" + "Minor (reclaiming) page faults"; or perf stat -e minor-faults.
- Mem-stall discrimination (S3): per-TID perf record/perf stat with PEBS-mem scoped to the frontier-decode spans — CYCLE_ACTIVITY.STALLS_MEM_ANY, MEM_LOAD_RETIRED.*, page-walk cycles, branch-MPKI. Run as a separate pass from the clean wall run
(PMU perturbs timing — join only on ratios/identities, never splice PMU times into wall — the Heisenberg rule).

7. Other harness scripts

- scripts/measure.sh — the portable interleaved-relative comparator (label=command contenders, sha-verified, ratio + TIE/WIN/LOSS, warns on a loaded box). Used when the full clean-bench is overkill.
- scripts/guest_ceiling_bench.sh — footprint-ceiling oracle A/B (baseline vs GZIPPY_FOOTPRINT_CEILING=1 vs rapidgzip), wall + maxRSS + minflt for file & pipe sinks, plus an IPC pass and a T1–T16 sweep.
- scripts/verify_isal.sh — proves the ISA-L static lib is linked + the production parallel-SM path actually executes (GZIPPY_BODY_FAIL_LOG=1), and sha-matches gzip(1).
- scripts/host_freeze_watchdog.sh / freeze_wrapper.sh — lighter standalone freeze-with-watchdog wrappers (hardcoded to freeze LXC 111 + 105, the camera/plex containers) for ad-hoc runs that don't need the full gate.

