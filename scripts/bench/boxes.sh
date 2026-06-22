#!/usr/bin/env bash
# boxes.sh — THE ONE PLACE that defines a perf box (ssh prefix + paths + freeze).
#
# WHY THIS FILE EXISTS:
#   The campaign runs the SAME rigs (standing.sh, kernel_gate.sh, the kernel-ab
#   harnesses) on TWO arches — Intel (neurotic) for cycle/wall truth and AMD/Zen2
#   (solvency) for the cross-arch LAW gate (BMI2 is microcoded on Zen2, so an
#   Intel-only win is NOT-YET-LAW until replicated on AMD). Rather than scatter the
#   ssh string + corpus paths + comparator paths across every script, ONE box
#   abstraction lives here. A rig sources this with `BOX=<name>` and gets a uniform
#   set of BOX_* variables. When AMD returns, `--box solvency` Just Works.
#
# Usage (inside a bash rig, AFTER sourcing guest.env if you need its other vars):
#   BOX="${BOX:-neurotic}" . scripts/bench/boxes.sh
#   $BOX_SSH "uname -a"                 # ssh prefix is word-split-safe under bash
#   scp $BOX_SCP_JFLAG file "$BOX_GUEST:$BOX_STAGE/"
#
# It is plain POSIX-ish `KEY=value` with one case — no side effects beyond setting
# BOX_* (safe to source repeatedly). It does NOT mutate any box (no governor/turbo
# writes); the FREEZE convention is DOCUMENTED in BOX_FREEZE_NOTE, applied by a
# human/operator before a LAW run (see scripts/bench/AMD-STAGING.md).

BOX="${BOX:-neurotic}"

case "$BOX" in
  neurotic|intel|i7)
    # --- Intel i7-13700T LXC behind the `neurotic` jump host (cycle/wall truth) ---
    BOX_NAME=neurotic
    BOX_ARCH=intel
    BOX_JUMP=neurotic                       # ssh ProxyJump alias (proven working)
    BOX_GUEST=root@10.30.0.199
    BOX_SSH="ssh -o ConnectTimeout=15 -J ${BOX_JUMP} ${BOX_GUEST}"
    BOX_SCP_JFLAG="-J ${BOX_JUMP}"          # scp jump flag
    BOX_SRC=/mnt/internal/gz-head           # git checkout root (rig does fetch+reset)
    BOX_TARGET=/dev/shm/kgate-target        # cargo target (RAM; / is 95% full)
    BOX_STAGE=/root/kernel-gate             # stable scp landing (survives in-tree reset)
    BOX_CORPUS_DIR=/root                    # <corpus>.gz live here
    BOX_RG=/root/oracle_c/rapidgzip-native  # native rapidgzip ELF (T>1 SOTA)
    BOX_IGZIP=/usr/bin/igzip                # ISA-L igzip (T1 SOTA)
    BOX_PINBASE=4                           # measurement P-core (cpu4; sibling excluded)
    BOX_FEATURES=gzippy-native              # cargo --features (pure-Rust, C-FFI off)
    BOX_FLAVOR=parallel-sm+pure             # expected GZIPPY_DEBUG build-flavor
    BOX_BRANCH=kernel-converge-A
    # Freeze: host-set on the Proxmox host; the LXC sees these READ-ONLY.
    BOX_FREEZE_NOTE="intel_pstate/no_turbo=1 + governor=performance (host-set; LXC read-only — assert, do not write)"
    ;;

  solvency|amd|zen2)
    # --- AMD EPYC 7282 Zen2 bare metal (the cross-arch LAW gate) ---
    # OFFLINE as of 2026-06-20. Paths below are BEST-KNOWN from memory
    # (reference_solvency_bench_box) and MUST be confirmed with one read-only ssh
    # the moment the box returns (see scripts/bench/AMD-STAGING.md). Do NOT bank an
    # AMD number until these are verified live.
    # CONFIRMED LIVE 2026-06-22 (T≥2-locate cycle): box reachable as root@10.0.2.240,
    # cargo 1.96 at /root/.cargo/bin (symlinked into /usr/local/bin so the rig's bare
    # `cargo` resolves under non-interactive ssh), rapidgzip-native 0.16.0 ELF built,
    # corpora in /root, gov=ondemand/boost=1 (default, NOT frozen).
    BOX_NAME=solvency
    BOX_ARCH=amd
    BOX_JUMP=""                             # direct — no jump host
    BOX_GUEST=root@10.0.2.240
    BOX_SSH="ssh -o ConnectTimeout=15 ${BOX_GUEST}"
    BOX_SCP_JFLAG=""                        # no jump
    BOX_SRC="${SOLVENCY_SRC:-/root/gz-head}"             # fresh clone, origin=GitHub
    BOX_TARGET="${SOLVENCY_TARGET:-/dev/shm/kgate-target}"
    BOX_STAGE="${SOLVENCY_STAGE:-/root/kernel-gate}"
    BOX_CORPUS_DIR="${SOLVENCY_CORPUS_DIR:-/root}"       # <corpus>.gz live here
    BOX_RG="${SOLVENCY_RG:-/root/gz-base/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip}"
    BOX_IGZIP="${SOLVENCY_IGZIP:-/usr/bin/igzip}"
    BOX_PINBASE="${SOLVENCY_PINBASE:-4}"    # a single physical core (no SMT sibling)
    BOX_FEATURES=gzippy-native
    BOX_FLAVOR=parallel-sm+pure
    BOX_BRANCH=kernel-converge-A
    # AMD freeze is DIFFERENT from Intel: there is NO intel_pstate/no_turbo. Pin
    # the governor to performance and DISABLE boost via cpufreq boost=0.
    BOX_FREEZE_NOTE="governor=performance + /sys/devices/system/cpu/cpufreq/boost=0 (AMD; NOT intel no_turbo). Zen2 BMI2 is MICROCODED — expect different IPC vs Intel."
    ;;

  *)
    echo "boxes.sh: unknown BOX='$BOX' (known: neurotic|solvency)" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
