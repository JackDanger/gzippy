I have the three scripts in full. I'll attack each contamination class against actual line behavior. No need to read further — the spec is self-contained.

# Disproof report — can parity.sh emit a clean summary line over contamination?

## (a) PIPE sink — **NO (closed in practice), one residual**

`timed()` redirects with `taskset -c "$MASK" "$@" >"$sink"` where `$sink` is `$ARTDIR/sink_gzippy.bin` / `sink_rapidgzip.bin`, `ARTDIR=/dev/shm/...` created by `mkdir -p`. That is a regular file on tmpfs — no pipe, no backpressure. The summary's `gmin` comes from `date +%s.%N` around a write to RAM. The phantom-pipe class is genuinely defeated for the normal case.

**Residual hole:** the script never *asserts* the sink is a regular file. `>"$sink"` happily opens a pre-existing **FIFO** or a **symlink** at that path (the `rm -f` runs *after* the loop, not before, so a planted node from a prior/hostile run survives the first iterations). A symlink→`/dev/null` would be caught by the sha check (empty→mismatch→abort), but a FIFO with a fast draining reader is a writev phantom that `sha256sum "$sink"` would then re-read inconsistently. Low-probability, but the wrapper's whole reason for existing is *structural* impossibility, and there is no `[ -f "$sink" ] && [ ! -L "$sink" ] && [ ! -p "$sink" ]` guard, nor a `rm -f` *before* the loop.

## (b) Unverified / wrong sha — **NO, airtight**

Every recorded iteration (`i>=1`) computes `gsha=$(sha256sum "$sink")` and compares to `REF_SHA`; any mismatch sets `DIVERGED=1`, and after the loop `if [ "$DIVERGED" -ne 0 ]; then fail ...; fi` exits **before** the summary block. The oracle itself is pinned: `REF_SHA=$(gzip -dc "$CORPUS"|sha256sum)` is cross-checked against `CORPUS_RAW_SHA256` and aborts on drift. A non-zero exit in `timed()` only emits a stderr WARN but still returns its sha — which cannot match a full-corpus sha, so a failed/partial run is caught. The warmup `iter0` skips the sha check *but also drops its time* (`continue` before `GZT=...`), so no unverified run ever reaches `gmin`. `sha=OK` in the summary is only printable when `DIVERGED==0`. This class is the strongest part of the design.

## (c) Seeded / non-window-absent run — **YES, reportable**

Section 1 is a **denylist**: it aborts only if one of ~10 named vars (`GZIPPY_SEED_WINDOWS`, `..._CAPTURE`, `..._NO_WINDOWS`, `GZIPPY_*_ORACLE`, `GZIPPY_BYPASS_*`, `GZIPPY_SLEEP_DECODE_NS`, `GZIPPY_SLOW_MODE`) is set. Any seeding/oracle env **not on that list** (a singular `GZIPPY_SEED_WINDOW`, a renamed/new oracle, anything inherited from `/root/.bashrc` or the SSH `AcceptEnv`/`SendEnv` path) flows straight into `env GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN"` and routes to the clean engine. The section-4 assertion only checks `path=ParallelSM` — a seeded run is *still* ParallelSM, so it does not prove window-absence. Result: a contaminated, binder-masked number prints a normal summary line. A denylist can never make this structurally impossible.

**Minimal hardening:** scrub to an **allowlist** — run the measured command under `env -i PATH=... GZIPPY_FORCE_PARALLEL_SM=1 "$GZIPPY_BIN"` (and `GZIPPY_DEBUG=1` only for the assertion), or `unset` every `GZIPPY_*` not in {`FORCE_PARALLEL_SM`,`DEBUG`} right after section 1. Optionally add a positive window-absent assertion (a counter/log line that proves the marker bootstrap ran).

## (d) Stale binary — **YES, reportable**

The whole `guest.env` pin (one `GUEST_SRC`, one `GZIPPY_BIN`) only guarantees build-and-measure agree on *location*, not on *freshness*. `--build` is **optional** (`DO_BUILD=0` default): without it the guest does nothing but `[ -x "$GZIPPY_BIN" ]` and measures whatever binary is sitting there — the header literally says "*then YOU own its freshness*." Edit source → run `parity.sh` without `--build` → summary prints from the previous build. No mechanical guard compares binary mtime to source mtime or binary↔HEAD.

Worse, the provenance line is actively misleading: `do_sync()` rsyncs with `--exclude '.git/'`, so on the guest `git rev-parse HEAD` (in `_parity_guest.sh`) reads either `NA` or a **stale hash from a prior clone** — it does *not* describe the working tree that was just synced. So even a careful reader can't use `head=` to detect the stale-binary case; it's decoupled from the measured bytes.

**Minimal hardening:** when `DO_BUILD=0`, abort unless `$GZIPPY_BIN` is newer than every tracked source file (`find src build.rs Cargo.* -newer "$GZIPPY_BIN" -print -quit` must be empty), or simply require `--build`. Stamp the git SHA into the binary at build (`env!`/`vergen`) and have the guest read it back from `--version` rather than from an absent `.git`.

## (e) Non-interleaved measure — **NO, satisfied (one bias residual)**

The loop body runs gzippy then rapidgzip **inside each iteration** `i`, so they are alternated per trial — exactly the interleave the rule demands; the `RATIO=rmin/gmin` is computed from per-trial-interleaved samples. Structurally there is no batch-all-gzippy-then-all-rapidgzip path. **Residual (not a contamination hole):** the order *within* a trial is fixed (gzippy always first), so a systematic first-mover effect (cold cache / page-cache warming of `$CORPUS` for the second runner) biases consistently. Randomizing the within-trial order would tighten it, but the claim ("alternated per trial") holds.

## (f) Thawed host — **YES, reportable**

Section 5 reads `scaling_governor` and `intel_pstate/no_turbo` and on mismatch only does `echo "## WARN ..."` — it **never aborts**. The summary block (`gzippy=..ms rg=..ms ratio=.. sha=OK verdict=..`) prints unconditionally afterward, absolute ms and a WIN/LOSS verdict included. So a thawed host yields a fully-formed perf line with a one-line warning buried above it.

This is sharpest on the *actual* target: `guest.env` states the box is an **LXC guest**, where `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` and `intel_pstate/no_turbo` are typically not exposed → both readbacks fall to `NA` → `WARN` every run → never a freeze guarantee. `taskset -c "$MASK"` is applied but never read back; affinity is assumed, not verified (offline cores would error→sha-miss→abort, so that sub-case is incidentally covered, but governor/turbo are not). The relative-ratio design *mitigates* frequency drift since interleaving hits both tools, but the spec's claim is "never **report** a number from a contaminated setup" — and it does report one.

**Minimal hardening:** make governor≠`$GOV` or no_turbo≠`$NO_TURBO` a `fail` (not WARN); if readback is `NA` on the LXC, require an explicit `HOST_FROZEN=1` acknowledgment or stamp `verdict=CONTAMINATED` and suppress WIN/LOSS. Read back effective affinity (`taskset -pc $$`) too.

---

## Cross-cutting note
`_parity_guest.sh` uses `set -u` but **not** `set -e` (it toggles `set +e/-e` only around specific calls). Most failures are caught downstream by the sha gate, but it means correctness rests on the explicit `fail`/sha checks, not on the shell aborting — so any *future* guard added without its own explicit check won't be load-bearing. And `sha=OK` is a hardcoded literal in the `printf`, not derived from a variable; it's correct today only because the code path is unreachable when `DIVERGED!=0` — fragile to reordering.

---

VERDICT: **CAN-REPORT-CONTAMINATED=YES**

Required hardening (in priority order):
- **(c)** Replace the seeding **denylist** with an **allowlist**: run the measured command via `env -i PATH=... GZIPPY_FORCE_PARALLEL_SM=1 $GZIPPY_BIN` (or `unset` every `GZIPPY_*` except FORCE_PARALLEL_SM/DEBUG); add a positive window-absent assertion, since `path=ParallelSM` does not prove the binder ran.
- **(d)** Close the stale-binary path: when `DO_BUILD=0`, abort unless `$GZIPPY_BIN` is newer than all tracked sources (or require `--build`); embed the git SHA in the binary and read it back via `--version` (current `git rev-parse HEAD` is meaningless because rsync excludes `.git/`).
- **(f)** Make governor/turbo mismatch (and `NA` readback on the LXC) a hard **`fail`** or force `verdict=CONTAMINATED`; the current WARN lets every thawed/LXC run print a verdict. Also read back effective `taskset` affinity.
- **(a)** Before the loop, `rm -f` the sinks and assert each is a regular, non-symlink, non-FIFO file (`[ ! -e ] || { [ -f ] && [ ! -L ] && [ ! -p ]; }`).
- **(e)** (optional, bias only) randomize within-trial runner order so the first-mover/page-cache effect doesn't always favor the same tool.

Airtight as written: **(b)** sha-verify and **(e)** per-trial interleave.
