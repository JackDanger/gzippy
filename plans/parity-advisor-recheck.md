I read both hardened scripts. Verdicts below cite exact lines in `scripts/bench/_parity_guest.sh`.

## (a) pipe/FIFO sink — **CLOSED**
`assert_regular_sink` (167–174) does `rm -f` (kills a planted FIFO/symlink — `rm` removes the *link/node*, not a target), recreates with `: > "$p"`, then asserts `[ -f ] && [ ! -L ] && [ ! -p ]` (171) for both sinks (173–174). The measured write at 181 is `>"$sink"` (truncate-open of that regular file), so it stays regular across all N iterations. The named class — *planted/inherited FIFO or symlink* — is structurally dead.
Residual (out of the stated model): the assertion is once, pre-loop; a **live** attacker swapping the node to a FIFO between 174 and the 181 writes is a TOCTOU that this doesn't re-check per-iteration. Not the enumerated class.

## (c) denylist misses renamed/inherited vars — **CLOSED**
The scrub at 49–55 is now an **allowlist**: it enumerates *every* `GZIPPY_*` from `env` (50) and `unset`s anything except `GZIPPY_FORCE_PARALLEL_SM`/`GZIPPY_DEBUG` (52–53). A renamed/new/inherited oracle is unset *before* line 193 regardless of whether the 60–63 denylist recognizes it — the denylist only controls the loud *hard-fail*, not the neutralization. So a renamed seeding var is neutralized either way.
Residuals (adjacent, not class-c): the sed regex `GZIPPY_[A-Z_0-9]*` (50) won't match a **lowercase**-named var, and the allowlist namespace is `GZIPPY_*` only — a non-`GZIPPY_`-prefixed oracle isn't scrubbed. Also `GZIPPY_DEBUG` is *kept* (52), so an inherited `GZIPPY_DEBUG=1` survives into the measured process (193) and adds stderr work. None is a `GZIPPY_SEED`-class hole.

## (d) stale binary when --build omitted — **CLOSED for the common case, STILL-OPEN at two edges**
Guard at 90–94: when `DO_BUILD != 1`, `find src build.rs Cargo.toml Cargo.lock -newer "$GZIPPY_BIN"` aborts (exit 6) if any tracked source out-dates the binary. The textbook "edited Rust, forgot `--build`" is caught.
But it is genuinely **STILL-OPEN** two ways:
- **Set is incomplete.** The `-newer` set is only `{src,build.rs,Cargo.toml,Cargo.lock}`. `vendor/` and `benches/` are in `RSYNC_PATHS` (parity.sh:103) and `vendor/` feeds `build.rs` (isal/rapidgzip C) — a vendor-only edit changes the binary but is **not** in the staleness set → stale binary measured silently.
- **Cross-host mtime.** The binary's mtime is the *guest* clock at build; source mtimes are the *owner's*, preserved by `rsync -a`. `-newer` compares across hosts. If the owner clock lags the guest, a fresh edit can timestamp *older* than the guest binary → false-negative → stale binary measured. (Owner-ahead skew only over-aborts, which is safe.)

## (f) thawed host only WARNed — **CLOSED (by default)**
128–134 is now a HARD FAIL (exit 13) when `host_thawed=1`; NA/hidden sysfs also fails closed (NA ≠ expected → thaw → fail). The silent-WARN class is gone on the default path.
Residual: `HOST_FROZEN=1` / `--host-frozen` (parity.sh:60) downgrades to WARN+proceed (130) on **pure operator assertion** — nothing re-verifies the box is actually frozen. On AMD/no-`intel_pstate` hosts `no_turbo` reads NA (121), forcing operators to habitually pass `--host-frozen`, which then also blanket-bypasses the governor check. So the override can re-admit a genuinely thawed host — but it's auditable in the printed provenance (153).

## VERDICT: **CAN-REPORT-CONTAMINATED = YES** (narrowly)

The four enumerated holes are closed *in their stated form*. But a contaminated number is still reachable, all via residuals of (d) and (f):
1. **(d)** a `vendor/`-only change with `--no-sync`/no-`--build` → stale binary, undetected (not in the `-newer` set).
2. **(d)** owner-behind-guest clock skew → fresh source looks older than binary → stale binary, undetected.
3. **(f)** `--host-frozen` on an actually-thawed (not merely unreadable) host → WARN+proceed.

To reach **NO**: add `vendor benches` to the line-91 `find` set (or sha-compare inputs instead of mtime), and make `--host-frozen` require a *positive* freeze readback rather than a bare assertion.
