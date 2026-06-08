# GUEST.md — the guest-access cheat-sheet (P1 of OWNER-HELP-SUGGESTIONS)

The perf guest is the campaign's **exclusive, single-tenant** measurement
resource. Concurrent runs skew both — never run two perf measurements at once.

All paths/values live in **`scripts/bench/guest.env`** (the single source of
truth). Source it; never re-derive a path inline:

```sh
. scripts/bench/guest.env
```

---

## The double-hop

The guest (`REDACTED_IP`) sits behind the jump host `neurotic`. Two forms exist
in the transcripts; **standardize on the shorter ProxyJump form**:

```sh
# CANONICAL (use this — it is what guest.env's $SSH_GUEST expands to):
ssh -o ConnectTimeout=15 -J neurotic root@REDACTED_IP "…"

# LEGACY nested form (avoid — longer, easy to mis-quote, wedges the channel):
ssh neurotic 'ssh REDACTED_IP "…"'
```

`ConnectTimeout=15` is mandatory: a dead hop must FAIL FAST, never hang the tool
channel (the "no wedged channel" rule). Wrap any remote command that can run long
in `timeout`.

`$SSH_JUMP` (`ssh neurotic`) is for staging files onto the jump host before the
second hop, the way the existing `run_*.sh` drivers do.

---

## ONE checkout root, ONE corpus (the stale-binary trap)

The owner previously used **three** guest roots (`/root/gzippy`,
`/root/gzippy-bench`, `/tmp/gz-ft-src`) and **two** corpus paths concurrently.
That is a silent correctness hazard: a measure can read a **stale binary** from a
root different than the one just rebuilt.

**The pin (guest.env):**

| key          | value                              | role                          |
|--------------|------------------------------------|-------------------------------|
| `GUEST_SRC`  | `/root/gzippy-bench`               | the ONE repo root (build cwd) |
| `GZIPPY_BIN` | `$GUEST_SRC/target/release/gzippy` | the ONE binary we measure     |
| `CORPUS`     | `/root/silesia.gz`                 | the ONE corpus                |

**RULE: rebuild and measure MUST use the same `$GUEST_SRC`.** `parity.sh`
enforces this mechanically (it builds in `$GUEST_SRC` and measures
`$GZIPPY_BIN` — they cannot drift). Do not hand-build in one root and measure in
another.

The other roots are NOT deleted (other agents/banked numbers may reference them).
We declare ONE canonical and have new tooling use only it. Adopting `GUEST_SRC`
on the live guest is the owner's next turn — this doc/config does **not**
forcibly mutate a guest the owner is mid-using.

### Which-build-is-which

| feature flag                  | meaning                                                        |
|-------------------------------|---------------------------------------------------------------|
| `gzippy-native` (the default) | unified pure-Rust decode, **NO C-FFI in the decode graph**, every arch. This is the SOLE production decode path and the parity target. |
| `gzippy-isal`                 | x86_64-only reference baseline: pure-Rust marker engine to a 32 KiB clean window, then HANDS OFF to ISA-L for the clean tail. Comparison only, NOT production. |
| `pure-rust-inflate`           | the legacy feature name; `gzippy-native` is a pure alias of it. A `gzippy-native` build is byte-identical to a `pure-rust-inflate` build. |

`parity.sh --feature gzippy-native` (default) measures production. Use
`--feature gzippy-isal` only for the explicit native-vs-isal comparison.

---

## Discovery: one command, not a retyped one-liner

```sh
scripts/bench/guest-status.sh
```

Prints host / nproc / governor / no_turbo / perf_event_paranoid / mem,
`$GUEST_SRC` HEAD+branch+dirty, **`$GZIPPY_BIN` mtime+sha** (the stale-binary
check), corpus presence+size+sha, and the rapidgzip version — in one double-hop.
Run it at the top of any measurement turn to confirm the box is sane and the
binary is the one you think it is.

---

## Host-lock convention (Rule 6: frozen host)

A trustworthy number requires a frozen host:

- **CPU pin:** `taskset -c $TASKSET` (`0,2,4,6,8,10,12,14` for T8 — even cores,
  no SMT-sibling contention). Both contenders see the same pin so the
  *interleaved* ratio is jitter-immune.
- **Governor:** `scaling_governor == performance` (`$GOV`).
- **Turbo:** `intel_pstate/no_turbo == 1` (`$NO_TURBO`) — neutralizes the
  busy-vs-sleep turbo-deflation artifact (Measurement PROCESS rule 2).

`parity.sh` reads these back and **warns loudly** if the host is not frozen, so a
number is never silently taken on a thawed box. (Setting the governor/no_turbo is
a privileged write done by the host-lock harness on `neurotic`; `parity.sh` only
asserts the state, it does not change it.)

---

## Corpus pin + regeneration (P3)

`CORPUS=/root/silesia.gz` is the ONLY corpus the parity wrapper reads. Its
**decompressed** sha256 (`CORPUS_RAW_SHA256` in guest.env) is the correctness
oracle every measured run is verified against (Rule 4 — a fast wrong-bytes win is
a loss).

```sh
scripts/bench/ensure-corpus.sh        # verify size+sha; regenerate if absent/wrong
```

`parity.sh` calls `ensure-corpus.sh` before measuring, so a fresh guest (or a
`/tmp` copy lost to a reboot) can never silently drift from banked numbers.

---

## Orphan hygiene

Every turn that spawns waits/subagents must leave NO orphans (`claude -p`,
detached `sleep`, stray guest `gzippy`/`rapidgzip`). Use:

```sh
scripts/bench/await.sh <file-or-pid>   # self-cleaning wait (never orphans)
scripts/bench/orphan-check.sh          # one-shot local+guest sweep
```

See OWNER-HELP-SUGGESTIONS.md P4.
