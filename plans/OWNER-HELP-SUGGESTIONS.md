# OWNER-HELP-SUGGESTIONS — accelerate the owner's build/measure/perturb loop

Read-only review of the campaign owner's per-turn transcripts
(`.../c0f5ffa8-.../subagents/agent-*.jsonl`, 100+ spawns) cross-referenced with
`plans/orchestrator-status.md`, `plans/SUPERVISOR-FEEDBACK.md`, and `scripts/bench/`.

**Goal of this doc:** remove the friction the owner re-pays *every turn*. All
artifacts below are ADDITIVE new files (new scripts under `scripts/bench/`, a new
reference doc under `plans/` or `docs/`, a guard wrapper) — none modify code on the
hot path, so an implementer can build them in a worktree without disturbing the live
owner. Ordered by leverage (biggest, most-repeated tax first).

EXCLUDED per supervisor scope: flaky-test fixes (separate agent).

---

## P0 — One canonical guest sync+build+measure-vs-rg wrapper (`scripts/bench/parity.sh`)

**Friction (highest-frequency tax).** Every measurement turn the owner *hand-assembles*
the full chain through the `ssh neurotic 'ssh REDACTED_IP …'` double-hop:
- double-hop appears in **62 of ~100** transcripts;
- source sync is reinvented every turn with *mutually incompatible* mechanisms —
  `tar czf - … | ssh … 'tar xzf -'`, `rsync -az -e "ssh -J neurotic"`,
  `scp /tmp/x.tgz neurotic:` then a second hop, partial single-file `tar czf - src/decompress/parallel/gzip_chunk.rs`. (All present verbatim in the largest transcripts.)
- guest build is re-typed each time, e.g.
  `timeout 600 ssh neurotic 'ssh REDACTED_IP "cd /tmp/gz-ft-src && RUSTFLAGS=\"-C target-cpu=native\" cargo build --release --no-default-features --features pure-rust-inflate …"'`
- the measure invocation (`GZIPPY_FORCE_PARALLEL_SM=1 … -d -c -p 8 silesia.gz`, best-of-N,
  sha-verify, vs `rapidgzip -p 8`) is reassembled inline every time.

**Artifact.** `scripts/bench/parity.sh [--build] [--feature pure-rust-inflate|gzippy-native] [-T 8] [-N 11] [--decompose]`:
1. rsync the working tree (tracked+modified `src build.rs Cargo.toml Cargo.lock benches scripts vendor`, `--exclude target/.git`) to ONE canonical guest checkout (see P1) over the `-J neurotic` jump in a single command;
2. build on guest with the pinned `RUSTFLAGS="-C target-cpu=native"` + the selected feature, under `scripts/cargo-lock.sh`, surfacing only `Compiling gzippy|Finished|error`;
3. host-lock + interleaved best-of-N measure of gzippy (`GZIPPY_FORCE_PARALLEL_SM=1`, regular-file sink, window-absent-preserving) vs `rapidgzip` at the given T, on the canonical corpus (P3);
4. sha-verify both outputs against the gzip oracle and ABORT loudly on mismatch (a fast wrong-bytes win is a loss — Rule 4);
5. with `--decompose`, additionally run `fulcrum_total` capture (`scripts/bench/fulcrum_total_capture.sh` already exists) and print the per-stage breakdown;
6. print one summary line: `gzippy=<ms>  rg=<ms>  ratio=<x>  sha=OK`.

**Why / time saved.** This is the single most-repeated multi-minute ritual. Collapsing
sync+build+measure+verify into one command removes ~5–15 min of hand-typing and the
class of transcription bugs (wrong feature, wrong sink, forgot sha, forgot host-lock)
from *every* measurement turn. It also enforces Rules 4 & 6 mechanically.

**Notes / collision risk.** New file only. Reuse existing `scripts/cargo-lock.sh`,
`scripts/leader-lock.sh`, `scripts/measure.sh`, `scripts/bench/fulcrum_total_capture.sh`
rather than reimplementing. Keep all guest paths sourced from the P1 config file so the
wrapper and the owner never disagree about *where*. Make every remote step `timeout`-wrapped.

---

## P1 — Pin ONE guest checkout location + a guest-access cheat-sheet (`plans/GUEST.md` + `scripts/bench/guest.env`)

**Friction (silent correctness hazard, not just slow).** The owner uses **three**
guest checkout roots concurrently — `/root/gzippy` (60 files), `/root/gzippy-bench`
(32), `/tmp/gz-ft-src` (8) — plus two corpus paths `/root/silesia.gz` (186) and
`/tmp/silesia.gz` (54). This is the "which-build-is-which" trap: a measure can read a
*stale* binary from a different root than the one just rebuilt, silently invalidating a
result. Each turn also re-probes guest state with a bespoke discovery one-liner
(`hostname; nproc; perf_event_paranoid; ls …; which rapidgzip; ls silesia …`) — that
probe alone shows up dozens of times, every variant slightly different.

**Artifact.**
- `scripts/bench/guest.env` — single source of truth: `GUEST=REDACTED_IP`,
  `JUMP=neurotic`, `GUEST_SRC=/root/gzippy-bench/src` (pick ONE, document the choice),
  `CORPUS=/root/silesia.gz`, `TASKSET="0,2,4,6,8,10,12,14"`, `GOV=performance`,
  `RG=$(which rapidgzip)`, `FULCRUM=/tmp/fulcrum`. Sourced by P0 and any future driver.
- `plans/GUEST.md` — the cheat-sheet: the canonical double-hop form
  (`ssh -J neurotic root@REDACTED_IP …` is shorter and already used in newer turns —
  standardize on it over the nested `ssh neurotic 'ssh …'`), the one discovery command
  (`scripts/bench/guest-status.sh` — hostname/nproc/governor/binary-mtime/corpus/rg-version
  in one shot), host-lock convention (taskset + governor), and the rule
  "rebuild and measure MUST use the same `$GUEST_SRC`."

**Why / time saved.** Eliminates the discovery-probe retype every turn (~1–2 min) AND
closes a real correctness gap (stale-binary measurement). A standing config means P0 and
the owner can never disagree about paths.

**Notes / collision risk.** New files only. Do NOT delete the other guest roots (other
runs/agents may reference them) — just declare the canonical one and have new tooling use it.

---

## P2 — Standing env-knob / oracle index (`plans/KNOBS.md`)

**Friction (wrong-premise prevention).** The owner drives a sprawling `GZIPPY_*`
surface — measured distinct knobs across transcripts include
`GZIPPY_FORCE_PARALLEL_SM` (323 uses), `GZIPPY_SEED_WINDOWS`, `GZIPPY_SLOW_MODE`,
`GZIPPY_POISON_RESERVE`, `GZIPPY_FOLD_CONTIG`, `GZIPPY_SLOW_MARKER_MODE`,
`GZIPPY_SLOW_KIND`, `GZIPPY_SLOW_DECODE`, `GZIPPY_ISAL_ENGINE_ORACLE`,
`GZIPPY_PERFECT_OVERLAP`, `GZIPPY_STALL_RESIDENCY_PROBE`, `GZIPPY_SEED_WINDOWS_CAPTURE`,
`GZIPPY_SKIP_WRITEV_SYSCALL`, `GZIPPY_FOLD_NODRAIN`, `GZIPPY_TIMELINE`, `GZIPPY_SLOW_STORE`,
`GZIPPY_DECODE_FREE`, `GZIPPY_WRITEV_CAP_KIB`, `GZIPPY_PREFETCH_CACHE_CAP`, ~25+ total.
Many are one-off oracles. There is no index, so the owner re-discovers/re-derives what
each measures (and which are perturbation knobs per Measurement-PROCESS rule 1 vs which
are oracles per rule 3). A misread knob = a wrong-premise turn.

**Artifact.** `plans/KNOBS.md`: a generated-then-curated table — knob name, where it's
read (`rg -n GZIPPY_X src/`), one line of what it measures, and its CLASS
(*perturbation* = slows a region to test criticality / *oracle* = removes a region to set
a ceiling / *instrument* = trace/counter / *behavior* = changes path). Generate the
skeleton mechanically: `rg -o 'GZIPPY_[A-Z_]+' src/ | sort -u` → stub each row with its
read-site; the implementer fills the one-line semantics from the surrounding code. Add a
top note pointing perturbation knobs at PROCESS rules 1–3 and oracle knobs at rule 3
(remove-and-measure, never extrapolate the slow-down slope).

**Why / time saved.** Prevents the most expensive failure mode (a whole turn spent on a
phantom because a knob was misunderstood) and saves the per-turn `rg` re-derivation.
Self-contained reference; near-zero collision risk.

---

## P3 — Pinned corpus + reproducible regenerate step (in `guest.env` + `plans/GUEST.md`)

**Friction.** silesia path is re-derived every turn (`/root/silesia.gz` 186 ×,
`/tmp/silesia.gz` 54 ×) and there's no recorded "how this corpus was produced" so a
fresh guest (or the `/tmp` copy after a reboot) can drift from the `/root` copy used in
banked numbers.

**Artifact.** Fold `CORPUS=/root/silesia.gz` into `guest.env` (P1) as the ONLY corpus
the parity wrapper reads, plus a `scripts/bench/ensure-corpus.sh` that checks the file's
size+sha against a pinned value and, if absent/mismatched, regenerates it from the
canonical source (`https://jackdanger.com/squishy/` per MEMORY) so any guest is
reproducible. P0 calls `ensure-corpus.sh` before measuring.

**Why / time saved.** Removes the path-retype and guarantees banked numbers are
comparable across turns/guests. Small, additive.

---

## P4 — Orphan-safe advisor-wait + sentinel helper (`scripts/bench/await.sh`) and a single orphan-sweep (`scripts/bench/orphan-check.sh`)

**Friction (recurring self-inflicted).** Two recurring patterns generate orphans the
owner then spends effort hunting:
- the keep-alive sentinel `nohup sleep 86400 >/dev/null 2>&1 &` (present verbatim);
- the advisor-wait spin `until [ -s plans/X-advisor-verdict.md ]; do sleep 3; done`
  and `until … || ! pgrep -f "claude -p --model opus"; do sleep 3; done` — these leave
  detached `sleep`/`claude -p` children when the turn ends.

The cost shows up as a repeated 5-line orphan-check ritual at the top/bottom of turns
(`pgrep -fl "claude -p"; pgrep -fl "sleep [0-9]"; … ; ssh … 'pgrep -fl gzippy …'`) and
the owner closing nearly *every* `orchestrator-status` entry with "NO orphan processes"
(SUPERVISOR-FEEDBACK confirms this is now the owner's standing discipline). It works, but
it's manual tax every turn.

**Artifact.**
- `scripts/bench/await.sh <file-or-pid> [--timeout S]` — a self-cleaning wait: traps
  EXIT/INT/TERM to `pkill -P $$` its own `sleep` children, so it can never orphan; polls
  for a verdict file or a pid exit and returns. Replaces the hand-rolled `until … sleep`
  loops and the `nohup sleep 86400` sentinel.
- `scripts/bench/orphan-check.sh` — the canonical one-shot sweep (local `claude -p` /
  `sleep [0-9]` / `cargo` / `rustc` + guest `gzippy`/`rapidgzip`/`measure.sh` via the
  jump host), printing `LOCAL_CLEAN` / `GUEST_CLEAN` or the offenders. One command
  instead of the re-typed 5-liner.

**Why / time saved.** Removes the orphan-creation source AND the per-turn cleanup retype
(~1 min/turn) while keeping the discipline the supervisor already values. Pure helpers,
no collision.

---

## P5 — Retire the per-turn one-off driver explosion into parametrized drivers

**Friction.** `scripts/bench/` already shows the symptom: many near-duplicate bespoke
drivers were created turn-by-turn — `guest_ceiling.sh`, `guest_clean_only.sh`,
`guest_engine_isolation.sh`, `guest_fulcrum_capture.sh`, `guest_same_sink_floor.sh`,
`guest_step0.sh`, each paired with a laptop `run_*.sh`. Each new perturbation/oracle turn
tends to spawn yet another pair instead of reusing one.

**Artifact.** A single parametrized `scripts/bench/oracle.sh --kind <clean-only|engine-isolation|same-sink|ceiling|stepN> [--slow KNOB=PCT] [-T] [-N]`
that dispatches to the right `GZIPPY_*` knob set (sourced from the P2 index) over the P0
sync+build+measure spine. New oracle = a new `--kind` case, not a new file pair.

**Why / time saved.** Stops the script sprawl, makes oracles consistent with the
parity spine (same host-lock, same sha-verify, same corpus), and means a new
perturbation is a flag, not a 100-line rewrite. Lower priority because the existing
drivers work; this is consolidation, build it after P0–P4 prove the spine.

**Notes / collision risk.** ADD `oracle.sh` alongside the existing drivers; leave the
current ones in place until banked numbers no longer reference them, then the owner can
delete per the "main hosts no dead code" rule.

---

## Implementation order for the implementer agent
1. **P1** (`guest.env` + `GUEST.md` + `guest-status.sh`) — everything else sources it.
2. **P3** (`ensure-corpus.sh`) — small, feeds P0.
3. **P0** (`parity.sh`) — the big win; depends on P1/P3.
4. **P4** (`await.sh` + `orphan-check.sh`) — independent, parallelizable with P0.
5. **P2** (`KNOBS.md`) — independent reference; generate skeleton via `rg`, curate.
6. **P5** (`oracle.sh`) — last, consolidation on top of the proven spine.

All are new files under `scripts/bench/` or `plans/`; none touch `src/`. Safe to build in
a worktree while the owner runs. Each remote command must be `timeout`-wrapped and use
`ssh -J neurotic root@REDACTED_IP` (the shorter, newer form), per the "no wedged channel"
rule. No multi-line python via Bash — any analysis stays in the existing
`scripts/fulcrum_total.py` / `*.py`.
