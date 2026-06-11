# INSTRUMENT REGISTRY — Measurement-Integrity Steward (2026-06-08)

Owner: MEASUREMENT-INTEGRITY STEWARD (standing-specialists.md role 1). One row per
measurement tool. Each row HASH-PINS the script/binary it was validated against and
records a GREP-TARGET-EXISTS check (do the assert/grep strings the tool keys on still
appear in current binary output?). **Hash mismatch OR a missing target string =>
the row reverts to UNVALIDATED** and the tool may not gate a verdict until re-run.

Pins are at branch `reimplement-isa-l` HEAD **d56cb0f5** (charter pin; confirmed via
`git rev-parse`). Script hashes are `shasum -a 256 | cut -c1-16` of the working-tree
file (NOTE: `_oracle_guest.sh` carries the uncommitted grep-bug fix — see row).

Binary identity is the load-bearing build identity, NOT the git hash (the spine
rsyncs `--exclude .git`, so a guest `git rev-parse` reads a stale clone). The gate
binaries are pinned by their bin_sha prefix as reported in each provenance block.

## Legend
- VALIDATED   = self-test passed against the pinned hash AND every grep/assert target
                string verified present in current binary output (source-confirmed).
- UNVALIDATED = hash drifted, a target string is missing, OR never self-tested on a
                quiet box; may be used EXPLORATORY-ONLY, never to bank a verdict.
- Confounds   = known ways the number lies even when the row is VALIDATED.

---

## 1. parity.sh / _parity_guest.sh — PRODUCTION whole-system parity (the spine)
- ROLE: interleaved best-of-N gzippy-vs-rapidgzip, regular-file /dev/shm sink,
  sha-verify every run, path=ParallelSM assert, contamination env-scrub, stale-binary
  content-fingerprint guard, host-freeze + instantaneous-procs_running quiet gate.
- SCRIPT HASH-PIN: parity.sh `139594fd5e5c92cd`, _parity_guest.sh `e0648e23e9db331f`.
- GREP/ASSERT TARGETS (verified present in binary @ d56cb0f5):
  - `path=` + `*ParallelSM*` — EXISTS. Emitter src/decompress/mod.rs:244/290/350
    (`path={:?}`, `DecodePath::ParallelSM` unit variant => Debug renders `ParallelSM`).
    mod.rs hash `05d485fdfc6ed845`.
  - `procs_running` (/proc/stat) — kernel field, host-stable.
  - `instructions`/error/`Finished`/`Compiling gzippy` — cargo + perf, host-stable.
- VALIDATION STATUS: **VALIDATED** for the JOB-1 native-T4 number (this is the
  instrument that produced native-T4 0.740x, kind=same-sink == production parity).
- CONFOUNDS: min-of-N has a downward bias; loaded box inflates absolute + spread
  (mitigated by the procs_running<=2.0 hard-fail); rg-version drift (pin: rg 0.16.0).

## 2. oracle.sh / _oracle_guest.sh — parametrized oracle/perturbation driver
- ROLE: one `--kind` per oracle on the parity spine. KINDS: same-sink (production
  control), ceiling (decode-removed floor, byte-exact), engine-isolation (ISA-L
  engine oracle == ocl_cf), clean-only (seeded, SHA-NOT-CHECKED), perturb (slow-inject).
  Same sync + host-lock + sink + sha policy as parity.sh.
- SCRIPT HASH-PIN: oracle.sh `3f9ea5ec61bdd76c`, _oracle_guest.sh `b0af63846137de2f`
  (carries the uncommitted `isal_oracle_chunks=`→`isal_chunks=` grep-bug fix; the FIX
  is in the working tree, so any build off the synced tree has the corrected runner).
- GREP/ASSERT TARGETS (verified present in binary @ d56cb0f5):
  - `path=`/`ParallelSM` — EXISTS (see row 1).
  - `isal_chunks=` (coverage line, in-script fallback==0 assert, _oracle_guest.sh:230)
    — **EXISTS**. Emitter src/decompress/parallel/chunk_fetcher.rs:870-871
    (`isal_chunks={} isal_fallbacks={}`, under GZIPPY_VERBOSE). chunk_fetcher.rs hash
    `1570867ab63aee2d`. Backing counters ISAL_ENGINE_ORACLE_CHUNKS (gzip_chunk.rs:386,
    incremented on every ISA-L clean-tail success) + ISAL_ENGINE_ORACLE_FALLBACKS
    (gzip_chunk.rs:680, incremented on Ok(false) fallback). The sed parses
    `isal_chunks=\([0-9]*\)` / `isal_fallbacks=\([0-9]*\)` MATCH the printf format.
  - **HISTORICAL BUG (FIXED)**: the prior runner grepped `isal_oracle_chunks=`, a
    label the binary NEVER emitted => the in-script fallback==0 / coverage assert
    silently hard-failed `coverage-unreadable` on EVERY prior engine-isolation attempt.
    Fixed this campaign (status JOB-1). The new target `isal_chunks=` is confirmed
    present; the error message text at :235 STILL says `isal_oracle_chunks=0` and :236
    `isal_oracle_fallbacks=` — those are cosmetic error strings, NOT grep targets
    (the live greps are `isal_chunks=`), so they do not re-introduce the bug. (Tidy-up
    only; non-load-bearing.)
- VALIDATION STATUS: **VALIDATED** for engine-isolation/ocl_cf @ the gate run
  (bin b9eb0a73, isal_chunks=14 fallbacks=0). Self-test surrogate: fallbacks==0 is the
  OFF==identity / no-contamination check (any clean chunk that fell to pure-Rust would
  increment the counter and VOID the ceiling); coverage>0 proves the engine ran.
- CONFOUNDS: ocl_cf is a BLEND not a pure-engine ceiling (advisor-confirmed
  gzip_chunk.rs:128-131,196-223 — only the clean 32KiB-window continuation goes
  through ISA-L FFI; markered prefix + chunk-0 bootstrap stay pure-Rust). So ocl_cf
  contains pure-Rust engine + FFI/handoff cost; the engine share it isolates is an
  UNDERESTIMATE and the non-engine residual derived from it is an UPPER BOUND.
  clean-only is SHA-NOT-CHECKED by design (garbage bytes) — never a parity number.

## 3. host/bench-lock.sh + lib_hostlock.sh — host (neurotic) freeze lifecycle
- ROLE: acquire = freeze ALL running LXCs except an access allowlist (Plex +
  transmission + sabnzbd + llama + immich…), no_turbo=1, uncore lock, governor=perf
  min==max pin, SETTLE, verify-quiet via instantaneous procs_running (NOT 1-min
  loadavg, which is a lagging EMA). watchdog auto-restores after TTL. release = thaw all.
- SCRIPT HASH-PIN: host/bench-lock.sh `fdc001f3214ffedf` (repo mirror; the LIVE copy
  is on neurotic at /root/bench-lock.sh — lib_hostlock.sh scp-syncs the mirror up each
  acquire, so repo<->host drift is closed at acquire time). lib_hostlock.sh sources it.
- GREP/ASSERT TARGETS: `BENCH_LOCK=quiet`/`loaded`/`FAIL` (its own output, host-stable);
  procs_running, pct list, sysfs paths — host/kernel, not binary output. N/A for the
  grep-target-drift class (does not key on gzippy binary strings).
- VALIDATION STATUS: **VALIDATED** in direction — the gate run reported runnable_avg
  1.00-1.50 (<=2.0), the freeze allowlist freezes the noisy-neighbor set the prior
  drift was blamed on. The repo mirror is the source-of-truth ONLY if the scp-sync
  succeeds; a host without write to /root/bench-lock.sh silently uses its existing copy
  => treat the LIVE host copy as authoritative and re-confirm it matches this hash
  before banking an absolute number.
- CONFOUNDS: Plex runs ON the host (outside any LXC freeze) — if a transcode arrives
  mid-run the box reloads; the procs_running gate is the catch (hard-fails > 2.0).
  loadavg in the provenance is a lagging EMA and is context-only, not a gate.

## 4. rss_vs_t.sh / _rss_vs_t_guest.sh — RSS-vs-T + per-thread bytes + perf MPKI
- ROLE: footprint/cache instrument (NOT a wall instrument). RSS-vs-T min-of-5,
  per-thread byte accounting, ballast positive-control, perf MPKI. sha-verify + path
  assert + quiet gate inherited from the spine.
- SCRIPT HASH-PIN: rss_vs_t.sh `06cc69d66d5e3728`, _rss_vs_t_guest.sh `2bed9e44718cbccc`.
- GREP/ASSERT TARGETS: `path=`/`ParallelSM` EXISTS (row 1); perf labels `instructions`,
  `LLC-load-misses`, `cache-misses`, `L1-dcache*` (perf stat output, host-dependent).
- VALIDATION STATUS: **UNVALIDATED for the MPKI label** (footprint counters VALID).
  - **KNOWN GREP-TARGET-DRIFT BUG (open):** _rss_vs_t_guest.sh:229-230 do
    `awk '/instructions/{...print $1...}'` / `awk '/LLC-load-misses/{...print $1...}'`.
    On a hybrid CPU at perf_event_paranoid=4, perf prefixes counters as
    `cpu_core/instructions/` and shifts the value column, so `$1` is the wrong field
    => the printed MPKI is `-nan` (the exact symptom the cache-residency turn logged,
    status:124). The RAW counters in perf_${label}.txt are valid; only the computed
    MPKI label is wrong. This is in the isal_oracle_chunks= class (a parse target that
    drifted out of the expected output shape) but is MEASUREMENT-LABEL-ONLY and did
    NOT touch the JOB-1 gate. Follow-up (measurement-only): make the awk match the
    `cpu_core/<event>/` prefix and pick the numeric column robustly.
- CONFOUNDS: RSS is wall-SLACK (footprint, not wall); the ballast control validates
  the perf instrument moved monotonically before any MPKI is trusted.

## 5. fulcrum_total_capture.sh + scripts/fulcrum_total.py — per-stage decompose
- ROLE: HYPOTHESIS GENERATOR (per CLAUDE.md: fulcrum is never the verdict; the verdict
  is a causal perturbation). Captures a window-absent production trace + GZIPPY_VERBOSE
  counters, decomposes per-stage slack/serial/starved.
- SCRIPT HASH-PIN: fulcrum_total_capture.sh `37e1ec2c136d5a00`, fulcrum_total.py
  `5769feb8d8bf6afc`.
- GREP/ASSERT TARGETS:
  - `path=`/`ParallelSM` (capture:65) — EXISTS (row 1).
  - `window_seeded|flip_to_clean|finished_no_flip|isal_oracle` (capture:94, `|| true`)
    — first THREE **EXIST** (chunk_fetcher.rs:855, GZIPPY_VERBOSE Unified-decoder line);
    `isal_oracle` does **NOT** appear in any runtime output (only in comments / fn
    names `finish_decode_chunk_isal_oracle` / test module names). BENIGN: it is the 4th
    alternative in a `grep -E ... || true` whose first three match, so it is a DEAD
    alternative, NOT a silent hard-fail (unlike the isal_oracle_chunks= class which was
    a sole grep target). Recommend dropping `|isal_oracle` for honesty; non-load-bearing.
  - trace_v2 spans (GZIPPY_TRACE_DETAIL) — emitter src/decompress/parallel/trace_v2.rs.
- VALIDATION STATUS: **VALIDATED as a hypothesis generator** (not a verdict tool).
  Any "lever is X" it suggests is UNCONFIRMED until a causal perturbation + freq-neutral
  control confirms it (Red-team's domain). The fulcrum_total.py self-time / idle-gap /
  busy+idle==span asserts were not re-run this turn — re-run before banking any
  attribution AS a number.
- CONFOUNDS: producer-side attribution is analyst-biasable and has manufactured phantom
  levers before (the combine_crc "62ms serial CRC" nested-span double-count). NEVER
  present a fulcrum attribution AS a verdict (charter rule; Steward refuses to bank it).

## 5b. fulcrum decide — scripts/fulcrum + bench/decide.sh + bench/_decide_guest.sh + fulcrum_decide.py (2026-06-11)
- ROLE: ONE-RUN ranked decision table (plans/fulcrum2-charter.md): per-cell
  interleaved sha-verified walls (canonical masks), trace+contig_prof captures, and
  the in-tree kill-switch knob A/Bs (same-binary, env-only) with effect-verification
  predicates; distribution health (largest-gap bimodality, RESOLVED/UNRESOLVED +
  N-needed); UNFROZEN refusal; per-row re-verify command; DO-THIS-NEXT line.
- GUEST PRIMITIVES: VERBATIM copies of the hash-pinned _parity_guest.sh functions
  (lib_decide_guest.sh, provenance markers per function). ANY edit to the spine's
  primitives must be mirrored there or THIS row reverts to UNVALIDATED.
- GUARD CHANGE (fulcrum_total.py, charter-derived): seeding guard refuses only
  ACTUAL oracle contamination (SEED_WINDOWS replay hits>0; isal_chunks>0 on a
  native build; legacy oracle labels). window_seeded>0 alone = production-seeded
  routing (M3+) and is ACCEPTED. Also fixes the isal_oracle_chunks= pattern bug
  (binary emits isal_chunks=, chunk_fetcher.rs:870).
- GREP/ASSERT TARGETS: `path=`/ParallelSM (row 1); `seeded_block=`/`exact_block=`
  (chunk_fetcher.rs:859 verbose line); `disttbl: builds=`/`reuses=`
  (contig_prof.rs:261-265, knob arm DEAD by design — marker_inflate.rs:2262/2266
  count only the amortized arm); `SEED_WINDOWS replay: hits=` (seed_windows.rs:311).
- VALIDATION STATUS: **VALIDATED 2026-06-11** — fulcrum_total --selftest (all
  retained + new guard matrix) and fulcrum_decide --selftest (33 asserts incl.
  known-null-knob => CAUSAL-NULL) pass; end-to-end run decide_20260611T052041Z
  (frozen, quiet, sha 16/16 cells+knobs) + independent N=21 re-verify
  decide_20260611T053005Z reproduced the tier-1 finding (slab_alloc T1
  -103.2/-99.9ms). Knob A/B deltas are CAUSAL for the knob-covered component only;
  HYPOTHESIS rows remain hypothesis generators (charter rule).
- CONFOUNDS: contig_prof cyc/iter is TSC-rate-bound — NOT comparable across
  clock states (the banked 34.9 cyc/iter vs frozen 65.6 reconciliation; shares ARE
  comparable). Knob deltas at knob-n<21 carry min-of-N bias; the re-verify command
  on each row prescribes N=21.

## 6. The gate falsifier docs (provenance, not an instrument)
- plans/low-t-gate-falsifier.md — pre-registered BEFORE measuring (rule 5); binding
  criteria F-ENGINE-CLOSABLE / F-NON-ENGINE / BLOCKED; measured table matches the
  banked numbers. plans/low-t-gate-advisor-verdict.md — UPHELD-WITH-CAVEATS.
- These are COMMIT-PINNED to HEAD d56cb0f5. If gzip_chunk.rs:128-131,196-223 (the
  blend region the advisor cited) moves, the 0.159/0.101 split reverts to OPEN.

---

## STALENESS TRIGGERS (re-run the self-test before this tool gates anything)
- Any script hash above changes (edit / new commit) => that row => UNVALIDATED.
- A grep/assert target string disappears from binary output (e.g. someone renames
  `isal_chunks=` or `path=`) => that row => UNVALIDATED. (Re-run the source check:
  `grep -rn '<target>' src/` must find a print/eprintln/format emitter, not just a
  comment / fn name / test module — the exact isal_oracle_chunks= failure shape.)
- Host changes / > N days / a different guest root => host-lock + spine rows revert.
- The binary bin_sha differs from the gate's (ocl_cf b9eb0a73 / native 710a6dc) =>
  the gate numbers are not this binary's; re-measure.
