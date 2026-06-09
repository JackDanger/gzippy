# THREAD-COORDINATION / SERIAL-PROCESSING DEVIATION HUNT — lead-auditor charter (2026-06-09)

## GOVERNING PRINCIPLE (user, 2026-06-09 — your north star)
"Anytime our performance is different, it's a sign that there is a difference in the COMPILED
CODE. It's important to know what that difference IS." A wall gap = a machine-instruction
difference somewhere. For the pipeline, that means: gzippy's coordination/serial code emits
work (syscalls, locks, copies, allocations, wait/wake) that rapidgzip's does not — or differs
in structure. Your job: find those SPECIFIC deviations, and where possible drill to the
compiled-code/syscall difference. A deviation named only at the source level is half-done;
tie it to the runtime work (instructions / syscalls / lock ops) it causes.

## THE HUNCH (user): gzippy-isal's thread coordination + serial processing may DEVIATE from
## the vendor (rapidgzip) pipeline — "just in case we introduced some kind of deviation."
This is the prime suspect for the campaign's LOW-T STRUCTURAL RESIDUAL: even with real ISA-L,
gzippy-isal loses T1 0.899x / T4 0.900x, and the gap is NOT the engine and NOT placement — it
is structural pipeline overhead (serial output + u16 marker bootstrap + chunk-0 + per-chunk).
A faithful-port deviation in coordination/serial processing would explain it.

## SCOPE A — THREAD COORDINATION (the parallel chunk pipeline)
Compare gzippy's `src/decompress/parallel/` (single_member.rs, sm_driver.rs, chunk_fetcher.rs,
gzip_chunk.rs, the thread pool, the prefetcher, the window map/publish) to rapidgzip's
`vendor/rapidgzip/.../` (ParallelGzipReader.hpp, GzipChunkFetcher.hpp, BlockFetcher.hpp, the
thread pool, the window map). Find structural DEVIATIONS, cite vendor file:line vs gzippy:
- chunk PARTITIONING + dispatch to the pool (size, count, how starts are guessed).
- the PREFETCHER / prefetch horizon (how far ahead, admission control, eviction — DIS-6 found
  prior placement fixes dead, but the COORDINATION structure vs vendor is the question here).
- WINDOW PUBLISH handoff (how a finished chunk's 32 KiB window reaches its successor; locks,
  copies, wait/wake) vs vendor setInitialWindow / the window map.
- the CONSUMER in-order collection loop vs vendor ParallelGzipReader.hpp:~600-650.
- any EXTRA locks / condvars / channel ops / thread-wakeups gzippy does that vendor doesn't.

## SCOPE B — SERIAL PROCESSING (the un-parallelizable tail)
Compare the SERIAL parts: the consumer's in-order OUTPUT (writev — note the OverlapWriter was
already found NON-FAITHFUL; verify it's OFF in production), the WINDOW-PUBLISH chain, the CRC
combine, the CHUNK-0 bootstrap, per-chunk FFI setup. Find gzippy-specific serial work rapidgzip
does NOT do (rg's writeFunctor is INLINE-synchronous in the consumer read loop,
ParallelGzipReader.hpp:621 — does gzippy match that, or add a thread/copy/alloc?). Map each of
the campaign's three residual terms (output / marker-bootstrap / chunk-0) to rg's equivalent and
name the divergence.

## DELIVERABLE
plans/pipeline-fidelity-verdict.md: a ranked list of DEVIATIONS (gzippy coordination/serial vs
rapidgzip), each with vendor file:line + gzippy file:line + the runtime work it causes
(syscall/lock/copy/alloc/wakeup) + a faithful-convergence target (delete-divergent / create-
matching). Rank by likely low-T wall impact. If you need parallel sub-agents (you likely CANNOT
spawn them — verify), report the specific fan-out to the SUPERVISOR and I will launch them.

## DISCIPLINES
git WORKTREE. Mostly SOURCE analysis (+ optionally a single trace via the existing fulcrum/log
tooling to confirm a deviation's runtime cost — but do NOT freeze the box / do NOT run wall
benchmarks; a residual-sizing oracle holds the box). Serialize any builds via cargo-lock.sh.
SOURCE-VERIFY first-hand against vendor (the campaign has mis-cited vendor before — read the
actual rapidgzip headers). No multi-line python via Bash. NO orphan processes. Hand ME the
ranked deviation list for my Opus gate; do NOT claim "advisor-vetted." Do NOT touch
src/decompress/parallel/ NAMES (a naming agent is queued) — READ only; propose convergence, do
not implement this turn. Update plans/orchestrator-status.md. STOP at the checkpoint and report.
