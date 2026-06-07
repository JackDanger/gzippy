# TIER-APPROACH MANDATE (supervisor → leader, user-set 2026-06-06)

This SUPERSEDES the "faithful but accepted-slow" done-bar for gzippy-native.
User will NOT accept less than a **1.0× TIE** of gzippy-native vs rapidgzip on
parallel single-member decode. Pure-Rust + **inline ASM is permitted**. The
premise "pure-Rust must be slower" is REJECTED.

## The method is TIERED — do NOT hill-climb individual levers

Hill-climbing levers (BMI2, then LUT, then refill, …) yields local optima. The
BMI2-TIE proved the leading "obvious" suspect was a no-op. Instead:

**TIER 1 — DESIGN an overall approach believed to reach 1.0×.**
Synthesize EVERYTHING measured (the uniform ~1.85× d_c/d_w/L_resolve ratio =
single-cause engine substitution; BMI2/ALU already-on TIE; multi-symbol LUT +
lean refill already present; the memory-bound hypothesis; the cache-residency
mandate; the one-engine fold). Produce ONE coherent design for a cache-resident
pure-Rust parallel decoder that can TIE ISA-L — an architecture, not a lever
list. Must address: refill width, copy_match, table layout/sharing across
threads, per-symbol branchiness, and the shared-tiny-hot-working-set mandate
(plans/gzippy-native-design-mandate.md).

**TIER 2 — PROVE the design BEFORE committing to it.**
Build executable tooling that PROVES the design can reach rapidgzip's wall
(~0.46–0.54s target). Acceptable forms: an executable performance/cache model
parameterized by REAL measured constants (per-T MPKI, mem-stall, working-set
size, decode-symbol throughput), a formal proof, OR a standalone toy decoder
benchmarked in isolation. The proof must show the design is bandwidth/latency
feasible — not assert it. Capture the real per-T cache-miss/mem-stall numbers
with a POSITIVE CONTROL first (the advisor caveat: ALU-TIE proves "not ALU", NOT
"is cache" — measure, don't assume).

**TIER 3 — ALIGN gzippy-native to the proven design.**
Only after TIER 2 proves feasibility, implement. Every commit dual-sha byte-exact
(both features == reference 028bd002…cb410f). Verify wall on the locked harness.

## STRUCTURE mandate (user-set, same message)

Make the file/dir/function structure keep us oriented so surprises like "BMI2 was
already on" can't recur:
- Put the two paths in **two subdirectories**: a `gzippy-isal` tree (faithful
  rapidgzip, FFI clean-tail handoff — reference baseline) and a `gzippy-native`
  tree (one pure-Rust cache-resident engine, ISA-L techniques grafted, no C-FFI
  in its decode graph).
- **Remove dead code** (e.g. unified.rs:122 `HAS_BMI2=false` dead placeholder,
  any flagged-off path we won't use). main hosts no dead code.
- Names must describe behavior (continue the naming-truth rename discipline). A
  technique that is ON must be visibly ON in the structure; no dead consts that
  read as live gates.

## 3-WAY FULCRUM (the deliverable that makes the answer concrete)

rapidgzip / gzippy-isal / gzippy-native, locked harness, wall + RSS +
per-thread-working-set + L2/L3 MPKI. Isolates pipeline-overhead vs engine-cost.
gzippy-isal must pass the HARDENED differential vs the FOLDED native driver +
dual-sha BEFORE it is used as the 3-way baseline (plans/isal-parity-gate-mandate.md).

## DISCIPLINES (non-negotiable, hard-won)
- Do minimal work in the supervisor; the LEADER delegates everything to its own
  subagents.
- Serialize ALL builds (scripts/cargo-lock.sh mkdir-mutex) — the Mac OOM'd once.
- Validate every instrument with a positive control before trusting a number.
- Pre-register a falsifier + bound the ceiling (oracle removal) before any
  optimization work-stretch (CLAUDE.md rule 3).
- Reject a lever only with a MECHANISM (rule 7), not a bare TIE.
- Independent advisor corroboration before any consequential/"done" claim.
- Numbers ONLY from the locked Fulcrum harness, never a hand script (rule 8).

## Status
Leader: read plans/orchestrator-status.md for campaign history, then execute
TIER 1 → 2 → 3 + the structure mandate. Report back the TIER-1 design + TIER-2
proof plan for supervisor/advisor review BEFORE TIER-3 implementation.
