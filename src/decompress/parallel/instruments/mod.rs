//! Campaign measurement instruments — NOT part of the rapidgzip decode pipeline;
//! env-gated, byte-transparent in production.
//!
//! These modules have NO vendor (rapidgzip) counterpart. They exist solely to
//! measure / perturb the production decode path under environment-variable gates
//! and are inert (byte-transparent) when those gates are unset. They are grouped
//! here so the production `parallel/` modules read as a clean structural mirror of
//! rapidgzip. `parallel/mod.rs` re-exports each one at the old
//! `parallel::<name>` path, so every hot-path hook call site is unchanged.
//!
//! | instrument        | what it measures / perturbs                              |
//! |-------------------|----------------------------------------------------------|
//! | `removal_oracle`  | STORE-removal + symbol-stream NODECODE replay ceiling    |
//! | `slow_knob`       | env-gated slow-injection (causal-perturbation pre-gate)  |
//! | `contig_prof`     | contig clean-loop rdtsc class profiler                   |
//! | `trace_jsonl`     | per-event JSONL trace emitter (formerly `trace`)         |
//! | `trace_timeline`  | span/timeline trace emitter (formerly `trace_v2`)        |

pub mod contig_prof;
pub mod removal_oracle;
pub mod slow_knob;
pub mod trace_jsonl;
pub mod trace_timeline;
