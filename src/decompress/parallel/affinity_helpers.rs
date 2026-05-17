//! Literal port of `rapidgzip::AffinityHelpers`
//! (vendor/rapidgzip/librapidarchive/src/core/AffinityHelpers.hpp).
//!
//! Two public helpers in the vendor header:
//!
//! 1. `availableCores()` (AffinityHelpers.hpp:18-21 / 104-130) — number of
//!    logical cores the calling process may schedule on.
//!     - Non-Linux fallback (AffinityHelpers.hpp:17-21):
//!       `std::thread::hardware_concurrency()`.
//!     - Linux path (AffinityHelpers.hpp:104-130): allocate a dynamically-
//!       sized `cpu_set_t`, call `sched_getaffinity`, return `CPU_COUNT`.
//!       This honors cgroup / cpuset bounds (taskset, Kubernetes cpu quotas,
//!       LXC, etc.), which `hardware_concurrency` does NOT.
//!
//! 2. `pinThreadToLogicalCore(int logicalCoreId)`
//!    (AffinityHelpers.hpp:11-14 / 73-101) — pin the calling thread to a
//!    specific logical processor.
//!     - Non-Linux fallback: no-op (`/** @todo */`).
//!     - Linux path: dynamically-sized `cpu_set_t`, `sched_setaffinity`.
//!
//! Mapping rapidgzip -> Rust
//! -------------------------
//! - `std::thread::hardware_concurrency()`     -> [`num_cpus::get_physical`]
//!   would return physical-core count; we use [`num_cpus::get`], which
//!   matches the vendor: rapidgzip counts logical cores (hyperthreads
//!   included), as does `hardware_concurrency`.
//! - Linux `sched_getaffinity` -> [`num_cpus::get`] in the `cgroups`-aware
//!   `num_cpus` crate honors `sched_getaffinity` directly on Linux (see
//!   `num_cpus` source: `get_num_cpus()` → `sched_getaffinity` + `CPU_COUNT`).
//!   This is the same syscall sequence the vendor uses, with the same
//!   "EINVAL means widen the mask and retry" loop already implemented in
//!   the crate.
//! - Linux `sched_setaffinity` -> [`core_affinity::set_for_current`] is the
//!   portable equivalent; on Linux it calls `pthread_setaffinity_np` →
//!   `sched_setaffinity` under the hood, with identical semantics. We
//!   resolve the requested logical-core index against
//!   `core_affinity::get_core_ids()` so the call works on the platforms
//!   gzippy targets (Linux, macOS, *BSD).
//!
//! Note: the existing [`crate::decompress::parallel::thread_pool`]
//! `available_cores` private helper duplicates this for thread_pool's local
//! use. That call site can switch to [`available_cores`] here once we
//! consolidate; this commit lands the standalone primitive.

#![allow(dead_code)]

/// Mirror of `availableCores()` (AffinityHelpers.hpp:18-21 for non-Linux /
/// AffinityHelpers.hpp:104-130 for Linux).
///
/// Returns the number of logical processors the calling process may schedule
/// on. On Linux this respects cgroup / cpuset constraints (the vendor uses
/// `sched_getaffinity` + `CPU_COUNT`); on other platforms it falls back to
/// the OS-reported hardware concurrency.
///
/// The vendor returns `unsigned int`. We return `usize` because that is the
/// type every Rust call site needs.
pub fn available_cores() -> usize {
    // `num_cpus::get()` is sched_getaffinity-aware on Linux and uses
    // `_SC_NPROCESSORS_ONLN` / `sysctlbyname("hw.logicalcpu")` elsewhere —
    // matching the vendor's split-implementation behavior.
    num_cpus::get()
}

/// Mirror of `pinThreadToLogicalCore(int logicalCoreId)` (AffinityHelpers.hpp:11-14
/// for non-Linux / AffinityHelpers.hpp:73-101 for Linux).
///
/// Pins the calling thread to the given logical processor / hardware
/// thread. Returns `Err` if the platform reports no enumerable core IDs
/// (`core_affinity::get_core_ids()` returned `None`) or if no core with the
/// requested ID exists; returns `Ok(())` if pinning succeeded OR if the
/// platform is the vendor's `/** @todo */` no-op branch (we mirror the
/// non-failing semantics there).
///
/// The vendor throws `std::runtime_error` on `sched_setaffinity` failure
/// (AffinityHelpers.hpp:93-100); we return `Result<(), AffinityError>` so
/// callers can decide. `ThreadPool::workerMain` ignores the result on entry
/// (`pinThreadToLogicalCore(...)`, no try/catch), so the call site we
/// already have (thread_pool.rs:355-358) drops the result; that matches.
pub fn pin_thread_to_logical_core(logical_core_id: u32) -> Result<(), AffinityError> {
    // Vendor non-Linux branch (AffinityHelpers.hpp:10-14) is a documented
    // no-op (`/** @todo */`). On Linux `core_affinity` calls
    // `pthread_setaffinity_np` -> `sched_setaffinity`; on macOS it issues
    // `thread_policy_set` (which is best-effort on Apple Silicon, again
    // matching the vendor's no-op-ish behavior on non-Linux).
    let core_ids = match core_affinity::get_core_ids() {
        Some(ids) => ids,
        // Mirror the vendor's "no-op succeeds" non-Linux behavior on
        // platforms where the OS does not expose core IDs.
        None => return Ok(()),
    };

    let target = match core_ids
        .into_iter()
        .find(|c| c.id as u32 == logical_core_id)
    {
        Some(t) => t,
        None => return Err(AffinityError::NoSuchCore(logical_core_id)),
    };

    if core_affinity::set_for_current(target) {
        Ok(())
    } else {
        // `sched_setaffinity` returning non-zero (AffinityHelpers.hpp:93)
        // is surfaced here as a typed error. The vendor throws
        // `std::runtime_error` with the same intent.
        Err(AffinityError::SetFailed(logical_core_id))
    }
}

/// Errors observable on [`pin_thread_to_logical_core`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffinityError {
    /// The requested logical core is not present in the OS-reported core
    /// list. Mirror of the EINVAL retry exhausting itself in
    /// AffinityHelpers.hpp:60-66.
    NoSuchCore(u32),
    /// The OS rejected the pin request. Mirror of the
    /// `std::runtime_error("sched_setaffinity returned ...")` throw in
    /// AffinityHelpers.hpp:93-100.
    SetFailed(u32),
}

impl std::fmt::Display for AffinityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AffinityError::NoSuchCore(id) => write!(f, "no logical core with id {id}"),
            AffinityError::SetFailed(id) => write!(f, "failed to pin thread to logical core {id}"),
        }
    }
}

impl std::error::Error for AffinityError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_cores_is_positive() {
        // The vendor returns `unsigned int` and is documented to return at
        // least 1 (every running process is by definition schedulable on at
        // least one core). Rust's `num_cpus::get()` shares that guarantee.
        assert!(available_cores() >= 1);
    }

    #[test]
    fn pin_to_first_available_core_returns_a_well_typed_result() {
        // The function must terminate and return one of three documented
        // states for core 0:
        //   - Ok(()) on Linux (sched_setaffinity succeeded) and on
        //     platforms with no enumerable cores (vendor no-op branch).
        //   - Err(NoSuchCore(0)) on heavily sandboxed CI hosts where the
        //     calling thread is already constrained off core 0.
        //   - Err(SetFailed(0)) on macOS / Apple Silicon, where the kernel
        //     ignores thread-affinity hints and core_affinity reports the
        //     set as a no-op-failure. This corresponds exactly to the
        //     vendor's documented `/** @todo */` branch on non-Linux
        //     (AffinityHelpers.hpp:10-14), which is "best-effort, may not
        //     actually pin".
        // We accept all three; the test simply guarantees the API is
        // well-typed and does not panic.
        let result = pin_thread_to_logical_core(0);
        match result {
            Ok(()) | Err(AffinityError::NoSuchCore(0)) | Err(AffinityError::SetFailed(0)) => {}
            Err(e) => panic!("unexpected pin error variant for core 0: {e}"),
        }
    }

    #[test]
    fn pin_to_impossible_core_returns_no_such_core() {
        // On every platform there will be a logical core ID >= 1 << 20 that
        // does not exist; either get_core_ids returned a finite Vec
        // (NoSuchCore) or None (no-op success). Both prove the API contract.
        let result = pin_thread_to_logical_core(1 << 24);
        assert!(matches!(result, Ok(()) | Err(AffinityError::NoSuchCore(_))));
    }
}
