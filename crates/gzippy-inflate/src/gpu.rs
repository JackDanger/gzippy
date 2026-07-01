//! §3.9 — GPU offload for very large DEFLATE inputs.
//!
//! For inputs ≥ 1 GiB compressed AND ≥ 1000 estimated blocks, dispatch
//! per-block decode to a GPU compute kernel. CPU does the block-finder
//! scan (~10 GB/s bandwidth-bound) and submits block ranges to the
//! GPU for parallel inflate.
//!
//! ## Target throughput
//!
//! Per `plans/unified-decoder.md` §3.9: realistic peak ~20 GB/s on
//! M3 Max. Done-when target: ≥ 15 GB/s.
//!
//! ## Backends
//!
//! - **Metal** (macOS) — via `metal` crate.
//! - **CUDA** (Linux + NVIDIA) — via `cust` crate.
//! - **Vulkan compute** (portable) — via `vulkano` or `wgpu`.
//!
//! Each behind its own feature flag (`gpu-metal`, `gpu-cuda`,
//! `gpu-vulkan`). v0.1 ships the API surface; concrete backends land
//! in v0.2+. Per plan §11 effort estimate: ~2 person-months per
//! backend.
//!
//! ## v0.1 scope
//!
//! Scaffold only:
//! - `GpuInflate` struct with `available_backends()` query.
//! - `GpuBackend` enum.
//! - `dispatch_blocks(blocks: &[BlockRange]) -> ...` API stub
//!   returning `Unavailable` on all targets until backends land.
//!
//! Production code that wants GPU offload calls
//! `GpuInflate::dispatch_blocks` and gracefully falls back to CPU
//! decode if `Unavailable` is returned.

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Supported GPU backends. Used to query at runtime which backends
/// are available on the host system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// Apple Metal (macOS / iPadOS).
    Metal,
    /// NVIDIA CUDA (Linux / Windows).
    Cuda,
    /// Khronos Vulkan compute (portable).
    Vulkan,
}

/// Per-block decode dispatch unit.
///
/// The CPU's block-finder scan produces a stream of these; each is
/// handed to the GPU kernel which decodes one block in parallel
/// with the others.
#[derive(Debug, Clone, Copy)]
pub struct BlockRange {
    /// Bit offset into the compressed stream where this block's
    /// header starts.
    pub start_bit: u64,
    /// Bit offset where the block ends (exclusive).
    pub end_bit: u64,
    /// Expected uncompressed byte size (estimated from prior block
    /// avg, or computed exactly post-decode). Used to pre-allocate
    /// per-block output GPU buffers.
    pub estimated_output_size: u32,
}

/// GPU inflate dispatcher.
pub struct GpuInflate {
    #[allow(dead_code)] // set by v0.2 when real backends land
    backend: Option<GpuBackend>,
}

impl Default for GpuInflate {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuInflate {
    /// Probe the system for available GPU backends. v0.1: always
    /// returns `None` (no backends compiled in).
    pub fn new() -> Self {
        Self { backend: None }
    }

    /// Which backends are compiled into THIS BUILD. v0.1: always
    /// empty Vec because no feature is wired up yet.
    pub fn available_backends() -> Vec<GpuBackend> {
        #[allow(unused_mut)]
        let mut v = Vec::new();
        #[cfg(feature = "gpu-metal")]
        v.push(GpuBackend::Metal);
        #[cfg(feature = "gpu-cuda")]
        v.push(GpuBackend::Cuda);
        #[cfg(feature = "gpu-vulkan")]
        v.push(GpuBackend::Vulkan);
        v
    }

    /// Dispatch a set of blocks to the GPU for parallel decode.
    /// v0.1: always returns `GpuError::Unavailable`. Callers fall
    /// back to CPU decode.
    ///
    /// Future API: returns a future or stream that yields
    /// `(BlockRange, Vec<u8>)` pairs as each block's decode
    /// completes on the GPU.
    pub fn dispatch_blocks(
        &self,
        _input: &[u8],
        _blocks: &[BlockRange],
    ) -> Result<Vec<Vec<u8>>, GpuError> {
        Err(GpuError::Unavailable)
    }

    /// Returns true if GPU offload is worth attempting for the
    /// given input. Per plan §3.9: ≥ 1 GiB compressed AND
    /// ≥ 1000 estimated blocks.
    pub fn should_offload(compressed_size: usize, estimated_blocks: usize) -> bool {
        compressed_size >= 1024 * 1024 * 1024 && estimated_blocks >= 1000
    }
}

/// GPU dispatch failure modes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuError {
    /// No GPU backend compiled in or no compatible device found.
    /// Caller falls back to CPU decode.
    Unavailable,
    /// Kernel launched but failed (driver error, OOM, etc).
    DispatchFailed,
    /// Decoded output failed correctness check on host.
    CorrectnessFailure,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_v0_1_always_unavailable() {
        let gpu = GpuInflate::new();
        let r = gpu.dispatch_blocks(&[], &[]);
        assert_eq!(r.unwrap_err(), GpuError::Unavailable);
    }

    #[test]
    fn should_offload_thresholds() {
        // Below 1 GiB → no offload regardless of block count.
        assert!(!GpuInflate::should_offload(500 * 1024 * 1024, 10_000));
        // Above 1 GiB but few blocks → no offload.
        assert!(!GpuInflate::should_offload(2 * 1024 * 1024 * 1024, 100));
        // Both thresholds met → offload.
        assert!(GpuInflate::should_offload(2 * 1024 * 1024 * 1024, 5_000));
    }

    #[test]
    fn available_backends_v0_1_empty() {
        // No GPU features compiled in by default.
        let backends = GpuInflate::available_backends();
        assert!(backends.is_empty());
    }
}
