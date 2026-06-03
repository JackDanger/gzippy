//! Unified deflate decoder — Phase 1 of the plan at
//! `plans/unified-decoder.md` (sign-off commit `e2ecace`).
//!
//! This file is the *type-system commitment* — three sealed-trait axes
//! (`DecodeMode`, `ArchProfile`, `OutputModel`) and the `Inflate<M, A, O>`
//! struct shape. One monomorphisation, `Inflate<Clean, Generic, Streaming>`,
//! is implemented today by **delegation to [`ResumableInflate2`]**. Phase 2
//! replaces the delegation with an interpreted Rust port of the hot loop
//! (lifting B1-B6 + T0/T3/T4/T5 in place). Phase 3+ adds dynasm JIT,
//! AOT-codegen, the markers/owned monomorphisations, etc.
//!
//! ## Why delegation in commit 1
//!
//! Per the sixth-pass advisor consult, commit 1 is the type-system
//! commitment, not the new hot loop. Wrapping `ResumableInflate2` (which
//! is validated by 638 lib tests + the real-silesia unit test) means
//! commit 1 is provably correct. The traits + builder are the load-bearing
//! deliverable; the inner implementation is replaceable in subsequent
//! commits without touching callers.
//!
//! ## Public surface
//!
//! - `Inflate` — the entry struct; constructed via `Inflate::builder()`
//! - `DecodeMode`, `ArchProfile`, `OutputModel` — sealed traits exposed
//!   only so external code can write `Inflate<M, A, O>` bounds. New
//!   impls land internally as monomorphisations are added.
//! - `Clean`, `Generic`, `Streaming` — the day-one marker types. Other
//!   markers (`Markers`, `X86_64Bmi2`, `AArch64Neon`, `OwnedOutput`)
//!   exist as `pub(crate)` placeholders; promoted to `pub` per the
//!   builder methods that select them.
//!
//! ## Counter for routing-trap test
//!
//! [`UNIFIED_INFLATE_RUNS`] increments on every `read_stream` call that
//! goes through the unified path (not via the legacy env-var fallback).
//! Mirrors the pattern at
//! `src/decompress/parallel/single_member.rs::MARKER_PIPELINE_RUNS`.

#![allow(dead_code)] // future-extensible scaffold; not all axes have callers yet

use std::sync::atomic::{AtomicU64, Ordering};

use super::resumable::{InflateStreamResult, ResumableInflate2};
use super::stopping_point::StoppingPoint;

// =============================================================================
// Sealed-trait scaffolding
// =============================================================================
//
// `Sealed` is a private trait that prevents external crates from impl'ing the
// axis traits. This keeps `Inflate<M: DecodeMode, ...>` extensible by us but
// closed to external mis-implementation — the surface is only ever exercised
// through the builder methods we expose.

mod sealed {
    pub trait Sealed {}
}

// -----------------------------------------------------------------------------
// Axis 1: DecodeMode — Clean (u8 output) vs Markers (u16 output with markers)
// -----------------------------------------------------------------------------

/// Output element type + marker-emission strategy.
///
/// - [`Clean`]: u8 output, no markers; used when the predecessor window is
///   known (parallel-SM Phase 2, BGZF, sequential SM).
/// - [`Markers`] (`pub(crate)` until phase 4 enables it): u16 output, marker
///   emission for cross-chunk back-references; used by parallel-SM
///   speculative decode.
pub trait DecodeMode: sealed::Sealed {
    /// Output element type. `u8` for Clean, `u16` for Markers.
    type Elem: Copy + Default + 'static;
    /// Whether the inner loop emits markers for back-refs past output start.
    const EMITS_MARKERS: bool;
    /// Bytes per output element. 1 for Clean, 2 for Markers.
    const ELEM_BYTES: usize;
}

/// Clean-mode marker: u8 output, no marker emission.
pub struct Clean;
impl sealed::Sealed for Clean {}
impl DecodeMode for Clean {
    type Elem = u8;
    const EMITS_MARKERS: bool = false;
    const ELEM_BYTES: usize = 1;
}

/// Markers-mode marker: u16 output with rapidgzip-shape back-ref markers.
/// `pub(crate)` until phase 4 wires the marker hot loop. Public exposure
/// of `Markers` is the structural commitment to support speculative
/// decode without rebuilding the trait surface later.
pub(crate) struct Markers;
impl sealed::Sealed for Markers {}
impl DecodeMode for Markers {
    type Elem = u16;
    const EMITS_MARKERS: bool = true;
    const ELEM_BYTES: usize = 2;
}

// -----------------------------------------------------------------------------
// Axis 2: ArchProfile — runtime-chosen SIMD / BMI2 dispatch
// -----------------------------------------------------------------------------

/// Architecture profile selected at decoder construction.
///
/// - [`Generic`]: portable scalar fallback (64-bit bitbuf).
/// - `X86_64Bmi2` (placeholder, phase 3): BZHI dispatch + 256-bit shift register.
/// - `AArch64Neon` (placeholder, phase 3): 128-bit NEON shift register.
pub trait ArchProfile: sealed::Sealed {
    const HAS_BMI2: bool;
    const HAS_AVX2: bool;
    const HAS_AVX512: bool;
    const HAS_NEON: bool;
    /// Bit-buffer width in bits (64, 128, 256, or 512).
    const BITBUF_BITS: u32;
}

/// Generic scalar profile — portable 64-bit bitbuf. Day-one default.
pub struct Generic;
impl sealed::Sealed for Generic {}
impl ArchProfile for Generic {
    const HAS_BMI2: bool = false;
    const HAS_AVX2: bool = false;
    const HAS_AVX512: bool = false;
    const HAS_NEON: bool = false;
    const BITBUF_BITS: u32 = 64;
}

/// x86_64 + BMI2 profile — `pub(crate)` placeholder.
pub(crate) struct X86_64Bmi2;
impl sealed::Sealed for X86_64Bmi2 {}
impl ArchProfile for X86_64Bmi2 {
    const HAS_BMI2: bool = true;
    const HAS_AVX2: bool = true;
    const HAS_AVX512: bool = false;
    const HAS_NEON: bool = false;
    const BITBUF_BITS: u32 = 128;
}

/// aarch64 + NEON profile — `pub(crate)` placeholder.
pub(crate) struct AArch64Neon;
impl sealed::Sealed for AArch64Neon {}
impl ArchProfile for AArch64Neon {
    const HAS_BMI2: bool = false;
    const HAS_AVX2: bool = false;
    const HAS_AVX512: bool = false;
    const HAS_NEON: bool = true;
    const BITBUF_BITS: u32 = 128;
}

// -----------------------------------------------------------------------------
// Axis 3: OutputModel — Streaming (resumable, yields on output-full) vs Owned
// -----------------------------------------------------------------------------

/// Caller's output-buffer ownership model.
///
/// - [`Streaming`]: caller passes a `&mut [u8]` per call; decoder yields on
///   fill (today's parallel-SM/BGZF/sequential pattern).
/// - `OwnedOutput` (placeholder, phase 5): decoder owns its output buffer
///   sized via ISIZE-trailer (files) or amortized-growth Vec (chunks).
pub trait OutputModel: sealed::Sealed {
    /// True if the inner loop must check `out_pos >= output.len()`
    /// every iteration (Streaming); false if pre-sized output makes
    /// the check unnecessary (Owned).
    const PER_ITER_YIELD_CHECKS: bool;
}

/// Streaming output — caller passes a buffer per `read_stream` call.
pub struct Streaming;
impl sealed::Sealed for Streaming {}
impl OutputModel for Streaming {
    const PER_ITER_YIELD_CHECKS: bool = true;
}

/// Owned output — `pub(crate)` placeholder for phase 5.
pub(crate) struct OwnedOutput;
impl sealed::Sealed for OwnedOutput {}
impl OutputModel for OwnedOutput {
    const PER_ITER_YIELD_CHECKS: bool = false;
}

// =============================================================================
// Routing counters (Step 2.5 instrumentation pattern)
// =============================================================================

/// Incremented on every `Inflate::read_stream` / `read_stream_starting_at`
/// call (delegates to [`ResumableInflate2`] today).
///
/// Read by routing-trap tests to prove production chunk decode uses the
/// unified inflate surface (mirrors `MARKER_PIPELINE_RUNS`).
pub static UNIFIED_INFLATE_RUNS: AtomicU64 = AtomicU64::new(0);

// =============================================================================
// Inflate<M, A, O> — the day-one type with one wired monomorphisation
// =============================================================================

/// Unified deflate decoder.
///
/// The three type parameters are the axes of the plan's monomorphisation
/// cross product. In Phase 1 only `Inflate<Clean, Generic, Streaming>` is
/// implemented — it delegates to the validated [`ResumableInflate2`]
/// hot loop. Subsequent phases add the other monomorphisations and
/// replace the delegation with a direct port of the hot loop.
pub struct Inflate<'a, M: DecodeMode, A: ArchProfile, O: OutputModel> {
    inner: ResumableInflate2<'a>,
    _mode: std::marker::PhantomData<(M, A, O)>,
}

impl<'a, M: DecodeMode, A: ArchProfile, O: OutputModel> Inflate<'a, M, A, O> {
    /// Bit-position of the next bit the decoder will read.
    pub fn bit_position(&self) -> usize {
        self.inner.bit_position()
    }

    /// True once the deflate stream's BFINAL block has fully decoded.
    pub fn at_end_of_stream(&self) -> bool {
        self.inner.at_end_of_stream()
    }
}

// -----------------------------------------------------------------------------
// Inflate<Clean, Generic, Streaming> — day-one wired implementation
// -----------------------------------------------------------------------------

impl<'a> Inflate<'a, Clean, Generic, Streaming> {
    /// Construct positioned at `bit_offset = 0` over the full input.
    pub fn new(input: &'a [u8], bit_offset: usize) -> std::io::Result<Self> {
        Self::with_until_bits(input, bit_offset, input.len() * 8)
    }

    /// Construct positioned at `bit_offset` with `until_bits` as the
    /// inclusive bit-cap. Matches `ResumableInflate2::with_until_bits`.
    pub fn with_until_bits(
        input: &'a [u8],
        bit_offset: usize,
        until_bits: usize,
    ) -> std::io::Result<Self> {
        let inner = ResumableInflate2::with_until_bits(input, bit_offset, until_bits)?;
        Ok(Self {
            inner,
            _mode: std::marker::PhantomData,
        })
    }

    /// Seed the sliding window (32 KiB max) from a clean predecessor.
    pub fn set_window(&mut self, window: &[u8]) -> std::io::Result<()> {
        self.inner.set_window(window)
    }

    /// Configure which `StoppingPoint`s cause an early return from
    /// `read_stream`. Same semantics as the underlying ResumableInflate2.
    pub fn set_stopping_points(&mut self, points: StoppingPoint) {
        self.inner.set_stopping_points(points)
    }

    /// Decode into `output`, yielding when output fills or a stop point
    /// fires. The unified routing wrapper.
    ///
    /// Phase 1 delegation note: today this counts a call and forwards to
    /// `ResumableInflate2::read_stream`. The kill-switch only changes
    /// which counter increments — the inner call is the same path. In
    /// phase 2 the inner call becomes a direct port; the surface stays
    /// identical.
    pub fn read_stream(&mut self, output: &mut [u8]) -> std::io::Result<InflateStreamResult> {
        UNIFIED_INFLATE_RUNS.fetch_add(1, Ordering::Relaxed);
        self.inner.read_stream(output)
    }

    /// Option A3 entry point — runs decode starting at `output[out_pos_start..]`
    /// with the contract that `output[0..out_pos_start]` already contains the
    /// predecessor's sliding-window image. The fast path in
    /// `copy_match_windowed` then matches every legal back-reference
    /// (max distance 32 KiB ≤ out_pos_start when prefix is 32 KiB),
    /// eliminating the `state.window` slow path entirely.
    ///
    /// Counters use the same UNIFIED_INFLATE_RUNS bucket so production
    /// telemetry continues to work.
    #[cfg(feature = "pure-rust-inflate")]
    pub fn read_stream_starting_at(
        &mut self,
        output: &mut [u8],
        out_pos_start: usize,
    ) -> std::io::Result<InflateStreamResult> {
        UNIFIED_INFLATE_RUNS.fetch_add(1, Ordering::Relaxed);
        self.inner.read_stream_starting_at(output, out_pos_start)
    }

    /// Snapshot of `stopped_at` after the last `read_stream` call.
    pub fn stopped_at(&self) -> StoppingPoint {
        self.inner.stopped_at()
    }

    /// True if the most recent block had `BFINAL=1`.
    pub fn is_final_block(&self) -> bool {
        self.inner.is_final_block()
    }

    /// Clear any pending `set_stopping_points` configuration so the next
    /// `read_stream` decodes through every boundary.
    pub fn clear_stop(&mut self) {
        self.inner.clear_stop()
    }

    /// The set of stopping points currently configured.
    pub fn points_to_stop_at(&self) -> StoppingPoint {
        self.inner.points_to_stop_at()
    }

    /// Block-type of the current block, if known (0=stored, 1=fixed,
    /// 2=dynamic). `None` between blocks.
    pub fn btype(&self) -> Option<u8> {
        self.inner.btype()
    }

    /// Number of compressed bits consumed from the start of the stream.
    pub fn tell_compressed(&self) -> usize {
        self.inner.tell_compressed()
    }

    /// The inclusive bit-cap configured at construction.
    pub fn encoded_until_bits(&self) -> usize {
        self.inner.encoded_until_bits()
    }

    /// Bytes remaining in the input buffer, starting at the byte-aligned
    /// position after `tell_compressed()`. Caller must confirm the
    /// cursor is byte-aligned before calling.
    pub fn remaining_input(&self) -> &'a [u8] {
        self.inner.remaining_input()
    }

    /// Consume `n` bytes from `remaining_input()` — typically used to
    /// advance past a gzip-member trailer the caller has just read.
    pub fn advance_input(&mut self, n: usize) {
        self.inner.advance_input(n)
    }

    /// Reset block / pending state to begin a new gzip stream from the
    /// current bit position (used by multi-member parallel chunk
    /// decoders that reuse the same allocation across members).
    pub fn reset_for_next_stream(&mut self) {
        self.inner.reset_for_next_stream()
    }

    /// True when an inner state-machine session is mid-block. Used by
    /// the production wrapper as a sanity check between yields.
    pub fn session_pending(&self) -> bool {
        self.inner.session_pending()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Type-system commitment: the day-one monomorphisation must compile
    /// and be constructible. If this stops compiling, the trait surface
    /// regressed.
    #[test]
    fn day_one_monomorphisation_compiles() {
        let input = b"\x03\x00"; // shortest valid deflate (empty stored block, BFINAL=1)
        let inflate =
            Inflate::<Clean, Generic, Streaming>::with_until_bits(input, 0, input.len() * 8)
                .expect("construct");
        // After construction, bit_position is at the start.
        assert_eq!(inflate.bit_position(), 0);
    }

    #[test]
    fn unified_inflate_runs_counter_increments() {
        // Build a tiny deflate stream we can decode.
        use std::io::Write;
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(b"hello unified").unwrap();
        let deflate = enc.finish().unwrap();

        let before = UNIFIED_INFLATE_RUNS.load(Ordering::Relaxed);

        let mut inflate =
            Inflate::<Clean, Generic, Streaming>::with_until_bits(&deflate, 0, deflate.len() * 8)
                .unwrap();
        inflate.set_window(&[]).unwrap();
        let mut out = vec![0u8; 64];
        let r = inflate.read_stream(&mut out).expect("decode");

        let after = UNIFIED_INFLATE_RUNS.load(Ordering::Relaxed);

        assert_eq!(&out[..r.bytes_written], b"hello unified");
        assert!(
            after > before,
            "UNIFIED_INFLATE_RUNS did not increment (was {before}, now {after})"
        );
    }

    /// Byte-perfect decode through the unified surface matches the
    /// underlying ResumableInflate2. This is the delegation-correctness
    /// gate: if it fails, the wrapper introduced a regression in the
    /// type or borrow surface even though the inner impl is unchanged.
    #[test]
    fn unified_decode_byte_perfect_vs_resumable() {
        use std::io::Write;
        let payload = b"the quick brown fox jumps over the lazy dog. ".repeat(100);
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(&payload).unwrap();
        let deflate = enc.finish().unwrap();

        // Decode via unified path.
        let mut got = Vec::new();
        let mut inflate =
            Inflate::<Clean, Generic, Streaming>::with_until_bits(&deflate, 0, deflate.len() * 8)
                .unwrap();
        inflate.set_window(&[]).unwrap();
        let mut scratch = vec![0u8; 256];
        loop {
            let r = inflate.read_stream(&mut scratch).expect("read");
            got.extend_from_slice(&scratch[..r.bytes_written]);
            if r.finished || r.bytes_written == 0 {
                break;
            }
        }
        assert_eq!(got, payload);
    }
}
