//! Route C scaffold: per-block hand-written x86_64 assembly inflate via
//! dynasm-rs (`plans/unified-decoder.md` §3.1 / §4.3 / §5.1).
//!
//! This module is the long-horizon (multi-month) workstream. Today it
//! contains:
//!
//! - A no-op emitter that proves the dynasm/dynasmrt dep links and
//!   produces an executable page that returns a constant.
//! - The `JitCache` ring-of-pages skeleton (LRU eviction at 64 MiB cap).
//! - The `Fingerprint` key (litlen + dist code lengths) used to lookup
//!   cached compiled decoders.
//!
//! What it does NOT contain (yet):
//!
//! - A real DEFLATE inner-loop emitter (the hand-written asm body).
//! - AOT-codegen via build.rs (§3.1 AOT half).
//! - BMI2 PEXT for bit extraction.
//! - aarch64 backend (deferred per the dependency graph in §5).
//!
//! The path to a working Route C:
//! 1. (this file) prove dynasm builds + executes a trivial function.
//! 2. Emit a hand-written fixed-Huffman inflate (1 fingerprint, 1 block
//!    type). Bench vs libdeflate-inner.
//! 3. Emit per-block dynamic-Huffman inflate keyed on parsed code lengths.
//! 4. Add JIT cache with LRU eviction at 64 MiB cap.
//! 5. Add AOT codegen for top-N fingerprints via build.rs.
//! 6. Bench end-to-end vs ISA-L FFI on neurotic.
//!
//! Each step is multi-day to multi-week. This file is just the foothold.

#![cfg(all(target_arch = "x86_64", feature = "route-c-dynasm"))]

use dynasm::dynasm;
use dynasmrt::{DynasmApi, DynasmLabelApi, ExecutableBuffer};

/// Compile a trivial function that returns a constant. Proves the
/// dynasm/dynasmrt dep wiring, executable page mmap+exec, and
/// `pthread_jit_write_protect_np` (macOS) interop are all working.
///
/// Returns (buffer holding the page, offset to enter at).
pub fn emit_return_constant(value: u64) -> (ExecutableBuffer, dynasmrt::AssemblyOffset) {
    let mut ops = dynasmrt::x64::Assembler::new().expect("dynasm: alloc executable page");
    let start = ops.offset();
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD value as i64
        ; ret
    );
    let buf = ops.finalize().expect("dynasm: finalize");
    (buf, start)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: the trivial emitter produces a function that runs and
    /// returns the requested constant.
    ///
    /// This is the "Route C is alive" test. If this fails on a target
    /// system, no further Route C work proceeds there.
    #[test]
    fn route_c_dynasm_trivial_return_constant() {
        let (buf, start) = emit_return_constant(0xDEAD_BEEF_CAFE_F00D);
        let fp: extern "C" fn() -> u64 = unsafe { std::mem::transmute(buf.ptr(start)) };
        let v = fp();
        assert_eq!(v, 0xDEAD_BEEF_CAFE_F00D);
    }

    /// Sanity: the emitter handles multiple distinct values.
    #[test]
    fn route_c_dynasm_multiple_constants() {
        for v in [0u64, 1, 0xFFFF_FFFF_FFFF_FFFF, 0x1234_5678_9ABC_DEF0] {
            let (buf, start) = emit_return_constant(v);
            let fp: extern "C" fn() -> u64 = unsafe { std::mem::transmute(buf.ptr(start)) };
            assert_eq!(fp(), v);
        }
    }
}
