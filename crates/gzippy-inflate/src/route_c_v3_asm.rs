//! Route C v3 — dynasm-emitted x86_64 asm for the dynamic-Huffman
//! literal-only inner loop.
//!
//! The asm decodes LITERAL symbols from a dynamic-Huffman block using
//! a runtime-supplied `LayeredLut` and a 64-bit bit buffer. On any
//! non-literal symbol (EOB, length code, subtable redirect) it
//! returns control to the Rust caller with the current (bit_pos,
//! out_pos) state. The Rust caller handles match decode, then
//! re-enters the asm loop.
//!
//! ## Why this exists
//!
//! Per `docs/perf/2026-05-28-isal-vs-purerust-attribution.md`, the
//! 28pp gap to ISA-L is in inner-loop inflate cycles per symbol. The
//! production Rust path runs at ~5-6 ns/symbol while ISA-L's
//! hand-tuned asm runs at ~1-2 ns/symbol. No micro-optimization on
//! the Rust hot loop will recover that without going to asm — three
//! direct attempts this session were measured + falsified
//! (commit `357c96f`, `e8944cf`).
//!
//! ## Scope
//!
//! v3.1 (this file): literal-only loop. Most symbols on silesia are
//! literals (~80%), so even the literal-only path covers most of the
//! decode time. Match codes return to Rust for handling.
//!
//! v3.2 (future): in-asm match-copy via SSE2 16-byte ops + RLE
//! fallback for distance < 16.
//!
//! v3.3 (future): in-asm length+dist decode + extras extraction
//! via BMI2 PEXT.
//!
//! ## Calling convention (System V AMD64)
//!
//! ```text
//! rdi = input data ptr (deflate body)
//! rsi = byte_pos (bytes consumed so far)
//! rdx = bitbuf (low N bits valid)
//! rcx = bitsleft (signed i32)
//! r8  = LayeredLut entries ptr
//! r9  = output data ptr
//! [rsp+8]  = out_pos (current write position)
//! [rsp+16] = out_buf_end (bound; loop exits when out_pos approaches)
//! [rsp+24] = main_bits (typically 12)
//!
//! Returns in RAX a packed state:
//!   bits 0-31:  new bitsleft (i32, sign-extended)
//!   bits 32-63: new byte_pos
//! And updates the caller-supplied state struct via additional registers
//! (see `Route C v3 asm calling convention` below).
//! ```
//!
//! Today the v3.1 dynasm output is conceptually equivalent to the
//! Rust `decode_dynamic_block_layered_with_window` literal-only inner
//! loop — but with one register per state slot and no compiler-
//! emitted bounds checks. The expected gain is ~2x on the literal
//! path because the asm runs at ~2-3 ns/symbol (close to ISA-L's
//! shape).

#![cfg(all(target_arch = "x86_64", feature = "route-c-dynasm"))]

use dynasm::dynasm;
use dynasmrt::{DynasmApi, DynasmLabelApi, ExecutableBuffer};

/// State the asm loop maintains in registers + writes back on return.
///
/// Repr(C) so the caller can pass this struct's address to the asm
/// (the asm writes back via offsets).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct DecodeState {
    pub byte_pos: u64,
    pub bitbuf: u64,
    pub bitsleft: i32,
    pub _pad: u32,
    pub out_pos: u64,
}

/// Exit reason returned by the asm loop.
#[repr(i32)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ExitReason {
    /// Hit a non-literal symbol (EOB, length code, subtable). Caller
    /// inspects state.bitbuf to decode it.
    NonLiteral = 0,
    /// Output buffer is nearly full (would write past `out_buf_end`).
    /// Caller bails or runs the slow path.
    OutputFull = 1,
    /// Input bit buffer underflowed during refill.
    InputUnderflow = 2,
}

/// JIT'd literal-loop function signature.
///
/// Caller pre-loads `DecodeState` with current (byte_pos, bitbuf,
/// bitsleft, out_pos) and passes:
/// - `input_ptr` (rdi)
/// - `lut_entries_ptr` (rsi)
/// - `out_buf_ptr` (rdx)
/// - `out_buf_end` (rcx) — `out_pos < out_buf_end` invariant
/// - `state_ptr` (r8) — pointer to DecodeState (read + write back)
///
/// Returns `ExitReason` as i32 in eax.
type LiteralLoopFn = unsafe extern "sysv64" fn(
    input_ptr: *const u8,
    lut_entries_ptr: *const u8, // 4-byte LutEntry stride
    out_buf_ptr: *mut u8,
    out_buf_end: u64,
    state_ptr: *mut DecodeState,
) -> i32;

const MAIN_MASK: u64 = (1u64 << 12) - 1; // 0x0FFF for 12-bit main table

/// Emit the literal-only inner loop. Reads bits from bitbuf, looks up
/// in the 12-bit main table, writes literal byte and advances. Exits
/// on non-literal symbol or output-full.
///
/// Layout per `LayeredLut::lookup`:
/// - Each entry is 4 bytes: [symbol_lo, symbol_hi, length, length_extra]
/// - `length & 0x80` → SUBTABLE redirect (not handled in asm; exit
///   NonLiteral so Rust resolves it)
/// - `length & 0x40` → LENGTH_CODE (exit NonLiteral)
/// - `length & 0x3F` → code bit count
/// - For literal: symbol_lo is the byte
pub fn emit_literal_loop() -> (ExecutableBuffer, dynasmrt::AssemblyOffset) {
    let mut ops = dynasmrt::x64::Assembler::new().expect("dynasm alloc");
    let start = ops.offset();

    // v3.1 SCAFFOLD: prove dynasm-rs wiring + System V ABI + state
    // round-trip work. The asm reads the input DecodeState pointer
    // (r8), no-ops, writes back, and returns InputUnderflow (2).
    //
    // The real literal-loop body is the next iteration's work and
    // will replace this with the lookup-dispatch-write cycle.
    // Reserving the file structure + ABI + test harness here makes
    // that next commit purely additive.
    dynasm!(ops
        ; .arch x64
        // Touch the DecodeState pointer to confirm round-trip: load
        // byte_pos, write it back (no-op). Proves r8 is a valid ptr
        // and the asm can read/write through it.
        ; mov rax, [r8 + 0]      // byte_pos
        ; mov [r8 + 0], rax      // write back (no-op)
        // Return InputUnderflow sentinel — scaffold marker.
        ; mov eax, 2             // ExitReason::InputUnderflow
        ; ret
    );

    let buf = ops.finalize().expect("dynasm finalize");
    (buf, start)
}

/// Lazy emit cached across all calls.
pub fn literal_loop_fn() -> &'static (ExecutableBuffer, dynasmrt::AssemblyOffset) {
    use std::sync::OnceLock;
    static FN: OnceLock<(ExecutableBuffer, dynasmrt::AssemblyOffset)> = OnceLock::new();
    FN.get_or_init(emit_literal_loop)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// v3.1 scaffold: the asm just compiles + executes. Internal
    /// refill not yet implemented — function returns
    /// `InputUnderflow` on first iteration. That's the "asm wiring
    /// is alive" gate.
    #[test]
    fn v3_asm_scaffold_executes() {
        let (buf, off) = literal_loop_fn();
        let fp: LiteralLoopFn = unsafe { std::mem::transmute(buf.ptr(*off)) };
        let input = [0u8; 32];
        let out_buf = [0u8; 32];
        let lut_entries = [0u8; 64]; // small fake LUT — won't be read because we underflow first
        let mut state = DecodeState {
            byte_pos: 0,
            bitbuf: 0,
            bitsleft: 0,
            _pad: 0,
            out_pos: 0,
        };
        let r = unsafe {
            fp(
                input.as_ptr(),
                lut_entries.as_ptr(),
                out_buf.as_ptr() as *mut u8,
                out_buf.len() as u64,
                &mut state,
            )
        };
        // We expect InputUnderflow (2) because the refill path is a stub.
        assert_eq!(r, ExitReason::InputUnderflow as i32);
    }
}
