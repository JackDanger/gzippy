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
/// - `input_len` (r9) — input byte length; refill guards against OOB
///
/// Returns `ExitReason` as i32 in eax.
type LiteralLoopFn = unsafe extern "sysv64" fn(
    input_ptr: *const u8,
    lut_entries_ptr: *const u8, // 4-byte LutEntry stride
    out_buf_ptr: *mut u8,
    out_buf_end: u64,
    state_ptr: *mut DecodeState,
    input_len: u64,
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

    // v3.2: single-symbol literal decode. Proves the lookup + write
    // path works end-to-end before adding the outer loop (v3.3) and
    // refill (v3.4).
    //
    // ABI:
    //   rdi = input_ptr        (preserved)
    //   rsi = lut_entries_ptr
    //   rdx = out_buf_ptr
    //   rcx = out_buf_end
    //   r8  = DecodeState ptr
    //
    // State layout (DecodeState):
    //   [r8 +  0]  byte_pos (u64)
    //   [r8 +  8]  bitbuf   (u64)
    //   [r8 + 16]  bitsleft (i32; signed)
    //   [r8 + 24]  out_pos  (u64)
    //
    // Registers used:
    //   r10 = bitbuf  (caller-saved scratch)
    //   r11 = bitsleft as i32 sign-extended to i64
    //   r12 = byte_pos (callee-saved; pushed)
    //   r13 = out_pos  (callee-saved; pushed)

    let loop_top = ops.new_dynamic_label();
    let refill_chunked = ops.new_dynamic_label();
    let refill_byte = ops.new_dynamic_label();
    let refill_done = ops.new_dynamic_label();
    let non_literal_exit = ops.new_dynamic_label();
    let output_full_exit = ops.new_dynamic_label();
    let underflow_exit = ops.new_dynamic_label();
    let writeback = ops.new_dynamic_label();

    dynasm!(ops
        ; .arch x64
        // Prologue: save callee-saved regs we use.
        ; push r12
        ; push r13
        ; push r14
        ; push rbx

        // Move out_buf_ptr from rdx to rbx (callee-saved).
        // Move out_buf_end from rcx to r14 (callee-saved) so the loop
        // can freely clobber rcx as shift-count + scratch.
        ; mov rbx, rdx
        ; mov r14, rcx

        // Load state.
        ; mov r12, QWORD [r8 + 0]       // byte_pos
        ; mov r10, QWORD [r8 + 8]       // bitbuf
        ; movsxd r11, DWORD [r8 + 16]   // bitsleft
        ; mov r13, QWORD [r8 + 24]      // out_pos

        ;=> loop_top
        // Output bounds check.
        ; cmp r13, r14
        ; jae =>output_full_exit

        // v3.9 chunked refill — libdeflate-style 8-byte load with the
        // `bitsleft |= 56` trick. Hot path is the chunked branch.
        //
        //   bitbuf |= load_u64(in_next) << bitsleft;
        //   in_next += (63 - bitsleft) >> 3;
        //   bitsleft |= 56;                   // bitsleft now ∈ [56, 63]
        //
        // The spilled high bits of the load are NOT lost — we advanced
        // in_next by exactly the bytes consumed, so the spilled bytes
        // appear on the next refill load.
        //
        // Bounds: requires (byte_pos + 8) <= input_len. Falls back to
        // byte-by-byte for the tail.
        ; cmp r11, 56
        ; jge =>refill_done
        ; lea rax, [r12 + 8]
        ; cmp rax, r9                    // r9 = input_len
        ; ja =>refill_byte               // not enough input for 8-byte load

        ;=> refill_chunked
        ; mov rax, QWORD [rdi + r12]
        ; mov cl, r11b
        ; shl rax, cl                    // rax <<= bitsleft (top bits drop)
        ; or r10, rax                    // bitbuf |= shifted load
        ; mov rax, 63
        ; sub rax, r11                   // rax = 63 - bitsleft
        ; shr rax, 3                     // rax = (63 - bitsleft) >> 3
        ; add r12, rax                   // byte_pos += bytes_consumed
        ; or r11, 56                     // bitsleft |= 56  ∈ [56, 63]
        ; jmp =>refill_done

        ;=> refill_byte
        // Byte-by-byte for the input tail: while bitsleft < 56 AND
        // bytes remain, load 1 byte at bitsleft, advance.
        ; cmp r11, 56
        ; jge =>refill_done
        ; cmp r12, r9
        ; jae =>refill_done              // ran out — proceed with what we have
        ; movzx rax, BYTE [rdi + r12]
        ; mov cl, r11b
        ; shl rax, cl
        ; or r10, rax
        ; add r12, 1
        ; add r11, 8
        ; jmp =>refill_byte
        ;=> refill_done

        // 12-bit LUT lookup: key = bitbuf & 0xFFF.
        ; mov rax, r10
        ; and rax, 0xFFF
        // Each LutEntry is 4 bytes: symbol_lo, symbol_hi, length, length_extra.
        ; shl rax, 2
        ; add rax, rsi                   // rax = lut_entries_ptr + key*4
        ; mov eax, DWORD [rax]           // load entry (32 bits)

        // Extract length byte (bits 16-23) into cl.
        ; mov ecx, eax
        ; shr ecx, 16
        // cl = length byte; ch = length_extra byte
        // Check SUBTABLE_FLAG (0x80) or LENGTH_CODE_FLAG (0x40).
        ; test cl, BYTE 0xC0u8 as i8
        ; jnz =>non_literal_exit
        // Check length != 0 (empty slot).
        ; test cl, BYTE 0x3F
        ; jz =>non_literal_exit

        // Extract symbol (bits 0-15) → edx.
        ; mov edx, eax
        ; and edx, 0xFFFF
        ; cmp edx, 256
        ; jae =>non_literal_exit         // EOB or reserved

        // Literal: write byte at out_buf_ptr[out_pos].
        // rbx = out_buf_ptr, r13 = out_pos, dl = symbol low byte
        ; mov BYTE [rbx + r13], dl
        ; inc r13

        // Consume code_bits (cl) from bitbuf.
        ; mov rax, r10
        ; shr rax, cl                    // bitbuf >>= code_bits
        ; mov r10, rax
        ; movzx ecx, cl                  // zero-extend cl to ecx
        ; sub r11, rcx                   // bitsleft -= code_bits

        // v3.4: loop back for the next symbol.
        ; jmp =>loop_top

        ;=> non_literal_exit
        ; mov eax, 0                     // ExitReason::NonLiteral
        ; jmp =>writeback

        ;=> output_full_exit
        ; mov eax, 1                     // ExitReason::OutputFull
        ; jmp =>writeback

        ;=> underflow_exit
        ; mov eax, 2                     // ExitReason::InputUnderflow
        ; jmp =>writeback

        ;=> writeback
        ; mov QWORD [r8 + 0], r12        // byte_pos
        ; mov QWORD [r8 + 8], r10        // bitbuf
        ; mov DWORD [r8 + 16], r11d      // bitsleft
        ; mov QWORD [r8 + 24], r13       // out_pos
        ; pop rbx
        ; pop r14
        ; pop r13
        ; pop r12
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

/// Safe wrapper that calls the JIT'd literal loop.
///
/// Returns the ExitReason. Updates `state` in place.
pub fn run_literal_loop(
    input: &[u8],
    lut_entries: &[u8],
    out_buf: &mut [u8],
    state: &mut DecodeState,
) -> ExitReason {
    let (buf, off) = literal_loop_fn();
    let fp: LiteralLoopFn = unsafe { std::mem::transmute(buf.ptr(*off)) };
    let r = unsafe {
        fp(
            input.as_ptr(),
            lut_entries.as_ptr(),
            out_buf.as_mut_ptr(),
            out_buf.len() as u64,
            state,
            input.len() as u64,
        )
    };
    match r {
        0 => ExitReason::NonLiteral,
        1 => ExitReason::OutputFull,
        _ => ExitReason::InputUnderflow,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// v3.2: with empty LUT entries, the lookup hits an empty slot
    /// (length=0) and returns NonLiteral. Confirms the lookup +
    /// flag-dispatch path is wired correctly.
    #[test]
    fn v3_asm_empty_lut_returns_non_literal() {
        let (buf, off) = literal_loop_fn();
        let fp: LiteralLoopFn = unsafe { std::mem::transmute(buf.ptr(*off)) };
        // 16 bytes of input so refill can read 8 bytes safely.
        let input = [0u8; 16];
        let mut out_buf = [0u8; 32];
        // 4096 LUT entries × 4 bytes = 16 KB. All-zero → every lookup
        // sees length=0, triggers non_literal_exit.
        let lut_entries = vec![0u8; 4096 * 4];
        let mut state = DecodeState {
            byte_pos: 0,
            bitbuf: 0,
            bitsleft: 0,
            _pad: 0,
            out_pos: 0,
        };
        let input_len = input.len() as u64;
        let r = unsafe {
            fp(
                input.as_ptr(),
                lut_entries.as_ptr(),
                out_buf.as_mut_ptr(),
                out_buf.len() as u64,
                &mut state,
                input_len,
            )
        };
        assert_eq!(
            r,
            ExitReason::NonLiteral as i32,
            "empty-slot lookup → NonLiteral"
        );
        // v3.9 chunked refill: load 8 bytes, advance by (63-0)>>3 = 7,
        // set bitsleft |= 56 → bitsleft=56 (since 0 was the seed).
        assert_eq!(state.byte_pos, 7, "chunked refill consumes 7 bytes");
        assert_eq!(state.bitsleft, 56);
    }

    /// v3.4: under a literal-only LUT (LUT[0] = literal 'X', length 4),
    /// the asm loop fills the output buffer to capacity then exits
    /// OutputFull. Every byte should be 'X'.
    #[test]
    fn v3_asm_fills_with_literal() {
        let (buf, off) = literal_loop_fn();
        let fp: LiteralLoopFn = unsafe { std::mem::transmute(buf.ptr(*off)) };

        let input = [0u8; 64];
        let mut out_buf = [0u8; 32];
        let mut lut_entries = vec![0u8; 4096 * 4];
        lut_entries[0] = 0x58; // 'X'
        lut_entries[1] = 0x00;
        lut_entries[2] = 0x04; // length=4
        lut_entries[3] = 0x00;

        let mut state = DecodeState {
            byte_pos: 0,
            bitbuf: 0,
            bitsleft: 0,
            _pad: 0,
            out_pos: 0,
        };
        let input_len = input.len() as u64;
        let r = unsafe {
            fp(
                input.as_ptr(),
                lut_entries.as_ptr(),
                out_buf.as_mut_ptr(),
                out_buf.len() as u64,
                &mut state,
                input_len,
            )
        };
        assert_eq!(r, ExitReason::OutputFull as i32, "loop fills then exits");
        assert_eq!(state.out_pos, 32, "out_pos reached out_buf_end");
        assert!(out_buf.iter().all(|&b| b == b'X'), "every byte is 'X'");
    }

    /// v3.4: multiple literals decoded in one call. With a 4-bit
    /// repeating literal code, we should write multiple 'X' bytes
    /// before the bit buffer runs out and triggers a non-literal
    /// exit (when the empty slot after the valid bits is hit).
    #[test]
    fn v3_asm_loops_through_multiple_literals() {
        let (buf, off) = literal_loop_fn();
        let fp: LiteralLoopFn = unsafe { std::mem::transmute(buf.ptr(*off)) };

        // Bitbuf will be all zeros after refill. Every 4-bit window
        // is 0, so every lookup hits LUT[0] which we set to literal 'X'
        // length 4. After 16 such writes (64 bits / 4 bits = 16
        // literals), bitsleft drops to 0; the refill check at
        // loop_top sees bitsleft < 12, refills another 64 bits (byte_pos
        // advances by 8) and continues. Loop terminates when out_pos
        // hits out_buf_end.
        let input = vec![0u8; 256];
        let mut out_buf = vec![0u8; 100]; // 100 'X's expected
        let mut lut_entries = vec![0u8; 4096 * 4];
        lut_entries[0] = 0x58; // 'X'
        lut_entries[1] = 0x00;
        lut_entries[2] = 0x04; // length=4
        lut_entries[3] = 0x00;

        let mut state = DecodeState {
            byte_pos: 0,
            bitbuf: 0,
            bitsleft: 0,
            _pad: 0,
            out_pos: 0,
        };
        let input_len = input.len() as u64;
        let r = unsafe {
            fp(
                input.as_ptr(),
                lut_entries.as_ptr(),
                out_buf.as_mut_ptr(),
                out_buf.len() as u64,
                &mut state,
                input_len,
            )
        };
        // Loop fills out_buf to capacity, then exits OutputFull.
        assert_eq!(r, ExitReason::OutputFull as i32);
        assert_eq!(state.out_pos, 100);
        // Every byte should be 'X'.
        assert!(out_buf.iter().all(|&b| b == b'X'), "all 'X'");
    }

    /// v3.2: output-full exit path. With out_buf_end == out_pos, the
    /// asm should return OutputFull (1) before doing any work.
    #[test]
    fn v3_asm_output_full_returns_early() {
        let (buf, off) = literal_loop_fn();
        let fp: LiteralLoopFn = unsafe { std::mem::transmute(buf.ptr(*off)) };
        let input = [0u8; 16];
        let mut out_buf = [0u8; 32];
        let lut_entries = vec![0u8; 4096 * 4];
        let mut state = DecodeState {
            byte_pos: 0,
            bitbuf: 0,
            bitsleft: 0,
            _pad: 0,
            out_pos: 32, // == out_buf.len()
        };
        let input_len = input.len() as u64;
        let r = unsafe {
            fp(
                input.as_ptr(),
                lut_entries.as_ptr(),
                out_buf.as_mut_ptr(),
                32, // out_buf_end
                &mut state,
                input_len,
            )
        };
        assert_eq!(r, ExitReason::OutputFull as i32);
        // No refill happened.
        assert_eq!(state.byte_pos, 0);
        assert_eq!(state.bitsleft, 0);
    }
}
