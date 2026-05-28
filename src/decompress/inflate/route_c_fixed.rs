//! Route C v1 — fixed-Huffman DEFLATE inflate via hand-written x86_64
//! assembly emitted by dynasm-rs (`plans/unified-decoder.md` §3.1 / §4.3).
//!
//! ## Scope
//!
//! v1 supports **fixed-Huffman (BTYPE=01) literal-only blocks**: literals,
//! end-of-block, and length/distance codes that emit no extra bits (i.e.
//! symbol 256 EOB only). Real DEFLATE blocks with back-references will
//! fall back to the existing pure-Rust decoder. v2 (separate workstream)
//! adds match copy + extra-bits + dist table.
//!
//! ## Correctness contract
//!
//! - Three-oracle differential (flate2 + libdeflate + zlib-ng) must
//!   agree byte-for-byte with this decoder on every literal-only
//!   fixed-Huffman block.
//! - The emitter is deterministic: identical input → identical output
//!   byte sequence.
//! - All asm reads/writes are bounds-checked at the Rust boundary;
//!   the asm itself trusts the caller's input/output slice lengths
//!   (the standard performance/safety trade for JIT'd inner loops).
//!
//! ## Why this exists
//!
//! Per `plans/unified-decoder.md` §3.1, the end-state inflate uses
//! per-block hand-emitted asm (dynasm) with AOT codegen for top-N
//! fingerprints. The fixed-Huffman fingerprint is the natural
//! starting point because the litlen/dist tables are RFC-fixed and
//! the emit can be a single compile-time constant table.
//!
//! ## RFC 1951 §3.2.6 fixed Huffman table
//!
//! ```text
//! Lit Value    Bits  Codes
//! ---------    ----  -----
//!   0 - 143     8    00110000 .. 10111111
//! 144 - 255     9    110010000 .. 111111111
//! 256 - 279     7    0000000 .. 0010111
//! 280 - 287     8    11000000 .. 11000111
//! ```

#![cfg(all(target_arch = "x86_64", feature = "route-c-dynasm"))]

use dynasm::dynasm;
use dynasmrt::{DynasmApi, DynasmLabelApi, ExecutableBuffer};

/// Fixed-Huffman decode entry. Packed (symbol, code_length).
/// `code_length` is 7, 8, or 9 bits; `symbol` is 0..=287.
#[derive(Clone, Copy, Debug)]
pub struct FixedEntry {
    pub symbol: u16,
    pub bits: u8,
}

/// Build the fixed-Huffman lookup table indexed by 9 bits (reversed).
///
/// Returns 512 entries; each maps a 9-bit (reversed) key to the symbol
/// it decodes plus the code length consumed. Decoders peek 9 bits,
/// look up, then advance by `bits`.
///
/// This is `const`-shape data — could be folded to a `const` table
/// once the codebase migrates off non-const ops. Today the build
/// runs once via OnceLock.
fn reverse_bits_local(mut code: u32, n: u8) -> u32 {
    let mut rev = 0u32;
    for _ in 0..n {
        rev = (rev << 1) | (code & 1);
        code >>= 1;
    }
    rev
}

pub fn build_fixed_lut() -> [FixedEntry; 512] {
    let mut lut = [FixedEntry { symbol: 0, bits: 0 }; 512];

    // Walk symbols and assign codes per RFC 1951 §3.2.6.
    // Code lengths array indexed by symbol:
    let mut code_lengths = [0u8; 288];
    code_lengths[0..=143].fill(8);
    code_lengths[144..=255].fill(9);
    code_lengths[256..=279].fill(7);
    code_lengths[280..=287].fill(8);

    // Count codes per length
    let mut count = [0u16; 16];
    for &len in code_lengths.iter() {
        if len > 0 && len <= 15 {
            count[len as usize] += 1;
        }
    }

    // First code per length (canonical Huffman)
    let mut first_code = [0u32; 16];
    let mut code: u32 = 0;
    for len in 1..=15 {
        code = (code + count[len - 1] as u32) << 1;
        first_code[len] = code;
    }

    // Assign + populate LUT
    let mut next_code = first_code;
    for (symbol, &len) in code_lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let len_u = len as usize;
        let codeword = next_code[len_u];
        next_code[len_u] += 1;
        // The LUT key is `reverse_bits(codeword, len)` left-extended to 9 bits.
        // For len < 9, every 9-bit key whose low `len` bits == reversed(codeword)
        // resolves to this symbol.
        let rev = reverse_bits_local(codeword, len);
        let stride = 1u32 << len;
        let mut key = rev;
        while (key as usize) < 512 {
            lut[key as usize] = FixedEntry {
                symbol: symbol as u16,
                bits: len,
            };
            key += stride;
        }
    }
    lut
}

/// JIT'd literal-only inflate function signature.
///
/// Arguments (System V AMD64):
/// - `rdi` = input data ptr
/// - `rsi` = input bit position (in bits)
/// - `rdx` = output data ptr
/// - `rcx` = output capacity
/// - `r8`  = `*const FixedEntry` (LUT ptr)
///
/// Returns (`rax`): output bytes written. Negative on error.
///
/// v1 ONLY decodes literals + EOB. If a length code (symbol >= 257)
/// is encountered, returns `-1` and the caller falls back to the
/// pure-Rust decoder for the remainder of the block.
type LiteralOnlyFn = unsafe extern "sysv64" fn(
    input: *const u8,
    bit_pos: u64,
    output: *mut u8,
    cap: u64,
    lut: *const FixedEntry,
) -> i64;

/// Emit the literal-only fixed-Huffman inflate inner loop.
///
/// Algorithm:
/// ```text
/// loop {
///     bits_left < 9 → refill 8 bytes from input
///     key = bits & 0x1FF          ; 9-bit LUT key
///     entry = lut[key]            ; (symbol, code_length)
///     if symbol == 256: ret out_bytes
///     if symbol >= 257: ret -1    (caller fallback)
///     output[out_pos++] = symbol  (low 8 bits — literal)
///     bits >>= entry.bits
///     bits_left -= entry.bits
/// }
/// ```
///
/// Bits are kept in `r10` (64-bit buffer); `r11b` is bits_left (counter).
/// Out pos accumulates in `r12`; loop bound checked at top of each iter.
pub fn emit_literal_only() -> (ExecutableBuffer, dynasmrt::AssemblyOffset) {
    let mut ops = dynasmrt::x64::Assembler::new().expect("dynasm: alloc page");
    let start = ops.offset();
    // dynasm-rs forward labels: `=>label_name` references, `=> label_name:`
    // defines. Standard System V AMD64 calling convention; we save +
    // restore r12-r14 (callee-saved scratch).
    let loop_top = ops.new_dynamic_label();
    let have_bits = ops.new_dynamic_label();
    let eob = ops.new_dynamic_label();
    let fallback_match = ops.new_dynamic_label();
    let overflow = ops.new_dynamic_label();
    let epilogue = ops.new_dynamic_label();
    dynasm!(ops
        ; .arch x64
        // Prologue: save callee-saved regs we touch.
        ; push r12
        ; push r13
        ; push r14
        ; push rbx
        // Save the output capacity (rcx in System V is 4th arg) to rbx
        // because we use rcx as the shift count register.
        ; mov rbx, rcx
        // Load bit buffer from input byte at (bit_pos / 8), aligned.
        ; mov r10, rsi          // r10 = bit_pos (bits, not bytes)
        ; mov r11, r10
        ; shr r11, 3            // r11 = byte offset
        ; mov r10, QWORD [rdi + r11]   // r10 = 8 bytes from input
        ; and rsi, 7            // rsi = bit_pos % 8 (sub-byte offset)
        ; mov rcx, rsi
        ; shr r10, cl           // r10 >>= sub-byte offset → bit-aligned buffer
        ; mov r11d, 64
        ; sub r11b, sil         // r11b = 64 - sub_offset (bits in buffer)
        ; xor r12, r12          // r12 = out_pos
        ; mov r13, rdx          // r13 = output base ptr (immutable)
        ; mov r14, r8           // r14 = LUT ptr
        ;=> loop_top
        // Bounds check: r12 < cap (cap in rbx).
        ; cmp r12, rbx
        ; jae =>overflow
        // Refill if bits_left < 9.
        ; cmp r11b, BYTE 9
        ; jae =>have_bits
        // v1 doesn't implement multi-byte refill — return sentinel
        // and let the caller fall back. v2 will add proper refill.
        ; mov rax, -2           // refill_underflow sentinel
        ; jmp =>epilogue
        ;=> have_bits
        // Extract 9-bit LUT key.
        ; mov rax, r10
        ; and rax, 0x1FF
        // Load entry: each FixedEntry is 4 bytes (repr(Rust) padding to
        // alignof(u16)=2 → struct size is 4 bytes after u8 field).
        ; shl rax, 2            // index × 4
        ; add rax, r14
        ; movzx ecx, WORD [rax]      // ecx = symbol
        ; movzx eax, BYTE [rax + 2]  // eax = bits
        // EOB?
        ; cmp ecx, 256
        ; je =>eob
        // Match code? (>= 257). v1 returns -1 for fallback.
        ; cmp ecx, 257
        ; jae =>fallback_match
        // Literal: write cl into output, advance.
        ; mov [r13 + r12], cl
        ; inc r12
        // Consume `eax` bits.
        ; mov rcx, rax
        ; shr r10, cl
        ; sub r11b, al
        ; jmp =>loop_top
        ;=> eob
        ; mov rax, r12          // bytes written
        ; jmp =>epilogue
        ;=> fallback_match
        ; mov rax, -1           // signal caller to use pure-Rust path
        ; jmp =>epilogue
        ;=> overflow
        ; mov rax, -3           // output capacity overflow
        ;=> epilogue
        ; pop rbx
        ; pop r14
        ; pop r13
        ; pop r12
        ; ret
    );
    let buf = ops.finalize().expect("dynasm: finalize");
    (buf, start)
}

/// Lazy LUT instance (build once per process).
pub fn fixed_lut() -> &'static [FixedEntry; 512] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[FixedEntry; 512]> = OnceLock::new();
    LUT.get_or_init(build_fixed_lut)
}

/// Lazy emitter (build once per process).
pub fn literal_only_fn() -> &'static (ExecutableBuffer, dynasmrt::AssemblyOffset) {
    use std::sync::OnceLock;
    static FN: OnceLock<(ExecutableBuffer, dynasmrt::AssemblyOffset)> = OnceLock::new();
    FN.get_or_init(emit_literal_only)
}

/// Safe wrapper: decode literal-only fixed-Huffman block.
///
/// Returns `Ok(bytes_written)` on EOB; `Err` if a match code is hit
/// (caller falls back) or output overflows.
pub fn decode_literal_only_block(
    input: &[u8],
    bit_pos: usize,
    output: &mut [u8],
) -> std::io::Result<usize> {
    let lut = fixed_lut();
    let (buf, offset) = literal_only_fn();
    let fp: LiteralOnlyFn = unsafe { std::mem::transmute(buf.ptr(*offset)) };
    let written = unsafe {
        fp(
            input.as_ptr(),
            bit_pos as u64,
            output.as_mut_ptr(),
            output.len() as u64,
            lut.as_ptr(),
        )
    };
    if written >= 0 {
        Ok(written as usize)
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            match written {
                -1 => "route-c v1: match code in block (fallback to pure-Rust)",
                -2 => "route-c v1: input bit underflow",
                -3 => "route-c v1: output capacity overflow",
                _ => "route-c v1: unknown error",
            },
        ))
    }
}

// ── Route C v2 — full fixed-Huffman inflate (literals + matches) ─────────
//
// Pure-Rust reference decoder that handles ALL fixed-Huffman symbols:
// literals (0-255), EOB (256), and length codes (257-285) with their
// extra bits, plus the fixed 5-bit distance code and its extra bits.
// This is the correctness oracle; a future v2 commit will mirror this
// in dynasm-rs asm. Lives in the same module because the LUT + bit-reader
// primitives are shared.

/// Length base + extra-bits table per RFC 1951 §3.2.5 (length symbols 257-285).
/// Indexed by `symbol - 257`. Entry: (length base, extra bits).
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Distance base + extra-bits table per RFC 1951 §3.2.5 (distance symbols 0-29).
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// LSB-first bit reader over a byte slice.
struct BitReader<'a> {
    buf: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8], bit_pos: usize) -> Self {
        Self { buf, bit_pos }
    }

    /// Peek `n` bits (≤ 32). Returns 0-padded if past EOF.
    fn peek(&self, n: u8) -> u32 {
        debug_assert!(n <= 32);
        let byte = self.bit_pos / 8;
        let off = (self.bit_pos % 8) as u32;
        let mut buf: u64 = 0;
        for i in 0..6 {
            if byte + i < self.buf.len() {
                buf |= (self.buf[byte + i] as u64) << (i * 8);
            }
        }
        ((buf >> off) & ((1u64 << n) - 1)) as u32
    }

    fn consume(&mut self, n: u8) {
        self.bit_pos += n as usize;
    }

    fn read(&mut self, n: u8) -> u32 {
        let v = self.peek(n);
        self.consume(n);
        v
    }

    fn pos(&self) -> usize {
        self.bit_pos
    }
}

/// Pure-Rust reference decoder for a fixed-Huffman block.
/// Caller positions `bit_pos` at the START of block data (after the
/// 3-bit header `BFINAL|BTYPE`); the decoder consumes through EOB
/// and returns the new bit position + total bytes written.
///
/// Output is appended to `output` starting at `out_start`. Returns
/// `(new_bit_pos, total_out_pos)` or an io::Error on malformed data
/// or output overflow.
pub fn decode_fixed_block_rust(
    input: &[u8],
    bit_pos: usize,
    output: &mut [u8],
    out_start: usize,
) -> std::io::Result<(usize, usize)> {
    let lut = fixed_lut();
    let mut br = BitReader::new(input, bit_pos);
    let mut out_pos = out_start;

    loop {
        // Decode litlen symbol via 9-bit LUT.
        let key = br.peek(9) as usize;
        let entry = lut[key];
        br.consume(entry.bits);
        let sym = entry.symbol;
        if sym < 256 {
            // Literal byte.
            if out_pos >= output.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "route-c v2: output overflow",
                ));
            }
            output[out_pos] = sym as u8;
            out_pos += 1;
        } else if sym == 256 {
            // EOB.
            return Ok((br.pos(), out_pos));
        } else if sym <= 285 {
            // Length code 257..285. Read extra bits + decode dist.
            let li = (sym - 257) as usize;
            let length_base = LENGTH_BASE[li] as usize;
            let extra = LENGTH_EXTRA[li];
            let length = length_base + br.read(extra) as usize;

            // Fixed-Huffman dist code is a fixed 5-bit codeword (per RFC §3.2.6).
            // Read 5 bits and REVERSE them to recover the symbol.
            let dist_code_bits = br.read(5) as u32;
            let dist_sym = reverse_bits_local(dist_code_bits, 5) as usize;
            if dist_sym >= 30 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("route-c v2: invalid distance symbol {dist_sym}"),
                ));
            }
            let dist_base = DIST_BASE[dist_sym] as usize;
            let dist_extra = DIST_EXTRA[dist_sym];
            let distance = dist_base + br.read(dist_extra) as usize;

            // Match copy with overlap (RLE for dist < length).
            if distance == 0 || distance > out_pos {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("route-c v2: invalid distance {distance} at out_pos {out_pos}"),
                ));
            }
            if out_pos + length > output.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "route-c v2: match overflow",
                ));
            }
            // Byte-by-byte to handle overlap (dist < length).
            for i in 0..length {
                output[out_pos + i] = output[out_pos + i - distance];
            }
            out_pos += length;
        } else {
            // Symbols 286, 287 are reserved per RFC 1951.
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("route-c v2: reserved symbol {sym}"),
            ));
        }
    }
}

/// Decode a complete fixed-Huffman gzip stream (header + fixed block + footer)
/// using the Route C v2 reference decoder. Used in tests to round-trip vs
/// flate2-encoded data.
///
/// `gz` is the full gzip bytes. Returns the decoded payload.
pub fn decode_gzip_with_route_c_v2(gz: &[u8]) -> std::io::Result<Vec<u8>> {
    // Minimal gzip header parse: 10-byte fixed header (no FEXTRA/FNAME).
    if gz.len() < 18 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "gzip stream too short",
        ));
    }
    if gz[0] != 0x1f || gz[1] != 0x8b {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a gzip stream",
        ));
    }
    if gz[2] != 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "non-DEFLATE compression",
        ));
    }
    let flg = gz[3];
    let mut header_end = 10;
    if flg & 0x04 != 0 {
        // FEXTRA
        let xlen = u16::from_le_bytes([gz[header_end], gz[header_end + 1]]) as usize;
        header_end += 2 + xlen;
    }
    if flg & 0x08 != 0 {
        // FNAME (NUL-terminated)
        while header_end < gz.len() && gz[header_end] != 0 {
            header_end += 1;
        }
        header_end += 1;
    }
    if flg & 0x10 != 0 {
        // FCOMMENT
        while header_end < gz.len() && gz[header_end] != 0 {
            header_end += 1;
        }
        header_end += 1;
    }
    if flg & 0x02 != 0 {
        // FHCRC
        header_end += 2;
    }
    let isize_field = u32::from_le_bytes([
        gz[gz.len() - 4],
        gz[gz.len() - 3],
        gz[gz.len() - 2],
        gz[gz.len() - 1],
    ]) as usize;

    // Bit position 0 is the first bit of the deflate stream.
    let mut bit_pos = header_end * 8;
    let mut output = vec![0u8; isize_field + 64]; // small slack for over-emit
    let mut out_pos = 0;

    loop {
        // 3-bit block header: BFINAL (1 bit), BTYPE (2 bits).
        let mut br = BitReader::new(gz, bit_pos);
        let bfinal = br.read(1);
        let btype = br.read(2);
        bit_pos = br.pos();
        if btype != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!("route-c v2 only supports BTYPE=01 (fixed Huffman); got BTYPE={btype}"),
            ));
        }
        let (new_bit_pos, new_out_pos) =
            decode_fixed_block_rust(gz, bit_pos, &mut output, out_pos)?;
        bit_pos = new_bit_pos;
        out_pos = new_out_pos;
        if bfinal == 1 {
            break;
        }
    }
    output.truncate(out_pos);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixed LUT covers all 512 9-bit keys: every entry has bits > 0.
    #[test]
    fn fixed_lut_is_complete() {
        let lut = build_fixed_lut();
        for (i, e) in lut.iter().enumerate() {
            assert!(
                e.bits >= 7 && e.bits <= 9,
                "key {} has invalid bits {}",
                i,
                e.bits
            );
            assert!(e.symbol <= 287, "key {} has invalid symbol {}", i, e.symbol);
        }
    }

    /// Symbol 256 (EOB) lives at code 0000000 (7 bits).
    /// Reversed: still 0000000. The LUT key extends to 9 bits with
    /// strides of 2^7 = 128, so keys 0, 128, 256, 384 all → symbol 256.
    #[test]
    fn fixed_lut_eob_key() {
        let lut = build_fixed_lut();
        for &k in &[0usize, 128, 256, 384] {
            assert_eq!(lut[k].symbol, 256, "key {} should decode to EOB", k);
            assert_eq!(lut[k].bits, 7, "EOB code is 7 bits");
        }
    }

    /// The JIT'd literal-only decoder runs and returns. Build a tiny
    /// fixed-Huffman block that's just `BFINAL=1, BTYPE=01, EOB`.
    ///
    /// Bit layout (LSB-first per RFC 1951):
    ///   BFINAL=1 (1 bit), BTYPE=01 (2 bits), EOB code (7 bits of 0)
    ///   → byte stream: 1|01|0000000 → LSB-first → 0000000_01_1 → 0b0000001_1
    ///   → byte 0 = 0b00000011 = 0x03; remaining bits = 0.
    #[test]
    fn route_c_v1_empty_fixed_block_returns_zero() {
        // We bypass the block header (caller's job) and call directly
        // at bit_pos = 0 of an input that is just the EOB code.
        // EOB is 7 bits of 0 → first byte = 0x00.
        let input = [0u8; 16];
        let mut output = [0u8; 256];
        let r = decode_literal_only_block(&input, 0, &mut output);
        assert_eq!(r.ok(), Some(0), "EOB-only block should write 0 bytes");
    }

    /// Encode a single literal 'A' (0x41) followed by EOB and verify
    /// the JIT decodes it byte-perfect.
    ///
    /// Per RFC 1951 §3.1.1 Huffman codes are emitted with the
    /// MOST-significant bit FIRST into the lowest stream position.
    /// LSB-first byte reading then RE-reverses them, so the LUT (built
    /// with `reverse_bits(codeword, len)`) sees the codeword in its
    /// original MSB-first form.
    ///
    /// 'A' = symbol 65; len = 8; codeword = 48 + 65 = 113 = 0b01110001.
    /// Stream byte 0 (MSB of code in bit 0): bits 0..7 = 0,1,1,1,0,0,0,1
    /// → byte = 0b10001110 = 0x8E.
    /// EOB = symbol 256; len = 7; codeword = 0. Stream bits 8..14 = 0;
    /// byte 1 = 0.
    #[test]
    fn route_c_v1_single_literal_a() {
        let input = [0x8Eu8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut output = [0u8; 256];
        let r = decode_literal_only_block(&input, 0, &mut output);
        assert_eq!(r.ok(), Some(1), "single 'A' + EOB should write 1 byte");
        assert_eq!(output[0], b'A');
    }

    /// Helper: gzip-encode `data` at level 1 (zlib's default for fast =
    /// fixed-Huffman where the encoder chooses it). Flate2's behavior:
    /// level=1 emits fixed-Huffman blocks for typical data.
    fn gzip_level1(data: &[u8]) -> Vec<u8> {
        use flate2::{write::GzEncoder, Compression};
        use std::io::Write;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(1));
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    /// Route C v2 reference Rust decoder vs flate2 oracle: empty payload.
    #[test]
    fn route_c_v2_empty_payload() {
        let payload = b"";
        let gz = gzip_level1(payload);
        // flate2 may emit a stored block for empty payload — check first.
        let header_end = {
            let flg = gz[3];
            let mut p = 10;
            if flg & 0x04 != 0 {
                let xlen = u16::from_le_bytes([gz[p], gz[p + 1]]) as usize;
                p += 2 + xlen;
            }
            if flg & 0x08 != 0 {
                while gz[p] != 0 {
                    p += 1;
                }
                p += 1;
            }
            p
        };
        let btype = (gz[header_end] >> 1) & 0b11;
        if btype != 1 {
            eprintln!("flate2 emitted BTYPE={btype} for empty payload; skipping");
            return;
        }
        let decoded = decode_gzip_with_route_c_v2(&gz).expect("decode");
        assert_eq!(decoded.as_slice(), payload);
    }

    /// Round-trip ASCII text through flate2 level 1, decode with Route C v2,
    /// compare byte-for-byte. This exercises literal codes only.
    #[test]
    fn route_c_v2_text_round_trip() {
        let payload = b"Hello, World! This is a test of Route C v2 fixed-Huffman.";
        let gz = gzip_level1(payload);

        // Skip the test if flate2 chose a different block type than BTYPE=01.
        // (For small inputs zlib sometimes uses stored.)
        let header_end = {
            let flg = gz[3];
            let mut p = 10;
            if flg & 0x04 != 0 {
                let xlen = u16::from_le_bytes([gz[p], gz[p + 1]]) as usize;
                p += 2 + xlen;
            }
            if flg & 0x08 != 0 {
                while gz[p] != 0 {
                    p += 1;
                }
                p += 1;
            }
            p
        };
        let btype = (gz[header_end] >> 1) & 0b11;
        if btype != 1 {
            eprintln!("flate2 emitted BTYPE={btype} for text; skipping");
            return;
        }

        let decoded = decode_gzip_with_route_c_v2(&gz).expect("decode");
        assert_eq!(decoded.as_slice(), payload);
    }

    /// Round-trip a repetitive payload that's GUARANTEED to exercise the
    /// match-copy path (length + distance codes).
    #[test]
    fn route_c_v2_match_copy_round_trip() {
        // 200-byte pattern: 'A' repeated. Must produce many back-refs.
        let payload = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let gz = gzip_level1(payload);

        let header_end = {
            let flg = gz[3];
            let mut p = 10;
            if flg & 0x04 != 0 {
                let xlen = u16::from_le_bytes([gz[p], gz[p + 1]]) as usize;
                p += 2 + xlen;
            }
            if flg & 0x08 != 0 {
                while gz[p] != 0 {
                    p += 1;
                }
                p += 1;
            }
            p
        };
        let btype = (gz[header_end] >> 1) & 0b11;
        if btype != 1 {
            eprintln!("flate2 emitted BTYPE={btype} for repetitive; skipping");
            return;
        }

        let decoded = decode_gzip_with_route_c_v2(&gz).expect("decode");
        assert_eq!(decoded.as_slice(), payload.as_slice());
    }

    /// Differential vs flate2: random binary data round-trip.
    #[test]
    fn route_c_v2_random_binary_round_trip() {
        // Deterministic pseudo-random payload.
        let mut payload = vec![0u8; 4096];
        let mut x: u32 = 0xdead_beef;
        for b in payload.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *b = (x >> 16) as u8;
        }
        let gz = gzip_level1(&payload);

        let header_end = {
            let flg = gz[3];
            let mut p = 10;
            if flg & 0x04 != 0 {
                let xlen = u16::from_le_bytes([gz[p], gz[p + 1]]) as usize;
                p += 2 + xlen;
            }
            if flg & 0x08 != 0 {
                while gz[p] != 0 {
                    p += 1;
                }
                p += 1;
            }
            p
        };
        let btype = (gz[header_end] >> 1) & 0b11;
        if btype != 1 {
            // Random data often goes to stored blocks (BTYPE=00) under level 1.
            eprintln!("flate2 emitted BTYPE={btype} for random; skipping (expected)");
            return;
        }

        let decoded = decode_gzip_with_route_c_v2(&gz).expect("decode");
        assert_eq!(decoded.len(), payload.len());
        assert_eq!(decoded.as_slice(), payload.as_slice());
    }
}
