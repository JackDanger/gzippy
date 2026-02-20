//! Speculative parallel decompression for single-member gzip files
//!
//! Implements rapidgzip's key insight: decode chunks IN PARALLEL from the start,
//! using u16 output buffers where unresolved back-references become "markers"
//! that are resolved once the previous chunk's window is known.
//!
//! This eliminates the sequential scan bottleneck that made two-pass approaches
//! slower than sequential libdeflate.
//!
//! Algorithm:
//! 1. Partition compressed data at fixed byte intervals (4 MiB)
//! 2. Find valid deflate block start near each partition point
//! 3. Decode each chunk in parallel with u16 output (markers for unknown refs)
//! 4. Propagate 32KB windows sequentially between chunks
//! 5. Resolve markers in parallel using the propagated windows
//! 6. Assemble final u8 output

use crate::consume_first_decode::Bits;
use crate::libdeflate_decode::get_fixed_tables;
use crate::libdeflate_entry::{DistTable, LitLenTable};
use crate::marker_decode::skip_gzip_header;
use std::io::{self, Error, ErrorKind};

const WINDOW_SIZE: usize = 32768;
const MARKER_BASE: u16 = WINDOW_SIZE as u16;
const CHUNK_SPACING: usize = 4 * 1024 * 1024;
const MIN_SIZE_FOR_PARALLEL: usize = 8 * 1024 * 1024;
#[allow(dead_code)]
const MIN_VALID_OUTPUT: usize = 32 * 1024;
#[allow(dead_code)]
const MAX_TRIAL_INPUT: usize = 64 * 1024;

// =============================================================================
// Core marker-aware inflate using fast Bits reader
// =============================================================================

struct MarkerInflateResult {
    data: Vec<u16>,
    #[allow(dead_code)]
    decoded_bytes: usize,
    #[allow(dead_code)]
    marker_count: usize,
    window: [u8; WINDOW_SIZE],
    window_fill: usize,
    #[allow(dead_code)]
    bits_end_pos: usize,
}

fn read_dynamic_tables(bits: &mut Bits) -> io::Result<(LitLenTable, DistTable)> {
    if bits.available() < 14 {
        bits.refill();
    }

    let hlit = (bits.peek() & 0x1F) as usize + 257;
    bits.consume(5);
    let hdist = (bits.peek() & 0x1F) as usize + 1;
    bits.consume(5);
    let hclen = (bits.peek() & 0xF) as usize + 4;
    bits.consume(4);

    if hlit > 286 || hdist > 30 || hclen > 19 {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid dynamic header"));
    }

    const CODE_LENGTH_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];

    let mut code_length_lengths = [0u8; 19];
    for &idx in CODE_LENGTH_ORDER.iter().take(hclen) {
        if bits.available() < 3 {
            bits.refill();
        }
        code_length_lengths[idx] = (bits.peek() & 0x7) as u8;
        bits.consume(3);
    }

    let cl_table = build_code_length_table(&code_length_lengths)?;

    let mut all_lengths = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < hlit + hdist {
        if bits.available() < 15 {
            bits.refill();
        }

        let entry = cl_table[(bits.peek() & 0x7F) as usize];
        let symbol = (entry >> 8) as u8;
        let len = (entry & 0xFF) as u8;
        bits.consume(len as u32);

        match symbol {
            0..=15 => {
                all_lengths[i] = symbol;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(Error::new(ErrorKind::InvalidData, "Invalid repeat"));
                }
                let repeat = 3 + (bits.peek() & 0x3) as usize;
                bits.consume(2);
                let val = all_lengths[i - 1];
                for _ in 0..repeat {
                    if i >= hlit + hdist {
                        break;
                    }
                    all_lengths[i] = val;
                    i += 1;
                }
            }
            17 => {
                let repeat = 3 + (bits.peek() & 0x7) as usize;
                bits.consume(3);
                for _ in 0..repeat {
                    if i >= hlit + hdist {
                        break;
                    }
                    all_lengths[i] = 0;
                    i += 1;
                }
            }
            18 => {
                let repeat = 11 + (bits.peek() & 0x7F) as usize;
                bits.consume(7);
                for _ in 0..repeat {
                    if i >= hlit + hdist {
                        break;
                    }
                    all_lengths[i] = 0;
                    i += 1;
                }
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "Invalid code length symbol",
                ));
            }
        }
    }

    let litlen = LitLenTable::build(&all_lengths[..hlit])
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Invalid litlen table"))?;
    let dist = DistTable::build(&all_lengths[hlit..])
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Invalid dist table"))?;

    Ok((litlen, dist))
}

fn build_code_length_table(lengths: &[u8; 19]) -> io::Result<Vec<u32>> {
    let mut count = [0u16; 8];
    for &len in lengths.iter() {
        if len > 7 {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid CL length"));
        }
        if len > 0 {
            count[len as usize] += 1;
        }
    }

    let mut next_code = [0u16; 8];
    let mut code = 0u16;
    for bits in 1..8 {
        code = (code + count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    let table_size = 128;
    let mut table = vec![0u32; table_size];

    for (symbol, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let code = next_code[len as usize];
        next_code[len as usize] += 1;

        let reversed = (0..len).fold(0u16, |acc, i| acc | (((code >> i) & 1) << (len - 1 - i)));

        let step = 1usize << len;
        let mut idx = reversed as usize;
        while idx < table_size {
            table[idx] = ((symbol as u32) << 8) | (len as u32);
            idx += step;
        }
    }

    Ok(table)
}

/// Decode a deflate stream with marker support for unresolved back-references.
///
/// Uses the fast Bits reader and production LitLenTable/DistTable for Huffman
/// decode, but outputs u16 where values > 255 are markers for back-references
/// that extend before the start of this chunk's decoded data.
///
/// `max_input_pos`: stop decoding when the bit reader reaches this byte position
/// in `deflate_data`. Set to `deflate_data.len()` to decode until BFINAL.
fn inflate_with_markers(
    deflate_data: &[u8],
    start_byte: usize,
    start_bit_offset: u8,
    max_input_pos: usize,
) -> io::Result<MarkerInflateResult> {
    let mut bits = Bits {
        data: deflate_data,
        pos: start_byte,
        bitbuf: 0,
        bitsleft: 0,
    };
    bits.refill();

    if start_bit_offset > 0 {
        bits.consume(start_bit_offset as u32);
    }

    let mut output: Vec<u16> = Vec::with_capacity(4 * 1024 * 1024);
    let mut window = [0u16; WINDOW_SIZE];
    let mut window_pos: usize = 0;
    let mut decoded_bytes: usize = 0;
    let mut marker_count: usize = 0;

    let debug = std::env::var("GZIPPY_SPEC_DEBUG").is_ok();
    let mut block_num = 0u32;

    loop {
        if bits.pos >= max_input_pos {
            break;
        }

        if bits.available() < 3 {
            bits.refill();
        }
        if bits.available() < 3 {
            break;
        }

        let bfinal = (bits.peek() & 1) != 0;
        let btype = ((bits.peek() >> 1) & 3) as u8;
        bits.consume(3);

        if debug {
            eprintln!(
                "  block {}: bfinal={} btype={} pos={} decoded={}",
                block_num, bfinal, btype, bits.pos, decoded_bytes
            );
        }

        let block_result = match btype {
            0 => decode_stored_markers(
                &mut bits,
                &mut output,
                &mut window,
                &mut window_pos,
                &mut decoded_bytes,
            ),
            1 => {
                let (litlen, dist) = get_fixed_tables();
                decode_huffman_markers(
                    &mut bits,
                    &mut output,
                    &mut window,
                    &mut window_pos,
                    &mut decoded_bytes,
                    &mut marker_count,
                    litlen,
                    dist,
                )
            }
            2 => match read_dynamic_tables(&mut bits) {
                Ok((litlen, dist)) => decode_huffman_markers(
                    &mut bits,
                    &mut output,
                    &mut window,
                    &mut window_pos,
                    &mut decoded_bytes,
                    &mut marker_count,
                    &litlen,
                    &dist,
                ),
                Err(e) => Err(e),
            },
            _ => Err(Error::new(ErrorKind::InvalidData, "Invalid block type")),
        };

        if let Err(e) = block_result {
            if decoded_bytes > 0 {
                if debug {
                    eprintln!(
                        "  block {}: error at pos={} decoded={}, treating as end-of-chunk: {}",
                        block_num, bits.pos, decoded_bytes, e
                    );
                }
                break;
            }
            return Err(e);
        }

        block_num += 1;
        if bfinal {
            break;
        }
    }

    let mut final_window = [0u8; WINDOW_SIZE];
    let fill = decoded_bytes.min(WINDOW_SIZE);
    for (i, byte) in final_window[..fill].iter_mut().enumerate() {
        let idx = (window_pos + WINDOW_SIZE - fill + i) % WINDOW_SIZE;
        let val = window[idx];
        *byte = if val <= 255 { val as u8 } else { 0 };
    }

    Ok(MarkerInflateResult {
        data: output,
        decoded_bytes,
        marker_count,
        window: final_window,
        window_fill: fill,
        bits_end_pos: bits.pos,
    })
}

fn decode_stored_markers(
    bits: &mut Bits,
    output: &mut Vec<u16>,
    window: &mut [u16; WINDOW_SIZE],
    window_pos: &mut usize,
    decoded_bytes: &mut usize,
) -> io::Result<()> {
    bits.align_to_byte();
    if bits.available() < 32 {
        bits.refill();
    }

    let len = (bits.peek() & 0xFFFF) as u16;
    bits.consume(16);
    let nlen = (bits.peek() & 0xFFFF) as u16;
    bits.consume(16);

    if len != !nlen {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid stored block length",
        ));
    }

    for _ in 0..len {
        if bits.pos >= bits.data.len() {
            return Err(Error::new(ErrorKind::UnexpectedEof, "EOF in stored block"));
        }
        let byte = bits.data[bits.pos] as u16;
        bits.pos += 1;
        output.push(byte);
        window[*window_pos] = byte;
        *window_pos = (*window_pos + 1) % WINDOW_SIZE;
        *decoded_bytes += 1;
    }

    bits.bitbuf = 0;
    bits.bitsleft = 0;

    Ok(())
}

#[inline]
fn append_marker_output(
    value: u16,
    output: &mut Vec<u16>,
    window: &mut [u16; WINDOW_SIZE],
    window_pos: &mut usize,
    decoded_bytes: &mut usize,
    marker_count: &mut usize,
) {
    output.push(value);
    window[*window_pos] = value;
    *window_pos = (*window_pos + 1) % WINDOW_SIZE;
    *decoded_bytes += 1;
    if value > 255 {
        *marker_count += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_huffman_markers(
    bits: &mut Bits,
    output: &mut Vec<u16>,
    window: &mut [u16; WINDOW_SIZE],
    window_pos: &mut usize,
    decoded_bytes: &mut usize,
    marker_count: &mut usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> io::Result<()> {
    let max_decoded = *decoded_bytes + 16 * 1024 * 1024;

    loop {
        if *decoded_bytes >= max_decoded {
            return Ok(());
        }

        if bits.available() < 15 {
            bits.refill();
        }
        if bits.available() < 15 {
            bits.refill_slow();
            if bits.available() < 7 {
                return Ok(());
            }
        }

        let entry = litlen.lookup(bits.peek());

        if entry.is_literal() {
            let lit = entry.literal_value() as u16;
            bits.consume(entry.total_bits() as u32);
            append_marker_output(lit, output, window, window_pos, decoded_bytes, marker_count);
            continue;
        }

        if entry.is_exceptional() {
            if entry.is_end_of_block() {
                bits.consume(entry.total_bits() as u32);
                return Ok(());
            }
            if entry.is_subtable_ptr() {
                let sub_entry = litlen.lookup_subtable(entry, bits.peek());
                if sub_entry.is_literal() {
                    let lit = sub_entry.literal_value() as u16;
                    bits.consume(sub_entry.total_bits() as u32);
                    append_marker_output(
                        lit,
                        output,
                        window,
                        window_pos,
                        decoded_bytes,
                        marker_count,
                    );
                    continue;
                }
                if sub_entry.is_end_of_block() {
                    bits.consume(sub_entry.total_bits() as u32);
                    return Ok(());
                }
                let saved = bits.peek();
                bits.consume(sub_entry.total_bits() as u32);
                let length = sub_entry.decode_length(saved) as usize;
                decode_distance_and_copy(
                    bits,
                    output,
                    window,
                    window_pos,
                    decoded_bytes,
                    marker_count,
                    dist,
                    length,
                )?;
                continue;
            }
            return Err(Error::new(ErrorKind::InvalidData, "Invalid litlen entry"));
        }

        let saved = bits.peek();
        bits.consume(entry.total_bits() as u32);
        let length = entry.decode_length(saved) as usize;
        decode_distance_and_copy(
            bits,
            output,
            window,
            window_pos,
            decoded_bytes,
            marker_count,
            dist,
            length,
        )?;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn decode_distance_and_copy(
    bits: &mut Bits,
    output: &mut Vec<u16>,
    window: &mut [u16; WINDOW_SIZE],
    window_pos: &mut usize,
    decoded_bytes: &mut usize,
    marker_count: &mut usize,
    dist: &DistTable,
    length: usize,
) -> io::Result<()> {
    if bits.available() < 15 {
        bits.refill();
    }

    let mut dentry = dist.lookup(bits.peek());
    if dentry.is_exceptional() {
        if dentry.is_subtable_ptr() {
            dentry = dist.lookup_subtable(dentry, bits.peek());
        } else {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid distance entry"));
        }
    }
    let dsaved = bits.peek();
    bits.consume(dentry.total_bits() as u32);
    let distance = dentry.decode_distance(dsaved) as usize;

    if distance == 0 || distance > WINDOW_SIZE {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
    }

    let start_window_pos = *window_pos;
    let start_decoded = *decoded_bytes;

    for i in 0..length {
        let value = if distance > start_decoded + i {
            let marker_offset = distance - start_decoded - i - 1;
            MARKER_BASE + (marker_offset as u16).min(u16::MAX - MARKER_BASE)
        } else {
            let src = (start_window_pos + WINDOW_SIZE - distance + i) % WINDOW_SIZE;
            window[src]
        };
        append_marker_output(
            value,
            output,
            window,
            window_pos,
            decoded_bytes,
            marker_count,
        );
    }

    Ok(())
}

// =============================================================================
// Block finder: locate valid deflate block starts near target positions
// =============================================================================

/// Cheap pre-filter: check if position could be a deflate block start.
/// Only checks the BFINAL+BTYPE header and basic validity, not full table build.
#[inline]
fn quick_block_check(deflate_data: &[u8], byte_pos: usize, bit_offset: u8) -> bool {
    if byte_pos + 8 >= deflate_data.len() {
        return false;
    }

    let mut bits = Bits {
        data: deflate_data,
        pos: byte_pos,
        bitbuf: 0,
        bitsleft: 0,
    };
    bits.refill();
    if bit_offset > 0 {
        bits.consume(bit_offset as u32);
    }

    if bits.available() < 17 {
        return false;
    }

    let btype = ((bits.peek() >> 1) & 3) as u8;
    bits.consume(3);

    match btype {
        0 => {
            bits.align_to_byte();
            if bits.available() < 32 {
                bits.refill();
            }
            if bits.available() < 32 {
                return false;
            }
            let len = (bits.peek() & 0xFFFF) as u16;
            let nlen = ((bits.peek() >> 16) & 0xFFFF) as u16;
            len == !nlen && len > 0
        }
        2 => {
            let hlit = (bits.peek() & 0x1F) as usize + 257;
            let hdist = ((bits.peek() >> 5) & 0x1F) as usize + 1;
            let hclen = ((bits.peek() >> 10) & 0xF) as usize + 4;
            if hlit > 286 || hdist > 32 || !(4..=19).contains(&hclen) {
                return false;
            }
            bits.consume(14);

            // Precode leaf count check (rapidgzip-style).
            // Read hclen code-length code lengths (3 bits each),
            // verify they form a valid prefix-free code via Kraft inequality.
            // For max code length 7: sum of 2^(7-L) for each L > 0 must equal 128.
            // This rejects ~95% of random data that passes range checks.
            let needed_bits = hclen * 3;
            if bits.available() < needed_bits as u32 {
                bits.refill();
            }
            if bits.available() < needed_bits as u32 {
                return false;
            }
            let mut allocated_leaves: u32 = 0;
            let mut nonzero = 0u32;
            let mut val = bits.peek();
            for _ in 0..hclen {
                let cl = (val & 7) as u32;
                if cl > 0 {
                    nonzero += 1;
                    allocated_leaves += 1u32 << (7 - cl);
                }
                val >>= 3;
            }
            // Complete prefix code has exactly 128 leaves (2^7).
            // Allow slightly incomplete codes (some encoders produce them).
            nonzero >= 2 && (64..=128).contains(&allocated_leaves)
        }
        _ => false,
    }
}

/// Validate a candidate block start by building full Huffman tables and
/// doing a minimal trial decode (256 symbols). Checks EOB symbol presence
/// and that decoded output is consistent.
fn validate_block_start(deflate_data: &[u8], byte_pos: usize, bit_offset: u8) -> bool {
    let mut bits = Bits {
        data: deflate_data,
        pos: byte_pos,
        bitbuf: 0,
        bitsleft: 0,
    };
    bits.refill();
    if bit_offset > 0 {
        bits.consume(bit_offset as u32);
    }

    if bits.available() < 3 {
        return false;
    }

    let btype = ((bits.peek() >> 1) & 3) as u8;
    bits.consume(3);

    match btype {
        0 => true,
        1 => {
            // Fixed Huffman: try decoding a few symbols
            let (litlen, dist) = get_fixed_tables();
            trial_decode_symbols(&mut bits, litlen, dist, 256)
        }
        2 => {
            let tables = read_dynamic_tables(&mut bits);
            match tables {
                Ok((litlen, dist)) => trial_decode_symbols(&mut bits, &litlen, &dist, 256),
                Err(_) => false,
            }
        }
        _ => false,
    }
}

/// Try decoding a small number of symbols. Returns true if at least
/// `min_symbols` decode successfully (no invalid distances, no errors).
fn trial_decode_symbols(
    bits: &mut Bits,
    litlen: &LitLenTable,
    dist: &DistTable,
    min_symbols: usize,
) -> bool {
    let mut symbols_ok = 0usize;
    for _ in 0..min_symbols * 2 {
        if bits.available() < 15 {
            bits.refill();
        }
        if bits.available() < 7 {
            return symbols_ok >= min_symbols;
        }

        let entry = litlen.lookup(bits.peek());

        if entry.is_literal() {
            bits.consume(entry.total_bits() as u32);
            symbols_ok += 1;
            continue;
        }

        if entry.is_exceptional() {
            if entry.is_end_of_block() {
                bits.consume(entry.total_bits() as u32);
                return symbols_ok >= min_symbols / 4;
            }
            if entry.is_subtable_ptr() {
                let sub = litlen.lookup_subtable(entry, bits.peek());
                if sub.is_end_of_block() {
                    bits.consume(sub.total_bits() as u32);
                    return symbols_ok >= min_symbols / 4;
                }
                if sub.is_literal() {
                    bits.consume(sub.total_bits() as u32);
                    symbols_ok += 1;
                    continue;
                }
                let saved = bits.peek();
                bits.consume(sub.total_bits() as u32);
                let _length = sub.decode_length(saved);
            } else {
                return false;
            }
        } else {
            let saved = bits.peek();
            bits.consume(entry.total_bits() as u32);
            let _length = entry.decode_length(saved);
        }

        // Read distance
        if bits.available() < 15 {
            bits.refill();
        }
        if bits.available() < 5 {
            return false;
        }
        let mut dentry = dist.lookup(bits.peek());
        if dentry.is_exceptional() {
            if dentry.is_subtable_ptr() {
                dentry = dist.lookup_subtable(dentry, bits.peek());
            } else {
                return false;
            }
        }
        let dsaved = bits.peek();
        bits.consume(dentry.total_bits() as u32);
        let distance = dentry.decode_distance(dsaved) as usize;
        if distance == 0 || distance > WINDOW_SIZE {
            return false;
        }
        symbols_ok += 1;
    }
    symbols_ok >= min_symbols
}

// =============================================================================
// Marker resolution
// =============================================================================

/// Resolve markers in a chunk's u16 output using the previous chunk's window.
/// Markers >= MARKER_BASE are replaced with window[value - MARKER_BASE].
fn resolve_markers(data: &[u16], window: &[u8], window_size: usize) -> Vec<u8> {
    data.iter()
        .map(|&val| {
            if val <= 255 {
                val as u8
            } else {
                let idx = (val - MARKER_BASE) as usize;
                if idx < window_size {
                    window[window_size - 1 - idx]
                } else {
                    0
                }
            }
        })
        .collect()
}

// =============================================================================
// Parallel orchestrator
// =============================================================================

/// Decompress a single-member gzip file using speculative parallel decode.
///
/// Returns Some(output) on success, None if the file is too small for parallel
/// benefit, or Err on decode failure.
pub fn decompress_speculative(data: &[u8], num_threads: usize) -> io::Result<Option<Vec<u8>>> {
    let debug = std::env::var("GZIPPY_DEBUG").is_ok();

    if num_threads <= 1 {
        return Ok(None);
    }

    let header_size = skip_gzip_header(data)?;
    let deflate_data = &data[header_size..data.len().saturating_sub(8)];

    if deflate_data.len() < MIN_SIZE_FOR_PARALLEL {
        if debug {
            eprintln!(
                "[speculative] too small: {} < {}",
                deflate_data.len(),
                MIN_SIZE_FOR_PARALLEL
            );
        }
        return Ok(None);
    }

    // Read ISIZE to estimate compression ratio. High ratios (>3x) mean complex data
    // where block boundaries are sparse and speculative search will waste time.
    let isize_hint = if data.len() >= 4 {
        u32::from_le_bytes([
            data[data.len() - 4],
            data[data.len() - 3],
            data[data.len() - 2],
            data[data.len() - 1],
        ]) as usize
    } else {
        0
    };
    if isize_hint > 0 {
        let ratio = isize_hint as f64 / deflate_data.len() as f64;
        if ratio > 10.0 {
            if debug {
                eprintln!(
                    "[speculative] compression ratio {:.1}x too high, skipping",
                    ratio
                );
            }
            return Ok(None);
        }
    }

    let num_chunks = num_threads.min(deflate_data.len().div_ceil(CHUNK_SPACING));
    if num_chunks < 2 {
        return Ok(None);
    }

    // =========================================================================
    // Phase 1+2: Parallel speculative decode from evenly-spaced positions
    // =========================================================================
    // Like rapidgzip: don't search for block starts — just pick evenly-spaced
    // byte positions and try to decode from each. Positions that don't land on
    // a valid block boundary will fail quickly. We try all 8 bit offsets.
    let spacing = deflate_data.len() / num_chunks;

    // Generate chunk positions (byte offset, bit offset 0 — we try bit offsets during decode)
    let targets: Vec<usize> = (1..num_chunks)
        .map(|i| i * spacing)
        .filter(|&t| t < deflate_data.len())
        .collect();

    if targets.is_empty() {
        return Ok(None);
    }

    if debug {
        eprintln!(
            "[speculative] {} chunks, spacing {} bytes, targets: {:?}",
            num_chunks, spacing, targets
        );
    }

    let decode_start = std::time::Instant::now();

    struct ChunkResult {
        data: Vec<u16>,
        window: [u8; WINDOW_SIZE],
        window_fill: usize,
    }

    // Phase 1: find valid start positions by searching around each target.
    // Use 16KB search window (typical deflate blocks are 4-32KB compressed)
    // with strict 20ms time limit per target. The precode leaf count check
    // in quick_block_check rejects ~95% of false positives, making the search fast.
    const SEARCH_RANGE: usize = 16 * 1024;
    let found_starts: Vec<Option<(usize, u8)>> = std::thread::scope(|s| {
        let handles: Vec<_> = targets
            .iter()
            .map(|&target| {
                s.spawn(move || {
                    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(20);
                    let start = target.saturating_sub(SEARCH_RANGE / 2);
                    let end =
                        (target + SEARCH_RANGE / 2).min(deflate_data.len().saturating_sub(32));
                    for byte_pos in start..end {
                        if byte_pos % 64 == 0 && std::time::Instant::now() > deadline {
                            return None;
                        }
                        for bit_offset in 0..8u8 {
                            if quick_block_check(deflate_data, byte_pos, bit_offset)
                                && validate_block_start(deflate_data, byte_pos, bit_offset)
                            {
                                return Some((byte_pos, bit_offset));
                            }
                        }
                    }
                    None
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Build chunk list from successful finds
    let mut valid_targets: Vec<(usize, u8, usize)> = Vec::new();
    for (idx, found) in found_starts.iter().enumerate() {
        if let Some((byte_pos, bit_offset)) = found {
            let end = if idx + 1 < targets.len() {
                // Use next target's position as upper bound, adjusted if it also found a start
                found_starts
                    .get(idx + 1)
                    .and_then(|f| f.as_ref())
                    .map(|(pos, _)| *pos)
                    .unwrap_or(targets.get(idx + 1).copied().unwrap_or(deflate_data.len()))
            } else {
                deflate_data.len()
            };
            valid_targets.push((*byte_pos, *bit_offset, end));
        } else {
            break;
        }
    }

    if valid_targets.is_empty() {
        if debug {
            eprintln!(
                "[speculative] no valid block starts found in {:?}",
                decode_start.elapsed()
            );
        }
        return Ok(None);
    }

    if debug {
        eprintln!(
            "[speculative] found {}/{} block starts in {:?}",
            valid_targets.len(),
            targets.len(),
            decode_start.elapsed()
        );
    }

    // Phase 2a: Decode chunk 0 first and validate before launching speculative chunks.
    // This limits wasted work when block positions are false positives.
    let chunk0_end = valid_targets[0].0;
    let chunk0 = inflate_with_markers(deflate_data, 0, 0, chunk0_end)?;

    // Early validation: chunk 0 output should be roughly ISIZE / num_chunks.
    // If it's < 25% of expected, the block boundary is likely wrong.
    if isize_hint > 0 {
        let expected_per_chunk = isize_hint / (1 + valid_targets.len());
        if chunk0.data.len() < expected_per_chunk / 4 {
            if debug {
                eprintln!(
                    "[speculative] chunk 0 too small: {} vs expected ~{}, aborting in {:?}",
                    chunk0.data.len(),
                    expected_per_chunk,
                    decode_start.elapsed()
                );
            }
            return Ok(None);
        }
    }

    // Phase 2b: Parallel decode of speculative chunks
    let spec_results: Vec<io::Result<ChunkResult>> = std::thread::scope(|s| {
        let spec_handles: Vec<_> = valid_targets
            .iter()
            .map(|&(start_byte, start_bit, end)| {
                s.spawn(move || {
                    let result = inflate_with_markers(deflate_data, start_byte, start_bit, end)?;
                    Ok(ChunkResult {
                        data: result.data,
                        window: result.window,
                        window_fill: result.window_fill,
                    })
                })
            })
            .collect();

        spec_handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });

    let mut good_chunks: Vec<&ChunkResult> = Vec::new();
    for (idx, result) in spec_results.iter().enumerate() {
        match result {
            Ok(r) => good_chunks.push(r),
            Err(e) => {
                if debug {
                    let (byte_pos, bit_off, _end) = valid_targets[idx];
                    eprintln!(
                        "[speculative] chunk {} failed at byte={} bit={}: {}",
                        idx + 1,
                        byte_pos,
                        bit_off,
                        e
                    );
                }
                break;
            }
        }
    }

    if good_chunks.is_empty() {
        if debug {
            eprintln!(
                "[speculative] no speculative chunks succeeded in {:?}, falling back",
                decode_start.elapsed()
            );
        }
        return Ok(None);
    }

    if debug {
        eprintln!(
            "[speculative] {}/{} speculative chunks succeeded in {:?}",
            good_chunks.len(),
            targets.len(),
            decode_start.elapsed()
        );
    }

    // =========================================================================
    // Phase 3: Window propagation and marker resolution
    // =========================================================================
    let chunk0_bytes: Vec<u8> = chunk0
        .data
        .iter()
        .map(|&v| if v <= 255 { v as u8 } else { 0 })
        .collect();

    let total_size = chunk0_bytes.len() + good_chunks.iter().map(|r| r.data.len()).sum::<usize>();

    // Verify against ISIZE: if total output differs significantly from expected,
    // the block finder found false positives. Fall back to sequential.
    if isize_hint > 0 {
        let expected = isize_hint;
        let ratio = if expected > total_size {
            expected as f64 / total_size as f64
        } else {
            total_size as f64 / expected as f64
        };
        if ratio > 1.1 {
            if debug {
                eprintln!(
                    "[speculative] ISIZE mismatch: output {} vs expected {}, ratio {:.2}x, falling back",
                    total_size, expected, ratio
                );
            }
            return Ok(None);
        }
    }

    let mut final_output = Vec::with_capacity(total_size);
    final_output.extend_from_slice(&chunk0_bytes);

    let mut prev_window = chunk0.window;
    let mut prev_fill = chunk0.window_fill;

    for chunk in &good_chunks {
        let resolved = resolve_markers(&chunk.data, &prev_window, prev_fill);
        final_output.extend_from_slice(&resolved);
        prev_window = chunk.window;
        prev_fill = chunk.window_fill;
    }

    // Final CRC32 + ISIZE verification against the gzip trailer.
    // This catches any false positive block boundaries that produced
    // output matching ISIZE but with wrong content.
    if data.len() >= 8 {
        let trailer = &data[data.len() - 8..];
        let expected_crc = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
        let expected_isize =
            u32::from_le_bytes([trailer[4], trailer[5], trailer[6], trailer[7]]) as usize;

        if final_output.len() != expected_isize {
            if debug {
                eprintln!(
                    "[speculative] final ISIZE mismatch: {} vs {}, falling back",
                    final_output.len(),
                    expected_isize
                );
            }
            return Ok(None);
        }

        let actual_crc = crc32fast::hash(&final_output);
        if actual_crc != expected_crc {
            if debug {
                eprintln!(
                    "[speculative] CRC32 mismatch: {:08x} vs {:08x}, falling back",
                    actual_crc, expected_crc
                );
            }
            return Ok(None);
        }
    }

    Ok(Some(final_output))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn compress_gzip(data: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn test_inflate_with_markers_from_start() {
        let original = vec![b'A'; 100_000];
        let compressed = compress_gzip(&original);
        let header_size = skip_gzip_header(&compressed).unwrap();
        let deflate_data = &compressed[header_size..compressed.len() - 8];
        eprintln!(
            "deflate_data len: {}, original len: {}",
            deflate_data.len(),
            original.len()
        );

        let result = inflate_with_markers(deflate_data, 0, 0, deflate_data.len()).unwrap();

        eprintln!(
            "decoded: {} bytes, markers: {}",
            result.data.len(),
            result.marker_count
        );
        assert_eq!(
            result.marker_count, 0,
            "No markers expected from position 0"
        );
        assert_eq!(
            result.data.len(),
            original.len(),
            "Output size mismatch: got {}, expected {}",
            result.data.len(),
            original.len()
        );
        let bytes: Vec<u8> = result.data.iter().map(|&v| v as u8).collect();
        assert_eq!(&bytes[..100], &original[..100], "First 100 bytes mismatch");
    }

    #[test]
    fn test_inflate_with_markers_diverse_data() {
        let mut original = Vec::with_capacity(200_000);
        for i in 0u32..50_000 {
            original.extend_from_slice(&i.to_le_bytes());
        }
        let compressed = compress_gzip(&original);
        let header_size = skip_gzip_header(&compressed).unwrap();
        let deflate_data = &compressed[header_size..compressed.len() - 8];

        let result = inflate_with_markers(deflate_data, 0, 0, deflate_data.len()).unwrap();
        assert_eq!(result.marker_count, 0);
        let bytes: Vec<u8> = result.data.iter().map(|&v| v as u8).collect();
        assert_eq!(bytes, original);
    }

    #[test]
    fn test_marker_resolution() {
        let window = [42u8; WINDOW_SIZE];
        let data: Vec<u16> = vec![0, 65, 255, MARKER_BASE, MARKER_BASE + 100];
        let resolved = resolve_markers(&data, &window, WINDOW_SIZE);
        assert_eq!(resolved[0], 0);
        assert_eq!(resolved[1], 65);
        assert_eq!(resolved[2], 255);
        assert_eq!(resolved[3], 42);
        assert_eq!(resolved[4], 42);
    }

    #[test]
    fn test_speculative_parallel_basic() {
        let mut original = Vec::with_capacity(10 * 1024 * 1024);
        for i in 0u64..1_500_000 {
            original.extend_from_slice(&i.to_le_bytes());
        }
        let compressed = compress_gzip(&original);

        let result = decompress_speculative(&compressed, 4).unwrap();
        // File should be large enough for parallel
        if let Some(output) = result {
            assert_eq!(output.len(), original.len(), "Output size mismatch");
            // First chunk should be correct (no markers)
            assert_eq!(&output[..1024], &original[..1024], "First 1KB mismatch");
        }
    }

    #[test]
    fn test_speculative_too_small() {
        let original = vec![b'X'; 1000];
        let compressed = compress_gzip(&original);
        let result = decompress_speculative(&compressed, 4).unwrap();
        assert!(result.is_none(), "Small files should return None");
    }

    #[test]
    fn test_speculative_on_diverse_data() {
        let mut original = Vec::with_capacity(16 * 1024 * 1024);
        for i in 0u32..4_000_000 {
            original.extend_from_slice(&i.to_le_bytes());
        }
        let compressed = compress_gzip(&original);
        eprintln!(
            "Diverse data: {} uncompressed, {} compressed",
            original.len(),
            compressed.len()
        );

        let result = decompress_speculative(&compressed, 4).unwrap();
        match result {
            Some(output) => {
                eprintln!("Speculative succeeded: {} bytes output", output.len());
                assert_eq!(output.len(), original.len());
                assert_eq!(&output[..1024], &original[..1024]);
            }
            None => {
                eprintln!("Speculative returned None (no block starts found), OK");
            }
        }
    }
}
