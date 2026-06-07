//! Encapsulated inner-DEFLATE-decode microbench (2026-05-28).
//!
//! Isolates the inner inflate loop from the parallel pipeline / markers /
//! CRC / page-fault churn so it can be measured with EXACT, noise-free
//! single-threaded perf counters (instructions/byte, cycles/byte) on any
//! core with no container freeze required.
//!
//! Decodes one raw-DEFLATE stream (silesia, gzip header/trailer stripped)
//! repeatedly through ONE decoder chosen by argv, so `perf stat` attributes
//! cleanly:
//!
//!   perf stat -e instructions,cpu_core/cycles/ -- inner_bench pure 5
//!   perf stat -e instructions,cpu_core/cycles/ -- inner_bench libdeflate 5
//!
//! Then instructions/byte = instructions / (iters * output_len).
use std::time::Instant;

fn load_raw_deflate(path: &str) -> Vec<u8> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    // Strip the gzip header (parse) + 8-byte trailer to get raw DEFLATE.
    let (_h, header) =
        gzippy::decompress::parallel::gzip_format::read_header(&data).expect("parse gzip header");
    data[header..data.len() - 8].to_vec()
}

#[cfg(feature = "pure-rust-inflate")]
fn decode_pure(deflate: &[u8], out: &mut [u8]) -> usize {
    use gzippy::decompress::inflate::resumable::ResumableInflate2;
    let mut d = ResumableInflate2::with_until_bits(deflate, 0, deflate.len() * 8).expect("init");
    d.set_window(&[]).expect("window");
    let mut total = 0usize;
    loop {
        let r = d.read_stream(&mut out[total..]).expect("read");
        total += r.bytes_written;
        if r.finished || r.bytes_written == 0 {
            break;
        }
    }
    total
}

fn decode_libdeflate(deflate: &[u8], out: &mut [u8]) -> usize {
    let mut d = libdeflater::Decompressor::new();
    d.deflate_decompress(deflate, out).expect("libdeflate")
}

// In-repo libdeflate-STYLE pure-Rust decoder (NON-resumable). The A/B that
// isolates whether ResumableInflate2's 2x-vs-libdeflate instruction gap is
// the RESUMABLE CONTRACT (yield checks / pending_match / snapshots) or the
// core algorithm: if this is ~libdeflate-fast, it's the contract.
#[cfg(feature = "pure-rust-inflate")]
fn decode_consume_first(deflate: &[u8], out: &mut [u8]) -> usize {
    gzippy::decompress::inflate::consume_first_decode::inflate_consume_first(deflate, out)
        .expect("consume_first")
}

// Window-absent BOOTSTRAP decoder (`marker_inflate::Block`) — the marker-path
// inner loop that decodes ~31% of silesia (the speculative window-absent
// chunks) at ~175 MB/s in production. Decoding from offset 0 with no window
// exercises the SAME per-symbol decode (LutLitLenCode litlen +
// get_distance_dynamic/apply_distance_extra + u16-ring writes) that runs on
// the marker path, isolating its instructions/byte from the parallel pipeline.
// This is the measurement gate for the bootstrap-unification lever.
#[cfg(feature = "pure-rust-inflate")]
fn decode_bootstrap(deflate: &[u8], out16: &mut Vec<u16>) -> usize {
    use gzippy::decompress::inflate::consume_first_decode::Bits;
    use gzippy::decompress::parallel::marker_inflate::Block;
    let mut block = Block::new();
    block.reset(None, None);
    let mut bits = Bits::new(deflate);
    out16.clear();
    loop {
        if block.read_header(&mut bits, false).is_err() {
            break;
        }
        while !block.eob() {
            if block.read(&mut bits, out16, usize::MAX).is_err() {
                return out16.len();
            }
        }
        if block.is_last_block() {
            break;
        }
    }
    out16.len()
}

// ── Candidate-discriminating decoders ─────────────────────────────────────
//
// These two modes replicate the PRODUCTION window-absent per-symbol loop
// (`deflate_block.rs::run_multi_cached_loop!`) — same LutLitLenCode
// litlen decode, same HuffmanCodingReversedBitsCached dist decode, same
// `get_distance_dynamic`, same multi-symbol unpack — but vary ONLY the store
// backend, so the A/B isolates the three candidates:
//
//   tbl_isal_u8  (A): ISA-L table + FLAT u8 store + flat match-copy, NO ring,
//                     NO markers, NO 64KiB re-entry (one straight pass).
//   tbl_isal_u16 (D): ISA-L table + u16 RING store + ring match-copy + the
//                     RING_SIZE-258 re-entry cap, but NO markers
//                     (CONTAINS_MARKERS=false equivalent: no backward scan).
//
// Then:
//   A vs `consume_first` (B)  → candidate (i) TABLE PRIMITIVE
//   A vs D                    → candidate (ii) u16 STORE + ring modulo + reentry
//   D vs `bootstrap` (C)      → candidate (iii)/marker bookkeeping residue
/// Flat match-copy into a u8 buffer (overlap-correct, no SIMD slop). Shared by
/// neither A nor D for the store comparison to be apples-to-apples we use a
/// matching ring/flat copy in each; this is A's flat copy.
#[cfg(feature = "pure-rust-inflate")]
#[inline(always)]
fn flat_copy_u8(out: &mut [u8], out_pos: usize, distance: usize, length: usize) {
    let src = out_pos - distance;
    if distance >= length {
        out.copy_within(src..src + length, out_pos);
    } else if distance == 1 {
        let b = out[src];
        out[out_pos..out_pos + length].fill(b);
    } else {
        for i in 0..length {
            out[out_pos + i] = out[src + (i % distance)];
        }
    }
}

/// A: ISA-L table + flat u8, no ring, no markers, single pass.
#[cfg(feature = "pure-rust-inflate")]
fn decode_tbl_isal_u8(deflate: &[u8], out: &mut [u8]) -> usize {
    use gzippy::decompress::inflate::consume_first_decode::Bits;
    use gzippy::decompress::parallel::huffman_reversed_bits_cached::HuffmanCodingReversedBitsCached;
    use gzippy::decompress::parallel::lut_huffman::LutLitLenCode;
    use gzippy::decompress::parallel::marker_inflate::{Block, END_OF_BLOCK_SYMBOL};
    use gzippy::decompress::parallel::rfc_tables::get_distance_dynamic;

    const MULTI_DISTANCE_OFFSET: u32 = 254;
    let mut bits = Bits::new(deflate);
    let mut header = Block::new();
    header.reset(None, None);
    let mut out_pos = 0usize;
    let mut litlen_hc = LutLitLenCode::new_empty();

    loop {
        if header.read_header(&mut bits, false).is_err() {
            break;
        }
        // Only DYNAMIC blocks exercise the ISA-L litlen table (matches prod).
        let lit = header.literal_cl[..header.literal_code_count].to_vec();
        let dist = header.literal_cl
            [header.literal_code_count..header.literal_code_count + header.distance_code_count]
            .to_vec();
        if !litlen_hc.rebuild_from(&lit) {
            break;
        }
        let mut dist_hc: HuffmanCodingReversedBitsCached<30> =
            HuffmanCodingReversedBitsCached::new();
        if dist_hc.initialize_from_lengths(&dist, false)
            != gzippy::decompress::parallel::error::Error::None
        {
            break;
        }
        let mut at_eob = false;
        while !at_eob {
            let decoded = litlen_hc.decode(&mut bits);
            if decoded.bit_count == 0 {
                return out_pos;
            }
            bits.consume(decoded.bit_count);
            let mut symbol = decoded.symbol;
            let mut symbol_count = decoded.sym_count;
            while symbol_count > 0 {
                let code = (symbol & 0xFFFF) as u16;
                if code <= 255 || symbol_count > 1 {
                    out[out_pos] = (code & 0xFF) as u8;
                    out_pos += 1;
                    symbol >>= 8;
                    symbol_count -= 1;
                    continue;
                }
                if code == END_OF_BLOCK_SYMBOL {
                    at_eob = true;
                    break;
                }
                let length = (symbol - MULTI_DISTANCE_OFFSET) as usize;
                if length == 0 {
                    symbol >>= 8;
                    symbol_count -= 1;
                    continue;
                }
                let distance = match get_distance_dynamic(&dist_hc, &mut bits) {
                    Ok(d) => d as usize,
                    Err(_) => return out_pos,
                };
                flat_copy_u8(out, out_pos, distance, length);
                out_pos += length;
                symbol >>= 8;
                symbol_count -= 1;
            }
        }
        if header.is_last_block() {
            break;
        }
    }
    out_pos
}

/// D: ISA-L table + u16 ring store + ring match-copy + RING_SIZE-258 re-entry
/// cap, NO markers (CONTAINS_MARKERS=false: no backward scan, no distance_marker).
/// Output is drained to a u8 buffer at each re-entry to mimic the production
/// per-call drain that the cap forces.
#[cfg(feature = "pure-rust-inflate")]
fn decode_tbl_isal_u16(deflate: &[u8], _out: &mut [u8]) -> usize {
    use gzippy::decompress::inflate::consume_first_decode::Bits;
    use gzippy::decompress::parallel::huffman_reversed_bits_cached::HuffmanCodingReversedBitsCached;
    use gzippy::decompress::parallel::lut_huffman::LutLitLenCode;
    use gzippy::decompress::parallel::marker_inflate::{Block, END_OF_BLOCK_SYMBOL};
    use gzippy::decompress::parallel::rfc_tables::get_distance_dynamic;

    const MULTI_DISTANCE_OFFSET: u32 = 254;
    const RING_SIZE: usize = 1 << 17; // 128 KiB u16 ring (prod ring is this order)
    const MAX_RUN_LENGTH: usize = 258;
    const N_MAX: usize = RING_SIZE - MAX_RUN_LENGTH; // per-call decode cap (re-entry cadence)

    let mut bits = Bits::new(deflate);
    let mut header = Block::new();
    header.reset(None, None);
    let mut ring: Vec<u16> = vec![0u16; RING_SIZE];
    let ring_ptr = ring.as_mut_ptr();
    let mut pos: usize = 0; // monotonic logical
    let mut out_pos = 0usize;
    let mut litlen_hc = LutLitLenCode::new_empty();

    #[inline(always)]
    unsafe fn ring_backref(ring_ptr: *mut u16, pos: usize, distance: usize, length: usize) {
        let src_phys = (pos + RING_SIZE - distance) % RING_SIZE;
        let dst_phys = pos % RING_SIZE;
        if distance >= length && src_phys + length <= RING_SIZE && dst_phys + length <= RING_SIZE {
            let src = ring_ptr.add(src_phys);
            let dst = ring_ptr.add(dst_phys);
            let mut i = 0usize;
            while i + 8 <= length {
                std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), 8);
                i += 8;
            }
            while i < length {
                dst.add(i).write(*src.add(i));
                i += 1;
            }
        } else if distance == 1 {
            let v = *ring_ptr.add((pos + RING_SIZE - 1) % RING_SIZE);
            for i in 0..length {
                ring_ptr.add((dst_phys + i) % RING_SIZE).write(v);
            }
        } else {
            for i in 0..length {
                let v = *ring_ptr.add((src_phys + i) % RING_SIZE);
                ring_ptr.add((dst_phys + i) % RING_SIZE).write(v);
            }
        }
    }

    loop {
        if header.read_header(&mut bits, false).is_err() {
            break;
        }
        let lit = header.literal_cl[..header.literal_code_count].to_vec();
        let dist = header.literal_cl
            [header.literal_code_count..header.literal_code_count + header.distance_code_count]
            .to_vec();
        if !litlen_hc.rebuild_from(&lit) {
            break;
        }
        let mut dist_hc: HuffmanCodingReversedBitsCached<30> =
            HuffmanCodingReversedBitsCached::new();
        if dist_hc.initialize_from_lengths(&dist, false)
            != gzippy::decompress::parallel::error::Error::None
        {
            break;
        }
        let mut at_eob = false;
        while !at_eob {
            // Per-call re-entry cap: decode at most N_MAX then "drain" (mimics
            // production read() boundary). We drain by advancing a logical
            // counter and resetting the emitted budget — the cap RE-RUNS the
            // setup (table refs reloaded, pos resynced) every ~128 KiB.
            let mut emitted = 0usize;
            let mut phys: u16 = (pos & (RING_SIZE - 1)) as u16;
            while emitted < N_MAX {
                let decoded = litlen_hc.decode(&mut bits);
                if decoded.bit_count == 0 {
                    return out_pos;
                }
                bits.consume(decoded.bit_count);
                let mut symbol = decoded.symbol;
                let mut symbol_count = decoded.sym_count;
                while symbol_count > 0 {
                    let code = (symbol & 0xFFFF) as u16;
                    if code <= 255 || symbol_count > 1 {
                        unsafe {
                            ring_ptr.add(phys as usize).write((code & 0xFF));
                        }
                        phys = phys.wrapping_add(1);
                        pos += 1;
                        emitted += 1;
                        out_pos += 1;
                        symbol >>= 8;
                        symbol_count -= 1;
                        continue;
                    }
                    if code == END_OF_BLOCK_SYMBOL {
                        at_eob = true;
                        break;
                    }
                    let length = (symbol - MULTI_DISTANCE_OFFSET) as usize;
                    if length == 0 {
                        symbol >>= 8;
                        symbol_count -= 1;
                        continue;
                    }
                    let distance = match get_distance_dynamic(&dist_hc, &mut bits) {
                        Ok(d) => d as usize,
                        Err(_) => return out_pos,
                    };
                    unsafe {
                        ring_backref(ring_ptr, pos, distance, length);
                    }
                    pos += length;
                    phys = (pos & (RING_SIZE - 1)) as u16;
                    emitted += length;
                    out_pos += length;
                    symbol >>= 8;
                    symbol_count -= 1;
                }
                if at_eob {
                    break;
                }
            }
            // (re-entry boundary — loop top reloads phys from pos)
        }
        if header.is_last_block() {
            break;
        }
    }
    // Touch the ring so the writes aren't dead-code-eliminated.
    std::hint::black_box(&ring);
    out_pos
}

/// E: CLEAN table (LitLenTable+DistTable) but ONE-literal-at-a-time into flat
/// u8 (NO multi-literal packing). Isolates the TABLE PRIMITIVE from the
/// packed-store optimization. If E ≈ B (consume_first), the win is the table
/// format/critical-path and is PORTABLE to the u16 ring. If E ≈ A (isal), the
/// win is the packing and is NOT portable to a ring.
#[cfg(feature = "pure-rust-inflate")]
fn decode_clean_u8_nopack(deflate: &[u8], out: &mut [u8]) -> usize {
    use gzippy::decompress::inflate::consume_first_decode::Bits;
    use gzippy::decompress::inflate::libdeflate_entry::{DistTable, LitLenTable};
    use gzippy::decompress::parallel::marker_inflate::Block;

    let mut bits = Bits::new(deflate);
    let mut header = Block::new();
    header.reset(None, None);
    let mut out_pos = 0usize;

    loop {
        if header.read_header(&mut bits, false).is_err() {
            break;
        }
        let lit = header.literal_cl[..header.literal_code_count].to_vec();
        let dist = header.literal_cl
            [header.literal_code_count..header.literal_code_count + header.distance_code_count]
            .to_vec();
        let litlen = match LitLenTable::build(&lit) {
            Some(t) => t,
            None => break,
        };
        let dist_t = match DistTable::build(&dist) {
            Some(t) => t,
            None => break,
        };
        loop {
            bits.refill();
            let saved = bits.peek();
            let entry = litlen.resolve(saved);
            if entry.is_exceptional() && entry.is_end_of_block() {
                bits.consume_entry(entry.raw());
                break;
            }
            if (entry.raw() as i32) < 0 {
                // literal — single write, no packing
                bits.consume_entry(entry.raw());
                out[out_pos] = entry.literal_value();
                out_pos += 1;
                continue;
            }
            // length code
            let length = entry.decode_length(saved);
            bits.consume_entry(entry.raw());
            bits.refill();
            let dsaved = bits.peek();
            let mut de = dist_t.lookup(dsaved);
            if de.is_subtable_ptr() {
                bits.consume(DistTable::TABLE_BITS as u32);
                de = dist_t.lookup_subtable(de, dsaved);
            }
            let dsaved2 = bits.peek();
            bits.consume_entry(de.raw());
            let distance = de.decode_distance(dsaved2) as usize;
            flat_copy_u8(out, out_pos, distance, length as usize);
            out_pos += length as usize;
        }
        if header.is_last_block() {
            break;
        }
    }
    out_pos
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "pure".into());
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let path = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "benchmark_data/silesia-gzip.tar.gz".into());

    let deflate = load_raw_deflate(&path);

    // Bootstrap (window-absent marker-path) mode uses a u16 output buffer.
    #[cfg(feature = "pure-rust-inflate")]
    if which == "bootstrap" {
        let mut out16: Vec<u16> = Vec::with_capacity(deflate.len() * 4);
        let t0 = Instant::now();
        let mut total = 0usize;
        let mut out_len = 0usize;
        for _ in 0..iters {
            out_len = decode_bootstrap(&deflate, &mut out16);
            total += out_len;
        }
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "decoder=bootstrap iters={iters} out_len={out_len} total_bytes={total} \
             wall={secs:.3}s {:.0} MB/s  (perf: instructions/{total} = ins/byte)",
            total as f64 / secs / 1e6,
        );
        println!("{total}");
        return;
    }

    // Over-allocate a generous output buffer ONCE (deflate ratio is well under
    // 8x). NO libdeflate sizing-probe: a probe decode contaminated every run
    // (~16% of a `consume_first` perf-stat was actually the one libdeflate
    // probe decode), poisoning the cross-decoder instruction ratio. The buffer
    // is faulted once and reused across iters — identical cost for every
    // decoder, so it cancels in the ratio.
    let mut out = vec![0u8; deflate.len() * 8];

    let t0 = Instant::now();
    let mut total = 0usize;
    let mut out_len = 0usize;
    for _ in 0..iters {
        out_len = match which.as_str() {
            "libdeflate" => decode_libdeflate(&deflate, &mut out),
            #[cfg(feature = "pure-rust-inflate")]
            "pure" => decode_pure(&deflate, &mut out),
            #[cfg(feature = "pure-rust-inflate")]
            "consume_first" => decode_consume_first(&deflate, &mut out),
            #[cfg(feature = "pure-rust-inflate")]
            "tbl_isal_u8" => decode_tbl_isal_u8(&deflate, &mut out),
            #[cfg(feature = "pure-rust-inflate")]
            "tbl_isal_u16" => decode_tbl_isal_u16(&deflate, &mut out),
            #[cfg(feature = "pure-rust-inflate")]
            "clean_u8_nopack" => decode_clean_u8_nopack(&deflate, &mut out),
            other => panic!("unknown decoder {other:?} (or pure-rust-inflate feature off)"),
        };
        total += out_len;
    }
    let secs = t0.elapsed().as_secs_f64();
    let mbps = total as f64 / secs / 1e6;
    eprintln!(
        "decoder={which} iters={iters} out_len={out_len} total_bytes={total} \
         wall={secs:.3}s {mbps:.0} MB/s  (perf: instructions/{total} = ins/byte)",
    );
    // Print total decoded bytes on stdout for scripting (ins/byte denominator).
    println!("{total}");
}
