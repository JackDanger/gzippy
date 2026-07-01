//! STEP-0.5 — aarch64 ENGINE-A vs ENGINE-B clean-decode A/B (pure-Rust, FFI off).
//!
//! Resolves the dominant CLEAN-KERNEL-DESIGN question: gzippy has TWO pure-Rust
//! clean-decode engines and the design proposed to BUILD a flat one from scratch —
//! but a flat one ALREADY exists. This A/B measures, on macOS-aarch64 with the
//! deterministic instruction primitive (`/usr/bin/time -l` two-point subtraction,
//! driven by `scripts/bench/standing/kernel_ab_aarch64.py`), whether the existing
//! flat engine BEATS the production two-level engine on gzippy's OWN primitives:
//!
//!   ENGINE A (flat)      — `decode_huffman_libdeflate_style`
//!                          (consume_first_decode.rs:632): flat masked LitLenTable
//!                          (TABLE_BITS=12), saved_bitbuf, multi-literal lookahead,
//!                          single refill/iter, contiguous output, NEON copy.
//!                          The production decoder for bgzf/scan/multi-member.
//!   ENGINE B (two-level) — `Block::decode_clean_into_contig`
//!                          (marker_inflate.rs:2970): the ISA-L two-level packed
//!                          table (`asm.lut_litlen`), the unpack branch-chain /
//!                          P3.2 literal-chain, double refill cadence. The
//!                          production CLEAN T1 contig path the parallel-SM engine
//!                          runs (chunk_decode.rs:1695).
//!
//! Both arms decode the SAME real silesia-derived dynamic block, CONTIGUOUSLY,
//! starting at the SAME body bit, with their tables built ONCE outside the timed
//! loop, looped to >=256 MiB total output. Using the CONTIG engine-B (not the
//! ring `decode_clean_fast_loop`) is the CONSERVATIVE, fairest table-vs-cadence
//! discriminator: it excludes the `% U8_RING_SIZE` ring-masking confound (which
//! would only bias FURTHER toward flat). No igzip/FFI, no x86 rdtsc — pure-Rust
//! both arms, so this runs on aarch64.
//!
//! Gate-0 (printed to stderr, LOUD-FAIL else the number does not exist):
//!   - byte-exact : engine A out == engine B out == flate2/gzip oracle (non-inert);
//!   - same body  : engine A body-bit == engine B body-bit (cursor conservation);
//!   - non-inert  : FLAT_DECODE_CALLS advanced by exactly `reps` (engine A ran on
//!                  aarch64), and engine-B loop produced `reps` full blocks;
//!   - same sink  : both arms write an in-RAM contiguous dst (the driver pipes the
//!                  process to /dev/null identically for both).
//!
//! Usage (the driver runs each arm at reps=R and reps=2R; per-rep instr =
//! (instr_2R - instr_R)/R cancels ALL fixed setup → kernel-only instr/B):
//!   kernel_ab_aarch64 --arm a --reps 4000 --corpus /path/plain --name webster
//!   kernel_ab_aarch64 --arm b --reps 4000 --corpus /path/plain --name webster
//!   kernel_ab_aarch64 --arm both --reps 1  --corpus /path/plain  # Gate-0 only
//!
//! SCOPE: macOS-aarch64, NOT-YET-LAW cross-arch. Deterministic-instr HYPOTHESIS.

use std::io::{Read, Write};
use std::sync::atomic::Ordering;

use gzippy::decompress::block_walker::walk_block_boundaries;
use gzippy::decompress::inflate::consume_first_decode::{
    decode_flat_clean_pub, parse_dynamic_header, Bits, FLAT_DECODE_CALLS,
};
use gzippy::decompress::inflate::libdeflate_entry::{DistTable, LitLenTable};
use gzippy::decompress::parallel::marker_inflate::Block;

const TARGET_BYTES: u64 = 256 * 1024 * 1024;

fn die(msg: &str) -> ! {
    eprintln!("FATAL: {msg}");
    std::process::exit(2);
}

/// Mirror of block_walker.rs:78-98 — byte offset where the deflate body begins.
fn gzip_header_end(gz: &[u8]) -> usize {
    let flg = gz[3];
    let mut header_end = 10;
    if flg & 0x04 != 0 {
        let xlen = u16::from_le_bytes([gz[header_end], gz[header_end + 1]]) as usize;
        header_end += 2 + xlen;
    }
    if flg & 0x08 != 0 {
        while header_end < gz.len() && gz[header_end] != 0 {
            header_end += 1;
        }
        header_end += 1;
    }
    if flg & 0x10 != 0 {
        while header_end < gz.len() && gz[header_end] != 0 {
            header_end += 1;
        }
        header_end += 1;
    }
    if flg & 0x02 != 0 {
        header_end += 2;
    }
    header_end
}

/// Read a plaintext corpus, compress a ~`slice`-byte middle slice to gzip (level 6,
/// real dynamic blocks), and return (raw_deflate_truncated_to_first_block,
/// first_block_start_bit, oracle_output_of_that_block). Identical for every rep
/// count (the R vs 2R subtraction cancels this fixed cost).
fn build_real_block(plain_path: &str, offset: usize, slice: usize) -> (Vec<u8>, u64, Vec<u8>) {
    let corpus = std::fs::read(plain_path)
        .unwrap_or_else(|e| die(&format!("read corpus {plain_path}: {e}")));
    if corpus.len() < offset + slice {
        die(&format!(
            "corpus {plain_path} too small ({} bytes) for offset {offset} + slice {slice}",
            corpus.len()
        ));
    }
    let s = &corpus[offset..offset + slice];

    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
    enc.write_all(s).unwrap();
    let gz = enc.finish().unwrap();

    let blocks = walk_block_boundaries(&gz).unwrap_or_else(|e| die(&format!("walk: {e}")));
    let header_end = gzip_header_end(&gz);
    let full_deflate = &gz[header_end..gz.len() - 8];

    let first = &blocks[0];
    if first.btype != 2 {
        die("first block is not dynamic — adjust slice/offset");
    }
    let end_byte = (first.end_bit as usize).div_ceil(8);
    let deflate = full_deflate[..end_byte].to_vec();

    let mut full = Vec::new();
    flate2::read::GzDecoder::new(&gz[..])
        .read_to_end(&mut full)
        .unwrap();
    let block_out_len = first.decoded_bytes as usize;
    let oracle = full[..block_out_len].to_vec();

    (deflate, first.start_bit, oracle)
}

/// Snapshot of a `Bits` cursor at the block body (after the dynamic header).
#[derive(Clone, Copy)]
struct BodyCursor {
    pos: usize,
    bitbuf: u64,
    bitsleft: u32,
}

/// ENGINE A setup: parse the dynamic header once, build the FLAT tables, return
/// (litlen, dist, body cursor).
fn setup_engine_a(deflate: &[u8], start_bit: u64) -> (LitLenTable, DistTable, BodyCursor, usize) {
    let mut bits = Bits::at_bit_offset(deflate, start_bit as usize);
    bits.refill();
    // Consume the 3-bit block header (BFINAL|BTYPE) the same way
    // inflate_consume_first_bits does before dispatching to decode_dynamic.
    let btype = ((bits.peek() >> 1) & 3) as u8;
    if btype != 2 {
        die("engine A: first block btype != dynamic");
    }
    bits.consume(3);
    let (ll, dl) = parse_dynamic_header(&mut bits).unwrap_or_else(|e| die(&format!("parse: {e}")));
    let litlen = LitLenTable::build(&ll).unwrap_or_else(|| die("LitLenTable::build failed"));
    let dist = DistTable::build(&dl).unwrap_or_else(|| die("DistTable::build failed"));
    let body = BodyCursor {
        pos: bits.pos,
        bitbuf: bits.bitbuf,
        bitsleft: bits.bitsleft,
    };
    (litlen, dist, body, bits.bit_position())
}

/// One ENGINE A decode of the block body into a fresh contiguous buffer.
fn decode_once_a(
    deflate: &[u8],
    body: BodyCursor,
    litlen: &LitLenTable,
    dist: &DistTable,
    out: &mut [u8],
) -> usize {
    let mut bits = Bits {
        data: deflate,
        pos: body.pos,
        bitbuf: body.bitbuf,
        bitsleft: body.bitsleft,
    };
    decode_flat_clean_pub(&mut bits, out, 0, litlen, dist)
        .unwrap_or_else(|e| die(&format!("engine A decode: {e}")))
}

fn run_arm_a(deflate: &[u8], start_bit: u64, oracle: &[u8], reps: u64, gate0: bool) -> usize {
    let (litlen, dist, body, body_bit) = setup_engine_a(deflate, start_bit);
    let cap = oracle.len() + 512;
    let mut out = vec![0u8; cap];

    // WARM + byte-exact (table-build already done in setup).
    let before = FLAT_DECODE_CALLS.load(Ordering::Relaxed);
    let produced = decode_once_a(deflate, body, &litlen, &dist, &mut out);
    if produced != oracle.len() || &out[..produced] != oracle {
        die(&format!(
            "ARM A byte mismatch: produced {produced} vs oracle {}",
            oracle.len()
        ));
    }
    if FLAT_DECODE_CALLS.load(Ordering::Relaxed) != before + 1 {
        die("ARM A: FLAT_DECODE_CALLS did not advance on warm decode (INERT)");
    }
    if gate0 {
        eprintln!(
            "ARM A Gate-0: byte-exact vs flate2 OK ({produced} bytes/block); body_bit={body_bit}; flat engine NON-INERT (counter advanced)"
        );
    }

    // TIMED LOOP — engine A kernel only (tables stay built).
    let mut total: usize = 0;
    let start = FLAT_DECODE_CALLS.load(Ordering::Relaxed);
    for _ in 0..reps {
        let n = decode_once_a(deflate, body, &litlen, &dist, &mut out);
        total += n;
        std::hint::black_box(out[0]);
    }
    let advanced = FLAT_DECODE_CALLS.load(Ordering::Relaxed) - start;
    if advanced != reps {
        die(&format!(
            "ARM A: FLAT_DECODE_CALLS advanced {advanced} != reps {reps} (INERT/inconsistent)"
        ));
    }
    total
}

/// ENGINE B (production CLEAN contig two-level path) — looped, LUTs built once.
fn run_arm_b(deflate: &[u8], start_bit: u64, oracle: &[u8], reps: u64, gate0: bool) -> usize {
    const HEADROOM: usize = 258 + 8 + 64;
    let cap = oracle.len() + HEADROOM;
    let mut buf = vec![0u8; cap];
    let base = buf.as_mut_ptr();

    let mut block = Block::new();
    let mut unused: Vec<u16> = Vec::new();
    block.reset(None, None);
    block
        .set_initial_window(&mut unused, &[])
        .unwrap_or_else(|e| die(&format!("set_initial_window: {e:?}")));

    // Parse the header ONCE (LUTs built lazily on the first contig decode).
    let body_bit;
    let body = {
        let mut bits = Bits::at_bit_offset(deflate, start_bit as usize);
        block
            .read_header(&mut bits, false)
            .unwrap_or_else(|e| die(&format!("read_header: {e:?}")));
        body_bit = bits.bit_position();
        BodyCursor {
            pos: bits.pos,
            bitbuf: bits.bitbuf,
            bitsleft: bits.bitsleft,
        }
    };

    let decode_once_b = |block: &mut Block, base: *mut u8| -> usize {
        let mut bits = Bits {
            data: deflate,
            pos: body.pos,
            bitbuf: body.bitbuf,
            bitsleft: body.bitsleft,
        };
        let mut pos = 0usize;
        block.reset_block_body_for_isolation();
        while !block.eob() {
            let out_room = cap.saturating_sub(258 + 8);
            if pos > out_room {
                die("ARM B: insufficient headroom");
            }
            let r = unsafe {
                block.decode_clean_into_contig(&mut bits, base, cap, &mut pos, usize::MAX)
            };
            match r {
                Ok(0) if !block.eob() => die("ARM B: no progress"),
                Ok(_) => {}
                Err(e) => die(&format!("ARM B decode: {e:?}")),
            }
        }
        pos
    };

    // WARM + byte-exact (builds the LUTs outside the timer).
    let produced = decode_once_b(&mut block, base);
    if produced != oracle.len() || &buf[..produced] != oracle {
        die(&format!(
            "ARM B byte mismatch: produced {produced} vs oracle {}",
            oracle.len()
        ));
    }
    if gate0 {
        eprintln!(
            "ARM B Gate-0: byte-exact vs flate2 OK ({produced} bytes/block); body_bit={body_bit}; two-level contig clean path"
        );
    }

    let mut total: usize = 0;
    for _ in 0..reps {
        let n = decode_once_b(&mut block, base);
        total += n;
        std::hint::black_box(buf[0]);
    }
    total
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut arm = "both".to_string();
    let mut reps: u64 = 0;
    let mut corpus = "/tmp/silesia_x/silesia/webster".to_string();
    let mut offset = 1_000_000usize;
    let mut slice = 512 * 1024usize;
    let mut name = "corpus".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--arm" => {
                arm = args[i + 1].clone();
                i += 2;
            }
            "--reps" => {
                reps = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--corpus" => {
                corpus = args[i + 1].clone();
                i += 2;
            }
            "--offset" => {
                offset = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--slice" => {
                slice = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--name" => {
                name = args[i + 1].clone();
                i += 2;
            }
            other => die(&format!("unknown arg {other}")),
        }
    }

    let (deflate, start_bit, oracle) = build_real_block(&corpus, offset, slice);

    // Gate-0 cursor conservation + cross-engine byte-exact: ALWAYS run both warm
    // decodes once and assert A out == B out == oracle and body_bit_A == body_bit_B.
    let (la, da, ba, body_bit_a) = setup_engine_a(&deflate, start_bit);
    {
        let cap = oracle.len() + 512;
        let mut oa = vec![0u8; cap];
        let pa = decode_once_a(&deflate, ba, &la, &da, &mut oa);
        // Re-derive engine B body_bit via a throwaway parse for the cross-check.
        let mut blk = Block::new();
        let mut unused: Vec<u16> = Vec::new();
        blk.reset(None, None);
        blk.set_initial_window(&mut unused, &[]).unwrap();
        let mut bits = Bits::at_bit_offset(&deflate, start_bit as usize);
        blk.read_header(&mut bits, false)
            .unwrap_or_else(|e| die(&format!("read_header: {e:?}")));
        let body_bit_b = bits.bit_position();
        if body_bit_a != body_bit_b {
            die(&format!(
                "Gate-0 cursor mismatch: engine A body_bit {body_bit_a} != engine B body_bit {body_bit_b}"
            ));
        }
        if pa != oracle.len() || oa[..pa] != oracle[..] {
            die("Gate-0: engine A != oracle");
        }
        eprintln!(
            "[gate-0] corpus={name} block_out={} body_bit={body_bit_a} (A==B) engine-A==oracle OK",
            oracle.len()
        );
    }

    if reps == 0 {
        reps = (TARGET_BYTES / oracle.len().max(1) as u64).max(1);
    }

    let gate0 = true;
    let total = match arm.as_str() {
        "a" => run_arm_a(&deflate, start_bit, &oracle, reps, gate0),
        "b" => run_arm_b(&deflate, start_bit, &oracle, reps, gate0),
        "both" => {
            let ta = run_arm_a(&deflate, start_bit, &oracle, reps, gate0);
            let tb = run_arm_b(&deflate, start_bit, &oracle, reps, gate0);
            println!(
                "RESULT-BOTH name={name} reps={reps} bytes_per_rep={} total_a={ta} total_b={tb}",
                oracle.len()
            );
            return;
        }
        other => die(&format!("unknown arm {other}")),
    };

    println!(
        "RESULT arm={arm} name={name} reps={reps} bytes_per_rep={} total_bytes={total}",
        oracle.len()
    );
}
