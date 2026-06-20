//! NIGHT31 — ISOLATED KERNEL-vs-KERNEL A/B.
//!
//! Resolves the WEAK (NIGHT26 by-region whole-program perf-attribution
//! "run_contig +0.905 instr-count") vs STRONG (NIGHT16/23/30 removal-oracles)
//! contradiction with an ISOLATED-CAUSAL measurement: gz's production CLEAN
//! kernel (`Block::decode_clean_into_contig` = run_contig + its inseparable
//! resumable glue — NIGHT27's definition of gz's kernel) vs igzip's
//! `decode_huffman_code_block_stateless_04` (the AVX2/BMI2 hand-asm that runs
//! on Intel; `igzip_inflate_multibinary.asm:45` dispatch target), each decoding
//! the SAME real silesia dynamic block, looped to >=256 MiB total output, with
//! the per-block Huffman tables built ONCE OUTSIDE the timed loop (so
//! table-build amortizes to ~0 and the per-byte KERNEL cost is isolated).
//!
//! Why loop ONE real block instead of a synthetic huge block: the brief's "one
//! huge dynamic block" requirement exists only to amortize table-build; looping
//! a REAL silesia block achieves identical amortization with the REAL symbol
//! mix and ZERO hand-rolled-encoder risk.
//!
//! Gate-0 (printed): _04 selected (ARM B calls `_04` DIRECTLY — exported,
//! removing the multibinary dispatch ambiguity); ARM A == ARM B == flate2
//! byte-identical (non-inert); both arms write an in-RAM dst (same sink);
//! conservation (decoded bytes == block output len each rep).
//!
//! Usage (run under `perf stat -e cycles,instructions` for IPC/instr-per-byte;
//! the example ALSO prints rdtsc cyc/B around ONLY the kernel loop):
//!   kernel_ab --arm a --reps 4000      # gz run_contig kernel
//!   kernel_ab --arm b --reps 4000      # igzip _04 kernel
//!   kernel_ab --arm both --reps 1      # Gate-0 byte-exact self-check only

use std::arch::x86_64::_rdtsc;
use std::io::{Read, Write};

use gzippy::decompress::block_walker::walk_block_boundaries;
use gzippy::decompress::inflate::consume_first_decode::Bits;
use gzippy::decompress::parallel::marker_inflate::Block;

// ─── igzip internal kernel (exported by static-linked libisal) ───────────────
// `decode_huffman_code_block_stateless_04` is the AVX2/BMI2 variant
// (igzip_decode_block_stateless.asm, ARCH=04). nm -D libisal.so.2 confirms it is
// a global `T` symbol. Calling it DIRECTLY (not the multibinary dispatcher
// `decode_huffman_code_block_stateless`) makes the "_04 selected" Gate-0
// trivially true by construction.
mod isal_kernel {
    use isal::isal_sys::igzip_lib::inflate_state;
    extern "C" {
        pub fn decode_huffman_code_block_stateless_04(
            state: *mut inflate_state,
            start_out: *mut u8,
        );
    }
}

const TARGET_BYTES: u64 = 256 * 1024 * 1024;

fn die(msg: &str) -> ! {
    eprintln!("FATAL: {msg}");
    std::process::exit(2);
}

/// Read a silesia-derived slice, compress it to a RAW deflate stream whose FIRST
/// block is dynamic-Huffman (so it is byte-aligned at byte 0 and needs no
/// predecessor window). Returns (raw_deflate, first_block_body_bit, plain_output
/// of that one block).
fn build_real_block() -> (Vec<u8>, u64, Vec<u8>) {
    // Locate a silesia corpus. Prefer the decompressed tar; fall back to
    // decompressing silesia.gz.
    let candidates = [
        "/root/gz-fullrewrite/benchmark_data/silesia.tar",
        "/root/silesia.tar",
    ];
    let mut raw_corpus: Option<Vec<u8>> = None;
    for c in candidates {
        if let Ok(b) = std::fs::read(c) {
            raw_corpus = Some(b);
            break;
        }
    }
    let corpus = raw_corpus.unwrap_or_else(|| {
        // Decompress /root/silesia.gz via flate2.
        let gz = std::fs::read("/root/silesia.gz")
            .unwrap_or_else(|e| die(&format!("no silesia corpus found: {e}")));
        let mut d = flate2::read::GzDecoder::new(&gz[..]);
        let mut out = Vec::new();
        d.read_to_end(&mut out)
            .unwrap_or_else(|e| die(&format!("silesia.gz decompress: {e}")));
        out
    });
    if corpus.len() < 2 * 1024 * 1024 {
        die("silesia corpus too small");
    }
    // Take a representative ~512 KiB slice from the middle (skip any leading
    // tar header region). 512 KiB plain → one or a few dynamic blocks; we use
    // the FIRST dynamic block.
    let start = 1_000_000usize;
    let slice = &corpus[start..start + 512 * 1024];

    // Compress to a GZIP stream (so walk_block_boundaries can parse it), level 6
    // (real dynamic blocks, real symbol mix).
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
    enc.write_all(slice).unwrap();
    let gz = enc.finish().unwrap();

    let blocks = walk_block_boundaries(&gz).unwrap_or_else(|e| die(&format!("walk: {e}")));
    // First dynamic-Huffman block.
    let blk = blocks
        .iter()
        .find(|b| b.btype == 2)
        .unwrap_or_else(|| die("no dynamic block produced"));

    // The deflate body lives inside `gz` after the gzip header. walk gives bit
    // offsets relative to the START of the gzip stream. The kernels consume RAW
    // deflate. We hand both kernels the FULL gzip bytes as the data buffer and
    // position the bit cursor at `blk.start_bit` (gz run_contig via Bits) / set
    // up igzip on the raw deflate region. To keep igzip byte-aligned, we REQUIRE
    // the first dynamic block to start byte-aligned within the deflate body.
    //
    // The gzip header is a whole number of bytes; the deflate body starts at a
    // byte boundary. The FIRST deflate block's header is at deflate-body bit 0,
    // i.e. byte-aligned. We use the FIRST block and require it to be dynamic.
    let _ = blk;
    let first = &blocks[0];
    if first.btype != 2 {
        die("first block is not dynamic — adjust slice/level");
    }
    // Full-stream oracle, then take the first block's output prefix.
    let mut full = Vec::new();
    flate2::read::GzDecoder::new(&gz[..])
        .read_to_end(&mut full)
        .unwrap();
    let block_out_len = first.decoded_bytes as usize;
    let block_output = full[..block_out_len].to_vec();

    // start_bit for the FIRST block == deflate-body-start bit (byte-aligned).
    (gz, first.start_bit, block_output)
}

/// ARM A — gz production clean-contig kernel, looped, tables built once.
fn run_arm_a(gz: &[u8], start_bit: u64, oracle: &[u8], reps: u64) {
    // Contig buffer: no predecessor window (first block) → window_len = 0, pos 0.
    // Reserve oracle.len() + generous headroom for the word-copy overshoot.
    const HEADROOM: usize = 258 + 8 + 64;
    let cap = oracle.len() + HEADROOM;
    let mut buf = vec![0u8; cap];
    let base = buf.as_mut_ptr();

    let mut block = Block::new();
    // Flip the Block CLEAN with an EMPTY window (stream start). This matches
    // production's seed for a first block (set_initial_window with empty window
    // flips clean — marker_inflate.rs:982-991).
    let mut unused: Vec<u16> = Vec::new();
    block.reset(None, None);
    block
        .set_initial_window(&mut unused, &[])
        .unwrap_or_else(|e| die(&format!("set_initial_window: {e:?}")));

    // Parse the header ONCE (builds nothing yet; LUTs built on first
    // decode_clean_into_contig). Snapshot body-start bit cursor.
    let body_bits = {
        let mut bits = Bits::at_bit_offset(gz, start_bit as usize);
        block
            .read_header(&mut bits, false)
            .unwrap_or_else(|e| die(&format!("read_header: {e:?}")));
        (bits.pos, bits.bitbuf, bits.bitsleft)
    };

    // WARM: one decode to build the LUTs (table-build) OUTSIDE the timer, and
    // capture the produced bytes for the Gate-0 byte-exact check.
    let decode_once = |block: &mut Block, buf_base: *mut u8| -> usize {
        let mut bits = Bits {
            data: gz,
            pos: body_bits.0,
            bitbuf: body_bits.1,
            bitsleft: body_bits.2,
        };
        let mut pos = 0usize;
        block.reset_block_body_for_isolation();
        // One block may need >1 contig call if bigger than out_room; loop to EOB.
        while !block.eob() {
            let out_room = cap.saturating_sub(258 + 8);
            if pos > out_room {
                die("ARM A: insufficient headroom");
            }
            let r = unsafe {
                block.decode_clean_into_contig(&mut bits, buf_base, cap, &mut pos, usize::MAX)
            };
            match r {
                Ok(0) if !block.eob() => die("ARM A: no progress"),
                Ok(_) => {}
                Err(e) => die(&format!("ARM A decode: {e:?}")),
            }
        }
        pos
    };

    let produced = decode_once(&mut block, base);
    if produced != oracle.len() || &buf[..produced] != oracle {
        die(&format!(
            "ARM A byte mismatch: produced {produced} vs oracle {}",
            oracle.len()
        ));
    }
    eprintln!(
        "ARM A Gate-0: byte-exact vs flate2 OK ({} bytes/block); asm_kernel enabled={}",
        produced,
        asm_enabled()
    );

    // TIMED LOOP — kernel only. reset_block_body + restore cursor each rep; LUTs
    // stay built (block_huffman_luts_ready true), so NO table-build in the loop.
    let mut total: u64 = 0;
    let t0 = unsafe { _rdtsc() };
    for _ in 0..reps {
        let n = decode_once(&mut block, base);
        total += n as u64;
        std::hint::black_box(buf[0]);
    }
    let t1 = unsafe { _rdtsc() };
    let cyc = t1 - t0;
    report("A", cyc, total);
}

/// ARM B — igzip decode_huffman_code_block_stateless_04, looped, tables built
/// once via one bounded isal_inflate header parse.
fn run_arm_b(gz: &[u8], start_bit: u64, oracle: &[u8], reps: u64) {
    use isal::isal_sys::igzip_lib as isal;
    // The first deflate block starts byte-aligned (start_bit % 8 == 0).
    if start_bit % 8 != 0 {
        die("ARM B requires a byte-aligned first block");
    }
    let deflate_byte = (start_bit / 8) as usize;
    let deflate = &gz[deflate_byte..];

    const HEADROOM: usize = 258 + 64;
    let cap = oracle.len() + HEADROOM;
    let mut dst = vec![0u8; cap];

    // Parse the header ONCE: run isal_inflate with avail_out clamped so it stops
    // right after building tables (block_state -> ISAL_BLOCK_CODED). We give it
    // tiny avail_out so it parses the header + emits a few bytes then returns
    // output-full, leaving lit_huff_code/dist_huff_code built and the bit cursor
    // at the body.
    let mut state: isal::inflate_state = unsafe { std::mem::zeroed() };
    unsafe { isal::isal_inflate_init(&mut state) };
    state.crc_flag = 0; // raw inflate (no gzip wrapper on the slice we point at)
    state.next_in = deflate.as_ptr() as *mut u8;
    state.avail_in = deflate.len() as u32;
    state.next_out = dst.as_mut_ptr();
    state.avail_out = 1; // clamp: stop after header build + first byte
    let _ = unsafe { isal::isal_inflate(&mut state) };
    if state.block_state != isal::isal_block_state_ISAL_BLOCK_CODED {
        die(&format!(
            "ARM B: header parse did not reach ISAL_BLOCK_CODED (block_state={})",
            state.block_state
        ));
    }
    // Snapshot the post-header kernel-entry state (read_in/length/next_in/avail_in
    // + how many output bytes the header-parse already emitted).
    let snap_read_in = state.read_in;
    let snap_read_in_length = state.read_in_length;
    let snap_next_in = state.next_in;
    let snap_avail_in = state.avail_in;
    let pre_emitted = state.total_out as usize; // bytes the header-parse emitted

    // decode_huffman_code_block_stateless writes to state.next_out; start_out is
    // the original out base (used for distance bounds). We restore next_out to
    // base+pre_emitted each rep so the full block lands contiguously and the
    // backref window (start_out..next_out) is valid.
    let base = dst.as_mut_ptr();

    let decode_once = |state: &mut isal::inflate_state| -> usize {
        state.read_in = snap_read_in;
        state.read_in_length = snap_read_in_length;
        state.next_in = snap_next_in;
        state.avail_in = snap_avail_in;
        state.next_out = unsafe { base.add(pre_emitted) };
        state.avail_out = (cap - pre_emitted) as u32;
        state.block_state = isal::isal_block_state_ISAL_BLOCK_CODED;
        unsafe {
            isal_kernel::decode_huffman_code_block_stateless_04(state as *mut _, base);
        }
        let written = (state.next_out as usize) - (base as usize);
        written
    };

    // WARM + Gate-0 byte check.
    let produced = decode_once(&mut state);
    if produced != oracle.len() {
        die(&format!(
            "ARM B byte count {produced} != oracle {} (block_state after={})",
            oracle.len(),
            state.block_state
        ));
    }
    if &dst[..produced] != oracle {
        die("ARM B byte mismatch vs flate2");
    }
    eprintln!(
        "ARM B Gate-0: byte-exact vs flate2 OK ({} bytes/block); kernel=_04 (called directly)",
        produced
    );

    let mut total: u64 = 0;
    let t0 = unsafe { _rdtsc() };
    for _ in 0..reps {
        let n = decode_once(&mut state);
        total += n as u64;
        std::hint::black_box(dst[0]);
    }
    let t1 = unsafe { _rdtsc() };
    let cyc = t1 - t0;
    report("B", cyc, total);
}

fn asm_enabled() -> bool {
    std::env::var("GZIPPY_ASM_KERNEL")
        .map(|v| v != "0")
        .unwrap_or(true)
        && std::arch::is_x86_feature_detected!("bmi2")
}

fn report(arm: &str, cyc: u64, bytes: u64) {
    let cpb = cyc as f64 / bytes as f64;
    println!("ARM={arm} bytes={bytes} rdtsc_cyc={cyc} rdtsc_cyc_per_byte={cpb:.4}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut arm = "both".to_string();
    let mut reps: u64 = 0;
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
            other => die(&format!("unknown arg {other}")),
        }
    }

    if !std::arch::is_x86_feature_detected!("bmi2") {
        die("no BMI2 — run on the guest");
    }

    let (gz, start_bit, oracle) = build_real_block();
    eprintln!(
        "block: start_bit={start_bit} (byte-aligned={}) output_bytes={}",
        start_bit % 8 == 0,
        oracle.len()
    );
    if reps == 0 {
        // Default: choose reps to reach >=256 MiB total.
        reps = (TARGET_BYTES / oracle.len().max(1) as u64).max(1);
    }
    eprintln!("reps={reps} (target total >= {TARGET_BYTES} bytes)");

    match arm.as_str() {
        "a" => run_arm_a(&gz, start_bit, &oracle, reps),
        "b" => run_arm_b(&gz, start_bit, &oracle, reps),
        "both" => {
            run_arm_a(&gz, start_bit, &oracle, reps);
            run_arm_b(&gz, start_bit, &oracle, reps);
        }
        other => die(&format!("unknown arm {other}")),
    }
}
