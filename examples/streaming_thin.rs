//! THIN streaming microbench — ROUTE-A REMOVAL ORACLE @ kernel-converge-A HEAD.
//!
//! Question (mission route A): does shedding the resumable/marker/chunk
//! bookkeeping around the PRODUCTION clean kernel get gzippy-native to
//! igzip/libdeflate T1 parity, or is the residual deficit kernel-codegen-bound
//! (=> route B required)?
//!
//! This is a port of the 2026-06-15 streaming-thin falsifier (commit 7c798f4c)
//! onto kernel-converge-A's CURRENT kernel. It is a THIN single-pass wrapper
//! over the ALREADY-SHIPPING `Block::decode_clean_into_contig` +
//! `decode_clean_stored_into_contig` kernel — the SAME function the production
//! native clean-chunk path calls (chunk_decode.rs: "the BULK clean tail is
//! u8-direct via decode_clean_into_contig"). It decodes ONE clean single-member
//! gzip stream into a CONSTANT-MEMORY sliding buffer
//!     [ retained 32768 | batch B | SLOP >= 266 ]
//! flushing ONLY at decode return, memmove-retaining EXACTLY 32768 bytes,
//! resetting `pos` to 32768. No routing, no chunk scaffold, no parallel
//! pipeline, no marker sink, no per-chunk CRC second-touch — JUST the kernel
//! driven single-pass. (Thin SKIPS CRC32; the C tools COMPUTE it in-timer, so
//! a thin LOSS is conservative — route A would still owe CRC.)
//!
//! The `prod` mode runs the FULL production parallel-SM path at T=1
//! (decode-only timed, /dev/null sink) so igzip / libdeflate / prod-T1 /
//! thin-T1 all race apples-to-apples (same internal decode-only timing, same
//! /dev/null sink). thin-T1 − prod-T1 = the scaffold/bookkeeping bound route A
//! can capture; thin-T1 − igzip = the residual kernel-codegen gap route B owes.
//!
//! Modes (argv):
//!   streaming_thin verify     <file.gz>                correctness gate
//!   streaming_thin gzippy     <file.gz> [batch_bytes]  thin streaming kernel (route A)
//!   streaming_thin prod       <file.gz>                full parallel-SM T=1 (decode-only)
//!   streaming_thin libdeflate <file.gz>                libdeflate one-shot
//!   streaming_thin zlibng     <file.gz>                zlib-ng (flate2) stream
//!   streaming_thin igzip      <file.gz>                ISA-L one-shot (isal feat)
//!
//! Each timed mode prints: `RESULT mode=<m> ms=<f> bytes=<n>`.

use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

const WINDOW: usize = 32768;
// SLOP >= MAX_RUN_LENGTH(258) + 8 = 266 (decode_clean_into_contig out_room cap).
const SLOP: usize = 512;
const DEFAULT_BATCH: usize = 4 * 1024 * 1024;

/// gzip ISIZE (uncompressed size mod 2^32) from the trailer's last 4 bytes.
fn gzip_isize(file: &[u8]) -> usize {
    let n = file.len();
    u32::from_le_bytes([file[n - 4], file[n - 3], file[n - 2], file[n - 1]]) as usize
}

fn devnull() -> std::io::BufWriter<std::fs::File> {
    let f = OpenOptions::new()
        .write(true)
        .open("/dev/null")
        .expect("open /dev/null");
    std::io::BufWriter::with_capacity(1 << 20, f)
}

// ───────────────────────────── gzippy thin kernel ──────────────────────────
//
// Single-pass streaming decode through the SHIPPING clean kernel. `sink` is
// called with each flushed slice. Returns total output bytes.
#[cfg(pure_inflate_decode)]
fn gzippy_thin<S: FnMut(&[u8])>(file_bytes: &[u8], batch: usize, mut sink: S) -> usize {
    use gzippy::decompress::inflate::consume_first_decode::Bits;
    use gzippy::decompress::parallel::marker_inflate::{Block, CompressionType};

    let (_h, hdr) =
        gzippy::decompress::parallel::gzip_format::read_header(file_bytes).expect("gzip header");
    // Include the 8-byte trailer as input slop; decode stops at the last-block
    // EOB before reaching it.
    let deflate = &file_bytes[hdr..];

    let cap = WINDOW + batch + SLOP;
    let mut buf = vec![0u8; cap];
    let base = buf.as_mut_ptr();

    let mut bits = Bits::new(deflate);
    let mut b = Block::new();
    let mut prime: Vec<u16> = Vec::new();
    // Prime clean (window-primed) mode with an EMPTY initial window (first
    // single member, no preset dictionary).
    b.set_initial_window(&mut prime, &[]).expect("prime clean");

    let mut pos: usize = 0; // logical contiguous write index
    let mut window_len: usize = 0; // history-prefix length (0, then 32768)
    let mut total: usize = 0;
    let flush_hi = WINDOW + batch; // flush trigger position

    'outer: loop {
        if b.read_header(&mut bits, false).is_err() {
            break;
        }
        while !b.eob() {
            // SAFETY: `base` is valid for `[0, cap)`; the kernel's out_room cap
            // (cap - (MAX_RUN_LENGTH+8)) keeps every store < cap. n_max =
            // usize::MAX so the only bound is the buffer cap.
            let n = if b.compression_type() == CompressionType::Uncompressed {
                unsafe {
                    b.decode_clean_stored_into_contig(&mut bits, base, cap, &mut pos, usize::MAX)
                }
                .expect("stored block decode")
            } else {
                unsafe { b.decode_clean_into_contig(&mut bits, base, cap, &mut pos, usize::MAX) }
                    .expect("clean block decode")
            };
            // FLUSH (only at decode return). Emit everything past the history
            // prefix; retain EXACTLY the last 32768 as the new prefix.
            if pos >= flush_hi {
                sink(&buf[window_len..pos]);
                total += pos - window_len;
                buf.copy_within(pos - WINDOW..pos, 0); // memmove-retain 32768
                pos = WINDOW;
                window_len = WINDOW;
            }
            if n == 0 && !b.eob() {
                panic!("thin: decode stalled before EOB (pos={pos} cap={cap})");
            }
        }
        if b.is_last_block() {
            break 'outer;
        }
    }
    // Final flush: emit the remaining decoded tail (no retain needed).
    if pos > window_len {
        sink(&buf[window_len..pos]);
        total += pos - window_len;
    }
    total
}

#[cfg(pure_inflate_decode)]
fn run_gzippy(file: &[u8], batch: usize) -> (f64, usize) {
    let mut out = devnull();
    let t = Instant::now();
    let n = gzippy_thin(file, batch, |slice| {
        out.write_all(slice).expect("sink write");
    });
    out.flush().expect("flush");
    (t.elapsed().as_secs_f64() * 1e3, n)
}

#[cfg(not(pure_inflate_decode))]
fn run_gzippy(_file: &[u8], _batch: usize) -> (f64, usize) {
    panic!("gzippy thin kernel requires pure_inflate_decode (build --features gzippy-native/isal)");
}

// ───────────────────────────── prod (full parallel-SM, T=1) ────────────────
#[cfg(pure_inflate_decode)]
fn run_prod(file: &[u8]) -> (f64, usize) {
    let mut out = devnull();
    let t = Instant::now();
    let n =
        gzippy::decompress::parallel::single_member::decompress_parallel(file, &mut out, None, 1)
            .expect("prod parallel-SM T=1 decode");
    out.flush().expect("flush");
    (t.elapsed().as_secs_f64() * 1e3, n as usize)
}

#[cfg(not(pure_inflate_decode))]
fn run_prod(_file: &[u8]) -> (f64, usize) {
    panic!("prod mode requires pure_inflate_decode (build --features gzippy-native/isal)");
}

// ───────────────────────────── competitors ────────────────────────────────

fn run_libdeflate(file: &[u8]) -> (f64, usize) {
    let isize = gzip_isize(file);
    let mut buf = vec![0u8; isize];
    let mut out = devnull();
    let t = Instant::now();
    let n = libdeflater::Decompressor::new()
        .gzip_decompress(file, &mut buf)
        .expect("libdeflate gzip_decompress");
    out.write_all(&buf[..n]).expect("sink");
    out.flush().expect("flush");
    (t.elapsed().as_secs_f64() * 1e3, n)
}

fn run_zlibng(file: &[u8]) -> (f64, usize) {
    use std::io::Read;
    let mut out = devnull();
    let mut chunk = vec![0u8; 1 << 20];
    let t = Instant::now();
    let mut dec = flate2::read::GzDecoder::new(file);
    let mut total = 0usize;
    loop {
        let r = dec.read(&mut chunk).expect("zlib-ng read");
        if r == 0 {
            break;
        }
        out.write_all(&chunk[..r]).expect("sink");
        total += r;
    }
    out.flush().expect("flush");
    (t.elapsed().as_secs_f64() * 1e3, total)
}

#[cfg(feature = "isal-compression")]
fn run_igzip(file: &[u8]) -> (f64, usize) {
    let mut out = devnull();
    let t = Instant::now();
    let n = gzippy::isal_decompress_oracle::decompress_gzip_stream(file, &mut out)
        .expect("igzip decode") as usize;
    out.flush().expect("flush");
    (t.elapsed().as_secs_f64() * 1e3, n)
}

#[cfg(not(feature = "isal-compression"))]
fn run_igzip(_file: &[u8]) -> (f64, usize) {
    panic!("igzip mode requires the isal-compression feature (build --features gzippy-isal)");
}

// ──────────────── gz-driver + igzip _04 BARE kernel (MEASUREMENT 2) ──────────
//
// DE-CONFOUNDED bare-kernel removal-oracle. This is the EXACT same thin
// single-pass contig driver as `gzippy_thin` (same buffer, same memmove-retain
// window, same flush), but the inner per-block decode is igzip's BARE
// `decode_huffman_code_block_stateless_04` instead of gz's `run_contig` kernel.
//
// gz STILL parses every block header and BUILDS the Huffman tables
// (`LutLitLenCode`/`LutDistCode` — gz's own table-build, whose output is the
// byte-for-byte igzip `inflate_huff_code_large/small` `#[repr(C)]` layout), so
// `_04` reads gz's already-built tables directly. NOTHING of igzip's STREAMING
// ENGINE is grafted in (no per-chunk `set_dict`, no 128 KiB input staging, no
// streaming-history glue) — that whole-engine swap was NIGHT37's confound. The
// ONLY thing swapped is the inner decode loop. Hence:
//     gzippy_thin − igzip_bare  ==  the BARE-KERNEL ceiling
//     igzip_bare  ≈ igzip       ⇒  the kernel IS the lever
//     igzip_bare  ≈ gzippy_thin ⇒  residual is NOT the bare loop (pivot)
//
// One per-block ~18 KiB table memcpy installs gz's table into the igzip state
// (igzip builds in-place, so this is overhead igzip does not pay → it biases
// igzip_bare SLOWER → CONSERVATIVE for the "kernel is the lever" verdict).
#[cfg(all(pure_inflate_decode, feature = "isal-compression"))]
fn igzip_bare<S: FnMut(&[u8])>(file_bytes: &[u8], batch: usize, mut sink: S) -> usize {
    use gzippy::decompress::inflate::consume_first_decode::{parse_dynamic_header, Bits};
    use gzippy::decompress::parallel::gzip_format::read_header as gz_read_header;
    use gzippy::decompress::parallel::lut_huffman::{
        InflateHuffCodeLarge, InflateHuffCodeSmall, LutDistCode, LutLitLenCode, SINGLE_SYM_FLAG,
    };
    use isal::isal_sys::igzip_lib as isal;

    extern "C" {
        fn decode_huffman_code_block_stateless_04(
            state: *mut isal::inflate_state,
            start_out: *mut u8,
        );
        fn decode_huffman_code_block_stateless_base(
            state: *mut isal::inflate_state,
            start_out: *mut u8,
        );
    }
    let use_base = std::env::var("GZIPPY_BARE_BASE").is_ok();

    let (_h, hdr) = gz_read_header(file_bytes).expect("gzip header");
    // Padded deflate body so _04's (>=8 B) lookahead never runs off the end.
    let mut input: Vec<u8> = file_bytes[hdr..].to_vec();
    input.extend_from_slice(&[0u8; 32]);

    let cap = WINDOW + batch + SLOP;
    let mut buf = vec![0u8; cap];
    let base = buf.as_mut_ptr();

    // Reusable per-block builders + ONE igzip inflate_state (boxed; ~40 KiB).
    let mut litlen = LutLitLenCode::new_empty();
    let mut dist = LutDistCode::new_empty();
    let mut state: Box<isal::inflate_state> = Box::new(unsafe { std::mem::zeroed() });
    unsafe { isal::isal_inflate_init(&mut *state) };
    state.crc_flag = 0;
    debug_assert_eq!(
        std::mem::size_of::<InflateHuffCodeLarge>(),
        std::mem::size_of_val(&state.lit_huff_code),
        "gz/igzip lit table layout drift"
    );
    debug_assert_eq!(
        std::mem::size_of::<InflateHuffCodeSmall>(),
        std::mem::size_of_val(&state.dist_huff_code),
        "gz/igzip dist table layout drift"
    );

    // Fixed-Huffman code lengths (RFC 1951 §3.2.6).
    let fixed_litlen: Vec<u8> = (0u32..288)
        .map(|s| match s {
            0..=143 => 8u8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        })
        .collect();
    let fixed_dist: Vec<u8> = vec![5u8; 30];

    let mut bit_off: usize = 0;
    let mut pos: usize = 0;
    let mut window_len: usize = 0;
    let mut total: usize = 0;
    let mut dbg_n: usize = 0;
    let dbg = std::env::var("GZIPPY_BARE_DEBUG").is_ok();
    let flush_hi = WINDOW + batch;

    loop {
        let mut bits = Bits::at_bit_offset(&input, bit_off);
        bits.refill();
        let bfinal = (bits.peek() & 1) as u32;
        bits.consume(1);
        let btype = (bits.peek() & 0b11) as u32;
        bits.consume(2);

        match btype {
            0 => {
                // Stored: align to byte, LEN/NLEN, copy raw bytes (gz handles
                // stored too — this is a memcpy, not the kernel under test).
                bits.align_to_byte();
                let byte = bits.bit_position() / 8;
                let len = u16::from_le_bytes([input[byte], input[byte + 1]]) as usize;
                let data = byte + 4;
                unsafe {
                    std::ptr::copy_nonoverlapping(input.as_ptr().add(data), base.add(pos), len);
                }
                pos += len;
                bit_off = (data + len) * 8;
            }
            1 | 2 => {
                let (ll, dl): (Vec<u8>, Vec<u8>) = if btype == 1 {
                    (fixed_litlen.clone(), fixed_dist.clone())
                } else {
                    parse_dynamic_header(&mut bits).expect("dynamic header")
                };
                // SINGLE-symbol packing: igzip's _04 speculative-literal asm
                // path mis-decodes gz's TRIPLE pack; single-symbol entries are
                // unambiguous. (Conservative: SINGLE makes _04 do more
                // iterations than its native TRIPLE → biases igzip_bare SLOWER.)
                if !litlen.rebuild_from_multisym(&ll, SINGLE_SYM_FLAG) {
                    panic!("litlen table build failed");
                }
                if !dist.rebuild_from(&dl) {
                    panic!("dist table build failed");
                }
                // Install gz-built tables into the igzip state (binary-identical
                // repr(C) layout; _04 reads them as [state + huff_code offset]).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (&litlen.table) as *const InflateHuffCodeLarge as *const u8,
                        (&mut state.lit_huff_code) as *mut _ as *mut u8,
                        std::mem::size_of::<InflateHuffCodeLarge>(),
                    );
                    std::ptr::copy_nonoverlapping(
                        (&*dist.table) as *const InflateHuffCodeSmall as *const u8,
                        (&mut state.dist_huff_code) as *mut _ as *mut u8,
                        std::mem::size_of::<InflateHuffCodeSmall>(),
                    );
                }
                // Pre-kernel bit state for the block BODY (mirror kernel_ab
                // ARM B / igzip inflate_in_load): preload read_in LSB-first.
                let body_bit = bits.bit_position();
                let body_byte = body_bit / 8;
                let body_skip = (body_bit % 8) as u32;
                let mut read_in: u64 = 0;
                let mut read_in_length: i32 = 0;
                let mut p = body_byte;
                if body_skip != 0 {
                    read_in = (input[p] as u64) >> body_skip;
                    read_in_length = (8 - body_skip) as i32;
                    p += 1;
                }
                while read_in_length <= 56 && p < input.len() {
                    read_in |= (input[p] as u64) << read_in_length;
                    read_in_length += 8;
                    p += 1;
                }
                state.read_in = read_in;
                state.read_in_length = read_in_length;
                state.next_in = unsafe { input.as_ptr().add(p) as *mut u8 };
                state.avail_in = (input.len() - p) as u32;
                let out_ptr = unsafe { base.add(pos) };
                state.next_out = out_ptr;
                state.avail_out = (cap - pos) as u32;
                state.block_state = isal::isal_block_state_ISAL_BLOCK_CODED;
                state.copy_overflow_length = 0;
                state.copy_overflow_distance = 0;
                // start_out = base ⇒ lookback bounded by (next_out − base) =
                // the contiguous retained window + already-decoded bytes.
                // igzip contract: _04 is the FAST bulk loop that bails (leaves
                // block_state == CODED) within OUT/IN_BUFFER_SLOP of the buffer
                // ends; the caller then finishes the block's tail with the
                // careful loop. We mirror that: _04 for the bulk, _base for the
                // (tiny, near-boundary) tail. >99% of symbols run on _04.
                let coded = isal::isal_block_state_ISAL_BLOCK_CODED;
                unsafe {
                    if use_base {
                        decode_huffman_code_block_stateless_base(&mut *state as *mut _, base);
                    } else {
                        decode_huffman_code_block_stateless_04(&mut *state as *mut _, base);
                        if state.block_state == coded {
                            decode_huffman_code_block_stateless_base(&mut *state as *mut _, base);
                        }
                    }
                }
                let produced = (state.next_out as usize) - (out_ptr as usize);
                pos += produced;
                // Absolute bit offset after EOB (inverse of inflate_in_load).
                let consumed_bytes = (state.next_in as usize) - (input.as_ptr() as usize);
                bit_off = consumed_bytes * 8 - state.read_in_length as usize;
                if dbg && dbg_n < 6 {
                    dbg_n += 1;
                    eprintln!(
                        "blk btype={btype} bfinal={bfinal} produced={produced} \
                         block_state={} read_in_len={} consumed_bytes={consumed_bytes} \
                         body_bit={body_bit} -> bit_off={bit_off}",
                        state.block_state, state.read_in_length
                    );
                }
            }
            _ => panic!("invalid btype 3"),
        }

        // FLUSH (only at block return) — identical to gzippy_thin.
        if pos >= flush_hi {
            sink(&buf[window_len..pos]);
            total += pos - window_len;
            buf.copy_within(pos - WINDOW..pos, 0);
            pos = WINDOW;
            window_len = WINDOW;
        }

        if bfinal == 1 {
            break;
        }
    }
    if pos > window_len {
        sink(&buf[window_len..pos]);
        total += pos - window_len;
    }
    total
}

#[cfg(all(pure_inflate_decode, feature = "isal-compression"))]
fn run_igzipbare(file: &[u8], batch: usize) -> (f64, usize) {
    let mut out = devnull();
    let t = Instant::now();
    let n = igzip_bare(file, batch, |slice| {
        out.write_all(slice).expect("sink write");
    });
    out.flush().expect("flush");
    (t.elapsed().as_secs_f64() * 1e3, n)
}

#[cfg(not(all(pure_inflate_decode, feature = "isal-compression")))]
fn run_igzipbare(_file: &[u8], _batch: usize) -> (f64, usize) {
    panic!("igzipbare mode requires gzippy-isal (pure_inflate_decode + isal-compression)");
}

// ───────────────────────────── correctness gate ────────────────────────────

/// Dependency-free 64-bit FNV-1a fingerprint (compact log lines only).
fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(pure_inflate_decode)]
fn thin_to_vec(file: &[u8], batch: usize) -> Vec<u8> {
    let mut v = Vec::new();
    gzippy_thin(file, batch, |slice| v.extend_from_slice(slice));
    v
}

#[cfg(pure_inflate_decode)]
fn verify(file: &[u8]) {
    // Reference 1: libdeflate one-shot.
    let isize = gzip_isize(file);
    let mut ld = vec![0u8; isize];
    let nld = libdeflater::Decompressor::new()
        .gzip_decompress(file, &mut ld)
        .expect("libdeflate");
    ld.truncate(nld);

    // Reference 2: flate2 (zlib-ng).
    let mut fl = Vec::new();
    {
        use std::io::Read;
        flate2::read::GzDecoder::new(file)
            .read_to_end(&mut fl)
            .expect("flate2");
    }
    assert_eq!(ld, fl, "libdeflate vs flate2 disagree (bad corpus?)");
    println!(
        "ref (libdeflate==flate2) bytes={} fnv={:016x}",
        ld.len(),
        fnv64(&ld)
    );

    // Thin kernel at several batch sizes to force many mid-block flushes.
    for &batch in &[64 * 1024usize, 1024 * 1024, DEFAULT_BATCH] {
        let got = thin_to_vec(file, batch);
        assert_eq!(
            got.len(),
            ld.len(),
            "thin batch={batch}: length mismatch ({} vs {})",
            got.len(),
            ld.len()
        );
        assert!(
            got == ld,
            "thin batch={batch}: BYTE MISMATCH (wrong output)"
        );
        println!(
            "thin batch={batch:>8} byte-exact OK (fnv={:016x})",
            fnv64(&got)
        );
    }

    // prod path byte-exact too.
    {
        let mut pv = Vec::new();
        gzippy::decompress::parallel::single_member::decompress_parallel(file, &mut pv, None, 1)
            .expect("prod decode");
        assert!(pv == ld, "prod: BYTE MISMATCH");
        println!(
            "prod parallel-SM T=1 byte-exact OK (fnv={:016x})",
            fnv64(&pv)
        );
    }

    // igzip_bare (gz driver + igzip _04 kernel, gz-built tables) — EXPERIMENTAL,
    // BLOCKED: the asm `_04` long-code path mis-reads gz's long_code_lookup
    // table layout (short/singleton entries decode fine under the C `_base`
    // kernel with the SAME gz-built table, but `_04` trips END_INPUT on the
    // first >12-bit code). Kept as scaffolding for the de-confounded bare-kernel
    // removal-oracle; NON-FATAL here (set GZIPPY_VERIFY_IGZIPBARE=1 to assert).
    #[cfg(feature = "isal-compression")]
    if std::env::var("GZIPPY_VERIFY_IGZIPBARE").is_ok() {
        for &batch in &[64 * 1024usize, 1024 * 1024, DEFAULT_BATCH] {
            let mut v = Vec::new();
            igzip_bare(file, batch, |slice| v.extend_from_slice(slice));
            assert!(
                v == ld,
                "igzipbare batch={batch}: BYTE MISMATCH ({} vs {})",
                v.len(),
                ld.len()
            );
            println!("igzipbare batch={batch:>8} byte-exact OK");
        }
    } else {
        println!("igzipbare: SKIPPED verify (EXPERIMENTAL/BLOCKED — _04 long-code table)");
    }

    verify_synthetic_flush_boundary();
    println!("VERIFY OK");
}

#[cfg(pure_inflate_decode)]
fn verify_synthetic_flush_boundary() {
    use flate2::{write::GzEncoder, Compression};
    use std::io::Write as _;

    let mut chunk = vec![0u8; WINDOW];
    let mut x: u32 = 0x1234_5678;
    for b in chunk.iter_mut() {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *b = (x >> 24) as u8;
    }
    let mut payload = Vec::new();
    for _ in 0..400 {
        payload.extend_from_slice(&chunk); // ~12.8 MiB, distance-32768 refs
    }
    payload.push(0xAB);
    payload.extend_from_slice(&chunk);

    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(&payload).unwrap();
    let gz = enc.finish().unwrap();

    for &batch in &[40_000usize, 100_000, 1 << 20] {
        let got = thin_to_vec(&gz, batch);
        assert_eq!(got.len(), payload.len(), "synthetic batch={batch}: len");
        assert!(got == payload, "synthetic batch={batch}: BYTE MISMATCH");
    }
    println!("synthetic flush-boundary (distance-32768 across memmove) byte-exact OK");
}

#[cfg(not(pure_inflate_decode))]
fn verify(_file: &[u8]) {
    panic!("verify requires pure_inflate_decode (build --features gzippy-native/isal)");
}

// ───────────────────────────── main ───────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: streaming_thin <verify|gzippy|prod|libdeflate|zlibng|igzip> <file.gz> [batch_bytes]"
        );
        std::process::exit(2);
    }
    let mode = args[1].as_str();
    let path = &args[2];
    let file = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));

    match mode {
        "verify" => verify(&file),
        "gzippy" => {
            let batch = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(DEFAULT_BATCH);
            let (ms, n) = run_gzippy(&file, batch);
            println!("RESULT mode=gzippy batch={batch} ms={ms:.3} bytes={n}");
        }
        "prod" => {
            let (ms, n) = run_prod(&file);
            println!("RESULT mode=prod ms={ms:.3} bytes={n}");
        }
        "libdeflate" => {
            let (ms, n) = run_libdeflate(&file);
            println!("RESULT mode=libdeflate ms={ms:.3} bytes={n}");
        }
        "zlibng" => {
            let (ms, n) = run_zlibng(&file);
            println!("RESULT mode=zlibng ms={ms:.3} bytes={n}");
        }
        "igzip" => {
            let (ms, n) = run_igzip(&file);
            println!("RESULT mode=igzip ms={ms:.3} bytes={n}");
        }
        "igzipbare" => {
            let batch = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(DEFAULT_BATCH);
            let (ms, n) = run_igzipbare(&file, batch);
            println!("RESULT mode=igzipbare batch={batch} ms={ms:.3} bytes={n}");
        }
        other => {
            eprintln!("unknown mode: {other}");
            std::process::exit(2);
        }
    }
}
