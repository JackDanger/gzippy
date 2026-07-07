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
    let n = gzippy::decompress::parallel::single_member::decompress_parallel(
        file, &mut out, None, 1, false,
    )
    .expect("prod parallel-SM T=1 decode");
    out.flush().expect("flush");
    (t.elapsed().as_secs_f64() * 1e3, n as usize)
}

#[cfg(not(pure_inflate_decode))]
fn run_prod(_file: &[u8]) -> (f64, usize) {
    panic!("prod mode requires pure_inflate_decode (build --features gzippy-native/isal)");
}

// Thread-parameterized prod for the T>1 load-bearing classification: same real
// production path at an arbitrary thread count, decode-only timed, /dev/null sink.
#[cfg(pure_inflate_decode)]
fn run_prodt(file: &[u8], threads: usize) -> (f64, usize) {
    let mut out = devnull();
    let t = Instant::now();
    let n = gzippy::decompress::parallel::single_member::decompress_parallel(
        file, &mut out, None, threads, false,
    )
    .expect("prodt parallel-SM decode");
    out.flush().expect("flush");
    (t.elapsed().as_secs_f64() * 1e3, n as usize)
}

#[cfg(not(pure_inflate_decode))]
fn run_prodt(_file: &[u8], _threads: usize) -> (f64, usize) {
    panic!("prodt mode requires pure_inflate_decode (build --features gzippy-native/isal)");
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
// DE-CONFOUNDED bare-inner-decode removal-oracle. This is the EXACT same thin
// single-pass contig driver as `gzippy_thin` (same buffer, same memmove-retain
// 32 KiB window, same per-batch flush), but the per-block INNER DECODE is
// igzip's own header-read + `_04` Huffman loop instead of gz's run_contig
// kernel + gz table-build.
//
// THE FIX (2026-06-21): the prior scaffold built gz's OWN Huffman tables
// (`LutLitLenCode`/`LutDistCode`) and fed them to `_04`. That worked for the C
// `_base` kernel but the asm `_04` long-code handler tripped `ISAL_END_INPUT`
// on gz's `long_code_lookup` layout (~130 B in). Sidestep gz's table format
// ENTIRELY: drive the patched ISA-L `isal_inflate` with the
// `END_OF_BLOCK_HEADER` stopping point — it runs igzip's OWN `read_header`
// (igzip-native, `_04`-compatible tables) and restores the bit reader to the
// block BODY (igzip_inflate.c:2354-2365 stop + :2616-2622 restore), then we
// call `_04` (+ `_base` careful tail) directly into gz's contig buffer. igzip
// owns the bit reader + table-build; gz owns the driver/window/contig. Hence:
//     gzippy_thin − igzip_bare  ==  the inner-decode (kernel+table-build) ceiling
//     igzip_bare  ≈ igzip       ⇒  the inner decode IS the lever
//     igzip_bare  ≈ gzippy_thin ⇒  residual is the gz driver/scaffold (pivot)
//
// Room invariant: we keep ≥ MAXBLK+SLOP free before every block (flush+memmove
// when `pos` reaches `WINDOW+batch`), so `_04` NEVER output-overflows. A
// remaining `CODED` after `_04` then means an input-end bail → `_base` safely
// finishes (it re-zeroes copy_overflow at entry, so calling it after a genuine
// OUTPUT overflow would corrupt — the room invariant forbids that case; an
// assert is the tripwire).
#[cfg(all(pure_inflate_decode, feature = "isal-compression"))]
fn igzip_bare<S: FnMut(&[u8])>(file_bytes: &[u8], batch: usize, mut sink: S) -> usize {
    use gzippy::decompress::parallel::gzip_format::read_header as gz_read_header;
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
    // Padded raw-deflate body so the kernel's (>=8 B) lookahead never runs off
    // the end; igzip owns the bit reader from here.
    let mut input: Vec<u8> = file_bytes[hdr..].to_vec();
    input.extend_from_slice(&[0u8; 64]);

    // MAXBLK = guaranteed free headroom before each block (so a single block's
    // output never overflows avail_out). Stored blocks are <=65535 B; real gzip
    // coded blocks are far under 8 MiB. The assert below is the tripwire if not.
    const MAXBLK: usize = 8 * 1024 * 1024;
    let cap = WINDOW + batch + MAXBLK + SLOP;
    let mut buf = vec![0u8; cap];
    let base = buf.as_mut_ptr();

    let mut state: Box<isal::inflate_state> = Box::new(unsafe { std::mem::zeroed() });
    unsafe { isal::isal_inflate_init(&mut *state) };
    state.crc_flag = 0; // raw deflate, no checksum (thin skips CRC too)
    state.hist_bits = 0; // full 32 KiB window
    state.next_in = input.as_ptr() as *mut u8;
    state.avail_in = input.len() as u32;
    state.read_in = 0;
    state.read_in_length = 0;
    state.block_state = isal::isal_block_state_ISAL_BLOCK_NEW_HDR;
    state.points_to_stop_at = isal::ISAL_STOPPING_POINT_END_OF_BLOCK_HEADER;

    let coded = isal::isal_block_state_ISAL_BLOCK_CODED;
    let type0 = isal::isal_block_state_ISAL_BLOCK_TYPE0;

    let mut pos: usize = 0;
    let mut window_len: usize = 0;
    let mut total: usize = 0;
    let mut dbg_n: usize = 0;
    let dbg = std::env::var("GZIPPY_BARE_DEBUG").is_ok();
    let flush_hi = WINDOW + batch;

    loop {
        // FLUSH BEFORE the block so MAXBLK+SLOP headroom is guaranteed.
        if pos >= flush_hi {
            sink(&buf[window_len..pos]);
            total += pos - window_len;
            buf.copy_within(pos - WINDOW..pos, 0); // memmove-retain 32768
            pos = WINDOW;
            window_len = WINDOW;
        }

        // Read ONLY the next block header (igzip-native, _04-compatible tables);
        // the stopping point breaks before any body decode and restores the bit
        // reader to the block body.
        state.avail_out = 0;
        let _ = unsafe { isal::isal_inflate(&mut *state) };
        let bfinal = state.bfinal;

        if state.block_state == coded {
            let out_ptr = unsafe { base.add(pos) };
            state.next_out = out_ptr;
            state.avail_out = (cap - pos) as u32;
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
            assert!(
                state.block_state != coded,
                "igzipbare: block output exceeded MAXBLK headroom (raise MAXBLK)"
            );
            let produced = (state.next_out as usize) - (out_ptr as usize);
            if dbg && dbg_n < 6 {
                dbg_n += 1;
                eprintln!(
                    "blk CODED bfinal={bfinal} produced={produced} block_state={}",
                    state.block_state
                );
            }
            pos += produced;
        } else if state.block_state == type0 {
            // Stored block: read_header byte-aligned the bit reader at the data
            // and set type0_block_len = LEN (<=65535). Reconstruct the absolute
            // body offset, copy, then reseat the bit reader past the stored data.
            let len = unsafe { state.__bindgen_anon_1.type0_block_len } as usize;
            let consumed = (state.next_in as usize) - (input.as_ptr() as usize);
            let buffered = (state.read_in_length / 8) as usize;
            let data_off = consumed - buffered;
            unsafe {
                std::ptr::copy_nonoverlapping(input.as_ptr().add(data_off), base.add(pos), len);
            }
            pos += len;
            let after = data_off + len;
            state.next_in = unsafe { input.as_ptr().add(after) as *mut u8 };
            state.avail_in = (input.len() - after) as u32;
            state.read_in = 0;
            state.read_in_length = 0;
            state.block_state = if bfinal == 1 {
                isal::isal_block_state_ISAL_BLOCK_INPUT_DONE
            } else {
                isal::isal_block_state_ISAL_BLOCK_NEW_HDR
            };
        } else {
            panic!(
                "igzipbare: unexpected block_state {} after header read",
                state.block_state
            );
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

// ──────── CHEAP-HEADER bare kernel (clean-scaffold sizing, 2026-06-21) ────────
//
// IDENTICAL to `igzip_bare` (same gz contig driver, same memmove-retain window,
// same per-batch flush, same igzip `_04`/`_base` body kernel) EXCEPT the per-block
// HEADER read calls the lean `gzippy_read_header_export` (= file-local static
// `read_header`, the SAME function igzip's monolith calls inline) instead of
// re-entering the full stateful `isal_inflate` with avail_out=0 + the
// END_OF_BLOCK_HEADER stopping point.
//
// WHY: `igzip_bare`'s per-block `isal_inflate(stop=header)` pays outer dispatch +
// tmp_out setup + read-buffer save/restore EVERY block — overhead the igzip
// monolith does NOT (it reads headers inline). That artifact CONTAMINATES the
// `igzipbare - igzip` residual (only an UPPER bound on scaffold). This variant
// removes it, so `igzipbarecheap - igzip` is a CLEAN gz-contig-driver-vs-monolith
// scaffold (still gz-favorable: gz arms skip CRC, igzip computes it). The removed
// artifact is then directly measurable as `igzipbare - igzipbarecheap` > A/A
// spread (the NON-INERT proof the header overhead was real and is gone).
#[cfg(all(pure_inflate_decode, feature = "isal-compression"))]
fn igzip_bare_cheap<S: FnMut(&[u8])>(file_bytes: &[u8], batch: usize, mut sink: S) -> usize {
    use gzippy::decompress::parallel::gzip_format::read_header as gz_read_header;
    use isal::isal_sys::igzip_lib as isal;

    extern "C" {
        fn gzippy_read_header_export(state: *mut isal::inflate_state) -> i32;
        fn decode_huffman_code_block_stateless_04(
            state: *mut isal::inflate_state,
            start_out: *mut u8,
        );
        fn decode_huffman_code_block_stateless_base(
            state: *mut isal::inflate_state,
            start_out: *mut u8,
        );
    }

    let (_h, hdr) = gz_read_header(file_bytes).expect("gzip header");
    let mut input: Vec<u8> = file_bytes[hdr..].to_vec();
    input.extend_from_slice(&[0u8; 64]);

    const MAXBLK: usize = 8 * 1024 * 1024;
    let cap = WINDOW + batch + MAXBLK + SLOP;
    let mut buf = vec![0u8; cap];
    let base = buf.as_mut_ptr();

    let mut state: Box<isal::inflate_state> = Box::new(unsafe { std::mem::zeroed() });
    unsafe { isal::isal_inflate_init(&mut *state) };
    state.crc_flag = 0;
    state.hist_bits = 0;
    state.next_in = input.as_ptr() as *mut u8;
    state.avail_in = input.len() as u32;
    state.read_in = 0;
    state.read_in_length = 0;
    state.block_state = isal::isal_block_state_ISAL_BLOCK_NEW_HDR;

    let coded = isal::isal_block_state_ISAL_BLOCK_CODED;
    let type0 = isal::isal_block_state_ISAL_BLOCK_TYPE0;

    let mut pos: usize = 0;
    let mut window_len: usize = 0;
    let mut total: usize = 0;
    let flush_hi = WINDOW + batch;

    loop {
        if pos >= flush_hi {
            sink(&buf[window_len..pos]);
            total += pos - window_len;
            buf.copy_within(pos - WINDOW..pos, 0);
            pos = WINDOW;
            window_len = WINDOW;
        }

        // LEAN inline header read (no isal_inflate re-entry). read_header sets
        // block_state=CODED (setup_{static,dynamic}_header) or =TYPE0 (stored),
        // advances the bit reader to the block body, and records bfinal/btype.
        let ret = unsafe { gzippy_read_header_export(&mut *state as *mut _) };
        assert_eq!(
            ret, 0,
            "igzipbarecheap: read_header ret={ret} (ISAL_END_INPUT?)"
        );
        let bfinal = state.bfinal;

        if state.block_state == coded {
            let out_ptr = unsafe { base.add(pos) };
            state.next_out = out_ptr;
            state.avail_out = (cap - pos) as u32;
            unsafe {
                decode_huffman_code_block_stateless_04(&mut *state as *mut _, base);
                if state.block_state == coded {
                    decode_huffman_code_block_stateless_base(&mut *state as *mut _, base);
                }
            }
            assert!(
                state.block_state != coded,
                "igzipbarecheap: block output exceeded MAXBLK headroom (raise MAXBLK)"
            );
            let produced = (state.next_out as usize) - (out_ptr as usize);
            pos += produced;
        } else if state.block_state == type0 {
            let len = unsafe { state.__bindgen_anon_1.type0_block_len } as usize;
            let consumed = (state.next_in as usize) - (input.as_ptr() as usize);
            let buffered = (state.read_in_length / 8) as usize;
            let data_off = consumed - buffered;
            unsafe {
                std::ptr::copy_nonoverlapping(input.as_ptr().add(data_off), base.add(pos), len);
            }
            pos += len;
            let after = data_off + len;
            state.next_in = unsafe { input.as_ptr().add(after) as *mut u8 };
            state.avail_in = (input.len() - after) as u32;
            state.read_in = 0;
            state.read_in_length = 0;
            state.block_state = if bfinal == 1 {
                isal::isal_block_state_ISAL_BLOCK_INPUT_DONE
            } else {
                isal::isal_block_state_ISAL_BLOCK_NEW_HDR
            };
        } else {
            panic!(
                "igzipbarecheap: unexpected block_state {} after header read",
                state.block_state
            );
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
fn run_igzipbarecheap(file: &[u8], batch: usize) -> (f64, usize) {
    let mut out = devnull();
    let t = Instant::now();
    let n = igzip_bare_cheap(file, batch, |slice| {
        out.write_all(slice).expect("sink write");
    });
    out.flush().expect("flush");
    (t.elapsed().as_secs_f64() * 1e3, n)
}

// ───── SCAFFOLD-LOCATE: per-sub-region removal oracles (MEASUREMENT 3) ────────
//
// CLEAN SCAFFOLD = (cheap − igzip)/igzip is the gz-contig-driver-vs-igzip-monolith
// overhead, holding the inner decode constant (BOTH arms run igzip read_header +
// `_04`/`_base`). This family REMOVES one gz-driver sub-region at a time from the
// `cheap` arm so the interleaved wall response BOUNDS that sub-region's share of
// the scaffold (removal-oracle = the verdict, not a code-read). Each ablation is
// byte-exact (`total` still == ref) and prints a NON-INERT proof to stderr
// (sink_calls / memmove_calls / input_copied / cap) so we can confirm the
// perturbation actually fired and differs from baseline.
//
//   None        = baseline cheap (== igzip_bare_cheap)
//   NoSink      = remove the per-flush output sink writes (/dev/null write_all)
//   NoZero      = allocate the contig buffer UNINITIALIZED (remove the ~cap zeroing)
//   NoInputCopy = borrow file_bytes[hdr..] directly (remove the input .to_vec memcpy)
//   BigBuf      = one uninit buffer sized to the whole output → copy_within window
//                 memmove + per-batch flush NEVER fire (single final flush)
#[cfg(all(pure_inflate_decode, feature = "isal-compression"))]
#[derive(Clone, Copy, PartialEq, Debug)]
enum Ablate {
    None,
    NoSink,
    NoZero,
    NoInputCopy,
    BigBuf,
}

#[cfg(all(pure_inflate_decode, feature = "isal-compression"))]
fn igzip_bare_cheap_ab<S: FnMut(&[u8])>(
    file_bytes: &[u8],
    batch: usize,
    ab: Ablate,
    mut sink: S,
) -> usize {
    use gzippy::decompress::parallel::gzip_format::read_header as gz_read_header;
    use isal::isal_sys::igzip_lib as isal;

    extern "C" {
        fn gzippy_read_header_export(state: *mut isal::inflate_state) -> i32;
        fn decode_huffman_code_block_stateless_04(
            state: *mut isal::inflate_state,
            start_out: *mut u8,
        );
        fn decode_huffman_code_block_stateless_base(
            state: *mut isal::inflate_state,
            start_out: *mut u8,
        );
    }

    let (_h, hdr) = gz_read_header(file_bytes).expect("gzip header");

    // INPUT: NoInputCopy borrows file_bytes[hdr..] (the trailing 8-byte gzip
    // trailer + EOF acts as read-ahead slop); every other arm copies + zero-pads.
    let owned: Vec<u8>;
    let (in_ptr, in_len, input_copied): (*const u8, usize, u32) = if ab == Ablate::NoInputCopy {
        let s = &file_bytes[hdr..];
        (s.as_ptr(), s.len(), 0)
    } else {
        let mut v: Vec<u8> = file_bytes[hdr..].to_vec();
        v.extend_from_slice(&[0u8; 64]);
        let (p, l) = (v.as_ptr(), v.len());
        owned = v;
        let _ = &owned; // keep alive
        (p, l, 1)
    };

    const MAXBLK: usize = 8 * 1024 * 1024;
    // BigBuf: whole output fits → no mid-decode flush/memmove ever.
    let isize_hint = gzip_isize(file_bytes);
    let cap = if ab == Ablate::BigBuf {
        WINDOW + isize_hint + MAXBLK + SLOP
    } else {
        WINDOW + batch + MAXBLK + SLOP
    };
    // NoZero / BigBuf: uninitialized alloc (skip the cap zeroing). Safe here:
    // every byte read back (window prefix + flushed slice) was written by decode
    // or copy_within before any read.
    let mut buf: Vec<u8> = if ab == Ablate::NoZero || ab == Ablate::BigBuf {
        let mut v = Vec::with_capacity(cap);
        unsafe { v.set_len(cap) };
        v
    } else {
        vec![0u8; cap]
    };
    let base = buf.as_mut_ptr();

    let mut state: Box<isal::inflate_state> = Box::new(unsafe { std::mem::zeroed() });
    unsafe { isal::isal_inflate_init(&mut *state) };
    state.crc_flag = 0;
    state.hist_bits = 0;
    state.next_in = in_ptr as *mut u8;
    state.avail_in = in_len as u32;
    state.read_in = 0;
    state.read_in_length = 0;
    state.block_state = isal::isal_block_state_ISAL_BLOCK_NEW_HDR;

    let coded = isal::isal_block_state_ISAL_BLOCK_CODED;
    let type0 = isal::isal_block_state_ISAL_BLOCK_TYPE0;

    let mut pos: usize = 0;
    let mut window_len: usize = 0;
    let mut total: usize = 0;
    // BigBuf disables the mid-decode flush trigger.
    let flush_hi = if ab == Ablate::BigBuf {
        usize::MAX
    } else {
        WINDOW + batch
    };
    let mut sink_calls: u64 = 0;
    let mut memmove_calls: u64 = 0;

    loop {
        if pos >= flush_hi {
            if ab != Ablate::NoSink {
                sink(&buf[window_len..pos]);
            }
            sink_calls += 1;
            total += pos - window_len;
            buf.copy_within(pos - WINDOW..pos, 0);
            memmove_calls += 1;
            pos = WINDOW;
            window_len = WINDOW;
        }

        let ret = unsafe { gzippy_read_header_export(&mut *state as *mut _) };
        assert_eq!(ret, 0, "cheap_ab: read_header ret={ret}");
        let bfinal = state.bfinal;

        if state.block_state == coded {
            let out_ptr = unsafe { base.add(pos) };
            state.next_out = out_ptr;
            state.avail_out = (cap - pos) as u32;
            unsafe {
                decode_huffman_code_block_stateless_04(&mut *state as *mut _, base);
                if state.block_state == coded {
                    decode_huffman_code_block_stateless_base(&mut *state as *mut _, base);
                }
            }
            assert!(
                state.block_state != coded,
                "cheap_ab: block exceeded headroom"
            );
            let produced = (state.next_out as usize) - (out_ptr as usize);
            pos += produced;
        } else if state.block_state == type0 {
            let len = unsafe { state.__bindgen_anon_1.type0_block_len } as usize;
            let consumed = (state.next_in as usize) - (in_ptr as usize);
            let buffered = (state.read_in_length / 8) as usize;
            let data_off = consumed - buffered;
            unsafe {
                std::ptr::copy_nonoverlapping(in_ptr.add(data_off), base.add(pos), len);
            }
            pos += len;
            let after = data_off + len;
            state.next_in = unsafe { in_ptr.add(after) as *mut u8 };
            state.avail_in = (in_len - after) as u32;
            state.read_in = 0;
            state.read_in_length = 0;
            state.block_state = if bfinal == 1 {
                isal::isal_block_state_ISAL_BLOCK_INPUT_DONE
            } else {
                isal::isal_block_state_ISAL_BLOCK_NEW_HDR
            };
        } else {
            panic!("cheap_ab: unexpected block_state {}", state.block_state);
        }

        if bfinal == 1 {
            break;
        }
    }
    if pos > window_len {
        if ab != Ablate::NoSink {
            sink(&buf[window_len..pos]);
        }
        sink_calls += 1;
        total += pos - window_len;
    }
    eprintln!(
        "ABLATE {ab:?}: sink_calls={sink_calls} memmove_calls={memmove_calls} input_copied={input_copied} cap={cap}"
    );
    total
}

#[cfg(all(pure_inflate_decode, feature = "isal-compression"))]
fn run_cheap_ab(file: &[u8], batch: usize, ab: Ablate) -> (f64, usize) {
    let mut out = devnull();
    let t = Instant::now();
    let n = igzip_bare_cheap_ab(file, batch, ab, |slice| {
        out.write_all(slice).expect("sink write");
    });
    out.flush().expect("flush");
    (t.elapsed().as_secs_f64() * 1e3, n)
}

#[cfg(all(pure_inflate_decode, feature = "isal-compression"))]
fn run_cheap_mode(file: &[u8], batch: usize, mode: &str) -> Option<(f64, usize)> {
    let ab = match mode {
        "cheap_nosink" => Ablate::NoSink,
        "cheap_nozero" => Ablate::NoZero,
        "cheap_noin" => Ablate::NoInputCopy,
        "cheap_big" => Ablate::BigBuf,
        _ => return None,
    };
    Some(run_cheap_ab(file, batch, ab))
}

#[cfg(not(all(pure_inflate_decode, feature = "isal-compression")))]
fn run_cheap_mode(_file: &[u8], _batch: usize, mode: &str) -> Option<(f64, usize)> {
    match mode {
        "cheap_nosink" | "cheap_nozero" | "cheap_noin" | "cheap_big" => {
            panic!("cheap-ablation modes require gzippy-isal")
        }
        _ => None,
    }
}

#[cfg(not(all(pure_inflate_decode, feature = "isal-compression")))]
fn run_igzipbarecheap(_file: &[u8], _batch: usize) -> (f64, usize) {
    panic!("igzipbarecheap mode requires gzippy-isal (pure_inflate_decode + isal-compression)");
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
        gzippy::decompress::parallel::single_member::decompress_parallel(
            file, &mut pv, None, 1, false,
        )
        .expect("prod decode");
        assert!(pv == ld, "prod: BYTE MISMATCH");
        println!(
            "prod parallel-SM T=1 byte-exact OK (fnv={:016x})",
            fnv64(&pv)
        );
    }

    // igzip_bare (gz contig driver + igzip read_header + igzip _04 kernel) — the
    // de-confounded bare-inner-decode oracle. ASSERTED byte-exact by default
    // (set GZIPPY_SKIP_IGZIPBARE=1 to skip, e.g. on a non-isal build path).
    #[cfg(feature = "isal-compression")]
    if std::env::var("GZIPPY_SKIP_IGZIPBARE").is_err() {
        for &batch in &[64 * 1024usize, 1024 * 1024, DEFAULT_BATCH] {
            let mut v = Vec::new();
            igzip_bare(file, batch, |slice| v.extend_from_slice(slice));
            assert_eq!(
                v.len(),
                ld.len(),
                "igzipbare batch={batch}: length mismatch ({} vs {})",
                v.len(),
                ld.len()
            );
            assert!(v == ld, "igzipbare batch={batch}: BYTE MISMATCH");
            println!("igzipbare batch={batch:>8} byte-exact OK");

            let mut vc = Vec::new();
            igzip_bare_cheap(file, batch, |slice| vc.extend_from_slice(slice));
            assert_eq!(
                vc.len(),
                ld.len(),
                "igzipbarecheap batch={batch}: length mismatch ({} vs {})",
                vc.len(),
                ld.len()
            );
            assert!(vc == ld, "igzipbarecheap batch={batch}: BYTE MISMATCH");
            println!("igzipbarecheap batch={batch:>8} byte-exact OK");
        }
    } else {
        println!("igzipbare: SKIPPED verify (GZIPPY_SKIP_IGZIPBARE set)");
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
            "usage: streaming_thin <verify|gzippy|prod|libdeflate|zlibng|igzip|igzipbare|igzipbarecheap> <file.gz> [batch_bytes]"
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
        #[cfg(all(feature = "profile-rebuild", pure_inflate_decode))]
        "profile" => {
            let iters = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(40);
            let mut total = 0usize;
            for _ in 0..iters {
                total += gzippy_thin(&file, DEFAULT_BATCH, |_s| {});
            }
            println!(
                "RESULT mode=profile iters={iters} bytes_per_iter={}",
                total / iters
            );
            gzippy::decompress::parallel::lut_huffman::dump_rebuild_profile();
        }
        "prodt" => {
            let threads = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1);
            let (ms, n) = run_prodt(&file, threads);
            println!("RESULT mode=prodt threads={threads} ms={ms:.3} bytes={n}");
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
        "igzipbarecheap" => {
            let batch = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(DEFAULT_BATCH);
            let (ms, n) = run_igzipbarecheap(&file, batch);
            println!("RESULT mode=igzipbarecheap batch={batch} ms={ms:.3} bytes={n}");
        }
        "cheap_nosink" | "cheap_nozero" | "cheap_noin" | "cheap_big" => {
            let batch = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(DEFAULT_BATCH);
            let (ms, n) = run_cheap_mode(&file, batch, mode).expect("cheap mode");
            println!("RESULT mode={mode} batch={batch} ms={ms:.3} bytes={n}");
        }
        other => {
            eprintln!("unknown mode: {other}");
            std::process::exit(2);
        }
    }
}
