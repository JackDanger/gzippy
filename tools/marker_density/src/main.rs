//! KILL-TEST for structural hypothesis (A): "bound the slow u16 marker
//! decode to the first <=32 KiB of each chunk, then run a fast u8 clean
//! decoder for the rest."
//!
//! That hypothesis only works if marker DENSITY decays fast with output
//! offset. This tool decodes silesia starting at deep block boundaries
//! (simulating chunks N>0 that begin with an UNKNOWN window) using the
//! production `Block` decoder with `reset(None, None)` (no window), then
//! reports the fraction of output u16 values that are markers
//! (>= MARKER_BASE) in successive output windows.
//!
//! It does NOT stop at the 32-KiB clean-handoff boundary the way
//! production does — it keeps decoding so we see the TRUE density curve,
//! distinguishing "markers genuinely propagate and stay dense" from
//! "the handoff predicate just never armed."
//!
//! Usage: gzippy-marker-density <file.gz> [num_starts] [bytes_per_start]

use gzippy::decompress::inflate::consume_first_decode::Bits;
use gzippy::decompress::parallel::blockfinder_validation::find_blocks_parallel;
use gzippy::decompress::parallel::marker_inflate::Block;
use std::fs;

const MARKER_BASE: u16 = 32_768; // == MAX_WINDOW_SIZE
const BUCKET: usize = 8 * 1024; // output bytes per density bucket
const N_BUCKETS: usize = 64; // 64 * 8KiB = 512 KiB profiled per start

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: gzippy-marker-density <file.gz> [num_starts] [bytes_per_start]");
        std::process::exit(2);
    }
    let path = &args[0];
    let num_starts: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let bytes_per_start: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(N_BUCKETS * BUCKET);

    let gz = fs::read(path).expect("read gz");
    // Strip the gzip header to get to the raw DEFLATE stream. Minimal
    // parse: magic(2) method(1) flags(1) mtime(4) xfl(1) os(1) = 10,
    // plus optional FEXTRA/FNAME/FCOMMENT. silesia-large is a plain
    // gzip with no extra fields, so 10 is correct; we still scan flags.
    let deflate_start_byte = gzip_header_len(&gz);
    let deflate = &gz[deflate_start_byte..];

    // Find candidate block boundaries across the whole stream.
    let boundaries = find_blocks_parallel(deflate, 16);
    let valid: Vec<usize> = boundaries
        .iter()
        .filter(|b| b.valid)
        .map(|b| b.bit_offset)
        .collect();
    eprintln!(
        "deflate bytes={} found {} valid block boundaries",
        deflate.len(),
        valid.len()
    );
    if valid.len() < 4 {
        eprintln!("too few boundaries to profile");
        std::process::exit(1);
    }

    // Sample `num_starts` boundaries spread across the file, skipping the
    // very first few (those decode with the real zero-window in
    // production, so they wouldn't emit markers).
    let mut starts: Vec<usize> = Vec::new();
    let lo = valid.len() / 20; // skip first 5%
    let hi = valid.len() - 2;
    if hi <= lo {
        eprintln!("boundary range too small");
        std::process::exit(1);
    }
    for k in 0..num_starts {
        let idx = lo + (k * (hi - lo)) / num_starts.max(1);
        starts.push(valid[idx]);
    }

    // Per-bucket accumulation across all starts.
    let mut marker_sum = vec![0u64; N_BUCKETS];
    let mut total_sum = vec![0u64; N_BUCKETS];
    // Track, per start, the output offset at which marker fraction first
    // drops below 1% and stays there for a full bucket (the "clean
    // handoff could fire here" offset).
    let mut first_clean_offsets: Vec<Option<usize>> = Vec::new();
    let mut decoded_ok = 0usize;

    let mut block = Block::new();
    for &start_bit in &starts {
        let mut output: Vec<u16> = Vec::with_capacity(bytes_per_start + (1 << 17));
        block.reset(Some(&mut output), None);
        let byte_offset = start_bit / 8;
        let bit_in_byte = (start_bit % 8) as u32;
        if byte_offset >= deflate.len() {
            continue;
        }
        let mut bits = Bits::new(&deflate[byte_offset..]);
        if bit_in_byte > 0 {
            bits.consume(bit_in_byte);
        }

        let ok = decode_forward(&mut block, &mut bits, &mut output, bytes_per_start);
        if !ok {
            // Speculative start landed on a false boundary; skip.
            continue;
        }
        decoded_ok += 1;

        // Bucketize this start's output marker fraction.
        let mut first_clean: Option<usize> = None;
        for b in 0..N_BUCKETS {
            let s = b * BUCKET;
            if s >= output.len() {
                break;
            }
            let e = (s + BUCKET).min(output.len());
            let slice = &output[s..e];
            let m = slice.iter().filter(|&&v| v >= MARKER_BASE).count() as u64;
            marker_sum[b] += m;
            total_sum[b] += slice.len() as u64;
            let frac = m as f64 / slice.len() as f64;
            if first_clean.is_none() && frac < 0.01 {
                first_clean = Some(s);
            }
        }
        first_clean_offsets.push(first_clean);
    }

    println!("# decoded_ok_starts={decoded_ok} of {}", starts.len());
    println!("# bucket_KiB  marker_fraction  (avg over starts)");
    for b in 0..N_BUCKETS {
        if total_sum[b] == 0 {
            break;
        }
        let frac = marker_sum[b] as f64 / total_sum[b] as f64;
        // Simple ASCII bar.
        let bar = "#".repeat((frac * 50.0).round() as usize);
        println!("{:>6}     {:>7.4}   {}", b * BUCKET / 1024, frac, bar);
    }

    // Distribution of "first offset where marker fraction < 1%".
    let mut clean_kib: Vec<i64> = first_clean_offsets
        .iter()
        .map(|o| match o {
            Some(off) => (*off / 1024) as i64,
            None => -1, // never got clean within profiled window
        })
        .collect();
    clean_kib.sort_unstable();
    let never = clean_kib.iter().filter(|&&v| v < 0).count();
    let got: Vec<i64> = clean_kib.iter().copied().filter(|&v| v >= 0).collect();
    println!("\n# first-offset-where-marker-frac<1% (KiB into chunk):");
    if !got.is_empty() {
        let med = got[got.len() / 2];
        let p90 = got[(got.len() * 9 / 10).min(got.len() - 1)];
        let maxv = *got.last().unwrap();
        println!(
            "#   median={med} KiB  p90={p90} KiB  max={maxv} KiB  never_clean={never}/{}",
            clean_kib.len()
        );
    } else {
        println!("#   NONE of the starts reached <1% marker fraction within the profiled window (never_clean={never})");
    }
    println!(
        "#\n# INTERPRETATION: if median first-clean offset is small (e.g. <= 32-64 KiB),"
    );
    println!("# hypothesis (A) is viable: the slow u16 region is bounded and a fast u8");
    println!("# decoder can take the rest. If it is large / many never-clean, (A) is DEAD:");
    println!("# markers propagate and stay dense, so there is no short bounded u16 prefix.");
}

/// Decode forward across blocks until we have at least `target` output
/// bytes or hit BFINAL / an error. Returns false if the FIRST block
/// header/body fails to decode (false speculative boundary).
fn decode_forward(block: &mut Block, bits: &mut Bits, output: &mut Vec<u16>, target: usize) -> bool {
    let mut first_block = true;
    loop {
        if block.read_header(bits, false).is_err() {
            return !first_block;
        }
        while !block.eob() {
            if block.read(bits, output, usize::MAX).is_err() {
                return !first_block;
            }
            if output.len() >= target {
                return true;
            }
        }
        first_block = false;
        if block.is_last_block() {
            return true;
        }
        if output.len() >= target {
            return true;
        }
    }
}

/// Minimal gzip-header length parse (magic + optional FEXTRA/FNAME/FCOMMENT/FHCRC).
fn gzip_header_len(gz: &[u8]) -> usize {
    assert!(gz.len() > 10 && gz[0] == 0x1f && gz[1] == 0x8b, "not gzip");
    let flags = gz[3];
    let mut pos = 10usize;
    if flags & 0x04 != 0 {
        // FEXTRA
        let xlen = (gz[pos] as usize) | ((gz[pos + 1] as usize) << 8);
        pos += 2 + xlen;
    }
    if flags & 0x08 != 0 {
        // FNAME (zero-terminated)
        while gz[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }
    if flags & 0x10 != 0 {
        // FCOMMENT
        while gz[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }
    if flags & 0x02 != 0 {
        // FHCRC
        pos += 2;
    }
    pos
}
