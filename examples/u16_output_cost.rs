//! Premortem mitigation A1: measure u16 output bandwidth cost vs. u8.
//!
//! Before sinking days into a u16-marker-emitting deflate decoder, prove that
//! the bandwidth doubling on the output side doesn't erase the parallelism
//! win. This synthetic benchmark models deflate's inner-loop memory access
//! pattern (literals + short/long back-references over a sliding 32 KB
//! window) in both u8 and u16 output modes — same control flow, same number
//! of symbols, only the per-symbol store width differs.
//!
//! Decision rule:
//! - If u16/u8 throughput ratio ≥ 0.5, the design is viable: a u16 marker
//!   pipeline that's "half the speed of u8 inflate per thread" still beats
//!   rapidgzip on CI because we get 4× from parallelism.
//! - If ratio < 0.3, the algorithm can't win on memory-bandwidth-limited
//!   hardware and the design must change.
//!
//! Run with: `cargo run --release --example u16_output_cost`.
//! (It's an `examples/` file, not a `benches/` harness — `cargo bench` won't
//! find it. Copilot review on PR #93 caught the original instruction.)

use std::time::Instant;

/// Number of output positions to produce per pass.
const OUTPUT_SYMBOLS: usize = 256 * 1024 * 1024;

/// Approximate fraction of symbols that are back-references (rest are
/// literals). Matches deflate's typical mix on natural text/binary.
const BACKREF_FRACTION: f32 = 0.4;

/// Distribution of back-reference lengths (length code histogram from
/// Silesia-like data). All in [3, 258] per RFC 1951.
const LENGTHS: &[u16] = &[3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 258];

/// Distribution of back-reference distances. Most refs are short (cache-local).
const DISTANCES: &[u16] = &[1, 4, 16, 64, 256, 1024, 4096, 16384, 32000];

#[inline(always)]
fn pseudo_random(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// Synthetic deflate-style decode into u8 output.
///
/// Approximates the inner loop of an inflate engine: read a "symbol" (here:
/// from a pseudo-random source, since the actual Huffman cost is not what
/// we're measuring), then either write a literal or copy a run from a
/// recent position. Mirrors the cache + bandwidth pattern of real inflate.
fn synthetic_decode_u8(output: &mut [u8]) -> usize {
    let mut rng: u64 = 0xfeedface;
    let mut pos = 0;
    while pos < output.len() {
        let r = pseudo_random(&mut rng);
        let is_backref = ((r as u32) as f32) * (1.0 / u32::MAX as f32) < BACKREF_FRACTION;

        if !is_backref || pos < 32 {
            output[pos] = (r >> 32) as u8;
            pos += 1;
        } else {
            let length = LENGTHS[((r >> 8) as usize) % LENGTHS.len()] as usize;
            let distance = DISTANCES[((r >> 16) as usize) % DISTANCES.len()] as usize;
            let distance = distance.min(pos);
            let length = length.min(output.len() - pos);
            // Byte-by-byte to keep behavior identical between u8 and u16 paths;
            // memcpy-style optimization is the same proportional factor for both.
            for k in 0..length {
                output[pos + k] = output[pos + k - distance];
            }
            pos += length;
        }
    }
    pos
}

/// Same synthetic decode into u16 output.
fn synthetic_decode_u16(output: &mut [u16]) -> usize {
    let mut rng: u64 = 0xfeedface;
    let mut pos = 0;
    while pos < output.len() {
        let r = pseudo_random(&mut rng);
        let is_backref = ((r as u32) as f32) * (1.0 / u32::MAX as f32) < BACKREF_FRACTION;

        if !is_backref || pos < 32 {
            output[pos] = (r >> 32) as u16;
            pos += 1;
        } else {
            let length = LENGTHS[((r >> 8) as usize) % LENGTHS.len()] as usize;
            let distance = DISTANCES[((r >> 16) as usize) % DISTANCES.len()] as usize;
            let distance = distance.min(pos);
            let length = length.min(output.len() - pos);
            for k in 0..length {
                output[pos + k] = output[pos + k - distance];
            }
            pos += length;
        }
    }
    pos
}

fn bench<F: FnMut() -> usize>(label: &str, mut f: F) -> f64 {
    // Warmup
    let _ = f();
    // Best of 3
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let n = f();
        let elapsed = t.elapsed().as_secs_f64();
        let mbps = (n as f64) / elapsed / 1e6;
        if elapsed < best {
            best = elapsed;
        }
        println!("  {label}: {mbps:>7.0} MB/s ({elapsed:.3}s for {n} symbols)");
    }
    (OUTPUT_SYMBOLS as f64) / best / 1e6
}

fn main() {
    println!("u16 output cost premortem (synthetic deflate access pattern)");
    println!(
        "Output symbols per pass: {} ({:.1} MiB u8, {:.1} MiB u16)",
        OUTPUT_SYMBOLS,
        OUTPUT_SYMBOLS as f64 / (1024.0 * 1024.0),
        (OUTPUT_SYMBOLS * 2) as f64 / (1024.0 * 1024.0),
    );
    println!();

    let mut buf_u8 = vec![0u8; OUTPUT_SYMBOLS];
    let mbps_u8 = bench("u8 ", || synthetic_decode_u8(&mut buf_u8));

    println!();
    let mut buf_u16 = vec![0u16; OUTPUT_SYMBOLS];
    let mbps_u16 = bench("u16", || synthetic_decode_u16(&mut buf_u16));

    println!();
    let ratio = mbps_u16 / mbps_u8;
    println!("u16/u8 throughput ratio: {ratio:.2}");
    println!();
    println!("Decision rule:");
    println!("  ≥ 0.50 → u16 marker decoder is viable; proceed with port.");
    println!("  0.30..0.50 → marginal; check on x86_64 CI before committing.");
    println!("  < 0.30 → memory bandwidth dominates; pivot to a different design.");
}
