//! Phase 0.6 spike: does iai-callgrind/cachegrind count SIMD instructions
//! deterministically and at a granularity that distinguishes SIMD from scalar?
//!
//! Test strategy: pair scalar and SIMD impls of two operations, decode their
//! cachegrind output, see whether:
//!
//!   (a) SIMD impl shows a different instruction count than scalar (good —
//!       cachegrind notices the SIMD instructions exist).
//!   (b) SIMD impl shows MUCH lower instruction count than scalar (cachegrind
//!       counts each SIMD instruction as one op but doesn't model widened
//!       throughput; the instruction-count metric will favor SIMD variants,
//!       so a ±2% baseline check on it will FAIL when a SIMD variant lands).
//!   (c) Counts are identical (cachegrind treats SIMD opaquely — useless
//!       for SIMD variant gating).
//!
//! Outcomes drive plan adjustment per Phase 0.6.

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

// ── Test 1: popcount ─────────────────────────────────────────────────────────

/// Scalar popcount over a fixed-length array.
#[inline(never)]
fn popcount_scalar(input: &[u64; 256]) -> u32 {
    let mut total: u32 = 0;
    for &x in input.iter() {
        total = total.wrapping_add(x.count_ones());
    }
    total
}

/// SIMD popcount using LLVM intrinsics via `count_ones()` on a u64 — which
/// on x86_64 with popcnt feature available compiles to `popcnt` (a scalar
/// SSE instruction, not vector). We use this as a control: instruction count
/// should still differ from a pure-bitops scalar implementation.
#[inline(never)]
fn popcount_popcnt_intrinsic(input: &[u64; 256]) -> u32 {
    // Explicit unsafe path using std::arch popcnt if available.
    #[cfg(target_feature = "popcnt")]
    unsafe {
        use std::arch::x86_64::_popcnt64;
        let mut total: u32 = 0;
        for &x in input.iter() {
            total = total.wrapping_add(_popcnt64(x as i64) as u32);
        }
        total
    }
    #[cfg(not(target_feature = "popcnt"))]
    {
        popcount_scalar(input)
    }
}

/// Bit-twiddling scalar popcount (Hamming weight, no popcnt instruction).
/// LLVM may still recognize and lower to popcnt; we use `black_box` to fight
/// that, or accept that recognition itself is interesting data.
#[inline(never)]
fn popcount_bittwiddle(input: &[u64; 256]) -> u32 {
    let mut total: u32 = 0;
    for &x_in in input.iter() {
        let mut x = black_box(x_in);
        x = x - ((x >> 1) & 0x5555555555555555);
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
        total = total.wrapping_add(((x.wrapping_mul(0x0101010101010101)) >> 56) as u32);
    }
    total
}

fn make_popcount_input() -> [u64; 256] {
    let mut out = [0u64; 256];
    for (i, slot) in out.iter_mut().enumerate() {
        // Pseudo-random fill so popcounts vary.
        *slot = (i as u64)
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0xBF58476D1CE4E5B9);
    }
    out
}

// ── Test 2: narrow u16→u8 (the actual SIMD path we'd use in Phase 5) ────────

/// Genuinely scalar narrow: u16[N] → u8[N], truncating cast.
/// `black_box` inside the loop body defeats LLVM's auto-vectorizer.
/// Without this, `target-cpu=x86-64-v3` lowers a naive zip-iter loop to
/// AVX2 packus, making the "scalar" baseline indistinguishable from the
/// hand-rolled AVX2 variant.
#[inline(never)]
fn narrow_scalar(src: &[u16; 1024], dst: &mut [u8; 1024]) {
    for i in 0..1024 {
        // black_box on the read prevents LLVM from recognizing the
        // memory-to-memory truncating-narrow pattern. Per-element load
        // and store are scalar.
        let v = black_box(src[i]);
        dst[i] = v as u8;
    }
}

/// SIMD narrow using AVX2 _mm256_packus_epi16. Saturating pack, but inputs
/// are all < 256 so saturation is a no-op. This is the EXACT path used in
/// `narrow_u16_to_u8_avx2` in production (chunk_fetcher.rs).
#[inline(never)]
#[cfg(target_feature = "avx2")]
fn narrow_avx2(src: &[u16; 1024], dst: &mut [u8; 1024]) {
    unsafe {
        use std::arch::x86_64::*;
        let mut i = 0;
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();
        // Process 32 u16 → 32 u8 per iteration via two _mm256_loadu_si256
        // + _mm256_packus_epi16 + permute.
        while i + 32 <= 1024 {
            let a = _mm256_loadu_si256(src_ptr.add(i) as *const _);
            let b = _mm256_loadu_si256(src_ptr.add(i + 16) as *const _);
            let packed = _mm256_packus_epi16(a, b);
            let permuted = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
            _mm256_storeu_si256(dst_ptr.add(i) as *mut _, permuted);
            i += 32;
        }
        while i < 1024 {
            *dst_ptr.add(i) = *src_ptr.add(i) as u8;
            i += 1;
        }
    }
}

#[cfg(not(target_feature = "avx2"))]
fn narrow_avx2(src: &[u16; 1024], dst: &mut [u8; 1024]) {
    narrow_scalar(src, dst);
}

fn make_narrow_input() -> Box<([u16; 1024], [u8; 1024])> {
    let mut input = [0u16; 1024];
    for (i, slot) in input.iter_mut().enumerate() {
        // Values < 256 so PACKUSWB saturation is a no-op.
        *slot = (i as u16) & 0xFF;
    }
    Box::new((input, [0u8; 1024]))
}

// ── iai-callgrind harness ────────────────────────────────────────────────────

#[library_benchmark]
fn bench_popcount_scalar() -> u32 {
    let input = make_popcount_input();
    black_box(popcount_scalar(black_box(&input)))
}

#[library_benchmark]
fn bench_popcount_popcnt() -> u32 {
    let input = make_popcount_input();
    black_box(popcount_popcnt_intrinsic(black_box(&input)))
}

#[library_benchmark]
fn bench_popcount_bittwiddle() -> u32 {
    let input = make_popcount_input();
    black_box(popcount_bittwiddle(black_box(&input)))
}

#[library_benchmark]
fn bench_narrow_scalar() -> u32 {
    let mut buf = make_narrow_input();
    narrow_scalar(&buf.0, &mut buf.1);
    // Return a digest so the result is observable.
    black_box(buf.1.iter().fold(0u32, |a, &b| a.wrapping_add(b as u32)))
}

#[library_benchmark]
fn bench_narrow_avx2() -> u32 {
    let mut buf = make_narrow_input();
    narrow_avx2(&buf.0, &mut buf.1);
    black_box(buf.1.iter().fold(0u32, |a, &b| a.wrapping_add(b as u32)))
}

library_benchmark_group!(
    name = popcount;
    benchmarks = bench_popcount_scalar, bench_popcount_popcnt, bench_popcount_bittwiddle
);

library_benchmark_group!(
    name = narrow;
    benchmarks = bench_narrow_scalar, bench_narrow_avx2
);

main!(library_benchmark_groups = popcount, narrow);
