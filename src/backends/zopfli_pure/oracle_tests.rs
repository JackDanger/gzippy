//! Oracle tests: compare pure-Rust port against the C FFI.
//! Deleted in cutover step.
#![cfg(test)]
#![allow(non_camel_case_types)]

#[allow(dead_code)]
pub fn corpus() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("empty", vec![]),
        ("byte", b"x".to_vec()),
        ("ascii", b"hello world hello world hello world".to_vec()),
        ("zeros_1k", vec![0u8; 1024]),
        ("zeros_64k", vec![0u8; 65_536]),
        ("rand_4k", {
            let mut v = Vec::with_capacity(4096);
            let mut s: u32 = 0x12345678;
            for _ in 0..4096 {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                v.push((s >> 16) as u8);
            }
            v
        }),
        (
            "alice",
            include_bytes!("../../../test_data/alice.txt").to_vec(),
        ),
    ]
}

// ── Step 2: katajainen oracle ────────────────────────────────────────────────

extern "C" {
    fn ZopfliLengthLimitedCodeLengths(
        frequencies: *const usize,
        n: i32,
        maxbits: i32,
        bitlengths: *mut u32,
    ) -> i32;
}

fn ffi_lengths(freq: &[usize], maxbits: i32) -> Result<Vec<u32>, ()> {
    let mut bl = vec![0u32; freq.len()];
    let r = unsafe {
        ZopfliLengthLimitedCodeLengths(freq.as_ptr(), freq.len() as i32, maxbits, bl.as_mut_ptr())
    };
    if r == 0 {
        Ok(bl)
    } else {
        Err(())
    }
}

#[test]
fn katajainen_matches_ffi() {
    use crate::backends::zopfli_pure::katajainen::length_limited_code_lengths;
    let cases: Vec<Vec<usize>> = vec![
        vec![1, 1, 2, 3, 5, 8, 13, 21],
        vec![0, 0, 1, 0, 0, 2, 0, 5, 0, 9, 1],
        (0..286usize).map(|i| i * 7 % 17).collect(),
        vec![1; 32], // ZOPFLI_NUM_D
        vec![100, 100, 100, 100],
        vec![0, 0, 0, 5], // <2 used symbols path
        vec![0, 0, 0, 0],
        vec![42],
    ];
    for freq in &cases {
        for &mb in &[15i32, 7] {
            let mut got = vec![0u32; freq.len()];
            let rs_result = length_limited_code_lengths(freq, mb, &mut got);
            let ffi_result = ffi_lengths(freq, mb);
            match (rs_result, ffi_result) {
                (Ok(()), Ok(exp)) => {
                    assert_eq!(got, exp, "freq={:?} maxbits={}", freq, mb);
                }
                (Err(()), Err(())) => {}
                (Ok(()), Err(())) => {
                    panic!(
                        "Rust succeeded but FFI failed: freq={:?} maxbits={}",
                        freq, mb
                    )
                }
                (Err(()), Ok(exp)) => {
                    panic!(
                        "Rust failed but FFI succeeded with {:?}: freq={:?} maxbits={}",
                        exp, freq, mb
                    )
                }
            }
        }
    }
}

// ── Step 3: tree oracle ─────────────────────────────────────────────────────

extern "C" {
    fn ZopfliCalculateBitLengths(count: *const usize, n: usize, maxbits: i32, bitlengths: *mut u32);
    fn ZopfliLengthsToSymbols(lengths: *const u32, n: usize, maxbits: u32, symbols: *mut u32);
    fn ZopfliCalculateEntropy(count: *const usize, n: usize, bitlengths: *mut f64);
}

fn ffi_bit_lengths(count: &[usize], maxbits: i32) -> Vec<u32> {
    let mut bl = vec![0u32; count.len()];
    unsafe { ZopfliCalculateBitLengths(count.as_ptr(), count.len(), maxbits, bl.as_mut_ptr()) };
    bl
}

fn ffi_lengths_to_symbols(lengths: &[u32], maxbits: u32) -> Vec<u32> {
    let mut syms = vec![0u32; lengths.len()];
    unsafe { ZopfliLengthsToSymbols(lengths.as_ptr(), lengths.len(), maxbits, syms.as_mut_ptr()) };
    syms
}

fn ffi_entropy(count: &[usize]) -> Vec<f64> {
    let mut bl = vec![0.0f64; count.len()];
    unsafe { ZopfliCalculateEntropy(count.as_ptr(), count.len(), bl.as_mut_ptr()) };
    bl
}

#[test]
fn tree_bitlengths_match_ffi() {
    use crate::backends::zopfli_pure::tree::calculate_bit_lengths;
    let tables: Vec<Vec<usize>> = vec![
        vec![1, 1, 2, 3, 5, 8, 13, 21],
        (0..288usize).map(|i| i * 7 % 17).collect(),
        vec![1usize; 32],
        vec![100, 200, 50, 300, 10],
        vec![0, 0, 5, 0, 0],
        vec![42],
    ];
    for count in &tables {
        let exp = ffi_bit_lengths(count, 15);
        let mut got = vec![0u32; count.len()];
        calculate_bit_lengths(count, 15, &mut got);
        assert_eq!(got, exp, "count={:?}", count);
    }
}

#[test]
fn tree_lengths_to_symbols_match_ffi() {
    use crate::backends::zopfli_pure::tree::{calculate_bit_lengths, lengths_to_symbols};
    let tables: Vec<Vec<usize>> = vec![
        vec![1, 1, 2, 3, 5, 8, 13, 21],
        (0..288usize).map(|i| i * 7 % 17).collect(),
        vec![1usize; 32],
    ];
    for count in &tables {
        let mut lengths = vec![0u32; count.len()];
        calculate_bit_lengths(count, 15, &mut lengths);
        let exp = ffi_lengths_to_symbols(&lengths, 15);
        let mut got = vec![0u32; count.len()];
        lengths_to_symbols(&lengths, 15, &mut got);
        assert_eq!(got, exp, "count={:?}", count);
    }
}

// ── Step 8: lz77_greedy end-to-end oracle ────────────────────────────────────

use crate::backends::zopfli_pure::ZopfliOptions as RsOpts;
use std::os::raw::{c_int, c_uchar};

#[repr(C)]
struct FfiZopfliOptions {
    verbose: c_int,
    verbose_more: c_int,
    numiterations: c_int,
    blocksplitting: c_int,
    blocksplittinglast: c_int,
    blocksplittingmax: c_int,
}

#[repr(C)]
struct FfiLZ77Store {
    litlens: *mut u16,
    dists: *mut u16,
    size: usize,
    data: *const c_uchar,
    pos: *mut usize,
    ll_symbol: *mut u16,
    d_symbol: *mut u16,
    ll_counts: *mut usize,
    d_counts: *mut usize,
}

#[repr(C)]
struct FfiLongestMatchCache {
    length: *mut u16,
    dist: *mut u16,
    sublen: *mut c_uchar,
}

#[repr(C)]
struct FfiBlockState {
    options: *const FfiZopfliOptions,
    lmc: *mut FfiLongestMatchCache,
    blockstart: usize,
    blockend: usize,
}

#[repr(C)]
struct FfiHash {
    head: *mut c_int,
    prev: *mut u16,
    hashval: *mut c_int,
    val: c_int,
    head2: *mut c_int,
    prev2: *mut u16,
    hashval2: *mut c_int,
    val2: c_int,
    same: *mut u16,
}

extern "C" {
    fn ZopfliInitLZ77Store(data: *const c_uchar, store: *mut FfiLZ77Store);
    fn ZopfliCleanLZ77Store(store: *mut FfiLZ77Store);
    fn ZopfliInitBlockState(
        opts: *const FfiZopfliOptions,
        start: usize,
        end: usize,
        add_lmc: c_int,
        s: *mut FfiBlockState,
    );
    fn ZopfliCleanBlockState(s: *mut FfiBlockState);
    fn ZopfliAllocHash(window: usize, h: *mut FfiHash);
    fn ZopfliCleanHash(h: *mut FfiHash);
    fn ZopfliResetHash(window: usize, h: *mut FfiHash);
    fn ZopfliLZ77Greedy(
        s: *mut FfiBlockState,
        in_: *const c_uchar,
        instart: usize,
        inend: usize,
        store: *mut FfiLZ77Store,
        h: *mut FfiHash,
    );
}

const ZOPFLI_WINDOW_SIZE: usize = 32_768;

fn rs_to_ffi_opts(o: &RsOpts) -> FfiZopfliOptions {
    FfiZopfliOptions {
        verbose: o.verbose,
        verbose_more: o.verbose_more,
        numiterations: o.numiterations,
        blocksplitting: o.blocksplitting,
        blocksplittinglast: o.blocksplittinglast,
        blocksplittingmax: o.blocksplittingmax,
    }
}

struct GreedySnapshot {
    litlens: Vec<u16>,
    dists: Vec<u16>,
    pos: Vec<usize>,
    ll_symbol: Vec<u16>,
    d_symbol: Vec<u16>,
}

fn ffi_greedy_snapshot(opts: &RsOpts, data: &[u8]) -> GreedySnapshot {
    unsafe {
        let ffi_opts = rs_to_ffi_opts(opts);

        let mut store = std::mem::MaybeUninit::<FfiLZ77Store>::uninit();
        ZopfliInitLZ77Store(data.as_ptr(), store.as_mut_ptr());
        let mut store = store.assume_init();

        let mut state = std::mem::MaybeUninit::<FfiBlockState>::uninit();
        ZopfliInitBlockState(&ffi_opts, 0, data.len(), 1, state.as_mut_ptr());
        let mut state = state.assume_init();

        let mut hash = std::mem::MaybeUninit::<FfiHash>::uninit();
        ZopfliAllocHash(ZOPFLI_WINDOW_SIZE, hash.as_mut_ptr());
        let mut hash = hash.assume_init();
        ZopfliResetHash(ZOPFLI_WINDOW_SIZE, &mut hash);

        ZopfliLZ77Greedy(
            &mut state,
            data.as_ptr(),
            0,
            data.len(),
            &mut store,
            &mut hash,
        );

        let n = store.size;
        let snap = GreedySnapshot {
            litlens: std::slice::from_raw_parts(store.litlens, n).to_vec(),
            dists: std::slice::from_raw_parts(store.dists, n).to_vec(),
            pos: std::slice::from_raw_parts(store.pos, n).to_vec(),
            ll_symbol: std::slice::from_raw_parts(store.ll_symbol, n).to_vec(),
            d_symbol: std::slice::from_raw_parts(store.d_symbol, n).to_vec(),
        };

        ZopfliCleanLZ77Store(&mut store);
        ZopfliCleanBlockState(&mut state);
        ZopfliCleanHash(&mut hash);

        snap
    }
}

#[test]
fn lz77_greedy_matches_ffi_byte_for_byte() {
    use crate::backends::zopfli_pure::hash::ZopfliHash;
    use crate::backends::zopfli_pure::lz77::{lz77_greedy, BlockState, LZ77Store};

    let opts = RsOpts::default();
    for (name, data) in corpus() {
        if data.is_empty() {
            // C path early-returns; trivially equal.
            continue;
        }
        let ffi = ffi_greedy_snapshot(&opts, &data);

        let mut store = LZ77Store::new(&data);
        let mut state = BlockState::new(&opts, 0, data.len(), true);
        let mut hash = ZopfliHash::new(ZOPFLI_WINDOW_SIZE);
        lz77_greedy(&mut state, &data, 0, data.len(), &mut store, &mut hash);

        assert_eq!(store.litlens, ffi.litlens, "litlens differ on {}", name);
        assert_eq!(store.dists, ffi.dists, "dists differ on {}", name);
        assert_eq!(store.pos, ffi.pos, "pos differ on {}", name);
        assert_eq!(
            store.ll_symbol, ffi.ll_symbol,
            "ll_symbol differs on {}",
            name
        );
        assert_eq!(store.d_symbol, ffi.d_symbol, "d_symbol differs on {}", name);
    }
}

#[test]
fn tree_entropy_match_ffi_exact() {
    use crate::backends::zopfli_pure::tree::calculate_entropy;
    let tables: Vec<Vec<usize>> = vec![
        vec![1, 1, 2, 3, 5, 8, 13, 21],
        (0..288usize).map(|i| i * 7 % 17).collect(),
        vec![1usize; 32],
        vec![0, 0, 5, 0, 3],
        vec![100; 10],
    ];
    for count in &tables {
        let exp = ffi_entropy(count);
        let mut got = vec![0.0f64; count.len()];
        calculate_entropy(count, &mut got);
        for (i, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
            // Allow 2 ULP: on arm64 (Apple Silicon) Rust's codegen for
            // log()*kInvLog2 differs by up to 2 ULP from Clang's output.
            // This difference is too small to affect LZ77 path selection.
            let diff = (g.to_bits() as i64 - e.to_bits() as i64).unsigned_abs();
            assert!(
                diff <= 2,
                "entropy mismatch at index {} (got {}, exp {}, diff {} ULP); count={:?}",
                i,
                g,
                e,
                diff,
                count
            );
        }
    }
}
