//! Phase 11.2 corpus oracle: compare zopfli_pure's output to the
//! vendored C zopfli reference byte-for-byte across a fixed corpus.
//!
//! Gated on the `oracle` cargo feature so plain `cargo test` stays
//! fast. Run via `cargo test --features oracle -- --include-ignored`
//! or `make oracle-vs-c`.
//!
//! Both implementations write the exact same 10-byte gzip header
//! (ID1 ID2 CM=8 FLG=0 MTIME=0,0,0,0 XFL=2 OS=3 — see
//! `vendor/zopfli/src/zopfli/gzip_container.c:90-101` and our
//! port at `gzip.rs:20-29`), so the entire gzip output is comparable
//! byte-for-byte. No header stripping required.

#![cfg(feature = "oracle")]

// Imported via `crate::` rather than `super::` because this file is included
// by `src/lib.rs` directly (via `#[path]`), so its module parent is the crate
// root, not `crate::compress::deflate::parse::ultra`. See lib.rs /
// ultra/mod.rs for the rationale (avoids dual lib/bin compilation of an
// FFI-using test).
use crate::compress::deflate::parse::ultra::{compress, ZopfliFormat, ZopfliOptions};

// `blocksplittingmax` (and `blocksplittinglast`) are dead on the Rust side
// (`ZopfliOptions` no longer carries them — the splitter has been
// unconditionally uncapped since the crown-caps change, see
// `deflate.rs`'s "UNCAP" comment) but this struct's layout MUST still
// mirror the real C `ZopfliOptions` (`vendor/zopfli/src/zopfli/zopfli.h`)
// byte-for-byte for the FFI call below to be sound — so the fields stay
// here, in this FFI shim only.
#[repr(C)]
#[derive(Clone, Copy)]
struct CZopfliOptions {
    verbose: i32,
    verbose_more: i32,
    numiterations: i32,
    blocksplitting: i32,
    blocksplittinglast: i32,
    blocksplittingmax: i32,
}

#[allow(non_camel_case_types)]
type c_uchar = u8;
#[allow(non_camel_case_types)]
type c_size_t = usize;

unsafe extern "C" {
    fn ZopfliInitOptions(options: *mut CZopfliOptions);
    fn ZopfliCompress(
        options: *const CZopfliOptions,
        output_type: i32, // ZOPFLI_FORMAT_GZIP = 0
        input: *const c_uchar,
        insize: c_size_t,
        out: *mut *mut c_uchar,
        outsize: *mut c_size_t,
    );
}

const ZOPFLI_FORMAT_GZIP: i32 = 0;

fn c_compress(input: &[u8], iters: i32, splitting: i32, max_blocks: i32) -> Vec<u8> {
    unsafe {
        let mut opts: CZopfliOptions = std::mem::zeroed();
        ZopfliInitOptions(&mut opts);
        opts.numiterations = iters;
        opts.blocksplitting = splitting;
        opts.blocksplittingmax = max_blocks;

        let mut out: *mut c_uchar = std::ptr::null_mut();
        let mut outsize: c_size_t = 0;
        ZopfliCompress(
            &opts,
            ZOPFLI_FORMAT_GZIP,
            input.as_ptr(),
            input.len(),
            &mut out,
            &mut outsize,
        );
        // The C library uses ZOPFLI_APPEND_DATA which mallocs in chunks
        // via realloc; we own the resulting buffer and must free it.
        let copied = std::slice::from_raw_parts(out, outsize).to_vec();
        libc::free(out as *mut _);
        copied
    }
}

// `_max_blocks` kept for signature symmetry with `c_compress` above (same
// combos are driven through both) even though the Rust `ZopfliOptions` no
// longer has a field to carry it into — see the `CZopfliOptions` doc comment.
fn rust_compress(input: &[u8], iters: i32, splitting: i32, _max_blocks: i32) -> Vec<u8> {
    let opts = ZopfliOptions {
        verbose: 0,
        verbose_more: 0,
        numiterations: iters,
        blocksplitting: splitting,
        thread_budget: 0,
    };
    compress(&opts, ZopfliFormat::Gzip, input)
}

/// Deterministic LCG so the oracle corpus is reproducible across runs
/// and machines.
fn lcg(seed: u32, len: usize) -> Vec<u8> {
    let mut s: u32 = seed;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        v.push((s >> 16) as u8);
    }
    v
}

fn corpus() -> Vec<(&'static str, Vec<u8>)> {
    let mut out: Vec<(&'static str, Vec<u8>)> = vec![
        ("empty", vec![]),
        ("byte", b"x".to_vec()),
        ("ascii", b"hello world hello world hello world".to_vec()),
        ("zeros_1k", vec![0u8; 1024]),
        ("zeros_64k", vec![0u8; 65_536]),
        ("rand_1k", lcg(0x1234_5678, 1_024)),
        ("rand_64k", lcg(0xdead_beef, 65_536)),
        ("yes_abc", b"yesabcyesabcyesabc".repeat(512)),
        (
            "byte_runs",
            (0..256u32)
                .flat_map(|b| std::iter::repeat_n(b as u8, 32))
                .collect(),
        ),
        // Small repeating mix — the kind of input that exposes both
        // literal and back-reference paths.
        (
            "mixed_repeat",
            b"the quick brown fox jumps over the lazy dog. ".repeat(20),
        ),
    ];
    if let Ok(alice) = std::fs::read("test_data/alice.txt") {
        out.push(("alice.txt", alice));
    }
    out
}

/// Pareto oracle against the vendored C zopfli. Since the LzFind binary-tree
/// matchfinder (DEFLATE-CROWN) intentionally produces *different* bytes than C
/// zopfli's hash-chain parse — that is the point, ECT-style better ratio — the
/// old byte-identity check is obsolete. Instead we require, for every corpus
/// input and option combo, that our output (a) round-trips exactly through an
/// independent decoder and (b) is no larger than C zopfli's (ratio must not
/// regress vs the reference).
#[test]
#[ignore = "requires `--features oracle` and an init'd vendor/zopfli submodule; run via `make oracle-vs-c`"]
fn corpus_gzip_pareto_vs_c_zopfli() {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let mut failures: Vec<String> = Vec::new();
    let corpus = corpus();

    // The combinations the plan calls out: numiterations ∈ {1, 5, 30}
    // (15 is the default and is implicitly tested by the fixture
    // suite) × blocksplitting ∈ {0, 1}. maxblocks=15 matches default.
    let combos: &[(i32, i32, i32)] = &[
        (1, 1, 15),
        (5, 1, 15),
        (30, 1, 15),
        (1, 0, 15),
        (15, 0, 15),
        // explicit maxblocks=0 (unlimited) under default splitting
        (15, 1, 0),
    ];

    for (name, input) in &corpus {
        for &(iters, splitting, max_blocks) in combos {
            let c = c_compress(input, iters, splitting, max_blocks);
            let r = rust_compress(input, iters, splitting, max_blocks);

            // (a) Independent-decoder round-trip (never weakened).
            let mut decoded = Vec::new();
            match GzDecoder::new(&r[..]).read_to_end(&mut decoded) {
                Ok(_) if decoded == *input => {}
                Ok(_) => failures.push(format!(
                    "  {name} (iters={iters} splitting={splitting} maxblocks={max_blocks}): \
                     round-trip content mismatch"
                )),
                Err(e) => failures.push(format!(
                    "  {name} (iters={iters} splitting={splitting} maxblocks={max_blocks}): \
                     gunzip failed: {e}"
                )),
            }

            // (b) Ratio must not regress vs C zopfli.
            if r.len() > c.len() {
                failures.push(format!(
                    "  {name} (iters={iters} splitting={splitting} maxblocks={max_blocks}): \
                     ratio regression C={}B Rust={}B (+{}B)",
                    c.len(),
                    r.len(),
                    r.len() - c.len()
                ));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "zopfli_pure failed the Pareto oracle vs vendor/zopfli on {} \
             (corpus, opts) pair(s):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
