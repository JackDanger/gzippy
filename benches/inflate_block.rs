//! Phase 1.2 microbench harness (see plans/inner-loop-execution.md).
//!
//! Measures gzippy's inner-loop decode rate on the `corpus/silesia_blocks/`
//! corpus. Two bench groups:
//!
//! - `inner_loop_only` — window pre-seeded; timed region is `read_header` +
//!   body loop. Optimization target.
//! - `from_scratch` — timed region includes `set_initial_window`. What
//!   production pays per block.
//!
//! Headline metric: aggregate MB/s across all 30 blocks (weighted by
//! decoded bytes). Per-block MB/s is a shape-regression detector only,
//! not the optimization target (per Opus advisor: large blocks hit
//! warm-cache, small blocks hit cold-start; per-block MB/s is misleading
//! as a primary metric).

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use gzippy::decompress::inflate::consume_first_decode::Bits;
use gzippy::decompress::parallel::deflate_block::Block;

#[path = "common/corpus_loader.rs"]
mod corpus_loader;
use corpus_loader::{load_corpus, corpus_dir, CorpusBlock};

/// Decode one block from a corpus entry with the window already seeded.
/// Buffers come from the caller (hoisted; allocator noise not measured).
#[inline(always)]
fn decode_seeded(
    block_data: &CorpusBlock,
    decoder: &mut Block,
    out_buf: &mut Vec<u16>,
    body_buf: &mut Vec<u16>,
) {
    *decoder = Block::new();
    out_buf.clear();
    decoder
        .set_initial_window(out_buf, &block_data.predecessor)
        .expect("set_initial_window");
    out_buf.clear();
    body_buf.clear();
    let mut bits = Bits::at_bit_offset(
        &block_data.compressed,
        block_data.bit_offset_in_compressed as usize,
    );
    decoder.read_header(&mut bits, false).expect("read_header");
    while !decoder.eob() {
        decoder
            .read(&mut bits, body_buf, usize::MAX)
            .expect("read body");
    }
}

/// Decode INCLUDING the set_initial_window cost (matches production
/// per-block pay).
#[inline(always)]
fn decode_from_scratch(
    block_data: &CorpusBlock,
    decoder: &mut Block,
    out_buf: &mut Vec<u16>,
    body_buf: &mut Vec<u16>,
) {
    *decoder = Block::new();
    out_buf.clear();
    body_buf.clear();
    decoder
        .set_initial_window(out_buf, &block_data.predecessor)
        .expect("set_initial_window");
    out_buf.clear();
    let mut bits = Bits::at_bit_offset(
        &block_data.compressed,
        block_data.bit_offset_in_compressed as usize,
    );
    decoder.read_header(&mut bits, false).expect("read_header");
    while !decoder.eob() {
        decoder
            .read(&mut bits, body_buf, usize::MAX)
            .expect("read body");
    }
}

fn bench_corpus_aggregate(c: &mut Criterion) {
    let corpus = load_corpus(corpus_dir());
    assert!(!corpus.is_empty(), "empty corpus directory");
    let total_decoded: u64 = corpus.iter().map(|b| b.decoded_expected.len() as u64).sum();
    let max_decoded = corpus
        .iter()
        .map(|b| b.decoded_expected.len())
        .max()
        .expect("non-empty corpus");

    let mut group = c.benchmark_group("inflate_block_aggregate");
    group.throughput(Throughput::Bytes(total_decoded));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(200);

    let mut decoder = Block::new();
    let mut out_buf: Vec<u16> = Vec::with_capacity(max_decoded);
    let mut body_buf: Vec<u16> = Vec::with_capacity(max_decoded);

    group.bench_function("inner_loop_only", |b| {
        b.iter(|| {
            for block_data in &corpus {
                decode_seeded(block_data, &mut decoder, &mut out_buf, &mut body_buf);
                black_box(&body_buf);
            }
        });
    });

    group.bench_function("from_scratch", |b| {
        b.iter(|| {
            for block_data in &corpus {
                decode_from_scratch(block_data, &mut decoder, &mut out_buf, &mut body_buf);
                black_box(&body_buf);
            }
        });
    });

    group.finish();
}

fn bench_corpus_per_block(c: &mut Criterion) {
    let corpus = load_corpus(corpus_dir());
    let max_decoded = corpus
        .iter()
        .map(|b| b.decoded_expected.len())
        .max()
        .expect("non-empty corpus");

    let mut group = c.benchmark_group("inflate_block_per_block");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(100);

    let mut decoder = Block::new();
    let mut out_buf: Vec<u16> = Vec::with_capacity(max_decoded);
    let mut body_buf: Vec<u16> = Vec::with_capacity(max_decoded);

    for block_data in &corpus {
        group.throughput(Throughput::Bytes(block_data.decoded_expected.len() as u64));
        group.bench_function(&block_data.file_name, |b| {
            b.iter(|| {
                decode_seeded(block_data, &mut decoder, &mut out_buf, &mut body_buf);
                black_box(&body_buf);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_corpus_aggregate, bench_corpus_per_block);
criterion_main!(benches);
