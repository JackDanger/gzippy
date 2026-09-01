//! T-SCOPE: **SHARED** by T=1 and T>1. This is the encoder ENGINE, not a driver.
//!
//! `CLAUDE.md`'s "three separate paths" is an ISOLATION constraint on the drivers,
//! not an instruction to keep three copies of a parser: T=1 calls into this engine
//! directly, and the T>1 driver (`compress/pipelined.rs`) calls the same engine once
//! per chunk. A change here moves BOTH boards, so it must be measured at T=1 AND at
//! a T>1 thread count before it is believed.
//!
//! Pure-Rust DEFLATE encoder — Increment 2 (hash-chain matchfinder + parsers).
//!
//! This module is the entry point for a from-scratch, pure-Rust DEFLATE/gzip
//! compressor whose structure transliterates libdeflate
//! (`vendor/libdeflate/lib/deflate_compress.c`). Increment 1 landed the proven
//! substrate — constant [`tables`], the word-oriented [`bitstream`], the
//! length-limited canonical [`huffman`] builder + dynamic-block header, the
//! [`block_split`] statistic, and the shared [`matchfinder`] primitives.
//!
//! Increment 2 adds REAL compression: the hash-chains matchfinder
//! ([`matchfinder::hc`], a port of `hc_matchfinder.h`), the level→params table
//! ([`level`]), and the greedy / lazy / lazy2 [`parse`]rs (levels 2-9). Each
//! block chooses the cheapest of a stored, static-Huffman, or dynamic-Huffman
//! encoding of the parsed literal/back-reference token stream. Matches share a
//! 32 KiB window across block boundaries, exactly as DEFLATE allows.
//!
//! Correctness is pinned by `src/tests/deflate_encoder_matches.rs`: byte-exact
//! roundtrip through flate2, libdeflate (FFI), and system `gzip -d` for every
//! implemented level, plus a proptest generator. As of Increment 7 this is the
//! SOLE production compress engine — every level 0–12, T1 and T>1 (via
//! `pipelined::compress_buffer_pure`) — with the C-FFI backends removed from the
//! routing graph.
//!
//! Dead-code audit (Stage E, docs/compressor-architecture.md §5-E,
//! 2026-07-21): the blanket `#![allow(dead_code)]` this module carried since
//! Increment 1 ("some substrate primitives are used only by later
//! increments") is REMOVED — near-optimal/ultra landed in Stages A-D, so the
//! excuse no longer holds, and a `cargo build --release` with the allow
//! stripped is now warning-clean. Five genuinely-unreferenced items found
//! that way were deleted (`BitWriter::with_capacity`/`buffered_bits`,
//! `HcMatchfinder::reset`, `tables::DEFLATE_MAX_NUM_SYMS`/
//! `DEFLATE_MAX_CODEWORD_LEN` — zero callers in production OR tests). One
//! item, `level::max_passthrough_size`, has test coverage but no production
//! call site (a libdeflate port never wired into the near-optimal entry
//! point); it keeps its own narrow `#[allow(dead_code)]` with a doc note
//! rather than a blanket module allow, so the compiler will flag anything
//! ELSE that goes dead in the future.

pub mod anatomy_counters;
pub mod anatomy_wall;
pub mod bitstream;
pub mod block_cost_probe;
pub mod block_split;
pub mod costs;
pub mod encode_types;
pub mod huffman;
pub mod level;
pub mod matchfinder;
pub mod parse;
pub mod tables;

use bitstream::BitWriter;
use tables::DEFLATE_BLOCKTYPE_UNCOMPRESSED;

/// Largest payload of a single stored (BTYPE=00) sub-block.
const MAX_STORED_SUBBLOCK: usize = 65535;

/// The deterministic gzip header emitted by `libdeflate-gzip -c`.
///
/// XFL is metadata, but it is part of byte-for-byte gzip compatibility: the
/// vendor marks its fastest level with 4 and its maximum-compression levels
/// with 2.
#[inline]
pub(crate) fn minimal_gzip_header(level: u32) -> [u8; 10] {
    let xfl = match level {
        1 => 4,
        8..=u32::MAX => 2,
        _ => 0,
    };
    [0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, xfl, 0xff]
}

/// Output-buffer capacity estimate for a one-shot compress of `len` bytes at
/// `level`, plus `framing_slack` bytes for whatever header/trailer the caller
/// adds around the raw DEFLATE stream (gzip header + CRC32 + ISIZE = 18 for
/// the gzip entry points; the raw-DEFLATE entry points pass a smaller slack).
///
/// Levels 1-12 keep the pre-existing `len/2 + slack` guess (real compression,
/// unknown ratio ahead of time). Level 0 is now genuine STORED-only (see
/// `deflate_into`'s `level == 0` branch): output is `len` plus 5 bytes of
/// framing per 65535-byte sub-block, so `len/2` would under-size by ~2x and
/// force `Vec` to repeatedly grow-and-copy while writing what is otherwise a
/// memcpy-class encode — exactly the constant-factor tax a stored mode must
/// not carry if it is to race pigz's memcpy at the wall.
#[inline]
fn estimate_output_cap(len: usize, level: u32, framing_slack: usize) -> usize {
    if level == 0 {
        len + len.div_ceil(MAX_STORED_SUBBLOCK).max(1) * 5 + framing_slack
    } else {
        len / 2 + framing_slack
    }
}

/// Compress `data` into a raw DEFLATE stream (no gzip/zlib framing) at `level`.
pub fn encode_deflate_bytes_to_vec(data: &[u8], level: u32) -> Vec<u8> {
    let cap = estimate_output_cap(data.len(), level, 64);
    crate::anatomy_count!(alloc_events);
    crate::anatomy_count!(alloc_bytes, cap);
    let mut out = Vec::with_capacity(cap);
    encode_deflate_bytes_to_sink(data, &[], level, &mut out);
    out
}

/// Number of trailing pad bytes a caller-owned buffer must carry past the
/// logical input end so the matchfinder's speculative word loads always stay in
/// bounds. Re-exported for the in-place T1 path, which pads its read buffer once
/// (`resize(len + PAD, 0)`) rather than copying the input into a second padded
/// buffer. Must equal [`parse::BUF_PAD`].
pub const INPLACE_TAIL_PAD: usize = parse::BUF_PAD;

/// Compress `data` into a raw DEFLATE stream, appending to `out`.
///
/// `dict` is an optional preset-dictionary window: its bytes are seeded into the
/// matchfinder so back-references in the coded output may point into it, but the
/// dictionary itself is not emitted. The decoder must have the identical window
/// preloaded. Pass `&[]` for no dictionary (the gzip/single-member case).
pub fn encode_deflate_bytes_to_sink(data: &[u8], dict: &[u8], level: u32, out: &mut Vec<u8>) {
    // A standalone single final block: BFINAL is set on the last internal
    // block and no sync-flush marker is appended. `bw.finish()` byte-aligns
    // the tail. This is the T1 / single-member framing.
    encode_deflate_segment_to_sink(data, dict, level, true, out, false);
}

/// Compress `data` into a raw DEFLATE stream for use as ONE CHUNK of a larger
/// concatenated single-member stream, appending to `out`.
///
/// Identical to [`encode_deflate_bytes_to_sink`] except for the stream-position semantics
/// controlled by `is_last`:
///
/// * `is_last == true` — this chunk closes the stream. The last internal block
///   carries `BFINAL=1` and NOTHING is appended after it; `bw.finish()`
///   byte-aligns the tail. With an empty `dict` this is byte-identical to
///   [`encode_deflate_bytes_to_sink`] (the single-member case).
/// * `is_last == false` — this chunk is followed by more chunks. Every internal
///   block (including the last) stays `BFINAL=0`, and a byte-aligned empty
///   stored block — the standard `Z_SYNC_FLUSH` marker
///   `[BFINAL=0][BTYPE=00][align][LEN=0000][NLEN=FFFF]` — is appended so the
///   chunk ends on a clean, byte-aligned block boundary.
///
/// The sync-flush suffix is the load-bearing correctness detail: independently
/// compressed chunks concatenate into ONE valid single-member DEFLATE stream
/// only when every non-final chunk ends byte-aligned, so a decoder reads chunk
/// N's tail and then chunk N+1's header with no stray bits between them. Each
/// chunk's back-references may point into `dict` (the preceding window, seeded
/// into the matchfinder but not emitted), which the decoder already holds as
/// the tail of the output decoded so far.
pub fn encode_deflate_segment_to_sink(
    data: &[u8],
    dict: &[u8],
    level: u32,
    is_last: bool,
    out: &mut Vec<u8>,
    parallel: bool,
) {
    // Output write-through: emit the DEFLATE stream straight into the caller's
    // `out` (adopting it as the sink) instead of building a second Vec and
    // copying it over. `mem::take` moves the existing buffer in and `finish`
    // moves it back, so the caller's bytes are preserved as the prefix and no
    // output-sized buffer is duplicated.
    let mut bw = BitWriter::from_vec(std::mem::take(out));

    if dict.is_empty() {
        // No preset dictionary: build a padded working buffer [data | pad] so
        // the matchfinder's speculative loads stay in bounds. (Callers holding a
        // buffer that already carries the pad — the T1 hot path — should use
        // `encode_gzip_slack_padded_to_vec` / `encode_deflate_slack_padded_to_sink` to skip this copy.)
        let cap = data.len() + parse::BUF_PAD;
        crate::anatomy_count!(alloc_events);
        crate::anatomy_count!(alloc_bytes, cap);
        let mut buf = Vec::with_capacity(cap);
        buf.extend_from_slice(data);
        buf.resize(data.len() + parse::BUF_PAD, 0);
        deflate_into(
            &mut bw,
            &buf,
            0,
            data.len(),
            data.len(),
            level,
            is_last,
            true,
            parallel,
        );
    } else {
        // Preset-dictionary chunk: prepend the dictionary into one padded buffer
        // [dict | data | pad] and parse over the data region with the dictionary
        // seeded ahead of it (matches may point back into it).
        let dict_len = dict.len();
        let in_end = dict_len + data.len();
        let cap = in_end + parse::BUF_PAD;
        crate::anatomy_count!(alloc_events);
        crate::anatomy_count!(alloc_bytes, cap);
        let mut buf = Vec::with_capacity(cap);
        buf.extend_from_slice(dict);
        buf.extend_from_slice(data);
        buf.resize(in_end + parse::BUF_PAD, 0);
        deflate_into(
            &mut bw,
            &buf,
            dict_len,
            in_end,
            data.len(),
            level,
            is_last,
            true,
            parallel,
        );
    }

    *out = bw.finish();
}

/// Compress `data` into a raw DEFLATE FRAGMENT for the T>1 bit-splicing
/// writer, appending to `out` and returning the [`bitstream::ChunkMeta`] the
/// splicer needs.
///
/// Differences from [`encode_deflate_segment_to_sink`], which this replaces on
/// the T>1 chunk path:
///
/// * a non-final chunk gets NO sync-flush seam and NO byte-align padding —
///   the returned `pad_bits` tells the writer thread exactly where the
///   fragment's bitstream ends, and the writer bit-splices the next fragment
///   directly onto it. The ~5-byte-per-chunk framing floor disappears from
///   the output entirely;
/// * `needs_alignment` reports whether the fragment contains a stored
///   (BTYPE=00) block. Stored LEN/NLEN byte-alignment is relative to the
///   fragment's own start, so such a fragment must be placed byte-aligned:
///   the splicer re-creates the old-style seam at that one boundary instead
///   of shifting the fragment. Detection is [`STORED_BLOCK_EMITTED`], a
///   thread-local set by the two cold stored-block emitters
///   ([`write_stored_subblock`], which the parser's stored-escape and
///   `emit_stored_block` both route through, and ultra's
///   `add_non_compressed_block`) — NOT a field on [`BitWriter`]: the first
///   version put it there and eroded T1 wall broadly (see the layout note in
///   `bitstream.rs`). Emission always happens on the thread running this
///   function (ultra's scoped threads squeeze LZ77 stores and join BEFORE
///   `add_lz77_block` writes bits), so reset-encode-read here is race-free.
///
/// The final chunk (`is_last`) still sets BFINAL on its last block; its
/// trailing pad, if any, is reported like any other so the splicer can shift
/// it too, then zero-pad the whole stream once at the very end.
pub fn encode_deflate_splice_chunk_to_sink(
    data: &[u8],
    dict: &[u8],
    level: u32,
    is_last: bool,
    out: &mut Vec<u8>,
    parallel: bool,
    input_total_len: usize,
) -> bitstream::ChunkMeta {
    STORED_BLOCK_EMITTED.with(|f| f.set(false));
    let mut bw = BitWriter::from_vec(std::mem::take(out));

    if dict.is_empty() {
        let cap = data.len() + parse::BUF_PAD;
        crate::anatomy_count!(alloc_events);
        crate::anatomy_count!(alloc_bytes, cap);
        let mut buf = Vec::with_capacity(cap);
        buf.extend_from_slice(data);
        buf.resize(data.len() + parse::BUF_PAD, 0);
        deflate_into(
            &mut bw,
            &buf,
            0,
            data.len(),
            input_total_len,
            level,
            is_last,
            false,
            parallel,
        );
    } else {
        let dict_len = dict.len();
        let in_end = dict_len + data.len();
        let cap = in_end + parse::BUF_PAD;
        crate::anatomy_count!(alloc_events);
        crate::anatomy_count!(alloc_bytes, cap);
        let mut buf = Vec::with_capacity(cap);
        buf.extend_from_slice(dict);
        buf.extend_from_slice(data);
        buf.resize(in_end + parse::BUF_PAD, 0);
        deflate_into(
            &mut bw,
            &buf,
            dict_len,
            in_end,
            input_total_len,
            level,
            is_last,
            false,
            parallel,
        );
    }

    let needs_alignment = STORED_BLOCK_EMITTED.with(|f| f.get());
    let (bytes, pad_bits) = bw.finish_unaligned();
    *out = bytes;
    bitstream::ChunkMeta {
        pad_bits,
        needs_alignment,
    }
}

thread_local! {
    /// Set whenever a stored (BTYPE=00) block is emitted on this thread; the
    /// T>1 splice-chunk encoder resets it before encoding and reads it after,
    /// because a fragment containing a stored block byte-aligns LEN/NLEN
    /// relative to its own start and therefore cannot be bit-shifted by the
    /// writer-thread splicer.
    ///
    /// WHY A THREAD-LOCAL AND NOT A `BitWriter` FIELD (bisect receipt,
    /// 2026-08-03): the first substrate version carried this as
    /// `align_sensitive: bool` on `BitWriter` with a store in
    /// `align_to_byte()`. `BitWriter` is on the T1 hot path at every level,
    /// and that change ALONE (splicer never invoked) moved armexe.elf L1/T1
    /// from wall ratio 0.576 to 0.615 vs gzip on the frozen Zen2 box. The two
    /// set-sites here are per-STORED-BLOCK cold paths (a TLS store apiece),
    /// and the reset/read run once per T>1 chunk — nothing on the per-token
    /// emit path is touched.
    ///
    /// Correctness relies on emission being single-threaded per chunk: both
    /// set-sites ([`write_stored_subblock`] and ultra's
    /// `add_non_compressed_block`) run on the thread that called
    /// [`encode_deflate_splice_chunk_to_sink`] (ultra's scoped threads
    /// produce LZ77 stores and join before any bit is written).
    static STORED_BLOCK_EMITTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Cold set-site hook for [`STORED_BLOCK_EMITTED`]. `pub(crate)` so ultra's
/// direct stored-block emitter (`add_non_compressed_block`, which bypasses
/// [`emit_stored_block`]) can report too.
#[inline]
pub(crate) fn note_stored_block_emitted() {
    STORED_BLOCK_EMITTED.with(|f| f.set(true));
}

fn deflate_into(
    bw: &mut BitWriter,
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    input_total_len: usize,
    level: u32,
    is_last: bool,
    sync_flush: bool,
    parallel: bool,
) {
    debug_assert!(buf.len() >= in_end + parse::BUF_PAD);
    if in_end == data_start {
        emit_stored_block(bw, &[], is_last);
    } else if level == 0 {
        emit_stored_block(bw, &buf[data_start..in_end], is_last);
    } else {
        // T>1 spends its parallel wall slack on a stronger parse — see
        // `level::params_parallel`. T1 is untouched.
        //
        // RETRATED 2026-08-30 (pre-merge): `61f0f01d` switched L8-L9 T>1 to the
        // regular parser claiming "the size is unchanged" — the ledger proves
        // otherwise: the near-optimal per-chunk parse is what made T4 L8-9
        // SMALLER than T1 (canary: text +23,413 B / tabular +6,503 B /
        // binary +1,759 B back to the regular parser, flipping binary/tabular/
        // text L9 T4 from won to lost against gzip/pigz/libdeflate). A won cell
        // that regresses blocks the merge, so the stronger parse comes back.
        //
        // OPEN LEVER (named, with numbers): on the frozen box, near-optimal L9
        // T>1 measured ~2.5x slower than libdeflate in wall (61f0f01d) while
        // the regular parser measured 0.58x. Neither gets both axes on L9 T4 —
        // the lever is a parse BETWEEN them (e.g. near-optimal with a bounded
        // effort, or a deeper regular search) measured on the size AND wall
        // boards before it ships.
        let params = if parallel {
            level::params_parallel(level)
        } else {
            level::params(level)
        };
        let budget = if parallel {
            encode_types::HeaderBudget::Generous
        } else {
            encode_types::HeaderBudget::Lean
        };
        parse::compress(
            buf,
            data_start,
            in_end,
            input_total_len,
            &params,
            is_last,
            budget,
            bw,
        );
    }

    // Non-final chunk in a CONCATENATED stream: close on a clean byte boundary
    // with a sync-flush marker so the next chunk's stream joins without stray
    // bits. Skipped when one continuous `BitWriter` spans every chunk.
    if !is_last && sync_flush {
        emit_stored_block(bw, &[], false);
    }
}

/// Encode `buf[..logical_len]` (no preset dictionary) as a single final DEFLATE
/// block, appended to `out` with no intermediate output buffer.
///
/// `buf` MUST already carry at least [`INPLACE_TAIL_PAD`] trailing zero bytes
/// past `logical_len` (`buf.len() >= logical_len + INPLACE_TAIL_PAD`). This is
/// the copy-free entry point: the caller pads its own read buffer once, so the
/// input is parsed IN PLACE rather than copied into a second padded buffer, and
/// the output is written through into `out`. Output is byte-identical to
/// `encode_deflate_bytes_to_sink(&buf[..logical_len], &[], level, out)`.
pub fn encode_deflate_slack_padded_to_sink(
    buf: &[u8],
    logical_len: usize,
    level: u32,
    out: &mut Vec<u8>,
) {
    encode_census::legacy_encode();
    assert!(
        buf.len() >= logical_len + INPLACE_TAIL_PAD,
        "encode_deflate_slack_padded_to_sink: buf must carry INPLACE_TAIL_PAD trailing pad bytes"
    );
    let mut bw = BitWriter::from_vec(std::mem::take(out));
    deflate_into(
        &mut bw,
        buf,
        0,
        logical_len,
        logical_len,
        level,
        true,
        true,
        false,
    );
    *out = bw.finish();
}

/// Compress `data` into a gzip-framed stream (gzip header + DEFLATE + CRC32 +
/// ISIZE). This is the variant the roundtrip oracles consume.
///
/// ⚠ THIS IS A CONVENIENCE WRAPPER, NOT A SECOND ENGINE. It pads a copy of
/// `data` and delegates to [`encode_gzip_slack_padded_to_vec`], so its bytes
/// are IDENTICAL to the shipped CLI path at every level, by construction.
///
/// It used to be a second engine, and that was a silent defect. It carried its
/// own pick-min dispatch that handled only the mmap levels (1/2/4) and had no
/// zlib branch, so at L5-L7 the zlib pick-min never ran: by the time the shared
/// dispatcher was reached, the 10-byte gzip header made its `bw.byte_len() == 0`
/// guard false. Nothing announced this. Measured on 2026-08-21 before the fix:
/// dickens L5 4,582,861 here vs 4,544,452 shipped (+38,409 B); access.log L6
/// +43,787 B; never smaller.
///
/// That mattered because FIVE test suites call this function — `size_invariants`
/// (which walks L0-L9 and owns ladder monotonicity), `anatomy_pins`,
/// `perf_shape`, `startup_cost`, `anatomy_wall` — and `anatomy_pins` described
/// it as "the production T1 entry point", which it was not. Those suites agreed
/// with production only by the accident of their fixtures. Delegation makes the
/// agreement structural instead of lucky.
///
/// Cost, MEASURED not assumed: for LARGE inputs this adds one copy of the input
/// plus the pad. For SMALL ones it is cheaper than what it replaced — the
/// startup pins moved DOWN, alloc_events 3 -> 2 and alloc_bytes 308 -> 291 on a
/// 1-byte input at L6 and L9, because the old path allocated three times. Either
/// way this is a convenience API with no CLI caller: the CLI reaches
/// `encode_gzip_slack_padded_to_vec` directly.
pub fn encode_gzip_bytes_to_vec(data: &[u8], level: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(data.len() + INPLACE_TAIL_PAD);
    buf.extend_from_slice(data);
    buf.resize(data.len() + INPLACE_TAIL_PAD, 0);
    encode_gzip_slack_padded_to_vec(&buf, data.len(), level)
}

/// Gzip-framed compression that parses IN PLACE over a caller-padded buffer.
///
/// `buf[..logical_len]` is the input; `buf` MUST carry at least
/// [`INPLACE_TAIL_PAD`] trailing zero bytes past `logical_len`. This is the
/// allocation-lean T1 entry point: the caller reads the input once (e.g. via
/// `read_to_end`) and pads that same buffer (`resize(len + INPLACE_TAIL_PAD,
/// 0)`), so the compressor neither copies the input into a second work buffer
/// nor builds a separate output buffer. Output is byte-identical to
/// `encode_gzip_bytes_to_vec(&buf[..logical_len], level)`.
/// Levels whose production encoder is our own libdeflate port (`compress::ldx`).
///
/// THE PORT IS THE PRODUCT. `ldx` is a per-decision transliteration of
/// `vendor/libdeflate/lib/deflate_compress.c`; it lived in the tree as a test
/// oracle while the shipped encoder grew a second parse per level to defend
/// size cells. Measured 2026-08-22, in-process, same build, no I/O:
///
/// ```text
///     ours / ldx        L0     L1     L2     L4     L6     L8     L9
///     wall (dickens)   4.08x  2.89x  1.64x  1.80x  1.72x  1.00x  1.00x
///     size              1.000  0.979  0.999  0.981  0.994  1.000  1.000
/// ```
///
/// L8/L9 are already the port (1.000x size, 1.00x wall) — that is the control
/// that confirms the reading. L0-L7 pay 1.6-4.1x for 0.1-2.3% of size.
///
/// End to end through the CLI, levels 1-7 routed here, `-p1`:
///
/// ```text
///     wall vs libdeflate     before        after
///     dickens      L1         3.25x        1.20x   (2.71x faster)
///     data.parquet L1         3.85x        1.20x   (3.21x faster)
///     dickens      L6         2.08x        1.23x
///     size vs libdeflate     0.972-0.999x  1.000x  (exact parity)
/// ```
///
/// Owner priority: wall outranks size, and tying on size is acceptable. This
/// ties EXACTLY and takes the wall.
///
/// 10-12 keep our own engine — the exotic ladder has no libdeflate counterpart
/// and `LdxCompressor::new` returns `None` above 9.
/// ONE ENCODE PER INPUT — the architectural invariant, COUNTED not asserted.
///
/// ⭐ OWNER, 2026-08-23: "Why do we even have pick-min? Isn't that the approach that
/// drove to parallel implementations which caused us to lose so much wall clock time?
/// I told you to start with the perfect port of the vendor we're competing against and
/// then to make optimizations that you could surpass in all cases. ... This project is
/// named after its speed. Compression can't get worse, but that is strictly secondary."
///
/// Whole-buffer pick-min encoded every input TWICE (THREE times at L1) and kept the
/// smaller result. It cost ~2x CLI wall at every level to defend 0.002-1.95% of size.
/// It is deleted; `tests/one_encode_only.rs` keeps it deleted.
///
/// NOT feature-gated on purpose: one relaxed atomic per whole-file encode is free (it
/// is per CALL, not per byte), so the guard runs in the DEFAULT test suite. A guard
/// that only runs under a feature flag is one that gets missed.
pub mod encode_census {
    use core::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// One per entry to the libdeflate port (the production encoder).
    pub static PORT_ENCODES: AtomicU64 = AtomicU64::new(0);
    /// One per entry to the legacy whole-buffer encoder.
    pub static LEGACY_ENCODES: AtomicU64 = AtomicU64::new(0);

    #[inline]
    pub(crate) fn port_encode() {
        PORT_ENCODES.fetch_add(1, Relaxed);
    }
    #[inline]
    pub(crate) fn legacy_encode() {
        LEGACY_ENCODES.fetch_add(1, Relaxed);
    }

    /// `(port, legacy)` since the last [`reset`].
    ///
    /// Read by `tests/one_encode_only.rs`. The bin target cannot see integration
    /// tests, so `-D dead-code` flags these there; they are the observability surface
    /// the guard is built on, not dead code.
    #[allow(dead_code)]
    pub fn snapshot() -> (u64, u64) {
        (PORT_ENCODES.load(Relaxed), LEGACY_ENCODES.load(Relaxed))
    }
    #[allow(dead_code)]
    pub fn reset() {
        PORT_ENCODES.store(0, Relaxed);
        LEGACY_ENCODES.store(0, Relaxed);
    }
}

#[inline]
pub(crate) fn level_uses_ldx(level: u32) -> bool {
    // ⭐ THE PORT IS THE BASELINE (owner, 2026-08-23) — with THREE measured
    // exceptions. Routing L1/L6/L7 to the port was tried on this branch
    // (`b28e96f3`) and the per-commit ledger gate went red immediately and
    // stayed red for 45 commits: `won_cells_stay_won` regresses FOUR cells
    // (binary:L6 vs gzip +1,614 B / vs pigz +887 B; text:L6 vs gzip +12,610 B
    // / vs pigz +12,090 B) and `fast_l1_ratio_multi_corpus` loses the L1 text
    // cell to pigz (43,980 vs 42,384 = 1.038x). A won cell that regresses is
    // a regression on a closed cell — the ledger is append-only and is never
    // edited to fit a result, so the routing comes back.
    //
    // Each exception is a MEASUREMENT with a named gate, not a preference —
    // and all three collapse the moment the port learns ONE knob, `good_match`
    // (shorten the chain walk once a match >= good_match is found; zlib/gzip/
    // pigz all use it, and libdeflate does not implement it). That is the
    // named follow-up lever (port `good_match` INTO ldx, then re-measure on
    // the frozen board and retire the exceptions one level at a time):
    //
    //   L1  our L1 is igzip-derived and BEATS pigz -1 on text where the port does not
    //       (43,980 vs 42,384 = 1.038x pigz). Gate: `fast_l1_ratio_multi_corpus`. #347.
    //
    //   L6  `won_cells_stay_won` (append-only) regresses FOUR cells if L6 routes here:
    //         binary:L6 vs gzip +1,614 B / vs pigz +887 B
    //         text:L6   vs gzip +12,610 B / vs pigz +12,090 B
    //       Those cells were won by the ZLIB arm = baseline + `good_match` 8 + chain
    //       128. Carrying those knobs on the level keeps the cells with ONE encode.
    //
    //   L7  follows from L6: the port has no `good_match`, so our L6 is stronger than
    //       the port's L7 (100, 130) and `ladder_is_monotone_t1` fires (305,775 >
    //       304,252 on text). L7 keeps the legacy encoder at its own measured-best
    //       single config (chain 256, `good_match` 32) — monotone, and 2 clause-3 flips
    //       against 4 for `params(7)`.
    //
    // Enforced by `tests/one_encode_only.rs`, which COUNTS encoder entries: a predicate
    // has lied about exactly this three times in this campaign.
    !matches!(level, 1 | 6 | 7) && level <= 9
}

pub fn encode_gzip_slack_padded_to_vec(buf: &[u8], logical_len: usize, level: u32) -> Vec<u8> {
    crate::anatomy_wall_root!({
        let cap = estimate_output_cap(logical_len, level, 32);
        crate::anatomy_count!(alloc_events);
        crate::anatomy_count!(alloc_bytes, cap);
        let mut out = Vec::with_capacity(cap);
        out.extend_from_slice(&minimal_gzip_header(level));

        // ONE production encoder for 0-9: our libdeflate port.
        // `compress_for_diff` emits RAW DEFLATE, which is exactly what belongs
        // between the header written above and the CRC/ISIZE written below.
        // Append straight into `out` — no scratch buffer, no zeroing, no copy.
        if !(level_uses_ldx(level)
            && crate::compress::ldx::compress_into(level, &buf[..logical_len], &mut out))
        {
            encode_deflate_slack_padded_to_sink(buf, logical_len, level, &mut out);
        }

        let crc =
            crate::anatomy_wall_time!(crc_ns, crc_calls, { crc32fast::hash(&buf[..logical_len]) });
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(logical_len as u32).to_le_bytes());
        out
    })
}

/// Compress `reader` into `writer` as a gzip stream at `level`.
///
/// ⚠ THIS BUFFERS THE WHOLE INPUT. `ldx` (the production T1 parser for L0-9) is
/// whole-buffer by construction, and L10-12 has no resumable parser, so every
/// level reads the reader to end before emitting a byte. The single-pass
/// streaming implementation that this entry point used to carry was deleted
/// 2026-08-30: it was unreachable for every production level, and keeping it
/// alive let the "genuinely streaming" doc claim survive for two weeks after
/// the routing that could have used it was gone. A true streaming T1 API
/// needs a resumable `ldx` port — that is the named open work item (see
/// `src/lib.rs`), and until it lands this function is the honest whole-buffer
/// single-threaded route.
#[allow(dead_code)] // library API entry point; no in-crate caller (the binary
                    // routes through `_sized`). Kept public per the module's
                    // dead-code policy: narrow allow + doc note.
pub fn encode_gzip_reader_to_writer<R: std::io::Read, W: std::io::Write>(
    reader: &mut R,
    writer: &mut W,
    level: u32,
) -> std::io::Result<u64> {
    encode_gzip_reader_to_writer_sized(reader, writer, level, None)
}

/// [`encode_gzip_reader_to_writer`] with an optional input-size hint.
///
/// The hint sizes the read buffer ONCE up front instead of letting
/// `read_to_end` grow it by doubling — an 8 MiB input otherwise touches ~21 MB
/// of fresh anonymous pages (every page a minor fault).
pub fn encode_gzip_reader_to_writer_sized<R: std::io::Read, W: std::io::Write>(
    reader: &mut R,
    writer: &mut W,
    level: u32,
    size_hint: Option<usize>,
) -> std::io::Result<u64> {
    let mut input = Vec::with_capacity(size_hint.map_or(0, |h| h + INPLACE_TAIL_PAD));
    reader.read_to_end(&mut input)?;
    let logical_len = input.len();
    input.resize(logical_len + INPLACE_TAIL_PAD, 0);
    let gz = encode_gzip_slack_padded_to_vec(&input, logical_len, level);
    writer.write_all(&gz)?;
    Ok(logical_len as u64)
}

/// Compress `data[..logical_len]` as one gzip stream, written to `writer`.
///
/// `data` MAY carry trailing slack past `logical_len` — if it carries at least
/// [`INPLACE_TAIL_PAD`] readable zero bytes there, the parse runs IN PLACE with
/// no copy of the input. The mmap route arranges exactly that (see
/// `io.rs::map_with_tail_pad`). Callers with no slack pass
/// `logical_len == data.len()` and get a single copy instead.
pub fn encode_gzip_unpadded_slice_to_writer<W: std::io::Write>(
    data: &[u8],
    logical_len: usize,
    writer: &mut W,
    level: u32,
) -> std::io::Result<u64> {
    debug_assert!(logical_len <= data.len());

    // L0 FAST PATH: write stored blocks directly to the writer, no intermediate Vec.
    // L0 is "stored" (no compression) — just copy the data into gzip blocks.
    // This avoids the extra copy of the entire input into an output Vec.
    if level == 0 {
        writer.write_all(&minimal_gzip_header(0))?;
        let crc = crc32fast::hash(&data[..logical_len]);
        if logical_len == 0 {
            // Empty input: must emit one empty stored block for valid DEFLATE.
            // BFINAL=1, BTYPE=0, LEN=0, NLEN=0xFFFF (matches deflate_compress_none).
            writer.write_all(&[0x01, 0x00, 0x00, 0xFF, 0xFF])?;
        } else {
            let mut in_next: usize = 0;
            while in_next < logical_len {
                let len = core::cmp::min(logical_len - in_next, u16::MAX as usize);
                let bfinal: u8 = if in_next + len == logical_len { 1 } else { 0 };
                writer.write_all(&[bfinal])?;
                writer.write_all(&(len as u16).to_le_bytes())?;
                writer.write_all(&((len as u16 ^ 0xFFFFu16).to_le_bytes()))?;
                writer.write_all(&data[in_next..in_next + len])?;
                in_next += len;
            }
        }
        writer.write_all(&crc.to_le_bytes())?;
        writer.write_all(&(logical_len as u32).to_le_bytes())?;
        return Ok(logical_len as u64);
    }

    // FAST PATH: the caller already gave us the pad, so parse IN PLACE.
    //
    // `data` is usually an mmap of the whole input. Copying it just to
    // append INPLACE_TAIL_PAD zero bytes costs a full memcpy of the file —
    // 51 MB on monorepo.tar to add 16 bytes. Measured 2026-08-21: our
    // explicit allocations ran at EXACTLY 1.50x the input on every corpus
    // file regardless of compressibility (1.0x this copy + 0.5x the output
    // reservation), and peak RSS at 2.5-2.7x the input.
    //
    // When the mapping already carries >= INPLACE_TAIL_PAD readable bytes
    // past `logical_len` — which `map_with_tail_pad` arranges by mapping
    // into the final partial page, where the kernel zero-fills past EOF —
    // those bytes ARE the pad the padded encoder requires, and the copy is
    // pure waste. `debug_assert` the zero-fill rather than trust it.
    if data.len() >= logical_len + INPLACE_TAIL_PAD {
        debug_assert!(
            data[logical_len..logical_len + INPLACE_TAIL_PAD]
                .iter()
                .all(|&b| b == 0),
            "slack bytes past logical_len must read as zero"
        );
        let gz = encode_gzip_slack_padded_to_vec(data, logical_len, level);
        writer.write_all(&gz)?;
        return Ok(logical_len as u64);
    }

    // SLOW PATH: no slack available (e.g. the input ends exactly on a page
    // boundary, so mapping further would fault). Copy once.
    let cap = logical_len + INPLACE_TAIL_PAD;
    crate::anatomy_count!(alloc_events);
    crate::anatomy_count!(alloc_bytes, cap);
    let mut input = Vec::with_capacity(cap);
    input.extend_from_slice(data);
    input.resize(cap, 0);
    let gz = encode_gzip_slack_padded_to_vec(&input, logical_len, level);
    writer.write_all(&gz)?;
    Ok(logical_len as u64)
}

/// Emit one or more stored (uncompressed, BTYPE=00) blocks covering `data`.
///
/// Port of the uncompressed-block emission in `deflate_flush_block` (~:1826).
/// A stored sub-block carries at most 65535 bytes, so long inputs use several;
/// `is_final` marks the last sub-block BFINAL.
///
/// `pub(crate)` (Stage E, docs/compressor-architecture.md §5-E): also the
/// single source of stored-block FRAMING for `compress::deflate64`'s
/// empty-input special case (BFINAL=1/BTYPE=00/LEN=0/NLEN=0xFFFF) — the
/// wire format is format-law, not tier-specific, so it dedupes across both
/// encoders exactly like the gzip wrapper.
pub(crate) fn emit_stored_block(bw: &mut BitWriter, data: &[u8], is_final: bool) {
    if data.is_empty() {
        write_stored_subblock(bw, &[], is_final);
        return;
    }
    let mut off = 0usize;
    while off < data.len() {
        let end = (off + MAX_STORED_SUBBLOCK).min(data.len());
        let last = end == data.len();
        write_stored_subblock(bw, &data[off..end], is_final && last);
        off = end;
    }
}

fn write_stored_subblock(bw: &mut BitWriter, sub: &[u8], bfinal: bool) {
    // The ONE physical-block emission site for every BTYPE=00 (stored) block
    // gzippy ever writes — not just the parser's cost-comparison "stored
    // wins" branch (`parse::emit_block`/`emit_block_static_or_stored`) but
    // ALSO the T>1 pipelined path's per-chunk sync-flush marker and the
    // empty-input special case (both call `emit_stored_block` directly from
    // `deflate_into`, above, bypassing the parser entirely). A first cut of
    // this counter lived in `parse/mod.rs`'s two `emit_block*` functions
    // instead and UNDERCOUNTED: a closed-loop check against fulcrum's
    // token-level block count (which counts every physical BTYPE=00 block
    // regardless of why it was emitted) found 15 stored blocks at the token
    // level vs 0 here on a T>1 pipelined run of `dd79_text6` L1 — exactly
    // the sync-flush markers between pipeline chunks. Counting HERE instead
    // (removed from the two `emit_block*` call sites, which route through
    // `emit_stored_block` -> here, so counting there too would double-count)
    // reconciles exactly against the token-level count on every path.
    crate::anatomy_count!(blocks_emitted_stored);
    // Same single-site property makes this the right hook for the T>1
    // splicer's stored-block tripwire (see `STORED_BLOCK_EMITTED`): one cold
    // TLS store per physical stored block, nothing on the per-token path.
    // (Ultra's `add_non_compressed_block` is the one emitter that bypasses
    // this function; it calls `note_stored_block_emitted` itself.)
    note_stored_block_emitted();
    debug_assert!(sub.len() <= MAX_STORED_SUBBLOCK);
    bw.add_bits(bfinal as u64, 1);
    bw.add_bits(DEFLATE_BLOCKTYPE_UNCOMPRESSED as u64, 2);
    bw.align_to_byte();
    let len = sub.len() as u16;
    bw.write_u16_le(len);
    bw.write_u16_le(!len);
    bw.write_aligned_bytes(sub);
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::process::{Command, Stdio};

    /// Deterministic mixed text+binary corpus of `len` bytes.
    fn mixed_corpus(len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let phrases: [&[u8]; 4] = [
            b"the quick brown fox jumps over the lazy dog; ",
            b"DEFLATE back-references span chunk boundaries. ",
            b"lorem ipsum dolor sit amet consectetur adipiscing; ",
            b"0123456789abcdef repeated structure repeated structure ",
        ];
        let mut i = 0usize;
        while v.len() < len {
            v.extend_from_slice(phrases[i % phrases.len()]);
            // Sprinkle pseudo-random binary bytes so blocks aren't trivially RLE.
            let x = (i.wrapping_mul(2654435761)) as u32;
            v.extend_from_slice(&x.to_le_bytes());
            i += 1;
        }
        v.truncate(len);
        v
    }

    /// Wrap a raw DEFLATE stream in minimal gzip framing over `original`.
    fn wrap_gzip(deflate: &[u8], original: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(deflate.len() + 18);
        out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff]);
        out.extend_from_slice(deflate);
        out.extend_from_slice(&crc32fast::hash(original).to_le_bytes());
        out.extend_from_slice(&(original.len() as u32).to_le_bytes());
        out
    }

    fn decode_flate2(gz: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(gz)
            .read_to_end(&mut out)
            .expect("flate2 failed to decode concatenated stream");
        out
    }

    /// `gzip -dc` decode; `None` only when no `gzip` binary is on PATH.
    fn decode_system_gzip(gz: &[u8]) -> Option<Vec<u8>> {
        let mut child = Command::new("gzip")
            .arg("-dc")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let mut stdin = child.stdin.take().unwrap();
        let buf = gz.to_vec();
        let writer = std::thread::spawn(move || {
            let _ = stdin.write_all(&buf);
        });
        let mut out = Vec::new();
        child.stdout.take().unwrap().read_to_end(&mut out).unwrap();
        writer.join().unwrap();
        assert!(child.wait().unwrap().success(), "gzip -dc exited non-zero");
        Some(out)
    }

    /// Three independently-compressed chunks (chunk k seeds the previous chunk's
    /// 32 KiB tail as its dictionary; only the last is `is_last`) concatenate
    /// into ONE valid single-member gzip stream that decodes byte-exact.
    #[test]
    fn three_chunks_concatenate_and_roundtrip() {
        let input = mixed_corpus(300_003);
        let n = input.len();
        let bounds = [0usize, n / 3, 2 * n / 3, n];

        for level in [1u32, 6, 9, 12] {
            let mut deflate = Vec::new();
            for c in 0..3 {
                let start = bounds[c];
                let end = bounds[c + 1];
                let dict_start = start.saturating_sub(32 * 1024);
                let dict = &input[dict_start..start];
                let is_last = c == 2;
                encode_deflate_segment_to_sink(
                    &input[start..end],
                    dict,
                    level,
                    is_last,
                    &mut deflate,
                    false,
                );
            }
            let gz = wrap_gzip(&deflate, &input);

            assert_eq!(
                decode_flate2(&gz),
                input,
                "flate2 roundtrip mismatch at L{level}"
            );
            if let Some(sys) = decode_system_gzip(&gz) {
                assert_eq!(sys, input, "gzip -dc roundtrip mismatch at L{level}");
            }
        }
    }

    /// `encode_deflate_segment_to_sink(data, &[], level, true, ..)` must be
    /// byte-identical to the single-block [`encode_deflate_bytes_to_sink`] (no sync marker,
    /// BFINAL set) — the regression guard the brief requires.
    #[test]
    fn is_last_no_dict_equals_compress_block() {
        let cases: [Vec<u8>; 3] = [Vec::new(), b"tiny".to_vec(), mixed_corpus(200_000)];
        for data in &cases {
            for level in [0u32, 1, 2, 6, 9, 12] {
                let mut streaming = Vec::new();
                encode_deflate_segment_to_sink(data, &[], level, true, &mut streaming, false);
                let mut block = Vec::new();
                encode_deflate_bytes_to_sink(data, &[], level, &mut block);
                assert_eq!(
                    streaming,
                    block,
                    "streaming(is_last=true) diverged from encode_deflate_bytes_to_sink at L{level}, len={}",
                    data.len()
                );
            }
        }
    }
}

#[cfg(test)]
mod inplace_tests {
    use super::*;

    /// Pad `data` into a fresh buffer the way the T1 hot path pads its read
    /// buffer in place, then return `(padded, logical_len)`.
    fn padded(data: &[u8]) -> (Vec<u8>, usize) {
        let mut buf = data.to_vec();
        buf.resize(data.len() + INPLACE_TAIL_PAD, 0);
        (buf, data.len())
    }

    /// The copy-free in-place gzip path must be byte-identical to the reference
    /// `encode_gzip_bytes_to_vec` (which builds a separate padded work buffer).
    fn assert_padded_gzip_matches(data: &[u8], level: u32) {
        let reference = encode_gzip_bytes_to_vec(data, level);
        let (buf, logical_len) = padded(data);
        let inplace = encode_gzip_slack_padded_to_vec(&buf, logical_len, level);
        assert_eq!(
            reference,
            inplace,
            "encode_gzip_slack_padded_to_vec diverged at L{level}, len={}",
            data.len()
        );
    }

    /// The raw-DEFLATE in-place path must match `encode_deflate_bytes_to_sink` (append form).
    fn assert_padded_block_matches(data: &[u8], level: u32) {
        let mut reference = Vec::new();
        encode_deflate_bytes_to_sink(data, &[], level, &mut reference);
        let (buf, logical_len) = padded(data);
        let mut inplace = Vec::new();
        encode_deflate_slack_padded_to_sink(&buf, logical_len, level, &mut inplace);
        assert_eq!(
            reference,
            inplace,
            "encode_deflate_slack_padded_to_sink diverged at L{level}, len={}",
            data.len()
        );
    }

    #[test]
    fn inplace_matches_reference_edge_sizes() {
        // Tiny inputs (< BUF_PAD), inputs exactly at the pad boundary, and a few
        // multiples — the sizes where a speculative tail load is most likely to
        // read into the pad region.
        let motif = b"the quick brown fox 0123456789 ";
        for &len in &[
            0usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 255, 256, 257, 511, 512,
            4096, 4097,
        ] {
            let data: Vec<u8> = motif.iter().cloned().cycle().take(len).collect();
            for level in [0u32, 1, 2, 6, 8, 9, 12] {
                assert_padded_gzip_matches(&data, level);
                assert_padded_block_matches(&data, level);
            }
        }
        // Incompressible tail sizes too (chain misses, no long matches).
        for &len in &[13usize, 16, 19, 258, 259, 300] {
            let data: Vec<u8> = (0..len as u32)
                .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
                .collect();
            for level in [1u32, 6, 9, 12] {
                assert_padded_gzip_matches(&data, level);
            }
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(192))]

        /// On ANY input — including empty, sub-BUF_PAD, and boundary-straddling
        /// lengths — the in-place path is byte-identical to the copy-based
        /// reference at every strategy class.
        #[test]
        fn inplace_byte_identical_proptest(data in gen_data()) {
            for level in [0u32, 1, 6, 9, 12] {
                assert_padded_gzip_matches(&data, level);
                assert_padded_block_matches(&data, level);
            }
        }
    }

    /// Adversarial generator biased toward the small / boundary lengths that
    /// exercise the near-EOF speculative loads, plus runs and repeats.
    fn gen_data() -> impl proptest::strategy::Strategy<Value = Vec<u8>> {
        use proptest::prelude::*;
        prop_oneof![
            // Short random (straddles the max_len<5 gate and BUF_PAD).
            proptest::collection::vec(any::<u8>(), 0..40),
            // Runs (deep chains / long matches near EOF).
            (any::<u8>(), 0usize..300).prop_map(|(b, n)| vec![b; n]),
            // Repeated motif of a boundary-ish length.
            (proptest::collection::vec(any::<u8>(), 1..20), 0usize..40).prop_map(|(seed, reps)| {
                seed.iter()
                    .cloned()
                    .cycle()
                    .take(seed.len() * reps)
                    .collect()
            }),
            // Larger mixed buffer.
            proptest::collection::vec(any::<u8>(), 0..2048),
        ]
    }
}

#[cfg(test)]
mod unpadded_slice_tests {
    use super::*;

    /// Deterministic bytes with tunable redundancy (the standard seam-test
    /// generator family): small `period` = compressible, huge `period` =
    /// literal-dominated.
    fn corpus(len: usize, period: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let mut s: u32 = 0x1234_5678;
        for i in 0..len {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let r = (s >> 16) as u8;
            v.push(if (i as u32) % period < period / 2 {
                b'a' + (r % 26)
            } else {
                r
            });
        }
        v
    }

    fn unpadded(data: &[u8], level: u32) -> Vec<u8> {
        let mut out = Vec::new();
        let n = encode_gzip_unpadded_slice_to_writer(data, data.len(), &mut out, level)
            .expect("unpadded-slice encode");
        assert_eq!(n, data.len() as u64, "reported input length");
        out
    }

    /// The mmap-route encoder must be BYTE-IDENTICAL to the whole-buffer
    /// encoder at every level, on lengths chosen to hit each internal seam:
    /// empty; below/at/above INPLACE_TAIL_PAD (in-place fast path vs. the
    /// copy-once fallback); small (single block); and two multi-hundred-KB
    /// sizes that run many blocks and several stored sub-blocks at L0.
    /// L10-12 cover the copy fallback arm.
    #[test]
    fn unpadded_slice_is_byte_identical_to_whole_buffer() {
        let sizes = [
            0usize,
            1,
            INPLACE_TAIL_PAD - 1,
            INPLACE_TAIL_PAD,
            INPLACE_TAIL_PAD + 1,
            4096,
            375_000,
            700_000,
        ];
        for &len in &sizes {
            for period in [8u32, 96, 1 << 30] {
                let data = corpus(len, period);
                for level in [0u32, 1, 2, 3, 6, 9] {
                    assert_eq!(
                        unpadded(&data, level),
                        encode_gzip_bytes_to_vec(&data, level),
                        "L{level} len={len} period={period}: unpadded-slice \
                         output diverged from whole-buffer"
                    );
                }
                // Fallback arm (no resumable parser): keep it to one modest
                // size per period — the mechanism is a copy, not a parse.
                if len == 4096 || len == 700_000 {
                    for level in [10u32, 12] {
                        assert_eq!(
                            unpadded(&data, level),
                            encode_gzip_bytes_to_vec(&data, level),
                            "L{level} len={len} period={period}: fallback arm diverged"
                        );
                    }
                }
            }
        }
    }

    /// Multi-megabyte inputs: several fast/lazy blocks plus (at L0) several
    /// stored sub-blocks — big enough to exercise the in-place slack fast
    /// path and the whole-buffer copy path on the same input.
    #[test]
    fn unpadded_slice_matches_whole_buffer_on_multi_megabyte_inputs() {
        // ~4.3 MiB: several blocks at every level, comfortably multi-MB.
        let data = corpus(4_194_240 + 100_003, 64);
        for level in [0u32, 1, 6] {
            assert_eq!(
                unpadded(&data, level),
                encode_gzip_bytes_to_vec(&data, level),
                "L{level} multi-MB: unpadded-slice output diverged"
            );
        }
    }
}

#[cfg(test)]
mod dict_tests {
    use super::*;

    /// An empty preset dictionary must yield byte-identical output to the
    /// no-dictionary path (regression guard on the seeding wiring).
    #[test]
    fn empty_dict_equals_no_dict() {
        let data: Vec<u8> = b"the pure-rust deflate encoder must roundtrip. ".repeat(400);
        for level in [2u32, 6, 9] {
            let mut with_empty = Vec::new();
            encode_deflate_bytes_to_sink(&data, &[], level, &mut with_empty);
            let no_dict = encode_deflate_bytes_to_vec(&data, level);
            assert_eq!(with_empty, no_dict, "empty dict diverged at L{level}");
        }
    }

    /// A dictionary whose bytes appear in the data must let the parser reference
    /// it, producing a strictly smaller stream than compressing without it.
    /// This exercises the `skip_bytes` dictionary-seeding path (matches point
    /// back into `buf[..data_start]`).
    #[test]
    fn matching_dict_shrinks_output() {
        // Data begins with content that only exists in the dictionary, so the
        // opening bytes can only be coded as matches into the seeded window.
        let dict: Vec<u8> =
            b"PRESET-DICTIONARY-CONTENT-abcdefghijklmnopqrstuvwxyz-0123456789-".repeat(30);
        let data: Vec<u8> = {
            let mut d = dict.clone(); // fully present in the dictionary window
            d.extend_from_slice(b" and then some novel trailing text to code as literals.");
            d
        };
        for level in [4u32, 6, 9] {
            let with_dict = {
                let mut v = Vec::new();
                encode_deflate_bytes_to_sink(&data, &dict, level, &mut v);
                v.len()
            };
            let without = encode_deflate_bytes_to_vec(&data, level).len();
            assert!(
                with_dict < without,
                "L{level}: dict-seeded {with_dict} not smaller than no-dict {without}",
            );
        }
    }
}
