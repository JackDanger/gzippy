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
pub mod block_split;
pub mod costs;
pub mod huffman;
pub mod level;
pub mod matchfinder;
pub mod parse;
pub mod tables;

use bitstream::BitWriter;
use tables::DEFLATE_BLOCKTYPE_UNCOMPRESSED;

/// Largest payload of a single stored (BTYPE=00) sub-block.
const MAX_STORED_SUBBLOCK: usize = 65535;

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
pub fn compress_oneshot(data: &[u8], level: u32) -> Vec<u8> {
    let cap = estimate_output_cap(data.len(), level, 64);
    crate::anatomy_count!(alloc_events);
    crate::anatomy_count!(alloc_bytes, cap);
    let mut out = Vec::with_capacity(cap);
    compress_block(data, &[], level, &mut out);
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
pub fn compress_block(data: &[u8], dict: &[u8], level: u32, out: &mut Vec<u8>) {
    // A standalone single final block: BFINAL is set on the last internal
    // block and no sync-flush marker is appended. `bw.finish()` byte-aligns
    // the tail. This is the T1 / single-member framing.
    compress_block_streaming(data, dict, level, true, out);
}

/// Compress `data` into a raw DEFLATE stream for use as ONE CHUNK of a larger
/// concatenated single-member stream, appending to `out`.
///
/// Identical to [`compress_block`] except for the stream-position semantics
/// controlled by `is_last`:
///
/// * `is_last == true` — this chunk closes the stream. The last internal block
///   carries `BFINAL=1` and NOTHING is appended after it; `bw.finish()`
///   byte-aligns the tail. With an empty `dict` this is byte-identical to
///   [`compress_block`] (the single-member case).
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
pub fn compress_block_streaming(
    data: &[u8],
    dict: &[u8],
    level: u32,
    is_last: bool,
    out: &mut Vec<u8>,
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
        // `compress_gzip_padded` / `deflate_padded_in_place` to skip this copy.)
        let cap = data.len() + parse::BUF_PAD;
        crate::anatomy_count!(alloc_events);
        crate::anatomy_count!(alloc_bytes, cap);
        let mut buf = Vec::with_capacity(cap);
        buf.extend_from_slice(data);
        buf.resize(data.len() + parse::BUF_PAD, 0);
        deflate_into(&mut bw, &buf, 0, data.len(), level, is_last, true);
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
        deflate_into(&mut bw, &buf, dict_len, in_end, level, is_last, true);
    }

    *out = bw.finish();
}

/// Shared parse core: encode `buf[data_start..in_end]` into `bw`, treating
/// `buf[..data_start]` as a seeded (but un-emitted) preset-dictionary window.
///
/// `buf` MUST carry at least [`parse::BUF_PAD`] trailing bytes past `in_end`
/// that read as ZERO (the matchfinder's speculative loads reach up to `in_end +
/// 1`, and the emitted bytes are byte-identical only when those pad bytes are
/// zero — matches are clamped to `in_end` so the pad never enters the output).
///
/// `level == 0` is a genuine STORED-only mode (BTYPE=00 every block, no
/// matchfinder/Huffman coding at all) — the same contract zlib/pigz/gzip give
/// `Z_NO_COMPRESSION`/`-0` (gzip(1) and libdeflate-gzip reject `-0` outright;
/// pigz treats it as stored). This bypasses `level::params`/`parse::compress`
/// entirely rather than routing through `Strategy::Fast0` (which does real
/// LZ77 + per-block static-or-stored Huffman coding — see `level.rs`'s
/// `Strategy::Fast0` doc comment): a real compressor is not "as fast as a
/// memcpy, at worst as small as stored" the way peers' L0 is, so it can never
/// win the "at-least-as-fast, at-least-as-small" contract L0 is measured
/// against. `Strategy::Fast0`/`fast::run::<true>` stay in the tree (still
/// unit-tested via `level::params(0)`) but are no longer reachable from this
/// production call path. Levels 1-12 are completely unaffected — this is the
/// only new branch, guarded on `level == 0` alone.
///
/// `sync_flush` controls whether a non-final chunk is closed with a
/// byte-aligning empty stored block. It is REQUIRED when chunks are encoded
/// into separate bit streams that are later concatenated (the T>1 path, and
/// [`compress_block_streaming`]'s contract). It must be OFF for a
/// single-threaded streaming encoder that keeps ONE continuous `BitWriter`
/// across chunks: there the chunk seam does not exist in the bitstream at all,
/// so aligning at it would both waste ~5 bytes per chunk and, more
/// importantly, make the output depend on the chunk size — destroying
/// byte-identity with the whole-buffer encoder.
fn deflate_into(
    bw: &mut BitWriter,
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    level: u32,
    is_last: bool,
    sync_flush: bool,
) {
    debug_assert!(buf.len() >= in_end + parse::BUF_PAD);
    if in_end == data_start {
        emit_stored_block(bw, &[], is_last);
    } else if level == 0 {
        emit_stored_block(bw, &buf[data_start..in_end], is_last);
    } else {
        let params = level::params(level);
        parse::compress(buf, data_start, in_end, &params, is_last, bw);
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
/// `compress_block(&buf[..logical_len], &[], level, out)`.
pub fn deflate_padded_in_place(buf: &[u8], logical_len: usize, level: u32, out: &mut Vec<u8>) {
    assert!(
        buf.len() >= logical_len + INPLACE_TAIL_PAD,
        "deflate_padded_in_place: buf must carry INPLACE_TAIL_PAD trailing pad bytes"
    );
    let mut bw = BitWriter::from_vec(std::mem::take(out));
    deflate_into(&mut bw, buf, 0, logical_len, level, true, true);
    *out = bw.finish();
}

/// Compress `data` into a gzip-framed stream (gzip header + DEFLATE + CRC32 +
/// ISIZE). This is the variant the roundtrip oracles consume.
///
/// Wrapped in [`crate::anatomy_wall_root!`] (the `anatomy-wall` feature's
/// root span, see `anatomy_wall` module docs): this is one of the two
/// production T1 entry points (the other being [`compress_gzip_padded`]) the
/// wall-clock phase timers measure against.
pub fn compress_gzip(data: &[u8], level: u32) -> Vec<u8> {
    crate::anatomy_wall_root!({
        let cap = estimate_output_cap(data.len(), level, 32);
        crate::anatomy_count!(alloc_events);
        crate::anatomy_count!(alloc_bytes, cap);
        let mut out = Vec::with_capacity(cap);
        // Minimal gzip header: magic, CM=8 (deflate), FLG=0, MTIME=0, XFL=0,
        // OS=255 (unknown).
        out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff]);

        compress_block(data, &[], level, &mut out);

        let crc = crate::anatomy_wall_time!(crc_ns, crc_calls, { crc32fast::hash(data) });
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out
    })
}

/// Gzip-framed compression that parses IN PLACE over a caller-padded buffer.
///
/// `buf[..logical_len]` is the input; `buf` MUST carry at least
/// [`INPLACE_TAIL_PAD`] trailing zero bytes past `logical_len`. This is the
/// allocation-lean T1 entry point: the caller reads the input once (e.g. via
/// `read_to_end`) and pads that same buffer (`resize(len + INPLACE_TAIL_PAD,
/// 0)`), so the compressor neither copies the input into a second work buffer
/// nor builds a separate output buffer. Output is byte-identical to
/// `compress_gzip(&buf[..logical_len], level)`.
pub fn compress_gzip_padded(buf: &[u8], logical_len: usize, level: u32) -> Vec<u8> {
    crate::anatomy_wall_root!({
        let cap = estimate_output_cap(logical_len, level, 32);
        crate::anatomy_count!(alloc_events);
        crate::anatomy_count!(alloc_bytes, cap);
        let mut out = Vec::with_capacity(cap);
        out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff]);

        deflate_padded_in_place(buf, logical_len, level, &mut out);

        let crc =
            crate::anatomy_wall_time!(crc_ns, crc_calls, { crc32fast::hash(&buf[..logical_len]) });
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(logical_len as u32).to_le_bytes());
        out
    })
}

/// Bytes of already-emitted input kept as matchfinder context across chunk
/// boundaries in the streaming encoder. DEFLATE's maximum back-reference
/// distance is 32768, so a chunk that carries the preceding 32 KiB can reach
/// every position the whole-buffer encoder could have reached from the same
/// offset — which is why streaming does not have to cost ratio.
const STREAM_HISTORY: usize = 32 * 1024;

/// Input consumed per streaming iteration: 65535 x 64 = 4_194_240 (~4 MiB).
///
/// ONE constant for every level, every input and every machine — no detection,
/// no per-archive tuning. Two measured properties chose it:
///
/// * **A multiple of [`MAX_STORED_SUBBLOCK`]**, so at level 0 the stored
///   sub-block boundaries fall exactly where the whole-buffer encoder puts
///   them and the streamed output is byte-identical, not merely equivalent.
/// * **Large enough that forced block boundaries stop costing ratio.** Each
///   chunk seam ends a DEFLATE block, so smaller chunks cost output size.
///   Swept over the 21-file corpus x L0-L9, restricted to the 7 files >= 25 MiB
///   (the ones genuinely multi-chunk at every sweep point), worst-case size
///   regression versus whole-buffer encoding:
///     1 MiB 0.0370% | 2 MiB 0.0411% | 4 MiB 0.0189% | 8 MiB 0.0196%
///   (level 3 excluded — its content detector is separately chunk-sensitive,
///   see `parse::gated`; that is a property of the detector, not of chunking,
///   and it is why level 3 does NOT take this path yet.)
///
/// 4 MiB is where the ratio cost flattens. Peak RSS is then ~4.3 MB against
/// gzip's 2.0 MB and libdeflate's 18.0 MB, versus the whole-buffer path's
/// 2.009x the input size.
pub const STREAM_CHUNK: usize = MAX_STORED_SUBBLOCK * 64;

/// Single-pass streaming with NO chunk seam in the output.
///
/// The plain chunked path calls the parser once per chunk, and each call emits
/// complete blocks over its own input range — so a block is forced to end at
/// every seam. That cost real bytes: against libdeflate, at the levels where
/// our output is otherwise byte-identical, multi-chunk files came out +66 to
/// +532 bytes larger, which flipped nine tied per-label SIZE cells to failing.
///
/// Here ONE [`parse::ParseState`] (matchfinder + `in_base` + `next_hashes`)
/// and ONE parse position span the whole file. Each pass parses only COMPLETE
/// blocks that had at least [`parse::STREAM_BLOCK_LOOKAHEAD`] bytes of input
/// behind them, and carries the unconsumed tail — always under ~305 KB —
/// into the next refill. That margin is what makes every block-boundary
/// decision identical to a whole-buffer encode.
///
/// Buffer layout is `[history | unconsumed | free]`, and it slides rather than
/// grows: once more than two windows of history accumulate, contents move down
/// by a whole number of `WINDOW_SIZE`s and `ParseState::shift_down` decrements
/// `in_base` by the same amount. Because the matchfinder stores every position
/// as `pos - in_base`, that is an O(1) pointer-rebase with no table rewrite —
/// the same trick zlib's sliding window uses.
fn stream_resumable<R: std::io::Read, W: std::io::Write>(
    reader: &mut R,
    writer: &mut W,
    level: u32,
    stream_chunk: usize,
) -> std::io::Result<u64> {
    use std::io::ErrorKind;

    let params = level::params(level);
    // Room for the most history the slide rule can leave behind (two windows),
    // the largest tail the parser can decline to consume, one refill, and the
    // matchfinder's speculative-load pad.
    let cap = 2 * matchfinder::hc::WINDOW_SIZE
        + parse::STREAM_BLOCK_LOOKAHEAD
        + stream_chunk
        + INPLACE_TAIL_PAD;
    crate::anatomy_count!(alloc_events);
    crate::anatomy_count!(alloc_bytes, cap);
    let mut buf = vec![0u8; cap];

    let mut state = parse::ParseState::new();
    let mut in_next = 0usize; // parse position, in buffer coordinates
    let mut avail = 0usize; // valid bytes in buf
    let mut eof = false;
    let mut crc = crc32fast::Hasher::new();
    let mut total: u64 = 0;

    let mut out = Vec::with_capacity(stream_chunk / 2 + 1024);
    out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff]);
    let mut bw = BitWriter::from_vec(out);

    // NOTE: no `anatomy_wall_cli!` here. The CLI route in `compress::mod`
    // already arms that span around this call, and arming a second one nests
    // it inside the first — both accumulate into `cli_ns`, doubling it and
    // leaving a `cli residual` of exactly 50%. That is what the first version
    // of this instrumentation did, and `cli_calls=2` is what gave it away.
    loop {
        // Refill. CRC covers exactly the new bytes, while they are still hot in
        // cache from the read — not a second sweep of the whole input.
        //
        // `read_input` and `stream_crc` are SIBLING outer regions, not nested:
        // timing them separately here keeps the production structure (crc
        // interleaved with the reads, so each chunk is checksummed while hot)
        // exactly as it is, while still letting the two costs be told apart.
        let fill_to = cap - INPLACE_TAIL_PAD;
        while !eof && avail < fill_to {
            let r = crate::anatomy_wall_time!(read_input_ns, read_input_calls, {
                reader.read(&mut buf[avail..fill_to])
            });
            match r {
                Ok(0) => eof = true,
                Ok(k) => {
                    crate::anatomy_wall_time!(stream_crc_ns, stream_crc_calls, {
                        crc.update(&buf[avail..avail + k]);
                    });
                    total += k as u64;
                    avail += k;
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }

        // Speculative loads read past the logical end; those bytes must read as
        // zero for the output to be byte-exact.
        buf[avail..avail + INPLACE_TAIL_PAD].fill(0);

        // Nothing left to parse. The whole-buffer path guards this inside
        // `deflate_into` (`in_end == data_start` emits an empty block); calling
        // the parser directly means guarding it here instead. Without this the
        // parser starts a block at the buffer end, reads the byte past it, and
        // walks off — an empty input, or an input whose last byte lands exactly
        // on a block boundary, panics in `calculate_min_match_len`.
        if in_next == avail {
            if eof {
                // Close the stream: a zero-length final block carrying BFINAL.
                emit_stored_block(&mut bw, &[], true);
                crate::anatomy_wall_time!(write_out_ns, write_out_calls, { bw.drain_to(writer) })?;
                break;
            }
        } else {
            // The `root` span: the encoder call proper. Fires once per pass
            // here rather than once per file, so `root_calls` is the pass
            // count — the inner regions still sum inside it, which is what the
            // level-1 conservation check needs.
            in_next = crate::anatomy_wall_root!({
                parse::compress_resumable(
                    &buf[..avail + INPLACE_TAIL_PAD],
                    &mut state,
                    in_next,
                    avail,
                    &params,
                    eof,
                    eof,
                    &mut bw,
                )
            });
            crate::anatomy_wall_time!(write_out_ns, write_out_calls, { bw.drain_to(writer) })?;
        }

        if eof {
            break;
        }

        // Slide: reclaim everything the matchfinder can no longer reference.
        // `max_shift` returns a whole number of windows and always leaves one
        // full window of history behind `in_base`.
        let shift = state.max_shift();
        if shift > 0 {
            buf.copy_within(shift..avail, 0);
            avail -= shift;
            in_next -= shift;
            state.shift_down(shift);
        }
        debug_assert!(
            avail < fill_to,
            "slide must free space or the loop cannot make progress"
        );
    }

    let mut tail = bw.finish();
    tail.extend_from_slice(&crc.finalize().to_le_bytes());
    tail.extend_from_slice(&(total as u32).to_le_bytes());
    crate::anatomy_wall_time!(write_out_ns, write_out_calls, { writer.write_all(&tail) })?;
    Ok(total)
}

/// Whether `level` may take the single-pass streaming encoder.
///
/// THE RULE: a level streams only when its output is PROVABLY unaffected by
/// streaming. Two ways to earn that, and no third:
///
/// * **Level 0** — stored blocks, and [`STREAM_CHUNK`] is a multiple of
///   [`MAX_STORED_SUBBLOCK`], so sub-block boundaries land exactly where the
///   whole-buffer encoder puts them.
/// * **A resumable parser** ([`parse::level_has_resumable_parser`]) — one
///   matchfinder and one parse position span the file, so block boundaries
///   come from the block splitter rather than from input refill.
///
/// Everything else keeps the whole-buffer path and its memory cost. That is a
/// deliberate ordering: being at-least-as-small at the level the user typed is
/// the contract, and peak RSS is not, so a level that cannot yet stream
/// without growing its output does not stream.
///
/// THIS RULE WAS LEARNED THE EXPENSIVE WAY, TWICE. The first streaming
/// version ran every level through a seamed chunk loop. At L2/4/5/6/7 that
/// cost +66 to +532 bytes against libdeflate — levels where the buffered path
/// was EXACTLY byte-identical to it — flipping nine tied per-label SIZE cells.
/// Those were fixed by making greedy/lazy resumable. Then the SAME defect was
/// found at L10-12 (NearOptimal, still seamed): +101 to +5944 bytes, six more
/// passing cells flipped, and they had gone unmeasured because the corpus
/// sweep only covered L0-L9. Both times the mistake was letting a level onto
/// the streaming path without a proof that its bytes could not change.
///
/// This is a level branch, not content detection — it reads only the number
/// the user typed. It should shrink by making more parsers resumable (Fast for
/// L1, NearOptimal for L10-12, and L3 once its content detector is gone), not
/// by relaxing the rule.
#[inline]
pub fn level_streams(level: u32) -> bool {
    level == 0 || parse::level_has_resumable_parser(level)
}

/// Compress `reader` into `writer` as a gzip stream in ONE pass, holding a
/// fixed ~1.1 MiB of buffer regardless of input size.
///
/// This is the streaming counterpart to [`compress_gzip_padded`], which
/// materializes the whole input in one `Vec` and the whole output in another.
/// Measured peak RSS of that approach on a 232.2 MiB input was 2.009x the
/// input at `-0` and input-plus-compressed-size at `-6`, against a flat 2.0 MB
/// for both gzip and pigz — a difference that stops being a ratio and starts
/// being a failure once the input approaches available memory.
///
/// Three structural properties, in the order they matter:
///
/// 1. **One continuous [`BitWriter`] spans every chunk.** Chunks are not
///    separately-encoded streams that get concatenated, so no sync-flush
///    marker is needed at the seams (`sync_flush = false`) and the seam leaves
///    no trace in the output. [`BitWriter::drain_to`] hands off complete bytes
///    after each chunk while the partial-bit accumulator carries over.
/// 2. **CRC is folded into the chunk pass**, over data still hot in cache,
///    instead of a separate monolithic sweep of the whole input. On the
///    whole-buffer path that sweep measured 29.3 ms on 232 MiB — a third of
///    the entire `-0` route.
/// 3. **The history window slides inside one buffer.** Layout is
///    `[STREAM_HISTORY | STREAM_CHUNK | INPLACE_TAIL_PAD]`; after each chunk
///    the trailing 32 KiB is `copy_within`'d down to the history slot. The
///    encoder parses in place and back-references reach into the history
///    exactly as they would mid-buffer.
///
/// The one-byte lookahead exists because DEFLATE must mark the final block
/// BFINAL *while encoding it*, and "did the reader end exactly on a chunk
/// boundary" is not knowable otherwise. Reading one byte past a full chunk
/// answers it; that byte becomes the first byte of the next chunk.
pub fn compress_gzip_streaming<R: std::io::Read, W: std::io::Write>(
    reader: &mut R,
    writer: &mut W,
    level: u32,
) -> std::io::Result<u64> {
    compress_gzip_streaming_chunked(reader, writer, level, STREAM_CHUNK)
}

/// [`compress_gzip_streaming`] with the chunk size supplied by the caller.
///
/// A MEASUREMENT SEAM, not a tuning knob: production has exactly one chunk
/// size, [`STREAM_CHUNK`], and no code path lets a user or an environment
/// variable choose another. It exists because chunk size trades peak memory
/// against output size (each chunk seam ends a block, and forced block
/// boundaries cost ratio), and that curve has to be measured to pick the
/// constant rather than guessed. `chunk` should be a multiple of
/// [`MAX_STORED_SUBBLOCK`] to keep level 0 byte-identical.
pub fn compress_gzip_streaming_chunked<R: std::io::Read, W: std::io::Write>(
    reader: &mut R,
    writer: &mut W,
    level: u32,
    chunk: usize,
) -> std::io::Result<u64> {
    use std::io::ErrorKind;
    let stream_chunk = chunk.max(MAX_STORED_SUBBLOCK);

    // Levels whose parser can resume take the SEAM-FREE path: one matchfinder
    // and one parse position carried across the whole file, with blocks ending
    // only where the block splitter says so.
    if level > 0 && parse::level_has_resumable_parser(level) {
        return stream_resumable(reader, writer, level, stream_chunk);
    }

    crate::anatomy_count!(alloc_events);
    crate::anatomy_count!(
        alloc_bytes,
        STREAM_HISTORY + stream_chunk + INPLACE_TAIL_PAD
    );
    let mut buf = vec![0u8; STREAM_HISTORY + stream_chunk + INPLACE_TAIL_PAD];

    // Valid history occupies `buf[STREAM_HISTORY - hist .. STREAM_HISTORY]`.
    // Starting at zero matters: seeding the matchfinder with the buffer's
    // initial zero fill would invent back-references to bytes that were never
    // in the input.
    let mut hist: usize = 0;
    let mut carry: Option<u8> = None;
    let mut crc = crc32fast::Hasher::new();
    let mut total: u64 = 0;

    let mut out = Vec::with_capacity(stream_chunk / 2 + 1024);
    out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff]);
    let mut bw = BitWriter::from_vec(out);

    loop {
        let data_at = STREAM_HISTORY;
        let mut n = 0usize;
        if let Some(b) = carry.take() {
            buf[data_at] = b;
            n = 1;
        }
        while n < stream_chunk {
            match reader.read(&mut buf[data_at + n..data_at + stream_chunk]) {
                Ok(0) => break,
                Ok(k) => n += k,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }

        // A short read means EOF. A full chunk is only final if one more read
        // returns nothing.
        let is_last = if n < stream_chunk {
            true
        } else {
            let mut one = [0u8; 1];
            loop {
                match reader.read(&mut one) {
                    Ok(0) => break true,
                    Ok(_) => {
                        carry = Some(one[0]);
                        break false;
                    }
                    Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                }
            }
        };

        crc.update(&buf[data_at..data_at + n]);
        total += n as u64;

        // The matchfinder's speculative loads read past the logical end; those
        // bytes must be zero for the output to be byte-exact. Re-zeroed every
        // iteration because the previous, longer chunk may have left data here.
        buf[data_at + n..data_at + n + INPLACE_TAIL_PAD].fill(0);

        // Hand the encoder `[history | chunk | pad]` with the history marked as
        // preset dictionary. Slicing from `STREAM_HISTORY - hist` (not 0) keeps
        // the not-yet-filled part of the history slot out of the dictionary.
        let region = &buf[STREAM_HISTORY - hist..data_at + n + INPLACE_TAIL_PAD];
        deflate_into(&mut bw, region, hist, hist + n, level, is_last, false);

        bw.drain_to(writer)?;

        // Slide the trailing window down for the next iteration.
        let keep = (hist + n).min(STREAM_HISTORY);
        let src_start = (data_at + n) - keep;
        buf.copy_within(src_start..data_at + n, STREAM_HISTORY - keep);
        hist = keep;

        if is_last {
            break;
        }
    }

    let mut tail = bw.finish();
    tail.extend_from_slice(&crc.finalize().to_le_bytes());
    tail.extend_from_slice(&(total as u32).to_le_bytes());
    writer.write_all(&tail)?;
    Ok(total)
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
                compress_block_streaming(&input[start..end], dict, level, is_last, &mut deflate);
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

    /// `compress_block_streaming(data, &[], level, true, ..)` must be
    /// byte-identical to the single-block [`compress_block`] (no sync marker,
    /// BFINAL set) — the regression guard the brief requires.
    #[test]
    fn is_last_no_dict_equals_compress_block() {
        let cases: [Vec<u8>; 3] = [Vec::new(), b"tiny".to_vec(), mixed_corpus(200_000)];
        for data in &cases {
            for level in [0u32, 1, 2, 6, 9, 12] {
                let mut streaming = Vec::new();
                compress_block_streaming(data, &[], level, true, &mut streaming);
                let mut block = Vec::new();
                compress_block(data, &[], level, &mut block);
                assert_eq!(
                    streaming,
                    block,
                    "streaming(is_last=true) diverged from compress_block at L{level}, len={}",
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
    /// `compress_gzip` (which builds a separate padded work buffer).
    fn assert_padded_gzip_matches(data: &[u8], level: u32) {
        let reference = compress_gzip(data, level);
        let (buf, logical_len) = padded(data);
        let inplace = compress_gzip_padded(&buf, logical_len, level);
        assert_eq!(
            reference,
            inplace,
            "compress_gzip_padded diverged at L{level}, len={}",
            data.len()
        );
    }

    /// The raw-DEFLATE in-place path must match `compress_block` (append form).
    fn assert_padded_block_matches(data: &[u8], level: u32) {
        let mut reference = Vec::new();
        compress_block(data, &[], level, &mut reference);
        let (buf, logical_len) = padded(data);
        let mut inplace = Vec::new();
        deflate_padded_in_place(&buf, logical_len, level, &mut inplace);
        assert_eq!(
            reference,
            inplace,
            "deflate_padded_in_place diverged at L{level}, len={}",
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
mod dict_tests {
    use super::*;

    /// An empty preset dictionary must yield byte-identical output to the
    /// no-dictionary path (regression guard on the seeding wiring).
    #[test]
    fn empty_dict_equals_no_dict() {
        let data: Vec<u8> = b"the pure-rust deflate encoder must roundtrip. ".repeat(400);
        for level in [2u32, 6, 9] {
            let mut with_empty = Vec::new();
            compress_block(&data, &[], level, &mut with_empty);
            let no_dict = compress_oneshot(&data, level);
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
                compress_block(&data, &dict, level, &mut v);
                v.len()
            };
            let without = compress_oneshot(&data, level).len();
            assert!(
                with_dict < without,
                "L{level}: dict-seeded {with_dict} not smaller than no-dict {without}",
            );
        }
    }
}
