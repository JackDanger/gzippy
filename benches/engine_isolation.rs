//! Engine isolation microbench — CLAUDE.md rule-3 "isolation oracle".
//!
//! Decodes ONE known-window CLEAN silesia deflate chunk three ways, byte-exact
//! each, and reports the per-variant CLEAN decode RATE (MB/s). This BOUNDS the
//! engine speed-up ceiling: it removes the parallel-SM scheduler/publish/marker
//! machinery and measures only the single-thread inner clean-decode rate.
//!
//! THREE VARIANTS (same input slice, same start_bit, same 32 KiB window, same N):
//!  (i)   VAR_I  scalar_u16 — gzippy's CURRENT clean inner loop:
//!        `marker_inflate::Block` with the window pre-primed via
//!        `set_initial_window` so `contains_marker_bytes == false` from the first
//!        block (a genuine clean decode). Output accumulates as `Vec<u16>` (one
//!        u16 per decoded byte through the u16 ring) and is narrowed u16->u8 ONCE
//!        at the end. This is the SCALAR u16 baseline.
//!  (ii)  VAR_II E1-partial — same `Block` clean inner loop, but the decode sink
//!        is u8-direct (`U8Sink`): the post-flip drain calls `push_clean_u8`,
//!        which here writes bytes STRAIGHT into a `Vec<u8>` with NO u16
//!        accumulation and NO final narrow pass. This halves the OUTPUT write
//!        traffic (u8 vs u16) and removes the narrow. NOTE: the inner ring itself
//!        is still u16 (a full E1 would make the ring u8 too) — so this bounds the
//!        OUTPUT-traffic component of E1, reported honestly as "E1-partial".
//!  (iii) VAR_III isal — `isal_decompress::decompress_deflate_from_bit`, the FFI
//!        ISA-L oracle (upper bound; FFI is a MEASUREMENT oracle only).
//!
//! Byte-exactness is the ABSOLUTE gate: all three outputs must be identical over
//! the first N bytes or the rate numbers are VOID.
//!
//! Self-test (RECALIBRATED round-2): on a clean single-thread chunk pure ISA-L
//! is ~3x gzippy's current scalar-u16 inner loop. The round-1 band [1.7x,2.6x]
//! was MIS-CALIBRATED — it was lifted from the 2.1-2.38x system-vs-system wall
//! ratio, but THIS bench's (iii) is a PURE ISA-L clean decode (no marker
//! machinery, no CRC), a purer/faster denominator that yields a LARGER honest
//! ratio (advisor-confirmed iii/ii ~= 3.10x, iii/i ~= 3.29x). PASS band
//! (iii)/(i) in [2.5x, 3.6x] (guest ratio; under Rosetta the absolute MB/s
//! differ but the ratio should still hold — note if it does not; the guest run
//! is authoritative).

#[cfg(all(
    target_arch = "x86_64",
    feature = "isal-compression",
    feature = "pure-rust-inflate"
))]
mod bench {
    use gzippy::decompress::inflate::consume_first_decode::Bits;
    use gzippy::decompress::parallel::lut_huffman::MAX_LIT_LEN_SYM;
    use gzippy::decompress::parallel::lut_huffman::{
        LutDistCode, LutLitLenCode, ISAL_DECODE_LONG_BITS, LARGE_FLAG_BIT, LARGE_SHORT_SYM_MASK,
        LARGE_SYM_COUNT_MASK, LARGE_SYM_COUNT_OFFSET,
    };
    use gzippy::decompress::parallel::marker_inflate::{
        Block, CompressionType, MarkerSink, DISTANCE_BASE, DISTANCE_EXTRA, END_OF_BLOCK_SYMBOL,
        MAX_WINDOW_SIZE,
    };
    use gzippy::isal_decompress_oracle::decompress_deflate_from_bit;
    use std::time::Instant;

    const SEED_PATH: &str = "/tmp/engine.seed";
    const CORPUS: &str = "benchmark_data/silesia-gzip.tar.gz";
    const REQUESTED_N: usize = 4 * 1024 * 1024;
    const ITERS: usize = 11; // best-of-N, N >= 9

    // ── GZSEEDW2 seed-file parse (mirror of seed_windows.rs:163-224) ──────────
    struct SeedEntry {
        start_bit: usize,
        window: Vec<u8>,
    }

    fn load_seed() -> Vec<SeedEntry> {
        let buf = std::fs::read(SEED_PATH)
            .unwrap_or_else(|e| panic!("cannot read seed {SEED_PATH}: {e} (run the capture step)"));
        assert!(
            buf.len() >= 16 && &buf[0..8] == b"GZSEEDW2",
            "bad seed magic"
        );
        let n = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
        let mut p = 16usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            assert!(p + 16 <= buf.len(), "truncated seed entry header");
            let off = u64::from_le_bytes(buf[p..p + 8].try_into().unwrap()) as usize;
            let len = u64::from_le_bytes(buf[p + 8..p + 16].try_into().unwrap()) as usize;
            p += 16;
            assert!(p + len <= buf.len(), "truncated seed entry body");
            out.push(SeedEntry {
                start_bit: off,
                window: buf[p..p + len].to_vec(),
            });
            p += len;
        }
        out.sort_by_key(|e| e.start_bit);
        out
    }

    /// Load the raw deflate slice (header stripped, 8-byte trailer dropped) —
    /// IDENTICAL base to what `sm_driver::read_parallel_sm` passes to
    /// `chunk_fetcher::drive`, i.e. the base that seed start_bits are relative to.
    fn load_deflate() -> Vec<u8> {
        let data = std::fs::read(CORPUS).unwrap_or_else(|e| panic!("cannot read {CORPUS}: {e}"));
        let (_h, header) = gzippy::decompress::parallel::gzip_format::read_header(&data)
            .expect("gzip header parse");
        data[header..data.len().saturating_sub(8)].to_vec()
    }

    // ── u8-direct sink for variant (ii) ───────────────────────────────────────
    // The clean-primed Block drains exclusively via `push_clean_u8` (drain's
    // contains_marker_bytes==false branch), so this sink only ever sees u8 bytes
    // and stores them directly — no u16 accumulation, no final narrow.
    struct U8Sink {
        data: Vec<u8>,
    }
    impl U8Sink {
        fn with_capacity(c: usize) -> Self {
            Self {
                data: Vec::with_capacity(c),
            }
        }
    }
    impl MarkerSink for U8Sink {
        #[inline]
        fn push_slice(&mut self, values: &[u16]) {
            // Defensive: a clean decode never hits this (drain uses push_clean_u8).
            for &v in values {
                debug_assert!((v as usize) < 256, "marker value {v:#x} on clean path");
                self.data.push(v as u8);
            }
        }
        #[inline]
        fn sink_len(&self) -> usize {
            self.data.len()
        }
        #[inline]
        fn as_slice(&self) -> &[u16] {
            &[]
        }
        #[inline]
        fn push_clean_u8(&mut self, bytes: &[u8]) {
            self.data.extend_from_slice(bytes);
        }
    }

    // ── Variant (i): scalar u16 clean decode → Vec<u16>, narrow once at end ────
    fn decode_var_i(deflate: &[u8], start_bit: usize, window: &[u8], target_n: usize) -> Vec<u8> {
        let mut block = Block::new();
        let mut dummy: Vec<u16> = Vec::new();
        block
            .set_initial_window(&mut dummy, window)
            .expect("prime window (i)");
        debug_assert!(
            !block.contains_marker_bytes(),
            "(i) window-primed block must be clean"
        );
        let mut sink: Vec<u16> = Vec::with_capacity(target_n + 4096);
        let mut bits = Bits::at_bit_offset(deflate, start_bit);
        loop {
            block.read_header(&mut bits, false).expect("(i) header");
            while !block.eob() {
                block
                    .read(&mut bits, &mut sink, usize::MAX)
                    .expect("(i) body");
            }
            if block.is_last_block() || sink.len() >= target_n {
                break;
            }
        }
        // Narrow u16 -> u8 (the variant-(i) final pass).
        sink.iter().map(|&v| v as u8).collect()
    }

    // ── Variant (ii): E1-partial u8-direct sink (no u16 accumulation/narrow) ───
    fn decode_var_ii(deflate: &[u8], start_bit: usize, window: &[u8], target_n: usize) -> Vec<u8> {
        let mut block = Block::new();
        let mut dummy: Vec<u16> = Vec::new();
        block
            .set_initial_window(&mut dummy, window)
            .expect("prime window (ii)");
        debug_assert!(
            !block.contains_marker_bytes(),
            "(ii) window-primed block must be clean"
        );
        let mut sink = U8Sink::with_capacity(target_n + 4096);
        let mut bits = Bits::at_bit_offset(deflate, start_bit);
        loop {
            block.read_header(&mut bits, false).expect("(ii) header");
            while !block.eob() {
                block
                    .read(&mut bits, &mut sink, usize::MAX)
                    .expect("(ii) body");
            }
            if block.is_last_block() || sink.sink_len() >= target_n {
                break;
            }
        }
        sink.data
    }

    // ── Variant (iii): ISA-L FFI oracle ───────────────────────────────────────
    fn decode_var_iii(deflate: &[u8], start_bit: usize, window: &[u8], target_n: usize) -> Vec<u8> {
        decompress_deflate_from_bit(deflate, start_bit, window, target_n)
            .expect("(iii) isal decode")
    }

    // ── Variant (iv): clean E2/E3/E4 engine via read_clean_e234 ────────────────
    // Drives the new `Block::read_clean_e234` clean-only sibling (const-generic
    // E2/E3/E4 flags) and drains via `drain_clean_u8` into a Vec<u8> directly —
    // same u8-direct sink as variant (ii), so the E-deltas isolate the inner
    // technique, not the output-traffic component already in (ii).
    fn decode_var_iv<const E2: bool, const E3: bool, const E4: bool>(
        deflate: &[u8],
        start_bit: usize,
        window: &[u8],
        target_n: usize,
    ) -> Vec<u8> {
        let mut block = Block::new();
        let mut dummy: Vec<u16> = Vec::new();
        block
            .set_initial_window(&mut dummy, window)
            .expect("prime window (iv)");
        debug_assert!(
            !block.contains_marker_bytes(),
            "(iv) window-primed block must be clean"
        );
        let mut sink: Vec<u8> = Vec::with_capacity(target_n + 4096);
        let mut bits = Bits::at_bit_offset(deflate, start_bit);
        loop {
            if block.read_header(&mut bits, false).is_err() {
                break;
            }
            while !block.eob() {
                if block
                    .read_clean_e234::<E2, E3, E4>(&mut bits, usize::MAX)
                    .is_err()
                {
                    return sink;
                }
                block.drain_clean_u8(&mut sink);
            }
            if block.is_last_block() || sink.len() >= target_n {
                break;
            }
        }
        sink
    }

    // ── Variant (v): FLAT-u8 packed-table + SPECULATIVE SOFTWARE-PIPELINED loop ─
    //
    // This is the inner-Huffman-kernel LEVER (plans/inner-huffman-kernel.md): igzip's
    // speculative pipeline (igzip_decode_block_stateless.asm:507-627), built on the
    // FLAT-u8 path the faithful u8 rewrite enabled. It reuses the EXISTING igzip
    // packed-flat-short-code table (LutLitLenCode/LutDistCode, lut_huffman.rs — trick
    // #1, already production-live) and adds the missing trick #2:
    //
    //  * FLAT LINEAR u8 output buffer with the 32 KiB window prepended as the head
    //    (NOT a ring): back-refs read out[out_pos - distance], always already-final
    //    u8 — no `% RING_SIZE`, no wrap special-case inside the fast region. This is
    //    igzip's "window is the tail of the same buffer" precondition (asm:518/591/605).
    //  * SPECULATIVE 8-byte packed-literal store: write all 8 bytes of the packed
    //    `sym` UNCONDITIONALLY (one u64 store), then advance out_pos by the ACTUAL
    //    sym_count (1-3). Wrong-guess bytes are overwritten next iteration. Branchless
    //    multi-literal output (asm:518-519).
    //  * SLOP-MARGIN HEADROOM GUARD: the fast loop runs ONLY while there is
    //    >= (16 + 258) bytes of output headroom AND >= 8 bytes of input slop; inside
    //    that region there are NO per-symbol bounds checks (asm:48,488-512). A careful
    //    per-symbol tail handles the boundary.
    //  * PRELOAD: the next lit/len symbol is decoded BEFORE the current symbol's
    //    back-ref branch resolves, hiding the dependent table-load latency (asm:524-525).
    //  * WORD/overlap-doubling back-ref copy on the flat u8 buffer (asm:558-627):
    //    8-byte word copy for distance>=8, RLE memset for distance==1, byte tail.
    //
    // Byte-exactness is the absolute gate: VAR_V must be SHA-equal to VAR_I scalar AND
    // VAR_III ISA-L over the swept clean chunks, or the rate is VOID.

    // RFC 1951 §3.2.6 fixed-Huffman code lengths (FIXED_LIT_LEN_LENGTHS is private in
    // marker_inflate; reconstruct the 288-entry litlen table + 30-entry dist here).
    fn fixed_litlen_lengths() -> [u8; 288] {
        let mut t = [0u8; 288];
        for (i, v) in t.iter_mut().enumerate() {
            *v = if i < 144 {
                8
            } else if i < 256 {
                9
            } else if i < 280 {
                7
            } else {
                8
            };
        }
        t
    }

    /// Build the igzip packed-flat-short-code tables for ONE block from the code
    /// lengths the driving `Block` parsed in `read_header`. Returns None on a
    /// stored block or an invalid table (the speculative variant only handles
    /// fixed/dynamic compressed blocks; stored blocks are handled inline).
    fn build_block_tables(block: &Block) -> Option<(LutLitLenCode, LutDistCode)> {
        let mut litlen = LutLitLenCode::new_empty();
        let mut dist = LutDistCode::new_empty();
        match block.compression_type() {
            CompressionType::FixedHuffman => {
                let ll = fixed_litlen_lengths();
                let dl = [5u8; 30];
                if !litlen.rebuild_from(&ll) || !dist.rebuild_from(&dl) {
                    return None;
                }
            }
            CompressionType::DynamicHuffman => {
                let split = block.literal_code_count;
                let end = split + block.distance_code_count;
                if end > block.literal_cl.len() {
                    return None;
                }
                let ll = &block.literal_cl[..split];
                let dl = &block.literal_cl[split..end];
                if !litlen.rebuild_from(ll) || !dist.rebuild_from(dl) {
                    return None;
                }
            }
            _ => return None,
        }
        Some((litlen, dist))
    }

    // Output headroom the fast loop reserves so it can over-write without a
    // per-symbol bounds check: up to 8 speculative literal bytes + a 258-byte
    // max-length back-ref + a 16-byte word-copy overshoot (igzip asm:511).
    const OUT_SLOP: usize = 8 + 258 + 16;
    // Input slop so the bit refill can always read an 8-byte word (igzip
    // IN_BUFFER_SLOP, asm:48).
    const IN_SLOP: usize = 8;

    /// Word/overlap-doubling back-ref copy on a FLAT u8 buffer. `out_pos` is the
    /// current write cursor; the source is `out_pos - distance`, always already-
    /// final bytes (flat linear, window prepended). Caller guarantees
    /// `out_pos + length + 16 <= out.len()` (headroom guard) so this over-writes
    /// freely. Mirrors igzip large_byte_copy / small_byte_copy (asm:603-627).
    #[inline(always)]
    unsafe fn flat_backref_copy(out: *mut u8, out_pos: usize, distance: usize, length: usize) {
        let dst0 = out.add(out_pos);
        let src0 = out.add(out_pos - distance);
        // Discriminator MIRRORS production emit_backref_ring_u8 (marker_inflate.rs
        // :2704): the 8-byte word copy is correct ONLY for NON-overlapping copies
        // (`distance >= length`), where the source run is fully `length` bytes
        // behind the dest so the rounded-up 8-byte stride never aliases a not-yet-
        // written byte. For `1 < distance < length` the copy overlaps and the word
        // copy would read ahead of the just-written pattern — must go byte-by-byte
        // (or distance-doubling). distance==1 is RLE.
        if distance >= length {
            if distance >= 8 {
                // Word copy, may overshoot up to 7 bytes (headroom-licensed).
                let mut src = src0;
                let mut dst = dst0;
                let mut copied = 0usize;
                while copied < length {
                    let w = (src as *const u64).read_unaligned();
                    (dst as *mut u64).write_unaligned(w);
                    src = src.add(8);
                    dst = dst.add(8);
                    copied += 8;
                }
            } else {
                // distance < 8 but non-overlap (distance >= length so length < 8):
                // exact byte copy.
                for i in 0..length {
                    *dst0.add(i) = *src0.add(i);
                }
            }
        } else if distance == 1 {
            // RLE memset.
            let b = *src0;
            std::ptr::write_bytes(dst0, b, length);
        } else {
            // Overlap (1 < distance < length): sequential self-replicating copy.
            for i in 0..length {
                *dst0.add(i) = *src0.add(i);
            }
        }
    }

    // ── BMI2 + AVX wide-copy primitives for VAR_VI ───────────────────────────
    //
    // VAR_VI = VAR_V's speculative flat-u8 pipeline + the two remaining igzip
    // techniques the kernel-bench had not yet measured:
    //   (1) BMI2 BZHI for the VARIABLE-width bit extraction in the hot path
    //       (distance extra-bits mask `peek & ((1<<extra)-1)` — exactly BZHI's
    //       purpose; the fixed 12/10-bit table masks lower to AND-imm already).
    //       SHRX for the variable consume shift (no flag dependency, frees the
    //       refill chain). Mirrors igzip's SHLX/SHRX/BZHI Haswell build.
    //   (2) MOVDQU/AVX wide overlap-doubling back-ref copy. igzip uses SSE xmm
    //       MOVDQU (16-byte) for the copy (asm:603-627); we add a 16-byte SSE
    //       path and a 32-byte AVX2 path for the long-match bulk, on the flat u8
    //       buffer (no ring, so the copy is a straight forward memmove-style
    //       run for non-overlapping distances).

    /// BMI2 BZHI — zero the high bits of `v` from bit `n` upward (keep low `n`).
    /// `peek & ((1<<n)-1)` with a single hardware instruction, no mask
    /// materialization, no `n==64` UB.
    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    unsafe fn bzhi64(v: u64, n: u32) -> u64 {
        core::arch::x86_64::_bzhi_u64(v, n)
    }

    /// AVX2/SSE wide overlap-doubling back-ref copy on a FLAT u8 buffer.
    /// Semantics identical to `flat_backref_copy` (caller guarantees
    /// `out_pos + length + 32 <= cap` headroom). For non-overlapping copies
    /// (`distance >= length`) it uses 32-byte AVX2 stores for the bulk, 16-byte
    /// SSE for the remainder; distance==1 is RLE memset; overlapping runs use
    /// the distance-doubling SSE technique (write the first `distance` bytes,
    /// then double the written prefix with 16-byte copies). This is igzip's
    /// MOVDQU overlap-copy generalized to AVX2 width.
    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    unsafe fn avx_backref_copy(out: *mut u8, out_pos: usize, distance: usize, length: usize) {
        use core::arch::x86_64::{
            _mm256_loadu_si256, _mm256_storeu_si256, _mm_loadu_si128, _mm_storeu_si128,
        };
        let dst0 = out.add(out_pos);
        let src0 = out.add(out_pos - distance);
        if distance == 1 {
            std::ptr::write_bytes(dst0, *src0, length);
            return;
        }
        if distance >= length {
            // Non-overlapping: straight wide copy, over-write licensed by slop.
            let mut copied = 0usize;
            while copied + 32 <= length {
                let v = _mm256_loadu_si256(src0.add(copied) as *const _);
                _mm256_storeu_si256(dst0.add(copied) as *mut _, v);
                copied += 32;
            }
            if copied < length {
                // One 32-byte tail store (over-writes up to 31 bytes; licensed).
                let v = _mm256_loadu_si256(src0.add(copied) as *const _);
                _mm256_storeu_si256(dst0.add(copied) as *mut _, v);
            }
            return;
        }
        // Overlapping (1 < distance < length): distance-doubling with SSE.
        // First materialize the `distance`-byte seed, then repeatedly copy the
        // already-written prefix forward in 16-byte chunks until length filled.
        if distance >= 16 {
            // Seed is already >=16 bytes of valid history behind dst0; copy
            // forward in 16-byte SSE stores. Because distance>=16 each 16-byte
            // load reads only already-written bytes.
            let mut copied = 0usize;
            while copied < length {
                let v = _mm_loadu_si128(src0.add(copied) as *const _);
                _mm_storeu_si128(dst0.add(copied) as *mut _, v);
                copied += 16;
            }
        } else {
            // Small overlap (2..15): byte-accurate self-replicating copy.
            for i in 0..length {
                *dst0.add(i) = *src0.add(i);
            }
        }
    }

    fn decode_var_v(deflate: &[u8], start_bit: usize, window: &[u8], target_n: usize) -> Vec<u8> {
        // Flat linear output: [0..MAX_WINDOW_SIZE) = prepended window history,
        // [MAX_WINDOW_SIZE..) = decoded bytes. Back-refs into history read the
        // window region directly (no ring). Reserve OUT_SLOP so the fast loop can
        // over-write past the logical end without a per-symbol bounds check.
        let base = MAX_WINDOW_SIZE;
        let cap = base + target_n + OUT_SLOP + 4096;
        let mut out: Vec<u8> = vec![0u8; cap];
        out[..base].copy_from_slice(&window[..base.min(window.len())]);
        let out_ptr = out.as_mut_ptr();

        // Drive a Block purely to PARSE block headers (BFINAL/BTYPE + dynamic
        // code lengths). The decode itself is done by the speculative loop below,
        // NOT by Block::read — Block here is a header parser only.
        let mut block = Block::new();
        {
            let mut dummy: Vec<u16> = Vec::new();
            // Prime so the header parser starts in clean mode (matches the other
            // variants); the speculative loop does not use the Block ring at all.
            block
                .set_initial_window(&mut dummy, window)
                .expect("prime window (v)");
        }

        let mut bits = Bits::at_bit_offset(deflate, start_bit);
        let in_end = deflate.len();
        // out_pos is the absolute write index into `out` (>= base).
        let mut out_pos = base;
        let target_end = base + target_n; // stop emitting once we reach target_n

        'blocks: loop {
            if block.read_header(&mut bits, false).is_err() {
                break;
            }
            match block.compression_type() {
                CompressionType::Uncompressed => {
                    // Stored block: byte-aligned literal copy. Read length, copy.
                    bits.align_to_byte();
                    // LEN (16) then NLEN (16). Pull via read_u16.
                    let len = bits.read_u16() as usize;
                    let _nlen = bits.read_u16();
                    for _ in 0..len {
                        if bits.available() < 8 {
                            bits.refill();
                        }
                        let b = (bits.peek() & 0xFF) as u8;
                        bits.consume(8);
                        unsafe {
                            *out_ptr.add(out_pos) = b;
                        }
                        out_pos += 1;
                    }
                    if block.is_last_block() || out_pos >= target_end {
                        break 'blocks;
                    }
                    continue 'blocks;
                }
                CompressionType::FixedHuffman | CompressionType::DynamicHuffman => {}
                CompressionType::Reserved => break 'blocks,
            }

            let (litlen, dist) = match build_block_tables(&block) {
                Some(t) => t,
                None => break 'blocks,
            };

            // ── SPECULATIVE SOFTWARE-PIPELINED FAST LOOP ──────────────────────
            // Runs while headroom (out) AND slop (in) permit unchecked over-
            // read/write (igzip asm:488-512). Preloads the next lit/len symbol
            // before resolving the current packet's back-ref branch.
            //
            // PACKET SEMANTICS (production marker_inflate.rs:1492-1602): one
            // `litlen.decode` returns a packet of `sym_count` elements packed
            // low-byte-first. Elements are LITERALS while their value <= 255; a
            // trailing element with value > 255 is a LENGTH code (igzip packs
            // literal + (literal|length)). So the packet is: a literal PREFIX
            // (speculative 8-byte store, advance by the count of leading
            // literals) followed by an OPTIONAL trailing length code (back-ref).
            let mut at_eob = false;
            bits.refill();
            let mut pre = litlen.decode(&mut bits); // PRELOAD
            'fast: loop {
                let out_ok = out_pos + OUT_SLOP < cap;
                let in_ok = bits.pos + IN_SLOP < in_end;
                if !(out_ok && in_ok) {
                    break;
                }
                let sym = pre.symbol;
                let sym_count = pre.sym_count;
                if pre.bit_count == 0 {
                    return out[base..out_pos.min(target_end)].to_vec();
                }
                bits.consume(pre.bit_count);

                // SPECULATIVE 8-byte store of the packed bytes (igzip asm:518):
                // write all up-to-3 packed bytes unconditionally, then advance by
                // the count of LEADING LITERALS only. Wrong bytes are overwritten
                // by the next packet (or by the back-ref below).
                unsafe {
                    let packed = (sym & 0x00FF_FFFF) as u64;
                    (out_ptr.add(out_pos) as *mut u64).write_unaligned(packed);
                }
                // Count leading literals (production unpack loop semantics).
                let mut s = sym;
                let mut remaining = sym_count;
                let mut lit_prefix = 0u32;
                let mut trailing_code: Option<u16> = None;
                while remaining > 0 {
                    let code = (s & 0xFFFF) as u16;
                    if code <= 255 || remaining > 1 {
                        // Literal (multi-pack always literal except the last
                        // element; the last element may be a length code).
                        if remaining == 1 && code > 255 {
                            trailing_code = Some(code);
                            break;
                        }
                        lit_prefix += 1;
                        remaining -= 1;
                        s >>= 8;
                        continue;
                    }
                    // remaining == 1, code > 255: trailing length/EOB.
                    trailing_code = Some(code);
                    break;
                }
                out_pos += lit_prefix as usize;

                if let Some(code) = trailing_code {
                    if code == END_OF_BLOCK_SYMBOL {
                        at_eob = true;
                        break 'fast;
                    }
                    if (code as u32) > MAX_LIT_LEN_SYM {
                        return out[base..out_pos.min(target_end)].to_vec();
                    }
                    let length = (code as usize).wrapping_sub(254);
                    if length != 0 {
                        let (dsym, dbits) = match dist.decode(&mut bits) {
                            Some(d) => d,
                            None => return out[base..out_pos.min(target_end)].to_vec(),
                        };
                        bits.consume(dbits);
                        if dsym as usize >= DISTANCE_BASE.len() {
                            return out[base..out_pos.min(target_end)].to_vec();
                        }
                        let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                        let distance = if extra > 0 {
                            if bits.available() < extra {
                                bits.refill();
                            }
                            let mask = (1u64 << extra) - 1;
                            let v = (bits.peek() & mask) as usize;
                            bits.consume(extra);
                            DISTANCE_BASE[dsym as usize] as usize + v
                        } else {
                            DISTANCE_BASE[dsym as usize] as usize
                        };
                        if distance == 0 || distance > out_pos {
                            return out[base..out_pos.min(target_end)].to_vec();
                        }
                        unsafe {
                            flat_backref_copy(out_ptr, out_pos, distance, length);
                        }
                        out_pos += length;
                    }
                }

                bits.refill();
                pre = litlen.decode(&mut bits); // PRELOAD next
                if out_pos >= target_end {
                    break 'fast;
                }
            }

            // ── CAREFUL TAIL: per-symbol, bounds-checked, to the block boundary ─
            // The fast loop ALWAYS consumed `pre`'s bits before preloading, so at
            // every `break` `pre` is a FRESH un-consumed decode. Process it (and
            // continue) with full bounds checks until EOB or target_end.
            if !at_eob {
                let mut cur = pre;
                'careful: loop {
                    if out_pos >= target_end {
                        break;
                    }
                    let sym = cur.symbol;
                    let sym_count = cur.sym_count;
                    if cur.bit_count == 0 {
                        return out[base..out_pos.min(target_end)].to_vec();
                    }
                    bits.consume(cur.bit_count);
                    let mut s = sym;
                    let mut remaining = sym_count;
                    // Unpack literals (bounds-checked, byte-by-byte).
                    while remaining > 0 {
                        let code = (s & 0xFFFF) as u16;
                        if code <= 255 || remaining > 1 {
                            if remaining == 1 && code > 255 {
                                break;
                            }
                            if out_pos >= cap {
                                return out[base..target_end.min(out_pos)].to_vec();
                            }
                            unsafe {
                                *out_ptr.add(out_pos) = (code & 0xFF) as u8;
                            }
                            out_pos += 1;
                            remaining -= 1;
                            s >>= 8;
                            continue;
                        }
                        break;
                    }
                    if remaining == 1 {
                        let code = (s & 0xFFFF) as u16;
                        if code == END_OF_BLOCK_SYMBOL {
                            // EOB: this block is done; the outer block loop reads
                            // the next header. (No need to set `at_eob` — it is
                            // only read to decide whether to ENTER this tail.)
                            break 'careful;
                        }
                        if (code as u32) > MAX_LIT_LEN_SYM {
                            return out[base..out_pos.min(target_end)].to_vec();
                        }
                        let length = (code as usize).wrapping_sub(254);
                        if length != 0 {
                            let (dsym, dbits) = match dist.decode(&mut bits) {
                                Some(d) => d,
                                None => return out[base..out_pos.min(target_end)].to_vec(),
                            };
                            bits.consume(dbits);
                            if dsym as usize >= DISTANCE_BASE.len() {
                                return out[base..out_pos.min(target_end)].to_vec();
                            }
                            let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                            let distance = if extra > 0 {
                                if bits.available() < extra {
                                    bits.refill();
                                }
                                let mask = (1u64 << extra) - 1;
                                let v = (bits.peek() & mask) as usize;
                                bits.consume(extra);
                                DISTANCE_BASE[dsym as usize] as usize + v
                            } else {
                                DISTANCE_BASE[dsym as usize] as usize
                            };
                            if distance == 0 || distance > out_pos {
                                return out[base..out_pos.min(target_end)].to_vec();
                            }
                            if out_pos + length + 16 > cap {
                                for i in 0..length {
                                    if out_pos + i >= cap {
                                        break;
                                    }
                                    unsafe {
                                        let v = *out_ptr.add(out_pos + i - distance);
                                        *out_ptr.add(out_pos + i) = v;
                                    }
                                }
                            } else {
                                unsafe {
                                    flat_backref_copy(out_ptr, out_pos, distance, length);
                                }
                            }
                            out_pos += length;
                        }
                    }
                    bits.refill();
                    cur = litlen.decode(&mut bits);
                }
            }

            if block.is_last_block() || out_pos >= target_end {
                break 'blocks;
            }
            // The bit cursor is positioned just after this block's EOB symbol, so
            // the next `read_header` parses the following block's BFINAL/BTYPE.
        }

        // Return ONLY the decoded region (drop the prepended window), clamped to
        // target_n.
        let end = out_pos.min(target_end);
        out[base..end].to_vec()
    }

    // ── Variant (vi): VAR_V + BMI2 BZHI/SHRX + AVX wide overlap copy ──────────
    //
    // Structurally IDENTICAL to decode_var_v (same speculative software-pipelined
    // fast loop, same careful tail, same packed-u32 multi-symbol table reuse —
    // trick #3 confirmed: it drives the SAME `litlen.decode` packed packets and
    // unpacks up to 3 packed literals per decode). The ONLY differences are the
    // two added igzip techniques:
    //   * distance extra-bits extracted via BMI2 BZHI (bzhi64) instead of a
    //     materialized `(1<<extra)-1` mask;
    //   * back-ref copy via `avx_backref_copy` (AVX2 32-byte / SSE 16-byte
    //     MOVDQU) instead of the 8-byte word `flat_backref_copy`.
    // On a non-x86_64 / non-AVX2 host it would fall back, but this bench only
    // compiles on x86_64 (cfg guard on `mod bench`); the AVX2 path is live on
    // the guest (avx2_detected=true) and validated by the byte-exact gate.
    #[cfg(target_arch = "x86_64")]
    fn decode_var_vi(deflate: &[u8], start_bit: usize, window: &[u8], target_n: usize) -> Vec<u8> {
        let base = MAX_WINDOW_SIZE;
        // Larger slop: AVX copy can over-write up to 31 bytes past `length`.
        let out_slop = OUT_SLOP + 32;
        let cap = base + target_n + out_slop + 4096;
        let mut out: Vec<u8> = vec![0u8; cap];
        out[..base].copy_from_slice(&window[..base.min(window.len())]);
        let out_ptr = out.as_mut_ptr();

        let mut block = Block::new();
        {
            let mut dummy: Vec<u16> = Vec::new();
            block
                .set_initial_window(&mut dummy, window)
                .expect("prime window (vi)");
        }

        let mut bits = Bits::at_bit_offset(deflate, start_bit);
        let in_end = deflate.len();
        let mut out_pos = base;
        let target_end = base + target_n;

        'blocks: loop {
            if block.read_header(&mut bits, false).is_err() {
                break;
            }
            match block.compression_type() {
                CompressionType::Uncompressed => {
                    bits.align_to_byte();
                    let len = bits.read_u16() as usize;
                    let _nlen = bits.read_u16();
                    for _ in 0..len {
                        if bits.available() < 8 {
                            bits.refill();
                        }
                        let b = (bits.peek() & 0xFF) as u8;
                        bits.consume(8);
                        unsafe {
                            *out_ptr.add(out_pos) = b;
                        }
                        out_pos += 1;
                    }
                    if block.is_last_block() || out_pos >= target_end {
                        break 'blocks;
                    }
                    continue 'blocks;
                }
                CompressionType::FixedHuffman | CompressionType::DynamicHuffman => {}
                CompressionType::Reserved => break 'blocks,
            }

            let (litlen, dist) = match build_block_tables(&block) {
                Some(t) => t,
                None => break 'blocks,
            };

            let mut at_eob = false;
            bits.refill();
            let mut pre = litlen.decode(&mut bits); // PRELOAD
            'fast: loop {
                let out_ok = out_pos + out_slop < cap;
                let in_ok = bits.pos + IN_SLOP < in_end;
                if !(out_ok && in_ok) {
                    break;
                }
                let sym = pre.symbol;
                let sym_count = pre.sym_count;
                if pre.bit_count == 0 {
                    return out[base..out_pos.min(target_end)].to_vec();
                }
                bits.consume(pre.bit_count);

                unsafe {
                    let packed = (sym & 0x00FF_FFFF) as u64;
                    (out_ptr.add(out_pos) as *mut u64).write_unaligned(packed);
                }
                let mut s = sym;
                let mut remaining = sym_count;
                let mut lit_prefix = 0u32;
                let mut trailing_code: Option<u16> = None;
                while remaining > 0 {
                    let code = (s & 0xFFFF) as u16;
                    if code <= 255 || remaining > 1 {
                        if remaining == 1 && code > 255 {
                            trailing_code = Some(code);
                            break;
                        }
                        lit_prefix += 1;
                        remaining -= 1;
                        s >>= 8;
                        continue;
                    }
                    trailing_code = Some(code);
                    break;
                }
                out_pos += lit_prefix as usize;

                if let Some(code) = trailing_code {
                    if code == END_OF_BLOCK_SYMBOL {
                        at_eob = true;
                        break 'fast;
                    }
                    if (code as u32) > MAX_LIT_LEN_SYM {
                        return out[base..out_pos.min(target_end)].to_vec();
                    }
                    let length = (code as usize).wrapping_sub(254);
                    if length != 0 {
                        let (dsym, dbits) = match dist.decode(&mut bits) {
                            Some(d) => d,
                            None => return out[base..out_pos.min(target_end)].to_vec(),
                        };
                        bits.consume(dbits);
                        if dsym as usize >= DISTANCE_BASE.len() {
                            return out[base..out_pos.min(target_end)].to_vec();
                        }
                        let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                        let distance = if extra > 0 {
                            if bits.available() < extra {
                                bits.refill();
                            }
                            // BMI2 BZHI: keep low `extra` bits, no mask materialize.
                            let v = unsafe { bzhi64(bits.peek(), extra) } as usize;
                            bits.consume(extra);
                            DISTANCE_BASE[dsym as usize] as usize + v
                        } else {
                            DISTANCE_BASE[dsym as usize] as usize
                        };
                        if distance == 0 || distance > out_pos {
                            return out[base..out_pos.min(target_end)].to_vec();
                        }
                        unsafe {
                            avx_backref_copy(out_ptr, out_pos, distance, length);
                        }
                        out_pos += length;
                    }
                }

                bits.refill();
                pre = litlen.decode(&mut bits); // PRELOAD next
                if out_pos >= target_end {
                    break 'fast;
                }
            }

            // ── CAREFUL TAIL (bounds-checked; uses scalar copy for safety) ─────
            if !at_eob {
                let mut cur = pre;
                'careful: loop {
                    if out_pos >= target_end {
                        break;
                    }
                    let sym = cur.symbol;
                    let sym_count = cur.sym_count;
                    if cur.bit_count == 0 {
                        return out[base..out_pos.min(target_end)].to_vec();
                    }
                    bits.consume(cur.bit_count);
                    let mut s = sym;
                    let mut remaining = sym_count;
                    while remaining > 0 {
                        let code = (s & 0xFFFF) as u16;
                        if code <= 255 || remaining > 1 {
                            if remaining == 1 && code > 255 {
                                break;
                            }
                            if out_pos >= cap {
                                return out[base..target_end.min(out_pos)].to_vec();
                            }
                            unsafe {
                                *out_ptr.add(out_pos) = (code & 0xFF) as u8;
                            }
                            out_pos += 1;
                            remaining -= 1;
                            s >>= 8;
                            continue;
                        }
                        break;
                    }
                    if remaining == 1 {
                        let code = (s & 0xFFFF) as u16;
                        if code == END_OF_BLOCK_SYMBOL {
                            break 'careful;
                        }
                        if (code as u32) > MAX_LIT_LEN_SYM {
                            return out[base..out_pos.min(target_end)].to_vec();
                        }
                        let length = (code as usize).wrapping_sub(254);
                        if length != 0 {
                            let (dsym, dbits) = match dist.decode(&mut bits) {
                                Some(d) => d,
                                None => return out[base..out_pos.min(target_end)].to_vec(),
                            };
                            bits.consume(dbits);
                            if dsym as usize >= DISTANCE_BASE.len() {
                                return out[base..out_pos.min(target_end)].to_vec();
                            }
                            let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                            let distance = if extra > 0 {
                                if bits.available() < extra {
                                    bits.refill();
                                }
                                let v = unsafe { bzhi64(bits.peek(), extra) } as usize;
                                bits.consume(extra);
                                DISTANCE_BASE[dsym as usize] as usize + v
                            } else {
                                DISTANCE_BASE[dsym as usize] as usize
                            };
                            if distance == 0 || distance > out_pos {
                                return out[base..out_pos.min(target_end)].to_vec();
                            }
                            if out_pos + length + 32 > cap {
                                for i in 0..length {
                                    if out_pos + i >= cap {
                                        break;
                                    }
                                    unsafe {
                                        let v = *out_ptr.add(out_pos + i - distance);
                                        *out_ptr.add(out_pos + i) = v;
                                    }
                                }
                            } else {
                                unsafe {
                                    avx_backref_copy(out_ptr, out_pos, distance, length);
                                }
                            }
                            out_pos += length;
                        }
                    }
                    bits.refill();
                    cur = litlen.decode(&mut bits);
                }
            }

            if block.is_last_block() || out_pos >= target_end {
                break 'blocks;
            }
        }

        let end = out_pos.min(target_end);
        out[base..end].to_vec()
    }

    // Decode the distance for a length code and copy the `length`-byte back-ref
    // into the flat u8 buffer (byte-identical to VAR_VI's careful-tail back-ref).
    // Returns false on a malformed distance (caller bails). Used by VAR_VII to
    // resolve the trailing length code the asm hot loop hands back.
    #[cfg(target_arch = "x86_64")]
    fn emit_one_backref(
        dist: &LutDistCode,
        bits: &mut Bits<'_>,
        out_ptr: *mut u8,
        out_pos: &mut usize,
        cap: usize,
        length: usize,
    ) -> bool {
        let (dsym, dbits) = match dist.decode(bits) {
            Some(d) => d,
            None => return false,
        };
        bits.consume(dbits);
        if dsym as usize >= DISTANCE_BASE.len() {
            return false;
        }
        let extra = DISTANCE_EXTRA[dsym as usize] as u32;
        let distance = if extra > 0 {
            if bits.available() < extra {
                bits.refill();
            }
            let v = unsafe { bzhi64(bits.peek(), extra) } as usize;
            bits.consume(extra);
            DISTANCE_BASE[dsym as usize] as usize + v
        } else {
            DISTANCE_BASE[dsym as usize] as usize
        };
        if distance == 0 || distance > *out_pos {
            return false;
        }
        if *out_pos + length + 32 > cap {
            for i in 0..length {
                if *out_pos + i >= cap {
                    break;
                }
                unsafe {
                    let v = *out_ptr.add(*out_pos + i - distance);
                    *out_ptr.add(*out_pos + i) = v;
                }
            }
        } else {
            unsafe {
                avx_backref_copy(out_ptr, *out_pos, distance, length);
            }
        }
        *out_pos += length;
        true
    }

    // VAR_VI's per-symbol careful tail, factored out so VAR_VII re-enters it after
    // the asm loop exits. Decodes via litlen.decode (the validated Rust short/long
    // path) until EOB, target_end, or out-of-room. Returns false to signal the
    // caller should bail (malformed / truncated) — true on clean EOB/target.
    #[cfg(target_arch = "x86_64")]
    fn run_careful_tail(
        litlen: &LutLitLenCode,
        dist: &LutDistCode,
        bits: &mut Bits<'_>,
        out_ptr: *mut u8,
        out_pos: &mut usize,
        cap: usize,
        target_end: usize,
    ) -> bool {
        let mut cur = litlen.decode(bits);
        loop {
            if *out_pos >= target_end {
                return true;
            }
            let sym = cur.symbol;
            let sym_count = cur.sym_count;
            if cur.bit_count == 0 {
                return false;
            }
            bits.consume(cur.bit_count);
            let mut s = sym;
            let mut remaining = sym_count;
            while remaining > 0 {
                let code = (s & 0xFFFF) as u16;
                if code <= 255 || remaining > 1 {
                    if remaining == 1 && code > 255 {
                        break;
                    }
                    if *out_pos >= cap {
                        return false;
                    }
                    unsafe {
                        *out_ptr.add(*out_pos) = (code & 0xFF) as u8;
                    }
                    *out_pos += 1;
                    remaining -= 1;
                    s >>= 8;
                    continue;
                }
                break;
            }
            if remaining == 1 {
                let code = (s & 0xFFFF) as u16;
                if code == END_OF_BLOCK_SYMBOL {
                    return true;
                }
                if (code as u32) > MAX_LIT_LEN_SYM {
                    return false;
                }
                let length = (code as usize).wrapping_sub(254);
                if length != 0 && !emit_one_backref(dist, bits, out_ptr, out_pos, cap, length) {
                    return false;
                }
            }
            bits.refill();
            cur = litlen.decode(bits);
        }
    }

    // ── Variant (vii): INLINE-ASM literal-run hot loop (igzip transliteration) ─
    //
    // This is the CHARTER Phase-2 prototype: an inline `core::arch::asm!` hot loop
    // that transliterates the part of igzip's AVX2 clean-decode loop LLVM provably
    // does NOT emit from idiomatic Rust (source-map plans/phase1-source-map, asm
    // igzip_decode_block_stateless.asm:507-556):
    //
    //   F1 — ONE-ITERATION-AHEAD literal-table gather hoisted ACROSS the back-edge
    //        (asm:540): the next symbol's short_code_lookup[12-bit] load is issued
    //        BEFORE the current symbol is retired, behind the loop-carried read_in
    //        dependency LLVM serializes on.
    //   F3 — UNCONDITIONAL, FLAG-FREE refill + consume via SHLX/SHRX (asm:528-531,
    //        543-547, 370): read_in is topped to 57-64 bits every iteration with NO
    //        branch on read_in_length (the IN_BUFFER_SLOP=8 invariant, asm:489).
    //   F4 — all loop-carried state PINNED in registers (read_in, read_in_length,
    //        next_in, next_out, the preloaded next_sym): no spill between symbols.
    //   C  — the 8-byte SPECULATIVE PACKED-LITERAL store + advance-by-count
    //        (asm:518-519).
    //
    // SCOPE (Phase-2 isolation): the asm handles ONLY the literal-dominant run —
    // short-code, flag-clear, last-symbol <= 255 (pure literals). On the FIRST
    // length code / long code / boundary it EXITS to the Rust careful tail
    // (decode_var_vi's exact tail logic), which resolves the back-ref, then
    // re-enters the asm loop. This isolates F1/F3/F4/C (the table-gather hoist +
    // branchless refill + packed store) WITHOUT a full-kernel asm rewrite, and is
    // byte-exact by construction: every byte the asm emits is a literal it decoded
    // identically to LutLitLenCode::decode's short path, and all len/dist/long/EOB
    // handling stays in the validated Rust path.
    //
    // Byte-exactness is the absolute gate: VAR_VII must be SHA-equal to VAR_I
    // scalar AND VAR_III ISA-L over the swept clean chunks, or the rate is VOID.
    #[cfg(target_arch = "x86_64")]
    #[allow(unused_assignments)]
    fn decode_var_vii(deflate: &[u8], start_bit: usize, window: &[u8], target_n: usize) -> Vec<u8> {
        let base = MAX_WINDOW_SIZE;
        let out_slop = OUT_SLOP + 32;
        let cap = base + target_n + out_slop + 4096;
        let mut out: Vec<u8> = vec![0u8; cap];
        out[..base].copy_from_slice(&window[..base.min(window.len())]);
        let out_ptr = out.as_mut_ptr();

        let mut block = Block::new();
        {
            let mut dummy: Vec<u16> = Vec::new();
            block
                .set_initial_window(&mut dummy, window)
                .expect("prime window (vii)");
        }

        let mut bits = Bits::at_bit_offset(deflate, start_bit);
        let in_end = deflate.len();
        let mut out_pos = base;
        let target_end = base + target_n;

        const LONG_MASK: u64 = (1u64 << ISAL_DECODE_LONG_BITS) - 1;

        'blocks: loop {
            if block.read_header(&mut bits, false).is_err() {
                break;
            }
            match block.compression_type() {
                CompressionType::Uncompressed => {
                    bits.align_to_byte();
                    let len = bits.read_u16() as usize;
                    let _nlen = bits.read_u16();
                    for _ in 0..len {
                        if bits.available() < 8 {
                            bits.refill();
                        }
                        let b = (bits.peek() & 0xFF) as u8;
                        bits.consume(8);
                        unsafe {
                            *out_ptr.add(out_pos) = b;
                        }
                        out_pos += 1;
                    }
                    if block.is_last_block() || out_pos >= target_end {
                        break 'blocks;
                    }
                    continue 'blocks;
                }
                CompressionType::FixedHuffman | CompressionType::DynamicHuffman => {}
                CompressionType::Reserved => break 'blocks,
            }

            let (litlen, dist) = match build_block_tables(&block) {
                Some(t) => t,
                None => break 'blocks,
            };
            let short_tbl = litlen.table.short_code_lookup.as_ptr();

            let mut at_eob = false;

            // ── INLINE-ASM literal-run loop ───────────────────────────────────
            // Loop-carried regs (mirror igzip's pinning, asm:108-136):
            //   read_in        = bits.bitbuf       (r9 in igzip)
            //   read_in_length = bits.bitsleft     (r10)
            //   next_in_pos    = bits.pos          (rbx, byte index)
            //   next_out       = out_pos           (r12)
            // The asm exits via `exit_code`:
            //   0 = hit a non-literal short symbol (len/EOB) — `pre_sym` holds the
            //       fresh (un-consumed) packed entry; bits already point at it.
            //   1 = hit the LONG-code flag — exit, let Rust `decode()` redo it.
            //   2 = boundary (out/in slop) — exit to careful tail.
            let mut exit_code: u64;
            let mut pre_sym: u64; // the packed short_code_lookup entry that stopped us
            {
                let mut read_in: u64 = bits.bitbuf;
                let mut read_in_length: i64 = bits.bitsleft as i64;
                let mut next_in_pos: u64 = bits.pos as u64;
                let mut next_out: u64 = out_pos as u64;
                let in_ptr = deflate.as_ptr();
                // slop-adjusted ends (igzip asm:488-489): stop the unchecked loop
                // while there is < OUT_SLOP output headroom or < IN_SLOP input.
                let mut out_limit = (cap - out_slop) as u64;
                let in_limit = (in_end - IN_SLOP) as u64;
                // DEBUG knob: force the asm fast loop to exit at the top guard so
                // the careful tail decodes EVERYTHING — isolates careful-vs-asm.
                // Self-test knob: force the asm fast loop to exit at the top guard
                // so the careful tail decodes EVERYTHING. Used to prove the careful
                // machinery (window-prefix + back-ref + dist decode) is byte-exact
                // in isolation from the asm (it is). OFF==identity.
                if std::env::var_os("GZIPPY_VII_CAREFUL_ONLY").is_some() {
                    out_limit = out_pos as u64; // next_out(==out_pos) >= out_limit -> jae 9f
                }

                let _ = LONG_MASK; // documented constant; encoded as an immediate below
                unsafe {
                    core::arch::asm!(
                        // ---- top guard (igzip asm:508-512): ONE check per iter ----
                        "2:",
                        "cmp {next_out}, {out_limit}",
                        "jae 9f",                       // out slop -> boundary exit (code 2)
                        "cmp {next_in_pos}, {in_limit}",
                        "jae 9f",
                        // ---- F3 unconditional refill — MATCHES Bits::refill (the
                        // libdeflate convention, consume_first_decode.rs:255-263):
                        //   bitbuf |= word << len ; pos += 7 - ((len>>3)&7) ;
                        //   bitsleft = len | 56   (NOTE: NOT len + bytes*8 — gzippy's
                        //   Bits keeps the real bit-count in the low byte and uses 56
                        //   as a high marker; consume does bitsleft -= n, available()
                        //   reads (bitsleft as u8)). This DIFFERS from igzip's own
                        //   read_in_length accounting; we mirror Bits, not igzip, so
                        //   the byte-exact gate vs LutLitLenCode::decode holds.
                        // {ril} holds the ORIGINAL len on entry; {cnt} is a scratch
                        // for the byte-advance (cnt is dead here, reused below).
                        "mov {tmp}, qword ptr [{in_ptr} + {next_in_pos}]",
                        "shlx {tmp}, {tmp}, {ril}",     // word << len (BMI2, no flags)
                        "or {read_in}, {tmp}",
                        "mov {cnt}, 63",
                        "sub {cnt}, {ril}",
                        "shr {cnt}, 3",                 // bytes = (63-len)/8 == 7-((len>>3)&7) for len in [0,56]
                        "add {next_in_pos}, {cnt}",
                        "or {ril}, 56",                 // bitsleft = ORIGINAL len | 56
                        // ---- table gather: idx = read_in & 0xFFF; sym=short[idx] ----
                        // (igzip asm:524-525,540; F1 = this load feeds NEXT iter too)
                        "mov {tmp}, {read_in}",
                        "and {tmp}, 0xFFF",             // LONG_MASK = (1<<12)-1
                        "mov {sym:e}, dword ptr [{short_tbl} + {tmp}*4]",
                        // ---- long-code flag? (LARGE_FLAG_BIT = 1<<25) -> exit 1 ----
                        "test {sym:e}, 0x2000000",
                        "jnz 7f",
                        // ---- bit_count (sym>>28); 0 => invalid -> exit 1 ----
                        "mov {tmp}, {sym}",
                        "shr {tmp}, 28",                // bit_count
                        "jz 7f",                        // bit_count==0 -> invalid, let Rust handle
                        // consume bit_count: read_in >>= bc (SHRX, F3); len -= bc
                        "shrx {read_in}, {read_in}, {tmp}",
                        "sub {ril}, {tmp}",
                        // ---- count = (sym>>26)&3 ; keep in {cnt} for the advance ----
                        "mov {cnt}, {sym}",
                        "shr {cnt}, 26",
                        "and {cnt}, 3",
                        // ---- last element = (sym & 0x00FF_FFFF) >> (8*(count-1)) ----
                        // CRITICAL: mask sym to the 24-bit packed field FIRST, else
                        // the high bits (bitcount/count/flag at bits 24-31) leak into
                        // the shifted result and a literal mis-tests as a length code.
                        // shift = 8*(count-1) in {tmp} (bit_count is dead, reuse {tmp}).
                        "lea {tmp}, [{cnt} - 1]",
                        "shl {tmp}, 3",                 // 8*(count-1)
                        "mov {exit_code}, {sym}",       // scratch copy (exit_code dead here)
                        "and {exit_code}, 0x1FFFFFF",   // LARGE_SHORT_SYM_MASK (25-bit packed)
                        "shrx {tmp}, {exit_code}, {tmp}", // tmp = packed >> shift
                        "and {tmp}, 0xFFFF",            // last element (matches careful tail `s & 0xFFFF`)
                        // if last > 255 => length/EOB code -> exit 0 (Rust resolves it)
                        "cmp {tmp}, 255",
                        "ja 8f",
                        // ---- C: speculative 8-byte packed-literal store (asm:518) ----
                        "mov qword ptr [{out_ptr} + {next_out}], {sym}",
                        "add {next_out}, {cnt}",        // advance by ACTUAL count
                        "jmp 2b",
                        // ---- exits (exit_code reuses {cnt}, written back via mov) ----
                        "7:",                            // long code / invalid
                        "mov {cnt}, 1",
                        "jmp 3f",
                        "8:",                            // non-literal (len/EOB) short:
                        // bit_count already consumed; packed entry is in {sym}
                        // (== pre_sym output). Rust emits leading literals + resolves
                        // the trailing length code. Do NOT re-decode the litlen.
                        "mov {cnt}, 0",
                        "jmp 3f",
                        "9:",                            // boundary
                        "mov {cnt}, 2",
                        "3:",
                        "mov {exit_code}, {cnt}",
                        in_ptr = in(reg) in_ptr,
                        short_tbl = in(reg) short_tbl,
                        out_ptr = in(reg) out_ptr,
                        out_limit = in(reg) out_limit,
                        in_limit = in(reg) in_limit,
                        read_in = inout(reg) read_in,
                        ril = inout(reg) read_in_length,
                        next_in_pos = inout(reg) next_in_pos,
                        next_out = inout(reg) next_out,
                        sym = out(reg) pre_sym,
                        tmp = out(reg) _,
                        cnt = out(reg) _,
                        exit_code = out(reg) exit_code,
                        options(nostack),
                    );
                }

                // Write the loop-carried bit state back into `bits`.
                bits.bitbuf = read_in;
                bits.bitsleft = read_in_length as u32;
                bits.pos = next_in_pos as usize;
                out_pos = next_out as usize;
            }

            // ── Resolve the asm exit in Rust, then run the careful tail ───────
            // exit_code 0: a packet whose LAST element is a length/EOB code stopped
            //   the fast loop. The asm took the `ja 8f` branch BEFORE the store, so
            //   it did NOT store any bytes and did NOT advance out_pos — but it DID
            //   consume bit_count. So here we (a) emit the `cnt-1` LEADING literals
            //   from the packed entry, then (b) resolve the trailing length code as
            //   a back-ref. (The bits are already past the litlen code.)
            // exit_code 1: long-code / invalid. The asm jumped BEFORE the shrx, so
            //   it did NOT consume — bits still point at this symbol; the careful
            //   tail re-decodes it cleanly via litlen.decode.
            // exit_code 2: boundary. Drop straight into the careful tail.
            if exit_code == 0 {
                let sym = pre_sym as u32;
                let cnt = ((sym >> LARGE_SYM_COUNT_OFFSET) & LARGE_SYM_COUNT_MASK).max(1) as usize;
                let packed = sym & LARGE_SHORT_SYM_MASK;
                // (a) emit the cnt-1 leading literals (low bytes of `packed`).
                let mut s = packed;
                for _ in 0..(cnt - 1) {
                    if out_pos >= cap {
                        return out[base..out_pos.min(target_end)].to_vec();
                    }
                    unsafe {
                        *out_ptr.add(out_pos) = (s & 0xFF) as u8;
                    }
                    out_pos += 1;
                    s >>= 8;
                }
                // (b) the last element is the length/EOB code.
                let last = (s & 0xFFFF) as u16;
                if last == END_OF_BLOCK_SYMBOL {
                    at_eob = true;
                } else if (last as u32) > MAX_LIT_LEN_SYM {
                    return out[base..out_pos.min(target_end)].to_vec();
                } else {
                    let length = (last as usize).wrapping_sub(254);
                    if length != 0
                        && !emit_one_backref(&dist, &mut bits, out_ptr, &mut out_pos, cap, length)
                    {
                        return out[base..out_pos.min(target_end)].to_vec();
                    }
                }
            }

            // Careful tail: per-symbol, bounds-checked, to the block boundary (or
            // target_end). Identical decode/back-ref logic to VAR_VI's tail.
            if !at_eob && out_pos < target_end {
                if !run_careful_tail(
                    &litlen,
                    &dist,
                    &mut bits,
                    out_ptr,
                    &mut out_pos,
                    cap,
                    target_end,
                ) {
                    return out[base..out_pos.min(target_end)].to_vec();
                }
            }

            if block.is_last_block() || out_pos >= target_end {
                break 'blocks;
            }
        }

        let end = out_pos.min(target_end);
        out[base..end].to_vec()
    }

    fn crc32(b: &[u8]) -> u32 {
        let mut h = crc32fast::Hasher::new();
        h.update(b);
        h.finalize()
    }

    fn stats(times: &[f64]) -> (f64, f64, f64) {
        // returns (min, median, sigma%)
        let mut s = times.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = s[0];
        let median = s[s.len() / 2];
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        let var = s.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / s.len() as f64;
        let sigma = var.sqrt();
        (min, median, 100.0 * sigma / mean.max(1e-12))
    }

    // Decode-variant table. Every entry has the SAME signature, so (i)/(ii)/
    // (iii) and the const-generic (iv) stacks live in one array and share the
    // interleaved timing + byte-exact gate. Order matters: index 0 = scalar
    // reference, index 2 = ISA-L oracle (both used as byte-exact denominators).
    type DecodeFn = fn(&[u8], usize, &[u8], usize) -> Vec<u8>;
    const NV: usize = 10;
    const VARIANTS: [(&str, DecodeFn); NV] = [
        ("VAR_I_scalar_u16", decode_var_i),
        ("VAR_II_E1u8_part", decode_var_ii),
        ("VAR_III_isal", decode_var_iii),
        // VAR_IV_E000 is the engine WITH NO TECHNIQUE ON (E2=E3=E4=false). It
        // is the byte-exactness anchor required by the round-2 charter: it must
        // be SHA-identical to (i) scalar AND (iii) ISA-L, proving the new
        // read_clean_e234 entry decodes byte-for-byte like the production
        // <false> path before any technique can be trusted.
        ("VAR_IV_E000", decode_var_iv::<false, false, false>),
        ("VAR_IV_E2", decode_var_iv::<true, false, false>),
        ("VAR_IV_E23", decode_var_iv::<true, true, false>),
        ("VAR_IV_E234", decode_var_iv::<true, true, true>),
        // VAR_V is the inner-Huffman-kernel LEVER: igzip packed-flat-short-code
        // table (trick #1) + the speculative software-pipelined loop (trick #2) on
        // a FLAT-u8 linear buffer (the missing piece — production stores one u8 per
        // iter with a ring modulo and never preloads). This is the variant the
        // pre-registered falsifier gates: PASS if (V)/(III isal) >= 0.85.
        ("VAR_V_specflat", decode_var_v),
        // VAR_VI = VAR_V + BMI2 BZHI/SHRX bit extraction + AVX2/SSE MOVDQU wide
        // overlap-copy back-ref. This is the variant the ceiling falsifier gates:
        // PASS (pure-Rust IS igzip-class) if (VI)/(III isal) >= 0.85.
        ("VAR_VI_specbmi2avx", decode_var_vi),
        // VAR_VII = the CHARTER Phase-2 INLINE-ASM transliteration: igzip's
        // literal-run hot loop as a `core::arch::asm!` block (one-iteration-ahead
        // table-gather hoist + unconditional flag-free SHLX/SHRX refill+consume +
        // register-pinned loop state + 8-byte packed-literal store), exiting to the
        // validated Rust careful tail on any length/long/boundary. This is the
        // variant the GO/NO-GO gate measures: GO if (VII)/(III isal) materially
        // exceeds (VI)/(III isal) toward igzip-class (~0.945 ocl_cf); PLATEAU if
        // (VII) ≈ (VI) (inline asm cannot beat LLVM here).
        ("VAR_VII_asm", decode_var_vii),
    ];

    /// Per-chunk result: median MB/s per variant (index-aligned with VARIANTS)
    /// and whether every variant was byte-exact vs scalar AND scalar vs ISA-L.
    struct ChunkResult {
        med_mbps: [f64; NV],
        exact: [bool; NV],
        all_equal: bool,
        r_iii_i: f64,
    }

    /// Run the full byte-exact gate + interleaved timing for one seed entry.
    /// Returns None when the chunk is unusable (wrong window size, not mid-
    /// stream, or decodes too little).
    fn run_chunk(deflate: &[u8], entry: &SeedEntry) -> Option<ChunkResult> {
        if entry.window.len() != MAX_WINDOW_SIZE {
            return None;
        }
        let start_bit = entry.start_bit;
        let window = &entry.window[..];
        if !(start_bit > 64 && start_bit / 8 < deflate.len()) {
            return None;
        }

        // N_actual from the scalar reference (clamps to BFINAL if early).
        let probe = decode_var_i(deflate, start_bit, window, REQUESTED_N);
        let n_actual = probe.len().min(REQUESTED_N);
        if n_actual < 64 * 1024 {
            return None;
        }

        // Decode every variant once for the byte-exact gate.
        let outs: Vec<Vec<u8>> = VARIANTS
            .iter()
            .map(|(_, f)| f(deflate, start_bit, window, n_actual))
            .collect();
        let scalar = &outs[0][..n_actual];
        let isal = &outs[2][..n_actual];
        let scalar_eq_isal = scalar == isal;
        // Length-safe exact check: variant must be >= n_actual long AND match
        // scalar over [0, n_actual) AND scalar must match ISA-L.
        let mut exact = [false; NV];
        for (k, o) in outs.iter().enumerate() {
            exact[k] = o.len() >= n_actual && &o[..n_actual] == scalar && scalar_eq_isal;
        }
        let all_equal = exact.iter().all(|&b| b);

        if !all_equal {
            eprintln!("BYTE-EXACT FAILURE chunk start_bit={start_bit}:");
            for (k, (label, _)) in VARIANTS.iter().enumerate() {
                if !exact[k] {
                    let common = outs[k].len().min(n_actual);
                    let fd = outs[k][..common]
                        .iter()
                        .zip(&scalar[..common])
                        .position(|(p, q)| p != q);
                    eprintln!(
                        "  {label} VOID len={} (n_actual={n_actual}) first_diff={:?} crc={:#010x} (scalar={:#010x})",
                        outs[k].len(),
                        fd,
                        crc32(&outs[k][..common]),
                        crc32(&scalar[..common])
                    );
                    if let Some(d) = fd {
                        let lo = d.saturating_sub(6);
                        let hi = (d + 10).min(common);
                        eprintln!("    scalar[{lo}..{hi}] = {:02x?}", &scalar[lo..hi]);
                        eprintln!("    {label}[{lo}..{hi}] = {:02x?}", &outs[k][lo..hi]);
                    }
                }
            }
        }

        // Warm-up (discarded) then interleaved best-of-N.
        for (_, f) in VARIANTS.iter() {
            let _ = f(deflate, start_bit, window, n_actual);
        }
        let mut times: [Vec<f64>; NV] = Default::default();
        for _ in 0..ITERS {
            for (k, (_, f)) in VARIANTS.iter().enumerate() {
                let s = Instant::now();
                let r = f(deflate, start_bit, window, n_actual);
                times[k].push(s.elapsed().as_secs_f64());
                std::hint::black_box(&r);
            }
        }

        let mbps = |secs: f64| (n_actual as f64) / secs / 1e6;
        let mut med_mbps = [0.0f64; NV];
        for k in 0..NV {
            let (_min, med, _sig) = stats(&times[k]);
            med_mbps[k] = mbps(med);
        }
        let r_iii_i = med_mbps[2] / med_mbps[0];

        // Per-chunk report.
        println!(
            "CHUNK start_bit={start_bit} N_bytes={n_actual} SHA_ALL_EQUAL={}",
            if all_equal { "yes" } else { "no" }
        );
        for (k, (label, _)) in VARIANTS.iter().enumerate() {
            if exact[k] {
                println!(
                    "  {:<17} MBps_med={:>6.0}  vs_i={:.3} vs_iii={:.3}",
                    label,
                    med_mbps[k],
                    med_mbps[k] / med_mbps[0],
                    med_mbps[k] / med_mbps[2]
                );
            } else {
                println!("  {:<17} VOID (byte-exact gate failed)", label);
            }
        }

        Some(ChunkResult {
            med_mbps,
            exact,
            all_equal,
            r_iii_i,
        })
    }

    pub fn run() {
        // Note the AVX2 status: under Rosetta x86-64-v2 this is false, so E2's
        // scalar word-copy fallback runs and the byte-exact gate validates IT;
        // the AVX2 path itself is only exercised (and measured) on the guest.
        eprintln!("avx2_detected={}", std::is_x86_feature_detected!("avx2"));
        let seed = load_seed();
        assert!(
            seed.len() >= 3,
            "need >=3 seed entries to sweep, got {}",
            seed.len()
        );
        let deflate = load_deflate();

        // Sweep chunks at 10/30/50/70/90% of the sorted-by-start_bit seed list.
        // run_chunk() skips entries without a 32 KiB window or too-short decode,
        // so we over-pick and keep the usable ones.
        let pct = [10usize, 30, 50, 70, 90];
        let mut results: Vec<ChunkResult> = Vec::new();
        let mut median_chunk_idx: Option<usize> = None;
        for (j, &p) in pct.iter().enumerate() {
            let idx = (seed.len().saturating_sub(1) * p) / 100;
            if let Some(r) = run_chunk(&deflate, &seed[idx]) {
                if j == 2 {
                    median_chunk_idx = Some(results.len());
                }
                results.push(r);
            }
        }
        assert!(
            !results.is_empty(),
            "no usable chunks in the sweep (all skipped)"
        );

        // Aggregate: median-of-per-chunk-medians + min/max spread per variant.
        let med_of = |vals: &mut Vec<f64>| -> f64 {
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vals[vals.len() / 2]
        };
        println!("\nAGGREGATE over {} chunk(s):", results.len());
        for (k, (label, _)) in VARIANTS.iter().enumerate() {
            // Only aggregate chunks where this variant passed the gate.
            let mut vals: Vec<f64> = results
                .iter()
                .filter(|r| r.exact[k])
                .map(|r| r.med_mbps[k])
                .collect();
            if vals.is_empty() {
                println!("  {:<17} VOID (no byte-exact chunk)", label);
                continue;
            }
            let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let med = med_of(&mut vals);
            println!(
                "  {:<17} MBps_med_of_med={:>6.0}  min={:>6.0} max={:>6.0}",
                label, med, min, max
            );
        }

        // Self-test on the MEDIAN chunk (the 50% pick), preserved from round-2:
        // (iii)/(i) should land in [2.5x, 3.6x] on the guest. Under Rosetta the
        // absolute MB/s are garbage so the ratio can drift — the guest run is
        // authoritative; we only HARD-gate byte-exactness here.
        let all_chunks_exact = results.iter().all(|r| r.all_equal);
        let sha_all = if all_chunks_exact { "yes" } else { "no" };
        let r_iii_i = median_chunk_idx
            .map(|i| results[i].r_iii_i)
            .unwrap_or(results[0].r_iii_i);
        let selftest = r_iii_i >= 2.5 && r_iii_i <= 3.6;
        println!(
            "\nSHA_ALL_EQUAL={}  SELFTEST={}  (median-chunk iii/i={:.3})",
            sha_all,
            if selftest { "PASS" } else { "FAIL" },
            r_iii_i
        );
        if !selftest {
            eprintln!(
                "SELFTEST note: (iii)/(i)={:.3} outside [2.5,3.6]. Under Rosetta the \
                 ratio can drift; the guest run is authoritative.",
                r_iii_i
            );
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    feature = "isal-compression",
    feature = "pure-rust-inflate"
))]
fn main() {
    bench::run();
}

#[cfg(not(all(
    target_arch = "x86_64",
    feature = "isal-compression",
    feature = "pure-rust-inflate"
)))]
fn main() {
    eprintln!("engine_isolation: requires x86_64 + isal-compression + pure-rust-inflate");
}
