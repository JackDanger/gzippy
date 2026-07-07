//! Decompression Routing Oracle
//!
//! Verifies that every file type (BGZF, multi-member, single-member, zlib)
//! is correctly detected, routed to the right decompression path, and produces
//! byte-identical output regardless of thread count.
//!
//! Layer 0: FileOracle — creates known test files of each type
//! Layer 1: Format Detection — verify has_bgzf_markers, is_likely_multi_member
//! Layer 2: Path Correctness — each path produces valid output
//! Layer 3: Output Identity — same output regardless of path or thread count

#[cfg(test)]
mod tests {
    use crate::compress::parallel::{compress_single_member, GzipHeaderInfo};
    use crate::decompress::format::has_bgzf_markers;
    use std::io::Write;

    // =========================================================================
    // Layer 0: FileOracle — create known test files
    // =========================================================================

    struct FileOracle {
        original: Vec<u8>,
        single_member_gz: Vec<u8>,
        multi_member_gz: Vec<u8>,
        bgzf_gz: Vec<u8>,
    }

    impl FileOracle {
        fn new(size: usize) -> Self {
            let original = make_test_data(size);

            let single_member_gz = compress_single_member_gzip(&original);
            let multi_member_gz = compress_multi_member_gzip(&original);
            let bgzf_gz = compress_bgzf_gzip(&original);

            Self {
                original,
                single_member_gz,
                multi_member_gz,
                bgzf_gz,
            }
        }
    }

    fn make_test_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xcafebabe;
        let phrases: &[&[u8]] = &[
            b"the quick brown fox jumps over the lazy dog. ",
            b"pack my box with five dozen liquor jugs! ",
            b"0123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOP\n",
        ];
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 2 {
                data.push((rng >> 16) as u8);
            } else {
                let phrase = phrases[((rng >> 24) as usize) % phrases.len()];
                let remaining = size - data.len();
                data.extend_from_slice(&phrase[..remaining.min(phrase.len())]);
            }
        }
        data.truncate(size);
        data
    }

    /// Single-member gzip (what `gzip` produces).
    fn compress_single_member_gzip(data: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    /// Multi-member gzip (what `pigz` produces — concatenated gzip members).
    fn compress_multi_member_gzip(data: &[u8]) -> Vec<u8> {
        let chunk_size = 256 * 1024;
        let mut output = Vec::new();
        for chunk in data.chunks(chunk_size) {
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(chunk).unwrap();
            output.extend_from_slice(&encoder.finish().unwrap());
        }
        output
    }

    /// BGZF gzip (what gzippy produces — blocks with FEXTRA markers).
    fn compress_bgzf_gzip(data: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        let header = GzipHeaderInfo::default();
        compress_single_member(&mut output, data, 1, &header).unwrap();
        output
    }

    /// Decompress using flate2 as a reference implementation.
    fn decompress_reference(gz_data: &[u8]) -> Vec<u8> {
        use flate2::read::MultiGzDecoder;
        use std::io::Read;
        let mut decoder = MultiGzDecoder::new(gz_data);
        let mut output = Vec::new();
        decoder.read_to_end(&mut output).unwrap();
        output
    }

    // =========================================================================
    // Layer 0 tests: Oracle self-consistency
    // =========================================================================

    #[test]
    fn test_file_oracle_roundtrip() {
        let oracle = FileOracle::new(2 * 1024 * 1024);

        let from_single = decompress_reference(&oracle.single_member_gz);
        let from_multi = decompress_reference(&oracle.multi_member_gz);
        let from_bgzf = decompress_reference(&oracle.bgzf_gz);

        assert_eq!(
            from_single, oracle.original,
            "single-member roundtrip failed"
        );
        assert_eq!(from_multi, oracle.original, "multi-member roundtrip failed");
        assert_eq!(from_bgzf, oracle.original, "bgzf roundtrip failed");

        eprintln!(
            "oracle: {}B original, single={}B, multi={}B, bgzf={}B",
            oracle.original.len(),
            oracle.single_member_gz.len(),
            oracle.multi_member_gz.len(),
            oracle.bgzf_gz.len()
        );
    }

    /// Regression (multi-member `-p1` audit): a multi-member stream that reaches
    /// the non-parallel CLI entry `decompress_single_member` (used for `-p1` and
    /// for small inputs the dispatcher won't parallelize) must decode
    /// member-by-member, not error. The old guard was a `debug_assert!` —
    /// compiled out in release — so `cat a.gz a.gz | gzippy -d -p1` produced
    /// **zero bytes / a terminal error** in a release build. Small fixture +
    /// low thread counts force the non-parallel branch.
    #[test]
    fn test_multi_member_decodes_via_single_member_entry() {
        let original = make_test_data(600 * 1024); // >2 256KiB members
        let mm = compress_multi_member_gzip(&original);
        assert!(
            crate::decompress::format::is_likely_multi_member(&mm),
            "fixture must classify as multi-member"
        );
        for threads in [1usize, 2, 4] {
            let mut out = Vec::new();
            crate::decompress::decompress_single_member(&mm, &mut out, threads)
                .unwrap_or_else(|e| panic!("multi-member -p{threads} must decode, got {e:?}"));
            assert_eq!(out, original, "multi-member -p{threads} must be byte-exact");
        }
    }

    /// Regression (a VALID multi-member stream that the detector MIS-FLAGS as
    /// single-member must still decode byte-exact — never a loud error, never
    /// truncation).
    ///
    /// `is_likely_multi_member` is a perf-bounded heuristic (format.rs): to
    /// avoid magic-byte false positives inside compressed data it REJECTS a
    /// candidate member boundary whose preceding ISIZE is 0 (format.rs:89).
    /// A genuine `cat empty.gz data.gz` stream — first member empty, so the
    /// only boundary's preceding ISIZE is 0 — therefore classifies as
    /// single-member and routes to the parallel-SM path. That is the exact
    /// "valid multi-member flagged single → errors at the embedded 2nd gzip
    /// header" report. Correctness is preserved ONLY by the trailing-member
    /// resume fallback (`single_member.rs` `trailing_member_after_first` +
    /// `sm_driver::read_parallel_sm_resume_multi`): the single-member attempt
    /// fails at the member boundary, the driver confirms a real trailing
    /// member, and resumes the remaining members per-member-CRC-verified.
    ///
    /// This locks that contract end-to-end through the production entry
    /// (`decompress_single_member`) at T1 (inline pool) and T2/T4
    /// (concurrency). If detection is ever tightened so this no longer
    /// mis-detects, the precondition assert will fire — update the premise
    /// then; the byte-exact guarantee must hold either way.
    #[test]
    fn test_misdetected_multi_member_resumes_byte_exact() {
        // `cat empty.gz data.gz` — empty FIRST member forces the ISIZE==0
        // mis-detection deterministically (one boundary, preceding ISIZE 0).
        let body = make_test_data(300 * 1024);
        let mut mm = Vec::new();
        for chunk in [&b""[..], &body[..]] {
            let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            enc.write_all(chunk).unwrap();
            mm.extend_from_slice(&enc.finish().unwrap());
        }
        // Precondition: this VALID multi-member stream MIS-DETECTS as
        // single-member (the reported-bug trigger / the point of the test).
        assert!(
            !crate::decompress::format::is_likely_multi_member(&mm),
            "fixture premise: empty-first multi-member must mis-detect as \
             single-member to exercise the resume fallback"
        );
        // ...and gzip(1)/flate2 decode it fine — it is a legal stream.
        let expected = decompress_reference(&mm);
        assert_eq!(
            expected, body,
            "reference oracle: empty member ++ data == data"
        );
        for threads in [1usize, 2, 4] {
            let mut out = Vec::new();
            crate::decompress::decompress_single_member(&mm, &mut out, threads).unwrap_or_else(
                |e| {
                    panic!(
                        "mis-detected multi-member -p{threads} must resume-decode \
                         byte-exact, got error {e:?}"
                    )
                },
            );
            assert_eq!(
                out, expected,
                "mis-detected multi-member -p{threads}: WRONG BYTES \
                 (trailing-member resume fallback broken)"
            );
        }
    }

    /// Regression (P0 correctness, mmiso locate 2026-07-06): a stored-dominated
    /// DOMINANT-FIRST multi-member stream decoded at T1 must be byte-exact.
    ///
    /// Shape: a large incompressible FIRST member (>16 MiB, so its stored blocks
    /// push the second member's header past the `is_likely_multi_member` 16 MiB
    /// scan window → the stream MIS-DETECTS as single-member) whose LAST deflate
    /// block is Huffman, followed by a small trailing member.
    ///
    /// Pre-fix, T1 routed to the single-member `StoredParallel` decoder (the
    /// `num_threads > 1`-gated dominant-first detector was skipped at T1). That
    /// decoder read the WHOLE-FILE trailer (== the small last member's ISIZE),
    /// walked the stored chain across into the first member's Huffman final block
    /// (`walk_stored_chain` → `HuffmanTail{ prefix_out = member-1 size }`), and
    /// `decode_with_huffman_tail` tripped `prefix_out > expected_size` → a
    /// TERMINAL `stored output size mismatch: expected <last-ISIZE>,
    /// got <member1-size>` → EMPTY output, exit 1, on a file `gzip -dc` and
    /// rapidgzip decode fine. Verified on both arches with the mmA2_stored.gz
    /// fixture (98.6%-dominant first member).
    ///
    /// The fix: the dominant-first multi-member detection now runs at EVERY
    /// thread count (T1 → sequential multi-member path); and `StoredParallel`
    /// DECLINES (not errors) if it is ever handed this shape.
    #[test]
    fn test_dominant_first_stored_multi_member_p1_byte_exact() {
        use std::io::Write as _;
        // First member: 17 MiB pseudo-random (→ STORED blocks) + a 4 KiB
        // compressible tail (→ a Huffman FINAL block, the exact bug trigger).
        // 17 MiB > the 16 MiB is_likely_multi_member scan window.
        let prefix_len = 17 * 1024 * 1024;
        let mut m1_payload = vec![0u8; prefix_len];
        let mut state = 0x1357_9bdf_2468_ace0u64;
        for b in &mut m1_payload {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (state >> 33) as u8;
        }
        m1_payload.extend(std::iter::repeat_n(b'Z', 4096)); // compressible tail

        // Hand-build member 1's deflate: STORED blocks (bfinal=0) over the random
        // prefix, then a flate2 dynamic-Huffman FINAL block over the tail.
        let mut deflate = Vec::new();
        let block = 65535usize;
        let mut off = 0;
        while off < prefix_len {
            let end = (off + block).min(prefix_len);
            let chunk = &m1_payload[off..end];
            deflate.push(0x00); // bfinal=0, btype=00
            let len = chunk.len() as u16;
            deflate.extend_from_slice(&len.to_le_bytes());
            deflate.extend_from_slice(&(!len).to_le_bytes());
            deflate.extend_from_slice(chunk);
            off = end;
        }
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(&m1_payload[prefix_len..]).unwrap();
        deflate.extend_from_slice(&enc.finish().unwrap());

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&m1_payload);
        let m1_crc = hasher.finalize();
        let mut member1 = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
        member1.extend_from_slice(&deflate);
        member1.extend_from_slice(&m1_crc.to_le_bytes());
        member1.extend_from_slice(&(m1_payload.len() as u32).to_le_bytes());

        // Small trailing member.
        let m2_payload = b"small trailing member payload\n".to_vec();
        let mut enc2 = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc2.write_all(&m2_payload).unwrap();
        let member2 = enc2.finish().unwrap();

        let mut mm = member1;
        mm.extend_from_slice(&member2);

        // Precondition: the >16 MiB first member makes it MIS-DETECT as
        // single-member (the exact trigger — the point of the test).
        assert!(
            !crate::decompress::format::is_likely_multi_member(&mm),
            "fixture premise: >16 MiB first member must mis-detect as single-member"
        );
        // Independent oracle sanity: gzip(1)/flate2 decode it fine.
        let expected = decompress_reference(&mm);
        let mut want = m1_payload.clone();
        want.extend_from_slice(&m2_payload);
        assert_eq!(expected, want, "reference oracle sanity");

        for threads in [1usize, 2, 4] {
            let mut out = Vec::new();
            crate::decompress::decompress_single_member(&mm, &mut out, threads).unwrap_or_else(
                |e| {
                    panic!(
                        "dominant-first stored multi-member -p{threads} must decode \
                         byte-exact (P0 regression), got error {e:?}"
                    )
                },
            );
            assert_eq!(
                out, want,
                "dominant-first stored multi-member -p{threads}: WRONG BYTES / P0 regression"
            );
        }
    }

    /// Engine-swap correctness (mmiso T1, 2026-07-06): the T1 `MultiMemberSeq`
    /// path now routes each member's decode through the ParallelSM chunk kernel
    /// (per-member slice + own-trailer verify) instead of the scalar
    /// `inflate_consume_first_bits`. A member with a CORRUPTED CRC32 must be a
    /// TERMINAL error (gzip semantics), NOT silently accepted — the exact
    /// guarantee the engine-swap must preserve. Corrupt the LAST member's CRC and
    /// assert the whole decode errors.
    #[test]
    fn test_multi_member_seq_t1_corrupt_crc_errors() {
        let original = make_low_entropy_data(600 * 1024); // 256 KiB chunks → ≥2 members
        let mut mm = compress_multi_member_gzip(&original);
        assert!(mm.len() > 16, "fixture must be a real multi-member stream");
        // CRC32 of the LAST member lives at bytes [len-8 .. len-4]. Flip one bit.
        let len = mm.len();
        mm[len - 8] ^= 0xFF;
        let mut out = Vec::new();
        let r = crate::decompress::decompress_multi_member_sequential(&mm, &mut out);
        assert!(
            r.is_err(),
            "corrupt member CRC must be a terminal error, got Ok ({} bytes)",
            out.len()
        );
    }

    /// Engine-swap correctness (mmiso T1): an EMPTY interior member (ISIZE=0)
    /// contributes zero bytes and the walk must continue past it byte-exact. The
    /// old scalar loop and the new chunk-kernel walk must agree with gzip(1).
    #[test]
    fn test_multi_member_seq_t1_empty_interior_member_byte_exact() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write as _;
        let gz_of = |bytes: &[u8]| -> Vec<u8> {
            let mut enc = GzEncoder::new(Vec::new(), Compression::new(6));
            enc.write_all(bytes).unwrap();
            enc.finish().unwrap()
        };
        let d1 = make_low_entropy_data(300 * 1024);
        let d2 = make_low_entropy_data(200 * 1024);
        let mut mm = gz_of(&d1);
        mm.extend_from_slice(&gz_of(b"")); // empty INTERIOR member (ISIZE=0)
        mm.extend_from_slice(&gz_of(&d2));

        let mut want = d1.clone();
        want.extend_from_slice(&d2);
        // Independent oracle.
        assert_eq!(decompress_reference(&mm), want, "reference oracle sanity");

        let mut out = Vec::new();
        crate::decompress::decompress_multi_member_sequential(&mm, &mut out)
            .expect("empty-interior multi-member must decode");
        assert_eq!(out, want, "empty interior member: not byte-exact");
    }

    /// Engine-swap correctness (mmiso T1): a COMPRESSIBLE multi-member stream
    /// (pigz-style, many 256 KiB members) routed through the real classifier at
    /// T1 must land on `MultiMemberSeq` AND decode byte-exact through the fast
    /// per-member chunk kernel. This is the perf-target shape; the >16 MiB
    /// dominant-first (mis-detect) case is covered by
    /// `test_dominant_first_stored_multi_member_p1_byte_exact`.
    #[test]
    fn test_multi_member_seq_t1_compressible_byte_exact() {
        let original = make_low_entropy_data(3 * 1024 * 1024); // → ~12 members
        let mm = compress_multi_member_gzip(&original);
        assert!(
            crate::decompress::format::is_likely_multi_member(&mm),
            "fixture must be detected multi-member"
        );
        assert_eq!(
            crate::decompress::classify_gzip(&mm, 1),
            crate::decompress::DecodePath::MultiMemberSeq,
            "compressible multi-member at T1 must classify MultiMemberSeq"
        );
        assert_eq!(
            decompress_reference(&mm),
            original,
            "reference oracle sanity"
        );

        // Through the production single-member entry at T1.
        let mut out = Vec::new();
        crate::decompress::decompress_single_member(&mm, &mut out, 1)
            .expect("compressible multi-member T1 must decode");
        assert_eq!(
            out, original,
            "compressible multi-member T1: not byte-exact"
        );
    }

    /// Advisor review of the eager window-chain port (queuePrefetchedChunkPostProcessing,
    /// f7868ab): the consumer now eagerly publishes each prefetched chunk's end-window via
    /// `get_last_window`. Vendor `queueChunkForPostProcessing` (GzipChunkFetcher.hpp:562-570)
    /// has a FOOTER SPECIAL-CASE — it emplaces an EMPTY window when a chunk ends exactly on a
    /// member footer — which gzippy lacks. Multi-member is hard-rerouted at `classify_gzip`
    /// (mod.rs:163) BEFORE the parallel-SM path, so this is routing-protected. But if a
    /// multi-member ever leaked in, a window computed ACROSS a member footer would mix
    /// member-A tail bytes into member-B back-reference resolution → SILENT WRONG BYTES
    /// (possibly with a passing CRC). Force the worst case directly through the parallel-SM
    /// decode (bypassing routing) and assert it NEVER produces wrong bytes: it must either
    /// error cleanly or be byte-exact. Run at T1 (inline pool) and T4 (concurrency).
    #[test]
    fn test_multi_member_forced_through_parallel_sm_never_wrong_bytes() {
        let original = make_low_entropy_data(24 * 1024 * 1024);
        let mm = compress_multi_member_gzip(&original); // 256 KiB members → many footers
        assert!(
            mm.len() > 10 * 1024 * 1024,
            "fixture must clear the parallel-SM size gate (got {} bytes)",
            mm.len()
        );
        let expected = decompress_reference(&mm);
        for threads in [1usize, 4] {
            let mut out = Vec::new();
            // A clean Err is acceptable (multi-member is not this path's job; routing
            // never sends it here in production). We only forbid SILENT CORRUPTION:
            // byte-exact is the only acceptable SUCCESS — proving the absent footer
            // special-case is genuinely safe under this routing.
            if crate::decompress::parallel::single_member::decompress_parallel(
                &mm, &mut out, None, threads,
            )
            .is_ok()
            {
                assert_eq!(
                    out, expected,
                    "parallel-SM forced on multi-member T={threads}: WRONG BYTES \
                     (footer-coincident eager window publish bug — vendor GzipChunkFetcher.hpp:562-570)"
                );
            }
        }
    }

    // =========================================================================
    // Layer 1: Format Detection
    // =========================================================================

    #[test]
    fn test_detect_bgzf() {
        let oracle = FileOracle::new(512 * 1024);

        assert!(
            has_bgzf_markers(&oracle.bgzf_gz),
            "BGZF data should be detected as BGZF"
        );
        assert!(
            !has_bgzf_markers(&oracle.single_member_gz),
            "single-member should NOT be detected as BGZF"
        );
        // Multi-member from flate2 won't have BGZF markers
        assert!(
            !has_bgzf_markers(&oracle.multi_member_gz),
            "multi-member (flate2) should NOT be detected as BGZF"
        );
    }

    #[test]
    fn test_detect_multi_member() {
        let oracle = FileOracle::new(2 * 1024 * 1024);

        let multi = crate::decompress::format::is_likely_multi_member(&oracle.multi_member_gz);
        let single = crate::decompress::format::is_likely_multi_member(&oracle.single_member_gz);
        let bgzf = crate::decompress::format::is_likely_multi_member(&oracle.bgzf_gz);

        eprintln!("detection: multi={} single={} bgzf={}", multi, single, bgzf);

        assert!(multi, "multi-member should be detected as multi-member");
        assert!(
            !single,
            "single-member should NOT be detected as multi-member"
        );
    }

    // =========================================================================
    // Layer 2: Path Correctness — each decompression path produces valid output
    // =========================================================================

    #[test]
    fn test_bgzf_path_correctness() {
        let oracle = FileOracle::new(2 * 1024 * 1024);

        // T1
        let output_t1 =
            crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&oracle.bgzf_gz, 1).unwrap();
        assert_eq!(
            output_t1, oracle.original,
            "BGZF T1 output doesn't match original"
        );

        // T4
        let output_t4 =
            crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&oracle.bgzf_gz, 4).unwrap();
        assert_eq!(
            output_t4, oracle.original,
            "BGZF T4 output doesn't match original"
        );

        eprintln!("BGZF path: T1 and T4 produce identical correct output");
    }

    #[test]
    fn test_multi_member_path_correctness() {
        let oracle = FileOracle::new(2 * 1024 * 1024);

        // Parallel path
        let output_par = crate::decompress::bgzf::decompress_multi_member_parallel_to_vec(
            &oracle.multi_member_gz,
            4,
        )
        .unwrap();
        assert_eq!(
            output_par, oracle.original,
            "multi-member parallel output doesn't match original"
        );

        // Sequential path
        let mut output_seq = Vec::new();
        crate::decompress::decompress_multi_member_sequential(
            &oracle.multi_member_gz,
            &mut output_seq,
        )
        .unwrap();
        assert_eq!(
            output_seq, oracle.original,
            "multi-member sequential output doesn't match original"
        );

        eprintln!("multi-member path: parallel and sequential produce identical correct output");
    }

    #[test]
    fn test_single_member_path_correctness() {
        let oracle = FileOracle::new(2 * 1024 * 1024);

        // Sequential fallback
        let mut output = Vec::new();
        crate::decompress::decompress_single_member_pure(&oracle.single_member_gz, &mut output)
            .unwrap();
        assert_eq!(
            output, oracle.original,
            "single-member libdeflate output doesn't match original"
        );

        eprintln!("single-member path: libdeflate produces correct output");
    }

    // =========================================================================
    // Production routing: a single-member input large enough to clear the
    // 10 MiB parallel-path gate must produce correct output via
    // `decompress_single_member`. On x86_64 this exercises the parallel ISA-L
    // `inflatePrime` path wired in at v0.3.0; on arm64 it exercises libdeflate.
    // Either way, bytes must match.
    // =========================================================================

    #[test]
    fn test_single_member_routing_multithread() {
        // Use mostly-random data so it doesn't compress past the 10 MiB gate.
        // Target: ~14 MiB compressed.
        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "test input must exceed parallel-path 10 MiB gate (got {} bytes)",
            compressed.len()
        );

        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        assert_eq!(
            output, original,
            "decompress_single_member(T=4) output mismatch — production routing broken"
        );
    }

    /// DoS/OOM regression: a tiny MALFORMED gzip must be REJECTED with a terminal
    /// Err — never trigger an unbounded allocation. This fixture (a 78-byte fuzz
    /// mutant, `robust_compare.py` m8) decoded to a phantom-block runaway under the
    /// parallel-SM path: both bit readers zero-pad past end-of-input, so the
    /// decoder fabricated literals forever, growing the output buffer until the
    /// process OOM-killed (measured: 78 bytes → >1.5 GiB before SIGKILL). The
    /// input-relative output ceiling (`input_len × MAX_DEFLATE_EXPANSION`) caps
    /// the runaway: decode now errors like gzip/igzip. Runs at T=4 (parallel) and
    /// T=1; both must return Err quickly with bounded memory (no hang, no OOM).
    #[test]
    fn test_malformed_input_rejected_not_oom() {
        // robust_compare.py mutant m8: valid gzip header, corrupted deflate body,
        // ISIZE trailer = 0x44444444 (~1.14 GiB claimed). A correct decoder errors.
        const M8: &[u8] = &[
            31, 139, 8, 0, 144, 239, 59, 106, 0, 255, 237, 198, 177, 1, 0, 32, 8, 0, 32, 45, 75,
            255, 191, 184, 177, 39, 96, 34, 114, 237, 58, 183, 39, 68, 68, 68, 68, 68, 68, 68, 68,
            68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
            68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
        ];
        for threads in [1usize, 4] {
            let mut out = Vec::new();
            let r = crate::decompress::decompress_single_member(M8, &mut out, threads);
            assert!(
                r.is_err(),
                "malformed input must be rejected at T={threads} (got Ok, {} bytes)",
                out.len()
            );
            // Bounded: the ceiling for a 78-byte input is ~127 KiB; the runaway is
            // caught after at most one reserve-floor fill (~4 MiB), never the GiBs
            // it produced before the fix.
            assert!(
                out.len() < 64 * 1024 * 1024,
                "malformed decode produced {} bytes at T={threads} — ceiling not bounding output",
                out.len()
            );
        }
    }

    /// DoS/OOM regression for the MULTI-MEMBER SEQUENTIAL path (Bug A). The
    /// `decompress_multi_member_sequential` growth loop doubled its output buffer
    /// without limit on a malformed deflate body: the inflate bit reader zero-pads
    /// past end-of-input and fabricates phantom literals forever, so a tiny input
    /// drove a multi-GB allocation (DoS). The `input_len × MAX_DEFLATE_EXPANSION`
    /// ceiling now rejects the runaway with a terminal `Err` and bounded memory.
    /// This path is DISTINCT from `test_malformed_input_rejected_not_oom` (which
    /// covers the parallel-SM path); this fuzz-class input reaches the sequential
    /// decoder directly.
    #[test]
    fn test_multi_member_sequential_oom_rejected() {
        // Valid gzip header + corrupted deflate body + ISIZE trailer claiming
        // ~1.14 GiB (0x44444444). A correct decoder must error, not OOM.
        const M8: &[u8] = &[
            31, 139, 8, 0, 144, 239, 59, 106, 0, 255, 237, 198, 177, 1, 0, 32, 8, 0, 32, 45, 75,
            255, 191, 184, 177, 39, 96, 34, 114, 237, 58, 183, 39, 68, 68, 68, 68, 68, 68, 68, 68,
            68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
            68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
        ];
        let mut out = Vec::new();
        let r = crate::decompress::decompress_multi_member_sequential(M8, &mut out);
        assert!(
            r.is_err(),
            "malformed multi-member input must be rejected (got Ok, {} bytes)",
            out.len()
        );
        // Bounded: ceiling for a ~78-byte residual is ~78 KiB + slack; the runaway
        // is caught long before the GiBs it produced before the fix.
        assert!(
            out.len() < 16 * 1024 * 1024,
            "malformed multi-member decode produced {} bytes — ceiling not bounding output",
            out.len()
        );
    }

    /// Byte-exactness guard for Bug A: the OOM ceiling must NOT reject a legitimate
    /// high-ratio multi-member stream. This builds a large (16 MiB) low-entropy
    /// corpus compressed as many concatenated gzip members (pigz-style) — a real
    /// high-compression case — and asserts the sequential decoder recovers it
    /// byte-for-byte.
    #[test]
    fn test_multi_member_sequential_large_high_ratio_byte_exact() {
        // 16 MiB of highly compressible data → many members, high per-member ratio
        // (repeated phrase compresses well past 4:1). Exercises the OOM ceiling's
        // generosity: a legitimate high-ratio stream must NOT be rejected.
        let mut original = Vec::with_capacity(16 * 1024 * 1024);
        let phrase = b"the quick brown fox jumps over the lazy dog 0123456789\n";
        while original.len() < 16 * 1024 * 1024 {
            original.extend_from_slice(phrase);
        }
        original.truncate(16 * 1024 * 1024);
        let compressed = compress_multi_member_gzip(&original);
        // Sanity: it really is multi-member and high-ratio.
        assert!(
            compressed.len() * 4 < original.len(),
            "fixture must be high-ratio (got {} → {})",
            original.len(),
            compressed.len()
        );
        let mut out = Vec::new();
        let n = crate::decompress::decompress_multi_member_sequential(&compressed, &mut out)
            .expect("valid large multi-member must decode");
        assert_eq!(n as usize, original.len(), "decoded length mismatch");
        assert_eq!(out, original, "large multi-member output not byte-exact");
    }

    /// DoS/hang regression for the INFLATE outer loop (Bug B). A non-final DEFLATE
    /// block followed by end-of-input made `inflate_consume_first_bits` spin
    /// forever: the bit reader zero-pads past EOF, so `bfinal`/`btype` read as 0,
    /// `decode_stored`'s trailing-zero tolerance returns Ok with no progress, and
    /// the outer loop never terminates (observed >18 min CPU). The post-input-end
    /// block cap now errors instead of hanging. Wrapped in an explicit thread with
    /// a timeout so a regression FAILS (does not hang the suite).
    #[test]
    fn test_inflate_nonterminating_stream_errors_not_hangs() {
        use std::sync::mpsc;
        use std::time::Duration;
        // Non-final empty stored block (BFINAL=0,BTYPE=00; LEN=0,NLEN=0xFFFF), then
        // EOF — bfinal never set. Also a bare non-final stored header at EOF.
        for body in [vec![0x00u8, 0x00, 0x00, 0xFF, 0xFF], vec![0x00u8]] {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                use crate::decompress::inflate::consume_first_decode::{
                    inflate_consume_first_bits, Bits,
                };
                // Large buffer so this is NOT the WriteZero/OOM path — a true spin
                // makes zero output progress and never fills it.
                let mut out = vec![0u8; 4 * 1024 * 1024];
                let mut bits = Bits::new(&body);
                let r = inflate_consume_first_bits(&mut bits, &mut out);
                let _ = tx.send(r.is_err());
            });
            match rx.recv_timeout(Duration::from_secs(10)) {
                Ok(is_err) => assert!(is_err, "malformed non-terminating stream must Err"),
                Err(_) => panic!("inflate hung on non-terminating stream (Bug B regressed)"),
            }
        }
    }

    /// Robustness regression for the MULTI-MEMBER PARALLEL path (`bgzf.rs`
    /// `decompress_multi_member_parallel_to_vecs`). A crafted multi-member-looking
    /// input made `scan_member_boundaries_fast` emit a member whose
    /// `deflate_start > start + length - 8`, so `data[deflate_start..deflate_end]`
    /// panicked ("slice index starts at 57 but ends at 56") — a fuzz-found crash
    /// (parallel_sm_roundtrip). The bounds guard now flags the member and returns
    /// a terminal Err instead of panicking. This is the exact 66-byte input the
    /// fuzzer produced; the decoder must return (Ok or Err) without panicking.
    #[test]
    fn test_multi_member_parallel_malformed_no_panic() {
        const CRASH: &[u8] = &[
            0, 0, 0, 189, 8, 0, 0, 0, 255, 172, 255, 0, 0, 0, 189, 8, 0, 0, 0, 255, 172, 255, 8,
            189, 0, 0, 0, 31, 139, 8, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 82, 8, 189, 0, 0,
            0, 31, 139, 8, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 82,
        ];
        // Must not panic (Ok or Err both acceptable — this is malformed input).
        // Same entry the fuzz harness reaches (decompress_with_threads wraps this).
        let mut out = Vec::new();
        let _ = crate::decompress::decompress_bytes(CRASH, &mut out, 8);
    }

    /// Opt-in routing proof (deletion-trap) for the T1-MONOLITH-STREAMING native
    /// path: with `GZIPPY_STREAM_MONOLITH=1`, a single-member decode at T==1 MUST
    /// be handled by `decode_and_stream_monolith_native` (counter fires) AND be
    /// byte-exact. The streaming monolith is OPT-IN (fulcrum optgate refused the
    /// wall win as INSTRUCTION-ONLY; production T1 default stays thin-T1) — this
    /// test locks the opt-in wiring + byte-exactness. Native build only.
    #[cfg(parallel_sm)]
    #[test]
    fn test_t1_routes_through_streaming_monolith() {
        use crate::decompress::parallel::chunk_decode::MONOLITH_STREAM_NATIVE_RUNS;
        use std::sync::atomic::Ordering;

        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);
        let before = MONOLITH_STREAM_NATIVE_RUNS.load(Ordering::Relaxed);
        let mut output = Vec::new();
        // SAFETY: test-only env toggle; all T1 paths are byte-exact so concurrent
        // tests are unaffected by the routing flip.
        unsafe { std::env::set_var("GZIPPY_STREAM_MONOLITH", "1") };
        let r = crate::decompress::decompress_single_member(&compressed, &mut output, 1);
        unsafe { std::env::remove_var("GZIPPY_STREAM_MONOLITH") };
        r.unwrap();
        let after = MONOLITH_STREAM_NATIVE_RUNS.load(Ordering::Relaxed);
        assert_eq!(output, original, "T1 streaming-monolith output mismatch");
        assert!(
            after > before,
            "T1 single-member did NOT route through the streaming monolith with \
             GZIPPY_STREAM_MONOLITH=1 (counter {before} -> {after}); opt-in wiring changed"
        );
    }

    /// COALESCE correctness lock (rapidgzip parity): the clean-tail decoder now
    /// warm-decodes ACROSS deflate block boundaries within a chunk, returning to
    /// the driver only at the first pre-header EOB whose bit position reaches the
    /// chunk's `stop_hint`. The danger this test guards is *successor-stranding*:
    /// if the chunk finalizes at a parsed-header cursor (instead of the clean
    /// pre-header EOB), the next chunk would resume mid-header and diverge —
    /// most visibly on fixed-Huffman blocks (no dynamic-table preamble to resync
    /// against). silesia (gzip-9, all-dynamic) cannot surface this; this fixture
    /// is fixed-Huffman-heavy (mixed BTYPE=00/01) and is swept across thread
    /// counts so chunk boundaries land on many different block boundaries.
    /// See `resumable.rs` coalesce branch + `chunk_decode.rs` resumable_resync drain.
    #[test]
    fn test_coalesce_fixed_huffman_multithread_byte_exact() {
        let original = make_btype01_heavy_data(32 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "fixed-Huffman fixture must exceed the 10 MiB parallel gate (got {} bytes)",
            compressed.len()
        );
        for threads in [2usize, 4, 8, 16] {
            let mut output = Vec::new();
            crate::decompress::decompress_single_member(&compressed, &mut output, threads)
                .unwrap_or_else(|e| panic!("coalesce decode failed at T={threads}: {e:?}"));
            assert_eq!(
                output.len(),
                original.len(),
                "coalesce T={threads} length mismatch (successor-stranding?)"
            );
            assert_eq!(
                output, original,
                "coalesce T={threads} output mismatch — block-boundary coalescing strands the successor chunk"
            );
        }
    }

    /// Manual bisect: compare byte-exact output with vs without `GZIPPY_DRAIN_LONE`.
    /// Run: `cargo test --release --lib diagnose_lone_drain_byte_diff -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn diagnose_lone_drain_byte_diff_coalesce() {
        // Match `test_coalesce_fixed_huffman_multithread_byte_exact` shape.
        let original = make_btype01_heavy_data(32 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);
        let threads = 2usize;

        eprintln!("--- pair-drain trace ---");
        let mut good = Vec::new();
        std::env::remove_var("GZIPPY_DRAIN_LONE");
        std::env::set_var("GZIPPY_TRACE_DRAIN", "1");
        crate::decompress::decompress_single_member(&compressed, &mut good, threads)
            .expect("pair-drain decode");

        eprintln!("--- lone-drain trace ---");
        std::env::set_var("GZIPPY_DRAIN_LONE", "1");
        let mut bad = Vec::new();
        let err = crate::decompress::decompress_single_member(&compressed, &mut bad, threads);
        std::env::remove_var("GZIPPY_DRAIN_LONE");
        std::env::remove_var("GZIPPY_TRACE_DRAIN");

        if err.is_ok() && bad == good {
            eprintln!("GZIPPY_DRAIN_LONE: byte-identical to pair drain (unexpected)");
            return;
        }

        eprintln!("good_len={} bad_len={} err={err:?}", good.len(), bad.len());
        let n = good.len().min(bad.len());
        for i in 0..n {
            if good[i] != bad[i] {
                eprintln!(
                    "first byte diff at offset {i}: good={:#04x} bad={:#04x} \
                     (good context {:?}, bad context {:?})",
                    good[i],
                    bad[i],
                    &good[i.saturating_sub(8)..i.min(good.len()).saturating_add(8).min(good.len())],
                    &bad[i.saturating_sub(8)..i.min(bad.len()).saturating_add(8).min(bad.len())],
                );
                break;
            }
        }
        if bad.len() != good.len() {
            eprintln!("length delta: {}", bad.len() as i64 - good.len() as i64);
        }
        assert!(
            err.is_err() || bad != good,
            "expected lone-drain divergence for diagnosis"
        );
    }

    /// Pure incompressible bytes (PRNG), compressed into STORED deflate blocks
    /// via `Compression::none()` — what `gzip` produces on random data.
    fn make_high_entropy_data(size: usize) -> Vec<u8> {
        let mut data = vec![0u8; size];
        let mut state: u64 = 0x1234_5678_9abc_def0;
        for b in &mut data {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (state >> 33) as u8;
        }
        data
    }

    /// Force stored deflate blocks (BTYPE=00) by compressing at level 0.
    fn compress_stored_gzip(data: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::none());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    /// Stored-dominated single-member input above the 10 MiB gate must route to
    /// the NON-speculative `StoredParallel` path on x86_64 + ISA-L/pure-rust
    /// (the speculative pipeline thrashes on stored data — FULCRUM 2026-05-29).
    /// Everywhere it must decode byte-exact through the full single-member path.
    #[test]
    fn test_stored_dominated_routes_parallel_and_decodes() {
        let original = make_high_entropy_data(16 * 1024 * 1024);
        let compressed = compress_stored_gzip(&original);
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "stored fixture must exceed the 10 MiB gate (got {} bytes)",
            compressed.len()
        );
        // It is genuinely incompressible (ratio gate would bail the speculative
        // path) AND its first block is stored (the StoredParallel trigger).
        assert!(
            crate::decompress::parallel::stored_split::first_block_is_stored(&compressed),
            "level-0 gzip must start with a stored block"
        );

        let path = crate::decompress::classify_gzip(&compressed, 4);
        // `parallel_sm` is the classifier's own gate (sm_cfg::PARALLEL_SM).
        #[cfg(parallel_sm)]
        assert_eq!(
            path,
            crate::decompress::DecodePath::StoredParallel,
            "stored-dominated input on a parallel-SM build must take the \
             non-speculative parallel stored-split path (not single-thread libdeflate)"
        );
        // On non-parallel-SM builds it must NOT pick StoredParallel (no regression
        // to the routing on those platforms).
        #[cfg(not(parallel_sm))]
        assert_ne!(path, crate::decompress::DecodePath::StoredParallel);

        // Byte-exact through the full single-member dispatcher at T1/T4/T8.
        // Serialize against the other parallel-SM tests via the shared lock so
        // concurrent decodes don't race the shared process-global pipeline
        // counters those tests snapshot.
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        for t in [1usize, 4, 8] {
            let mut out = Vec::new();
            crate::decompress::decompress_single_member(&compressed, &mut out, t).unwrap();
            assert_eq!(out, original, "stored-parallel output mismatch at T={t}");
        }
    }

    /// A stream whose FIRST block is stored but which then contains Huffman
    /// blocks (mixed entropy) routes to StoredParallel by the cheap first-block
    /// heuristic, but the decoder must detect the Huffman block, decline
    /// (NotStoredDominated), and the dispatcher must still produce byte-exact
    /// output via the safe one-shot path. This is the correctness guard for the
    /// heuristic mis-fire.
    #[test]
    fn test_mixed_entropy_first_block_stored_decodes_byte_exact() {
        // Incompressible prefix (forces a stored first block) + compressible
        // suffix (forces Huffman blocks). Build as one member by concatenating
        // the raw payloads and compressing at default level: zlib emits stored
        // blocks for the random region and Huffman for the zero region.
        let mut original = make_high_entropy_data(12 * 1024 * 1024);
        original.resize(original.len() + 8 * 1024 * 1024, 0u8);
        let compressed = compress_single_member_gzip(&original);
        assert!(compressed.len() > 10 * 1024 * 1024);
        assert!(
            crate::decompress::parallel::stored_split::first_block_is_stored(&compressed),
            "random-prefix member must start with a stored block"
        );

        // Whatever path it takes, output must be byte-exact. Hold the SM test
        // lock so the T1 one-shot decode doesn't race the no-silent-fallback
        // counter snapshots (see test_stored_dominated_routes_parallel_and_decodes).
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        for t in [1usize, 4, 8] {
            let mut out = Vec::new();
            crate::decompress::decompress_single_member(&compressed, &mut out, t).unwrap();
            assert_eq!(
                out, original,
                "mixed-entropy output mismatch at T={t} — StoredParallel fallback broken"
            );
        }
    }

    /// Compressible input (Huffman first block) must NEVER route to
    /// StoredParallel — the compressible routing is unaffected by this change.
    #[test]
    fn test_compressible_input_not_routed_stored_parallel() {
        // Highly compressible: repeating phrases → dynamic Huffman first block.
        let original = make_test_data(16 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);
        assert!(
            !crate::decompress::parallel::stored_split::first_block_is_stored(&compressed),
            "compressible gzip must NOT start with a stored block"
        );
        let path = crate::decompress::classify_gzip(&compressed, 4);
        assert_ne!(
            path,
            crate::decompress::DecodePath::StoredParallel,
            "compressible input must not take the stored-split path"
        );
    }

    /// B3 proof: parallel SM byte-perfect with `--no-default-features
    /// --features pure-rust-inflate` (no isal-sys in the dependency graph).
    #[test]
    #[cfg(all(pure_inflate_decode, not(feature = "isal-compression")))]
    fn test_pure_rust_parallel_sm_e2e() {
        use std::sync::atomic::Ordering;

        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);
        assert!(compressed.len() > 10 * 1024 * 1024);

        let before = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        assert_eq!(output, original);
        let after = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);
        assert!(
            after > before,
            "pure-rust-inflate must run parallel SM (MARKER_PIPELINE_RUNS {before} -> {after})"
        );
    }

    // =========================================================================
    // GATE-A guard (2026-06-30, `perf/m1-litlen-preload`) — locks the
    // correctness precondition of the load-before-refill preload reorder in
    // `decode_huffman_fastloop_bounded_pipelined` (consume_first_decode.rs).
    //
    // The reorder (preload next litlen entry from the CURRENT bitbuf, THEN
    // refill) is byte-identical to the base ONLY when
    // `bitsleft >= LitLenTable::TABLE_BITS` at the preload site — the refill
    // preserves the low TABLE_BITS index bits only when it ORs strictly above
    // them. A `debug_assert!` at that site (the "instrumented count that must be
    // 0 on the fast path") makes any future cfg/code drift that breaks the
    // invariant PANIC in a debug/test build instead of silently corrupting
    // output.
    //
    // This test drives the PRODUCTION clean fast path (T1, window-present,
    // no-marker) through `single_member::decompress_parallel` on match-heavy
    // (low-entropy) data — every length-match iteration hits the guarded preload
    // — and asserts byte-exact roundtrip output. Two-layer coverage:
    //   * debug build: the `debug_assert!` fires precisely on any bitsleft<11
    //     preload (this test panics), and
    //   * release build (`cargo test --release`): a violated invariant corrupts
    //     the decode, so the byte-exact `assert_eq!` below fails.
    // On non-`parallel_sm` builds the pipelined kernel is not the decode path,
    // so the test is compiled out.
    // =========================================================================
    #[cfg(parallel_sm)]
    #[test]
    fn test_litlen_preload_invariant_clean_fastloop_byte_exact() {
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        // Match-heavy so the fastloop's length-match path (which contains the
        // guarded load-before-refill preload) runs many thousands of times.
        // Large enough to clear FASTLOOP_MARGIN and drive a real fastloop.
        let original = make_low_entropy_data(8 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);

        // T1 (num_threads = 1) → single clean chunk-0 decode on the pipelined
        // Rust kernel (marker_inflate flat-contig path); no u16-marker resolve.
        let mut output = Vec::new();
        crate::decompress::parallel::single_member::decompress_parallel(
            &compressed,
            &mut output,
            None,
            1,
        )
        .expect("clean T1 parallel-SM decode must succeed");

        assert_eq!(
            output, original,
            "load-before-refill preload produced wrong bytes on the clean fastloop \
             — the bitsleft>=TABLE_BITS invariant is violated (see the debug_assert \
             in decode_huffman_fastloop_bounded_pipelined)"
        );
    }

    // =========================================================================
    // Deletion-trap killer — the routing-assertion test from the v0.6 marker
    // decoder premortem.
    //
    // Every prior marker-based attempt in this codebase has been deleted
    // during cleanup because it lived outside the production CLI path. The
    // only thing that prevents the next cleanup from doing the same is a
    // test that fails when production routing *silently falls back* away
    // from the marker pipeline. Output-equivalence tests don't catch that —
    // gzippy will still produce correct output via the ISA-L sequential
    // fallback even if the marker pipeline is gone.
    //
    // `MARKER_PIPELINE_RUNS` is a process-global counter incremented on
    // every successful run of `parallel::single_member::decompress_parallel`.
    // We snapshot it around a real CLI-shaped decode and assert it moved.
    // On platforms where the parallel path is correctly gated off (arm64,
    // non-ISA-L builds) the test is a no-op.
    // =========================================================================
    #[test]
    fn test_marker_pipeline_actually_runs_on_x86_64_isal() {
        use std::sync::atomic::Ordering;

        // Serialize against any other test that calls `decompress_parallel`
        // concurrently — under `cargo test`'s default parallel execution,
        // another increment between `before` and `after` would mask a real
        // silent-fallback regression with a false-positive pass. (Copilot
        // review on PR #94.)
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "fixture must exceed 10 MiB parallel gate (got {} bytes)",
            compressed.len()
        );

        let before = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        assert_eq!(
            output, original,
            "output bytes wrong — but routing might also be broken; check the next assertion"
        );
        let after = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);

        // Only assert the counter moved on platforms where the parallel
        // marker pipeline is the intended path. Elsewhere the routing
        // correctly steers to libdeflate/zlib-ng and `after == before`.
        #[cfg(parallel_sm)]
        assert!(
            after > before,
            "MARKER_PIPELINE_RUNS did not increment ({before} -> {after}); \
             routing fell back silently. This is the failure mode that has \
             caused every prior marker-decoder to be deleted as 'dead code.' \
             Check that `decompress_single_member`'s parallel gate is \
             reachable (ISA-L available, num_threads > 1, data > 10 MiB)."
        );
        #[cfg(not(parallel_sm))]
        let _ = (before, after); // suppress unused-vars on non-target platforms
    }

    // =========================================================================
    // ONE-engine routing trap (M3, DIV-1 part 1) — paired with
    // `SEEDED_BLOCK_CHUNKS` / `SEEDED_WRAPPER_CHUNKS`.
    //
    // Pre-M3 this trap asserted `UNIFIED_INFLATE_RUNS` moved: every seeded
    // chunk's clean tail ran the SECOND engine (`StreamingInflateWrapper` →
    // `Inflate<Clean, Generic, Streaming>`). M3 reverses that contract on
    // gzippy-native: window-seeded INEXACT chunks (and chunk 0) decode on the
    // ONE `deflate::Block` engine (vendor GzipChunk.hpp:454-458), seeded via
    // `set_initial_window` → `decode_clean_into_contig`. The mark of "the
    // Block route is wired into production" is `SEEDED_BLOCK_CHUNKS` moving
    // on a real CLI-shaped decode — and NO seeded chunk silently taking the
    // wrapper arm (`SEEDED_WRAPPER_CHUNKS` unchanged; that arm exists only
    // for `GZIPPY_SEEDED_BLOCK=0`, the isal build, and the ISA-L oracle).
    //
    // M4 re-routed the until-exact paths onto the same ONE Block engine
    // (see `exact_block_engine_owns_until_exact_on_parallel_sm` below);
    // `unified::Inflate` remains compiled as the kill-switch / isal-build /
    // ISA-L-oracle arm only.
    //
    // Same lock + same fixture as `test_marker_pipeline_actually_runs...`
    // so the two traps don't race each other under parallel test
    // execution.
    // =========================================================================
    #[test]
    #[cfg(all(pure_inflate_decode, not(feature = "isal-compression")))]
    fn seeded_block_engine_runs_on_parallel_sm() {
        use std::sync::atomic::Ordering;

        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "fixture must exceed 10 MiB parallel gate (got {} bytes)",
            compressed.len()
        );

        let block_before =
            crate::decompress::parallel::chunk_decode::SEEDED_BLOCK_CHUNKS.load(Ordering::Relaxed);
        let wrapper_before = crate::decompress::parallel::chunk_decode::SEEDED_WRAPPER_CHUNKS
            .load(Ordering::Relaxed);
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        assert_eq!(output, original, "byte-perfect output");
        let block_after =
            crate::decompress::parallel::chunk_decode::SEEDED_BLOCK_CHUNKS.load(Ordering::Relaxed);
        let wrapper_after = crate::decompress::parallel::chunk_decode::SEEDED_WRAPPER_CHUNKS
            .load(Ordering::Relaxed);

        assert!(
            block_after > block_before,
            "SEEDED_BLOCK_CHUNKS did not increment ({block_before} -> {block_after}); \
             window-seeded inexact chunks must decode on the ONE `deflate::Block` \
             engine (M3, vendor GzipChunk.hpp:454-458)"
        );
        assert_eq!(
            wrapper_after, wrapper_before,
            "SEEDED_WRAPPER_CHUNKS moved ({wrapper_before} -> {wrapper_after}); a seeded \
             inexact chunk took the second engine (`StreamingInflateWrapper`) on \
             gzippy-native without the kill-switch"
        );
    }

    // =========================================================================
    // M4 (DIV-1 part 2) deletion trap: UNTIL-EXACT decodes never take the
    // wrapper engine on gzippy-native.
    //
    // Whether a given production decode schedules any until-exact chunk is
    // timing-dependent (it needs an on-demand decode with a published window
    // AND a confirmed stop boundary), so `EXACT_BLOCK_CHUNKS > 0` cannot be
    // asserted deterministically here (the exact_block_parity nets and the
    // guest-corpus counter dump prove that side). What IS deterministic: on
    // gzippy-native with the kill-switch at its default, NO until-exact chunk
    // may decode on the wrapper (`EXACT_WRAPPER_CHUNKS` unchanged — that arm
    // exists only for `GZIPPY_EXACT_BLOCK=0`, the isal build, and the ISA-L
    // oracle).
    // =========================================================================
    #[test]
    #[cfg(all(pure_inflate_decode, not(feature = "isal-compression")))]
    fn exact_block_engine_owns_until_exact_on_parallel_sm() {
        use std::sync::atomic::Ordering;

        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);

        let wrapper_before =
            crate::decompress::parallel::chunk_decode::EXACT_WRAPPER_CHUNKS.load(Ordering::Relaxed);
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        assert_eq!(output, original, "byte-perfect output");
        let wrapper_after =
            crate::decompress::parallel::chunk_decode::EXACT_WRAPPER_CHUNKS.load(Ordering::Relaxed);

        assert_eq!(
            wrapper_after, wrapper_before,
            "EXACT_WRAPPER_CHUNKS moved ({wrapper_before} -> {wrapper_after}); an until-exact \
             chunk took the second engine (`StreamingInflateWrapper`) on gzippy-native \
             without the kill-switch (M4, labeled deviation — see \
             `finish_decode_chunk_exact_block_native`)"
        );
    }

    // =========================================================================
    // m_unsplitBlocks emplacement reaches production
    //
    // Vendor `GzipChunkFetcher.hpp:393` populates `m_unsplitBlocks` inside
    // `appendSubchunksToIndexes` for every chunk that produced 2+ subchunks.
    // gzippy ports the emplace side as scaffolding for the seekable-reader
    // path (no production read site exists yet). Without a deletion-trap,
    // the emplace branch could rot silently as "dead code." This test runs
    // a single decode large enough to force multiple subchunks per chunk
    // and asserts the counter moved.
    // =========================================================================
    #[cfg(parallel_sm)]
    #[test]
    fn test_unsplit_blocks_emplaces_on_multi_subchunk_decode() {
        use std::sync::atomic::Ordering;

        // 24 MiB low-entropy fixture compresses to >10 MiB so the parallel
        // gate fires. With the default split_chunk_size of 4 MiB and ~24
        // MiB decompressed per chunk-partition, multi-subchunk chunks are
        // the common case (vendor's "chunk size > spacing" path).
        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);

        let before = crate::decompress::parallel::chunk_fetcher::UNSPLIT_BLOCKS_EMPLACED
            .load(Ordering::Relaxed);
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        assert_eq!(output, original);
        let after = crate::decompress::parallel::chunk_fetcher::UNSPLIT_BLOCKS_EMPLACED
            .load(Ordering::Relaxed);

        #[cfg(parallel_sm)]
        assert!(
            after > before,
            "UNSPLIT_BLOCKS_EMPLACED did not increment ({before} -> {after}); \
             the m_unsplitBlocks emplace branch at chunk_fetcher.rs (mirror of \
             GzipChunkFetcher.hpp:393) is unreachable. This catches the same \
             silent-fallback failure class as the marker-pipeline deletion trap."
        );
        #[cfg(not(parallel_sm))]
        let _ = (before, after);
    }

    // =========================================================================
    // Prefetch dispatch reaches the last-chunk stop hint
    //
    // Vendor BlockFetcher.hpp:533-535 accepts `file_size_in_bits` as the
    // worker's `nextOffset` for the LAST prefetch in a file. Without
    // this asymmetric lookup, gzippy's `lookup_block_offset(idx+1)`
    // returned `None` on `GetReturnCode::Failure` and the prefetch loop
    // skipped the last chunk — observed on the 221 MB / 3-partition
    // fixture (bench-2026-05-18). This trap asserts the new
    // `lookup_next_block_offset` accepts the file-size sentinel during
    // a real parallel SM decode.
    // =========================================================================
    #[cfg(parallel_sm)]
    #[test]
    fn test_prefetch_next_filesize_accept_fires() {
        use std::sync::atomic::Ordering;

        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        // Needs gzip file >= 32 MiB so `adjusted_chunk_size_bytes` keeps the
        // default 4 MiB spacing, AND compressed size < 9 × 4 MiB so the first
        // prefetch batch (indexes 1..=8) reaches `lookup_next(9)` Failure.
        let original = make_low_entropy_data(56 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);

        let before = crate::decompress::parallel::chunk_fetcher::PREFETCH_NEXT_FILESIZE_ACCEPT
            .load(Ordering::Relaxed);
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        assert_eq!(output, original);
        let after = crate::decompress::parallel::chunk_fetcher::PREFETCH_NEXT_FILESIZE_ACCEPT
            .load(Ordering::Relaxed);

        #[cfg(parallel_sm)]
        assert!(
            after > before,
            "PREFETCH_NEXT_FILESIZE_ACCEPT did not increment ({before} -> {after}); \
             the asymmetric lookup_next_block_offset at chunk_fetcher.rs (mirror of \
             vendor BlockFetcher.hpp:533-535) is unreachable. Without this asymmetry, \
             the last prefetch in any file is skipped — visible on the 3-partition \
             fixture as a 1.24-CPU serial bottleneck on a 16-core machine."
        );
        #[cfg(not(parallel_sm))]
        let _ = (before, after);
    }

    // =========================================================================
    // Performance regression guard — the CI gap that masked v0.3.0.
    //
    // The single-member parallel path must not be SLOWER than the single-thread
    // ISA-L baseline on the same input. v0.3.0–v0.5.0 had a buggy speculation
    // design that re-decoded the entire stream sequentially in phase 2; the
    // parallel path ran at ~1.75× the elapsed time of pure sequential.
    //
    // The v0.6 marker pipeline has no per-physical-core routing gate (the
    // earlier core floor was dropped).
    // On ≥4-physical-core hardware parallel comfortably beats sequential
    // (tight assertion: ratio < 1.5 catches v0.3.0-class 1.75× regression).
    // On <4 physical cores (e.g. 2-core CI runners) parallel-at-T=4 pays
    // Amdahl tax that sequential T=1 doesn't, so ratios in the 1.5–2.0×
    // range are structural — but we still assert ratio < 3.0 there so a
    // *catastrophic* regression (5×+) on small-core hardware doesn't slip
    // through (Opus advisor feedback on PR #97: removing the assertion
    // entirely lost regression protection on the most common CI class).
    // =========================================================================
    // Perf gates (1a/1b) run on neurotic via `make test-x86_64` — see
    // former plans/rust-rapidgzip.md validation gate. Ignored in CI while the
    // unported primitives (#1–#5) are still landing.
    #[test]
    #[ignore = "perf gate — run on neurotic, not GHA (former plans/rust-rapidgzip.md)"]
    #[cfg(parallel_sm)]
    fn test_single_member_parallel_not_slower_than_sequential() {
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        // Same fixture shape as the routing-correctness test above. Clears the
        // 10 MiB parallel-path gate.
        let original = make_low_entropy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip(&original);
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "fixture must exceed 10 MiB parallel gate (got {} bytes)",
            compressed.len()
        );

        // Helper: best of 3 runs, to dampen noise.
        let bench = |threads: usize| -> std::time::Duration {
            let mut best = std::time::Duration::MAX;
            for _ in 0..3 {
                let mut sink = Vec::with_capacity(original.len());
                let t = std::time::Instant::now();
                crate::decompress::decompress_single_member(&compressed, &mut sink, threads)
                    .expect("decompress");
                let elapsed = t.elapsed();
                assert_eq!(
                    sink.len(),
                    original.len(),
                    "T={threads} produced wrong byte count"
                );
                best = best.min(elapsed);
            }
            best
        };

        let seq = bench(1);
        let threads = num_cpus::get_physical().min(16);
        let par = bench(threads);

        let ratio = par.as_secs_f64() / seq.as_secs_f64().max(1e-9);
        let seq_mbps = (original.len() as f64) / seq.as_secs_f64() / 1e6;
        let par_mbps = (original.len() as f64) / par.as_secs_f64() / 1e6;
        let physical = num_cpus::get_physical();
        eprintln!(
            "single-member ({physical} physical cores, T={threads}): \
             sequential={seq_mbps:.0} MB/s  parallel={par_mbps:.0} MB/s  ratio={ratio:.2}"
        );

        // Validation Gate 1a (synthetic): parallel must beat sequential
        // by 2× on ≥4 physical cores (former plans/rust-rapidgzip.md Track A).
        let threshold = if physical >= 4 { 0.5 } else { 3.0 };
        assert!(
            ratio < threshold,
            "parallel single-member must achieve ratio < {threshold:.1} vs sequential \
             on {physical}-physical-core hardware (T={threads}): \
             par={par:?} seq={seq:?} ratio={ratio:.2}"
        );
    }

    /// Real silesia corpus — Validation Gate 1b (former plans/rust-rapidgzip.md A4).
    #[test]
    #[ignore = "perf gate — run on neurotic, not GHA (former plans/rust-rapidgzip.md)"]
    #[cfg(parallel_sm)]
    fn test_single_member_parallel_silesia() {
        use crate::tests::datasets;

        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let Some(compressed) = datasets::load_silesia_gzip() else {
            eprintln!("skip: benchmark_data/silesia-gzip.tar.gz not present");
            return;
        };
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "silesia gzip must exceed 10 MiB parallel gate (got {} bytes)",
            compressed.len()
        );

        let bench = |threads: usize| -> std::time::Duration {
            let mut best = std::time::Duration::MAX;
            for _ in 0..3 {
                let mut sink = Vec::new();
                let t = std::time::Instant::now();
                crate::decompress::decompress_single_member(&compressed, &mut sink, threads)
                    .expect("decompress");
                best = best.min(t.elapsed());
                assert!(!sink.is_empty(), "T={threads} produced empty output");
            }
            best
        };

        let threads = num_cpus::get_physical().min(16);
        let seq = bench(1);
        let par = bench(threads);
        let ratio = par.as_secs_f64() / seq.as_secs_f64().max(1e-9);
        let physical = num_cpus::get_physical();
        eprintln!(
            "silesia ({physical} physical cores, T={threads}): ratio={ratio:.2} \
             seq={seq:?} par={par:?}"
        );

        let threshold = if physical >= 4 { 0.5 } else { 3.0 };
        assert!(
            ratio < threshold,
            "silesia parallel must achieve ratio < {threshold:.1} vs sequential \
             on {physical}-core hardware (T={threads}): ratio={ratio:.2}"
        );
    }

    /// Repeated CRC gate on real silesia — catches rare (~1–2%) prefetch-path
    /// corruption at T≥2 (GZIPPY_NO_PREFETCH=1 is clean; see fname-header test).
    #[test]
    #[cfg(parallel_sm)]
    fn test_silesia_gzip_parallel_sm_crc_stress() {
        use crate::tests::datasets;

        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let Some(compressed) = datasets::load_silesia_gzip() else {
            eprintln!("skip: benchmark_data/silesia-gzip.tar.gz not present");
            return;
        };
        let threads = 8usize.min(num_cpus::get().max(2));
        for run in 0..12 {
            let mut out = Vec::new();
            crate::decompress::decompress_single_member(&compressed, &mut out, threads)
                .unwrap_or_else(|e| panic!("silesia CRC stress run {run} failed: {e:?}"));
            assert_eq!(
                out.len(),
                211_968_000,
                "silesia stress run {run}: ISIZE mismatch"
            );
            assert_eq!(
                crc32fast::hash(&out),
                0xf9ac_f2fe,
                "silesia stress run {run}: payload CRC32 mismatch"
            );
        }
    }

    /// mmap input slice (CLI file path) without fd writev.
    #[test]
    #[cfg(parallel_sm)]
    fn test_silesia_parallel_sm_mmap_slice() {
        let path = std::path::Path::new("benchmark_data/silesia-gzip.tar.gz");
        if !path.exists() {
            eprintln!("skip: silesia-gzip.tar.gz missing");
            return;
        }
        let file = std::fs::File::open(path).expect("open silesia");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap silesia") };
        let mut out = Vec::with_capacity(211_968_000);
        crate::decompress::decompress_single_member(&mmap[..], &mut out, 8)
            .expect("mmap parallel SM");
        assert_eq!(out.len(), 211_968_000);
        assert_eq!(crc32fast::hash(&out), 0xf9ac_f2fe);
    }

    /// CLI-shaped entry: mmap input + fd writev (no MARKER_PIPELINE_TEST_LOCK).
    #[test]
    #[cfg(all(parallel_sm, unix))]
    fn test_silesia_parallel_sm_mmap_fd_cli_shape() {
        use std::fs::File;

        use std::os::unix::io::AsRawFd;

        let path = std::path::Path::new("benchmark_data/silesia-gzip.tar.gz");
        if !path.exists() {
            eprintln!("skip: silesia-gzip.tar.gz missing");
            return;
        }
        let file = File::open(path).expect("open silesia");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap silesia") };
        let mut sink = Vec::new();
        let drain = tempfile::NamedTempFile::new().expect("tempfile");
        let out_fd = Some(drain.as_raw_fd());
        let nbytes =
            crate::decompress::decompress_single_member_fd(&mmap[..], &mut sink, out_fd, 8)
                .expect("mmap+fd parallel SM");
        assert_eq!(nbytes, 211_968_000);
        let got = std::fs::read(drain.path()).expect("read drain file");
        assert_eq!(got.len(), 211_968_000);
        assert_eq!(crc32fast::hash(&got), 0xf9ac_f2fe);
    }

    /// High-entropy synthetic proxy — not gate 2 (real silesia.tar.gz).
    ///
    /// PRNG-compressed-via-flate2-best is adversarial: dense literals,
    /// few back-refs, few block boundaries → speculation pathology. The
    /// 0.5× gate encodes the **production** (isal-compression) bar.
    /// **Expected RED on `--features pure-rust-inflate` until Phase B
    /// lands** (`former plans/pure-rust-perf.md`); pure-Rust inflate at ~334
    /// MB/s vs ISA-L's ~800 MB/s leaves no headroom for speculation
    /// overhead on this fixture. Use `test_single_member_parallel_silesia`
    /// (real silesia.tar.gz) as the pure-Rust Phase B gate.
    #[test]
    #[ignore = "perf gate — run on neurotic, not GHA (former plans/rust-rapidgzip.md)"]
    #[cfg(parallel_sm)]
    fn test_single_member_parallel_silesia_class_not_slower_than_sequential() {
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let mut original = vec![0u8; 24 * 1024 * 1024];
        let mut state = 0x12345678u64;
        for b in &mut original {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (state >> 32) as u8;
        }
        let mut compressed = Vec::new();
        {
            use std::io::Write;
            let mut enc =
                flate2::write::GzEncoder::new(&mut compressed, flate2::Compression::best());
            enc.write_all(&original).unwrap();
            enc.finish().unwrap();
        }
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "silesia-class fixture must exceed 10 MiB parallel gate (got {} bytes)",
            compressed.len()
        );

        let bench = |threads: usize| -> std::time::Duration {
            let mut best = std::time::Duration::MAX;
            for _ in 0..3 {
                let mut sink = Vec::with_capacity(original.len());
                let t = std::time::Instant::now();
                crate::decompress::decompress_single_member(&compressed, &mut sink, threads)
                    .expect("decompress");
                best = best.min(t.elapsed());
                assert_eq!(sink.len(), original.len());
            }
            best
        };

        let threads = num_cpus::get_physical().min(16);
        let seq = bench(1);
        let par = bench(threads);
        let ratio = par.as_secs_f64() / seq.as_secs_f64().max(1e-9);
        let physical = num_cpus::get_physical();
        eprintln!("silesia-class ({physical} physical cores, T={threads}): ratio={ratio:.2}");

        let threshold = if physical >= 4 { 0.5 } else { 3.0 };
        assert!(
            ratio < threshold,
            "silesia-class parallel must achieve ratio < {threshold:.1} vs sequential \
             on {physical}-core hardware (T={threads}): ratio={ratio:.2}"
        );
    }

    /// Deletion-trap: proves slow-path boundary search routes through the
    /// async `RawBlockFinderCoordinator`, not a silent sequential fallback.
    #[cfg(parallel_sm)]
    #[test]
    fn test_coordinator_boundary_search_runs_on_x86_64_isal() {
        use std::sync::atomic::Ordering;

        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let original = make_btype01_heavy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip_l1(&original);
        assert!(compressed.len() > 10 * 1024 * 1024);

        let before = crate::decompress::parallel::chunk_fetcher::COORDINATOR_BOUNDARY_SEARCH_RUNS
            .load(Ordering::Relaxed);
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        assert_eq!(output, original);
        let after = crate::decompress::parallel::chunk_fetcher::COORDINATOR_BOUNDARY_SEARCH_RUNS
            .load(Ordering::Relaxed);

        #[cfg(parallel_sm)]
        {
            assert!(
                after > before,
                "COORDINATOR_BOUNDARY_SEARCH_RUNS did not increment ({before} -> {after}); \
                 vendor no-window path (block finder + tryToDecode) must run on this fixture."
            );
        }
        #[cfg(not(parallel_sm))]
        let _ = (before, after);
    }

    /// 60% random / 40% short repetition — compresses to ~60% of original.
    /// Sized to clear the 10 MiB parallel-path gate without being huge.
    fn make_low_entropy_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xfeedface;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                data.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26 + b'a' as u64) as u8;
                let repeat = ((rng >> 40) % 8 + 2) as usize;
                for _ in 0..repeat.min(size - data.len()) {
                    data.push(byte);
                }
            }
        }
        data.truncate(size);
        data
    }

    /// Mixed-entropy data designed to push zlib L1 into emitting many short
    /// blocks (a mix of BTYPE=00/01/10). Specifically: 70% random bytes
    /// (uncompressible — short stored or stored-mixed blocks),
    /// 30% short repeats from a small phrase set (compresses well — short
    /// dynamic / fixed-Huffman blocks). At L1 with this entropy mix, zlib
    /// emits many small blocks rather than one giant one; the resulting
    /// compressed size is large enough (~12 MiB at 24 MiB original) to
    /// clear the 10 MiB parallel gate.
    ///
    /// This is the failure class the cross-chunk consistency correction
    /// sweep addresses: phase 1a's speculative pick lands near a block
    /// boundary, but not always *at* one — without correction the
    /// pipeline silently fell back to libdeflate.
    fn make_btype01_heavy_data(size: usize) -> Vec<u8> {
        let phrases: &[&[u8]] = &[b"abc", b"foo bar ", b"the quick brown ", b"hello ", b"xyz "];
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xb0bd1ec0de;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 100 < 70 {
                // Random byte.
                data.push((rng >> 16) as u8);
            } else {
                // Short phrase repetition.
                let phrase = phrases[(rng as usize) % phrases.len()];
                let to_take = phrase.len().min(size - data.len());
                data.extend_from_slice(&phrase[..to_take]);
            }
        }
        data.truncate(size);
        data
    }

    /// gzip-encode at level 1 (fastest / most fixed-Huffman emissions).
    fn compress_single_member_gzip_l1(data: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(1));
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    /// Companion to `test_marker_pipeline_actually_runs_on_x86_64_isal` —
    /// same shape (decompress_single_member at T=4, counter snapshot), but
    /// against a fixture engineered to maximize fixed-Huffman (BTYPE=01)
    /// block density. The rapidgzip-port path handles this via
    /// authoritative re-dispatch when the speculative start mismatches
    /// (see `chunk_fetcher::authoritative_dispatch`).
    #[test]
    fn test_marker_pipeline_runs_on_btype01_heavy_input() {
        use std::sync::atomic::Ordering;

        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let original = make_btype01_heavy_data(24 * 1024 * 1024);
        let compressed = compress_single_member_gzip_l1(&original);
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "fixture must exceed 10 MiB parallel gate (got {} bytes)",
            compressed.len()
        );

        let before_runs = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);
        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        assert_eq!(output, original, "byte-perfect output");
        let after_runs = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);

        #[cfg(parallel_sm)]
        assert!(
            after_runs > before_runs,
            "MARKER_PIPELINE_RUNS did not increment ({before_runs} -> {after_runs}) on \
             BTYPE=01-heavy fixture — parallel path silently fell back to \
             sequential libdeflate."
        );
        #[cfg(not(parallel_sm))]
        let _ = (before_runs, after_runs);
    }

    /// Mirrors benchmarks.yml `random-data` (10 MiB urandom, L9 T4 compress,
    /// parallel SM decompress). Reproduces CI's `ExactStopMissed` at the 10 MiB
    /// partition when fixed-Huffman tail blocks were skipped by BlockFinder.
    #[cfg(parallel_sm)]
    #[test]
    fn test_random_10mb_pipelined_l9_roundtrip_parallel_sm() {
        use crate::compress::pipelined::PipelinedGzEncoder;
        use std::io::Cursor;

        let mut original = vec![0u8; 10 * 1024 * 1024];
        let mut state = 0xc0ffeeu64;
        for b in &mut original {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (state >> 32) as u8;
        }

        let encoder = PipelinedGzEncoder::new(9, 4);
        let mut compressed = Vec::new();
        encoder
            .compress(Cursor::new(&original), &mut compressed)
            .unwrap();
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "fixture must exceed 10 MiB parallel gate (got {} bytes)",
            compressed.len()
        );

        let mut output = Vec::new();
        crate::decompress::decompress_single_member(&compressed, &mut output, 4).unwrap();
        assert_eq!(
            output, original,
            "byte-perfect roundtrip on CI-shaped fixture"
        );
    }

    #[cfg(parallel_sm)]
    mod fname_header_parallel_sm {
        use super::*;

        /// Wrap raw deflate bytes in a gzip frame that includes the FNAME
        /// header field (the optional filename string that `gzip(1)` always
        /// adds and that `flate2`'s default GzEncoder omits). The
        /// parallel-SM path's gzip-header parser must skip FNAME correctly
        /// to land on the deflate body at the right bit offset.
        ///
        /// Header layout (RFC 1952):
        ///   1F 8B          magic
        ///   08             CM = deflate
        ///   08             FLG with FNAME bit set
        ///   4 bytes        MTIME (zero)
        ///   00             XFL
        ///   03             OS = Unix
        ///   <name>\0       null-terminated filename
        ///   <deflate>
        ///   4 bytes        CRC32 (little-endian)
        ///   4 bytes        ISIZE (little-endian, mod 2^32)
        fn wrap_gzip_with_fname(deflate_body: &[u8], original: &[u8], filename: &str) -> Vec<u8> {
            let mut out = Vec::with_capacity(deflate_body.len() + filename.len() + 32);
            out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x08, 0, 0, 0, 0, 0, 0x03]);
            out.extend_from_slice(filename.as_bytes());
            out.push(0);
            out.extend_from_slice(deflate_body);
            let crc = crc32fast::hash(original);
            out.extend_from_slice(&crc.to_le_bytes());
            out.extend_from_slice(&(original.len() as u32).to_le_bytes());
            out
        }

        /// Raw-deflate the input via flate2 (without gzip framing).
        fn raw_deflate_level(data: &[u8], level: u32) -> Vec<u8> {
            let mut enc =
                flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
            enc.write_all(data).unwrap();
            enc.finish().unwrap()
        }

        /// Regression test for the T≥9 parallel-SM corruption on FNAME-headered
        /// gzip (neurotic 2026-05-19, `benchmark_data/silesia-large.gz`).
        ///
        /// Root cause was a race: workers pre-emptively published tail windows
        /// to `WindowMap` at the same key from different decode paths; at T≥9
        /// the later publisher clobbered the predecessor window and the last
        /// ~80 KiB decoded wrong. Fixed by publishing windows only on the
        /// consumer thread (`chunk_fetcher.rs` — no worker-side publish).
        ///
        /// gzip(1) always adds FNAME; bench fixtures from `gzippy -c` omit it,
        /// which is why CI missed this until the hand-rolled FNAME fixture.
        ///
        /// x86_64 + isal-compression only — other platforms use libdeflate for
        /// single-member and would pass without exercising parallel-SM.
        #[test]
        fn test_parallel_sm_handles_fname_header() {
            // Reproducer conditions (empirically on neurotic):
            //   - FNAME header (gzip(1) CLI output)
            //   - compressed size > 10 MiB (parallel routing gate)
            //   - T >= 9 (deep enough prefetch to hit the race before fix)
            let original = make_low_entropy_data(200 * 1024 * 1024);
            let deflate = raw_deflate_level(&original, 6);
            let fixture = wrap_gzip_with_fname(&deflate, &original, "silesia-large.bin");
            assert!(
                fixture.len() > 10 * 1024 * 1024,
                "fixture must exceed 10 MiB parallel gate (got {} bytes)",
                fixture.len()
            );

            // Use T = num_cpus (or 16, whichever is smaller — neurotic has
            // 16 physical, smaller machines might not reproduce the race).
            let threads = num_cpus::get().clamp(9, 16);

            let mut output = Vec::with_capacity(original.len());
            crate::decompress::decompress_single_member(&fixture, &mut output, threads).unwrap();

            assert_eq!(
                output.len(),
                original.len(),
                "decoded length mismatch on FNAME-headered fixture (threads={threads})"
            );
            assert_eq!(
                crc32fast::hash(&output),
                crc32fast::hash(&original),
                "decoded bytes diverge from original on FNAME-headered fixture (threads={threads})"
            );
        }
    } // fname_header_parallel_sm

    // =========================================================================
    // Optimization counter tests previously lived here (OptimizationCounters
    // snapshots for v0.6 phase-1 internals). The rapidgzip-port replaces the
    // phase-based pipeline with the chunk_fetcher prefetch loop + worker
    // pool; the relevant per-event observability is now in
    // `src/decompress/parallel/trace.rs` (GZIPPY_LOG_FILE=path).
    // =========================================================================

    // The four OptimizationCounters-based tests (test_isal_handoff_fires_*,
    // test_bootstrap_bounded_*, test_isal_produces_bulk_*, test_phase1a_*)
    // were deleted with the rapidgzip-port. Their assertions probed v0.6
    // internal counters that no longer exist; the relevant per-event
    // observability is now in `src/decompress/parallel/trace.rs`
    // (GZIPPY_LOG_FILE=path → scripts/parallel_sm_log_summary.py).

    // =========================================================================
    // Layer 3: Output Identity — same output regardless of thread count
    // =========================================================================

    #[test]
    fn test_bgzf_thread_independence() {
        let oracle = FileOracle::new(4 * 1024 * 1024);

        let outputs: Vec<Vec<u8>> = (1..=8)
            .map(|t| {
                crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&oracle.bgzf_gz, t)
                    .unwrap()
            })
            .collect();

        for (i, output) in outputs.iter().enumerate() {
            assert_eq!(
                output,
                &oracle.original,
                "BGZF T{} output differs from original",
                i + 1
            );
        }

        // Verify all outputs are identical to each other
        for i in 1..outputs.len() {
            assert_eq!(
                outputs[0],
                outputs[i],
                "BGZF T1 and T{} produce different output",
                i + 1
            );
        }

        eprintln!("BGZF: T1-T8 all produce identical output");
    }

    #[test]
    fn test_cross_format_output_identity() {
        let oracle = FileOracle::new(2 * 1024 * 1024);

        let from_bgzf =
            crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(&oracle.bgzf_gz, 4).unwrap();
        let from_multi = crate::decompress::bgzf::decompress_multi_member_parallel_to_vec(
            &oracle.multi_member_gz,
            4,
        )
        .unwrap();

        let mut from_single = Vec::new();
        crate::decompress::decompress_single_member_pure(
            &oracle.single_member_gz,
            &mut from_single,
        )
        .unwrap();

        assert_eq!(
            from_bgzf, oracle.original,
            "BGZF decompressed differs from original"
        );
        assert_eq!(
            from_multi, oracle.original,
            "multi-member decompressed differs from original"
        );
        assert_eq!(
            from_single, oracle.original,
            "single-member decompressed differs from original"
        );

        eprintln!("all three formats produce byte-identical output from the same input");
    }

    // =========================================================================
    // Layer 1b: classify_gzip routing table assertions
    //
    // These tests are the canonical check that the routing table in
    // decompression.rs::classify_gzip matches the actual file formats.
    // If routing changes, update CLAUDE.md first, then fix these tests.
    // =========================================================================

    #[test]
    fn test_classify_gzippy_parallel() {
        let oracle = FileOracle::new(512 * 1024);
        // gzippy-parallel output ("GZ" FEXTRA) → GzippyParallel regardless of threads
        assert_eq!(
            crate::decompress::classify_gzip(&oracle.bgzf_gz, 1),
            crate::decompress::DecodePath::GzippyParallel,
            "gzippy-parallel T1 should classify as GzippyParallel"
        );
        assert_eq!(
            crate::decompress::classify_gzip(&oracle.bgzf_gz, 4),
            crate::decompress::DecodePath::GzippyParallel,
            "gzippy-parallel T4 should classify as GzippyParallel"
        );
    }

    #[test]
    fn test_classify_multi_member() {
        let oracle = FileOracle::new(2 * 1024 * 1024);
        // T1 multi-member → sequential
        assert_eq!(
            crate::decompress::classify_gzip(&oracle.multi_member_gz, 1),
            crate::decompress::DecodePath::MultiMemberSeq,
            "multi-member T1 should classify as MultiMemberSeq"
        );
        // T4 multi-member → parallel
        assert_eq!(
            crate::decompress::classify_gzip(&oracle.multi_member_gz, 4),
            crate::decompress::DecodePath::MultiMemberPar,
            "multi-member T4 should classify as MultiMemberPar"
        );
    }

    #[test]
    fn test_classify_single_member() {
        use crate::decompress::{classify_gzip, DecodePath};
        let oracle = FileOracle::new(512 * 1024);
        let path = classify_gzip(&oracle.single_member_gz, 4);
        // Single-member must route to one of the pure-Rust single-member paths.
        // The C-FFI one-shot decode backends (libdeflate / zlib-ng) and ISA-L
        // decode are deleted from the decode graph.
        assert!(
            matches!(path, DecodePath::ParallelSM | DecodePath::StoredParallel),
            "single-member should classify as a single-member path, got {:?}",
            path
        );
    }
}
