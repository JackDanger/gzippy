//! Broad multi-oracle differential net for the pure-Rust inflate path.
//!
//! WHY (campaign): the upcoming copy-free-to-final clean-tail refactor and the
//! inner-Huffman rate work (BMI2 PEXT/BZHI, wider multi-literal, packed-u32 LUT)
//! are the riskiest correctness-sensitive rewrites in the campaign. This file
//! is the STANDING wide-coverage backstop so a subtle byte divergence in any of
//! those lands RED here, not on a user's archive.
//!
//! It differs from the existing nets as follows:
//!   - `three_oracle_diff.rs` validates the SINGLE-THREADED inflate primitive
//!     (`decompress_bytes(.., 1)`), NOT the parallel-SM chunk pipeline.
//!   - `pure_rust_inflate_corpus.rs` validates the parallel-SM pipeline but on
//!     four fixed shapes.
//!
//! This file widens the parallel-SM (production) differential to many corpus
//! shapes AND compression levels, against THREE independent oracles
//! (flate2/zlib-ng, libdeflate, zlib-ng raw FFI), with byte + CRC32 + ISIZE
//! verification on every case. Two implementations beat one; a shared bug in a
//! single oracle cannot mask a real production divergence.
//!
//! All cases force the parallel-SM (`ParallelSM`) routing pick — verified via
//! the `MARKER_PIPELINE_RUNS` counter — so the seam / fold / marker-resolution
//! machinery the refactor touches is actually exercised end-to-end.
//!
//! Run: `cargo test --release --features pure-rust-inflate diff_multi_oracle`.
//! These are sized to clear the 10 MiB parallel gate but stay laptop-cheap.

#[cfg(test)]
#[cfg(all(pure_inflate_decode, not(feature = "isal-compression")))]
mod tests {
    use std::io::{Read, Write};
    use std::sync::atomic::Ordering;

    // ── Oracles ──────────────────────────────────────────────────────────────

    /// Oracle 1: flate2 (zlib-ng). MultiGzDecoder tolerates single-member.
    fn oracle_flate2(gz: &[u8]) -> Vec<u8> {
        let mut decoder = flate2::read::MultiGzDecoder::new(gz);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).expect("flate2 oracle decode");
        out
    }

    /// Oracle 2: libdeflate one-shot FFI. Independent author/codebase.
    fn oracle_libdeflate(gz: &[u8], exact_size: usize) -> Vec<u8> {
        let mut out = vec![0u8; exact_size];
        let mut decoder = crate::backends::libdeflate::DecompressorEx::new();
        let r = decoder
            .gzip_decompress_ex(gz, &mut out)
            .expect("libdeflate oracle decode");
        out.truncate(r.output_size);
        out
    }

    /// Oracle 3: zlib-ng raw FFI via inflateInit2(31) (gzip auto-detect).
    /// A distinct entry point from flate2's wrapper — catches a flate2-wrapper
    /// bug that the zlib-ng-under-flate2 path would share.
    fn oracle_zlibng_raw(gz: &[u8], size_hint: usize) -> Vec<u8> {
        use libz_ng_sys as zng;
        use std::mem;
        use std::ptr;

        let mut strm: zng::z_stream = unsafe {
            let mut m = mem::MaybeUninit::<zng::z_stream>::uninit();
            ptr::write_bytes(m.as_mut_ptr(), 0, 1);
            m.assume_init()
        };
        let ver = unsafe { zng::zlibVersion() };
        let ret = unsafe {
            zng::inflateInit2_(&mut strm, 31, ver, mem::size_of::<zng::z_stream>() as i32)
        };
        assert_eq!(ret, zng::Z_OK, "zlib-ng inflateInit2 failed: {ret}");

        strm.next_in = gz.as_ptr() as *mut _;
        strm.avail_in = gz.len() as u32;
        let mut output = vec![0u8; size_hint.max(64 * 1024)];
        let mut out_pos = 0usize;
        loop {
            if out_pos >= output.len() {
                output.resize(output.len() * 2, 0);
            }
            strm.next_out = unsafe { output.as_mut_ptr().add(out_pos) };
            strm.avail_out = (output.len() - out_pos) as u32;
            let r = unsafe { zng::inflate(&mut strm, zng::Z_NO_FLUSH) };
            let written = (output.len() - out_pos) - strm.avail_out as usize;
            out_pos += written;
            if r == zng::Z_STREAM_END {
                break;
            }
            if r == zng::Z_OK || r == zng::Z_BUF_ERROR {
                if written == 0 && strm.avail_in == 0 {
                    break;
                }
                continue;
            }
            panic!("zlib-ng inflate error {r}");
        }
        unsafe { zng::inflateEnd(&mut strm) };
        output.truncate(out_pos);
        output
    }

    /// ISIZE (mod 2^32) from the gzip trailer — exact for our sub-4-GiB inputs.
    fn isize_from_trailer(gz: &[u8]) -> usize {
        let tail = &gz[gz.len() - 4..];
        u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]) as usize
    }

    /// CRC32 (gzip polynomial) from the gzip trailer (bytes -8..-4).
    fn crc32_from_trailer(gz: &[u8]) -> u32 {
        let t = &gz[gz.len() - 8..gz.len() - 4];
        u32::from_le_bytes([t[0], t[1], t[2], t[3]])
    }

    /// The CORE assertion: all three oracles agree, then gzippy's
    /// production parallel-SM decode is byte-identical to them, with
    /// CRC32 + ISIZE re-verified against the gzip trailer independently
    /// (the pipeline verifies internally too — this is a second check on
    /// a separately-computed CRC so a CRC-combine bug surfaces here).
    fn assert_parallel_sm_matches_all_oracles(gz: &[u8], label: &str) {
        assert!(
            gz.len() > 10 * 1024 * 1024,
            "{label}: must exceed 10 MiB compressed to route parallel-SM (got {})",
            gz.len()
        );

        let exact = isize_from_trailer(gz);
        let ref_flate2 = oracle_flate2(gz);
        let ref_libdeflate = oracle_libdeflate(gz, exact);
        let ref_zlibng = oracle_zlibng_raw(gz, exact);

        assert_eq!(
            ref_flate2, ref_libdeflate,
            "{label}: oracle disagreement flate2 vs libdeflate — fixture is suspect"
        );
        assert_eq!(
            ref_flate2, ref_zlibng,
            "{label}: oracle disagreement flate2 vs zlib-ng-raw — fixture is suspect"
        );
        let reference = ref_flate2;
        assert_eq!(
            reference.len(),
            exact,
            "{label}: ISIZE trailer ({exact}) disagrees with decoded length ({})",
            reference.len()
        );

        // Independent CRC32 over the reference bytes must match the trailer.
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&reference);
        let computed_crc = hasher.finalize();
        assert_eq!(
            computed_crc,
            crc32_from_trailer(gz),
            "{label}: independently-computed CRC32 disagrees with gzip trailer CRC \
             (oracle/fixture corruption)"
        );

        let _lock = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let before = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);

        let mut got = Vec::with_capacity(reference.len());
        crate::decompress::decompress_single_member(gz, &mut got, 4)
            .unwrap_or_else(|e| panic!("{label}: parallel-SM decode failed: {e}"));

        let after = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);
        assert!(
            after > before,
            "{label}: parallel-SM did not run (MARKER_PIPELINE_RUNS {before}->{after}); \
             routing fell through to a sequential backend — the seam/fold path was NOT exercised"
        );

        assert_eq!(
            got.len(),
            reference.len(),
            "{label}: length mismatch (got {} vs ref {})",
            got.len(),
            reference.len()
        );
        if got != reference {
            let d = got
                .iter()
                .zip(reference.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(got.len().min(reference.len()));
            panic!(
                "{label}: byte mismatch at offset {d} (got=0x{:02x} ref=0x{:02x})",
                got.get(d).copied().unwrap_or(0),
                reference.get(d).copied().unwrap_or(0)
            );
        }
    }

    // ── Corpus generators (deterministic) ────────────────────────────────────

    fn gz_at(payload: &[u8], level: u32) -> Vec<u8> {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    /// Semi-compressible (block-unique motif + PRNG tail); copied shape from
    /// `pure_rust_inflate_corpus` so the ratio clears the parallel gate while
    /// staying above the unprofitable floor.
    fn semicompressible(raw: usize, motif_frac: f64) -> Vec<u8> {
        const BLK: usize = 4096;
        let mut out = Vec::with_capacity(raw);
        let mut rng: u64 = 0x0bad_c0de_1234_5678;
        let mut block_index: u64 = 0;
        while out.len() < raw {
            block_index += 1;
            let mut motif = [0u8; 24];
            for (j, b) in motif.iter_mut().enumerate() {
                *b = (block_index.wrapping_mul(31).wrapping_add(j as u64) & 0xff) as u8;
            }
            let motif_len = ((BLK as f64) * motif_frac) as usize;
            let mut written = 0usize;
            while written < motif_len && out.len() < raw {
                let take = motif.len().min(motif_len - written);
                out.extend_from_slice(&motif[..take]);
                written += take;
            }
            let tail = BLK.saturating_sub(motif_len);
            for _ in 0..tail {
                if out.len() >= raw {
                    break;
                }
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                out.push((rng >> 24) as u8);
            }
        }
        out.truncate(raw);
        out
    }

    /// English-prose-like text: short Huffman codes (3-5 bit), heavy
    /// back-references at modest distances — the multi-literal/fast-loop
    /// shape the inner-Huffman rate work will rewrite.
    fn prose_like(raw: usize) -> Vec<u8> {
        const WORDS: &[&str] = &[
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "and",
            "then",
            "runs",
            "into",
            "forest",
            "where",
            "shadows",
            "dance",
            "between",
            "ancient",
            "trees",
            "while",
            "river",
            "flows",
            "gently",
            "toward",
            "distant",
            "mountains",
        ];
        let mut out = Vec::with_capacity(raw);
        let mut rng: u64 = 0xfeed_beef_1234_5678;
        while out.len() < raw {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let w = WORDS[(rng >> 33) as usize % WORDS.len()];
            out.extend_from_slice(w.as_bytes());
            out.push(b' ');
            if (rng >> 40).is_multiple_of(16) {
                out.extend_from_slice(b". \n");
            }
        }
        out.truncate(raw);
        out
    }

    /// Mixed entropy: ~60% PRNG / 40% short repeats → BTYPE 00/01/10 mix,
    /// many small blocks, many dynamic-header table builds.
    fn mixed(raw: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(raw);
        let mut rng: u64 = 0xfeed_face_c0de_d00d;
        while data.len() < raw {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                data.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26 + b'a' as u64) as u8;
                let repeat = ((rng >> 40) % 8 + 2) as usize;
                for _ in 0..repeat.min(raw - data.len()) {
                    data.push(byte);
                }
            }
        }
        data.truncate(raw);
        data
    }

    /// Long-distance back-reference stress that STILL routes parallel-SM.
    /// The routing gate needs BOTH compressed-size > 10 MiB AND ratio
    /// (uncompressed/compressed) >= 1.15 AND the first block NOT stored.
    ///
    /// Key constraint: a back-ref's distance must be <= 32768, so the matched
    /// content has to be < 32 KiB back INCLUDING intervening bytes. To get a
    /// genuine NEAR-max-distance back-ref, use BASE + PAD < 32768 with BASE
    /// repeated each period: the base then matches its previous occurrence at
    /// distance (BASE + PAD) ~= 30 KiB (just under the 32 KiB window edge — the
    /// distances that, post-flip, reach across the clean-tail seam into the
    /// oldest pre-flip window). The PAD of fresh PRNG bytes keeps the ratio in
    /// the ~2:1 band (clears the 1.15 floor without compressing below 10 MiB)
    /// and ensures the first block is dynamic/fixed (not stored), so routing
    /// picks ParallelSM rather than StoredParallel.
    fn max_distance_backrefs(raw: usize) -> Vec<u8> {
        const BASE: usize = 16 * 1024;
        const PAD: usize = 14 * 1024; // BASE+PAD = 30 KiB < 32768 → near-max-distance match
        let mut base = vec![0u8; BASE];
        let mut rng: u64 = 0xa5a5_5a5a_dead_c0de;
        for b in &mut base {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (rng >> 24) as u8;
        }
        let mut out = Vec::with_capacity(raw);
        while out.len() < raw {
            // Identical base each period → near-32 KiB-distance back-ref to its
            // previous occurrence (distance ~= BASE + PAD).
            let take = (raw - out.len()).min(base.len());
            out.extend_from_slice(&base[..take]);
            // PRNG literal tail (incompressible) holds the ratio + keeps the
            // first block non-stored.
            for _ in 0..PAD {
                if out.len() >= raw {
                    break;
                }
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                out.push((rng >> 24) as u8);
            }
        }
        out.truncate(raw);
        out
    }

    // ── Tests: one shape × multiple levels, all three oracles ─────────────────

    #[test]
    fn diff_semicompressible_l1() {
        let p = semicompressible(60 * 1024 * 1024, 0.30);
        assert_parallel_sm_matches_all_oracles(&gz_at(&p, 1), "semicompressible_l1");
    }

    #[test]
    fn diff_semicompressible_l6() {
        let p = semicompressible(72 * 1024 * 1024, 0.40);
        let gz = gz_at(&p, 6);
        if gz.len() <= 10 * 1024 * 1024 {
            eprintln!("diff_semicompressible_l6: compressed below gate, skipping");
            return;
        }
        assert_parallel_sm_matches_all_oracles(&gz, "semicompressible_l6");
    }

    #[test]
    fn diff_prose_like_l6() {
        // Prose compresses hard; need a large raw payload to clear 10 MiB gz.
        let p = prose_like(120 * 1024 * 1024);
        let gz = gz_at(&p, 6);
        if gz.len() <= 10 * 1024 * 1024 {
            eprintln!("diff_prose_like_l6: compressed below gate, skipping");
            return;
        }
        assert_parallel_sm_matches_all_oracles(&gz, "prose_like_l6");
    }

    #[test]
    fn diff_mixed_l1() {
        let p = mixed(24 * 1024 * 1024);
        assert_parallel_sm_matches_all_oracles(&gz_at(&p, 1), "mixed_l1");
    }

    #[test]
    fn diff_mixed_l9() {
        let p = mixed(28 * 1024 * 1024);
        let gz = gz_at(&p, 9);
        if gz.len() <= 10 * 1024 * 1024 {
            eprintln!("diff_mixed_l9: compressed below gate, skipping");
            return;
        }
        assert_parallel_sm_matches_all_oracles(&gz, "mixed_l9");
    }

    #[test]
    fn diff_max_distance_backrefs_l9() {
        // L9 maximizes match-finding → genuine near-32768-distance back-refs.
        // The literal padding keeps the ratio in the parallel-SM band (mostly
        // PRNG → ~no compression → stays well above 10 MiB and clears the
        // unprofitable floor) while the periodic base supplies the max-distance
        // back-refs that span the flip seam.
        let p = max_distance_backrefs(48 * 1024 * 1024);
        assert_parallel_sm_matches_all_oracles(&gz_at(&p, 9), "max_distance_backrefs_l9");
    }
}
