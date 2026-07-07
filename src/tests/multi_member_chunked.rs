//! Tests for the `MultiMemberChunked` decode path (dominant-member multi-member
//! streams decoded with the full within-member parallel engine per member).
//!
//! Covers §6 of the MM-ParallelSM design (the subset the interim
//! member-walk implementation exercises): multi-member differentials
//! (uneven / balanced / few-large / empty-first / empty-interior /
//! trailing-garbage / mixed-GZ / big-first), routing assertions, and the
//! deletion-trap counters (`MULTI_MEMBER_PIPELINE_RUNS`,
//! `MISROUTE_REENTRY_APPLIED`).

#[cfg(test)]
mod tests {
    use crate::decompress::{classify_gzip, decompress_gzip_to_vec, DecodePath};
    use std::io::Write;

    fn gz_member(payload: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    /// gzippy's own "GZ"-FEXTRA parallel format (a pure-GZ member).
    fn gz_subfield_member(payload: &[u8], _threads: usize) -> Vec<u8> {
        use crate::compress::parallel::{compress_single_member, GzipHeaderInfo};
        let mut out = Vec::new();
        compress_single_member(&mut out, payload, 1, &GzipHeaderInfo::default()).unwrap();
        out
    }

    fn concat(members: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        for m in members {
            out.extend_from_slice(m);
        }
        out
    }

    fn semi_compressible(n: usize, seed: u64) -> Vec<u8> {
        let words: [&[u8]; 6] = [
            b"the quick brown fox ",
            b"jumps over lazy dog ",
            b"0123456789abcdef ",
            b"gzippy multi member ",
            b"deflate stream test ",
            b"lorem ipsum dolor ",
        ];
        let mut out = Vec::with_capacity(n + 32);
        let mut s = seed | 1;
        while out.len() < n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            out.extend_from_slice(words[(s >> 33) as usize % words.len()]);
        }
        out.truncate(n);
        out
    }

    fn reference(gz: &[u8]) -> Vec<u8> {
        use flate2::read::MultiGzDecoder;
        use std::io::Read;
        let mut d = MultiGzDecoder::new(gz);
        let mut out = Vec::new();
        d.read_to_end(&mut out).unwrap();
        out
    }

    /// Every T decodes byte-exact vs flate2's MultiGzDecoder.
    fn assert_bytes_all_threads(gz: &[u8], expected: &[u8], label: &str) {
        for t in [1usize, 2, 4, 8] {
            let got = decompress_gzip_to_vec(gz, t)
                .unwrap_or_else(|e| panic!("{label} T{t} decode failed: {e:?}"));
            assert_eq!(got, expected, "{label} T{t} byte mismatch");
        }
    }

    #[test]
    fn uneven_three_small_one_dominant_byte_exact() {
        // 3 small members + 1 dominant. STAGE-2d: a dominant-member distribution
        // routes to the whole-file GRID (fast_path_ok is false — the dominant
        // member would pin a single worker on the member-per-worker path), so its
        // deflate blocks spread across all workers. (Previously locked to
        // MultiMemberPar for the regressing member-walk; the grid is a NEW path.)
        let members = vec![
            gz_member(&semi_compressible(64 * 1024, 1)),
            gz_member(&semi_compressible(64 * 1024, 2)),
            gz_member(&semi_compressible(64 * 1024, 3)),
            gz_member(&semi_compressible(48 * 1024 * 1024, 4)),
        ];
        let gz = concat(&members);
        let expected = reference(&gz);
        assert!(
            crate::decompress::format::is_likely_multi_member(&gz),
            "uneven fixture must classify multi-member"
        );
        assert_eq!(
            classify_gzip(&gz, 8),
            DecodePath::MultiMemberGrid,
            "dominant-member distribution routes the whole-file grid"
        );
        assert_bytes_all_threads(&gz, &expected, "uneven");
    }

    #[test]
    fn balanced_many_members_byte_exact_and_routes_fastpath() {
        let members: Vec<Vec<u8>> = (0..40)
            .map(|i| gz_member(&semi_compressible(1024 * 1024, 100 + i)))
            .collect();
        let gz = concat(&members);
        let expected = reference(&gz);
        // 40 balanced members, no dominant → member-per-worker fast path.
        assert_eq!(
            classify_gzip(&gz, 8),
            DecodePath::MultiMemberPar,
            "balanced stream must keep the member-per-worker fast path"
        );
        assert_bytes_all_threads(&gz, &expected, "balanced");
    }

    #[test]
    fn few_large_members_byte_exact() {
        // 2 large members at T8 → fewer members than workers, so member-per-worker
        // cannot saturate the pool (fast_path_ok is false when n < t_eff). STAGE-2d
        // routes to the whole-file GRID, which splits WITHIN members across all 8
        // workers. Byte-exact. (Previously locked to MultiMemberPar.)
        let members = vec![
            gz_member(&semi_compressible(12 * 1024 * 1024, 200)),
            gz_member(&semi_compressible(12 * 1024 * 1024, 201)),
        ];
        let gz = concat(&members);
        let expected = reference(&gz);
        assert_eq!(classify_gzip(&gz, 8), DecodePath::MultiMemberGrid);
        assert_bytes_all_threads(&gz, &expected, "few_large");
    }

    #[test]
    fn empty_interior_member_byte_exact() {
        let members = vec![
            gz_member(&semi_compressible(512 * 1024, 400)),
            gz_member(b""),
            gz_member(&semi_compressible(512 * 1024, 401)),
        ];
        let gz = concat(&members);
        let expected = reference(&gz);
        assert_bytes_all_threads(&gz, &expected, "empty_interior");
    }

    #[test]
    fn empty_first_member_byte_exact() {
        // Empty first member → preceding-ISIZE==0 makes detection classify this
        // SINGLE-member; the false-single re-entry decodes it correctly.
        let members = vec![
            gz_member(b""),
            gz_member(&semi_compressible(2 * 1024 * 1024, 500)),
        ];
        let gz = concat(&members);
        let expected = reference(&gz);
        assert_bytes_all_threads(&gz, &expected, "empty_first");
    }

    #[test]
    fn trailing_garbage_after_member_boundary_clean_stops() {
        // A multi-member stream (dominant first member so it routes chunked)
        // followed by non-gzip junk at a member boundary decodes to the members'
        // bytes and clean-stops at the junk (gzip(1) semantics; design §3.2/§6.1e).
        let p1 = semi_compressible(8 * 1024 * 1024, 600);
        let p2 = semi_compressible(256 * 1024, 601);
        let clean = concat(&[gz_member(&p1), gz_member(&p2)]);
        let expected = reference(&clean);
        let mut gz = clean.clone();
        gz.extend_from_slice(b"not a gzip member, trailing junk\x00\x01\x02\x03");
        for t in [1usize, 2, 4, 8] {
            let got = decompress_gzip_to_vec(&gz, t)
                .unwrap_or_else(|e| panic!("trailing_garbage T{t}: {e:?}"));
            assert_eq!(got, expected, "trailing_garbage T{t} must stop cleanly");
        }
    }

    #[test]
    fn mixed_gz_plus_plain_routes_chunked_byte_exact() {
        // gzippy "GZ" member ++ a plain gzip member: the GZ coverage walk finds
        // member 2 lacks the subfield → route MultiMemberChunked (not the GZ
        // fast path). Bytes must round-trip.
        let gz_part = gz_subfield_member(&semi_compressible(2 * 1024 * 1024, 700), 1);
        assert!(
            crate::decompress::format::has_bgzf_markers(&gz_part),
            "GZ member must carry the subfield"
        );
        let plain = gz_member(&semi_compressible(1024 * 1024, 701));
        let gz = concat(&[gz_part.clone(), plain]);
        assert_eq!(
            classify_gzip(&gz, 8),
            DecodePath::MultiMemberChunked,
            "mixed GZ++plain must route MultiMemberChunked"
        );
        let expected = reference(&gz);
        assert_bytes_all_threads(&gz, &expected, "mixed_gz");
    }

    #[test]
    fn pure_gz_file_still_routes_gzippy_parallel() {
        // A genuine gzippy multi-block "GZ" file must NOT be dragged onto the
        // chunked path by the coverage walk (RISK-3 regression guard).
        let gz = gz_subfield_member(&semi_compressible(8 * 1024 * 1024, 800), 4);
        assert!(crate::decompress::format::has_bgzf_markers(&gz));
        assert_eq!(
            classify_gzip(&gz, 8),
            DecodePath::GzippyParallel,
            "pure-GZ file must keep the GzippyParallel route"
        );
        // And decode byte-exact through the full router.
        let expected = semi_compressible(8 * 1024 * 1024, 800);
        assert_bytes_all_threads(&gz, &expected, "pure_gz");
    }

    /// Deletion-trap: the chunked pipeline counter advances on a
    /// MultiMemberChunked decode, and MISROUTE_REENTRY_APPLIED stays 0 for a
    /// cleanly-detected multi-member stream (it fires only on false-singles).
    #[cfg(parallel_sm)]
    #[test]
    fn deletion_trap_chunked_pipeline_runs_advances() {
        use crate::decompress::parallel::single_member::{
            MARKER_PIPELINE_TEST_LOCK, MISROUTE_REENTRY_APPLIED,
        };
        use crate::decompress::parallel::sm_driver::MULTI_MEMBER_PIPELINE_RUNS;
        use std::sync::atomic::Ordering;

        let _guard = MARKER_PIPELINE_TEST_LOCK.lock().unwrap();
        // A mixed GZ ++ plain concatenation is the deterministic MultiMemberChunked
        // route; it exercises the member-walk chunked pipeline.
        let gz = concat(&[
            gz_subfield_member(&semi_compressible(2 * 1024 * 1024, 1), 1),
            gz_member(&semi_compressible(1024 * 1024, 2)),
        ]);
        assert_eq!(classify_gzip(&gz, 8), DecodePath::MultiMemberChunked);

        let runs_before = MULTI_MEMBER_PIPELINE_RUNS.load(Ordering::Relaxed);
        let reentry_before = MISROUTE_REENTRY_APPLIED.load(Ordering::Relaxed);
        let _ = decompress_gzip_to_vec(&gz, 8).unwrap();
        let runs_after = MULTI_MEMBER_PIPELINE_RUNS.load(Ordering::Relaxed);
        let reentry_after = MISROUTE_REENTRY_APPLIED.load(Ordering::Relaxed);

        assert!(
            runs_after > runs_before,
            "MULTI_MEMBER_PIPELINE_RUNS must advance on a chunked decode"
        );
        assert_eq!(
            reentry_after, reentry_before,
            "a cleanly-detected multi-member stream must NOT trip the misroute re-entry"
        );
    }

    /// Deletion-trap: a false-single (empty-first member) trips the misroute
    /// re-entry, and a genuine single-member decode does NOT.
    #[cfg(parallel_sm)]
    #[test]
    fn deletion_trap_misroute_reentry_fires_on_false_single_only() {
        use crate::decompress::parallel::single_member::{
            MARKER_PIPELINE_TEST_LOCK, MISROUTE_REENTRY_APPLIED,
        };
        use std::sync::atomic::Ordering;

        let _guard = MARKER_PIPELINE_TEST_LOCK.lock().unwrap();

        // False single: empty first member + a real trailing member.
        let false_single = concat(&[
            gz_member(b""),
            gz_member(&semi_compressible(2 * 1024 * 1024, 900)),
        ]);
        assert!(
            !crate::decompress::format::is_likely_multi_member(&false_single),
            "empty-first stream must mis-detect as single-member"
        );
        let before = MISROUTE_REENTRY_APPLIED.load(Ordering::Relaxed);
        let _ = decompress_gzip_to_vec(&false_single, 8).unwrap();
        let after = MISROUTE_REENTRY_APPLIED.load(Ordering::Relaxed);
        assert!(
            after > before,
            "false-single stream must trip MISROUTE_REENTRY_APPLIED"
        );

        // Genuine single member: counter must NOT advance.
        let single = gz_member(&semi_compressible(4 * 1024 * 1024, 901));
        assert_eq!(classify_gzip(&single, 8), DecodePath::ParallelSM);
        let before2 = MISROUTE_REENTRY_APPLIED.load(Ordering::Relaxed);
        let _ = decompress_gzip_to_vec(&single, 8).unwrap();
        let after2 = MISROUTE_REENTRY_APPLIED.load(Ordering::Relaxed);
        assert_eq!(
            after2, before2,
            "a genuine single-member decode must NOT trip the misroute re-entry"
        );
    }

    /// Routing identity: single-member corpus inputs keep their pre-existing
    /// route (the new multi-member branches are gated on multi-member detection).
    #[test]
    fn single_member_routing_unchanged() {
        for size in [4096usize, 256 * 1024, 4 * 1024 * 1024] {
            let gz = gz_member(&semi_compressible(size, size as u64));
            let path = classify_gzip(&gz, 8);
            assert!(
                matches!(path, DecodePath::ParallelSM | DecodePath::StoredParallel),
                "single-member size={size} must route single-member, got {path:?}"
            );
        }
    }
}
