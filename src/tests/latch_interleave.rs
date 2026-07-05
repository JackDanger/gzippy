//! Same-process tiny/big decode interleave (correctness seam for the
//! tiny-file system-allocator lever, task #189/#199, lever-2b).
//!
//! Lever-2b is PER-DECODE and thread-local (`rpmalloc_alloc::SystemHugeScope`,
//! RAII): a tiny (≤8 MiB output) thin-T1 decode backs its huge chunk-output
//! reserve with the system allocator; every other decode is untouched. Unlike
//! the reverted lever-2a process-global latch there is NO cross-decode state —
//! the #199 advisor's "tiny latches the process, a later big decode is stuck"
//! seam is removed by construction. This test proves that end-to-end on the
//! REAL production entry (`decompress_single_member`, T1) in BOTH orders:
//!
//!   1. tiny→big and big→tiny both produce byte-exact output;
//!   2. the tiny decode really is system-backed (SYS_HUGE_ALLOCS moved — the
//!      Gate-0 non-inert proof);
//!   3. the big decode is NOT system-backed (counter did not move — the scope
//!      cannot leak past its RAII guard), in both orders;
//!   4. provenance is pointer-keyed: blocks allocated under the scope are freed
//!      through the system backend even though the free happens after the scope
//!      ended (exercised implicitly — a cross-backend free aborts the process).
//!
//! Serialized on `MARKER_PIPELINE_TEST_LOCK` (shared with the other decode
//! tests) because SYS_HUGE_ALLOCS is a process-global counter.

#[cfg(test)]
mod tests {
    use std::io::Write;

    fn gz(data: &[u8]) -> Vec<u8> {
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    /// Deterministic, moderately-compressible bytes (so a >8 MiB output stays a
    /// small .gz and the decode is quick).
    fn make_data(size: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(size);
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let phrases: &[&[u8]] = &[
            b"the quick brown fox jumps over the lazy dog. ",
            b"pack my box with five dozen liquor jugs! ",
            b"lorem ipsum dolor sit amet consectetur adipiscing elit. ",
        ];
        while v.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 33) % 5 < 1 {
                v.push((rng >> 24) as u8);
            } else {
                v.extend_from_slice(phrases[((rng >> 40) as usize) % phrases.len()]);
            }
        }
        v.truncate(size);
        v
    }

    fn decode(gz_bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        crate::decompress::decompress_single_member(gz_bytes, &mut out, 1)
            .expect("decompress_single_member(T=1) failed");
        out
    }

    fn sys_huge_allocs() -> u64 {
        crate::decompress::parallel::rpmalloc_alloc::SYS_HUGE_ALLOCS
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[ignore = "reads the process-global SYS_HUGE_ALLOCS counter (delta asserts); \
                a concurrent tiny T1 decode in another test could bump it. \
                Run serially: cargo test --release latch_interleave -- --ignored --test-threads=1"]
    #[test]
    fn tiny_big_interleave_same_process_byte_exact_no_cross_decode_state() {
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let tiny_raw = make_data(96 * 1024); // <= 8 MiB output ⇒ system-backed scope
        let big_raw = make_data(12 * 1024 * 1024); // > 8 MiB output ⇒ rpmalloc, no scope
        let tiny_gz = gz(&tiny_raw);
        let big_gz = gz(&big_raw);

        // The T1 manual buffer pool retains the (64 MiB-reserved) output buffer
        // across same-process decodes; a pool HIT allocates nothing at all, so
        // the Sys counter only moves on a pool MISS. Drain before each
        // "fresh tiny decode" assertion so the miss (the CLI fresh-process
        // shape, which the lever targets) is what we exercise.
        use crate::decompress::parallel::chunk_buffer_pool::drain_pools_for_test;

        // ── Order A: tiny first, then big ─────────────────────────────────
        drain_pools_for_test();
        let c0 = sys_huge_allocs();
        let tiny_out = decode(&tiny_gz);
        assert_eq!(tiny_out, tiny_raw, "tiny decode not byte-exact (order A)");
        let c1 = sys_huge_allocs();
        #[cfg(feature = "arena-allocator")]
        assert!(
            c1 > c0,
            "Gate-0 non-inert proof failed: tiny thin-T1 decode did not serve any \
             system-backed huge allocation (scope inert?) c0={c0} c1={c1}"
        );

        // Drain so the big decode takes a FRESH huge allocation (the sharpest
        // leak check: a leaked scope would make this fresh alloc system-backed).
        // The drain itself drops the tiny decode's pooled Sys-backed buffer —
        // exercising the pointer-keyed cross-scope free (Sys block freed via
        // Global long after its scope ended; a cross-backend free would abort).
        drain_pools_for_test();
        let c1b = sys_huge_allocs();
        let big_out = decode(&big_gz);
        assert_eq!(
            big_out, big_raw,
            "big decode not byte-exact after a tiny decode (order A)"
        );
        let c2 = sys_huge_allocs();
        assert_eq!(
            c2, c1b,
            "scope leak: a big (>8 MiB) decode served system-backed huge allocations \
             after a tiny decode ran in the same process"
        );

        // ── Order B: big first, then tiny ─────────────────────────────────
        drain_pools_for_test();
        let c3 = sys_huge_allocs();
        let big_out2 = decode(&big_gz);
        assert_eq!(big_out2, big_raw, "big decode not byte-exact (order B)");
        let c4 = sys_huge_allocs();
        assert_eq!(
            c4, c3,
            "scope leak: a big-first decode served system-backed huge allocations"
        );

        // Fresh tiny decode after a big one: the per-decode scope must re-arm
        // (this is exactly the state 2a's process-global latch got WRONG-way:
        // it would have latched rpmalloc forever after a big-first decode).
        drain_pools_for_test();
        let tiny_out2 = decode(&tiny_gz);
        assert_eq!(
            tiny_out2, tiny_raw,
            "tiny decode not byte-exact after a big decode (order B)"
        );
        let c5 = sys_huge_allocs();
        #[cfg(feature = "arena-allocator")]
        assert!(
            c5 > c4,
            "per-decode scope failed order B: fresh tiny decode after a big decode did \
             not serve a system-backed huge allocation (cross-decode state crept back?)"
        );

        // And WITHOUT a drain: a tiny decode reusing the pooled buffer allocates
        // nothing (counter still) and stays byte-exact — the pool-hit path.
        let c6 = sys_huge_allocs();
        let tiny_out3 = decode(&tiny_gz);
        assert_eq!(tiny_out3, tiny_raw, "pool-hit tiny decode not byte-exact");
        assert_eq!(
            sys_huge_allocs(),
            c6,
            "pool-hit tiny decode should reuse the pooled buffer, not allocate"
        );

        // Cross-order byte identity (no state-dependent output).
        assert_eq!(tiny_out, tiny_out2, "tiny output differs between orders");
        assert_eq!(big_out, big_out2, "big output differs between orders");
        let _ = (c0, c1, c1b, c2, c3, c4, c5, c6); // silence unused on non-arena builds
    }
}
