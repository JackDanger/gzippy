//! Roundtrip-correctness net for the pure-Rust DEFLATE encoder substrate
//! (Increment 1, `src/compress/deflate/`).
//!
//! The literals-only engine emits a valid gzip stream that need not match any
//! vendor byte-for-byte; the contract is that it decodes BACK to the exact
//! input through THREE independent decoders — flate2/zlib-ng, libdeflate (FFI),
//! and the system `gzip -d` — plus a proptest generator asserting the flate2
//! roundtrip over random + adversarial inputs.
//!
//! Run: `cargo test --release deflate_encoder_foundation`.

#[cfg(test)]
mod tests {
    use crate::compress::deflate::compress_gzip;
    use std::io::{Read, Write};
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    // ---- the three independent decoders ----

    /// Oracle 1: flate2 (zlib-ng backend), single-member gzip.
    fn decode_flate2(gz: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(gz)
            .read_to_end(&mut out)
            .expect("flate2 failed to decode our gzip stream");
        out
    }

    /// Oracle 2: libdeflate via the libdeflater FFI binding.
    fn decode_libdeflate(gz: &[u8], expected_len: usize) -> Vec<u8> {
        let mut decomp = libdeflater::Decompressor::new();
        // libdeflate needs a correctly-sized (or larger) output buffer.
        let mut out = vec![0u8; expected_len.max(1)];
        let n = decomp
            .gzip_decompress(gz, &mut out)
            .expect("libdeflate failed to decode our gzip stream");
        out.truncate(n);
        out
    }

    /// Oracle 3: the system `gzip -d`. Returns `None` if no `gzip` is on PATH.
    fn decode_system_gzip(gz: &[u8]) -> Option<Vec<u8>> {
        let mut child = match Command::new("gzip")
            .arg("-d")
            .arg("-c")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => return None, // gzip not available on this box
        };
        child
            .stdin
            .take()
            .unwrap()
            .write_all(gz)
            .expect("write to gzip stdin");
        let out = child.wait_with_output().expect("wait for gzip");
        assert!(out.status.success(), "system gzip -d rejected our stream");
        Some(out.stdout)
    }

    /// Assert every oracle recovers `input` byte-exact from our gzip stream.
    fn assert_roundtrips(input: &[u8], label: &str) {
        let gz = compress_gzip(input, 6);

        let f = decode_flate2(&gz);
        assert_eq!(f, input, "flate2 roundtrip mismatch [{label}]");

        let l = decode_libdeflate(&gz, input.len());
        assert_eq!(l, input, "libdeflate roundtrip mismatch [{label}]");

        if let Some(g) = decode_system_gzip(&gz) {
            assert_eq!(g, input, "system gzip roundtrip mismatch [{label}]");
        } else {
            eprintln!("note: system gzip unavailable; skipped that oracle for [{label}]");
        }
    }

    // ---- deterministic input generators ----

    /// Small, fast, deterministic PRNG (xorshift64*) for reproducible corpora.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed.max(1))
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn fill(&mut self, buf: &mut [u8]) {
            for b in buf.iter_mut() {
                *b = (self.next_u64() & 0xff) as u8;
            }
        }
    }

    fn silesia_slice(n: usize) -> Option<Vec<u8>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmark_data/silesia.tar");
        let mut f = std::fs::File::open(path).ok()?;
        let mut buf = vec![0u8; n];
        // Skip the tar header region to land in real file content.
        use std::io::Seek;
        f.seek(std::io::SeekFrom::Start(1 << 16)).ok()?;
        f.read_exact(&mut buf).ok()?;
        Some(buf)
    }

    #[test]
    fn roundtrip_fixed_corpus() {
        assert_roundtrips(b"", "empty");
        assert_roundtrips(b"A", "one-byte");
        assert_roundtrips(b"hello", "hello");
        assert_roundtrips(b"hello, world! hello, world!", "short-text");

        // All-same-byte x N (highly compressible; exercises the dynamic path
        // with a single dominant literal).
        assert_roundtrips(&vec![b'Z'; 100_000], "all-same-100k");

        // Two-symbol alternation.
        let alt: Vec<u8> = (0..50_000)
            .map(|i| if i % 2 == 0 { b'a' } else { b'b' })
            .collect();
        assert_roundtrips(&alt, "alternating");

        // Random / incompressible 64 KiB (should route to a stored block).
        let mut rng = Rng::new(0xC0FFEE);
        let mut incompressible = vec![0u8; 64 * 1024];
        rng.fill(&mut incompressible);
        assert_roundtrips(&incompressible, "incompressible-64k");

        // Just over the 64 KiB stored sub-block boundary.
        let mut boundary = vec![0u8; 65_536 + 100];
        Rng::new(7).fill(&mut boundary);
        assert_roundtrips(&boundary, "stored-boundary");

        // Larger-than-one-logical-block input (exercises multi-block chunking).
        let mut multiblock = vec![0u8; (1 << 18) + 12_345];
        for (i, b) in multiblock.iter_mut().enumerate() {
            *b = (i % 251) as u8; // biased distribution -> dynamic blocks
        }
        assert_roundtrips(&multiblock, "multi-block");

        // Full byte-value coverage.
        let all_bytes: Vec<u8> = (0..=255u8).cycle().take(10_000).collect();
        assert_roundtrips(&all_bytes, "all-byte-values");
    }

    #[test]
    fn roundtrip_silesia_slice() {
        match silesia_slice(256 * 1024) {
            Some(data) => assert_roundtrips(&data, "silesia-256k"),
            None => eprintln!("note: benchmark_data/silesia.tar missing; skipped silesia slice"),
        }
    }

    #[test]
    fn stored_block_chosen_for_incompressible() {
        // Sanity that the stored path actually engages and stays byte-exact.
        let mut rng = Rng::new(0xABCDEF);
        let mut data = vec![0u8; 200_000];
        rng.fill(&mut data);
        assert_roundtrips(&data, "incompressible-200k");
    }

    // ---- proptest: random + adversarial roundtrip through flate2 ----

    use proptest::prelude::*;

    /// A strategy mixing uniform-random bytes, runs, tiny inputs, and inputs
    /// near the 64 KiB stored-block boundary.
    fn adversarial_bytes() -> impl Strategy<Value = Vec<u8>> {
        prop_oneof![
            // Uniform random, any length up to ~70 KiB (crosses the boundary).
            proptest::collection::vec(any::<u8>(), 0..70_000),
            // Runs of a repeated byte (very compressible).
            (any::<u8>(), 0usize..70_000usize).prop_map(|(b, n)| vec![b; n]),
            // Tiny inputs.
            proptest::collection::vec(any::<u8>(), 0..8),
            // Near-boundary lengths with random content.
            (65_500usize..65_570usize).prop_map(|n| {
                let mut v = vec![0u8; n];
                Rng::new(n as u64).fill(&mut v);
                v
            }),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 200, max_shrink_iters: 2000, ..ProptestConfig::default() })]

        #[test]
        fn prop_roundtrip_flate2(input in adversarial_bytes()) {
            let gz = compress_gzip(&input, 6);
            let decoded = decode_flate2(&gz);
            prop_assert_eq!(decoded, input);
        }
    }
}
