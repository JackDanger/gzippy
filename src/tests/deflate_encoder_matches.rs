//! Roundtrip-correctness net for the pure-Rust DEFLATE encoder with REAL
//! matches (Increment 2: hash-chain matchfinder + greedy/lazy/lazy2 parsers,
//! `src/compress/deflate/`).
//!
//! A wrong match length or distance corrupts the DEFLATE stream in ways the
//! Huffman-only Increment-1 net could never produce. The contract: for EVERY
//! implemented level (2..=9), the gzip-framed output of `compress_gzip` decodes
//! BACK to the exact input through THREE independent decoders — flate2/zlib-ng,
//! libdeflate (FFI), and the system `gzip -d` — across an adversarial corpus
//! that exercises long matches, the full 32 KiB window, and distance codes,
//! plus a proptest generator over random/run/repeat/near-boundary inputs.
//!
//! Run: `cargo test --release deflate_encoder_matches`.

#[cfg(test)]
mod tests {
    use crate::compress::deflate::{compress_gzip, compress_oneshot};
    use std::io::{Read, Write};
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    /// Levels the Increment-2 engine implements with real match finding.
    const LEVELS: [u32; 8] = [2, 3, 4, 5, 6, 7, 8, 9];

    // ---- the three independent decoders ----

    fn decode_flate2(gz: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(gz)
            .read_to_end(&mut out)
            .expect("flate2 failed to decode our gzip stream");
        out
    }

    fn decode_libdeflate(gz: &[u8], expected_len: usize) -> Vec<u8> {
        let mut decomp = libdeflater::Decompressor::new();
        let mut out = vec![0u8; expected_len.max(1)];
        let n = decomp
            .gzip_decompress(gz, &mut out)
            .expect("libdeflate failed to decode our gzip stream");
        out.truncate(n);
        out
    }

    /// Returns `None` only when no `gzip` binary is on PATH.
    ///
    /// Feeds stdin from a dedicated thread while the parent reads stdout, so a
    /// large (multi-MiB) decompressed output cannot deadlock against the pipe
    /// buffer (writer blocked on a full stdin pipe while gzip is blocked on a
    /// full stdout pipe).
    fn decode_system_gzip(gz: &[u8]) -> Option<Vec<u8>> {
        let mut child = Command::new("gzip")
            .arg("-d")
            .arg("-c")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let mut stdin = child.stdin.take().unwrap();
        let gz_owned = gz.to_vec();
        let writer = std::thread::spawn(move || {
            let _ = stdin.write_all(&gz_owned);
            // Drop closes stdin, signaling EOF to gzip.
        });
        let out = child.wait_with_output().expect("wait for gzip");
        writer.join().expect("gzip stdin writer thread");
        assert!(out.status.success(), "system gzip -d rejected our stream");
        Some(out.stdout)
    }

    /// Is a usable `gzip` on PATH? Determined once; if it is, every roundtrip
    /// MUST exercise it (so the oracle is never silently skipped on a box that
    /// has it).
    fn gzip_available() -> bool {
        use std::sync::OnceLock;
        static AVAIL: OnceLock<bool> = OnceLock::new();
        *AVAIL.get_or_init(|| {
            // Round-trip a known stream through our encoder and system gzip.
            let gz = compress_gzip(b"gzip availability probe", 6);
            decode_system_gzip(&gz)
                .map(|d| d == b"gzip availability probe")
                .unwrap_or(false)
        })
    }

    /// Assert every available oracle recovers `input` byte-exact at `level`.
    /// Returns the number of oracles that actually decoded + compared.
    fn assert_roundtrips_level(input: &[u8], label: &str, level: u32) -> u32 {
        let gz = compress_gzip(input, level);
        let mut oracles = 0u32;

        let f = decode_flate2(&gz);
        assert_eq!(f, input, "flate2 mismatch [{label} L{level}]");
        oracles += 1;

        let l = decode_libdeflate(&gz, input.len());
        assert_eq!(l, input, "libdeflate mismatch [{label} L{level}]");
        oracles += 1;

        if gzip_available() {
            let g = decode_system_gzip(&gz).expect("gzip was available at probe time");
            assert_eq!(g, input, "system gzip mismatch [{label} L{level}]");
            oracles += 1;
        }

        // flate2 + libdeflate must always run; gzip too when present. None may
        // silently no-op.
        let expected = if gzip_available() { 3 } else { 2 };
        assert_eq!(
            oracles, expected,
            "an oracle was silently skipped [{label} L{level}]"
        );
        oracles
    }

    /// Run the roundtrip at every implemented level.
    fn assert_roundtrips(input: &[u8], label: &str) {
        for &level in &LEVELS {
            assert_roundtrips_level(input, label, level);
        }
    }

    // ---- deterministic input generators ----

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
        use std::io::Seek;
        f.seek(std::io::SeekFrom::Start(1 << 16)).ok()?;
        f.read_exact(&mut buf).ok()?;
        Some(buf)
    }

    #[test]
    fn oracles_present() {
        // flate2 + libdeflate are compiled-in; verify they actually decode our
        // stream, and report gzip availability loudly.
        let gz = compress_gzip(b"the quick brown fox", 6);
        assert_eq!(decode_flate2(&gz), b"the quick brown fox");
        assert_eq!(decode_libdeflate(&gz, 19), b"the quick brown fox");
        if gzip_available() {
            eprintln!("oracles: flate2 + libdeflate + system gzip (3 active)");
        } else {
            eprintln!("note: system gzip unavailable; flate2 + libdeflate only (2 active)");
        }
    }

    #[test]
    fn roundtrip_small_and_edge() {
        assert_roundtrips(b"", "empty");
        assert_roundtrips(b"A", "one-byte");
        assert_roundtrips(b"ab", "two-byte");
        assert_roundtrips(b"abc", "three-byte");
        assert_roundtrips(b"abcd", "four-byte");
        assert_roundtrips(b"hello", "hello");
        assert_roundtrips(b"hellohello", "hello-x2");
        assert_roundtrips(&b"hello".repeat(1000), "hello-x1000");
        assert_roundtrips(b"hello, world! hello, world!", "short-text");
    }

    #[test]
    fn roundtrip_repetitive_long_matches() {
        // All-same-byte: maximal-length (258) matches back-to-back.
        assert_roundtrips(&vec![b'Z'; 300_000], "all-same-300k");

        // A repeated 100-byte block: long matches at a fixed small offset.
        let block: Vec<u8> = (0..100u32).map(|i| (i * 7) as u8).collect();
        let repeated: Vec<u8> = block.iter().cloned().cycle().take(250_000).collect();
        assert_roundtrips(&repeated, "block100-repeat");

        // Two-symbol alternation.
        let alt: Vec<u8> = (0..80_000)
            .map(|i| if i % 2 == 0 { b'a' } else { b'b' })
            .collect();
        assert_roundtrips(&alt, "alternating");
    }

    #[test]
    fn roundtrip_incompressible() {
        // Random 64 KiB: no matches — should route to stored blocks.
        let mut rng = Rng::new(0xC0FFEE);
        let mut incompressible = vec![0u8; 64 * 1024];
        rng.fill(&mut incompressible);
        assert_roundtrips(&incompressible, "incompressible-64k");

        // Just over the 64 KiB stored sub-block boundary.
        let mut boundary = vec![0u8; 65_536 + 100];
        Rng::new(7).fill(&mut boundary);
        assert_roundtrips(&boundary, "stored-boundary");
    }

    #[test]
    fn roundtrip_distant_repeats_full_window() {
        // A pattern near the start, then ~31 KiB of unrelated filler, then the
        // pattern again: forces a large-distance match (offset in the tens of
        // thousands) that exercises the high offset-slot distance codes.
        let pattern: Vec<u8> = (0..400u32)
            .map(|i| (i.wrapping_mul(131) ^ 0x5A) as u8)
            .collect();
        let mut rng = Rng::new(0xD157A17);
        let mut filler = vec![0u8; 31_000];
        rng.fill(&mut filler);

        let mut data = Vec::new();
        data.extend_from_slice(&pattern);
        data.extend_from_slice(&filler);
        data.extend_from_slice(&pattern); // distant repeat (offset ~31.4 KiB)
        data.extend_from_slice(&filler[..5000]);
        data.extend_from_slice(&pattern); // another distant repeat
        assert_roundtrips(&data, "distant-repeats");

        // Mixed text with recurring distant phrases across the window.
        let phrase = b"the pure-rust deflate encoder must roundtrip byte for byte. ";
        let mut text = Vec::new();
        for i in 0..2000 {
            text.extend_from_slice(phrase);
            if i % 7 == 0 {
                text.extend_from_slice(format!("[marker {i}] ").as_bytes());
            }
        }
        assert_roundtrips(&text, "recurring-phrases");
    }

    #[test]
    fn roundtrip_multiblock_and_all_bytes() {
        // Larger than one soft-max block (300 KiB) with a biased distribution.
        let mut multiblock = vec![0u8; 700_000];
        for (i, b) in multiblock.iter_mut().enumerate() {
            *b = ((i * 31 + (i >> 8)) % 251) as u8;
        }
        assert_roundtrips(&multiblock, "multi-block");

        // Full byte-value coverage with structure.
        let all_bytes: Vec<u8> = (0..=255u8).cycle().take(20_000).collect();
        assert_roundtrips(&all_bytes, "all-byte-values");
    }

    #[test]
    fn roundtrip_silesia_slice() {
        match silesia_slice(1 << 20) {
            Some(data) => assert_roundtrips(&data, "silesia-1MiB"),
            None => eprintln!("note: benchmark_data/silesia.tar missing; skipped silesia slice"),
        }
    }

    /// Correctness-adjacent RATIO sanity (NOT a perf claim): the new engine must
    /// be in the same ballpark as flate2 at each level, else the parser is
    /// losing matches.
    #[test]
    fn ratio_sanity_vs_flate2() {
        let Some(data) = silesia_slice(1 << 20) else {
            eprintln!("note: silesia.tar missing; skipped ratio sanity");
            return;
        };
        for &level in &[2u32, 6, 9] {
            let ours = compress_oneshot(&data, level).len();

            let mut enc =
                flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
            enc.write_all(&data).unwrap();
            let theirs = enc.finish().unwrap().len();

            let ratio = ours as f64 / theirs as f64;
            eprintln!("ratio L{level}: ours={ours} flate2={theirs} (ours/flate2 = {ratio:.3})",);
            // Generous bound — this only catches a parser that is badly losing
            // matches, not small coding differences.
            assert!(
                ratio < 1.30,
                "L{level}: pure-Rust output {ours} is {ratio:.2}x flate2 {theirs} — matches likely lost",
            );
        }
    }

    // ---- proptest: random + adversarial roundtrip through flate2 ----

    use proptest::prelude::*;

    /// A strategy mixing uniform-random bytes, runs, repeated blocks (real
    /// matches), and near-window-boundary lengths.
    fn adversarial_bytes() -> impl Strategy<Value = Vec<u8>> {
        prop_oneof![
            // Uniform random, any length up to ~70 KiB.
            proptest::collection::vec(any::<u8>(), 0..70_000),
            // Runs of a repeated byte (maximal-length matches).
            (any::<u8>(), 0usize..70_000usize).prop_map(|(b, n)| vec![b; n]),
            // Repeated small blocks (matches at various offsets).
            (
                proptest::collection::vec(any::<u8>(), 1..300),
                1usize..400usize
            )
                .prop_map(|(block, reps)| block
                    .iter()
                    .cloned()
                    .cycle()
                    .take(block.len() * reps)
                    .collect()),
            // Random prefix + its own repeat far away (distant matches).
            (
                proptest::collection::vec(any::<u8>(), 50..2000),
                0usize..40_000usize
            )
                .prop_map(|(pat, gap)| {
                    let mut v = pat.clone();
                    v.resize(v.len() + gap, 0u8);
                    v.extend_from_slice(&pat);
                    v
                }),
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
        #![proptest_config(ProptestConfig { cases: 240, max_shrink_iters: 4000, ..ProptestConfig::default() })]

        #[test]
        fn prop_roundtrip_flate2(input in adversarial_bytes(), level in 2u32..=9u32) {
            let gz = compress_gzip(&input, level);
            let decoded = decode_flate2(&gz);
            prop_assert_eq!(decoded, input);
        }
    }
}
