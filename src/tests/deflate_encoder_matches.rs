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
        silesia_slice_at(1 << 16, n)
    }

    /// A `n`-byte slice of `benchmark_data/silesia.tar` starting at byte
    /// `offset`. Distinct offsets land in different member files (silesia is a
    /// tar of heterogeneous corpora — text, binaries, DB dumps), giving the
    /// multi-corpus ratio guard genuinely different byte statistics.
    fn silesia_slice_at(offset: u64, n: usize) -> Option<Vec<u8>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmark_data/silesia.tar");
        let mut f = std::fs::File::open(path).ok()?;
        let mut buf = vec![0u8; n];
        use std::io::Seek;
        f.seek(std::io::SeekFrom::Start(offset)).ok()?;
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

    // ======================================================================
    // Increment 3 — near-optimal parser (L10-12) correctness net.
    // ======================================================================

    /// The near-optimal levels: bt matchfinder + iterative min-cost-path DP.
    const NEAR_LEVELS: [u32; 3] = [10, 11, 12];

    /// libdeflate raw-DEFLATE size at `level` (the ratio oracle).
    fn libdeflate_deflate_size(data: &[u8], level: i32) -> usize {
        let lvl = libdeflater::CompressionLvl::new(level).expect("valid level");
        let mut c = libdeflater::Compressor::new(lvl);
        let bound = c.deflate_compress_bound(data.len());
        let mut out = vec![0u8; bound];
        c.deflate_compress(data, &mut out)
            .expect("libdeflate compress")
    }

    /// Roundtrip a deep near-optimal stream through all three oracles. Uses a
    /// silesia slice of at least 2×SOFT_MAX_BLOCK_LENGTH so the driver spans
    /// multiple blocks AND exercises the rewind-on-split path.
    #[test]
    fn near_optimal_deep_roundtrip_three_oracles() {
        // 2 * SOFT_MAX_BLOCK_LENGTH = 600_000; take a bit more so a real split
        // and rewind occur.
        let Some(data) = silesia_slice(700_000) else {
            eprintln!("note: silesia.tar missing; skipped near-optimal deep roundtrip");
            return;
        };
        for &level in &NEAR_LEVELS {
            let n = assert_roundtrips_level(&data, "silesia-700k-near-optimal", level);
            // Confirm all present oracles actually ran (never a silent skip).
            let expected = if gzip_available() { 3 } else { 2 };
            assert_eq!(n, expected, "L{level}: an oracle was skipped");
        }
    }

    /// Smaller near-optimal roundtrips exercising the adversarial corners.
    #[test]
    fn near_optimal_roundtrip_adversarial() {
        for &level in &NEAR_LEVELS {
            assert_roundtrips_level(b"", "empty", level);
            assert_roundtrips_level(b"a", "one-byte", level);
            assert_roundtrips_level(b"abcabcabc", "tiny-repeat", level);
            // Highly redundant: exercises the long-match-skip path.
            assert_roundtrips_level(&vec![b'Q'; 400_000], "all-same-400k", level);
            let block: Vec<u8> = (0..137u32).map(|i| (i * 13) as u8).collect();
            let repeated: Vec<u8> = block.iter().cloned().cycle().take(650_000).collect();
            assert_roundtrips_level(&repeated, "block137-repeat-650k", level);
            // Incompressible: exercises the only-literals / stored path.
            let mut incompressible = vec![0u8; 90_000];
            Rng::new(0xBADCAFE).fill(&mut incompressible);
            assert_roundtrips_level(&incompressible, "incompressible-90k", level);
            // Window-straddling distant repeat.
            let pattern: Vec<u8> = (0..500u32)
                .map(|i| (i.wrapping_mul(151) ^ 0x33) as u8)
                .collect();
            let mut d = pattern.clone();
            d.resize(d.len() + 33_000, 0u8);
            d.extend_from_slice(&pattern);
            assert_roundtrips_level(&d, "window-straddle", level);
        }
    }

    /// THE coverage proof: the near-optimal L12 output must reach the
    /// compression CEILING — strictly smaller than libdeflate L9 AND within
    /// 0.1% of libdeflate L12 — on a silesia slice. A correct-but-too-big DP
    /// (wrong cost model / lost matches) roundtrips fine but FAILS here.
    #[test]
    fn near_optimal_reaches_ratio_ceiling() {
        let Some(data) = silesia_slice(4_000_000) else {
            eprintln!("note: silesia.tar missing; skipped ratio-ceiling assertion");
            return;
        };
        let ours_l12 = compress_oneshot(&data, 12).len();
        let libdeflate_l9 = libdeflate_deflate_size(&data, 9);
        let libdeflate_l12 = libdeflate_deflate_size(&data, 12);

        eprintln!(
            "ratio ceiling (silesia 4 MiB): ours_L12={ours_l12} \
             libdeflate_L9={libdeflate_l9} libdeflate_L12={libdeflate_l12} \
             (ours/L9={:.4}, ours/L12={:.4})",
            ours_l12 as f64 / libdeflate_l9 as f64,
            ours_l12 as f64 / libdeflate_l12 as f64,
        );

        // (1) Beats libdeflate L9 — the ceiling gzippy could not previously reach.
        assert!(
            ours_l12 < libdeflate_l9,
            "L12 near-optimal {ours_l12} not smaller than libdeflate L9 {libdeflate_l9}"
        );
        // (2) Matches libdeflate L12 to within 0.1%.
        let bound = (libdeflate_l12 as f64 * 1.001) as usize;
        assert!(
            ours_l12 <= bound,
            "L12 near-optimal {ours_l12} exceeds libdeflate L12 {libdeflate_l12} by >0.1% (bound {bound})"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 96, max_shrink_iters: 3000, ..ProptestConfig::default() })]

        /// Roundtrip near-optimal output over the adversarial generator. Fewer
        /// cases than the L2-9 proptest because the DP is heavier.
        #[test]
        fn prop_near_optimal_roundtrip_flate2(input in adversarial_bytes(), level in 10u32..=12u32) {
            let gz = compress_gzip(&input, level);
            let decoded = decode_flate2(&gz);
            prop_assert_eq!(decoded, input);
        }
    }

    // ======================================================================
    // Increment 4 — igzip-class one-pass FAST path (L1) correctness + sanity.
    // ======================================================================
    //
    // L1 is a NEW mode: a single static-Huffman block emitted directly from a
    // chainless single-probe matchfinder. It is not byte-identical to any tool;
    // the only correctness contract is that its gzip stream decodes BACK to the
    // exact input through every oracle. Ratio is a soft "actually compresses,
    // no wild blowup" sanity — NOT a perf/ratio claim (the caller gates that).

    /// gzip-framed size of `data` under flate2/zlib-ng at `level` (the stable,
    /// always-present ratio reference, comparable to `gzip -level`).
    fn flate2_gzip_size(data: &[u8], level: u32) -> usize {
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        e.write_all(data).unwrap();
        e.finish().unwrap().len()
    }

    /// `pigz -{level} -c` output size via the system binary, or `None` when
    /// `pigz` is not on PATH (CI without pigz) so the caller skips the pigz
    /// assertion rather than failing. This is the actual rival the Lever-1
    /// fast-L1 ratio target is stated against.
    fn pigz_gzip_size(data: &[u8], level: u32) -> Option<usize> {
        let mut child = Command::new("pigz")
            .arg(format!("-{level}"))
            .arg("-c")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let mut stdin = child.stdin.take().unwrap();
        let buf = data.to_vec();
        let w = std::thread::spawn(move || {
            let _ = stdin.write_all(&buf);
        });
        let out = child.wait_with_output().ok()?;
        w.join().ok()?;
        if !out.status.success() {
            return None;
        }
        Some(out.stdout.len())
    }

    /// The L1 fast path must roundtrip-decode through all three oracles across
    /// the full adversarial corner corpus.
    #[test]
    fn fast_l1_roundtrip_three_oracles() {
        // empty / tiny / hello×N.
        assert_roundtrips_level(b"", "empty", 1);
        assert_roundtrips_level(b"A", "one-byte", 1);
        assert_roundtrips_level(b"ab", "two-byte", 1);
        assert_roundtrips_level(b"abc", "three-byte", 1);
        assert_roundtrips_level(b"abcd", "four-byte", 1);
        assert_roundtrips_level(b"hello", "hello", 1);
        assert_roundtrips_level(&b"hello".repeat(1000), "hello-x1000", 1);
        assert_roundtrips_level(b"hello, world! hello, world!", "short-text", 1);

        // all-same-byte (maximal back-to-back matches) + a small-block repeat.
        assert_roundtrips_level(&vec![b'Z'; 300_000], "all-same-300k", 1);
        let block: Vec<u8> = (0..100u32).map(|i| (i * 7) as u8).collect();
        let repeated: Vec<u8> = block.iter().cloned().cycle().take(250_000).collect();
        assert_roundtrips_level(&repeated, "block100-repeat", 1);

        // incompressible-64K (only-literals path; may expand slightly — must
        // still roundtrip). Plus a just-over-64KiB boundary case.
        let mut incompressible = vec![0u8; 64 * 1024];
        Rng::new(0xC0FFEE).fill(&mut incompressible);
        assert_roundtrips_level(&incompressible, "incompressible-64k", 1);
        let mut boundary = vec![0u8; 65_536 + 100];
        Rng::new(7).fill(&mut boundary);
        assert_roundtrips_level(&boundary, "stored-boundary", 1);

        // long matches + distant window-straddling repeats (high offset slots).
        let pattern: Vec<u8> = (0..400u32)
            .map(|i| (i.wrapping_mul(131) ^ 0x5A) as u8)
            .collect();
        let mut rng = Rng::new(0xD157A17);
        let mut filler = vec![0u8; 31_000];
        rng.fill(&mut filler);
        let mut data = Vec::new();
        data.extend_from_slice(&pattern);
        data.extend_from_slice(&filler);
        data.extend_from_slice(&pattern); // distant repeat (~31 KiB back)
        data.extend_from_slice(&filler[..5000]);
        data.extend_from_slice(&pattern);
        assert_roundtrips_level(&data, "distant-repeats", 1);

        // multi-chunk / large text with recurring phrases.
        let phrase = b"the pure-rust deflate encoder must roundtrip byte for byte. ";
        let mut text = Vec::new();
        for i in 0..4000 {
            text.extend_from_slice(phrase);
            if i % 7 == 0 {
                text.extend_from_slice(format!("[marker {i}] ").as_bytes());
            }
        }
        assert_roundtrips_level(&text, "recurring-phrases", 1);

        // >= 1 MiB silesia slice through all three oracles; assert none skipped.
        if let Some(sil) = silesia_slice(1 << 20) {
            let n = assert_roundtrips_level(&sil, "silesia-1MiB", 1);
            let expected = if gzip_available() { 3 } else { 2 };
            assert_eq!(n, expected, "L1: an oracle was skipped");
        } else {
            eprintln!("note: silesia.tar missing; skipped L1 silesia roundtrip");
        }
    }

    /// CONTENT-ADAPTIVE CHAIN MATCHING (2026-07-22 mission) correctness net:
    /// `l1-tune`'s `chain_enabled` lever switches a block's matcher from the
    /// chainless single-probe finder to the hash-chains finder mid-parse —
    /// a genuinely different code path (`fast::chain_block`/`hc_catchup` in
    /// `parse/fast.rs`) that the OTHER tests in this file never exercise
    /// (they never call `tune::set`, so `chain_enabled` stays at its `false`
    /// env-var default). Runs the SAME three-oracle roundtrip net as
    /// [`fast_l1_roundtrip_three_oracles`] but with chain mode forced on
    /// across a (threshold, depth) grid, over both the adversarial corner
    /// corpus (empty/tiny/all-same/distant-repeats/boundary cases — the
    /// finder-switch mid-block-loop is exactly where an off-by-one in the
    /// `hc_catchup` contiguity contract would corrupt output) and a
    /// generated literal-dense (`binary_corpus`) corpus sized to guarantee
    /// multiple 64 KiB blocks so the one-block-lag detector actually flips
    /// mid-file.
    ///
    /// Test isolation: `tune::set` mutates PROCESS-GLOBAL state (by design —
    /// see `tune::set`'s doc comment), so this restores the prior snapshot
    /// via a `Drop` guard even on panic, keeping every OTHER test in this
    /// binary unaffected once this test returns. A transient window where
    /// truly-concurrent tests could observe `chain_enabled=true` remains
    /// (inherent to the global-tune design also used by `examples/
    /// l1_search.rs`) — benign here because chain mode is monotonically
    /// size-non-increasing and never changes roundtrip correctness, so it
    /// cannot flip a passing assertion in another test to failing; no
    /// existing test in this file asserts an exact byte count that chain
    /// mode could perturb.
    #[cfg(feature = "l1-tune")]
    #[test]
    fn chain_mode_roundtrip_adversarial() {
        use crate::compress::deflate::parse::tune::{self, L1Tune};

        struct RestoreTune(L1Tune);
        impl Drop for RestoreTune {
            fn drop(&mut self) {
                tune::set(self.0);
            }
        }
        let _restore = RestoreTune(tune::get());

        let base = tune::get();
        let literal_dense = binary_corpus(400_000); // several 64 KiB blocks

        for threshold in [50u32, 80] {
            for depth in [4u32, 32] {
                tune::set(L1Tune {
                    chain_enabled: true,
                    chain_lit_threshold_pct: threshold,
                    chain_max_search_depth: depth,
                    ..base
                });
                let tag = format!("chain-t{threshold}-d{depth}");

                // Adversarial corner cases (mirrors
                // `fast_l1_roundtrip_three_oracles`, chain-mode-forced).
                assert_roundtrips_level(b"", &format!("{tag}-empty"), 1);
                assert_roundtrips_level(b"A", &format!("{tag}-one-byte"), 1);
                assert_roundtrips_level(&vec![b'Z'; 300_000], &format!("{tag}-all-same"), 1);
                let mut incompressible = vec![0u8; 64 * 1024 + 37];
                Rng::new(0xC0FFEE ^ threshold as u64 ^ depth as u64).fill(&mut incompressible);
                assert_roundtrips_level(&incompressible, &format!("{tag}-incompressible"), 1);
                let pattern: Vec<u8> = (0..400u32)
                    .map(|i| (i.wrapping_mul(131) ^ 0x5A) as u8)
                    .collect();
                let mut rng = Rng::new(0xD157A17 ^ threshold as u64 ^ depth as u64);
                let mut filler = vec![0u8; 31_000];
                rng.fill(&mut filler);
                let mut data = Vec::new();
                data.extend_from_slice(&pattern);
                data.extend_from_slice(&filler);
                data.extend_from_slice(&pattern);
                data.extend_from_slice(&filler[..5000]);
                data.extend_from_slice(&pattern);
                assert_roundtrips_level(&data, &format!("{tag}-distant-repeats"), 1);

                // The literal-dense corpus this lever targets: enough blocks
                // (400_000 / 65536 ~= 6) for the one-block-lag detector to
                // both fire (block 2+) and, at threshold=80, sometimes NOT
                // fire (exercising the chainless<->chain toggle both ways in
                // one `run()` call, the `hc_catchup` resync path).
                let n = assert_roundtrips_level(&literal_dense, &format!("{tag}-bindense"), 1);
                let expected = if gzip_available() { 3 } else { 2 };
                assert_eq!(n, expected, "{tag}: an oracle was silently skipped");
            }
        }
    }

    /// HASH3-PROBE lever roundtrip differential (2026-07-22 campaign): the
    /// last unmeasured member of the L1 probe-adding family — see
    /// `parse::tune::L1Tune::hash3_enabled`'s doc comment for the full
    /// mechanism (a genuine 3-byte-key `head3` table making length-3
    /// matches VISIBLE, unlike the falsified accept-flip which reused the
    /// 4-byte-hash candidate). Sweeps table size, both probe policies
    /// (miss-only vs always-probe), and both insert policies (always vs
    /// literal-only-sparse) across the same adversarial corners
    /// `chain_mode_roundtrip_adversarial` uses (empty, one byte, all-same,
    /// incompressible, distant-repeats, and a literal-dense corpus sized to
    /// exercise multiple 64 KiB blocks) — every combination MUST roundtrip
    /// byte-for-byte through every available oracle, since this lever
    /// changes ACCEPT decisions (a new class of match becomes emittable),
    /// the single place a bit-exact bug would most likely hide.
    ///
    /// Test isolation: same `Drop`-guard restore pattern as
    /// `chain_mode_roundtrip_adversarial` (see that test's doc comment for
    /// why this is safe against the shared process-global `tune` cell).
    #[cfg(feature = "l1-tune")]
    #[test]
    fn hash3_probe_roundtrip_adversarial() {
        use crate::compress::deflate::parse::tune::{self, L1Tune};

        struct RestoreTune(L1Tune);
        impl Drop for RestoreTune {
            fn drop(&mut self) {
                tune::set(self.0);
            }
        }
        let _restore = RestoreTune(tune::get());

        let base = tune::get();
        let literal_dense = binary_corpus(400_000);

        for bits in [12u32, 15] {
            for always_probe in [false, true] {
                for insert_always in [true, false] {
                    for max_dist in [0usize, 4096, 32768] {
                        tune::set(L1Tune {
                            hash3_enabled: true,
                            hash3_bits: bits,
                            hash3_always_probe: always_probe,
                            hash3_max_dist: max_dist,
                            hash3_insert_always: insert_always,
                            // Explicit `false`: this test targets the
                            // HASH3-PROBE lever in isolation across its own
                            // axes (bits/policy/insert/max_dist). Since the
                            // 2026-07-22 ship promotion, `base` (the
                            // process default) now has `hash3_gated: true`
                            // — leaving the gate on here would make most
                            // combinations exercise the GATED composition
                            // instead (covered separately by
                            // `hash3_gate_roundtrip_adversarial` below),
                            // silently narrowing this test's own coverage.
                            hash3_gated: false,
                            ..base
                        });
                        let tag = format!(
                            "hash3-b{bits}-ap{}-ia{}-d{max_dist}",
                            always_probe as u8, insert_always as u8
                        );

                        assert_roundtrips_level(b"", &format!("{tag}-empty"), 1);
                        assert_roundtrips_level(b"A", &format!("{tag}-one-byte"), 1);
                        assert_roundtrips_level(b"AB", &format!("{tag}-two-bytes"), 1);
                        assert_roundtrips_level(b"ABC", &format!("{tag}-three-bytes"), 1);
                        assert_roundtrips_level(
                            &vec![b'Z'; 300_000],
                            &format!("{tag}-all-same"),
                            1,
                        );
                        // Dense 3-byte-periodic repeats: the exact shape a
                        // genuine hash3 candidate should surface most often
                        // (a length-3-ish run at a short, then a long,
                        // distance).
                        let mut triples = Vec::new();
                        for i in 0..100_000u32 {
                            triples.extend_from_slice(&[
                                (i % 251) as u8,
                                ((i / 3) % 251) as u8,
                                ((i * 7) % 251) as u8,
                            ]);
                        }
                        assert_roundtrips_level(&triples, &format!("{tag}-triples"), 1);
                        let mut incompressible = vec![0u8; 64 * 1024 + 37];
                        Rng::new(0xC0FFEE ^ bits as u64 ^ max_dist as u64)
                            .fill(&mut incompressible);
                        assert_roundtrips_level(
                            &incompressible,
                            &format!("{tag}-incompressible"),
                            1,
                        );
                        let pattern: Vec<u8> = (0..400u32)
                            .map(|i| (i.wrapping_mul(131) ^ 0x5A) as u8)
                            .collect();
                        let mut rng = Rng::new(0xD157A17 ^ bits as u64 ^ max_dist as u64);
                        let mut filler = vec![0u8; 31_000];
                        rng.fill(&mut filler);
                        let mut data = Vec::new();
                        data.extend_from_slice(&pattern);
                        data.extend_from_slice(&filler);
                        data.extend_from_slice(&pattern);
                        data.extend_from_slice(&filler[..5000]);
                        data.extend_from_slice(&pattern);
                        assert_roundtrips_level(&data, &format!("{tag}-distant-repeats"), 1);

                        let n =
                            assert_roundtrips_level(&literal_dense, &format!("{tag}-bindense"), 1);
                        let expected = if gzip_available() { 3 } else { 2 };
                        assert_eq!(n, expected, "{tag}: an oracle was silently skipped");
                    }
                }
            }
        }
    }

    /// HASH3-GATE composition roundtrip differential (2026-07-22 "compose
    /// the two proven l1-tune levers" mission): `hash3_gated` reuses the
    /// content-adaptive chain matching lever's literal-fraction detector to
    /// gate the HASH3-PROBE lever's per-block probe/insert, a genuinely
    /// different code path (`process_position_l1`'s `hash3_touch`/
    /// `hash3_active` split, and `run`'s independent `hash3_active_next`
    /// one-block-lag state machine) from either lever alone — neither
    /// `chain_mode_roundtrip_adversarial` nor `hash3_probe_roundtrip_
    /// adversarial` exercises the gate transition itself (a block flipping
    /// from probe-silent to probe-active mid-file, and the warm-insert vs
    /// sparse-insert policies at that transition). Same adversarial-corner
    /// + literal-dense-multi-block shape as the two lever tests above,
    /// swept across threshold x warm-insert x initial-active.
    ///
    /// Test isolation: same `Drop`-guard restore pattern as the two lever
    /// tests above.
    #[cfg(feature = "l1-tune")]
    #[test]
    fn hash3_gate_roundtrip_adversarial() {
        use crate::compress::deflate::parse::tune::{self, L1Tune};

        struct RestoreTune(L1Tune);
        impl Drop for RestoreTune {
            fn drop(&mut self) {
                tune::set(self.0);
            }
        }
        let _restore = RestoreTune(tune::get());

        let base = tune::get();
        // Sized for several 64 KiB blocks so the one-block-lag gate both
        // arms (active->inactive and inactive->active) within one file.
        let literal_dense = binary_corpus(400_000);

        for threshold in [50u32, 80] {
            for warm_insert in [true, false] {
                for initial_active in [true, false] {
                    tune::set(L1Tune {
                        hash3_enabled: true,
                        hash3_bits: 15,
                        hash3_always_probe: false,
                        hash3_max_dist: 32768,
                        hash3_insert_always: true,
                        hash3_gated: true,
                        hash3_gate_lit_threshold_pct: threshold,
                        hash3_gate_warm_insert: warm_insert,
                        hash3_gate_initial_active: initial_active,
                        ..base
                    });
                    let tag = format!(
                        "hash3gate-t{threshold}-w{}-i{}",
                        warm_insert as u8, initial_active as u8
                    );

                    assert_roundtrips_level(b"", &format!("{tag}-empty"), 1);
                    assert_roundtrips_level(b"A", &format!("{tag}-one-byte"), 1);
                    assert_roundtrips_level(b"AB", &format!("{tag}-two-bytes"), 1);
                    assert_roundtrips_level(b"ABC", &format!("{tag}-three-bytes"), 1);
                    assert_roundtrips_level(&vec![b'Z'; 300_000], &format!("{tag}-all-same"), 1);
                    let mut triples = Vec::new();
                    for i in 0..100_000u32 {
                        triples.extend_from_slice(&[
                            (i % 251) as u8,
                            ((i / 3) % 251) as u8,
                            ((i * 7) % 251) as u8,
                        ]);
                    }
                    assert_roundtrips_level(&triples, &format!("{tag}-triples"), 1);
                    let mut incompressible = vec![0u8; 64 * 1024 + 37];
                    Rng::new(0xC0FFEE ^ threshold as u64 ^ warm_insert as u64)
                        .fill(&mut incompressible);
                    assert_roundtrips_level(&incompressible, &format!("{tag}-incompressible"), 1);
                    let pattern: Vec<u8> = (0..400u32)
                        .map(|i| (i.wrapping_mul(131) ^ 0x5A) as u8)
                        .collect();
                    let mut rng = Rng::new(0xD157A17 ^ threshold as u64 ^ warm_insert as u64);
                    let mut filler = vec![0u8; 31_000];
                    rng.fill(&mut filler);
                    let mut data = Vec::new();
                    data.extend_from_slice(&pattern);
                    data.extend_from_slice(&filler);
                    data.extend_from_slice(&pattern);
                    data.extend_from_slice(&filler[..5000]);
                    data.extend_from_slice(&pattern);
                    assert_roundtrips_level(&data, &format!("{tag}-distant-repeats"), 1);

                    // Mixed content: a low-literal-fraction (match-heavy)
                    // prefix followed by a high-literal-fraction (bin-like)
                    // suffix, sized to span multiple blocks — exercises the
                    // gate flipping ACTIVE mid-file (`initial_active=false`
                    // starts silent, the bin suffix should trip it on).
                    let mut mixed = text_corpus(300_000);
                    mixed.extend_from_slice(&binary_corpus(300_000));
                    let n = assert_roundtrips_level(&mixed, &format!("{tag}-mixed"), 1);
                    let expected = if gzip_available() { 3 } else { 2 };
                    assert_eq!(n, expected, "{tag}: an oracle was silently skipped");

                    let n2 = assert_roundtrips_level(&literal_dense, &format!("{tag}-bindense"), 1);
                    assert_eq!(n2, expected, "{tag}: an oracle was silently skipped");
                }
            }
        }
    }

    /// gzip-framed size of `data` under libdeflate at `level` (the tighter ratio
    /// reference: `size_fastL1 <= libdeflate -1` is the aspiration, `<= gzip -1`
    /// the hard floor).
    fn libdeflate_gzip_size(data: &[u8], level: i32) -> usize {
        let lvl = libdeflater::CompressionLvl::new(level).expect("valid level");
        let mut c = libdeflater::Compressor::new(lvl);
        let bound = c.gzip_compress_bound(data.len());
        let mut out = vec![0u8; bound];
        c.gzip_compress(data, &mut out).expect("libdeflate gzip")
    }

    /// A text-heavy generated corpus (~`n` bytes): recurring English phrases with
    /// periodic markers, so the byte statistics differ sharply from the binary
    /// corpus below and from raw silesia slices.
    fn text_corpus(n: usize) -> Vec<u8> {
        let phrases: [&[u8]; 4] = [
            b"the pure-rust deflate encoder must roundtrip byte for byte. ",
            b"lempel-ziv parsing finds the longest match at each position. ",
            b"dynamic huffman codes adapt to the local symbol frequencies. ",
            b"a stored block escapes when the data will not compress at all. ",
        ];
        let mut out = Vec::with_capacity(n + 128);
        let mut i = 0usize;
        while out.len() < n {
            out.extend_from_slice(phrases[i % phrases.len()]);
            if i.is_multiple_of(11) {
                out.extend_from_slice(format!("<<marker {i} @ {}>> ", out.len()).as_bytes());
            }
            i += 1;
        }
        out.truncate(n);
        out
    }

    /// A binary-heavy generated corpus (~`n` bytes): little-endian record structs
    /// with a slowly-varying key and a repeating payload, i.e. compressible but
    /// with very different statistics from text (few printable bytes, structural
    /// periodicity at the record stride rather than the word stride).
    fn binary_corpus(n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n + 64);
        let payload: [u8; 12] = [
            0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33, 0xFF, 0xEE, 0xDD, 0xCC,
        ];
        let mut key: u32 = 0;
        while out.len() < n {
            out.extend_from_slice(&key.to_le_bytes());
            out.extend_from_slice(&(key.wrapping_mul(2654435761)).to_le_bytes());
            out.extend_from_slice(&payload);
            // Slowly advance the key so records mostly repeat but drift.
            if key.is_multiple_of(5) {
                key = key.wrapping_add(1);
            }
        }
        out.truncate(n);
        out
    }

    /// THE ratio guard (BLOCKING, multi-corpus). On THREE distinct corpora with
    /// genuinely different byte statistics — a text-heavy slice, a binary-heavy
    /// slice, and silesia slices at distinct offsets (different member files) —
    /// the fast-L1 gzip stream MUST be no larger than `gzip -1`. This is the
    /// guard the earlier single-corpus sanity lacked: it missed a 28% blowup that
    /// a per-corpus assert would have caught. Reports fast-L1 vs gzip-1 vs
    /// libdeflate-1 (and a rough in-process compress-time vs libdeflate-1) for
    /// every corpus.
    #[test]
    fn fast_l1_ratio_multi_corpus() {
        // Always-present generated corpora, plus silesia offsets when the tar is
        // available (distinct offsets land in different file types).
        let mut corpora: Vec<(String, Vec<u8>)> = vec![
            ("text-heavy-2MiB".to_string(), text_corpus(2_000_000)),
            ("binary-heavy-2MiB".to_string(), binary_corpus(2_000_000)),
        ];
        for (label, off) in [
            ("silesia@64KiB", 1u64 << 16),
            ("silesia@40MiB", 40 << 20),
            ("silesia@120MiB", 120 << 20),
        ] {
            if let Some(d) = silesia_slice_at(off, 4_000_000) {
                corpora.push((label.to_string(), d));
            }
        }
        // The generated text+binary corpora are always present and carry the
        // guard; silesia slices are additive coverage on machines that have the
        // tar (CI runners don't ship benchmark_data/silesia.tar).
        if corpora.len() < 3 {
            eprintln!(
                "fast_l1_ratio_multi_corpus: silesia.tar unavailable, running on \
                 {} generated corpora only",
                corpora.len()
            );
        }

        let mut failures = Vec::new();
        for (label, data) in &corpora {
            // Rough in-process compress time (non-gated sanity that it's fast).
            let t0 = std::time::Instant::now();
            let ours = compress_gzip(data, 1).len();
            let ours_ms = t0.elapsed().as_secs_f64() * 1e3;

            let tl = std::time::Instant::now();
            let ld1 = libdeflate_gzip_size(data, 1);
            let ld1_ms = tl.elapsed().as_secs_f64() * 1e3;

            let gzip1 = flate2_gzip_size(data, 1);
            // pigz -1 (the stated Lever-1 rival). None when pigz is absent.
            let pigz1 = pigz_gzip_size(data, 1);
            let pigz_str = match pigz1 {
                Some(p) => format!("pigz-1={p} (fastL1/pigz1={:.3})", ours as f64 / p as f64),
                None => "pigz-1=<absent>".to_string(),
            };

            eprintln!(
                "L1 [{label}]: raw={} fastL1={ours} ({:.3}x raw, {ours_ms:.1}ms)  \
                 gzip-1={gzip1} (fastL1/gzip1={:.3})  {pigz_str}  \
                 libdeflate-1={ld1} (fastL1/ld1={:.3}, {ld1_ms:.1}ms)",
                data.len(),
                ours as f64 / data.len() as f64,
                ours as f64 / gzip1 as f64,
                ours as f64 / ld1 as f64,
            );

            // (1) It actually compresses (unless the corpus is incompressible,
            // which none of these are).
            assert!(
                ours < data.len(),
                "L1 [{label}]: output {ours} not smaller than raw {}",
                data.len()
            );
            // (2) THE guard: no larger than gzip -1 on EVERY corpus.
            if ours > gzip1 {
                failures.push(format!(
                    "[{label}] fastL1={ours} > gzip-1={gzip1} ({:.3}x)",
                    ours as f64 / gzip1 as f64
                ));
            }
            // (3) Lever-1 target: fast-L1 must be no larger than pigz -1 on the
            // TEXT corpus — the cell the lever set out to close (widening the
            // single-probe hash 8K->64K flips text from ~1.007x to ~0.97x pigz).
            // Checked only when pigz is on PATH. Silesia is RECORDED (eprintln
            // above) but NOT asserted (mixed content, no single guaranteed
            // relation to pigz-1 across offsets).
            //
            // TIGHTENED 2026-07-22 (HASH3-GATE promotion to the L1 default,
            // commit `7f735755`): the composed HASH3-PROBE + literal-fraction
            // GATE lever closes the chainless-vs-pigz binary gap that used to
            // sit here un-asserted (~4% baseline on the real `dd79_bin6`
            // breadth corpus, see `L1_HASH3_BITS`'s doc comment) — pigz-1 is
            // now a real bound on binary content too, not just text. This
            // generated `binary_corpus` fixture is a different byte
            // distribution from the real breadth file the lever was measured
            // against (it does not literally SURPASS pigz-1 here, unlike the
            // real corpus) but the composed lever brings it to within ~0.3%,
            // down from a ~4% pre-lever deficit — asserted at a tight 1.01x
            // ceiling (not the strict `<=` the TEXT class gets) so this guard
            // still catches a real regression of the gate/probe lever without
            // over-claiming a bound this particular synthetic generator
            // doesn't quite cross.
            if let Some(p) = pigz1 {
                if label.contains("text") && ours > p {
                    failures.push(format!(
                        "[{label}] fastL1={ours} > pigz-1={p} ({:.3}x) — TEXT ratio \
                         target regressed",
                        ours as f64 / p as f64
                    ));
                }
                if label.contains("binary") && (ours as f64) > (p as f64) * 1.01 {
                    failures.push(format!(
                        "[{label}] fastL1={ours} > pigz-1*1.01={:.0} ({:.3}x) — the \
                         HASH3-GATE composed-lever binary ratio target regressed",
                        p as f64 * 1.01,
                        ours as f64 / p as f64
                    ));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "fast-L1 exceeded gzip -1 on {}/{} corpora: {}",
            failures.len(),
            corpora.len(),
            failures.join("; ")
        );
    }

    /// The STORED-block escape must fire on incompressible data: a per-block
    /// dynamic/static Huffman coding of random bytes would EXPAND the input
    /// (>= 1x), so the cheapest-of selector must fall back to a stored block,
    /// keeping the gzip output within a tiny per-block header of the raw size.
    /// (RFC-fixed one-block coding could not escape — this is what the ratio
    /// blowup was.)
    #[test]
    fn fast_l1_incompressible_uses_stored_escape() {
        // A few incompressible sizes spanning the 64 KiB block boundary.
        for &n in &[64 * 1024usize, 200_000, 1 << 20] {
            let mut data = vec![0u8; n];
            Rng::new(0xBADC0DE ^ n as u64).fill(&mut data);

            let gz = compress_gzip(&data, 1);
            // Roundtrips (stored blocks must still decode).
            assert_eq!(
                decode_flate2(&gz),
                data,
                "stored-escape roundtrip failed (n={n})"
            );

            // Stored coding adds ~6 header bytes per <=64 KiB sub-block (~n/5000),
            // so a stored stream is <= ~1.0003x n. A Huffman coding of random
            // bytes would run ~1.05x+ n. A 1.016x ceiling (n + n/64 + 256) cleanly
            // proves the STORED escape fired and not a Huffman expansion.
            let ceiling = n + n / 64 + 256;
            assert!(
                gz.len() <= ceiling,
                "incompressible n={n}: fast-L1 gzip {} exceeds stored ceiling {ceiling} \
                 — the STORED escape did not fire (Huffman-expanded instead)",
                gz.len()
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 240, max_shrink_iters: 4000, ..ProptestConfig::default() })]

        /// Roundtrip the L1 fast path over the adversarial generator: every input
        /// must decode back byte-exact through flate2.
        #[test]
        fn prop_fast_l1_roundtrip_flate2(input in adversarial_bytes()) {
            let gz = compress_gzip(&input, 1);
            let decoded = decode_flate2(&gz);
            prop_assert_eq!(decoded, input);
        }

        /// The stored escape must fire for EVERY random (incompressible) block:
        /// output stays within a small per-block header of the raw size.
        #[test]
        fn prop_fast_l1_incompressible_stored_escape(
            data in proptest::collection::vec(any::<u8>(), 4096..80_000)
        ) {
            let n = data.len();
            let gz = compress_gzip(&data, 1);
            prop_assert_eq!(decode_flate2(&gz), data.clone());
            let ceiling = n + n / 64 + 256;
            prop_assert!(
                gz.len() <= ceiling,
                "n={}: gz {} > stored ceiling {} — stored escape did not fire",
                n, gz.len(), ceiling
            );
        }
    }
}
