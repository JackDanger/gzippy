//! Baseline measurement for the tight Huffman decoder work.
//!
//! Compares `ResumableInflate2` (gzippy's pure-Rust resumable inflate)
//! against `flate2` (zlib-ng under the hood) on three workload shapes:
//!
//! - **silesia_text**: a tar-shaped concatenation of the in-tree text
//!   fixtures (`sample.utf8-*`). Literal-heavy; exercises the literal
//!   fast path + dynamic Huffman tables. ~300 KiB.
//! - **silesia_repetitive**: repeating phrases. Match-heavy; exercises
//!   `copy_match_windowed`'s back-reference path.
//! - **synthetic_random**: high-entropy. Mostly stored blocks; exercises
//!   the `resume_decode_stored_resumable` path.
//!
//! Run:
//! ```text
//! cargo bench --features pure-rust-inflate --bench tight_huffman_baseline
//! ```
//!
//! Records baseline numbers in `/tmp/gzippy-tight-huffman-baseline.txt`
//! that subsequent perf landings can A/B against.

#[cfg(feature = "pure-rust-inflate")]
mod bench {
    use gzippy::decompress::inflate::resumable::{
        ResumableInflate2, BODY_RESUMABLE_FASTLOOP_ENTERS, READ_STREAM_BYTES_OUT,
        READ_STREAM_CALLS, READ_STREAM_OUTPUT_BUF_BYTES,
    };
    use std::io::{Read, Write};
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    fn make_deflate(payload: &[u8], level: u32) -> Vec<u8> {
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    /// Best-effort identifier of the active target-cpu for the bench
    /// header. Doesn't query LLVM directly — just reports the rustc
    /// target triple and whether `target-cpu=native` is likely set
    /// (presence of any `target_feature` beyond the baseline).
    fn get_target_cpu() -> &'static str {
        #[cfg(target_feature = "bmi2")]
        return "x86_64+bmi2 (likely native)";
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        return "aarch64+neon (likely native)";
        #[cfg(not(any(
            target_feature = "bmi2",
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        return "generic (no native flags detected)";
    }

    /// Best-of-N timing after `warmup` warmup iterations. Both timers
    /// follow the same shape: allocate output buffer outside the timed
    /// region; construct any decoder state outside the timed region;
    /// time only the decode. Advisor item E1-E3: bench harness must be
    /// symmetric.
    /// Monolithic decode: one giant output buffer, decoder fills in one
    /// call. Favors decoders that benefit from large contiguous writes.
    fn time_resumable_monolithic(
        deflate: &[u8],
        expected_out: usize,
        iters: usize,
        warmup: usize,
    ) -> (f64, u64) {
        let mut out = vec![0u8; expected_out + 1024];
        for _ in 0..warmup {
            let mut decoder =
                ResumableInflate2::with_until_bits(deflate, 0, deflate.len() * 8).unwrap();
            decoder.set_window(&[]).unwrap();
            let mut total = 0usize;
            loop {
                let r = decoder.read_stream(&mut out[total..]).unwrap();
                total += r.bytes_written;
                if r.finished || r.bytes_written == 0 {
                    break;
                }
            }
        }
        let fastloop_before = BODY_RESUMABLE_FASTLOOP_ENTERS.load(Ordering::Relaxed);
        let mut best_ns_per_byte = f64::MAX;
        for _ in 0..iters {
            let mut decoder =
                ResumableInflate2::with_until_bits(deflate, 0, deflate.len() * 8).unwrap();
            decoder.set_window(&[]).unwrap();
            let start = Instant::now();
            let mut total = 0usize;
            loop {
                let r = decoder.read_stream(&mut out[total..]).unwrap();
                total += r.bytes_written;
                if r.finished || r.bytes_written == 0 {
                    break;
                }
            }
            let elapsed_ns = start.elapsed().as_nanos() as f64;
            let ns_per_byte = elapsed_ns / total as f64;
            if ns_per_byte < best_ns_per_byte {
                best_ns_per_byte = ns_per_byte;
            }
        }
        let fastloop_after = BODY_RESUMABLE_FASTLOOP_ENTERS.load(Ordering::Relaxed);
        (best_ns_per_byte, fastloop_after - fastloop_before)
    }

    /// Chunked decode: read into a 64 KiB output window at a time, like
    /// a streaming consumer would. Tests whether the win from monolithic
    /// decode is real or just bench-construction artifact (advisor item B1).
    fn time_resumable_chunked(
        deflate: &[u8],
        expected_out: usize,
        iters: usize,
        warmup: usize,
    ) -> f64 {
        const CHUNK: usize = 64 * 1024;
        let mut out = vec![0u8; CHUNK];
        for _ in 0..warmup {
            let mut decoder =
                ResumableInflate2::with_until_bits(deflate, 0, deflate.len() * 8).unwrap();
            decoder.set_window(&[]).unwrap();
            let mut total = 0usize;
            loop {
                let r = decoder.read_stream(&mut out).unwrap();
                total += r.bytes_written;
                if r.finished {
                    break;
                }
                if r.bytes_written == 0 && total >= expected_out {
                    break;
                }
            }
        }
        let mut best_ns_per_byte = f64::MAX;
        for _ in 0..iters {
            let mut decoder =
                ResumableInflate2::with_until_bits(deflate, 0, deflate.len() * 8).unwrap();
            decoder.set_window(&[]).unwrap();
            let start = Instant::now();
            let mut total = 0usize;
            loop {
                let r = decoder.read_stream(&mut out).unwrap();
                total += r.bytes_written;
                if r.finished {
                    break;
                }
                if r.bytes_written == 0 {
                    break;
                }
            }
            let elapsed_ns = start.elapsed().as_nanos() as f64;
            let ns_per_byte = elapsed_ns / total as f64;
            if ns_per_byte < best_ns_per_byte {
                best_ns_per_byte = ns_per_byte;
            }
        }
        best_ns_per_byte
    }

    fn time_flate2(deflate: &[u8], expected_out: usize, iters: usize, warmup: usize) -> f64 {
        let mut out = vec![0u8; expected_out + 1024];
        for _ in 0..warmup {
            let mut decoder = flate2::read::DeflateDecoder::new(deflate);
            let _ = decoder.read(&mut out).unwrap();
        }
        let mut best_ns_per_byte = f64::MAX;
        for _ in 0..iters {
            // Symmetric with time_resumable: decoder construction outside
            // the timed region. This corrects the previous asymmetry
            // (advisor item E1).
            let mut decoder = flate2::read::DeflateDecoder::new(deflate);
            let start = Instant::now();
            let total = decoder.read(&mut out).unwrap();
            let elapsed_ns = start.elapsed().as_nanos() as f64;
            let ns_per_byte = elapsed_ns / total as f64;
            if ns_per_byte < best_ns_per_byte {
                best_ns_per_byte = ns_per_byte;
            }
        }
        best_ns_per_byte
    }

    /// libdeflate one-shot decode via the safe `libdeflater` wrapper.
    /// Wrap input with gzip header/trailer once outside the timed
    /// region (libdeflater exposes gzip decode, not raw deflate).
    ///
    /// libdeflate is the second vendor with viable arm64 paths
    /// (ISA-L is x86_64-only). Third comparison column — vs flate2
    /// (zlib-ng) and now vs libdeflate.
    fn time_libdeflate(payload: &[u8], iters: usize, warmup: usize) -> f64 {
        use libdeflater::Decompressor;
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(payload).unwrap();
        let gz = enc.finish().unwrap();
        let mut out = vec![0u8; payload.len() + 1024];
        for _ in 0..warmup {
            let mut decoder = Decompressor::new();
            let _ = decoder.gzip_decompress(&gz, &mut out).unwrap();
        }
        let mut best_ns_per_byte = f64::MAX;
        for _ in 0..iters {
            let mut decoder = Decompressor::new();
            let start = Instant::now();
            let n = decoder.gzip_decompress(&gz, &mut out).unwrap();
            let elapsed_ns = start.elapsed().as_nanos() as f64;
            let ns_per_byte = elapsed_ns / n as f64;
            if ns_per_byte < best_ns_per_byte {
                best_ns_per_byte = ns_per_byte;
            }
        }
        best_ns_per_byte
    }

    /// English-like text with vocabulary diversity. Compresses to ~50-60%
    /// at L6 — the realistic "Huffman-bound" workload (dynamic-Huffman
    /// blocks with many distinct literal/length codes; moderate match
    /// frequency so the decoder spends time in the Huffman lookup, NOT
    /// just in `copy_match_fast`).
    fn make_text(size_kb: usize) -> Vec<u8> {
        // 280 unique words; cycling produces diverse-enough literal
        // frequencies to force dynamic Huffman with many codes.
        const WORDS: &[&str] = &[
            "the",
            "of",
            "and",
            "to",
            "a",
            "in",
            "for",
            "is",
            "on",
            "that",
            "by",
            "this",
            "with",
            "I",
            "you",
            "it",
            "not",
            "or",
            "be",
            "are",
            "from",
            "at",
            "as",
            "your",
            "all",
            "have",
            "new",
            "more",
            "an",
            "was",
            "we",
            "will",
            "home",
            "can",
            "us",
            "about",
            "if",
            "page",
            "my",
            "has",
            "search",
            "free",
            "but",
            "our",
            "one",
            "other",
            "do",
            "no",
            "information",
            "time",
            "they",
            "site",
            "he",
            "up",
            "may",
            "what",
            "which",
            "their",
            "news",
            "out",
            "use",
            "any",
            "there",
            "see",
            "only",
            "so",
            "his",
            "when",
            "contact",
            "here",
            "business",
            "who",
            "web",
            "also",
            "now",
            "help",
            "get",
            "view",
            "online",
            "first",
            "am",
            "been",
            "would",
            "how",
            "were",
            "me",
            "services",
            "some",
            "these",
            "click",
            "its",
            "like",
            "service",
            "than",
            "find",
            "price",
            "date",
            "back",
            "top",
            "people",
            "had",
            "list",
            "name",
            "just",
            "over",
            "state",
            "year",
            "day",
            "into",
            "email",
            "two",
            "health",
            "world",
            "next",
            "used",
            "go",
            "work",
            "last",
            "most",
            "products",
            "music",
            "buy",
            "data",
            "make",
            "them",
            "should",
            "product",
            "system",
            "post",
            "her",
            "city",
            "add",
            "policy",
            "number",
            "such",
            "please",
            "available",
            "copyright",
            "support",
            "message",
            "after",
            "best",
            "software",
            "then",
            "jan",
            "good",
            "video",
            "well",
            "where",
            "info",
            "rights",
            "public",
            "books",
            "high",
            "school",
            "through",
            "each",
            "links",
            "she",
            "review",
            "years",
            "order",
            "very",
            "privacy",
            "book",
            "items",
            "company",
            "read",
            "group",
            "sex",
            "need",
            "many",
            "user",
            "said",
            "de",
            "does",
            "set",
            "under",
            "general",
            "research",
            "university",
            "January",
            "mail",
            "full",
            "map",
            "reviews",
            "program",
            "life",
            "know",
            "games",
            "way",
            "days",
            "management",
            "part",
            "could",
            "great",
            "United",
            "hotel",
            "real",
            "item",
            "international",
            "center",
            "ebay",
            "must",
            "store",
            "travel",
            "comments",
            "made",
            "development",
            "report",
            "off",
            "member",
            "details",
            "line",
            "terms",
            "before",
            "hotels",
            "did",
            "send",
            "right",
            "type",
            "because",
            "local",
            "those",
            "using",
            "results",
            "office",
            "education",
            "national",
            "car",
            "design",
            "take",
            "posted",
            "internet",
            "address",
            "community",
            "within",
            "States",
            "area",
            "want",
            "phone",
            "shipping",
            "reserved",
            "subject",
            "between",
            "forum",
            "family",
            "long",
            "based",
            "code",
            "show",
            "even",
            "black",
            "check",
            "special",
            "prices",
            "website",
            "index",
            "being",
            "women",
            "much",
            "sign",
            "file",
            "link",
            "open",
            "today",
            "technology",
            "south",
            "case",
            "project",
            "same",
            "pages",
            "version",
            "section",
            "own",
            "found",
            "sports",
            "house",
            "related",
            "security",
            "both",
            "county",
            "American",
            "photo",
            "game",
            "members",
            "power",
            "while",
            "care",
            "network",
        ];
        let mut data = Vec::with_capacity(size_kb * 1024);
        let mut i = 0usize;
        let mut line_len = 0usize;
        while data.len() < size_kb * 1024 {
            let w = WORDS[i % WORDS.len()];
            i = i.wrapping_add(1).wrapping_mul(2654435761);
            if line_len + w.len() + 1 > 72 {
                data.push(b'\n');
                line_len = 0;
            } else if line_len > 0 {
                data.push(b' ');
                line_len += 1;
            }
            data.extend_from_slice(w.as_bytes());
            line_len += w.len();
        }
        data.truncate(size_kb * 1024);
        data
    }

    /// Repetitive but with enough variation to force dynamic Huffman
    /// rather than degenerate-to-stored. Compresses to ~10-20% — match-
    /// heavy but with real Huffman decoding interleaved.
    fn make_repetitive(size_kb: usize) -> Vec<u8> {
        const PHRASES: &[&[u8]] = &[
            b"the quick brown fox jumps over the lazy dog ",
            b"pack my box with five dozen liquor jugs ",
            b"how vexingly quick daft zebras jump ",
            b"sphinx of black quartz judge my vow ",
        ];
        let mut data = Vec::with_capacity(size_kb * 1024);
        let mut i = 0usize;
        while data.len() < size_kb * 1024 {
            data.extend_from_slice(PHRASES[i & 3]);
            i += 1;
        }
        data.truncate(size_kb * 1024);
        data
    }

    /// Mixed-entropy: 60% pseudo-text, 40% PRNG bytes. Forces zlib L6
    /// into emitting a stream of dynamic-Huffman blocks at varied
    /// densities. The "realistic web traffic" shape.
    fn make_mixed(size_kb: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size_kb * 1024);
        let mut rng: u64 = 0xfeed_face;
        while data.len() < size_kb * 1024 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                // ASCII printable
                let c = ((rng >> 16) % 95 + 32) as u8;
                data.push(c);
            } else {
                // Short repeat
                let len = ((rng >> 24) % 4 + 2) as usize;
                let b = ((rng >> 32) % 26 + b'a' as u64) as u8;
                for _ in 0..len.min(size_kb * 1024 - data.len()) {
                    data.push(b);
                }
            }
        }
        data.truncate(size_kb * 1024);
        data
    }

    fn make_random(size_kb: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size_kb * 1024);
        let mut rng: u64 = 0xcafef00d_deadbeef;
        for _ in 0..size_kb * 1024 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((rng >> 24) as u8);
        }
        data
    }

    pub fn run() {
        const ITERS: usize = 30; // advisor item E6: best-of-30 after warmup
        const WARMUP: usize = 5;

        // Cases: small + large variants. Small fits L1+L2 (best-case);
        // large forces L2/L3 misses and reflects realistic workloads.
        // (Advisor item E5: 524 KiB alone hides cache-miss costs.)
        let cases: Vec<(&str, Vec<u8>, u32)> = vec![
            // Huffman-dominated text — small (fits L2).
            ("text_words", make_text(512), 6),
            // Same shape — large (forces L2/L3 misses).
            ("text_words_8MiB", make_text(8 * 1024), 6),
            // L1 produces fixed-Huffman blocks for small inputs;
            // exercises T2's fixed-table specialization target.
            ("text_fixed_L1", make_text(512), 1),
            // Mixed entropy (~57% compression).
            ("mixed", make_mixed(512), 6),
            ("mixed_8MiB", make_mixed(8 * 1024), 6),
            // Match-heavy — match-copy dominated (not Huffman).
            ("repetitive", make_repetitive(512), 6),
            // Stored-block dominated.
            ("random", make_random(512), 1),
        ];

        println!("\n=== Tight Huffman Decoder Baseline ===");
        println!(
            "Build: pure-rust-inflate, release, target-cpu={}",
            get_target_cpu()
        );
        println!(
            "Best-of-{} after {} warmup. Decoder construction outside timed region.",
            ITERS, WARMUP
        );
        println!(
            "Three vendors: flate2 (zlib-ng), libdeflate (one-shot). Our decoder: MONO + CHUNKED."
        );
        println!();
        println!(
            "{:<18} {:>9} {:>8} {:>9} {:>9} {:>6} {:>9} {:>6} {:>9} {:>6}",
            "case",
            "raw",
            "compr%",
            "rust_mono",
            "rust_chunk",
            "vs_z_m",
            "flate2",
            "vs_z_c",
            "libdef",
            "vs_ld"
        );
        println!("{}", "-".repeat(110));

        let mut report = String::new();
        report.push_str("# Tight Huffman Decoder Baseline\n");
        report.push_str(&format!(
            "# Build: pure-rust-inflate, target-cpu={}. Best-of-{} runs after {} warmup.\n\n",
            get_target_cpu(),
            ITERS,
            WARMUP
        ));
        report.push_str("| case | raw | compr% | rust_mono | rust_chunk | vs zlib mono | flate2 | vs zlib chunk | libdef | vs libdef |\n");
        report.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");

        BODY_RESUMABLE_FASTLOOP_ENTERS.store(0, Ordering::Relaxed);
        let mut total_fastloop_entries: u64 = 0;
        for (label, payload, level) in &cases {
            let deflate = make_deflate(payload, *level);
            let compress_pct = (deflate.len() as f64 / payload.len() as f64) * 100.0;
            let (mono_ns, fastloop_delta) =
                time_resumable_monolithic(&deflate, payload.len(), ITERS, WARMUP);
            let chunk_ns = time_resumable_chunked(&deflate, payload.len(), ITERS, WARMUP);
            let flate2_ns = time_flate2(&deflate, payload.len(), ITERS, WARMUP);
            let libdef_ns = time_libdeflate(payload, ITERS, WARMUP);
            let vs_z_m = mono_ns / flate2_ns;
            let vs_z_c = chunk_ns / flate2_ns;
            let vs_ld = chunk_ns / libdef_ns;
            total_fastloop_entries += fastloop_delta;
            println!(
                "{:<18} {:>9} {:>7.1}% {:>9.2} {:>9.2} {:>5.2}x {:>9.2} {:>5.2}x {:>9.2} {:>5.2}x",
                label,
                payload.len(),
                compress_pct,
                mono_ns,
                chunk_ns,
                vs_z_m,
                flate2_ns,
                vs_z_c,
                libdef_ns,
                vs_ld
            );
            report.push_str(&format!(
                "| {} | {} | {:.1}% | {:.2} | {:.2} | {:.2}× | {:.2} | {:.2}× | {:.2} | {:.2}× |\n",
                label,
                payload.len(),
                compress_pct,
                mono_ns,
                chunk_ns,
                vs_z_m,
                flate2_ns,
                vs_z_c,
                libdef_ns,
                vs_ld
            ));
        }

        // Per advisor item B7: FASTLOOP must fire across the corpus. If
        // not, the bench is meaningless because we're measuring the
        // safe-loop path, not the optimized fastloop.
        println!();
        println!(
            "FASTLOOP entries across all cases: {} (must be > 0; else the bench is meaningless)",
            total_fastloop_entries
        );
        assert!(
            total_fastloop_entries > 0,
            "BODY_RESUMABLE_FASTLOOP_ENTERS = 0 — fastloop never fired across {} cases. \
             Either the corpus is too small (no case > FASTLOOP_MARGIN), or the fastloop \
             entry condition regressed.",
            cases.len()
        );

        // Step 2.5: read_stream calling-shape diagnostic. Counts
        // accumulate across MONO + CHUNKED + warmups for all cases —
        // so this is "across the whole bench run", not per-case.
        // Direction-of-travel only; absolute numbers reflect the bench
        // mix, not production.
        let calls = READ_STREAM_CALLS.load(Ordering::Relaxed);
        let bytes_out = READ_STREAM_BYTES_OUT.load(Ordering::Relaxed);
        let buf_bytes = READ_STREAM_OUTPUT_BUF_BYTES.load(Ordering::Relaxed);
        let avg_out = if calls > 0 { bytes_out / calls } else { 0 };
        let avg_buf = if calls > 0 { buf_bytes / calls } else { 0 };
        let fill_pct = if buf_bytes > 0 {
            (bytes_out as f64 / buf_bytes as f64) * 100.0
        } else {
            0.0
        };
        println!();
        println!("read_stream calling-shape (bench-wide, mono+chunked+warmups combined):");
        println!("  total calls:     {}", calls);
        println!(
            "  avg bytes/call:  {} ({:.1} KiB)",
            avg_out,
            avg_out as f64 / 1024.0
        );
        println!(
            "  avg buffer/call: {} ({:.1} KiB)",
            avg_buf,
            avg_buf as f64 / 1024.0
        );
        println!("  buffer fill %:   {:.1}%", fill_pct);
        println!(
            "  (Step 2.5: for production-shape characterization, run \
             `cargo test --release --features pure-rust-inflate -- \
             step25 --ignored --nocapture` on a host where parallel-SM \
             is gated on, i.e. x86_64.)"
        );

        let _ = std::fs::write("/tmp/gzippy-tight-huffman-baseline.txt", &report);
        println!();
        println!("Report saved to /tmp/gzippy-tight-huffman-baseline.txt");
        println!();
        println!("Reading the ratio:");
        println!("  1.0×  = pure-Rust matches flate2 (zlib-ng)");
        println!("  >1.0× = pure-Rust slower (e.g. 2.0× = takes twice as long per byte)");
        println!("  <1.0× = pure-Rust faster");
        println!();
        println!("flate2 with default features uses zlib-ng. ISA-L is typically 1.5-2×");
        println!("faster than zlib-ng on text; matching flate2 is the floor, beating it");
        println!("is the target.");
    }
}

#[cfg(feature = "pure-rust-inflate")]
fn main() {
    bench::run();
}

#[cfg(not(feature = "pure-rust-inflate"))]
fn main() {
    eprintln!("tight_huffman_baseline requires --features pure-rust-inflate");
}
