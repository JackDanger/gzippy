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
    use gzippy::decompress::inflate::resumable::ResumableInflate2;
    use std::io::{Read, Write};
    use std::time::Instant;

    fn make_deflate(payload: &[u8], level: u32) -> Vec<u8> {
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    /// Best-of-N timing after `warmup` warmup iterations. Both timers
    /// follow the same shape: allocate output buffer outside the timed
    /// region; construct any decoder state outside the timed region;
    /// time only the decode. Advisor item E1-E3: bench harness must be
    /// symmetric.
    fn time_resumable(deflate: &[u8], expected_out: usize, iters: usize, warmup: usize) -> f64 {
        let mut out = vec![0u8; expected_out + 1024];
        // Warmup: prime caches, JIT, allocator.
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
        let mut best_ns_per_byte = f64::MAX;
        for _ in 0..iters {
            // Decoder construction is outside the timed region (mirrors
            // flate2 harness which also constructs outside).
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

    /// English-like text with vocabulary diversity. Compresses to ~50-60%
    /// at L6 — the realistic "Huffman-bound" workload (dynamic-Huffman
    /// blocks with many distinct literal/length codes; moderate match
    /// frequency so the decoder spends time in the Huffman lookup, NOT
    /// just in `copy_match_fast`).
    fn make_text(size_kb: usize) -> Vec<u8> {
        // 280 unique words; cycling produces diverse-enough literal
        // frequencies to force dynamic Huffman with many codes.
        const WORDS: &[&str] = &[
            "the", "of", "and", "to", "a", "in", "for", "is", "on", "that", "by", "this", "with",
            "I", "you", "it", "not", "or", "be", "are", "from", "at", "as", "your", "all", "have",
            "new", "more", "an", "was", "we", "will", "home", "can", "us", "about", "if", "page",
            "my", "has", "search", "free", "but", "our", "one", "other", "do", "no", "information",
            "time", "they", "site", "he", "up", "may", "what", "which", "their", "news", "out",
            "use", "any", "there", "see", "only", "so", "his", "when", "contact", "here", "business",
            "who", "web", "also", "now", "help", "get", "view", "online", "first", "am", "been",
            "would", "how", "were", "me", "services", "some", "these", "click", "its", "like",
            "service", "than", "find", "price", "date", "back", "top", "people", "had", "list",
            "name", "just", "over", "state", "year", "day", "into", "email", "two", "health",
            "world", "next", "used", "go", "work", "last", "most", "products", "music", "buy",
            "data", "make", "them", "should", "product", "system", "post", "her", "city", "add",
            "policy", "number", "such", "please", "available", "copyright", "support", "message",
            "after", "best", "software", "then", "jan", "good", "video", "well", "where", "info",
            "rights", "public", "books", "high", "school", "through", "each", "links", "she",
            "review", "years", "order", "very", "privacy", "book", "items", "company", "read",
            "group", "sex", "need", "many", "user", "said", "de", "does", "set", "under", "general",
            "research", "university", "January", "mail", "full", "map", "reviews", "program", "life",
            "know", "games", "way", "days", "management", "part", "could", "great", "United", "hotel",
            "real", "item", "international", "center", "ebay", "must", "store", "travel", "comments",
            "made", "development", "report", "off", "member", "details", "line", "terms", "before",
            "hotels", "did", "send", "right", "type", "because", "local", "those", "using", "results",
            "office", "education", "national", "car", "design", "take", "posted", "internet", "address",
            "community", "within", "States", "area", "want", "phone", "shipping", "reserved", "subject",
            "between", "forum", "family", "long", "based", "code", "show", "even", "black", "check",
            "special", "prices", "website", "index", "being", "women", "much", "sign", "file", "link",
            "open", "today", "technology", "south", "case", "project", "same", "pages", "version",
            "section", "own", "found", "sports", "house", "related", "security", "both", "county",
            "American", "photo", "game", "members", "power", "while", "care", "network",
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
        println!("Build: pure-rust-inflate, release, target_feature unset (portable)");
        println!("Best-of-{} after {} warmup. Decoder construction outside timed region.", ITERS, WARMUP);
        println!();
        println!(
            "{:<20} {:>12} {:>12} {:>10} {:>12} {:>10} {:>10}",
            "case", "raw_bytes", "deflate_bytes", "compress%", "rust_ns/B", "flate2_ns/B", "ratio"
        );
        println!("{}", "-".repeat(100));

        let mut report = String::new();
        report.push_str("# Tight Huffman Decoder Baseline\n");
        report.push_str(&format!(
            "# Build: pure-rust-inflate, release. Best-of-{} runs after {} warmup.\n\n",
            ITERS, WARMUP
        ));
        report.push_str("| case | raw_bytes | deflate_bytes | compress% | rust_ns/B | flate2_ns/B | ratio (rust/flate2) |\n");
        report.push_str("|---|---:|---:|---:|---:|---:|---:|\n");

        for (label, payload, level) in &cases {
            let deflate = make_deflate(payload, *level);
            let compress_pct = (deflate.len() as f64 / payload.len() as f64) * 100.0;
            let rust_ns_per_byte = time_resumable(&deflate, payload.len(), ITERS, WARMUP);
            let flate2_ns_per_byte = time_flate2(&deflate, payload.len(), ITERS, WARMUP);
            let ratio = rust_ns_per_byte / flate2_ns_per_byte;
            println!(
                "{:<20} {:>12} {:>12} {:>9.1}% {:>12.2} {:>10.2} {:>9.2}x",
                label,
                payload.len(),
                deflate.len(),
                compress_pct,
                rust_ns_per_byte,
                flate2_ns_per_byte,
                ratio,
            );
            report.push_str(&format!(
                "| {} | {} | {} | {:.1}% | {:.2} | {:.2} | {:.2}× |\n",
                label,
                payload.len(),
                deflate.len(),
                compress_pct,
                rust_ns_per_byte,
                flate2_ns_per_byte,
                ratio,
            ));
        }

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
