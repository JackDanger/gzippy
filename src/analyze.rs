//! --analyze: an ANSI compression fingerprint that explains itself.
//!
//! Runs greedy LZ77 with a 32 KiB window and byte-verified matches — the
//! same decisions deflate's fast path makes — then prints:
//!
//!   1. a stat card with qualitative labels (HIGH, LOW, EXCELLENT, ...),
//!   2. a position-aware canvas with an offset ruler, where glyph = LZ77
//!      match density and colour = mean byte value (blue→red rainbow),
//!   3. a byte-value colour legend so cells on the canvas are decodable,
//!   4. a match-length histogram ("how long is each reused sequence?"),
//!   5. a back-reference-distance histogram ("how far back do matches
//!      reach?"),
//!   6. a plain-English interpretation paragraph.
//!
//! Documented in gzippy(1) under EASTER EGGS.

use crate::cli::GzippyArgs;
use std::io::{self, Read, Write};

pub fn maybe_run(args: &GzippyArgs) -> Option<i32> {
    if args.analyze {
        Some(run_analyze(args))
    } else {
        None
    }
}

fn run_analyze(args: &GzippyArgs) -> i32 {
    let source = match load_input(args) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("gzippy: --analyze: {}", e);
            return 1;
        }
    };

    let (raw, compressed_len) = match maybe_decompress(&source) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("gzippy: --analyze: decompression failed: {}", e);
            return 1;
        }
    };

    if raw.is_empty() {
        eprintln!("gzippy: --analyze: empty input");
        return 1;
    }

    let (cols, rows) = if args.analyze_full {
        let (c, r) = term_size().unwrap_or((80, 24));
        (c.clamp(60, 200), r.saturating_sub(24).clamp(10, 100))
    } else {
        (80usize, 20usize)
    };

    let analysis = analyze(&raw);

    let mut out = io::BufWriter::new(io::stdout().lock());
    let name = args
        .files
        .first()
        .map(String::as_str)
        .filter(|s| *s != "-")
        .unwrap_or("(stdin)");
    render_banner(&mut out, name, raw.len());
    render_stats(&mut out, &raw, &analysis, compressed_len);
    render_canvas(&mut out, &raw, &analysis, cols, rows);
    render_color_legend(&mut out);
    render_length_histogram(&mut out, &analysis);
    render_distance_histogram(&mut out, &analysis);
    render_verdict(&mut out, &raw, &analysis, compressed_len);
    let _ = out.flush();
    0
}

// ── input plumbing ────────────────────────────────────────────────────────

fn load_input(args: &GzippyArgs) -> io::Result<Vec<u8>> {
    match args.files.first().map(String::as_str) {
        None | Some("-") => {
            let mut buf = Vec::new();
            io::stdin().lock().read_to_end(&mut buf)?;
            Ok(buf)
        }
        Some(path) => std::fs::read(path),
    }
}

fn maybe_decompress(data: &[u8]) -> io::Result<(Vec<u8>, Option<usize>)> {
    if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
        return Ok((data.to_vec(), None));
    }
    use flate2::read::MultiGzDecoder;
    let mut decoder = MultiGzDecoder::new(data);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out)?;
    Ok((out, Some(data.len())))
}

// ── analysis ──────────────────────────────────────────────────────────────

struct Analysis {
    entropy: f64,
    peak_match: usize,
    total_match_bytes: u64,
    total_match_distance: u64,
    n_matches: u64,
    /// Per-byte: 1 if the byte lies inside a back-reference of length >=3.
    in_match: Vec<u8>,
    /// Counts of matches by length bucket: 3-4, 5-8, 9-16, 17-32, 33-64, 65-128, 129-258.
    len_buckets: [u64; 7],
    /// Counts of matches by distance bucket: 1-8, 9-64, 65-512, 513-4K, 4K-32K.
    dist_buckets: [u64; 5],
}

/// Greedy LZ77 with a 32 KiB window — deflate's L1 matcher, roughly.
fn analyze(raw: &[u8]) -> Analysis {
    const WINDOW: usize = 32 * 1024;
    const MAX_MATCH: usize = 258;
    const HASH_BITS: u32 = 15;
    const HASH_SIZE: usize = 1 << HASH_BITS;

    let mut table = vec![u32::MAX; HASH_SIZE];
    let mut in_match = vec![0u8; raw.len()];
    let mut len_buckets = [0u64; 7];
    let mut dist_buckets = [0u64; 5];
    let mut total_match_bytes = 0u64;
    let mut total_match_distance = 0u64;
    let mut n_matches = 0u64;
    let mut peak = 0usize;

    let mut i = 0usize;
    while i + 2 < raw.len() {
        let trigram = ((raw[i] as u32) << 16) | ((raw[i + 1] as u32) << 8) | raw[i + 2] as u32;
        let h = trigram
            .wrapping_mul(0x1E35_A7BD)
            .wrapping_shr(32 - HASH_BITS) as usize;
        let prev = table[h];
        table[h] = i as u32;

        if prev != u32::MAX {
            let p = prev as usize;
            let dist = i - p;
            if (3..WINDOW).contains(&dist)
                && raw[p] == raw[i]
                && raw[p + 1] == raw[i + 1]
                && raw[p + 2] == raw[i + 2]
            {
                let mut len = 3usize;
                let cap = MAX_MATCH.min(raw.len() - i).min(raw.len() - p);
                while len < cap && raw[p + len] == raw[i + len] {
                    len += 1;
                }
                for slot in in_match.iter_mut().skip(i).take(len) {
                    *slot = 1;
                }
                len_buckets[len_bucket(len)] += 1;
                dist_buckets[dist_bucket(dist)] += 1;
                total_match_bytes += len as u64;
                total_match_distance += dist as u64;
                n_matches += 1;
                peak = peak.max(len);
                i += len;
                continue;
            }
        }
        i += 1;
    }

    Analysis {
        entropy: shannon_entropy(raw),
        peak_match: peak,
        total_match_bytes,
        total_match_distance,
        n_matches,
        in_match,
        len_buckets,
        dist_buckets,
    }
}

fn len_bucket(len: usize) -> usize {
    match len {
        3..=4 => 0,
        5..=8 => 1,
        9..=16 => 2,
        17..=32 => 3,
        33..=64 => 4,
        65..=128 => 5,
        _ => 6,
    }
}

fn dist_bucket(dist: usize) -> usize {
    match dist {
        0..=8 => 0,
        9..=64 => 1,
        65..=512 => 2,
        513..=4096 => 3,
        _ => 4,
    }
}

fn shannon_entropy(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    let n = bytes.len() as f64;
    let mut h = 0.0;
    for &c in counts.iter() {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.log2();
        }
    }
    h
}

// ── rendering ─────────────────────────────────────────────────────────────

fn render_banner(out: &mut impl Write, name: &str, bytes: usize) {
    let rule: String = "─".repeat(78);
    let _ = writeln!(
        out,
        "\x1b[1mgzippy --analyze\x1b[0m  {}  ({})",
        name,
        human_bytes(bytes as u64)
    );
    let _ = writeln!(out, "{}", rule);
}

/// Big human-facing stat block. Each row: numeric, a 10-char bar, a
/// qualitative label, and a plain-English hint.
fn render_stats(out: &mut impl Write, raw: &[u8], a: &Analysis, compressed_len: Option<usize>) {
    let cover = a.total_match_bytes as f64 / raw.len().max(1) as f64;
    let (ent_label, ent_hint) = entropy_label(a.entropy);
    let (cov_label, cov_hint) = cover_label(cover);
    let avg_len = if a.n_matches == 0 {
        0.0
    } else {
        a.total_match_bytes as f64 / a.n_matches as f64
    };
    let avg_dist = if a.n_matches == 0 {
        0.0
    } else {
        a.total_match_distance as f64 / a.n_matches as f64
    };

    let _ = writeln!(
        out,
        "  entropy    {}  {:>5.2}/8   {:<10}  ({})",
        bar(a.entropy, 8.0, 10),
        a.entropy,
        ent_label,
        ent_hint
    );
    let _ = writeln!(
        out,
        "  LZ77 cover {}  {:>5.1}%    {:<10}  ({})",
        bar(cover, 1.0, 10),
        cover * 100.0,
        cov_label,
        cov_hint
    );
    if a.n_matches > 0 {
        let (match_count, count_unit) = human_count(a.n_matches);
        let _ = writeln!(
            out,
            "  matches    {:.2}{}  avg length {:.1} B  avg back-distance {}",
            match_count,
            count_unit,
            avg_len,
            human_bytes(avg_dist as u64),
        );
    }
    let (ratio_line, ratio_hint) = ratio_outlook(cover, compressed_len, raw.len());
    let _ = writeln!(out, "  {}  ({})", ratio_line, ratio_hint);
    let _ = writeln!(out);
}

/// 10-character progress bar using 1/8-block characters.
fn bar(value: f64, max: f64, width: usize) -> String {
    let frac = (value / max).clamp(0.0, 1.0);
    let eighths = (frac * width as f64 * 8.0).round() as usize;
    let full = eighths / 8;
    let part = eighths % 8;
    let partial = [
        '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}',
        '\u{2588}',
    ];
    let mut s = String::new();
    s.push('[');
    for _ in 0..full {
        s.push('\u{2588}');
    }
    if part > 0 && full < width {
        s.push(partial[part - 1]);
        for _ in 0..(width - full - 1) {
            s.push(' ');
        }
    } else {
        for _ in 0..(width - full) {
            s.push(' ');
        }
    }
    s.push(']');
    s
}

fn entropy_label(e: f64) -> (&'static str, &'static str) {
    if e < 2.5 {
        ("VERY LOW", "near-constant — RLE would do better than gzip")
    } else if e < 4.5 {
        ("LOW", "text, source code, or structured logs")
    } else if e < 6.5 {
        ("MEDIUM", "mixed text and binary, or a data file")
    } else if e < 7.8 {
        ("HIGH", "compiled binary, image, or packed data")
    } else {
        ("MAX", "random or already compressed — gzip will not help")
    }
}

fn cover_label(c: f64) -> (&'static str, &'static str) {
    if c < 0.05 {
        ("NONE", "nothing repeats — probably random or encrypted")
    } else if c < 0.30 {
        ("LOW", "only short sequences repeat")
    } else if c < 0.60 {
        ("MEDIUM", "moderate reuse — expect a modest shrink")
    } else if c < 0.85 {
        ("HIGH", "lots of reuse — gzip will do well")
    } else {
        (
            "EXTREME",
            "most bytes come for free — this file is very squishy",
        )
    }
}

/// "ratio" line of the stats card. If the input was gzip, show the real
/// observed ratio. Otherwise project a rough band from the LZ77 cover.
fn ratio_outlook(
    cover: f64,
    compressed_len: Option<usize>,
    raw_len: usize,
) -> (String, &'static str) {
    if let Some(gz) = compressed_len {
        let pct = 100.0 * gz as f64 / raw_len.max(1) as f64;
        (
            format!(
                "gz file   {}  is {:>5.1}% of raw ({} → {})",
                bar(pct / 100.0, 1.0, 10),
                pct,
                human_bytes(gz as u64),
                human_bytes(raw_len as u64),
            ),
            "observed ratio of the input you handed us",
        )
    } else {
        // Empirical bands mapping LZ77 cover → typical deflate ratio.
        let (lo, hi) = if cover < 0.05 {
            (96.0, 100.0)
        } else if cover < 0.30 {
            (70.0, 92.0)
        } else if cover < 0.60 {
            (50.0, 70.0)
        } else if cover < 0.85 {
            (25.0, 50.0)
        } else {
            (8.0, 25.0)
        };
        let mid = (lo + hi) / 2.0;
        (
            format!(
                "est. gzip {}  ~{:.0}% of raw (range {:.0}–{:.0}%)",
                bar(mid / 100.0, 1.0, 10),
                mid,
                lo,
                hi
            ),
            "rough projection from cover; run `gzippy -l` for the real number",
        )
    }
}

/// Canvas with a left-side offset ruler. Each cell glyph encodes match
/// density; each cell colour encodes mean byte value via a monotonic
/// blue→red rainbow, so similar colours mean similar byte values.
fn render_canvas(out: &mut impl Write, raw: &[u8], a: &Analysis, cols: usize, rows: usize) {
    let prefix_w = 7; // "xxx.xU "
    let canvas_cols = cols.saturating_sub(prefix_w + 2).max(40);
    let cells = canvas_cols * rows;
    let bytes_per_cell = raw.len().div_ceil(cells).max(1);
    let bytes_per_row = bytes_per_cell * canvas_cols;

    let _ = writeln!(
        out,
        "  canvas   each cell = {}, left ruler shows byte offset in the file",
        human_bytes(bytes_per_cell as u64)
    );

    for row in 0..rows {
        let row_offset = row * bytes_per_row;
        if row_offset >= raw.len() {
            break;
        }
        let label = human_offset(row_offset as u64);
        let _ = write!(out, "{:>6} │ ", label);
        for col in 0..canvas_cols {
            let idx = row * canvas_cols + col;
            let start = idx * bytes_per_cell;
            if start >= raw.len() {
                let _ = write!(out, " ");
                continue;
            }
            let end = (start + bytes_per_cell).min(raw.len());
            let chunk = &raw[start..end];

            let mean: u32 =
                chunk.iter().map(|&b| b as u32).sum::<u32>() / chunk.len().max(1) as u32;
            let covered: u32 = a.in_match[start..end].iter().map(|&b| b as u32).sum();
            let density = covered as f64 / chunk.len() as f64;

            let ansi = byte_to_xterm(mean as u8);
            let glyph = density_glyph(density);
            let _ = write!(out, "\x1b[38;5;{}m{}", ansi, glyph);
        }
        let _ = writeln!(out, "\x1b[0m");
    }
    let _ = writeln!(out);
}

/// Monotonic blue → cyan → green → yellow → red rainbow over byte value.
fn byte_to_xterm(byte: u8) -> u8 {
    let x = byte as f64 / 255.0;
    let (r, g, b) = if x < 0.25 {
        (0.0, x * 4.0, 1.0)
    } else if x < 0.5 {
        (0.0, 1.0, 1.0 - (x - 0.25) * 4.0)
    } else if x < 0.75 {
        ((x - 0.5) * 4.0, 1.0, 0.0)
    } else {
        (1.0, 1.0 - (x - 0.75) * 4.0, 0.0)
    };
    let qr = (r * 5.0).round() as u8;
    let qg = (g * 5.0).round() as u8;
    let qb = (b * 5.0).round() as u8;
    16 + 36 * qr + 6 * qg + qb
}

/// Density glyph ramp biased toward the 70–100% range where most files sit.
fn density_glyph(density: f64) -> char {
    if density < 0.05 {
        ' '
    } else if density < 0.25 {
        '\u{00B7}'
    } else if density < 0.50 {
        '\u{2591}'
    } else if density < 0.70 {
        '\u{2592}'
    } else if density < 0.92 {
        '\u{2593}'
    } else {
        '\u{2588}'
    }
}

fn render_color_legend(out: &mut impl Write) {
    // 50-cell gradient; ticks at 0, 12, 25, 37, 50.
    let width = 50usize;
    let _ = writeln!(out, "  colour in the canvas = mean byte value:");
    let _ = write!(out, "  0x00 ");
    for i in 0..width {
        let b = (i as f64 / (width - 1) as f64 * 255.0).round() as u8;
        let _ = write!(out, "\x1b[38;5;{}m\u{2588}", byte_to_xterm(b));
    }
    let _ = writeln!(out, "\x1b[0m 0xFF");
    let _ = writeln!(
        out,
        "       └─ zeros ─┘ └─ ASCII ─┘ └─ text ─┘ └─ hi-bit ─┘"
    );
    let _ = writeln!(
        out,
        "       0x00       0x40        0x80       0xC0         0xFF"
    );
    let _ = writeln!(out);
}

fn render_length_histogram(out: &mut impl Write, a: &Analysis) {
    let labels = [
        " 3 -   4 B",
        " 5 -   8 B",
        " 9 -  16 B",
        "17 -  32 B",
        "33 -  64 B",
        "65 - 128 B",
        "129- 258 B",
    ];
    let hints = [
        "literal repeats; common everywhere",
        "short phrases; instruction bundles",
        "words, variable names, small keys",
        "lines of code, struct layouts",
        "function prologues, boilerplate",
        "long runs; uncommon but impactful",
        "deflate's hard cap (258 bytes)",
    ];
    let _ = writeln!(
        out,
        "  match-length histogram — how long is each reused sequence?"
    );
    render_bucket_rows(out, &a.len_buckets, &labels, &hints, a.n_matches);
    let _ = writeln!(out);
}

fn render_distance_histogram(out: &mut impl Write, a: &Analysis) {
    let labels = [
        "  1 -   8 B",
        "  9 -  64 B",
        " 65 - 512 B",
        "513 -  4 KB",
        "  4 - 32 KB",
    ];
    let hints = [
        "adjacent bytes; \"aaaa\", tight loops",
        "within a line or small struct",
        "within a function or paragraph",
        "across a file section",
        "long-range reuse across the file",
    ];
    let _ = writeln!(
        out,
        "  back-reference-distance histogram — how far back do matches reach?"
    );
    render_bucket_rows(out, &a.dist_buckets, &labels, &hints, a.n_matches);
    let _ = writeln!(out);
}

fn render_bucket_rows(
    out: &mut impl Write,
    buckets: &[u64],
    labels: &[&str],
    hints: &[&str],
    total: u64,
) {
    let total = total.max(1);
    let max_bar = 22usize;
    for (i, &count) in buckets.iter().enumerate() {
        let pct = 100.0 * count as f64 / total as f64;
        let blocks = ((pct / 100.0) * max_bar as f64 * 8.0).round() as usize;
        let full = blocks / 8;
        let part = blocks % 8;
        let partial = [
            '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}',
            '\u{2588}',
        ];
        let mut bar = String::new();
        for _ in 0..full {
            bar.push('\u{2588}');
        }
        if part > 0 && full < max_bar {
            bar.push(partial[part - 1]);
        }
        let _ = writeln!(
            out,
            "    {}  {:<max_bar$}  {:>5.1}%  {}",
            labels[i],
            bar,
            pct,
            hints[i],
            max_bar = max_bar,
        );
    }
}

/// Plain-English interpretation. The rules are pattern-matching on a small
/// grid of entropy × coverage; deliberately conservative — this is a
/// fingerprint, not a file-type oracle.
fn render_verdict(out: &mut impl Write, raw: &[u8], a: &Analysis, compressed_len: Option<usize>) {
    let cover = a.total_match_bytes as f64 / raw.len().max(1) as f64;
    let avg_dist = if a.n_matches == 0 {
        0.0
    } else {
        a.total_match_distance as f64 / a.n_matches as f64
    };

    let nature = if a.entropy > 7.8 && cover < 0.05 {
        "random, encrypted, or already-compressed data."
    } else if a.entropy < 2.5 {
        "nearly constant data — huge runs of one byte."
    } else if a.entropy < 4.5 && cover > 0.7 {
        "text, source code, or structured logs — deflate's sweet spot."
    } else if a.entropy > 6.5 && cover > 0.6 {
        "a compiled binary, media container, or other structured binary."
    } else if cover < 0.2 {
        "data with little internal repetition."
    } else {
        "structured data of some kind."
    };

    // Only characterise reach when there are enough matches to make the
    // statistic meaningful. A handful of hash-collision hits on random data
    // would otherwise get described as "global duplication."
    let reach_significant = a.n_matches >= 100 && cover > 0.05;
    let reach = if !reach_significant {
        String::new()
    } else if avg_dist < 64.0 {
        "Matches are very local — reuse happens between neighbouring bytes.".to_string()
    } else if avg_dist < 512.0 {
        "Matches live within a function or small section.".to_string()
    } else if avg_dist < 4096.0 {
        "Matches span the file section — phrase-level reuse.".to_string()
    } else {
        "Matches reach across the whole file — global duplication.".to_string()
    };

    let cap = if a.peak_match >= 258 {
        "Peak match hit deflate's 258-byte cap, so some region is heavily duplicated."
    } else if a.peak_match >= 64 {
        "The longest match is solid but short of the 258-byte cap."
    } else if a.peak_match >= 16 {
        "Only short matches — LZ77 will help, but Huffman coding does the rest."
    } else {
        "No long matches found; almost nothing repeats."
    };

    let _ = writeln!(out, "  reading this fingerprint:");
    let _ = writeln!(out, "    • shape   {}", nature);
    if !reach.is_empty() {
        let _ = writeln!(out, "    • reach   {}", reach);
    }
    let _ = writeln!(out, "    • peak    {}", cap);
    if compressed_len.is_some() {
        let _ = writeln!(
            out,
            "    • ratio   real compression ratio is shown above (observed, not estimated)."
        );
    }
}

// ── formatting helpers ────────────────────────────────────────────────────

fn human_bytes(n: u64) -> String {
    const K: u64 = 1024;
    if n < K {
        format!("{} B", n)
    } else if n < K * K {
        format!("{:.1} KB", n as f64 / K as f64)
    } else if n < K * K * K {
        format!("{:.1} MB", n as f64 / (K * K) as f64)
    } else {
        format!("{:.2} GB", n as f64 / (K * K * K) as f64)
    }
}

fn human_offset(n: u64) -> String {
    const K: u64 = 1024;
    if n < K * 10 {
        format!("{}B", n)
    } else if n < K * K * 10 {
        format!("{:.0}K", n as f64 / K as f64)
    } else if n < K * K * K * 10 {
        format!("{:.1}M", n as f64 / (K * K) as f64)
    } else {
        format!("{:.1}G", n as f64 / (K * K * K) as f64)
    }
}

fn human_count(n: u64) -> (f64, &'static str) {
    if n < 1_000 {
        (n as f64, "")
    } else if n < 1_000_000 {
        (n as f64 / 1_000.0, "K")
    } else {
        (n as f64 / 1_000_000.0, "M")
    }
}

fn term_size() -> Option<(usize, usize)> {
    let cols = std::env::var("COLUMNS").ok()?.parse::<usize>().ok()?;
    let rows = std::env::var("LINES").ok()?.parse::<usize>().ok()?;
    Some((cols.max(20), rows.max(5)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_zero_for_constant() {
        assert!(shannon_entropy(&[7u8; 1024]) < 1e-6);
    }

    #[test]
    fn entropy_near_eight_for_uniform() {
        let raw: Vec<u8> = (0..4096).map(|i| (i & 0xff) as u8).collect();
        assert!(shannon_entropy(&raw) > 7.9);
    }

    #[test]
    fn analyze_repeats_extend() {
        let raw: Vec<u8> = b"ABCDEFGHIJABCDEFGHIJABCDEFGHIJ".to_vec();
        let a = analyze(&raw);
        assert!(a.peak_match >= 10);
        assert!(a.n_matches >= 1);
    }

    #[test]
    fn analyze_random_sparse_vs_repetitive_dense() {
        let mut rnd = Vec::with_capacity(4096);
        let mut x: u32 = 0xDEADBEEF;
        for _ in 0..4096 {
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            rnd.push((x >> 16) as u8);
        }
        let rep: Vec<u8> = b"ABCDEFGHIJ".repeat(410);
        let ar = analyze(&rnd);
        let arep = analyze(&rep);
        let cov_rnd = ar.total_match_bytes as f64 / rnd.len() as f64;
        let cov_rep = arep.total_match_bytes as f64 / rep.len() as f64;
        assert!(cov_rep > cov_rnd * 5.0);
    }

    #[test]
    fn bucket_boundaries() {
        assert_eq!(len_bucket(3), 0);
        assert_eq!(len_bucket(4), 0);
        assert_eq!(len_bucket(5), 1);
        assert_eq!(len_bucket(8), 1);
        assert_eq!(len_bucket(9), 2);
        assert_eq!(len_bucket(258), 6);
        assert_eq!(dist_bucket(1), 0);
        assert_eq!(dist_bucket(8), 0);
        assert_eq!(dist_bucket(9), 1);
        assert_eq!(dist_bucket(32768), 4);
    }

    #[test]
    fn qualitative_labels_cover_range() {
        assert_eq!(entropy_label(1.0).0, "VERY LOW");
        assert_eq!(entropy_label(3.0).0, "LOW");
        assert_eq!(entropy_label(5.5).0, "MEDIUM");
        assert_eq!(entropy_label(7.0).0, "HIGH");
        assert_eq!(entropy_label(7.99).0, "MAX");
        assert_eq!(cover_label(0.01).0, "NONE");
        assert_eq!(cover_label(0.2).0, "LOW");
        assert_eq!(cover_label(0.5).0, "MEDIUM");
        assert_eq!(cover_label(0.75).0, "HIGH");
        assert_eq!(cover_label(0.9).0, "EXTREME");
    }

    #[test]
    fn byte_palette_is_monotonic_across_quartiles() {
        // Values stepped across the rainbow should give distinct colour codes.
        let a = byte_to_xterm(0);
        let b = byte_to_xterm(64);
        let c = byte_to_xterm(128);
        let d = byte_to_xterm(192);
        let e = byte_to_xterm(255);
        let set = std::collections::HashSet::from([a, b, c, d, e]);
        assert!(
            set.len() >= 4,
            "rainbow should produce distinct hues, got {:?}",
            [a, b, c, d, e]
        );
    }
}
