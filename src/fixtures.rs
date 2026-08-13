//! Deterministic corpus fixtures for the fingerprint suite.
//!
//! The real campaign corpus (squishy + silesia) does NOT belong in this
//! open-source repo, so tests pin against SYNTHETIC inputs generated here by
//! seeded integer arithmetic — identical bytes on every platform, every run,
//! forever. Each fixture imitates one content class the corpus taught us to
//! care about; none imitates any real file.
//!
//! These are for MECHANISM tests (fingerprints, ledgers, ratchets), not for
//! promotion: the promotion rule still runs on the real corpus on the boxes.

/// One fixture: a name and its deterministic bytes.
pub const NAMES: &[&str] = &["text", "tabular", "binary", "noise"];

/// Fixture size. Big enough for multiple DEFLATE blocks at every level and
/// several T>1 chunks at -p4; small enough that the whole suite runs in
/// seconds.
pub const LEN: usize = 1 << 20;

/// Small-file sizes: 4 KiB / 16 KiB / 64 KiB / 256 KiB prefixes of the same
/// generators (see [`generate_sized`] — a shorter output is a byte-exact
/// prefix of a longer one). The board never grades sub-1-MiB inputs, but
/// 4-100 KB files are the high-frequency real-world case;
/// tests/smallfile_pins.rs pins our output size per (fixture, size, level)
/// on this grid.
pub const SMALL_SIZES: &[usize] = &[4 << 10, 16 << 10, 64 << 10, 256 << 10];

/// The content classes graded at small sizes: one match-rich text class and
/// one binary class. `tabular`/`noise` add little discrimination below 1 MiB
/// (tabular is a denser text; noise is stored blocks at every size).
pub const SMALL_NAMES: &[&str] = &["text", "binary"];

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Generate a fixture by name at the standard [`LEN`]. Panics on an unknown
/// name (test-support code; a typo should fail loudly).
pub fn generate(name: &str) -> Vec<u8> {
    generate_sized(name, LEN)
}

/// Generate a fixture by name at an arbitrary `len` — the SAME seeded
/// generator as [`generate`], run longer or shorter. For any two lengths the
/// shorter output is a byte-exact prefix of the longer one (each generator
/// only ever appends), so size-scaling tests (tests/perf_shape.rs) vary ONLY
/// the input length, never the content class. `generate(name)` ==
/// `generate_sized(name, LEN)` byte-for-byte; the frozen-hash test below
/// guards that identity against generator drift.
pub fn generate_sized(name: &str, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len + 128);
    match name {
        // Prose-ish: words from a small lexicon, sentence structure, newlines.
        // Match-rich with a wide-ish literal alphabet — the aozora/dickens class.
        "text" => {
            const WORDS: &[&str] = &[
                "the", "of", "and", "to", "in", "was", "it", "his", "that", "he", "her", "with",
                "for", "had", "is", "you", "not", "be", "she", "on", "at", "by", "which", "have",
                "from", "this", "him", "they", "were", "all", "are", "but", "said", "one", "when",
                "there", "them", "would", "been", "will", "who", "more", "no", "if", "out", "so",
                "what", "up", "their", "then", "time", "into", "little", "about", "could", "than",
                "like", "other", "some", "only", "over", "such", "down", "your",
            ];
            let mut rng = XorShift(0x74657874_00000001);
            let mut words_in_sentence = 0u32;
            while out.len() < len {
                let w = WORDS[(rng.next() % WORDS.len() as u64) as usize];
                if words_in_sentence == 0 {
                    let mut c = w.as_bytes().to_vec();
                    c[0] = c[0].to_ascii_uppercase();
                    out.extend_from_slice(&c);
                } else {
                    out.extend_from_slice(w.as_bytes());
                }
                words_in_sentence += 1;
                let r = rng.next() % 100;
                if r < 8 && words_in_sentence > 3 {
                    out.extend_from_slice(b". ");
                    words_in_sentence = 0;
                    if rng.next().is_multiple_of(4) {
                        out.push(b'\n');
                    }
                } else if r < 12 {
                    out.extend_from_slice(b", ");
                } else {
                    out.push(b' ');
                }
            }
        }
        // CSV-ish: repeating field skeletons, incrementing ids, enum strings.
        // Extremely match-dense, tiny literal alphabet — the data.csv class
        // (where the length-3 rule says DISABLE).
        "tabular" => {
            const STATUS: &[&str] = &["active", "inactive", "pending", "active", "active"];
            let mut rng = XorShift(0x74616275_00000002);
            let mut id = 100_000u64;
            out.extend_from_slice(b"id,timestamp,region,status,value,flag\n");
            while out.len() < len {
                id += 1;
                let ts = 1_700_000_000 + (rng.next() % 86_400);
                let region = (rng.next() % 4) as usize;
                let status = STATUS[(rng.next() % STATUS.len() as u64) as usize];
                let value = rng.next() % 100_000;
                let flag = rng.next() % 2;
                out.extend_from_slice(
                    format!(
                        "{id},{ts},region-{region:02},{status},{}.{:02},{flag}\n",
                        value / 100,
                        value % 100
                    )
                    .as_bytes(),
                );
            }
        }
        // Executable-ish: repeating 16-byte record headers, varying LE words,
        // zero runs, wide literal alphabet — the armexe/tool.bin class (where
        // the length-3 rule says ENABLE and earns real bytes).
        "binary" => {
            let mut rng = XorShift(0x62696e61_00000003);
            while out.len() < len {
                out.extend_from_slice(&[0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00]);
                out.extend_from_slice(&(rng.next() as u32).to_le_bytes());
                out.extend_from_slice(&((out.len() as u32) ^ 0xdeadbeef).to_le_bytes());
                for _ in 0..(rng.next() % 6 + 2) {
                    out.extend_from_slice(&rng.next().to_le_bytes());
                }
                let zeros = (rng.next() % 48) as usize;
                out.resize(out.len() + zeros, 0);
            }
        }
        // Incompressible: raw generator output — the movie.mp4 class (stored
        // blocks, framing dominates).
        "noise" => {
            let mut rng = XorShift(0x6e6f6973_00000004);
            while out.len() < len {
                out.extend_from_slice(&rng.next().to_le_bytes());
            }
        }
        other => panic!("unknown fixture '{other}' (declared: {NAMES:?})"),
    }
    out.truncate(len);
    out
}

// ============================================================================
// Response-surface sampler — the parameterized end of the generator family.
//
// The named fixtures above are POINTS chosen to imitate corpus classes. The
// sampler below is the SPACE those points live in: content is generated along
// five explicit axes so `examples/surface_probe.rs` can walk the space,
// measure ratio-vs-rival at each point, and flag CLIFFS (adjacent points where
// the verdict flips or jumps). Each cliff is a generalization boundary with
// its content coordinates named — the failure modes a NEW archive type would
// hit, discovered before any user hits them.
//
// The sampler is measurement support: tests pin the generator BYTES (so a
// surface is comparable run-to-run) but never the ratios (the surface is a
// measurement, not a ratchet).
// ============================================================================

/// One point in content space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceParams {
    /// Target order-0 entropy of fresh literals, bits/byte (2..=8). The
    /// generator hits it by mixing a hot subset with the full alphabet; the
    /// probe reports the MEASURED entropy beside the target.
    pub entropy_bits: u8,
    /// Back-reference distance: every emitted match copies from exactly this
    /// many bytes back (16..=8192), so the axis walks match distance from
    /// "well inside every window" to "chunk-boundary scale".
    pub period: u16,
    /// Match-length profile: false = short (3..=8), true = long (32..=258).
    pub long_matches: bool,
    /// Fresh-literal alphabet size (4..=256).
    pub alphabet: u16,
    /// Record structure: insert a fixed 16-byte skeleton every ~256 bytes
    /// (the CSV/JSONL "field grid" shape) or not.
    pub records: bool,
}

impl SurfaceParams {
    /// Stable id naming the coordinates: `e{bits}_p{period}_{short|long}_a{n}_r{0|1}`.
    pub fn id(&self) -> String {
        format!(
            "e{}_p{}_{}_a{}_r{}",
            self.entropy_bits,
            self.period,
            if self.long_matches { "long" } else { "short" },
            self.alphabet,
            self.records as u8
        )
    }
}

/// The declared axis grids. Adjacency for cliff detection is one step along
/// one of these lists with every other coordinate held fixed.
pub const SURFACE_ENTROPY: &[u8] = &[2, 4, 6, 8];
pub const SURFACE_PERIODS: &[u16] = &[16, 128, 1024, 8192];

/// The declared sample: 60 deterministic points.
///  - 32: full entropy x period grid, alphabet 256, both match profiles.
///  - 12: small alphabet (16), entropy 2/3/4 x all periods, short matches.
///  -  8: record structure ON, alphabet 64, entropy 3/5, all periods, short.
///  -  8: record structure ON vs OFF pairs at alphabet 64, entropy 3/5,
///        period 128/8192, long matches (the records axis under long matches).
///  Adjacent pairs along every axis exist inside each block, so cliff
///  detection has one-step neighbours everywhere.
pub fn surface_points() -> Vec<SurfaceParams> {
    let mut pts = Vec::with_capacity(60);
    for &long_matches in &[false, true] {
        for &entropy_bits in SURFACE_ENTROPY {
            for &period in SURFACE_PERIODS {
                pts.push(SurfaceParams {
                    entropy_bits,
                    period,
                    long_matches,
                    alphabet: 256,
                    records: false,
                });
            }
        }
    }
    for &entropy_bits in &[2u8, 3, 4] {
        for &period in SURFACE_PERIODS {
            pts.push(SurfaceParams {
                entropy_bits,
                period,
                long_matches: false,
                alphabet: 16,
                records: false,
            });
        }
    }
    for &entropy_bits in &[3u8, 5] {
        for &period in SURFACE_PERIODS {
            pts.push(SurfaceParams {
                entropy_bits,
                period,
                long_matches: false,
                alphabet: 64,
                records: true,
            });
        }
    }
    for &entropy_bits in &[3u8, 5] {
        for &period in &[128u16, 8192] {
            for &records in &[false, true] {
                pts.push(SurfaceParams {
                    entropy_bits,
                    period,
                    long_matches: true,
                    alphabet: 64,
                    records,
                });
            }
        }
    }
    pts
}

/// Mixture weight q such that drawing uniform-from-hot (size `hot`) with
/// probability q and uniform-from-`alphabet` otherwise has order-0 entropy
/// `target` bits. Binary search; entropy is monotone decreasing in q.
fn surface_mix(target: f64, hot: usize, alphabet: usize) -> f64 {
    let ent = |q: f64| -> f64 {
        let p_hot = q / hot as f64 + (1.0 - q) / alphabet as f64;
        let p_cold = (1.0 - q) / alphabet as f64;
        let mut h = -(hot as f64) * p_hot * p_hot.log2();
        if alphabet > hot && p_cold > 0.0 {
            h -= (alphabet - hot) as f64 * p_cold * p_cold.log2();
        }
        h
    };
    if ent(0.0) <= target {
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..48 {
        let mid = (lo + hi) / 2.0;
        if ent(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Generate a surface point. Append-only, so as with [`generate_sized`] a
/// shorter output is a byte-exact prefix of a longer one — the pinned test
/// hashes 64 KiB while the probe runs 1 MiB of the same stream.
pub fn surface_generate(p: &SurfaceParams, len: usize) -> Vec<u8> {
    assert!((2..=8).contains(&p.entropy_bits));
    assert!(p.alphabet >= 4 && p.alphabet <= 256);
    let alphabet = p.alphabet as usize;
    let target = f64::from(p.entropy_bits).min((alphabet as f64).log2());
    // Hot set: small enough that q -> 1 undershoots the lowest target, large
    // enough to give the search room; never more than half the alphabet.
    let hot = (1usize << (p.entropy_bits.saturating_sub(1))).clamp(2, alphabet / 2);
    let q = surface_mix(target, hot, alphabet);
    let q_scaled = (q * f64::from(u32::MAX)) as u64;

    // Seed derived from the coordinates so every point is its own stream.
    let mut seed = 0x5375_7266_0000_0001u64;
    for v in [
        p.entropy_bits as u64,
        p.period as u64,
        p.long_matches as u64,
        p.alphabet as u64,
        p.records as u64,
    ] {
        seed = (seed ^ v).wrapping_mul(0x100_0000_01b3);
    }
    let mut rng = XorShift(seed | 1);

    let mut out: Vec<u8> = Vec::with_capacity(len + 512);
    let mut since_record = 0usize;
    let mut recno = 0u32;
    while out.len() < len {
        if p.records && since_record >= 240 {
            // 16-byte skeleton: a fixed frame with a slow counter — the
            // "field grid" every structured format carries.
            out.extend_from_slice(format!("\n#R{recno:08}|F0|\t").as_bytes());
            recno += 1;
            since_record = 0;
            continue;
        }
        let dist = p.period as usize;
        if out.len() >= dist && rng.next() % 100 < 35 {
            // one back-reference at exactly `period` distance (may overlap
            // itself when the length exceeds the distance — RLE-like, legal)
            let l = if p.long_matches {
                32 + (rng.next() % 227) as usize
            } else {
                3 + (rng.next() % 6) as usize
            };
            for _ in 0..l {
                let b = out[out.len() - dist];
                out.push(b);
            }
            since_record += l;
        } else {
            // one fresh literal from the entropy-controlled distribution
            let sym = if (rng.next() & 0xffff_ffff) < q_scaled {
                rng.next() % hot as u64
            } else {
                rng.next() % alphabet as u64
            };
            out.push(sym as u8);
            since_record += 1;
        }
    }
    out.truncate(len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The small-file grid (tests/smallfile_pins.rs) grades SHORTER RUNS of
    /// the same generators, which is only honest if a shorter output really
    /// is a byte-exact prefix of the standard fixture — vary the length,
    /// never the content class. Guard that here for every declared cell.
    #[test]
    fn small_sizes_are_prefixes_of_the_standard_fixtures() {
        for &name in SMALL_NAMES {
            let full = generate(name);
            for &size in SMALL_SIZES {
                assert!(size < LEN, "{name}: small size {size} is not small");
                assert_eq!(
                    generate_sized(name, size),
                    full[..size],
                    "{name}@{size} is not a prefix of the {LEN}-byte fixture — \
                     the generator appends non-deterministically?"
                );
            }
        }
    }

    /// The pinned-fingerprint files are only meaningful if these bytes never
    /// change. Guard the generators with content hashes: a change to any
    /// generator MUST be a conscious act that also regenerates every pin.
    #[test]
    fn fixtures_are_frozen() {
        for &name in NAMES {
            let data = generate(name);
            assert_eq!(data.len(), LEN, "{name}");
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for &b in &data {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
            let expect = match name {
                "text" => 0xd2ad5cb3d9f2ac83u64,
                "tabular" => 0x8f132a1f79ec4511,
                "binary" => 0xfe903199456d928d,
                "noise" => 0xcdfd5fb185201167,
                _ => unreachable!(),
            };
            if expect != 0 {
                assert_eq!(h, expect, "fixture '{name}' bytes changed — every pinned fingerprint file is now stale; regenerate via examples/fingerprint_tool.rs");
            } else {
                eprintln!("fixture {name}: fnv={h:#018x} (pin this)");
            }
        }
    }

    /// Pin the response-surface GENERATOR, never its ratios: a surface TSV is
    /// only comparable to an earlier one if each point's bytes are identical.
    /// One sha256 over (id, per-point sha256 at 64 KiB) for all declared
    /// points; on mismatch every per-point line is printed for repinning.
    /// 64 KiB is a prefix of the probe's 1 MiB stream (append-only property
    /// guarded in surface_points_are_prefix_stable).
    #[test]
    fn surface_generators_are_frozen() {
        let pts = surface_points();
        assert_eq!(pts.len(), 60, "declared sample size changed");
        // Ids must be unique or the TSV/cliff coordinates are ambiguous.
        let mut manifest = String::new();
        for p in &pts {
            let sha = crate::holdout::sha256_hex(&surface_generate(p, 64 << 10));
            manifest.push_str(&format!("{}\t{}\n", p.id(), sha));
        }
        let got = crate::holdout::sha256_hex(manifest.as_bytes());
        let want = "e2f23a2e680ac6c8ca39834819ef1d604ba6080a95bd7f1df5eb2252a76872f2";
        if got != want {
            eprint!("{manifest}");
            assert_eq!(
                got, want,
                "surface generator bytes changed — every archived surface TSV \
                 is now incomparable; repin the manifest sha above consciously"
            );
        }
    }

    /// The probe hashes 64 KiB but measures 1 MiB; that is only one stream if
    /// generation is append-only. Guard the prefix property on a spread of
    /// points (cheap; full grid would re-generate 62 MiB).
    #[test]
    fn surface_points_are_prefix_stable() {
        for p in surface_points().iter().step_by(9) {
            let long = surface_generate(p, 256 << 10);
            let short = surface_generate(p, 64 << 10);
            assert_eq!(short[..], long[..64 << 10], "{}", p.id());
        }
    }

    /// The entropy axis must actually order the content: measured order-0
    /// entropy of fresh literals rises strictly with the axis coordinate.
    /// (Matches copy earlier bytes, so measure a matchless configuration.)
    #[test]
    fn surface_entropy_axis_is_monotone() {
        let mut last = -1.0f64;
        for &e in SURFACE_ENTROPY {
            let p = SurfaceParams {
                entropy_bits: e,
                period: 8192,
                long_matches: false,
                alphabet: 256,
                records: false,
            };
            // period 8192 with 64 KiB still emits matches; measure literals
            // via a histogram of the whole output — matches only repeat
            // earlier literals, so the histogram stays distribution-shaped.
            let data = surface_generate(&p, 64 << 10);
            let mut hist = [0u64; 256];
            for &b in &data {
                hist[b as usize] += 1;
            }
            let n = data.len() as f64;
            let h: f64 = hist
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let pr = c as f64 / n;
                    -pr * pr.log2()
                })
                .sum();
            assert!(
                h > last,
                "entropy axis not monotone: e{e} measured {h:.3} <= previous {last:.3}"
            );
            last = h;
        }
    }
}
