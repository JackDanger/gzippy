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

/// Generate a fixture by name. Panics on an unknown name (test-support code;
/// a typo should fail loudly).
pub fn generate(name: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(LEN + 128);
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
            while out.len() < LEN {
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
            while out.len() < LEN {
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
            while out.len() < LEN {
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
            while out.len() < LEN {
                out.extend_from_slice(&rng.next().to_le_bytes());
            }
        }
        other => panic!("unknown fixture '{other}' (declared: {NAMES:?})"),
    }
    out.truncate(LEN);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
