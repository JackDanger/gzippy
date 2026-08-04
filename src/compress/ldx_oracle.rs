//! Per-decision divergence oracle: our shipped encoder vs the `ldx` port.
//!
//! `ldx` (`src/compress/ldx/`) is a byte-exact port of libdeflate's encoder,
//! verified byte-identical at L0-L9. Whole-file byte diffs can only say
//! "file X is +N bytes at level L". This module turns that into "the FIRST
//! divergent DECISION is at uncompressed position P" — literal-vs-match,
//! match-length, or match-distance — by compressing with both encoders and
//! decoding both outputs back into token streams with the in-tree block
//! walker ([`crate::decompress::block_walker::walk_deflate_tokens`]).
//!
//! NOT a port of any C function; this is tooling on top of the port, which is
//! why it lives beside `ldx/` rather than inside it (the port's module rule
//! is "if you cannot point at the C, it does not belong here").
//!
//! Nothing here is routed into any shipping path.

#![allow(dead_code)] // public surface used by examples/ldx_divergence.rs + tests/ldx_oracle.rs

use crate::decompress::block_walker::{walk_deflate_tokens, DeflateEvent};

/// One decoded token with its uncompressed start position and the index of
/// the DEFLATE block it sits in. `dist == 0` means literal (covers 1 byte);
/// otherwise a match covering `len` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub pos: u64,
    pub block: u32,
    pub len: u16,
    pub dist: u16,
}

impl Token {
    pub fn is_literal(&self) -> bool {
        self.dist == 0
    }

    /// Human-readable form; `input` supplies the literal's byte value.
    pub fn describe(&self, input: &[u8]) -> String {
        if self.is_literal() {
            let b = input.get(self.pos as usize).copied().unwrap_or(0);
            if b.is_ascii_graphic() || b == b' ' {
                format!("literal '{}'", b as char)
            } else {
                format!("literal 0x{b:02x}")
            }
        } else {
            format!("match({},{})", self.len, self.dist)
        }
    }
}

/// Per-stream summary from one tokenization pass.
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    pub compressed_len: usize,
    /// Uncompressed bytes covered by each block, in stream order.
    pub block_uncompressed_lens: Vec<u64>,
    /// `(uncompressed_start_pos, btype)` of every block, in stream order.
    /// btype: 0 stored, 1 fixed, 2 dynamic.
    pub block_starts: Vec<(u64, u8)>,
    pub literals: u64,
    pub matches: u64,
    pub total_uncompressed: u64,
}

impl StreamStats {
    pub fn block_count(&self) -> usize {
        self.block_uncompressed_lens.len()
    }

    /// Index of the block containing uncompressed position `pos` (the last
    /// block whose start is <= pos).
    pub fn block_index_at(&self, pos: u64) -> usize {
        self.block_starts
            .partition_point(|&(p, _)| p <= pos)
            .saturating_sub(1)
    }
}

/// Decode a raw DEFLATE stream into positioned tokens. Stored-block payloads
/// are expanded into per-byte literal tokens: a stored byte is a forced
/// literal decision at that position.
pub fn tokenize(deflate: &[u8]) -> std::io::Result<(Vec<Token>, StreamStats)> {
    let mut tokens = Vec::new();
    let mut stats = StreamStats {
        compressed_len: deflate.len(),
        ..Default::default()
    };
    let mut pos: u64 = 0;
    let mut block: i64 = -1;
    let total = walk_deflate_tokens(deflate, &mut |ev| match ev {
        DeflateEvent::BlockStart { btype, .. } => {
            block += 1;
            stats.block_uncompressed_lens.push(0);
            stats.block_starts.push((pos, btype));
        }
        DeflateEvent::Literal => {
            tokens.push(Token {
                pos,
                block: block as u32,
                len: 1,
                dist: 0,
            });
            *stats.block_uncompressed_lens.last_mut().unwrap() += 1;
            stats.literals += 1;
            pos += 1;
        }
        DeflateEvent::Match { len, dist } => {
            tokens.push(Token {
                pos,
                block: block as u32,
                len,
                dist,
            });
            *stats.block_uncompressed_lens.last_mut().unwrap() += len as u64;
            stats.matches += 1;
            pos += len as u64;
        }
        DeflateEvent::StoredBytes { len } => {
            for _ in 0..len {
                tokens.push(Token {
                    pos,
                    block: block as u32,
                    len: 1,
                    dist: 0,
                });
                stats.literals += 1;
                pos += 1;
            }
            *stats.block_uncompressed_lens.last_mut().unwrap() += len as u64;
        }
    })?;
    stats.total_uncompressed = total;
    debug_assert_eq!(pos, total);
    Ok((tokens, stats))
}

/// What kind of decision diverged first.
#[derive(Debug, Clone, Copy)]
pub enum DivergenceKind {
    /// Both streams start a token at this position and the tokens differ.
    Token { ours: Token, ldx: Token },
    /// The block framing differs at this position: `Some(btype)` if that
    /// stream starts a block here, `None` if it is mid-block.
    BlockBoundary { ours: Option<u8>, ldx: Option<u8> },
}

/// The first place the two encoders decided differently.
#[derive(Debug, Clone, Copy)]
pub struct FirstDivergence {
    /// Absolute uncompressed position of the decision.
    pub pos: u64,
    pub kind: DivergenceKind,
}

/// Aggregate divergence between two token streams over the same input.
///
/// The four aligned classes count positions where BOTH streams start a token
/// and the tokens differ. `misaligned_starts` counts token starts present in
/// only one stream — the positional cascade that follows an aligned
/// divergence (e.g. ours took match(5) where ldx took a literal: ldx's next
/// four token starts have no counterpart in ours).
#[derive(Debug, Clone, Default)]
pub struct DivergenceReport {
    pub first: Option<FirstDivergence>,
    pub we_literal_they_match: u64,
    pub we_match_they_literal: u64,
    pub both_match_different_len: u64,
    pub both_match_different_dist: u64,
    pub misaligned_starts: u64,
    /// Positions where the block framing differs: a block starts there in
    /// exactly one stream, or in both with a different btype. Two streams
    /// can be token-identical and still differ ONLY here (e.g. stored-block
    /// grids of different sizes over incompressible data).
    pub block_boundary: u64,
    pub ours: StreamStats,
    pub ldx: StreamStats,
}

impl DivergenceReport {
    /// Total divergent positions: aligned class counts, cascade starts, and
    /// block-framing differences.
    pub fn total_divergent(&self) -> u64 {
        self.we_literal_they_match
            + self.we_match_they_literal
            + self.both_match_different_len
            + self.both_match_different_dist
            + self.misaligned_starts
            + self.block_boundary
    }

    pub fn is_zero(&self) -> bool {
        self.total_divergent() == 0 && self.first.is_none()
    }
}

/// Compare two raw DEFLATE streams that decode the same input.
pub fn compare(ours_deflate: &[u8], ldx_deflate: &[u8]) -> std::io::Result<DivergenceReport> {
    let (a, ours_stats) = tokenize(ours_deflate)?;
    let (b, ldx_stats) = tokenize(ldx_deflate)?;
    if ours_stats.total_uncompressed != ldx_stats.total_uncompressed {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "streams decode different lengths: ours {} vs ldx {}",
                ours_stats.total_uncompressed, ldx_stats.total_uncompressed
            ),
        ));
    }
    let mut r = DivergenceReport {
        ours: ours_stats,
        ldx: ldx_stats,
        ..Default::default()
    };

    // Block-framing diff: merge the two block-start lists by position.
    let mut first_boundary: Option<FirstDivergence> = None;
    {
        let (sa, sb) = (&r.ours.block_starts, &r.ldx.block_starts);
        let (mut i, mut j) = (0usize, 0usize);
        while i < sa.len() || j < sb.len() {
            let pa = sa.get(i).map(|&(p, _)| p);
            let pb = sb.get(j).map(|&(p, _)| p);
            let (pos, ours_bt, ldx_bt) = match (pa, pb) {
                (Some(pa), Some(pb)) if pa == pb => {
                    let (a_bt, b_bt) = (sa[i].1, sb[j].1);
                    i += 1;
                    j += 1;
                    if a_bt == b_bt {
                        continue;
                    }
                    (pa, Some(a_bt), Some(b_bt))
                }
                (Some(pa), None) => {
                    let bt = sa[i].1;
                    i += 1;
                    (pa, Some(bt), None)
                }
                (Some(pa), Some(pb)) if pa < pb => {
                    let bt = sa[i].1;
                    i += 1;
                    (pa, Some(bt), None)
                }
                (_, Some(pb)) => {
                    let bt = sb[j].1;
                    j += 1;
                    (pb, None, Some(bt))
                }
                (None, None) => unreachable!(),
            };
            r.block_boundary += 1;
            if first_boundary.is_none() {
                first_boundary = Some(FirstDivergence {
                    pos,
                    kind: DivergenceKind::BlockBoundary {
                        ours: ours_bt,
                        ldx: ldx_bt,
                    },
                });
            }
        }
    }

    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let (ta, tb) = (a[i], b[j]);
        if ta.pos < tb.pos {
            r.misaligned_starts += 1;
            i += 1;
            continue;
        }
        if tb.pos < ta.pos {
            r.misaligned_starts += 1;
            j += 1;
            continue;
        }
        // Aligned start: compare the decisions.
        if ta.len == tb.len && ta.dist == tb.dist {
            i += 1;
            j += 1;
            continue;
        }
        match (ta.is_literal(), tb.is_literal()) {
            (true, false) => r.we_literal_they_match += 1,
            (false, true) => r.we_match_they_literal += 1,
            (false, false) if ta.len != tb.len => r.both_match_different_len += 1,
            (false, false) => r.both_match_different_dist += 1,
            // Two literals at the same position decode the same byte and
            // carry no (len,dist) freedom — they cannot differ.
            (true, true) => unreachable!("two literals at one position are equal"),
        }
        if r.first.is_none() {
            r.first = Some(FirstDivergence {
                pos: ta.pos,
                kind: DivergenceKind::Token { ours: ta, ldx: tb },
            });
        }
        i += 1;
        j += 1;
    }
    // One stream ran out of tokens first only if lengths mismatched, which
    // was rejected above — but count any tail defensively.
    r.misaligned_starts += (a.len() - i) as u64 + (b.len() - j) as u64;
    // The FIRST divergence is the earliest of the token diff and the framing
    // diff; on a position tie the token diff is the more informative one.
    r.first = match (r.first, first_boundary) {
        (Some(t), Some(b)) if b.pos < t.pos => Some(b),
        (None, b) => b,
        (t, _) => t,
    };
    Ok(r)
}

/// End-to-end: compress `input` at `level` with the shipped T1 path and with
/// ldx, then compare. Returns `None` when ldx does not implement `level`
/// (the exotic levels 10-12).
pub fn divergence_at_level(input: &[u8], level: u32) -> Option<std::io::Result<DivergenceReport>> {
    let ldx = crate::compress::ldx::compress_for_diff(level, input)?;
    let ours = crate::compress::deflate::encode_deflate_bytes_to_vec(input, level);
    Some(compare(&ours, &ldx))
}
