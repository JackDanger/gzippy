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

// ───────────────────────────────────────────────────────────────────────────
// Exact bit accounting of the size gap
//
// The census above counts divergent POSITIONS. This layer weighs them: it
// partitions both token streams into maximal identical/divergent regions,
// costs every token from its OWN block's real Huffman tables (code length +
// extra bits, via `block_walker::walk_deflate_block_lens`), and attributes
// the whole size gap across six classes. The attribution is EXACT, not
// estimated: per side, header + EOB + padding + identical-token bits +
// divergent-region bits sums to 8 × the compressed byte size, so the class
// deltas sum to 8 × (size_ours − size_ldx) with ZERO residual. A nonzero
// residual means this module's model of the streams has drifted from
// reality — `tests/divergence_accounting.rs` pins it at zero.
// ───────────────────────────────────────────────────────────────────────────

use crate::decompress::block_walker::{walk_deflate_block_lens, BlockLens};

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Length-symbol index (0..=28) for a match length in 3..=258: the last
/// index whose base is <= `len`, honoring the dedicated 258 code.
fn len_sym(len: u16) -> usize {
    if len == 258 {
        return 28;
    }
    let mut i = 0;
    while i + 1 < 29 && LENGTH_BASE[i + 1] <= len {
        i += 1;
    }
    i.min(27)
}

/// Distance-symbol index (0..=29) for a match distance in 1..=32768.
fn dist_sym(dist: u16) -> usize {
    let mut i = 0;
    while i + 1 < 30 && DIST_BASE[i + 1] <= dist {
        i += 1;
    }
    i
}

/// Exact bit cost of one token under its own block's real code lengths.
/// Stored-block tokens cost 8 bits per byte (framing is counted separately).
fn token_bits(t: &Token, input: &[u8], blocks: &[BlockLens]) -> u64 {
    let b = &blocks[t.block as usize];
    if b.btype == 0 {
        return 8 * t.len.max(1) as u64;
    }
    if t.dist == 0 {
        b.litlen[input[t.pos as usize] as usize] as u64
    } else {
        let li = len_sym(t.len);
        let di = dist_sym(t.dist);
        b.litlen[257 + li] as u64
            + LENGTH_EXTRA[li] as u64
            + b.dist[di] as u64
            + DIST_EXTRA[di] as u64
    }
}

/// The four aligned divergence classes, in attribution order. A divergent
/// REGION is keyed by the class of its first divergent decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceClass {
    /// Ours took a literal where ldx took a match.
    WeLitTheyMatch = 0,
    /// Both took a match, different lengths.
    DiffLen = 1,
    /// Both took a match, same length, different distances.
    DiffDist = 2,
    /// Ours took a match where ldx took a literal.
    WeMatchTheyLit = 3,
}

impl DivergenceClass {
    pub const ALL: [DivergenceClass; 4] = [
        DivergenceClass::WeLitTheyMatch,
        DivergenceClass::DiffLen,
        DivergenceClass::DiffDist,
        DivergenceClass::WeMatchTheyLit,
    ];

    pub fn name(self) -> &'static str {
        match self {
            DivergenceClass::WeLitTheyMatch => "we_lit_they_match",
            DivergenceClass::DiffLen => "diff_len",
            DivergenceClass::DiffDist => "diff_dist",
            DivergenceClass::WeMatchTheyLit => "we_match_they_lit",
        }
    }

    fn of(a: &Token, b: &Token) -> DivergenceClass {
        match (a.is_literal(), b.is_literal()) {
            (true, false) => DivergenceClass::WeLitTheyMatch,
            (false, true) => DivergenceClass::WeMatchTheyLit,
            (false, false) if a.len != b.len => DivergenceClass::DiffLen,
            _ => DivergenceClass::DiffDist,
        }
    }
}

/// Bit totals of the divergent regions whose FIRST divergent decision has one
/// class. `bits_ours`/`bits_ldx` are the exact Huffman bits each stream spent
/// inside those regions.
#[derive(Debug, Clone, Copy, Default)]
pub struct RegionTotals {
    pub regions: u64,
    pub bits_ours: u64,
    pub bits_ldx: u64,
}

impl RegionTotals {
    pub fn delta_bits(&self) -> i64 {
        self.bits_ours as i64 - self.bits_ldx as i64
    }
}

/// Exact bit accounting of `ours` vs `ldx` over the same input. All `[u64;
/// 2]` fields are `[ours, ldx]`.
#[derive(Debug, Clone, Default)]
pub struct BitAccounting {
    pub ours_bytes: u64,
    pub ldx_bytes: u64,
    /// Block header bits: 3-bit prelude + dynamic table description, or
    /// stored alignment + LEN/NLEN.
    pub header_bits: [u64; 2],
    /// One EOB symbol per non-stored block, at that block's real code length.
    pub eob_bits: [u64; 2],
    /// Zero-fill after the final block, up to the compressed byte boundary.
    pub padding_bits: [u64; 2],
    /// Bits both streams spent on IDENTICAL tokens (same pos/len/dist) —
    /// any delta here is pure Huffman-table drift.
    pub ident_bits: [u64; 2],
    pub ident_tokens: u64,
    /// Divergent-region totals, indexed by `DivergenceClass as usize`.
    pub regions: [RegionTotals; 4],
    /// Every ALIGNED divergent decision pair (both streams start a token at
    /// the same position and the tokens differ), for histogramming. The
    /// first pair of every region is included.
    pub aligned_pairs: Vec<(DivergenceClass, Token, Token)>,
}

impl BitAccounting {
    /// Attribution classes, in the order `attribution_bits` reports them:
    /// the four region classes, then Huffman-table drift on identical
    /// tokens, then framing (headers + EOB + final padding).
    pub const ATTRIBUTION_CLASSES: [&'static str; 6] = [
        "we_lit_they_match",
        "diff_len",
        "diff_dist",
        "we_match_they_lit",
        "table_drift",
        "headers_eob",
    ];

    /// Signed per-class bit deltas (ours − ldx), in
    /// [`Self::ATTRIBUTION_CLASSES`] order. Sums exactly to
    /// [`Self::gap_bits`] when the accounting is sound.
    pub fn attribution_bits(&self) -> [i64; 6] {
        let d = |f: [u64; 2]| f[0] as i64 - f[1] as i64;
        [
            self.regions[0].delta_bits(),
            self.regions[1].delta_bits(),
            self.regions[2].delta_bits(),
            self.regions[3].delta_bits(),
            d(self.ident_bits),
            d(self.header_bits) + d(self.eob_bits) + d(self.padding_bits),
        ]
    }

    /// The whole gap: 8 × (size_ours − size_ldx), in bits.
    pub fn gap_bits(&self) -> i64 {
        8 * (self.ours_bytes as i64 - self.ldx_bytes as i64)
    }

    /// `gap_bits − Σ attribution_bits`. EXACTLY zero when this module's
    /// model of the two streams matches reality.
    pub fn residual_bits(&self) -> i64 {
        self.gap_bits() - self.attribution_bits().iter().sum::<i64>()
    }

    /// Per-side accounted bits: header + EOB + padding + identical tokens +
    /// divergent regions. Equals `8 × compressed bytes` per side when sound
    /// (a stronger check than `residual_bits == 0`, which two compensating
    /// per-side errors could fake).
    pub fn side_accounted_bits(&self) -> [u64; 2] {
        let mut out = [0u64; 2];
        for (s, o) in out.iter_mut().enumerate() {
            *o = self.header_bits[s] + self.eob_bits[s] + self.padding_bits[s] + self.ident_bits[s];
        }
        for r in &self.regions {
            out[0] += r.bits_ours;
            out[1] += r.bits_ldx;
        }
        out
    }

    pub fn side_total_bits(&self) -> [u64; 2] {
        [8 * self.ours_bytes, 8 * self.ldx_bytes]
    }
}

/// Exact bit accounting of two raw DEFLATE streams over the same `input`.
///
/// Alignment: identical tokens are consumed pairwise; at the first differing
/// token both streams are position-aligned (identical prefixes cover equal
/// spans), a region opens keyed by that pair's class, and tokens are consumed
/// lowest-position-first until both streams agree on an identical token
/// again. Every token is therefore counted exactly once, in exactly one
/// bucket, at its own block's real code lengths.
pub fn account(
    input: &[u8],
    ours_deflate: &[u8],
    ldx_deflate: &[u8],
) -> std::io::Result<BitAccounting> {
    let bad = |m: String| std::io::Error::new(std::io::ErrorKind::InvalidData, m);
    let (toks_o, stats_o) = tokenize(ours_deflate)?;
    let (toks_l, stats_l) = tokenize(ldx_deflate)?;
    if stats_o.total_uncompressed != input.len() as u64
        || stats_l.total_uncompressed != input.len() as u64
    {
        return Err(bad(format!(
            "streams do not decode the input length {}: ours {} vs ldx {}",
            input.len(),
            stats_o.total_uncompressed,
            stats_l.total_uncompressed
        )));
    }
    let blocks_o = walk_deflate_block_lens(ours_deflate)?;
    let blocks_l = walk_deflate_block_lens(ldx_deflate)?;

    let mut acc = BitAccounting {
        ours_bytes: ours_deflate.len() as u64,
        ldx_bytes: ldx_deflate.len() as u64,
        ..Default::default()
    };
    for (side, (blocks, comp_len)) in [
        (&blocks_o, ours_deflate.len()),
        (&blocks_l, ldx_deflate.len()),
    ]
    .into_iter()
    .enumerate()
    {
        acc.header_bits[side] = blocks.iter().map(|b| b.body_start_bit - b.start_bit).sum();
        acc.eob_bits[side] = blocks
            .iter()
            .filter(|b| b.btype != 0)
            .map(|b| b.litlen[256] as u64)
            .sum();
        acc.padding_bits[side] = 8 * comp_len as u64 - blocks.last().unwrap().end_bit;
    }

    let same = |a: &Token, b: &Token| a.pos == b.pos && a.len == b.len && a.dist == b.dist;
    let (mut i, mut j) = (0usize, 0usize);
    while i < toks_o.len() && j < toks_l.len() {
        let (ta, tb) = (toks_o[i], toks_l[j]);
        if same(&ta, &tb) {
            acc.ident_bits[0] += token_bits(&ta, input, &blocks_o);
            acc.ident_bits[1] += token_bits(&tb, input, &blocks_l);
            acc.ident_tokens += 1;
            i += 1;
            j += 1;
            continue;
        }
        // Identical prefixes cover equal spans, so a divergence always opens
        // position-aligned; a violation is a bug in THIS module, not parse
        // movement.
        assert_eq!(ta.pos, tb.pos, "divergent region must start aligned");
        let class = DivergenceClass::of(&ta, &tb);
        acc.regions[class as usize].regions += 1;
        loop {
            let pa = toks_o.get(i).map(|t| t.pos).unwrap_or(u64::MAX);
            let pb = toks_l.get(j).map(|t| t.pos).unwrap_or(u64::MAX);
            if pa == u64::MAX && pb == u64::MAX {
                break;
            }
            if pa == pb && pa != u64::MAX && same(&toks_o[i], &toks_l[j]) {
                break; // re-synced
            }
            if pa == pb {
                let (xa, xb) = (toks_o[i], toks_l[j]);
                acc.aligned_pairs
                    .push((DivergenceClass::of(&xa, &xb), xa, xb));
            }
            if pa <= pb {
                acc.regions[class as usize].bits_ours += token_bits(&toks_o[i], input, &blocks_o);
                i += 1;
            } else {
                acc.regions[class as usize].bits_ldx += token_bits(&toks_l[j], input, &blocks_l);
                j += 1;
            }
        }
    }
    assert!(
        i == toks_o.len() && j == toks_l.len(),
        "alignment left unconsumed tokens: ours {}/{}, ldx {}/{}",
        i,
        toks_o.len(),
        j,
        toks_l.len()
    );
    Ok(acc)
}

/// End-to-end: compress `input` at `level` with the shipped T1 path and with
/// ldx, then run [`account`]. `None` when ldx does not implement `level`.
pub fn account_at_level(input: &[u8], level: u32) -> Option<std::io::Result<BitAccounting>> {
    let ldx = crate::compress::ldx::compress_for_diff(level, input)?;
    let ours = crate::compress::deflate::encode_deflate_bytes_to_vec(input, level);
    Some(account(input, &ours, &ldx))
}
