//! Canonical Huffman code construction + the DEFLATE dynamic-block header —
//! ONE module tree shared by every tier (level engine L0-12 and the crown
//! `parse::ultra` engine).
//!
//! Three builders/tiers, each in its own submodule (see
//! `docs/compressor-architecture.md` §2 for the target shape this stage
//! implements):
//!
//! - [`fast`] — libdeflate's APPROXIMATE length-limited builder. Hot path for
//!   every level and near-optimal; carries L2-12 byte-identity with
//!   libdeflate.
//! - [`optimal`] — the EXACT (Katajainen package-merge) length-limited
//!   builder plus RLE-aware Huffman-count shaping. Used by the crown engine
//!   (`parse::ultra`) wherever a block can afford exactness.
//! - [`header`] — the level engine's dynamic-block header (precode/RLE)
//!   builder + emitter, built on [`fast::make_huffman_code`]. Ultra
//!   constructs+emits its own dynamic header via a parallel code path
//!   (`parse::ultra::deflate`/`deflate_size`) that shares the wire format but
//!   not the cost-accounting shape; see `header`'s module doc and
//!   `docs/compressor-architecture.md` §3 for why the two are not merged in
//!   this stage.
//!
//! [`HuffmanCode`] is the shared output type: both `fast::make_huffman_code`
//! and `header::build_dynamic_header`'s precode consume/produce it.

pub mod fast;
pub mod header;
pub mod optimal;

pub use fast::make_huffman_code;
pub use header::build_dynamic_header;

/// A canonical Huffman code: per-symbol length (0 = unused) and bit-reversed
/// codeword (right-justified, only the low `len` bits are meaningful).
#[derive(Clone)]
pub struct HuffmanCode {
    pub lens: Vec<u8>,
    pub codewords: Vec<u32>,
}
