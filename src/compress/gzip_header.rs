//! [Gzippy's gzip header metadata](`GzipHeaderInfo`) — FNAME/FCOMMENT/MTIME
//! (RFC 1952). Extracted from `compress::parallel` (audit item #3): production
//! code imported this shared type from inside a quarantined C-FFI oracle
//! module, which made the one-concept-per-module rule lie.

/// Metadata for gzip header FNAME and MTIME fields
#[derive(Clone, Debug, Default)]
pub struct GzipHeaderInfo {
    /// Original filename (basename only) for FNAME field
    pub filename: Option<String>,
    /// File modification time as Unix timestamp for MTIME field
    pub mtime: u32,
    /// Optional comment for FCOMMENT field
    pub comment: Option<String>,
}

impl GzipHeaderInfo {
    /// Serialize as a gzip member header (RFC 1952): CM=8, FLG set from the
    /// FNAME/FCOMMENT fields actually present, MTIME little-endian, XFL=0,
    /// OS=0xff (unknown — the value every gzippy encode path emits).
    ///
    /// This is THE file-output header for every thread count (issue #309):
    /// the T>1 pipelined path builds it directly and the T1 file dispatch in
    /// `compress/io.rs` splices it over the fixed minimal header the T1
    /// encoders emit. A default `GzipHeaderInfo` serializes to exactly that
    /// 10-byte minimal header, so `-n` (no name, no time) output is unchanged.
    pub fn to_member_header(&self) -> Vec<u8> {
        let mut header = Vec::with_capacity(64);
        let mut flags: u8 = 0x00;
        if self.filename.is_some() {
            flags |= 0x08; // FNAME
        }
        if self.comment.is_some() {
            flags |= 0x10; // FCOMMENT
        }
        header.extend_from_slice(&[0x1f, 0x8b, 0x08, flags]);
        header.extend_from_slice(&self.mtime.to_le_bytes());
        header.extend_from_slice(&[0x00, 0xff]); // XFL, OS
        if let Some(ref name) = self.filename {
            header.extend_from_slice(name.as_bytes());
            header.push(0);
        }
        if let Some(ref comment) = self.comment {
            header.extend_from_slice(comment.as_bytes());
            header.push(0);
        }
        header
    }
}
