//! Literal port of `rapidgzip::IndexFileFormat`
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/IndexFileFormat.hpp:36-1060).
//!
//! Read/write seekable gzip indexes. gzippy initially ports only the
//! **indexed_gzip GZIDX** format (IndexFileFormat.hpp:39-66, 408-685) —
//! the layout actually used by `indexed_gzip` and consumed by tools like
//! `gztool`. The `bgzip`/`gztool` BGZI / "gzipind" variants and the
//! WindowMap parallel-decompression machinery (IndexFileFormat.hpp:236-405,
//! 688-960) are explicitly out of scope here and will land in follow-up
//! commits — none are wired into gzippy's CLI today.
//!
//! Format (mirror of the comment block at IndexFileFormat.hpp:39-66):
//!
//! ```text
//! 00  "GZIDX"             # Index File ID
//! 05  uint8_t             # File Version (0 or 1)
//! 06  uint8_t             # Flags (Unused)
//! 07  uint64_t LE         # Compressed Size (bytes)
//! 15  uint64_t LE         # Uncompressed Size (bytes)
//! 23  uint32_t LE         # Spacing
//! 27  uint32_t LE         # Window Size (must be 32768 for indexed_gzip
//!                         # compatibility — IndexFileFormat.hpp:471-474)
//! 31  uint32_t LE         # Number of Checkpoints
//! 35
//!
//! Per checkpoint:
//! 00  uint64_t LE         # Compressed Offset, rounded UP in bytes
//! 08  uint64_t LE         # Uncompressed Offset (bytes)
//! 16  uint8_t             # Bits (0-7); compressed_bit_offset =
//!                         # compressed_byte_offset * 8 - bits
//! 17  uint8_t             # Data Flag (1 if window data follows)
//! 18
//!
//! Window data (only for checkpoints whose Data Flag is 1) follows the
//! checkpoint table, each window exactly `windowSizeInBytes` bytes.
//! ```
//!
//! NOTE — this module owns serialization only; it does not interact with
//! WindowMap or any decompression pipeline. The caller supplies and
//! consumes the raw window bytes per checkpoint.

#![allow(dead_code)]

use std::io::{self, Read, Write};

/// Mirror of `MAGIC_BYTES` (IndexFileFormat.hpp:410).
pub const MAGIC_BYTES: &[u8; 5] = b"GZIDX";

/// The only supported window size for the indexed_gzip format
/// (IndexFileFormat.hpp:471-474). Any other value is rejected on read.
pub const WINDOW_SIZE_BYTES: u32 = 32 * 1024;

/// Mirror of `Checkpoint` (IndexFileFormat.hpp:68-81).
///
/// `line_offset` is only meaningful when [`GzipIndex::has_line_offsets`]
/// is set (IndexFileFormat.hpp:72) — the indexed_gzip GZIDX format does
/// not carry line offsets, so it stays zero on read.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Checkpoint {
    /// Mirror of `compressedOffsetInBits` (IndexFileFormat.hpp:70).
    pub compressed_offset_in_bits: u64,
    /// Mirror of `uncompressedOffsetInBytes` (IndexFileFormat.hpp:71).
    pub uncompressed_offset_in_bytes: u64,
    /// Mirror of `lineOffset` (IndexFileFormat.hpp:72).
    pub line_offset: u64,
}

/// Mirror of `IndexFormat` (IndexFileFormat.hpp:84-89).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexFormat {
    /// `INDEXED_GZIP` — the GZIDX format implemented here.
    IndexedGzip = 0,
    /// `GZTOOL` — the gztool "gzipind" format. Stub variant; reader/writer
    /// not yet ported.
    Gztool = 1,
    /// `GZTOOL_WITH_LINES` — the gztool "gzipind X" format with line
    /// offsets. Stub variant; reader/writer not yet ported.
    GztoolWithLines = 2,
}

/// Mirror of `NewlineFormat` (IndexFileFormat.hpp:92-96).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NewlineFormat {
    /// `LINE_FEED` — `\n`.
    #[default]
    LineFeed = 0,
    /// `CARRIAGE_RETURN` — `\r`.
    CarriageReturn = 1,
}

/// Mirror of `GzipIndex` (IndexFileFormat.hpp:116-170).
///
/// `windows` is a `Vec<Option<Vec<u8>>>` parallel to `checkpoints`: each
/// entry holds either the 32-KiB window bytes for that checkpoint or
/// `None` when the checkpoint has no window data attached (e.g. the very
/// last checkpoint or any whose Data Flag was 0 on read).
///
/// The rapidgzip type owns a `std::shared_ptr<WindowMap>` instead
/// (IndexFileFormat.hpp:151). This port keeps the simpler representation
/// because the WindowMap thread-safety/Zlib-compression layer is not
/// needed for pure serialization; a follow-up will adapt to WindowMap
/// when the parallel reader port wires this in.
#[derive(Clone, Debug, Default)]
pub struct GzipIndex {
    /// Mirror of `compressedSizeInBytes` (IndexFileFormat.hpp:139).
    /// `u64::MAX` means "unknown" / not-yet-finalized.
    pub compressed_size_in_bytes: u64,
    /// Mirror of `uncompressedSizeInBytes` (IndexFileFormat.hpp:140).
    pub uncompressed_size_in_bytes: u64,
    /// Mirror of `checkpointSpacing` (IndexFileFormat.hpp:146).
    pub checkpoint_spacing: u32,
    /// Mirror of `windowSizeInBytes` (IndexFileFormat.hpp:147).
    /// Must equal [`WINDOW_SIZE_BYTES`] for the indexed_gzip format
    /// (IndexFileFormat.hpp:471-474).
    pub window_size_in_bytes: u32,
    /// Mirror of `checkpoints` (IndexFileFormat.hpp:149). Sorted ascending
    /// by `compressed_offset_in_bits`.
    pub checkpoints: Vec<Checkpoint>,
    /// One entry per checkpoint: the 32-KiB window bytes (or `None` if
    /// no window data was stored). Parallel to `checkpoints`.
    pub windows: Vec<Option<Vec<u8>>>,
    /// Mirror of `hasLineOffsets` (IndexFileFormat.hpp:153).
    pub has_line_offsets: bool,
    /// Mirror of `newlineFormat` (IndexFileFormat.hpp:154).
    pub newline_format: NewlineFormat,
}

impl GzipIndex {
    fn new() -> Self {
        Self {
            compressed_size_in_bytes: u64::MAX,
            uncompressed_size_in_bytes: u64::MAX,
            checkpoint_spacing: 0,
            window_size_in_bytes: 0,
            checkpoints: Vec::new(),
            windows: Vec::new(),
            has_line_offsets: false,
            newline_format: NewlineFormat::LineFeed,
        }
    }
}

/// Errors emitted by the GZIDX reader/writer.
///
/// Mirror of the various `std::invalid_argument` / `std::runtime_error`
/// throws inside `readGzipIndex` / `writeGzipIndex`
/// (IndexFileFormat.hpp:413-684).
#[derive(Debug)]
pub enum IndexError {
    /// Wrong magic — first 5 bytes were not `b"GZIDX"`. Mirror of
    /// IndexFileFormat.hpp:438-439.
    InvalidMagic,
    /// `formatVersion > 1`. Mirror of IndexFileFormat.hpp:445-447.
    UnsupportedFormatVersion(u8),
    /// `windowSizeInBytes != 32 KiB`. Mirror of IndexFileFormat.hpp:471-474.
    InvalidWindowSize(u32),
    /// Read returned fewer bytes than requested. Mirror of the
    /// `checkedRead` `throw std::runtime_error` at IndexFileFormat.hpp:201-203.
    PrematureEof,
    /// A checkpoint's compressed offset (after `bits` correction) sits
    /// past the end of the archive. Mirror of IndexFileFormat.hpp:486-488.
    CompressedOffsetOutOfRange,
    /// A checkpoint's uncompressed offset sits past the uncompressed
    /// size. Mirror of IndexFileFormat.hpp:492-494.
    UncompressedOffsetOutOfRange,
    /// `bits >= 8` for a checkpoint. Mirror of IndexFileFormat.hpp:497-499.
    DenormalBitOffset,
    /// `bits > 0` while the byte-rounded offset is 0 (negative effective
    /// offset). Mirror of IndexFileFormat.hpp:500-503.
    DenormalNegativeBitOffset,
    /// The caller asked to write a window that isn't the right size.
    /// Mirror of IndexFileFormat.hpp:615-617.
    WindowWrongSize { got: usize, expected: u32 },
    /// Generic underlying I/O error.
    Io(io::Error),
}

impl From<io::Error> for IndexError {
    fn from(e: io::Error) -> Self {
        IndexError::Io(e)
    }
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::InvalidMagic => write!(f, "Magic bytes do not match! Expected 'GZIDX'."),
            IndexError::UnsupportedFormatVersion(v) => {
                write!(f, "Index format version {v} is newer than supported.")
            }
            IndexError::InvalidWindowSize(s) => {
                write!(f, "Only a window size of 32 KiB is supported (got {s}).")
            }
            IndexError::PrematureEof => write!(f, "Premature end of index file."),
            IndexError::CompressedOffsetOutOfRange => {
                write!(f, "Checkpoint compressed offset is after the file end.")
            }
            IndexError::UncompressedOffsetOutOfRange => {
                write!(f, "Checkpoint uncompressed offset is after the file end.")
            }
            IndexError::DenormalBitOffset => {
                write!(
                    f,
                    "Denormal compressed offset for checkpoint: bit offset >= 8."
                )
            }
            IndexError::DenormalNegativeBitOffset => write!(
                f,
                "Denormal bits for checkpoint: effectively negative offset."
            ),
            IndexError::WindowWrongSize { got, expected } => write!(
                f,
                "Window data size {got} does not match expected {expected}."
            ),
            IndexError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for IndexError {}

/// Mirror of `checkedRead` (IndexFileFormat.hpp:191-204): read exactly
/// `buf.len()` bytes from `r`, or return [`IndexError::PrematureEof`].
fn checked_read<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<(), IndexError> {
    r.read_exact(buf).map_err(|e| match e.kind() {
        io::ErrorKind::UnexpectedEof => IndexError::PrematureEof,
        _ => IndexError::Io(e),
    })
}

/// Mirror of `readValue<T>` (IndexFileFormat.hpp:208-217), specialised
/// for the little-endian fixed-width integers the GZIDX format uses.
/// Note that rapidgzip relies on the host endianness matching the writer's,
/// but in practice all writers are LE; this port enforces LE on read so
/// indexes round-trip on big-endian hosts too.
fn read_u8<R: Read>(r: &mut R) -> Result<u8, IndexError> {
    let mut b = [0u8; 1];
    checked_read(r, &mut b)?;
    Ok(b[0])
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, IndexError> {
    let mut b = [0u8; 4];
    checked_read(r, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, IndexError> {
    let mut b = [0u8; 8];
    checked_read(r, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Faithful port of `indexed_gzip::readGzipIndex`
/// (IndexFileFormat.hpp:413-593), simplified: synchronous, no thread pool,
/// no WindowMap compression — the on-disk layout is read verbatim.
///
/// `archive_size`, if `Some`, must equal the `compressedSizeInBytes`
/// field stored in the index (IndexFileFormat.hpp:459-464). Pass `None`
/// to skip that check.
pub fn read_gzip_index<R: Read>(
    r: &mut R,
    archive_size: Option<u64>,
) -> Result<GzipIndex, IndexError> {
    let mut magic = [0u8; 5];
    checked_read(r, &mut magic)?;
    if &magic != MAGIC_BYTES {
        return Err(IndexError::InvalidMagic);
    }

    let format_version = read_u8(r)?;
    if format_version > 1 {
        return Err(IndexError::UnsupportedFormatVersion(format_version));
    }
    let _reserved_flags = read_u8(r)?;

    let mut index = GzipIndex::new();
    index.compressed_size_in_bytes = read_u64_le(r)?;
    index.uncompressed_size_in_bytes = read_u64_le(r)?;
    index.checkpoint_spacing = read_u32_le(r)?;
    index.window_size_in_bytes = read_u32_le(r)?;

    if let Some(s) = archive_size {
        if s != index.compressed_size_in_bytes {
            // Mirror of IndexFileFormat.hpp:459-464 — fall through as a
            // CompressedOffsetOutOfRange-class invariant violation.
            return Err(IndexError::CompressedOffsetOutOfRange);
        }
    }

    if index.window_size_in_bytes != WINDOW_SIZE_BYTES {
        return Err(IndexError::InvalidWindowSize(index.window_size_in_bytes));
    }

    let checkpoint_count = read_u32_le(r)? as usize;
    index.checkpoints = Vec::with_capacity(checkpoint_count);
    index.windows = Vec::with_capacity(checkpoint_count);
    let mut data_flags: Vec<bool> = Vec::with_capacity(checkpoint_count);

    // Mirror of IndexFileFormat.hpp:480-527.
    for _ in 0..checkpoint_count {
        let mut cp = Checkpoint::default();

        // Compressed offset, rounded down in bytes; bit correction below.
        let compressed_bytes = read_u64_le(r)?;
        if compressed_bytes > index.compressed_size_in_bytes {
            return Err(IndexError::CompressedOffsetOutOfRange);
        }
        cp.compressed_offset_in_bits = compressed_bytes.saturating_mul(8);

        cp.uncompressed_offset_in_bytes = read_u64_le(r)?;
        if cp.uncompressed_offset_in_bytes > index.uncompressed_size_in_bytes {
            return Err(IndexError::UncompressedOffsetOutOfRange);
        }

        let bits = read_u8(r)?;
        if bits >= 8 {
            return Err(IndexError::DenormalBitOffset);
        }
        if bits > 0 {
            if cp.compressed_offset_in_bits == 0 {
                return Err(IndexError::DenormalNegativeBitOffset);
            }
            cp.compressed_offset_in_bits -= bits as u64;
        }

        // Per-format data flag (IndexFileFormat.hpp:507-516).
        let has_window = if format_version == 0 {
            // Version 0 implicitly attaches a window to every checkpoint
            // except the first.
            !index.checkpoints.is_empty()
        } else {
            read_u8(r)? != 0
        };
        data_flags.push(has_window);

        index.checkpoints.push(cp);
    }

    // Now read window blobs in checkpoint order (IndexFileFormat.hpp:562-586).
    let window_size = index.window_size_in_bytes as usize;
    for has_window in &data_flags {
        if *has_window {
            let mut buf = vec![0u8; window_size];
            checked_read(r, &mut buf)?;
            index.windows.push(Some(buf));
        } else {
            index.windows.push(None);
        }
    }

    Ok(index)
}

/// Faithful port of `indexed_gzip::writeGzipIndex`
/// (IndexFileFormat.hpp:596-684), simplified for our `Vec<Option<...>>`
/// window representation. Writes magic + header + checkpoint table +
/// window blobs in their canonical GZIDX layout.
///
/// `index.windows.len()` must equal `index.checkpoints.len()`.
pub fn write_gzip_index<W: Write>(index: &GzipIndex, w: &mut W) -> Result<(), IndexError> {
    if index.windows.len() != index.checkpoints.len() {
        return Err(IndexError::WindowWrongSize {
            got: index.windows.len(),
            expected: index.checkpoints.len() as u32,
        });
    }
    // Mirror of IndexFileFormat.hpp:615-617: every present window must
    // be exactly `windowSizeInBytes` bytes.
    let win_size = WINDOW_SIZE_BYTES as usize;
    for w in index.windows.iter().flatten() {
        if w.len() != win_size {
            return Err(IndexError::WindowWrongSize {
                got: w.len(),
                expected: WINDOW_SIZE_BYTES,
            });
        }
    }

    w.write_all(MAGIC_BYTES)?;
    w.write_all(&[1u8])?; // format version
    w.write_all(&[0u8])?; // reserved flags

    // Mirror of IndexFileFormat.hpp:624-634: spacing defaults to
    // max(WINDOW_SIZE_BYTES, min_adjacent_diff). We honor any caller-set
    // spacing >= WINDOW_SIZE_BYTES; otherwise compute the minimum
    // checkpoint-spacing as the smallest gap between adjacent
    // uncompressed offsets.
    let mut checkpoint_spacing = index.checkpoint_spacing;
    if !index.checkpoints.is_empty() && checkpoint_spacing < WINDOW_SIZE_BYTES {
        let mut min_gap: u64 = 0;
        for pair in index.checkpoints.windows(2) {
            let gap = pair[1].uncompressed_offset_in_bytes - pair[0].uncompressed_offset_in_bytes;
            if min_gap == 0 || gap < min_gap {
                min_gap = gap;
            }
        }
        checkpoint_spacing = std::cmp::max(WINDOW_SIZE_BYTES, min_gap as u32);
    }

    w.write_all(&index.compressed_size_in_bytes.to_le_bytes())?;
    w.write_all(&index.uncompressed_size_in_bytes.to_le_bytes())?;
    w.write_all(&checkpoint_spacing.to_le_bytes())?;
    w.write_all(&WINDOW_SIZE_BYTES.to_le_bytes())?;
    w.write_all(&(index.checkpoints.len() as u32).to_le_bytes())?;

    // Checkpoint table (IndexFileFormat.hpp:642-655).
    for (cp, win) in index.checkpoints.iter().zip(index.windows.iter()) {
        let bits_off = cp.compressed_offset_in_bits % 8;
        let byte_off = cp.compressed_offset_in_bits / 8 + if bits_off == 0 { 0 } else { 1 };
        w.write_all(&byte_off.to_le_bytes())?;
        w.write_all(&cp.uncompressed_offset_in_bytes.to_le_bytes())?;
        let bits = if bits_off == 0 {
            0u8
        } else {
            (8 - bits_off) as u8
        };
        w.write_all(&[bits])?;
        let has_window: u8 = if win.is_some() { 1 } else { 0 };
        w.write_all(&[has_window])?;
    }

    // Window blobs (IndexFileFormat.hpp:657-683).
    for bytes in index.windows.iter().flatten() {
        w.write_all(bytes)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index() -> GzipIndex {
        let mut idx = GzipIndex {
            compressed_size_in_bytes: 1024 * 1024,
            uncompressed_size_in_bytes: 4 * 1024 * 1024,
            checkpoint_spacing: WINDOW_SIZE_BYTES,
            window_size_in_bytes: WINDOW_SIZE_BYTES,
            ..Default::default()
        };
        // Two checkpoints — second has a window blob, first does not.
        idx.checkpoints.push(Checkpoint {
            compressed_offset_in_bits: 0,
            uncompressed_offset_in_bytes: 0,
            line_offset: 0,
        });
        idx.windows.push(None);
        idx.checkpoints.push(Checkpoint {
            compressed_offset_in_bits: 8 * 100_000 + 3,
            uncompressed_offset_in_bytes: 500_000,
            line_offset: 0,
        });
        let mut window = vec![0u8; WINDOW_SIZE_BYTES as usize];
        for (i, b) in window.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        idx.windows.push(Some(window));
        idx
    }

    #[test]
    fn round_trip_indexed_gzip() {
        let original = make_index();
        let mut buf: Vec<u8> = Vec::new();
        write_gzip_index(&original, &mut buf).unwrap();
        let parsed =
            read_gzip_index(&mut &buf[..], Some(original.compressed_size_in_bytes)).unwrap();
        assert_eq!(
            parsed.compressed_size_in_bytes,
            original.compressed_size_in_bytes
        );
        assert_eq!(
            parsed.uncompressed_size_in_bytes,
            original.uncompressed_size_in_bytes
        );
        assert_eq!(parsed.window_size_in_bytes, WINDOW_SIZE_BYTES);
        assert_eq!(parsed.checkpoints, original.checkpoints);
        assert_eq!(parsed.windows, original.windows);
    }

    #[test]
    fn rejects_bad_magic() {
        let buf = b"WRONG\x01\x00".to_vec();
        let err = read_gzip_index(&mut &buf[..], None).unwrap_err();
        assert!(matches!(err, IndexError::InvalidMagic));
    }

    #[test]
    fn rejects_bad_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC_BYTES);
        buf.push(99); // bad version
        buf.push(0);
        let err = read_gzip_index(&mut &buf[..], None).unwrap_err();
        assert!(matches!(err, IndexError::UnsupportedFormatVersion(99)));
    }

    #[test]
    fn rejects_wrong_window_size() {
        let mut idx = make_index();
        idx.window_size_in_bytes = 16 * 1024; // wrong
        let mut buf = Vec::new();
        // Manually write the header with the wrong window size so we hit
        // the read path's check.
        buf.extend_from_slice(MAGIC_BYTES);
        buf.push(1);
        buf.push(0);
        buf.extend_from_slice(&idx.compressed_size_in_bytes.to_le_bytes());
        buf.extend_from_slice(&idx.uncompressed_size_in_bytes.to_le_bytes());
        buf.extend_from_slice(&idx.checkpoint_spacing.to_le_bytes());
        buf.extend_from_slice(&16384u32.to_le_bytes()); // bad window size
        buf.extend_from_slice(&0u32.to_le_bytes());
        let err = read_gzip_index(&mut &buf[..], None).unwrap_err();
        assert!(matches!(err, IndexError::InvalidWindowSize(16384)));
    }

    #[test]
    fn checkpoint_bit_offset_round_trips() {
        let mut idx = make_index();
        // Force a denormal-friendly bit offset.
        idx.checkpoints[1].compressed_offset_in_bits = 8 * 12345 + 5; // 5 bits past byte 12345
        let mut buf = Vec::new();
        write_gzip_index(&idx, &mut buf).unwrap();
        let parsed = read_gzip_index(&mut &buf[..], None).unwrap();
        assert_eq!(
            parsed.checkpoints[1].compressed_offset_in_bits,
            8 * 12345 + 5
        );
    }
}
