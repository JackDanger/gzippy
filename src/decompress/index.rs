//! Indexed/random-access decompression for gzip files.
//!
//! Builds a seek index at configurable intervals, allowing decompression to start
//! from arbitrary byte offsets without reading the entire file from the beginning.
//! Each checkpoint records: compressed bit offset, uncompressed offset, and the 32KB
//! LZ77 window needed to resume inflate.
//!
//! Index format:
//!   Magic: "GZIDX\x01" (6 bytes, version=1)
//!   deflate_offset: u32
//!   total_uncompressed_size: u64
//!   point_count: u32
//!   reserved: u16
//!   [IndexPoint * point_count]
//!
//! IndexPoint (32784 bytes each):
//!   compressed_bit_offset: u64
//!   uncompressed_offset: u64
//!   window: [u8; 32768]

use std::io::Write;

use crate::error::{GzippyError, GzippyResult};

const MAGIC: &[u8] = b"GZIDX\x01";
const WINDOW_SIZE: usize = 32768;

/// A single checkpoint in the index.
#[derive(Clone)]
pub struct IndexPoint {
    pub compressed_bit_offset: u64,
    pub uncompressed_offset: u64,
    pub window: [u8; WINDOW_SIZE],
}

/// A complete seek index.
pub struct SeekIndex {
    pub points: Vec<IndexPoint>,
    pub total_uncompressed_size: u64,
    pub deflate_offset: usize,
}

/// Build a seek index for a gzip file.
///
/// Scans the deflate stream and records checkpoints at regular intervals.
/// Returns an error if the file is multi-member or has unsupported format.
pub fn build_index(gzip_data: &[u8], interval_bytes: usize) -> GzippyResult<SeekIndex> {
    // Check magic bytes
    if gzip_data.len() < 2 || gzip_data[0] != 0x1f || gzip_data[1] != 0x8b {
        return Err(GzippyError::invalid_argument("Not a gzip file"));
    }

    // Reject multi-member and gzippy-parallel formats
    if crate::decompress::format::is_likely_multi_member(gzip_data) {
        return Err(GzippyError::invalid_argument(
            "Multi-member gzip files are not yet supported for indexing",
        ));
    }

    if crate::decompress::format::has_bgzf_markers(gzip_data) {
        return Err(GzippyError::invalid_argument(
            "gzippy-parallel files (BGZF format) have built-in block boundaries; \
             use BGZF block headers directly instead of this index",
        ));
    }

    // Parse gzip header to find where deflate data starts
    let deflate_offset = crate::decompress::format::parse_gzip_header_size(gzip_data)
        .ok_or_else(|| GzippyError::invalid_argument("Invalid gzip header"))?;

    if deflate_offset >= gzip_data.len() {
        return Err(GzippyError::invalid_argument(
            "Gzip header extends past file end",
        ));
    }

    let deflate_data = &gzip_data[deflate_offset..];

    // Scan the deflate stream to get checkpoints
    let scan_result = crate::decompress::scan_inflate::scan_deflate_fast(
        deflate_data,
        interval_bytes,
        0, // expected_output_size hint
    )
    .map_err(|e| GzippyError::decompression(format!("Scan failed: {}", e)))?;

    // Convert ScanCheckpoint to IndexPoint
    let mut points = Vec::new();
    for checkpoint in scan_result.checkpoints {
        // Compute absolute bit offset from deflate start.
        // input_byte_pos is the byte position in the deflate stream,
        // bitsleft is the number of bits still in the buffer (not yet consumed).
        // So consumed bits = input_byte_pos * 8 - bitsleft
        let input_bits = (checkpoint.input_byte_pos as u64).saturating_mul(8);
        let compressed_bit_offset = input_bits.saturating_sub(checkpoint.bitsleft as u64);

        let mut window = [0u8; WINDOW_SIZE];
        let window_len = checkpoint.window.len().min(WINDOW_SIZE);
        window[..window_len].copy_from_slice(&checkpoint.window[..window_len]);

        points.push(IndexPoint {
            compressed_bit_offset,
            uncompressed_offset: checkpoint.output_offset as u64,
            window,
        });
    }

    // Ensure we have at least one checkpoint at offset 0 or the beginning
    // This allows seeking even if the entire file is in a single deflate block
    if points.is_empty() {
        // Create a single checkpoint at offset 0 (beginning)
        // This requires decompressing from the start
        points.push(IndexPoint {
            compressed_bit_offset: 0,
            uncompressed_offset: 0,
            window: [0u8; WINDOW_SIZE],
        });
    }

    Ok(SeekIndex {
        points,
        total_uncompressed_size: scan_result.total_output_size as u64,
        deflate_offset,
    })
}

/// Serialize an index to a writer.
pub fn serialize_index(index: &SeekIndex, writer: &mut dyn Write) -> GzippyResult<()> {
    // Magic
    writer.write_all(MAGIC).map_err(GzippyError::Io)?;

    // Header
    writer
        .write_all(&(index.deflate_offset as u32).to_le_bytes())
        .map_err(GzippyError::Io)?;
    writer
        .write_all(&index.total_uncompressed_size.to_le_bytes())
        .map_err(GzippyError::Io)?;
    writer
        .write_all(&(index.points.len() as u32).to_le_bytes())
        .map_err(GzippyError::Io)?;
    writer
        .write_all(&[0u8; 2]) // reserved
        .map_err(GzippyError::Io)?;

    // Points
    for point in &index.points {
        writer
            .write_all(&point.compressed_bit_offset.to_le_bytes())
            .map_err(GzippyError::Io)?;
        writer
            .write_all(&point.uncompressed_offset.to_le_bytes())
            .map_err(GzippyError::Io)?;
        writer.write_all(&point.window).map_err(GzippyError::Io)?;
    }

    Ok(())
}

/// Load an index from a slice.
pub fn load_index(data: &[u8]) -> GzippyResult<SeekIndex> {
    if data.len() < MAGIC.len() + 4 + 8 + 4 + 2 {
        return Err(GzippyError::parse("Index file too small"));
    }

    // Check magic
    if &data[..MAGIC.len()] != MAGIC {
        return Err(GzippyError::parse("Invalid index file magic"));
    }

    let mut offset = MAGIC.len();

    // Parse header
    let deflate_offset = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    offset += 4;

    let total_uncompressed_size = u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]);
    offset += 8;

    let point_count = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    offset += 4;

    offset += 2; // skip reserved

    // Parse points
    let point_size = 8 + 8 + WINDOW_SIZE;
    let expected_size = offset + point_count * point_size;
    if data.len() < expected_size {
        return Err(GzippyError::parse(format!(
            "Index file truncated: expected {} bytes, got {}",
            expected_size,
            data.len()
        )));
    }

    let mut points = Vec::new();
    for _ in 0..point_count {
        let compressed_bit_offset = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        let uncompressed_offset = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        let mut window = [0u8; WINDOW_SIZE];
        window.copy_from_slice(&data[offset..offset + WINDOW_SIZE]);
        offset += WINDOW_SIZE;

        points.push(IndexPoint {
            compressed_bit_offset,
            uncompressed_offset,
            window,
        });
    }

    // Validate monotonicity
    for i in 1..points.len() {
        if points[i].uncompressed_offset <= points[i - 1].uncompressed_offset {
            return Err(GzippyError::parse(
                "Index points are not monotonically increasing",
            ));
        }
    }

    Ok(SeekIndex {
        points,
        total_uncompressed_size,
        deflate_offset,
    })
}

/// Seek to an uncompressed offset and decompress from there.
///
/// # Arguments
/// * `gzip_data` - Full gzip file data
/// * `index` - The seek index
/// * `uncompressed_offset` - Target byte offset in the uncompressed stream
/// * `max_bytes` - Maximum number of bytes to decompress (or u64::MAX for all)
/// * `writer` - Output writer
pub fn seek_decompress<W: Write>(
    gzip_data: &[u8],
    index: &SeekIndex,
    uncompressed_offset: u64,
    max_bytes: u64,
    writer: &mut W,
) -> GzippyResult<u64> {
    // Validate that the gzip file matches the index
    if gzip_data.len() < 2 || gzip_data[0] != 0x1f || gzip_data[1] != 0x8b {
        return Err(GzippyError::invalid_argument("Not a gzip file"));
    }

    // Handle special case: seeking to offset 0 (decompress from beginning)
    if uncompressed_offset == 0 {
        if index.points.is_empty() {
            return Err(GzippyError::invalid_argument("No checkpoints in index"));
        }
        let checkpoint = &index.points[0];
        if checkpoint.uncompressed_offset != 0 {
            return Err(GzippyError::invalid_argument(format!(
                "Cannot seek to 0; earliest checkpoint is at {}",
                checkpoint.uncompressed_offset
            )));
        }
        // Fall through to decompress from first checkpoint at offset 0
    }

    // Find the checkpoint at or before the target offset
    let checkpoint_idx = index
        .points
        .binary_search_by_key(&uncompressed_offset, |p| p.uncompressed_offset)
        .unwrap_or_else(|idx| {
            if idx == 0 {
                return 0; // Will be checked below
            }
            // Find the checkpoint just before our target
            idx - 1
        });

    // Validate checkpoint bounds
    if checkpoint_idx >= index.points.len() {
        return Err(GzippyError::invalid_argument(format!(
            "Requested offset {} is beyond uncompressed size {}",
            uncompressed_offset, index.total_uncompressed_size
        )));
    }

    let checkpoint = &index.points[checkpoint_idx];

    // Validate that the checkpoint is at or before the requested offset
    // (unless we're seeking to offset 0 and the first checkpoint is at 0)
    if checkpoint.uncompressed_offset > uncompressed_offset {
        return Err(GzippyError::invalid_argument(format!(
            "Seek offset {} is before earliest checkpoint at {}",
            uncompressed_offset, checkpoint.uncompressed_offset
        )));
    }

    // Calculate how many bytes to skip at the start of the decompressed block
    let skip_bytes = (uncompressed_offset - checkpoint.uncompressed_offset) as usize;

    // Get deflate data, starting from where the checkpoint's deflate data begins
    let deflate_data = &gzip_data[index.deflate_offset..];

    // Decompress from the checkpoint's bit offset
    // We need to decompress at least skip_bytes + up to max_bytes
    let max_output =
        skip_bytes.saturating_add(max_bytes.min(u64::MAX - skip_bytes as u64) as usize);
    let max_output = max_output.max(skip_bytes + 1); // At least enough to skip

    let (output, _) = crate::backends::inflate_bit::decompress_deflate_from_bit_with_end(
        deflate_data,
        checkpoint.compressed_bit_offset as usize,
        &checkpoint.window,
        max_output,
    )
    .ok_or_else(|| GzippyError::decompression("Failed to decompress from checkpoint"))?;

    // Skip the bytes we don't need
    let remaining = if skip_bytes < output.len() {
        &output[skip_bytes..]
    } else {
        &[]
    };

    // Write up to max_bytes
    let to_write = remaining.len().min(max_bytes as usize);
    writer.write_all(&remaining[..to_write])?;
    writer.flush()?;

    Ok(to_write as u64)
}
