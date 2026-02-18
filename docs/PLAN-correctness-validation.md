# Correctness Validation

## Goal
Make gzippy safe for production use by adding integrity validation that gzip/pigz users expect.

## What's needed

### 1. CRC32 validation in BGZF parallel path
- `bgzf.rs:inflate_into_libdeflate()` uses `libdeflate_deflate_decompress` (raw deflate)
- This strips the gzip header/trailer, so CRC32 is never checked
- Fix: after decompressing each BGZF block, read the CRC32 from the trailer and validate
- The single-member path (`gzip_decompress_ex`) already validates CRC32 internally

### 2. Symlink handling
- `compression.rs` and `decompression.rs` only check `is_dir()` and `is_file()`
- Symlinks, device files, FIFOs, and sockets are not handled
- gzip behavior: skip symlinks without `-f`, follow with `-f`; skip devices/FIFOs always
- Add checks before opening files in both compress and decompress paths

### 3. Truncated file detection
- libdeflate returns `BadData` for truncated streams, but edge cases may exist
- The BGZF path may silently succeed on partial data if block markers parse correctly
- Add explicit size checks and trailer presence validation

### 4. FHCRC validation
- Code parses the FHCRC flag (0x02) but never validates the 2-byte header CRC
- Low priority since almost no tools set this flag

## Approach
- Add `crc32fast` computation alongside BGZF parallel decompression
- Compare computed CRC against trailer CRC after each block
- Add file type detection (symlink/device/FIFO) early in compress_file/decompress_file
- Add integration tests for corrupt data, truncated files, symlinks
