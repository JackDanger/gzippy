//! §3.13 mmap direct decode — output writes land directly into an
//! mmap'd file region; the kernel handles writeback asynchronously.
//!
//! For GB-scale inputs the in-RAM working set is just per-chunk
//! decoder state + tail window (~32 KiB per worker) — the bulk
//! output bytes go straight into the page cache as the decoder
//! emits them, and the kernel pages out to the backing file under
//! memory pressure.
//!
//! ## Sizing
//!
//! The output file is pre-truncated to `ISIZE` (from the gzip
//! trailer, mod 2^32). For multi-member files where the true output
//! size exceeds 4 GiB, the caller MUST pre-size larger and trim on
//! finalize; today's `MmapWriter::open_pre_sized` accepts an exact
//! size and the call site picks it (e.g. via the `ultra_fast_inflate`
//! multi-member detection in `decompress::io`).
//!
//! ## Why a custom Write impl
//!
//! `memmap2::MmapMut` doesn't implement `Write` directly. We wrap
//! it with a cursor that tracks `pos` and bounds-checks each call;
//! at finalize the file is `set_len`'d to the actual byte count
//! (in case the caller over-estimated).

use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

pub struct MmapWriter {
    /// The file handle is kept alive so the underlying fd stays open
    /// for the lifetime of the mapping.
    file: File,
    /// Mutable mmap covering `[0, capacity)`.
    mmap: MmapMut,
    /// Current write cursor.
    pos: usize,
    /// Original mapped length; on finalize, the file is truncated to
    /// `pos` so the on-disk size matches the actual bytes written.
    capacity: usize,
}

impl MmapWriter {
    /// Open `path` for writing, pre-size to `expected_size`, and
    /// return a writer that drops bytes into the mmap'd region.
    ///
    /// `expected_size` should match the gzip trailer's ISIZE (or a
    /// safe upper bound for multi-member files). The file is opened
    /// O_TRUNC so any existing contents are lost.
    pub fn open_pre_sized<P: AsRef<Path>>(path: P, expected_size: usize) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        // Empty file → 0-length mmap which can't be written. For zero-
        // sized output, skip the mmap and treat the writer as
        // immediately full.
        if expected_size == 0 {
            return Ok(Self {
                file,
                mmap: MmapOptions::new().len(0).map_anon()?,
                pos: 0,
                capacity: 0,
            });
        }
        file.set_len(expected_size as u64)?;
        // SAFETY: `file` is the file the mmap will be backed by; we
        // hold it for the lifetime of MmapWriter, so the underlying
        // fd doesn't close out from under the mapping.
        let mmap = unsafe { MmapOptions::new().len(expected_size).map_mut(&file)? };
        Ok(Self {
            file,
            mmap,
            pos: 0,
            capacity: expected_size,
        })
    }

    /// Actual bytes written so far.
    #[allow(dead_code)] // unit-test surface; production reads via finalize()
    pub fn bytes_written(&self) -> usize {
        self.pos
    }

    /// Finalize: msync the mapping to disk and truncate the file to
    /// the actual bytes written.
    pub fn finalize(self) -> io::Result<usize> {
        if self.capacity > 0 {
            self.mmap.flush()?;
        }
        // Truncate to actual size in case the caller over-estimated.
        if self.pos < self.capacity {
            self.file.set_len(self.pos as u64)?;
        }
        Ok(self.pos)
    }
}

impl Write for MmapWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.capacity == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "MmapWriter: zero-sized output",
            ));
        }
        let remaining = self.capacity - self.pos;
        if remaining == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "MmapWriter: output region full (ISIZE undersized?)",
            ));
        }
        let take = buf.len().min(remaining);
        self.mmap[self.pos..self.pos + take].copy_from_slice(&buf[..take]);
        self.pos += take;
        Ok(take)
    }

    fn flush(&mut self) -> io::Result<()> {
        // Don't msync per-call — that would defeat the async-writeback
        // win. Caller invokes `finalize()` to flush + truncate.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use tempfile::tempdir;

    #[test]
    fn mmap_writer_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.bin");
        let mut w = MmapWriter::open_pre_sized(&path, 1024).unwrap();
        let payload = b"hello mmap world";
        w.write_all(payload).unwrap();
        assert_eq!(w.bytes_written(), payload.len());
        let n = w.finalize().unwrap();
        assert_eq!(n, payload.len());

        let mut got = Vec::new();
        File::open(&path).unwrap().read_to_end(&mut got).unwrap();
        assert_eq!(got, payload);
    }

    #[test]
    fn mmap_writer_truncates_on_finalize() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.bin");
        // Pre-size 1024 but only write 16.
        let mut w = MmapWriter::open_pre_sized(&path, 1024).unwrap();
        w.write_all(b"hello mmap world").unwrap();
        w.finalize().unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), 16);
    }

    #[test]
    fn mmap_writer_zero_sized() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("zero.bin");
        let w = MmapWriter::open_pre_sized(&path, 0).unwrap();
        let n = w.finalize().unwrap();
        assert_eq!(n, 0);
        // File should exist and be empty.
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), 0);
    }

    #[test]
    fn mmap_writer_partial_write_returns_n() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("small.bin");
        let mut w = MmapWriter::open_pre_sized(&path, 4).unwrap();
        let buf = b"hello world"; // larger than capacity
        let n = w.write(buf).unwrap();
        assert_eq!(n, 4, "should write only 4 bytes (capacity)");
        let next = w.write(&buf[4..]).unwrap_err();
        assert_eq!(next.kind(), io::ErrorKind::WriteZero);
    }
}
