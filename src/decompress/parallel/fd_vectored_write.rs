//! Zero-copy vectored output for the parallel-SM consumer.
//!
//! ## Why this exists (the S4 lever)
//!
//! The in-order consumer (`chunk_fetcher::drain_one_pending`) is the
//! single point that materializes every decoded byte to the output
//! sink. Before this module it did so via `BufWriter::write_all` —
//! a single-core `memcpy` of all ~503 MB of freshly cross-core-decoded
//! payload through a 1 MiB `BufWriter` staging buffer. A
//! positive-controlled removal oracle (S4, 2026-06-01) proved that
//! nulling that write yields **+27% T8 wall** (OFF 1.81× → ON 1.32×
//! vs rapidgzip), thread-local and orthogonal to the dead granule
//! levers. See `plans/wall-progress.md` 2026-06-01 "S4 CONSUMER-NULL
//! CEILING = GO".
//!
//! rapidgzip writes the SAME bytes via `writev` over iovecs of its
//! decode buffers (zero userspace gather; the consumer pays ~syscall
//! only): `ChunkData.hpp:794 writeAll` → `DecodedData.hpp:530 toIoVec`
//! → `FileUtils.hpp:765 writeAllToFdVector`. This module mirrors
//! `writeAllToFdVector`: it `writev`'s the chunk's payload iovecs
//! (`narrowed`, then `data[prefix..]`) DIRECTLY to the output fd,
//! eliminating the `BufWriter` gather copy entirely on the fast path.
//!
//! ## Plumbing choice (DOCUMENTED per mission deliverable)
//!
//! The consumer is generic over `W: std::io::Write`; the concrete
//! sinks are `BufWriter<File>`, `BufWriter<StdoutLock>` (and the
//! `CountingWriter`-wrapped stdin variant), `MmapWriter`, and test
//! writers. Three designs were considered:
//!
//!  1. A `ChunkVectoredWrite` trait with a blanket default impl
//!     (sequential) + specialized impls for fd-backed writers.
//!     REJECTED: Rust coherence forbids specializing a blanket impl
//!     without the unstable `specialization` feature; without the
//!     blanket, the bound propagates onto every public-API caller
//!     (incl. `Vec<u8>` from `decompress_gzip_to_vec` and arbitrary
//!     user/test writers), forcing an impl for each.
//!
//!  2. A concrete `enum` writer wrapping the sink. REJECTED:
//!     `decompress_gzip_libdeflate` dispatches the SAME `writer` to
//!     both the SM path AND the bgzf / multi-member paths, so wrapping
//!     would pollute those unrelated paths too.
//!
//!  3. **CHOSEN:** thread a single `out_fd: Option<RawFd>` parameter
//!     down only the SM consumer chain (`decompress_single_member_for`
//!     → `decompress_parallel` → `read_parallel_sm` → `drive` →
//!     `drive_impl` → `consumer_loop` → `drain_one_pending`). It
//!     touches only SM-internal functions, leaves the generic `W:
//!     Write` signature and every other path / public API untouched,
//!     and sidesteps all coherence battles. `io.rs` computes the fd
//!     once (via `AsRawFd`) for the fd-backed sinks and passes `None`
//!     for `MmapWriter` / pipes-where-unbeneficial / test writers,
//!     which keep the existing `write_all` path.
//!
//! ## The no-split-write-paths invariant (CORRECTNESS HAZARD)
//!
//! You MUST NOT `writev` some bytes direct-to-fd while other bytes sit
//! in the `BufWriter`: the `BufWriter` holds buffered bytes not yet at
//! the fd, so a direct `writev` would jump AHEAD of them and corrupt /
//! interleave the output. The parallel-SM path writes ONLY decoded
//! chunk payloads — no header/trailer is ever written (those are
//! parsed). So when `out_fd` is `Some`, the consumer routes ALL
//! payload writes through `writev` to the fd and NEVER touches the
//! `BufWriter` for payload. `io.rs` flushes the (empty) `BufWriter`
//! once before handing off, so the buffer is provably empty and the
//! fd cursor is where the next byte belongs.

// `parallel_sm` can be enabled on x86_64-windows; `writev` is unix
// only. The consumer only ever passes `Some(fd)` on unix (the fd is
// computed via `AsRawFd` under `#[cfg(unix)]` in `io.rs`), so this
// whole module is unix-gated and the non-unix consumer takes the
// buffered fallback.
#![cfg(all(parallel_sm, unix))]

use std::io;

/// `writev` the two payload slices (`narrowed`, then `data`) directly
/// to `fd` (a raw file descriptor, `i32`), looping to handle partial
/// writes. Mirror of vendor `FileUtils.hpp:765 writeAllToFdVector`.
///
/// `narrowed` may be empty (a clean chunk with no markers → only the
/// `data` iovec is submitted). `data` may be empty (degenerate). When
/// both are empty this is a no-op.
///
/// `writev` is permitted to short-write (it returns the number of
/// bytes consumed across the iovecs); we advance the base/len of the
/// first not-fully-written iovec and re-issue until every byte is on
/// the fd. Byte order is `narrowed` then `data`, matching the
/// `write_all` sequence it replaces.
pub fn writev_all_to_fd(fd: i32, narrowed: &[u8], data: &[u8]) -> io::Result<()> {
    // Build up to two iovecs, skipping empty slices so we never submit
    // a zero-length iovec (harmless but pointless) and so an all-empty
    // call short-circuits.
    let mut iovs: [libc::iovec; 2] = unsafe { std::mem::zeroed() };
    let mut n_iovs = 0usize;
    if !narrowed.is_empty() {
        iovs[n_iovs] = libc::iovec {
            iov_base: narrowed.as_ptr() as *mut libc::c_void,
            iov_len: narrowed.len(),
        };
        n_iovs += 1;
    }
    if !data.is_empty() {
        iovs[n_iovs] = libc::iovec {
            iov_base: data.as_ptr() as *mut libc::c_void,
            iov_len: data.len(),
        };
        n_iovs += 1;
    }
    if n_iovs == 0 {
        return Ok(());
    }

    // `iov_ptr` walks forward over the iovec array as leading iovecs
    // are fully drained; `iovs[idx]` carries the (possibly advanced)
    // partial state of the current head iovec.
    let mut idx = 0usize;
    while idx < n_iovs {
        let remaining = n_iovs - idx;
        // SAFETY: `iovs[idx..n_iovs]` are valid iovecs pointing into
        // the live `narrowed` / `data` slices (borrowed for the whole
        // call); `remaining` is in 1..=2.
        let written = unsafe { libc::writev(fd, iovs.as_ptr().add(idx), remaining as libc::c_int) };
        if written < 0 {
            let err = io::Error::last_os_error();
            // Retry on EINTR exactly like a well-behaved write loop.
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        // `writev` returning 0 with iovecs of nonzero total length
        // would be a pathological/closed sink; treat as WriteZero to
        // avoid an infinite loop (matches `Write::write_all` semantics).
        if written == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "writev wrote zero bytes",
            ));
        }
        // Consume `written` bytes across the head iovec(s), advancing
        // `idx` past any fully-drained iovecs and partially advancing
        // the new head.
        let mut to_consume = written as usize;
        while idx < n_iovs && to_consume > 0 {
            let head_len = iovs[idx].iov_len;
            if to_consume >= head_len {
                // This iovec is fully written.
                to_consume -= head_len;
                idx += 1;
            } else {
                // Partial: advance base + shrink len, then re-issue.
                // SAFETY: `to_consume < head_len`, so the new base is
                // still within the original slice.
                iovs[idx].iov_base =
                    unsafe { (iovs[idx].iov_base as *mut u8).add(to_consume) as *mut libc::c_void };
                iovs[idx].iov_len = head_len - to_consume;
                to_consume = 0;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};
    use std::os::unix::io::AsRawFd;

    fn roundtrip(narrowed: &[u8], data: &[u8]) -> Vec<u8> {
        let mut tmp = tempfile::tempfile().expect("tempfile");
        writev_all_to_fd(tmp.as_raw_fd(), narrowed, data).expect("writev");
        tmp.seek(SeekFrom::Start(0)).unwrap();
        let mut out = Vec::new();
        tmp.read_to_end(&mut out).unwrap();
        out
    }

    #[test]
    fn both_slices() {
        let out = roundtrip(b"AAA", b"BBBB");
        assert_eq!(out, b"AAABBBB");
    }

    #[test]
    fn empty_narrowed_clean_chunk() {
        // Clean chunk → 1 iovec, data only.
        let out = roundtrip(b"", b"hello world");
        assert_eq!(out, b"hello world");
    }

    #[test]
    fn empty_data() {
        let out = roundtrip(b"only-markers", b"");
        assert_eq!(out, b"only-markers");
    }

    #[test]
    fn both_empty_is_noop() {
        let out = roundtrip(b"", b"");
        assert_eq!(out, b"");
    }

    #[test]
    fn large_payload_byte_order_preserved() {
        // Larger than a typical pipe buffer to exercise the partial
        // -write advance loop when the sink is a pipe.
        let narrowed: Vec<u8> = (0..50_000u32).map(|i| (i % 251) as u8).collect();
        let data: Vec<u8> = (0..200_000u32).map(|i| ((i * 7) % 253) as u8).collect();
        let out = roundtrip(&narrowed, &data);
        let mut expected = narrowed.clone();
        expected.extend_from_slice(&data);
        assert_eq!(out, expected);
    }

    #[test]
    fn pipe_partial_writev_advance() {
        // A pipe forces short writev's (pipe buffer is ~64 KiB), which
        // exercises the base/len advance loop and the multi-iovec
        // consume path. A reader thread drains so the writer never
        // blocks forever.
        use std::io::Read as _;
        let mut fds = [0 as libc::c_int; 2];
        let rc = unsafe { libc::pipe(fds.as_mut_ptr()) };
        assert_eq!(rc, 0, "pipe()");
        let (read_fd, write_fd) = (fds[0], fds[1]);

        let narrowed: Vec<u8> = (0..70_000u32).map(|i| (i % 251) as u8).collect();
        let data: Vec<u8> = (0..300_000u32).map(|i| ((i * 13) % 247) as u8).collect();
        let mut expected = narrowed.clone();
        expected.extend_from_slice(&data);
        let total = expected.len();

        let reader = std::thread::spawn(move || {
            let mut f = unsafe { std::fs::File::from_raw_fd_compat(read_fd) };
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            buf
        });

        writev_all_to_fd(write_fd, &narrowed, &data).expect("writev to pipe");
        // Close the write end so the reader sees EOF.
        unsafe { libc::close(write_fd) };

        let got = reader.join().unwrap();
        assert_eq!(got.len(), total);
        assert_eq!(got, expected);
    }

    // Small shim: `File::from_raw_fd` is unsafe + needs the trait in
    // scope; wrap it so the test reads cleanly.
    trait FromRawFdCompat {
        unsafe fn from_raw_fd_compat(fd: libc::c_int) -> Self;
    }
    impl FromRawFdCompat for std::fs::File {
        unsafe fn from_raw_fd_compat(fd: libc::c_int) -> Self {
            use std::os::unix::io::FromRawFd;
            std::fs::File::from_raw_fd(fd)
        }
    }
}
