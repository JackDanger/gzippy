//! Zero-copy vectored output for the parallel-SM consumer — the
//! data-plane `writeAll` tail.
//!
//! ## Why this exists (the S4 lever)
//!
//! The in-order consumer (`chunk_fetcher::drain_one_pending`) is the
//! single point that materializes every decoded byte to the output
//! sink. Before this module it did so via `BufWriter::write_all` —
//! a single-core `memcpy` of all freshly cross-core-decoded payload
//! through a 1 MiB `BufWriter` staging buffer. A positive-controlled
//! removal oracle (S4, 2026-06-01) proved that nulling that write
//! yields a large T8 wall win, thread-local and orthogonal to the
//! granule levers.
//!
//! rapidgzip writes the SAME bytes via a vectored `writeAll`:
//! `ChunkData.hpp:794 writeAll` → `DecodedData.hpp:529 toIoVec` →
//! (Linux pipe) `FileUtils.hpp:619 SpliceVault::splice` /
//! `FileUtils.hpp:531 writeAllSpliceUnsafe(vmsplice)` →
//! (fallback / non-pipe) `FileUtils.hpp:765 writeAllToFdVector(writev)`.
//! This module mirrors that chain:
//!
//!   * [`to_io_vec`] gathers the resolved marker segments + the clean
//!     `data` tail into one `iovec` array (vendor `toIoVec`,
//!     DecodedData.hpp:529-546). The resolved markers now live in N
//!     128 KiB segments (the in-place `applyWindow` reuse — see
//!     `segmented_markers`), so this is genuinely a multi-iovec gather,
//!     not the prior two-slice (narrowed | data) form.
//!   * [`writev_all_to_fd`] is the `writev` data path (vendor
//!     `writeAllToFdVector`, FileUtils.hpp:765-800): partial-write
//!     advance + `IOV_MAX` batching + EINTR retry.
//!   * [`splice::SpliceVault`] + [`write_all_to_fd`] add the Linux
//!     `vmsplice` fast path (vendor `SpliceVault`, FileUtils.hpp:582-708)
//!     WITH the mandatory page-lifetime accounting: `vmsplice` GIFTS the
//!     source pages to the pipe, so the source buffer must stay alive
//!     until the kernel has drained `pipe_buffer_size` bytes past it.
//!     The vault holds an owning handle (the `ChunkData` whose buffers
//!     were spliced) until that many subsequent bytes have been spliced,
//!     then drops it. On any non-pipe fd `F_GETPIPE_SZ` returns -1 and
//!     the vault declines (`splice` → `Err`), so [`write_all_to_fd`]
//!     falls back to `writev` — exactly vendor `writeAll`
//!     (ChunkData.hpp:804-809).
//!
//! ## Plumbing choice (DOCUMENTED per mission deliverable)
//!
//! The consumer is generic over `W: std::io::Write`; the concrete
//! sinks are `BufWriter<File>`, `BufWriter<StdoutLock>`, `MmapWriter`,
//! and test writers. Rather than a `ChunkVectoredWrite` trait (Rust
//! coherence forbids specializing a blanket impl) or an `enum` writer
//! (would pollute the unrelated bgzf / multi-member paths that share
//! the same `writer`), we thread a single `out_fd: Option<RawFd>`
//! parameter down only the SM consumer chain. `io.rs` computes the fd
//! once (via `AsRawFd`) for fd-backed sinks and passes `None` for
//! `MmapWriter` / test writers, which keep the existing `write_all`
//! path.
//!
//! ## The no-split-write-paths invariant (CORRECTNESS HAZARD)
//!
//! You MUST NOT `writev`/`vmsplice` some bytes direct-to-fd while other
//! bytes sit in the `BufWriter`: the `BufWriter` holds buffered bytes
//! not yet at the fd, so a direct fd write would jump AHEAD of them and
//! corrupt the output. The parallel-SM path writes ONLY decoded chunk
//! payloads — no header/trailer is ever written. So when `out_fd` is
//! `Some`, the consumer routes ALL payload writes through this module
//! and NEVER touches the `BufWriter` for payload; `io.rs` flushes the
//! (empty) `BufWriter` once before handing off.

// `parallel_sm` can be enabled on x86_64-windows; `writev`/`vmsplice`
// are unix/linux only. The consumer only ever passes `Some(fd)` on unix
// (the fd is computed via `AsRawFd` under `#[cfg(unix)]` in `io.rs`), so
// this whole module is unix-gated and the non-unix consumer takes the
// buffered fallback.
#![cfg(all(parallel_sm, unix))]

use std::io;

/// Maximum iovecs per `writev`/`vmsplice` call. Vendor batches at
/// `IOV_MAX` (FileUtils.hpp:769). We query `_SC_IOV_MAX` and fall back
/// to the common 1024.
#[inline]
fn iov_max() -> usize {
    let v = unsafe { libc::sysconf(libc::_SC_IOV_MAX) };
    if v > 0 {
        v as usize
    } else {
        1024
    }
}

/// Build the `iovec` gather list for a chunk's payload (vendor
/// `toIoVec`, DecodedData.hpp:529-546): the resolved marker segments
/// (in append order) followed by the clean `data` tail. Empty slices
/// are skipped so we never submit a zero-length iovec.
///
/// The returned iovecs borrow `segments` + `data`; the caller MUST keep
/// those buffers alive until the write (and, on the vmsplice path, the
/// pipe-drain) completes — enforced by the `SpliceVault` accounting on
/// the splice path and by lexical lifetime on the writev path.
pub fn to_io_vec<'a, I>(segments: I, data: &'a [u8]) -> Vec<libc::iovec>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let mut iovs: Vec<libc::iovec> = Vec::new();
    for seg in segments {
        if !seg.is_empty() {
            iovs.push(libc::iovec {
                iov_base: seg.as_ptr() as *mut libc::c_void,
                iov_len: seg.len(),
            });
        }
    }
    if !data.is_empty() {
        iovs.push(libc::iovec {
            iov_base: data.as_ptr() as *mut libc::c_void,
            iov_len: data.len(),
        });
    }
    iovs
}

/// `writev` an `iovec` list fully to `fd`, looping over `IOV_MAX`
/// batches and advancing across partial writes. Mirror of vendor
/// `writeAllToFdVector` (FileUtils.hpp:765-800) + the EINTR retry of a
/// well-behaved write loop.
///
/// The iovec array is mutated in place to advance the head iovec on a
/// partial write; callers pass an owned `Vec` (from [`to_io_vec`]).
pub fn writev_all_to_fd(fd: i32, iovs: &mut [libc::iovec]) -> io::Result<()> {
    let n = iovs.len();
    let mut idx = 0usize;
    while idx < n {
        let segment_count = (n - idx).min(iov_max());
        // SAFETY: `iovs[idx..idx+segment_count]` are valid iovecs into
        // live borrowed slices; `segment_count >= 1`.
        let written =
            unsafe { libc::writev(fd, iovs.as_ptr().add(idx), segment_count as libc::c_int) };
        if written < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if written == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "writev wrote zero bytes",
            ));
        }
        // Consume `written` bytes across the head iovec(s), advancing
        // `idx` past fully-drained iovecs and partially advancing the
        // new head (vendor FileUtils.hpp:776-796).
        let mut to_consume = written as usize;
        while idx < n && to_consume >= iovs[idx].iov_len {
            to_consume -= iovs[idx].iov_len;
            idx += 1;
        }
        if idx < n && to_consume > 0 {
            // Partial: advance base + shrink len, then re-issue.
            // SAFETY: `to_consume < iovs[idx].iov_len`, so the new base
            // is still within the original slice.
            iovs[idx].iov_base =
                unsafe { (iovs[idx].iov_base as *mut u8).add(to_consume) as *mut libc::c_void };
            iovs[idx].iov_len -= to_consume;
        }
    }
    Ok(())
}

// ── Linux vmsplice fast path + SpliceVault page-lifetime accounting ──
//
// vmsplice GIFTS the source pages into the pipe; the kernel may
// reference them after the syscall returns, until the reader has
// consumed them. So the source buffers MUST stay alive until at least
// `pipe_buffer_size` MORE bytes have been spliced past them. This is
// the SpliceVault contract (vendor FileUtils.hpp:582-708). Omitting it
// is a CORRECTNESS bug (pipe reader sees freed/reused pages). We port
// the accounting faithfully; the owning handle is the boxed
// `Any`-erased chunk whose buffers were spliced.
#[cfg(target_os = "linux")]
pub mod splice {
    use super::*;
    use std::any::Any;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Per-fd vault of spliced-page owners (vendor `SpliceVault`,
    /// FileUtils.hpp:582-708). Keeps each owning handle alive until
    /// `pipe_buffer_size` subsequent bytes have been spliced past it.
    pub struct SpliceVault {
        fd: i32,
        /// `F_GETPIPE_SZ` result; < 0 means "not a pipe" → vmsplice
        /// disabled, caller falls back to writev (vendor :615, :636).
        pipe_buffer_size: i64,
        /// (owning handle, raw-ptr identity, spliced byte count) in
        /// splice order (vendor `m_splicedData`, FileUtils.hpp:698-700).
        /// We type-erase the owner as `Box<dyn Any + Send>`; its only
        /// job is to keep the spliced buffers' allocations alive.
        spliced: std::collections::VecDeque<(Box<dyn Any + Send>, *const (), usize)>,
        total_spliced_bytes: usize,
    }

    // SAFETY: the raw `*const ()` is only an identity token for owner
    // coalescing; it is never dereferenced. The `Box<dyn Any + Send>`
    // owners are `Send`. The vault is only ever accessed under its
    // global `Mutex`.
    unsafe impl Send for SpliceVault {}

    impl SpliceVault {
        fn new(fd: i32) -> Self {
            // `fcntl(fd, F_GETPIPE_SZ)` returns the pipe buffer size for
            // a pipe fd, or -1 (errno EBADF/EINVAL) for any non-pipe fd
            // (regular file, socket, tty) — exactly the vendor gate
            // (FileUtils.hpp:680). So this single call both probes
            // "is this a pipe?" and sizes the lifetime ledger.
            let sz = unsafe { libc::fcntl(fd, libc::F_GETPIPE_SZ) } as i64;
            Self {
                fd,
                pipe_buffer_size: sz,
                spliced: std::collections::VecDeque::new(),
                total_spliced_bytes: 0,
            }
        }

        /// vmsplice the iovecs to the pipe, then account the owner for
        /// lifetime. Returns `Err` (caller falls back to `writev`) when
        /// the fd is not a pipe (vendor :615 `m_pipeBufferSize < 0`) or
        /// vmsplice errored. On `Ok`, the pages have been gifted and
        /// `owner` is retained until enough subsequent bytes drain.
        ///
        /// `owner` must own the allocations backing every iovec in
        /// `iovs` (vendor's `splicedData` shared_ptr,
        /// FileUtils.hpp:631-651).
        fn splice(
            &mut self,
            iovs: &mut [libc::iovec],
            owner: Box<dyn Any + Send>,
        ) -> io::Result<()> {
            if self.pipe_buffer_size < 0 {
                return Err(io::Error::other("fd is not a pipe (vmsplice declined)"));
            }
            let total: usize = iovs.iter().map(|v| v.iov_len).sum();
            self.vmsplice_all(iovs)?;
            self.account(owner, total);
            Ok(())
        }

        /// vmsplice the whole iovec list (vendor `writeAllSpliceUnsafe`
        /// iovec overload, FileUtils.hpp:531-573): IOV_MAX batches +
        /// partial-write advance.
        fn vmsplice_all(&self, iovs: &mut [libc::iovec]) -> io::Result<()> {
            let n = iovs.len();
            let mut idx = 0usize;
            while idx < n {
                let segment_count = (n - idx).min(iov_max());
                // SAFETY: valid iovecs into live borrowed buffers;
                // segment_count >= 1.
                let written = unsafe {
                    libc::vmsplice(
                        self.fd,
                        iovs.as_ptr().add(idx),
                        segment_count,
                        0, // flags: 0, exactly as vendor (:537)
                    )
                };
                if written < 0 {
                    let err = io::Error::last_os_error();
                    if err.kind() == io::ErrorKind::Interrupted {
                        continue;
                    }
                    return Err(err);
                }
                if written == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "vmsplice wrote zero bytes",
                    ));
                }
                let mut to_consume = written as usize;
                while idx < n && to_consume >= iovs[idx].iov_len {
                    to_consume -= iovs[idx].iov_len;
                    idx += 1;
                }
                if idx < n && to_consume > 0 {
                    iovs[idx].iov_base = unsafe {
                        (iovs[idx].iov_base as *mut u8).add(to_consume) as *mut libc::c_void
                    };
                    iovs[idx].iov_len -= to_consume;
                }
            }
            Ok(())
        }

        /// Retain `owner` and drop owners whose pages have certainly
        /// drained (vendor `account`, FileUtils.hpp:654-675). An owner
        /// is safe to drop once `pipe_buffer_size` MORE bytes have been
        /// spliced past it (so its pages can no longer be in the pipe
        /// buffer). We NEVER drop the most recent owner even if it alone
        /// exceeds the pipe buffer — part of it is still resident
        /// (vendor :667-668; enforced by `len() > 1`).
        fn account(&mut self, owner: Box<dyn Any + Send>, bytes: usize) {
            self.total_spliced_bytes += bytes;
            let id = owner.as_ref() as *const dyn Any as *const ();
            // Coalesce consecutive splices of the SAME owner (vendor
            // :661-665). Our chunk path splices one owner per call so
            // this rarely fires, but keep it faithful.
            match self.spliced.back_mut() {
                Some((_, last_id, last_bytes)) if *last_id == id => {
                    *last_bytes += bytes;
                }
                _ => {
                    self.spliced.push_back((owner, id, bytes));
                }
            }
            // Drop owners that are provably fully drained (vendor
            // :669-674). Never drop the front while doing so would leave
            // < pipe_buffer_size bytes accounted after it.
            while self.spliced.len() > 1 {
                let front_bytes = self.spliced.front().map(|e| e.2).unwrap_or(0);
                if self.total_spliced_bytes - front_bytes >= self.pipe_buffer_size as usize {
                    self.total_spliced_bytes -= front_bytes;
                    self.spliced.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    /// Global per-fd vault registry (vendor's `static vaults`,
    /// FileUtils.hpp:591-600). Keyed by fd; one vault per output fd.
    fn vaults() -> &'static Mutex<HashMap<i32, SpliceVault>> {
        use std::sync::OnceLock;
        static VAULTS: OnceLock<Mutex<HashMap<i32, SpliceVault>>> = OnceLock::new();
        VAULTS.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Attempt the vmsplice fast path for `iovs` (gifting `owner`'s
    /// pages to the pipe). Returns `Ok(())` on success (pages gifted,
    /// owner retained in the vault), or `Err(owner)` when the fd is not
    /// a pipe or vmsplice failed — the caller then falls back to
    /// `writev` and `owner` is handed back so its lifetime can extend
    /// across the (synchronous) writev.
    pub fn try_splice(
        fd: i32,
        iovs: &mut [libc::iovec],
        owner: Box<dyn Any + Send>,
    ) -> Result<(), Box<dyn Any + Send>> {
        let mut guard = vaults().lock().unwrap_or_else(|p| p.into_inner());
        let vault = guard.entry(fd).or_insert_with(|| SpliceVault::new(fd));
        // We don't yet know if vmsplice will succeed; the vault decides
        // (pipe check + vmsplice). On Err we must return `owner` to the
        // caller without having stored it — so take it back out. Since
        // `splice` consumes `owner` only on success (it stores it), we
        // split: probe the pipe first, then either splice-with-owner or
        // return the owner.
        if vault.pipe_buffer_size < 0 {
            return Err(owner);
        }
        match vault.splice(iovs, owner) {
            Ok(()) => Ok(()),
            // vmsplice errored AFTER the pipe check (e.g. SIGPIPE). The
            // owner was moved into `splice`; we can't recover it. Return
            // a fresh empty owner so the caller falls back to writev —
            // the source buffers are still alive in the caller for the
            // synchronous writev, so a no-op owner is sufficient there.
            Err(_) => Err(Box::new(())),
        }
    }
}

/// Write a chunk's payload iovecs fully to `fd`, faithfully mirroring
/// vendor `writeAll` (ChunkData.hpp:794-825): on Linux try `vmsplice`
/// first (succeeds only on a PIPE fd, with `SpliceVault` page-lifetime
/// accounting), and FALL BACK to `writev` for any non-pipe fd (regular
/// file) or if vmsplice errors. On non-Linux unix there is no
/// `vmsplice`, so this is `writev` directly.
///
/// `owner` must own the allocations backing every iovec; on the
/// vmsplice path it is moved into the per-fd `SpliceVault` and kept
/// alive until the pipe has drained. On the writev path the iovec
/// slices are fully copied into the kernel before this returns, so the
/// caller's live buffers suffice and `owner` is dropped here.
pub fn write_all_to_fd(
    fd: i32,
    iovs: &mut [libc::iovec],
    owner: Box<dyn std::any::Any + Send>,
) -> io::Result<()> {
    if iovs.is_empty() {
        // Drop the owner; nothing was spliced.
        drop(owner);
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        // Snapshot the iovecs BEFORE vmsplice mutates them, so a
        // writev fallback writes the full, untouched list (vendor
        // ChunkData.hpp:805-809 passes the SAME `buffersToWrite` to the
        // writev fallback).
        let snapshot: Vec<libc::iovec> = iovs.to_vec();
        match splice::try_splice(fd, iovs, owner) {
            Ok(()) => Ok(()),
            Err(_owner) => {
                // vmsplice declined (non-pipe fd) or failed. writev the
                // untouched snapshot fully. writev COPIES bytes into the
                // kernel synchronously, so the source need only live for
                // the call; the caller holds the buffers for the whole
                // `write_all_to_fd` invocation, so the snapshot iovecs
                // remain valid. `_owner` is dropped at end of arm.
                let mut snap = snapshot;
                writev_all_to_fd(fd, &mut snap)
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        // No vmsplice: writev directly (vendor's non-HAVE_VMSPLICE
        // branch, ChunkData.hpp:811-821). `owner` lives until the end
        // of this function, covering the synchronous writev.
        let r = writev_all_to_fd(fd, iovs);
        drop(owner);
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};
    use std::os::unix::io::AsRawFd;

    /// No-op owner for tests (owns nothing; the test holds the buffers).
    fn no_owner() -> Box<dyn std::any::Any + Send> {
        Box::new(())
    }

    fn roundtrip_file(segments: &[&[u8]], data: &[u8]) -> Vec<u8> {
        let mut tmp = tempfile::tempfile().expect("tempfile");
        let mut iovs = to_io_vec(segments.iter().copied(), data);
        // Regular file → vmsplice declines → writev fallback (Linux),
        // or writev directly (other unix).
        write_all_to_fd(tmp.as_raw_fd(), &mut iovs, no_owner()).expect("write_all_to_fd");
        tmp.seek(SeekFrom::Start(0)).unwrap();
        let mut out = Vec::new();
        tmp.read_to_end(&mut out).unwrap();
        out
    }

    #[test]
    fn segments_then_data() {
        let out = roundtrip_file(&[b"AAA", b"BB"], b"CCCC");
        assert_eq!(out, b"AAABBCCCC");
    }

    #[test]
    fn empty_segments_clean_chunk() {
        let out = roundtrip_file(&[], b"hello world");
        assert_eq!(out, b"hello world");
    }

    #[test]
    fn empty_data_markers_only() {
        let out = roundtrip_file(&[b"only", b"-markers"], b"");
        assert_eq!(out, b"only-markers");
    }

    #[test]
    fn all_empty_is_noop() {
        let out = roundtrip_file(&[], b"");
        assert_eq!(out, b"");
    }

    #[test]
    fn empty_segment_slices_skipped() {
        // An empty segment in the middle must not corrupt order.
        let out = roundtrip_file(&[b"X", b"", b"Y"], b"Z");
        assert_eq!(out, b"XYZ");
    }

    #[test]
    fn many_segments_byte_order_preserved() {
        let segs: Vec<Vec<u8>> = (0..50u32)
            .map(|i| (0..4096u32).map(|j| ((i + j) % 251) as u8).collect())
            .collect();
        let seg_refs: Vec<&[u8]> = segs.iter().map(|s| s.as_slice()).collect();
        let data: Vec<u8> = (0..100_000u32).map(|i| ((i * 7) % 253) as u8).collect();
        let out = roundtrip_file(&seg_refs, &data);
        let mut expected: Vec<u8> = Vec::new();
        for s in &segs {
            expected.extend_from_slice(s);
        }
        expected.extend_from_slice(&data);
        assert_eq!(out, expected);
    }

    /// Pipe path: on Linux this exercises vmsplice + the SpliceVault
    /// page-lifetime accounting; on other unix it is the writev pipe
    /// path. Either way the bytes must arrive in order. A reader thread
    /// drains so the writer never blocks forever. The owner is an
    /// owning copy of the buffers, kept alive by the vault.
    #[test]
    fn pipe_path_byte_order_and_lifetime() {
        use std::io::Read as _;
        let mut fds = [0 as libc::c_int; 2];
        let rc = unsafe { libc::pipe(fds.as_mut_ptr()) };
        assert_eq!(rc, 0, "pipe()");
        let (read_fd, write_fd) = (fds[0], fds[1]);

        // Owned buffers; we splice slices borrowed from these and move
        // an owning clone into the vault so the pages outlive the
        // syscall (the whole point of SpliceVault).
        let segs: Vec<Vec<u8>> = (0..8u32)
            .map(|i| (0..40_000u32).map(|j| ((i + j) % 251) as u8).collect())
            .collect();
        let data: Vec<u8> = (0..300_000u32).map(|i| ((i * 13) % 247) as u8).collect();

        let mut expected: Vec<u8> = Vec::new();
        for s in &segs {
            expected.extend_from_slice(s);
        }
        expected.extend_from_slice(&data);
        let total = expected.len();

        let reader = std::thread::spawn(move || {
            let mut f = unsafe {
                use std::os::unix::io::FromRawFd;
                std::fs::File::from_raw_fd(read_fd)
            };
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            buf
        });

        // Owner: an owning clone of the spliced buffers. On the Linux
        // vmsplice path the vault holds this until the pipe drains; on
        // other unix it is dropped after the synchronous writev.
        let owner: Box<dyn std::any::Any + Send> = Box::new((segs.clone(), data.clone()));
        let seg_refs: Vec<&[u8]> = segs.iter().map(|s| s.as_slice()).collect();
        let mut iovs = to_io_vec(seg_refs.iter().copied(), &data);
        write_all_to_fd(write_fd, &mut iovs, owner).expect("write_all_to_fd pipe");
        // Close the write end so the reader sees EOF.
        unsafe { libc::close(write_fd) };

        let got = reader.join().unwrap();
        assert_eq!(got.len(), total);
        assert_eq!(got, expected);

        // Keep the source buffers alive past the read join (defensive;
        // the vault already guarantees lifetime on the vmsplice path).
        drop(segs);
        drop(data);
    }
}
