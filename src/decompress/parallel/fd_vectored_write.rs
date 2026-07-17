//! Zero-copy vectored output for the parallel-SM consumer — the
//! data-plane `writeAll` tail.
//!
//! ## Why this exists
//!
//! The in-order consumer (`chunk_fetcher::drain_one_pending`) is the
//! single point that materializes every decoded byte to the output
//! sink. Before this module it did so via `BufWriter::write_all` —
//! a single-core `memcpy` of all freshly cross-core-decoded payload
//! through a 1 MiB `BufWriter` staging buffer.
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
    // No per-call byte cap.
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
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    /// Number of chunk payloads that engaged the vmsplice fast path
    /// (i.e. the fd was a pipe AND vmsplice fully succeeded). The byte-
    /// identity GATE asserts this is `> 0` on a pipe sink and `0` on a
    /// file sink. Read with [`vmsplice_engagements`].
    pub(crate) static VMSPLICE_ENGAGEMENTS: AtomicU64 = AtomicU64::new(0);

    /// Read the vmsplice-engagement counter.
    #[allow(dead_code)] // diagnostic accessor; consumed by the byte-identity gate test only
    pub fn vmsplice_engagements() -> u64 {
        VMSPLICE_ENGAGEMENTS.load(Ordering::Relaxed)
    }

    /// Outcome of a vmsplice attempt for one chunk payload.
    pub enum SpliceOutcome {
        /// All segments gifted; owner retained in the vault.
        Spliced,
        /// fd is not a pipe, OR vmsplice failed BEFORE any byte was
        /// gifted (vendor `i == 0` → `return errno`). The intact owner
        /// is handed back so the caller can `writev`-fall-back safely.
        Declined(Box<dyn Any + Send>),
        /// vmsplice failed AFTER gifting some segments (vendor `i > 0`
        /// → throw). FATAL: the caller must propagate this error and
        /// must NOT writev-fall-back (would duplicate the gifted
        /// prefix). The owner has already been retained in the vault.
        Fatal(io::Error),
    }

    /// A vmsplice failure plus how many bytes were already gifted to the
    /// pipe before it occurred.
    struct VmspliceFail {
        err: io::Error,
        bytes_gifted: usize,
    }

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
        ) -> SpliceOutcome {
            if self.pipe_buffer_size < 0 {
                return SpliceOutcome::Declined(owner);
            }
            let total: usize = iovs.iter().map(|v| v.iov_len).sum();
            match self.vmsplice_all(iovs) {
                Ok(()) => {
                    self.account(owner, total);
                    SpliceOutcome::Spliced
                }
                // Vendor `writeAllSpliceUnsafe` iovec overload
                // (FileUtils.hpp:535-547): on a vmsplice error with
                // NOTHING yet gifted (i == 0) it `return errno` →
                // the caller (`splice`/`writeAll`) takes the writev
                // fallback. The real `owner` is still fully intact
                // (no pages handed to the kernel), so hand it back.
                Err(VmspliceFail {
                    err: _,
                    bytes_gifted: 0,
                }) => SpliceOutcome::Declined(owner),
                // i > 0: some segments were ALREADY gifted to the pipe
                // before the failure. Vendor THROWS here (a hard fatal,
                // never a writev fallback — re-writing the full list
                // would DUPLICATE the already-gifted prefix). We mirror
                // the throw with a fatal `Err`, and — crucially — we
                // RETAIN the owner in the vault so its already-gifted
                // pages stay alive while the kernel may still reference
                // them (the use-after-free guard). We account the FULL
                // `total` so the owner is held for the maximal window.
                Err(VmspliceFail { err, .. }) => {
                    self.account(owner, total);
                    SpliceOutcome::Fatal(err)
                }
            }
        }

        /// vmsplice the whole iovec list (vendor `writeAllSpliceUnsafe`
        /// iovec overload, FileUtils.hpp:531-573): IOV_MAX batches +
        /// partial-write advance.
        ///
        /// On error, reports how many bytes were SUCCESSFULLY gifted to
        /// the pipe before the failure (`bytes_gifted`), so the caller
        /// can distinguish a clean pre-splice failure (== 0 → safe
        /// writev fallback) from a mid-list failure (> 0 → fatal, the
        /// already-gifted pages must NOT be re-written and must stay
        /// alive). This is the data the vendor encodes implicitly via
        /// the `i == 0` test (FileUtils.hpp:540).
        fn vmsplice_all(&self, iovs: &mut [libc::iovec]) -> Result<(), VmspliceFail> {
            let n = iovs.len();
            let mut idx = 0usize;
            let mut bytes_gifted = 0usize;
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
                    return Err(VmspliceFail { err, bytes_gifted });
                }
                if written == 0 {
                    return Err(VmspliceFail {
                        err: io::Error::new(io::ErrorKind::WriteZero, "vmsplice wrote zero bytes"),
                        bytes_gifted,
                    });
                }
                bytes_gifted += written as usize;
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
    /// pages to the pipe). The three outcomes mirror the vendor split
    /// (FileUtils.hpp:535-547 + ChunkData.hpp:806-808):
    ///   * [`SpliceOutcome::Spliced`] — all pages gifted, owner retained
    ///     in the vault until the pipe drains. Increments the engagement
    ///     counter.
    ///   * [`SpliceOutcome::Declined`] — not a pipe, or vmsplice failed
    ///     before any byte was gifted (`i == 0`). Owner handed back
    ///     INTACT; caller does the safe writev fallback.
    ///   * [`SpliceOutcome::Fatal`] — vmsplice failed mid-list (`i > 0`),
    ///     some pages already gifted. Caller MUST propagate (no writev
    ///     fallback). Owner already retained in the vault so the gifted
    ///     pages stay alive.
    pub fn try_splice(
        fd: i32,
        iovs: &mut [libc::iovec],
        owner: Box<dyn Any + Send>,
    ) -> SpliceOutcome {
        let mut guard = vaults().lock().unwrap_or_else(|p| p.into_inner());
        let vault = guard.entry(fd).or_insert_with(|| SpliceVault::new(fd));
        let outcome = vault.splice(iovs, owner);
        if matches!(outcome, SpliceOutcome::Spliced) {
            VMSPLICE_ENGAGEMENTS.fetch_add(1, Ordering::Relaxed);
        }
        outcome
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
#[cfg(any(test, target_os = "linux"))]
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
        use splice::SpliceOutcome;
        // Snapshot the iovecs BEFORE vmsplice mutates them, so a
        // writev fallback writes the full, untouched list (vendor
        // ChunkData.hpp:805-809 passes the SAME `buffersToWrite` to the
        // writev fallback). The snapshot is ONLY ever used on the
        // `Declined` path, where NO byte was gifted — so writing the
        // full list cannot duplicate anything.
        let snapshot: Vec<libc::iovec> = iovs.to_vec();
        match splice::try_splice(fd, iovs, owner) {
            SpliceOutcome::Spliced => Ok(()),
            SpliceOutcome::Declined(_owner) => {
                // Not a pipe, or vmsplice failed before gifting any byte.
                // The owner is intact and the caller holds the source
                // buffers for the whole `write_all_to_fd` invocation, so
                // the snapshot iovecs are valid. writev COPIES bytes into
                // the kernel synchronously. `_owner` drops at arm end.
                let mut snap = snapshot;
                writev_all_to_fd(fd, &mut snap)
            }
            SpliceOutcome::Fatal(err) => {
                // vmsplice gifted some segments then failed (vendor
                // `i > 0` → throw). Propagate hard — NO writev fallback
                // (would duplicate the already-gifted prefix). The owner
                // is retained in the vault so the gifted pages stay alive
                // (no use-after-free).
                Err(err)
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

/// True when `fd` is a pipe (`F_GETPIPE_SZ` succeeds on Linux).
#[cfg(target_os = "linux")]
#[inline]
pub fn is_pipe_fd(fd: i32) -> bool {
    #[cfg(target_os = "linux")]
    {
        (unsafe { libc::fcntl(fd, libc::F_GETPIPE_SZ) }) >= 0
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = fd;
        false
    }
}

/// Consumer zero-copy output: gather-`writev` (or Linux-pipe `vmsplice`)
/// then recycle `chunk`'s rpmalloc-backed buffers.
///
/// Regular files and non-Linux unix use synchronous `writev` — the kernel
/// copies bytes before return, so `chunk` stays on the stack with no
/// per-chunk `Box<dyn Any>` owner. Linux pipes box `chunk` for
/// `SpliceVault` page-lifetime accounting (vendor `writeAll` pipe arm).
// Reached only from the Linux-only pipe-fd branch in `chunk_fetcher`; on
// non-Linux unix it compiles but is unused.
#[allow(dead_code)]
pub fn write_chunk_payload_to_fd(
    fd: i32,
    iovs: &mut [libc::iovec],
    mut chunk: crate::decompress::parallel::chunk_data::ChunkData,
) -> io::Result<()> {
    if iovs.is_empty() {
        chunk.recycle_decoded_buffers();
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    if is_pipe_fd(fd) {
        write_all_to_fd(fd, iovs, Box::new(chunk))?;
        return Ok(());
    }
    let r = writev_all_to_fd(fd, iovs);
    chunk.recycle_decoded_buffers();
    r
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
    ///
    /// IGNORE'd under the default parallel `cargo test`: this writes ~620 KiB
    /// to a kernel pipe while a reader thread drains it. Under a saturated
    /// runner (the bin test binary co-runs subprocess-spawning CLI tests on a
    /// 2-core CI host) the reader thread can be starved of a core, the
    /// Linux vmsplice writer then blocks on the full pipe, and the whole test
    /// binary wedges for the 6 h job timeout. Run it serially with
    /// `--ignored --test-threads=1`, where the reader always gets scheduled.
    #[ignore = "blocking pipe write deadlocks if the reader thread is starved under parallel cargo test; run serially with --ignored --test-threads=1"]
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

    /// EARLY-READER-DEATH.
    ///
    /// On Linux: open a pipe, shrink its buffer, splice ONE payload that
    /// fits (so the vault retains an owner), then CLOSE the read end with
    /// the pipe non-empty and no reader. A second large `write_all_to_fd`
    /// then hits EPIPE on vmsplice. The fixed code must (a) return a
    /// clean `Err` (no panic / no segfault / no double-drop UAF) and
    /// (b) NOT silently duplicate already-written bytes. We ignore
    /// SIGPIPE so the syscall returns EPIPE rather than killing us.
    ///
    /// NOTE: the read end is closed before the failing write, but Phase 1
    /// still issues a *blocking* vmsplice into a 4 KiB pipe with no reader;
    /// under a core-starved parallel run that splice can block long enough to
    /// wedge the whole bin test binary (observed: 6 h job timeout). It also
    /// flips SIGPIPE to SIG_IGN process-wide, polluting sibling tests. Run it
    /// serially with `--ignored --test-threads=1`.
    #[cfg(target_os = "linux")]
    #[ignore = "blocking vmsplice into a tiny pipe + process-global SIGPIPE=SIG_IGN; unsafe under parallel cargo test, run serially with --ignored --test-threads=1"]
    #[test]
    fn early_reader_death_is_clean_error_no_duplicate() {
        unsafe {
            libc::signal(libc::SIGPIPE, libc::SIG_IGN);
        }
        let mut fds = [0 as libc::c_int; 2];
        let rc = unsafe { libc::pipe(fds.as_mut_ptr()) };
        assert_eq!(rc, 0, "pipe()");
        let (read_fd, write_fd) = (fds[0], fds[1]);

        // Shrink the pipe buffer so a small splice fills it.
        unsafe {
            libc::fcntl(write_fd, libc::F_SETPIPE_SZ, 4096);
        }
        let pipe_sz = unsafe { libc::fcntl(write_fd, libc::F_GETPIPE_SZ) };
        assert!(pipe_sz > 0, "F_GETPIPE_SZ");

        // Phase 1: splice a payload that fits in the (now small) pipe
        // buffer. This succeeds (vmsplice engages) and leaves an owner
        // in the vault — so the later failure path must not corrupt it.
        let first: Vec<u8> = (0..(pipe_sz as usize / 2))
            .map(|i| (i % 251) as u8)
            .collect();
        let owner1: Box<dyn std::any::Any + Send> = Box::new(first.clone());
        let mut iov1 = to_io_vec(std::iter::empty::<&[u8]>(), &first);
        write_all_to_fd(write_fd, &mut iov1, owner1).expect("phase-1 splice fits");

        // Kill the reader: close the read end while the pipe holds data.
        unsafe { libc::close(read_fd) };

        // Phase 2: a large payload. The pipe is non-empty and now has no
        // reader → vmsplice returns EPIPE. Must be a clean Err, never a
        // panic/segfault, never a duplicating writev fallback of an
        // already-gifted prefix.
        let big: Vec<u8> = (0..(pipe_sz as usize * 8))
            .map(|i| ((i * 13) % 247) as u8)
            .collect();
        let owner2: Box<dyn std::any::Any + Send> = Box::new(big.clone());
        let mut iov2 = to_io_vec(std::iter::empty::<&[u8]>(), &big);
        let r = write_all_to_fd(write_fd, &mut iov2, owner2);
        assert!(
            r.is_err(),
            "writing to a pipe with a dead reader must fail cleanly"
        );

        unsafe { libc::close(write_fd) };
        drop(first);
        drop(big);
    }

    /// The engagement counter must advance on a pipe sink and stay put
    /// on a file sink (used by the byte-identity GATE).
    #[cfg(target_os = "linux")]
    #[ignore = "shared process-global counter races under parallel cargo test; run serially with --ignored --test-threads=1"]
    #[test]
    fn vmsplice_counter_pipe_vs_file() {
        use std::io::Read as _;
        // File sink: must NOT engage vmsplice.
        let before_file = splice::vmsplice_engagements();
        let _ = roundtrip_file(&[b"abc"], b"def");
        assert_eq!(
            splice::vmsplice_engagements(),
            before_file,
            "file sink must not engage vmsplice"
        );

        // Pipe sink: must engage vmsplice at least once.
        let before_pipe = splice::vmsplice_engagements();
        let mut fds = [0 as libc::c_int; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        let (read_fd, write_fd) = (fds[0], fds[1]);
        let data: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let dclone = data.clone();
        let reader = std::thread::spawn(move || {
            let mut f = unsafe {
                use std::os::unix::io::FromRawFd;
                std::fs::File::from_raw_fd(read_fd)
            };
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            buf
        });
        let owner: Box<dyn std::any::Any + Send> = Box::new(data.clone());
        let mut iovs = to_io_vec(std::iter::empty::<&[u8]>(), &data);
        write_all_to_fd(write_fd, &mut iovs, owner).expect("pipe write");
        unsafe { libc::close(write_fd) };
        let got = reader.join().unwrap();
        assert_eq!(got, dclone);
        assert!(
            splice::vmsplice_engagements() > before_pipe,
            "pipe sink must engage vmsplice"
        );
    }
}
