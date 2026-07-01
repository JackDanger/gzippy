#![cfg(parallel_sm)]

//! Vendor `std::shared_ptr<ChunkData>` alias semantics for the prefetch cache.
//!
//! rapidgzip's `queueChunkForPostProcessing` lambda captures
//! `std::shared_ptr<ChunkData>` and calls `chunkData->applyWindow` **in place**
//! (GzipChunkFetcher.hpp:579-582) while the prefetch cache holds another ref
//! to the same allocation. gzippy previously cloned [`ChunkData`] on
//! `Arc::try_unwrap` miss and returned the clone through the post-process
//! channel — same bytes after resolve, but extra memcpy and a second heap
//! object vs vendor.
//!
//! ## Safety contract (`unsafe impl Sync`)
//!
//! At most one thread holds `&mut ChunkData` (pool post-process). Concurrent
//! `&ChunkData` reads are allowed only when:
//!
//! 1. **Head chunk:** consumer finished `getLastWindow` publish and queued
//!    `applyWindow`, then blocks on `future.get()` without touching this
//!    chunk again until the future completes (vendor `waitForReplacedMarkers`
//!    :516).
//! 2. **Prefetch cache entry:** no thread reads the entry while pool mutates
//!    it; the consumer scan skips in-flight entries (`m_markersBeingReplaced`
//!    :533) before inspecting marker state.
//! 3. **After `future.get()`:** consumer takes unique ownership via
//!    `Arc::try_unwrap` when the cache entry was removed on `get()`.

use std::cell::UnsafeCell;
use std::ops::Deref;
use std::sync::Arc;

use super::chunk_data::ChunkData;

/// Shared chunk payload — mirror of `std::shared_ptr<ChunkData>`.
pub struct SharedChunkData {
    inner: UnsafeCell<ChunkData>,
}

unsafe impl Sync for SharedChunkData {}

pub type ChunkArc = Arc<SharedChunkData>;

impl SharedChunkData {
    pub fn new(chunk: ChunkData) -> ChunkArc {
        Arc::new(Self {
            inner: UnsafeCell::new(chunk),
        })
    }

    #[inline]
    pub fn get(&self) -> &ChunkData {
        // SAFETY: read-only alias; writer excluded by contract above.
        unsafe { &*self.inner.get() }
    }

    /// Pool-only in-place mutation (`applyWindow` and fused resolve tail).
    pub fn with_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut ChunkData) -> R,
    {
        // SAFETY: exclusive writer on pool thread; no concurrent reader on this arc.
        let chunk = unsafe { &mut *self.inner.get() };
        f(chunk)
    }

    /// Take owned [`ChunkData`] when this is the last `Arc` ref.
    pub fn into_inner(this: Self) -> ChunkData {
        // SAFETY: unique ownership — no other `Arc` refs exist.
        this.inner.into_inner()
    }

    /// `Arc::try_unwrap` when unique, else deep-clone (cache still holds a ref).
    pub fn take_or_clone(arc: ChunkArc) -> ChunkData {
        match Arc::try_unwrap(arc) {
            Ok(this) => Self::into_inner(this),
            Err(arc) => arc.get().clone(),
        }
    }
}

impl Deref for SharedChunkData {
    type Target = ChunkData;

    fn deref(&self) -> &ChunkData {
        self.get()
    }
}
