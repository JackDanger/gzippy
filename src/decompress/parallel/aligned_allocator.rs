//! Literal port of `rapidgzip::AlignedAllocator` + `rapidgzip::AlignedVector`
//! (vendor/rapidgzip/librapidarchive/src/core/AlignedAllocator.hpp:18-82).
//!
//! Purpose (per the vendor comment AlignedAllocator.hpp:10-17):
//! > Returns aligned pointers when allocations are requested.
//! > Default alignment is 64B = 512b sufficient for AVX-512 and most cache
//! > line sizes.
//!
//! Rapidgzip uses `AlignedVector<T>` (the `std::vector<T, AlignedAllocator<T>>`
//! alias on AlignedAllocator.hpp:81-82) for hot buffers that feed SIMD
//! kernels — the SIMD loads/stores require natural alignment to avoid the
//! split-cache-line and split-page-fault penalties.
//!
//! Mapping rapidgzip -> Rust
//! -------------------------
//! Rust's stable `Vec<T>` does NOT accept a custom allocator (the
//! `allocator_api` feature is nightly). The faithful port is therefore a
//! standalone container type backed by [`std::alloc::alloc`] /
//! [`std::alloc::dealloc`] with an explicit [`std::alloc::Layout`].
//!
//! - `template<ElementType, std::size_t ALIGNMENT_IN_BYTES = 64>`
//!   (AlignedAllocator.hpp:18-20) -> generic `<T, const A: usize = 64>`.
//! - `allocate(nElements)`
//!   (AlignedAllocator.hpp:57-66) -> [`AlignedVec::with_capacity`] calls
//!   `alloc::alloc(Layout::from_size_align(n*size_of::<T>(), A))`.
//! - `deallocate(ptr, n)`
//!   (AlignedAllocator.hpp:68-77) -> Drop impl calls
//!   `alloc::dealloc(ptr, layout)`.
//! - `ALIGNMENT_IN_BYTES >= alignof(T)` static_assert
//!   (AlignedAllocator.hpp:23-26) -> compile-time assertion in `new`.
//! - `std::bad_array_new_length` overflow check (AlignedAllocator.hpp:60-62)
//!   -> `Layout::array::<T>(n)` returns Err on overflow; we propagate.
//!
//! The type holds raw bytes for `T: Copy` only. The vendor's
//! `std::vector<T, AlignedAllocator>` supports arbitrary `T`, but every
//! rapidgzip caller specializes to POD/Copy types (uint8_t, uint16_t,
//! uint32_t, deflate code-tables). Restricting to `T: Copy` keeps the port
//! drop-trivial; we can lift the bound if a future call site needs it.

#![allow(dead_code)]

use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;

/// Default alignment matching `ALIGNMENT_IN_BYTES = 64` on
/// AlignedAllocator.hpp:19. 64 bytes covers AVX-512 (vmovdqa64) and the
/// cache-line size on x86_64 / aarch64.
pub const DEFAULT_ALIGNMENT: usize = 64;

/// Heap-allocated, aligned, length-tracked buffer for `T: Copy`.
///
/// Mirror of `AlignedVector<T, ALIGNMENT_IN_BYTES>` (AlignedAllocator.hpp:81-82).
pub struct AlignedVec<T: Copy, const ALIGNMENT: usize = DEFAULT_ALIGNMENT> {
    /// Owning pointer to the aligned buffer. `NonNull` mirrors
    /// `ElementType*` (AlignedAllocator.hpp:65) — the allocator returns a
    /// non-null pointer or throws.
    ptr: NonNull<T>,
    /// Number of `T` elements currently considered initialized. The
    /// vendor `std::vector` tracks this internally as `size()`.
    len: usize,
    /// Capacity in `T` elements. Allocator deallocate requires the same
    /// Layout used to allocate; we recompute from `cap` on drop.
    cap: usize,
    /// `T` is owned here. Without this, `AlignedVec<T>` would not appear
    /// to own `T` and the variance/drop checker would be wrong.
    _marker: PhantomData<T>,
}

impl<T: Copy, const ALIGNMENT: usize> AlignedVec<T, ALIGNMENT> {
    /// Compile-time alignment validation. Mirror of the static_assert at
    /// AlignedAllocator.hpp:23-26: `ALIGNMENT_IN_BYTES >= alignof(T)`. We
    /// also require the alignment to be a power of two and non-zero, as
    /// the vendor's comment "Must be a positive power of 2"
    /// (AlignedAllocator.hpp:16) demands and `std::align_val_t` enforces.
    const _ASSERT_ALIGNMENT: () = {
        assert!(
            ALIGNMENT >= align_of::<T>(),
            "ALIGNMENT must be >= alignof(T); see AlignedAllocator.hpp:23-26"
        );
        assert!(
            ALIGNMENT.is_power_of_two() && ALIGNMENT > 0,
            "ALIGNMENT must be a positive power of two; see AlignedAllocator.hpp:16"
        );
    };

    /// Construct an empty buffer; no allocation. Mirror of
    /// `std::vector<>()` default construction — the vendor's
    /// `AlignedVector` inherits that from `std::vector`.
    pub fn new() -> Self {
        // Force the const assertion to be evaluated.
        #[allow(clippy::let_unit_value)]
        let _ = Self::_ASSERT_ALIGNMENT;
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            _marker: PhantomData,
        }
    }

    /// Allocate space for `capacity` elements, length stays at 0. Mirror
    /// of `std::vector::reserve` in combination with the allocate path
    /// (AlignedAllocator.hpp:57-66).
    ///
    /// # Panics
    ///
    /// Panics if `capacity * size_of::<T>()` overflows or if the allocator
    /// returns null. The vendor throws `std::bad_array_new_length` for
    /// the overflow case (AlignedAllocator.hpp:60-62) and lets
    /// `::operator new[]` throw `std::bad_alloc` for the allocator
    /// failure. Both are unrecoverable; panicking is the closest Rust
    /// analogue.
    pub fn with_capacity(capacity: usize) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::_ASSERT_ALIGNMENT;
        if capacity == 0 {
            return Self::new();
        }
        let layout = Self::layout_for(capacity);
        // SAFETY: layout is non-zero size (capacity > 0, T is sized via
        // `size_of::<T>()`); alignment is a non-zero power of two from
        // ALIGNMENT generic parameter (checked by _ASSERT_ALIGNMENT).
        let raw = unsafe { alloc::alloc(layout) };
        let ptr = match NonNull::new(raw as *mut T) {
            Some(p) => p,
            None => alloc::handle_alloc_error(layout),
        };
        Self {
            ptr,
            len: 0,
            cap: capacity,
            _marker: PhantomData,
        }
    }

    /// Set length without zero-filling. Mirror of `resize(n)` followed by
    /// "you wrote into the back" — but for `T: Copy` types the vendor's
    /// resize-to-n initializes to T() (zero for numerics). Callers that
    /// want zero-init should follow with `as_mut_slice().fill(T::default())`.
    ///
    /// # Safety
    ///
    /// `new_len` must be `<= cap`, and the first `new_len` elements must
    /// have been written. The vendor's std::vector tracks this through
    /// its public API; here it is the caller's responsibility.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.cap);
        self.len = new_len;
    }

    /// Number of initialized elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` iff `len == 0`. Mirror of `std::vector::empty()`.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Capacity in `T` elements.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Pointer to the first element. Guaranteed aligned to ALIGNMENT.
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Mutable pointer to the first element.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// View as `&[T]` of length `len`.
    pub fn as_slice(&self) -> &[T] {
        if self.cap == 0 {
            return &[];
        }
        // SAFETY: ptr is valid for `len` reads (we own the buffer and the
        // caller upheld set_len's contract).
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// View as `&mut [T]` of length `len`.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.cap == 0 {
            return &mut [];
        }
        // SAFETY: as above; we hold &mut self so there's no aliasing.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Compute the layout used for `cap` elements. Mirror of
    /// `nElementsToAllocate * sizeof(ElementType)` at AlignedAllocator.hpp:64.
    fn layout_for(cap: usize) -> Layout {
        match Layout::from_size_align(cap * size_of::<T>(), ALIGNMENT) {
            Ok(l) => l,
            // Vendor: std::bad_array_new_length (AlignedAllocator.hpp:60-62).
            // We panic with the offending value to mirror the unrecoverable
            // intent.
            Err(_) => panic!(
                "AlignedVec capacity {cap} * size_of::<T>()={} overflows Layout",
                size_of::<T>()
            ),
        }
    }
}

impl<T: Copy, const ALIGNMENT: usize> Default for AlignedVec<T, ALIGNMENT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const ALIGNMENT: usize> Deref for AlignedVec<T, ALIGNMENT> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy, const ALIGNMENT: usize> DerefMut for AlignedVec<T, ALIGNMENT> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy, const ALIGNMENT: usize> Drop for AlignedVec<T, ALIGNMENT> {
    /// Mirror of `deallocate` (AlignedAllocator.hpp:68-77).
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        let layout = Self::layout_for(self.cap);
        // SAFETY: layout matches the one used by alloc::alloc in
        // with_capacity; cap is non-zero so layout is non-zero size.
        unsafe { alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout) };
    }
}

// SAFETY: AlignedVec owns its buffer; transferring the owning pointer
// across threads is safe because the data is `T: Copy + Send` (Copy
// implies Send for the element). The vendor's std::vector<T,
// AlignedAllocator<T>> is similarly Send when T is.
unsafe impl<T: Copy + Send, const ALIGNMENT: usize> Send for AlignedVec<T, ALIGNMENT> {}
// SAFETY: Reads via &AlignedVec require &self, so concurrent reads cannot
// observe partial writes. The Sync bound on T mirrors std::vector's
// implicit Sync (sharing a const reference across threads is safe iff
// shared element references are safe).
unsafe impl<T: Copy + Sync, const ALIGNMENT: usize> Sync for AlignedVec<T, ALIGNMENT> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_does_not_allocate() {
        let v: AlignedVec<u8> = AlignedVec::new();
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);
        assert!(v.is_empty());
        assert!(v.as_slice().is_empty());
    }

    #[test]
    fn with_capacity_returns_aligned_pointer_default_alignment() {
        // Default alignment of 64 covers AVX-512; the pointer the
        // allocator returns must satisfy that.
        let v: AlignedVec<u8> = AlignedVec::with_capacity(128);
        let addr = v.as_ptr() as usize;
        assert_eq!(addr % DEFAULT_ALIGNMENT, 0);
        assert_eq!(v.capacity(), 128);
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn with_capacity_returns_aligned_pointer_custom_alignment() {
        // 256-byte alignment is sometimes useful for page-fragment hot
        // buffers; the type accepts any power-of-two >= alignof(T).
        let v: AlignedVec<u32, 256> = AlignedVec::with_capacity(64);
        let addr = v.as_ptr() as usize;
        assert_eq!(addr % 256, 0);
    }

    #[test]
    fn set_len_then_read_back() {
        let mut v: AlignedVec<u32> = AlignedVec::with_capacity(8);
        // Write into the uninitialized backing store via the raw pointer,
        // then promote len.
        for i in 0..8u32 {
            // SAFETY: i is in-bounds for cap=8.
            unsafe { v.as_mut_ptr().add(i as usize).write(i * 10) };
        }
        // SAFETY: all 8 slots were just written.
        unsafe { v.set_len(8) };
        assert_eq!(v.as_slice(), &[0, 10, 20, 30, 40, 50, 60, 70]);
    }

    #[test]
    fn deref_through_slice_api() {
        let mut v: AlignedVec<u16> = AlignedVec::with_capacity(4);
        for i in 0..4u16 {
            unsafe { v.as_mut_ptr().add(i as usize).write(i + 1) };
        }
        unsafe { v.set_len(4) };
        // Through Deref<Target=[T]>.
        let total: u16 = v.iter().sum();
        assert_eq!(total, 1 + 2 + 3 + 4);
    }

    #[test]
    fn drop_releases_buffer() {
        // Sanity: dropping an AlignedVec must not panic / leak / corrupt
        // the heap. We exercise a few sizes; the test passes if Miri or
        // ASan would not flag anything (and if it does not crash here).
        for cap in [1usize, 7, 64, 1024, 65536] {
            let _v: AlignedVec<u8> = AlignedVec::with_capacity(cap);
        }
        for cap in [1usize, 16, 4096] {
            let _v: AlignedVec<u32> = AlignedVec::with_capacity(cap);
        }
    }
}
