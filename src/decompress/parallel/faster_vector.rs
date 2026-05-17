//! Literal port of `rapidgzip::FasterVector`
//! (vendor/rapidgzip/librapidarchive/src/core/FasterVector.hpp:120-128).
//!
//! Despite the header file weighing in at 454 lines, the **active** portion
//! is just one of two `using` aliases (FasterVector.hpp:120-128):
//!
//! ```c++
//! #if 1
//! #ifdef LIBRAPIDARCHIVE_WITH_RPMALLOC
//! template<typename T>
//! using FasterVector = std::vector<T, RpmallocAllocator<T> >;
//! #else
//! template<typename T>
//! using FasterVector = std::vector<T>;
//! #endif
//! #else
//! // ... 320 lines of an experimental no-init-on-resize vector ...
//! #endif
//! ```
//!
//! The `#else` arm is dead code, kept in the vendor for archaeology. The
//! comment block at FasterVector.hpp:146-158 explains why the experimental
//! variant was reverted:
//!
//! > This was supposed to be a faster std::vector alternative that saves
//! > time by not initializing its contents on resize ... However, it leads
//! > to almost double the memory usage with wikidata.json (12 GB -> 16 GB)
//! > even when disabling rounding to powers of 2 when reserving in
//! > @ref insert and when disabling alignment and disabling the
//! > shrink_to_fit check.
//!
//! So the faithful port is the **active** typedef, not the 320-line dead
//! class. Mirror it as a Rust type alias.
//!
//! Mapping rapidgzip -> Rust
//! -------------------------
//! - `std::vector<T, RpmallocAllocator<T>>` (FasterVector.hpp:124) when
//!   `LIBRAPIDARCHIVE_WITH_RPMALLOC` is defined ->
//!   [`Vec<T>`]. Gzippy does NOT vendor rpmalloc (see Cargo.toml — the
//!   system allocator is in play), which makes this branch equivalent to
//!   the system-allocator branch in our build.
//! - `std::vector<T>` (FasterVector.hpp:127) -> [`Vec<T>`]. Same.
//!
//! Either way, the active vendor type is `std::vector<T>` against the
//! prevailing allocator, and the Rust analogue is `Vec<T>` against the
//! prevailing global allocator. Calling sites do not need a distinct
//! `FasterVector` type in Rust — `Vec<T>` already is "the vector the build
//! uses" — but exposing the alias here keeps the structural correspondence
//! to the vendor obvious and gives a single place to swap in a non-default
//! allocator later (e.g. mimalloc or jemalloc, the closest Rust analogues
//! of rpmalloc) without touching call sites.
//!
//! Note on the experimental class
//! ------------------------------
//! The 320-line `#if 0`-branch class (FasterVector.hpp:139-453) was
//! considered: it manages a raw `T*` + `size` + `capacity`, supports
//! POD-only operations, uses [`std::memmove`] / [`std::memcpy`], and
//! deliberately skips the value-initialization that `std::vector::resize`
//! performs. The vendor has already determined empirically that this hurts
//! more than it helps. Porting it would copy a known-worse implementation
//! into gzippy. The "PORT, DON'T INNOVATE" rule is satisfied by the
//! active code: we port the active typedef. If a future advisor wants the
//! dead variant for completeness, it can be added as
//! [`ExperimentalFasterVector`] in a follow-up commit and gated behind
//! a feature flag the same way the `#if 0` gates the C++ variant.

#![allow(dead_code)]

/// Mirror of the active vendor typedef
/// `using FasterVector = std::vector<T>` (FasterVector.hpp:127), or
/// `std::vector<T, RpmallocAllocator<T>>` (FasterVector.hpp:124) under the
/// rpmalloc build, both of which resolve to the same shape in our
/// allocator world.
///
/// This is intentionally a type alias rather than a newtype. The vendor
/// type alias is transparent to callers — every `FasterVector<T>` operator
/// is just a forwarded `std::vector<T>` method — and gzippy callers should
/// be able to drop in a `Vec<T>` anywhere a port writes `FasterVector<T>`.
pub type FasterVector<T> = Vec<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_behaves_as_vec_pushing_growing() {
        // FasterVector is std::vector at heart; the rapidgzip call sites
        // use push_back, size(), data(), operator[]. The Rust equivalents
        // are push, len, as_ptr/as_mut_ptr, indexing. All work because
        // the alias IS Vec<T>.
        let mut v: FasterVector<u32> = FasterVector::with_capacity(0);
        for value in [10u32, 20, 30] {
            v.push(value);
        }
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 10);
        assert_eq!(v[2], 30);
    }

    #[test]
    fn alias_supports_with_capacity_and_resize() {
        // Vendor `FasterVector(size_t size, optional<T>)` -> resize.
        // Mirror with Vec::with_capacity + resize.
        let mut v: FasterVector<u8> = FasterVector::with_capacity(64);
        assert_eq!(v.len(), 0);
        assert!(v.capacity() >= 64);

        v.resize(32, 0xAB);
        assert_eq!(v.len(), 32);
        assert!(v.iter().all(|b| *b == 0xAB));
    }

    #[test]
    fn alias_supports_iter_via_data_pointer() {
        // The vendor exposes data()/begin()/end(); Rust's Vec gives
        // as_ptr/iter equivalents. Verify iteration semantics line up.
        let v: FasterVector<u16> = (1u16..=5).collect();
        let sum: u16 = v.iter().sum();
        assert_eq!(sum, 15);
        assert_eq!(unsafe { *v.as_ptr().add(0) }, 1);
        assert_eq!(unsafe { *v.as_ptr().add(4) }, 5);
    }

    #[test]
    fn move_semantics_are_default_for_vec() {
        // Vendor `FasterVector` deletes copy and defaults move
        // (FasterVector.hpp:212-213). Vec is non-Copy / non-Clone by
        // default (well, it implements Clone, but moves are the natural
        // ownership transfer).
        let mut a: FasterVector<i64> = vec![1, 2, 3];
        let b = std::mem::take(&mut a);
        assert!(a.is_empty());
        assert_eq!(b, vec![1, 2, 3]);
    }
}
