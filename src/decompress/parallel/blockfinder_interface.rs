//! Literal port of `rapidgzip::blockfinder::Interface`
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/blockfinder/Interface.hpp:1-17).
//!
//! The C++ class is a pure abstract base with:
//!
//! ```cpp
//! namespace rapidgzip::blockfinder
//! {
//! class Interface
//! {
//! public:
//!     virtual ~Interface() = default;
//!     [[nodiscard]] virtual size_t find() = 0;
//! };
//! }
//! ```
//!
//! In Rust this maps to a small trait with a single `find()` method
//! returning the bit offset (or `NO_MORE_BLOCKS`, the sentinel mirror
//! of C++'s `std::numeric_limits<std::size_t>::max()`).

#![allow(dead_code)]

/// Sentinel value returned by [`BlockFinderInterface::find`] when no
/// further block boundary can be produced — mirror of C++'s
/// `std::numeric_limits<std::size_t>::max()` (used e.g. in
/// `PigzStringView.hpp:63-65` and `Bgzf.hpp:200-202`).
pub const NO_MORE_BLOCKS: usize = usize::MAX;

/// Faithful port of `rapidgzip::blockfinder::Interface`
/// (Interface.hpp:8-16). Implementors return the next deflate block
/// offset **in bits** from the start of the file, or [`NO_MORE_BLOCKS`]
/// when exhausted.
///
/// The trait is object-safe so that the future generic block-fetcher
/// can hold a `Box<dyn BlockFinderInterface + Send>`, just as
/// rapidgzip stores a `std::unique_ptr<blockfinder::Interface>`.
pub trait BlockFinderInterface {
    /// Mirror of `[[nodiscard]] virtual size_t find() = 0`
    /// (Interface.hpp:14-15). Bit offset of the next deflate block, or
    /// [`NO_MORE_BLOCKS`] on exhaustion.
    fn find(&mut self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A finder that yields a fixed sequence of bit offsets — proves
    /// the trait is object-safe (i.e. usable behind `dyn`).
    struct Fixed {
        offsets: std::vec::IntoIter<usize>,
    }

    impl BlockFinderInterface for Fixed {
        fn find(&mut self) -> usize {
            self.offsets.next().unwrap_or(NO_MORE_BLOCKS)
        }
    }

    #[test]
    fn yields_offsets_then_sentinel() {
        let mut f = Fixed {
            offsets: vec![80, 160, 240].into_iter(),
        };
        assert_eq!(f.find(), 80);
        assert_eq!(f.find(), 160);
        assert_eq!(f.find(), 240);
        assert_eq!(f.find(), NO_MORE_BLOCKS);
        assert_eq!(f.find(), NO_MORE_BLOCKS);
    }

    #[test]
    fn object_safe_behind_dyn() {
        let mut boxed: Box<dyn BlockFinderInterface> = Box::new(Fixed {
            offsets: vec![1, 2].into_iter(),
        });
        assert_eq!(boxed.find(), 1);
        assert_eq!(boxed.find(), 2);
        assert_eq!(boxed.find(), NO_MORE_BLOCKS);
    }
}
