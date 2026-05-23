//! Deflate decoder stopping-point flags (patched ISA-L / rapidgzip semantics).
//!
//! Mirrors `rapidgzip::StoppingPoint` (`definitions.hpp:92-100`) and the
//! patched ISA-L `ISAL_STOPPING_POINT_*` constants used by the parallel
//! single-member wrapper.

#![allow(dead_code)] // END_OF_STREAM_HEADER / ALL wired at Track B3 wrapper parity

/// Bit-flag set: callers OR variants to request early return from inflate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
#[repr(transparent)]
pub struct StoppingPoint(pub u32);

impl StoppingPoint {
    pub const NONE: Self = Self(0);
    pub const END_OF_STREAM_HEADER: Self = Self(1);
    pub const END_OF_STREAM: Self = Self(2);
    pub const END_OF_BLOCK_HEADER: Self = Self(4);
    pub const END_OF_BLOCK: Self = Self(8);
    pub const ALL: Self = Self(0xFFFF_FFFF);

    #[inline]
    pub const fn contains(self, other: StoppingPoint) -> bool {
        (self.0 & other.0) != 0
    }

    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for StoppingPoint {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
