//! Named alternatives that replace boolean parameters in the encoder.
//!
//! Both types below exist because a `bool` in these positions is a footgun that
//! has already fired. `deflate_into` and the parse runners took `is_last` and
//! `consume_all` as adjacent booleans; they are INDEPENDENT, and conflating
//! them made every non-final segment of the parallel path stop consuming its
//! input. Five concatenation tests caught it, but only because those tests
//! exist — nothing in the types said the two could not be swapped.
//!
//! See `docs/encoder-architecture.md`: no boolean parameters where the two
//! values are not obviously opposites.

/// Whether the block being emitted closes the DEFLATE stream.
///
/// This is purely about the BFINAL bit. A segment in the middle of a
/// CONCATENATED stream (the T>1 path) is [`BlockRole::Interior`] even though
/// it consumes all of its own input — which is exactly the distinction
/// [`InputMode`] carries and this type does not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockRole {
    /// More blocks follow; BFINAL stays 0.
    Interior,
    /// Closes the stream; the last block carries BFINAL=1.
    Final,
}

impl BlockRole {
    #[inline]
    pub fn is_final(self) -> bool {
        matches!(self, BlockRole::Final)
    }
}

/// Whether the parser must consume every byte it was handed.
///
/// [`InputMode::Drain`] is the whole-buffer contract: this is all the input
/// there will ever be for this call, parse to the end. [`InputMode::Bounded`]
/// is the single-pass streaming contract: the buffer will be refilled, so stop
/// at the last block boundary that still had a full lookahead behind it and
/// leave the tail for the caller to carry forward. Only `Bounded` can return a
/// position short of the end.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InputMode {
    /// Parse to the end of the supplied range.
    Drain,
    /// The buffer will be refilled; stop at a block boundary and report where.
    Bounded,
}

impl InputMode {
    #[inline]
    pub fn must_drain(self) -> bool {
        matches!(self, InputMode::Drain)
    }
}

/// How much wall time this encode path can afford to spend shrinking a block
/// header. A property of the PATH, not of the level — which is why it is not a
/// field on [`LevelParams`](super::level::LevelParams).
///
/// `Lean` is the DEFAULT so that any construction site nobody threads fails
/// SAFE (shaping off, T1 bytes unchanged) rather than silently opting in.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum HeaderBudget {
    /// Wall-critical path (T1). Build codes from the true histogram only.
    #[default]
    Lean,
    /// Wall-slack path (T>1). Also try the RLE-shaped histogram, keep it when
    /// strictly cheaper.
    Generous,
}

impl HeaderBudget {
    #[inline]
    pub fn may_shape(self) -> bool {
        matches!(self, HeaderBudget::Generous)
    }
}
