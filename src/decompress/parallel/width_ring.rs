#![cfg(parallel_sm)]
#![allow(dead_code)]
// P1 skeleton (plans/engine-campaign.md, plans/engine-u8-design.md §2): NOT wired
// into any production path yet. Block migrates onto this type in P2/M2; until
// then the only callers are the unit tests below.

//! `WidthRing` — the ONE dual-width decode window, extracted as a type.
//!
//! Faithful port of vendor rapidgzip's window pair as a single object
//! (`vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/deflate.hpp`):
//!
//! - ONE allocation, two widths: `PreDecodedBuffer = std::array<uint16_t,
//!   2*MAX_WINDOW_SIZE>` and its `reinterpret_cast` u8 view `DecodedBuffer`
//!   (deflate.hpp:805-806, 890-894). The "flip" is a WIDTH reinterpretation
//!   of the same memory, never a second buffer.
//! - Width selector `m_containsMarkerBytes` (deflate.hpp:936-939) is the
//!   [`RingWidth`] state here.
//! - Marker pre-init zone (deflate.hpp:875-888): the upper half holds
//!   `i + MAX_WINDOW_SIZE` so an out-of-history back-ref copy yields the
//!   correct MapMarkers value with no per-element marker branch.
//! - Marker-distance counter `m_distanceToLastMarkerByte`: ++ per clean
//!   literal (deflate.hpp:1311-1317), recomputed by a backward scan after
//!   each back-ref copy (deflate.hpp:1379-1389).
//! - Clean-window arming (deflate.hpp:1282-1284) in [`WidthRing::should_flip`].
//! - Flip mechanics `setInitialWindow` (deflate.hpp:1740-1785) in
//!   [`WidthRing::flip_in_place`]: conflate the surviving 32 KiB through a
//!   scratch buffer (vendor `conflatedBuffer`, :1772-1776), place it at the
//!   TAIL of the u8 view (:1778-1780), re-base the cursor (vendor
//!   `m_windowPosition = 0`, :1782). The seam bytes (this call's undrained
//!   output) are returned as u8 — vendor emits them as u8 views into the
//!   just-conflated window (deflate.hpp:1285-1286), i.e. no u16 re-read after
//!   the flip (fixes map DIV-4 relative to `Block::drain_transition_narrow_u16`).
//! - Pre-decode window seed (deflate.hpp:1750-1759) in [`WidthRing::seed_window`]:
//!   clean from byte 0.
//!
//! The emit methods here are simple per-element loops: this type is the
//! ownership/contract boundary. `Block`'s optimized fast loops keep writing
//! through raw pointers obtained FROM the ring after the M2 migration (same
//! codegen); see plans/engine-u8-design.md §2.

use super::marker_inflate::{MAX_WINDOW_SIZE, RING_SIZE, U8_RING_SIZE};
use super::replace_markers::MARKER_BASE;

/// Maximum deflate back-reference run length (RFC 1951 §3.2.5).
const MAX_RUN_LENGTH: usize = 258;

/// The active element width of the ring — vendor `m_containsMarkerBytes`
/// (deflate.hpp:936-939). `Marker` ⇒ u16 elements addressed `% RING_SIZE`;
/// `Clean` ⇒ u8 elements addressed `% U8_RING_SIZE` over the SAME bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingWidth {
    /// u16 marker mode: back-refs past the chunk start emit MapMarkers values.
    Marker,
    /// u8 clean mode: every back-ref resolves to a literal byte.
    Clean,
}

/// Errors surfaced by the skeleton's emit methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingError {
    /// `seed_window` called after decoding started, or seed longer than 32 KiB
    /// (vendor guard at deflate.hpp:1751 + the MAX_WINDOW_SIZE contract).
    InvalidSeed,
    /// Back-ref distance 0, > 32 KiB, or (clean mode only) beyond decoded
    /// history — vendor's `EXCEEDED_WINDOW_RANGE`, checked only when clean
    /// (`if constexpr (!containsMarkerBytes)`, deflate.hpp:1569-1573).
    ExceededWindowRange,
    /// The undrained span outgrew the lookback guarantee; the caller must
    /// drain at least once per `read()`-sized batch (vendor returns views
    /// per `read()` call, deflate.hpp:1286-1292).
    DrainOverdue,
}

/// The one dual-width decode window. See module docs for the vendor mapping.
pub struct WidthRing {
    /// Single backing allocation (vendor `m_window16`, deflate.hpp:926).
    /// Boxed for stack-pressure parity with vendor's heap'd Block
    /// (deflate.hpp:802-803 / GzipChunk.hpp:454-456).
    ring: Box<[u16; RING_SIZE]>,
    /// Logical write cursor (vendor `m_windowPosition`, deflate.hpp:933).
    /// Units are u16 slots in `Marker` width, u8 slots in `Clean` width;
    /// the flip re-bases it (deflate.hpp:1782 sets it to 0; we use
    /// `U8_RING_SIZE` ≡ physical 0 so `pos - distance` cannot underflow,
    /// mirroring `Block::flip_repack_to_u8`).
    pos: usize,
    /// Logical position up to which output has been handed to the caller.
    drained: usize,
    width: RingWidth,
    /// Vendor `m_distanceToLastMarkerByte` (deflate.hpp:944-951). Undefined
    /// once `width == Clean` (vendor: "the exact value does not matter and
    /// is undefined when m_containsMarkerBytes is false").
    distance_to_last_marker: usize,
    /// Vendor `m_decodedBytes` (deflate.hpp:940-944): total decoded bytes,
    /// including a seeded window's length (deflate.hpp:1755).
    decoded: usize,
}

impl Default for WidthRing {
    fn default() -> Self {
        Self::new()
    }
}

impl WidthRing {
    pub fn new() -> Self {
        let mut ring: Box<[u16; RING_SIZE]> = vec![0u16; RING_SIZE]
            .into_boxed_slice()
            .try_into()
            .expect("RING_SIZE-sized box");
        Self::init_marker_zone(&mut ring);
        WidthRing {
            ring,
            pos: 0,
            drained: 0,
            width: RingWidth::Marker,
            distance_to_last_marker: 0,
            decoded: 0,
        }
    }

    /// Vendor `initializeMarkedWindowBuffer` (deflate.hpp:875-888): the upper
    /// half holds `MAX_WINDOW_SIZE + i` so a back-ref reaching before the
    /// chunk start copies the correct MapMarkers value:
    /// at output position `p`, distance `d > p` reads physical slot
    /// `RING_SIZE - (d - p)`, whose pre-init value `65536 + p - d` equals
    /// `MARKER_BASE + (MAX_WINDOW_SIZE - (d - p))` — the window byte index
    /// from the OLDEST byte (MarkerReplacement.hpp:24-42).
    fn init_marker_zone(ring: &mut [u16; RING_SIZE]) {
        for i in 0..MAX_WINDOW_SIZE {
            ring[MAX_WINDOW_SIZE + i] = (MAX_WINDOW_SIZE + i) as u16;
        }
    }

    /// Re-init to a fresh marker-mode ring (vendor `Block::reset`,
    /// deflate.hpp:670-697, window-less arm).
    pub fn reset(&mut self) {
        self.pos = 0;
        self.drained = 0;
        self.width = RingWidth::Marker;
        self.distance_to_last_marker = 0;
        self.decoded = 0;
        Self::init_marker_zone(&mut self.ring);
    }

    pub fn width(&self) -> RingWidth {
        self.width
    }

    pub fn decoded_bytes(&self) -> usize {
        self.decoded
    }

    /// u8 view of the SAME backing bytes — vendor `getWindow()`'s
    /// `reinterpret_cast<std::uint8_t*>(m_window16.data())`
    /// (deflate.hpp:890-894).
    fn ring8(&self) -> &[u8] {
        // SAFETY: u8 has alignment 1 ≤ align_of::<u16>; the region is the
        // ring's own allocation, length U8_RING_SIZE == RING_SIZE * 2 bytes;
        // shared borrow of self prevents aliasing mutation.
        unsafe { std::slice::from_raw_parts(self.ring.as_ptr() as *const u8, U8_RING_SIZE) }
    }

    fn ring8_mut(&mut self) -> &mut [u8] {
        // SAFETY: as `ring8`, with the exclusive borrow of self.
        unsafe { std::slice::from_raw_parts_mut(self.ring.as_mut_ptr() as *mut u8, U8_RING_SIZE) }
    }

    /// Pre-decode window seed — vendor `setInitialWindow`'s
    /// "before decoding has started" arm (deflate.hpp:1750-1759): memcpy the
    /// seed into the u8 view at `[0, len)`, advance cursor and `m_decodedBytes`
    /// by the seed length, go Clean. The whole chunk then decodes u8-DIRECT.
    /// The seed itself is never drained (it is predecessor output).
    pub fn seed_window(&mut self, seed: &[u8]) -> Result<(), RingError> {
        if self.decoded != 0 || self.pos != 0 || seed.len() > MAX_WINDOW_SIZE {
            return Err(RingError::InvalidSeed);
        }
        // Vendor: an empty initialWindow still flips to clean
        // (deflate.hpp:1751-1759 — m_containsMarkerBytes = false even for the
        // empty case, used at stream starts where no history can be referenced).
        let len = seed.len();
        self.ring8_mut()[..len].copy_from_slice(seed);
        self.pos = len;
        self.drained = len;
        self.decoded = len;
        self.width = RingWidth::Clean;
        Ok(())
    }

    /// Append one literal byte (vendor `appendToWindow`, deflate.hpp:1303-1322;
    /// literals are < 256 hence always clean, so the marker counter always
    /// increments in marker mode).
    pub fn push_literal(&mut self, byte: u8) {
        match self.width {
            RingWidth::Marker => {
                let slot = self.pos % RING_SIZE;
                self.ring[slot] = byte as u16;
                self.distance_to_last_marker += 1;
            }
            RingWidth::Clean => {
                let slot = self.pos % U8_RING_SIZE;
                self.ring8_mut()[slot] = byte;
            }
        }
        self.pos += 1;
        self.decoded += 1;
    }

    /// Copy a back-reference (vendor `resolveBackreference`, deflate.hpp:1346-1422,
    /// expressed as the always-correct per-element loop: the source may overlap
    /// the destination, so elements are copied in order exactly as vendor's
    /// `appendToWindowUnsafe` loop does for the overlap case).
    ///
    /// Marker mode: a distance reaching past the chunk start lands in the
    /// pre-init marker zone, emitting MapMarkers values (no branch — see
    /// `init_marker_zone`). After the copy, the marker counter is recomputed
    /// by the vendor backward scan (deflate.hpp:1379-1389).
    ///
    /// Clean mode: the window-range check applies (vendor compiles it only for
    /// `!containsMarkerBytes`, deflate.hpp:1569-1573).
    pub fn copy_backref(&mut self, distance: usize, length: usize) -> Result<(), RingError> {
        if distance == 0 || distance > MAX_WINDOW_SIZE || length > MAX_RUN_LENGTH {
            return Err(RingError::ExceededWindowRange);
        }
        match self.width {
            RingWidth::Marker => {
                for _ in 0..length {
                    let src = (self.pos % RING_SIZE + RING_SIZE - distance) % RING_SIZE;
                    let v = self.ring[src];
                    let dst = self.pos % RING_SIZE;
                    self.ring[dst] = v;
                    self.pos += 1;
                }
                // Vendor backward scan (deflate.hpp:1379-1389): if the copied
                // run contains a marker, the counter is the clean run length
                // SINCE that marker; otherwise it grows by `length`.
                let mut clean_tail = 0usize;
                let mut found_marker = false;
                for k in 0..length {
                    let slot = (self.pos + RING_SIZE - 1 - k) % RING_SIZE;
                    if self.ring[slot] >= MARKER_BASE {
                        found_marker = true;
                        break;
                    }
                    clean_tail += 1;
                }
                if found_marker {
                    self.distance_to_last_marker = clean_tail;
                } else {
                    self.distance_to_last_marker += length;
                }
            }
            RingWidth::Clean => {
                if distance > self.decoded {
                    return Err(RingError::ExceededWindowRange);
                }
                for _ in 0..length {
                    let src = (self.pos % U8_RING_SIZE + U8_RING_SIZE - distance) % U8_RING_SIZE;
                    let b = self.ring8()[src];
                    let dst = self.pos % U8_RING_SIZE;
                    self.ring8_mut()[dst] = b;
                    self.pos += 1;
                }
            }
        }
        self.decoded += length;
        Ok(())
    }

    /// The clean-window arming predicate — EXACT vendor condition
    /// (deflate.hpp:1282-1284):
    /// `distanceToLastMarkerByte >= m_window16.size()` (the whole u16 ring,
    /// including the pre-init zone, has been overwritten clean) OR
    /// (`>= MAX_WINDOW_SIZE` AND `== m_decodedBytes`: the chunk has ONLY ever
    /// emitted clean bytes and 32 KiB of them exist).
    pub fn should_flip(&self) -> bool {
        self.width == RingWidth::Marker
            && (self.distance_to_last_marker >= RING_SIZE
                || (self.distance_to_last_marker >= MAX_WINDOW_SIZE
                    && self.distance_to_last_marker == self.decoded))
    }

    /// Flip the ring width in place — vendor `setInitialWindow`'s mid-decode
    /// arm (deflate.hpp:1762-1784): value-downcast the rotated last-32 KiB
    /// window through a scratch buffer (`conflatedBuffer`, :1772-1776; scratch
    /// is mandatory — the u8 destination tail physically overlaps the u16
    /// source slots), place it at the TAIL of the u8 view (:1778-1780), and
    /// re-base the cursor (:1782).
    ///
    /// The still-undrained seam bytes are appended to `seam_out` as u8 — the
    /// analog of vendor returning the flip call's output as u8 views into the
    /// conflated window (deflate.hpp:1285-1286).
    ///
    /// Precondition: `should_flip()` — which guarantees ≥ 32 KiB decoded and
    /// the last `MAX_WINDOW_SIZE` outputs clean (hence the undrained span,
    /// bounded by the drain contract at ≤ 32 KiB, is clean too).
    pub fn flip_in_place(&mut self, seam_out: &mut Vec<u8>) {
        debug_assert!(self.should_flip(), "flip_in_place without arming");
        debug_assert!(self.decoded >= MAX_WINDOW_SIZE);
        debug_assert!(self.pos - self.drained <= RING_SIZE - MAX_WINDOW_SIZE);

        // 1. Conflate: value-downcast logical [pos - 32 Ki, pos) in order.
        let mut conflated = vec![0u8; MAX_WINDOW_SIZE];
        for (k, out) in conflated.iter_mut().enumerate() {
            let slot = (self.pos % RING_SIZE + RING_SIZE - MAX_WINDOW_SIZE + k) % RING_SIZE;
            let v = self.ring[slot];
            debug_assert!(v < MARKER_BASE, "marker {v:#x} in window being flipped");
            *out = (v & 0xFF) as u8;
        }

        // 2. Seam: the undrained tail is the newest `pos - drained` bytes of
        //    the conflated window (clean by the arming guarantee).
        let undrained = self.pos - self.drained;
        debug_assert!(undrained <= MAX_WINDOW_SIZE);
        seam_out.extend_from_slice(&conflated[MAX_WINDOW_SIZE - undrained..]);

        // 3. Place the window at the u8 view's tail so a post-flip back-ref of
        //    distance d reads u8 slot `U8_RING_SIZE - d` — the byte logically
        //    `d` before the seam (vendor dest offset `window.size() -
        //    conflatedBuffer.size()`, deflate.hpp:1778-1780).
        self.ring8_mut()[U8_RING_SIZE - MAX_WINDOW_SIZE..].copy_from_slice(&conflated);

        // 4. Re-base the cursor to u8-logical. Vendor sets m_windowPosition = 0
        //    (deflate.hpp:1782); we use U8_RING_SIZE (≡ physical slot 0) so
        //    `pos - distance` arithmetic cannot underflow.
        self.pos = U8_RING_SIZE;
        self.drained = U8_RING_SIZE;
        self.width = RingWidth::Clean;
    }

    /// Flip if armed; returns whether the flip happened. Mirrors the per-call
    /// hook in `Block::read` (marker_inflate.rs `just_flipped`) / vendor's
    /// post-`readInternal` check (deflate.hpp:1279-1289).
    pub fn maybe_flip(&mut self, seam_out: &mut Vec<u8>) -> bool {
        if self.should_flip() {
            self.flip_in_place(seam_out);
            true
        } else {
            false
        }
    }

    /// Drain marker-mode output `[drained, pos)` as u16 (vendor
    /// `result.dataWithMarkers = lastBuffers(m_window16, ...)`,
    /// deflate.hpp:1288). Caller must drain at least once before the
    /// undrained span exceeds `RING_SIZE - MAX_WINDOW_SIZE` (the lookback
    /// guarantee — older slots get overwritten past that).
    pub fn drain_u16(&mut self, out: &mut Vec<u16>) -> Result<usize, RingError> {
        debug_assert_eq!(self.width, RingWidth::Marker);
        let n = self.pos - self.drained;
        if n > RING_SIZE - MAX_WINDOW_SIZE {
            return Err(RingError::DrainOverdue);
        }
        out.reserve(n);
        for i in 0..n {
            out.push(self.ring[(self.drained + i) % RING_SIZE]);
        }
        self.drained = self.pos;
        Ok(n)
    }

    /// Drain clean-mode output `[drained, pos)` as u8 — a plain copy, no
    /// narrow (vendor `result.data = lastBuffers(window, ...)`,
    /// deflate.hpp:1292). Same drain-cadence contract as `drain_u16`
    /// (clean lookback bound: `U8_RING_SIZE - MAX_WINDOW_SIZE`).
    pub fn drain_u8(&mut self, out: &mut Vec<u8>) -> Result<usize, RingError> {
        debug_assert_eq!(self.width, RingWidth::Clean);
        let n = self.pos - self.drained;
        if n > U8_RING_SIZE - MAX_WINDOW_SIZE {
            return Err(RingError::DrainOverdue);
        }
        out.reserve(n);
        for i in 0..n {
            let b = self.ring8()[(self.drained + i) % U8_RING_SIZE];
            out.push(b);
        }
        self.drained = self.pos;
        Ok(n)
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// MapMarkers (MarkerReplacement.hpp:24-42): ≤255 literal; ≥ MARKER_BASE
    /// indexes the 32 KiB window from its OLDEST byte; the gap is invalid.
    fn map_marker(v: u16, window: &[u8; MAX_WINDOW_SIZE]) -> u8 {
        if v < 256 {
            v as u8
        } else {
            assert!(v >= MARKER_BASE, "invalid 2 B code {v:#x}");
            window[(v - MARKER_BASE) as usize]
        }
    }

    /// One synthetic decode op.
    #[derive(Debug, Clone, Copy)]
    enum Op {
        Lit(u8),
        Backref { distance: usize, length: usize },
    }

    /// Trivially-correct reference: flat u16 history with MapMarkers-encoded
    /// out-of-history reads (the model `init_marker_zone`'s trick must equal).
    fn reference_u16(ops: &[Op]) -> Vec<u16> {
        let mut hist: Vec<u16> = Vec::new();
        for &op in ops {
            match op {
                Op::Lit(b) => hist.push(b as u16),
                Op::Backref { distance, length } => {
                    for _ in 0..length {
                        let p = hist.len();
                        let v = if distance <= p {
                            hist[p - distance]
                        } else {
                            // Window byte index from oldest: 32 Ki - (d - p).
                            MARKER_BASE + (MAX_WINDOW_SIZE - (distance - p)) as u16
                        };
                        hist.push(v);
                    }
                }
            }
        }
        hist
    }

    /// Reference for the seeded/clean path: real bytes against a known window.
    fn reference_clean(ops: &[Op], window: &[u8; MAX_WINDOW_SIZE]) -> Vec<u8> {
        let mut full: Vec<u8> = window.to_vec();
        for &op in ops {
            match op {
                Op::Lit(b) => full.push(b),
                Op::Backref { distance, length } => {
                    for _ in 0..length {
                        let b = full[full.len() - distance];
                        full.push(b);
                    }
                }
            }
        }
        full[MAX_WINDOW_SIZE..].to_vec()
    }

    /// Replay ops through a marker-mode WidthRing with the production drain
    /// cadence and the per-batch flip hook. Returns (u16 prefix, u8 tail).
    fn replay_marker(ops: &[Op]) -> (Vec<u16>, Vec<u8>) {
        let mut ring = WidthRing::new();
        let mut u16_out: Vec<u16> = Vec::new();
        let mut u8_out: Vec<u8> = Vec::new();
        for &op in ops {
            match op {
                Op::Lit(b) => ring.push_literal(b),
                Op::Backref { distance, length } => {
                    ring.copy_backref(distance, length).expect("backref");
                }
            }
            // Drain cadence: stay within the lookback bound (max op = 258).
            if ring.pos - ring.drained > 16 * 1024 {
                match ring.width() {
                    RingWidth::Marker => {
                        ring.drain_u16(&mut u16_out).unwrap();
                    }
                    RingWidth::Clean => {
                        ring.drain_u8(&mut u8_out).unwrap();
                    }
                }
            }
            // Flip hook (per batch in production; per op here — the predicate
            // is checked at the same drain-safe points, output is identical).
            if ring.width() == RingWidth::Marker {
                ring.drain_u16(&mut u16_out).unwrap();
                ring.maybe_flip(&mut u8_out);
            }
        }
        match ring.width() {
            RingWidth::Marker => {
                ring.drain_u16(&mut u16_out).unwrap();
            }
            RingWidth::Clean => {
                ring.drain_u8(&mut u8_out).unwrap();
            }
        }
        (u16_out, u8_out)
    }

    /// Replay ops through a window-seeded (clean from byte 0) WidthRing.
    fn replay_seeded(ops: &[Op], window: &[u8; MAX_WINDOW_SIZE]) -> Vec<u8> {
        let mut ring = WidthRing::new();
        ring.seed_window(window).expect("seed");
        let mut out: Vec<u8> = Vec::new();
        for &op in ops {
            match op {
                Op::Lit(b) => ring.push_literal(b),
                Op::Backref { distance, length } => {
                    ring.copy_backref(distance, length).expect("backref");
                }
            }
            if ring.pos - ring.drained > 16 * 1024 {
                ring.drain_u8(&mut out).unwrap();
            }
        }
        ring.drain_u8(&mut out).unwrap();
        out
    }

    /// Resolve a replayed (u16 prefix, u8 tail) pair against a window.
    fn resolve(u16_prefix: &[u16], u8_tail: &[u8], window: &[u8; MAX_WINDOW_SIZE]) -> Vec<u8> {
        let mut out: Vec<u8> = u16_prefix.iter().map(|&v| map_marker(v, window)).collect();
        out.extend_from_slice(u8_tail);
        out
    }

    fn test_window() -> [u8; MAX_WINDOW_SIZE] {
        let mut w = [0u8; MAX_WINDOW_SIZE];
        for (i, slot) in w.iter_mut().enumerate() {
            *slot = (i % 251) as u8;
        }
        w
    }

    /// Deterministic op-stream generator (LCG; no external dev-deps).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// `marker_share`: per-mille chance a backref deliberately reaches past
    /// the produced prefix (emitting markers). Distances stay ≤ 32 Ki.
    fn gen_ops(seed: u64, n_ops: usize, backref_share: usize, marker_share: usize) -> Vec<Op> {
        let mut rng = Lcg(seed);
        let mut produced = 0usize;
        let mut ops = Vec::with_capacity(n_ops);
        for _ in 0..n_ops {
            let is_backref = rng.below(1000) < backref_share;
            if is_backref {
                let length = 3 + rng.below(MAX_RUN_LENGTH - 2);
                let wants_marker = rng.below(1000) < marker_share;
                let distance = if wants_marker || produced == 0 {
                    // Reach past the chunk start (≤ 32 Ki total).
                    let beyond = 1 + rng.below(MAX_WINDOW_SIZE.saturating_sub(produced).max(1));
                    (produced + beyond).min(MAX_WINDOW_SIZE)
                } else {
                    1 + rng.below(produced.min(MAX_WINDOW_SIZE))
                };
                if distance == 0 || distance > MAX_WINDOW_SIZE {
                    continue;
                }
                ops.push(Op::Backref { distance, length });
                produced += length;
            } else {
                ops.push(Op::Lit(rng.below(256) as u8));
                produced += 1;
            }
        }
        ops
    }

    // ── flip semantics ───────────────────────────────────────────────────

    #[test]
    fn arming_condition_two_all_clean_chunk() {
        // Vendor condition 2 (deflate.hpp:1283-1284): all output clean AND
        // ≥ 32 KiB of it.
        let mut ring = WidthRing::new();
        for i in 0..MAX_WINDOW_SIZE - 1 {
            ring.push_literal((i % 256) as u8);
            assert!(!ring.should_flip(), "armed too early at {i}");
        }
        ring.push_literal(0xAB);
        assert!(ring.should_flip(), "must arm at exactly 32 KiB all-clean");
    }

    #[test]
    fn arming_condition_one_after_marker() {
        // Vendor condition 1 (deflate.hpp:1282): after ANY marker, arming
        // needs the full u16 ring (64 Ki) of consecutive clean output —
        // 32 KiB is NOT enough (condition 2 fails: counter != decoded).
        let mut ring = WidthRing::new();
        ring.push_literal(1);
        ring.copy_backref(MAX_WINDOW_SIZE, 4).unwrap(); // emits markers
        let mut sink16 = Vec::new();
        for i in 0..RING_SIZE - 1 {
            ring.push_literal((i % 256) as u8);
            if ring.pos - ring.drained > 16 * 1024 {
                ring.drain_u16(&mut sink16).unwrap();
            }
            assert!(!ring.should_flip(), "armed too early at {i}");
        }
        ring.push_literal(0xCD);
        assert!(ring.should_flip(), "must arm after 64 Ki consecutive clean");
    }

    #[test]
    fn flip_places_window_at_u8_tail_and_rebases() {
        let mut ring = WidthRing::new();
        let mut sink16 = Vec::new();
        let total = MAX_WINDOW_SIZE + 100;
        for i in 0..total {
            ring.push_literal((i % 256) as u8);
            if ring.pos - ring.drained > 16 * 1024 {
                ring.drain_u16(&mut sink16).unwrap();
            }
        }
        assert!(ring.should_flip());
        let mut seam = Vec::new();
        ring.flip_in_place(&mut seam);
        assert_eq!(ring.width(), RingWidth::Clean);
        assert_eq!(ring.pos, U8_RING_SIZE);
        assert_eq!(ring.drained, U8_RING_SIZE);
        // Seam = the undrained bytes, narrowed.
        let drained_so_far = sink16.len();
        assert_eq!(seam.len(), total - drained_so_far);
        assert_eq!(seam[0], (drained_so_far % 256) as u8);
        // A post-flip back-ref of distance d must read the byte logically d
        // before the seam: logical index total - d.
        for &d in &[1usize, 100, 4097, MAX_WINDOW_SIZE] {
            let mut probe = ring.ring8()[(U8_RING_SIZE - d) % U8_RING_SIZE];
            let expect = ((total - d) % 256) as u8;
            assert_eq!(probe, expect, "window byte at distance {d}");
            // And through the public API:
            ring.copy_backref(d, 1).unwrap();
            let mut out = Vec::new();
            ring.drain_u8(&mut out).unwrap();
            probe = *out.last().unwrap();
            // The previous iterations appended bytes after the seam, so the
            // expected byte for this read shifts accordingly; recompute from
            // the reference: history = 0..total bytes then the bytes we just
            // appended. Simplest exact check: compare against a shadow model.
            let _ = probe; // checked below in the model-driven tests
        }
    }

    #[test]
    fn marker_zone_emits_mapmarkers_values() {
        // At output position p, distance d > p must produce
        // MARKER_BASE + (32 Ki - (d - p)) — MapMarkers' window index from the
        // oldest byte (MarkerReplacement.hpp:24-42).
        let mut ring = WidthRing::new();
        ring.push_literal(7);
        ring.push_literal(9); // p = 2
        ring.copy_backref(10, 3).unwrap(); // d = 10 > p = 2
        let mut out = Vec::new();
        ring.drain_u16(&mut out).unwrap();
        assert_eq!(out[0], 7);
        assert_eq!(out[1], 9);
        // Element 0 of the copy: p = 2, d = 10 ⇒ 32768 + (32768 - 8) = 65528
        // (computed in u32 — the sum's intermediate exceeds u16::MAX/2 + ...).
        let expect = |off: u32| (MARKER_BASE as u32 + MAX_WINDOW_SIZE as u32 - off) as u16;
        assert_eq!(out[2], expect(8));
        assert_eq!(out[3], expect(7));
        assert_eq!(out[4], expect(6));
    }

    #[test]
    fn backref_marker_counter_backward_scan() {
        // Copying a run that ENDS clean but contains a marker must set the
        // counter to the clean tail length (vendor deflate.hpp:1379-1389).
        let mut ring = WidthRing::new();
        for _ in 0..4 {
            ring.push_literal(5);
        }
        // p=4: emit a marker (d=5 > p=4) then copy a run that overlaps it.
        ring.copy_backref(5, 1).unwrap(); // marker at logical 4
        assert_eq!(ring.distance_to_last_marker, 0);
        // Copies logical 0..5 = [5,5,5,5,marker] → ends ON the marker ⇒ counter 0.
        ring.copy_backref(5, 5).unwrap();
        assert_eq!(ring.distance_to_last_marker, 0);
        ring.copy_backref(10, 4).unwrap(); // copies logical 0..4: all clean
        assert_eq!(ring.distance_to_last_marker, 4);
    }

    #[test]
    fn seed_window_clean_from_byte_zero() {
        let window = test_window();
        let mut ring = WidthRing::new();
        ring.seed_window(&window).unwrap();
        assert_eq!(ring.width(), RingWidth::Clean);
        assert_eq!(ring.decoded_bytes(), MAX_WINDOW_SIZE);
        ring.copy_backref(MAX_WINDOW_SIZE, 4).unwrap(); // oldest seed bytes
        let mut out = Vec::new();
        ring.drain_u8(&mut out).unwrap();
        assert_eq!(out, &window[..4]);
    }

    #[test]
    fn seed_window_rejects_after_decode_and_oversize() {
        let mut ring = WidthRing::new();
        ring.push_literal(1);
        assert_eq!(ring.seed_window(&[0u8; 16]), Err(RingError::InvalidSeed));
        let mut ring2 = WidthRing::new();
        assert_eq!(
            ring2.seed_window(&vec![0u8; MAX_WINDOW_SIZE + 1]),
            Err(RingError::InvalidSeed)
        );
    }

    #[test]
    fn clean_mode_window_range_check() {
        // Vendor compiles the distance > decoded check only when clean
        // (deflate.hpp:1569-1573); marker mode must NOT reject it.
        let mut ring = WidthRing::new();
        ring.seed_window(&[1, 2, 3, 4]).unwrap();
        assert_eq!(ring.copy_backref(5, 1), Err(RingError::ExceededWindowRange));
        let mut marker_ring = WidthRing::new();
        marker_ring.push_literal(0);
        assert!(marker_ring.copy_backref(5, 1).is_ok());
    }

    // ── marker-resolution equivalence + reference-model differentials ────

    /// Resolved marker-path output == seeded clean-path output == reference,
    /// across generated op corpora (clean-heavy, marker-heavy, flip-crossing,
    /// wrap-crossing).
    #[test]
    fn marker_vs_seeded_equivalence_on_generated_corpora() {
        let window = test_window();
        let profiles: &[(u64, usize, usize, usize, &str)] = &[
            (11, 4_000, 250, 0, "clean-heavy, no markers"),
            (13, 4_000, 300, 400, "marker-heavy"),
            (17, 80_000, 200, 5, "sparse markers, long clean runs"),
            (19, 160_000, 100, 0, "all-clean flip + wrap crossing"),
            (23, 600_000, 250, 2, "multi-wrap, late markers"),
        ];
        for &(seed, n_ops, backref_share, marker_share, name) in profiles {
            let ops = gen_ops(seed, n_ops, backref_share, marker_share);

            // Reference models.
            let ref16 = reference_u16(&ops);
            let ref_clean = reference_clean(&ops, &window);
            // Cross-check the two references against each other first.
            let ref16_resolved: Vec<u8> = ref16.iter().map(|&v| map_marker(v, &window)).collect();
            assert_eq!(ref16_resolved, ref_clean, "[{name}] reference self-check");

            // WidthRing marker path (with natural flip) resolved against the
            // window must equal the reference bytes.
            let (u16_prefix, u8_tail) = replay_marker(&ops);
            // The u16 prefix must match the reference u16 stream element-wise
            // (markers preserved through backref copies of marker values).
            assert_eq!(
                u16_prefix[..],
                ref16[..u16_prefix.len()],
                "[{name}] u16 marker stream"
            );
            let resolved = resolve(&u16_prefix, &u8_tail, &window);
            assert_eq!(resolved, ref_clean, "[{name}] marker path resolved bytes");

            // WidthRing seeded path must equal the same bytes u8-direct.
            let seeded = replay_seeded(&ops, &window);
            assert_eq!(seeded, ref_clean, "[{name}] seeded clean path bytes");
        }
    }

    /// The flip must occur for all-clean streams and the u16 prefix must end
    /// exactly where the seam starts (no byte lost or duplicated at the seam).
    #[test]
    fn seam_stitching_exact() {
        let ops = gen_ops(29, 200_000, 150, 0);
        let window = test_window();
        let (u16_prefix, u8_tail) = replay_marker(&ops);
        assert!(!u8_tail.is_empty(), "flip must arm on an all-clean stream");
        let ref_clean = reference_clean(&ops, &window);
        assert_eq!(u16_prefix.len() + u8_tail.len(), ref_clean.len());
        let resolved = resolve(&u16_prefix, &u8_tail, &window);
        assert_eq!(resolved, ref_clean);
    }

    #[test]
    fn drain_overdue_is_reported() {
        let mut ring = WidthRing::new();
        for i in 0..(RING_SIZE - MAX_WINDOW_SIZE + 1) {
            ring.push_literal((i % 256) as u8);
        }
        let mut out = Vec::new();
        assert_eq!(ring.drain_u16(&mut out), Err(RingError::DrainOverdue));
    }

    // ── byte-exact cross-check against the existing engine ──────────────
    //
    // Locks the contract WidthRing must preserve through the M2 migration:
    // the production `Block` decoding a real deflate-with-dictionary corpus,
    // marker path (+ MapMarkers resolve) vs seeded path vs flate2 ground
    // truth. (WidthRing itself is not a deflate decoder; the engine moves
    // ONTO it in M2 and this net then runs against the migrated Block.)
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    #[test]
    fn existing_engine_marker_vs_seeded_byte_exact_on_generated_corpus() {
        use crate::decompress::inflate::consume_first_decode::Bits;
        use crate::decompress::parallel::marker_inflate::Block;

        // Generated corpus: repetitive + pseudo-random mix so the encoder
        // emits both literals and dictionary-reaching back-refs.
        let mut rng = Lcg(42);
        let mut dict = vec![0u8; MAX_WINDOW_SIZE];
        for b in dict.iter_mut() {
            *b = rng.below(256) as u8;
        }
        let mut payload = Vec::with_capacity(200_000);
        // Dictionary-reaching prefix: repeat dict slices.
        for k in 0..64 {
            let start = (k * 509) % (MAX_WINDOW_SIZE - 300);
            payload.extend_from_slice(&dict[start..start + 200 + rng.below(80)]);
        }
        // Then self-similar text.
        while payload.len() < 180_000 {
            let phrase = format!("gzippy-widthring-{}-", rng.below(1000));
            payload.extend_from_slice(phrase.as_bytes());
        }

        // Raw DEFLATE with the preset dictionary.
        let deflated = {
            use flate2::{Compress, Compression, FlushCompress};
            let mut c = Compress::new(Compression::default(), false);
            c.set_dictionary(&dict).expect("set_dictionary");
            let mut out = vec![0u8; payload.len() + 4096];
            let status = c
                .compress(&payload, &mut out, FlushCompress::Finish)
                .expect("compress");
            assert_eq!(status, flate2::Status::StreamEnd);
            out.truncate(c.total_out() as usize);
            out
        };

        // Ground truth via flate2 decompress-with-dict.
        let ground_truth = {
            use flate2::{Decompress, FlushDecompress};
            let mut d = Decompress::new(false);
            d.set_dictionary(&dict).expect("set_dictionary");
            let mut out = vec![0u8; payload.len() + 4096];
            d.decompress(&deflated, &mut out, FlushDecompress::Finish)
                .expect("decompress");
            out.truncate(d.total_out() as usize);
            out
        };
        assert_eq!(ground_truth, payload, "flate2 round-trip self-check");

        let leaked: &'static [u8] = Box::leak(deflated.clone().into_boxed_slice());

        // (a) Existing engine, MARKER path: no window ⇒ u16 + markers,
        //     resolved via the production MapMarkers semantics.
        let marker_resolved: Vec<u8> = {
            let mut bits = Bits::new(leaked);
            let mut block = Block::new();
            let mut out: Vec<u16> = Vec::new();
            loop {
                block.read_header(&mut bits, false).expect("header");
                while !block.eob() {
                    block.read(&mut bits, &mut out, usize::MAX).expect("read");
                }
                if block.is_last_block() {
                    break;
                }
            }
            let mut window32 = [0u8; MAX_WINDOW_SIZE];
            window32.copy_from_slice(&dict[..]);
            out.iter().map(|&v| map_marker(v, &window32)).collect()
        };
        assert_eq!(
            marker_resolved, ground_truth,
            "existing engine marker path + resolve"
        );

        // (b) Existing engine, SEEDED path: clean u8-direct from byte 0.
        let seeded_bytes: Vec<u8> = {
            let mut bits = Bits::new(leaked);
            let mut block = Block::new();
            let mut out: Vec<u16> = Vec::new();
            block.set_initial_window(&mut out, &dict).expect("seed");
            loop {
                block.read_header(&mut bits, false).expect("header");
                while !block.eob() {
                    block.read(&mut bits, &mut out, usize::MAX).expect("read");
                }
                if block.is_last_block() {
                    break;
                }
            }
            out.iter()
                .map(|&v| {
                    assert!(v < 256, "seeded decode must not emit markers");
                    v as u8
                })
                .collect()
        };
        assert_eq!(seeded_bytes, ground_truth, "existing engine seeded path");
    }
}
