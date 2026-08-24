//! C: `vendor/libdeflate/lib/matchfinder_common.h` — the pieces every matchfinder
//! shares: the position type, the sliding-window rebase, the hash, and `lz_extend`.

/// C: `#define MATCHFINDER_WINDOW_ORDER 15` (set by the compressor's build)
pub(crate) const MATCHFINDER_WINDOW_ORDER: u32 = 15;

/// C: `#define MATCHFINDER_WINDOW_SIZE (1UL << MATCHFINDER_WINDOW_ORDER)` (:47)
pub(crate) const MATCHFINDER_WINDOW_SIZE: i32 = 1 << MATCHFINDER_WINDOW_ORDER;

/// C: `typedef s16 mf_pos_t;` (:49)
///
/// **A SIGNED 16-bit position, and the signedness is the whole design.** Positions are
/// relative to a base that slides; an entry older than the window becomes negative and
/// is rejected by the single test `cur_node <= cutoff`. Widening this to `u16` or
/// `i32` would need an extra validity flag and change the rebase.
pub(crate) type MfPos = i16;

/// C: `#define MATCHFINDER_INITVAL ((mf_pos_t)-MATCHFINDER_WINDOW_SIZE)` (:51)
///
/// -32768 — which is exactly `i16::MIN`, so an empty slot is permanently out of
/// bounds and no "is this slot occupied" branch is ever needed.
///
/// **Negate FIRST, then narrow.** The C is `((mf_pos_t)-MATCHFINDER_WINDOW_SIZE)`:
/// `-32768` is computed in `int` and then cast to `s16`. Writing it the other way
/// round — `-(32768 as i16)` — narrows 32768 to `i16::MIN` and then negates it, which
/// overflows. Rust catches that at compile time; C would have wrapped back to
/// `-32768` and nobody would ever have noticed the reversed order.
pub(crate) const MATCHFINDER_INITVAL: MfPos = (-MATCHFINDER_WINDOW_SIZE) as MfPos;

/// C: `matchfinder_init(mf_pos_t *data, size_t size)` (:108)
///
/// Essentially an optimized `memset`.
pub(crate) fn matchfinder_init(data: &mut [MfPos]) {
    for e in data.iter_mut() {
        *e = MATCHFINDER_INITVAL;
    }
}

/// C: `matchfinder_rebase(mf_pos_t *data, size_t size)` (:140)
///
/// Slide the matchfinder by `MATCHFINDER_WINDOW_SIZE` bytes. This must be called just
/// after each `MATCHFINDER_WINDOW_SIZE` bytes have been run through the matchfinder.
///
/// This subtracts `MATCHFINDER_WINDOW_SIZE` bytes from each entry, making the entries
/// be relative to the current position rather than the position
/// `MATCHFINDER_WINDOW_SIZE` bytes prior. To avoid integer underflows, entries that
/// would become less than `-MATCHFINDER_WINDOW_SIZE` stay at
/// `-MATCHFINDER_WINDOW_SIZE`, keeping them permanently out of bounds.
///
/// # The branchless form is not an optimisation to skip
///
/// For a 32768-byte window the C uses `data[i] = 0x8000 | (data[i] & ~(data[i] >> 15))`:
/// clear all bits if the value was already negative, then set the sign bit. That is
/// signed-saturating subtraction of 32768, and it is what the vectorised
/// architecture-specific overrides implement. `data[i] >> 15` is an ARITHMETIC shift
/// of a signed value — all-ones for negatives, zero for non-negatives. A logical shift
/// gives 0 or 1 and the mask collapses; that single wrong shift silently corrupts
/// every window slide.
pub(crate) fn matchfinder_rebase(data: &mut [MfPos]) {
    if MATCHFINDER_WINDOW_SIZE == 32768 {
        // Branchless version for 32768-byte windows.
        for e in data.iter_mut() {
            *e = (0x8000u16 as i16) | (*e & !(*e >> 15));
        }
    } else {
        // C: `if (data[i] >= 0) data[i] -= (mf_pos_t)-MATCHFINDER_WINDOW_SIZE;
        //     else data[i] = (mf_pos_t)-MATCHFINDER_WINDOW_SIZE;`
        //
        // NOTE, and it is NOT a port bug: the C subtracts `MATCHFINDER_INITVAL`,
        // which is `-WINDOW_SIZE`, so the non-negative arm reads as `data[i] +=
        // WINDOW_SIZE` — the opposite direction from the function's own comment and
        // from the branchless arm above. This branch is UNREACHABLE for DEFLATE
        // (`MATCHFINDER_WINDOW_ORDER` is 15, so `MATCHFINDER_WINDOW_SIZE` is 32768 and
        // the branchless arm always wins), which is presumably why it has never
        // mattered. Ported exactly as written rather than "corrected": this module's
        // contract is to reproduce the C, and silently fixing an unreachable branch
        // would make a future diff against upstream look like OUR divergence.
        // `wrapping_sub` because at width 15 the subtraction would overflow `i16`.
        for e in data.iter_mut() {
            if *e >= 0 {
                *e = e.wrapping_sub(MATCHFINDER_INITVAL);
            } else {
                *e = MATCHFINDER_INITVAL;
            }
        }
    }
}

/// C: `lz_hash(u32 seq, unsigned num_bits)` (:169)
///
/// Given a sequence prefix held in the low-order bits of a 32-bit value, multiply by a
/// carefully-chosen large constant. Discard any bits of the product that don't fit in
/// a 32-bit value, but take the next-highest `num_bits` bits of the product as the
/// hash value, as those have the most randomness.
///
/// The multiply MUST wrap — that is what "discard any bits that don't fit" means.
#[inline(always)]
pub(crate) fn lz_hash(seq: u32, num_bits: u32) -> u32 {
    seq.wrapping_mul(0x1E35_A7BD) >> (32 - num_bits)
}

/// C: `lz_extend(strptr, matchptr, start_len, max_len)` (:178)
///
/// Return the number of bytes at `matchptr` that match the bytes at `strptr`, up to a
/// maximum of `max_len`. Initially, `start_len` bytes are matched.
///
/// The C unrolls four word comparisons, then loops by words, then finishes byte by
/// byte, and locates the first differing byte with a bit-scan of the XOR. The word
/// path is reproduced with `u64` loads and `trailing_zeros`, which is `bsfw` on
/// little-endian; the result is identical to the byte loop, so the shape here is about
/// codegen, not about the answer.
#[inline(always)]
pub(crate) fn lz_extend(
    buf: &[u8],
    strptr: usize,
    matchptr: usize,
    start_len: u32,
    max_len: u32,
) -> u32 {
    const WORDBYTES: u32 = 8;
    let mut len = start_len;

    // C relies on this caller contract rather than clamping its inner-loop limit.
    // The parsers establish it by capping `max_len` at bytes remaining from
    // `strptr`; `matchptr` is an earlier input position.  Keep that proof live in
    // debug builds, but do not turn it into release work on every extension.
    debug_assert!(strptr <= buf.len() && matchptr <= buf.len());
    debug_assert!(max_len as usize <= buf.len() - strptr);
    debug_assert!(max_len as usize <= buf.len() - matchptr);
    debug_assert!(start_len <= max_len);

    #[inline(always)]
    unsafe fn load_word(buf: &[u8], i: usize) -> u64 {
        // SAFETY: the caller proves that `[i, i + WORDBYTES)` is in `buf`.
        u64::from_le(unsafe { (buf.as_ptr().add(i) as *const u64).read_unaligned() })
    }

    // C: four `COMPARE_WORD_STEP`s before the regular word loop.  This is not
    // merely an unroll hint: spelling all four keeps the same fast-path control
    // flow and lets LLVM schedule the independent unaligned loads as C does.
    if max_len - len >= 4 * WORDBYTES {
        macro_rules! compare_word_step {
            () => {{
                // SAFETY: the enclosing guard leaves at least four full words.
                let v_word = unsafe {
                    load_word(buf, matchptr + len as usize) ^ load_word(buf, strptr + len as usize)
                };
                if v_word != 0 {
                    return len + (v_word.trailing_zeros() >> 3);
                }
                len += WORDBYTES;
            }};
        }
        compare_word_step!();
        compare_word_step!();
        compare_word_step!();
        compare_word_step!();
    }

    while len + WORDBYTES <= max_len {
        // SAFETY: the loop condition and entry clamp prove both loads fit.
        let v_word = unsafe {
            load_word(buf, matchptr + len as usize) ^ load_word(buf, strptr + len as usize)
        };
        if v_word != 0 {
            // C: `len += bsfw(v_word) >> 3` on little-endian targets.
            return len + (v_word.trailing_zeros() >> 3);
        }
        len += WORDBYTES;
    }

    while len < max_len {
        // SAFETY: `len < max_len` and the entry clamp prove both loads fit.
        if unsafe {
            *buf.get_unchecked(matchptr + len as usize) != *buf.get_unchecked(strptr + len as usize)
        } {
            break;
        }
        len += 1;
    }
    len
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty slot must be permanently out of bounds: `MATCHFINDER_INITVAL` is
    /// `i16::MIN`, and `cutoff = cur_pos - WINDOW_SIZE` can never go below it for any
    /// legal `cur_pos` in `0..=WINDOW_SIZE`. That is what lets the matchfinder use one
    /// comparison instead of an occupancy check.
    #[test]
    fn the_init_value_is_permanently_out_of_bounds() {
        assert_eq!(MATCHFINDER_INITVAL, i16::MIN);
        // Compare in i32, BEFORE the narrowing cast. Comparing the i16 values would be
        // trivially true (nothing is below `i16::MIN`) and would prove nothing; the
        // real claim is that `cur_pos - WINDOW_SIZE` never wraps on its way into i16,
        // so the cutoff the matchfinder actually uses is the value it intended.
        for cur_pos in 0..=MATCHFINDER_WINDOW_SIZE {
            let cutoff_i32 = cur_pos - MATCHFINDER_WINDOW_SIZE;
            assert!(
                cutoff_i32 >= MATCHFINDER_INITVAL as i32,
                "cur_pos={cur_pos}: cutoff {cutoff_i32} is below the init value"
            );
            assert_eq!(
                cutoff_i32 as i16 as i32, cutoff_i32,
                "cur_pos={cur_pos}: the cutoff does not survive narrowing to i16"
            );
        }
    }

    /// The branchless rebase must equal signed-saturating subtraction of 32768 over
    /// the WHOLE i16 domain — all 65,536 values, not a sample. This is the one line
    /// where an arithmetic-vs-logical shift mistake compiles and is wrong.
    #[test]
    fn the_branchless_rebase_is_saturating_subtraction_over_all_of_i16() {
        let mut data: Vec<MfPos> = (i16::MIN..=i16::MAX).collect();
        let before = data.clone();
        matchfinder_rebase(&mut data);

        for (i, (&got, &was)) in data.iter().zip(before.iter()).enumerate() {
            let want: i16 = if was >= 0 {
                // was - 32768, saturating at -32768.
                (was as i32 - 32768).max(-32768) as i16
            } else {
                -32768
            };
            assert_eq!(
                got, want,
                "entry {i}: rebase({was}) gave {got}, want {want}"
            );
        }
    }

    /// After a rebase every entry is still a valid `mf_pos_t` and no previously
    /// out-of-window entry has become in-window — a slide must never resurrect a
    /// stale position, which would produce an offset past the window and an invalid
    /// stream.
    #[test]
    fn a_rebase_never_resurrects_a_stale_position() {
        let mut data: Vec<MfPos> = (i16::MIN..=i16::MAX).collect();
        matchfinder_rebase(&mut data);
        for &e in &data {
            assert!(e <= 0 || e < MATCHFINDER_WINDOW_SIZE as i16);
        }
        // Everything that was negative stays at the floor.
        for (i, &e) in data.iter().enumerate() {
            let was = i16::MIN.wrapping_add(i as i16);
            if was < 0 {
                assert_eq!(e, i16::MIN);
            }
        }
    }

    /// `lz_hash` must wrap, and must use the high bits of the product. A
    /// non-wrapping multiply panics in debug; taking the LOW bits instead of the high
    /// ones compiles and produces a much worse hash.
    #[test]
    fn lz_hash_wraps_and_takes_the_high_bits() {
        // A value whose product overflows u32 many times over.
        let h = lz_hash(0xFFFF_FFFF, 15);
        assert!(h < (1 << 15));

        // The shift takes bits 17..32 of the wrapped product.
        for seq in [0u32, 1, 0x1234_5678, 0xDEAD_BEEF, u32::MAX] {
            let want = seq.wrapping_mul(0x1E35_A7BD) >> 17;
            assert_eq!(lz_hash(seq, 15), want, "seq={seq:#x}");
        }
    }

    /// `lz_extend` must agree with a naive byte-by-byte comparison — over the word
    /// path, the tail path, and every boundary between them.
    #[test]
    fn lz_extend_matches_a_naive_byte_comparison() {
        let naive = |buf: &[u8], s: usize, m: usize, start: u32, max: u32| -> u32 {
            let mut len = start;
            while len < max && buf[m + len as usize] == buf[s + len as usize] {
                len += 1;
            }
            len
        };

        // A buffer where a match runs for a controlled number of bytes.
        for run in 0..80usize {
            let mut buf = vec![0u8; 600];
            // matchptr region at 0, strptr region at 256.
            for i in 0..run {
                buf[i] = (i % 7) as u8;
                buf[256 + i] = (i % 7) as u8;
            }
            buf[run] = 0xAA;
            buf[256 + run] = 0xBB;

            for max in [4u32, 8, 16, 17, 33, 64, 100, 258] {
                let start = 0u32;
                assert_eq!(
                    lz_extend(&buf, 256, 0, start, max),
                    naive(&buf, 256, 0, start, max),
                    "run={run} max={max}"
                );
            }
        }
    }

    /// The `start_len` argument means "these bytes are already known to match" —
    /// `lz_extend` must not re-verify them, and must not go below `start_len` even
    /// when they DON'T match. The ht matchfinder relies on this: it passes 4 after
    /// checking a 4-byte sequence equality.
    #[test]
    fn lz_extend_trusts_start_len() {
        let mut buf = vec![0u8; 400];
        // Deliberately make the first 4 bytes differ.
        buf[0..4].copy_from_slice(&[1, 2, 3, 4]);
        buf[64..68].copy_from_slice(&[9, 9, 9, 9]);
        // ...but bytes 4..20 agree.
        for i in 4..20 {
            buf[i] = 0x55;
            buf[64 + i] = 0x55;
        }
        buf[20] = 1;
        buf[84] = 2;

        assert_eq!(lz_extend(&buf, 64, 0, 4, 258), 20);
    }
}
