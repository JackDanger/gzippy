//! Cost model for the near-optimal parser.
//!
//! Faithful transliteration of the cost machinery in libdeflate
//! `vendor/libdeflate/lib/deflate_compress.c`: the BIT_COST-scaled
//! [`DeflateCosts`] (`:385-395`), the [`OptimumNode`] graph node (`:417-434`),
//! `deflate_set_costs_from_codes` (`:2924-2957`), the `default_litlen_costs`
//! table + `deflate_choose_default_litlen_costs` (`:2986-3160`),
//! `deflate_set_default_costs` / `deflate_adjust_costs` / `deflate_set_initial_costs`
//! (`:3185-3311`), and the dense `offset_slot_full` map
//! (`deflate_init_offset_slot_full`, `:3851-3868`).
//!
//! BIT_COST scales bit costs by 16 so the search can weigh fractional-bit
//! symbol costs. The map / table / weighting are ported verbatim; the strict
//! ratio assertion in the encoder test net is the guard that a subtle cost error
//! (wrong NOSTAT scaling, off-by-one in the length loop, wrong slot map) has not
//! silently inflated the output.

use super::block_split::NUM_OBSERVATION_TYPES;
use super::tables::{
    length_slot, DEFLATE_MAX_MATCH_LEN, DEFLATE_MIN_MATCH_LEN, DEFLATE_NUM_OFFSET_SYMS,
    LENGTH_EXTRA_BITS, OFFSET_EXTRA_BITS, OFFSET_SLOT_BASE,
};

/// Cost scaling factor (`BIT_COST`, `:140`). A symbol with `k` bits of entropy
/// costs `k * BIT_COST`, so fractional bits can be represented.
pub const BIT_COST: u32 = 16;

/// Assumed bit cost of a literal / length / offset symbol that was unused in the
/// previous pass (`:149-151`). Keeps such symbols usable in the next pass.
pub const LITERAL_NOSTAT_BITS: u32 = 13;
pub const LENGTH_NOSTAT_BITS: u32 = 13;
pub const OFFSET_NOSTAT_BITS: u32 = 10;

/// `OPTIMUM_OFFSET_SHIFT` / `OPTIMUM_LEN_MASK` (`:430-431`): `item` packs the
/// length (or 1 for a literal) in the low bits and the offset (or literal byte)
/// in the high bits.
pub const OPTIMUM_OFFSET_SHIFT: u32 = 9;
pub const OPTIMUM_LEN_MASK: u32 = (1 << OPTIMUM_OFFSET_SHIFT) - 1;

const NUM_LITERALS: usize = 256;
const MAX_MATCH_LEN: usize = DEFLATE_MAX_MATCH_LEN as usize;
const MIN_MATCH_LEN: usize = DEFLATE_MIN_MATCH_LEN as usize;
/// Number of REAL offset slots (`ARRAY_LEN(deflate_offset_slot_base)` = 30). The
/// litlen/offset alphabets reserve 32 offset symbols but only 30 are ever used.
const NUM_OFFSET_SLOTS: usize = OFFSET_SLOT_BASE.len();
/// `DEFLATE_MAX_MATCH_OFFSET` = 32768; the dense offset-slot map has one entry
/// per possible offset (index 0 unused).
pub const MAX_MATCH_OFFSET: usize = 32768;

/// The BIT_COST-scaled cost model (`struct deflate_costs`).
#[derive(Clone)]
pub struct DeflateCosts {
    /// Cost of each literal byte.
    pub literal: [u32; NUM_LITERALS],
    /// Cost of each match length (indexed by length, `3..=258`).
    pub length: [u32; MAX_MATCH_LEN + 1],
    /// Cost of a match offset of each offset slot.
    pub offset_slot: [u32; DEFLATE_NUM_OFFSET_SYMS],
}

impl Default for DeflateCosts {
    fn default() -> Self {
        DeflateCosts {
            literal: [0; NUM_LITERALS],
            length: [0; MAX_MATCH_LEN + 1],
            offset_slot: [0; DEFLATE_NUM_OFFSET_SYMS],
        }
    }
}

/// A node in the min-cost-path graph (`struct deflate_optimum_node`).
#[derive(Clone, Copy, Default)]
pub struct OptimumNode {
    /// Minimum cost (BIT_COST-scaled) to reach the end of the block from here.
    pub cost_to_end: u32,
    /// The literal/match chosen from here on the min-cost path. Low
    /// `OPTIMUM_OFFSET_SHIFT` bits = length (1 for a literal); high bits = offset
    /// (or the literal byte).
    pub item: u32,
}

/// The dense offset → offset-slot map (`offset_slot_full`), one entry per offset
/// value `1..=32768` (`deflate_init_offset_slot_full`).
pub struct OffsetSlotFull {
    map: Box<[u8]>,
}

impl OffsetSlotFull {
    pub fn new() -> Self {
        let mut map = vec![0u8; MAX_MATCH_OFFSET + 1].into_boxed_slice();
        for slot in 0..NUM_OFFSET_SLOTS {
            let base = OFFSET_SLOT_BASE[slot];
            let end = base + (1 << OFFSET_EXTRA_BITS[slot]);
            for off in base..end {
                map[off as usize] = slot as u8;
            }
        }
        OffsetSlotFull { map }
    }

    #[inline]
    pub fn slot(&self, offset: u32) -> usize {
        self.map[offset as usize] as usize
    }

    /// Unchecked twin of [`Self::slot`] for the near-optimal DP hot loop
    /// (`find_min_cost_path`), which has already proven `offset` is a valid
    /// DEFLATE match offset (`1..=MAX_MATCH_OFFSET`) before calling this.
    ///
    /// # Safety
    /// `offset as usize` must be `< self.map.len()` (`== MAX_MATCH_OFFSET + 1`),
    /// i.e. `offset <= MAX_MATCH_OFFSET`.
    #[inline]
    pub unsafe fn slot_unchecked(&self, offset: u32) -> usize {
        debug_assert!(
            (offset as usize) < self.map.len(),
            "offset {offset} out of range for offset_slot_full (len {})",
            self.map.len()
        );
        *self.map.get_unchecked(offset as usize) as usize
    }
}

impl Default for OffsetSlotFull {
    fn default() -> Self {
        Self::new()
    }
}

impl DeflateCosts {
    /// `deflate_set_costs_from_codes`: derive the cost model from built codeword
    /// lengths (`litlen_lens[288]`, `offset_lens[32]`).
    pub fn set_from_codes(&mut self, litlen_lens: &[u8], offset_lens: &[u8]) {
        // Literals.
        for i in 0..NUM_LITERALS {
            let bits = if litlen_lens[i] != 0 {
                litlen_lens[i] as u32
            } else {
                LITERAL_NOSTAT_BITS
            };
            self.literal[i] = bits * BIT_COST;
        }

        // Lengths.
        for len in MIN_MATCH_LEN..=MAX_MATCH_LEN {
            let slot = length_slot(len as u32) as usize;
            let litlen_sym = super::tables::DEFLATE_FIRST_LEN_SYM + slot;
            let mut bits = if litlen_lens[litlen_sym] != 0 {
                litlen_lens[litlen_sym] as u32
            } else {
                LENGTH_NOSTAT_BITS
            };
            bits += LENGTH_EXTRA_BITS[slot] as u32;
            self.length[len] = bits * BIT_COST;
        }

        // Offset slots.
        for i in 0..NUM_OFFSET_SLOTS {
            let mut bits = if offset_lens[i] != 0 {
                offset_lens[i] as u32
            } else {
                OFFSET_NOSTAT_BITS
            };
            bits += OFFSET_EXTRA_BITS[i] as u32;
            self.offset_slot[i] = bits * BIT_COST;
        }
    }

    /// `deflate_set_default_costs`.
    pub fn set_default(&mut self, lit_cost: u32, len_sym_cost: u32) {
        for i in 0..NUM_LITERALS {
            self.literal[i] = lit_cost;
        }
        for len in MIN_MATCH_LEN..=MAX_MATCH_LEN {
            self.length[len] = default_length_cost(len as u32, len_sym_cost);
        }
        for slot in 0..NUM_OFFSET_SLOTS {
            self.offset_slot[slot] = default_offset_slot_cost(slot);
        }
    }

    /// `deflate_adjust_costs_impl`: blend current costs toward the default costs
    /// with a similarity-derived `change_amount` (0..=3).
    fn adjust_impl(&mut self, lit_cost: u32, len_sym_cost: u32, change_amount: i32) {
        for i in 0..NUM_LITERALS {
            adjust_cost(&mut self.literal[i], lit_cost, change_amount);
        }
        for len in MIN_MATCH_LEN..=MAX_MATCH_LEN {
            let def = default_length_cost(len as u32, len_sym_cost);
            adjust_cost(&mut self.length[len], def, change_amount);
        }
        for slot in 0..NUM_OFFSET_SLOTS {
            let def = default_offset_slot_cost(slot);
            adjust_cost(&mut self.offset_slot[slot], def, change_amount);
        }
    }
}

/// `deflate_adjust_cost` (`:3207-3220`).
#[inline]
fn adjust_cost(cost_p: &mut u32, default_cost: u32, change_amount: i32) {
    let c = *cost_p;
    *cost_p = match change_amount {
        0 => (default_cost + 3 * c) / 4,
        1 => (default_cost + c) / 2,
        2 => (5 * default_cost + 3 * c) / 8,
        _ => (3 * default_cost + c) / 4,
    };
}

/// `deflate_default_length_cost`.
#[inline]
fn default_length_cost(len: u32, len_sym_cost: u32) -> u32 {
    let slot = length_slot(len) as usize;
    let num_extra_bits = LENGTH_EXTRA_BITS[slot] as u32;
    len_sym_cost + num_extra_bits * BIT_COST
}

/// `deflate_default_offset_slot_cost`. All offset symbols assumed equiprobable
/// over ~30 slots: `int(-log2(1/30) * BIT_COST)` = `4*BIT_COST + 907*BIT_COST/1000`.
#[inline]
fn default_offset_slot_cost(slot: usize) -> u32 {
    let num_extra_bits = OFFSET_EXTRA_BITS[slot] as u32;
    let offset_sym_cost = 4 * BIT_COST + (907 * BIT_COST) / 1000;
    offset_sym_cost + num_extra_bits * BIT_COST
}

/// `deflate_choose_default_litlen_costs`: pick `(lit_cost, len_sym_cost)` from
/// the data's literal diversity and greedy-parse match frequency.
///
/// `block` is the block's bytes; `match_len_freqs[len]` is the approximate
/// greedy match-length histogram gathered during matchfinding.
pub fn choose_default_litlen_costs(
    block: &[u8],
    match_len_freqs: &[u32],
    max_search_depth: u32,
) -> (u32, u32) {
    let block_length = block.len() as u32;
    let mut freqs = [0u32; NUM_LITERALS];
    let cutoff = block_length >> 11; // ignore very rare literals
    for &b in block {
        freqs[b as usize] += 1;
    }
    let mut num_used_literals = 0u32;
    for &f in freqs.iter() {
        if f > cutoff {
            num_used_literals += 1;
        }
    }
    if num_used_literals == 0 {
        num_used_literals = 1;
    }

    // Estimate literal/match balance using the greedy match histogram with the
    // min_len heuristic applied.
    let mut literal_freq = block_length as i64;
    let mut match_freq = 0i64;
    let start = super::parse::choose_min_match_len(num_used_literals, max_search_depth) as usize;
    for len in start..match_len_freqs.len() {
        match_freq += match_len_freqs[len] as i64;
        literal_freq -= (len as i64) * (match_len_freqs[len] as i64);
    }
    if literal_freq < 0 {
        literal_freq = 0;
    }

    let i = if match_freq > literal_freq {
        2 // many matches
    } else if match_freq * 4 > literal_freq {
        1 // neutral
    } else {
        0 // few matches
    };

    let lit_cost = DEFAULT_LITLEN_COSTS[i].used_lits_to_lit_cost[num_used_literals as usize] as u32;
    let len_sym_cost = DEFAULT_LITLEN_COSTS[i].len_sym_cost as u32;
    (lit_cost, len_sym_cost)
}

/// `deflate_set_initial_costs`: default costs for the first block, else blend
/// with the previous block's costs by similarity.
#[allow(clippy::too_many_arguments)]
pub fn set_initial_costs(
    costs: &mut DeflateCosts,
    block: &[u8],
    match_len_freqs: &[u32],
    max_search_depth: u32,
    is_first_block: bool,
    // cross-block similarity inputs for adjust_costs:
    cur_observations: &[u32; NUM_OBSERVATION_TYPES],
    cur_num_observations: u32,
    prev_observations: &[u32; NUM_OBSERVATION_TYPES],
    prev_num_observations: u32,
) {
    let (lit_cost, len_sym_cost) =
        choose_default_litlen_costs(block, match_len_freqs, max_search_depth);
    if is_first_block {
        costs.set_default(lit_cost, len_sym_cost);
    } else {
        adjust_costs(
            costs,
            lit_cost,
            len_sym_cost,
            cur_observations,
            cur_num_observations,
            prev_observations,
            prev_num_observations,
        );
    }
}

/// `deflate_adjust_costs`: decide how much the current block differs from the
/// previous one (SAD of observation distributions) and mix default vs current.
#[allow(clippy::too_many_arguments)]
fn adjust_costs(
    costs: &mut DeflateCosts,
    lit_cost: u32,
    len_sym_cost: u32,
    cur_observations: &[u32; NUM_OBSERVATION_TYPES],
    cur_num_observations: u32,
    prev_observations: &[u32; NUM_OBSERVATION_TYPES],
    prev_num_observations: u32,
) {
    let mut total_delta: u64 = 0;
    for i in 0..NUM_OBSERVATION_TYPES {
        let prev = prev_observations[i] as u64 * cur_num_observations as u64;
        let cur = cur_observations[i] as u64 * prev_num_observations as u64;
        total_delta += prev.abs_diff(cur);
    }
    let cutoff = (prev_num_observations as u64 * cur_num_observations as u64 * 200) / 512;

    if total_delta > 3 * cutoff {
        costs.set_default(lit_cost, len_sym_cost);
    } else if 4 * total_delta > 9 * cutoff {
        costs.adjust_impl(lit_cost, len_sym_cost, 3);
    } else if 2 * total_delta > 3 * cutoff {
        costs.adjust_impl(lit_cost, len_sym_cost, 2);
    } else if 2 * total_delta > cutoff {
        costs.adjust_impl(lit_cost, len_sym_cost, 1);
    } else {
        costs.adjust_impl(lit_cost, len_sym_cost, 0);
    }
}

/// The `default_litlen_costs[]` lookup table, copied verbatim from
/// `deflate_compress.c:2986-3102` (generated by
/// `scripts/gen_default_litlen_costs.py`). Indexed by estimated match
/// probability (0 = few matches / 0.25, 1 = neutral / 0.5, 2 = many / 0.75).
struct DefaultLitlenCosts {
    used_lits_to_lit_cost: [u8; 257],
    len_sym_cost: u8,
}

#[rustfmt::skip]
static DEFAULT_LITLEN_COSTS: [DefaultLitlenCosts; 3] = [
    DefaultLitlenCosts { // match_prob = 0.25
        used_lits_to_lit_cost: [
            6, 6, 22, 32, 38, 43, 48, 51,
            54, 57, 59, 61, 64, 65, 67, 69,
            70, 72, 73, 74, 75, 76, 77, 79,
            80, 80, 81, 82, 83, 84, 85, 85,
            86, 87, 88, 88, 89, 89, 90, 91,
            91, 92, 92, 93, 93, 94, 95, 95,
            96, 96, 96, 97, 97, 98, 98, 99,
            99, 99, 100, 100, 101, 101, 101, 102,
            102, 102, 103, 103, 104, 104, 104, 105,
            105, 105, 105, 106, 106, 106, 107, 107,
            107, 108, 108, 108, 108, 109, 109, 109,
            109, 110, 110, 110, 111, 111, 111, 111,
            112, 112, 112, 112, 112, 113, 113, 113,
            113, 114, 114, 114, 114, 114, 115, 115,
            115, 115, 115, 116, 116, 116, 116, 116,
            117, 117, 117, 117, 117, 118, 118, 118,
            118, 118, 118, 119, 119, 119, 119, 119,
            120, 120, 120, 120, 120, 120, 121, 121,
            121, 121, 121, 121, 121, 122, 122, 122,
            122, 122, 122, 123, 123, 123, 123, 123,
            123, 123, 124, 124, 124, 124, 124, 124,
            124, 125, 125, 125, 125, 125, 125, 125,
            125, 126, 126, 126, 126, 126, 126, 126,
            127, 127, 127, 127, 127, 127, 127, 127,
            128, 128, 128, 128, 128, 128, 128, 128,
            128, 129, 129, 129, 129, 129, 129, 129,
            129, 129, 130, 130, 130, 130, 130, 130,
            130, 130, 130, 131, 131, 131, 131, 131,
            131, 131, 131, 131, 131, 132, 132, 132,
            132, 132, 132, 132, 132, 132, 132, 133,
            133, 133, 133, 133, 133, 133, 133, 133,
            133, 134, 134, 134, 134, 134, 134, 134,
            134,
        ],
        len_sym_cost: 109,
    },
    DefaultLitlenCosts { // match_prob = 0.5
        used_lits_to_lit_cost: [
            16, 16, 32, 41, 48, 53, 57, 60,
            64, 66, 69, 71, 73, 75, 76, 78,
            80, 81, 82, 83, 85, 86, 87, 88,
            89, 90, 91, 92, 92, 93, 94, 95,
            96, 96, 97, 98, 98, 99, 99, 100,
            101, 101, 102, 102, 103, 103, 104, 104,
            105, 105, 106, 106, 107, 107, 108, 108,
            108, 109, 109, 110, 110, 110, 111, 111,
            112, 112, 112, 113, 113, 113, 114, 114,
            114, 115, 115, 115, 115, 116, 116, 116,
            117, 117, 117, 118, 118, 118, 118, 119,
            119, 119, 119, 120, 120, 120, 120, 121,
            121, 121, 121, 122, 122, 122, 122, 122,
            123, 123, 123, 123, 124, 124, 124, 124,
            124, 125, 125, 125, 125, 125, 126, 126,
            126, 126, 126, 127, 127, 127, 127, 127,
            128, 128, 128, 128, 128, 128, 129, 129,
            129, 129, 129, 129, 130, 130, 130, 130,
            130, 130, 131, 131, 131, 131, 131, 131,
            131, 132, 132, 132, 132, 132, 132, 133,
            133, 133, 133, 133, 133, 133, 134, 134,
            134, 134, 134, 134, 134, 134, 135, 135,
            135, 135, 135, 135, 135, 135, 136, 136,
            136, 136, 136, 136, 136, 136, 137, 137,
            137, 137, 137, 137, 137, 137, 138, 138,
            138, 138, 138, 138, 138, 138, 138, 139,
            139, 139, 139, 139, 139, 139, 139, 139,
            140, 140, 140, 140, 140, 140, 140, 140,
            140, 141, 141, 141, 141, 141, 141, 141,
            141, 141, 141, 142, 142, 142, 142, 142,
            142, 142, 142, 142, 142, 142, 143, 143,
            143, 143, 143, 143, 143, 143, 143, 143,
            144,
        ],
        len_sym_cost: 93,
    },
    DefaultLitlenCosts { // match_prob = 0.75
        used_lits_to_lit_cost: [
            32, 32, 48, 57, 64, 69, 73, 76,
            80, 82, 85, 87, 89, 91, 92, 94,
            96, 97, 98, 99, 101, 102, 103, 104,
            105, 106, 107, 108, 108, 109, 110, 111,
            112, 112, 113, 114, 114, 115, 115, 116,
            117, 117, 118, 118, 119, 119, 120, 120,
            121, 121, 122, 122, 123, 123, 124, 124,
            124, 125, 125, 126, 126, 126, 127, 127,
            128, 128, 128, 129, 129, 129, 130, 130,
            130, 131, 131, 131, 131, 132, 132, 132,
            133, 133, 133, 134, 134, 134, 134, 135,
            135, 135, 135, 136, 136, 136, 136, 137,
            137, 137, 137, 138, 138, 138, 138, 138,
            139, 139, 139, 139, 140, 140, 140, 140,
            140, 141, 141, 141, 141, 141, 142, 142,
            142, 142, 142, 143, 143, 143, 143, 143,
            144, 144, 144, 144, 144, 144, 145, 145,
            145, 145, 145, 145, 146, 146, 146, 146,
            146, 146, 147, 147, 147, 147, 147, 147,
            147, 148, 148, 148, 148, 148, 148, 149,
            149, 149, 149, 149, 149, 149, 150, 150,
            150, 150, 150, 150, 150, 150, 151, 151,
            151, 151, 151, 151, 151, 151, 152, 152,
            152, 152, 152, 152, 152, 152, 153, 153,
            153, 153, 153, 153, 153, 153, 154, 154,
            154, 154, 154, 154, 154, 154, 154, 155,
            155, 155, 155, 155, 155, 155, 155, 155,
            156, 156, 156, 156, 156, 156, 156, 156,
            156, 157, 157, 157, 157, 157, 157, 157,
            157, 157, 157, 158, 158, 158, 158, 158,
            158, 158, 158, 158, 158, 158, 159, 159,
            159, 159, 159, 159, 159, 159, 159, 159,
            160,
        ],
        len_sym_cost: 84,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::deflate::huffman::make_huffman_code;
    use crate::compress::deflate::tables::{
        offset_slot as offset_slot_fn, static_litlen_freqs, static_offset_freqs,
        DEFLATE_NUM_LITLEN_SYMS, MAX_LITLEN_CODEWORD_LEN, MAX_OFFSET_CODEWORD_LEN,
    };

    #[test]
    fn offset_slot_full_matches_condensed_map() {
        let full = OffsetSlotFull::new();
        for off in 1..=32768u32 {
            assert_eq!(
                full.slot(off),
                offset_slot_fn(off) as usize,
                "offset {off}: dense map disagrees with condensed offset_slot()"
            );
        }
    }

    #[test]
    fn default_table_dimensions() {
        assert_eq!(DEFAULT_LITLEN_COSTS.len(), 3);
        for t in &DEFAULT_LITLEN_COSTS {
            assert_eq!(t.used_lits_to_lit_cost.len(), 257);
        }
        // Spot-check the documented endpoints of each subtable.
        assert_eq!(DEFAULT_LITLEN_COSTS[0].used_lits_to_lit_cost[0], 6);
        assert_eq!(DEFAULT_LITLEN_COSTS[0].used_lits_to_lit_cost[256], 134);
        assert_eq!(DEFAULT_LITLEN_COSTS[0].len_sym_cost, 109);
        assert_eq!(DEFAULT_LITLEN_COSTS[2].used_lits_to_lit_cost[256], 160);
        assert_eq!(DEFAULT_LITLEN_COSTS[2].len_sym_cost, 84);
    }

    /// Independent hand-rolled bit counter: given per-symbol codeword lengths,
    /// the exact whole-bit cost of a small literal/match token list, verified
    /// against `set_from_codes` producing consistent scaled costs.
    #[test]
    fn set_from_codes_scales_true_bit_lengths() {
        // Build the RFC static codes and derive costs from them.
        let litcode = make_huffman_code(
            DEFLATE_NUM_LITLEN_SYMS,
            MAX_LITLEN_CODEWORD_LEN,
            &static_litlen_freqs(),
        );
        let offcode = make_huffman_code(
            DEFLATE_NUM_OFFSET_SYMS,
            MAX_OFFSET_CODEWORD_LEN,
            &static_offset_freqs(),
        );
        let mut costs = DeflateCosts::default();
        costs.set_from_codes(&litcode.lens, &offcode.lens);

        // Static code: literals 0..=143 are 8 bits, 144..=255 are 9 bits.
        assert_eq!(costs.literal[0], 8 * BIT_COST);
        assert_eq!(costs.literal[200], 9 * BIT_COST);

        // A length-3 match: length slot 0 (litlen sym 257) has 7-bit static
        // codeword and 0 extra bits.
        let ls = length_slot(3) as usize;
        let sym = super::super::tables::DEFLATE_FIRST_LEN_SYM + ls;
        let expected_len_bits = litcode.lens[sym] as u32 + LENGTH_EXTRA_BITS[ls] as u32;
        assert_eq!(costs.length[3], expected_len_bits * BIT_COST);

        // Offset slot with extra bits: slot's cost = (codeword len + extra) * BIT_COST.
        let os = offset_slot_fn(1000) as usize;
        let expected_off_bits = offcode.lens[os] as u32 + OFFSET_EXTRA_BITS[os] as u32;
        assert_eq!(costs.offset_slot[os], expected_off_bits * BIT_COST);

        // A hand-rolled bit count for a fixed token list, computed two ways:
        // (a) directly from codeword lengths, (b) from the scaled cost model / BIT_COST.
        // tokens: literal 'A'(65), match len 3 off 1000, literal 'z'(122).
        let direct_bits = litcode.lens[65] as u32
            + (litcode.lens[sym] as u32 + LENGTH_EXTRA_BITS[ls] as u32)
            + (offcode.lens[os] as u32 + OFFSET_EXTRA_BITS[os] as u32)
            + litcode.lens[122] as u32;
        let model_bits =
            (costs.literal[65] + costs.length[3] + costs.offset_slot[os] + costs.literal[122])
                / BIT_COST;
        assert_eq!(direct_bits, model_bits);
    }

    #[test]
    fn nostat_costs_for_unused_symbols() {
        // A code where symbol 5 is unused (len 0) must get the NOSTAT cost.
        let mut lens = vec![0u8; DEFLATE_NUM_LITLEN_SYMS];
        lens[0] = 4;
        let offlens = vec![0u8; DEFLATE_NUM_OFFSET_SYMS];
        let mut costs = DeflateCosts::default();
        costs.set_from_codes(&lens, &offlens);
        assert_eq!(costs.literal[5], LITERAL_NOSTAT_BITS * BIT_COST);
        assert_eq!(costs.literal[0], 4 * BIT_COST);
        // Unused offset slot 0 => (OFFSET_NOSTAT_BITS + extra(0)) * BIT_COST.
        assert_eq!(costs.offset_slot[0], OFFSET_NOSTAT_BITS * BIT_COST);
    }
}
