//! DEFLATE length/distance symbol and extra-bit tables.
//! Port of vendor/zopfli/src/zopfli/symbols.h

// These constants and functions are used by modules unlocked in later steps.
#![allow(dead_code)]

pub const ZOPFLI_NUM_LL: usize = 288;
pub const ZOPFLI_NUM_D: usize = 32;
pub const ZOPFLI_MAX_MATCH: usize = 258;
pub const ZOPFLI_MIN_MATCH: usize = 3;
pub const ZOPFLI_WINDOW_SIZE: usize = 32_768;
pub const ZOPFLI_WINDOW_MASK: usize = ZOPFLI_WINDOW_SIZE - 1;
pub const ZOPFLI_MASTER_BLOCK_SIZE: usize = 1_000_000;
pub const ZOPFLI_LARGE_FLOAT: f64 = 1e30;
pub const ZOPFLI_CACHE_LENGTH: usize = 8;
pub const ZOPFLI_MAX_CHAIN_HITS: i32 = 8192;

/// Extra bits for the given distance (DEFLATE spec).
pub fn dist_extra_bits(dist: i32) -> i32 {
    if dist < 5 {
        return 0;
    }
    // log2(dist - 1) - 1
    let l = 31 ^ (dist - 1).leading_zeros() as i32;
    l - 1
}

/// Extra bits value for the given distance (DEFLATE spec).
pub fn dist_extra_bits_value(dist: i32) -> i32 {
    if dist < 5 {
        return 0;
    }
    let l = 31 ^ (dist - 1).leading_zeros() as i32; // log2(dist - 1)
    (dist - (1 + (1 << l))) & ((1 << (l - 1)) - 1)
}

/// Distance symbol for the given distance (DEFLATE spec).
pub fn dist_symbol(dist: i32) -> i32 {
    if dist < 5 {
        return dist - 1;
    }
    let l = 31 ^ (dist - 1).leading_zeros() as i32; // log2(dist - 1)
    let r = ((dist - 1) >> (l - 1)) & 1;
    l * 2 + r
}

/// Extra bits for the given length (DEFLATE spec).
pub fn length_extra_bits(l: i32) -> i32 {
    const TABLE: [i32; 259] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0,
    ];
    TABLE[l as usize]
}

/// Extra bits value for the given length (DEFLATE spec).
pub fn length_extra_bits_value(l: i32) -> i32 {
    const TABLE: [i32; 259] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2,
        3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0,
        1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4,
        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1,
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0,
    ];
    TABLE[l as usize]
}

/// Length symbol for the given length (DEFLATE spec). Returns symbol in [257, 285].
pub fn length_symbol(l: i32) -> i32 {
    const TABLE: [i32; 259] = [
        0, 0, 0, 257, 258, 259, 260, 261, 262, 263, 264, 265, 265, 266, 266, 267, 267, 268, 268,
        269, 269, 269, 269, 270, 270, 270, 270, 271, 271, 271, 271, 272, 272, 272, 272, 273, 273,
        273, 273, 273, 273, 273, 273, 274, 274, 274, 274, 274, 274, 274, 274, 275, 275, 275, 275,
        275, 275, 275, 275, 276, 276, 276, 276, 276, 276, 276, 276, 277, 277, 277, 277, 277, 277,
        277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 278, 278, 278, 278, 278, 278, 278, 278,
        278, 278, 278, 278, 278, 278, 278, 278, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279,
        279, 279, 279, 279, 279, 279, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280,
        280, 280, 280, 280, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281,
        281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281,
        282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282,
        282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 283, 283, 283, 283,
        283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283,
        283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 284, 284, 284, 284, 284, 284, 284, 284,
        284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284,
        284, 284, 284, 284, 284, 285,
    ];
    TABLE[l as usize]
}

/// Extra bits for the given length symbol. Indexed by `s - 257`.
pub fn length_symbol_extra_bits(s: i32) -> i32 {
    const TABLE: [i32; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    TABLE[(s - 257) as usize]
}

/// Extra bits for the given distance symbol.
pub fn dist_symbol_extra_bits(s: i32) -> i32 {
    const TABLE: [i32; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];
    TABLE[s as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbols_match_ffi() {
        assert_eq!(length_symbol(3), 257);
        assert_eq!(length_symbol(258), 285);
        assert_eq!(length_symbol(11), 265);
        assert_eq!(dist_symbol(1), 0);
        assert_eq!(dist_symbol(4), 3);
        assert_eq!(dist_symbol(5), 4);
        assert_eq!(dist_symbol(32_768), 29);
        assert_eq!(length_extra_bits(3), 0);
        assert_eq!(length_extra_bits(11), 1);
        assert_eq!(length_extra_bits(258), 0);
        assert_eq!(dist_extra_bits(1), 0);
        assert_eq!(dist_extra_bits(5), 1);
        assert_eq!(dist_extra_bits(32_768), 13);
        for d in 1..=32_768 {
            assert!(dist_extra_bits(d) + d.count_ones() as i32 >= 0);
        }
        for l in 3..=258 {
            let s = length_symbol(l);
            assert!((257..=285).contains(&s));
            assert_eq!(length_symbol_extra_bits(s), length_extra_bits(l));
        }
    }
}
