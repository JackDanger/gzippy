//! Bit-length / symbol conversion and entropy calculation.
//! Port of Google Zopfli tree.c

use super::katajainen::length_limited_code_lengths;

pub fn calculate_bit_lengths(count: &[usize], maxbits: i32, bitlengths: &mut [u32]) {
    length_limited_code_lengths(count, maxbits, bitlengths)
        .expect("ZopfliLengthLimitedCodeLengths failed");
}

pub fn lengths_to_symbols(lengths: &[u32], maxbits: u32, symbols: &mut [u32]) {
    let n = lengths.len();
    for s in symbols.iter_mut() {
        *s = 0;
    }

    let mut bl_count = vec![0usize; maxbits as usize + 1];
    let mut next_code = vec![0usize; maxbits as usize + 1];

    for &len in lengths.iter() {
        bl_count[len as usize] += 1;
    }

    let mut code: usize = 0;
    bl_count[0] = 0;
    for bits in 1..=maxbits as usize {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    for i in 0..n {
        let len = lengths[i] as usize;
        if len != 0 {
            symbols[i] = next_code[len] as u32;
            next_code[len] += 1;
        }
    }
}

// Must match C exactly — uses ln * kInvLog2, NOT log2().
// The C code uses the truncated constant 1.4426950408889 (not LOG2_E =
// 1.4426950408889634), which produces slightly different results.
#[allow(clippy::approx_constant)]
const K_INV_LOG2: f64 = 1.4426950408889;

pub fn calculate_entropy(count: &[usize], bitlengths: &mut [f64]) {
    let n = count.len();
    let sum: u32 = count.iter().map(|&c| c as u32).sum();
    let log2sum = (if sum == 0 {
        (n as f64).ln()
    } else {
        (sum as f64).ln()
    }) * K_INV_LOG2;

    for i in 0..n {
        if count[i] == 0 {
            bitlengths[i] = log2sum;
        } else {
            bitlengths[i] = log2sum - (count[i] as f64).ln() * K_INV_LOG2;
        }
        if bitlengths[i] < 0.0 && bitlengths[i] > -1e-5 {
            bitlengths[i] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_to_symbols_basic() {
        // 4-symbol code with lengths [2, 2, 3, 3] → symbols [0, 1, 4, 5]
        let lengths = [2u32, 2, 3, 3];
        let mut symbols = [0u32; 4];
        lengths_to_symbols(&lengths, 3, &mut symbols);
        // Canonical Huffman: 00, 01, 100, 101
        assert_eq!(symbols[0], 0b00); // 0
        assert_eq!(symbols[1], 0b01); // 1
        assert_eq!(symbols[2], 0b100); // 4
        assert_eq!(symbols[3], 0b101); // 5
    }

    #[test]
    fn entropy_nonnegative() {
        let counts = vec![1usize, 2, 3, 4, 5, 6, 7, 8];
        let mut bl = vec![0.0f64; 8];
        calculate_entropy(&counts, &mut bl);
        for &b in &bl {
            assert!(b >= 0.0, "negative entropy {}", b);
        }
    }

    #[test]
    fn entropy_uniform_is_log2n() {
        // For n uniform symbols the entropy per symbol is log2(n).
        let n = 8;
        let counts = vec![1usize; n];
        let mut bl = vec![0.0f64; n];
        calculate_entropy(&counts, &mut bl);
        let expected = (n as f64).log2();
        for &b in &bl {
            assert!(
                (b - expected).abs() < 1e-9,
                "expected {} got {}",
                expected,
                b
            );
        }
    }
}
