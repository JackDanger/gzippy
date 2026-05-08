//! Oracle tests: compare pure-Rust port against the C FFI.
//! Deleted in cutover step.
#![cfg(test)]

#[allow(dead_code)]
pub fn corpus() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("empty", vec![]),
        ("byte", b"x".to_vec()),
        ("ascii", b"hello world hello world hello world".to_vec()),
        ("zeros_1k", vec![0u8; 1024]),
        ("zeros_64k", vec![0u8; 65_536]),
        ("rand_4k", {
            let mut v = Vec::with_capacity(4096);
            let mut s: u32 = 0x12345678;
            for _ in 0..4096 {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                v.push((s >> 16) as u8);
            }
            v
        }),
        (
            "alice",
            include_bytes!("../../../test_data/alice.txt").to_vec(),
        ),
    ]
}

// ── Step 2: katajainen oracle ────────────────────────────────────────────────

extern "C" {
    fn ZopfliLengthLimitedCodeLengths(
        frequencies: *const usize,
        n: i32,
        maxbits: i32,
        bitlengths: *mut u32,
    ) -> i32;
}

fn ffi_lengths(freq: &[usize], maxbits: i32) -> Result<Vec<u32>, ()> {
    let mut bl = vec![0u32; freq.len()];
    let r = unsafe {
        ZopfliLengthLimitedCodeLengths(freq.as_ptr(), freq.len() as i32, maxbits, bl.as_mut_ptr())
    };
    if r == 0 {
        Ok(bl)
    } else {
        Err(())
    }
}

#[test]
fn katajainen_matches_ffi() {
    use crate::backends::zopfli_pure::katajainen::length_limited_code_lengths;
    let cases: Vec<Vec<usize>> = vec![
        vec![1, 1, 2, 3, 5, 8, 13, 21],
        vec![0, 0, 1, 0, 0, 2, 0, 5, 0, 9, 1],
        (0..286usize).map(|i| i * 7 % 17).collect(),
        vec![1; 32], // ZOPFLI_NUM_D
        vec![100, 100, 100, 100],
        vec![0, 0, 0, 5], // <2 used symbols path
        vec![0, 0, 0, 0],
        vec![42],
    ];
    for freq in &cases {
        for &mb in &[15i32, 7] {
            let mut got = vec![0u32; freq.len()];
            let rs_result = length_limited_code_lengths(freq, mb, &mut got);
            let ffi_result = ffi_lengths(freq, mb);
            match (rs_result, ffi_result) {
                (Ok(()), Ok(exp)) => {
                    assert_eq!(got, exp, "freq={:?} maxbits={}", freq, mb);
                }
                (Err(()), Err(())) => {} // both agree it's an error
                (Ok(()), Err(())) => {
                    panic!(
                        "Rust succeeded but FFI failed: freq={:?} maxbits={}",
                        freq, mb
                    )
                }
                (Err(()), Ok(exp)) => {
                    panic!(
                        "Rust failed but FFI succeeded with {:?}: freq={:?} maxbits={}",
                        exp, freq, mb
                    )
                }
            }
        }
    }
}
