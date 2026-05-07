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
