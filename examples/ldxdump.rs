//! Dump raw DEFLATE produced by the `ldx` port: `ldxdump <level> < in > out`.
//!
//! Exists solely to run the port's rung-3 gate — a byte-for-byte differential against
//! libdeflate's own `libdeflate_deflate_compress`. Nothing ships from here.
fn main() {
    use std::io::{Read, Write};
    let level: u32 = std::env::args()
        .nth(1)
        .expect("usage: ldxdump <level>")
        .parse()
        .unwrap();
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input).unwrap();
    let out = gzippy::compress::ldx::compress_for_diff(level, &input)
        .unwrap_or_else(|| panic!("level {level} is not ported"));
    std::io::stdout().write_all(&out).unwrap();
}
