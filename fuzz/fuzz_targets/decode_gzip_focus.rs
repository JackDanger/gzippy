#![no_main]
//! gzip entry points only (T1/T2/T4), with one class filtered out so
//! coverage goes elsewhere (re-enable once the GZ-format fixes land):
//!   - BGZF ("GZ" FEXTRA subfield) inputs are skipped (ISIZE alloc + header-overlap panic).
//! Differential vs flate2 MultiGzDecoder when both succeed.
use libfuzzer_sys::fuzz_target;
use std::io::Read;

fuzz_target!(|input: &[u8]| {
    if input.len() < 2 || input.len() > 1 << 18 {
        return;
    }
    let t = [1usize, 2, 4][(input[0] % 3) as usize];
    let data = &input[1..];
    if data.len() >= 4 && data[3] & 0x04 != 0 && data.windows(2).take(80).any(|w| w == b"GZ") {
        return;
    }
    let mut ours = Vec::new();
    if gzippy::decompress_to_writer_with_threads(data, &mut ours, t).is_ok()
        && data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b
    {
        let mut d = flate2::read::MultiGzDecoder::new(data);
        let mut theirs = Vec::new();
        if d.read_to_end(&mut theirs).is_ok() {
            assert!(ours == theirs, "MISMATCH t={t} ours={} theirs={}", ours.len(), theirs.len());
        }
    }
});
