#![no_main]
//! Arbitrary bytes into every public decode entry point (from the 2026-11-21
//! decoder audit, which found two hangs and the OOM class with it).
//! Selector byte picks the arm so libFuzzer can specialise per arm.
//! Differential: when gzippy returns Ok, flate2's MultiGzDecoder must have
//! produced the same bytes (an Ok with different bytes = silently wrong output).
use libfuzzer_sys::fuzz_target;
use std::io::Read;

fn flate2_multi(data: &[u8]) -> (Vec<u8>, bool) {
    let mut d = flate2::read::MultiGzDecoder::new(data);
    let mut out = Vec::new();
    let ok = d.read_to_end(&mut out).is_ok();
    (out, ok)
}

fuzz_target!(|input: &[u8]| {
    if input.len() < 1 || input.len() > 1 << 20 {
        return;
    }
    let sel = input[0];
    let data = &input[1..];
    match sel % 6 {
        0 | 1 | 2 => {
            let t = [1usize, 2, 4][(sel % 3) as usize];
            let mut sink = Vec::new();
            let r = gzippy::decompress_to_writer_with_threads(data, &mut sink, t);
            if let Ok(_) = r {
                if data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b {
                    let (f, fok) = flate2_multi(data);
                    if fok {
                        assert!(sink == f, "MISMATCH vs flate2 t={t} ours={} theirs={}", sink.len(), f.len());
                    }
                }
            }
        }
        3 => {
            let _ = gzippy::decompress_raw(data);
        }
        4 => {
            let _ = gzippy::decompress_deflate64(data);
        }
        _ => {
            // mutate a valid header around raw deflate: header + data, trailer from fuzz tail
            let mut gz = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
            gz.extend_from_slice(data);
            let t = if sel & 0x40 != 0 { 4 } else { 1 };
            let mut sink = Vec::new();
            if gzippy::decompress_to_writer_with_threads(&gz, &mut sink, t).is_ok() {
                let (f, fok) = flate2_multi(&gz);
                if fok {
                    assert!(sink == f, "MISMATCH(wrapped) vs flate2 t={t} ours={} theirs={}", sink.len(), f.len());
                }
            }
        }
    }
});
