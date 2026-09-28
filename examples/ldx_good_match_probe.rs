//! Size probe for the ldx `good_match` lever (the L6/L7 exception-retirement cell).
//!
//! For each corpus file and level 5/6/7, prints:
//!   prod  — production T1 size via `compress_with_threads(.., 1)`
//!           (L5 = the port with the new zlib-arm config; L6/L7 = the legacy
//!            zlib arm, i.e. the config the port must beat to retire the
//!            routing exception)
//!   port  — the ldx port at that level (`compress_for_diff`, raw DEFLATE;
//!           +18 B gzip overhead is added for comparison)
//!
//! The lever works if `port - prod <= 0` at L6 and L7 across the corpus
//! (the port matches or beats the legacy arm), with `port - prod == 0` at
//! L5 as the probe's self-check (both are the same encoder there).
//!
//! Named cells: the FOUR `won_cells_stay_won` L6 regressions (binary vs
//! gzip/pigz, text vs gzip/pigz) and the L7 ladder cell — see
//! `level_uses_ldx` in src/compress/deflate/mod.rs.

use std::fs;
use std::io::Read;

fn decompress_if_gz(data: Vec<u8>) -> Vec<u8> {
    if data.len() > 2 && data[0] == 0x1f && data[1] == 0x8b {
        let mut d = flate2::read::GzDecoder::new(&data[..]);
        let mut out = Vec::new();
        d.read_to_end(&mut out)
            .expect("gzip input failed to decode");
        out
    } else {
        data
    }
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: ldx_good_match_probe <corpus-dir>");
    let mut entries: Vec<std::fs::DirEntry> =
        fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());

    for e in entries {
        let path = e.path();
        if !path.is_file() {
            continue;
        }
        let raw = fs::read(&path).expect("read");
        if raw.len() > 100 * 1024 * 1024 {
            continue;
        }
        let data = decompress_if_gz(raw);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        for level in [5u32, 6, 7] {
            let prod = gzippy::compress_with_threads(&data, level as u8, 1)
                .unwrap()
                .len();
            let port_raw = gzippy::compress::ldx::compress_for_diff(level, &data)
                .unwrap_or_else(|| panic!("port level {level} missing"))
                .len();
            let port = port_raw + 18; // gzip header + crc/size trailer
            println!(
                "{}\tL{}\tprod={}\tport={}\tdelta={:+}",
                name,
                level,
                prod,
                port,
                port as i64 - prod as i64
            );
        }
    }
}
