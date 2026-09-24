//! L1 lever probe: the ldx port L1 vs the production (legacy) L1.
//!
//! Prints per corpus file: the port's raw DEFLATE size (`compress_for_diff`,
//! level 1) with its payload sha256, and the production L1 gzip size with
//! payload sha256. The caller compares `port_raw + 18` against
//! `libdeflate-gzip -1` (10-byte header + 8-byte trailer) for the size and
//! against the vendor payload sha for byte identity.
//!
//! Named cell: text L1 T1 — the census says the port is +739 B vs
//! libdeflate there; if the port is really byte-identical to the vendor,
//! that pin is stale and the L1 routing flip is a pure size win on the
//! board.

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
        .expect("usage: ldx_l1_probe <corpus-dir>");
    let mut entries: Vec<std::fs::DirEntry> =
        fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let name = e.file_name().to_string_lossy().to_string();
        let mut data = Vec::new();
        fs::File::open(e.path())
            .unwrap()
            .read_to_end(&mut data)
            .unwrap();
        data = decompress_if_gz(data);
        if data.is_empty() {
            continue;
        }
        let raw = gzippy::compress::ldx::compress_for_diff(1, &data)
            .unwrap_or_else(|| panic!("{name}: ldx L1 missing"));
        let prod = gzippy::compress_with_threads(&data, 1, 1).unwrap();
        std::fs::write(format!("/tmp/l1probe/{name}.port_raw"), &raw).unwrap();
        std::fs::write(format!("/tmp/l1probe/{name}.prod_gz"), &prod).unwrap();
        println!(
            "{name}\tinput={}\tport_raw={}\tprod_gz={}",
            data.len(),
            raw.len(),
            prod.len()
        );
    }
}
