//! Cachegrind target for tests/ir_vs_ldx.rs: `ir_runner <ours|ldx> <level> <file>`.
//!
//! One binary, two engines, so the ours-vs-ldx Ir comparison shares its process
//! startup, file read, and allocator behaviour — the only difference cachegrind
//! sees is the encoder itself. Nothing ships from here.
//!
//! Known asymmetry, deliberate: `ours` is the shipped T1 gzip path (header +
//! crc32 + trailer via `compress_with_threads(_, _, 1)`), while `ldx` is raw
//! DEFLATE (`compress_for_diff`, no framing, no crc), so the framing term
//! counts against us. Do NOT read ldx as a stand-in for the libdeflate C
//! binary's cost: the rustc build of the port spends ~1.5x the Ir of the C
//! build (trainer, 2026-08-04: text L1 ldx 62.02 Ir/B vs libdeflate-gzip
//! 40.60 Ir/B — 65.0M vs 42.6M instructions on the 1 MiB fixture). This is a
//! SAME-TOOLCHAIN algorithm comparison; the real-binary goal is the rival leg
//! in tests/fingerprints/rivals_ir.tsv.
//!
//! Third arm `ldx_gzip` closes that asymmetry ladder from the other side: the
//! SAME gzip framing `ours` emits (header `deflate/mod.rs:78`, crc32fast +
//! ISIZE) wrapped around the port's raw `compress_into` — i.e. the shipped
//! bytes with the shipped framing, minus the T1 driver route (whole-buffer
//! read+copy, INPLACE_TAIL_PAD sizing, in-place parse, flush). So
//! `ours / ldx_gzip` isolates the driver tax and `ldx_gzip / ldx` isolates the
//! framing term, both in the same runnable binary. Probe-only, like the arms
//! above; not part of either pin file's goal sheet.
fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: ir_runner <ours|ldx_gzip|ldx> <level> <file>";
    let engine = args.next().expect(usage);
    let level: u32 = args.next().expect(usage).parse().expect(usage);
    let path = args.next().expect(usage);
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let out = match engine.as_str() {
        "ours" => gzippy::compress_with_threads(&data, level as u8, 1).expect("compress failed"),
        "ldx" => gzippy::compress::ldx::compress_for_diff(level, &data)
            .unwrap_or_else(|| panic!("ldx does not support level {level}")),
        "ldx_gzip" => {
            let mut out = Vec::new();
            // Identical header to ours: magic, CM=8, FLG=0, MTIME=0, XFL=0,
            // OS=0xff (src/compress/deflate/mod.rs:78).
            out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff]);
            if !gzippy::compress::ldx::compress_into(level, &data, &mut out) {
                panic!("ldx does not support level {level}");
            }
            // Identical trailer to ours: crc32fast over the input + ISIZE
            // (src/compress/deflate/mod.rs:708, README form at :857).
            out.extend_from_slice(&crc32fast::hash(&data).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out
        }
        other => panic!("unknown engine '{other}' — {usage}"),
    };
    // Print the size so the compression cannot be optimized away.
    println!("{}", out.len());
}
