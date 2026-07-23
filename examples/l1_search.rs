//! L1-band ratio close-out — empirical config-space search (2026-07-22
//! campaign, see `Cargo.toml`'s `l1-tune` feature doc comment).
//!
//! Sweeps the L1 fast-path matchfinder's tunable knobs (lazy-peek gate,
//! insert-depth, block length, and the new conditional-2nd-bucket-probe
//! lever) via `gzippy::compress::deflate::parse::tune::set` — no rebuild and
//! no respawn per candidate — and reports COMPRESSED SIZE (deterministic,
//! fast, no wall-clock needed) against libdeflate-1/pigz-1/gzip-1 on three
//! corpus groups:
//!   - `text`  — 6 generated text-heavy files (`text_corpus`, ported from
//!     `src/tests/deflate_encoder_matches.rs`'s ratio-guard test).
//!   - `bin`   — 6 generated binary-heavy files (`binary_corpus`, same port).
//!   - `sil`   — 40 slices of `benchmark_data/silesia.tar` at distinct
//!     offsets (real heterogeneous text+binary content), when the tar is
//!     present; skipped with a loud notice otherwise.
//! These are NOT byte-identical to whatever box-side `text6`/`bin6`/`sil40`
//! fixtures a prior session may have used (that definition isn't checked
//! into the repo) — they are a faithful, reproducible LOCAL stand-in built
//! from the same generators the shipped ratio-guard test already uses, so
//! every number here is real and re-runnable, not fabricated.
//!
//! Usage:
//!   cargo run --release --features l1-tune --example l1_search -- size
//!     -> grid search on size only (default if no subcommand given).
//!   cargo run --release --features l1-tune --example l1_search -- wall <config-name> <reps>
//!     -> in-process median wall-ms for ONE named config (for an external
//!        interleaved A/B driver — see the campaign report for the harness).
//!   cargo run --release --features l1-tune --example l1_search -- list
//!     -> print the config grid (name + knob values) without compressing.
//!
//! Build: cargo build --release --features l1-tune --example l1_search

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use gzippy::compress::deflate::compress_gzip;
use gzippy::compress::deflate::parse::tune::{self, L1Tune};

// ---- corpus generators (ported verbatim from
// src/tests/deflate_encoder_matches.rs's fast_l1_ratio_multi_corpus so the
// size numbers here are directly comparable to that shipped ratio guard) ----

fn text_corpus(n: usize) -> Vec<u8> {
    let phrases: [&[u8]; 4] = [
        b"the pure-rust deflate encoder must roundtrip byte for byte. ",
        b"lempel-ziv parsing finds the longest match at each position. ",
        b"dynamic huffman codes adapt to the local symbol frequencies. ",
        b"a stored block escapes when the data will not compress at all. ",
    ];
    let mut out = Vec::with_capacity(n + 128);
    let mut i = 0usize;
    while out.len() < n {
        out.extend_from_slice(phrases[i % phrases.len()]);
        if i.is_multiple_of(11) {
            out.extend_from_slice(format!("<<marker {i} @ {}>> ", out.len()).as_bytes());
        }
        i += 1;
    }
    out.truncate(n);
    out
}

fn binary_corpus(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n + 64);
    let payload: [u8; 12] = [
        0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33, 0xFF, 0xEE, 0xDD, 0xCC,
    ];
    let mut key: u32 = 0;
    while out.len() < n {
        out.extend_from_slice(&key.to_le_bytes());
        out.extend_from_slice(&(key.wrapping_mul(2654435761)).to_le_bytes());
        out.extend_from_slice(&payload);
        if key.is_multiple_of(5) {
            key = key.wrapping_add(1);
        }
    }
    out.truncate(n);
    out
}

fn silesia_tar_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmark_data/silesia.tar")
}

/// `n` bytes of `benchmark_data/silesia.tar` starting at `offset`, or `None`
/// if the tar is absent/short. Distinct offsets land in different member
/// files (tar entries), giving heterogeneous real-world content.
fn silesia_slice_at(offset: u64, n: usize) -> Option<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let path = silesia_tar_path();
    let mut f = std::fs::File::open(path).ok()?;
    f.seek(SeekFrom::Start(offset)).ok()?;
    let mut buf = vec![0u8; n];
    let read = f.read(&mut buf).ok()?;
    if read < n / 2 {
        // Too close to EOF to be a representative slice.
        return None;
    }
    buf.truncate(read);
    Some(buf)
}

struct Corpus {
    group: &'static str,
    label: String,
    data: Vec<u8>,
}

fn build_corpora() -> Vec<Corpus> {
    let mut out = Vec::new();
    // text: 6 distinct sizes (distinct phrase-cycle lengths -> distinct byte
    // statistics per "file").
    for (i, n) in [
        500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000,
    ]
    .into_iter()
    .enumerate()
    {
        out.push(Corpus {
            group: "text",
            label: format!("text{i}-{n}B"),
            data: text_corpus(n),
        });
    }
    // bin: 6 distinct sizes.
    for (i, n) in [
        500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000,
    ]
    .into_iter()
    .enumerate()
    {
        out.push(Corpus {
            group: "bin",
            label: format!("bin{i}-{n}B"),
            data: binary_corpus(n),
        });
    }
    // sil: 40 slices spread across the tar (real heterogeneous content).
    let tar_len = std::fs::metadata(silesia_tar_path()).map(|m| m.len()).ok();
    if let Some(tar_len) = tar_len {
        let slice_n = 4_000_000usize;
        let step = (tar_len / 40).max(slice_n as u64 + 1);
        for i in 0..40u64 {
            let off = i * step;
            if let Some(d) = silesia_slice_at(off, slice_n) {
                out.push(Corpus {
                    group: "sil",
                    label: format!("sil{i}@{off}"),
                    data: d,
                });
            }
        }
    } else {
        eprintln!(
            "l1_search: benchmark_data/silesia.tar absent — sil group skipped \
             (text+bin groups still run)"
        );
    }
    out
}

// ---- reference encoders (same as the shipped ratio-guard test) ----

fn flate2_gzip_size(data: &[u8], level: u32) -> usize {
    let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
    e.write_all(data).unwrap();
    e.finish().unwrap().len()
}

fn libdeflate_gzip_size(data: &[u8], level: i32) -> usize {
    let lvl = libdeflater::CompressionLvl::new(level).expect("valid level");
    let mut c = libdeflater::Compressor::new(lvl);
    let bound = c.gzip_compress_bound(data.len());
    let mut out = vec![0u8; bound];
    c.gzip_compress(data, &mut out).expect("libdeflate gzip")
}

fn pigz_gzip_size(data: &[u8], level: u32) -> Option<usize> {
    let mut child = Command::new("pigz")
        .arg(format!("-{level}"))
        .arg("-c")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take().unwrap();
    let buf = data.to_vec();
    let w = std::thread::spawn(move || {
        let _ = stdin.write_all(&buf);
    });
    let out = child.wait_with_output().ok()?;
    w.join().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout.len())
}

// ---- the config grid ----

/// Shipped defaults (`fast.rs` consts as of the 2026-07-22 campaign start):
/// lazy_peek_max_len=4, lazy_peek_min_dist=8192, insert_depth=3,
/// block_length=65536, bucket2 off.
fn baseline() -> L1Tune {
    L1Tune {
        lazy_peek_max_len: 4,
        lazy_peek_min_dist: 8192,
        insert_depth: 3,
        block_length: 65536,
        bucket2_enabled: false,
        bucket2_gate_max_len: 8,
    }
}

fn named_configs() -> Vec<(String, L1Tune)> {
    let base = baseline();
    let mut v: Vec<(String, L1Tune)> = vec![("baseline".to_string(), base)];

    // Axis A: lazy-peek max_len (accepted-match length gate).
    for max_len in [3, 4, 6, 8, 12, 16] {
        v.push((
            format!("peekmax{max_len}"),
            L1Tune {
                lazy_peek_max_len: max_len,
                ..base
            },
        ));
    }

    // Axis B: lazy-peek min_dist (distance gate; 0 = always peek on a short
    // match, >= WINDOW effectively disables the peek).
    for min_dist in [0usize, 1024, 2048, 4096, 8192, 16384, 32768] {
        v.push((
            format!("peekdist{min_dist}"),
            L1Tune {
                lazy_peek_min_dist: min_dist,
                ..base
            },
        ));
    }

    // Axis C: insert-depth (LIMIT_HASH_UPDATE interior inserts per accepted
    // match).
    for depth in [
        1usize,
        2,
        3,
        4,
        6,
        8,
        12,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        usize::MAX,
    ] {
        v.push((
            format!("insdepth{}", if depth == usize::MAX { 9999 } else { depth }),
            L1Tune {
                insert_depth: depth,
                ..base
            },
        ));
    }

    // Axis D: block length (64 KiB boundary effect).
    for bl in [16384usize, 32768, 65536, 131072, 262144] {
        v.push((
            format!("block{bl}"),
            L1Tune {
                block_length: bl,
                ..base
            },
        ));
    }

    // Axis E: bucket2 (new lever) — gate_max_len sweep.
    for gate in [3u32, 4, 6, 8, 12, 16] {
        v.push((
            format!("bucket2gate{gate}"),
            L1Tune {
                bucket2_enabled: true,
                bucket2_gate_max_len: gate,
                ..base
            },
        ));
    }

    // Hand-picked combined candidates (not just the greedy best-of-axis
    // fold in `combined_configs`): moderate insert-depth (the dominant bin
    // lever, but MAX is likely too wall-costly) paired with bucket2 (the
    // dominant sil lever at near-zero apparent size cost elsewhere).
    for depth in [8usize, 16, 24, 32] {
        for gate in [8u32, 16] {
            v.push((
                format!("hand-depth{depth}-gate{gate}"),
                L1Tune {
                    insert_depth: depth,
                    bucket2_enabled: true,
                    bucket2_gate_max_len: gate,
                    ..base
                },
            ));
        }
    }

    v
}

/// Combined candidates built from the best single-axis result on EACH axis
/// (populated by `pick_combined` after the single-axis sweep reports).
fn combined_configs(picks: &[(&str, L1Tune)]) -> Vec<(String, L1Tune)> {
    let base = baseline();
    let mut out = Vec::new();
    // combined-1: fold every non-baseline field from each axis's best pick
    // into a single stacked candidate.
    let mut c1 = base;
    for (_, p) in picks {
        if p.lazy_peek_max_len != base.lazy_peek_max_len {
            c1.lazy_peek_max_len = p.lazy_peek_max_len;
        }
        if p.lazy_peek_min_dist != base.lazy_peek_min_dist {
            c1.lazy_peek_min_dist = p.lazy_peek_min_dist;
        }
        if p.insert_depth != base.insert_depth {
            c1.insert_depth = p.insert_depth;
        }
        if p.block_length != base.block_length {
            c1.block_length = p.block_length;
        }
        if p.bucket2_enabled {
            c1.bucket2_enabled = true;
            c1.bucket2_gate_max_len = p.bucket2_gate_max_len;
        }
    }
    out.push(("combined-best-of-axis".to_string(), c1));
    out
}

struct RefSizes {
    ld1: usize,
    gzip1: usize,
    pigz1: Option<usize>,
}

fn ref_sizes(data: &[u8]) -> RefSizes {
    RefSizes {
        ld1: libdeflate_gzip_size(data, 1),
        gzip1: flate2_gzip_size(data, 1),
        pigz1: pigz_gzip_size(data, 1),
    }
}

fn run_size_search() {
    let corpora = build_corpora();
    if corpora.is_empty() {
        eprintln!("l1_search: no corpora available");
        std::process::exit(1);
    }
    eprintln!(
        "l1_search: {} corpus files ({} text, {} bin, {} sil)",
        corpora.len(),
        corpora.iter().filter(|c| c.group == "text").count(),
        corpora.iter().filter(|c| c.group == "bin").count(),
        corpora.iter().filter(|c| c.group == "sil").count(),
    );
    for c in &corpora {
        eprintln!("  [{}] {} ({} bytes)", c.group, c.label, c.data.len());
    }

    eprintln!("l1_search: computing reference sizes (libdeflate-1/gzip-1/pigz-1)...");
    let refs: Vec<RefSizes> = corpora.iter().map(|c| ref_sizes(&c.data)).collect();
    let groups = ["text", "bin", "sil"];
    for g in groups {
        let idxs: Vec<usize> = corpora
            .iter()
            .enumerate()
            .filter(|(_, c)| c.group == g)
            .map(|(i, _)| i)
            .collect();
        if idxs.is_empty() {
            continue;
        }
        let ld1: usize = idxs.iter().map(|&i| refs[i].ld1).sum();
        let gzip1: usize = idxs.iter().map(|&i| refs[i].gzip1).sum();
        let pigz1: Option<usize> = if idxs.iter().all(|&i| refs[i].pigz1.is_some()) {
            Some(idxs.iter().map(|&i| refs[i].pigz1.unwrap()).sum())
        } else {
            None
        };
        eprintln!(
            "  ref[{g}]: ld1={ld1} gzip1={gzip1} pigz1={}",
            pigz1.map(|v| v.to_string()).unwrap_or("<absent>".into())
        );
    }

    let configs = named_configs();
    println!("config\tgroup\tsize\tld1_ratio\tgzip1_ratio\tpigz1_ratio");
    let mut axis_best: Vec<(&str, L1Tune)> = Vec::new();
    // Track per-axis-prefix best config (lowest max group/ld1 ratio) for the
    // combined-candidate step.
    use std::collections::HashMap;
    let mut axis_prefix_best: HashMap<&'static str, (f64, L1Tune)> = HashMap::new();
    let axis_of = |name: &str| -> &'static str {
        if name.starts_with("peekmax") {
            "peekmax"
        } else if name.starts_with("peekdist") {
            "peekdist"
        } else if name.starts_with("insdepth") {
            "insdepth"
        } else if name.starts_with("block") {
            "block"
        } else if name.starts_with("bucket2") {
            "bucket2"
        } else {
            "other"
        }
    };

    for (name, tune_cfg) in &configs {
        tune::set(*tune_cfg);
        let sizes: Vec<usize> = corpora
            .iter()
            .map(|c| compress_gzip(&c.data, 1).len())
            .collect();
        let mut worst_ld1_ratio = 0f64;
        for g in groups {
            let idxs: Vec<usize> = corpora
                .iter()
                .enumerate()
                .filter(|(_, c)| c.group == g)
                .map(|(i, _)| i)
                .collect();
            if idxs.is_empty() {
                continue;
            }
            let size: usize = idxs.iter().map(|&i| sizes[i]).sum();
            let ld1: usize = idxs.iter().map(|&i| refs[i].ld1).sum();
            let gzip1: usize = idxs.iter().map(|&i| refs[i].gzip1).sum();
            let pigz1: Option<usize> = if idxs.iter().all(|&i| refs[i].pigz1.is_some()) {
                Some(idxs.iter().map(|&i| refs[i].pigz1.unwrap()).sum())
            } else {
                None
            };
            let ld1_ratio = size as f64 / ld1 as f64;
            let gzip1_ratio = size as f64 / gzip1 as f64;
            let pigz1_ratio = pigz1.map(|p| size as f64 / p as f64);
            worst_ld1_ratio = worst_ld1_ratio.max(ld1_ratio);
            println!(
                "{name}\t{g}\t{size}\t{ld1_ratio:.4}\t{gzip1_ratio:.4}\t{}",
                pigz1_ratio
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or("NA".into())
            );
        }
        let ax = axis_of(name);
        let e = axis_prefix_best.entry(ax).or_insert((f64::MAX, *tune_cfg));
        if worst_ld1_ratio < e.0 {
            *e = (worst_ld1_ratio, *tune_cfg);
        }
    }

    for (ax, (_, cfg)) in axis_prefix_best.iter() {
        if *ax != "other" {
            axis_best.push((ax, *cfg));
        }
    }
    let combined = combined_configs(&axis_best);
    for (name, tune_cfg) in &combined {
        tune::set(*tune_cfg);
        let sizes: Vec<usize> = corpora
            .iter()
            .map(|c| compress_gzip(&c.data, 1).len())
            .collect();
        for g in groups {
            let idxs: Vec<usize> = corpora
                .iter()
                .enumerate()
                .filter(|(_, c)| c.group == g)
                .map(|(i, _)| i)
                .collect();
            if idxs.is_empty() {
                continue;
            }
            let size: usize = idxs.iter().map(|&i| sizes[i]).sum();
            let ld1: usize = idxs.iter().map(|&i| refs[i].ld1).sum();
            let gzip1: usize = idxs.iter().map(|&i| refs[i].gzip1).sum();
            let pigz1: Option<usize> = if idxs.iter().all(|&i| refs[i].pigz1.is_some()) {
                Some(idxs.iter().map(|&i| refs[i].pigz1.unwrap()).sum())
            } else {
                None
            };
            println!(
                "{name}\t{g}\t{size}\t{:.4}\t{:.4}\t{}",
                size as f64 / ld1 as f64,
                size as f64 / gzip1 as f64,
                pigz1
                    .map(|p| format!("{:.4}", size as f64 / p as f64))
                    .unwrap_or("NA".into())
            );
        }
    }

    eprintln!("\nl1_search: config grid (for `wall` mode):");
    for (name, cfg) in configs.iter().chain(combined.iter()) {
        eprintln!(
            "  {name}: peek_max={} peek_dist={} depth={} block={} bucket2={}({})",
            cfg.lazy_peek_max_len,
            cfg.lazy_peek_min_dist,
            if cfg.insert_depth == usize::MAX {
                "ALL".to_string()
            } else {
                cfg.insert_depth.to_string()
            },
            cfg.block_length,
            cfg.bucket2_enabled,
            cfg.bucket2_gate_max_len
        );
    }
}

fn run_wall(name: &str, reps: usize) {
    // `wall` mode looks up one of the single-axis named configs by name, or
    // (since the "combined-*" candidates are only meaningful after a `size`
    // run's printed axis picks) accepts an explicit inline spec instead:
    // "spec:peekmax=6,peekdist=4096,depth=4,block=65536,bucket2=1,gate=8"
    let all: Vec<(String, L1Tune)> = named_configs();
    let cfg = if let Some(spec) = name.strip_prefix("spec:") {
        parse_spec(spec)
    } else {
        all.iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| *c)
            .unwrap_or_else(|| {
                eprintln!("l1_search wall: unknown config '{name}', using baseline");
                baseline()
            })
    };
    tune::set(cfg);

    let corpora = build_corpora();
    // Warm up (page cache / allocator).
    for c in &corpora {
        std::hint::black_box(compress_gzip(&c.data, 1).len());
    }
    let mut ms = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let mut total = 0usize;
        for c in &corpora {
            total += compress_gzip(&c.data, 1).len();
        }
        std::hint::black_box(total);
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = ms[ms.len() / 2];
    println!("config={name} reps={reps} median_ms={median:.3} all_ms={ms:?}");
}

fn parse_spec(spec: &str) -> L1Tune {
    let mut cfg = baseline();
    for kv in spec.split(',') {
        let mut it = kv.splitn(2, '=');
        let k = it.next().unwrap_or("");
        let v = it.next().unwrap_or("");
        match k {
            "peekmax" => cfg.lazy_peek_max_len = v.parse().unwrap(),
            "peekdist" => cfg.lazy_peek_min_dist = v.parse().unwrap(),
            "depth" => cfg.insert_depth = v.parse().unwrap(),
            "block" => cfg.block_length = v.parse().unwrap(),
            "bucket2" => cfg.bucket2_enabled = v == "1" || v == "true",
            "gate" => cfg.bucket2_gate_max_len = v.parse().unwrap(),
            _ => eprintln!("l1_search: unknown spec key '{k}'"),
        }
    }
    cfg
}

fn run_list() {
    for (name, cfg) in named_configs() {
        println!(
            "{name}\tpeekmax={}\tpeekdist={}\tdepth={}\tblock={}\tbucket2={}\tgate={}",
            cfg.lazy_peek_max_len,
            cfg.lazy_peek_min_dist,
            cfg.insert_depth,
            cfg.block_length,
            cfg.bucket2_enabled,
            cfg.bucket2_gate_max_len
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("wall") => {
            let name = args
                .get(2)
                .cloned()
                .unwrap_or_else(|| "baseline".to_string());
            let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);
            run_wall(&name, reps);
        }
        Some("list") => run_list(),
        _ => run_size_search(),
    }
}
