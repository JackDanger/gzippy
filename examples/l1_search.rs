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
//!   cargo run --release --features l1-tune --example l1_search -- breadth <cfg1;cfg2;...>
//!     -> per-file size/ratio/gate table over the ~/www/gzippy-bench/corpus
//!        breadth files (19+, excluding dd79_text6/dd79_bin6) for each named
//!        config (';'-separated — spec:k=v,k=v... configs use ',' internally),
//!        plus a WIN/LOSS flip report vs the FIRST config in the list
//!        (2026-07-22 hash3+detector composition mission).
//!   cargo run --release --features l1-tune --example l1_search -- file <path> <cfg1;cfg2;...>
//!     -> T1 (`compress_gzip`) size/ratio table for ONE explicit file — the
//!        targeted single-fixture micro-sweep tool (e.g. dd79_bin6, which
//!        `breadth` deliberately excludes).
//!   cargo run --release --features l1-tune --example l1_search -- filemt <path> <threads> <cfg1;cfg2;...>
//!     -> same, but through the REAL T>1 production path (`compress_bytes`
//!        -> `PipelinedGzEncoder::compress_buffer_pure`, 512KB per-chunk
//!        `compress_block_streaming`) to measure the per-chunk gate-state-
//!        reset shape directly instead of inferring it from T1 numbers.
//!
//! Build: cargo build --release --features l1-tune --example l1_search

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use gzippy::compress::compress_bytes;
use gzippy::compress::deflate::compress_gzip;
use gzippy::compress::deflate::parse::tune::{self, L1Tune};

// ---- corpus generators (ported verbatim from
// src/tests/deflate_encoder_matches.rs's fast_l1_ratio_multi_corpus). NO
// LONGER USED by `build_corpora` (2026-07-22 coordinator correction): the
// REAL dd79_text6/dd79_bin6 fixtures are available locally and are what the
// "13 non-WIN cells" claim was actually measured against — see
// `build_corpora`'s doc comment. Kept (not deleted) as a documented fallback
// / for a future breadth pass beyond the single dd79 fixture pair.
#[allow(dead_code)]
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

#[allow(dead_code)]
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

/// Breadth corpus: every file in `~/www/gzippy-bench/corpus` EXCEPT the
/// `dd79_text6`/`dd79_bin6` fixtures already covered by `build_corpora`'s
/// named text/bin groups — the "19+ breadth files" the composition mission
/// (2026-07-22) requires every candidate be swept against, the same
/// fixture directory the HASH3-PROBE and CONTENT-ADAPTIVE CHAIN MATCHING
/// lever reports used for their breadth sweeps.
fn breadth_corpus_dir() -> PathBuf {
    PathBuf::from("/Users/jackdanger/www/gzippy-bench/corpus")
}

fn build_breadth_corpora() -> Vec<Corpus> {
    let dir = breadth_corpus_dir();
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("l1_search: breadth corpus dir {dir:?} unreadable: {e}");
            return out;
        }
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n != "dd79_text6" && n != "dd79_bin6")
        .collect();
    names.sort();
    for name in names {
        match std::fs::read(dir.join(&name)) {
            Ok(d) => out.push(Corpus {
                group: "breadth",
                label: name,
                data: d,
            }),
            Err(e) => eprintln!("l1_search: breadth file {name} unreadable: {e}"),
        }
    }
    out
}

/// The REAL fixtures the "13 non-WIN cells" claim was measured against
/// (coordinator correction, 2026-07-22): `dd79_text6`/`dd79_bin6` are exact
/// 6 MiB (6,291,456-byte) files in `~/www/gzippy-bench/corpus/`, not the
/// generated stand-ins this tool used in its first pass (that synthetic
/// `bin` corpus turned out to be a much harder, unrepresentative regime —
/// 2.28x ld1 vs the real fixture's measured 1.009-1.049x band, see below).
fn dd79_fixture_path(name: &str) -> PathBuf {
    PathBuf::from("/Users/jackdanger/www/gzippy-bench/corpus").join(name)
}

fn read_file_or_none(path: &PathBuf) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

fn build_corpora() -> Vec<Corpus> {
    let mut out = Vec::new();

    match read_file_or_none(&dd79_fixture_path("dd79_text6")) {
        Some(d) => out.push(Corpus {
            group: "text",
            label: "dd79_text6".to_string(),
            data: d,
        }),
        None => eprintln!(
            "l1_search: gzippy-bench/corpus/dd79_text6 absent — text group skipped; \
             falling back to the synthetic generator would NOT be the real fixture, \
             refusing to substitute silently"
        ),
    }
    match read_file_or_none(&dd79_fixture_path("dd79_bin6")) {
        Some(d) => out.push(Corpus {
            group: "bin",
            label: "dd79_bin6".to_string(),
            data: d,
        }),
        None => eprintln!(
            "l1_search: gzippy-bench/corpus/dd79_bin6 absent — bin group skipped; \
             falling back to the synthetic generator would NOT be the real fixture, \
             refusing to substitute silently"
        ),
    }
    // sil: ONE real 40 MiB (41,943,040-byte) contiguous slice of
    // benchmark_data/silesia.tar — "sil40" by the same naming convention as
    // the 6 MiB text6/bin6 fixtures (a size, not a file count; the first
    // pass's 40-separate-4MB-slices reading was wrong).
    let sil_n = 40 * 1024 * 1024;
    match silesia_slice_at(0, sil_n) {
        Some(d) => out.push(Corpus {
            group: "sil",
            label: format!("silesia@0+{sil_n}B"),
            data: d,
        }),
        None => {
            eprintln!("l1_search: benchmark_data/silesia.tar absent or short — sil group skipped")
        }
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
        chain_enabled: false,
        chain_lit_threshold_pct: 80,
        chain_max_search_depth: 16,
        hash3_enabled: false,
        hash3_bits: 13,
        hash3_always_probe: false,
        hash3_max_dist: 4096,
        hash3_insert_always: true,
        hash3_gated: false,
        hash3_gate_lit_threshold_pct: 80,
        hash3_gate_warm_insert: true,
        hash3_gate_initial_active: true,
    }
}

/// The measured-best HASH3-PROBE knobs (bits=15, max_dist=32768,
/// insert_always=true, miss-only probe) from the 2026-07-22 HASH3-PROBE
/// lever report — the composition mission's starting point ("compose the
/// two proven l1-tune levers"), not re-derived here.
fn hash3_best() -> L1Tune {
    L1Tune {
        hash3_enabled: true,
        hash3_bits: 15,
        hash3_always_probe: false,
        hash3_max_dist: 32768,
        hash3_insert_always: true,
        ..baseline()
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

    // Axis F: CONTENT-ADAPTIVE CHAIN MATCHING (2026-07-22 mission) —
    // literal-density threshold x chain search-depth grid. Threshold is a
    // PERCENT (literal fraction of the preceding block); depth is
    // `max_search_depth` for the hash-chains finder on a fired block.
    for threshold in [50u32, 65, 80, 90] {
        for depth in [4u32, 8, 16, 32, 64, 128] {
            v.push((
                format!("chain-t{threshold}-d{depth}"),
                L1Tune {
                    chain_enabled: true,
                    chain_lit_threshold_pct: threshold,
                    chain_max_search_depth: depth,
                    ..base
                },
            ));
        }
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

    // Axis G: HASH3-PROBE (the last unmeasured member of the L1 probe-adding
    // family — see `tune::L1Tune::hash3_enabled`'s doc comment). Table-size x
    // max-dist grid at the cheaper "miss-only" probe policy first (policy
    // (a)), insert_always=true (the cheaper insert policy, tried first).
    for bits in [12u32, 13, 14, 15] {
        for max_dist in [256usize, 1024, 4096, 16384, 32768] {
            v.push((
                format!("hash3-b{bits}-d{max_dist}"),
                L1Tune {
                    hash3_enabled: true,
                    hash3_bits: bits,
                    hash3_always_probe: false,
                    hash3_max_dist: max_dist,
                    hash3_insert_always: true,
                    ..base
                },
            ));
        }
    }
    // Axis G2: probe-policy x insert-policy grid at a fixed mid-size table
    // (bits=13) and mid max_dist (4096), to isolate each axis independently
    // of the table-size sweep above.
    for always_probe in [false, true] {
        for insert_always in [true, false] {
            v.push((
                format!(
                    "hash3-policy-probe{}-ins{}",
                    always_probe as u8, insert_always as u8
                ),
                L1Tune {
                    hash3_enabled: true,
                    hash3_bits: 13,
                    hash3_always_probe: always_probe,
                    hash3_max_dist: 4096,
                    hash3_insert_always: insert_always,
                    ..base
                },
            ));
        }
    }
    // Hand-picked combined: hash3 (miss-only, insert-always, mid table)
    // stacked on top of the dominant insert-depth/bucket2 combo above, to
    // see whether the hash3 lever composes with the rest of the frontier.
    for depth in [8usize, 16] {
        for bits in [13u32, 14] {
            v.push((
                format!("hand-hash3-depth{depth}-b{bits}"),
                L1Tune {
                    insert_depth: depth,
                    hash3_enabled: true,
                    hash3_bits: bits,
                    hash3_always_probe: false,
                    hash3_max_dist: 4096,
                    hash3_insert_always: true,
                    ..base
                },
            ));
        }
    }

    // Axis H: HASH3-GATE composition (2026-07-22 "compose the two proven
    // l1-tune levers" mission). Layers the CONTENT-ADAPTIVE CHAIN
    // MATCHING lever's free literal-fraction detector onto the
    // measured-best HASH3-PROBE knobs (`hash3_best()`), so hash3 only
    // PROBES on blocks the preceding block's literal fraction flags as
    // bin-like. Threshold x warm-insert x initial-active grid — the
    // mission's named open question is exactly whether warm-insert
    // (keep `head3` populated through a gated-off stretch) beats sparse
    // gating (cheaper, but cold on re-activation).
    //
    // 47/48/49 added (2026-07-24 targeted micro-sweep, closing the
    // `dd79_bin6` promotion-gate blocker): `threshold=50` is a real
    // per-file knife-edge on `dd79_bin6` specifically (the 2026-07-22
    // aggregate-only sweep's "47-51 plateau" call missed this — see
    // `tune::L1Tune::hash3_gate_lit_threshold_pct`'s doc comment).
    // `hash3gate-t48-w0-i1` is the measured-best composed config found by
    // that sweep (T1 AND T4 WIN on `dd79_bin6` vs pigz-1, zero breadth
    // flips) — also now `L1Tune::from_env()`'s default, so a plain
    // `l1-tune`-feature build with no env override reproduces it.
    let h3 = hash3_best();
    for threshold in [47u32, 48, 49, 50, 65, 80, 90] {
        for warm_insert in [true, false] {
            for initial_active in [true, false] {
                v.push((
                    format!(
                        "hash3gate-t{threshold}-w{}-i{}",
                        warm_insert as u8, initial_active as u8
                    ),
                    L1Tune {
                        hash3_gated: true,
                        hash3_gate_lit_threshold_pct: threshold,
                        hash3_gate_warm_insert: warm_insert,
                        hash3_gate_initial_active: initial_active,
                        ..h3
                    },
                ));
            }
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
        } else if name.starts_with("hash3") {
            "hash3"
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

/// Like `run_wall` but times EACH corpus file separately and prints one line
/// per corpus (needed because the mission's per-cell kill rule is per
/// CORPUS CLASS — bin vs sil vs text each have a different pigz-1 wall
/// budget; the combined-loop `wall` mode conflates them, dominated by
/// whichever corpus is largest).
fn run_wall_percorpus(name: &str, reps: usize) {
    let all: Vec<(String, L1Tune)> = named_configs();
    let cfg = if let Some(spec) = name.strip_prefix("spec:") {
        parse_spec(spec)
    } else {
        all.iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| *c)
            .unwrap_or_else(|| {
                eprintln!("l1_search wallpc: unknown config '{name}', using baseline");
                baseline()
            })
    };
    tune::set(cfg);

    let corpora = build_corpora();
    for c in &corpora {
        std::hint::black_box(compress_gzip(&c.data, 1).len());
        let mut ms = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            let sz = compress_gzip(&c.data, 1).len();
            std::hint::black_box(sz);
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = ms[ms.len() / 2];
        println!(
            "config={name} corpus={}[{}] reps={reps} median_ms={median:.3}",
            c.label, c.group
        );
    }
}

/// Breadth accounting for the composition mission (2026-07-22): for each
/// named config (comma-separated in `names`), compresses every breadth
/// corpus file and reports size + ratios vs ld1/gzip1/pigz1, PLUS two gate
/// columns per the mission's "both-column accounting":
///   - strict: PASS iff size <= ld1*1.05 AND (pigz1 unavailable OR
///     size <= pigz1*1.05) — every RIVAL L1 encoder, both axes, with the
///     lever family's usual 1.05 slack.
///   - family: PASS iff size <= pigz1 (when available) AND size <= gzip1 —
///     the GNU-gzip-compatible family, no slack, "size <= everyone".
/// Also classifies each (config, file) as WIN/LOSS/TIE vs pigz-1 (ratio <
/// 1.0 / > 1.0 / == 1.0, matching the HASH3-PROBE/chain-mode lever
/// reports' own WIN/LOSS terminology) and, relative to the FIRST config in
/// `names` (the baseline for comparison), flags any WIN->LOSS or LOSS->WIN
/// flip — the exact accounting the mission's "avoid regressions" /
/// "gain the bin flips" bars need.
fn run_breadth(names: &str) {
    // Split on ';' (NOT ',') because a `spec:k=v,k=v,...` config's own
    // key=value pairs are comma-separated — a ',' outer delimiter would
    // shred a single spec into unparseable fragments.
    let cfg_names: Vec<&str> = names.split(';').collect();
    let configs: Vec<(String, L1Tune)> = cfg_names
        .iter()
        .map(|n| {
            let all = named_configs();
            let cfg = if let Some(spec) = n.strip_prefix("spec:") {
                parse_spec(spec)
            } else {
                all.iter()
                    .find(|(nm, _)| nm == n)
                    .map(|(_, c)| *c)
                    .unwrap_or_else(|| {
                        eprintln!("l1_search breadth: unknown config '{n}', using baseline");
                        baseline()
                    })
            };
            (n.to_string(), cfg)
        })
        .collect();

    let corpora = build_breadth_corpora();
    if corpora.is_empty() {
        eprintln!("l1_search breadth: no breadth corpus files found");
        std::process::exit(1);
    }
    eprintln!("l1_search breadth: {} files", corpora.len());

    let refs: Vec<RefSizes> = corpora.iter().map(|c| ref_sizes(&c.data)).collect();

    // config -> file -> size
    let mut sizes: Vec<Vec<usize>> = Vec::with_capacity(configs.len());
    for (_, cfg) in &configs {
        tune::set(*cfg);
        sizes.push(
            corpora
                .iter()
                .map(|c| compress_gzip(&c.data, 1).len())
                .collect(),
        );
    }

    println!("config\tfile\tsize\tld1_ratio\tgzip1_ratio\tpigz1_ratio\tstrict\tfamily\tvs_pigz");
    for (ci, (name, _)) in configs.iter().enumerate() {
        for (fi, c) in corpora.iter().enumerate() {
            let sz = sizes[ci][fi];
            let r = &refs[fi];
            let ld1_ratio = sz as f64 / r.ld1 as f64;
            let gzip1_ratio = sz as f64 / r.gzip1 as f64;
            let pigz1_ratio = r.pigz1.map(|p| sz as f64 / p as f64);
            let strict = ld1_ratio <= 1.05 && pigz1_ratio.map(|v| v <= 1.05).unwrap_or(true);
            let family = pigz1_ratio.map(|v| v <= 1.0).unwrap_or(true) && gzip1_ratio <= 1.0;
            let vs_pigz = match pigz1_ratio {
                Some(v) if v < 1.0 => "WIN",
                Some(v) if v > 1.0 => "LOSS",
                Some(_) => "TIE",
                None => "NA",
            };
            println!(
                "{name}\t{}\t{sz}\t{ld1_ratio:.4}\t{gzip1_ratio:.4}\t{}\t{}\t{}\t{vs_pigz}",
                c.label,
                pigz1_ratio
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or("NA".into()),
                if strict { "PASS" } else { "FAIL" },
                if family { "PASS" } else { "FAIL" },
            );
        }
    }

    // Flip accounting: config[0] is the reference baseline.
    if configs.len() > 1 {
        eprintln!("\nl1_search breadth: flip accounting vs '{}'", configs[0].0);
        for (fi, c) in corpora.iter().enumerate() {
            let r = &refs[fi];
            let base_pigz = r
                .pigz1
                .map(|p| sizes[0][fi] as f64 / p as f64)
                .unwrap_or(f64::NAN);
            for ci in 1..configs.len() {
                let cand_pigz = r
                    .pigz1
                    .map(|p| sizes[ci][fi] as f64 / p as f64)
                    .unwrap_or(f64::NAN);
                let base_win = base_pigz < 1.0;
                let cand_win = cand_pigz < 1.0;
                if base_win != cand_win {
                    eprintln!(
                        "  FLIP [{}] {}: {} -> {} (pigz ratio {:.4} -> {:.4})",
                        configs[ci].0,
                        c.label,
                        if base_win { "WIN" } else { "LOSS" },
                        if cand_win { "WIN" } else { "LOSS" },
                        base_pigz,
                        cand_pigz
                    );
                }
            }
        }
    }
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
            "hash3" => cfg.hash3_enabled = v == "1" || v == "true",
            "hash3bits" => cfg.hash3_bits = v.parse().unwrap(),
            "hash3always" => cfg.hash3_always_probe = v == "1" || v == "true",
            "hash3maxdist" => cfg.hash3_max_dist = v.parse().unwrap(),
            "hash3insertalways" => cfg.hash3_insert_always = v == "1" || v == "true",
            "hash3gated" => cfg.hash3_gated = v == "1" || v == "true",
            "hash3gatethreshold" => cfg.hash3_gate_lit_threshold_pct = v.parse().unwrap(),
            "hash3gatewarm" => cfg.hash3_gate_warm_insert = v == "1" || v == "true",
            "hash3gateinit" => cfg.hash3_gate_initial_active = v == "1" || v == "true",
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

/// Targeted single-file micro-sweep support (2026-07-24 hash3-gate bin6
/// close-out mission): sweeps a ';'-separated list of named/`spec:`
/// configs (same grammar as `breadth`) against ONE explicit file path (not
/// restricted to the breadth corpus dir's exclusion of dd79_text6/dd79_bin6
/// — the mission's target fixture IS one of those) and prints size +
/// ratios vs ld1/gzip1/pigz1 per config, T1 only (`compress_gzip`, same
/// engine `run()` call as `breadth`/`size`).
fn run_file(path: &str, names: &str) {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("l1_search file: {path} unreadable: {e}");
            std::process::exit(1);
        }
    };
    let cfg_names: Vec<&str> = names.split(';').collect();
    let configs: Vec<(String, L1Tune)> = cfg_names
        .iter()
        .map(|n| {
            let all = named_configs();
            let cfg = if let Some(spec) = n.strip_prefix("spec:") {
                parse_spec(spec)
            } else {
                all.iter()
                    .find(|(nm, _)| nm == n)
                    .map(|(_, c)| *c)
                    .unwrap_or_else(|| {
                        eprintln!("l1_search file: unknown config '{n}', using baseline");
                        baseline()
                    })
            };
            (n.to_string(), cfg)
        })
        .collect();

    let r = ref_sizes(&data);
    eprintln!(
        "l1_search file: {path} ({} bytes) ld1={} gzip1={} pigz1={}",
        data.len(),
        r.ld1,
        r.gzip1,
        r.pigz1.map(|v| v.to_string()).unwrap_or("<absent>".into())
    );
    println!("config\tsize\tld1_ratio\tgzip1_ratio\tpigz1_ratio\tvs_pigz");
    for (name, cfg) in &configs {
        tune::set(*cfg);
        let sz = compress_gzip(&data, 1).len();
        let ld1_ratio = sz as f64 / r.ld1 as f64;
        let gzip1_ratio = sz as f64 / r.gzip1 as f64;
        let pigz1_ratio = r.pigz1.map(|p| sz as f64 / p as f64);
        let vs_pigz = match pigz1_ratio {
            Some(v) if v < 1.0 => "WIN",
            Some(v) if v > 1.0 => "LOSS",
            Some(_) => "TIE",
            None => "NA",
        };
        println!(
            "{name}\t{sz}\t{ld1_ratio:.4}\t{gzip1_ratio:.4}\t{}\t{vs_pigz}",
            pigz1_ratio
                .map(|v| format!("{v:.4}"))
                .unwrap_or("NA".into())
        );
    }
}

/// Multi-threaded shape of the SAME single-file sweep: routes through
/// `compress_bytes` (the real production T>1 entry point,
/// `PipelinedGzEncoder::compress_buffer_pure` → 512KB-block
/// `compress_block_streaming` per chunk) instead of the T1-only
/// `compress_gzip`, so the mission's "does the per-chunk gate-state reset
/// cost bin6 at T4" question is measured on the ACTUAL T>1 code path, not
/// inferred. `tune::set` is a process-global `RwLock`, so it applies to
/// every worker thread `compress_bytes` spawns.
fn run_file_mt(path: &str, threads: usize, names: &str) {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("l1_search filemt: {path} unreadable: {e}");
            std::process::exit(1);
        }
    };
    let cfg_names: Vec<&str> = names.split(';').collect();
    let configs: Vec<(String, L1Tune)> = cfg_names
        .iter()
        .map(|n| {
            let all = named_configs();
            let cfg = if let Some(spec) = n.strip_prefix("spec:") {
                parse_spec(spec)
            } else {
                all.iter()
                    .find(|(nm, _)| nm == n)
                    .map(|(_, c)| *c)
                    .unwrap_or_else(|| {
                        eprintln!("l1_search filemt: unknown config '{n}', using baseline");
                        baseline()
                    })
            };
            (n.to_string(), cfg)
        })
        .collect();

    let r = ref_sizes(&data);
    eprintln!(
        "l1_search filemt: {path} T{threads} ({} bytes) ld1={} gzip1={} pigz1={}",
        data.len(),
        r.ld1,
        r.gzip1,
        r.pigz1.map(|v| v.to_string()).unwrap_or("<absent>".into())
    );
    println!("config\tthreads\tsize\tld1_ratio\tgzip1_ratio\tpigz1_ratio\tvs_pigz");
    for (name, cfg) in &configs {
        tune::set(*cfg);
        let mut out = Vec::new();
        compress_bytes(std::io::Cursor::new(&data), &mut out, 1, threads).expect("compress");
        let sz = out.len();
        let ld1_ratio = sz as f64 / r.ld1 as f64;
        let gzip1_ratio = sz as f64 / r.gzip1 as f64;
        let pigz1_ratio = r.pigz1.map(|p| sz as f64 / p as f64);
        let vs_pigz = match pigz1_ratio {
            Some(v) if v < 1.0 => "WIN",
            Some(v) if v > 1.0 => "LOSS",
            Some(_) => "TIE",
            None => "NA",
        };
        println!(
            "{name}\t{threads}\t{sz}\t{ld1_ratio:.4}\t{gzip1_ratio:.4}\t{}\t{vs_pigz}",
            pigz1_ratio
                .map(|v| format!("{v:.4}"))
                .unwrap_or("NA".into())
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("file") => {
            let path = args.get(2).cloned().unwrap_or_default();
            let names = args
                .get(3)
                .cloned()
                .unwrap_or_else(|| "baseline".to_string());
            run_file(&path, &names);
        }
        Some("filemt") => {
            let path = args.get(2).cloned().unwrap_or_default();
            let threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
            let names = args
                .get(4)
                .cloned()
                .unwrap_or_else(|| "baseline".to_string());
            run_file_mt(&path, threads, &names);
        }
        Some("wall") => {
            let name = args
                .get(2)
                .cloned()
                .unwrap_or_else(|| "baseline".to_string());
            let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);
            run_wall(&name, reps);
        }
        Some("wallpc") => {
            let name = args
                .get(2)
                .cloned()
                .unwrap_or_else(|| "baseline".to_string());
            let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);
            run_wall_percorpus(&name, reps);
        }
        Some("list") => run_list(),
        Some("breadth") => {
            let names = args
                .get(2)
                .cloned()
                .unwrap_or_else(|| "baseline".to_string());
            run_breadth(&names);
        }
        _ => run_size_search(),
    }
}
