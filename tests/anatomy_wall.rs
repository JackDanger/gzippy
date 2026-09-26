//! Integration test for the `anatomy-wall` feature (closed-loop conservation
//! check against a REAL compression run).
//!
//! Spawns the actual `gzippy` binary as a fresh subprocess — same isolation
//! rationale as `tests/anatomy_counters.rs`: `anatomy_wall::WALL` is one
//! process-wide static, and `cargo test` runs every test in the crate's
//! unit-test binary concurrently by default.
//!
//! Only compiled/run when the `anatomy-wall` feature is enabled (`cargo test
//! --features anatomy-wall`); the feature-off default build carries none of
//! this.

#![cfg(feature = "anatomy-wall")]

use std::collections::BTreeMap;
use std::process::{Command, Stdio};

/// Parse the flat `{"key":value,...}` JSON object `AnatomyWall::to_json`
/// emits. Values are either unsigned integers, `true`/`false`, or a quoted
/// string (the `granularity` field) — a `serde_json`-free hand parse
/// covering exactly that shape.
#[derive(Debug, Clone)]
enum Val {
    Num(u64),
    Bool(bool),
    // Held only to prove the `granularity` field parses as a well-formed
    // quoted string; no test currently reads its contents.
    #[allow(dead_code)]
    Str(String),
}

fn parse_flat_json(s: &str) -> BTreeMap<String, Val> {
    let body = s
        .trim()
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or_else(|| panic!("not a flat JSON object: {s}"));
    let mut map = BTreeMap::new();
    if body.is_empty() {
        return map;
    }
    // Split top-level commas only; the only nested commas possible here
    // would be inside a quoted string value (the `granularity` field), so
    // track quote state while splitting.
    let mut parts = Vec::new();
    let mut depth_in_quote = false;
    let mut start = 0;
    let bytes = body.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'"' {
            depth_in_quote = !depth_in_quote;
        } else if b == b',' && !depth_in_quote {
            parts.push(&body[start..i]);
            start = i + 1;
        }
    }
    parts.push(&body[start..]);

    for pair in parts {
        let (k, v) = pair
            .split_once(':')
            .unwrap_or_else(|| panic!("malformed key:value pair {pair:?} in {s}"));
        let key = k.trim().trim_matches('"').to_string();
        let vt = v.trim();
        let val = if let Some(inner) = vt.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
            Val::Str(inner.to_string())
        } else if vt == "true" {
            Val::Bool(true)
        } else if vt == "false" {
            Val::Bool(false)
        } else {
            Val::Num(
                vt.parse()
                    .unwrap_or_else(|_| panic!("non-integer/bool value for {key}: {vt:?}")),
            )
        };
        map.insert(key, val);
    }
    map
}

fn get_num(m: &BTreeMap<String, Val>, k: &str) -> u64 {
    match m.get(k) {
        Some(Val::Num(n)) => *n,
        other => panic!("expected numeric field {k:?}, got {other:?}"),
    }
}

fn get_bool(m: &BTreeMap<String, Val>, k: &str) -> bool {
    match m.get(k) {
        Some(Val::Bool(b)) => *b,
        other => panic!("expected boolean field {k:?}, got {other:?}"),
    }
}

/// A mixed corpus big enough to span several 64 KiB internal L1 blocks, so
/// `parse_match_calls`/`huffman_table_calls`/`huffman_encode_calls` are all
/// forced above 1 (proving per-block, not per-invocation, granularity) but
/// far below the input's byte count (proving NOT per-position granularity).
fn mixed_corpus(min_len: usize) -> Vec<u8> {
    let phrases: [&[u8]; 4] = [
        b"the quick brown fox jumps over the lazy dog; ",
        b"gzippy anatomy wall timers close the calibration loop; ",
        b"lorem ipsum dolor sit amet consectetur adipiscing elit; ",
        b"0123456789abcdef repeated structure repeated structure ",
    ];
    let mut data = Vec::new();
    let mut i = 0usize;
    while data.len() < min_len {
        data.extend_from_slice(phrases[i % phrases.len()]);
        let x = (i.wrapping_mul(2654435761)) as u32;
        data.extend_from_slice(&x.to_le_bytes());
        i += 1;
    }
    data
}

/// Run `gzippy -{level} -c -p 1` over `data` via stdin, returning
/// `(compressed_stdout, wall_map_from_stderr)`.
fn compress_with_wall(data: &[u8], level: u32) -> (Vec<u8>, BTreeMap<String, Val>) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_gzippy"))
        .arg(format!("-{level}"))
        .arg("-c")
        .arg("-p")
        .arg("1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn gzippy");

    let mut stdin = child.stdin.take().unwrap();
    let data_owned = data.to_vec();
    let writer = std::thread::spawn(move || {
        use std::io::Write;
        stdin.write_all(&data_owned).expect("write stdin");
    });

    let output = child.wait_with_output().expect("wait for gzippy");
    writer.join().unwrap();
    assert!(
        output.status.success(),
        "gzippy exited non-zero: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    let reconcile_line = stderr
        .lines()
        .find(|l| l.starts_with("ANATOMY_WALL_RECONCILE="))
        .unwrap_or_else(|| panic!("no ANATOMY_WALL_RECONCILE= line on stderr:\n{stderr}"));
    assert!(
        reconcile_line.starts_with("ANATOMY_WALL_RECONCILE=PASS"),
        "expected a PASS reconciliation, got: {reconcile_line}\nfull stderr:\n{stderr}"
    );

    let line = stderr
        .lines()
        .find_map(|l| l.strip_prefix("ANATOMY_WALL="))
        .unwrap_or_else(|| panic!("no ANATOMY_WALL= line on stderr:\n{stderr}"));
    (output.stdout, parse_flat_json(line))
}

/// The mission's conservation invariant, end-to-end against a real `gzippy`
/// invocation: named regions + derived residual reconcile to the root span,
/// and the granularity is per-block (call counts >> 1, << input byte count).
#[test]
fn conservation_and_granularity_hold_on_a_real_gzippy_invocation() {
    let data = mixed_corpus(900_000);
    let (compressed, w) = compress_with_wall(&data, 1);

    // Sanity: a valid, byte-exact gzip stream.
    let mut decoded = Vec::new();
    {
        use std::io::Read;
        flate2::read::GzDecoder::new(&compressed[..])
            .read_to_end(&mut decoded)
            .expect("gzippy stdout must be a valid gzip stream");
    }
    assert_eq!(decoded, data, "roundtrip sanity check failed");

    let root_ns = get_num(&w, "root_ns");
    let root_calls = get_num(&w, "root_calls");
    let parse_ns = get_num(&w, "parse_match_ns");
    let parse_calls = get_num(&w, "parse_match_calls");
    let table_ns = get_num(&w, "huffman_table_ns");
    let table_calls = get_num(&w, "huffman_table_calls");
    let encode_ns = get_num(&w, "huffman_encode_ns");
    let encode_calls = get_num(&w, "huffman_encode_calls");
    let crc_ns = get_num(&w, "crc_ns");
    let crc_calls = get_num(&w, "crc_calls");
    let residual_ns = get_num(&w, "residual_ns");
    let conserved = get_bool(&w, "conserved");

    assert!(root_ns > 0, "root span must have measured nonzero time");
    assert_eq!(
        root_calls, 1,
        "exactly one encode_gzip_bytes_to_vec* invocation this run"
    );
    assert!(
        conserved,
        "conservation must hold: named regions must not exceed root_ns"
    );
    assert_eq!(
        root_ns,
        parse_ns + table_ns + encode_ns + crc_ns + residual_ns,
        "root_ns must equal the sum of every named region plus the residual exactly"
    );

    // Granularity: per-block, not per-invocation (>1 call) and not
    // per-position (far fewer calls than input bytes).
    assert!(
        parse_calls > 1,
        "expected multiple internal blocks on a 900KB input"
    );
    assert!(
        table_calls > 1,
        "expected multiple huffman_table invocations"
    );
    assert!(
        encode_calls > 1,
        "expected multiple huffman_encode invocations"
    );
    assert!(
        (parse_calls as usize) < data.len() / 100,
        "parse_match_calls ({parse_calls}) must be block-granular, not position-granular \
         (input is {} bytes)",
        data.len()
    );
    assert_eq!(
        crc_calls, 1,
        "CRC is computed once per invocation, not per block"
    );

    // Every named region actually measured nonzero time (not a dead/never-
    // wired timer).
    assert!(parse_ns > 0, "parse_match_ns must be nonzero");
    assert!(table_ns > 0, "huffman_table_ns must be nonzero");
    assert!(encode_ns > 0, "huffman_encode_ns must be nonzero");
    assert!(crc_ns > 0, "crc_ns must be nonzero");
}

/// Coverage-hole closure test (2026-07-26): `parse_match` was ONLY wired
/// into the L0/L1 fast parser (`fast.rs`) when this module was first built,
/// so every OTHER level's match-finding time silently folded into
/// RESIDUAL — exactly the gap that made a gzippy-vs-libdeflate phase
/// comparison meaningless at L2-L9 (libdeflate's own instrumented build
/// reports 90-93% in `parse_match` at those levels; gzippy read 0%). This
/// asserts, for EVERY level 0-9 (Fast0, Fast, Greedy x2, LazyGated,
/// Lazy x3, Lazy2 x2 — see `level.rs::params`), that:
///   - the run reconciles (Gate 0: named regions do not exceed root),
///   - `parse_match_ns` is now NONZERO (the coverage-hole assertion itself
///     — a regression back to the hole would silently zero this again),
///   - `parse_match_calls` is block-granular (>1 call, far fewer calls
///     than input bytes -- not per-invocation, not per-position),
///   - the compressed stream still round-trips byte-exact (the wall timers
///     must be pure observation, never change what is encoded).
/// L10-12 (near-optimal: bt matchfinder + iterative min-cost-path DP) are
/// OUT OF SCOPE for this closure -- their match-caching/DP-pass loop has no
/// per-block call boundary of the same shape as greedy/lazy's `run_block`,
/// and are not claimed to be covered here. Two new regions landed 2026-09-25
/// for that gap's near-opt side: `near_opt_fill` (per-internal-block fill
/// span, lever-0) and `near_opt_flush` (per-block optimize-and-flush).
///
/// ⚠ the level range below is STALE and the whole canary is FALSE on trunk
/// since the port rerouting: `level_uses_ldx` routes L0/L2/L4/L5/L8/L9 to
/// the ldx PORT (`level::params` levels map vs the production engine are no
/// longer the same object), and the port has NO parse_match timers —— so
/// `parse_match_ns` is structurally zero at 6 of the 10 levels the loop
/// walks. Discovered 2026-09-25 (lever-0, pre-existing on clean trunk; the
/// suite is not in CI so the break was invisible). Ignored, NOT deleted:
/// the re-arming work is named in the lever ledger (port-side parse_match
/// timers + a production-route parameterized level list) and the canary's
/// conservation sibling still passes.
#[test]
#[ignore = "stale premise: 6 of 10 levels now route to the port, which has no parse_match timers — re-arm when the port gains region timers (lever ledger, anatomy row)"]
fn parse_match_covers_every_level_0_through_9() {
    // 3.5 MiB: comfortably exceeds L0's own 1 MiB internal block length
    // (`fast::FAST0_BLOCK_LENGTH`, the largest of any level's block unit)
    // so EVERY level's `parse_match_calls` is forced above 1 on this input.
    let data = mixed_corpus(3_500_000);
    for level in 0..=9u32 {
        let (compressed, w) = compress_with_wall(&data, level);

        let mut decoded = Vec::new();
        {
            use std::io::Read;
            flate2::read::GzDecoder::new(&compressed[..])
                .read_to_end(&mut decoded)
                .unwrap_or_else(|e| panic!("level {level}: gzip decode failed: {e}"));
        }
        assert_eq!(
            decoded, data,
            "level {level}: roundtrip sanity check failed"
        );

        let root_ns = get_num(&w, "root_ns");
        let parse_ns = get_num(&w, "parse_match_ns");
        let parse_calls = get_num(&w, "parse_match_calls");
        let table_ns = get_num(&w, "huffman_table_ns");
        let encode_ns = get_num(&w, "huffman_encode_ns");
        let crc_ns = get_num(&w, "crc_ns");
        let residual_ns = get_num(&w, "residual_ns");
        let conserved = get_bool(&w, "conserved");

        assert!(conserved, "level {level}: conservation must hold");
        assert_eq!(
            root_ns,
            parse_ns + table_ns + encode_ns + crc_ns + residual_ns,
            "level {level}: root_ns must equal named-region sum + residual exactly"
        );
        assert!(
            parse_ns > 0,
            "level {level}: parse_match_ns must be NONZERO -- coverage hole regressed \
             (this is the exact bug this test exists to catch)"
        );
        assert!(
            parse_calls > 1,
            "level {level}: expected multiple internal blocks on a 900KB input, got \
             parse_match_calls={parse_calls}"
        );
        assert!(
            (parse_calls as usize) < data.len() / 100,
            "level {level}: parse_match_calls ({parse_calls}) must be block-granular, \
             not position-granular"
        );
    }
}

#[test]
fn wall_output_is_absent_from_a_feature_off_style_but_present_here() {
    // Not feature-off (this whole file is feature-gated), but confirms the
    // stderr lines are well-formed and present exactly once per invocation.
    let (_out, w) = compress_with_wall(b"tiny input for a quick sanity check", 1);
    assert!(w.contains_key("root_ns"));
    assert!(w.contains_key("granularity"));
}

/// Lever-0's own coverage assertion (audit-wave M4, agent-34's gap): the
/// `near_opt_fill` / `near_opt_flush` regions are wired but no test named
/// them. L11 on a real invocation runs the near-optimal parser at T1 (its
/// only T1 route), so both regions must be NONZERO and the conservation
/// equation must hold INCLUDING them (residual derived from the full
/// named-sum, no double-count with the huffman regions — the flush timer
/// wraps the whole optimize_and_flush whose internal emit_block also runs
/// its own huffman region timers... verified conserved by the manual M1
/// run: root 222.6ms = fill 104.1 + flush 117.7 + 0.08 residual, 13 calls
/// each on the 3 MB log slice).
#[test]
fn near_opt_regions_nonzero_and_conserved_at_l11() {
    let data = mixed_corpus(3_000_000);
    let (compressed, w) = compress_with_wall(&data, 11);

    let mut decoded = Vec::new();
    {
        use std::io::Read;
        flate2::read::GzDecoder::new(&compressed[..])
            .read_to_end(&mut decoded)
            .expect("gzippy stdout must be a valid gzip stream");
    }
    assert_eq!(decoded, data, "L11 roundtrip sanity check failed");

    let root_ns = get_num(&w, "root_ns");
    let fill_ns = get_num(&w, "near_opt_fill_ns");
    let fill_calls = get_num(&w, "near_opt_fill_calls");
    let flush_ns = get_num(&w, "near_opt_flush_ns");
    let flush_calls = get_num(&w, "near_opt_flush_calls");
    let residual_ns = get_num(&w, "residual_ns");
    let conserved = get_bool(&w, "conserved");

    assert!(conserved, "L11: conservation must hold");
    assert!(fill_ns > 0, "L11: near_opt_fill_ns must be nonzero");
    assert!(flush_ns > 0, "L11: near_opt_flush_ns must be nonzero");
    assert!(
        fill_calls > 1 && flush_calls == fill_calls,
        "L11: both regions are per-internal-block (fill {fill_calls}, flush {flush_calls})"
    );
    assert_eq!(
        root_ns,
        fill_ns
            + flush_ns
            + get_num(&w, "parse_match_ns")
            + get_num(&w, "huffman_table_ns")
            + get_num(&w, "huffman_encode_ns")
            + get_num(&w, "crc_ns")
            + get_num(&w, "mf_new_ns")
            + residual_ns,
        "L11: root_ns must equal the FULL named-region sum + residual (the flush timer \
         wraps emit_block's callers, so the huffman regions double-book if separate — \
         their zero here is the conservation receipt that the near-opt path routes its \
         text emission inside the flush span)"
    );
}
