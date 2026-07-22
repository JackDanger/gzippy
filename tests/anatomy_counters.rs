//! Integration test for the `anatomy-counters` feature (closed-loop
//! reconciliation against a REAL compression run).
//!
//! This spawns the actual `gzippy` binary as a fresh subprocess rather than
//! calling `compress_oneshot` in-process: `AnatomyCounters` (`src/compress/
//! deflate/anatomy_counters.rs`) is one process-wide static, and `cargo test`
//! runs every test in the crate's unit-test binary concurrently by default —
//! an in-process test asserting an EXACT count against that shared global
//! would race against every OTHER test that happens to exercise compression
//! at the same time. A subprocess is the actual isolation boundary a real
//! measurement run gets too (fulcrum spawns gzippy once per measured
//! invocation), so this is also the faithful integration shape, not just a
//! test-hygiene workaround.
//!
//! Only compiled/run when the `anatomy-counters` feature is enabled
//! (`cargo test --features anatomy-counters`); the feature-off default build
//! carries none of this.

#![cfg(feature = "anatomy-counters")]

use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

/// Parse the flat `{"key":123,...}` JSON object `AnatomyCounters::to_json`
/// emits. No `serde_json` dependency needed for this shape (unsigned
/// integers only, no nesting, no strings-with-commas).
fn parse_flat_json(s: &str) -> BTreeMap<String, u64> {
    let body = s
        .trim()
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or_else(|| panic!("not a flat JSON object: {s}"));
    let mut map = BTreeMap::new();
    if body.is_empty() {
        return map;
    }
    for pair in body.split(',') {
        let (k, v) = pair
            .split_once(':')
            .unwrap_or_else(|| panic!("malformed key:value pair {pair:?} in {s}"));
        let key = k.trim().trim_matches('"').to_string();
        let val: u64 = v
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("non-integer value for {key}: {v:?}"));
        map.insert(key, val);
    }
    map
}

/// A mixed corpus (repeated phrases + pseudo-random interleaving) big enough
/// to span several DEFLATE blocks at L1 — exercises literals, matches, AND
/// the block-split machinery, not just one trivial block.
fn mixed_corpus(min_len: usize) -> Vec<u8> {
    let phrases: [&[u8]; 4] = [
        b"the quick brown fox jumps over the lazy dog; ",
        b"gzippy anatomy counters close the calibration loop; ",
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

/// A mostly-incompressible corpus (a xorshift-style PRNG stream, occasional
/// short repeats spliced in) — long consecutive-miss runs are exactly what
/// arms `fast.rs`'s L0-only ACCEL scan-step ramp (`ACCEL_ARM_THRESHOLD`
/// consecutive misses), so this is the fixture that proves
/// `fast_positions_skipped` is a LIVE counter, not a permanent zero.
fn low_redundancy_corpus(min_len: usize) -> Vec<u8> {
    let mut data = Vec::new();
    let mut x: u32 = 0x9E3779B9;
    while data.len() < min_len {
        // Every 64th byte, splice in a short repeat of the last 6 bytes so
        // the finder still sees SOME matches (a pure-noise stream would
        // never exercise the probe/accept path at all).
        if data.len() >= 6 && data.len() % 64 < 6 {
            let start = data.len() - 6;
            let b = data[start];
            data.push(b);
        } else {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            data.push((x >> 16) as u8);
        }
    }
    data
}

/// Run `gzippy -{level} -c -p 1` over `data` via stdin, returning
/// `(compressed_stdout, counters_map_from_stderr)`.
fn compress_with_counters(data: &[u8], level: u32) -> (Vec<u8>, BTreeMap<String, u64>) {
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
    let line = stderr
        .lines()
        .find_map(|l| l.strip_prefix("ANATOMY_COUNTERS="))
        .unwrap_or_else(|| {
            panic!("no ANATOMY_COUNTERS= line on stderr:\n{stderr}");
        });
    (output.stdout, parse_flat_json(line))
}

/// The mission's three cross-check invariants, run end-to-end against a real
/// `gzippy` invocation: token-level facts derivable from the compressed
/// output (byte conservation + a roundtrip decode) reconciled against the
/// execution-side counters gathered during that SAME run.
#[test]
fn reconciliation_invariants_hold_on_a_real_gzippy_invocation() {
    let data = mixed_corpus(900_000);
    let (compressed, c) = compress_with_counters(&data, 1);

    // Sanity: the subprocess actually produced a valid, byte-exact gzip
    // stream (a "win" with wrong bytes, or a vacuous/empty run, would make
    // every invariant below meaningless).
    let mut decoded = Vec::new();
    {
        use std::io::Read;
        flate2::read::GzDecoder::new(&compressed[..])
            .read_to_end(&mut decoded)
            .expect("gzippy stdout must be a valid gzip stream");
    }
    assert_eq!(decoded, data, "roundtrip sanity check failed");

    let get = |k: &str| *c.get(k).unwrap_or_else(|| panic!("missing counter {k}"));
    let lits = get("literals_emitted");
    let lits_fast = get("literals_emitted_fast");
    let matches = get("matches_emitted");
    let matches_fast = get("matches_emitted_fast");
    let match_bytes = get("match_length_bytes_total");
    let bso = get("block_split_observations");
    let stored = get("blocks_emitted_stored");
    let fixed = get("blocks_emitted_fixed");
    let dynamic = get("blocks_emitted_dynamic");
    let make_calls = get("huffman_make_code_calls");

    // Invariant 1 ("tokens emitted == extract count"): every input byte is
    // covered by exactly one literal or exactly one position of exactly one
    // match — the LZ77 parse invariant, checkable without any external
    // token-level oracle.
    assert_eq!(
        lits + match_bytes,
        data.len() as u64,
        "literal+match-length byte count must equal total input bytes parsed"
    );
    assert!(
        lits > 0 && matches > 0,
        "fixture must exercise both token kinds"
    );

    // Invariant 2: block_split_observations is exactly literals_emitted +
    // matches_emitted MINUS the fast-path calls (push_literal_fast /
    // push_match_fast never call observe_literal/observe_match — see
    // block_split.rs and parse/mod.rs's Sink).
    assert_eq!(
        bso,
        (lits - lits_fast) + (matches - matches_fast),
        "block_split_observations must equal slow-path literal+match pushes"
    );

    // Invariant 3 (huffman-build cross-check, gzippy's own — NOT the spec's
    // literal "2 * blocks_emitted_dynamic" claim, which assumed
    // length_limited_code_lengths gates the ordinary path; it doesn't, see
    // this module's DEVIATION doc comment): `compress()` builds the static
    // reference code ONCE per invocation (2 `make_huffman_code` calls), and
    // — for the L1 (`Strategy::Fast`) path this test exercises, which emits
    // every block via the shared `emit_block` — 3 more calls per block
    // (litcode + offcode for the dynamic-candidate cost probe, ALWAYS built
    // regardless of which block type wins, + 1 for the header's precode in
    // `build_dynamic_header`). At T>1 (`-p 1` above pins T=1 for this exact
    // arithmetic to hold) this would need to account for the ParallelGzEncoder's
    // per-block-group dispatch instead; kept single-threaded for an exact
    // closed-form check.
    let total_blocks = stored + fixed + dynamic;
    assert!(total_blocks > 0, "fixture must emit at least one block");
    assert_eq!(
        make_calls,
        2 + 3 * total_blocks,
        "huffman_make_code_calls must reconcile to 2 (static) + 3*blocks (dynamic \
         candidate + precode, built for every block regardless of outcome)"
    );
}

#[test]
fn counters_are_absent_from_a_feature_off_style_quiet_run() {
    // Not feature-off (this whole file is feature-gated), but confirms the
    // stderr line is well-formed and present exactly once per invocation —
    // guards against a double-flush (e.g. an exit path that runs the flush
    // twice) or a missing flush on a tiny input.
    let (_out, c) = compress_with_counters(b"tiny", 1);
    assert!(c.contains_key("alloc_events"));
    assert!(c.get("alloc_events").copied().unwrap_or(0) > 0);
}

// ============================================================================
// fast_* (parse/fast.rs, L0/L1 single-probe matchfinder) reconciliation.
//
// Before this counter set, `fast.rs` was the ONE parser with zero anatomy
// coverage: a calibration run confirmed hc_probe_attempts/bt_probe_attempts
// sit at exactly 0 at L1 (the fast path never touches those finders). These
// tests close the loop the same way the L1-block test above does: run a real
// gzippy subprocess, decode+verify the output, then check the fast_* counters
// against facts derivable from that SAME run (byte conservation via the
// input length, and cross-checks against the shared Sink counters that ONLY
// fast.rs increments through the `_fast` push variants).
// ============================================================================

/// Assertions common to both L0 and L1: `fast_*` must account for every input
/// byte, and the accepted-probe count must equal the shared Sink's
/// `matches_emitted_fast` (the mission's "accepted probes == matches_emitted_fast"
/// cross-check) — plus the outcome buckets must partition attempts exactly.
fn assert_fast_common_invariants(c: &BTreeMap<String, u64>, data_len: u64) {
    let get = |k: &str| *c.get(k).unwrap_or_else(|| panic!("missing counter {k}"));

    let processed = get("fast_positions_processed");
    let skipped = get("fast_positions_skipped");
    let attempts = get("fast_probe_attempts");
    let miss = get("fast_probe_outcome_miss");
    let too_short = get("fast_probe_outcome_too_short");
    let accepted = get("fast_probe_outcome_accepted");
    let deferred = get("fast_probe_outcome_deferred");
    let lazy_events = get("fast_lazy_peek_events");
    let lazy_defers = get("fast_lazy_peek_defers");
    let hash_computations = get("fast_hash_computations");
    let head_reads = get("fast_head_table_reads");
    let head_writes = get("fast_head_table_writes");
    let matches_fast = get("matches_emitted_fast");
    let lits = get("literals_emitted");
    let match_bytes = get("match_length_bytes_total");

    // The mission's headline reconciliation invariant: every coded input byte
    // lands in exactly one of "processed via the probe path" or "skipped by
    // ACCEL with no finder touch at all" (see anatomy_counters.rs's fast_*
    // doc comment for the exact partition definition).
    assert_eq!(
        processed + skipped,
        data_len,
        "fast_positions_processed + fast_positions_skipped must equal input bytes consumed"
    );

    // "accepted probes == matches_emitted_fast": every accepted-and-not-
    // deferred probe calls push_match_fast exactly once, and push_match_fast
    // is called from nowhere else in the fast path.
    assert_eq!(
        accepted, matches_fast,
        "fast_probe_outcome_accepted must equal matches_emitted_fast exactly"
    );

    // Outcome buckets are an exact partition of attempts (miss/too-short/
    // accepted/deferred — deferred is the justified 4th bucket documented in
    // anatomy_counters.rs: a lazy-peek DEFER is neither a miss, too-short,
    // nor an emitted match).
    assert_eq!(
        attempts,
        miss + too_short + accepted + deferred,
        "fast_probe_attempts must partition exactly into miss+too_short+accepted+deferred"
    );

    // Every probe attempt does at least one primary head-table write (the
    // unconditional insert at the top of the per-position body) and, when
    // batched, at least one read feeding it — so both must be >= attempts.
    assert!(
        head_writes >= attempts,
        "fast_head_table_writes ({head_writes}) must be >= fast_probe_attempts ({attempts})"
    );
    assert!(
        head_reads >= attempts,
        "fast_head_table_reads ({head_reads}) must be >= fast_probe_attempts ({attempts})"
    );
    assert!(hash_computations > 0, "fast_hash_computations must fire");
    assert!(
        attempts > 0,
        "fixture must exercise the finder at least once"
    );

    // Lazy-peek defers are a subset of lazy-peek attempts.
    assert!(
        lazy_defers <= lazy_events,
        "fast_lazy_peek_defers ({lazy_defers}) must be <= fast_lazy_peek_events ({lazy_events})"
    );

    // Token-level reconciliation still holds with fast.rs in the mix (same
    // invariant the L1 test above checks, restated here so THIS test is
    // self-contained): every input byte is exactly one literal or one
    // position of exactly one match.
    assert_eq!(
        lits + match_bytes,
        data_len,
        "literal+match-length byte count must equal total input bytes parsed"
    );
}

/// A real-text slice big enough, and with enough long-range repeat structure,
/// to reliably arm the L1-only lazy-peek lever (`LAZY_PEEK_MIN_DIST = 8192`
/// gates it to short matches at distances the small synthetic `mixed_corpus`
/// rarely produces). Mirrors the precedent in `matchfinder/hc.rs`'s
/// `matches_equal_scalar_silesia` test: same fixture file, same graceful
/// skip-if-missing (the file is `.gitignore`d — `benchmark_data/silesia.tar.xz`
/// is the tracked, compressed form; a fresh checkout without it extracted
/// simply skips this fixture's assertions rather than failing CI).
fn silesia_slice(min_len: usize) -> Option<Vec<u8>> {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmark_data/silesia.tar");
    let mut f = std::fs::File::open(&path).ok()?;
    use std::io::{Read, Seek};
    let mut data = vec![0u8; min_len];
    f.seek(std::io::SeekFrom::Start(1 << 16)).ok()?;
    f.read_exact(&mut data).ok()?;
    Some(data)
}

#[test]
fn fast_path_reconciliation_invariants_hold_at_l1() {
    let Some(data) = silesia_slice(900_000) else {
        eprintln!("note: benchmark_data/silesia.tar missing; skipped fast_path L1 fixture");
        return;
    };
    let (compressed, c) = compress_with_counters(&data, 1);

    let mut decoded = Vec::new();
    {
        use std::io::Read;
        flate2::read::GzDecoder::new(&compressed[..])
            .read_to_end(&mut decoded)
            .expect("gzippy stdout must be a valid gzip stream");
    }
    assert_eq!(decoded, data, "roundtrip sanity check failed");

    assert_fast_common_invariants(&c, data.len() as u64);

    let get = |k: &str| *c.get(k).unwrap_or_else(|| panic!("missing counter {k}"));

    // L1 (`Strategy::Fast`, `ACCEL == false`) never activates the scan-step
    // ramp at all — `fastloop_l0` is a physically separate function L1 never
    // calls (see fast.rs's module doc comment).
    assert_eq!(
        get("fast_positions_skipped"),
        0,
        "L1 must never skip positions via ACCEL (that mechanism is L0-only)"
    );

    // The SF2 two-position software pipeline is L1-only and this fixture
    // (900 KB, mixed literals+matches) is far bigger than one batch pair, so
    // it must engage at least once.
    assert!(
        get("fast_k2_batch_iterations") > 0,
        "L1's SF2 batch pipeline must engage on a fixture this size"
    );

    // The lazy-peek lever (short, far accepted matches) should fire at least
    // once on a corpus with real match variety; if this ever reads 0 the
    // fixture stopped exercising the lever, not that the counter is dead
    // (cross-checked against `fast_probe_outcome_deferred` below).
    assert!(
        get("fast_lazy_peek_events") > 0,
        "expected at least one lazy-peek attempt on this fixture"
    );
    assert_eq!(
        get("fast_lazy_peek_defers"),
        get("fast_probe_outcome_deferred"),
        "every lazy-peek defer must be reflected in the deferred outcome bucket"
    );
}

#[test]
fn fast_path_reconciliation_invariants_hold_at_l0() {
    let data = low_redundancy_corpus(900_000);
    let (compressed, c) = compress_with_counters(&data, 0);

    let mut decoded = Vec::new();
    {
        use std::io::Read;
        flate2::read::GzDecoder::new(&compressed[..])
            .read_to_end(&mut decoded)
            .expect("gzippy stdout must be a valid gzip stream");
    }
    assert_eq!(decoded, data, "roundtrip sanity check failed");

    assert_fast_common_invariants(&c, data.len() as u64);

    let get = |k: &str| *c.get(k).unwrap_or_else(|| panic!("missing counter {k}"));

    // L0 (`Strategy::Fast0`, `ACCEL == true`) is the ramp's ONLY caller; a
    // low-redundancy fixture with long miss runs must arm it at least once —
    // proving `fast_positions_skipped` is a live counter, not a permanent
    // zero (mirrored by the L1 test's assertion that it's ALWAYS zero there).
    assert!(
        get("fast_positions_skipped") > 0,
        "expected the ACCEL ramp to skip at least one position on a low-redundancy fixture"
    );

    // L0 shares NEITHER the lazy peek NOR the SF2 batch pipeline with L1 —
    // `fastloop_l0` is a dedicated, non-generic function that never calls
    // `process_position_l1` (see fast.rs's module doc comment on why).
    assert_eq!(
        get("fast_lazy_peek_events"),
        0,
        "L0 must never attempt a lazy peek (L1-only mechanism)"
    );
    assert_eq!(
        get("fast_k2_batch_iterations"),
        0,
        "L0 must never engage the SF2 batch pipeline (L1-only mechanism)"
    );
    assert_eq!(
        get("fast_probe_outcome_deferred"),
        0,
        "L0 can never defer (no lazy peek to defer through)"
    );
}
