//! PERF-SHAPE pins: the deterministic shadow of parallel efficiency, plus the
//! charter's zero-per-chunk-allocation rule, as feature-gated tests.
//!
//! Wall-clock never goes in a unit test (proven flaky); counted work is the
//! perf proxy here, with the standing calibration receipt: counters price
//! DIRECTION, not magnitude (a measured -21% Ir moved wall 5.4% while a -27%
//! write count moved wall 0%). Two invariant families:
//!
//! 1. **Parallel-overhead budget** (`parallel_overhead_budget`): for every
//!    synthetic fixture (src/fixtures.rs) x levels {1,2,6,9}, measure TOTAL
//!    counter work — the counters are one process-global set of atomics, so
//!    an in-process T4 run sums over all threads/chunks by construction — at
//!    T1 (`encode_gzip_bytes_to_vec`, the production T1 entry point, same as
//!    tests/anatomy_pins.rs) and at T4
//!    (`PipelinedGzEncoder::compress_buffer_pure`, the SOLE production T>1
//!    compress path — `src/compress/mod.rs` routes every T>1 CLI/library
//!    invocation there; compressed from MEMORY, never stdin pipes — the
//!    stdin trap silently routes T1 at every -p). The per-counter T4/T1
//!    ratio is pinned as a ONE-SIDED CEILING in
//!    tests/fingerprints/perf_shape.tsv: current ratio <= pinned ratio.
//!    LESS overhead passes and prompts a deliberate pin-tightening message
//!    (the seam-tax convention from tests/t4_pins.rs). Duplicated per-chunk
//!    work — e.g. the per-chunk dictionary re-hash that seeds each chunk's
//!    matchfinder — moves this ratio instantly.
//!
//!    NOTE the ratio is NOT pure orchestration overhead: T>1 deliberately
//!    routes through `level::params_parallel` (a stronger parse the T4 wall
//!    slack pays for — see `deflate_into`), so part of the pinned ratio is
//!    that intentional extra work. The pin still serves its purpose: any
//!    CHANGE in duplicated/extra work per chunk moves the ratio and fails
//!    the ceiling (or invites a tightening).
//!
//! 2. **Allocation invariance** (`allocation_invariance`): the charter's
//!    "zero per-chunk allocation" rule as a test. One compressible fixture
//!    ("text"), same seeded generator (`fixtures::generate_sized`), at 1 MiB
//!    and 8 MiB, T1 and T4, levels {1,6}: `alloc_events` must be O(1) in
//!    input size at T1 (MEASURED: zero growth from 1 MiB to 8 MiB); at T4
//!    the MEASURED current behavior is ~3.3-3.7 counted allocation events
//!    per chunk, dominated by the per-chunk padded `[dict|data|pad]` copy
//!    buffer (see the constants below for the mechanism and the encoded
//!    tightest-true bound — allocation is NOT O(1) at T4 today). `alloc_bytes`
//!    legitimately scales linearly with input (the padded [data|pad] work
//!    buffer and the output-capacity estimate are both O(n)), so bytes are
//!    pinned as a bytes-per-input-byte ceiling instead — said here so nobody
//!    mistakes the linear term for a leak.
//!
//! ## Determinism gate
//!
//! A T4 counter total is only pinnable if it is deterministic across two
//! identical in-process runs. Every cell is measured TWICE at T4 (and at T1)
//! and any counter that disagrees must be listed in
//! `T4_NONDETERMINISTIC_COUNTERS` with the reason — the gate fails loudly on
//! an unlisted one, so nondeterminism is a finding, not a flake. As of pin
//! generation, ALL exposed counters are deterministic at T4: the chunk grid
//! is a pure function of (input_len, threads, level), each chunk's work is a
//! pure function of (chunk bytes, dictionary bytes), and worker scheduling
//! only changes WHICH thread does the work, never how much of it there is.
//!
//! ## Why in-process, and why the shared lock
//!
//! `AnatomyCounters` is one process-wide static; `cargo test` runs the tests
//! in this binary concurrently. Both tests take `COUNTER_LOCK` for their
//! whole measurement span, so reset/measure/snapshot never interleave.
//! (tests/anatomy_pins.rs solves this with a single `#[test]`; two distinct
//! invariant families deserve two named tests, hence the mutex.)
//!
//! To regenerate the ratio pins after an INTENTIONAL change (the TSV diff is
//! the lever's mechanism, reviewable line by line — commit it in the same
//! PR):
//!
//! ```text
//! UPDATE_PERF_SHAPE=1 cargo test --release --test perf_shape --features anatomy-counters
//! ```
//!
//! (An env var in TEST tooling is the standard snapshot-test pattern; the
//! no-env-knobs rule binds the SHIPPED binary, and this entire file plus the
//! counters it reads do not exist in a default build.)

#![cfg(feature = "anatomy-counters")]

use gzippy::compress::deflate::anatomy_counters;
use gzippy::compress::deflate::encode_gzip_bytes_to_vec;
use gzippy::compress::pipelined::PipelinedGzEncoder;
use gzippy::fixtures;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;

/// The pinned level grid for the overhead-budget family.
const LEVELS: &[u32] = &[1, 2, 6, 9];

/// T>1 thread count under test. 4 matches the campaign's canonical T4
/// coordinate (the board's failing-class thread count).
const T4: usize = 4;

const REGEN_CMD: &str =
    "UPDATE_PERF_SHAPE=1 cargo test --release --test perf_shape --features anatomy-counters";

/// Counters whose T4 totals are NOT deterministic across identical runs,
/// excluded from the pin surface — each entry must name why. The determinism
/// gate fails loudly on any unlisted disagreement.
///
/// EMPTY as of pin generation: every exposed counter was measured
/// deterministic at T4 (two identical in-process runs, every fixture x level
/// — see the module doc for why the work volume is schedule-independent).
const T4_NONDETERMINISTIC_COUNTERS: &[&str] = &[];

/// Serializes the two tests in this binary against the one process-wide
/// counter static. Poisoning is ignored deliberately: a panic in one test
/// must not turn the other into a spurious failure.
static COUNTER_LOCK: Mutex<()> = Mutex::new(());

fn pins_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fingerprints/perf_shape.tsv")
}

fn update_mode() -> bool {
    std::env::var("UPDATE_PERF_SHAPE").as_deref() == Ok("1")
}

/// Parse the flat `{"key":123,...}` JSON `AnatomyCounters::to_json` emits
/// (same shape, same parser rationale as tests/anatomy_pins.rs).
fn parse_flat_json(s: &str) -> BTreeMap<String, u64> {
    let body = s
        .trim()
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or_else(|| panic!("not a flat JSON object: {s}"));
    let mut map = BTreeMap::new();
    for pair in body.split(',').filter(|p| !p.is_empty()) {
        let (k, v) = pair
            .split_once(':')
            .unwrap_or_else(|| panic!("malformed key:value pair {pair:?}"));
        map.insert(
            k.trim().trim_matches('"').to_string(),
            v.trim()
                .parse()
                .unwrap_or_else(|_| panic!("non-integer value for {k}: {v:?}")),
        );
    }
    map
}

/// Production T1 compression, in-process (the entry point anatomy_pins pins).
fn compress_t1(data: &[u8], level: u32) -> Vec<u8> {
    encode_gzip_bytes_to_vec(data, level)
}

/// Production T>1 compression, in-process, from memory. This is the function
/// `src/compress/mod.rs::compress_with_pipeline_sized` hands every T>1
/// invocation to (route `PURE_PARALLEL_PIPELINE`) — standard single-member
/// gzip with sync-flush chunk seams.
fn compress_t4(data: &[u8], level: u32) -> Vec<u8> {
    let encoder = PipelinedGzEncoder::new(level, T4);
    let mut out = Vec::new();
    encoder
        .compress_buffer_pure(data, &mut out)
        .expect("compress_buffer_pure failed");
    out
}

/// Reset the global counters, run `compress`, snapshot every counter, THEN
/// verify the stream roundtrips (decode after the snapshot so the decoder
/// can never perturb what we pinned).
fn measure(data: &[u8], compress: impl Fn() -> Vec<u8>, what: &str) -> BTreeMap<String, u64> {
    anatomy_counters::reset();
    let compressed = compress();
    let snapshot = parse_flat_json(&anatomy_counters::COUNTERS.to_json());

    let mut decoded = Vec::new();
    {
        use std::io::Read;
        flate2::read::GzDecoder::new(&compressed[..])
            .read_to_end(&mut decoded)
            .unwrap_or_else(|e| panic!("{what}: invalid gzip stream: {e}"));
    }
    assert_eq!(decoded, data, "{what}: roundtrip failed");
    snapshot
}

/// Measure one arm TWICE and demand exact determinism on every counter not
/// explicitly excluded. Returns the (excluded-counters-removed) snapshot.
fn measure_deterministic(
    data: &[u8],
    compress: impl Fn() -> Vec<u8>,
    what: &str,
) -> BTreeMap<String, u64> {
    let a = measure(data, &compress, what);
    let b = measure(data, &compress, what);
    let mut disagreements: Vec<String> = a
        .iter()
        .filter(|(k, va)| b.get(*k) != Some(va))
        .map(|(k, va)| {
            format!(
                "    {k}: run A {va}, run B {}",
                b.get(k).copied().unwrap_or(0)
            )
        })
        .filter(|line| {
            !T4_NONDETERMINISTIC_COUNTERS
                .iter()
                .any(|ex| line.trim_start().starts_with(&format!("{ex}:")))
        })
        .collect();
    disagreements.sort();
    assert!(
        disagreements.is_empty(),
        "\nNONDETERMINISTIC COUNTER(S) at {what} — two identical in-process runs disagreed\n\
         and the counter is not listed in T4_NONDETERMINISTIC_COUNTERS:\n{}\n\
         A pinnable counter must be exact. Find the race, or exclude the counter in\n\
         T4_NONDETERMINISTIC_COUNTERS with a comment naming why.\n",
        disagreements.join("\n")
    );
    let mut out = a;
    for ex in T4_NONDETERMINISTIC_COUNTERS {
        out.remove(*ex);
    }
    out
}

fn warm_up() {
    // Retire once-per-PROCESS work (OnceLock statics: StaticCodes etc. — the
    // exact 50-vs-48 trap anatomy_pins documents) on BOTH paths at every
    // pinned level before any measurement.
    for &level in LEVELS {
        let _ = compress_t1(b"warm-up: retire OnceLock init work", level);
        let _ = compress_t4(b"warm-up: retire OnceLock init work", level);
    }
}

fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn ratio_str(t1: u64, t4: u64) -> String {
    if t1 == 0 {
        if t4 == 0 {
            "0/0".to_string()
        } else {
            format!("{t4}/0 (inf)")
        }
    } else {
        format!("{:.4}", t4 as f64 / t1 as f64)
    }
}

// ---------------------------------------------------------------------------
// Family 1: the parallel-overhead budget.
// ---------------------------------------------------------------------------

/// One pinned row: total counter work at T1 and at T4 for a cell.
#[derive(Clone, Copy)]
struct RatioPin {
    t1: u64,
    t4: u64,
}

type RatioGrid = BTreeMap<(String, u32), BTreeMap<String, RatioPin>>;

fn tsv_header() -> String {
    "# perf_shape.tsv — the parallel-overhead budget: TOTAL counter work (sum over\n\
     # all threads/chunks; the counters are one process-global atomic set) for the\n\
     # production T1 entry point (encode_gzip_bytes_to_vec) and the production T>1\n\
     # pipeline (PipelinedGzEncoder::compress_buffer_pure, 4 threads, from memory)\n\
     # per fixture (src/fixtures.rs) x level {1,2,6,9} x counter.\n\
     #\n\
     # THE ASSERTION IS ONE-SIDED: current T4/T1 ratio <= pinned T4/T1 ratio\n\
     # (cross-multiplied exactly, no floats). Less overhead PASSES and asks for a\n\
     # deliberate pin tightening; more fails naming the counter. Rows where t1=0\n\
     # pin an absolute T4 ceiling instead (a ratio over zero is unbounded).\n\
     # All-zero counters carry no row.\n\
     #\n\
     # Counters price DIRECTION, not magnitude (-21% Ir moved wall 5.4%; -27%\n\
     # writes moved 0%): a ratio move names a mechanism, never a wall verdict.\n\
     #\n\
     # Regenerate (the diff is the change's mechanism — commit it in the same PR):\n\
     #   UPDATE_PERF_SHAPE=1 cargo test --release --test perf_shape --features anatomy-counters\n\
     fixture\tlevel\tcounter\tt1\tt4\n"
        .to_string()
}

fn write_pins(grid: &RatioGrid) {
    let mut tsv = tsv_header();
    for ((fixture, level), counters) in grid {
        for (counter, pin) in counters {
            let _ = writeln!(tsv, "{fixture}\t{level}\t{counter}\t{}\t{}", pin.t1, pin.t4);
        }
    }
    std::fs::write(pins_path(), tsv).expect("write perf_shape.tsv");
}

fn read_pins() -> RatioGrid {
    let tsv = std::fs::read_to_string(pins_path()).unwrap_or_else(|e| {
        panic!(
            "{} missing ({e}) — generate it once:\n    {REGEN_CMD}",
            pins_path().display()
        )
    });
    let mut grid = RatioGrid::new();
    for line in tsv.lines() {
        if line.starts_with('#') || line.starts_with("fixture\t") || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 5, "malformed pin row: {line:?}");
        grid.entry((cols[0].to_string(), cols[1].parse().unwrap()))
            .or_default()
            .insert(
                cols[2].to_string(),
                RatioPin {
                    t1: cols[3].parse().unwrap(),
                    t4: cols[4].parse().unwrap(),
                },
            );
    }
    grid
}

/// `true` iff ratio t4a/t1a <= t4b/t1b, compared exactly by cross-multiplying
/// in u128 (both denominators must be > 0).
fn ratio_le(t4a: u64, t1a: u64, t4b: u64, t1b: u64) -> bool {
    (t4a as u128) * (t1b as u128) <= (t4b as u128) * (t1a as u128)
}

#[test]
fn parallel_overhead_budget() {
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    warm_up();

    // ── Measure every cell: T1 and T4, each twice (the determinism gate). ──
    let mut current = RatioGrid::new();
    for &name in fixtures::NAMES {
        let data = fixtures::generate(name);
        for &level in LEVELS {
            let t1 = measure_deterministic(
                &data,
                || compress_t1(&data, level),
                &format!("{name}:L{level}:T1"),
            );
            let t4 = measure_deterministic(
                &data,
                || compress_t4(&data, level),
                &format!("{name}:L{level}:T4"),
            );
            let mut cell: BTreeMap<String, RatioPin> = BTreeMap::new();
            for (k, &v1) in &t1 {
                let v4 = t4.get(k).copied().unwrap_or(0);
                if v1 > 0 || v4 > 0 {
                    cell.insert(k.clone(), RatioPin { t1: v1, t4: v4 });
                }
            }
            for (k, &v4) in &t4 {
                if !t1.contains_key(k) && v4 > 0 {
                    cell.insert(k.clone(), RatioPin { t1: 0, t4: v4 });
                }
            }
            current.insert((name.to_string(), level), cell);
        }
    }

    if update_mode() {
        write_pins(&current);
        eprintln!(
            "perf_shape: regenerated {} ({} cells)",
            pins_path().display(),
            current.len()
        );
        return;
    }

    // ── Grade every cell against the ceiling. ─────────────────────────────
    let pinned = read_pins();
    let mut failures = Vec::new();
    let mut tightenable = Vec::new();
    for (cell, counters) in &current {
        let Some(pins) = pinned.get(cell) else {
            failures.push(format!(
                "UNPINNED CELL {}:L{}:T4/T1 — no rows in the TSV; regenerate pins:\n    {REGEN_CMD}",
                cell.0, cell.1
            ));
            continue;
        };
        // (counter, pinned, current, ratio_increase) for rows over the ceiling.
        let mut over: Vec<(&str, RatioPin, RatioPin, f64)> = Vec::new();
        let mut under = 0usize;
        for (counter, cur) in counters {
            let Some(pin) = pins.get(counter) else {
                failures.push(format!(
                    "UNPINNED COUNTER {counter} at {}:L{} (t1={}, t4={}) — a new counter \
                     joined the surface; regenerate pins:\n    {REGEN_CMD}",
                    cell.0, cell.1, cur.t1, cur.t4
                ));
                continue;
            };
            let ok = if pin.t1 == 0 {
                // A ratio over zero is unbounded; the pin is an absolute T4
                // ceiling for counters the T1 path never touches.
                cur.t4 <= pin.t4
            } else if cur.t1 == 0 {
                // Pinned T1 work vanished entirely: only an all-zero current
                // cell stays under any finite pinned ratio.
                cur.t4 == 0
            } else {
                ratio_le(cur.t4, cur.t1, pin.t4, pin.t1)
            };
            if !ok {
                let pin_ratio = if pin.t1 == 0 {
                    f64::INFINITY
                } else {
                    pin.t4 as f64 / pin.t1 as f64
                };
                let cur_ratio = if cur.t1 == 0 {
                    f64::INFINITY
                } else {
                    cur.t4 as f64 / cur.t1 as f64
                };
                over.push((counter, *pin, *cur, cur_ratio - pin_ratio));
            } else {
                let strictly_less = if pin.t1 == 0 {
                    cur.t4 < pin.t4
                } else if cur.t1 == 0 {
                    pin.t4 > 0
                } else {
                    !ratio_le(pin.t4, pin.t1, cur.t4, cur.t1)
                };
                if strictly_less {
                    under += 1;
                }
            }
        }
        if under > 0 {
            tightenable.push(format!("{}:L{} ({under} counter(s))", cell.0, cell.1));
        }
        if !over.is_empty() {
            // Sorted by ratio increase, biggest mechanism first.
            over.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
            let mut msg = format!(
                "PARALLEL OVERHEAD GREW {}:L{}:T4/T1 — {} counter(s) over the pinned ceiling\n    \
                 {:<30} {:>16} {:>16} {:>10} {:>10}\n",
                cell.0,
                cell.1,
                over.len(),
                "counter",
                "t1 now",
                "t4 now",
                "pinned",
                "now"
            );
            for (counter, pin, cur, _) in over {
                let _ = writeln!(
                    msg,
                    "    {counter:<30} {:>16} {:>16} {:>10} {:>10}",
                    commas(cur.t1),
                    commas(cur.t4),
                    ratio_str(pin.t1, pin.t4),
                    ratio_str(cur.t1, cur.t4),
                );
            }
            failures.push(msg);
        }
    }
    for cell in pinned.keys() {
        if !current.contains_key(cell) {
            failures.push(format!(
                "STALE PIN CELL {}:L{}:T4/T1 — pinned but no longer measured (fixture or \
                 level grid changed); regenerate pins:\n    {REGEN_CMD}",
                cell.0, cell.1
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "\n{}\n\nThe T4/T1 work ratio grew past its pinned ceiling: some per-chunk work is\n\
         being duplicated (or added) at T>1 that was not before — e.g. an extra\n\
         dictionary re-hash, a widened per-chunk parse. Find the mechanism the\n\
         counters above name, or — if the extra work is a deliberate lever —\n\
         regenerate the pins in the same PR (the diff is your mechanism):\n    {REGEN_CMD}\n\
         Remember: a count ratio is a MECHANISM, not a price — wall verdicts still\n\
         need paired runs on the frozen box.\n",
        failures.join("\n")
    );

    if !tightenable.is_empty() {
        eprintln!(
            "perf_shape: parallel overhead SHRANK below its pinned ceiling on {} cell(s):\n  {}\n\
             An improvement to bank — tighten the pins deliberately in this PR:\n    {REGEN_CMD}",
            tightenable.len(),
            tightenable.join("\n  ")
        );
    }
}

// ---------------------------------------------------------------------------
// Family 2: allocation invariance (the charter's zero-per-chunk-allocation
// rule as a test).
// ---------------------------------------------------------------------------

/// Levels for the allocation family: the shallowest and the charter's default
/// (a shallow AND a deep level — hard stop #3).
const ALLOC_LEVELS: &[u32] = &[1, 6];

const MIB: usize = 1 << 20;
const SMALL: usize = MIB; // 1 MiB
const LARGE: usize = 8 * MIB; // 8 MiB

/// T1 `alloc_events` must be O(1) in input size: growing the input 8x may add
/// at most this many allocation events. MEASURED current behavior at pin time
/// is EXACTLY ZERO growth (both sizes allocate the same fixed set: the output
/// vec + the padded [data|pad] work buffer + the header scratch's first
/// growth); 2 is slack for a legitimately amortized one-off (e.g. a scratch
/// buffer crossing a capacity doubling), NOT for anything per-block or
/// per-chunk — a single extra per-block allocation shows up as hundreds of
/// events and fails loudly.
const T1_EVENT_GROWTH_SLACK: u64 = 2;

/// T4 `alloc_events` per chunk, ceiling. MEASURED current behavior (text,
/// 1 MiB -> 8 MiB, i.e. 4 -> 10 chunks): events grow 20 over 6 extra chunks
/// at L1 (3.33/chunk) and 22 over 6 at L6 (3.67/chunk). The dominant term is
/// the padded `[dict | data | pad]` copy buffer
/// `encode_deflate_segment_to_sink` builds for EVERY chunk
/// (deflate/mod.rs:173-175); the rest is per-chunk-worker scratch growth
/// (e.g. the header scratch's first growth, reused within but not across
/// chunk invocations). This is a REAL, measured violation of the charter's
/// zero-per-chunk-allocation ideal (STEP 1 wording), encoded here as the
/// tightest true bound so any FURTHER per-chunk allocation fails the test —
/// and so that removing the per-chunk copy (a named candidate lever) turns
/// this ceiling slack into a visible tightening opportunity.
const T4_EVENTS_PER_CHUNK_CEILING: u64 = 4;

/// `alloc_bytes` per input byte, ceiling (x1000, integer arithmetic).
/// `alloc_bytes` legitimately scales linearly with input at BOTH thread
/// counts — the padded work buffer is input-sized and the output-capacity
/// estimate is ~len/2 at L1+ (`estimate_output_cap`) — so O(1) bytes is not
/// the invariant; bytes-per-input-byte is. MEASURED at pin time (text,
/// L1/L6): T1 = 1.5004x-1.5006x (one input-sized padded copy + the ~len/2
/// output cap), T4 = 1.036x-1.096x (per-chunk padded copies sum to ~input
/// size + one 32 KiB dictionary prefix per chunk; no whole-file output-cap
/// vec). Ceiling 2.0x: one MORE input-sized copy (worst arm 1.5 -> 2.5)
/// or a doubled output-cap estimate (1.5 -> 2.0006) both fail it.
const BYTES_PER_INPUT_BYTE_CEILING_X1000: u64 = 2000;

/// One measured allocation arm.
struct AllocArm {
    events: u64,
    bytes: u64,
    /// Chunk count, derived from the counters themselves: every non-final
    /// T>1 chunk closes with a sync-flush EMPTY STORED block, and on this
    /// compressible fixture at these levels stored never wins a real block,
    /// so `blocks_emitted_stored + 1 == chunks`. At T1 there are no seams and
    /// stored never wins, so this reads 1 (a single "chunk").
    chunks: u64,
}

fn alloc_arm(data: &[u8], level: u32, threads: usize) -> AllocArm {
    let what = format!("alloc:{}B:L{level}:T{threads}", data.len());
    let snap = measure_deterministic(
        data,
        || {
            if threads == 1 {
                compress_t1(data, level)
            } else {
                compress_t4(data, level)
            }
        },
        &what,
    );
    let get = |k: &str| {
        snap.get(k)
            .copied()
            .unwrap_or_else(|| panic!("missing counter {k}"))
    };
    AllocArm {
        events: get("alloc_events"),
        bytes: get("alloc_bytes"),
        chunks: get("blocks_emitted_stored") + 1,
    }
}

#[test]
fn allocation_invariance() {
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    warm_up();

    let small = fixtures::generate_sized("text", SMALL);
    let large = fixtures::generate_sized("text", LARGE);

    let mut failures = Vec::new();
    for &level in ALLOC_LEVELS {
        for threads in [1usize, T4] {
            let a = alloc_arm(&small, level, threads);
            let b = alloc_arm(&large, level, threads);
            let coord = format!("text:L{level}:T{threads}");
            // Visible under --nocapture: the measured arms behind the bounds,
            // so re-deriving the constants never needs an edit-and-fail loop.
            eprintln!(
                "perf_shape alloc {coord}: {SMALL} B -> events {}, bytes {}, chunks {}; \
                 {LARGE} B -> events {}, bytes {}, chunks {}",
                a.events, a.bytes, a.chunks, b.events, b.bytes, b.chunks
            );

            // ── alloc_events: O(1) at T1; O(chunks) at T4, per-chunk-capped. ──
            let event_growth = b.events.saturating_sub(a.events);
            let chunk_growth = b.chunks.saturating_sub(a.chunks);
            let implied_per_chunk = if chunk_growth > 0 {
                event_growth as f64 / chunk_growth as f64
            } else {
                f64::NAN
            };
            if threads == 1 {
                if event_growth > T1_EVENT_GROWTH_SLACK {
                    failures.push(format!(
                        "ALLOCATION GROWTH {coord}: alloc_events is not O(1) in input size\n    \
                         alloc_events   {SMALL} B input: {}   {LARGE} B input: {}   growth: {} \
                         (allowed slack: {T1_EVENT_GROWTH_SLACK})\n    \
                         implied per-chunk allocations: {:.2} (chunks {} -> {})\n    \
                         Every allocation must be O(1) in input size at T1 (the charter's\n    \
                         zero-per-chunk-allocation rule) — something now allocates per block,\n    \
                         per chunk, or per input byte.",
                        commas(a.events),
                        commas(b.events),
                        commas(event_growth),
                        implied_per_chunk,
                        a.chunks,
                        b.chunks,
                    ));
                }
            } else {
                // T4: events/chunk is the measured contract (see
                // T4_EVENTS_PER_CHUNK_CEILING for the mechanism).
                let ceiling = a.events + T4_EVENTS_PER_CHUNK_CEILING * chunk_growth;
                if b.events > ceiling {
                    failures.push(format!(
                        "ALLOCATION GROWTH {coord}: per-chunk alloc_events over ceiling\n    \
                         alloc_events   {SMALL} B input: {}   {LARGE} B input: {}   growth: {}\n    \
                         chunks         {} -> {} (growth {})\n    \
                         implied per-chunk allocations: {:.2}   ceiling: {} per chunk\n    \
                         The T>1 pipeline budget is {} counted allocation(s) per chunk (the\n    \
                         padded [dict|data|pad] copy buffer + header scratch — see\n    \
                         T4_EVENTS_PER_CHUNK_CEILING). A NEW per-chunk allocation joined it.",
                        commas(a.events),
                        commas(b.events),
                        commas(event_growth),
                        a.chunks,
                        b.chunks,
                        chunk_growth,
                        implied_per_chunk,
                        T4_EVENTS_PER_CHUNK_CEILING,
                        T4_EVENTS_PER_CHUNK_CEILING,
                    ));
                }
            }

            // ── alloc_bytes: linear in input by design; pin the slope. ──
            for (len, arm, size_name) in [(SMALL, &a, "small"), (LARGE, &b, "large")] {
                let ceiling = (len as u64) * BYTES_PER_INPUT_BYTE_CEILING_X1000 / 1000;
                if arm.bytes > ceiling {
                    failures.push(format!(
                        "ALLOCATION VOLUME {coord} ({size_name}): alloc_bytes per input byte over ceiling\n    \
                         alloc_bytes {} for {} input bytes = {:.3} bytes/input byte \
                         (ceiling {:.3})\n    \
                         alloc_bytes scales linearly by DESIGN (padded work buffer + output\n    \
                         cap estimate); this failure means the input is being copied at least\n    \
                         one more whole time than at pin time.",
                        commas(arm.bytes),
                        commas(len as u64),
                        arm.bytes as f64 / len as f64,
                        BYTES_PER_INPUT_BYTE_CEILING_X1000 as f64 / 1000.0,
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "\n{}\n\nAllocation shape changed. These bounds encode MEASURED current behavior\n\
         (see the constants' doc comments for the mechanism behind each number);\n\
         if a deliberate change legitimately moves one, update the constant in the\n\
         same PR with a comment naming the new mechanism.\n",
        failures.join("\n\n")
    );
}
