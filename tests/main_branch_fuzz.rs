//! Main-branch-only comprehensive fuzz harness for the v0.6 marker pipeline.
//!
//! # Why this lives here, not in `src/decompress/parallel/deflate_block.rs`
//!
//! The unit-test module already has `fuzz_diff_against_oracle` with 200 trials
//! that catches generic decoder regressions. This file is a deliberately
//! louder, more comprehensive sibling that runs **only on `push` to main**
//! (CI sets `GZIPPY_MAIN_FUZZ=1` for those builds; PRs leave it empty). On
//! a PR or feature branch the test exits early with a "skipped" message.
//!
//! # The "fails if empty" reminder beacon
//!
//! When CI on main runs this test and the trial counter returns 0, the test
//! **panics with a specific message** naming the work that needs filling
//! in. This is intentional: the test is a checked-in TODO that screams.
//!
//! - **Today (PR #90 follow-up)**: `run_comprehensive_fuzz` returns 0.
//!   CI on main goes red. That's the reminder.
//! - **After it's filled in**: returns the number of trials actually run
//!   (target: ~10⁴ randomized scenarios across block-type bias, bit-
//!   aligned starts, end_bit_limit boundaries, marker pipeline end-to-
//!   end with various window sizes, BTYPE=01 density). CI on main goes
//!   green; PR builds are unaffected.
//!
//! # Failure-mode catalogue this should exercise
//!
//! Drawn from `docs/marker-decoder-premortem.md`:
//!
//! - **A1/A6**: u16 store-width cost and per-thread throughput — assert
//!   end-to-end throughput stays within a band of the oracle's u8 decoder.
//! - **B1/B2**: cross-chunk marker emission for back-refs that straddle
//!   chunk boundaries; marker propagation through chunk-local copies.
//! - **B3**: RFC-1951 distance > 32768 always produces an explicit error,
//!   never garbage output.
//! - **B4**: leftover markers post-`replace_markers` fail-fast via
//!   `u16_to_u8`.
//! - **B5**: CRC=0 and ISIZE=0 trailers always verify (regression-tested
//!   in `single_member::tests` today; widen to randomized fixtures).
//! - **B7**: false-positive boundary that ISA-L accepts but a full
//!   `deflate_block::Block` trial-decode rejects — synthesize an
//!   adversarial bit-stream and confirm rejection.
//! - **C1**: `MARKER_PIPELINE_RUNS` increments on every successful run
//!   (already covered by routing tests at fixed fixtures; widen here
//!   across randomized inputs that should all take the parallel path).
//! - **BTYPE=01 dense regions**: synthesize inputs where most boundaries
//!   are fixed-Huffman (the failure class blocking the Silesia bench);
//!   assert pipeline still finds them once the BlockFinder enhancement
//!   lands.

#[test]
fn fuzz_diff_against_oracle_comprehensive() {
    // PR-branch / feature-branch path: env var unset OR empty → exit
    // silently. The CI workflow sets GZIPPY_MAIN_FUZZ to '' (empty
    // string, not unset) on non-main branches via GitHub Actions
    // ternary syntax, so `var_os().is_none()` would let those through.
    // Require the value to be exactly "1" to be conservative.
    if std::env::var("GZIPPY_MAIN_FUZZ").ok().as_deref() != Some("1") {
        eprintln!(
            "fuzz_diff_against_oracle_comprehensive: skipped (set GZIPPY_MAIN_FUZZ=1 \
             to run; only fires on `push` to main in CI)"
        );
        return;
    }

    let trials = run_comprehensive_fuzz();

    // The "fail if empty" beacon: if no trials were actually executed, the
    // body hasn't been written yet. Panic with a message that points the
    // reader at the work to do, rather than silently passing.
    assert!(
        trials > 0,
        "fuzz_diff_against_oracle_comprehensive ran 0 trials — this is the \
         placeholder body that landed with PR #90's structural-improvements \
         batch. Fill `run_comprehensive_fuzz()` in `tests/main_branch_fuzz.rs` \
         to exercise the failure-mode catalogue named in the module doc. The \
         premortem (docs/marker-decoder-premortem.md) lists the exact classes \
         this should cover."
    );
}

/// Comprehensive fuzz body. **Stub** — implement before relying on this test.
///
/// Returns the number of trials executed. The `#[test]` above asserts this
/// is non-zero, so an unimplemented body fails CI on main loudly. See the
/// module doc for the scenario list this should cover.
fn run_comprehensive_fuzz() -> usize {
    // TODO: implement. Suggested shape:
    //
    //   let mut trials = 0;
    //   for seed in 0..N {
    //       trials += scenario_random_bytes_round_trip(seed);
    //       trials += scenario_bit_aligned_chunk_boundary(seed);
    //       trials += scenario_btype01_heavy(seed);
    //       trials += scenario_crc_zero_fixture(seed);
    //       // …
    //   }
    //   trials
    //
    // Each scenario asserts byte-identity against the production u8 oracle.
    0
}
