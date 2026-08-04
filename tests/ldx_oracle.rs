//! Invariants of the ldx per-decision divergence oracle
//! (`src/compress/ldx_oracle.rs`, driven by `examples/ldx_divergence.rs`).
//!
//! Inputs are the SYNTHETIC fixtures from `src/fixtures.rs` only — never
//! corpus files (this is an open-source repo; no private data).
//!
//! Three invariants:
//! (a) ldx-vs-ldx divergence is zero — the oracle cannot invent a diff;
//! (b) when our output bytes equal ldx's, the oracle reports zero divergence
//!     (consistency: byte-identity implies decision-identity);
//! (c) when the outputs differ, the first divergence has a position within
//!     the input and the class breakdown is non-empty.

use gzippy::compress::ldx_oracle::{compare, tokenize};
use gzippy::compress::{deflate::encode_deflate_bytes_to_vec, ldx::compress_for_diff};
use gzippy::fixtures;

/// Levels exercised: a stored level, the ht path (L1), greedy (2-3), lazy
/// (6), and near-optimal (9). Full 0-9 would also pass but costs runtime.
const LEVELS: &[u32] = &[0, 1, 2, 3, 6, 9];

/// (a) The oracle reports zero divergence when both streams are ldx itself.
#[test]
fn ldx_vs_ldx_divergence_is_zero() {
    for &name in fixtures::NAMES {
        let input = fixtures::generate(name);
        for &level in LEVELS {
            let ldx = compress_for_diff(level, &input)
                .unwrap_or_else(|| panic!("level {level} not ported"));
            let r = compare(&ldx, &ldx).expect("token walk failed");
            assert!(
                r.is_zero(),
                "fixture {name} L{level}: ldx-vs-ldx reported {} divergent positions",
                r.total_divergent()
            );
            assert!(r.first.is_none(), "fixture {name} L{level}");
        }
    }
}

/// (b) Byte-identical outputs must mean zero reported divergence, and
/// (c) differing outputs must mean a first divergence inside the input with
/// a non-empty class breakdown. Every (fixture, level) pair lands in exactly
/// one of the two arms, so both invariants are exercised by one sweep.
#[test]
fn byte_identity_implies_zero_divergence_and_diffs_are_located() {
    let mut identical_cells = 0u32;
    let mut divergent_cells = 0u32;
    for &name in fixtures::NAMES {
        let input = fixtures::generate(name);
        for &level in LEVELS {
            let ldx = compress_for_diff(level, &input)
                .unwrap_or_else(|| panic!("level {level} not ported"));
            let ours = encode_deflate_bytes_to_vec(&input, level);
            let r = compare(&ours, &ldx).expect("token walk failed");
            if ours == ldx {
                identical_cells += 1;
                assert!(
                    r.is_zero(),
                    "fixture {name} L{level}: bytes identical but oracle reported {} divergent positions (first: {:?})",
                    r.total_divergent(),
                    r.first
                );
            } else {
                divergent_cells += 1;
                let f = r.first.unwrap_or_else(|| {
                    panic!(
                        "fixture {name} L{level}: outputs differ ({} vs {} B) but oracle found no divergent decision",
                        ours.len(),
                        ldx.len()
                    )
                });
                assert!(
                    (f.pos as usize) <= input.len(),
                    "fixture {name} L{level}: first divergence pos {} > input len {}",
                    f.pos,
                    input.len()
                );
                let class_total = r.we_literal_they_match
                    + r.we_match_they_literal
                    + r.both_match_different_len
                    + r.both_match_different_dist
                    + r.block_boundary;
                assert!(
                    class_total > 0,
                    "fixture {name} L{level}: outputs differ but class breakdown is empty"
                );
                assert!(
                    r.total_divergent() >= class_total,
                    "fixture {name} L{level}: total < class sum"
                );
            }
        }
    }
    // The sweep must have exercised at least one arm each way OR be all-tie;
    // record which so a silent routing change is visible in test output.
    eprintln!(
        "oracle sweep over synthetic fixtures {:?} x levels {LEVELS:?}: {identical_cells} byte-identical cells, {divergent_cells} divergent cells",
        fixtures::NAMES
    );
    assert_eq!(
        identical_cells + divergent_cells,
        (fixtures::NAMES.len() * LEVELS.len()) as u32
    );
}

/// The tokenizer itself must account for every input byte: token coverage
/// (sum of literal/match lengths, positions strictly increasing) equals the
/// input length, on both encoders' outputs.
#[test]
fn tokenizer_accounts_for_every_byte() {
    let input = fixtures::generate("text");
    for &level in &[1u32, 6] {
        for stream in [
            encode_deflate_bytes_to_vec(&input, level),
            compress_for_diff(level, &input).unwrap(),
        ] {
            let (tokens, stats) = tokenize(&stream).expect("token walk failed");
            assert_eq!(stats.total_uncompressed as usize, input.len(), "L{level}");
            let mut pos = 0u64;
            for t in &tokens {
                assert_eq!(t.pos, pos, "L{level}: token starts must tile the input");
                pos += if t.is_literal() { 1 } else { t.len as u64 };
            }
            assert_eq!(pos as usize, input.len(), "L{level}");
            assert_eq!(
                stats.block_uncompressed_lens.iter().sum::<u64>() as usize,
                input.len(),
                "L{level}: block sizes must sum to the input"
            );
        }
    }
}
