// Test infrastructure. All content is cfg(test) only.
// alloc_counter installs a #[global_allocator] for per-thread allocation counting.

#[macro_use]
pub mod utils; // assert_slices_eq! and other test macros
pub mod alloc_counter; // counting allocator — #[global_allocator] in test builds

pub mod datasets; // benchmark corpus (silesia, enwik, etc.)
pub mod fixtures; // generated test fixtures (deterministic random data)

pub mod alloc_budget;
pub mod compress_oracle;
pub mod correctness;
pub mod deflate_encoder_foundation;
pub mod deflate_encoder_matches;
pub mod diff_multi_oracle;
pub mod diff_ratio;
pub mod golden;
pub mod index;
pub mod inflate_fuzz_loop;
pub mod inflate_oracle;
pub mod inflate_proptest;
pub mod latch_interleave;
pub mod multi_member_chunked;
pub mod phantom_eos_probe;
#[cfg(feature = "pure-rust-encoder")]
pub mod pure_parallel_encoder;
pub mod pure_rust_inflate_corpus;
pub mod resumable_correctness;
pub mod routing;
pub mod seam_crossing;
pub mod selector_byte_transparency;
pub mod three_oracle_diff;
pub mod trace_parity;
