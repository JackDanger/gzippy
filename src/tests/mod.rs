// Test infrastructure. All content is cfg(test) only.
// alloc_counter installs a #[global_allocator] for per-thread allocation counting.

#[macro_use]
pub mod utils;        // assert_slices_eq! and other test macros
pub mod alloc_counter; // counting allocator — #[global_allocator] in test builds

pub mod datasets;    // benchmark corpus (silesia, enwik, etc.)
pub mod fixtures;    // generated test fixtures (deterministic random data)

pub mod alloc_budget;
pub mod compress_oracle;
pub mod correctness;
pub mod diff_ratio;
pub mod golden;
pub mod hot_path;
pub mod inflate_oracle;
pub mod pipeline;
pub mod routing;
