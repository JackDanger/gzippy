// Experimental parallel decompression paths — none are wired into production.
// See CLAUDE.md "Experiments" section for performance measurements and caveats.
//
// Speculative parallel (parallel_single_member): 88–148 MB/s vs 600–2000 MB/s sequential.
// arm64 warning: speculative paths are 16× slower on low-redundancy data.

pub mod block_finder;
pub mod block_finder_lut;
pub mod bmi2;
pub mod consume_first_decode;
pub mod consume_first_table;
pub mod double_literal;
pub mod jit_decode;
pub mod libdeflate_decode;
pub mod libdeflate_entry;
pub mod marker_decode;
pub mod parallel_single_member;
pub mod specialized_decode;
pub mod speculative_parallel;
pub mod ultra_fast_inflate;
pub mod ultra_inflate;
pub mod vector_huffman;
