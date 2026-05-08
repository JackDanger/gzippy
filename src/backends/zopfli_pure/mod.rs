//! Pure-Rust port of Google Zopfli. Built bottom-up; oracle-tested
//! against the C FFI at `crate::backends::zopfli_compress` until cutover.

#![allow(dead_code)]

pub mod cache; // Step 5
pub mod deflate_size; // Step 9
pub mod hash; // Step 4
pub mod katajainen; // Step 2
pub mod lz77; // Steps 6-8 (built incrementally)
pub mod squeeze; // Steps 10-12 (built incrementally)
pub mod symbols; // Step 1
pub mod tree; // Step 3

/// Options used throughout the program. Mirrors C `ZopfliOptions`. Default
/// matches `ZopfliInitOptions`.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct ZopfliOptions {
    pub verbose: i32,
    pub verbose_more: i32,
    pub numiterations: i32,
    pub blocksplitting: i32,
    pub blocksplittinglast: i32,
    pub blocksplittingmax: i32,
}

impl Default for ZopfliOptions {
    fn default() -> Self {
        Self {
            verbose: 0,
            verbose_more: 0,
            numiterations: 15,
            blocksplitting: 1,
            blocksplittinglast: 0,
            blocksplittingmax: 15,
        }
    }
}
// pub mod blocksplitter; // unlock at Step 13
// pub mod deflate;       // unlock at Steps 14-15
// pub mod gzip;          // unlock at Step 16

#[cfg(test)]
mod oracle_tests;
