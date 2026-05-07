//! Pure-Rust port of Google Zopfli. Built bottom-up; oracle-tested
//! against the C FFI at `crate::backends::zopfli_compress` until cutover.

pub mod symbols; // Step 1
                 // pub mod katajainen;  // unlock at Step 2
                 // pub mod tree;        // unlock at Step 3
                 // pub mod hash;        // unlock at Step 4
                 // pub mod cache;       // unlock at Step 5
                 // pub mod lz77;        // unlock at Steps 6-8
                 // pub mod deflate_size; // unlock at Step 9
                 // pub mod squeeze;     // unlock at Steps 10-12
                 // pub mod blocksplitter; // unlock at Step 13
                 // pub mod deflate;     // unlock at Steps 14-15
                 // pub mod gzip;        // unlock at Step 16

#[cfg(test)]
mod oracle_tests;
