#![cfg(parallel_sm)]

//! Port of `rapidgzip::deflate::getUsedWindowSymbols`
//! (vendor/.../gzip/deflate.hpp:1846-1988).

use crate::decompress::inflate::consume_first_decode::Bits;
use crate::decompress::parallel::marker_inflate::{Block, BlockError, MarkerSink, MAX_WINDOW_SIZE};

#[derive(Default)]
struct DiscardSink {
    len: usize,
}

impl MarkerSink for DiscardSink {
    fn push_slice(&mut self, values: &[u16]) {
        self.len += values.len();
    }
    fn sink_len(&self) -> usize {
        self.len
    }
    fn as_slice(&self) -> &[u16] {
        &[]
    }
}

/// Bitmap of which bytes in the 32 KiB DEFLATE window are referenced when
/// decoding up to 32 KiB forward from `start_bit` in `deflate_data`.
/// Index 0 is the oldest window byte (vendor `getUsedWindowSymbols`).
pub fn get_used_window_symbols(deflate_data: &[u8], start_bit: usize) -> Vec<bool> {
    let mut used = vec![false; MAX_WINDOW_SIZE];
    if start_bit / 8 >= deflate_data.len() {
        return used;
    }

    let mut bits = Bits::at_bit_offset(deflate_data, start_bit);
    let mut block = Block::new();
    block.set_track_backreferences(true);

    let mut n_bytes_read = 0usize;
    let mut sink = DiscardSink::default();

    while n_bytes_read < MAX_WINDOW_SIZE {
        match block.read_header(&mut bits, false) {
            Ok(()) => {}
            Err(BlockError::EndOfFile) => break,
            Err(_) => break,
        }

        while n_bytes_read < MAX_WINDOW_SIZE && !block.eob() {
            let n = match block.read(&mut bits, &mut sink, MAX_WINDOW_SIZE - n_bytes_read) {
                Ok(n) => n,
                Err(_) => break,
            };
            if n == 0 && block.eob() {
                break;
            }
            n_bytes_read += n;
        }

        for reference in block.backreferences() {
            if (reference.distance as usize) < n_bytes_read {
                continue;
            }
            let distance_from_end = reference.distance as usize - n_bytes_read;
            if distance_from_end > MAX_WINDOW_SIZE {
                continue;
            }
            if reference.length == 0 {
                continue;
            }
            let start_offset = MAX_WINDOW_SIZE - distance_from_end;
            for i in 0..reference.length as usize {
                if start_offset + i < MAX_WINDOW_SIZE {
                    used[start_offset + i] = true;
                }
            }
        }

        if block.eos() {
            break;
        }
    }

    used
}
