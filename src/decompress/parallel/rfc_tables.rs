#![allow(dead_code)]

//! Port of `rapidgzip/gzip/RFCTables.hpp` length/distance helpers used by
//! `HuffmanCodingShortBitsCachedDeflate` LUT fill and slow-path decode.

use crate::decompress::parallel::error::Error;
use crate::decompress::parallel::huffman_base::LsbBitReader;
use crate::decompress::parallel::huffman_symbols_per_length::HuffmanCodingSymbolsPerLength;

/// Mirror of `distanceLUT` (RFCTables.hpp:64-65).
pub const DISTANCE_LUT: [u16; 30] = {
    let mut lut = [0u16; 30];
    let mut i = 0usize;
    while i < 4 {
        lut[i] = (i + 1) as u16;
        i += 1;
    }
    while i < 30 {
        let extra = (i as u16 - 2) / 2;
        lut[i] = 1 + (1 << (extra + 1)) + ((i as u16 % 2) << extra);
        i += 1;
    }
    lut
};

/// Mirror of `calculateLength` (RFCTables.hpp:71-77).
#[inline]
pub const fn calculate_length(code: u16) -> u16 {
    debug_assert!(code < 285 - 261);
    let extra_bits = code / 4;
    3 + (1 << (extra_bits + 2)) + ((code % 4) << extra_bits)
}

/// Mirror of `getLength` (RFCTables.hpp:97-114).
pub fn get_length<R: LsbBitReader>(code: u16, bit_reader: &mut R) -> Result<u16, Error> {
    if code <= 264 {
        return Ok(code - 257 + 3);
    }
    if code < 285 {
        let lc = code - 261;
        let extra_bits = lc / 4;
        let extra = if extra_bits > 0 {
            bit_reader
                .read(extra_bits as u8)
                .map_err(|_| Error::InvalidHuffmanCode)? as u16
        } else {
            0
        };
        return Ok(calculate_length(lc) + extra);
    }
    if code == 285 {
        return Ok(258);
    }
    Err(Error::InvalidHuffmanCode)
}

/// Mirror of `getLengthMinus3` (RFCTables.hpp:117-123).
pub fn get_length_minus3<R: LsbBitReader>(code: u16, bit_reader: &mut R) -> Result<u8, Error> {
    let len = get_length(code, bit_reader)?;
    Ok((len - 3) as u8)
}

/// Mirror of `getDistance` for dynamic Huffman (RFCTables.hpp:126-157).
pub fn get_distance_dynamic<R: LsbBitReader, const MAX_DIST: usize>(
    distance_hc: &HuffmanCodingSymbolsPerLength<MAX_DIST>,
    bit_reader: &mut R,
) -> Result<u16, Error> {
    let decoded = distance_hc
        .decode(bit_reader)
        .ok_or(Error::InvalidHuffmanCode)?;
    let mut distance = decoded;

    if distance <= 3 {
        distance += 1;
    } else if distance <= 29 {
        let extra_bits_count = (distance - 2) / 2;
        let extra_bits = if extra_bits_count > 0 {
            bit_reader
                .read(extra_bits_count as u8)
                .map_err(|_| Error::InvalidHuffmanCode)? as u16
        } else {
            0
        };
        distance = DISTANCE_LUT[distance as usize] + extra_bits;
    } else {
        return Err(Error::InvalidHuffmanCode);
    }
    Ok(distance)
}
