//! Cranelift JIT Huffman Decoder
//!
//! Generates native machine code for each unique Huffman table fingerprint.
//! The compiled decoder has symbol values as immediate constants, eliminating
//! table lookups entirely.
//!
//! ## Architecture
//!
//! For each fingerprint, we generate a function with signature:
//! ```ignore
//! fn decode(bitbuf: u64, out: *mut u8, out_pos: usize) -> (u8, u32)
//! // Returns (symbol, bits_consumed) for a single symbol decode
//! ```
//!
//! The generated code is a series of comparisons and branches that directly
//! emit symbol values without memory lookups.

#![allow(dead_code)]

use std::collections::HashMap;
use std::mem;

use cranelift_codegen::ir::types::*;
use cranelift_codegen::ir::{AbiParam, InstBuilder, Signature, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::jit_decode::TableFingerprint;

/// Type alias for the compiled decode function
/// Returns (symbol, bits_consumed) where symbol has LITERAL_FLAG (0x8000) if literal
type DecodeFn = unsafe extern "C" fn(bitbuf: u64) -> u64;

/// Cranelift JIT compiler for Huffman decoders
pub struct CraneliftJIT {
    /// The JIT module containing compiled functions
    module: JITModule,
    /// Context for building functions
    ctx: Context,
    /// Builder context (reusable)
    builder_ctx: FunctionBuilderContext,
    /// Cache of compiled decoders by fingerprint
    decoders: HashMap<TableFingerprint, DecodeFn>,
    /// Statistics
    compile_count: usize,
    compile_time_us: u64,
}

/// Flag to indicate literal in return value
const LITERAL_FLAG: u64 = 0x8000;
/// Shift for bits consumed in return value
const BITS_SHIFT: u64 = 16;

impl CraneliftJIT {
    /// Create a new JIT compiler
    pub fn new() -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| e.to_string())?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| e.to_string())?;

        let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| e.to_string())?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);

        Ok(Self {
            ctx: module.make_context(),
            module,
            builder_ctx: FunctionBuilderContext::new(),
            decoders: HashMap::new(),
            compile_count: 0,
            compile_time_us: 0,
        })
    }

    /// Get or compile a decoder for the given table
    pub fn get_decoder(&mut self, litlen_lens: &[u8]) -> Option<DecodeFn> {
        let fingerprint = TableFingerprint::from_litlen_lengths(litlen_lens);

        if let Some(&func) = self.decoders.get(&fingerprint) {
            return Some(func);
        }

        // Compile a new decoder
        let start = std::time::Instant::now();
        let func = self.compile_decoder(litlen_lens, fingerprint)?;
        self.compile_time_us += start.elapsed().as_micros() as u64;
        self.compile_count += 1;

        self.decoders.insert(fingerprint, func);
        Some(func)
    }

    /// Compile a decoder for the given code lengths
    fn compile_decoder(
        &mut self,
        litlen_lens: &[u8],
        fingerprint: TableFingerprint,
    ) -> Option<DecodeFn> {
        // Build the Huffman table entries first
        let entries = build_decode_entries(litlen_lens)?;

        // Create function signature: fn(bitbuf: u64) -> u64
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(I64)); // bitbuf
        sig.returns.push(AbiParam::new(I64)); // packed (symbol, bits)

        let func_name = format!("decode_{:016x}", fingerprint.as_u64());
        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Local, &sig)
            .ok()?;

        self.ctx.func.signature = sig;
        self.ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let bitbuf = builder.block_params(entry_block)[0];

            // Generate the decode logic as a decision tree
            // For simplicity, we generate a flat lookup (similar to table)
            // A more advanced version would generate an optimal decision tree

            // Mask to 11 bits for lookup
            let mask = builder.ins().iconst(I64, 0x7FF);
            let index = builder.ins().band(bitbuf, mask);

            // For now, generate a simple series of comparisons
            // This is a basic implementation - a production version would use
            // jump tables or optimal decision trees

            let result = generate_decode_tree(&mut builder, index, &entries);
            builder.ins().return_(&[result]);

            builder.finalize();
        }

        // Compile the function
        self.module.define_function(func_id, &mut self.ctx).ok()?;

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().ok()?;

        let code_ptr = self.module.get_finalized_function(func_id);
        Some(unsafe { mem::transmute::<*const u8, DecodeFn>(code_ptr) })
    }

    /// Get compilation statistics
    pub fn stats(&self) -> (usize, u64) {
        (self.compile_count, self.compile_time_us)
    }
}

impl Default for CraneliftJIT {
    fn default() -> Self {
        Self::new().expect("Failed to create Cranelift JIT")
    }
}

/// A decode table entry for JIT compilation
#[derive(Clone, Copy)]
struct DecodeEntry {
    symbol: u16,
    bits: u8,
    is_literal: bool,
    is_eob: bool,
}

/// Build decode entries from code lengths
fn build_decode_entries(litlen_lens: &[u8]) -> Option<Vec<DecodeEntry>> {
    const TABLE_BITS: usize = 11;
    const TABLE_SIZE: usize = 1 << TABLE_BITS;

    let mut entries = vec![
        DecodeEntry {
            symbol: 0,
            bits: 0,
            is_literal: false,
            is_eob: false
        };
        TABLE_SIZE
    ];

    // Count codes at each length
    let mut bl_count = [0u32; 16];
    for &len in litlen_lens {
        if len > 0 && len <= 15 {
            bl_count[len as usize] += 1;
        }
    }

    // Compute first code for each length
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..16 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign codes to symbols
    for (symbol, &len) in litlen_lens.iter().enumerate() {
        if len == 0 || len as usize > TABLE_BITS {
            continue;
        }
        let len = len as usize;

        let codeword = next_code[len];
        next_code[len] += 1;

        // Reverse bits
        let mut reversed = 0u32;
        let mut c = codeword;
        for _ in 0..len {
            reversed = (reversed << 1) | (c & 1);
            c >>= 1;
        }

        // Fill table entries
        let stride = 1usize << len;
        let mut idx = reversed as usize;
        while idx < TABLE_SIZE {
            entries[idx] = DecodeEntry {
                symbol: symbol as u16,
                bits: len as u8,
                is_literal: symbol < 256,
                is_eob: symbol == 256,
            };
            idx += stride;
        }
    }

    Some(entries)
}

/// Generate the decode tree as Cranelift IR
fn generate_decode_tree(
    builder: &mut FunctionBuilder,
    index: cranelift_codegen::ir::Value,
    entries: &[DecodeEntry],
) -> cranelift_codegen::ir::Value {
    // For this basic implementation, we'll use a lookup table approach
    // embedded as a series of comparisons. A production version would
    // generate an optimal decision tree or jump table.

    // Find unique entries to reduce comparisons
    let mut unique_entries: Vec<(usize, DecodeEntry)> = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        if entry.bits > 0 {
            // Check if this pattern is the canonical one for this symbol
            let is_canonical = entries[..i]
                .iter()
                .all(|e| e.symbol != entry.symbol || e.bits != entry.bits);
            if is_canonical {
                unique_entries.push((i, *entry));
            }
        }
    }

    // Generate a default return value (error case)
    let default_result = builder.ins().iconst(I64, 0);

    // If no unique entries, return default
    if unique_entries.is_empty() {
        return default_result;
    }

    // For simplicity, generate linear comparisons
    // A real implementation would use a switch/jump table
    let mut result = default_result;

    for (pattern, entry) in unique_entries.iter().take(64) {
        // Only handle first 64 unique patterns to limit code size
        let pattern_val = builder.ins().iconst(I64, *pattern as i64);
        let mask = builder.ins().iconst(I64, (1i64 << entry.bits) - 1);
        let masked_index = builder.ins().band(index, mask);
        let cmp = builder.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::Equal,
            masked_index,
            pattern_val,
        );

        // Compute the result for this entry
        let mut packed: u64 = entry.symbol as u64;
        if entry.is_literal {
            packed |= LITERAL_FLAG;
        }
        packed |= (entry.bits as u64) << BITS_SHIFT;

        let entry_result = builder.ins().iconst(I64, packed as i64);
        result = builder.ins().select(cmp, entry_result, result);
    }

    result
}

/// Decode a single symbol using JIT-compiled decoder
#[inline(always)]
pub fn decode_jit(decoder: DecodeFn, bitbuf: u64) -> (u16, u8, bool, bool) {
    let packed = unsafe { decoder(bitbuf) };
    let symbol = (packed & 0x7FFF) as u16;
    let is_literal = (packed & LITERAL_FLAG) != 0;
    let bits = ((packed >> BITS_SHIFT) & 0x1F) as u8;
    let is_eob = symbol == 256;
    (symbol, bits, is_literal, is_eob)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cranelift_jit_creation() {
        let jit = CraneliftJIT::new();
        assert!(
            jit.is_ok(),
            "Failed to create Cranelift JIT: {:?}",
            jit.err()
        );
    }

    #[test]
    fn test_cranelift_jit_compile() {
        let mut jit = match CraneliftJIT::new() {
            Ok(j) => j,
            Err(e) => {
                eprintln!("Skipping test - Cranelift JIT not available: {}", e);
                return;
            }
        };

        // Fixed Huffman code lengths
        let mut litlen_lens = vec![0u8; 288];
        litlen_lens[..144].fill(8);
        litlen_lens[144..256].fill(9);
        litlen_lens[256] = 7; // EOB
        litlen_lens[257..280].fill(7);
        litlen_lens[280..288].fill(8);

        let start = std::time::Instant::now();
        let decoder = jit.get_decoder(&litlen_lens);
        let compile_time = start.elapsed();

        assert!(decoder.is_some(), "Failed to compile decoder");
        eprintln!("[JIT] Compile time: {:?}", compile_time);

        let (count, time_us) = jit.stats();
        eprintln!("[JIT] Stats: {} compiles, {} µs total", count, time_us);
    }

    #[test]
    fn bench_cranelift_jit() {
        let mut jit = match CraneliftJIT::new() {
            Ok(j) => j,
            Err(e) => {
                eprintln!("Skipping benchmark - Cranelift JIT not available: {}", e);
                return;
            }
        };

        // Fixed Huffman code lengths
        let mut litlen_lens = vec![0u8; 288];
        litlen_lens[..144].fill(8);
        litlen_lens[144..256].fill(9);
        litlen_lens[256] = 7;
        litlen_lens[257..280].fill(7);
        litlen_lens[280..288].fill(8);

        let decoder = jit.get_decoder(&litlen_lens).unwrap();

        // Benchmark decode speed
        let iterations = 10_000_000;
        let test_patterns: Vec<u64> = (0..1000).map(|i| i * 7919 % 2048).collect();

        let start = std::time::Instant::now();
        let mut total_bits = 0u64;
        for _ in 0..iterations / 1000 {
            for &pattern in &test_patterns {
                let (_, bits, _, _) = decode_jit(decoder, pattern);
                total_bits += bits as u64;
            }
        }
        let elapsed = start.elapsed();

        let decodes_per_sec = iterations as f64 / elapsed.as_secs_f64();
        eprintln!("\n[BENCH] Cranelift JIT Decode:");
        eprintln!("[BENCH]   {} M decodes/sec", decodes_per_sec / 1_000_000.0);
        eprintln!("[BENCH]   Total bits decoded: {}", total_bits);
    }
}
