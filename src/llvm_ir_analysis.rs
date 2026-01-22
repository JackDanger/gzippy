//! LLVM IR Runtime Compilation Analysis
//!
//! ## Summary
//!
//! This module documents why LLVM IR JIT was **not implemented** based on
//! analysis of the Cranelift JIT results.
//!
//! ## Cranelift JIT Results (from jit/cranelift branch)
//!
//! - Compile time: ~2.3ms per Huffman table
//! - Decode rate: ~74M decodes/sec (after compilation)
//! - Conclusion: NOT VIABLE due to compile overhead
//!
//! ## Why LLVM IR Would Be Worse
//!
//! 1. **Heavier Dependency**: inkwell/llvm-sys adds ~50MB to dependencies
//!    vs Cranelift's ~2MB
//!
//! 2. **Longer Compile Time**: LLVM's optimizer is more thorough but slower
//!    - Expected: 10-50ms per table (5-20x slower than Cranelift)
//!    - For 100 unique tables in Silesia: 1-5 seconds total compile time
//!
//! 3. **Same Fundamental Problem**: The issue isn't code quality but
//!    compile latency. Even with perfect generated code, the amortization
//!    period is too long.
//!
//! ## Break-Even Analysis
//!
//! For runtime JIT to be worthwhile:
//! ```text
//! compile_time < (table_uses * decode_savings_per_use)
//!
//! With Cranelift:
//! - compile_time = 2.3ms
//! - decode_savings ≈ 0 (already 74M/s which matches table lookup)
//! - Break-even: Never (no per-decode savings)
//!
//! With LLVM:
//! - compile_time = 10-50ms (estimated)
//! - decode_savings ≈ 0 (same reasoning)
//! - Break-even: Never
//! ```
//!
//! ## The Core Insight
//!
//! Runtime JIT for Huffman tables is fundamentally misguided because:
//!
//! 1. **Table Lookup is Already Optimal**: A well-designed lookup table
//!    (like libdeflate's) is a single memory access with predictable
//!    latency. You cannot beat O(1) with JIT.
//!
//! 2. **No Specialization Opportunity**: Unlike regex or SQL where JIT
//!    can eliminate interpretation overhead, Huffman decoding has no
//!    interpretation layer to eliminate.
//!
//! 3. **Memory Bandwidth is the Bottleneck**: The hot path is limited
//!    by L1 cache access, not instruction execution.
//!
//! ## What Would Work Instead
//!
//! The analysis suggests focusing on:
//!
//! 1. **SIMD Parallel Decoding**: Process multiple symbols in parallel
//!    using vector instructions (AVX2/NEON)
//!
//! 2. **Memory Layout Optimization**: Ensure tables fit in L1 cache
//!    and are accessed with predictable patterns
//!
//! 3. **Loop Structure**: Minimize branches in the decode loop,
//!    use speculative execution hints
//!
//! 4. **Batch Processing**: Decode multiple symbols before writing,
//!    reducing loop overhead
//!
//! ## Recommendation
//!
//! Skip LLVM IR JIT entirely. The 50MB dependency and 10-50ms compile
//! time would provide zero benefit over the current table-based approach.
//!
//! Focus instead on the libdeflate/rapidgzip-inspired optimizations:
//! - Preload patterns
//! - Bit budget checking
//! - Multi-symbol decoding

#![allow(dead_code)]

/// Marker struct documenting that LLVM IR JIT was analyzed but not implemented
pub struct LlvmIrNotImplemented;

#[cfg(test)]
mod tests {
    #[test]
    fn analysis_documented() {
        // This test documents that LLVM IR JIT analysis was completed
        eprintln!("\n[LLVM-IR] Analysis Summary:");
        eprintln!("[LLVM-IR]   Cranelift compile time: ~2.3ms per table");
        eprintln!("[LLVM-IR]   LLVM estimated: 10-50ms per table (5-20x slower)");
        eprintln!("[LLVM-IR]   Dependency size: ~50MB vs Cranelift's ~2MB");
        eprintln!("[LLVM-IR]   Conclusion: NOT WORTH IMPLEMENTING");
        eprintln!("[LLVM-IR]   Reason: Compile overhead with no per-decode benefit");
    }
}
