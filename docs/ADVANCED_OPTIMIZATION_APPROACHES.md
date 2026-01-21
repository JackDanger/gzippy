# Advanced Mathematical Approaches to Exceed libdeflate

**Goal:** Achieve >100% of libdeflate's throughput on dynamic Huffman data

**Current Status:** 69% of libdeflate (940 MB/s vs 1,422 MB/s)

---

## Why Rust Can Beat C

These approaches leverage Rust features that have no equivalent in C:

### 1. Const Generics + Compile-Time Evaluation
```rust
// Rust: Generate specialized code at compile time
impl<const BITS: usize> Table<BITS> {
    pub const fn build() -> Self { /* computed at compile time */ }
}

// C: Must use runtime initialization or external code generation
// No way to parameterize on integer constants at compile time
```

### 2. Guaranteed No-Aliasing (&mut)
```rust
// Rust: Compiler KNOWS these don't alias
fn process(input: &[u8], output: &mut [u8]) {
    // SIMD can freely read input while writing output
}

// C: Even with restrict, compilers are conservative
// Many optimizations disabled due to potential aliasing
```

### 3. Type-Level Integers (const generics)
```rust
// Rust: Specialize for exact bit widths
fn decode<const WIDTH: u8>(bits: u64) -> (u16, u8) {
    // Compiler generates optimal code for WIDTH=7, 8, 9 separately
}

// C: Must use runtime branches or macro explosion
```

---

## Approach 1: Algebraic Normal Form (ANF) Decoding

### Mathematical Foundation

A Huffman tree is a Boolean function:
```
f: {0,1}^D → Symbol × BitLength
```

Where D is the maximum code depth (9 for fixed Huffman).

Any Boolean function can be represented in Algebraic Normal Form:
```
f(x₁,...,xₙ) = ⊕ aₛ · ∏(xⱼ for j∈S)  for all S⊆[n]
```

Where:
- ⊕ is XOR
- ∏ is AND
- aₛ ∈ {0,1} are the ANF coefficients

### Why This Is Fast

1. **Branchless Evaluation**: ANF uses only XOR and AND - no branches
2. **SIMD-Friendly**: XOR/AND operations vectorize naturally
3. **Constant-Time**: Every evaluation takes the same number of cycles

### Implementation Strategy

```rust
// Precompute ANF coefficients at compile time
const ANF_SYMBOL_BITS: [[u64; 9]; 8] = compute_anf_coefficients();

// Evaluate in constant time using bitwise ops
#[inline(always)]
fn anf_decode(bits: u64) -> u8 {
    let mut symbol = 0u8;
    for i in 0..8 {
        // Monomial selection via AND
        let selected = bits & ANF_SYMBOL_BITS[i];
        // Parity via POPCNT (single instruction)
        symbol |= ((selected.count_ones() & 1) as u8) << i;
    }
    symbol
}
```

### Benchmark Results

```
Algebraic: 1599.0 M ops/sec
Standard:  1050.8 M ops/sec
Ratio:     1.52x
```

**The algebraic approach is 52% faster for isolated lookups!**

---

## Approach 2: Interleaved Finite State Machine (FSM)

### Core Insight

Traditional Huffman decoding is sequential:
```
decode(symbol₀) → decode(symbol₁) → decode(symbol₂) → ...
```

But we can process multiple bit positions in parallel:
```
Lane 0: decode at bit offset 0
Lane 1: decode at bit offset 8
Lane 2: decode at bit offset 16
...
Lane 7: decode at bit offset 56
```

### The Problem: Variable-Length Codes

After decoding, bit positions don't align:
- Lane 0 consumes 7 bits → next at bit 7
- Lane 1 consumes 9 bits → next at bit 17
- etc.

### The Solution: Prefix-Sum Reconciliation

1. **Parallel Decode**: Each lane decodes independently
2. **Prefix-Sum**: Compute cumulative bits consumed
3. **Scatter**: Write outputs to correct positions

```rust
// SIMD parallel decode (8 lanes)
let transitions = gather_transitions(&table, &states, &input_bits);

// Prefix sum for output positions
let output_offsets = prefix_sum(&bits_consumed);

// Scatter to correct positions
for i in 0..8 {
    if transitions[i].output != NONE {
        output[output_offsets[i]] = transitions[i].output as u8;
    }
}
```

### Why This Beats Sequential

On literal-heavy data (60-80% literals):
- Sequential: 1 symbol per ~10 cycles (table lookup + branches)
- Parallel: 8 symbols per ~30 cycles (SIMD gather + prefix sum)

**Theoretical speedup: 8×10/30 = 2.67x**

### Rust-Specific Optimizations

1. **Guaranteed SIMD safety**: &mut aliasing rules allow aggressive vectorization
2. **Compile-time FSM generation**: const fn builds transition tables at build time
3. **Type-specialized lanes**: const generics for 8-lane, 16-lane, 32-lane versions

---

## Approach 3: Hybrid Adaptive Decoder

Combine ANF and FSM based on data characteristics:

```rust
pub fn decode_adaptive(bits: &mut Bits, output: &mut [u8]) -> Result<usize> {
    let mut pattern = 0xFFFF_FFFF_FFFF_FFFFu64; // Recent symbol types

    loop {
        let literal_ratio = pattern.count_ones();

        if literal_ratio > 56 {
            // >87% literals: Use ANF (branchless, no state)
            decode_anf_batch(bits, output, &mut out_pos);
        } else if literal_ratio > 40 {
            // 62-87% literals: Use interleaved FSM
            decode_fsm_interleaved(bits, output, &mut out_pos);
        } else {
            // <62% literals: Use sequential (matches dominate)
            decode_sequential(bits, output, &mut out_pos);
        }

        // Update pattern...
    }
}
```

---

## Implementation Roadmap

### Phase 1: ANF Decoder (Current)
- [x] Build ANF lookup table at compile time
- [x] Benchmark isolated lookups (1.52x speedup)
- [ ] Integrate into full decode loop
- [ ] Handle 9-bit codes (symbols 144-255)

### Phase 2: Interleaved FSM
- [x] Define FSM state encoding
- [x] Build transition table structure
- [ ] Implement AVX2 gather for parallel lookup
- [ ] Implement prefix-sum reconciliation
- [ ] Handle match/distance separately

### Phase 3: Hybrid Integration
- [ ] Add adaptive mode switching
- [ ] Profile on diverse datasets
- [ ] Tune thresholds for optimal switching

---

## Expected Performance

| Approach | Literal Lookups | Full Decode | vs libdeflate |
|----------|----------------|-------------|---------------|
| Current  | 1050 M/s       | 940 MB/s    | 69%           |
| ANF      | 1599 M/s       | ~1200 MB/s  | ~85%          |
| FSM      | N/A            | ~1500 MB/s  | ~105%         |
| Hybrid   | adaptive       | ~1600 MB/s  | ~112%         |

**Target: 1600+ MB/s (112%+ of libdeflate)**

---

## Mathematical Details

### ANF via Möbius Transform

The ANF coefficients are computed via the Möbius transform:
```
a_S = ⊕ f(T) for all T ⊆ S
```

For a Huffman tree, f(T) is the symbol at leaf T (or 0 if T is internal).

### FSM State Minimization

The FSM can be minimized using:
1. **Hopcroft's algorithm**: Minimize DFA states
2. **State encoding**: Pack state into 16 bits for SIMD efficiency
3. **Transition compression**: Use delta encoding for sparse tables

### SIMD Prefix Sum

Efficient prefix sum using AVX2:
```asm
; Input: ymm0 = [a, b, c, d, e, f, g, h]
; Output: ymm0 = [0, a, a+b, a+b+c, ...]

vpsllq ymm1, ymm0, 32      ; Shift left by 32 bits
vpaddq ymm0, ymm0, ymm1    ; Add pairs
vpermq ymm1, ymm0, 0x93    ; Rotate
vpaddq ymm0, ymm0, ymm1    ; Add quads
```

This computes prefix sum in O(log n) steps.

---

## Conclusion

By leveraging Rust's unique capabilities:
1. **Const generics** for compile-time table generation
2. **&mut aliasing guarantees** for SIMD optimization
3. **Type-level integers** for specialized code paths

We can implement mathematical optimizations that are impossible in C, achieving **>100% of libdeflate's throughput**.

The key insight: Huffman decoding is fundamentally a Boolean function evaluation problem, and Boolean functions have highly optimized algebraic representations that Rust can exploit at compile time.
