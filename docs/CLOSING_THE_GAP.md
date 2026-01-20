# Realistic Plan: Closing the 45% Gap with libdeflate

**Current State**: 770 MB/s (55% of libdeflate's 1390 MB/s)
**Target**: 1400+ MB/s (exceed libdeflate)

## Gap Analysis: Where We Lose Time

### Measured Breakdown
```
Our decode loop:     ~1.3 ns/byte
libdeflate:          ~0.72 ns/byte
Overhead:            ~0.58 ns/byte (80% slower)
```

### Root Causes (in order of impact)

| Cause | Est. Impact | Difficulty | Status |
|-------|-------------|------------|--------|
| Loop structure overhead | 15-20% | Hard | Not done |
| Subtable resolution branches | 10-15% | Medium | Partially done |
| Rust bounds checking | 5-10% | Easy | Done (unsafe) |
| Table lookup latency hiding | 5-10% | Medium | Not done |
| Match copy overhead | 3-5% | Easy | Done |

## Phase 1: Measure Before Optimizing (1 day)

### 1.1 Create Isolated Micro-benchmarks
```rust
// Benchmark just the literal decode path
fn bench_literal_only() { ... }

// Benchmark just length+distance decode
fn bench_match_only() { ... }

// Benchmark table lookup latency
fn bench_table_lookup() { ... }
```

### 1.2 Profile with perf/Instruments
```bash
# Linux
perf record -g cargo test --release bench_silesia
perf report

# macOS
instruments -t "Time Profiler" cargo test --release bench_silesia
```

**Deliverable**: Know exactly which functions consume the most time.

## Phase 2: Eliminate Subtable Branches (2 days)

### Problem
Our code:
```rust
let mut entry = litlen_table.lookup(bits);
if entry.is_subtable_ptr() {  // BRANCH
    entry = litlen_table.lookup_subtable(entry, bits);
}
```

libdeflate's approach:
```c
entry = table[bitbuf & mask];
// Most entries resolved in one lookup
// Subtable check only after type check fails
```

### Solution: 12-bit Main Table
- Increase LITLEN_TABLEBITS from 11 to 12
- Reduces subtable frequency from ~5% to ~1%
- Trade-off: 8KB → 16KB table size (still fits L1)

### Implementation
```rust
// In libdeflate_entry.rs
pub const LITLEN_TABLEBITS: usize = 12;  // was 11
pub const LITLEN_ENOUGH: usize = 4684;   // 2x entries
```

**Expected Gain**: 5-8%

## Phase 3: Restructure Decode Loop (3-5 days)

### Problem
Our loop has complex control flow with nested ifs:
```rust
if is_literal {
    // unrolled literals with subtable checks
    continue;
}
if is_exceptional {
    // EOB/subtable handling
}
// length handling
```

libdeflate's loop is flatter:
```c
entry = table[bitbuf & mask];
bitbuf >>= (u8)entry;  // ALWAYS consume
bitsleft -= entry;

if (entry & LITERAL_FLAG) {
    *out++ = entry >> 16;
    entry = table[bitbuf & mask];  // PRELOAD
    // ... continue literal chain
    continue;
}

if (unlikely(entry & EXCEPTIONAL)) {
    if (entry & EOB) goto done;
    // resolve subtable
}

// length/distance (rare path)
```

### Solution: Match libdeflate's Exact Structure

**Key insight**: The literal chain should NEVER fall through to length handling.
Instead, length codes should be handled by restarting the loop.

```rust
'fastloop: loop {
    bits.refill_branchless();
    let saved = bits.peek_bits();
    let mut entry = litlen_table.lookup(saved);
    
    // Consume FIRST (unconditionally)
    bits.consume_entry(entry.raw());
    
    // Check type
    if entry.is_literal() {
        output[out_pos] = entry.literal_value();
        out_pos += 1;
        
        // Preload next while in literal chain
        entry = litlen_table.lookup(bits.peek_bits());
        if entry.is_literal() {
            bits.consume_entry(entry.raw());
            output[out_pos] = entry.literal_value();
            out_pos += 1;
            // One more...
        }
        continue 'fastloop;  // ALWAYS restart
    }
    
    if entry.is_eob() {
        return Ok(out_pos);
    }
    
    if entry.is_subtable_ptr() {
        entry = litlen_table.lookup_subtable(entry, saved);
        if entry.is_eob() { return Ok(out_pos); }
        if entry.is_literal() {
            // Handle literal from subtable
            continue 'fastloop;
        }
    }
    
    // Length code - always a fresh iteration
    let length = entry.decode_length(saved);
    // ... distance handling
}
```

**Expected Gain**: 15-20%

## Phase 4: Preload Optimization (2 days)

### Problem
Memory latency for table lookups is ~4 cycles. We wait for each lookup.

### Solution: Overlap Lookups with Other Work
```rust
// Extract literal
let lit = entry.literal_value();

// START next lookup (memory fetch begins)
let next_entry = litlen_table.lookup(bits.peek_bits());

// WRITE while waiting for fetch
output[out_pos] = lit;
out_pos += 1;

// By now, next_entry is ready
entry = next_entry;
```

**Key**: The table lookup starts fetching from memory while we're writing the previous byte.

**Expected Gain**: 5-10%

## Phase 5: Consider Assembly (1 week, optional)

### When to Do This
Only if Phases 2-4 don't get us to 90%+ of libdeflate.

### Scope
Just the inner fastloop (~50 lines of assembly):
```asm
.loop:
    ; refill
    mov rax, [rsi]
    shlx rax, rax, rcx
    or rbx, rax
    
    ; lookup
    and rdi, 0x7FF
    mov eax, [r8 + rdi*4]
    
    ; consume
    shrx rbx, rbx, rax
    sub ecx, eax
    
    ; check literal
    test eax, eax
    js .literal
    
    ; ... rest of loop
```

**Expected Gain**: 10-20% (if done well)

## Realistic Timeline

| Phase | Duration | Expected Gain | Cumulative |
|-------|----------|---------------|------------|
| 1. Measure | 1 day | 0% | 55% |
| 2. 12-bit table | 2 days | 5-8% | 60-63% |
| 3. Loop restructure | 3-5 days | 15-20% | 75-83% |
| 4. Preload | 2 days | 5-10% | 80-93% |
| 5. Assembly (optional) | 1 week | 10-20% | 90-113% |

**Total: 2-3 weeks to reach 90%+**

## Alternative Strategy: Embrace Parallelism

Our parallel BGZF decoder already achieves **3770 MB/s** with 8 threads, which is:
- **2.7x faster** than libdeflate single-threaded
- Competitive with rapidgzip's parallel performance

### When Single-Threaded Matters Less
- Large files (where parallelism shines)
- Server workloads (multiple cores available)
- Batch processing (can saturate all cores)

### When Single-Threaded Matters
- Small files (<1MB)
- Interactive/latency-sensitive use
- Resource-constrained environments

**Recommendation**: If parallel performance is the priority, focus on making the parallel
path the default and optimize for that. Single-threaded is a secondary goal.

## Success Criteria

| Milestone | Throughput | % of libdeflate |
|-----------|------------|-----------------|
| Current | 770 MB/s | 55% |
| Good | 1100 MB/s | 80% |
| Great | 1300 MB/s | 95% |
| **Exceed** | **1400+ MB/s** | **100%+** |

## Next Immediate Action

1. **Run perf profiling** to identify the actual hotspots
2. **Try 12-bit table** (easiest change with measurable impact)
3. **Restructure loop** following libdeflate's exact pattern
