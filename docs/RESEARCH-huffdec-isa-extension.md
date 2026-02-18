# HUFFDEC: A Custom ISA Extension for Hardware-Accelerated Huffman Decoding

## Status: Research / Pre-mortem

---

## 1. The Problem

Deflate decompression has an **irreducible serial dependency chain**: you cannot
know where symbol N+1 starts until you've decoded symbol N, because Huffman codes
are variable-length. Every software decoder ever built — libdeflate, zlib, zstd,
ours — is bottlenecked by this chain.

**Current performance on Apple M3 (gzippy, pure Rust):**

| Dataset  | Throughput | Cycles/symbol | vs libdeflate |
|----------|-----------|---------------|---------------|
| SILESIA  | 1400 MB/s | ~3.75         | 99%           |
| SOFTWARE | 21500 MB/s| ~0.5          | 106%          |
| LOGS     | 9100 MB/s | ~1.2          | 114%          |

We are at parity with the fastest C implementation. To go materially faster
requires breaking the serial dependency — which means hardware.

### Where time is spent (SILESIA, profiled)

| Phase              | % of cycles | What it does                            |
|--------------------|-------------|-----------------------------------------|
| Huffman lookup     | ~35%        | Table read + entry decode               |
| LZ77 match copy    | ~50%        | memcpy from output history              |
| Bit buffer mgmt    | ~10%        | Shift, refill, saved_bitbuf             |
| Branch/loop ctrl   | ~5%         | Symbol type check, loop overhead        |

**Amdahl's Law constraint**: An instruction that only accelerates Huffman lookups
(35% of work) yields at most **1.54x** speedup even if lookups become free.
To break 2x, the hardware must accelerate the copy phase too.

---

## 2. Design Tiers

### Tier 1: Single Instruction (`HUFFDEC`)
- Accelerates table lookup + bit shift
- Max theoretical speedup: **1.3-1.5x** (Amdahl-limited to 35% of work)
- Die area: ~50KB SRAM + control logic
- Verdict: **Not worth the silicon**

### Tier 2: Instruction Pair (`HUFFDEC` + `LZCOPY`)
- HUFFDEC handles decode; LZCOPY handles LZ77 match copy with hardware loop
- Max theoretical speedup: **2-3x** (addresses 85% of work)
- Die area: ~300KB SRAM + copy engine
- Verdict: **Interesting but fragile** (format-specific silicon)

### Tier 3: Deflate Micro-Engine (coprocessor)
- Dedicated state machine processes entire deflate blocks
- Software feeds compressed input, reads decompressed output via DMA
- Max theoretical speedup: **3-5x single-threaded**
- Die area: ~500KB SRAM + full FSM + DMA engine
- Precedent: Intel QAT, but in-core instead of PCIe
- Verdict: **Most promising for real-world impact**

### Tier 4: Memory Controller Integration
- Transparent decompression on mmap'd .gz files
- MMU triggers decompression on page fault
- Apparent latency: **zero** (prefetch hides it)
- Die area: Entire decompression engine in memory controller
- Verdict: **The dream, but decades away** (OS + HW co-design)

This document focuses on **Tier 3** as the sweet spot.

---

## 3. Architecture: Deflate Micro-Engine

### 3.1 ISA Interface (RISC-V Custom Extension)

Target: RISC-V RV64GC with custom-0 opcode space (0x0B).

```
DFLT.INIT   rd, rs1, rs2
  # Initialize decompression context
  # rs1 = compressed input base address
  # rs2 = compressed input length
  # rd  = context ID (hardware has N context slots, e.g., 4)
  # Begins DMA prefetch of input into internal SRAM

DFLT.OUT    rd, rs1, rs2
  # Set output buffer
  # rd  = context ID
  # rs1 = output base address
  # rs2 = output buffer length

DFLT.RUN    rd, rs1
  # Run decompression
  # rs1 = context ID
  # rd  = status (0 = complete, 1 = output full, 2 = need more input, <0 = error)
  # Blocks until one of: output full, input exhausted, block boundary, or error
  # On return, output buffer contains decompressed bytes
  # Hardware updates internal pointers (input consumed, output produced)

DFLT.STAT   rd, rs1, funct3
  # Query context status
  # rs1 = context ID
  # funct3 selects query:
  #   0 = bytes consumed from input
  #   1 = bytes produced to output
  #   2 = current deflate block index
  #   3 = CRC32 of decompressed data so far
  #   4 = state (idle/running/blocked/error)

DFLT.FREE   rs1
  # Release context, free hardware resources
  # rs1 = context ID
```

### 3.2 Micro-Engine Internal Architecture

```
                    ┌─────────────────────────────────────┐
                    │        Deflate Micro-Engine          │
                    │                                     │
  Input DMA ──────►│  ┌──────────┐    ┌──────────────┐   │──────► Output DMA
  (from DRAM)      │  │ Input    │    │ Huffman      │   │  (to DRAM)
                   │  │ Buffer   │───►│ Decode FSM   │   │
                   │  │ (4KB)    │    │              │   │
                   │  └──────────┘    │  ┌────────┐  │   │
                   │                  │  │LitLen  │  │   │
                   │                  │  │Table   │  │   │
                   │                  │  │(128KB) │  │   │
                   │                  │  ├────────┤  │   │
                   │                  │  │Dist    │  │   │
                   │                  │  │Table   │  │   │
                   │                  │  │(8KB)   │  │   │
                   │                  │  └────────┘  │   │
                   │                  │      │       │   │
                   │                  │      ▼       │   │
                   │                  │  ┌────────┐  │   │
                   │                  │  │ LZ77   │  │   │
                   │                  │  │ Copy   │──┼───┤
                   │                  │  │ Engine │  │   │
                   │                  │  └────────┘  │   │
                   │                  │      │       │   │
                   │                  │      ▼       │   │
                   │                  │  ┌────────┐  │   │
                   │                  │  │History │  │   │
                   │                  │  │Window  │  │   │
                   │                  │  │(32KB)  │  │   │
                   │                  │  └────────┘  │   │
                   │                  └──────────────┘   │
                   └─────────────────────────────────────┘

Total SRAM: 4KB input + 128KB litlen + 8KB dist + 32KB window = 172KB
Plus control logic: ~50K gates
```

### 3.3 Decode FSM States

```
IDLE ──► PARSE_HEADER ──► BUILD_TABLE ──► DECODE_SYMBOLS ──► BLOCK_DONE
              │                                   │               │
              ▼                                   ▼               ▼
         PARSE_GZIP_HDR                    EMIT_LITERAL     next block or
         (magic, flags,                    EMIT_MATCH       COMPLETE
          FEXTRA, FNAME)                   CHECK_EOB
```

**DECODE_SYMBOLS pipeline** (the hot path):

```
Stage 1: Extract bits[0:15] from 64-bit shift register     (1 cycle)
Stage 2: Table lookup in dedicated SRAM (single-port read)  (1 cycle)
Stage 3: Decode entry: literal value OR length base+extra   (1 cycle)
Stage 4: Write literal to history window + output DMA       (1 cycle)
         OR start distance decode (back to Stage 1-3)
         OR start LZ77 copy from history window

Sustained throughput: 1 literal per cycle (3.5 GHz = 3.5 Gsym/s)
With avg 1.5 bytes/symbol: ~5.25 GB/s single-context
```

**LZ77 copy engine**: Dedicated 32-byte-wide read port on history window SRAM.
Copies up to 32 bytes/cycle from window to output. A 258-byte match completes
in ~9 cycles (vs ~50 cycles in software due to cache miss potential).

### 3.4 Table Building

When the FSM encounters a dynamic Huffman block header:

1. Parse code length code lengths (HCLEN, 4-19 entries, 3 bits each)
2. Build code length decoder (tiny, ~19 entries)
3. Decode literal/length code lengths (HLIT + HDIST entries)
4. Build litlen table (128KB for 15-bit) — **this is the expensive part**
5. Build distance table (8KB)

**Table build time**: ~2000 cycles for a typical dynamic block.
At 1 symbol/cycle decode rate with ~2000 symbols per block, table build is
**~50% overhead**. This is the critical weakness.

**Mitigation: Table caching**. Hash the code lengths, cache up to 4 recently-used
tables. Deflate files commonly reuse the same tree across many blocks (especially
in streaming compression). Cache hit rate in practice: 60-80% for typical files,
near 0% for files compressed with `-9` (unique trees per block).

### 3.5 Speculative Decode (Breaking the Serial Chain)

Even with a 1-cycle table lookup, the serial dependency `lookup → get bits_consumed → shift → next lookup` creates a 2-cycle minimum per symbol (lookup + shift, with the shift feeding back).

**Hardware speculation** can break this:

```
Cycle N:   Lookup symbol S[i] from bits[0:15]        → result R[i]
           SIMULTANEOUSLY:
           Lookup from bits[1:16]  → speculative R[i+1][if_1bit]
           Lookup from bits[2:17]  → speculative R[i+1][if_2bit]
           ...
           Lookup from bits[15:30] → speculative R[i+1][if_15bit]

Cycle N+1: R[i] resolves: S[i] consumed K bits
           Select R[i+1][if_Kbit] → this is S[i+1], already decoded!
           Start 15 speculative lookups for S[i+2]
```

This requires a **15-port read** on the table SRAM (or 15 banked copies).
15 copies of 128KB = 1.92 MB of SRAM. That's enormous — roughly the size of
an M3's entire L2 cache.

**Practical alternative**: Use the probability distribution. 80%+ of symbols
have codes 7-9 bits long. Speculate on 3 likely lengths (7, 8, 9) instead of
all 15. 3-port SRAM (384KB) is feasible. Misprediction penalty: 1 cycle
(fall back to non-speculative path). Expected throughput:
**0.8 × 1 cycle + 0.2 × 2 cycles = 1.2 cycles/symbol average**.

At 3.5 GHz: **~4.4 GB/s single-context**.

---

## 4. Software Simulation Path

Before committing to silicon, prove the concept in software:

### 4.1 Speculative SIMD Decode (no custom hardware needed)

```rust
// Speculative decode: try all 15 possible next-symbol offsets
fn decode_speculative_simd(bitbuf: u64, table: &[u32; 32768]) -> [(u32, u8); 15] {
    // ARM SVE: svld1_gather with 15 indices
    // x86 AVX-512: vpgatherdd with 16 indices
    let mut results = [(0u32, 0u8); 15];
    for offset in 1..=15 {
        let shifted = bitbuf >> offset;
        let idx = (shifted & 0x7FFF) as usize;  // 15-bit table
        let entry = table[idx];
        results[offset - 1] = (entry, offset as u8);
    }
    results
}
```

This is implementable TODAY with SVE/AVX-512 gather instructions. If it achieves
>1.5x speedup over the current decode loop, the hardware case is strong.

### 4.2 Benchmark Protocol

1. Implement speculative SIMD decode in `consume_first_decode.rs`
2. Benchmark on SILESIA (match-heavy), SOFTWARE (literal-heavy), LOGS (mixed)
3. Measure: symbols/cycle, cache miss rate, branch misprediction rate
4. Compare to theoretical 1 sym/cycle hardware ceiling

**Success criteria**: >1.8x throughput improvement justifies hardware exploration.
<1.3x means software is already close enough and hardware isn't worth it.

---

## 5. Full Pre-Mortem

*It's 2036. The HUFFDEC extension shipped in RISC-V cores 3 years ago. It was
a failure. Here is every reason why.*

### 5.1 Amdahl's Law Killed It

Even Tier 3 (full micro-engine) only addresses the decode phase. In real
workloads:

- **Network-bound**: Data arrives at 10-25 Gbps (1.2-3.1 GB/s). The software
  decoder at 1.4 GB/s already can't saturate a 25G NIC. Adding hardware decode
  doesn't help — the bottleneck is the network, not the CPU.

- **Storage-bound**: NVMe SSDs deliver 7 GB/s sequential. The micro-engine's
  5.25 GB/s can't keep up with raw storage bandwidth. You'd need 2+ contexts
  to saturate one SSD. But at that point, software with 4 threads (4 × 1.4 =
  5.6 GB/s) already matches.

- **Decompression is rarely the bottleneck**: In web servers, database engines,
  and data pipelines, decompression is <5% of total CPU time. Doubling its speed
  saves <2.5% end-to-end. The silicon would have been better spent on more
  cache or wider issue.

### 5.2 The Table Loading Tax

Real-world deflate blocks are small:

| Source              | Avg block size | Symbols/block | Table build % |
|---------------------|---------------|---------------|---------------|
| Web content (CDN)   | 4-8 KB        | 800-1600      | 55-70%        |
| Log files           | 8-16 KB       | 1600-3200     | 38-55%        |
| Tarballs (gzip -6)  | 16-32 KB      | 3200-6400     | 24-38%        |
| Archives (gzip -9)  | 32-64 KB      | 6400-12800    | 14-24%        |

For web content — the most common use case — **more than half** the micro-engine's
time is spent building tables, not decoding symbols. The 1 symbol/cycle decode
rate is irrelevant when table build takes 2000 cycles per 1200-symbol block.

Table caching helps for streaming compression (same tree reused), but HTTP
responses from different servers have different trees. Cache hit rate in
production CDN workload: measured at **23%**. Not enough.

**Failure**: The effective throughput for CDN decompression was 2.1 GB/s, not
the advertised 5.25 GB/s. Only 1.5x over software — not enough to justify
custom silicon.

### 5.3 Format Obsolescence

By 2033:
- **zstd** became the default for HTTP (RFC 9659 Content-Encoding: zstd)
- **Brotli** dominates WASM and web asset delivery
- **LZ4** is standard for real-time streaming and database pages
- **gzip** lives only in legacy APIs and `.tar.gz` archives

The HUFFDEC extension accelerates only deflate. It cannot decode zstd (uses ANS,
not Huffman), brotli (uses context-dependent Huffman + transforms), or LZ4
(no Huffman at all).

The silicon became a **stranded asset** — die area permanently allocated to a
shrinking use case. Unlike AES-NI (AES will be relevant for decades), deflate
is being actively displaced.

**Counter-argument**: gzip is the cockroach of formats. It's been "dying" since
2015 and still accounts for 60%+ of compressed web traffic in 2026. But the
trend is real, and silicon investments amortize over 10+ year chip lifecycles.

### 5.4 Software Speculative Decode Matched It

In 2031, ARM shipped SVE2 with 512-bit vectors (8 × 64-bit lanes). The
speculative SIMD decode approach (Section 4.1) was implemented in software:

- 8-way gather: decode current symbol + 7 speculative next-symbols in parallel
- Select correct next-symbol based on current symbol's bit length
- Effective rate: **0.7 symbols/cycle** (vs 1.0 for hardware)

The software approach achieved **70% of the hardware throughput with zero silicon
cost**. Combined with 8 cores: 8 × 0.7 × 1.4 GB/s = **7.84 GB/s** — exceeding
the hardware micro-engine's 5.25 GB/s single-context.

The remaining 30% gap didn't justify the design, validation, and tapeout costs.

### 5.5 Validation Was a Nightmare

The deflate specification has **47 distinct edge cases** that the hardware must
handle correctly:

- Empty blocks (BTYPE=00 with LEN=0)
- Fixed Huffman blocks (BTYPE=01, hardcoded tree)
- Dynamic blocks with degenerate trees (1 symbol, all same length)
- Blocks with no literals (all matches)
- Blocks with no matches (all literals)
- Distance code 0 (invalid but some encoders emit it)
- Length 258 with distance 1 (RLE, 258 bytes of same byte)
- Back-reference crossing block boundary
- BFINAL=1 on first block
- Code lengths that don't form a valid Huffman tree
- Over-subscribed Huffman trees
- Under-subscribed Huffman trees (some decoders allow, some don't)
- HLIT=29, HDIST=29 (maximum code count)
- HLIT=0 (no literal codes? technically invalid but...)
- Code length code with only code 16/17/18 (repeat codes)
- Window full after match (32KB boundary)
- Match that would extend past end of block
- Concatenated gzip members
- Gzip header with FEXTRA, FNAME, FCOMMENT, FHCRC in various combinations
- CRC32 mismatch
- ISIZE mismatch
- Truncated input in every possible state

A silicon bug in edge case #31 (under-subscribed trees with repeat code 16 at
the start of the distance alphabet) was discovered 14 months after tapeout.
The microcode patch added 3 cycles of overhead to every DFLT.RUN invocation,
reducing sustained throughput by 8%. The patch couldn't be reverted because
curl, nginx, and the Linux kernel all relied on the corrected behavior.

### 5.6 Security Surface

The micro-engine introduced two novel attack vectors:

1. **Decompression bomb amplification**: A crafted 42-byte gzip input
   decompresses to 4.5 PB (the classic "42.zip" bomb). In software, the OS can
   kill the process. With the micro-engine's DMA engine, the output write
   bypassed normal memory protection, causing a kernel panic when the output
   buffer overflowed into kernel address space. CVE-2034-XXXX, CVSS 9.8.

2. **Side-channel via DFLT.STAT**: The `bytes_consumed` counter leaked
   information about the Huffman tree structure, enabling a chosen-ciphertext
   attack on TLS 1.3 with gzip compression (a CRIME/BREACH variant). The
   counter had to be disabled in security-sensitive contexts, eliminating one of
   the micro-engine's key features (progress monitoring).

### 5.7 Power Budget

| Component              | Static power | Dynamic power (active) |
|------------------------|-------------|----------------------|
| LitLen table SRAM      | 180 mW      | 340 mW               |
| Distance table SRAM    | 12 mW       | 22 mW                |
| History window SRAM    | 45 mW       | 85 mW                |
| Input buffer SRAM      | 6 mW        | 11 mW                |
| Decode FSM logic       | 8 mW        | 95 mW                |
| DMA engines            | 4 mW        | 120 mW               |
| **Total**              | **255 mW**  | **673 mW**           |

255 mW static power — **always on**, whether decompressing or not. On a 5W
mobile SoC power budget, this is 5.1% of total power consumed by a unit that's
active <0.1% of the time.

Power gating eliminates static power but adds 200+ cycle wake-up latency.
For short decompression bursts (common in mobile: decompress a 2KB API
response), the wake-up + table-load overhead exceeded the time saved by
hardware decode.

**On mobile, software was faster for payloads under 16KB** — which is 73% of
HTTP responses.

### 5.8 The Opportunity Cost

The 172KB of dedicated SRAM could instead have been:
- **172KB more L1 data cache** (2x capacity on some cores): ~15% IPC improvement
  across ALL workloads, not just decompression
- **172KB of register file**: enabling 2x more hardware threads
- **172KB of branch predictor tables**: reducing mispredictions for ALL code

A 15% IPC improvement benefits every program. A 3x decompression speedup
benefits only the <5% of cycles spent decompressing. The expected-value
calculation never favored HUFFDEC.

### 5.9 Multi-Core Already Solved It

By the time HUFFDEC shipped (2033), standard server CPUs had 128+ cores.
Parallel decompression (BGZF-style block splitting):

- 16 cores × 1.4 GB/s = **22.4 GB/s** (software, no custom silicon)
- 1 HUFFDEC context = **5.25 GB/s** (best case)

Even with 4 hardware contexts (4 × 5.25 = 21 GB/s), software parallelism on
commodity hardware was competitive. And the software approach worked with ANY
compression format, not just deflate.

### 5.10 Nobody Used the Instructions

The instructions required:
1. RISC-V core with the extension (limited availability)
2. Modified libc/zlib to use DFLT.* instructions
3. OS kernel support for the DMA regions
4. Driver for power management of the micro-engine

In practice, only 2 of 7 major RISC-V core vendors implemented the extension.
zlib-ng added support 18 months after first silicon. musl libc never did.
Go's compress/flate used it, but only on Linux (no FreeBSD or macOS support).
Python's gzip module never got a binding.

**Ecosystem reach after 3 years: ~12% of decompression workloads on RISC-V
platforms**, which were themselves ~8% of server deployments. Net impact on
global decompression throughput: **<1%**.

---

## 6. What Would Actually Work Instead

### 6.1 The Pragmatic Path: Speculative SIMD Decode (Software)

Implementable today. No silicon changes. Works on existing ARM SVE2 / x86
AVX-512 hardware.

```
Current:    lookup → shift → lookup → shift → ...
                4 cycles/symbol (serial)

Speculative: lookup + 7 gathers → shift → select + 7 gathers → shift → ...
                ~1.5 cycles/symbol (ILP via gather)
```

**Expected speedup**: 2-2.5x on literal-heavy data, 1.3-1.5x on match-heavy.
**Cost**: 0 (software only, runs on existing hardware).
**Risk**: Low (if it doesn't work, revert; no stranded silicon).

### 6.2 The Medium Path: Programmable Bit-Manipulation Unit

Instead of a fixed-function deflate decoder, add a **configurable table-lookup
accelerator** to the CPU:

```
TBLLOAD  table_id, base, size, entry_width
  # Load arbitrary lookup table into dedicated SRAM
  # Configurable entry width: 8/16/32/64 bits

TBLOOK   rd, rs1, table_id, mask_bits
  # rd = table[rs1 & ((1 << mask_bits) - 1)]
  # Single-cycle lookup in dedicated SRAM

TBGATHER rd_vec, rs_vec, table_id, mask_bits
  # Vector gather: rd_vec[i] = table[rs_vec[i] & mask]
  # N-way parallel lookup (N = vector width / entry_width)
```

This accelerates deflate, zstd (FSE/ANS tables), brotli (context-model tables),
and any algorithm that does table-driven decoding. Not format-specific.
Smaller die area (just fast SRAM + gather logic, no FSM).

### 6.3 The Long Path: Transparent Compressed Memory

Make decompression invisible by integrating it into the memory hierarchy:

1. Filesystem stores .gz files with block index metadata
2. mmap() a .gz file → MMU maps it with "compressed" page attribute
3. Page fault → hardware decompresses one block (64KB) into physical page
4. Subsequent accesses hit the cached decompressed page (zero latency)
5. Prefetcher can decompress ahead based on sequential access pattern

This requires co-design of: filesystem, OS virtual memory, MMU page table
entries, and decompression hardware. It's a 10-year project, not a 2-year ISA
extension. But it makes the performance question moot — decompression latency
becomes invisible, hidden behind prefetch.

**Precedent**: Apple's APFS already does this for LZFSE-compressed system files,
with hardware acceleration in the SSD controller. Extending to user-space .gz
files with in-core decompression is architecturally feasible.

---

## 7. Recommendation

**Don't build HUFFDEC.** Instead:

1. **Now (2026)**: Implement speculative SIMD decode in gzippy using ARM
   NEON/SVE gather or x86 AVX-512 gather. Measure. If >1.5x, publish paper.

2. **Medium-term**: Advocate for TBLOOK/TBGATHER (format-agnostic table
   accelerator) in RISC-V extensions working group. Useful beyond compression.

3. **Long-term**: Work with OS/filesystem teams on transparent compressed
   memory. This is the only approach that truly makes decompression "free."

The custom ISA extension is the wrong abstraction level. It's too specific
(only deflate), too expensive (172KB SRAM), and too fragile (47 edge cases in
silicon). The right answer is either lower (general table-lookup acceleration)
or higher (OS-level transparent decompression).

---

## 8. If We Built It Anyway: Minimum Viable Design

For completeness, here's the smallest useful hardware acceleration — not a
full micro-engine, but a single instruction that's actually worth the silicon:

### `HUFFDEC rd, rs1, rs2` (RISC-V R-type, custom-0)

```
Encoding: funct7=0000000 | rs2 | rs1 | funct3=000 | rd | 0001011

Operands:
  rs1 = bit buffer (u64, lower bits are current codes)
  rs2 = table base address (in normal memory, not dedicated SRAM)
  rd  = result: bits[31:8] = decoded value, bits[7:0] = bits consumed

Semantics:
  mask = rs1 & 0x7FFF            // 15-bit index
  entry = *(u32*)(rs2 + mask*4)  // table lookup (uses L1 cache, not SRAM)
  rd = entry                     // entry format matches LitLenEntry
```

**That's it.** It's just a specialized indexed load: `rd = *(rs2 + (rs1 & MASK) * 4)`.

**Why even bother?** Because the CPU can optimize this specific pattern:
- It KNOWS the access is to a 128KB-aligned region (can bypass TLB for table)
- It KNOWS the index is 15 bits (can skip address calculation overflow check)
- It can fuse with the subsequent `SRL rs1, rs1, rd` (shift by bits consumed)
- The fusion eliminates 1 cycle from the critical path

**Speedup**: ~15-20%. Barely worth it. But it costs almost nothing — no SRAM,
no FSM, just a specialized load micro-op and a fusion rule.

This is the honest answer: the minimum useful hardware acceleration for Huffman
decode is embarrassingly small, and everything bigger has a fatal flaw.

---

## Appendix A: The Serial Dependency Chain (Formal)

```
Symbol decode iteration i:

  entry_i    = TABLE[bitbuf_i & MASK]           // memory load, 3-4 cycles
  bits_i     = entry_i & 0xFF                   // extract, 1 cycle
  bitbuf_i+1 = bitbuf_i >> bits_i              // shift, 1 cycle (depends on entry_i)
  entry_i+1  = TABLE[bitbuf_i+1 & MASK]         // load, 3-4 cycles (depends on bitbuf_i+1)

  Critical path: entry_i → bits_i → bitbuf_i+1 → entry_i+1
  Length: 3-4 + 1 + 1 + 3-4 = 8-10 cycles for 2 symbols = 4-5 cycles/symbol
```

Modern OoO CPUs hide some of this via speculative execution and store
forwarding, achieving ~3.75 cycles/symbol in practice (measured on M3). The
hardware serial dependency floor is **2 cycles/symbol** (1-cycle lookup + 1-cycle
shift, perfectly pipelined). We're within 2x of the hardware floor already.

## Appendix B: Comparison to Existing Hardware Accelerators

| Accelerator       | Interface | Throughput  | Format     | Die area |
|-------------------|-----------|------------|------------|----------|
| Intel QAT 2.0     | PCIe 4.0  | 24 GB/s    | deflate    | ~20mm2   |
| AMD Zen 4 (none)  | -         | -          | -          | -        |
| Apple ANE         | MMIO      | N/A        | LZFSE only | ~5mm2    |
| Arm Neoverse CCA  | -         | -          | -          | -        |
| HUFFDEC (proposed)| ISA instr | 5.25 GB/s  | deflate    | ~3mm2    |
| TBLOOK (proposed) | ISA instr | ~3 GB/s    | any table  | ~1mm2    |

Intel QAT exists and achieves 24 GB/s — but it's a discrete PCIe accelerator
with 50-100us setup latency. Useless for small payloads. The in-core approach
avoids setup latency but can't match QAT's raw throughput (QAT has 20mm2 of
dedicated silicon; we're proposing 3mm2).

## Appendix C: Prior Art

- **US Patent 7,339,502** (IBM, 2008): Hardware Huffman decoder for JPEG
- **US Patent 9,432,633** (Intel, 2016): QAT deflate acceleration architecture
- **Xilinx GZIP IP** (2020): FPGA-based gzip, 8 GB/s at 300 MHz
- **NVIDIA nvCOMP** (2021): GPU-based decompression (LZ4, Snappy, zstd; no deflate)
- **Samsung CXL-based decompression** (2023): Near-memory decompression accelerator
- **rapidgzip** (Knaust, 2023): Software parallel decompression via block-finder
