# ISA-L FFI for L0-L3 Compression

## Goal
Close the compression speed gap at fast levels. ISA-L's L0-L3 uses AVX-512/AVX2 hand-tuned
assembly for LZ77 matching, achieving 2-3 GB/s — 3-5x faster than zlib-ng at these levels.

## Current state
- L1-L5: zlib-ng via flate2 (~400-600 MB/s)
- L6-L9: pipelined zlib-ng with dictionary sharing (37-78% faster than pigz)
- L10-L12: libdeflate ultra compression (near-zopfli quality)

## What the git history tells us
- `jackdanger/gzip-build` evaluated ISA-L and rejected it for portability (autotools)
- But we already link libdeflate-sys via FFI for decompression
- The `isa-l` submodule exists in the repo root (currently untracked)
- ISA-L's compression is its strength; its decompression was rejected because
  libdeflate is simpler and nearly as fast

## Approach

### Option A: isa-l-sys crate (preferred if it exists)
- Check if a Rust `-sys` crate for ISA-L compression exists
- Would provide automatic build integration like libdeflate-sys

### Option B: Build ISA-L ourselves
- The `isa-l` submodule already exists in the repo
- ISA-L supports cmake in addition to autotools (cmake is more portable)
- Build via build.rs with cc crate or cmake crate
- Link statically

### Integration plan
1. Add ISA-L as optional dependency behind a feature flag (`isal-compression`)
2. Create `src/isal_compress.rs` with FFI bindings to `isal_deflate`
3. Route L0-L3 compression through ISA-L when available
4. Fall back to zlib-ng if ISA-L not available (e.g., ARM without NEON fast path)
5. Keep L4-L9 on zlib-ng (ISA-L only has L0-L3)
6. Keep L10-L12 on libdeflate

### Compression level mapping
```
Level  Current backend    New backend (x86)   New backend (ARM)
0      zlib-ng store      ISA-L L0            zlib-ng (ISA-L has no ARM SIMD for compress)
1      zlib-ng L2         ISA-L L1            zlib-ng L2
2      zlib-ng L2         ISA-L L2            zlib-ng L2
3      zlib-ng L3         ISA-L L3            zlib-ng L3
4-9    zlib-ng L4-L9      zlib-ng L4-L9       zlib-ng L4-L9
10-12  libdeflate         libdeflate          libdeflate
```

### ARM considerations
- ISA-L compression uses SIMD only on x86 (AVX2/AVX-512)
- On ARM, ISA-L falls back to C — no faster than zlib-ng
- So this feature primarily benefits x86/x86_64 platforms
- Consider making it x86-only to avoid unnecessary complexity on ARM

## Validation
- Compress with ISA-L, decompress with gzip/pigz/igzip — verify round-trip
- Benchmark L1-L3 compression speed vs pigz -1/-2/-3 and igzip
- Verify compression ratio is acceptable (ISA-L L0-L1 produce larger output)
- Verify parallel compression still works with ISA-L backend
