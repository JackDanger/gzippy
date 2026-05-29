# Production decode call-graph (pure-Rust, x86_64)

Authoritative map of *what actually runs* on the pure-Rust decode path, to end
the "multiple lookalike implementations, lost track of which is production"
problem. The **live path** below is from the runtime span call-tree
(`GZIPPY_TIMELINE=… ; timeline_analyze.py --tree`, T8 silesia-large) — it shows
what executes, with no guessing across the lookalike decoders. Regenerate any
time; the tree is ground truth.

## Entry & routing (cfg-gated)
```
CLI → decompress::decompress_gzip_libdeflate → classify_gzip
    → DecodePath::IsalParallelSM            (x86_64, threads>1, compressed>4 MiB)
    → parallel::single_member::decompress_parallel → chunk_fetcher::drive
```
- `PARALLEL_SM = cfg!(x86_64 && (isal-compression || pure-rust-inflate))`.
- pure-rust build (`--no-default-features --features pure-rust-inflate`):
  `USE_ISAL_INFLATE = false` → the clean inner decoder is **ResumableInflate2**
  (Rust), not C ISA-L. THIS is the shipping target.
- isal-compression build: same pipeline, but `decode_chunk_isal_impl` uses the
  C ISA-L FFI for the clean decode. (Used only as the perf oracle now.)

## Live worker decode path (runtime call-tree)
```
pool.run_task → worker.decode_chunk → worker.scan_run → worker.scan_candidate
  ├─ window-ABSENT chunk (~31% on silesia): decode_chunk_marker_bootstrap_then_isal
  │    1. bootstrap_with_deflate_block → deflate_block::Block      [u16 marker ring]
  │    2. chunk.append_markered(&bootstrap.markers)                [COPY #1 — gzippy-only]
  │    3. decode_chunk_isal_impl → ResumableInflate2::read_stream  [u8 clean tail]
  │    4. absorb_isal_tail(&mut chunk, tail)                       [COPY #2 — gzippy-only]
  └─ window-KNOWN chunk: decode_chunk_isal_impl → ResumableInflate2 [u8 clean]
→ post_process.task: apply_window (resolve u16 markers) + narrow_u16_to_u8 [COPY #3] + crc
→ consumer.drain → write_all(chunk.narrowed) + write_all(chunk.data)
```
rapidgzip's equivalent decodes each chunk **one-pass** into 128 KiB-segmented
`data_with_markers`/`data` buffers — it has **no** append_markered / absorb_isal_tail
(COPY #1/#2). Those 382 ms of merge copies are gzippy-specific (confirmed three
ways: fault-sampling 66% in memmove, the span trace, and this call-tree).

## The TWO production inner decoders (pure-Rust)
| decoder | file | role |
|---|---|---|
| `ResumableInflate2` | `src/decompress/inflate/resumable.rs` | clean u8 resumable FASTLOOP; window-known chunks + bootstrap ISA-L tail |
| `deflate_block::Block` | `src/decompress/parallel/deflate_block.rs` | window-absent u16 marker ring; the bootstrap |

## NOT on the production path (do not optimize; candidates for deletion task #6)
- `consume_first` (`inflate/consume_first_decode.rs`): `examples/inner_bench` + the
  inline match-copy helpers `ResumableInflate2` calls; non-resumable. (Helpers ARE live.)
- `isal_lut_bulk`: env-gated `GZIPPY_ISAL_PURE_BULK` (OFF).
- Cluster `combined_lut` / `packed_lut` / `simd_huffman` / `two_level_table` /
  `ultra_fast_inflate` / `double_literal` / `jit_decode` / `specialized_decode`:
  mutually-referencing; **not observed on the production call-tree**. Reachability
  from production is UNVERIFIED by grep (they reference each other) — confirm
  unreachable before deletion (golden + fuzz gate). `vector_huffman` already deleted.
