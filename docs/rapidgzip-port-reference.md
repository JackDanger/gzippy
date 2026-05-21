# rapidgzip port reference (archived)

> **Archived May 2026.** This file previously held a ~1,600-line vendor↔gzippy
> line-by-line matrix for the `feat/cross-chunk-retry` rapidgzip port effort.
> The unwired “museum” modules it described (`deflate_block`, `parallel_gzip_reader`,
> `fast_marker_inflate`, Huffman table ports, split `blockfinder_*` files,
> `gzip_reader`, `index_file_format`, etc.) were **removed from the tree** so
> production routing stays obvious. Recover the old document from git history
> if you need the detailed mapping (e.g. `git log -p -- docs/rapidgzip-port-reference.md`).
>
> **Production parallel single-member path (current):**
>
> ```
> decompress::decompress_single_member (x86, T>1, >10 MiB)
>   → parallel::single_member::decompress_parallel
>   → sm_driver::read_parallel_sm
>   → chunk_fetcher::drive
>   → gzip_chunk::decode_chunk_isal_inexact  (on-demand + speculative prefetch)
>   → inflate_wrapper::IsalInflateWrapper
> ```
>
> Supporting modules kept under `src/decompress/parallel/`: `block_finder`,
> `block_fetcher`, `block_map`, `chunk_data`, `window_map`, `apply_window`,
> `replace_markers`, `prefetcher`, `statistics`, `thread_pool`, `gzip_format`, …
>
> For design intent and type layout, see `docs/rapidgzip-port-design.md` (also
> marked historical where it diverges from the tree). For meta-review of the
> port trajectory, see `docs/codex-meta-audit.md`.
