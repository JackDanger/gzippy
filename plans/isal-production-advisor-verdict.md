I've inspected the code first-hand. Here's my verdict.

## VERDICT: UPHELD-WITH-CAVEATS

### Break attempts that FAILED (claims survived)

**Partial-mutation on fallback (claim 3).** I traced every `Ok(false)` return in `finish_decode_chunk_isal_oracle`. All five (is_available, window-len≠32KiB, FFI `None`, until_exact no-exact-boundary, over-decode decline) return **before** `chunk.data.commit`, `note_inner_decoded_bytes`, `append_block_boundary_at`, and `finalize_with_deflate`. `writable_tail_reserve` grows spare capacity only — uncommitted ISA-L bytes are overwritten by the pure path. So a decline is a genuinely clean handoff, not a half-mutated chunk. Survived.

**Over-decode accept corrupts (claim 2a).** The `end_bit <= stop_hint_bits` accept branch is provably reachable only via `ISAL_BLOCK_FINISH` (BFINAL): input-exhaustion stops give `avail_in==0` → `end_bit ≈ slice_end*8 ≫ stop_hint` (slice extends +256 KiB). Stronger still — `decompress_deflate_from_bit_into` returns `None` when `bits_remaining > 64` after FINISH, so the accept fires only when the member ends within ~64 bits of `slice_end`, i.e. the true final chunk. Narrower and safer than the commit claims. Survived.

**Subchunk accounting drift (claim 2b).** Incremental `note_inner_decoded_bytes(off-prev_off)` + final `keep_len-prev_off` telescopes to exactly `keep_len`; `off=min(keep_len)` and ascending boundaries keep `prev_off` monotone. No total drift. Survived.

### Genuine CAVEATS (real holes, non-corrupting)

1. **Faithfulness is conditional, not categorical (claims 1/3).** The until_exact-no-exact-boundary and stored/fixed-block over-decode cases route the clean tail to **pure-Rust where rapidgzip-WITH_ISAL uses `IsalInflateWrapper`**. On all-dynamic silesia this is 0–1 chunks. But on a **stored-block- or fixed-Huffman-heavy corpus, declines become the common case** — `gzippy-isal` silently degrades to `gzippy-native` (zero ISA-L coverage), and the "production ISA-L clean tail" / TIE-vs-rg perf claim would not hold there. Correctness is safe (counted, byte-exact), but "faithful WITH_ISAL" is generous; it's "faithful *where ISA-L's EOB-stop contract is honored*." The `ISAL_ENGINE_ORACLE_FALLBACKS==0` gate only proves this on the all-dynamic corpus — it cannot certify faithfulness on the very input class that triggers declines.

2. **Verification is asserted, not reproduced.** The dual-sha pin and 10/10 differential gate run only on the x86_64 ISA-L guest; I could not re-execute them on this arm64/darwin host. Claim 2 rests on their harness, which I read but did not run.

**Bottom line:** the byte-exactness and fallback-safety machinery is structurally sound and survives adversarial reading. The "faithful" label is the soft spot — defensible on the target corpus, overstated for stored/fixed-block inputs where it silently provides no ISA-L coverage. UPHELD-WITH-CAVEATS.
