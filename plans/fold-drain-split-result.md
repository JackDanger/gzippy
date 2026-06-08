# FOLD second-touch split — advisor-owed decomposition (owner turn 2026-06-08)

The disproof advisor (plans/clean-rate-ceiling-advisor-verdict.md) REFUTED C1/C3 on
the model that ocl_cf's 36ms (440→404) bundles {symbol-rate + ring-write + ring→data
drain memcpy + CRC}, estimating the 200MB second-touch at 10-20ms (up to HALF the
"engine" headroom). It required running GZIPPY_FOLD_NODRAIN / FOLD_NOCRC to split the
36ms before bounding the STEP-2 engine techniques.

## RAN IT (frozen host, matched /dev/shm sink, best-of-N min — min is jitter-robust)

N=21 (the box was under intermittent load; med/spread inflate but the best-of-N MIN
converges — all arms pick their cleanest run):

| arm | min wall | what |
|---|---|---|
| native_OFF (production, byte-exact == pin) | 0.4348 s | full path |
| NODRAIN (skip ring→data drain memcpy) | 0.4354 s | drain removed |
| NODRAIN+NOCRC (skip drain + per-byte CRC) | 0.4342 s | both removed |
| rapidgzip | 0.3722 s | target |

**Deltas (min-based):**
- drain memcpy = OFF − NODRAIN = **−0.6 ms** (noise; sign-negative)
- per-byte CRC = NODRAIN − NODRAIN+NOCRC = **1.2 ms** (negligible)
- **total second-touch (drain + CRC) = OFF − NODRAIN+NOCRC = 0.6 ms** (noise floor)

(Two earlier N=11/N=15 attempts under heavier load gave incoherent ±5ms sign-unstable
deltas — consistent with the term being BELOW the noise floor, not with a 10-20ms term
that would survive best-of-N min.)

## VERDICT: the advisor's magnitude is REFUTED by measurement; C1/C3 RESTORED

The ring-drain + CRC second-touch is **~0-1 ms, NOT 10-20 ms.** Reconciliation
(source-verified, gzip_chunk.rs:860-919): the bulk drain memcpy was ALREADY deleted in
a prior BANKED turn (commit 0f5bc85b, copy-free-to-final Stage 2). The production FOLD
path decodes the clean tail STRAIGHT into the pre-reserved contiguous chunk.data; the
only remaining "drain" is `extend_from_slice` from a warm ring slice into a contiguous
pre-reserved tail = a single sequential memcpy of warm data = ~0.6ms, not a cold
200MB second-touch. The code comment's "UPPER BOUND on intrinsic symbol rate" caveat is
technically true but quantitatively ~1ms, so:

**ocl_cf 404ms IS a valid speed-UP ceiling for the STEP-2 pure-Rust engine techniques,
to within ~1ms.** The 36ms (440→404) is ~97% pure-Rust-vs-ISA-L SYMBOL RATE. The four
authorized techniques (table _mm_prefetch, single-level L1 table, static-Huffman,
FASTLOOP yield-elision) attack exactly that symbol rate and CAN approach 404ms.

C2 stands (advisor UPHELD-WITH-CAVEATS): even at the ideal ISA-L clean rate, gzippy is
0.945× rg = 21ms short — that residual is OUTSIDE the clean engine (marker-region
pure-Rust decode + bootstrap structure + placement), a SEPARATE term that bounds the
no-FFI 1.0× bar independently.

## Frozen-host matched picture (the decisive numbers)
| run | wall | ratio vs rg |
|---|---|---|
| rapidgzip 0.16.0 | 376-383 ms | 1.000× |
| ocl_cf (ISA-L clean engine, copy-free, coverage 14/0) | 404 ms | 0.945× |
| production native (pure-Rust) | 434-440 ms | 0.857-0.870× |

Engine symbol-rate lever = ~36ms = 0.870×→0.945×. Non-engine residual = ~21ms =
0.945×→1.0×.
