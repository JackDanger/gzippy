# 2026-11-09 — L12-T>1 size vs libdeflate -12 — CLOSED as a tie at every thread count (per-thread table)

Wave-4's "+1,352 B" worry does not reproduce from the current binary on the stamped corpus (16,085,866 B, sha `1010fa2f…`; ldc -12 = 3,726,533, sha `63071572…`).

**Per-thread size table (gzippy -12, sizes exact):**
| `-p` | size | vs ldc -12 |
|---|---|---|
| 1 | 3,728,679 | +1,146 B (+0.031%) |
| 2 / 4 | 3,728,785 | +0.034% |
| 6 | 3,728,260 | +0.046% |
| 8 | 3,728,585 | +0.055% |
| 10 / 12 | 3,729,166 | +0.071% |
| 14-32 (plateau) | 3,730,144 | **+3,611 B (+0.097%)** — worst case, 115 B inside the 0.1% band ceiling (3,730,259) |

Roundtrips: python gzip + zlib(wbits 31) + vendored `gzip -t` on all 11 outputs; decompressed sha == corpus sha.

**Rules applied:** all cells inside the recorded 0.1% tie band → the cell is a TIE (`TIE BAND`, charter v2). **CLARIFICATION when citing:** the historical figure "L12-T>1 = 3,728,679" is the `-p1` output — the T>1 sizes are thread-partition-dependent (deterministic per `-p`, byte-identical reruns at -p4/-14), so cite the per-thread table, not one number. No passes repricing triggered (the band holds at the saturated grid with 115 B margin).

Sweep negative results recorded: `igzip -0` label verified correct in the tree (`vendor-technique-index.md` E7 already states igzip L0 uses a trained static table — no "stored" claim exists anywhere to fix); `nice_match_length = 96` confirmed to have exactly one production site (the L9 route arm, `level.rs:595`) — no other level inherited it (L7 130, L8 258/max, L10 75, L11 150, L12 max verified).
