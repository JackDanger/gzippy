# 2026-11-12 — FIRST Linux frozen-box WALL census + the renegotiated-band parity matrix — termag `jdanger-rust-builds`

The wall axis of the per-label board, first time adjudicated off-Mac: fulcrum `wallcensus` under a **guard-held box freeze** (`FREEZE=OK boost=NA/sigstop[tenants]+watchdog, ttl=64800` — the first attempt's 7200s default TTL watchdog force-released mid-run and the instrument's own receipt says "re-run with a bigger ttl"; run resumed from cached cells, VOID never trusted), vanilla W2 binary (sha `a69c0eabab60…`), identified W3 (`gzippy=e2c73d4456 dirty=0`), n=9 interleaved arch-alternating reps per cell, L1-12 × T1,2,4 × 11 TUNE members, four rivals (gzip 1.12, pigz 2.8, libdeflate-gzip 1.19, vendored igzip 2.31.1). Companion: the parity census (fulcrum parity, L0-9 × T1,2,4,8 × --roundtrip, `gzippy=4058269002…` post-#472 tree) whose divergence table doubles as the renegotiated-band audit (below).

## Wall headline (1,025 measured; 462 rival-absent by `goal::rival_accepts_level`; 97 VOID rows each naming its own reason in census.json — void is never a result)

| rival | measured | ours-slower beyond band | ours-faster/tied | worst loss |
|---|---:|---:|---:|---|
| gzip 1.12 | 280 | **1** | 279 | `data.json` L9-T4 **+33.9%** |
| pigz 2.8 | 337 | **26** | — | (cells tabulated; all graded per charter) |
| libdeflate 1.19 | 369 | **96** | — | `movie.mp4` L1-T1 **+99.4%** |
| igzip 2.31.1 | 63 | **38** | — | `movie.mp4` L3-T1 +72.3% |

161 slower / 889 won-or-tied. The L1-T1 +40-99% cells vs libdeflate = the standing rustc-vs-C Ir-tax bucket, **now enumerated per-cell on Linux**; the L6-T2 data.csv +92% vs libdeflate is a new kind of finding (chunked-pipeline seam cost on CSV cadence at exactly 2 threads); `data.json` L9-T4 +33.9% is the single named near-opt hot cell. Full per-cell table: `census.tsv` + fulcrum's own `summary.txt`.

## Parity matrix = the renegotiated band's first real-world audit

The parity census's own premise ("T>1 == T1") is the *pre-charter-v2* contract, superseded by #447 (T>1 may emit different bytes than T1). Its output still grades the CURRENT renegotiated bands exactly:

- **L1-8: T2 == T4 byte-identical, 11/11 members at every level** — the preserved band holds on Linux arm64 everywhere it is claimed.
- **L9: T2 ≠ T4 on 11/11** — the wave-9 (#458) flip is doing exactly what the renegotiated contract says (T2 = chunked Lazy2-600; T3/T4 = near-opt vec).
- **T4 vs T8: diverges for most members at most levels** — expected: T8 was never inside the renegotiated cross-`-p` band (thread_byte_parity asserts the T2/T3/T4 band only). Recorded as an OPEN band-scope question, not a failure.
- **L0: T2/T4/T8 == T1 on 33/33** (STORED mode, routing-inert) and `IDENTICAL` settled the route-proof needle story: #470's needle + #472's `-v`-gates-the-route-print fix are what made this run possible at all — both receipted in their PRs.

## Notes

- The `Ng the -p2 flip 90-level T2 streams` renegotiation question (size-side) is the wave-in-progress complement to this receipt: the SIZE side of the same band failed per #471's census, so the L10-12-T2 family has both a size exposure and (now) its wall picture.
- Raw receipts beside this README: `census.tsv`/`census.json`/`summary.txt`/`meta.json` (wall) + `census/parity.tsv` (440-cell divergence matrix with T1/T2/T4/T8 sha columns).
- The instrument fixes that made this run possible are receipts-in-code: #470 (ParityNeedle), #472 (`-v` route print — gate was `GZIPPY_DEBUG`-only; double function-local OnceLock falsified by the committed `set_then_route_assert_agrees` probe).
