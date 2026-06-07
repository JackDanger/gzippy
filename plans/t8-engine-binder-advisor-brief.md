# DISPROOF BRIEF — is the T8 binder the ENGINE (per-thread decode), or scheduling/consumer-wait?

You are an INDEPENDENT DISPROOF advisor. Read-only. Try to BREAK the claim below.
Do NOT trust the campaign charter's prior conclusion; it is exactly what this brief overturns.

## CONTEXT
gzippy is a Rust gzip decoder racing rapidgzip (C++, vendor/rapidgzip/) to a 1.0x wall TIE on
silesia (211968000 bytes decoded) at T8. The parallel single-member path mirrors rapidgzip's
GzipChunkFetcher/BlockFetcher: N workers decode chunks, an in-order consumer publishes windows +
combines CRC + writes. Both expose identical --verbose pool stats (gzippy ported the stat names
1:1 from BlockFetcher.hpp:95-118).

## THE PRIOR CHARTER CONCLUSION I AM OVERTURNING
The charter (CURRENT STATE, HEAD fb3baec0) concluded: "gzippy's perfectly-parallel decode FLOOR
(Theoretical-Optimal 0.117s) is ALREADY ~= rapidgzip's ENTIRE wall (0.130s). The whole 1.70x gap
is the scheduling/serial term: pool-fill gap + in-order consumer future::get wait (~0.077s) ~=
~0.10s = the dominant T8 binder. rapidgzip ties DESPITE the same engine gap by overlapping decode."

## MY MEASUREMENT THIS TURN (first-hand, locked guest REDACTED_IP via ssh neurotic->guest, 16c
gov=performance, box load ~2.5 so use INTERNAL SPANS not wall absolutes; gzippy-mk2 byte-exact
sha 028bd002...cb410f via path=ParallelSM; rapidgzip 0.16.x --verbose; 3 runs each, stable):

Per-tool --verbose pool stats, T8 (CPUS=0,2,4,6,8,10,12,14), silesia:
| metric                       | gzippy        | rapidgzip     | ratio |
| decodeBlock (SUM over workers) | 0.93 s      | 0.50 s        | 1.86x |
| Theoretical-Optimal (=/8)    | 0.118 s       | 0.068 s       | 1.74x |
| Total Real Decode Duration   | 0.139 s       | 0.086 s       | 1.61x |
| std::future::get (consumer wait) | 0.077-0.082 s | 0.062-0.067 s | ~1.25x|
| Pool Fill Factor             | 85%           | 78%           |       |

decodeBlock is the SAME span in both (vendor BlockFetcher.hpp:117 futureWaitTotalTime +
decodeBlock timer; gzippy ported it). gzippy's body_rate is 269 MB/s; header 23.8ms (2.5%).

## THE CLAIM (what I will act on if it survives your disproof)
1. The charter's premise is a UNIT ERROR: it compared gzippy's decode FLOOR (0.118s) to
   rapidgzip's WALL (0.130s) and concluded "floor ~= rg wall, so gap is scheduling." But
   rapidgzip's decode FLOOR is 0.068s (its own Theoretical-Optimal), NOT 0.130s. The correct
   comparison is floor-to-floor: 0.118 vs 0.068 = 1.74x. The engine IS the gap.
2. The T8 binder is the ENGINE (per-thread decode throughput): decodeBlock 0.93s vs 0.50s = 1.86x.
   This matches the long-observed CONSTANT ~1.7x ratio at BOTH T1 and T8 (flat-across-T = the
   signature of a per-thread throughput gap, which the charter itself stated).
3. The consumer future::get gap (0.077 vs 0.062 = 1.25x) is a MINORITY and largely DOWNSTREAM of
   the slow engine (the consumer waits longer because each chunk takes longer to decode). It is
   not the dominant binder.

## VENDOR SOURCE I VERIFIED (BlockFetcher.hpp:246-329)
rapidgzip's get() also pumps prefetchNewBlocks() in a `while(wait_for(1ms))` loop during the
future wait (line 314-316), exactly as gzippy (chunk_fetcher.rs:1289 Lever H pump). So the
consumer-overlap STRUCTURE is already faithfully ported; future::get is non-zero in BOTH. There
is no missing overlap mechanism that would explain a 1.7x wall gap.

## FAILED ORACLE (full disclosure — why I did NOT bound the ceiling via bypass)
I tried the decode-bypass oracle (GZIPPY_BYPASS_DECODE: memcpy-replay captured chunks, decode~=0,
full coordination chain, BYTE-EXACT sha verified) AND the GZIPPY_SLEEP_DECODE_NS=0 variant
(zeroed correct-size chunk, full chain). BOTH gave T8 wall ~1050-1080ms = 3.6-5.5x SLOWER than
real decode (290ms). A decode-FREE run cannot be slower than full decode for a true coordination
floor => both oracles are CONFOUNDED: they bypass the buffer pool (Buffer pool u8: hits=0) and
do fresh full-size zeroed allocs per chunk that page-fault and hold up to 33 ChunkData (660MB)
live at once, plus single-threaded CRC over 212MB of zeros un-overlapped with decode. So I bound
the binder by the FLOOR-TO-FLOOR span comparison instead (decodeBlock 1.86x), not by removal.

## YOUR JOB — try to BREAK claims 1-3. Specifically:
- D-A: Is decodeBlock measured comparably? Could gzippy's decodeBlock span include work
  rapidgzip's excludes (e.g. window-publish, marker-resolve, alloc) inflating it spuriously?
  (gzippy decodeBlock is the worker run_decode_task timer; check whether marker/u16 post-flip
  work counts in it. 57.5% of body bytes are "post_flip_u16" per the trace — though a prior
  advisor found that counter is MIS-NAMED and actually counts marker-FREE u8 blocks.)
- D-B: Is the floor-to-floor comparison valid given different chunk sizes / chunk counts?
  gzippy: 18 chunks T8. rapidgzip: 17 fetched. Could a chunking difference (not engine speed)
  explain decodeBlock 1.86x? (Same input bytes decoded in both.)
- D-C: Could the 1.25x future::get gap actually be the LEADING term once you account for the
  fact that decodeBlock is a SUM over workers (so /8 understates tail effects)? i.e. is the wall
  gap really tracking the floor or the consumer wait?
- D-D: Is "engine 1.86x slower" consistent with the byte-exact TIE-vs-self and the prior
  ENGINE-BENCH-ROUND-2 PLATEAU finding (pure-Rust engine ~2.4x slower than ISA-L in isolation)?
  Does that CORROBORATE (engine is the binder, and it plateaus in pure-Rust) or CONTRADICT?
- D-E: Given the engine is the binder and round-2 found pure-Rust+ASM plateaus at ~2.4x ISA-L,
  is the 1.0x bar reachable at all in pure-Rust without FFI? (This may be a user-constraint FORK.)

Verdict format: UPHELD / UPHELD-WITH-CAVEATS / REFUTED for each claim, with the specific
disproof attempt and what survived. Source-verify any file:line you rely on. Be adversarial.
