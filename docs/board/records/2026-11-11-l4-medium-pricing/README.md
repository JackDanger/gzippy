# L4 zlib-ng `deflate_medium` rung — pricing record for the open cell `data.sqlite:L4:T1`

Pricing run, **2026-11-11** (probe/l4-deflate-medium, worktree
`.review-tmp/wA-medium/`, parent `main` @ `30252fba`, probe-scoped, default
OFF, nothing pushed). Prober/builder: wave-13 pricing agent (laptop, M1
aarch64). Design authority: `docs/board/attack/l4l5/nn-deflate-medium-rung.md`
(the attack memo); methodology template: the L5-depth32 falsifier record
(`docs/board/records/2026-11-11-l5-depth32-falsifier/README.md`).

## 0. The card and the rung

- Cell: `data.sqlite:L4:T1`, ours-vs-rivals `+0.13%` (gzip -4) / `+0.17%`
  (pigz -4) — the §2b board survivor. The T4 leg of this cell is already
  closed by the shipped L4-at-T>1 strategy step (Lazy + depth x4,
  `level.rs` `default_params_parallel_route`, the "L4-only clean subset"
  receipt); the T1 leg is the open one.
- Rung: zlib-ng's `deflate_medium` family ported INSIDE the ldx engine — the
  memo's §3.1/§3.3 shape: greedy's knobs, accept predicate and emission
  helpers, plus the look-ahead probe whose result is cached as the next
  iteration's current, plus the bounded `fizzle_matches` boundary slide. Per
  the memo's own read of `:185 (early_exit = s->level < 5)`: zlib-ng's L4 runs
  medium with the probe OFF; the compression win the rung is named for comes
  from the probes and fizzle — the L5 shape. This rung probes at L4.
- Vendor source read at `vendor/zlib-ng/deflate_medium.c` (265 lines, cited
  `:NNN` throughout the module header).

## PROVENANCE (verify before quoting any number)

```
branch  probe/l4-deflate-medium (worktree .review-tmp/wA-medium)
  commits: a7c65dd8 (the rung port) -> 03013b83 (stats atomics + family test)
           -> d54c1960 (comment share fix); tip d54c1960
parent  main @ 30252fba (origin/main identical: 30252fbafb4cf8bfccb7d716e7ce89f29aa242a5)
gzippy --version  vanilla-bin: gzippy 1.0.0 (parallel-sm+pure)
                  tune-bin:    gzippy 1.0.0 (parallel-sm+pure+INSTRUMENTED(ladder-tune))
sha256 vanilla release bin : 898b39843062fbb55a2fe15f8daca5471edbecaf85af465cdc39ef93325f77ab
sha256 tune release bin    : 7d30ecb4bb81158f4096839d0087f87a9ff92c13a6520e1af95ab1fb33e48a35
  (post-03013b83 build, the one the wall battery v3 ran; the SIZE sweep ran on
   the pre-atomics build sha 495f1a7256f61134071c286793204d23612bfd48c5f2ca552b21644e8af89e54 —
   identical output bytes re-verified on data.sqlite: sha 3a1add9b84bbb9f4…92bc019 from both
   builds)
builds: cargo build --release [--features ladder-tune] --target-dir target-{vanilla,tune}
       (isolated per arm — the user-global target-dir must not be shared; CLAUDE.md rule)
rivals: vendor/gzip (sha 186fd9d72…), vendor/pigz v2.8 (sha 9822a14a…),
        vendor/libdeflate @ 22f6e02 built fresh (scratch/libdeflate-cbuild)
oracle: /usr/bin/zcat (Apple gzip 479.0.0.700.1 — gzippy-shim check: --version
        names no gzippy; stdin-driven)
member input shas: scratch/member-input-sha256.tsv (written by the sweep)
sweep log/script + wall log/script: scratch/{sweep.log,sweep.py,wall.log,wall.py}
committed test suite: vanilla release `cargo test` 39/39 suites ok (worktree);
  ladder-tune lib suite: 1228 passed / 2 FAILED, both failures reproduce on
  a clean main @ 30252fba `--features ladder-tune` build
  (`l1_match_reach_is_t1_only`, `l9_retune_vs_l8_alias_routing_is_deterministic`)
  — the wave-9 params_parallel T>1 override, pre-existing, not this branch
```

A/A CONTROL (route assertion): on all 31 member files, the tune binary with
`GZIPPY_LDX_MEDIUM` UNSET produced byte-identical output (sha256 equality) to
the vanilla binary at `-4 -p1`. The probe changes nothing by default; the env
reaches exactly the shipped L4 encode; a shipped build compiles the module
away and reads no env (non-negotiable #3 untouched, ladder-tune discipline).

## 1. The port — what shipped on the branch, and the two soundness fixes over the memo

`src/compress/ldx/compress_medium.rs` (new, ladder-tune-gated; module cfg'd
out of shipped builds), `compress.rs` dispatch thorttle (L4 only; the L2 arm
splits out with byte-identical bodies), `mod.rs` registration. The eight
documented deviations from the vendor live in the module header; two of them
are CORRECTIONS to the memo's own sketch, and are the load-bearing findings
of this port:

1. **The probed position must never be re-searched, so the memo's per-block
   cache reset is UNSOUND and the cache carries across block boundaries.**
   `hc_matchfinder_longest_match` INSERTS the string it searches (`:225-233`
   of `hc_matchfinder.rs`), so a probed position is in the tables the moment
   the probe returns; a fresh search at that position would self-link
   `next_tab[pos] = pos` (the memo's own §2 warning) and later walk it as an
   offset-0 "match". Resetting `next_live` at block boundaries would produce
   exactly that double insert on the block's first unit. The vendor carries
   `next_match` across `FLUSH_BLOCK` for the same reason; the port carries the
   one cached candidate across the boundary (block state — split stats,
   sequence store, the min-match heuristic — stays per-block).
2. **The commit rule admits `c_len == 0`.** The vendor's slide guard runs
   BEFORE its decrement, so the chain can end at `c.match_length == 0` and
   `:169`'s `<= 1` commit admits it (current emits no symbol; the slid match
   starts at the current position). The memo's §3.3 sketch claims `c_len >= 1`
   always and stops at 1; the vendor source is the semantics authority and the
   port follows it. Measured weight: c0 commits are rarer (~1-10% of commits
   by member) but nonzero everywhere.

Both fixes are exercised by the module's unit tests (round-trip across block
shapes through an independent decoder; byte-identity to greedy on a verified
matchless 4 KiB stream; a slide-favoring stream where the rung moves bytes).

The other deviations, verbatim-tracked per the memo: (a) the len-3 accept
predicate stays greedy's (`> 3 || offset <= 4096`) — the vendor's
`WANT_MIN_MATCH=4` ban is not adopted (the far-len-3 lever family stays
intact); (b) the `16 * max_insert_length` insert cap is not adopted (wall
knob, not correctness); (c) the `orgstart` dedup IS required and implemented
(vendor `:54-64`); (d) the vendor's slide cap 256 is taken verbatim (`:155`);
(e) tail guards map to in-buffer bounds (`in_end - q >= 5` probe floor, the
slide's window-edge limit mapped to "slid source stays in the buffer").

## 2. The measurement box — members and limits

- 15-member bench set **of record** (the L5-depth32 record's corpus, re-staged
  in this worktree and byte-compared member-for-member against that worktree's
  staged copies — `scratch/members/`): 12 silesia members + `software.archive`
  + `logs.txt` (221,249,619 B) + `logs.txt-card26m` (26,214,398 B slice).
- The campaign corpus root (`~/www/gzippy-bench/corpus`, 16 members) is ALSO
  swept: it carries the REAL `data.sqlite` (48,308,224 B,
  sha `21a73d3d…67c2634`) — the target cell's own file, present on this box —
  plus the W2 thin-band members (`aozora.txt`, `dickens`, `data.json`,
  `tool.bin` — `symbols.dwarf` is not on this box), the L5 sibling card class
  (`minjs.min.js`), and the near-incompressible class (`movie.mp4`,
  `photo.jpg`, `weights.safetensors`, `ecoli.fastq`, …).
- All sizes at `-4 -p1`, raw streams, byte counts from the produced stream,
  sha256 of every arm's stream, every arm round-tripped through the
  independent oracle back to the member bytes (both arms — ship AND medium).

## 3. Sizes — per-member table (deterministic bytes, `-4 -p1`, every arm round-trips)

Raw per-arm sizes + sha256 per member x arm: `scratch/sizes.tsv`; member input
shas: `scratch/member-input-sha256.tsv`; sweep log: `scratch/sweep.log`
(A/A pre-pass FIRST, then the 31-member sweep; every medium row stamped
`rt-ok` through `/usr/bin/zcat`; the ship arm is stamped in the TSV column
`ship_rt` too — all rows `rt-ok`).

Sorted to put this record's question first (subset shown; full table in
`sizes.tsv`, all 31 members):

| member | in B | ship (ours L4) | medium L4 | Δ ship | Δ pigz -4 | Δ gzip -4 |
|---|---:|---:|---:|---:|---:|---:|
| data.sqlite (THE CELL) | 48,308,224 | 14,804,403 | 13,753,190 | -1,051,213 (-7.10%) | -1,026,714 (-6.95%) | -1,031,411 (-6.98%) |
| access.log | 26,214,398 | 2,857,413 | 2,654,961 | -202,452 (-7.09%) | -418,057 (-13.60%) | -409,579 (-13.37%) |
| logs.txt | 221,249,619 | 12,386,135 | 10,239,512 | -2,146,623 (-17.33%) | -2,558,004 (-19.99%) | -2,536,143 (-19.85%) |
| logs.txt-card26m | 26,214,398 | 1,473,523 | 1,216,322 | -257,201 (-17.45%) | -306,512 (-20.13%) | -303,940 (-19.99%) |
| data.json | 14,215,394 | 1,736,197 | 1,599,316 | -136,881 (-7.88%) | -191,406 (-10.69%) | -186,698 (-10.45%) |
| nci | 33,553,445 | 3,431,861 | 3,204,179 | -227,682 (-6.63%) | -513,250 (-13.81%) | -509,518 (-13.72%) |
| xml | 5,345,280 | 747,620 | 705,492 | -42,128 (-5.63%) | -101,181 (-12.54%) | -104,262 (-12.88%) |
| global mix: sil40 | 40,000,000 | 15,852,476 | 15,674,093 | -178,383 (-1.13%) | -496,452 (-3.07%) | -488,757 (-3.02%) |
| webster | 41,458,703 | 12,601,951 | 12,344,264 | -257,687 (-2.04%) | -670,412 (-5.15%) | -644,505 (-4.96%) |
| mozilla | 51,220,480 | 19,145,403 | 18,883,030 | -262,373 (-1.37%) | -621,754 (-3.19%) | -633,699 (-3.25%) |
| dickens | 10,192,446 | 3,972,887 | 3,935,501 | -37,386 (-0.94%) | -161,166 (-3.93%) | -156,217 (-3.82%) |
| tool.bin (W2 band) | 62,480,352 | 21,400,590 | 20,957,814 | -442,776 (-2.07%) | -725,267 (-3.34%) | -771,934 (-3.55%) |
| aozora.txt (W2 band) | 12,003,648 | 4,197,277 | 4,118,650 | -78,627 (-1.87%) | -240,844 (-5.52%) | -236,038 (-5.42%) |
| movie.mp4 | 12,942,257 | 12,891,391 | 12,890,594 | -797 (-0.006%) | -20,221 (-0.157%) | -14,400 (-0.112%) |
| photo.jpg | 6,511,067 | 6,473,476 | 6,473,437 | -39 (-0.0006%) | -16,568 (-0.255%) | -7,835 (-0.121%) |
| x-ray | 8,474,240 | 6,058,381 | 6,056,817 | -1,564 (-0.026%) | -36,434 (-0.598%) | -27,123 (-0.446%) |
| weights.safetensors | 269,060,552 | 213,295,620 | 213,295,197 | -423 (-0.0002%) | -460,517 (-0.215%) | +23,117 (+0.0108%) |

Remaining members (all NON-regressions vs ship, all at-or-below both rivals
except weights/gzip-4 above): mr -4,468 (-0.122%), osdb -105,203 (-2.78%),
sao -23,240 (-0.43%), ooffice -31,176 (-1.00%), samba -90,339 (-1.63%),
reymont -43,796 (-2.20%), minjs.min.js -24,669 (-2.18%), markup.xml
-56,158 (-2.48%), monorepo.tar -272,467 (-2.64%), armexe.elf -6,794
(-1.16%), winexe.exe -26,132 (-1.68%), text-1MB -3,620 (-0.87%),
ecoli.fastq -4,067 (-0.089%).

The band question, answered on all 31 members:

- **Regression watch (±0.1% band vs the worktree's own shipped binary):**
  ZERO members grew. The smallest moves are weights.safetensors (-423 B),
  photo.jpg (-39 B), movie.mp4 (-797 B) — all NEGATIVE.
- **Cell closure:** `data.sqlite:L4:T1` goes from +0.13/+0.17% ABOVE gzip/pigz
  to 7.0% BELOW both — a 1.03 MB absolute margin, closed ~50x over.
- The only member where medium still trails a rival is
  `weights.safetensors` vs gzip -4 at +0.0108% — inside the ±0.1% tie band,
  and the same verdict class the shipped L4 already had there (+0.0111%).
  No pass→fail verdict class flips vs any of the three rivals on any of the
  31 members: the libdeflate-L4 exact-tie members (ld4 == ship bytes) all
  move DOWN (tie → win), never up.
- Aggregate over all 31 members: ship 403,323,971 → medium 397,253,380
  (-6,070,591 B, -1.51%); vs gzip -4 406,927,369 (-2.38%); vs pigz -4
  407,457,229 (-2.50%).

## 4. Mechanism — the size delta is signed against a MEASURED deed rate

`GZIPPY_LDX_MEDIUM_STATS=1` (probe instrument in the rung, stderr, one line
per encode — the "count it; never infer it" rule):

| member | in MB | probes | probe finds | pretest passes | slide steps | commit1 | commit0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| data.sqlite | 48.3 | 9,217,281 | 6,143,176 | 3,182,564 | 9,534,393 | 1,076,688 | 3,358 |
| logs.txt | 221.2 | 7,779,475 | 4,498,778 | 1,703,010 | 69,861,258 | 991,289 | 96,561 |
| webster | 41.5 | 7,298,466 | 4,373,692 | 365,812 | 1,921,492 | 120,302 | 52,573 |
| dickens | 10.2 | 2,126,954 | 1,565,734 | 116,571 | 199,949 | 27,101 | 6,514 |
| data.json | 14.2 | 1,078,378 | 516,246 | 132,699 | 2,873,533 | 46,012 | 27,257 |
| aozora.txt | 12.0 | 2,273,867 | 1,560,417 | 389,666 | 718,750 | 55,880 | 14,098 |
| tool.bin | 62.5 | 14,941,645 | 6,861,924 | 1,135,362 | 4,666,134 | 444,618 | 62,695 |
| photo.jpg | 6.5 | 6,426,163 | 85,992 | 119 | 469 | 36 | 4 |
| weights.safetensors | 269.1 | 212,568,674 | 64,545,481 | 73,391 | 9,053 | 1,182 | 11 |

- **Causal signature matches the size column:** members with a large size win
  have a high committed-slide rate (data.sqlite: 1.08M commits = 11.7% of its
  probes; logs.txt: 1.09M commits with ~64 slide-steps per commit — long
  same-distance repeat runs absorbed into single matches); members with
  near-zero movement are commit-free (photo.jpg: 40 commits, -39 B;
  weights.safetensors: 1,193 commits over 269 MB, -423 B).
- Wall risk shape follows the SAME counts: the slide only runs on
  pretest-passed probes, so the added work is 2 byte-loads per probe plus
  O(k) bytes compared per slide attempt — measured max
  69.9M pair-loads on logs.txt (the churn member; each run still finishes in
  hundreds of ms at -p1 — the wall table below).
- c0-commits (the deviation the memo's sketch excluded; my port found it in
  the vendor) are rare everywhere (0.03-21% of commits) but nonzero — the
  commit-rule boundary matters to honest vendoring, not to any measured cell.

## 4b. The throttle boundary — T>1 receipts (the -p2/-p4 flip feature trap)

The probe is throttled to L4 **at T1**, and the T>1 inertness is proven, not
assumed — with the wave-9 A/A-trap caveat documented in `level.rs:419-428`
(the ladder-tune feature itself replaces the whole T>1 route with
`params_inner(11) + ladder_tune.apply`, so ANY ladder-tune build already
differs from a shipped binary at -p2/-p4; clean main `--features ladder-tune`
shows the same remix):

```
dickens -4 -p4:  vanilla 3,850,283  vs  MAIN-tune(no features diff) 3,683,397
                 ^ the pre-existing ladder-tune T>1 route remix (clean main @
                   30252fba --features ladder-tune reproduces char-12 divergence)
branch-tune(-p4 no env)   == main-tune(-p4 no env)     — byte-identical: the branch
                                                          adds ZERO T>1 delta
branch-tune(-p4 env)      == branch-tune(-p4 no env)  — byte-identical: GZIPPY_LDX_MEDIUM
                                                          is INERT at T>1 (the parallel
                                                          pipeline never constructs an
                                                          LdxCompressor)
same three controls at
-p2 (dickens):                              identical outcomes
                 (vanilla 3,850,283; main-tune 3,683,397; branch-tune env==noenv
                  3,683,397)
```

So the T>1 bytes are owned by the feature's known remix, not by this probe;
shifted binaries never touch the shipped -p4 route (vanilla suite = 39/39
suites incl. the `thread_byte_parity`/`t4_pins` gates). An eventual LEVER to
arm medium at T>1 would be its own chunked-route decision — out of scope
here (memo §3.1).

## 5. Wall — interleaved medians at -L4 -p1 (HYPOTHESIS-TIER)

THREE interleaved arms (the v3 design in `scratch/wall3.py`; raw per-rep
samples + load stamps in `scratch/wall.log`):

- `ship`   : vanilla binary (shipped bytes, shipped codegen)
- `tune`   : ladder-tune binary, env unset — SAME bytes as ship (the A/A above),
             different codegen (the cfg'd software-pipelined chain walk)
- `medium` : ladder-tune binary + `GZIPPY_LDX_MEDIUM=1`

The mechanism's honest wall cost is the SAME-BINARY pair `medium/tune` (the
instrument's codegen premium cancels); `medium/ship` is what an arming would
inherit only if it ALSO adopted the pipelined walk (it would not), so it is
the pessimistic upper bound. Battery ran with load1 3.5-4.0 (below the
wave-census's co-runner storm band but not idle; read every number as
hypothesis-tier per charter, sub-5% = local tie).

| member | ship ms | tune ms | medium ms | MECHANISM (medium/tune) | full (medium/ship) | instrument (tune/ship) |
|---|---:|---:|---:|---:|---:|---:|
| data.sqlite | 229.3 | 242.5 | 273.4 | **+12.8%** | +19.3% | +5.8% |
| logs.txt | 239.2 | 231.4 | 298.9 | +29.2% | +24.9% | -3.3% |
| logs.txt-card26m | 33.8 | 32.8 | 41.1 | +25.6% | +21.8% | -3.0% |
| access.log (campaign) | 60.8 | 61.0 | 69.1 | +13.4% | +13.7% | +0.3% |
| data.json | 36.4 | 37.0 | 42.2 | +14.3% | +16.1% | +1.6% |
| tool.bin | 321.2 | 338.8 | 377.4 | +11.4% | +17.5% | +5.5% |
| aozora.txt | 74.7 | 78.1 | 89.6 | +14.8% | +20.0% | +4.5% |
| minjs.min.js | 21.6 | 22.4 | 24.6 | +9.5% | +13.8% | +3.9% |
| xml | 18.3 | 18.4 | 20.9 | +14.0% | +14.3% | +0.3% |
| nci | 78.5 | 79.2 | 94.3 | +19.0% | +20.0% | +0.9% |
| mozilla | 265.9 | 282.6 | 315.4 | +11.6% | +18.6% | +6.3% |
| dickens | 71.4 | 75.1 | 81.8 | +8.9% | +14.6% | +5.2% |
| mr | 54.6 | 57.1 | 64.1 | +12.3% | +17.4% | +4.6% |
| osdb | 53.9 | 56.2 | 61.2 | +8.9% | +13.5% | +4.2% |
| ooffice | 48.6 | 51.6 | 56.2 | +9.0% | +15.6% | +6.1% |
| reymont | 37.9 | 40.2 | 44.7 | +11.4% | +17.9% | +5.9% |
| samba | 86.9 | 90.9 | 101.4 | +11.6% | +16.7% | +4.6% |
| sao | 66.9 | 71.5 | 79.3 | +11.1% | +18.5% | +6.8% |
| software.archive | 236.0 | 240.7 | 246.7 | +2.5% | +4.5% | +2.0% |
| webster | 214.3 | 224.2 | 248.4 | +10.8% | +15.9% | +4.6% |
| x-ray | 65.3 | 69.5 | 75.4 | +8.5% | +15.6% | +6.6% |
| weights.safetensors | 2356.8 | 2560.3 | 2760.1 | +7.8% | +17.1% | +8.6% |

(Every row above is a wall.log entry: raw per-rep samples for all three
arms + per-rep load1 stamps sit in `scratch/wall.log` [v3 pass]; access.log
was appended in its own quiet-window pass, same protocol.)

Reading:

- The mechanism's wall cost at -p1 lands +2.5%…+29.2% (log class worst at
  +29.2% because the slide work is proportional to the deed table's
  slide_steps: 69.9M chained byte-pair loads on logs.txt), near +8-15% on
  most members, ~0% where the parse is tie-dense or literal-only
  (software.archive +2.5%, weights +7.8% on 269 MB where fizzle almost never
  commits).
- **The memo's +0-3% predicted envelope is falsified** at these coordinates:
  the cost model ("same search count; marginal = fizzle's 2-byte pretest plus
  O(k) compares") underpriced the slide chain's cache behavior — the
  backward byte-pair walks (in[src0-j] vs in[q-j]) chase a window that
  mispredicts and misses; at logs.txt that is ~140M extra loads on a
  9.6-tokens/1-KB-class parse.
- BUT the per-label bar prices ours vs THE RIVALS: with the rung ON,
  these members remain FASTER than both rivals at -p1 AND smaller:

| member | ship (ours) | medium (ours) | gzip -4 | pigz -4 -p1 |
|---|---:|---:|---:|---:|
| data.sqlite | 229 ms | 273 ms | 480 ms | 450 ms |
| logs.txt | 231 ms | 299 ms | 677 ms | 527 ms |
| tool.bin | 321 ms | 377 ms | 740 ms | 668 ms |

  (single quiet-window runs per rival, round-robin x3, medians; pigz pinned
  to -p1 — its default is all-core and the first measurement 137/90 ms was
  the multicore row.)
- The W2 thin-band question ("wall within noise"): NOT within noise vs our
  own binary (+8-15% mechanism cost every member), and the W2 trap rows
  themselves (libdeflate:*:L8/L9:T1:wall) sit at coordinates this rung never
  touches (T>1 route inert per 4b; L8/L9 routes untouched by construction —
  the env proves byte-inert there). An arming decision is a wall-led
  adjudication the frozen box owns; these rows say the price is REAL, not a
  tie.

## 7. Verdict — what this probe prices, and the recommendation

SIZE (deterministic, all 31 members, both arms round-tripped):

- The cell CLOSES: `data.sqlite:L4:T1` moves from +0.13/+0.17% to
  -6.98%/-6.95% vs gzip/pigz -4 — 1.03 MB absolute margin, closed over 50x.
- Zero size regressions anywhere (the smallest moves: weights -423 B,
  photo.jpg -39 B, movie.mp4 -797 B); no tie flips to a loss; the
  libdeflate-L4 exact ties all move DOWN (tie -> win).
- Aggregate -6.07 MB over the 31 members (-1.51%), -2.38% vs gzip -4
  aggregate, -2.50% vs pigz -4 aggregate.
- Ladder shape (memo D3): medium-L4 beats our own L5 on 12 members (log
  class -5..-17%) and our own L9 on 1 (weights, pre-existing inversion);
  the synthetic lattice ties modestly. An arming needs the conscious
  one-level ladder-shape parseattention the memo's D3 planned.

WALL (hypothesis-tier, same-binary mechanism pair):

- NOT within noise: +2.5%..+29.2% T1 wall on the mechanism pair, mean
  ~+11-12%. The memo's +0-3% envelope is falsified (the slide's backward
  byte-pair walk is the price, and it scales with the deed table).
- Against the SAME-LEVEL rivals, all measured cells STAY won on BOTH axes
  with the rung on: ours-medium is 1.45-2.0x the wall of gzip -4's
  advantage... (i.e., ours remains faster than BOTH rivals at each measured
  member).

RECOMMENDATION for the lever PR (this probe does not ship anything):

1. **SHARD into two PRs.** (a) the rung's port + tests + probe hook as an
   INERT lever base; (b) the arming table (L4 -> medium at T1) as its own
   lever judged by the frozen-box `fulcrum try --levels 4 --threads 1,4`
   clause machinery — the T1 wall premium measured here (+8-15% most
   members, +29% logs) is the clause-7 live question, and the -0.95..-7.1%
   per-member size column is the clause-5 capital it spends.
2. The -p1 SIZE table above is the frozen-box sweep's expectation: 31/31
   non-regressing, 3 hairline members essentially unchanged (weights/photo/
   movie |< -0.01%|).
3. The value is on the LOG/sqlite class. If the owner prices the class under
   clause 5, the probe's per-member row set gives the exact cell map the box
   will sweep (sizes.tsv in scratch/).
4. Follow-ups when (and only when) arming: `ir_budget`/`ir_vs_ldx`/
   `anatomy_pins` L4 rows (the medium parse changes the Ir profile);
   `parity-census.sh` after any parse-layer merge; a deed-counter check per
   member on the box before the wall battery (the pricing record's table is
   the baseline).

Zero shipped parameters changed; the probe's binary identity, the A/A
control, and every stream's independent-decoder round-trip are recorded in
`scratch/` and quoted above verbatim.
