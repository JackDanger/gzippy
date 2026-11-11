# L5 depth-32 ladder rung — falsifier receipt for the board-worst cell `access.log:L5:T1`

Run date: **2026-11-11** (wave-12 pricing run, `probe/l5-depth32`, worktree
`.review-tmp/wA-l5/`, parent `main` @ `7e0d08cdaf384f5d429b7930633d6e9b26237524`).
Prober/builder: wave-12 pricing agent (laptop, M1 aarch64, load 2.4-4.6 during
runs).

## 0. The card and the mechanism

- Card (frozen board, plan `plan-2026-09-one-encoder.md` §2b): the 1,320-cell
  census peaked at `access.log L5 vs gzip -5 +1.07%` / `pigz -5 +0.75%`.
  Wave-12 card M1's registered falsifier was a zero-code LADDER sweep on the
  lazy channel.
- Shipped L5-T1 route: the ldx port's own level table, `Lazy(16,30)` with
  good_match=0 — `src/compress/ldx/compress.rs` (`5 => (16, 30, 0)`, the
  "UNCHANGED libdeflate config" row; libdeflate routes L5 lazy at
  `deflate_compress.c:3946-3948`). gzip -5's L5 is `deflate_slow` with
  good=8 / nice=32 / chain=32 (zlib-ng `deflate.c:120`), i.e. chains to a
  cap of 32 with good_match quartering at `>>2`.
- The question priced here: **can a depth bump 16→32 (+ gzip's good_match
  quartering as the "good8-equivalent" arm) recover the access.log-class
  deficit, and what does it cost on knife-edge members?**

## 1. THE GRAMMAR LIMITATION (this is the receipt's first-class finding)

**`GZIPPY_LADDER` cannot express an L5 depth bump on the shipped route.**
Measured, not inferred: on the `--features ladder-tune` build,
`GZIPPY_LADDER=lazy:32:30` changed the bytes of **0/15** members at L5 `-p1`,
while on the SAME binary the ldx-level triple hook
`GZIPPY_LDX_CFG=5:32:30` changed **15/15** members (reproducible envs in
`scripts/sweep.py`; raw shas in `sizes.tsv`). Cause, code-read: `GZIPPY_LADDER`
lives in `src/compress/deflate/level.rs`'s `ladder_tune::apply` and only
overrides `params_baseline`/`params_parallel` — the LEGACY parser's params
object; since the `L5` exceptions retired, L5-T1 routes to
`LdxCompressor::new` (`level_uses_ldx` = everything except L1, 2..=9), whose
own table reads its own env hook `GZIPPY_LDX_CFG=<level>:<depth>:<nice>[:<good>]`
(`ldx/compress.rs`, also `ladder-tune`-gated, default-off, read once at
compressor construction — one encode per process at `-p1`). The card M1
falsifier as registered (`GZIPPY_LADDER=lazy:32:30`) is a DEAD LEGAL WRITE —
it measures nothing because it reaches nothing at this route. The zero-code
sweep that DOES bisect the shipped route is `GZIPPY_LDX_CFG`.

Both `/LADDER` (strategy depth nice [passes]) and `/LDX_CFG` (per-level
triple with optional good) have one more grammar limitation either way:
**neither supports thread scoping** — there is no `-p` field — so this rung
was swept at `-p1` only and is labeled T1. `-p4` behaviour was not measured
(the card is T1; the rung's T>1 cost is unpriced and would need its own leg).

## 2. Controls FIRST — the A/A sha verification

- Control arm: ladder-tune build + `GZIPPY_LDX_CFG=5:16:30`
  (depth 16, nice 30 — the shipped triple verbatim).
- Result: **byte-identical (sha256 equality) to the vanilla shipped binary
  (`--target-dir target-ship`, no features, no env) on 15/15 members.**
  This is the route assertion as well as the A/A proof: the -p1 L5 stream IS
  the port's, under the `ladder-tune` build, and the env hook reaches exactly
  the shipped encode with the control knob returning shipped bytes.
- Perturbation proof the env is LIVE: `5:32:30` differs from shipped on
  15/15 (and `ladG32` differs on 0/15 on the same binary). A control that
  "passes" because it never reached the encoder is the documented vacuous-A/A
  trap — the first pass of `scripts/sweep.py` applied NO env (its
  all-identical verdicts, 15/15 "identical" on every arm including the
  depth-32 arm, were only caught by a manual sensitivity check after the
  fact); the script was fixed and the sweep re-run, and the first pass's
  outputs are VOID, not banked. The archived `sweep.log`/`sizes.tsv` are the
  RE-RUN (live-env) artifacts.
- Validity: every ladder-arm stream roundtrips through an independent decoder
  (`/usr/bin/zcat`, stdin — not a gzippy shim) back to the exact member bytes:
  15/15 ok for every non-rival arm (the sweep aborts on the first
  roundtrip failure; none occurred). VALID GZIP bar holds on the probe.
- Binary provenance: `git_rev=7e0d08cdaf384f5d429b7930633d6e9b26237524`
  (sha `f5afff2f…b18527d9` vanilla; `bd1a65d1…29efe0e4` ladder-tune;
  full stamp in the `[stamp]` line of `sweep.log`). The
  `+INSTRUMENTED(ladder-tune)` suffix appears in `gzippy --version` on the
  tune binary only.

## 3. The rungs swept (all at `-5 -p1`, zero-code, env-only)

| arm | env (ladder-tune build) | mechanism |
|---|---|---|
| ship | none (vanilla binary) | shipped Lazy(16,30), good 0 |
| lad16 | `GZIPPY_LDX_CFG=5:16:30` | control |
| lad20 | `GZIPPY_LDX_CFG=5:20:30` | depth rung |
| lad24 | `GZIPPY_LDX_CFG=5:24:30` | depth rung |
| lad32 | `GZIPPY_LDX_CFG=5:32:30` | gzip -5 chain cap |
| lad32g8 | `GZIPPY_LDX_CFG=5:32:30:8` | -5 chain cap + good_match=8 (`>> 2` quartering, `hc_matchfinder_longest_match:221`) |
| ladG32 | `GZIPPY_LADDER=lazy:32:30` | the card's registered falsifier grammar (inert) |

Rivals (built locally from the worktree's submodule pins, `-p1` where the
CLI has threads; sizes only): `vendor/gzip` @ `94cfaabe` (distrotech-gzip),
`vendor/pigz` @ `fe4894f` (v2.8), `vendor/libdeflate` @ `22f6e02` (v1.25+).

**Corpus caveat (member substitution):** `access.log` does NOT exist on this
box. The card's file class is represented by `benchmark_data/logs.txt`
(221,249,619 B) and a card-scale slice `logs.txt-card26m` (exactly
**26,214,398 B**, the canonical access.log input size recorded in
`docs/board/records/z119-graded-len3/sizes.tsv`), both cut from logs.txt.
Both are LOG-CLASS stand-ins, not the card's file. The other 12 members are
`silesia.tar`'s members; `software.archive` comes from `benchmark_data/`.

## 4. Size table (deterministic; deltas vs shipped L5-T1)

Raw per-arm sizes + sha256: `sizes.tsv` (one row per member x arm).

Gzippy shipped `ship` = `Lazy(16,30)`; delta columns are `arm - ship` bytes
and % of ship. Raw absolute sizes:

| member | ship | lad32 | lad32g8 | gzip -5 | pigz -5 | libdeflate -5 |
|---|---:|---:|---:|---:|---:|---:|
| dickens | 3,894,073 | 3,862,146 | 3,862,754 | 3,946,208 | 3,950,777 | 3,894,044 |
| logs.txt | 12,092,008 | 12,092,948 | 12,092,611 | 12,422,400 | 12,437,650 | 12,092,008 |
| logs.txt-card26m | 1,437,521 | 1,437,562 | 1,437,555 | 1,478,079 | 1,479,676 | 1,437,521 |
| mozilla | 18,860,320 | 18,808,878 | 18,816,562 | 19,161,130 | 19,199,408 | 18,878,893 |
| mr | 3,664,107 | 3,648,501 | 3,648,436 | 3,714,278 | 3,701,213 | 3,664,263 |
| nci | 3,268,206 | 3,216,211 | 3,243,224 | 3,378,337 | 3,381,836 | 3,268,206 |
| ooffice | 3,074,465 | 3,067,771 | 3,068,215 | 3,115,523 | 3,119,121 | 3,079,295 |
| osdb | 3,664,358 | 3,659,176 | 3,659,217 | 3,797,209 | 3,758,928 | 3,666,494 |
| reymont | 1,921,129 | 1,882,442 | 1,885,283 | 1,929,516 | 1,934,756 | 1,921,131 |
| samba | 5,414,888 | 5,387,399 | 5,393,285 | 5,519,785 | 5,509,430 | 5,416,731 |
| sao | 5,350,747 | 5,334,316 | 5,334,291 | 5,370,681 | 5,370,527 | 5,352,230 |
| software.archive | 5,182,578 | 5,178,223 | 5,178,223 | 5,139,189 | 5,166,773 | 5,182,578 |
| webster | 12,311,615 | 12,194,704 | 12,218,050 | 12,433,635 | 12,458,830 | 12,310,794 |
| x-ray | 5,980,662 | 5,980,558 | 5,980,558 | 6,046,340 | 6,055,866 | 5,998,418 |
| xml | 718,574 | 703,490 | 709,182 | 731,266 | 729,839 | 718,633 |

Deltas vs `ship` (lad20 / lad24 / lad32 / lad32g8), aggregate
lad20 −146,558 B, lad24 −253,993 B, lad32 −380,926 B, lad32g8 −307,805 B:

| member | lad20 | lad24 | lad32 | lad32g8 | gzip -5 vs ship | pigz -5 vs ship | libdeflate -5 vs ship |
|---|---:|---:|---:|---:|---:|---:|---:|
| dickens | −12,712 (−0.326%) | −20,938 (−0.538%) | −31,927 (−0.820%) | −31,319 (−0.804%) | +52,135 (+1.339%) | +56,704 (+1.456%) | −29 (−0.001%) |
| logs.txt | +455 (+0.004%) | +843 (+0.007%) | +940 (+0.008%) | +603 (+0.005%) | +330,392 (+2.732%) | +345,642 (+2.858%) | 0 (+0.000%) |
| logs.txt-card26m | +40 (+0.003%) | +41 (+0.003%) | +41 (+0.003%) | +34 (+0.002%) | +40,558 (+2.821%) | +42,155 (+2.933%) | 0 (+0.000%) |
| mozilla | −18,830 (−0.100%) | −33,465 (−0.177%) | −51,442 (−0.273%) | −43,758 (−0.232%) | +300,810 (+1.595%) | +339,088 (+1.798%) | +18,573 (+0.099%) |
| mr | −6,814 (−0.186%) | −11,204 (−0.306%) | −15,606 (−0.426%) | −15,671 (−0.428%) | +50,171 (+1.369%) | +37,106 (+1.013%) | +156 (+0.004%) |
| nci | −22,723 (−0.695%) | −35,777 (−1.095%) | −51,995 (−1.591%) | −24,982 (−0.764%) | +110,131 (+3.370%) | +113,630 (+3.477%) | 0 (+0.000%) |
| ooffice | −2,567 (−0.084%) | −4,602 (−0.150%) | −6,694 (−0.218%) | −6,250 (−0.203%) | +41,058 (+1.336%) | +44,656 (+1.453%) | +4,830 (+0.157%) |
| osdb | −2,562 (−0.070%) | −3,963 (−0.108%) | −5,182 (−0.141%) | −5,141 (−0.140%) | +132,851 (+3.626%) | +94,570 (+2.581%) | +2,136 (+0.058%) |
| reymont | −12,588 (−0.655%) | −22,692 (−1.181%) | −38,687 (−2.014%) | −35,846 (−1.866%) | +8,387 (+0.437%) | +13,627 (+0.709%) | +2 (+0.000%) |
| samba | −10,072 (−0.186%) | −19,266 (−0.356%) | −27,489 (−0.508%) | −21,603 (−0.399%) | +104,897 (+1.937%) | +94,542 (+1.746%) | +1,843 (+0.034%) |
| sao | −6,297 (−0.118%) | −11,046 (−0.206%) | −16,431 (−0.307%) | −16,456 (−0.308%) | +19,934 (+0.373%) | +19,780 (+0.370%) | +1,483 (+0.028%) |
| software.archive | −2,342 (−0.045%) | −7,231 (−0.140%) | −4,355 (−0.084%) | −4,355 (−0.084%) | −43,389 (−0.837%) | −15,805 (−0.305%) | 0 (+0.000%) |
| webster | −43,536 (−0.354%) | −74,799 (−0.608%) | −116,911 (−0.950%) | −93,565 (−0.760%) | +122,020 (+0.991%) | +147,215 (+1.196%) | −821 (−0.007%) |
| x-ray | −108 (−0.002%) | −109 (−0.002%) | −104 (−0.002%) | −104 (−0.002%) | +65,678 (+1.098%) | +75,204 (+1.258%) | +17,756 (+0.297%) |
| xml | −5,902 (−0.821%) | −9,785 (−1.362%) | −15,084 (−2.099%) | −9,392 (−1.307%) | +12,692 (+1.766%) | +11,265 (+1.568%) | +59 (+0.008%) |

Read-through:

- Depth is a large size WIN on 13/15 members (xml −2.10%, reymont −2.01%,
  nci −1.59%, webster −0.95%, dickens −0.82%) and slightly BIGGER on the two
  log-class members (+0.003-0.008%, all inside the ±0.1% tie band = local
  size ties on log class).
- `lad24` vs `lad32`: lad32 ≤ lad24 on 11/15 members; only mr/x-ray and the
  log-class members move ≤ ~40 B between the 24 and 32 rungs (card26m
  saturates at 24 already — its 1,437,562 B is byte-equal in size at both
  rungs). The good_match=8 gate (`lad32g8`) trades size back for wall:
  vs plain lad32 it gives up nci +27,013 B / mozilla +7,684 B / dickens
  +608 B while landing slightly SMALLER on card26m (−7 B) and mr (−65 B).
- **On the card's class the rung moves the WRONG way.** `logs.txt-card26m`
  (the access.log-family stand-in): ship 1,437,521 → lad32 1,437,562
  (**+41 B, +0.003%**) → lad32g8 +34 B. gzip -5 is +2.82% BIGGER than our
  shipped L5 there, i.e. on our own log-class stand-ins the shipped L5
  ALREADY beats both rivals by 2.7-2.9%, and hiding deeper chains into the
  same log text makes us marginally BIGGER, not smaller. The
  libdeflate -5 exact-tie members (`logs.txt`, `logs.txt-card26m`,
  `software.archive`, `nci`: 0 delta at ship) break exact byte identity with
  libdeflate at the deeper rung but stay on the small side everywhere —
  no pass→fail size flip on any of the 15 local members.
- Mechanism note (matches the D-line in `hh-hairline-cells.md`): chains
  longer than 16 DO exist here (15/15 members change bytes), but on this
  text the added finds displace better shorter matches
  (deeper is NOT signed-smaller). The record from `engine.wasm L8` and the
  L2 x4-depth record had already warned depth is not monotone; this rung is
  the L5 instantiation of that same non-monotonicity.

## 5. Wall battery (interleaved medians, alternating arms — HYPOTHESIS-TIER)

Laptop-frozen-box caveat per charter: **every number in this section is
hypothesis-tier**; a sub-5% same-box wall delta is a LOCAL TIE, and the
frozen box adjudicates. 7-9 reps per pair, alternating arm order per rep
(`wall.log` retains raw per-rep timings; blocks 2/3 = same-binary env-only
pairs, its honest instrument; also the reason: block 1's cross-binary
calibration `ship` vs `l16` at byte-identical config measured **+8.78%**
(card26m) and **+5.01%** (webster) of pure build/instrument premium, so
cross-binary pairings overstate the depth cost; both calibrations reported
inline below and in `wall.log`).

Same-binary (target-tune binary; env-only lever l16=shipped-equivalent
config):

| member | arm pair | medians (A ms → B ms) | wall delta |
|---|---|---|---:|
| logs.txt-card26m | l16 → lad32 | 36.9 → 38.9 | **+5.27%** |
| logs.txt (221 MB) | l16 → lad32 | 300.7 → 310.5 | **+3.25%** |
| software.archive | l16 → lad32 | 292.5 → 293.4 | **+0.32%** (local tie) |
| webster | l16 → lad32 | 270.5 → 313.4 | **+15.86%** |
| dickens | l16 → lad32 | 92.7 → 111.2 | **+19.90%** |
| mozilla | l16 → lad32 | 324.5 → 364.5 | **+12.31%** |
| xml | l16 → lad32 | 21.0 → 23.4 | **+11.26%** |
| logs.txt-card26m | l16 → lad32g8 | 37.8 → 39.5 | +4.61% |
| webster | l16 → lad32g8 | 278.7 → 309.8 | +11.17% |
| logs.txt-card26m | lad32 → lad32g8 | 39.3 → 39.3 | −0.13% |
| webster | lad32 → lad32g8 | 325.1 → 311.2 | −4.30% |

Reading:
- The depth-32 wall price is **+0.3% (software.archive, local tie) to
  +19.9% at T1** depending on member; 5/7 same-binary pairs are ABOVE the
  5%-local-tie threshold. Note: on the text members that is a wall COST for
  a size win — it convicts on 2026-09-05's terms ("wall never loses; size
  may spend ≤1%/cell") on every text member.
- good_match=8 quartering trades back some of plain-depth-32's size for
  wall (webster: −4.30% wall for +23,346 B vs plain lad32; card26m: −0.13%
  wall for −7 B) — not a rescue, and on the rest of the corpus it gives up
  real size (nci +27,013 B, dickens +608 B) for at most a few % of wall.
- On the card's class, the honest same-box wall line is +3-5%: a local tie
  at the sub-5% threshold on the log members, near-tie enough that the
  frozen box could read it as affordable on the ONLY cells with any
  measured deficit... but the size leg there does not materialize (+0.003%
  the wrong way), so there is nothing to spend the wall on.

## 6. Verdict per charter

Cell under judge: `access.log:L5:T1` vs gzip -5 (+1.07%) / pigz -5 (+0.75%).

1. **The registered falsifier did not fire — it could not.**
   `GZIPPY_LADDER`'s lazy channel is INERT at the shipped L5-T1 route
   (0/15 members byte-changed, with the same build's `GZIPPY_LDX_CFG` hook
   changing 15/15 as the live control). Its grammar (strategy:depth:nice[:nearopt
   passes]) cannot express the L5 depth bump on the route that produces
   the shipped stream; the route-surface probe is `GZIPPY_LDX_CFG`.
2. **On every locally available member the L5 depth-32 rung (with or
   without the gzip-style good_match quartering) does NOT recover a
   log-class deficit.** The only log-class members present move +34 B to
   +940 B (+0.002-0.008% bigger — inside the ±0.1% tie band by 12-50x, but
   the wrong SIGN).
3. **The rung is a large size improvement everywhere else** — aggregate
   −380,926 B over the 15-member local corpus (−2.01% reymont, −2.10% xml,
   −1.59% nci, −0.95% webster, −0.82% dickens; 13/15 members at-or-below
   shipped; zero pass→fail sizing flips vs any local rival) — but it rides a
   **+0.3% to +19.9% same-binary T1 wall cost, 5/7 pairs above the sub-5%
   local tie threshold (every text member sits at 11-20%)**. At the frozen
   box this is a `wall never loses` conviction UNLESS the cells it would
   close are judged on the frozen corpus (the local corpus is neither TUNE
   nor GATE, and access.log is not here at all).
4. The card's own deficit (+1.07% vs gzip) is **not reproducible locally**:
   the local access.log-family stand-ins WIN against gzip -5 by +2.7-2.9%
   at shipped knobs. The frozen-box access.log may behave differently (its
   content — sparse server logs with very long waits between repeats —
   cannot be assumed from these stand-ins). The cheapest un-falsifying leg
   would therefore be re-running THE SAME env sweep at the frozen box on
   the REAL access.log member with `GZIPPY_LDX_CFG=5:32:30` and
   `5:32:30:8` (both zero-code), judged against the T1 whole-buffer path.

**WALL VERDICT TIER: hypothesis-tier throughout (laptop, unfrozen,
co-runner load 2.4-4.6; all wall deltas quoted are medians of interleaved
alternating reps). No claim in this receipt is a wall conviction; per
charter a local tie reads as NO-SHIP and every wall number is
recommendation-tier only.**

## 7. Recommendation: NO-SHIP (as registered; re-open only at the frozen box, on the real member)

- **Do not arm the depth-32 rung for `access.log:L5:T1` on the strength of
  any local number.** The local evidence says (a) the rung does NOT pay on
  log class, (b) it pays ONLY by spending 11-20% T1 wall on every text
  member elsewhere, and the card's +1.07% is not locally reproducible.
  SHIP would be a guess about a file this box does not have.
- The profitable shape of the mechanism is not the log class: it gains
  [silesia-style text/binary] 0.1-2.1% of size at ~2x chain cost, which is
  the opposite of the winner-typed shape the charter prices here.
- If the rung is ever re-derived, the correct zero-code surface is
  `GZIPPY_LDX_CFG=5:32:30[:8]` (NOT `GZIPPY_LADDER`; fix the card text that
  registered the dead grammar), with `tie-guard.sh` and the 22-file enum
  as the next gates — the exact-tie members here (logs, software.archive,
  nci vs libdeflate at 0.0000%) are the clause-3 cage the arming step must
  clear on the frozen corpus.
- Knife-edge members from the plan's W2 thin-margin band
  (`plan-2026-10-29-sweep.md:131-135`: `libdeflate:*:L8/L9:T1:wall`
  thin-margin rejects on aozora, data.json, dickens, symbols.dwarf,
  tool.bin, Δ up to +1.0%): only `dickens` of those five exists locally,
  and at the L5 coordinate it reads the WORST wall cost of the sweep
  (+19.9% for −0.82% size) — exactly the reject-shape that band convicts.
  The other four W2 members are not on this box; an arming step must price
  them on the frozen corpus before any whole-level rung is considered.
- Zero shipped parameters changed for this receipt: probe-only (`--features
  ladder-tune` which is default-off), own worktree, own target dirs,
  root checkout untouched. No `GZIPPY_*` env reaches production builds.

## 8. Provenance appendix

- Branch/worktree: `probe/l5-depth32` @ `.review-tmp/wA-l5/`,
  `git rev 7e0d08cdaf384f5d429b7930633d6e9b26237524` (= parent `main`).
- Binaries: `target-ship/release/gzippy` sha
  `f5afff2f2c91ac5c8ff8f3df69689545f26c42cd42dbeccd18a1db4bf18527d9`
  (vanilla, no features); `target-tune/release/gzippy` sha
  `bd1a65d15a1dc84f84b4bd21c80075cd164d48dfbcb51ca653a9774a29efe0e4`
  (`+INSTRUMENTED(ladder-tune)`).
- Builds: `cargo build --release [--features ladder-tune] --target-dir
  <worktree>/target-{ship,tune}` — the explicit per-arm target dir is
  REQUIRED because the user-global `~/.cargo/config.toml` pins one shared
  target-dir; two arms built into it would collide into one binary and lie.
- Rivals: `vendor/gzip` @ 94cfaabe (distrotech-gzip), `vendor/pigz` @
  fe4894f (v2.8), `vendor/libdeflate` @ 22f6e02 (v1.25-5). Built fresh in
  the worktree (`./configure && make`, `make`, `cmake -B cbuild ... &&
  cmake --build --target libdeflate-gzip`).
- Corpus: 12 silesia members (+`software.archive`, `logs.txt`) staged to
  `scratch/members/`, plus `logs.txt-card26m` = first 26,214,398 B of
  logs.txt. Sha256 of each member available alongside arms in `sizes.tsv`.
- Commands: `python3 scripts/sweep.py` (sizes + sha + roundtrip + rival
  rows), `python3 scripts/pivot.py` (table), `python3 scripts/wall.py`,
  `python3 scripts/wall3.py` (interleaved wall, raw reps in `wall.log`).
- `-p4` was NOT swept: neither env grammar carries a thread field, so no
  thread-scoped rung exists; the cell is judged at T1 only (card is T1).
