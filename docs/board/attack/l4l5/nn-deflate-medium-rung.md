# The L4 rung — zlib-ng's `deflate_medium` family inside the port

Attack memo, 2026-09-05, branch `brain/2026-09-05-attack`. Vendor sources read at
`vendor/zlib-ng/deflate_medium.c` (all `:NNN` line cites below are that file unless
noted). The "frozen lattice" failing numbers and the "won-with-margin census tier"
come from the parallel planning session's cross-agent handoff — flagged
`(parent-supplied)` wherever they are not reproducible from this checkout yet.
Everything else was read in this worktree today; every file cite is real.

---

## 0. TL;DR

| field | value |
|---|---|
| The defect | After the one-encoder pivot (pick-min deleted, one encode per input), L4 = port greedy `(16,30)` produces **larger** output than L3 = legacy lazy `(12,14)` + far-len-3 cost gate on the synthetic lattice: `tabular` 271,505 > 262,033 (**+9,472 B**), `binary` 663,583 > 661,353 (**+2,230 B**) — re-opening `KNOWN_SAGS` pairs `("tabular",3)`, `("binary",3)` (`tests/size_invariants.rs:39-84`; the rows were healed by mmap pick-min `#330/#331` at this commit and the pick-min deletion re-opens them per plan §2). (`parent-supplied` lattice numbers) |
| The lever | Port zlib-ng's `deflate_medium` **family** as an L4-only rung inside `ldx` — plan doc `docs/plan-2026-09-one-encoder.md` §5 names it explicitly ("vendor-precedented candidate … an L4 rung inside the port"). |
| The discovered economics | In **our** port this rung is nearly free: because `hc_matchfinder_longest_match` inserts the string at `in_next` while it searches (`hc_matchfinder.rs:225-233`), the medium loop's look-ahead probe is exactly the search greedy would have done at the position after the match — **same search count, same insert count, same table discipline as greedy; the only added work is `fizzle_matches`' byte compares**. Vendor pay-off without lazy's cost. |
| Budget | Legacy-era measurement: full lazy at L4 = **+17.7% wall** (`docs/target-encoder-and-gap-analysis.md` G4). This rung predicts ≈ greedy wall + ε. Census context (parent-supplied): L4 wall rows win with margin in the ratio ≤ 0.80 tier = clause-5 margin-as-capital. T4 slack measured 249–330% (`src/compress/deflate/level.rs:417`). |
| Falsifier | `cargo test --release` (lattice + fingerprints) → `measure_tables` → frozen-box `fulcrum try <ref> --levels 3,4,5 --threads 1,4` → `board-size.sh all`. Full sequence §5, in frozen order. |
| The one thing it must never touch | L3. Its lazy `(12,14)` + far-len-3 state is PR #364's lever and its won cells are ledger rows (`tests/fingerprints/ledger.tsv`); the rung lands after #364 in the merge order (handoff §3.1) and edits only `src/compress/ldx/`. |

---

## 1. The algorithm, line by line (`vendor/zlib-ng/deflate_medium.c`, 265 lines)

`deflate_medium` is one Intel-authored strategy (van de Ven, `:3-6`) that is **neither
greedy nor lazy**: it emits the match at the current position — but it keeps the
match at the *position after the current match* warmed in a `next_match` cache, and
it owns a bounded boundary-fixing rule between the two.

### 1.1 The data structure

`struct match { match_start, match_length, strstart, orgstart }` — 4× u16 (`:16-21`).
Two instances live for the whole stream: `current_match` (aligned, cacheline
comments `:179-181`) and `next_match`. `orgstart` records where the match's own
insert sweep had already inserted — the dedup anchor.

### 1.2 `emit_match` (`:23-44`)

Matches shorter than `WANT_MIN_MATCH = 4` (`vendor/zlib-ng/zutil.h:59-60`) are
**emitted as literals** — zlib-ng's medium never takes a len-3 match. Longer ones
tally `dist = strstart − match_start`, `len − 3`.

### 1.3 `insert_match` (`:47-91`) — medium's insert discipline

Inserts hash entries for the match body with two zlib-ng wrinkles:

* **Dedup via `orgstart`** (`:54-64`, `:74-82`): positions already inserted during a
  previous iteration's tail (or that predate the match after a slide) are skipped —
  the branch lattice peeking at `match.orgstart`.
* **The cap** (`:70`): the body is inserted only when
  `match_len <= 16 * s->max_insert_length` (max_insert_length = level's `max_lazy`
  = 12 at L4 → cap 192). Longer matches insert **only the last hash position**
  (`:83-90`: `quick_insert_string` at `strstart + 2 − STD_MIN_MATCH`) so the next
  position can still find the match tail.

This is the third member of the `LIMIT_HASH_UPDATE` family documented in
`docs/vendor-technique-index.md` P10 (igzip: +0/+1/+2 only; zlib-ng fast: ≤
`max_insert_length`; zlib-ng medium: ≤ `16 * max_insert`; libdeflate `skip_bytes`:
**everything** — "buying ratio with insert cost").

### 1.4 `find_best_match` (`:93-121`)

One hash-chain probe: if head `hash_head` gives `dist <= MAX_DIST(s)` (= `w_size −
MIN_LOOKAHEAD` = 32768 − 262 = 32506, `deflate.h:380,385`) and `dist > 0` and head
≠ 0 → plain `longest_match` (chain walks governed by the level's `max_chain`,
early exit at `nice_match_length`). A `match_length < WANT_MIN_MATCH` result is
collapsed to **1 (literal)** and a self-restart guard `match_start >= strstart`
dissolves likewise (`:110-113`). No len-3, no offset gate: *everything* under 4
bytes is a literal by `emit_match`.

### 1.5 `fizzle_matches` (`:128-176`) — the actual algorithm `"medium"`

Preconditions at the call site (`:233-237`): `current.match_length > 1`, and
`c.match_length − 1 ≤ next.match_start` and `≤ next.strstart`.

1. **One-byte pre-test** (`:134-139`): the byte *before* the next match's source
   must equal the byte one past `current`'s start-if-slid — `*match == *orig`.
   Mismatch → return, nothing happened (this makes the common case 2 byte loads).
2. **Slide loop** (`:150-167`): move the boundary between the two matches leftward
   one byte at a time — `n.strstart--, n.match_start--, n.match_length++,
   c.match_length--` — while two matched bytes continue to match, with four abort
   guards: `c.match_length < 1` (`:151-153`), `n.strstart <= limit` where
   `limit = strstart − MAX_DIST` window-edge floor (`:145,153-155`),
   `n.match_length >= 256` cap (`:155-156`), `n.match_start <= 1` distance floor
   (`:157-158`).
3. **Commit rule** (`:169-175`): accept *only* `if (changed && c.match_length <= 1
   && n.match_length != 2)` — the entire current match has dissolved to ≤ 1
   literal and the next match grew by the exact same byte count at the **same
   offset** (both `strstart` and `match_start` shifted by `k`, so
   `dist = strstart − match_start` is invariant). Without a commit the slide is
   fully rolled back (`:173-175`).

Note what fizzle is *not*: it is not lazy's promote. There is **no**
"if next is better, emit a literal and defer" branch anywhere in `deflate_medium`
(`:210-244` re-checked twice): zlib-ng's medium never defers a match wholesale and
never promotes a next match to current on a score rule. Its only adaptive act is
the slide — and the slide's *outcome* is exactly the shape libdeflate's
near-optimal frontier produced in the G11 anatomy probe: "moved the second match N
positions LEFT and dissolved the first into literals" (`docs/target-encoder-and-gap-analysis.md:687-697`,
falsified the blanket too-far rule precisely because the winning move is
*contextual* — it depends on a better match existing one position on; the slide
looks at exactly that context, bounded, with no table, no DP).

### 1.6 The main loop (`:178-264`)

```
early_exit = (level < 5)                       # :184-185
loop:
  fill lookahead
  if !early_exit && next.cache is live: current = next; next dead   # :210-214
  else: quick_insert_string(strstart); current = find_best_match   # :214-221
  if lookahead > current.len + 3: insert_match(current)            # interior inserts
  if !early_exit:                       # the look-ahead one         # :227-244
      strstart += current.match_length
      quick_insert_string(strstart)     # probe position gets inserted
      next_swap = find_best_match(...)  # one chain walk, cached
      fizzle_matches(&current, &next)   # optional slide, committed or rolled back
      strstart restore
  emit current; strstart += current.match_length; FLUSH on block windows
```

So per emitted unit the flow is: **search at p (or consume the cache), interior
inserts, one look-ahead walk at p+L whose result is *re-used* as the next
iteration's current** — the search at the position after the match happens once
and once only.

### 1.7 Where it sits in zlib-ng's own ladder (`vendor/zlib-ng/deflate.c:105-130`)

```
      good lazy  nice chain  func
/*3*/ {4,    6,  16,   6} deflate_medium
/*4*/ {4,   12,  32,  24} deflate_medium   /* "lazy matches" comment in C */
/*5*/ {8,   16,  32,  32} deflate_medium
/*6*/ {8,   16, 128, 128} deflate_medium
/*7-9*/                              deflate_slow
```

`early_exit = level < 5` means zlib-ng's own L4 is medium in *greedy shape*
(probe off): the rung exists so L3–L6 use one cheap parser and only L5–6 pay the
probe+cache. The compression win the rung is named for comes from the probes and
fizzle, i.e. the L5 shape.

---

## 2. Mechanism → what the ldx port already has

Port map (all verified in this worktree):

| mechanism | zlib-ng site | in the port today | gap for the rung |
|---|---|---|---|
| greedy loop skeleton | `deflate_medium` loop | `ldx/compress_greedy.rs:69-163` — block loop, `calculate_min_match_len` once per block, `hc_matchfinder_longest_match` + `hc_matchfinder_skip_bytes`, `sequences`/`split`/`flush` helpers | **none** — donate this file as the template for `compress_medium.rs` |
| per-position search that also inserts | `quick_insert_string` + `longest_match` split | `hc_matchfinder_longest_match` fuses them: head/`next_tab` writes at `:225-233`, next-hash computation `:236-239` | **this is why the medium mapping is an exact budget match** — see §4 |
| interior inserts | `insert_match` orgstart dedup + 16×cap | `hc_matchfinder_skip_bytes` (`hc_matchfinder.rs:421-478`) inserts every interior position (libdeflate policy), with the tail early-out (`:432-434`) — **but greedy never re-examines a position, so libdeflate has no double-insert problem; a committed slide does** (the slid next-match absorbs the current match's already-inserted body). The **`orgstart` dedup is therefore REQUIRED in the rung** (§3.3): without it, a committed match's `skip_bytes` re-inserts its absorbed interior and a double insert self-links (`next_tab[pos] = pos`), which poisons future chain walks with same-node loops. The `16 * max_insert_length` cap stays rejected (a wall knob, not a correctness knob — v2) |
| look-ahead-one probe | `:227-231` probe inserts at `p+L`, full depth | no equivalent in greedy; lazy has a **half-depth** probe with promote-scoring (`ldx/compress_lazy.rs:222-243`: `depth >> 1`, `4*Δlen + Δbsr32(offset) > 2`) | the rung's probe: full depth, no promote scoring, cached |
| boundary fix | `fizzle_matches` | **absent** (G11, "located … NOT built", `grep -rn medium src/compress/deflate/level.rs` = nothing) | the one mechanism this lever ports |
| deferred-search cache | `next_match` double-buffer | no equivalent upstream; lazy re-searches a promoted position only *logically* (the half-depth search result gets promoted, never re-walked) | local `(len, offset)` + liveness flag — trivial |
| pending the far-len-3 gate | n/a in zlib-ng medium | **legacy arm already has it**: `src/compress/deflate/parse/far_len3.rs` (exact-trigram pricing, fails-closed, `INERT`) wired per level through `src/compress/deflate/level.rs:172-193,352-356,732-960` and `parse/greedy.rs`/`parse/lazy.rs` | **do not port it into the rung in v1.** The port's L4 keeps libdeflate's own fixed clause `len>3 || offset ≤ 4096` (`compress_greedy.rs:118`). One lever at a time: the far-len-3 porting to ldx is PR #364 (`lever/ldx-len3`, tip `f8b9c7e5`), still open per handoff §4.5 |
| slide-cap constant | 256 (`:155`) | `DEFLATE_MAX_MATCH_LEN = 258` | keep the vendor's 256 cap (faithful, conservative); document as the single intentional adaptation |
| min-match heuristic | none (zlib-ng uses `WANT_MIN_MATCH=4` policy) | `ldx/min_match.rs` — `calculate_min_match_len` (per-block, 4 KiB scan) + depth<16 clamp | the rung keeps the **greedy** cadence: once per block, depth 16 ⇒ no shallow clamp (`min_match.rs:50-59` is inert at depth 16) |

What the port already knows that makes this rung look exactly like this:

* **L3's rung shape is proven precedent**: our shipping L3 is lazy at libdeflate's
  L3 knobs and wins on 20/22 files (`compress_greedy.rs` module docs `:9-15`).
  The `#363` good_match receipt (`(128,65,8)/(256,130,32)`) is the "promote the
  decision rule, keep the table" template this lever copies.
* The failing lattice pairs are precisely the shape fizzle repairs: G11's frontier
  example (lit x2 + match len-5@2482410 vs ours match len-4@2482408 then len-3
  len-5@2482412) and the falsified blanket too-far rule (dickens +14,791 B) recorded
  at `docs/target-encoder-and-gap-analysis.md:701-706`.

### What the ten mechanisms are worth, mechanically

Greedy's losing scenario on the lattice fixtures is: current match accepted
long-but-myopic; one-few bytes later a *correct* match exists whose source continues
through the discarded bytes. Lazy fixes it by deferring (emits literal, promotes
next) — which changes **every** decision point, pays a half-depth walk per non-nice
match, and adopts-parse-difference bookkeeping. Fizzle fixes the *same* scenario
only when the discarded bytes genuinely concatenate into the source of the next
match (the `*match == *orig` chain) — commit is structural, not cost-modelled: it
fires when it *can*, not when it *should*, and the one rung's verdict is whether
"can" is frequent enough on the fixtures. That is the whole risk of the lever
(§6 D1).

---

## 3. Design — the L4 rung

### 3.1 Where it sits

* **Route**: L4 only. Inside `src/compress/ldx/`, new `compress_medium.rs`; the
  route changes exactly one match arm in `ldx/compress.rs:171-179` (`2..=4` greedy
  → `2..=3` greedy, `4` medium). `LdxCompressor::new`'s map at
  `ldx/compress.rs:114-135` stays namespace: L4 keeps `(16,30)` — **the knobs
  column does not move with this lever**.
* **Ladder position after landing** (one-encoder reality: L2/L4/L5 routes to the
  port; L3 remains the legacy lazy+far-len-3 exception under #364):
  `L3 = legacy Lazy(12,14)+far_len3` < `L4 = port Medium(16,30)` ≤ `L5 = port Lazy(16,30)`.
  The rung must *heal* `("tabular",3)`/`("binary",3)` (target ≤ 262,033 / ≤ 661,353)
  without opening any unlisted sag at any adjacent pair, including upward
  (`L5 > L4` would be a new unlisted sag).
* **Sibling memos' grade context** (parent-supplied): this rung class already graded
  `minjs.min.js L5 +0.19%` and `data.sqlite L4 +0.13%` on the real corpus — the two
  named L4/L5 failbar rows of plan §2b. Note L5 is *not* re-routed by this lever;
  minjs L5 closes only if a later decision extends the same rung to L5 (same
  mechanism, opt-in follow-up, never in this PR — plan §1's one-level-at-a-time).

### 3.2 Knobs (and what they are pinned to)

| knob | value | why unchanged |
|---|---|---|
| `max_search_depth` | **16** | libdeflate's L4 verbatim (`compress.rs:126`). The lever is the decision rule, not the depth. zlib-ng's own L4 chain is 24 (`deflate.c:122`) — recorded, **not adopted**; a depth sweep would conflate the falsifier. Also depth ≥ 16 keeps `choose_min_match_len`'s shallow clamp inert (`min_match.rs:51-59`) so L4's min-match semantics don't move. |
| `nice_match_length` | **30** | libdeflate's L4 verbatim (`compress.rs:126`). Probe walks at full depth (medium has no `depth >> 1`; lazy's half-depth is libdeflate's, not medium's). |
| min-match | greedy cadence: `calculate_min_match_len` once per block (`min_len − 1` as the finder's best_len floor) | NOT lazy's mid-block recalc — medium's identical no-recalc keeps this a decision-rule rung; the cadence machinery is `recalculate_min_match_len` (`min_match.rs:104-120`) and belongs to the lazy lineage. The far-len-3 gate stays legacy-only (§2 table). |
| len-3 policy | keep greedy's `(len > 3 || offset ≤ 4096)`-shaped accept at the rung's own emit boundary | zlib-ng's medium bans len-3 (`WANT_MIN_MATCH=4`) — adopting *that* would collide with the entire far-len-3 lever family (dd79_bin6 history: `far_len3.rs:3-30`) and double the blast radius. Refuse the zlib-ng semantic fork in v1. |
| slide cap | `n.match_length >= 256` abort (vendor `:155`) | vendored verbatim; the rung's contract. `max_len` for the probe stays `DEFLATE_MAX_MATCH_LEN` via the existing `adjust_max_and_nice_len` (`sequences.rs:145-150`). |

### 3.3 The loop, in port terms (the one-turn implementer's shape)

New file `src/compress/ldx/compress_medium.rs`, `MediumState` = reuse `GreedyState`
(`compress_greedy.rs:35-48`; hash-chain finder + sequences already parameter-free):

```rust
// per block (donor: compress_greedy.rs:88-99): in_block_begin, choose_max_block_end,
// init_block_split_stats, deflate_begin_sequences, min_len = calculate_min_match_len(…)
//
// per iteration, with p = in_next:
// cur_org = the first position of this match's span that has NOT yet been
// inserted into the hash tables (the vendor's `orgstart`; REQUIRED — see above).
let (mut cur_len, cur_off, mut cur_org): (u32, u32, usize);
if next_live {
    (cur_len, cur_off, cur_org) = cached_next;   // FREE — last iteration's probe
    next_live = false;
} else {
    cur_len = hc_matchfinder_longest_match(mf, in, &mut in_cur_base, p,
        min_len - 1, max_len, nice_len, max_search_depth, &mut next_hashes, &mut cur_off);
    cur_org = p;
    // this call inserts p and leaves next_hashes valid for p+1 (hc_matchfinder.rs:225-239)
}
// The emitted unit is 1 byte when the candidate is REJECTED (greedy accept
// predicate fails: none, or len-3 at offset > 4096 — §3.2 keeps that predicate),
// else the match length. zlib-ng reaches the same shape with WANT_MIN_MATCH
// collapsing (:108-118); ours collapses at the predicate, not at 4.
let unit_len = if cur_len >= min_len && (cur_len > 3 || cur_off <= 4096) {
    cur_len as usize
} else {
    1 // emit a single literal at p; the interim positions are examined normally
};
// ── the look-ahead one (inserts first, then probe BEFORE emit; full depth) ──
let q = p + unit_len;
if q + 1 < in_end {
    // interior inserts for the emitted unit, orgstart-bounded — the medium
    // replacement for greedy's unconditional skip_bytes (vendor insert_match
    // :47-91; greedy never re-examines a position, medium's commits do):
    //   positions [p+1 .. q-1] are inserted EXCEPT those < cur_org.
    let first = core::cmp::max(p + 1, cur_org);
    if unit_len > 1 && first < q {
        hc_matchfinder_skip_bytes(mf, in, &mut in_cur_base, first, in_end,
            (q - first) as u32, &mut next_hashes);   // inserts first..q-1, hashes → q
        // (hc's own near-tail early-out at hc_matchfinder.rs:432-434 is preserved
        //  by calling it rather than hand-inserting.)
    }
    let mut n_len = 0u32;
    let mut n_off = 0u32;
    n_len = hc_matchfinder_longest_match(mf, in, &mut in_cur_base, q,
        min_len - 1, max_len, nice_len, max_search_depth, &mut next_hashes, &mut n_off);
    let mut next_live_candidate = n_len >= min_len;   // "no match" returns best_len = min_len − 1
    // ── fizzle — PORT OF deflate_medium.c:128-176 VERBATIM in (position, len, match_start) form ──
    // Local struct mirroring `struct match` (:16-21): { match_start, match_length, strstart, orgstart }.
    // From (len, offset): strstart = position, match_start = position − offset.
    // dist = strstart − match_start is INVARIANT under the slide (both decrease by j);
    // length grows by j. Commit is only when current collapses to exactly 1.
    if next_live_candidate && unit_len > 1 {
        let src0 = q - n_off as usize;                // next's source position (match_start)
        // vendor pre-conditions at the call site (:233-237):
        //   unit_len > 1  and  (unit_len − 1) <= src0  [match_start floor]
        if unit_len - 1 <= src0 {
            // vendor pre-test (:134-139) — one byte of a shifted boundary:
            //   match position = src0 − (unit_len − 1),  orig position = q − (unit_len − 1) = p + 1
            if in[src0 - unit_len + 1] == in[p + 1] {
                // slide loop (:145-167), one j at a time, pointers step back together:
                //   step-j test:  in[src0 − 1 − j] == in[q − 1 − j]
                //   guards IN ORDER: c_fail = (unit_len − j) < 1 → break (:151)
                //                    n_str-after-slide ≤ limit → break (:153; limit =
                //                    q > MAX_DIST ? q − MAX_DIST : 0, :145 — port: break when
                //                    q − j < n_off, i.e. the slid source start would go negative)
                //                    n_len + j >= 256 → break (:155)
                //                    src0 − j <= 1 → break (:157)
                //   accept step: n_str -= 1; n_src -= 1; n_len2 += 1; c_len -= 1; changed++
                // commit rule (:169-175): changed && c_len <= 1 && n_len2 != 2
                //   c_len = unit_len − j; note c_len ≥ 1 ALWAYS (the < 1 guard breaks first),
                //   so c_len ≤ 1 means c_len == 1 i.e. j == unit_len − 1 exactly.
                //   On commit: cur emits its 1 byte as a literal (unit collapses);
                //   cached_next := (n_len2, unchanged n_off, orgstart = q + 1) at strstart
                //   q − j = p + 1 — the orgstart marks that positions [p+1 … q] are already
                //   in the tables (p+1..q−1 from the unit's own sweep above, q from the
                //   probe's search), so n's later emit only inserts [q+1 …]. next_live
                //   stays true. Non-commit changes roll back ENTIRELY (:173-175):
                //   mutate local copies only; originals stay untouched (vendor :141-142).
            }
        }
    }
}
// emit (donor: compress_greedy.rs:118-147) — the (possibly slid) current unit:
//   match → deflate_choose_match(cur_len, cur_off) + the orgstart-bounded insert
//           sweep above (first = max(p+1, cur_org)) + p += cur_len
//   literal → deflate_choose_literal + p += 1
//   the COMMIT case above is exactly the literal branch (c_len == 1) followed by
//   the cached n' being consumed as the next current.
```

* `q + 1 < in_end` (the `probe_guards`): mirror zlib-ng's two look-ahead gates (`:223,227`) — probe only
  when `q + 1 < in_end` leaves room for the 4-byte load horizon and, per
  `:227`, the probe position is not inside the final `MIN_LOOKAHEAD` stretch.
  Port simplification, stated in-code: no absolute window edge exists in the port
  (the matchfinder self-slides, `hc_matchfinder.rs:189-193`), so the guard is the
  tail-bounds one only.
* **Block boundary**: reset `next_live = false` at each block start (soft-max
  300 000 B grid). zlib-ng's cache survives block flushes; ours does not, on
  purpose: it costs ≤ 1 extra full walk per 300 KB block and it keeps each block a
  deterministic unit for Phase 1's exact state carry (plan §3) — the replayed
  matchfinder table must not depend on a hidden cross-block cache.
* The commit case advances `in_next` to the slid match's start (the new current is
  the cached slide at `p + 1`), so the next emission's insert sweep runs
  `first = max(p_next + 1, q + 1)` — the slide absorbed interior positions that
  are already in the tables, and `orgstart` is what keeps the one-insert-per-byte
  invariant intact (a double insert self-links `next_tab[pos] = pos` and poisons
  future chain walks — see the §2 interior-inserts row).

### 3.4 What it must NOT disturb (each row is a ledger/search fact)

| pinned surface | why | how the rung respects it |
|---|---|---|
| L3 parse = legacy `Lazy(12,14)` + far-len-3 gate, switches in `deflate/level.rs:762-791` (`far_len3_gate: true`, greedy shadow `false`); its won cells are ledger rows in `tests/fingerprints/ledger.tsv` | The L3 lever receipt is PR #364 (`lever/ldx-len3`); the open board cell `access.log L3 vs libdeflate +0.42%` is *that* lever's target (plan §2b; handoff §4.5) | rung touches **nothing under `src/compress/deflate/`**; merge order: stack slices → #363 → #364 → this rung (handoff §3.1, branch-garden-manifest) |
| L5–L9 routes and outputs | one-encoder ladder authority; only L4 reroutes | single match-arm split in `compress.rs`; L5/L8/L9 routings untouched |
| Byte-tie cage with libdeflate | port hunks preserve the whole-cell byte ties at L4 when the fizzle never fires; the tie cage is the campaign's zero-tolerance rule | the **mechanism diff is enumerable**: on every non-commit position, medium output = greedy output byte-for-byte (same table discipline, same emission helpers). The tie cage survives exactly where fizzle never fires |
| `KNOWN_SAGS` list discipline | healing a listed pair (or opening an unlisted pair) fails the suite in that direction | do not pre-list; the rung's commit deletes `("tabular",3)`/`("binary",3)` only if the stack had listed them in its own landing PR |
| Phase 1 state carry + splicer (T>1 = T1 bytes) | plan §3 | the rung is inside the per-chunk encoder, whose seeding/state model does not add cross-chunk state; run `scripts/campaign/parity-census.sh` once on the rung ref — it is the named arbiter for any parse-layer touch |
| Splitter/precode/flush/ht-matchfinder/near-optimal/ultra | different engines, different falsifiers | not touched; `SOFT_MAX_BLOCK_LENGTH`, `SEQ_STORE_LENGTH`, `should_end_block` unchanged |

### 3.5 LOADING ORDER — the `HEAP-CHAIN WALK` constraint

The phrase arrives from the far_len3 PR receipts (cross-agent planning context);
its in-repo content is three orders, all verified today, and the rung may not
violate any:

1. **Inside every matchfinder call, reads precede writes over the same slot.** In
   `hc_matchfinder_longest_match` the canonical sequence is: read hash heads
   (`:213-218`) → write the new node & links (`:225-233`) → compute next hash
   (`:236-239`) → walk the chain via `next_tab` (`:274-298`). The chain walk reads
   a node *before* the reply's store writes (`next_tab[pos]` points to the
   position being inserted). **Call the shipped functions; never hand-roll a fused
   probe walk** — the medium probe must be two plain calls
   (`skip_bytes` then `longest_match`) in greedy's call order, so instruction
   mix/codegen stays the shape the L9-beats-the-C receipt (0.88×, `hc_matchfinder.rs:84-89`
   comment block) was measured on.
2. **Per-block init order** (far_len3's receipt pattern, `parse/lazy.rs:214-232`):
   per-block machinery initializes *before* the first decision of the block and
   re-arms on the widening cadence. The rung's per-block init is the greedy block
   prologue verbatim (`compress_greedy.rs:88-99`); the only new state is loop-local
   (`next_live`, the cached `(len, offset)`), initialized dead.
3. **Branch loading order** (`docs/branch-garden-manifest.md`): stack slices
   (`perf/t1-output-cap`) → `lever/ldx-good-match` (#363, `df31c2a5`) →
   `lever/ldx-len3` (#364, `f8b9c7e5`) → this rung's ref. Never land the rung on a
   base that predates #364: L3's lever is the one this rung's ladder test reads
   against, and it must be the *shipped* L3, not a pending one.

---

## 4. Cost model — deeds per byte, not vibes

**The count identity.** Using the hc contract verified above, both greedy L4 and
medium-L4-probe-on perform exactly:

* **1 chain walk per position the parse actually lands on** (match starts +
  literal positions). Greedy: search at p, then `skip_bytes` over the body, then
  search at p+L when it arrives there in the next iteration.
  Medium: search at p, `skip_bytes` over the body (same count flag for flag:
  both call `skip_bytes(…, length − 1)`), and the search at q=p+L is executed
  *ahead of the emission* and **consumed from the cache next iteration** — never
  re-searched (§1.6 zlib-ng `:210-214`). One walk per examined position, both.
* **1 hash insert per byte**, both: every passed position inserts exactly once
  (search-insert at p and at q; the orgstart-bounded sweep for interiors — the
  dedup is what makes that true across committed slides, §3.3). Insert count and
  table evolution are greedy-identical under the v1 rule
  (§3.2-3.3). Committed slides add nothing new: the slid region was going to be
  inserted/covered either way; the boundary just moved.
* **The rung's marginal cost** = `fizzle_matches`' work: 2 byte loads on the
  pre-test for every probed position, then `k` more per slide attempt, and it
  commits only under the collapse rule. `_quick_exit` iterations and tail
  short-circuits are end-of-buffer-only (< `MIN_LOOKAHEAD` bytes per stream).

| parse | full-depth walks / byte | half-depth walks / byte | hash inserts / byte | extra per position |
|---|---|---|---|---|
| port greedy L4 (today) | 1 per visited position | 0 | 1 (body-1 per match) | — |
| port medium L4 (this rung) | 1 per visited position (probe = next visit, cached) | 0 | 1 | 2 byte loads + O(k) slide reads per probe |
| port lazy L4 (rejected rung for wall) | 1 per visited position | 1 per non-nice match (`depth >> 1`, `compress_lazy.rs:233`) | 1 | bsr32 promote scoring + literal plumbing |

**Predicted wall** ≈ greedy's wall + fizzle overhead ≈ **+0–3% vs today's L4** at
T1, with the same store/read mix. Anchors that make this an acceptable spend:

* Legacy record of the *rejected* alternative: full lazy at L4 = **+17.7% wall**
  (G4, `target-encoder-and-gap-analysis.md:440`) — measured pre-port; in-process
  port regime stands at 1.03–1.11× of C libdeflate overall (plan §0), but the
  relative gap between probe-on and lazy also persists (lazy's additional
  full-position probe outlives codegen improvements).
* The branch's census rows for L4 sit in the win-with-margin tier, wall ratios
  ≤ 0.80 (parent-supplied census tiering) → promotion-rule clause 5 prices margin
  as capital: a confirmed real erosion is acceptable while
  `post_ratio ≤ min(0.80, 1 − 3 × layout_floor(cell))`
  (`docs/promotion-rule.md:64-74`), and `data.parquet`/`movie.mp4`-class thin cells
  keep the flat 0.005 budget (they are not winning cells at T1 in the old census;
  current post-pivot state has TUNE L4 at zero fails, parent-supplied).
* T4: G29 already measured **249–330% wall slack** against running a stronger
  parse at T4 (`level.rs:417`); the rung's ≤3% is invisible there by an order of
  magnitude. The T1 leg is the actionable one, and it is covered by (a) the count
  identity above (fizzle only) and (b) the clause-5 floor.
* The instrument already exists for verification, not prediction:
  `fulcrum anatomy` on a build with `--features anatomy-counters` — the behavior
  of the count lines is pinned (`hc_probe_attempts`,
  `hc_probe_outcome_{miss,too_short,accepted}`, `hc_chain_table_reads`,
  `hc_head_table_reads/writes`, `hc_positions_skipped`, `hc_hash_computations`;
  `hc_matchfinder.rs:114-167`, `anatomy-counters.rs`). Assert the rung's
  attempts-per-byte delta vs greedy on `tabular` before promotion ("counted, never
  inferred" — plan §3).

---

## 5. Falsifier plan — the frozen sequence, in order

Nothing past stage 1 touches the box; everything is gate-composable and is
judged by the standing rules (`docs/promotion-rule.md` — clause 3 pass→fail flips
absolute; clause 5 margin-floor; clause 7 wall verdicts from the frozen box only).

1. **Build gates (any box, in-process, seconds):**
   `cargo test --release`
   * `ladder_is_monotone_t1` (`tests/size_invariants.rs:139-172`): the 4 fixtures
     `["text","tabular","binary","noise"]` at T1, every adjacent pair. Promotion
     needs `("tabular",3)`: new L4 ≤ 262,033 and `("binary",3)`: ≤ 661,353
     (`parent-supplied` lattice numbers), no new sag pairs anywhere, no healing of
     unknown sags.
   * `noise_expansion_bounded_t1/_t4` + the incompressible slack ratchet
     (`:101-106,209-217`).
   * `won_cells_stay_won` + `fingerprints_match_pins` (`tests/fingerprint_suite.rs:100,155`):
     **expect these to pass unchanged** — the pin grid is `LEVELS = [1,2,6,9]`
     (`:31`), which does not include L4, and the rung does not touch L1/L2/L6/L9
     parses. A failure here = the rung leaked outside L4 → fix the routing before
     anything else.
2. **Measure tables (print, not gate):**
   `cargo test --test size_invariants measure_tables -- --ignored --nocapture`
   — per-pair ladder and per-level slack; record `tabular`/`binary` L3→L4/L5 deltas
   in the PR body. Add `("tabular",3)`/`("binary",3)` ONLY if the in-flight stack
   had listed them; else list nothing.
3. **Mechanism diff (the lever's own fingerprint):**
   `cargo run --release --example fingerprint_tool -- pin-ours` → the regenerated
   `tests/fingerprints/ours.tsv` diff IS the mechanism for review: changed
   cells should be L4-only (well, L{4} here, plus whatever cells the pin grid
   covers). Any L1/L2/L6/L9 movement → wrong level got routed; stop.
   `examples/ldx_divergence.rs` remains the stage-2 diff oracle for block-level
   spans against plain-libdeflate L4.
4. **The frozen box (solvency, AMD Zen2, wave-run, single quiet box):**
   the frozen `fulcrum try` ORDER is:
   1. `fulcrum try <base-ref> --rescore` (or reuse the banked
      `size-all-ee0c1d2c`/post-stack census artifact) — establishing operated-base
      won cells for L3/L4/L5.
   2. `fulcrum try <rung-ref> --levels 3,4,5 --threads 1,4`
      (scoped levels like the #363/#364 template: handoff §4.5 "scoped try levels
      …"; barrier-namespace `_threads` from plan §3 and campaign `wave-runner.sh`).
   3. Judge under `docs/promotion-rule.md` clauses 1–8 with `try.json`
      (`adjudication.clause*`), including the confirm-short-circuit semantics for
      wall suspects (`:54-74,121-125`).
   4. On SHIP: merge order from §3.5 row 3; run `scripts/campaign/board-size.sh all`,
      then once on the ref `scripts/campaign/parity-census.sh` (guaranteeing T1==T4
      bytes per chunk grid after a parse-layer touch), then wait for the wave
      queue — never run try runs on a busy box or off-solvency.
   Env/harness facts for the box are concretized in handoff §5
   (`CAMPAIGN_FULCRUM`, `CAMPAIGN_CORPUS_ROOT`, `/root/wave-runner.sh`,
   `/root/wave-queue.txt`).
5. **Stop rule**: any clause-3 CONFIRMED flip, or any clause-5 conviction chain
   that cannot be acquitted by LAYOUT-ARTIFACT, = record closed, do not re-sample
   (plan §8's stop rule; promotion-rule §134's noise floor ~1.5% at n=15 says
   raise `n`, never read tea leaves).

---

## 6. The three ways this lever dies — with the pinch tests

Like the L2 len-3 lane did (full receipt trail in `parse/far_len3.rs:11-30`:
UNCONDITIONAL drop = 53 flips incl. weights.safetensors +203,233 B; MEAN-literal
cost gate = 28 flips — fixed-constant policies die on the tie-guard lattice), the
rung dies in one of exactly three ways, each pre-pinned with its test:

### D1 — Fizzle is too weak: the sag survives

**How it dies:** the rung lands, `tabular` stays e.g. ~266-270 KB (the +9,472 B
gap not closed). Mechanism: the commit rule demands c collapse ≤ 1 with the
*byte-pair equality chain back to p* — on tabular data whose "correct" second
match has a *different* source than the one that can absorb the first match's
bytes (mixed-offset tables, alternating columns of different repeats), lazy's
promote buys deferral that fizzle structurally cannot express. fizzle's outcome set
is a SUBSET of lazy's *plus* the merged-long-match case, but the subset need not
contain enough of the tabular 9,472 B.

**Pinch test (in-process, one command):** `cargo test --test size_invariants
measure_tables -- --ignored --nocapture` on the rung build vs base build; the
fizzle count comes free from anatomy (`--features anatomy-counters`,
counter `medium_fizzle_slide_steps`/`commits` if added per §4):

| pinch | receipt that RULES OUT D1 | receipt that CONFIRMS D1 |
|---|---|---|
| `tabular` L4 bytes after rung | ≤ 262,033 (any margin; ladder must stay monotone vs L5) | > 262,033 AND commits recorded near zero (commit rate < ~1/512 observations) ⇒ slide never fires on the class |
| `binary` L4 bytes | ≤ 661,353 (the +2,230 B is loose) | indifferent here; binary can stay as-is |

**If D1:** the rung is recorded closed-at-both-coordinates *with the rung's receipt*
and the §3 plan-B rung takes over pre-specified (no new design turn needed):

> **Plan B (mechanism-successor, vendor-precedented, pre-registered):** route L4 to
> the port's own `deflate_compress_lazy` (16,30) **with the `#363` good_match
> brake extension to L4** — the plan doc §5's own alternative ("extension of
> #363's `good_match` port to L4"). libdeflate's L5 is already lazy(16,30), so
> byte-identical L4==L5 rows still satisfy `hi ≤ lo` (the ladder test allows
> equality), the knobs come from libdeflate's own map, and the wall capital is
> already measured in the legacy record at +17.7% (re-measured under the port
> per plan §6's codegen regime). Falsifier: same §5 sequence with levels 3,4,5.

### D2 — Tie-cage bleed on the real corpus (the L2 lane's death replay)

**How it dies:** on the real corpus, fizzle commits land on cells whose current
state is a byte tie or hairline pass, and clause 3 convicts: "previously-passing
per-label cell that fails blocks the ship". The board's hairline band
(`photo.jpg L1-L3 +0.04%`, `weights.safetensors L7-L9 +0.00-0.02%`, `movie.mp4 L6
+0.0008%`) is exactly the cell-class the min-3 lane bled onto at L2 in its
falsified run. The rung's *existing* defense is that it is address-scoped: only
commit positions emit different bytes — but a commit on a near-tie cell is enough
to flip a tie INTO a loss permanently (size cells are exact integers;
promotion-rule §50-52).

**Pinch test (one command each, all pre-run on macOS build, no box access):**
* `cargo run --release --example fingerprint_tool -- pin-ours` before AND after:
  the diff enumerates every touched cell stream. Fizzle commits with a **nonzero
  count but zero effects** on non-L4 rows ⇒ correct scoping. **Any** row change
  outside L4 = leak (the suite at stage 1 catches it).
* `cargo test --test size_invariants` per fixture with the `tabular`/`binary`
  path-pairs enumerated: `hex-bleed on tabular` (the min-3 lane's own signature:
  *release* a far-len-3-style rule and watch the tie bands) is excluded by design
  because fizzle has **no constants to parametrize** — its commits are rule-bound
  by byte pairs, not thresholds. That is the mechanistic difference from the
  dead lane; state it in the PR.
* On the frozen box: `fulcrum why photo.jpg:L0004:T0001` style lookups for any
  cell the try verdict lists (§5), and cross-layout confirm machinery runs
  automatically — LAYOUT-ARTIFACT acquittals shrink clause-6 harm, real ones
  block; never interpret a confirm-UNDECIDED as a pass (`:57-62`).

**If D2:** revert is automatic (clause 3 absolute). The rung variant to try
otherwise (opt-in, single lever, next PR): keep libdeflate's insert discipline =
byte-identity safety net (which v1 does), then bar FIZZLE commits from changing
the *framing* (i.e., only break-ties where the re-encoded stream stays inside the
tie cage) — by construction v1 has no such gate, and adding one recreates the
family of falsified "too-far" gates (the G11 `dickens +14,791 B` blanket). So the
recorded disposition after a D2 flip: **Park the rung's corpus acceptance at the
bottom of the tie cage rather than modify the commit rule**; the rule's
`n.match_length != 2`-style guard already pins the degenerate 2-merges.

### D3 — the wall & ladder second-order effects

**How it dies:** two independent legs, both conservatively pricable:
1. **T1 loss on near-incompressibles at L4** (historically the L4 wall-loss class:
   5 of 51 census losses pre-port sat at L4: `movie.mp4`, `data.parquet`,
   `armexe.elf`, `tool.bin`, `symbols.dwarf`; 0 at T4 — superseded census
   `docs/board/wall-census-complete.md:62-71`, still the worst-class denominators).
   Even at fizzle-only cost, a +1-3% wall margin over libdeflate on those cells
   can convert thin ties/tier-margin into confirmed erosion (clause 5).
2. **L5 overshoot**: the rung must also not make L4 *better than L5* anywhere.
   Medium's slide can, in rare merge-cases, produce a smaller L4 than lazy(16,30)
   L5 on a fixture (fizzle dissolving the match lazy would have deferred around
   differently), which opens an UNLISTED sag (lattice fail) or flips the
   win-direction in the compressed-size bar. The lattice's equal-sizes case
   (L4==L5 byte-identical) is legal; a *larger*-L5 row at any fixture is not.

**Pinch test (in-process, pre-box):** `measure_tables` L4 vs L5 per fixture;
require `L5 ≤ L4` on all four (lattice), and the anatomy counter
`medium_fizzle_commits` should be nonzero on `tabular` while `noise` stays
commit-free (incompressible data should never waste fizzle probes).
**Pinch test (box, paired, `fulcrum ab`-style):** the frozen-box try in §5
already carves exactly this out: `--levels 3,4,5 --threads 1,4` covers the T4
starvation question implicitly (G29 measured the slack envelope 249-330% —
the T4 leg is not the wall's active coordinate), so the actionable wall bar is
the **T1 low-margin library cells**: `movie.mp4`/`data.parquet`-class at L4 T1
must hold clause-5 floors or the verdict is NO-SHIP by the rule, and the confirm
chain adjudicates artifact-vs-real per coordinate.

**If D3:** stop; record; the appendix plan-B rung inherits the same wall ceiling
— a lazy-at-L4 costs MORE wall, so Plan B would need the wall receipt *first*
via its own scoped try; if that also fails, the class is closed without spending
more wall (plan §8 item 3's stop rule).

---

## 7. One-turn implementation checklist (for the coder)

| # | what | where |
|---|---|---|
| 1 | new parser file: `deflate_compress_medium(c, p: &mut GreedyState, …, max_search_depth, nice_match_length)` porting the §3.3 shape from `compress_greedy.rs`; `MediumState` optional (reuse `GreedyState`) | `src/compress/ldx/compress_medium.rs` (new) |
| 2 | route L4: split the `2..=4` arm → `2..=3` greedy, `4 => deflate_compress_medium(…, 16, 30)`; level map entry stays number-for-number | `src/compress/ldx/compress.rs:114-135` (map comment), `:171-179` (route) |
| 3 | register the module | `src/compress/ldx/mod.rs:88-105` |
| 4 | fizzle in `(len, offset)` form with the four vendored aborts + `!= 2` guard, AND the `orgstart` field threaded through the cached match (fresh find: `orgstart = p`; commit: `orgstart = q + 1`), plus the orgstart-bounded interior sweep in the emit path (§3.3); inline `#[inline(always)]` | new file; see §1.5 receipts and the §2 interior-inserts row (the self-link hazard the dedup prevents) |
| 5 | pin regeneration (only if the rung's pins appear): `fingerprint_tool -- pin-ours` in-PR | `tests/fingerprints/ours.tsv` |
| 6 | tests: run the full §5 stage-1 gates + `examples/ldx_divergence.rs` block diff vs base; record `measured_tables` in the PR body | — |
| 7 | per-block cache reset + probe guards documented in the rung's header comment with the C line refs (port-rules compliance: same arithmetic/types, no idiomatic cleanups; the stated deviations from zlib-ng = the len-3 predicate kept instead of `WANT_MIN_MATCH=4` (§3.2), the `16 × max_insert_length` insert cap not adopted while `orgstart` dedup IS required (§2/§3.3), and near-block-end guards mapped to in-buffer bounds (§3.3)) | module doc-comment |
| 8 | branch: rebase onto post-#364; PR description carries §5's receipts verbatim + the graded cell list (data.sqlite L4 +0.13/+0.17%) | — |

Est. size: ~300 lines including the vendored-derivative comments and 3 new tests
(rung round-trips through `LdxCompressor::new(4)`; fizzle unit test exercising
commit and rollback paths; `effort_rises_with_level`-style L3≤L4≤L5 pin per
`compress_greedy.rs:268-285` style).

---

## 8. Honest gaps

* The lattice deltas (`tabular` 271,505 vs 262,033; `binary` 663,583 vs 661,353),
  the win-with-margin census tier (≤0.80 at L4 rows), and the graded real-corpus
  cells (`minjs.min.js L5 +0.19%`, `data.sqlite L4 +0.13%`) are **parent-supplied
  cross-agent data**, not reproducible from this checkout's HEAD (the pick-min
  deletion that re-opens the sags is still in flight on the unmerged stack; this
  branch's `KNOWN_SAGS` does not yet carry the rows). Whoever lands the rung
  re-derives them from the in-flight stack before touching `KNOWN_SAGS`.
* `HEAP-CHAIN WALK` is translated above (§3.5) to the three in-repo orders it can
  mean; PR #364's body holds the receipts for the exact phrasing — read it at
  merge time, not now.
* The claim "fizzle alone closes the tabular sag" is a **hypothesis, not a
  measurement** — D1's pinch table is the falsifier; Plan B is pre-registered so
  the turn does not dead-end.
* This memo touches nothing: it is read+write-only planning per the campaign's
  no-cargo-runs rule for planning agents; no build, no box, no git mutation was
  performed while producing it.
