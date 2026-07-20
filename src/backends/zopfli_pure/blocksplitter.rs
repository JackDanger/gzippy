//! Block splitter: greedy bisection of an LZ77 stream into cheaper blocks.
//! Port of Google Zopfli blocksplitter.c.

use super::deflate_size::{calculate_block_size, calculate_block_size_auto_type};
use super::hash::ZopfliHash;
use super::lz77::{lz77_greedy, BlockState, LZ77Store};
use super::symbols::{ZOPFLI_LARGE_FLOAT, ZOPFLI_WINDOW_SIZE};
use super::ZopfliOptions;
use std::collections::HashMap;

/// Split-point search granularity for the recursive bisection. The C code
/// hard-codes 9; keep it.
const FIND_MINIMUM_NUM: usize = 9;

/// Estimated bit cost of encoding `lz77[lstart..lend]` as a single block.
/// Mirrors C `EstimateCost`; just dispatches to the auto-type sizer.
#[inline]
fn estimate_cost(lz77: &LZ77Store<'_>, lstart: usize, lend: usize) -> f64 {
    calculate_block_size_auto_type(lz77, lstart, lend)
}

/// `cost(start, i) + cost(i, end)` — the cost of splitting at `i`.
#[inline]
fn split_cost(i: usize, lz77: &LZ77Store<'_>, start: usize, end: usize) -> f64 {
    estimate_cost(lz77, start, i) + estimate_cost(lz77, i, end)
}

/// Finds the `i` in `[start, end)` that minimises `f(i)`. For wide ranges
/// (`>= 1024`) uses a 9-way recursive narrowing; for narrow ones, brute force.
/// Returns `(arg_min, min)` so callers don't need an out-pointer.
///
/// The 9 `f(p[i])` evaluations per narrowing iteration are independent, so
/// we run them in parallel via `std::thread::scope` whenever
/// `thread_budget != 1`. Budget `1` keeps the original serial loop (used
/// when an outer pool is already saturating CPUs — see `ZopfliOptions`).
/// Brute-force branch and the per-iteration argmin remain serial; both are
/// trivial relative to one `f()` call.
fn find_minimum<F: Fn(usize) -> f64 + Sync>(
    f: F,
    start: usize,
    end: usize,
    thread_budget: u32,
) -> (usize, f64) {
    if end - start < 1024 {
        let mut best = ZOPFLI_LARGE_FLOAT;
        let mut result = start;
        for i in start..end {
            let v = f(i);
            if v < best {
                best = v;
                result = i;
            }
        }
        return (result, best);
    }

    let mut start = start;
    let mut end = end;
    let mut p = [0usize; FIND_MINIMUM_NUM];
    let mut vp = [0f64; FIND_MINIMUM_NUM];
    let mut pos = start;
    let mut lastbest = ZOPFLI_LARGE_FLOAT;

    loop {
        if end - start <= FIND_MINIMUM_NUM {
            break;
        }

        for (i, slot) in p.iter_mut().enumerate() {
            *slot = start + (i + 1) * ((end - start) / (FIND_MINIMUM_NUM + 1));
        }
        if thread_budget == 1 {
            for i in 0..FIND_MINIMUM_NUM {
                vp[i] = f(p[i]);
            }
        } else {
            let f_ref = &f;
            let p_snap = p;
            let computed: [f64; FIND_MINIMUM_NUM] = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..FIND_MINIMUM_NUM)
                    .map(|i| scope.spawn(move || f_ref(p_snap[i])))
                    .collect();
                let mut out = [0f64; FIND_MINIMUM_NUM];
                for (i, h) in handles.into_iter().enumerate() {
                    out[i] = h.join().unwrap();
                }
                out
            });
            vp = computed;
        }
        let mut besti = 0;
        let mut best = vp[0];
        for (i, &v) in vp.iter().enumerate().skip(1) {
            if v < best {
                best = v;
                besti = i;
            }
        }
        if best > lastbest {
            break;
        }

        start = if besti == 0 { start } else { p[besti - 1] };
        end = if besti == FIND_MINIMUM_NUM - 1 {
            end
        } else {
            p[besti + 1]
        };

        pos = p[besti];
        lastbest = best;
    }

    (pos, lastbest)
}

/// Insertion-sort a single value into the already-sorted `out`. Mirrors
/// `AddSorted` byte-for-byte (one push, one in-place shift).
fn add_sorted(value: usize, out: &mut Vec<usize>) {
    out.push(value);
    let n = out.len();
    for i in 0..n - 1 {
        if out[i] > value {
            for j in (i + 1..n).rev() {
                out[j] = out[j - 1];
            }
            out[i] = value;
            break;
        }
    }
}

/// Picks the longest `[start, end)` interval bounded by existing splitpoints
/// (or by `[0, lz77size - 1]`) whose `start` is not yet marked done. C
/// `FindLargestSplittableBlock` returns a bool + out-params; we return
/// `Option<(start, end)>`.
fn find_largest_splittable_block(
    lz77size: usize,
    done: &[u8],
    splitpoints: &[usize],
) -> Option<(usize, usize)> {
    let mut longest = 0;
    let mut found: Option<(usize, usize)> = None;
    let n = splitpoints.len();
    for i in 0..=n {
        let start = if i == 0 { 0 } else { splitpoints[i - 1] };
        let end = if i == n { lz77size - 1 } else { splitpoints[i] };
        if done[start] == 0 && end - start > longest {
            longest = end - start;
            found = Some((start, end));
        }
    }
    found
}

/// Optionally print the split points in decimal + hex (gated on `verbose`).
/// Mirrors `PrintBlockSplitPoints`; we walk the LZ77 to recover byte
/// positions for display.
fn print_block_split_points(lz77: &LZ77Store<'_>, lz77splitpoints: &[usize]) {
    if lz77splitpoints.is_empty() {
        eprintln!("block split points:  (hex:)");
        return;
    }

    let mut byte_points: Vec<usize> = Vec::with_capacity(lz77splitpoints.len());
    let mut pos: usize = 0;
    for i in 0..lz77.size() {
        let length = if lz77.dists[i] == 0 {
            1
        } else {
            lz77.litlens[i] as usize
        };
        if byte_points.len() < lz77splitpoints.len() && lz77splitpoints[byte_points.len()] == i {
            byte_points.push(pos);
            if byte_points.len() == lz77splitpoints.len() {
                break;
            }
        }
        pos += length;
    }
    debug_assert_eq!(byte_points.len(), lz77splitpoints.len());

    let mut s = String::from("block split points: ");
    for p in &byte_points {
        s.push_str(&format!("{} ", p));
    }
    s.push_str("(hex:");
    for p in &byte_points {
        s.push_str(&format!(" {:x}", p));
    }
    s.push(')');
    eprintln!("{}", s);
}

/// Greedily splits the LZ77 stream into blocks. Each iteration finds the
/// best single-position split of the largest as-yet-unsplittable interval
/// and accepts it iff it strictly reduces the auto-type block cost.
/// Returns the LZ77-index split points in sorted order.
pub fn block_split_lz77(
    options: &ZopfliOptions,
    lz77: &LZ77Store<'_>,
    maxblocks: usize,
) -> Vec<usize> {
    let mut splitpoints: Vec<usize> = Vec::new();

    if lz77.size() < 10 {
        return splitpoints;
    }

    let mut done = vec![0u8; lz77.size()];
    let mut lstart = 0usize;
    let mut lend = lz77.size();
    let mut numblocks: usize = 1;

    loop {
        if maxblocks > 0 && numblocks >= maxblocks {
            break;
        }

        debug_assert!(lstart < lend);
        let (llpos, splitcost) = find_minimum(
            |i| split_cost(i, lz77, lstart, lend),
            lstart + 1,
            lend,
            options.thread_budget,
        );

        debug_assert!(llpos > lstart);
        debug_assert!(llpos < lend);

        let origcost = estimate_cost(lz77, lstart, lend);

        if splitcost > origcost || llpos == lstart + 1 || llpos == lend {
            done[lstart] = 1;
        } else {
            add_sorted(llpos, &mut splitpoints);
            numblocks += 1;
        }

        match find_largest_splittable_block(lz77.size(), &done, &splitpoints) {
            None => break,
            Some((s, e)) => {
                lstart = s;
                lend = e;
            }
        }

        if lend - lstart < 10 {
            break;
        }
    }

    if options.verbose != 0 {
        print_block_split_points(lz77, &splitpoints);
    }

    splitpoints
}

// ── Recursive exact-cost splitter (optimal-parse frontier / ECT port) ────────

/// Recursive divide-and-conquer block splitter over the LZ77 index range.
/// Mirrors the optimal-parse frontier's `Splitter` (which produces block
/// counts identical to ECT): at each node find the best interior split via
/// 9-way narrowing *plus a brute-force scan over the final ≤9-position window*
/// (the precision the greedy `find_minimum` omits — it breaks out of the
/// narrowing without evaluating those last positions), and if it strictly
/// reduces the exact block cost, split and recurse into BOTH halves.
///
/// `auto_type == false` uses dynamic-only cost (`calculate_block_size(..,2)`),
/// matching ECT's `SplitCost` (3 + GetDynamicLengths + CalculateTreeSize);
/// `true` uses the stored/fixed/dynamic best-of-three (`_auto_type`).
struct RecursiveSplitter<'a, 'b> {
    lz77: &'a LZ77Store<'b>,
    memo: HashMap<(usize, usize), f64>,
    auto_type: bool,
}

impl RecursiveSplitter<'_, '_> {
    /// Exact block cost of `lz77[a..b]`, memoized per `(a, b)`.
    fn cost(&mut self, a: usize, b: usize) -> f64 {
        if let Some(&c) = self.memo.get(&(a, b)) {
            return c;
        }
        let c = if self.auto_type {
            calculate_block_size_auto_type(self.lz77, a, b)
        } else {
            calculate_block_size(self.lz77, a, b, 2)
        };
        self.memo.insert((a, b), c);
        c
    }

    /// Minimise `cost(a,s)+cost(s,b)` over `s` in `(a, b)` with zopfli's 9-way
    /// narrowing, then brute-force the final window. Returns `(best_s,
    /// best_c)`; `best_c == ZOPFLI_LARGE_FLOAT` ⇒ no interior split exists.
    fn best_split_in(&mut self, a: usize, b: usize) -> (usize, f64) {
        if b < a + 2 {
            return (a, ZOPFLI_LARGE_FLOAT);
        }
        let start = a + 1;
        let end = b - 1; // inclusive search bounds for s
        const NUM: usize = FIND_MINIMUM_NUM;
        if end - start < NUM {
            let mut best_s = start;
            let mut best_c = ZOPFLI_LARGE_FLOAT;
            for s in start..=end {
                let c = self.cost(a, s) + self.cost(s, b);
                if c < best_c {
                    best_c = c;
                    best_s = s;
                }
            }
            return (best_s, best_c);
        }
        let mut lo = start;
        let mut hi = end;
        let mut best_s = start;
        let mut best_c = ZOPFLI_LARGE_FLOAT;
        let mut last = ZOPFLI_LARGE_FLOAT;
        loop {
            if hi - lo < NUM {
                for s in lo..=hi {
                    let c = self.cost(a, s) + self.cost(s, b);
                    if c < best_c {
                        best_c = c;
                        best_s = s;
                    }
                }
                break;
            }
            let step = (hi - lo) / (NUM + 1);
            let mut p = [0usize; NUM];
            let mut vp = [0f64; NUM];
            let mut bi = 0;
            for i in 0..NUM {
                p[i] = lo + (i + 1) * step;
                vp[i] = self.cost(a, p[i]) + self.cost(p[i], b);
                if vp[i] < vp[bi] {
                    bi = i;
                }
            }
            if vp[bi] < best_c {
                best_c = vp[bi];
                best_s = p[bi];
            }
            if vp[bi] >= last {
                break;
            }
            last = vp[bi];
            lo = if bi == 0 { lo } else { p[bi - 1] };
            hi = if bi == NUM - 1 { hi } else { p[bi + 1] };
        }
        (best_s, best_c)
    }

    /// Split `(a, b)` at its best interior boundary iff that strictly reduces
    /// the exact cost, then recurse into both halves. `cap` bounds the total
    /// number of interior split points collected.
    fn recurse(&mut self, a: usize, b: usize, splits: &mut Vec<usize>, cap: usize) {
        if splits.len() >= cap || b < a + 2 {
            return;
        }
        let whole = self.cost(a, b);
        let (best_s, best_c) = self.best_split_in(a, b);
        if best_c < whole && best_s > a && best_s < b {
            splits.push(best_s);
            self.recurse(a, best_s, splits, cap);
            self.recurse(best_s, b, splits, cap);
        }
    }
}

/// Recursive exact-cost split of an already-built LZ77 stream. Returns the
/// sorted LZ77-index split points. `maxblocks` caps the block count (as in the
/// greedy splitter: at most `maxblocks - 1` interior splits); `0` ⇒ unbounded.
/// `auto_type` selects the cost model (see `RecursiveSplitter`).
pub fn block_split_lz77_recursive(
    lz77: &LZ77Store<'_>,
    maxblocks: usize,
    auto_type: bool,
) -> Vec<usize> {
    if lz77.size() < 10 {
        return Vec::new();
    }
    let cap = if maxblocks > 0 {
        maxblocks.saturating_sub(1)
    } else {
        usize::MAX
    };
    let mut sp = RecursiveSplitter {
        lz77,
        memo: HashMap::new(),
        auto_type,
    };
    let mut splits: Vec<usize> = Vec::new();
    sp.recurse(0, lz77.size(), &mut splits, cap);
    splits.sort_unstable();
    splits.dedup();
    splits
}

/// Convenience wrapper: greedy-parse `in_[instart..inend]`, split the LZ77
/// stream, then convert the LZ77 indices back to byte positions in `in_`.
pub fn block_split(
    options: &ZopfliOptions,
    in_: &[u8],
    instart: usize,
    inend: usize,
    maxblocks: usize,
) -> Vec<usize> {
    let mut store = LZ77Store::new(in_);
    let mut state = BlockState::new(options, instart, inend, false);
    let mut hash = ZopfliHash::new(ZOPFLI_WINDOW_SIZE);

    lz77_greedy(&mut state, in_, instart, inend, &mut store, &mut hash);

    let lz77splitpoints = block_split_lz77(options, &store, maxblocks);

    if lz77splitpoints.is_empty() {
        return Vec::new();
    }

    // Translate LZ77 indices back to byte positions in `in_`.
    let mut byte_points: Vec<usize> = Vec::with_capacity(lz77splitpoints.len());
    let mut pos = instart;
    for i in 0..store.size() {
        let length = if store.dists[i] == 0 {
            1
        } else {
            store.litlens[i] as usize
        };
        if byte_points.len() < lz77splitpoints.len() && lz77splitpoints[byte_points.len()] == i {
            byte_points.push(pos);
            if byte_points.len() == lz77splitpoints.len() {
                break;
            }
        }
        pos += length;
    }
    debug_assert_eq!(byte_points.len(), lz77splitpoints.len());
    byte_points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sorted_keeps_sorted() {
        let mut v: Vec<usize> = Vec::new();
        for n in [5, 1, 3, 7, 4, 2, 6] {
            add_sorted(n, &mut v);
        }
        assert_eq!(v, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn add_sorted_handles_duplicates() {
        let mut v: Vec<usize> = Vec::new();
        for n in [3, 3, 1, 3] {
            add_sorted(n, &mut v);
        }
        assert_eq!(v, vec![1, 3, 3, 3]);
    }

    #[test]
    fn find_minimum_brute_force_branch() {
        // < 1024 → brute-force scan, returns the global minimum.
        let f = |i: usize| ((i as f64) - 7.5).powi(2);
        let (arg, _val) = find_minimum(f, 0, 16, 1);
        // 7 and 8 tie; brute force picks the *first* (smaller) index.
        assert_eq!(arg, 7);
    }

    #[test]
    fn find_minimum_serial_and_parallel_agree() {
        // ≥ 1024 → 9-way recursive narrowing. Same closure under
        // budget=1 (serial path) and budget=0 (parallel path) must
        // produce identical (arg, min). The function has many local
        // minima; pick a smooth quadratic so the narrowing actually
        // converges.
        let f = |i: usize| ((i as f64) - 5_000.5).powi(2);
        let (arg_serial, val_serial) = find_minimum(f, 0, 10_000, 1);
        let (arg_parallel, val_parallel) = find_minimum(f, 0, 10_000, 0);
        assert_eq!(arg_serial, arg_parallel);
        assert_eq!(val_serial.to_bits(), val_parallel.to_bits());
    }

    #[test]
    fn find_largest_splittable_block_skips_done() {
        // Intervals: [0,2), [2,7), [7,9). [2,7) is the largest (length 5).
        // For size=10 the last interval ends at lz77size - 1 = 9.
        let done = vec![0u8; 10];
        let splits = vec![2usize, 7];
        assert_eq!(
            find_largest_splittable_block(10, &done, &splits),
            Some((2, 7))
        );

        // Mark interval [2, 7)'s start as done → next-largest wins.
        let mut done2 = done.clone();
        done2[2] = 1;
        // Remaining: [0,2) length 2 and [7,9) length 2 — tie. Strict `>`
        // means the first encountered wins.
        assert_eq!(
            find_largest_splittable_block(10, &done2, &splits),
            Some((0, 2))
        );

        // All starts done → None.
        let done3 = vec![1u8; 10];
        assert_eq!(find_largest_splittable_block(10, &done3, &splits), None);
    }
}
