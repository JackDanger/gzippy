//! Length-limited Huffman code lengths via bounded package-merge.
//! Port of Google Zopfli katajainen.c

#[derive(Clone, Copy)]
struct Node {
    weight: usize,
    count: i32,
    tail: i32, // arena index, or -1 for None
}

/// Computes length-limited Huffman code lengths.
///
/// Returns Ok(()) on success, Err(()) if (1<<maxbits) < numsymbols or weight overflow.
pub fn length_limited_code_lengths(
    frequencies: &[usize],
    maxbits: i32,
    bitlengths: &mut [u32],
) -> Result<(), ()> {
    let n = frequencies.len();

    for b in bitlengths.iter_mut() {
        *b = 0;
    }

    let mut leaves: Vec<Node> = Vec::with_capacity(n);
    for (i, &f) in frequencies.iter().enumerate() {
        if f > 0 {
            leaves.push(Node {
                weight: f,
                count: i as i32,
                tail: -1,
            });
        }
    }
    let numsymbols = leaves.len() as i32;

    if (1 << maxbits) < numsymbols {
        return Err(());
    }
    if numsymbols == 0 {
        return Ok(());
    }
    if numsymbols == 1 {
        bitlengths[leaves[0].count as usize] = 1;
        return Ok(());
    }
    if numsymbols == 2 {
        bitlengths[leaves[0].count as usize] += 1;
        bitlengths[leaves[1].count as usize] += 1;
        return Ok(());
    }

    for leaf in &leaves {
        if leaf.weight >= (1usize << (usize::BITS - 9)) {
            return Err(());
        }
    }

    leaves.sort_by(|a, b| a.weight.cmp(&b.weight).then(a.count.cmp(&b.count)));

    let maxbits = if numsymbols - 1 < maxbits {
        numsymbols - 1
    } else {
        maxbits
    } as usize;

    let arena_size = maxbits * 2 * numsymbols as usize;
    let mut arena: Vec<Node> = vec![
        Node {
            weight: 0,
            count: 0,
            tail: -1
        };
        arena_size
    ];
    let mut pool_next: usize = 0;

    let mut lists: Vec<[i32; 2]> = vec![[0i32; 2]; maxbits];

    let node0 = pool_next as i32;
    pool_next += 1;
    arena[node0 as usize] = Node {
        weight: leaves[0].weight,
        count: 1,
        tail: -1,
    };

    let node1 = pool_next as i32;
    pool_next += 1;
    arena[node1 as usize] = Node {
        weight: leaves[1].weight,
        count: 2,
        tail: -1,
    };

    lists.fill([node0, node1]);

    let num_runs = 2 * numsymbols as usize - 4;
    for _ in 0..num_runs - 1 {
        boundary_pm(
            &mut lists,
            &leaves,
            numsymbols,
            &mut arena,
            &mut pool_next,
            maxbits - 1,
        );
    }
    boundary_pm_final(
        &mut lists,
        &leaves,
        numsymbols,
        &mut arena,
        &mut pool_next,
        maxbits - 1,
    );

    extract_bit_lengths(lists[maxbits - 1][1], &arena, &leaves, bitlengths);

    Ok(())
}

fn init_node(weight: usize, count: i32, tail: i32, arena: &mut [Node], idx: usize) {
    arena[idx] = Node {
        weight,
        count,
        tail,
    };
}

fn boundary_pm(
    lists: &mut [[i32; 2]],
    leaves: &[Node],
    numsymbols: i32,
    arena: &mut [Node],
    pool_next: &mut usize,
    index: usize,
) {
    let lastcount = arena[lists[index][1] as usize].count;

    if index == 0 && lastcount >= numsymbols {
        return;
    }

    let newchain = *pool_next as i32;
    *pool_next += 1;
    let oldchain_idx = lists[index][1];

    lists[index][0] = oldchain_idx;
    lists[index][1] = newchain;

    if index == 0 {
        let lw = leaves[lastcount as usize].weight;
        init_node(lw, lastcount + 1, -1, arena, newchain as usize);
    } else {
        let sum =
            arena[lists[index - 1][0] as usize].weight + arena[lists[index - 1][1] as usize].weight;
        if lastcount < numsymbols && sum > leaves[lastcount as usize].weight {
            let lw = leaves[lastcount as usize].weight;
            let old_tail = arena[oldchain_idx as usize].tail;
            init_node(lw, lastcount + 1, old_tail, arena, newchain as usize);
        } else {
            let prev1 = lists[index - 1][1];
            init_node(sum, lastcount, prev1, arena, newchain as usize);
            boundary_pm(lists, leaves, numsymbols, arena, pool_next, index - 1);
            boundary_pm(lists, leaves, numsymbols, arena, pool_next, index - 1);
        }
    }
}

fn boundary_pm_final(
    lists: &mut [[i32; 2]],
    leaves: &[Node],
    numsymbols: i32,
    arena: &mut [Node],
    pool_next: &mut usize,
    index: usize,
) {
    let lastcount = arena[lists[index][1] as usize].count;
    let sum =
        arena[lists[index - 1][0] as usize].weight + arena[lists[index - 1][1] as usize].weight;

    if lastcount < numsymbols && sum > leaves[lastcount as usize].weight {
        let newchain_idx = *pool_next as i32;
        let oldchain_tail = arena[lists[index][1] as usize].tail;
        arena[newchain_idx as usize] = Node {
            weight: 0,
            count: lastcount + 1,
            tail: oldchain_tail,
        };
        lists[index][1] = newchain_idx;
    } else {
        let prev1 = lists[index - 1][1];
        arena[lists[index][1] as usize].tail = prev1;
    }
}

fn extract_bit_lengths(chain: i32, arena: &[Node], leaves: &[Node], bitlengths: &mut [u32]) {
    let mut counts = [0i32; 16];
    let mut end: usize = 16;
    let mut ptr: usize = 15;
    let mut value: u32 = 1;

    let mut node = chain;
    while node != -1 {
        end -= 1;
        counts[end] = arena[node as usize].count;
        node = arena[node as usize].tail;
    }

    let mut val = counts[15];
    while ptr >= end {
        let prev_count = if ptr > 0 { counts[ptr - 1] } else { 0 };
        while val > prev_count {
            bitlengths[leaves[(val - 1) as usize].count as usize] = value;
            val -= 1;
        }
        ptr = ptr.wrapping_sub(1);
        if ptr == usize::MAX {
            break;
        }
        value += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_huffman() {
        let freq = vec![1usize, 2, 4, 8];
        let mut bl = vec![0u32; 4];
        length_limited_code_lengths(&freq, 15, &mut bl).unwrap();
        for &b in &bl {
            assert!(b > 0);
        }
        let kraft: f64 = bl.iter().map(|&b| 2f64.powi(-(b as i32))).sum();
        assert!(kraft <= 1.0 + 1e-9, "Kraft inequality violated: {}", kraft);
    }

    #[test]
    fn single_symbol() {
        let freq = vec![0usize, 5, 0];
        let mut bl = vec![0u32; 3];
        length_limited_code_lengths(&freq, 15, &mut bl).unwrap();
        assert_eq!(bl[1], 1);
        assert_eq!(bl[0], 0);
        assert_eq!(bl[2], 0);
    }

    #[test]
    fn two_symbols() {
        let freq = vec![3usize, 0, 7];
        let mut bl = vec![0u32; 3];
        length_limited_code_lengths(&freq, 15, &mut bl).unwrap();
        assert_eq!(bl[0], 1);
        assert_eq!(bl[2], 1);
    }

    #[test]
    fn all_zero_frequencies() {
        let freq = vec![0usize; 10];
        let mut bl = vec![0u32; 10];
        length_limited_code_lengths(&freq, 15, &mut bl).unwrap();
        assert!(bl.iter().all(|&b| b == 0));
    }

    #[test]
    fn respects_maxbits() {
        let freq: Vec<usize> = (1..=32).collect();
        let mut bl = vec![0u32; 32];
        length_limited_code_lengths(&freq, 7, &mut bl).unwrap();
        for &b in &bl {
            assert!(b <= 7, "bitlength {} exceeds maxbits 7", b);
        }
    }
}
