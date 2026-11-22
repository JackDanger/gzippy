//! VALID stream: x86_64 engine-A load-before-refill preload (byte identity).
//
// On x86_64 the flat fastloop (`decode_huffman_fastloop_bounded_pipelined`)
// uses a 12-bit litlen / 9-bit dist main table. Its refill-before-offset guard
// was `bitsleft < 32`, so a 9-bit offset codeword with 13 extra bits could
// leave bitsleft == 10 at the next-litlen PRELOAD; when the previous refill
// had left bitsleft == 63 only 11 valid bits remained and bit 11 of the
// 12-bit index was a shifted-in zero. A following 12-bit literal then decoded
// as the wrong symbol: valid stream, wrong bytes, CRC error. This stream is
// built so that every match is lit11 lit11 len284 dist29 lit12 (the exact
// trigger) with an LCG varying the bit phase. Receipt: the same bytes (crc32
// and length pinned below) were produced by the audit's craft script and
// decoded WRONG by an x86_64-apple-darwin release build of 52210553 ("CRC32
// mismatch", exit 1) while gzip, libdeflate-gzip and aarch64 gzippy agree.
// It only exercises the bug on x86_64 targets that run the pure-Rust
// fastloop (not the Linux asm kernel), but it is a valid-stream byte-identity
// check everywhere.

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

#[derive(Default)]
struct BitW {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}
impl BitW {
    fn bits(&mut self, v: u64, n: u32) {
        self.acc |= v << self.n;
        self.n += n;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    /// Huffman codes go MSB-first.
    fn code(&mut self, (c, n): (u32, u32)) {
        let r = c.reverse_bits() >> (32 - n);
        self.bits(r as u64, n);
    }
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

fn canonical(lengths: &[u32]) -> Vec<(u32, u32)> {
    let maxl = *lengths.iter().max().unwrap() as usize;
    let mut bl = vec![0u32; maxl + 1];
    for &l in lengths {
        if l > 0 {
            bl[l as usize] += 1;
        }
    }
    let mut next = vec![0u32; maxl + 1];
    let mut code = 0u32;
    for b in 1..=maxl {
        code = (code + if b > 1 { bl[b - 1] } else { 0 }) << 1;
        next[b] = code;
    }
    lengths
        .iter()
        .map(|&l| {
            if l == 0 {
                return (0, 0);
            }
            let c = next[l as usize];
            next[l as usize] += 1;
            (c, l)
        })
        .collect()
}

/// Returns (gzip member, expected payload).
fn preload_trigger_stream() -> (Vec<u8>, Vec<u8>) {
    let mut r = Lcg(1);
    let mut ll = vec![0u32; 286];
    for (c, l) in ll.iter_mut().enumerate().take(256) {
        *l = match c {
            0..=63 => 12,
            64..=127 => 11,
            128..=135 => 8,
            _ => 9,
        };
    }
    ll[256] = 3;
    ll[257] = 2;
    ll[265] = 3;
    ll[284] = 4;
    ll[285] = 3;
    let mut dl = vec![0u32; 30];
    for (s, l) in [0usize, 1, 2, 3, 10, 11, 12, 13].into_iter().zip(1u32..=8) {
        dl[s] = l;
    }
    dl[28] = 9;
    dl[29] = 9;
    let mut pre = vec![0u32; 19];
    for s in [0usize, 18] {
        pre[s] = 3;
    }
    for s in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 17] {
        pre[s] = 4;
    }
    let (llc, dlc, prc) = (canonical(&ll), canonical(&dl), canonical(&pre));
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut w = BitW::default();
    w.bits(1, 1); // BFINAL
    w.bits(2, 2); // dynamic
    w.bits(29, 5); // HLIT = 286
    w.bits(29, 5); // HDIST = 30
    w.bits(15, 4); // HCLEN = 19
    for s in ORDER {
        w.bits(pre[s] as u64, 3);
    }
    let seq: Vec<u32> = ll.iter().chain(dl.iter()).copied().collect();
    let mut i = 0;
    while i < seq.len() {
        if seq[i] == 0 {
            let mut j = i;
            while j < seq.len() && seq[j] == 0 && j - i < 138 {
                j += 1;
            }
            let run = j - i;
            if run >= 11 {
                w.code(prc[18]);
                w.bits((run - 11) as u64, 7);
                i = j;
                continue;
            }
            if run >= 3 {
                w.code(prc[17]);
                w.bits((run - 3) as u64, 3);
                i = j;
                continue;
            }
        }
        w.code(prc[seq[i] as usize]);
        i += 1;
    }
    let mut out: Vec<u8> = Vec::new();
    let lit = |w: &mut BitW, out: &mut Vec<u8>, c: u64| {
        w.code(llc[c as usize]);
        out.push(c as u8);
    };
    for _ in 0..33000 {
        let c = 128 + r.next() % 128;
        lit(&mut w, &mut out, c);
    }
    for _ in 0..4000 {
        for _ in 0..r.next() % 4 {
            let c = 136 + r.next() % 120;
            lit(&mut w, &mut out, c);
        }
        let c = 64 + r.next() % 64;
        lit(&mut w, &mut out, c);
        let c = 64 + r.next() % 64;
        lit(&mut w, &mut out, c);
        let ev = r.next() % 31;
        w.code(llc[284]);
        w.bits(ev, 5);
        let dv = r.next() % 8192;
        let dist = 24577 + dv as usize;
        w.code(dlc[29]);
        w.bits(dv, 13);
        for _ in 0..(227 + ev) {
            let b = out[out.len() - dist];
            out.push(b);
        }
        let c = r.next() % 64;
        lit(&mut w, &mut out, c);
    }
    w.code(llc[256]);
    let raw = w.finish();
    let mut gz = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
    gz.extend_from_slice(&raw);
    gz.extend_from_slice(&crc32fast::hash(&out).to_le_bytes());
    gz.extend_from_slice(&(out.len() as u32).to_le_bytes());
    (gz, out)
}

#[test]
fn x86_preload_bitsleft_10_stream_decodes_exactly() {
    let (gz, want) = preload_trigger_stream();
    // Pin the bytes to the receipt stream (craft_preload2.py output).
    assert_eq!((gz.len(), crc32fast::hash(&gz)), (76425, 0x54d4_ce3c));
    assert_eq!((want.len(), crc32fast::hash(&want)), (1018463, 0x5fa9_f89e));
    // Independent decoder agrees the stream is valid.
    let mut via_flate2 = Vec::new();
    std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(&gz[..]), &mut via_flate2)
        .unwrap();
    assert!(
        via_flate2 == want,
        "flate2 disagrees: crafted stream invalid"
    );
    for t in [1usize, 4] {
        let got = gzippy::decompress_with_threads(&gz, t).expect("valid stream rejected");
        assert!(got == want, "-p{t}: valid stream decoded to wrong bytes");
    }
}
