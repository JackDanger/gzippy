use std::time::Instant;
fn best<F: FnMut() -> usize>(mut f: F, n: usize) -> (f64, usize) {
    let mut b = f64::MAX; let mut s = 0;
    for _ in 0..n { let t = Instant::now(); s = f(); let e = t.elapsed().as_secs_f64(); if e < b { b = e; } }
    (b, s)
}
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let d = std::fs::read(&path).unwrap();
    let name = std::path::Path::new(&path).file_name().unwrap().to_string_lossy().to_string();
    for l in [1u32, 2, 4, 6, 9] {
        let mut buf = Vec::with_capacity(d.len() + d.len()/16 + 1024);
        let (t_ours, s_ours) = best(|| { buf.clear();
            gzippy::compress::ldx::compress_into(l, &d, &mut buf); buf.len() }, 7);
        let (t_c, s_c) = best(|| {
            let mut c = libdeflater::Compressor::new(libdeflater::CompressionLvl::new(l as i32).unwrap());
            let mut o = vec![0u8; c.deflate_compress_bound(d.len())];
            c.deflate_compress(&d, &mut o).unwrap()
        }, 7);
        println!("{name:<14} L{l:<2} ldx={:>8.2}ms C={:>8.2}ms  ldx/C={:>5.3}x  bytes {}",
                 t_ours*1000.0, t_c*1000.0, t_ours/t_c,
                 if s_ours == s_c { "IDENTICAL".to_string() } else { format!("{s_ours} vs {s_c}") });
    }
}
