fn main() {
    let c = "/Users/jackdanger/www/gzippy-bench/corpus";
    let mut n = 0;
    for e in std::fs::read_dir(c).unwrap() {
        let p = e.unwrap().path();
        let d = std::fs::read(&p).unwrap();
        for l in 0..=9u32 {
            if gzippy::compress::ldx::compress_for_diff(l, &d).is_some() {
                n += 1;
            }
        }
    }
    println!("ldx compressions completed with debug_asserts active: {n}");
}
