fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d = std::fs::read(&a[1]).unwrap();
    let l: u32 = a[2].parse().unwrap();
    let n: usize = a[3].parse().unwrap();
    let mut acc = 0usize;
    for _ in 0..n {
        acc += gzippy::compress::ldx::compress_for_diff(l, &d)
            .unwrap()
            .len();
    }
    println!("{acc}");
}
