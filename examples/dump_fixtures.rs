//! Materialize the frozen synthetic fixtures to a directory (receipt tooling).
fn main() {
    let dir = std::env::args().nth(1).expect("usage: dump_fixtures <dir>");
    for &name in gzippy::fixtures::NAMES {
        let data = gzippy::fixtures::generate(name);
        std::fs::write(format!("{dir}/{name}"), &data).unwrap();
    }
}
