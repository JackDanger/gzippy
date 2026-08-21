//! The NEVER-TUNE holdout corpus, generated from seeds.
//!
//! WHY. The campaign tunes on an 11-file TUNE set and promotes on a GATE set,
//! but both were chosen from the same 22 files, so neither can answer "have we
//! overfit to the board?". This module defines a third population: ~12 content
//! classes whose archive TYPES are absent from the tuning corpus (tar-of-source,
//! JSONL logs, protobuf-like TLV, wide CSV, VM-image mix, Chinese prose, XML
//! feed, FASTA DNA, MIME/base64, Apache logs, pointer heaps, Markdown). Every
//! byte is produced by seeded integer arithmetic — identical on every platform,
//! every run, forever — so the set is reproducible anywhere, never stored
//! in-repo as data, and definitionally never used for tuning.
//!
//! THE CONTRACT (the whole point of the instrument):
//!   * No parameter, threshold, or level map may ever be fitted against these
//!     files. They are graded by `scripts/campaign/holdout.sh` and compared to
//!     the tuning-board win-rate; a materially lower holdout win-rate is the
//!     overfit alarm.
//!   * The generator bytes are pinned by sha256 in [`PINS`] below, and
//!     `examples/holdout_gen.rs` REFUSES to materialize a member whose bytes do
//!     not match. A generator change is therefore a conscious repin, and it
//!     voids every previously recorded win-rate.
//!
//! See docs/generalization-instruments.md for the protocol.

/// Holdout member names. The extension names the archive type each imitates.
pub const NAMES: &[&str] = &[
    "src.tar",      // USTAR tarball of synthetic C-like source files
    "events.jsonl", // JSONL logs, different schema/statistics than data.json
    "proto.tlv",    // protobuf-like varint TLV binary with nested messages
    "wide.csv",     // CSV, 24 columns, different column statistics than data.csv
    "vm.img",       // page-granular mix: zeros / code / config / pointers / noise
    "cjk.txt",      // UTF-8 Chinese prose, different statistics than aozora
    "feed.xml",     // attribute-heavy nested XML feed
    "dna.fasta",    // 4-letter-alphabet genomic text with N runs
    "mail.mime",    // MIME mail with base64 attachment bodies
    "apache.log",   // Apache combined access log
    "heap.bin",     // 8-byte-aligned pointer-like u64 dump
    "repo.md",      // Markdown with code fences and tables
];

/// Uniform holdout member size: 4 MiB — several DEFLATE blocks at every level
/// and many T>1 chunks at -p4, while the whole 12-file grade stays laptop-fast.
pub const LEN: usize = 4 << 20;

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const WORDS: &[&str] = &[
    "buffer", "offset", "length", "stream", "packet", "handle", "window", "symbol", "table",
    "index", "cursor", "result", "status", "config", "worker", "signal", "socket", "header",
    "record", "column", "parser", "output", "input", "block", "chunk", "queue", "cache", "frame",
    "token", "field", "value", "count", "state", "flags", "error", "retry", "batch", "shard",
];

fn ident(rng: &mut XorShift) -> String {
    let a = WORDS[rng.below(WORDS.len() as u64) as usize];
    let b = WORDS[rng.below(WORDS.len() as u64) as usize];
    format!("{a}_{b}")
}

/// Synthetic C-like source text, shared by the tarball and Markdown members.
fn c_source(rng: &mut XorShift, target: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(target + 256);
    out.extend_from_slice(b"#include <stdint.h>\n#include <string.h>\n\n");
    while out.len() < target {
        let fname = ident(rng);
        let p1 = ident(rng);
        let p2 = ident(rng);
        out.extend_from_slice(
            format!("static int {fname}(const uint8_t *{p1}, size_t {p2}) {{\n").as_bytes(),
        );
        let stmts = 3 + rng.below(9);
        for _ in 0..stmts {
            let v = ident(rng);
            match rng.below(5) {
                0 => out.extend_from_slice(
                    format!("    uint32_t {v} = {};\n", rng.below(65536)).as_bytes(),
                ),
                1 => out.extend_from_slice(
                    format!("    if ({p2} < {}) return -1;\n", rng.below(4096)).as_bytes(),
                ),
                2 => out.extend_from_slice(
                    format!(
                        "    {v} = ({v} << {}) ^ {p1}[{}];\n",
                        rng.below(13),
                        rng.below(64)
                    )
                    .as_bytes(),
                ),
                3 => out.extend_from_slice(
                    format!("    /* {} {} */\n", ident(rng), ident(rng)).as_bytes(),
                ),
                _ => out.extend_from_slice(
                    format!("    memcpy(&{v}, {p1} + {}, sizeof {v});\n", rng.below(256))
                        .as_bytes(),
                ),
            }
        }
        out.extend_from_slice(
            format!("    return (int)({p2} & 0x{:x});\n}}\n\n", rng.below(255)).as_bytes(),
        );
    }
    out.truncate(target);
    out
}

/// One USTAR header + padded data for `content` at `path`.
fn tar_entry(out: &mut Vec<u8>, path: &str, content: &[u8]) {
    let mut h = [0u8; 512];
    h[..path.len().min(100)].copy_from_slice(&path.as_bytes()[..path.len().min(100)]);
    h[100..107].copy_from_slice(b"0000644");
    h[108..115].copy_from_slice(b"0000000");
    h[116..123].copy_from_slice(b"0000000");
    let size = format!("{:011o}", content.len());
    h[124..135].copy_from_slice(size.as_bytes());
    h[136..147].copy_from_slice(b"14400000000"); // fixed mtime
    h[148..156].copy_from_slice(b"        "); // checksum field = spaces while summing
    h[156] = b'0'; // regular file
    h[257..262].copy_from_slice(b"ustar");
    h[263..265].copy_from_slice(b"00");
    h[265..269].copy_from_slice(b"root");
    h[297..301].copy_from_slice(b"root");
    h[329..336].copy_from_slice(b"0000000");
    h[337..344].copy_from_slice(b"0000000");
    let sum: u32 = h.iter().map(|&b| b as u32).sum();
    let chk = format!("{sum:06o}\0 ");
    h[148..156].copy_from_slice(chk.as_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(content);
    let pad = (512 - content.len() % 512) % 512;
    out.resize(out.len() + pad, 0);
}

/// A protobuf field tag: `field << 3 | wire_type` (0 varint, 1 fixed64,
/// 2 length-delimited, 5 fixed32). A helper rather than inline shifts so the
/// wire type stays readable at every call site.
fn tag(field: u64, wire: u64) -> u64 {
    field << 3 | wire
}

fn varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_line(rng: &mut XorShift, out: &mut Vec<u8>) {
    // 76 output chars = 57 input bytes, the MIME line width.
    for _ in 0..19 {
        let v = rng.next() & 0xff_ffff;
        out.push(B64[(v >> 18 & 63) as usize]);
        out.push(B64[(v >> 12 & 63) as usize]);
        out.push(B64[(v >> 6 & 63) as usize]);
        out.push(B64[(v & 63) as usize]);
    }
    out.push(b'\n');
}

/// Generate a holdout member by name at the standard [`LEN`]. Panics on an
/// unknown name (test-support code; a typo should fail loudly).
pub fn generate(name: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(LEN + 4096);
    match name {
        "src.tar" => {
            let mut rng = XorShift(0x686f6c64_74617201);
            let mut fileno = 0u32;
            // Leave room for the two zero end-of-archive blocks tar writes.
            while out.len() + 8192 < LEN {
                let dir = WORDS[rng.below(WORDS.len() as u64) as usize];
                let flen = 1500 + rng.below(6000) as usize;
                let content = c_source(&mut rng, flen);
                tar_entry(
                    &mut out,
                    &format!("proj/src/{dir}/{}_{fileno}.c", ident(&mut rng)),
                    &content,
                );
                fileno += 1;
            }
            out.resize(LEN, 0); // end-of-archive zero fill
        }
        "events.jsonl" => {
            const LEVELS: &[&str] = &["debug", "info", "info", "info", "warn", "error"];
            const SVCS: &[&str] = &[
                "ingest-gw",
                "auth-svc",
                "billing-core",
                "search-idx",
                "notif-fanout",
            ];
            let mut rng = XorShift(0x686f6c64_6a736f02);
            let mut ts = 1_770_000_000_000u64; // ms epoch
            while out.len() < LEN {
                ts += rng.below(900);
                let lvl = LEVELS[rng.below(LEVELS.len() as u64) as usize];
                let svc = SVCS[rng.below(SVCS.len() as u64) as usize];
                let trace = rng.next();
                let span = rng.next() as u32;
                let dur = rng.below(90_000) as f64 / 100.0;
                let code = [200u64, 200, 200, 201, 204, 400, 404, 429, 500][rng.below(9) as usize];
                out.extend_from_slice(
                    format!(
                        "{{\"ts\":{ts},\"level\":\"{lvl}\",\"service\":\"{svc}\",\"trace_id\":\"{trace:016x}\",\"span_id\":\"{span:08x}\",\"http\":{{\"status\":{code},\"dur_ms\":{dur:.2}}},\"attrs\":{{\"{}\":{},\"{}\":\"{}\"}}}}\n",
                        ident(&mut rng), rng.below(1_000_000),
                        ident(&mut rng), ident(&mut rng),
                    )
                    .as_bytes(),
                );
            }
        }
        "proto.tlv" => {
            let mut rng = XorShift(0x686f6c64_746c7603);
            while out.len() < LEN {
                // message: several fields, one length-delimited nested message
                varint(&mut out, tag(1, 0)); // field 1 varint
                varint(&mut out, rng.below(1 << 32));
                varint(&mut out, tag(2, 0));
                varint(&mut out, rng.below(64));
                let s = ident(&mut rng);
                varint(&mut out, tag(3, 2)); // field 3 bytes
                varint(&mut out, s.len() as u64);
                out.extend_from_slice(s.as_bytes());
                let mut nested = Vec::with_capacity(64);
                varint(&mut nested, tag(1, 1)); // fixed64
                nested.extend_from_slice(&rng.next().to_le_bytes());
                varint(&mut nested, tag(2, 5)); // fixed32
                nested.extend_from_slice(&(rng.next() as u32).to_le_bytes());
                let reps = rng.below(6);
                for _ in 0..reps {
                    varint(&mut nested, tag(3, 0));
                    varint(&mut nested, rng.below(1 << 20));
                }
                varint(&mut out, tag(4, 2)); // field 4 nested
                varint(&mut out, nested.len() as u64);
                out.extend_from_slice(&nested);
            }
        }
        "wide.csv" => {
            let mut rng = XorShift(0x686f6c64_63737604);
            out.extend_from_slice(b"run_id,epoch,lr,loss,acc,f1,auc,grad_norm,batch,gpu,host,tag,dur_s,mem_gb,tokens,skip,seed,phase,opt,sched,clip,warm,decay,note\n");
            let mut run = 7000u64;
            while out.len() < LEN {
                run += rng.below(3);
                let loss = rng.below(9_000_000) as f64 / 1e6;
                let acc = rng.below(1_000_000) as f64 / 1e6;
                let phase = ["train", "eval", "", "train"][rng.below(4) as usize];
                let note = if rng.below(11) == 0 {
                    format!(
                        "\"spike, {} at step {}\"",
                        ident(&mut rng),
                        rng.below(100_000)
                    )
                } else {
                    String::new()
                };
                out.extend_from_slice(
                    format!(
                        "{run},{},{:.1e},{loss:.6},{acc:.6},{:.4},{:.4},{:.3e},{},gpu{},node-{:03},{},{}.{:03},{:.2},{},{},{},{phase},adamw,cosine,{:.1},{},{:.0e},{note}\n",
                        rng.below(300),
                        (1 + rng.below(9000)) as f64 * 1e-7,
                        rng.below(10_000) as f64 / 1e4,
                        rng.below(10_000) as f64 / 1e4,
                        rng.below(1_000_000) as f64 / 1e2,
                        32 << rng.below(4),
                        rng.below(8),
                        rng.below(64),
                        ident(&mut rng),
                        rng.below(4000),
                        rng.below(1000),
                        4.0 + rng.below(6000) as f64 / 100.0,
                        rng.below(2_000_000_000),
                        rng.below(2),
                        rng.below(100_000),
                        rng.below(50) as f64 / 10.0,
                        rng.below(10_000),
                        (1 + rng.below(99)) as f64 * 1e-6,
                    )
                    .as_bytes(),
                );
            }
        }
        "vm.img" => {
            let mut rng = XorShift(0x686f6c64_766d6905);
            while out.len() < LEN {
                let page_kind = rng.below(100);
                let start = out.len();
                if page_kind < 30 {
                    out.resize(start + 4096, 0); // zero page
                } else if page_kind < 55 {
                    // code-ish page: opcode-biased bytes with rel32 fields
                    while out.len() < start + 4096 {
                        match rng.below(10) {
                            0..=3 => {
                                out.push([0x48, 0x8b, 0x89, 0xe8, 0x0f][rng.below(5) as usize]);
                                out.push(rng.next() as u8 & 0x3f);
                            }
                            4..=6 => {
                                out.push(0xe8); // call rel32
                                out.extend_from_slice(
                                    &((rng.below(1 << 16) as i32 - 32768).to_le_bytes()),
                                );
                            }
                            7 => out.extend_from_slice(&[0x55, 0x48, 0x89, 0xe5]), // prologue
                            8 => out.extend_from_slice(&[0x5d, 0xc3]),             // epilogue
                            _ => out.push(0x90),
                        }
                    }
                } else if page_kind < 70 {
                    // ascii config fragment page
                    while out.len() < start + 4096 {
                        out.extend_from_slice(
                            format!(
                                "{}.{}={}\n",
                                ident(&mut rng),
                                ident(&mut rng),
                                rng.below(100_000)
                            )
                            .as_bytes(),
                        );
                    }
                } else if page_kind < 85 {
                    // pointer-table page
                    let base = 0xffff_8000_0000_0000u64 | (rng.below(1 << 30) << 12);
                    while out.len() < start + 4096 {
                        out.extend_from_slice(&(base + (rng.below(512) << 12)).to_le_bytes());
                    }
                } else {
                    while out.len() < start + 4096 {
                        out.extend_from_slice(&rng.next().to_le_bytes()); // noise page
                    }
                }
                out.truncate(start + 4096);
            }
        }
        "cjk.txt" => {
            let mut rng = XorShift(0x686f6c64_636a6b06);
            let mut sentence_len = 0u32;
            while out.len() < LEN {
                // Zipf-ish skew over common-CJK codepoints: square the draw so
                // low offsets (a small "common character" set) dominate, giving
                // different symbol statistics than aozora's Japanese mix.
                let r = rng.below(1 << 20) as f64 / (1 << 20) as f64;
                let cp = 0x4e00 + (r * r * 3200.0) as u32;
                let mut buf = [0u8; 4];
                out.extend_from_slice(char::from_u32(cp).unwrap().encode_utf8(&mut buf).as_bytes());
                sentence_len += 1;
                let d = rng.below(100);
                if d < 4 && sentence_len > 6 {
                    out.extend_from_slice("。".as_bytes());
                    sentence_len = 0;
                    if rng.below(5) == 0 {
                        out.extend_from_slice("\n\n".as_bytes());
                    }
                } else if d < 10 {
                    out.extend_from_slice("，".as_bytes());
                } else if d < 11 {
                    out.extend_from_slice("、".as_bytes());
                }
            }
        }
        "feed.xml" => {
            let mut rng = XorShift(0x686f6c64_786d6c07);
            out.extend_from_slice(b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<feed xmlns=\"http://example.invalid/ns/feed\">\n");
            let mut id = 40_000u64;
            while out.len() < LEN {
                id += 1;
                let t1 = ident(&mut rng);
                let t2 = ident(&mut rng);
                out.extend_from_slice(
                    format!(
                        "  <entry id=\"e{id}\" rev=\"{}\" lang=\"en\">\n    <title>{t1} {t2}</title>\n    <updated ts=\"{}\"/>\n    <author uid=\"u{:05}\" role=\"{}\"/>\n",
                        rng.below(30),
                        1_770_000_000 + rng.below(10_000_000),
                        rng.below(90_000),
                        ["editor", "bot", "member"][rng.below(3) as usize],
                    )
                    .as_bytes(),
                );
                let paras = 1 + rng.below(3);
                for _ in 0..paras {
                    out.extend_from_slice(b"    <p>");
                    let words = 8 + rng.below(30);
                    for w in 0..words {
                        if w > 0 {
                            out.push(b' ');
                        }
                        out.extend_from_slice(
                            WORDS[rng.below(WORDS.len() as u64) as usize].as_bytes(),
                        );
                    }
                    out.extend_from_slice(b"</p>\n");
                }
                out.extend_from_slice(
                    format!(
                        "    <link href=\"/{}/{}/{id}\" type=\"text/html\"/>\n  </entry>\n",
                        ident(&mut rng),
                        ident(&mut rng)
                    )
                    .as_bytes(),
                );
            }
        }
        "dna.fasta" => {
            let mut rng = XorShift(0x686f6c64_646e6108);
            let mut contig = 0u32;
            while out.len() < LEN {
                contig += 1;
                out.extend_from_slice(
                    format!(
                        ">contig_{contig:04} synthetic holdout assembly len={}\n",
                        20_000 + rng.below(40_000)
                    )
                    .as_bytes(),
                );
                let lines = 200 + rng.below(500);
                for _ in 0..lines {
                    for _ in 0..70 {
                        // biased base mix, occasional N run
                        let b = match rng.below(100) {
                            0..=31 => b'A',
                            32..=55 => b'T',
                            56..=77 => b'G',
                            78..=97 => b'C',
                            _ => b'N',
                        };
                        out.push(b);
                    }
                    out.push(b'\n');
                    if out.len() >= LEN {
                        break;
                    }
                }
            }
        }
        "mail.mime" => {
            let mut rng = XorShift(0x686f6c64_6d696d09);
            let mut msg = 0u32;
            while out.len() < LEN {
                msg += 1;
                out.extend_from_slice(
                    format!(
                        "From: {}@example.invalid\nTo: {}@example.invalid\nSubject: {} {} report {msg}\nMIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=\"=_b{:016x}\"\n\n--=_b{:016x}\nContent-Type: text/plain; charset=us-ascii\n\n",
                        ident(&mut rng), ident(&mut rng), ident(&mut rng), ident(&mut rng),
                        rng.next(), rng.0,
                    )
                    .as_bytes(),
                );
                let words = 40 + rng.below(120);
                for w in 0..words {
                    if w > 0 {
                        out.push(if rng.below(12) == 0 { b'\n' } else { b' ' });
                    }
                    out.extend_from_slice(WORDS[rng.below(WORDS.len() as u64) as usize].as_bytes());
                }
                out.extend_from_slice(
                    format!(
                        "\n\n--=_b{:016x}\nContent-Type: application/octet-stream\nContent-Transfer-Encoding: base64\n\n",
                        rng.0
                    )
                    .as_bytes(),
                );
                let b64_lines = 60 + rng.below(220);
                for _ in 0..b64_lines {
                    base64_line(&mut rng, &mut out);
                }
                out.extend_from_slice(format!("--=_b{:016x}--\n\n", rng.0).as_bytes());
            }
        }
        "apache.log" => {
            const PATHS: &[&str] = &[
                "/api/v2/items",
                "/api/v2/users",
                "/static/app.js",
                "/static/main.css",
                "/health",
                "/api/v2/search",
                "/favicon.ico",
                "/api/v2/orders",
            ];
            const UAS: &[&str] = &[
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "curl/8.4.0",
                "python-requests/2.31.0",
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15",
            ];
            let mut rng = XorShift(0x686f6c64_6c6f670a);
            let mut sec = 0u64;
            while out.len() < LEN {
                sec += rng.below(3);
                let day = 1 + sec / 86_400;
                let h = sec / 3600 % 24;
                let m = sec / 60 % 60;
                let s = sec % 60;
                let meth = ["GET", "GET", "GET", "POST", "PUT"][rng.below(5) as usize];
                let path = PATHS[rng.below(PATHS.len() as u64) as usize];
                let q = if rng.below(3) == 0 {
                    format!("?page={}&limit={}", rng.below(400), 25 << rng.below(3))
                } else {
                    String::new()
                };
                let code = [200u64, 200, 200, 200, 301, 304, 404, 500][rng.below(8) as usize];
                out.extend_from_slice(
                    format!(
                        "10.{}.{}.{} - - [{:02}/Mar/2026:{h:02}:{m:02}:{s:02} +0000] \"{meth} {path}{q} HTTP/1.1\" {code} {} \"-\" \"{}\"\n",
                        rng.below(256), rng.below(256), rng.below(256), day,
                        rng.below(600_000),
                        UAS[rng.below(UAS.len() as u64) as usize],
                    )
                    .as_bytes(),
                );
            }
        }
        "heap.bin" => {
            let mut rng = XorShift(0x686f6c64_68700b0b);
            let heap_base = 0x0000_7f3a_c000_0000u64;
            while out.len() < LEN {
                match rng.below(10) {
                    // pointer into a shared heap region: high bytes identical
                    0..=5 => out.extend_from_slice(
                        &(heap_base + (rng.below(1 << 24) & !0xf)).to_le_bytes(),
                    ),
                    6 => out.extend_from_slice(&0u64.to_le_bytes()),
                    7 => out.extend_from_slice(&rng.below(4096).to_le_bytes()), // small int
                    8 => out.extend_from_slice(&(rng.below(1 << 16) | 1).to_le_bytes()), // tagged
                    _ => out.extend_from_slice(
                        // vtable-like: separate shared region
                        &(0x0000_5610_0000_0000u64 + (rng.below(1 << 16) << 4)).to_le_bytes(),
                    ),
                }
            }
        }
        "repo.md" => {
            let mut rng = XorShift(0x686f6c64_6d640c0c);
            while out.len() < LEN {
                out.extend_from_slice(
                    format!("## {} {}\n\n", ident(&mut rng), ident(&mut rng)).as_bytes(),
                );
                let words = 30 + rng.below(90);
                for w in 0..words {
                    if w > 0 {
                        out.push(b' ');
                    }
                    if rng.below(14) == 0 {
                        out.extend_from_slice(format!("`{}`", ident(&mut rng)).as_bytes());
                    } else {
                        out.extend_from_slice(
                            WORDS[rng.below(WORDS.len() as u64) as usize].as_bytes(),
                        );
                    }
                }
                out.extend_from_slice(b"\n\n```c\n");
                let src_len = 400 + rng.below(1200) as usize;
                let src = c_source(&mut rng, src_len);
                out.extend_from_slice(&src);
                out.extend_from_slice(b"\n```\n\n| name | default | notes |\n|---|---|---|\n");
                let rows = 2 + rng.below(6);
                for _ in 0..rows {
                    out.extend_from_slice(
                        format!(
                            "| `{}` | {} | {} {} |\n",
                            ident(&mut rng),
                            rng.below(4096),
                            ident(&mut rng),
                            ident(&mut rng)
                        )
                        .as_bytes(),
                    );
                }
                out.push(b'\n');
            }
        }
        other => panic!("unknown holdout member '{other}' (declared: {NAMES:?})"),
    }
    out.truncate(LEN);
    out
}

// ---------------------------------------------------------------------------
// sha256 — a compact, dependency-free implementation for pinning generator
// bytes. Test/instrument support only; nothing on any hot path uses it. Emits
// the same hex `shasum -a 256` does, so a pin can be re-derived on any box with
// a one-line shell command, without cargo.
// ---------------------------------------------------------------------------

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// sha256 of `data` as lowercase hex (identical to `shasum -a 256`).
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bitlen = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    let mut w = [0u32; 64];
    #[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)] // clippy 1.98; see bitstream.rs
    #[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)] // clippy 1.98
    for chunk in msg.chunks_exact(64) {
        for (i, word) in w.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

/// The pinned sha256 of every holdout member — the SINGLE source of truth.
/// `examples/holdout_gen.rs` verifies every member against this table at
/// materialization time and REFUSES on mismatch, so `scripts/campaign/holdout.sh`
/// cannot grade drifted bytes even when no test has run. A generator change must
/// repin here, consciously, and every previously recorded win-rate is void.
pub const PINS: &[(&str, &str)] = &[
    (
        "src.tar",
        "81d0b1da89124306402cbc6f0e5a0fea287ef5cca958a418fba7fe231941422d",
    ),
    (
        "events.jsonl",
        "17bab6eca48960f0da59323b21a2c8d89ca49184a9ef6d4bc6f6d9c16ea7038f",
    ),
    (
        "proto.tlv",
        "17170bb52b7fdebf046c57a65db8a29be9e0c67a4c38b9f77a465e41bfea5b18",
    ),
    (
        "wide.csv",
        "f94e4a06c1d4e3b9d8f2e3225ab22924811205fb1c6ae0e89ce106aa2e8eb756",
    ),
    (
        "vm.img",
        "9e13dec396dfc8a9b91937ef1adf91ca12fc4fc997d5e45962df8dd367b64e77",
    ),
    (
        "cjk.txt",
        "b4a1763592246dac231354d49e13a0bfae627a8ed66c84e431d5c39a69e58f2e",
    ),
    (
        "feed.xml",
        "690b880fa7700576d4c576398c5f718ae65de7322baff697c9eb31676ef809f9",
    ),
    (
        "dna.fasta",
        "517d2179c86e10adff3ff34ea945b444025e3395ff2af3ca67cef676a9588ac0",
    ),
    (
        "mail.mime",
        "ddf5201093ed4e4a2354d0e72b2a2c0d9b3b69e1056edc3b42b12cdbfdc0746b",
    ),
    (
        "apache.log",
        "763e3f62e1dd685ffb4908cacb931143f2a2b17262995184460a2fc00b49b04f",
    ),
    (
        "heap.bin",
        "135d765f078264957248ef5ecf281514a55e15f0baa37fae1995070870cb519a",
    ),
    (
        "repo.md",
        "2d9da539993ee833824a83d8d104f342d377a6c971004e86cd81be46e835702f",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// sha256 self-check against a known vector, so a bug in the local
    /// implementation cannot silently re-pin every generator.
    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// The holdout is only a holdout if its bytes never drift: every recorded
    /// win-rate and the whole "never tuned on" contract assume these exact
    /// bytes. A generator change MUST be a conscious act that repins [`PINS`]
    /// — and it voids every holdout number taken before it.
    #[test]
    fn holdout_members_are_frozen() {
        assert_eq!(PINS.len(), NAMES.len());
        for &name in NAMES {
            let data = generate(name);
            assert_eq!(data.len(), LEN, "{name}");
            let got = sha256_hex(&data);
            let want = PINS
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, s)| *s)
                .unwrap();
            if want.is_empty() {
                eprintln!("holdout {name}: sha256={got} (pin this)");
            } else {
                assert_eq!(
                    got, want,
                    "holdout member '{name}' bytes changed — every recorded holdout \
                     win-rate is now incomparable; repin PINS consciously"
                );
            }
        }
    }
}
