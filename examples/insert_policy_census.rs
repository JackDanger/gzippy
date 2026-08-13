//! insert_policy_census — price an INTERIOR-INSERT POLICY on real token streams.
//!
//! The blocked question (PR #320's parked verdict): dense interior insertion
//! closes the L1 reach cliff (3.4467 -> 0.9950 vs libdeflate) but its bill is
//! ~0.72 Ir per extra insert and lands on files that have no slack
//! (`pigz:data.json:L1:T1:wall` 0.70 -> 1.05). The proposed cheaper mechanism is
//! LENGTH-KEYED density: index the whole interior only of matches at least
//! `L` bytes long, keep igzip's 8-position prefix elsewhere.
//!
//! Whether that is cheap is a property of the CONTENT, and it is countable
//! before anything is built: tokenize what the shipped encoder actually emits
//! and count, per policy, how many interior head-table writes it would perform.
//!
//!   cargo run --release --example insert_policy_census -- \
//!       [--level N] [--tsv OUT] <target ...>
//!
//! target := path | fixture:NAME | holdout:NAME | surface:ID
//!
//! Columns: `bytes`, `lit_rate` (literals per input byte — the (b) candidate's
//! cost base), `mlen_mean`, then for each policy the interior writes and the
//! EXTRA writes over `main`'s cap-8, per input byte. `cov>=L` is the fraction
//! of input bytes lying inside an accepted match of length >= L: the payoff
//! reach of a length-keyed policy, and its cost, are both that number.
use gzippy::compress::ldx_oracle::tokenize;

const THRESHOLDS: [u16; 5] = [16, 32, 64, 128, 258];

struct Row {
    name: String,
    bytes: u64,
    literals: u64,
    matches: u64,
    matched_bytes: u64,
    /// Interior writes under main's cap-8 policy.
    cap8: u64,
    /// Interior writes under full density.
    dense: u64,
    /// Interior writes under length-keyed density, per THRESHOLDS entry.
    keyed: [u64; THRESHOLDS.len()],
    /// Input bytes inside a match of length >= THRESHOLDS[i].
    cov: [u64; THRESHOLDS.len()],
}

fn load(target: &str) -> (String, Vec<u8>) {
    if let Some(n) = target.strip_prefix("fixture:") {
        (format!("fixture:{n}"), gzippy::fixtures::generate(n))
    } else if let Some(n) = target.strip_prefix("holdout:") {
        (format!("holdout:{n}"), gzippy::holdout::generate(n))
    } else if let Some(id) = target.strip_prefix("surface:") {
        let p = gzippy::fixtures::surface_points()
            .into_iter()
            .find(|p| p.id() == id)
            .unwrap_or_else(|| panic!("no surface point {id}"));
        (
            format!("surface:{id}"),
            gzippy::fixtures::surface_generate(&p, 1 << 20),
        )
    } else {
        let name = std::path::Path::new(target)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| target.to_string());
        (name, std::fs::read(target).expect("read target"))
    }
}

fn census(name: String, input: &[u8], level: u32) -> Row {
    let ours = gzippy::compress::deflate::encode_deflate_bytes_to_vec(input, level);
    let (tokens, _) = tokenize(&ours).expect("tokenize our own stream");
    let mut r = Row {
        name,
        bytes: input.len() as u64,
        literals: 0,
        matches: 0,
        matched_bytes: 0,
        cap8: 0,
        dense: 0,
        keyed: [0; THRESHOLDS.len()],
        cov: [0; THRESHOLDS.len()],
    };
    for t in &tokens {
        if t.is_literal() {
            r.literals += 1;
            continue;
        }
        r.matches += 1;
        let len = t.len as u64;
        r.matched_bytes += len;
        // The shipped loop indexes positions `pos+1 .. pos+length` (capped);
        // that is `length - 1` candidates, of which main takes the first 8.
        let interior = len - 1;
        r.cap8 += interior.min(8);
        r.dense += interior;
        for (i, &thr) in THRESHOLDS.iter().enumerate() {
            if t.len >= thr {
                r.keyed[i] += interior;
                r.cov[i] += len;
            } else {
                r.keyed[i] += interior.min(8);
            }
        }
    }
    r
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut level = 1u32;
    let mut tsv: Option<String> = None;
    let mut targets: Vec<String> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--level" => level = args.next().expect("--level N").parse().expect("level"),
            "--tsv" => tsv = Some(args.next().expect("--tsv PATH")),
            other => targets.push(other.to_string()),
        }
    }
    if targets.is_empty() {
        eprintln!("usage: insert_policy_census [--level N] [--tsv OUT] <target ...>");
        std::process::exit(2);
    }

    let rows: Vec<Row> = targets
        .iter()
        .map(|t| {
            let (name, data) = load(t);
            census(name, &data, level)
        })
        .collect();

    let mut out = String::new();
    out.push_str("name\tlevel\tbytes\tlit_rate\tmlen_mean\tcap8_pb\tdense_pb\textra_dense_pb");
    for thr in THRESHOLDS {
        out.push_str(&format!("\textra_k{thr}_pb\tcov_ge{thr}"));
    }
    out.push('\n');
    for r in &rows {
        let b = r.bytes as f64;
        let mlen = if r.matches > 0 {
            r.matched_bytes as f64 / r.matches as f64
        } else {
            0.0
        };
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.4}\t{:.1}\t{:.4}\t{:.4}\t{:.4}",
            r.name,
            level,
            r.bytes,
            r.literals as f64 / b,
            mlen,
            r.cap8 as f64 / b,
            r.dense as f64 / b,
            (r.dense - r.cap8) as f64 / b,
        ));
        for i in 0..THRESHOLDS.len() {
            out.push_str(&format!(
                "\t{:.4}\t{:.4}",
                (r.keyed[i] - r.cap8) as f64 / b,
                r.cov[i] as f64 / b
            ));
        }
        out.push('\n');
    }
    print!("{out}");
    if let Some(p) = tsv {
        std::fs::write(&p, &out).expect("write tsv");
        eprintln!("wrote {p}");
    }
}
