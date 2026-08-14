//! Hot-symbol SIZE canary: catch layout-lottery triggers before a box run.
//!
//! The campaign's costliest verdict-noise source is the layout lottery (see
//! memory: an inert refactor eroded ~90 wall cells 1.5-6% through code
//! placement alone). The lottery is TRIGGERED by code-size changes shifting
//! function placement — an accidental inlining/outlining flip in a hot
//! function is exactly such a trigger, and today it is invisible until a
//! frozen-box wall run moves for "no reason". This canary pins the machine-
//! code SIZE of ~10 load-bearing hot-path symbols and fails when one drifts
//! more than +/-20% — a laptop-visible tripwire for "the optimizer changed
//! its mind about this function", BEFORE any wall verdict is spent on it.
//!
//! NOT feature-gated: `cargo test --release --test symbol_canary` runs it
//! against the current release-profile source, no special features.
//!
//! ## Which binary gets measured, and why (MEASURED, 2026-08-04, this Mac)
//!
//! `[profile.release]` sets `strip = true`, so the SHIPPED binary
//! (`env!("CARGO_BIN_EXE_gzippy")`) has no symbol table for `nm` to read.
//! The obvious fix — an identical build with `strip = false` — was tested
//! and does NOT yield byte-identical text: `-C strip` feeds the compiler's
//! crate-id hashing, which reseeds fat-LTO layout decisions. Receipts:
//!
//! - Two `strip = false` builds through DIFFERENT profile names AND
//!   different target dirs: `__text` byte-IDENTICAL (1,164,884 B) — the
//!   build is deterministic and path-independent.
//! - The `strip = true` release build vs the `strip = false` twin: same
//!   `__text` length, same function COUNT (1,466 in `LC_FUNCTION_STARTS`),
//!   but ~10% of function boundaries shifted by small amounts (first at
//!   fn #44, 16 B) — slightly different per-function codegen, ~11% of
//!   bytes differing.
//!
//! So EXACT shipped-binary symbol sizes are unattainable from any
//! symbol-bearing build, and this canary instead grades the DETERMINISTIC
//! sibling: the `release-syms` profile (Cargo.toml: `inherits = "release"`,
//! `strip = false` — nothing ships from it). Same source, same flags,
//! same optimizer; per-function sizes differ from shipped by well under the
//! +/-20% band, and any source-level inlining flip (which moves a function
//! 2x or makes it vanish) moves both builds together. The pins therefore
//! catch exactly the event class they exist for.
//!
//! Freshness: when the graded test binary itself carries symbols (a
//! `cargo test --profile release-syms` run) it is graded directly. Otherwise
//! the test runs `cargo build --profile release-syms` — a no-op when the
//! twin is fresh (cargo is the staleness oracle), a real rebuild when it is
//! stale — PROVIDED the twin already exists. A checkout that never built it
//! (e.g. default CI) SKIPS with a printed one-time build command instead of
//! paying a surprise multi-minute LTO build inside a test.
//!
//! ## Method
//!
//! Per-symbol sizes are derived portably by sorting `nm -n` text-section
//! symbol addresses and taking successive deltas (the last symbol is bounded
//! by the text section's end, extracted by an in-test Mach-O/ELF parser — no
//! objdump/readelf dependency; verified on this Mac against both GNU and
//! llvm nm). Names are demangled by `nm`'s demangle flag when accepted, with
//! a light in-test legacy demangler as fallback; selection is substring-
//! based either way (Rust mangling embeds the plain identifier).
//!
//! Pins live in `tests/fingerprints/symbol_sizes.tsv`, KEYED BY TARGET
//! TRIPLE: sizes are toolchain- and arch-specific, so each box pins its own
//! rows. Triples with no pinned rows SKIP with a printed notice (not a
//! failure). Regenerate for the current triple (other triples' rows are
//! preserved):
//!
//! ```text
//! cargo build --profile release-syms
//! UPDATE_SYMBOL_SIZES=1 cargo test --release --test symbol_canary
//! ```
//!
//! The +/-20% band is a first guess, to be tuned with experience: tight
//! enough to catch an inlining flip, loose enough to ignore ordinary codegen
//! drift from small source edits (and the sub-2% strip-vs-no-strip jitter
//! measured above).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Relative drift tolerance per symbol. First-guess band — see module doc.
const TOLERANCE: f64 = 0.20;

/// The watched hot-path symbols, selected by substring. Each must match
/// EXACTLY ONE text symbol in the binary (verified at selection time and on
/// every run — a substring going 0-match means the function was inlined away
/// or renamed; going multi-match means an outlined copy appeared; both are
/// exactly the layout-relevant events this canary exists to catch, so both
/// FAIL rather than skip).
///
/// Coverage across the level ladder: the L0/L1 fast parser, the greedy
/// (L2-L3) and lazy (L4-L7) parser loops, the near-optimal (L8+) flush, the
/// lzfind matchfinder, the block driver + block emission + body-bit emitter,
/// the dynamic-header and huffman-code builders, and both monomorphizations
/// of the T>1 pipelined entry (stdout — the benchmark path — and file).
const SUBSTRINGS: &[&str] = &[
    // #286 wired bucket2/COST-GATE through `run_resumable` and the old
    // `parse::fast::run` symbol dissolved into the two monomorphized hot
    // loops below (the watch went 0-match, exactly the rename event the
    // canary exists to catch — caught 2026-08-08, adjusted here with the
    // pins regenerated in the same commit).
    // #320's T1-only REACH route monomorphizes `fastloop_l1<const REACH>` into
    // two symbols; pin each variant — a plain `fastloop_l1` substring also
    // hits `fastloop_l1_lean` and fails the duplicate-symbol guard.
    "fastloop_l1::<false>",
    "fastloop_l1::<true>",
    "fast::fastloop_l1_lean",
    "greedy::run_resumable",
    "lazy::run_resumable",
    "optimize_and_flush",
    "MatchFinder>::get_matches",
    "deflate_into",
    "emit_block",
    "emit_sequences",
    "build_dynamic_header",
    "make_huffman_code_into",
    "compress_buffer_pure::<std::io::buffered::bufwriter::BufWriter<std::io::stdio::Stdout>>",
    "compress_buffer_pure::<std::io::buffered::bufwriter::BufWriter<std::fs::File>>",
];

const REGEN_CMD: &str = "cargo build --profile release-syms && \
     UPDATE_SYMBOL_SIZES=1 cargo test --release --test symbol_canary";

fn pins_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fingerprints/symbol_sizes.tsv")
}

/// The target triple this test binary was compiled for, reconstructed from
/// `std::env::consts` (integration tests don't get the raw TARGET env).
/// First cut covers the fleet's actual boxes (darwin + linux-gnu); an
/// unlisted (arch, os) pair yields "<arch>-<os>", which simply never matches
/// a pinned triple and skips.
fn current_triple() -> String {
    let arch = std::env::consts::ARCH;
    match std::env::consts::OS {
        "macos" => format!("{arch}-apple-darwin"),
        "linux" => format!("{arch}-unknown-linux-gnu"),
        os => format!("{arch}-{os}"),
    }
}

// ─── Text-section extraction ─────────────────────────────────────────────────

fn u16le(b: &[u8], off: usize) -> u64 {
    u16::from_le_bytes(b[off..off + 2].try_into().unwrap()) as u64
}
fn u32le(b: &[u8], off: usize) -> u64 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as u64
}
fn u64le(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

/// The primary text section's (virtual address, size). Mach-O 64 and ELF 64,
/// little-endian — the two formats this project ships on. Used to clip the
/// nm symbol list to real code addresses and to bound the LAST symbol's
/// size, so a section boundary can never masquerade as a size. Panics on
/// anything else (a canary that cannot see cannot guard).
fn text_bounds(path: &Path) -> (u64, u64) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(b.len() >= 64, "{} too small to be a binary", path.display());
    match u32::from_le_bytes(b[0..4].try_into().unwrap()) {
        0xfeed_facf => {
            // Mach-O 64: 32-byte header, then ncmds load commands.
            let ncmds = u32le(&b, 16) as usize;
            let mut off = 32;
            for _ in 0..ncmds {
                let cmd = u32le(&b, off);
                let cmdsize = u32le(&b, off + 4) as usize;
                // LC_SEGMENT_64 (0x19): segname[16] @ +8, nsects u32 @ +64,
                // 80-byte section headers from +72.
                if cmd == 0x19 && b[off + 8..off + 24].starts_with(b"__TEXT\0") {
                    let nsects = u32le(&b, off + 64) as usize;
                    for s in 0..nsects {
                        let so = off + 72 + s * 80;
                        if b[so..so + 16].starts_with(b"__text\0") {
                            return (u64le(&b, so + 32), u64le(&b, so + 40));
                        }
                    }
                }
                off += cmdsize;
            }
            panic!("no __TEXT,__text section in {}", path.display());
        }
        0x464c_457f => {
            // ELF 64 LE ("\x7fELF", class 2, LE): find ".text" via shstrtab.
            assert!(
                b[4] == 2 && b[5] == 1,
                "{}: only 64-bit little-endian ELF is supported",
                path.display()
            );
            let shoff = u64le(&b, 0x28) as usize;
            let shentsize = u16le(&b, 0x3a) as usize;
            let shnum = u16le(&b, 0x3c) as usize;
            let shstrndx = u16le(&b, 0x3e) as usize;
            let stroff = u64le(&b, shoff + shstrndx * shentsize + 0x18) as usize;
            for i in 0..shnum {
                let sh = shoff + i * shentsize;
                let name_off = stroff + u32le(&b, sh) as usize;
                let name_end = b[name_off..].iter().position(|&c| c == 0).unwrap() + name_off;
                if &b[name_off..name_end] == b".text" {
                    return (u64le(&b, sh + 0x10), u64le(&b, sh + 0x20));
                }
            }
            panic!("no .text section in {}", path.display());
        }
        magic => panic!(
            "{}: unrecognized binary magic {magic:#010x} (expected Mach-O 64 or ELF 64 LE)",
            path.display()
        ),
    }
}

// ─── Symbol table reading ────────────────────────────────────────────────────

/// Light demangler for display: legacy Rust `_ZN<len><seg>...E` → `a::b::c`,
/// dropping the trailing hash segment. Anything else passes through.
/// (Matching is substring-based, so demangling is cosmetic for simple names —
/// but nm's own demangle flag is preferred because the pinned generic
/// substrings, e.g. `compress_buffer_pure::<...BufWriter<std::fs::File>>`,
/// only appear in fully demangled form.)
fn demangle_lite(name: &str) -> String {
    let inner = name
        .trim_start_matches('_')
        .strip_prefix("ZN")
        .map(|s| s.strip_suffix('E').unwrap_or(s));
    let Some(mut s) = inner else {
        return name.to_string();
    };
    let mut segs = Vec::new();
    while !s.is_empty() {
        let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
        let Ok(n) = digits.parse::<usize>() else {
            break;
        };
        let rest = &s[digits.len()..];
        if n == 0 || rest.len() < n {
            break;
        }
        segs.push(&rest[..n]);
        s = &rest[n..];
    }
    if segs.is_empty() {
        return name.to_string();
    }
    if segs.last().is_some_and(|l| l.starts_with("17h")) {
        segs.pop();
    }
    segs.join("::")
}

/// Run `nm -n` (numeric-sorted) on `bin`, preferring a demangling flag, and
/// parse `(addr, name)` rows for text symbols (type t/T). Tries PATH `nm`
/// first, then the platform's `/usr/bin/nm` — a PATH nm from a different
/// toolchain family (GNU binutils on a Mac) may or may not read the local
/// format, so "first invocation that yields plausible symbols" wins.
fn read_text_symbols(bin: &Path) -> Vec<(u64, String)> {
    let candidates: &[(&str, &[&str])] = &[
        ("nm", &["-n", "--demangle"]),
        ("nm", &["-n"]),
        ("/usr/bin/nm", &["-n", "--demangle"]),
        ("/usr/bin/nm", &["-n"]),
    ];
    for (tool, flags) in candidates {
        let Ok(out) = Command::new(tool).args(*flags).arg(bin).output() else {
            continue;
        };
        if !out.status.success() {
            continue;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let mut syms = Vec::new();
        for line in text.lines() {
            // "ADDR TYPE NAME" — undefined symbols have no ADDR column.
            let mut parts = line.splitn(3, ' ');
            let (Some(addr), Some(ty), Some(name)) = (parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            if !matches!(ty, "t" | "T") {
                continue;
            }
            let Ok(addr) = u64::from_str_radix(addr, 16) else {
                continue;
            };
            syms.push((addr, demangle_lite(name.trim())));
        }
        // A symbol-bearing gzippy binary has >1000 text symbols; a stripped
        // one has ~1. The threshold just rejects stripped/garbled output.
        if syms.len() >= 50 {
            return syms;
        }
    }
    Vec::new()
}

/// Portable per-symbol sizes: sort text-symbol addresses, successive deltas;
/// the final symbol is bounded by the text section's end.
fn symbol_sizes(bin: &Path, mut syms: Vec<(u64, String)>) -> Vec<(String, u64)> {
    let (text_addr, text_len) = text_bounds(bin);
    let text_end = text_addr + text_len;
    syms.retain(|(a, _)| *a >= text_addr && *a < text_end);
    syms.sort();
    let mut out = Vec::with_capacity(syms.len());
    for i in 0..syms.len() {
        let next = if i + 1 < syms.len() {
            syms[i + 1].0
        } else {
            text_end
        };
        out.push((syms[i].1.clone(), next - syms[i].0));
    }
    out
}

/// Resolve one watch substring to exactly one (symbol, size).
fn resolve<'a>(sizes: &'a [(String, u64)], substr: &str) -> Result<(&'a str, u64), String> {
    let hits: Vec<&(String, u64)> = sizes.iter().filter(|(n, _)| n.contains(substr)).collect();
    // Disambiguation for watches that are a strict prefix of a sibling symbol
    // (e.g. `fastloop_l1` vs `fastloop_l1_lean`, where NO plain substring can
    // name the shorter one uniquely): when the substring matches several
    // symbols but exactly ONE symbol *ends with* it, that suffix match is the
    // named function. A genuine outlined/duplicated copy still fails below,
    // because a copy of the same function matches the suffix too (`.._0`,
    // `..::hXX` variants do not end with the plain path).
    if hits.len() > 1 {
        let suffix_hits: Vec<&&(String, u64)> =
            hits.iter().filter(|(n, _)| n.ends_with(substr)).collect();
        if suffix_hits.len() == 1 {
            return Ok((suffix_hits[0].0.as_str(), suffix_hits[0].1));
        }
    }
    match hits.len() {
        1 => Ok((hits[0].0.as_str(), hits[0].1)),
        0 => Err(format!(
            "watch substring {substr:?} matches NO text symbol — the function was fully \
             inlined away or renamed (itself a layout-relevant event). If intended, adjust \
             SUBSTRINGS/pins and regenerate:\n    {REGEN_CMD}"
        )),
        n => Err(format!(
            "watch substring {substr:?} matches {n} symbols (an outlined/duplicated copy \
             appeared — itself a layout-relevant event):\n{}\nIf intended, tighten the \
             substring or regenerate:\n    {REGEN_CMD}",
            hits.iter()
                .map(|(name, sz)| format!("    {sz:>8} B  {name}"))
                .collect::<Vec<_>>()
                .join("\n")
        )),
    }
}

// ─── The test ────────────────────────────────────────────────────────────────

#[test]
fn hot_symbol_sizes_within_band() {
    let shipped = PathBuf::from(env!("CARGO_BIN_EXE_gzippy"));

    // Pick the graded binary: the test-provided binary itself ONLY when it
    // was built by the release-syms profile (a `cargo test --profile
    // release-syms` run) — keyed on the profile dir NAME, not on symbol
    // presence, because other unstripped profiles (quicktest, bench) carry
    // symbols too but with LTO off their sizes are a different codebase's.
    // Otherwise the release-syms twin — refreshed through cargo (a no-op
    // when fresh) when it exists, skipped with the one-time build command
    // when it does not (so a fresh checkout / default CI is not
    // surprise-charged a full LTO build inside a test). See module doc for
    // why the twin, and why its sizes are trustworthy despite `strip`
    // reseeding exact layout.
    let syms: Vec<(u64, String)>;
    let graded: PathBuf;
    if shipped.parent().and_then(Path::file_name) == Some("release-syms".as_ref()) {
        syms = read_text_symbols(&shipped);
        assert!(
            !syms.is_empty(),
            "symbol_canary: {} was built by the release-syms profile but has no readable \
             text symbols — is `nm` installed?",
            shipped.display()
        );
        graded = shipped;
    } else {
        let target_dir = shipped
            .parent()
            .and_then(Path::parent)
            .expect("CARGO_BIN_EXE has a target dir");
        let twin = target_dir
            .join("release-syms")
            .join(shipped.file_name().unwrap());
        if !twin.exists() {
            eprintln!(
                "symbol_canary: SKIP — the release binary is stripped (release profile sets \
                 strip = true) and no release-syms twin exists at {}.\nBuild it once (a few \
                 minutes, cached afterwards), then rerun:\n    cargo build --profile \
                 release-syms && cargo test --release --test symbol_canary",
                twin.display()
            );
            return;
        }
        // The twin exists: let cargo decide whether it is stale (fingerprint
        // check; sub-second no-op when fresh, a real rebuild when the source
        // changed — grading a stale twin would attribute old sizes to new
        // code, which is exactly the lie this canary must not tell).
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
        let status = Command::new(&cargo)
            .args(["build", "--profile", "release-syms"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .status()
            .unwrap_or_else(|e| panic!("spawn `{cargo} build --profile release-syms`: {e}"));
        assert!(
            status.success(),
            "`cargo build --profile release-syms` failed — cannot refresh the symbol source"
        );
        syms = read_text_symbols(&twin);
        assert!(
            !syms.is_empty(),
            "symbol_canary: release-syms twin at {} has no readable text symbols — is `nm` \
             installed, and does the release-syms profile still set strip = false?",
            twin.display()
        );
        graded = twin;
    }
    let sizes = symbol_sizes(&graded, syms);
    let triple = current_triple();

    // ── Regeneration: rewrite this triple's rows, preserve the others. ────
    if std::env::var("UPDATE_SYMBOL_SIZES").as_deref() == Ok("1") {
        let mut rows: Vec<String> = Vec::new();
        if let Ok(existing) = std::fs::read_to_string(pins_path()) {
            for line in existing.lines() {
                if line.starts_with('#') || line.starts_with("triple\t") || line.trim().is_empty() {
                    continue;
                }
                if !line.starts_with(&format!("{triple}\t")) {
                    rows.push(line.to_string());
                }
            }
        }
        for substr in SUBSTRINGS {
            let (name, size) = resolve(&sizes, substr).unwrap_or_else(|e| panic!("{e}"));
            rows.push(format!("{triple}\t{substr}\t{size}\t{name}"));
        }
        rows.sort();
        let header = "\
# symbol_sizes.tsv — machine-code sizes of load-bearing hot-path symbols,\n\
# pinned per target triple. Guarded by tests/symbol_canary.rs (default\n\
# features): each symbol's current size must be within +/-20% of its pin —\n\
# a laptop-visible tripwire for accidental inlining/outlining flips, which\n\
# trigger the layout lottery that erodes wall cells on the frozen boxes.\n\
# The +/-20% band is a first guess; tune it with experience.\n\
#\n\
# Sizes are successive `nm -n` address deltas over the release-syms build\n\
# (profile `release-syms` = release + strip=false; deterministic — two\n\
# builds across different target dirs measured byte-identical). The shipped\n\
# stripped binary's exact layout differs slightly (-C strip reseeds fat-LTO\n\
# layout; measured same function count, ~10% of boundaries shifted by tens\n\
# of bytes), so these pins track the deterministic sibling — an inlining\n\
# flip moves both together. See tests/symbol_canary.rs module doc.\n\
#\n\
# Rows exist only for triples pinned on their own box; other triples SKIP.\n\
# x86_64-unknown-linux-gnu rows should be generated on the trainer box with\n\
# the same command. Regenerate for the current triple (others preserved):\n\
#   cargo build --profile release-syms\n\
#   UPDATE_SYMBOL_SIZES=1 cargo test --release --test symbol_canary\n\
triple\tsubstring\tsize\tsymbol\n";
        std::fs::write(pins_path(), format!("{header}{}\n", rows.join("\n")))
            .expect("write symbol_sizes.tsv");
        eprintln!(
            "symbol_canary: regenerated {} for {triple} ({} symbols)",
            pins_path().display(),
            SUBSTRINGS.len()
        );
        return;
    }

    // ── Compare against pins for this triple. ─────────────────────────────
    let tsv = std::fs::read_to_string(pins_path()).unwrap_or_else(|e| {
        panic!(
            "{} missing ({e}) — generate it once:\n    {REGEN_CMD}",
            pins_path().display()
        )
    });
    let mut pinned: BTreeMap<String, u64> = BTreeMap::new();
    for line in tsv.lines() {
        if line.starts_with('#') || line.starts_with("triple\t") || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert!(cols.len() >= 3, "malformed pin row: {line:?}");
        if cols[0] == triple {
            pinned.insert(cols[1].to_string(), cols[2].parse().unwrap());
        }
    }
    if pinned.is_empty() {
        eprintln!(
            "symbol_canary: SKIP — no pinned rows for target triple {triple} in {}.\nGenerate \
             them on this box:\n    {REGEN_CMD}",
            pins_path().display()
        );
        return;
    }

    let mut failures = Vec::new();
    for (substr, &old) in &pinned {
        match resolve(&sizes, substr) {
            Err(e) => failures.push(e),
            Ok((name, new)) => {
                let drift = (new as f64 - old as f64) / old as f64;
                if drift.abs() > TOLERANCE {
                    failures.push(format!(
                        "symbol {substr} {} {:+.1}% — layout-shift risk; if intended, \
                         regenerate with UPDATE_SYMBOL_SIZES=1 (pinned {old} B, now {new} B; \
                         band +/-{:.0}%)\n    full symbol: {name}",
                        if drift > 0.0 { "grew" } else { "shrank" },
                        drift * 100.0,
                        TOLERANCE * 100.0
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "\nHOT-SYMBOL SIZE CANARY [{triple}] — {} symbol(s) out of band:\n\n{}\n\n\
         A hot function's machine-code size moved past the +/-{:.0}% band. This is the\n\
         trigger class for the layout lottery (inert-looking changes eroding wall cells\n\
         on the frozen boxes). If the size change is INTENTIONAL, regenerate the pins in\n\
         this PR:\n    {REGEN_CMD}\n",
        failures.len(),
        failures.join("\n\n"),
        TOLERANCE * 100.0
    );
}
