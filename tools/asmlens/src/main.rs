//! asmlens — machine-code analysis lens for the inflate hot loop.
//!
//! Disassembles a symbol from an ELF binary (x86_64 OR aarch64, auto-detected),
//! reconstructs basic blocks, detects loops via back-edges, overlays
//! per-instruction DYNAMIC weight from `perf script -F ip` samples, maps
//! instructions to source lines via DWARF, and diffs two symbols (e.g. our
//! `decode_huffman_libdeflate_style` vs libdeflate's `deflate_decompress_bmi2`,
//! both linked into examples/inner_bench).
//!
//! Multi-arch on purpose: same tool for the x86 bench box and the arm64 (M4)
//! native Rust-decoder builds we'll optimize later.
//!
//! CAVEAT: the per-source-line DWARF attribution is UNRELIABLE for our Rust at
//! -O3 (heavy inlining destroys the line table — expect line-0 buckets and
//! occasional misattribution to unrelated inlined sites). Trust the static
//! structure, opcode histogram, loop detection, and per-FUNCTION dynamic totals;
//! treat per-line as a hint only. (libdeflate's C line table is clean.)
//!
//! Usage:
//!   asmlens <binary> --sym <substr> [--sym <substr2>] [--perf <perf-ip-file>]
//!
//! perf samples:  perf record -e instructions:u -c 100000 -- <bin> ...
//!                perf script -F ip > samples.txt
use anyhow::{bail, Context, Result};
use capstone::prelude::*;
use capstone::{Capstone, InsnGroupType};
use object::{Object, ObjectSection, ObjectSymbol};
use std::collections::{BTreeMap, BTreeSet, HashSet};

struct Insn {
    ip: u64,
    mnem: String,
    is_branch: bool,
    is_call: bool,
    target: Option<u64>, // intra-function near-branch target
}

struct SymInfo {
    name: String,
    addr: u64,
    size: u64,
}

fn find_symbol(obj: &object::File, query: &str) -> Result<SymInfo> {
    let mut best: Option<SymInfo> = None;
    for s in obj.symbols() {
        let Ok(n) = s.name() else { continue };
        if s.size() > 0 && n.contains(query) {
            if best.as_ref().map_or(true, |b| n.len() < b.name.len()) {
                best = Some(SymInfo { name: n.to_string(), addr: s.address(), size: s.size() });
            }
        }
    }
    best.with_context(|| format!("symbol matching {query:?} not found"))
}

fn code_bytes<'a>(obj: &'a object::File, addr: u64, size: u64) -> Result<&'a [u8]> {
    for sec in obj.sections() {
        let start = sec.address();
        if addr >= start && addr + size <= start + sec.size() {
            let data = sec.data()?;
            let off = (addr - start) as usize;
            return Ok(&data[off..off + size as usize]);
        }
    }
    bail!("no section contains [{addr:#x}, {:#x})", addr + size)
}

fn build_capstone(obj: &object::File) -> Result<Capstone> {
    let cs = match obj.architecture() {
        object::Architecture::X86_64 => Capstone::new()
            .x86()
            .mode(arch::x86::ArchMode::Mode64)
            .detail(true)
            .build(),
        object::Architecture::Aarch64 => Capstone::new()
            .arm64()
            .mode(arch::arm64::ArchMode::Arm)
            .detail(true)
            .build(),
        a => bail!("unsupported architecture {a:?}"),
    }
    .map_err(|e| anyhow::anyhow!("capstone build: {e}"))?;
    Ok(cs)
}

fn disasm(cs: &Capstone, bytes: &[u8], base: u64, end: u64) -> Result<Vec<Insn>> {
    let insns = cs.disasm_all(bytes, base).map_err(|e| anyhow::anyhow!("disasm: {e}"))?;
    let mut out = Vec::new();
    for i in insns.iter() {
        let detail = cs.insn_detail(i).map_err(|e| anyhow::anyhow!("detail: {e}"))?;
        let groups: Vec<u32> = detail.groups().iter().map(|g| g.0 as u32).collect();
        let is_branch = groups.contains(&(InsnGroupType::CS_GRP_JUMP))
            || groups.contains(&(InsnGroupType::CS_GRP_BRANCH_RELATIVE));
        let is_call = groups.contains(&(InsnGroupType::CS_GRP_CALL));
        // Direct branch target: op_str is a single hex address ("0x...").
        let target = if is_branch {
            let op = i.op_str().unwrap_or("").trim();
            let hex = op.trim_start_matches("0x");
            u64::from_str_radix(hex, 16)
                .ok()
                .filter(|t| *t >= base && *t < end)
        } else {
            None
        };
        out.push(Insn {
            ip: i.address(),
            mnem: i.mnemonic().unwrap_or("?").to_string(),
            is_branch,
            is_call,
            target,
        });
    }
    Ok(out)
}

/// IPs inside a loop: any back-edge (target <= branch ip) marks [target, ip].
fn loop_ips(insns: &[Insn]) -> HashSet<u64> {
    let mut in_loop = HashSet::new();
    for ins in insns {
        if let Some(t) = ins.target {
            if t <= ins.ip {
                for x in insns.iter().filter(|i| i.ip >= t && i.ip <= ins.ip) {
                    in_loop.insert(x.ip);
                }
            }
        }
    }
    in_loop
}

/// Parse `perf script -F ip,sym,symoff` — lines are `<runtime_ip> <sym>+0x<off>`.
/// ASLR-proof: we key by (symbol, offset-within-symbol), NOT the relocated IP.
/// Returns sym-name -> (offset -> sample count).
fn parse_perf_symoff(path: &str) -> Result<BTreeMap<String, BTreeMap<u64, u64>>> {
    let txt = std::fs::read_to_string(path).with_context(|| format!("read perf {path}"))?;
    let mut out: BTreeMap<String, BTreeMap<u64, u64>> = BTreeMap::new();
    for line in txt.lines() {
        // drop the leading runtime ip, keep `<sym>+0x<off>`
        let rest = line.trim().splitn(2, char::is_whitespace).nth(1).unwrap_or("").trim();
        let Some((sym, off_s)) = rest.rsplit_once("+0x") else { continue };
        let Ok(off) = u64::from_str_radix(off_s.trim(), 16) else { continue };
        *out.entry(sym.to_string()).or_default().entry(off).or_insert(0) += 1;
    }
    Ok(out)
}

struct Report {
    sym: SymInfo,
    n_static: usize,
    n_loop: usize,
    branches: usize,
    calls: usize,
    opcodes: BTreeMap<String, usize>,
    dyn_total: u64,
    dyn_loop: u64,
    src_lines: BTreeMap<String, u64>,
}

fn analyze(
    obj: &object::File,
    cs: &Capstone,
    query: &str,
    perf: Option<&BTreeMap<String, BTreeMap<u64, u64>>>,
    a2l: Option<&addr2line::Loader>,
) -> Result<Report> {
    let sym = find_symbol(obj, query)?;
    let sym_addr = sym.addr;
    let end = sym.addr + sym.size;
    let insns = disasm(cs, code_bytes(obj, sym.addr, sym.size)?, sym.addr, end)?;
    let in_loop = loop_ips(&insns);

    // Merge perf offset->count for every perf symbol whose (demangled) name
    // contains the query. Keyed by offset within the symbol = ASLR-proof.
    let offcount: BTreeMap<u64, u64> = match perf {
        Some(p) => {
            let mut m = BTreeMap::new();
            for (s, offs) in p {
                if s.contains(query) {
                    for (o, c) in offs { *m.entry(*o).or_insert(0) += c; }
                }
            }
            m
        }
        None => BTreeMap::new(),
    };

    let mut opcodes = BTreeMap::new();
    let (mut branches, mut calls) = (0, 0);
    let (mut dyn_total, mut dyn_loop) = (0u64, 0u64);
    let mut src_lines: BTreeMap<String, u64> = BTreeMap::new();

    for ins in &insns {
        *opcodes.entry(ins.mnem.clone()).or_insert(0) += 1;
        if ins.is_branch { branches += 1; }
        if ins.is_call { calls += 1; }
        if perf.is_some() {
            let w = offcount.get(&(ins.ip - sym_addr)).copied().unwrap_or(0);
            dyn_total += w;
            if in_loop.contains(&ins.ip) { dyn_loop += w; }
            if w > 0 {
                if let Some(l) = a2l {
                    if let Ok(Some(loc)) = l.find_location(ins.ip) {
                        let key = format!(
                            "{}:{}",
                            loc.file.unwrap_or("?").rsplit('/').next().unwrap_or("?"),
                            loc.line.unwrap_or(0)
                        );
                        *src_lines.entry(key).or_insert(0) += w;
                    }
                }
            }
        }
    }
    Ok(Report { sym, n_static: insns.len(), n_loop: in_loop.len(), branches, calls, opcodes, dyn_total, dyn_loop, src_lines })
}

fn print_report(r: &Report, perf: bool) {
    println!("──────────────────────────────────────────────────────────────");
    println!("  {}", r.sym.name);
    println!("  addr={:#x} size={}  static_instrs={}  loop_body={}  branches={}  calls={}",
        r.sym.addr, r.sym.size, r.n_static, r.n_loop, r.branches, r.calls);
    if perf {
        let pct = if r.dyn_total > 0 { 100.0 * r.dyn_loop as f64 / r.dyn_total as f64 } else { 0.0 };
        println!("  dynamic samples: total={} in_loop={} ({:.1}% in hot loop)", r.dyn_total, r.dyn_loop, pct);
        if !r.src_lines.is_empty() {
            println!("  top source lines by dynamic instruction weight:");
            let mut v: Vec<_> = r.src_lines.iter().collect();
            v.sort_by(|a, b| b.1.cmp(a.1));
            for (line, w) in v.iter().take(12) {
                println!("    {:>8} ({:>4.1}%)  {}", w, 100.0 * **w as f64 / r.dyn_total.max(1) as f64, line);
            }
        }
    }
    let mut ops: Vec<_> = r.opcodes.iter().collect();
    ops.sort_by(|a, b| b.1.cmp(a.1));
    print!("  top opcodes:");
    for (op, n) in ops.iter().take(10) { print!(" {}={}", op, n); }
    println!();
}

fn print_diff(a: &Report, b: &Report) {
    println!("\n════════════════════ DIFF (A / B) ════════════════════");
    println!("  static instrs : {:>6} / {:>6}  ({:.2}x)", a.n_static, b.n_static, a.n_static as f64 / b.n_static.max(1) as f64);
    println!("  loop-body     : {:>6} / {:>6}", a.n_loop, b.n_loop);
    println!("  branches      : {:>6} / {:>6}  ({:.2}x)", a.branches, b.branches, a.branches as f64 / b.branches.max(1) as f64);
    if a.dyn_total > 0 && b.dyn_total > 0 {
        println!("  dynamic instrs: {:>9} / {:>9}  ({:.2}x)  [perf samples ∝ retired instrs]",
            a.dyn_total, b.dyn_total, a.dyn_total as f64 / b.dyn_total.max(1) as f64);
    }
    println!("  opcode deltas (A − B, top movers):");
    let keys: BTreeSet<&String> = a.opcodes.keys().chain(b.opcodes.keys()).collect();
    let mut deltas: Vec<(String, i64, usize, usize)> = keys.iter().map(|k| {
        let av = *a.opcodes.get(*k).unwrap_or(&0);
        let bv = *b.opcodes.get(*k).unwrap_or(&0);
        ((*k).clone(), av as i64 - bv as i64, av, bv)
    }).collect();
    deltas.sort_by(|x, y| y.1.abs().cmp(&x.1.abs()));
    for (op, d, av, bv) in deltas.iter().take(14) {
        println!("    {:<10} {:+6}   (A {:>4}  B {:>4})", op, d, av, bv);
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: asmlens <binary> --sym <substr> [--sym <substr2>] [--perf <perf-ip-file>]");
        eprintln!("note: per-line source attribution is unreliable at -O3 for Rust (inlining);");
        eprintln!("      trust static/opcode/per-function-dynamic; treat per-line as a hint.");
        std::process::exit(2);
    }
    let bin = &args[0];
    let mut syms = Vec::new();
    let mut perf_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sym" => { syms.push(args.get(i + 1).context("--sym needs a value")?.clone()); i += 2; }
            "--perf" => { perf_path = Some(args.get(i + 1).context("--perf needs a value")?.clone()); i += 2; }
            other => bail!("unknown arg {other}"),
        }
    }
    if syms.is_empty() { bail!("need at least one --sym"); }

    let data = std::fs::read(bin).with_context(|| format!("read {bin}"))?;
    let obj = object::File::parse(&*data)?;
    let cs = build_capstone(&obj)?;
    let perf = perf_path.as_deref().map(parse_perf_symoff).transpose()?;
    let a2l = addr2line::Loader::new(bin).ok();
    if a2l.is_none() { eprintln!("(note: no DWARF — build with debuginfo for source lines)"); }

    let mut reports = Vec::new();
    for q in &syms {
        let r = analyze(&obj, &cs, q, perf.as_ref(), a2l.as_ref())?;
        print_report(&r, perf.is_some());
        reports.push(r);
    }
    if reports.len() == 2 { print_diff(&reports[0], &reports[1]); }
    Ok(())
}
