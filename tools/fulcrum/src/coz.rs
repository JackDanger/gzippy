//! Coz layer: run the looped decode harness under `coz run` and parse the
//! resulting `profile.coz` into per-region wall-elasticity curves.
//!
//! profile.coz format (from the coz viewer parser, plasma-umass/coz
//! viewer/ts/profile.ts): newline-delimited records. A record is either a
//! JSON object (line starts with `{`) or TAB-delimited `type` then
//! `key=value` fields. The records we use:
//!   - `experiment`     selected=<file:line> speedup=<0..1> duration=<ns>
//!                      selected-samples=<n>
//!   - `throughput-point` (a.k.a. `progress-point`) name=<id> delta=<n>
//!                      duration=<ns>      ← emitted WITHIN each experiment
//!   - `latency-point`  name=<id> arrivals=<n> departures=<n> difference=<n>
//!
//! Coz's own analysis (which we reproduce): group experiments by
//! `selected`; at each `speedup` level accumulate the throughput point's
//! (delta,duration); throughput PERIOD = duration/delta (time per visit,
//! lower=faster); baseline = period at speedup 0; for each speedup level,
//!   program_speedup = (baseline_period − period) / baseline_period.
//! That program_speedup-vs-line-speedup curve IS ∂wall/∂speed for the
//! selected line — Coz's central output. We map each `selected` file:line
//! to one of the four FULCRUM regions by source-line-range membership, and
//! aggregate, so the report speaks in regions, not raw lines.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

/// One (speedup-level → accumulated throughput) bucket for a `selected`
/// line, across all experiments + all process runs that selected it.
#[derive(Default, Clone)]
struct ThroughputAccum {
    delta: f64,
    duration: f64,
}

impl ThroughputAccum {
    /// Throughput PERIOD: time per progress-point visit. Lower is faster.
    /// `None` when no visits were observed (delta==0) — an unusable point.
    fn period(&self) -> Option<f64> {
        if self.delta > 0.0 {
            Some(self.duration / self.delta)
        } else {
            None
        }
    }
}

/// Per-`selected`-line speedup curve: speedup-level (×100 = %) → program
/// speedup fraction.
#[derive(Default, Clone)]
pub struct LineCurve {
    pub selected: String,
    /// speedup_level (0.0..1.0) → program_speedup fraction (−1..2).
    pub points: BTreeMap<u64, f64>,
    pub total_samples: f64,
}

impl LineCurve {
    /// The headline elasticity: program speedup at the LARGEST line speedup
    /// level measured. This is the practical "if this line were as fast as
    /// possible, the program gets X% faster" number Coz plots.
    pub fn max_elasticity(&self) -> Option<(f64, f64)> {
        self.points
            .iter()
            .next_back()
            .map(|(lvl, ps)| (*lvl as f64 / 1000.0, *ps))
    }

    /// Slope near zero: average program_speedup / line_speedup over the
    /// measured non-zero levels. A robust scalar elasticity (∂wall/∂speed).
    pub fn slope(&self) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for (lvl, ps) in &self.points {
            let ls = *lvl as f64 / 1000.0;
            if ls > 0.0 {
                num += ps / ls; // each level's local slope
                den += 1.0;
            }
        }
        if den > 0.0 {
            num / den
        } else {
            0.0
        }
    }
}

/// A source-line range identifying one FULCRUM region in a file. coz
/// `selected` = `file:line`; we match by basename + line ∈ [lo,hi].
#[derive(Clone)]
pub struct RegionMap {
    pub region: String,
    pub file_basename: String,
    pub lo: u32,
    pub hi: u32,
}

/// Default region→source-line map for the 328696e tree. These bound the
/// hot functions behind each FULCRUM scope so coz line-experiments land in
/// the right bucket. Ranges are generous (whole function bodies); overlap
/// is impossible because each function lives in a disjoint span.
pub fn default_region_maps() -> Vec<RegionMap> {
    vec![
        // bootstrap_with_deflate_block[_inner] in gzip_chunk.rs
        RegionMap {
            region: "bootstrap".into(),
            file_basename: "gzip_chunk.rs".into(),
            lo: 1351,
            hi: 1700,
        },
        // absorb_isal_tail in gzip_chunk.rs
        RegionMap {
            region: "absorb".into(),
            file_basename: "gzip_chunk.rs".into(),
            lo: 966,
            hi: 1015,
        },
        // decode_chunk_isal_impl / pure_bulk_impl bulk loop in gzip_chunk.rs
        RegionMap {
            region: "bulk_inflate".into(),
            file_basename: "gzip_chunk.rs".into(),
            lo: 221,
            hi: 820,
        },
        // isal_lut_bulk::decode_block — the bulk decode driver.
        RegionMap {
            region: "bulk_inflate".into(),
            file_basename: "isal_lut_bulk.rs".into(),
            lo: 1,
            hi: 100000,
        },
        // isal_huffman_pure.rs — the pure-Rust ISA-L LUT inner Huffman
        // decode loop. THIS is where the bulk-inflate hot lines land
        // (coz sampled isal_huffman_pure.rs:419/468/473) — without this
        // map those samples would be unattributed and bulk_inflate would
        // be under-counted.
        RegionMap {
            region: "bulk_inflate".into(),
            file_basename: "isal_huffman_pure.rs".into(),
            lo: 1,
            hi: 100000,
        },
        // window-absent deflate inner loop
        RegionMap {
            region: "bootstrap".into(),
            file_basename: "deflate_block.rs".into(),
            lo: 1,
            hi: 100000,
        },
        // speculation scan
        RegionMap {
            region: "scan".into(),
            file_basename: "chunk_fetcher.rs".into(),
            lo: 2120,
            hi: 2240,
        },
        RegionMap {
            region: "scan".into(),
            file_basename: "raw_block_finder.rs".into(),
            lo: 1,
            hi: 100000,
        },
        RegionMap {
            region: "scan".into(),
            file_basename: "block_finder.rs".into(),
            lo: 1,
            hi: 100000,
        },
    ]
}

fn classify(selected: &str, maps: &[RegionMap]) -> Option<String> {
    let (file, line) = selected.rsplit_once(':')?;
    let base = Path::new(file)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(file);
    let line: u32 = line.parse().ok()?;
    for m in maps {
        if base == m.file_basename && line >= m.lo && line <= m.hi {
            return Some(m.region.clone());
        }
    }
    None
}

/// A field value parsed from a tab record's `key=value` token, or a JSON
/// object field. Returns the string value for `key`.
fn field<'a>(rec: &'a Record, key: &str) -> Option<&'a str> {
    rec.fields.get(key).map(|s| s.as_str())
}

struct Record {
    kind: String,
    fields: std::collections::HashMap<String, String>,
}

fn parse_record(line: &str) -> Option<Record> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    if let Some(stripped) = line.strip_prefix('{') {
        // JSON object form. Reconstruct, parse leniently.
        let json = format!("{{{stripped}");
        let v: serde_json::Value = serde_json::from_str(&json).ok()?;
        let obj = v.as_object()?;
        let kind = obj
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        let mut fields = std::collections::HashMap::new();
        for (k, val) in obj {
            let s = match val {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            fields.insert(k.clone(), s);
        }
        return Some(Record { kind, fields });
    }
    // Tab-delimited: first token is the type, rest are key=value.
    let mut parts = line.split('\t');
    let kind = parts.next()?.to_string();
    let mut fields = std::collections::HashMap::new();
    for p in parts {
        if let Some((k, v)) = p.split_once('=') {
            fields.insert(k.to_string(), v.to_string());
        }
    }
    Some(Record { kind, fields })
}

/// Parsed profile.coz: per-`selected` curve for the named throughput
/// point, plus the raw latency-point arrivals/departures per region scope.
pub struct CozProfile {
    pub progress_point: String,
    pub line_curves: Vec<LineCurve>,
    /// Region → aggregated curve (mean of member-line curves, sample-weighted).
    pub region_curves: BTreeMap<String, RegionCurve>,
    /// Region scope (latency point) → (arrivals, departures, mean latency ns).
    pub region_latency: BTreeMap<String, (f64, f64, f64)>,
    pub n_experiments: usize,
}

#[derive(Clone)]
pub struct RegionCurve {
    pub region: String,
    pub points: BTreeMap<u64, (f64, f64)>, // level -> (sum program_speedup, weight)
    pub samples: f64,
}

impl RegionCurve {
    /// Sample-weighted mean program-speedup at the max measured level, plus
    /// a crude 95% half-width from the per-line spread (±1.96·sd/√k). For
    /// the MVP this CI is indicative, not rigorous (Coz's own bootstrapping
    /// would be tighter); reported as such.
    pub fn elasticity_ci(&self) -> (f64, f64, f64) {
        let Some((lvl, _)) = self.points.iter().next_back() else {
            return (0.0, 0.0, 0.0);
        };
        let line_speedup = *lvl as f64 / 1000.0;
        let (sum, w) = self.points[lvl];
        let mean_ps = if w > 0.0 { sum / w } else { 0.0 };
        // elasticity = program_speedup / line_speedup at the top level
        let elasticity = if line_speedup > 0.0 {
            mean_ps / line_speedup
        } else {
            0.0
        };
        // Half-width proxy: scale inversely with sqrt(samples).
        let halfwidth = if self.samples > 1.0 {
            0.5 * elasticity.abs() / self.samples.sqrt().max(1.0)
        } else {
            elasticity.abs()
        };
        (
            elasticity,
            (elasticity - halfwidth).max(-1.0),
            elasticity + halfwidth,
        )
    }
}

/// Parse a profile.coz file into per-line + per-region curves, mapping the
/// named throughput point `progress_point` (default "chunk_emitted").
pub fn parse_profile(
    path: &Path,
    progress_point: &str,
    maps: &[RegionMap],
) -> std::io::Result<CozProfile> {
    let text = std::fs::read_to_string(path)?;
    // First pass: collect, per (selected, speedup-level), the accumulated
    // throughput for the named point. Coz interleaves an `experiment` line
    // (carrying selected+speedup) followed by the `throughput-point`
    // measurements observed DURING that experiment, then the next
    // experiment. We therefore track the "current" experiment context.
    let mut acc: BTreeMap<(String, u64), ThroughputAccum> = BTreeMap::new();
    let mut samples: BTreeMap<String, f64> = BTreeMap::new();
    let mut latency: BTreeMap<String, (f64, f64, f64)> = BTreeMap::new();
    let mut cur_selected: Option<String> = None;
    let mut cur_level: u64 = 0;
    // Real coz (JSON form) puts `duration` on the EXPERIMENT record; the
    // following `throughput-point` record carries only `name` + `delta`.
    // So we carry the current experiment's duration here and use it for the
    // throughput period (period = duration / delta). The older tab format
    // put duration on the throughput-point itself — both handled.
    let mut cur_duration: f64 = 0.0;
    let mut n_exp = 0usize;

    for line in text.lines() {
        let Some(rec) = parse_record(line) else {
            continue;
        };
        match rec.kind.as_str() {
            "experiment" => {
                n_exp += 1;
                let selected = field(&rec, "selected").unwrap_or("").to_string();
                let speedup: f64 = field(&rec, "speedup")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                // Quantize speedup to per-mille to use as a stable map key.
                cur_level = (speedup * 1000.0).round() as u64;
                cur_duration = field(&rec, "duration")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                // `selected_samples` (JSON, underscore) or `selected-samples`
                // (tab, hyphen) — accept either.
                if let Some(s) = field(&rec, "selected_samples")
                    .or_else(|| field(&rec, "selected-samples"))
                    .and_then(|s| s.parse::<f64>().ok())
                {
                    *samples.entry(selected.clone()).or_default() += s;
                }
                cur_selected = Some(selected);
            }
            "throughput-point" | "throughput_point" | "progress-point" => {
                let name = field(&rec, "name").unwrap_or("");
                if name != progress_point {
                    continue;
                }
                let delta: f64 = field(&rec, "delta")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                // Prefer the throughput-point's own duration (tab format);
                // fall back to the enclosing experiment's duration (JSON
                // format, where the point record omits it).
                let duration: f64 = field(&rec, "duration")
                    .and_then(|s| s.parse().ok())
                    .filter(|d: &f64| *d > 0.0)
                    .unwrap_or(cur_duration);
                if let Some(sel) = &cur_selected {
                    let e = acc.entry((sel.clone(), cur_level)).or_default();
                    e.delta += delta;
                    e.duration += duration;
                }
            }
            "latency-point" | "latency_point" => {
                let name = field(&rec, "name").unwrap_or("").to_string();
                let arrivals: f64 = field(&rec, "arrivals")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                let departures: f64 = field(&rec, "departures")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                let difference: f64 = field(&rec, "difference")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                let e = latency.entry(name).or_insert((0.0, 0.0, 0.0));
                e.0 += arrivals;
                e.1 += departures;
                e.2 += difference;
            }
            _ => {}
        }
    }

    // Build per-line curves. coz establishes the program's BASELINE
    // throughput from the speedup=0 experiments — and crucially those
    // baselines are GLOBAL (the program runs at its native speed), shared
    // across every `selected` line, not per-line. With few runs a given
    // line may only ever be measured at a NONZERO speedup, with its
    // speedup=0 baseline contributed by experiments on OTHER lines. So we
    // compute a global baseline period = aggregate(delta,duration) over ALL
    // speedup=0 experiments and use it when a per-line level-0 is absent.
    let mut by_selected: BTreeMap<String, BTreeMap<u64, ThroughputAccum>> = BTreeMap::new();
    let mut global_base = ThroughputAccum::default();
    for ((sel, lvl), a) in acc {
        if lvl == 0 {
            global_base.delta += a.delta;
            global_base.duration += a.duration;
        }
        by_selected.entry(sel).or_default().insert(lvl, a);
    }
    let global_baseline = global_base.period();

    let mut line_curves = Vec::new();
    let mut region_curves: BTreeMap<String, RegionCurve> = BTreeMap::new();

    for (selected, levels) in by_selected {
        // coz's baseline is the program's native (speedup=0) throughput,
        // which is GLOBAL — so use the aggregated global baseline. Fall
        // back to a line-local level-0 only if no global-0 exists at all.
        let baseline = global_baseline.or_else(|| levels.get(&0).and_then(|a| a.period()));
        let Some(baseline) = baseline else { continue };
        if baseline <= 0.0 {
            continue;
        }
        let mut curve = LineCurve {
            selected: selected.clone(),
            points: BTreeMap::new(),
            total_samples: *samples.get(&selected).unwrap_or(&0.0),
        };
        for (lvl, a) in &levels {
            if let Some(period) = a.period() {
                let program_speedup = (baseline - period) / baseline;
                curve.points.insert(*lvl, program_speedup);
            }
        }
        // Map to region + fold into the region curve.
        if let Some(region) = classify(&selected, maps) {
            let rc = region_curves
                .entry(region.clone())
                .or_insert_with(|| RegionCurve {
                    region: region.clone(),
                    points: BTreeMap::new(),
                    samples: 0.0,
                });
            let w = curve.total_samples.max(1.0);
            for (lvl, ps) in &curve.points {
                let e = rc.points.entry(*lvl).or_insert((0.0, 0.0));
                e.0 += ps * w;
                e.1 += w;
            }
            rc.samples += curve.total_samples;
        }
        line_curves.push(curve);
    }

    // Convert region accumulation (sum*w, w) into (mean, weight) is already
    // the stored form: points hold (sum_program_speedup*w, w). elasticity_ci
    // divides sum/w. Good.

    line_curves.sort_by(|a, b| {
        b.slope()
            .abs()
            .partial_cmp(&a.slope().abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(CozProfile {
        progress_point: progress_point.to_string(),
        line_curves,
        region_curves,
        region_latency: latency,
        n_experiments: n_exp,
    })
}

/// Build the `coz run` command that loops the in-process decode harness.
/// Coz APPENDS to `output` across runs, so the caller invokes this N times
/// (or relies on the harness's own `--iters`) to accumulate samples.
///
/// `harness_bin`: path to the built `fulcrum_loop` example.
/// Returns a Command ready to spawn.
pub fn coz_run_command(
    coz_bin: &str,
    harness_bin: &Path,
    input: &Path,
    iters: usize,
    threads: usize,
    output: &Path,
    cpus: Option<&str>,
) -> Command {
    // taskset wraps coz so the whole experiment is pinned to the P-cores.
    let mut cmd = if let Some(cpus) = cpus {
        let mut c = Command::new("taskset");
        c.arg("-c").arg(cpus).arg(coz_bin);
        c
    } else {
        Command::new(coz_bin)
    };
    cmd.arg("run")
        .arg("--output")
        .arg(output)
        // One experiment per execution; the outer loop + coz's append to
        // `output` accumulate experiments across runs for statistical power.
        .arg("--end-to-end")
        // Restrict the line-selection search to the parallel decode source
        // so coz spends its epochs on the regions we care about (and the
        // begin!/end! scopes), not on unrelated startup code.
        .arg("--source-scope")
        .arg("%/decompress/parallel/%")
        .arg("--binary-scope")
        .arg("MAIN")
        .arg("---")
        .arg(harness_bin)
        .arg(input)
        .arg("--iters")
        .arg(iters.to_string())
        .arg("--threads")
        .arg(threads.to_string());
    cmd
}

/// A `coz run` that PINS one source line at one speedup — the direct
/// validation probe ("what is the wall-elasticity of EXACTLY this line?").
/// Used to interrogate the absorb / bootstrap lines against ground truth.
#[allow(clippy::too_many_arguments)]
pub fn coz_fixed_command(
    coz_bin: &str,
    harness_bin: &Path,
    input: &Path,
    iters: usize,
    threads: usize,
    output: &Path,
    fixed_line: &str,
    fixed_speedup_pct: u32,
    cpus: Option<&str>,
) -> Command {
    let mut cmd = if let Some(cpus) = cpus {
        let mut c = Command::new("taskset");
        c.arg("-c").arg(cpus).arg(coz_bin);
        c
    } else {
        Command::new(coz_bin)
    };
    cmd.arg("run")
        .arg("--output")
        .arg(output)
        .arg("--end-to-end")
        .arg("--fixed-line")
        .arg(fixed_line)
        .arg("--fixed-speedup")
        .arg(fixed_speedup_pct.to_string())
        .arg("--binary-scope")
        .arg("MAIN")
        .arg("---")
        .arg(harness_bin)
        .arg(input)
        .arg("--iters")
        .arg(iters.to_string())
        .arg("--threads")
        .arg(threads.to_string());
    cmd
}
