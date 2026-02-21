//! Cloud fleet benchmarking: spin up EC2 on-demand instances, run benchmarks, tear down.
//!
//! Architecture:
//!   1. Launch x86_64 + arm64 EC2 instances in parallel
//!   2. User-data builds gzippy, gzippy-dev, and all competitor tools
//!   3. Benchmark data staged to /dev/shm (RAM-backed, no EBS bottleneck)
//!   4. SSH: `gzippy-dev bench --json` — same tool, same code, local or cloud
//!   5. Two-phase: sweep first, then precision re-runs for close races
//!   6. Strict scoring: gzippy must be >= every competitor, or it's a loss
//!
//! Uses `aws-vault exec personal` for credentials.
//!
//! Also runs local Mac benchmarks in parallel with the cloud fleet.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::bench;
use crate::bench::BenchDirection;
use std::time::{Duration, Instant};

const REGION: &str = "us-east-1";
const X86_TYPE: &str = "c7i.4xlarge";
const ARM64_TYPE: &str = "c8g.4xlarge";
const TAG_KEY: &str = "gzippy-bench";
const SSH_USER: &str = "ubuntu";
const SETUP_TIMEOUT: Duration = Duration::from_secs(20 * 60);
// Match CI runner core count for comparable Tmax results
const BENCH_TMAX_THREADS: usize = 4;

// Phase 1: initial sweep (higher counts since each instance handles 1 dataset)
const SWEEP_MIN_TRIALS: u32 = 30;
const SWEEP_MAX_TRIALS: u32 = 100;
const SWEEP_TARGET_CV: f64 = 0.015;

// Phase 2: precision re-run for close races
const PRECISION_MIN_TRIALS: u32 = 50;
const PRECISION_MAX_TRIALS: u32 = 200;
const PRECISION_TARGET_CV: f64 = 0.005;
const CLOSE_RACE_THRESHOLD: f64 = 3.0;

// ─── AWS CLI ──────────────────────────────────────────────────────────────────

fn aws(args: &[&str]) -> Result<String, String> {
    let output = Command::new("aws-vault")
        .args(["exec", "personal", "-d", "12h", "--"])
        .arg("aws")
        .arg("--region")
        .arg(REGION)
        .args(args)
        .output()
        .map_err(|e| format!("Failed to run aws-vault: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("aws {}: {}", args.first().unwrap_or(&""), stderr));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

// ─── SSH ──────────────────────────────────────────────────────────────────────

fn ssh_opts(key: &Path) -> Vec<String> {
    vec![
        "-o".into(), "StrictHostKeyChecking=no".into(),
        "-o".into(), "UserKnownHostsFile=/dev/null".into(),
        "-o".into(), "ConnectTimeout=10".into(),
        "-o".into(), "ServerAliveInterval=30".into(),
        "-o".into(), "ServerAliveCountMax=10".into(),
        "-o".into(), "LogLevel=ERROR".into(),
        "-i".into(), key.to_string_lossy().into(),
    ]
}

fn ssh(ip: &str, key: &Path, cmd: &str) -> Result<String, String> {
    let output = Command::new("ssh")
        .args(ssh_opts(key))
        .arg(format!("{SSH_USER}@{ip}"))
        .arg(cmd)
        .output()
        .map_err(|e| format!("SSH to {ip}: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() && stdout.trim().is_empty() {
        return Err(format!("SSH command failed on {ip}: {stderr}"));
    }
    Ok(stdout)
}

fn ssh_ok(ip: &str, key: &Path) -> bool {
    Command::new("ssh")
        .args(ssh_opts(key))
        .arg(format!("{SSH_USER}@{ip}"))
        .arg("true")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// ─── Resource lifecycle ───────────────────────────────────────────────────────

struct CleanupState {
    key_name: Option<String>,
    key_path: Option<PathBuf>,
    sg_id: Option<String>,
    instance_ids: Vec<String>,
}

impl CleanupState {
    fn new() -> Self {
        Self { key_name: None, key_path: None, sg_id: None, instance_ids: Vec::new() }
    }

    fn run(&self) {
        if !self.instance_ids.is_empty() {
            let ids: Vec<&str> = self.instance_ids.iter().map(|s| s.as_str()).collect();
            println!("  Terminating {} instance(s)...", ids.len());
            let mut args = vec!["ec2", "terminate-instances", "--instance-ids"];
            args.extend(ids.iter());
            let _ = aws(&args);
        }
        if let Some(sg) = &self.sg_id {
            if !self.instance_ids.is_empty() {
                std::thread::sleep(Duration::from_secs(5));
            }
            println!("  Deleting security group {sg}...");
            let _ = aws(&["ec2", "delete-security-group", "--group-id", sg]);
        }
        if let Some(name) = &self.key_name {
            println!("  Deleting key pair {name}...");
            let _ = aws(&["ec2", "delete-key-pair", "--key-name", name]);
        }
        if let Some(path) = &self.key_path {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn create_key_pair(session: &str) -> Result<(String, PathBuf), String> {
    let name = format!("{TAG_KEY}-{session}");
    let path = std::env::temp_dir().join(format!("{name}.pem"));
    let material = aws(&[
        "ec2", "create-key-pair",
        "--key-name", &name,
        "--query", "KeyMaterial",
        "--output", "text",
    ])?;
    std::fs::write(&path, &material).map_err(|e| format!("Write key: {e}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("chmod key: {e}"))?;
    }
    Ok((name, path))
}

fn find_vpc_and_subnet() -> Result<(String, String), String> {
    let vpc_id = aws(&[
        "ec2", "describe-vpcs",
        "--filters", "Name=isDefault,Values=true",
        "--query", "Vpcs[0].VpcId",
        "--output", "text",
    ])?;
    let vpc_id = if vpc_id.is_empty() || vpc_id == "None" {
        aws(&["ec2", "describe-vpcs", "--query", "Vpcs[0].VpcId", "--output", "text"])?
    } else {
        vpc_id
    };
    if vpc_id.is_empty() || vpc_id == "None" {
        return Err("No VPC found".into());
    }

    let subnet_id = aws(&[
        "ec2", "describe-subnets",
        "--filters",
            &format!("Name=vpc-id,Values={vpc_id}"),
            "Name=map-public-ip-on-launch,Values=true",
        "--query", "Subnets[0].SubnetId",
        "--output", "text",
    ])?;
    if subnet_id.is_empty() || subnet_id == "None" {
        let subnet_id = aws(&[
            "ec2", "describe-subnets",
            "--filters", &format!("Name=vpc-id,Values={vpc_id}"),
            "--query", "Subnets[0].SubnetId", "--output", "text",
        ])?;
        return Ok((vpc_id, subnet_id));
    }
    Ok((vpc_id, subnet_id))
}

fn find_ubuntu_ami(arch: &str) -> Result<String, String> {
    let arch_filter = match arch {
        "x86_64" => "amd64",
        "arm64" => "arm64",
        _ => return Err(format!("Unknown arch: {arch}")),
    };
    let name_pattern = format!("ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-{arch_filter}-server-*");
    aws(&[
        "ec2", "describe-images",
        "--owners", "099720109477",
        "--filters", &format!("Name=name,Values={name_pattern}"), "Name=state,Values=available",
        "--query", "sort_by(Images, &CreationDate)[-1].ImageId",
        "--output", "text",
    ])
}

fn create_security_group(session: &str, vpc_id: &str) -> Result<String, String> {
    let name = format!("{TAG_KEY}-{session}");
    let sg_id = aws(&[
        "ec2", "create-security-group",
        "--group-name", &name,
        "--description", "Ephemeral SG for gzippy cloud benchmarks",
        "--vpc-id", vpc_id,
        "--query", "GroupId", "--output", "text",
    ])?;
    let my_ip = get_my_public_ip().unwrap_or_else(|_| "0.0.0.0/0".to_string());
    let cidr = if my_ip.contains('/') { my_ip } else { format!("{my_ip}/32") };
    aws(&[
        "ec2", "authorize-security-group-ingress",
        "--group-id", &sg_id, "--protocol", "tcp", "--port", "22", "--cidr", &cidr,
    ])?;
    Ok(sg_id)
}

fn get_my_public_ip() -> Result<String, String> {
    let output = Command::new("curl")
        .args(["-s", "--max-time", "5", "https://checkip.amazonaws.com"])
        .output().map_err(|e| format!("curl: {e}"))?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn launch_on_demand(
    instance_type: &str, ami: &str, key: &str, sg: &str,
    subnet: &str, userdata_b64: &str, session: &str,
) -> Result<String, String> {
    let tag_spec = format!(
        "ResourceType=instance,Tags=[{{Key=Name,Value={TAG_KEY}-{session}}},{{Key={TAG_KEY},Value={session}}}]"
    );
    aws(&[
        "ec2", "run-instances",
        "--instance-type", instance_type,
        "--image-id", ami,
        "--key-name", key,
        "--security-group-ids", sg,
        "--subnet-id", subnet,
        "--associate-public-ip-address",
        "--block-device-mappings",
            "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":50,\"VolumeType\":\"gp3\",\"Iops\":3000,\"Throughput\":250}}]",
        "--user-data", userdata_b64,
        "--tag-specifications", &tag_spec,
        "--query", "Instances[0].InstanceId",
        "--output", "text",
    ])
}

fn wait_running(ids: &[&str]) -> Result<(), String> {
    let mut args = vec!["ec2", "wait", "instance-running", "--instance-ids"];
    args.extend(ids);
    aws(&args)?;
    Ok(())
}

fn get_public_ip(id: &str) -> Result<String, String> {
    aws(&[
        "ec2", "describe-instances",
        "--instance-ids", id,
        "--query", "Reservations[0].Instances[0].PublicIpAddress",
        "--output", "text",
    ])
}

fn wait_for_ssh(ip: &str, key: &Path, label: &str) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(5 * 60);
    loop {
        if ssh_ok(ip, key) {
            println!("  [{label}] SSH ready");
            return Ok(());
        }
        if Instant::now() > deadline {
            return Err(format!("[{label}] SSH timeout after 5 min"));
        }
        std::thread::sleep(Duration::from_secs(8));
        print!(".");
        let _ = std::io::stdout().flush();
    }
}

fn wait_for_setup(ip: &str, key: &Path, label: &str) -> Result<(), String> {
    let deadline = Instant::now() + SETUP_TIMEOUT;
    let start = Instant::now();
    println!("  [{label}] Waiting for setup...");
    loop {
        if let Ok(out) = ssh(ip, key, "cat /tmp/gzippy-ready 2>/dev/null || echo NOT_READY") {
            if out.trim() == "READY" {
                println!("  [{label}] Setup complete ({:.0}s)", start.elapsed().as_secs_f64());
                return Ok(());
            }
        }
        if Instant::now() > deadline {
            let log = ssh(ip, key, "tail -30 /var/log/cloud-init-output.log 2>/dev/null").unwrap_or_default();
            eprintln!("  [{label}] Last setup output:\n{log}");
            return Err(format!("[{label}] Setup timeout"));
        }
        let elapsed = start.elapsed().as_secs();
        if elapsed.is_multiple_of(30) && elapsed > 0 {
            let progress = ssh(ip, key, "tail -1 /var/log/cloud-init-output.log 2>/dev/null | head -c 120").unwrap_or_default();
            let progress = progress.trim();
            if !progress.is_empty() {
                println!("  [{label}] ({elapsed}s) {progress}");
            }
        }
        std::thread::sleep(Duration::from_secs(8));
    }
}

// ─── User-data script ─────────────────────────────────────────────────────────

fn user_data_script(commit: &str, repo_url: &str) -> String {
    format!(r#"#!/bin/bash
set -euxo pipefail
exec > /var/log/cloud-init-output.log 2>&1

echo "=== Installing dependencies ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential cmake git curl pigz nasm python3 \
    zlib1g-dev pkg-config unzip wget xz-utils autoconf automake libtool

echo "=== Installing Rust ==="
sudo -u {SSH_USER} bash -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'

echo "=== Cloning gzippy at {commit} ==="
cd /home/{SSH_USER}
sudo -u {SSH_USER} git clone --recursive {repo_url} gzippy
cd gzippy
sudo -u {SSH_USER} git checkout {commit}
sudo -u {SSH_USER} git submodule update --init --recursive

echo "=== Building all tools ==="
sudo -u {SSH_USER} bash -c 'source $HOME/.cargo/env && ./scripts/build-tools.sh --all'

echo "=== Building gzippy-dev ==="
sudo -u {SSH_USER} bash -c 'source $HOME/.cargo/env && cargo build --release --manifest-path tools/devtool/Cargo.toml --target-dir target'

echo "=== Preparing benchmark data ==="
sudo -u {SSH_USER} bash -c './scripts/prepare_benchmark_data.sh'

echo "=== Compressing for decompression benchmarks ==="
cd /home/{SSH_USER}/gzippy
GZIPPY=./target/release/gzippy
DATA_DIR=benchmark_data

for DATASET in silesia software logs; do
    case $DATASET in
        silesia) RAW="$DATA_DIR/silesia.tar" ;;
        software) RAW="$DATA_DIR/software.archive" ;;
        logs) RAW="$DATA_DIR/logs.txt" ;;
    esac
    [ ! -f "$RAW" ] && echo "SKIP: $RAW" && continue
    echo "Compressing $DATASET..."
    sudo -u {SSH_USER} bash -c "gzip -1 -c $RAW > $DATA_DIR/$DATASET-gzip.gz"
    sudo -u {SSH_USER} bash -c "source \$HOME/.cargo/env && $GZIPPY -1 -c $RAW > $DATA_DIR/$DATASET-bgzf.gz"
    sudo -u {SSH_USER} bash -c "./pigz/pigz -1 -c $RAW > $DATA_DIR/$DATASET-pigz.gz"
done

# Copy ALL data to /dev/shm — eliminates EBS I/O bottleneck
echo "=== Staging data to RAM (/dev/shm) ==="
for f in $DATA_DIR/silesia.tar $DATA_DIR/software.archive $DATA_DIR/logs.txt \
         $DATA_DIR/*-gzip.gz $DATA_DIR/*-bgzf.gz $DATA_DIR/*-pigz.gz; do
    [ -f "$f" ] && cp "$f" /dev/shm/
done
chown {SSH_USER}:{SSH_USER} /dev/shm/* 2>/dev/null || true
ls -lh /dev/shm/

# Flat bin directory with all tools + gzippy-dev
mkdir -p /home/{SSH_USER}/gzippy/bin
cd /home/{SSH_USER}/gzippy/bin
ln -sf ../target/release/gzippy gzippy
ln -sf ../target/release/gzippy-dev gzippy-dev
ln -sf ../pigz/pigz pigz
ln -sf ../pigz/unpigz unpigz
[ -f ../isa-l/build/igzip ] && ln -sf ../isa-l/build/igzip igzip || true
[ -f ../rapidgzip/librapidarchive/build/src/tools/rapidgzip ] && \
    ln -sf ../rapidgzip/librapidarchive/build/src/tools/rapidgzip rapidgzip || true
ln -sf /usr/bin/gzip gzip
chown -R {SSH_USER}:{SSH_USER} /home/{SSH_USER}/gzippy/bin

echo "READY" > /tmp/gzippy-ready
echo "=== Setup complete ==="
"#)
}

// ─── Benchmark execution via gzippy-dev on remote host ────────────────────────

#[derive(Clone)]
#[allow(dead_code)]
struct BenchResult {
    platform: String,
    dataset: String,
    archive: String,
    threads: String,
    tool: String,
    speed_mbps: f64,
    cv: f64,
    trials: u32,
    direction: String,
}

#[allow(clippy::too_many_arguments)]
fn remote_bench(
    ip: &str, key: &Path, label: &str,
    min_trials: u32, max_trials: u32, target_cv: f64,
    dataset: Option<&str>, archive: Option<&str>, threads: Option<usize>,
    direction: Option<&str>,
) -> Result<Vec<BenchResult>, String> {
    let repo = format!("/home/{SSH_USER}/gzippy");

    let mut cmd = format!(
        "cd {repo} && export PATH={repo}/bin:$PATH && \
         export TMPDIR=/dev/shm && \
         source $HOME/.cargo/env && \
         gzippy-dev bench --json \
         --min-trials {min_trials} --max-trials {max_trials} --target-cv {target_cv}"
    );
    if let Some(ds) = dataset { cmd += &format!(" --dataset {ds}"); }
    if let Some(ar) = archive { cmd += &format!(" --archive {ar}"); }
    if let Some(th) = threads { cmd += &format!(" --threads {th}"); }
    if let Some(dir) = direction { cmd += &format!(" --direction {dir}"); }

    let output = ssh(ip, key, &cmd)?;

    let json_line = output.lines()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| format!("[{label}] No JSON in bench output. Raw:\n{output}"))?;

    parse_bench_json(json_line, label)
}

fn parse_bench_json(json_str: &str, platform: &str) -> Result<Vec<BenchResult>, String> {
    let parsed: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| format!("JSON parse error: {e}"))?;

    let items = parsed.get("results")
        .and_then(|v| v.as_array())
        .ok_or("Missing 'results' array in JSON")?;

    let mut results = Vec::new();
    for item in items {
        let status = item.get("status").and_then(|v| v.as_str()).unwrap_or("fail");
        if status != "pass" { continue; }

        results.push(BenchResult {
            platform: platform.to_string(),
            dataset: item.get("dataset").and_then(|v| v.as_str()).unwrap_or("?").into(),
            archive: item.get("archive").and_then(|v| v.as_str()).unwrap_or("?").into(),
            threads: item.get("threads").and_then(|v| v.as_str()).unwrap_or("?").into(),
            tool: item.get("tool").and_then(|v| v.as_str()).unwrap_or("?").into(),
            speed_mbps: item.get("speed_mbps").and_then(|v| v.as_f64()).unwrap_or(0.0),
            cv: item.get("cv").and_then(|v| v.as_f64()).unwrap_or(0.0),
            trials: item.get("trials").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            direction: item.get("direction").and_then(|v| v.as_str()).unwrap_or("decompress").into(),
        });
    }
    Ok(results)
}

fn scenario_key(r: &BenchResult) -> String {
    format!("{}-{}-{}", r.dataset, r.archive, r.threads)
}

fn run_benchmarks_on(
    ip: &str, key: &Path, label: &str, dataset: Option<&str>,
    direction: &str,
) -> Result<Vec<BenchResult>, String> {
    let ds_label = dataset.unwrap_or("all");
    let dir_label = direction;

    println!("\n  [{label}] Phase 1: {dir_label} sweep {ds_label} ({SWEEP_MIN_TRIALS}-{SWEEP_MAX_TRIALS} trials, CV<{:.0}%, Tmax={BENCH_TMAX_THREADS})",
        SWEEP_TARGET_CV * 100.0);

    println!("  [{label}] T1...");
    let t1 = remote_bench(
        ip, key, label,
        SWEEP_MIN_TRIALS, SWEEP_MAX_TRIALS, SWEEP_TARGET_CV,
        dataset, None, Some(1), Some(direction),
    )?;
    for r in &t1 {
        println!("    {}-{} {} {}: {:.1} MB/s (CV {:.1}%)", r.dataset, r.archive, r.threads, r.tool, r.speed_mbps, r.cv * 100.0);
    }

    println!("  [{label}] Tmax ({BENCH_TMAX_THREADS} threads)...");
    let tmax = remote_bench(
        ip, key, label,
        SWEEP_MIN_TRIALS, SWEEP_MAX_TRIALS, SWEEP_TARGET_CV,
        dataset, None, Some(BENCH_TMAX_THREADS), Some(direction),
    )?;
    for r in &tmax {
        println!("    {}-{} {} {}: {:.1} MB/s (CV {:.1}%)", r.dataset, r.archive, r.threads, r.tool, r.speed_mbps, r.cv * 100.0);
    }

    let mut sweep: Vec<BenchResult> = Vec::new();
    sweep.extend(t1);
    sweep.extend(tmax);

    // Phase 2: Precision re-runs for close races
    let close_races = find_close_races(&sweep);
    if close_races.is_empty() {
        println!("  [{label}] Phase 2: No close races, all decisive");
        return Ok(sweep);
    }

    println!("\n  [{label}] Phase 2: {} close race(s) ({PRECISION_MIN_TRIALS}-{PRECISION_MAX_TRIALS} trials, CV<{:.1}%)",
        close_races.len(), PRECISION_TARGET_CV * 100.0);

    let mut all = sweep;

    for (ds, arch, thr) in &close_races {
        let threads: Option<usize> = Some(if thr == "T1" { 1 } else { BENCH_TMAX_THREADS });
        println!("  [{label}] PRECISION {ds}-{arch} {thr}...");

        let precision = remote_bench(
            ip, key, label,
            PRECISION_MIN_TRIALS, PRECISION_MAX_TRIALS, PRECISION_TARGET_CV,
            Some(ds), Some(arch), threads, Some(direction),
        )?;

        for r in &precision {
            println!("    {}: {:.1} MB/s (CV {:.2}%, {} trials)", r.tool, r.speed_mbps, r.cv * 100.0, r.trials);
        }

        let key_prefix = format!("{ds}-{arch}-{thr}");
        all.retain(|r| scenario_key(r) != key_prefix);
        all.extend(precision);
    }

    Ok(all)
}

fn find_close_races(results: &[BenchResult]) -> Vec<(String, String, String)> {
    let mut races = Vec::new();
    let mut scenarios: Vec<(String, String, String)> = results.iter()
        .map(|r| (r.dataset.clone(), r.archive.clone(), r.threads.clone()))
        .collect();
    scenarios.sort();
    scenarios.dedup();

    for (ds, arch, thr) in scenarios {
        let scenario: Vec<&BenchResult> = results.iter()
            .filter(|r| r.dataset == ds && r.archive == arch && r.threads == thr)
            .collect();

        let gzippy = scenario.iter().find(|r| r.tool == "gzippy");
        let best_competitor = scenario.iter()
            .filter(|r| r.tool != "gzippy")
            .max_by(|a, b| a.speed_mbps.partial_cmp(&b.speed_mbps).unwrap());

        if let (Some(g), Some(b)) = (gzippy, best_competitor) {
            let gap_pct = ((g.speed_mbps / b.speed_mbps) - 1.0) * 100.0;
            if gap_pct.abs() < CLOSE_RACE_THRESHOLD {
                races.push((ds, arch, thr));
            }
        }
    }
    races
}

// ─── Results ──────────────────────────────────────────────────────────────────

fn print_results(results: &[BenchResult]) {
    let has_decomp = results.iter().any(|r| r.direction == "decompress");
    let has_comp = results.iter().any(|r| r.direction == "compress");

    println!("\n╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  CLOUD FLEET RESULTS — STRICT SCORING (gzippy must win or it's a loss)             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝\n");

    let mut total_wins = 0u32;
    let mut total_losses = 0u32;

    if has_decomp {
        println!("  ══ DECOMPRESSION ══");
        let mut platforms: Vec<&str> = results.iter()
            .filter(|r| r.direction == "decompress")
            .map(|r| r.platform.as_str()).collect();
        platforms.sort();
        platforms.dedup();

        for platform in &platforms {
            println!("\n  ── {platform} ──");
            println!("  {:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>6} Verdict",
                "Scenario", "gzippy", "unpigz", "igzip", "rapidgzip", "gzip", "CV%");
            println!("  {}", "─".repeat(100));

            let plat_results: Vec<&BenchResult> = results.iter()
                .filter(|r| r.direction == "decompress" && r.platform == *platform)
                .collect();

            let mut scenarios: Vec<(String, String, String)> = plat_results.iter()
                .map(|r| (r.dataset.clone(), r.archive.clone(), r.threads.clone()))
                .collect();
            scenarios.sort();
            scenarios.dedup();

            for (dataset, archive, threads) in &scenarios {
                let scenario: Vec<&BenchResult> = plat_results.iter()
                    .filter(|r| r.dataset == *dataset && r.archive == *archive && r.threads == *threads)
                    .copied()
                    .collect();

                let get = |tool: &str| -> Option<&BenchResult> {
                    scenario.iter().find(|r| r.tool == tool).copied()
                };
                let fmt_speed = |tool: &str| -> String {
                    get(tool).map(|r| format!("{:.1}", r.speed_mbps)).unwrap_or_else(|| "—".to_string())
                };

                let gzippy = get("gzippy");
                let gzippy_cv = gzippy.map(|r| r.cv).unwrap_or(0.0);

                let competitors = ["unpigz", "pigz", "igzip", "rapidgzip", "gzip"];
                let best = competitors.iter()
                    .filter_map(|t| get(t))
                    .max_by(|a, b| a.speed_mbps.partial_cmp(&b.speed_mbps).unwrap());

                let (verdict, is_win) = if let (Some(g), Some(b)) = (gzippy, best) {
                    let gap = ((g.speed_mbps / b.speed_mbps) - 1.0) * 100.0;
                    if gap >= 0.0 {
                        (format!("WIN +{:.1}% vs {}", gap, b.tool), true)
                    } else {
                        (format!("LOSS {:.1}% vs {}", gap, b.tool), false)
                    }
                } else {
                    ("—".to_string(), false)
                };

                if gzippy.is_some() {
                    if is_win { total_wins += 1; } else { total_losses += 1; }
                }

                let scenario_name = format!("{dataset}-{archive} {threads}");
                let unpigz_speed = get("unpigz").or(get("pigz"));
                println!("  {:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>5.1} {}",
                    scenario_name,
                    fmt_speed("gzippy"),
                    unpigz_speed.map(|r| format!("{:.1}", r.speed_mbps)).unwrap_or_else(|| "—".to_string()),
                    fmt_speed("igzip"),
                    fmt_speed("rapidgzip"),
                    fmt_speed("gzip"),
                    gzippy_cv * 100.0,
                    verdict,
                );
            }
        }
    }

    if has_comp {
        println!("\n  ══ COMPRESSION ══");
        let mut platforms: Vec<&str> = results.iter()
            .filter(|r| r.direction == "compress")
            .map(|r| r.platform.as_str()).collect();
        platforms.sort();
        platforms.dedup();

        for platform in &platforms {
            println!("\n  ── {platform} ──");
            println!("  {:<25} {:>10} {:>10} {:>10} {:>10} {:>6} Verdict",
                "Scenario", "gzippy", "pigz", "igzip", "gzip", "CV%");
            println!("  {}", "─".repeat(85));

            let plat_results: Vec<&BenchResult> = results.iter()
                .filter(|r| r.direction == "compress" && r.platform == *platform)
                .collect();

            let mut scenarios: Vec<(String, String, String)> = plat_results.iter()
                .map(|r| (r.dataset.clone(), r.archive.clone(), r.threads.clone()))
                .collect();
            scenarios.sort();
            scenarios.dedup();

            for (dataset, level, threads) in &scenarios {
                let scenario: Vec<&BenchResult> = plat_results.iter()
                    .filter(|r| r.dataset == *dataset && r.archive == *level && r.threads == *threads)
                    .copied()
                    .collect();

                let get = |tool: &str| -> Option<&BenchResult> {
                    scenario.iter().find(|r| r.tool == tool).copied()
                };
                let fmt_speed = |tool: &str| -> String {
                    get(tool).map(|r| format!("{:.1}", r.speed_mbps)).unwrap_or_else(|| "—".to_string())
                };

                let gzippy = get("gzippy");
                let gzippy_cv = gzippy.map(|r| r.cv).unwrap_or(0.0);

                let competitors = ["pigz", "igzip", "gzip"];
                let best = competitors.iter()
                    .filter_map(|t| get(t))
                    .max_by(|a, b| a.speed_mbps.partial_cmp(&b.speed_mbps).unwrap());

                let (verdict, is_win) = if let (Some(g), Some(b)) = (gzippy, best) {
                    let gap = ((g.speed_mbps / b.speed_mbps) - 1.0) * 100.0;
                    if gap >= 0.0 {
                        (format!("WIN +{:.1}% vs {}", gap, b.tool), true)
                    } else {
                        (format!("LOSS {:.1}% vs {}", gap, b.tool), false)
                    }
                } else {
                    ("—".to_string(), false)
                };

                if gzippy.is_some() {
                    if is_win { total_wins += 1; } else { total_losses += 1; }
                }

                let scenario_name = format!("{dataset} {level} {threads}");
                println!("  {:<25} {:>10} {:>10} {:>10} {:>10} {:>5.1} {}",
                    scenario_name,
                    fmt_speed("gzippy"),
                    fmt_speed("pigz"),
                    fmt_speed("igzip"),
                    fmt_speed("gzip"),
                    gzippy_cv * 100.0,
                    verdict,
                );
            }
        }
    }

    // Summary
    let total = total_wins + total_losses;
    println!("\n  ══════════════════════════════════════════════════════");
    println!("  WINS: {total_wins}/{total}    LOSSES: {total_losses}/{total}");
    if total_losses > 0 {
        println!("\n  ── LOSSES ──");
        for r_dir in ["decompress", "compress"] {
            let dir_results: Vec<&BenchResult> = results.iter()
                .filter(|r| r.direction == r_dir)
                .collect();
            let mut scenarios: Vec<(String, String, String, String)> = dir_results.iter()
                .map(|r| (r.platform.clone(), r.dataset.clone(), r.archive.clone(), r.threads.clone()))
                .collect();
            scenarios.sort();
            scenarios.dedup();

            for (plat, ds, arch, thr) in &scenarios {
                let scenario: Vec<&&BenchResult> = dir_results.iter()
                    .filter(|r| r.platform == *plat && r.dataset == *ds && r.archive == *arch && r.threads == *thr)
                    .collect();
                let gzippy = scenario.iter().find(|r| r.tool == "gzippy").copied();
                let best = scenario.iter()
                    .filter(|r| r.tool != "gzippy")
                    .max_by(|a, b| a.speed_mbps.partial_cmp(&b.speed_mbps).unwrap())
                    .copied();
                if let (Some(g), Some(b)) = (gzippy, best) {
                    let gap = ((g.speed_mbps / b.speed_mbps) - 1.0) * 100.0;
                    if gap < 0.0 {
                        println!("    [{r_dir}] [{plat}] {ds}-{arch} {thr}: gzippy {:.1} vs {} {:.1} ({:+.1}%)",
                            g.speed_mbps, b.tool, b.speed_mbps, gap);
                    }
                }
            }
        }
    } else {
        println!("  CLEAN SWEEP — gzippy wins every scenario!");
    }
    println!();
}

fn dump_results_json(results: &[BenchResult], wall_time_secs: f64) {
    let items: Vec<serde_json::Value> = results.iter().map(|r| {
        serde_json::json!({
            "platform": r.platform,
            "dataset": r.dataset,
            "archive": r.archive,
            "threads": r.threads,
            "tool": r.tool,
            "speed_mbps": r.speed_mbps,
            "cv": r.cv,
            "trials": r.trials,
        })
    }).collect();

    // Build scorecard: per-scenario win/loss with gap
    let mut scenarios: Vec<(String, String, String, String)> = results.iter()
        .map(|r| (r.platform.clone(), r.dataset.clone(), r.archive.clone(), r.threads.clone()))
        .collect();
    scenarios.sort();
    scenarios.dedup();

    let mut scorecard: Vec<serde_json::Value> = Vec::new();
    for (plat, ds, arch, thr) in &scenarios {
        let scenario: Vec<&BenchResult> = results.iter()
            .filter(|r| r.platform == *plat && r.dataset == *ds && r.archive == *arch && r.threads == *thr)
            .collect();
        let gzippy = scenario.iter().find(|r| r.tool == "gzippy");
        let best = scenario.iter()
            .filter(|r| r.tool != "gzippy")
            .max_by(|a, b| a.speed_mbps.partial_cmp(&b.speed_mbps).unwrap());
        if let (Some(g), Some(b)) = (gzippy, best) {
            let gap = ((g.speed_mbps / b.speed_mbps) - 1.0) * 100.0;
            scorecard.push(serde_json::json!({
                "platform": plat,
                "scenario": format!("{ds}-{arch} {thr}"),
                "gzippy_mbps": g.speed_mbps,
                "best_competitor": b.tool,
                "competitor_mbps": b.speed_mbps,
                "gap_pct": (gap * 10.0).round() / 10.0,
                "verdict": if gap >= 0.0 { "WIN" } else { "LOSS" },
            }));
        }
    }

    let wins = scorecard.iter().filter(|s| s["verdict"] == "WIN").count();
    let losses = scorecard.iter().filter(|s| s["verdict"] == "LOSS").count();

    let output = serde_json::json!({
        "timestamp": chrono_now(),
        "wall_time_secs": wall_time_secs,
        "total_results": items.len(),
        "wins": wins,
        "losses": losses,
        "total_scenarios": wins + losses,
        "results": items,
        "scorecard": scorecard,
    });

    let json_path = "cloud-results.json";
    if let Ok(json_str) = serde_json::to_string_pretty(&output) {
        if std::fs::write(json_path, &json_str).is_ok() {
            println!("  Results written to {json_path} ({} scenarios, {wins}W/{losses}L)", wins + losses);
        }
    }
}

fn chrono_now() -> String {
    let output = Command::new("date").arg("+%Y-%m-%dT%H:%M:%S%z")
        .output().ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    output
}

// ─── Public API ───────────────────────────────────────────────────────────────

pub fn bench() -> Result<(), String> {
    let n_instances = DATASETS.len() * 2 * 2; // datasets × archs × directions
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  CLOUD BENCHMARK FLEET — {} INSTANCES (decompress + compress)           ║", n_instances);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    let _ = Command::new("aws-vault").arg("--version")
        .output().map_err(|_| "aws-vault not found")?;

    let commit = cmd_output("git", &["rev-parse", "HEAD"])?;
    let commit_short = &commit[..8.min(commit.len())];
    let branch = cmd_output("git", &["branch", "--show-current"]).unwrap_or_else(|_| "detached".into());
    let raw_repo_url = cmd_output("git", &["remote", "get-url", "origin"])?;
    let repo_url = if raw_repo_url.starts_with("git@github.com:") {
        raw_repo_url.replace("git@github.com:", "https://github.com/")
    } else {
        raw_repo_url.clone()
    };

    let remote_check = cmd_output("git", &["branch", "-r", "--contains", "HEAD"]);
    if remote_check.map(|s| s.trim().is_empty()).unwrap_or(true) {
        return Err(format!(
            "HEAD ({commit_short}) is not pushed. Run `git push` first."
        ));
    }

    println!("  Branch:    {branch}");
    println!("  Commit:    {commit_short}");
    println!("  Fleet:     {n_instances} instances ({} x86_64 {X86_TYPE} + {} arm64 {ARM64_TYPE})", n_instances / 2, n_instances / 2);
    println!("  Layout:    1 instance per (arch, dataset, direction) — fully parallel");
    println!("  Benchmark: gzippy-dev bench --json (same code locally and in cloud)");
    println!("  Threads:   T1 + Tmax={BENCH_TMAX_THREADS} (matches CI runner core count)");
    println!("  Sweep:     {SWEEP_MIN_TRIALS}-{SWEEP_MAX_TRIALS} trials, CV<{:.1}%", SWEEP_TARGET_CV * 100.0);
    println!("  Precision: {PRECISION_MIN_TRIALS}-{PRECISION_MAX_TRIALS} trials, CV<{:.1}% (close races <{CLOSE_RACE_THRESHOLD}%)", PRECISION_TARGET_CV * 100.0);
    println!("  I/O:       /dev/shm (RAM-backed, no EBS bottleneck)");
    println!("  Scoring:   STRICT — gzippy must be >= every competitor");
    println!();

    let session = generate_session_id();
    let mut cleanup = CleanupState::new();
    let result = run_fleet(&commit, &repo_url, &session, &mut cleanup);

    println!("\n  Cleaning up...");
    cleanup.run();
    println!("  Done.");
    result
}

const DATASETS: &[&str] = &["silesia", "software", "logs"];

fn run_fleet(
    commit: &str, repo_url: &str, session: &str, cleanup: &mut CleanupState,
) -> Result<(), String> {
    let total_start = Instant::now();

    print!("  VPC/subnet... ");
    let _ = std::io::stdout().flush();
    let (vpc_id, subnet_id) = find_vpc_and_subnet()?;
    println!("{vpc_id}/{subnet_id}");

    print!("  SSH key... ");
    let _ = std::io::stdout().flush();
    let (key_name, key_path) = create_key_pair(session)?;
    cleanup.key_name = Some(key_name.clone());
    cleanup.key_path = Some(key_path.clone());
    println!("OK");

    print!("  Security group... ");
    let _ = std::io::stdout().flush();
    let sg_id = create_security_group(session, &vpc_id)?;
    cleanup.sg_id = Some(sg_id.clone());
    println!("{sg_id}");

    print!("  AMIs... ");
    let _ = std::io::stdout().flush();
    let x86_ami = find_ubuntu_ami("x86_64")?;
    let arm64_ami = find_ubuntu_ami("arm64")?;
    println!("x86={x86_ami} arm64={arm64_ami}");

    let userdata = user_data_script(commit, repo_url);
    let userdata_b64 = base64_encode(&userdata);

    struct Instance {
        id: String,
        ip: String,
        arch: &'static str,
        dataset: &'static str,
        instance_type: &'static str,
        direction: &'static str,
    }
    let mut instances: Vec<Instance> = Vec::new();

    // Launch instances: one per (arch, dataset, direction) for full parallelism
    // This gives us 12 instances: 3 datasets × 2 archs × 2 directions
    let directions = ["decompress", "compress"];
    for &dataset in DATASETS {
        for (arch, itype, ami) in [
            ("x86_64", X86_TYPE, &x86_ami),
            ("arm64", ARM64_TYPE, &arm64_ami),
        ] {
            for &direction in &directions {
                let label = format!("{arch}/{dataset}/{direction}");
                print!("  Launching {label} ({itype})... ");
                let _ = std::io::stdout().flush();
                let id = launch_on_demand(itype, ami, &key_name, &sg_id, &subnet_id, &userdata_b64, session)?;
                cleanup.instance_ids.push(id.clone());
                println!("{id}");
                instances.push(Instance { id, ip: String::new(), arch, dataset, instance_type: itype, direction });
            }
        }
    }

    let all_ids: Vec<String> = instances.iter().map(|i| i.id.clone()).collect();
    let id_refs: Vec<&str> = all_ids.iter().map(|s| s.as_str()).collect();

    print!("  Waiting for {} instances... ", instances.len());
    let _ = std::io::stdout().flush();
    wait_running(&id_refs)?;
    println!("OK");

    for inst in &mut instances {
        inst.ip = get_public_ip(&inst.id)?;
        println!("  {}/{}/{}: {} ({})", inst.arch, inst.dataset, inst.direction, inst.ip, inst.instance_type);
    }

    // Wait for SSH + setup on all instances in parallel
    {
        let handles: Vec<_> = instances.iter().map(|inst| {
            let ip = inst.ip.clone();
            let key = key_path.clone();
            let label = format!("{}/{}/{}", inst.arch, inst.dataset, inst.direction);
            std::thread::spawn(move || -> Result<(), String> {
                wait_for_ssh(&ip, &key, &label)?;
                wait_for_setup(&ip, &key, &label)
            })
        }).collect();
        for h in handles {
            h.join().map_err(|_| "setup thread panic")??;
        }
    }

    println!(
        "\n  ═══ BENCHMARKS ({} cloud instances + local Mac in parallel) ═══",
        instances.len()
    );

    let cloud_handles: Vec<_> = instances
        .iter()
        .map(|inst| {
            let ip = inst.ip.clone();
            let key = key_path.clone();
            let label = format!("{}/{}/{}", inst.arch, inst.dataset, inst.direction);
            let dataset = inst.dataset.to_string();
            let direction = inst.direction.to_string();
            std::thread::spawn(move || run_benchmarks_on(&ip, &key, &label, Some(&dataset), &direction))
        })
        .collect();

    // Local Mac benchmark runs both decompress + compress in parallel with cloud
    let local_handle = std::thread::spawn(|| -> Result<Vec<BenchResult>, String> {
        let args = bench::BenchArgs {
            min_trials: SWEEP_MIN_TRIALS,
            max_trials: SWEEP_MAX_TRIALS,
            target_cv: SWEEP_TARGET_CV,
            direction: BenchDirection::Both,
            ..Default::default()
        };
        let (platform, results) = bench::run_and_collect(&args)?;
        let label = format!("local-{platform}");
        Ok(results
            .into_iter()
            .filter(|r| r.status == "pass")
            .map(|r| BenchResult {
                platform: label.clone(),
                dataset: r.dataset,
                archive: r.archive,
                threads: r.threads,
                tool: r.tool,
                speed_mbps: r.speed_mbps,
                cv: r.cv,
                trials: r.trials,
                direction: r.direction,
            })
            .collect())
    });

    let mut all = Vec::new();
    for h in cloud_handles {
        all.extend(h.join().map_err(|_| "cloud bench thread panic")??);
    }
    match local_handle.join() {
        Ok(Ok(local_results)) => {
            println!("\n  [local] {} results collected", local_results.len());
            all.extend(local_results);
        }
        Ok(Err(e)) => eprintln!("\n  [local] Benchmark failed: {e}"),
        Err(_) => eprintln!("\n  [local] Benchmark thread panicked"),
    }

    let elapsed = total_start.elapsed();
    println!("\n  Total wall time: {:.0}s ({:.1} min)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);

    print_results(&all);
    dump_results_json(&all, elapsed.as_secs_f64());
    Ok(())
}

// ─── Utilities ────────────────────────────────────────────────────────────────

fn cmd_output(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program).args(args)
        .output().map_err(|e| format!("{program}: {e}"))?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn generate_session_id() -> String {
    use std::time::SystemTime;
    let t = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis();
    format!("{:x}", t & 0xFFFFFFFF)
}

fn base64_encode(input: &str) -> String {
    let mut child = Command::new("base64")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("base64 command not found");
    {
        child.stdin.take().unwrap().write_all(input.as_bytes()).unwrap();
    }
    let output = child.wait_with_output().expect("base64 failed");
    String::from_utf8_lossy(&output.stdout).replace('\n', "")
}

pub fn cleanup_all() -> Result<(), String> {
    println!("  Searching for leaked gzippy-bench resources...");
    let result = aws(&[
        "ec2", "describe-instances",
        "--filters",
            &format!("Name=tag-key,Values={TAG_KEY}"),
            "Name=instance-state-name,Values=pending,running,stopping,stopped",
        "--query", "Reservations[].Instances[].InstanceId",
        "--output", "text",
    ])?;
    let ids: Vec<&str> = result.split_whitespace().filter(|s| !s.is_empty()).collect();
    if ids.is_empty() {
        println!("  No leaked instances.");
    } else {
        println!("  Terminating {} instance(s): {:?}", ids.len(), ids);
        let mut args = vec!["ec2", "terminate-instances", "--instance-ids"];
        args.extend(ids);
        aws(&args)?;
    }
    let result = aws(&[
        "ec2", "describe-security-groups",
        "--filters", &format!("Name=group-name,Values={TAG_KEY}-*"),
        "--query", "SecurityGroups[].GroupId", "--output", "text",
    ])?;
    for sg in result.split_whitespace().filter(|s| !s.is_empty()) {
        println!("  Deleting security group {sg}...");
        let _ = aws(&["ec2", "delete-security-group", "--group-id", sg]);
    }
    let result = aws(&[
        "ec2", "describe-key-pairs",
        "--filters", &format!("Name=key-name,Values={TAG_KEY}-*"),
        "--query", "KeyPairs[].KeyName", "--output", "text",
    ])?;
    for key in result.split_whitespace().filter(|s| !s.is_empty()) {
        println!("  Deleting key pair {key}...");
        let _ = aws(&["ec2", "delete-key-pair", "--key-name", key]);
    }
    println!("  Cleanup complete.");
    Ok(())
}
