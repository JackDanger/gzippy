//! Cloud fleet benchmarking: spin up EC2 on-demand instances, run benchmarks, tear down.
//!
//! Optimized for lowest latency:
//!   - On-demand instances (no spot capacity waits)
//!   - Larger instances (faster builds, more cores)
//!   - RAM-backed I/O (/dev/shm) to eliminate EBS bottleneck
//!   - Both platforms run fully in parallel
//!   - Two-phase benchmarks: sweep then precision re-runs for close races
//!
//! Uses `aws-vault exec personal` for credentials.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const REGION: &str = "us-east-1";
// 16 vCPU for fast builds; benchmarks use -p4 to match CI runner core count
const X86_TYPE: &str = "c6i.4xlarge";
const ARM64_TYPE: &str = "c7g.4xlarge";
const BENCH_THREADS: usize = 4;
const TAG_KEY: &str = "gzippy-bench";
const SSH_USER: &str = "ubuntu";
const SETUP_TIMEOUT: Duration = Duration::from_secs(20 * 60);

// Phase 1: initial sweep
const SWEEP_MIN_TRIALS: u32 = 15;
const SWEEP_MAX_TRIALS: u32 = 60;
const SWEEP_TARGET_CV: f64 = 0.02;

// Phase 2: precision re-run for close races
const PRECISION_MIN_TRIALS: u32 = 50;
const PRECISION_MAX_TRIALS: u32 = 200;
const PRECISION_TARGET_CV: f64 = 0.005;
// If gzippy is within this % of a competitor, it's a "close race" needing re-run
const CLOSE_RACE_THRESHOLD: f64 = 3.0;

// ─── AWS CLI ──────────────────────────────────────────────────────────────────

fn aws(args: &[&str]) -> Result<String, String> {
    let output = Command::new("aws-vault")
        .args(["exec", "personal", "--"])
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

#[allow(dead_code)]
fn ssh_stream(
    ip: &str, key: &Path, cmd: &str, prefix: &str,
) -> Result<Vec<String>, String> {
    let mut child = Command::new("ssh")
        .args(ssh_opts(key))
        .arg(format!("{SSH_USER}@{ip}"))
        .arg(cmd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("SSH spawn to {ip}: {e}"))?;

    let stdout = child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);
    let mut lines = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Read error: {e}"))?;
        println!("  {prefix} {line}");
        lines.push(line);
    }
    Ok(lines)
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
    let arch_filter = match arch { "x86_64" => "amd64", "arm64" => "arm64", _ => return Err(format!("Unknown arch: {arch}")) };
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
        if elapsed % 30 == 0 && elapsed > 0 {
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

# Flat bin directory
mkdir -p /home/{SSH_USER}/gzippy/bin
cd /home/{SSH_USER}/gzippy/bin
ln -sf ../target/release/gzippy gzippy
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

// ─── Benchmark execution ─────────────────────────────────────────────────────

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
}

fn scenario_key(r: &BenchResult) -> String {
    format!("{}-{}-{}-{}", r.platform, r.dataset, r.archive, r.threads)
}

fn run_one_benchmark(
    ip: &str, key: &Path, label: &str, repo: &str,
    dataset: &str, archive: &str, threads: usize, threads_label: &str,
    min_trials: u32, max_trials: u32, target_cv: f64,
) -> Result<Vec<BenchResult>, String> {
    let raw_file = match dataset {
        "silesia" => "/dev/shm/silesia.tar",
        "software" => "/dev/shm/software.archive",
        "logs" => "/dev/shm/logs.txt",
        _ => return Err(format!("Unknown dataset: {dataset}")),
    };
    let compressed = format!("/dev/shm/{dataset}-{archive}.gz");
    let json_file = format!("/dev/shm/bench-{dataset}-{archive}-{threads}.json");

    // Verify files exist
    let check = ssh(ip, key, &format!(
        "test -f {compressed} && test -f {raw_file} && echo OK || echo MISSING"
    ))?;
    if check.trim() != "OK" {
        eprintln!("  [{label}] SKIP {dataset}-{archive} {threads_label}: files missing");
        return Ok(Vec::new());
    }

    let run_cmd = format!(
        "cd {repo} && source $HOME/.cargo/env && \
         TMPDIR=/dev/shm python3 scripts/benchmark_decompression.py \
         --binaries {repo}/bin \
         --compressed-file {compressed} \
         --original-file {raw_file} \
         --threads {threads} \
         --archive-type {dataset}-{archive} \
         --min-trials {min_trials} \
         --max-trials {max_trials} \
         --target-cv {target_cv} \
         --output {json_file} 2>&1"
    );
    let _ = ssh(ip, key, &run_cmd);

    let output = ssh(ip, key, &format!("cat {json_file} 2>/dev/null || echo '{{}}' "))?;
    let trimmed = output.trim();

    let mut results = Vec::new();
    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(parsed) => {
            if let Some(items) = parsed.get("results").and_then(|v| v.as_array()) {
                for item in items {
                    let obj = match item.as_object() { Some(o) => o, None => continue };
                    let tool = obj.get("tool").and_then(|v| v.as_str()).unwrap_or("?").to_string();
                    if obj.get("status").and_then(|v| v.as_str()) != Some("pass") {
                        let err = obj.get("error").and_then(|v| v.as_str()).unwrap_or("failed");
                        println!("    {:<12} {}", tool, err);
                        continue;
                    }
                    let speed = obj.get("speed_mbps").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let cv = obj.get("cv").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let trials = obj.get("trials").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

                    results.push(BenchResult {
                        platform: label.to_string(),
                        dataset: dataset.to_string(),
                        archive: archive.to_string(),
                        threads: threads_label.to_string(),
                        tool,
                        speed_mbps: speed,
                        cv,
                        trials,
                    });
                }
            }
        }
        Err(e) => {
            eprintln!("  [{label}] JSON error for {dataset}-{archive} {threads_label}: {e}");
        }
    }
    Ok(results)
}

fn run_benchmarks_on(
    ip: &str, key: &Path, label: &str,
) -> Result<Vec<BenchResult>, String> {
    let datasets = ["silesia", "software", "logs"];
    let archives = ["gzip", "bgzf", "pigz"];
    let thread_configs = [1usize, BENCH_THREADS];
    let repo = format!("/home/{SSH_USER}/gzippy");

    // ── Phase 1: Initial sweep ──
    println!("\n  [{label}] ── Phase 1: Sweep ({SWEEP_MIN_TRIALS}-{SWEEP_MAX_TRIALS} trials, CV<{:.0}%) ──",
        SWEEP_TARGET_CV * 100.0);

    let mut all_results: Vec<BenchResult> = Vec::new();

    for dataset in &datasets {
        for archive in &archives {
            for &threads in &thread_configs {
                let tl = if threads == 1 { "T1" } else { "Tmax" };
                print!("  [{label}] {dataset}-{archive} {tl}  ");
                let _ = std::io::stdout().flush();

                let results = run_one_benchmark(
                    ip, key, label, &repo,
                    dataset, archive, threads, tl,
                    SWEEP_MIN_TRIALS, SWEEP_MAX_TRIALS, SWEEP_TARGET_CV,
                )?;

                for r in &results {
                    print!("{}:{:.0}  ", r.tool, r.speed_mbps);
                }
                println!();

                all_results.extend(results);
            }
        }
    }

    // ── Phase 2: Precision re-runs for close races ──
    let close_races = find_close_races(&all_results, label);
    if close_races.is_empty() {
        println!("  [{label}] ── Phase 2: No close races, all decisive ──");
    } else {
        println!("\n  [{label}] ── Phase 2: Re-running {} close race(s) ({PRECISION_MIN_TRIALS}-{PRECISION_MAX_TRIALS} trials, CV<{:.1}%) ──",
            close_races.len(), PRECISION_TARGET_CV * 100.0);

        for (dataset, archive, threads_label) in &close_races {
            let threads: usize = if threads_label == "T1" { 1 } else { BENCH_THREADS };
            print!("  [{label}] PRECISION {dataset}-{archive} {threads_label}  ");
            let _ = std::io::stdout().flush();

            let results = run_one_benchmark(
                ip, key, label, &repo,
                dataset, archive, threads, threads_label,
                PRECISION_MIN_TRIALS, PRECISION_MAX_TRIALS, PRECISION_TARGET_CV,
            )?;

            for r in &results {
                print!("{}:{:.1}  ", r.tool, r.speed_mbps);
            }
            println!();

            // Replace sweep results with precision results
            let scenario = format!("{label}-{dataset}-{archive}-{threads_label}");
            all_results.retain(|r| scenario_key(r) != scenario);
            all_results.extend(results);
        }
    }

    Ok(all_results)
}

/// Find scenarios where gzippy is within CLOSE_RACE_THRESHOLD% of any competitor
fn find_close_races(results: &[BenchResult], platform: &str) -> Vec<(String, String, String)> {
    let mut races = Vec::new();
    let mut scenarios: Vec<(String, String, String)> = results.iter()
        .filter(|r| r.platform == platform)
        .map(|r| (r.dataset.clone(), r.archive.clone(), r.threads.clone()))
        .collect();
    scenarios.sort();
    scenarios.dedup();

    for (ds, arch, thr) in scenarios {
        let scenario: Vec<&BenchResult> = results.iter()
            .filter(|r| r.platform == platform && r.dataset == ds && r.archive == arch && r.threads == thr)
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
    println!("\n╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  CLOUD FLEET RESULTS — STRICT SCORING (no parity, gzippy must win or it's a loss) ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝\n");

    let mut platforms: Vec<&str> = results.iter().map(|r| r.platform.as_str()).collect();
    platforms.sort();
    platforms.dedup();

    let mut total_wins = 0u32;
    let mut total_losses = 0u32;

    for platform in &platforms {
        println!("  ── {platform} ──");
        println!("  {:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>6} {}",
            "Scenario", "gzippy", "unpigz", "igzip", "rapidgzip", "gzip", "CV%", "Verdict");
        println!("  {}", "─".repeat(100));

        let plat_results: Vec<&BenchResult> = results.iter()
            .filter(|r| r.platform == *platform)
            .collect();

        let mut scenarios: Vec<(String, String, String)> = plat_results.iter()
            .map(|r| (r.dataset.clone(), r.archive.clone(), r.threads.clone()))
            .collect();
        scenarios.sort();
        scenarios.dedup();

        for (dataset, archive, threads) in &scenarios {
            let scenario: Vec<&&BenchResult> = plat_results.iter()
                .filter(|r| r.dataset == *dataset && r.archive == *archive && r.threads == *threads)
                .collect();

            let get = |tool: &str| -> Option<&BenchResult> {
                scenario.iter().find(|r| r.tool == tool).copied().copied()
            };
            let fmt_speed = |tool: &str| -> String {
                get(tool).map(|r| format!("{:.1}", r.speed_mbps)).unwrap_or_else(|| "—".to_string())
            };

            let gzippy = get("gzippy");
            let gzippy_cv = gzippy.map(|r| r.cv).unwrap_or(0.0);

            // Find best competitor and compute gap
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
        println!();
    }

    let total = total_wins + total_losses;
    println!("  ══════════════════════════════════════════════════════");
    println!("  WINS: {total_wins}/{total}    LOSSES: {total_losses}/{total}");
    if total_losses > 0 {
        println!("\n  ── LOSSES (gzippy slower than best competitor) ──");
        let mut scenarios: Vec<(String, String, String, String)> = results.iter()
            .map(|r| (r.platform.clone(), r.dataset.clone(), r.archive.clone(), r.threads.clone()))
            .collect();
        scenarios.sort();
        scenarios.dedup();

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
                if gap < 0.0 {
                    println!("    [{plat}] {ds}-{arch} {thr}: gzippy {:.1} vs {} {:.1} ({:+.1}%)",
                        g.speed_mbps, b.tool, b.speed_mbps, gap);
                }
            }
        }
    } else {
        println!("  CLEAN SWEEP — gzippy wins every scenario!");
    }
    println!();
}

// ─── Public API ───────────────────────────────────────────────────────────────

pub fn bench() -> Result<(), String> {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  CLOUD BENCHMARK FLEET — ON-DEMAND, FULL PARALLELISM                   ║");
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

    println!("  Branch:   {branch}");
    println!("  Commit:   {commit_short}");
    println!("  x86_64:   {X86_TYPE} on-demand (16 vCPU, bench with -p{BENCH_THREADS})");
    println!("  arm64:    {ARM64_TYPE} on-demand (16 vCPU, bench with -p{BENCH_THREADS})");
    println!("  Sweep:    {SWEEP_MIN_TRIALS}-{SWEEP_MAX_TRIALS} trials, CV<{:.0}%", SWEEP_TARGET_CV * 100.0);
    println!("  Precision: {PRECISION_MIN_TRIALS}-{PRECISION_MAX_TRIALS} trials, CV<{:.1}% (for close races <{CLOSE_RACE_THRESHOLD}%)", PRECISION_TARGET_CV * 100.0);
    println!("  I/O:      /dev/shm (RAM-backed, no EBS bottleneck)");
    println!("  Scoring:  STRICT — gzippy must be >= every competitor, no parity threshold");
    println!();

    let session = generate_session_id();
    let mut cleanup = CleanupState::new();
    let result = run_fleet(&commit, &repo_url, &session, &mut cleanup);

    println!("\n  Cleaning up...");
    cleanup.run();
    println!("  Done.");
    result
}

fn run_fleet(
    commit: &str, repo_url: &str, session: &str, cleanup: &mut CleanupState,
) -> Result<(), String> {
    let total_start = Instant::now();

    // All resource creation runs sequentially (fast API calls)
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

    // AMI lookup in parallel
    print!("  AMIs... ");
    let _ = std::io::stdout().flush();
    let x86_ami = find_ubuntu_ami("x86_64")?;
    let arm64_ami = find_ubuntu_ami("arm64")?;
    println!("x86={x86_ami} arm64={arm64_ami}");

    let userdata = user_data_script(commit, repo_url);
    let userdata_b64 = base64_encode(&userdata);

    // Launch both instances
    print!("  Launching x86_64 ({X86_TYPE})... ");
    let _ = std::io::stdout().flush();
    let x86_id = launch_on_demand(X86_TYPE, &x86_ami, &key_name, &sg_id, &subnet_id, &userdata_b64, session)?;
    cleanup.instance_ids.push(x86_id.clone());
    println!("{x86_id}");

    print!("  Launching arm64 ({ARM64_TYPE})... ");
    let _ = std::io::stdout().flush();
    let arm64_id = launch_on_demand(ARM64_TYPE, &arm64_ami, &key_name, &sg_id, &subnet_id, &userdata_b64, session)?;
    cleanup.instance_ids.push(arm64_id.clone());
    println!("{arm64_id}");

    // Wait for both to be running
    print!("  Waiting for running state... ");
    let _ = std::io::stdout().flush();
    wait_running(&[&x86_id, &arm64_id])?;
    println!("OK");

    let x86_ip = get_public_ip(&x86_id)?;
    let arm64_ip = get_public_ip(&arm64_id)?;
    println!("  x86_64: {x86_ip}");
    println!("  arm64:  {arm64_ip}");

    // SSH + setup in parallel
    {
        let k1 = key_path.clone(); let ip1 = x86_ip.clone();
        let k2 = key_path.clone(); let ip2 = arm64_ip.clone();
        let t1 = std::thread::spawn(move || wait_for_ssh(&ip1, &k1, "x86_64"));
        let t2 = std::thread::spawn(move || wait_for_ssh(&ip2, &k2, "arm64"));
        t1.join().map_err(|_| "thread panic")??;
        t2.join().map_err(|_| "thread panic")??;
    }
    {
        let k1 = key_path.clone(); let ip1 = x86_ip.clone();
        let k2 = key_path.clone(); let ip2 = arm64_ip.clone();
        let t1 = std::thread::spawn(move || wait_for_setup(&ip1, &k1, "x86_64"));
        let t2 = std::thread::spawn(move || wait_for_setup(&ip2, &k2, "arm64"));
        t1.join().map_err(|_| "thread panic")??;
        t2.join().map_err(|_| "thread panic")??;
    }

    // Run benchmarks on both platforms in parallel
    println!("\n  ═══ BENCHMARKS (both platforms in parallel) ═══");

    let k1 = key_path.clone(); let ip1 = x86_ip.clone();
    let k2 = key_path.clone(); let ip2 = arm64_ip.clone();

    let t1 = std::thread::spawn(move || run_benchmarks_on(&ip1, &k1, "x86_64"));
    let t2 = std::thread::spawn(move || run_benchmarks_on(&ip2, &k2, "arm64"));

    let mut all = Vec::new();
    all.extend(t1.join().map_err(|_| "x86 thread panic")??);
    all.extend(t2.join().map_err(|_| "arm64 thread panic")??);

    let elapsed = total_start.elapsed();
    println!("\n  Total wall time: {:.0}s ({:.1} min)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);

    print_results(&all);
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
