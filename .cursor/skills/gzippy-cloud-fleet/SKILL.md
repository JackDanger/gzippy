---
name: gzippy-cloud-fleet
description: Launch and manage the gzippy EC2 benchmark fleet for high-precision performance measurements. Use when running cloud benchmarks, debugging fleet setup, cleaning up AWS resources, or needing authoritative performance numbers on dedicated hardware.
---

# gzippy Cloud Fleet Benchmarking

## Quick Start

```bash
source .env                      # AWS credentials (or aws-vault)
gzippy-dev cloud bench           # Launch fleet, benchmark, tear down
```

If `.env` doesn't exist, use `aws-vault exec personal -d 12h -- bash` to get
a shell with credentials, then export `AWS_*` vars into `.env`.

## Architecture

**12 instances**, fully parallel:
- 6 × `c7i.4xlarge` (x86_64, Intel Sapphire Rapids)
- 6 × `c8g.4xlarge` (arm64, AWS Graviton4)

Each instance handles ONE `(arch, dataset, direction)` combination:

| Instance | Arch | Dataset | Direction |
|----------|------|---------|-----------|
| 1 | x86_64 | silesia | decompress |
| 2 | x86_64 | silesia | compress |
| 3 | arm64 | silesia | decompress |
| 4 | arm64 | silesia | compress |
| 5-8 | ... | software | ... |
| 9-12 | ... | logs | ... |

All benchmark data goes to `/dev/shm` (RAM-backed) to eliminate EBS I/O.

## What the Fleet Does

1. **Creates infrastructure**: VPC check, SSH key pair, security group
2. **Launches instances**: 12 on-demand instances with user-data script
3. **User-data script** (runs on each instance):
   - Installs build deps + Rust
   - Clones repo at HEAD commit (must be pushed first)
   - Inits submodules: isa-l, libdeflate, pigz, rapidgzip, zopfli (NOT gzip)
   - Builds all competitor tools + gzippy + gzippy-dev
   - Prepares benchmark data, compresses for decompression tests
   - Stages everything to `/dev/shm`
   - Writes `/tmp/gzippy-ready` marker
4. **Runs benchmarks**: `gzippy-dev bench --json` on each instance
5. **Streams results**: Prints to stdout as they arrive
6. **Tears down**: Terminates instances, deletes key pair, security group

## Reading Output

Results stream in real-time with very low noise:
```
silesia-bgzf T1 gzippy: 575.2 MB/s (CV 0.3%)
silesia-bgzf T1 igzip: 521.0 MB/s (CV 0.3%)
silesia-bgzf T1 rapidgzip: 477.4 MB/s (CV 0.2%)
```

**CV < 1.5%** on dedicated instances vs 5-30% on shared CI runners.
These numbers are authoritative for absolute speed comparisons.

## Troubleshooting

### Fleet setup fails / instances stuck in "Waiting for setup..."

SSH into an instance to check cloud-init logs:
```bash
# Find the SSH key (written to OS temp dir)
ls $(python3 -c "import tempfile; print(tempfile.gettempdir())")/gzippy-bench-*.pem

# Get an instance IP
source .env && aws ec2 describe-instances --region us-east-1 \
  --filters "Name=tag-key,Values=gzippy-bench" "Name=instance-state-name,Values=running" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text

# SSH in and check
ssh -i <KEY> ubuntu@<IP> "tail -50 /var/log/cloud-init-output.log"
```

**Common failures:**
- `gnulib uses git:// protocol`: Fixed — we skip the gzip submodule
- `gzippy: No such file or directory`: Need `cargo build --release` in user-data
- `cmake failed`: Missing build dep in apt-get list

### Leaked resources after kill -9

```bash
source .env && gzippy-dev cloud cleanup
```

This finds and deletes:
- Running instances tagged `gzippy-bench`
- Security groups named `gzippy-bench-*`
- Key pairs named `gzippy-bench-*`

### AWS credentials

The `aws()` function in `cloud.rs` auto-detects:
- If `AWS_ACCESS_KEY_ID` is in env → calls `aws` directly
- Otherwise → wraps with `aws-vault exec personal -d 12h --`

## Key Files

| File | Purpose |
|------|---------|
| `tools/devtool/src/cloud.rs` | Fleet orchestration (launch, SSH, benchmark, cleanup) |
| `tools/devtool/src/bench.rs` | Benchmark logic (shared by local + cloud) |
| `scripts/build-tools.sh` | Builds competitor tools (used in user-data) |
| `scripts/prepare_benchmark_data.sh` | Creates silesia/software/logs datasets |

## Cost

~$0.50-1.00 per fleet run (12 instances × ~20 min at on-demand pricing).
Instances auto-terminate after benchmarks complete or on error.
