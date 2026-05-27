# gzippy - The Fastest Parallel Gzip
# Build and test infrastructure
#
# Quick tests (<30s) run with 'make' or 'make quick' - for AI tools and iteration
# Full perf tests (10+ min) run with 'make perf-full' - for humans at release time

# Build configuration - submodules are in ./vendor/{gzip,pigz,isa-l,rapidgzip,libdeflate}
GZIP_DIR := ./vendor/gzip
PIGZ_DIR := ./vendor/pigz
ISAL_DIR := ./vendor/isa-l
RAPIDGZIP_DIR := ./vendor/rapidgzip
GZIPPY_DIR := .
TEST_DATA_DIR := test_data
RESULTS_DIR := test_results

# Build targets
GZIPPY_BIN := $(GZIPPY_DIR)/target/release/gzippy
UNGZIPPY_BIN := $(GZIPPY_DIR)/target/release/ungzippy
PIGZ_BIN := $(PIGZ_DIR)/pigz
IGZIP_BIN := $(ISAL_DIR)/build/igzip
RAPIDGZIP_BIN := $(RAPIDGZIP_DIR)/librapidarchive/build/src/tools/rapidgzip

# Prefer source-built gzip; on macOS fall back to brew gzip then /usr/bin/gzip
GZIP_BIN := $(shell if [ -x $(GZIP_DIR)/gzip ]; then echo $(GZIP_DIR)/gzip; elif [ -x /opt/homebrew/bin/gzip ]; then echo /opt/homebrew/bin/gzip; else which gzip; fi)
SYSTEM_GZIP := $(shell which gzip)
# macOS ships its own NEON-accelerated gzip at /usr/bin/gzip (Apple gzip 479)
# Set APPLE_GZIP when it differs from GZIP_BIN so bench targets compare both
APPLE_GZIP := $(if $(filter Darwin,$(shell uname)),/usr/bin/gzip,)

# Time budgets (seconds). Each nontrivial target wraps its work in
# `timeout` so a hang or runaway build fails the target instead of
# blocking forever. The number is the expectation: blowing the budget is
# a regression to investigate, not a knob to bump.
#   QUICK    — local quick-check stage, typically <30s
#   BENCH_SM — neurotic: build + single-member benchmark
#   TEST_X86 — neurotic: build + the x86_64-gated test subset
#   SHIP     — neurotic: build + gzippy-dev bench + L11 head-to-head
QUICK_TIMEOUT    := 300
BENCH_SM_TIMEOUT := 1200
TEST_X86_TIMEOUT := 1500
SHIP_TIMEOUT     := 2700

.PHONY: all build quick quick-wallclock update-baselines perf-full test-data test-data-quick clean help validate deps ship ship-local route-check oracle-vs-c bench-sm \
	profile-single-member-decompression-x86_64 profile-single-member-decompression-arm64 profile-decompression-x86_64

# =============================================================================
# Default target: quick benchmark for fast iteration (< 30 seconds)
# =============================================================================
all: quick

# =============================================================================
# Build targets
# =============================================================================

build: $(GZIPPY_BIN) $(UNGZIPPY_BIN)

deps: $(PIGZ_BIN) $(IGZIP_BIN) $(RAPIDGZIP_BIN)
	@# Try to build gzip, but don't fail if it doesn't work
	@$(MAKE) $(GZIP_DIR)/gzip 2>/dev/null || true
	@echo "✓ Dependencies ready (gzip, pigz, igzip, rapidgzip)"

$(GZIP_DIR)/gzip:
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "⚠ gzip submodule needs autotools — skipping source build on macOS"; \
		echo "  Using $(GZIP_BIN) (run 'brew install gzip' for GNU gzip)"; \
	else \
		echo "Building gzip from source..."; \
		cd $(GZIP_DIR) && find . -name "*.in" -exec touch {} \; 2>/dev/null; \
		touch configure aclocal.m4 Makefile.in 2>/dev/null || true; \
		cd $(GZIP_DIR) && ./configure --quiet 2>/dev/null || true; \
		if $(MAKE) -C $(GZIP_DIR) -j4 2>/dev/null; then \
			echo "✓ Built gzip from source"; \
		else \
			echo "⚠ gzip build failed, using system gzip: $(SYSTEM_GZIP)"; \
		fi; \
	fi

$(PIGZ_BIN):
	@echo "Building pigz from source..."
	@$(MAKE) -C $(PIGZ_DIR) pigz 2>&1 || (echo "  Cleaning and rebuilding..." && $(MAKE) -C $(PIGZ_DIR) clean >/dev/null 2>&1 && $(MAKE) -C $(PIGZ_DIR) pigz)
	@echo "✓ Built pigz"

$(IGZIP_BIN):
	@echo "Building igzip (ISA-L) from source..."
	@mkdir -p $(ISAL_DIR)/build
	@cd $(ISAL_DIR)/build && cmake .. >/dev/null 2>&1 && make -j4 igzip 2>&1 | grep -E "(Built|error)" || true
	@echo "✓ Built igzip"

$(RAPIDGZIP_BIN):
	@echo "Building rapidgzip from source..."
	@cd $(RAPIDGZIP_DIR) && git submodule update --init --recursive >/dev/null 2>&1 || true
	@mkdir -p $(RAPIDGZIP_DIR)/librapidarchive/build
	@cd $(RAPIDGZIP_DIR)/librapidarchive/build && cmake .. >/dev/null 2>&1 && make -j4 rapidgzip 2>&1 | grep -E "(Built|Linking|error)" || true
	@echo "✓ Built rapidgzip"

$(GZIPPY_BIN): FORCE
	@echo "Building gzippy..."
	@cd $(GZIPPY_DIR) && cargo build --release 2>&1 | grep -E "(Compiling gzippy|Finished|error)"
	@test -f $(GZIPPY_BIN) || (echo "✗ gzippy build failed"; exit 1)
	@echo "✓ Built gzippy"

# Create ungzippy symlink (like unpigz)
$(UNGZIPPY_BIN): $(GZIPPY_BIN)
	@ln -sf gzippy $(UNGZIPPY_BIN)
	@echo "✓ Created ungzippy symlink"

FORCE:

# =============================================================================
# Quick test suite — deterministic, layered, <30 seconds.
# Replaces wall-clock benchmarks with proxies that fail specifically:
#   Stage 1: correctness + routing smoke (cargo test)
#   Stage 2: allocation budget (any new alloc on hot path = fail)
#   Stage 3: differential ratio vs libdeflate (cancels thermal noise)
#   Stage 4: hot-path hit rates in bgzf decoder
# See benchmarks/baselines.json for thresholds. Run 'make update-baselines'
# after an intentional perf change.
# =============================================================================
quick: $(GZIPPY_BIN)
	@echo "══ make quick ══════════════════════════════════════════"
	@echo "── Stage 1: correctness + routing smoke ────────────────"
	@set -o pipefail; timeout $(QUICK_TIMEOUT) cargo test --release correctness 2>&1 | tail -3
	@set -o pipefail; timeout $(QUICK_TIMEOUT) cargo test --release routing 2>&1 | tail -3
	@echo "── Stage 2: allocation budget ──────────────────────────"
	@set -o pipefail; timeout $(QUICK_TIMEOUT) cargo test --release alloc_budget 2>&1 | tail -3
	@echo "── Stage 3: differential ratio vs libdeflate ───────────"
	@set -o pipefail; timeout $(QUICK_TIMEOUT) cargo test --release diff_ratio 2>&1 | tail -3
	@echo "── Stage 4: hot-path hit rates ─────────────────────────"
	@set -o pipefail; timeout $(QUICK_TIMEOUT) cargo test --release hot_path 2>&1 | tail -3
	@echo "════════════════════════════════════════════════════════"
	@echo "✓ make quick passed"

# Record current measurements as new baselines. Run after intentional perf changes.
update-baselines: $(GZIPPY_BIN)
	@echo "Recording baselines — review benchmarks/baselines.json before committing."
	@RECORD_BASELINES=1 cargo test --release alloc_budget -- --nocapture 2>&1 | grep -E "baseline:|RECORD" || true
	@RECORD_BASELINES=1 cargo test --release diff_ratio -- --nocapture 2>&1 | grep -E "baseline:|RECORD" || true
	@RECORD_BASELINES=1 cargo test --release hot_path -- --nocapture 2>&1 | grep -E "baseline:|RECORD" || true
	@echo "Done."

# =============================================================================
# Wall-clock benchmark (~30 seconds) - manual use before 'make ship'
# =============================================================================
quick-wallclock: $(GZIPPY_BIN) $(UNGZIPPY_BIN) $(PIGZ_BIN) deps
	@python3 scripts/perf.py --sizes 1,10 --levels 6 --threads 1,4

# =============================================================================
# Route check: generate test files and show which decompression path is taken
# for each T1/T4 × 1MB/10MB combo, with timing vs pigz.
# Run before ANY decompression code change to verify routing.
# =============================================================================
route-check: $(GZIPPY_BIN) $(PIGZ_BIN)
	@python3 scripts/route_check.py $(GZIPPY_BIN) $(PIGZ_BIN)

# =============================================================================
# Single-member decompression profiling (correctness gate + CPU/alloc artifacts)
# =============================================================================
profile-single-member-decompression-x86_64:
	@chmod +x scripts/profile_single_member_decompression_x86_64.sh
	@bash scripts/profile_single_member_decompression_x86_64.sh

profile-single-member-decompression-arm64:
	@chmod +x scripts/profile_single_member_decompression_arm64.sh
	@bash scripts/profile_single_member_decompression_arm64.sh

profile-decompression-x86_64: profile-single-member-decompression-x86_64

# Remote homelab box (Intel i9-14000), reached via an SSH jump host. This
# is the only place the isal-compression / parallel single-member path
# builds and runs — local arm64 macOS cannot. Used by `ship`, `bench-sm`,
# and `test-x86_64`.
NEUROTIC_SSH := ssh -o ConnectTimeout=15 -J neurotic root@REDACTED_IP

# Shell snippet (run on neurotic) that hard-syncs the /root/gzippy checkout
# to origin/$BRANCH — $BRANCH must be set in the calling recipe's shell.
# The safe.directory exception is required because the checkout is not
# owned by the ssh user (git refuses "dubious ownership" repos otherwise).
NEUROTIC_SYNC := git config --global --add safe.directory /root/gzippy; git fetch origin '$$BRANCH'; git checkout -f -B '$$BRANCH' 'origin/$$BRANCH'; git reset --hard 'origin/$$BRANCH'

# =============================================================================
# Ship: the "are we good?" gate. Always tests the *current branch*.
#
# Runs the cheap local checks first, then expensive homelab bench last so
# a fmt/test/clippy/oracle failure aborts in <60s instead of >20min.
#
# Step 1: refuse if working tree dirty / detached HEAD
# Step 2: cargo fmt --check + test --release + clippy (default + oracle features)
# Step 3: corpus oracle vs vendor C zopfli (66 byte-equality checks; L11 ratio gate)
# Step 4: L11 head-to-head wall-clock + deflate equality vs C zopfli
# Step 5: cross-tool roundtrip smoke (gzippy ↔ gzip/pigz/igzip when available)
# Step 6: push current branch + homelab fetches THIS branch + bench
#
# Target: $(NEUROTIC_SSH) (homelab intel i9-14000)
# Time: ~25 minutes total (~30s through step 5; ~20-25 min on step 6)
# Use `make ship-local` to run steps 1-5 only and skip the homelab.
# =============================================================================
ship: ship-precheck ship-local
	@BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	echo ""; \
	echo "── Step 6/6: push '$$BRANCH' + neurotic homelab benchmarks ──"; \
	if ! git rev-parse origin/$$BRANCH >/dev/null 2>&1 \
	    || [ -n "$$(git log origin/$$BRANCH..HEAD 2>/dev/null)" ]; then \
	  echo "  pushing $$BRANCH to origin..."; \
	  git push origin $$BRANCH || (echo "PUSH FAILED — aborting ship" && exit 1); \
	else \
	  echo "  origin/$$BRANCH already up to date"; \
	fi; \
	echo "  connecting to neurotic..."; \
	timeout $(SHIP_TIMEOUT) $(NEUROTIC_SSH) "set -e; cd gzippy; \
	  echo '  fetching origin/$$BRANCH...'; \
	  $(NEUROTIC_SYNC); \
	  git submodule update --init --recursive 2>&1 | tail -3; \
	  echo ''; echo '  ── disk-space precheck ──'; \
	  AVAIL_GB=\$$(df -BG . | tail -1 | awk '{gsub(/G/,\"\",\$$4); print \$$4}'); \
	  echo \"  root fs: \$${AVAIL_GB} GB available\"; \
	  if [ \"\$$AVAIL_GB\" -lt 5 ]; then \
	    if [ -d /mnt/internal ] && [ -w /mnt/internal ]; then \
	      ALT=/mnt/internal/gzippy-target; mkdir -p \"\$$ALT\"; \
	      ALT_GB=\$$(df -BG \"\$$ALT\" | tail -1 | awk '{gsub(/G/,\"\",\$$4); print \$$4}'); \
	      echo \"  root fs squeezed; relocating cargo target to \$$ALT (\$${ALT_GB} GB free) via symlink\"; \
	      rm -rf target; ln -sfn \"\$$ALT\" target; \
	    else \
	      echo \"ERROR: root fs has only \$${AVAIL_GB} GB free; bench needs >=5 GB; aborting\" >&2; \
	      echo \"       free space (e.g. 'cargo clean' in /root/hvac-pr*/) and re-run\" >&2; \
	      exit 1; \
	    fi; \
	  fi; \
	  rm -rf /dev/shm/gzippy-bench-* 2>/dev/null || true; \
	  echo ''; echo '  ── building gzippy + gzippy-dev from $$BRANCH ──'; \
	  cargo build --release 2>&1 | grep -E 'Compiling gzippy |Finished|error' || true; \
	  cargo build --release --manifest-path tools/devtool/Cargo.toml --target-dir target 2>&1 \
	    | grep -E 'Compiling gzippy-dev|Finished|error' || true; \
	  [ -x target/release/gzippy     ] || { echo 'ERROR: target/release/gzippy missing after build' >&2; exit 1; }; \
	  [ -x target/release/gzippy-dev ] || { echo 'ERROR: target/release/gzippy-dev missing after build' >&2; exit 1; }; \
	  BD=benchmark_data; BIN=target/release/gzippy; \
	  [ -f \"\$$BD/silesia.tar\" ] || { [ -f \"\$$BD/silesia.tar.xz\" ] && echo 'extracting silesia.tar' && xz -dk \"\$$BD/silesia.tar.xz\" -c > \"\$$BD/silesia.tar\"; }; \
	  for DS in silesia software logs; do \
	    case \$$DS in \
	      silesia)  RAW=\$$BD/silesia.tar;; \
	      software) RAW=\$$BD/software.archive;; \
	      logs)     RAW=\$$BD/logs.txt;; \
	    esac; \
	    [ -f \"\$$RAW\" ] || { echo \"skipping \$$DS (\$$RAW not found)\"; continue; }; \
	    [ -s \"\$$BD/\$$DS-gzip.gz\" ] || { echo \"creating \$$BD/\$$DS-gzip.gz\"; gzip -1 -c \"\$$RAW\" > \"\$$BD/\$$DS-gzip.gz\"; }; \
	    [ -s \"\$$BD/\$$DS-bgzf.gz\" ] || { echo \"creating \$$BD/\$$DS-bgzf.gz\"; \$$BIN -1 -c \"\$$RAW\" > \"\$$BD/\$$DS-bgzf.gz\"; }; \
	    [ -s \"\$$BD/\$$DS-pigz.gz\" ] || { PIGZ=\$$([ -x vendor/pigz/pigz ] && echo vendor/pigz/pigz || echo pigz); echo \"creating \$$BD/\$$DS-pigz.gz\"; \$$PIGZ -1 -c \"\$$RAW\" > \"\$$BD/\$$DS-pigz.gz\"; }; \
	  done; \
	  echo \"  archives: \$$(ls \$$BD/*-{gzip,bgzf,pigz}.gz 2>/dev/null | wc -l) files ready\"; \
	  echo ''; echo '  ── running gzippy-dev bench (--direction both: covers L1/6/9 + L11 micro-corpus) ──'; \
	  TMPDIR=/dev/shm ./target/release/gzippy-dev bench --direction both; \
	  echo ''; echo '  ── L11 head-to-head vs vendor C zopfli (homelab) ──'; \
	  bash scripts/ship_l11_headtohead.sh; \
	  rm -rf /dev/shm/gzippy-bench-* 2>/dev/null || true"
	@echo ""
	@echo "✓ ship complete on branch $$(git rev-parse --abbrev-ref HEAD)"

# =============================================================================
# bench-sm: single-member x86_64 Tmax gzippy vs rapidgzip on neurotic.
#
# Skips all local quality gates (fmt/test/clippy/oracle). Pushes the current
# branch, SSHs to neurotic, builds gzippy + rapidgzip, runs
# scripts/benchmark_single_member.py against silesia.tar compressed at -9.
# Use this for tight iteration on the parallel single-member path without
# the 20-minute full ship round-trip.
#
# Time: ~3-5 minutes (build ~1 min, bench ~2-3 min for 10 trials).
# =============================================================================
bench-sm: ship-precheck
	@BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	echo ""; \
	echo "── bench-sm: x86_64 Tmax single-member  gzippy vs rapidgzip ──"; \
	if ! git rev-parse origin/$$BRANCH >/dev/null 2>&1 \
	    || [ -n "$$(git log origin/$$BRANCH..HEAD 2>/dev/null)" ]; then \
	  echo "  pushing $$BRANCH to origin..."; \
	  git push origin $$BRANCH || (echo "PUSH FAILED — aborting bench-sm" >&2 && exit 1); \
	else \
	  echo "  origin/$$BRANCH already up to date"; \
	fi; \
	echo "  connecting to neurotic..."; \
	timeout $(BENCH_SM_TIMEOUT) $(NEUROTIC_SSH) "set -e; cd gzippy; \
	  echo '  fetching origin/$$BRANCH...'; \
	  $(NEUROTIC_SYNC); \
	  echo '  building gzippy (--features isal-compression)...'; \
	  cargo build --release --features isal-compression 2>&1 | grep -E 'Compiling gzippy |Finished|error' || true; \
	  RAPIDGZIP=vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip; \
	  if [ ! -x \"\$$RAPIDGZIP\" ]; then \
	    echo '  building rapidgzip (first time only)...'; \
	    cd vendor/rapidgzip && git submodule update --init --recursive >/dev/null 2>&1 || true; \
	    mkdir -p librapidarchive/build; \
	    cd librapidarchive/build && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1; \
	    make -j\$$(nproc) rapidgzip 2>&1 | grep -E 'Linking|Built|error' || true; \
	    cd /root/gzippy; \
	  fi; \
	  BD=benchmark_data; \
	  [ -f \"\$$BD/silesia.tar\" ] || { [ -f \"\$$BD/silesia.tar.xz\" ] && xz -dk \"\$$BD/silesia.tar.xz\" -c > \"\$$BD/silesia.tar\"; }; \
	  SL=\$$BD/silesia-large.bin; \
	  SLG=\$$BD/silesia-large.gz; \
	  [ -s \"\$$SL\" ] || { \
	    echo '  building silesia-large.bin (~500MB, one-time)...'; \
	    cat \"\$$BD/silesia.tar\" \"\$$BD/silesia.tar\" > \"\$$SL\"; \
	    head -c \$$((76 * 1024 * 1024)) \"\$$BD/silesia.tar\" >> \"\$$SL\"; \
	  }; \
	  [ -s \"\$$SLG\" ] || { echo '  compressing silesia-large.bin at gzip -9 (one-time, ~60s)...'; gzip -9 -c \"\$$SL\" > \"\$$SLG\"; }; \
	  BDIR=/tmp/bench-sm-bin; mkdir -p \"\$$BDIR\"; \
	  cp target/release/gzippy \"\$$BDIR/\"; \
	  cp \"\$$RAPIDGZIP\" \"\$$BDIR/\" 2>/dev/null || true; \
	  [ -x vendor/pigz/unpigz ] && cp vendor/pigz/unpigz \"\$$BDIR/\" || true; \
	  THREADS=\$$(nproc); \
	  echo ''; \
	  python3 scripts/benchmark_single_member.py \
	    --binaries \"\$$BDIR\" \
	    --compressed-file \"\$$SLG\" \
	    --original-file \"\$$SL\" \
	    --threads \"\$$THREADS\" \
	    --output /tmp/bench-sm-result.json; \
	  echo ''; \
	  echo 'Full JSON: /tmp/bench-sm-result.json'"
	@echo ""
	@echo "✓ bench-sm complete on branch $$(git rev-parse --abbrev-ref HEAD)"

# =============================================================================
# bench-sm-pure-rust: same as bench-sm but builds gzippy with
# `--no-default-features --features pure-rust-inflate` so the parallel-SM
# path goes through ResumableInflate2 instead of the ISA-L C FFI. This is
# the A/B that settles whether the tight-Huffman work has reached
# vendor-competitive performance with ISA-L.
#
# Produces two `gzippy` binaries on neurotic:
#   /tmp/bench-sm-bin/gzippy-isal   (current production path)
#   /tmp/bench-sm-bin/gzippy-purerust (pure-Rust decoder path)
# Both are run against rapidgzip; reports both ratios.
# =============================================================================
bench-sm-pure-rust: ship-precheck
	@BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	echo ""; \
	echo "── bench-sm-pure-rust: A/B gzippy{isal,pure-rust} vs rapidgzip ──"; \
	if ! git rev-parse origin/$$BRANCH >/dev/null 2>&1 \
	    || [ -n "$$(git log origin/$$BRANCH..HEAD 2>/dev/null)" ]; then \
	  echo "  pushing $$BRANCH to origin..."; \
	  git push origin $$BRANCH || (echo "PUSH FAILED — aborting" >&2 && exit 1); \
	else \
	  echo "  origin/$$BRANCH already up to date"; \
	fi; \
	echo "  connecting to neurotic..."; \
	timeout $(BENCH_SM_TIMEOUT) $(NEUROTIC_SSH) "set -e; cd gzippy; \
	  echo '  fetching origin/$$BRANCH...'; \
	  $(NEUROTIC_SYNC); \
	  BDIR=/tmp/bench-sm-bin; mkdir -p \"\$$BDIR\"; \
	  echo '  building gzippy (--features isal-compression)...'; \
	  cargo build --release --features isal-compression 2>&1 | grep -E 'Compiling gzippy |Finished|error' || true; \
	  cp target/release/gzippy \"\$$BDIR/gzippy-isal\"; \
	  echo '  building gzippy (--no-default-features --features pure-rust-inflate)...'; \
	  cargo build --release --no-default-features --features pure-rust-inflate 2>&1 | grep -E 'Compiling gzippy |Finished|error' || true; \
	  cp target/release/gzippy \"\$$BDIR/gzippy-purerust\"; \
	  RAPIDGZIP=vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip; \
	  cp \"\$$RAPIDGZIP\" \"\$$BDIR/\" 2>/dev/null || true; \
	  BD=benchmark_data; \
	  SL=\$$BD/silesia-large.bin; \
	  SLG=\$$BD/silesia-large.gz; \
	  THREADS=\$$(nproc); \
	  echo ''; \
	  bench_one() { \
	    local label=\"\$$1\" bin=\"\$$2\" args=\"\$$3\"; \
	    echo \"=== \$$label ===\"; \
	    for trial in 1 2 3 4 5; do \
	      local t0=\$$(date +%s.%N); \
	      \"\$$bin\" \$$args > /dev/null; \
	      local t1=\$$(date +%s.%N); \
	      local elapsed=\$$(awk \"BEGIN{printf \\\"%.3f\\\", \$$t1 - \$$t0}\"); \
	      local mbps=\$$(awk \"BEGIN{printf \\\"%.0f\\\", 503.6 / \$$elapsed}\"); \
	      echo \"  trial \$$trial: \$${elapsed}s = \$${mbps} MB/s\"; \
	    done; \
	    echo ''; \
	  }; \
	  bench_one 'A: gzippy-isal (current production, ISA-L FFI)' \"\$$BDIR/gzippy-isal\" \"-d -c -p \$$THREADS \$$SLG\"; \
	  bench_one 'B: gzippy-purerust (this branch, ResumableInflate2)' \"\$$BDIR/gzippy-purerust\" \"-d -c -p \$$THREADS \$$SLG\"; \
	  bench_one 'C: rapidgzip (reference)' \"\$$BDIR/rapidgzip\" \"-d -P \$$THREADS -c \$$SLG\"; \
	  echo 'Input: silesia-large.gz (503.6 MB raw, gzip -9)'"
	@echo ""
	@echo "✓ bench-sm-pure-rust complete on branch $$(git rev-parse --abbrev-ref HEAD)"

# =============================================================================
# test-x86_64: verify the x86_64-only code on the homelab box. The parallel
# single-member decode path (ISA-L) is gated on x86_64 + the
# isal-compression feature and does not build on local arm64 macOS. This
# pushes the branch, hard-syncs the homelab checkout, and runs the
# `routing` oracle — the suite that exercises that path end-to-end; the
# rest of the tests are arch-independent and run locally. Hard-capped by
# `timeout` (TEST_X86_TIMEOUT); typically ~8-12 min.
# =============================================================================
.PHONY: test-x86_64
test-x86_64: ship-precheck
	@BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	echo ""; \
	echo "── test-x86_64: cargo test --features isal-compression on neurotic ──"; \
	echo "  pushing $$BRANCH to origin..."; \
	git push origin $$BRANCH || (echo "PUSH FAILED — aborting" >&2 && exit 1); \
	echo "  connecting to neurotic..."; \
	timeout $(TEST_X86_TIMEOUT) $(NEUROTIC_SSH) "set -e; cd gzippy; \
	  $(NEUROTIC_SYNC); \
	  git submodule update --init vendor/isa-l vendor/isal-rs 2>&1 | tail -3; \
	  echo '  building + testing the x86_64-gated path...'; \
	  cargo test --release --features isal-compression routing"
	@echo ""
	@echo "✓ test-x86_64 passed on branch $$(git rev-parse --abbrev-ref HEAD)"

# `ship-precheck`: dirty-tree refusal. Pulled out of `ship-local` so a
# dirty tree fails in <1s instead of after 30s of local checks. Only
# runs as a prereq of `ship` (which pushes); `ship-local` is permitted
# to run on a dirty tree because nothing leaves the box.
.PHONY: ship-precheck
ship-precheck:
	@BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	if [ "$$BRANCH" = "HEAD" ]; then echo "ERROR: detached HEAD; checkout a branch first" >&2; exit 1; fi; \
	if [ -n "$$(git status --porcelain | grep -v 'vendor/zopfli')" ]; then \
	  echo "ERROR: uncommitted changes — homelab needs to fetch a committed branch" >&2; \
	  echo "       run 'git status' and commit (or stash) before 'make ship'" >&2; \
	  echo "       (use 'make ship-local' to run local checks against a dirty tree)" >&2; \
	  exit 1; \
	fi
	@echo "  ✓ branch=$$(git rev-parse --abbrev-ref HEAD), tree clean"

# `ship-local`: steps 1-5 only. Use to validate locally without the
# 20-minute homelab round-trip. Same checks as `ship`'s prefix; if this
# is green, `ship` only adds the homelab decompression scoreboard.
.PHONY: ship-local
ship-local: $(GZIPPY_BIN)
	@BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	if [ "$$BRANCH" = "HEAD" ]; then echo "ERROR: detached HEAD; checkout a branch first" >&2; exit 1; fi; \
	echo "══════════════════════════════════════════════════════"; \
	echo "  SHIP (local): branch=$$BRANCH"; \
	echo "══════════════════════════════════════════════════════"
	@echo ""
	@echo "── Step 1/6: branch sanity ──"
	@BRANCH=$$(git rev-parse --abbrev-ref HEAD); echo "  branch: $$BRANCH"
	@echo "  (working tree may be dirty; the dirty-tree check is in 'ship', not 'ship-local')"
	@echo ""
	@echo "── Step 2/6: cargo fmt + test + clippy (default + oracle features) ──"
	@cargo fmt --check || (echo "FORMAT FAILED — run 'cargo fmt'" >&2 && exit 1)
	@cargo test --release 2>&1 | tail -2 | (grep -q "0 failed" && echo "  ✓ cargo test --release") \
	  || (echo "TESTS FAILED — see 'cargo test --release'" >&2 && exit 1)
	@cargo clippy --release --all-targets -- -D warnings 2>&1 | tail -1 \
	  | (grep -q "Finished" && echo "  ✓ cargo clippy (default features)") \
	  || (echo "CLIPPY FAILED — see 'cargo clippy --release --all-targets -- -D warnings'" >&2 && exit 1)
	@cargo clippy --release --features oracle -- -D warnings 2>&1 | tail -1 \
	  | (grep -q "Finished" && echo "  ✓ cargo clippy (--features oracle)") \
	  || (echo "CLIPPY FAILED (oracle features) — see 'cargo clippy --release --features oracle -- -D warnings'" >&2 && exit 1)
	@echo ""
	@echo "── Step 3/6: corpus oracle vs vendor C zopfli (L11 ratio gate) ──"
	@$(MAKE) --no-print-directory oracle-vs-c 2>&1 | tail -3
	@echo ""
	@echo "── Step 4/6: L11 head-to-head wall-clock + deflate equality ──"
	@bash scripts/ship_l11_headtohead.sh
	@echo ""
	@echo "── Step 5/6: cross-tool roundtrip smoke ──"
	@bash scripts/ship_roundtrip_smoke.sh
	@echo ""
	@echo "✓ ship-local complete (steps 1-5 of 6 — homelab bench skipped)"

# =============================================================================
# Phase 11.2 corpus oracle: compares zopfli_pure's gzip output to the
# vendored C zopfli library byte-for-byte across a fixed corpus and a
# matrix of (numiterations, blocksplitting, maxblocks). The C library is
# only built under the `oracle` feature; production binaries are
# untouched. Run before merging any zopfli_pure change.
# =============================================================================
oracle-vs-c:
	@git submodule update --init vendor/zopfli
	@echo "── corpus oracle: zopfli_pure vs vendor/zopfli ──"
	cargo test --release --features oracle corpus_gzip_output_matches_c_zopfli -- --include-ignored --nocapture

# =============================================================================
# AWS credentials for cloud fleet benchmarks (optional — cloud.rs auto-uses
# aws-vault exec gzippy-dev when no env creds are present)
# =============================================================================
.env:
	aws-vault exec gzippy-dev --no-session -d 12h -- env | grep '^AWS_' > .env
	@echo "AWS credentials written to .env (gzippy-dev account)"

# =============================================================================
# Full performance tests (10+ minutes) - for humans at release time
# =============================================================================
perf-full: $(GZIPPY_BIN) $(UNGZIPPY_BIN) $(PIGZ_BIN) deps
	@mkdir -p $(RESULTS_DIR)
	@python3 scripts/perf.py --full 2>&1 | tee $(RESULTS_DIR)/perf_full_$$(date +%Y%m%d_%H%M%S).log

# Generate test data files using Python script
# Uses test_data/text-1MB.txt (Proust) as seed for highly-compressible text
test-data:
	@python3 scripts/generate_test_data.py --output-dir $(TEST_DATA_DIR) --size 10
	@python3 scripts/generate_test_data.py --output-dir $(TEST_DATA_DIR) --size 100

# Generate just 10MB test files (faster for quick testing)
test-data-quick:
	@python3 scripts/generate_test_data.py --output-dir $(TEST_DATA_DIR) --size 10

# =============================================================================
# Validation target - cross-tool compression/decompression matrix
# =============================================================================
validate: $(GZIPPY_BIN) $(UNGZIPPY_BIN) $(PIGZ_BIN) deps
	@python3 scripts/validate.py

# Validation with JSON output (run tests, save results)
validate-json: $(GZIPPY_BIN) $(UNGZIPPY_BIN) $(PIGZ_BIN) deps
	@mkdir -p $(RESULTS_DIR)
	@python3 scripts/validate.py --json -o $(RESULTS_DIR)/validation.json
	@echo "✓ Results saved to $(RESULTS_DIR)/validation.json"

# Run validation + generate charts (full workflow)
validation-chart: validate-json render-chart

# Render charts from existing JSON (fast iteration on chart rendering)
render-chart:
	@if [ ! -f $(RESULTS_DIR)/validation.json ]; then \
		echo "Error: $(RESULTS_DIR)/validation.json not found. Run 'make validate-json' first."; \
		exit 1; \
	fi
	@echo ""
	@python3 scripts/validation_chart.py $(RESULTS_DIR)/validation.json
	@python3 scripts/validation_chart.py $(RESULTS_DIR)/validation.json --html > $(RESULTS_DIR)/validation.html
	@echo ""
	@echo "✓ HTML chart: $(RESULTS_DIR)/validation.html"

# =============================================================================
# Lint target
# =============================================================================
lint:
	@echo "Running rustfmt..."
	@cargo fmt --all
	@echo "Running clippy..."
	@cargo clippy --release -- -D warnings
	@echo "✓ Lint passed"

lint-check:
	@echo "Checking formatting..."
	@cargo fmt --all --check
	@cargo clippy --release -- -D warnings
	@echo "✓ Lint check passed"

# =============================================================================
# Install target
# =============================================================================
install: $(GZIPPY_BIN) $(UNGZIPPY_BIN)
	@echo "Installing to /usr/local/bin..."
	@install -m 755 $(GZIPPY_BIN) /usr/local/bin/gzippy
	@ln -sf gzippy /usr/local/bin/ungzippy
	@echo "✓ Installed gzippy and ungzippy"

# =============================================================================
# Cleanup
# =============================================================================
clean:
	@echo "Cleaning..."
	@rm -rf $(TEST_DATA_DIR) $(RESULTS_DIR) $(BENCH_RESULTS_DIR) $(BENCH_BIN_DIR)
	@$(MAKE) -C $(PIGZ_DIR) clean >/dev/null 2>&1 || true
	@$(MAKE) -C $(GZIP_DIR) clean >/dev/null 2>&1 || true
	@cd $(GZIPPY_DIR) && cargo clean >/dev/null 2>&1
	@echo "✓ Cleaned"

# =============================================================================
# Benchmark Data Preparation (matches CI)
# =============================================================================

BENCHMARK_DIR := ./benchmark_data
BENCH_RESULTS_DIR := ./benchmark_results
BENCH_BIN_DIR := ./bench_bin
THREADS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Setup bin directory for multi-tool benchmarks
bench-bin: $(GZIPPY_BIN) $(PIGZ_BIN) $(IGZIP_BIN)
	@mkdir -p $(BENCH_BIN_DIR)
	@cp -f $(GZIPPY_BIN) $(BENCH_BIN_DIR)/
	@cp -f $(PIGZ_BIN) $(BENCH_BIN_DIR)/ 2>/dev/null || true
	@cp -f $(PIGZ_DIR)/unpigz $(BENCH_BIN_DIR)/ 2>/dev/null || true
	@cp -f $(IGZIP_BIN) $(BENCH_BIN_DIR)/ 2>/dev/null || true
	@cp -f $(RAPIDGZIP_BIN) $(BENCH_BIN_DIR)/ 2>/dev/null || true
	@cp -f $(GZIP_BIN) $(BENCH_BIN_DIR)/gzip 2>/dev/null || true
ifneq ($(APPLE_GZIP),$(GZIP_BIN))
ifneq ($(APPLE_GZIP),)
	@cp -f $(APPLE_GZIP) $(BENCH_BIN_DIR)/apple-gzip 2>/dev/null || true
endif
endif
	@echo "✓ Benchmark binaries ready in $(BENCH_BIN_DIR)/"

# Benchmark data files
SILESIA_TAR := $(BENCHMARK_DIR)/silesia.tar
SILESIA_GZ := $(BENCHMARK_DIR)/silesia-gzip.tar.gz
SOFTWARE := $(BENCHMARK_DIR)/software.archive
SOFTWARE_GZ := $(BENCHMARK_DIR)/software.archive.gz
LOGS := $(BENCHMARK_DIR)/logs.txt
LOGS_GZ := $(BENCHMARK_DIR)/logs.txt.gz

.PHONY: bench-data bench-bin bench-decompress bench-decompress-all bench-compress bench-compress-all
.PHONY: bench-decompress-silesia bench-decompress-silesia-all
.PHONY: bench-decompress-software bench-decompress-software-all
.PHONY: bench-decompress-logs bench-decompress-logs-all
.PHONY: bench-compress-silesia-l1 bench-compress-silesia-l1-all
.PHONY: bench-compress-silesia-l6 bench-compress-silesia-l6-all
.PHONY: bench-compress-silesia-l9 bench-compress-silesia-l9-all
.PHONY: bench-compress-software-l1 bench-compress-software-l1-all
.PHONY: bench-compress-software-l6 bench-compress-software-l6-all
.PHONY: bench-compress-software-l9 bench-compress-software-l9-all
.PHONY: bench-compress-logs-l1 bench-compress-logs-l1-all
.PHONY: bench-compress-logs-l6 bench-compress-logs-l6-all
.PHONY: bench-compress-logs-l9 bench-compress-logs-l9-all
.PHONY: bench bench-all bench-exhaustive

bench-data:
	@chmod +x scripts/prepare_benchmark_data.sh
	./scripts/prepare_benchmark_data.sh all

# =============================================================================
# DECOMPRESSION BENCHMARKS
# =============================================================================

# --- Silesia (mixed binary/text) ---
bench-decompress-silesia: $(GZIPPY_BIN) bench-data
	@echo "=== Decompression: silesia (gzippy only) ==="
	RUSTFLAGS="-C target-cpu=native" cargo test --release bench_cf_silesia -- --nocapture

bench-decompress-silesia-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_decompression.py \
		--binaries $(BENCH_BIN_DIR) --compressed-file $(SILESIA_GZ) --original-file $(SILESIA_TAR) \
		--threads $(THREADS) --archive-type silesia \
		--output $(BENCH_RESULTS_DIR)/decompress-silesia.json

# --- Software (source code patterns) ---
bench-decompress-software: $(GZIPPY_BIN) bench-data
	@echo "=== Decompression: software (gzippy only) ==="
	RUSTFLAGS="-C target-cpu=native" cargo test --release bench_cf_software -- --nocapture

bench-decompress-software-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_decompression.py \
		--binaries $(BENCH_BIN_DIR) --compressed-file $(SOFTWARE_GZ) --original-file $(SOFTWARE) \
		--threads $(THREADS) --archive-type software \
		--output $(BENCH_RESULTS_DIR)/decompress-software.json

# --- Logs (repetitive data) ---
bench-decompress-logs: $(GZIPPY_BIN) bench-data
	@echo "=== Decompression: logs (gzippy only) ==="
	RUSTFLAGS="-C target-cpu=native" cargo test --release bench_cf_logs -- --nocapture

bench-decompress-logs-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_decompression.py \
		--binaries $(BENCH_BIN_DIR) --compressed-file $(LOGS_GZ) --original-file $(LOGS) \
		--threads $(THREADS) --archive-type logs \
		--output $(BENCH_RESULTS_DIR)/decompress-logs.json

# --- Combined decompression ---
bench-decompress: bench-decompress-silesia bench-decompress-software bench-decompress-logs

bench-decompress-all: bench-decompress-silesia-all bench-decompress-software-all bench-decompress-logs-all
	@echo ""
	@echo "=== Decompression Results ==="
	@for f in $(BENCH_RESULTS_DIR)/decompress-*.json; do \
		[ -f "$$f" ] && echo "--- $$(basename $$f .json) ---" && \
		python3 -c "import json,sys; d=json.load(open('$$f')); [print(f'  {r[\"tool\"]}: {r.get(\"speed_mbps\",0):.1f} MB/s') for r in d.get('results',[]) if 'error' not in r]" 2>/dev/null || true; \
	done

# =============================================================================
# COMPRESSION BENCHMARKS
# =============================================================================

# --- Silesia L1 (fast) ---
bench-compress-silesia-l1: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: silesia L1 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -1 -p$(THREADS) -c $(SILESIA_TAR) > /dev/null'

bench-compress-silesia-l1-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(SILESIA_TAR) \
		--level 1 --threads $(THREADS) --content-type silesia \
		--output $(BENCH_RESULTS_DIR)/compress-silesia-l1.json

# --- Silesia L6 (default) ---
bench-compress-silesia-l6: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: silesia L6 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -6 -p$(THREADS) -c $(SILESIA_TAR) > /dev/null'

bench-compress-silesia-l6-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(SILESIA_TAR) \
		--level 6 --threads $(THREADS) --content-type silesia \
		--output $(BENCH_RESULTS_DIR)/compress-silesia-l6.json

# --- Silesia L9 (best) ---
bench-compress-silesia-l9: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: silesia L9 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -9 -p$(THREADS) -c $(SILESIA_TAR) > /dev/null'

bench-compress-silesia-l9-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(SILESIA_TAR) \
		--level 9 --threads $(THREADS) --content-type silesia \
		--output $(BENCH_RESULTS_DIR)/compress-silesia-l9.json

# --- Software L1/L6/L9 ---
bench-compress-software-l1: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: software L1 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -1 -p$(THREADS) -c $(SOFTWARE) > /dev/null'

bench-compress-software-l1-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(SOFTWARE) \
		--level 1 --threads $(THREADS) --content-type software \
		--output $(BENCH_RESULTS_DIR)/compress-software-l1.json

bench-compress-software-l6: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: software L6 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -6 -p$(THREADS) -c $(SOFTWARE) > /dev/null'

bench-compress-software-l6-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(SOFTWARE) \
		--level 6 --threads $(THREADS) --content-type software \
		--output $(BENCH_RESULTS_DIR)/compress-software-l6.json

bench-compress-software-l9: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: software L9 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -9 -p$(THREADS) -c $(SOFTWARE) > /dev/null'

bench-compress-software-l9-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(SOFTWARE) \
		--level 9 --threads $(THREADS) --content-type software \
		--output $(BENCH_RESULTS_DIR)/compress-software-l9.json

# --- Logs L1/L6/L9 ---
bench-compress-logs-l1: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: logs L1 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -1 -p$(THREADS) -c $(LOGS) > /dev/null'

bench-compress-logs-l1-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(LOGS) \
		--level 1 --threads $(THREADS) --content-type logs \
		--output $(BENCH_RESULTS_DIR)/compress-logs-l1.json

bench-compress-logs-l6: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: logs L6 ==="
	@echo "--- gzippy T1 ---"
	@time sh -c '$(GZIPPY_BIN) -6 -p1 -c $(LOGS) > /dev/null'
	@echo "--- gzippy T$(THREADS) ---"
	@time sh -c '$(GZIPPY_BIN) -6 -p$(THREADS) -c $(LOGS) > /dev/null'
	@echo "--- $(GZIP_BIN) (T1) ---"
	@time sh -c '$(GZIP_BIN) -6 -c $(LOGS) > /dev/null'
ifneq ($(APPLE_GZIP),$(GZIP_BIN))
ifneq ($(APPLE_GZIP),)
	@echo "--- Apple gzip $(APPLE_GZIP) (T1) ---"
	@time sh -c '$(APPLE_GZIP) -6 -c $(LOGS) > /dev/null'
endif
endif

bench-compress-logs-l6-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(LOGS) \
		--level 6 --threads $(THREADS) --content-type logs \
		--output $(BENCH_RESULTS_DIR)/compress-logs-l6.json

bench-compress-logs-l9: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: logs L9 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -9 -p$(THREADS) -c $(LOGS) > /dev/null'

bench-compress-logs-l9-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(LOGS) \
		--level 9 --threads $(THREADS) --content-type logs \
		--output $(BENCH_RESULTS_DIR)/compress-logs-l9.json

# --- Combined compression ---
bench-compress: bench-compress-silesia-l6 bench-compress-software-l6 bench-compress-logs-l6

bench-compress-all: \
	bench-compress-silesia-l1-all bench-compress-silesia-l6-all bench-compress-silesia-l9-all \
	bench-compress-software-l1-all bench-compress-software-l6-all bench-compress-software-l9-all \
	bench-compress-logs-l1-all bench-compress-logs-l6-all bench-compress-logs-l9-all
	@echo ""
	@echo "=== Compression Results ==="
	@for f in $(BENCH_RESULTS_DIR)/compress-*.json; do \
		[ -f "$$f" ] && echo "--- $$(basename $$f .json) ---" && \
		python3 -c "import json; d=json.load(open('$$f')); [print(f'  {r[\"tool\"]}: {r.get(\"speed_mbps\",0):.1f} MB/s, ratio {r.get(\"ratio\",0):.3f}') for r in d.get('results',[]) if 'error' not in r]" 2>/dev/null || true; \
	done

# =============================================================================
# Combined benchmark targets
# =============================================================================

# Quick: gzippy only, L6, all datasets
bench: bench-decompress bench-compress
	@echo ""
	@echo "=== Quick Benchmark Complete ==="

# Full: all tools, all levels
bench-all: bench-decompress-all bench-compress-all
	@echo ""
	@echo "=== Full Benchmark Complete ==="
	@echo "Results in $(BENCH_RESULTS_DIR)/"

bench-exhaustive: bench-all

# =============================================================================
# Help
# =============================================================================
help:
	@printf '%s\n' \
		'gzippy - The Fastest Parallel Gzip' \
		'======================================' \
		'' \
		'The one command:' \
		'  make ship         Full gate (uses CURRENT branch, not main):' \
		'                    fmt + test + clippy + oracle + L11 head-to-head +' \
		'                    cross-tool roundtrip + push branch + homelab bench.' \
		'                    Fail-fast (~30s through local steps; ~25min total).' \
		'  make ship-local   Same as `ship` but skips the homelab step (steps 1-5 only).' \
		'  make test-x86_64  cargo test --features isal-compression on the homelab box' \
		'                    (the x86_64-only parallel single-member path).' \
		'  make oracle-vs-c  Phase 11.2 corpus oracle: zopfli_pure vs vendor/zopfli' \
		'' \
		'Quick commands (for AI tools and iteration):' \
		'  make              Build and run quick benchmark (< 30 seconds)' \
		'  make quick        Same as above' \
		'  make route-check  Show decompression routing + timing for T1/T4 x 1MB/10MB' \
		'  make profile-single-member-decompression-x86_64  Homelab: SM decode + perf' \
		'  make profile-single-member-decompression-arm64   Local M1: SM libdeflate + samply' \
		'  make profile-decompression-x86_64              Alias for x86_64 target above' \
		'  make build        Build gzippy and ungzippy' \
		'  make deps         Build gzip and pigz from submodules' \
		'  make validate     Run validation suite (adaptive 3-17 trials)' \
		'  make lint         Run rustfmt and clippy (auto-fix)' \
		'  make lint-check   Check formatting without changes' \
		'' \
		'Benchmarks (gzippy only - fast):' \
		'  make bench                       Quick benchmark (L6, all datasets)' \
		'  make bench-decompress-silesia    Decompress silesia' \
		'  make bench-decompress-software   Decompress software' \
		'  make bench-decompress-logs       Decompress logs' \
		'  make bench-compress-silesia-l6   Compress silesia L6' \
		'  make bench-compress-software-l6  Compress software L6' \
		'  make bench-compress-logs-l6      Compress logs L6' \
		'' \
		'Benchmarks (all tools compared - exhaustive):' \
		'  make bench-all                       Full comparison (all tools, all levels)' \
		'  make bench-decompress-all            All decompression (3 datasets)' \
		'  make bench-decompress-silesia-all    Decompress silesia (all tools)' \
		'  make bench-decompress-software-all   Decompress software (all tools)' \
		'  make bench-decompress-logs-all       Decompress logs (all tools)' \
		'  make bench-compress-all              All compression (3 datasets x 3 levels)' \
		'  make bench-compress-silesia-l6-all   Compress silesia L6 (all tools)' \
		'  make bench-compress-software-l6-all  Compress software L6 (all tools)' \
		'  make bench-compress-logs-l6-all      Compress logs L6 (all tools)' \
		'' \
		'Data preparation:' \
		'  make bench-data   Prepare benchmark datasets (silesia, software, logs)' \
		'' \
		'Charting (separate test running from rendering):' \
		'  make validate-json     Run tests, save JSON to test_results/' \
		'  make render-chart      Generate charts from existing JSON (fast)' \
		'  make validation-chart  Both: run tests + generate charts' \
		'' \
		'Full testing (for humans at release time):' \
		'  make perf-full    Comprehensive performance tests (10+ minutes)' \
		'  make test-data    Generate all test data files' \
		'' \
		'Installation:' \
		'  make install      Install gzippy and ungzippy to /usr/local/bin' \
		'' \
		'Maintenance:' \
		'  make clean        Remove all build artifacts and test data' \
		'  make help         Show this message' \
		'' \
		'Binaries:' \
		'  gzippy            Compress (default) or decompress with -d' \
		'  ungzippy          Decompress (like gunzip/unpigz)'
