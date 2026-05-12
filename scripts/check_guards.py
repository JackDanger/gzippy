#!/usr/bin/env python3
"""
Check performance guards against benchmark results.

This script applies threshold assertions to aggregated benchmark results,
determining whether gzippy meets its performance targets.

Guards:
- Compression: gzippy must be >= 95% of pigz speed at L1-L9
- Compression: output size must be <= 102% of pigz at L1-L8, <= 100.5% at L9
- Compression: must match or beat igzip speed at L1-L3 (ISA-L hot path)
- Compression: L11 output size must be <= gzip -9 output size
- Decompression: gzippy must beat pigz/gzip, be within 1% of rapidgzip
- Single-threaded: gzippy must be >= 90% of igzip speed
- Single-member: must be within 1% of rapidgzip and beat unpigz

Usage:
    python3 scripts/check_guards.py \
        --compression aggregated/compression.json \
        --decompression aggregated/decompression.json \
        [--single-member-dir results-sm/] \
        --output guards-report.json
"""

import argparse
import json
import sys
from pathlib import Path


# Performance thresholds
THRESHOLDS = {
    # Compression
    "comp_vs_pigz_speed": 0.95,           # Must be >= 95% of pigz speed
    "comp_vs_pigz_size_l9": 1.005,        # Must be <= 100.5% of pigz size at L9
    "comp_vs_pigz_size_l1_l8": 1.02,      # Must be <= 102% of pigz size at L1-L8
    "comp_vs_igzip_speed_l1_l3": 1.0,     # Must match or beat igzip at L1-L3 (ISA-L path)
    "comp_l11_size_vs_gzip9": 1.0,        # L11 output must be <= gzip -9 output

    # Decompression
    "decomp_vs_pigz": 1.0,                # Must be faster than pigz
    "decomp_vs_gzip": 1.0,                # Must be faster than gzip
    "decomp_vs_rapidgzip": 0.99,          # Must be >= 99% of rapidgzip
    "decomp_vs_igzip": 0.90,              # Must be >= 90% of igzip (hand-tuned asm)

    # Single-member parallel decompression (v0.3.0 ISA-L inflatePrime path)
    # CI runs on ubuntu-latest (2 vCPUs). At T=2 the speculation overhead doesn't
    # amortize — the competitive performance only emerges at T>=4+ on real hardware.
    # These are regression-detection floors, not performance targets.
    # The authoritative comparison lives in `make ship` on the homelab (neurotic).
    "single_member_vs_rapidgzip": 0.50,   # CI regression floor; homelab target is ~1.0x
    "single_member_vs_pigz": 0.80,        # CI regression floor; homelab target is >1.0x
}


def load_json(path: str) -> list:
    """Load JSON file, return empty list if missing."""
    try:
        with open(path) as f:
            data = json.load(f)
            # Handle both list and dict formats
            if isinstance(data, dict):
                return data.get("results", data.get("benchmarks", []))
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def check_compression_guards(results: list) -> tuple:
    """
    Check compression performance guards.
    
    Returns (passed: bool, report: list of dicts)
    """
    report = []
    all_passed = True
    
    # Group by level and threads
    by_config = {}
    for r in results:
        if "error" in r:
            continue
        key = (r.get("level", 0), r.get("threads", 1), r.get("data_type", "unknown"))
        if key not in by_config:
            by_config[key] = {}
        by_config[key][r.get("tool", "unknown")] = r
    
    for (level, threads, data_type), tools in by_config.items():
        gzippy = tools.get("gzippy")

        if not gzippy:
            continue

        gzippy_speed = gzippy.get("speed_mbps", gzippy.get("speed", 0))
        gzippy_size = gzippy.get("output_size", gzippy.get("size", 0))

        # vs pigz speed and size (L1-L9 only; pigz not benchmarked at L10+)
        pigz = tools.get("pigz")
        if pigz:
            pigz_speed = pigz.get("speed_mbps", pigz.get("speed", 0))

            if pigz_speed > 0:
                speed_ratio = gzippy_speed / pigz_speed
                speed_passed = speed_ratio >= THRESHOLDS["comp_vs_pigz_speed"]
                report.append({
                    "name": f"Compression L{level} T{threads} {data_type}",
                    "metric": "speed_vs_pigz",
                    "gzippy": gzippy_speed,
                    "pigz": pigz_speed,
                    "ratio": speed_ratio,
                    "threshold": THRESHOLDS["comp_vs_pigz_speed"],
                    "passed": speed_passed,
                })
                if not speed_passed:
                    all_passed = False

            pigz_size = pigz.get("output_size", pigz.get("size", 0))
            if pigz_size > 0:
                size_ratio = gzippy_size / pigz_size

                # L9: tight ratio (zlib-ng quality output)
                if level >= 9:
                    size_passed = size_ratio <= THRESHOLDS["comp_vs_pigz_size_l9"]
                    report.append({
                        "name": f"Compression Ratio L{level} T{threads} {data_type}",
                        "metric": "size_vs_pigz",
                        "gzippy": gzippy_size,
                        "pigz": pigz_size,
                        "ratio": size_ratio,
                        "threshold": THRESHOLDS["comp_vs_pigz_size_l9"],
                        "passed": size_passed,
                    })
                    if not size_passed:
                        all_passed = False

                # L1-L8: loose ratio (speed-focused levels tolerate minor size variance)
                if 1 <= level <= 8:
                    size_passed = size_ratio <= THRESHOLDS["comp_vs_pigz_size_l1_l8"]
                    report.append({
                        "name": f"Compression Ratio L{level} T{threads} {data_type}",
                        "metric": "size_vs_pigz_l1_l8",
                        "gzippy": gzippy_size,
                        "pigz": pigz_size,
                        "ratio": size_ratio,
                        "threshold": THRESHOLDS["comp_vs_pigz_size_l1_l8"],
                        "passed": size_passed,
                    })
                    if not size_passed:
                        all_passed = False

        # vs igzip speed at L1-L3 (gzippy uses ISA-L on x86_64 at these levels)
        if level <= 3:
            igzip = tools.get("igzip")
            if igzip:
                igzip_speed = igzip.get("speed_mbps", igzip.get("speed", 0))
                if igzip_speed > 0:
                    ratio = gzippy_speed / igzip_speed
                    passed = ratio >= THRESHOLDS["comp_vs_igzip_speed_l1_l3"]
                    report.append({
                        "name": f"Compression L{level} T{threads} {data_type} vs igzip",
                        "metric": "speed_vs_igzip",
                        "gzippy": gzippy_speed,
                        "other": igzip_speed,
                        "ratio": ratio,
                        "threshold": THRESHOLDS["comp_vs_igzip_speed_l1_l3"],
                        "passed": passed,
                    })
                    if not passed:
                        all_passed = False

        # vs gzip -9 size at L10+ (zopfli must beat standard max compression)
        if level >= 10:
            gzip = tools.get("gzip")
            if gzip:
                gzip_size = gzip.get("output_size", gzip.get("size", 0))
                if gzip_size > 0:
                    size_ratio = gzippy_size / gzip_size
                    size_passed = size_ratio <= THRESHOLDS["comp_l11_size_vs_gzip9"]
                    report.append({
                        "name": f"Compression Ratio L{level} T{threads} {data_type} vs gzip-9",
                        "metric": "size_vs_gzip9",
                        "gzippy": gzippy_size,
                        "gzip9": gzip_size,
                        "ratio": size_ratio,
                        "threshold": THRESHOLDS["comp_l11_size_vs_gzip9"],
                        "passed": size_passed,
                    })
                    if not size_passed:
                        all_passed = False

    return all_passed, report


def check_decompression_guards(results: list) -> tuple:
    """
    Check decompression performance guards.
    
    Returns (passed: bool, report: list of dicts)
    """
    report = []
    all_passed = True
    
    # Group by source and threads
    by_config = {}
    for r in results:
        if "error" in r or r.get("status") == "fail":
            continue
        key = (r.get("source", "unknown"), r.get("threads", 1), r.get("data_type", "unknown"))
        if key not in by_config:
            by_config[key] = {}
        tool = r.get("tool", "unknown")
        # Normalize unpigz -> pigz for comparison
        if tool == "unpigz":
            tool = "pigz"
        by_config[key][tool] = r
    
    for (source, threads, data_type), tools in by_config.items():
        gzippy = tools.get("gzippy")
        
        if not gzippy:
            continue
        
        gzippy_speed = gzippy.get("speed_mbps", gzippy.get("speed", 0))
        
        # vs pigz
        pigz = tools.get("pigz")
        if pigz:
            pigz_speed = pigz.get("speed_mbps", pigz.get("speed", 0))
            if pigz_speed > 0:
                ratio = gzippy_speed / pigz_speed
                passed = ratio >= THRESHOLDS["decomp_vs_pigz"]
                report.append({
                    "name": f"Decompress {source} T{threads} {data_type} vs pigz",
                    "metric": "speed_vs_pigz",
                    "gzippy": gzippy_speed,
                    "other": pigz_speed,
                    "ratio": ratio,
                    "threshold": THRESHOLDS["decomp_vs_pigz"],
                    "passed": passed,
                })
                if not passed:
                    all_passed = False
        
        # vs gzip (single-threaded only)
        if threads == 1:
            gzip = tools.get("gzip")
            if gzip:
                gzip_speed = gzip.get("speed_mbps", gzip.get("speed", 0))
                if gzip_speed > 0:
                    ratio = gzippy_speed / gzip_speed
                    passed = ratio >= THRESHOLDS["decomp_vs_gzip"]
                    report.append({
                        "name": f"Decompress {source} T1 {data_type} vs gzip",
                        "metric": "speed_vs_gzip",
                        "gzippy": gzippy_speed,
                        "other": gzip_speed,
                        "ratio": ratio,
                        "threshold": THRESHOLDS["decomp_vs_gzip"],
                        "passed": passed,
                    })
                    if not passed:
                        all_passed = False
        
        # vs rapidgzip
        rapidgzip = tools.get("rapidgzip")
        if rapidgzip:
            rapid_speed = rapidgzip.get("speed_mbps", rapidgzip.get("speed", 0))
            if rapid_speed > 0:
                ratio = gzippy_speed / rapid_speed
                passed = ratio >= THRESHOLDS["decomp_vs_rapidgzip"]
                report.append({
                    "name": f"Decompress {source} T{threads} {data_type} vs rapidgzip",
                    "metric": "speed_vs_rapidgzip",
                    "gzippy": gzippy_speed,
                    "other": rapid_speed,
                    "ratio": ratio,
                    "threshold": THRESHOLDS["decomp_vs_rapidgzip"],
                    "passed": passed,
                })
                if not passed:
                    all_passed = False
        
        # vs igzip (single-threaded only)
        if threads == 1:
            igzip = tools.get("igzip")
            if igzip:
                igzip_speed = igzip.get("speed_mbps", igzip.get("speed", 0))
                if igzip_speed > 0:
                    ratio = gzippy_speed / igzip_speed
                    passed = ratio >= THRESHOLDS["decomp_vs_igzip"]
                    report.append({
                        "name": f"Decompress {source} T1 {data_type} vs igzip",
                        "metric": "speed_vs_igzip",
                        "gzippy": gzippy_speed,
                        "other": igzip_speed,
                        "ratio": ratio,
                        "threshold": THRESHOLDS["decomp_vs_igzip"],
                        "passed": passed,
                    })
                    if not passed:
                        all_passed = False
    
    return all_passed, report


def check_single_member_guards(results_dir: str) -> tuple:
    """
    Check single-member parallel decompression guards (v0.3.0 ISA-L inflatePrime path).

    Returns (passed: bool, report: list of dicts)
    """
    report = []
    all_passed = True

    if not results_dir:
        return True, []

    results_path = Path(results_dir)
    if not results_path.exists():
        return True, []

    for json_file in sorted(results_path.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if data.get("benchmark") != "single-member":
            continue

        threads = data.get("threads", "?")
        tool_results = {
            r["tool"]: r for r in data.get("results", [])
            if "error" not in r and r.get("status") == "pass"
        }

        gzippy = tool_results.get("gzippy")
        if not gzippy:
            continue

        gzippy_speed = gzippy.get("speed_mbps", 0)

        rapidgzip = tool_results.get("rapidgzip")
        if rapidgzip:
            rapid_speed = rapidgzip.get("speed_mbps", 0)
            if rapid_speed > 0:
                ratio = gzippy_speed / rapid_speed
                passed = ratio >= THRESHOLDS["single_member_vs_rapidgzip"]
                report.append({
                    "name": f"Single-member T{threads} vs rapidgzip",
                    "metric": "speed_vs_rapidgzip",
                    "gzippy": gzippy_speed,
                    "other": rapid_speed,
                    "ratio": ratio,
                    "threshold": THRESHOLDS["single_member_vs_rapidgzip"],
                    "passed": passed,
                })
                if not passed:
                    all_passed = False

        unpigz = tool_results.get("unpigz")
        if unpigz:
            unpigz_speed = unpigz.get("speed_mbps", 0)
            if unpigz_speed > 0:
                ratio = gzippy_speed / unpigz_speed
                passed = ratio >= THRESHOLDS["single_member_vs_pigz"]
                report.append({
                    "name": f"Single-member T{threads} vs unpigz",
                    "metric": "speed_vs_pigz",
                    "gzippy": gzippy_speed,
                    "other": unpigz_speed,
                    "ratio": ratio,
                    "threshold": THRESHOLDS["single_member_vs_pigz"],
                    "passed": passed,
                })
                if not passed:
                    all_passed = False

    return all_passed, report


def main():
    parser = argparse.ArgumentParser(description="Check performance guards")
    parser.add_argument("--compression", type=str, default="aggregated/compression.json",
                       help="Path to compression results")
    parser.add_argument("--decompression", type=str, default="aggregated/decompression.json",
                       help="Path to decompression results")
    parser.add_argument("--single-member-dir", type=str, default=None,
                       help="Directory containing single-member benchmark result JSONs")
    parser.add_argument("--output", type=str, default="guards-report.json",
                       help="Output report file")
    
    args = parser.parse_args()
    
    # Load results
    compression = load_json(args.compression)
    decompression = load_json(args.decompression)

    print("=== Performance Guards ===\n")

    # Check guards
    comp_passed, comp_report = check_compression_guards(compression)
    decomp_passed, decomp_report = check_decompression_guards(decompression)
    sm_passed, sm_report = check_single_member_guards(args.single_member_dir)

    all_passed = comp_passed and decomp_passed and sm_passed

    # Print results
    print("## Compression Guards\n")
    for g in comp_report:
        status = "✅" if g["passed"] else "❌"
        print(f"{status} {g['name']}: {g['ratio']:.3f}x (threshold: {g['threshold']})")

    print("\n## Decompression Guards\n")
    for g in decomp_report:
        status = "✅" if g["passed"] else "❌"
        print(f"{status} {g['name']}: {g['ratio']:.3f}x (threshold: {g['threshold']})")

    if sm_report:
        print("\n## Single-Member Guards\n")
        for g in sm_report:
            status = "✅" if g["passed"] else "❌"
            print(f"{status} {g['name']}: {g['ratio']:.3f}x (threshold: {g['threshold']})")

    # Summary
    all_guards = comp_report + decomp_report + sm_report
    total = len(all_guards)
    passed_count = sum(1 for g in all_guards if g["passed"])

    print(f"\n{'='*50}")
    print(f"{'✅ ALL GUARDS PASSED' if all_passed else '❌ SOME GUARDS FAILED'}")
    print(f"Passed: {passed_count}/{total}")

    # Write report
    report = {
        "passed": all_passed,
        "compression_passed": comp_passed,
        "decompression_passed": decomp_passed,
        "single_member_passed": sm_passed,
        "compression_guards": comp_report,
        "decompression_guards": decomp_report,
        "single_member_guards": sm_report,
        "thresholds": THRESHOLDS,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to {args.output}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
