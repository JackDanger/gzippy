#!/usr/bin/env python3
"""
Aggregate benchmark results from multiple artifact directories.

This script scans a directory for benchmark JSON files and aggregates them
into unified compression.json and decompression.json files.

Usage:
    python3 scripts/aggregate_benchmark_results.py \
        --input-dir results/ \
        --output-dir aggregated/

Expected input structure:
    results/
        bench-text-10mb-l1-single/benchmark-results.json
        bench-text-100mb-l6-max/benchmark-results.json
        benchmark-results-text/decompression.json
        benchmark-results-tarball/decompression.json
        ...

Output structure:
    aggregated/
        compression.json   (all compression results)
        decompression.json (all decompression results)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any


def find_json_files(directory: Path) -> list:
    """Find all JSON files in directory tree."""
    return list(directory.rglob("*.json"))


def load_json(path: Path) -> Any:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None


def is_compression_result(data: dict) -> bool:
    """Check if this looks like a compression benchmark result."""
    if not isinstance(data, dict):
        return False

    # benchmark_compression.py / benchmark_ci.py
    if "archive_type" in data:
        return False
    if "level" in data and "threads" in data and "results" in data:
        results = data["results"]
        if isinstance(results, list) and results:
            first = results[0]
            return isinstance(first, dict) and (
                "output_size" in first or first.get("operation") == "compress"
            )
    if "level" in data and "threads" in data and "size_mb" in data:
        return True

    return False


def is_decompression_result(data: dict) -> bool:
    """Check if this looks like a decompression benchmark result."""
    if not isinstance(data, dict):
        return False

    # benchmark_decompression.py
    if "archive_type" in data and "results" in data:
        results = data["results"]
        if isinstance(results, list) and results:
            first = results[0]
            return isinstance(first, dict) and (
                "speed_mbps" in first or first.get("operation") == "decompress"
            )

    # Legacy format
    if "benchmarks" in data:
        benchmarks = data["benchmarks"]
        if isinstance(benchmarks, list) and benchmarks:
            return "source" in benchmarks[0] and "speed_mbps" in benchmarks[0]

    return False


def normalize_compression_result(data: dict, source_file: str) -> list:
    """
    Normalize a compression result to a standard format.
    
    Returns a list of individual benchmark results.
    """
    results = []
    
    # Extract data_type from filename or path
    data_type = "unknown"
    source_lower = source_file.lower()
    if "silesia" in source_lower:
        data_type = "silesia"
    elif "software" in source_lower:
        data_type = "software"
    elif "logs" in source_lower:
        data_type = "logs"
    elif "text" in source_lower:
        data_type = "text"
    elif "tarball" in source_lower:
        data_type = "tarball"
    
    if data.get("content_type"):
        data_type = data["content_type"]

    level = data.get("level", 0)
    threads = data.get("threads", 1)

    # benchmark_compression.py / benchmark_ci.py
    if "results" in data:
        for r in data.get("results", []):
            if "error" in r:
                continue
            speed = r.get("speed_mbps", r.get("speed", 0))
            output_size = r.get("output_size", r.get("size", 0))
            results.append({
                "tool": r.get("tool", "unknown"),
                "level": r.get("level", level),
                "threads": r.get("threads", threads),
                "time": r.get("median", r.get("time", 0)),
                "size": output_size,
                "output_size": output_size,
                "speed": speed,
                "speed_mbps": speed,
                "data_type": data_type,
                "size_mb": data.get("size_mb", 0),
            })
    elif "level" in data:
        # Single result with nested results
        for r in data.get("results", [data]):
            results.append({
                "tool": r.get("tool", "unknown"),
                "level": data.get("level", r.get("level", 0)),
                "threads": data.get("threads", r.get("threads", 1)),
                "time": r.get("median", r.get("time", 0)),
                "size": r.get("output_size", r.get("size", 0)),
                "speed": r.get("speed", 0),
                "data_type": data_type,
                "size_mb": data.get("size_mb", 0),
            })
    
    return results


def normalize_decompression_result(data: dict, source_file: str) -> list:
    """
    Normalize a decompression result to a standard format.
    
    Returns a list of individual benchmark results.
    """
    results = []
    
    # Extract data_type from filename or path
    data_type = "unknown"
    source_lower = source_file.lower()
    if "silesia" in source_lower:
        data_type = "silesia"
    elif "software" in source_lower:
        data_type = "software"
    elif "logs" in source_lower:
        data_type = "logs"
    elif "text" in source_lower:
        data_type = "text"
    elif "tarball" in source_lower:
        data_type = "tarball"
    
    archive_type = data.get("archive_type", "unknown")
    if archive_type != "unknown":
        data_type = archive_type.split("-", 1)[0]
    threads = data.get("threads", 1)

    entries = data.get("results", data.get("benchmarks", []))
    for b in entries:
        if isinstance(b, dict) and "error" in b:
            continue
        source = b.get("archive_type", b.get("source", archive_type))
        speed = b.get("speed_mbps", b.get("speed", 0))
        results.append({
            "tool": b.get("tool", "unknown"),
            "source": source,
            "level": b.get("level", 0),
            "threads": b.get("threads", threads),
            "speed": speed,
            "speed_mbps": speed,
            "time": b.get("mean", b.get("mean_time", b.get("median", 0))),
            "trials": b.get("trials", 0),
            "cv": b.get("cv", 0),
            "status": b.get("status", "unknown"),
            "data_type": data_type,
        })

    return results


def aggregate_results(input_dir: Path) -> tuple:
    """
    Aggregate all benchmark results from input directory.
    
    Returns (compression_results, decompression_results).
    """
    compression = []
    decompression = []
    
    json_files = find_json_files(input_dir)
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    for json_file in json_files:
        print(f"Processing: {json_file}")
        data = load_json(json_file)
        
        if data is None:
            continue
        
        source_str = str(json_file)
        
        if is_compression_result(data):
            print(f"  -> Compression result")
            compression.extend(normalize_compression_result(data, source_str))
        elif is_decompression_result(data):
            print(f"  -> Decompression result")
            decompression.extend(normalize_decompression_result(data, source_str))
        else:
            print(f"  -> Skipped (unknown format)")
    
    return compression, decompression


def deduplicate_results(results: list, keys: list) -> list:
    """
    Remove duplicate results based on key fields.
    
    If duplicates exist, keep the one with more trials or lower CV.
    """
    seen = {}
    
    for r in results:
        key = tuple(r.get(k) for k in keys)
        
        if key in seen:
            existing = seen[key]
            # Prefer result with more trials or lower CV
            if r.get("trials", 0) > existing.get("trials", 0):
                seen[key] = r
            elif r.get("cv", 1) < existing.get("cv", 1):
                seen[key] = r
        else:
            seen[key] = r
    
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument("--input-dir", type=str, default="results",
                       help="Directory containing benchmark artifacts")
    parser.add_argument("--output-dir", type=str, default="aggregated",
                       help="Output directory for aggregated results")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate all results
    compression, decompression = aggregate_results(input_dir)
    
    # Deduplicate
    if compression:
        compression = deduplicate_results(
            compression, 
            ["tool", "level", "threads", "data_type", "size_mb"]
        )
    
    if decompression:
        decompression = deduplicate_results(
            decompression,
            ["tool", "source", "level", "threads", "data_type"]
        )
    
    # Write outputs
    compression_file = output_dir / "compression.json"
    decompression_file = output_dir / "decompression.json"
    
    with open(compression_file, 'w') as f:
        json.dump(compression, f, indent=2)
    print(f"\nWrote {len(compression)} compression results to {compression_file}")
    
    with open(decompression_file, 'w') as f:
        json.dump(decompression, f, indent=2)
    print(f"Wrote {len(decompression)} decompression results to {decompression_file}")

    if not compression and not decompression:
        print("ERROR: no benchmark results parsed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
