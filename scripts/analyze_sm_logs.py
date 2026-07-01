#!/usr/bin/env python3
"""Analyze parallel SM log files for per-stage timing and decode-NOT-STARTED stall analysis."""

import json
import sys
from pathlib import Path


def analyze_log(path, label):
    events = [json.loads(l) for l in open(path) if l.strip()]

    submits = [e for e in events if e["ev"] == "submit_decode"]
    decodes = [e for e in events if e["ev"] == "decode_ok"]
    consumes = [e for e in events if e["ev"] == "consume_done"]
    post_procs = [e for e in events if e["ev"] == "post_process_span"]
    drive_end = [e for e in events if e["ev"] == "drive_end"]
    drive_begin = [e for e in events if e["ev"] == "drive_begin"]
    loop_summ = [e for e in events if e["ev"] == "consumer_loop_summary"]
    accepts = [e for e in events if e["ev"] == "speculative_accept"]
    mismatches = [e for e in events if e["ev"] == "speculative_mismatch"]

    spec_prefetch = [s for s in submits if s.get("is_speculative_prefetch")]
    confirmed_submit = [s for s in submits if not s.get("is_speculative_prefetch")]

    print(f"=== {label} ===")
    if drive_begin:
        d = drive_begin[0]
        print(f"  Input bytes: {d['input_bytes']:,}  Pool size: {d['pool_size']}  Chunk size: {d['chunk_size']:,}")
    if drive_end:
        d = drive_end[0]
        total_s = d["duration_us"] / 1e6
        decoded_MB = d["decoded_bytes"] / 1e6
        print(f"  Wall: {total_s:.3f}s  Decoded: {decoded_MB:.0f}MB  CRC32: {d['crc32']:#010x}")

    print(f"  Chunks submitted: {len(submits)} (speculative={len(spec_prefetch)} confirmed={len(confirmed_submit)})")
    print(f"  Chunks decoded ok: {len(decodes)}")
    print(f"  Speculative accepts: {len(accepts)}  mismatches: {len(mismatches)}")
    print(f"  Post-process spans: {len(post_procs)}")

    # Consumer per-chunk times from consume_done events
    if consumes:
        total_recv_ms = sum(c["recv_us"] for c in consumes) / 1000
        total_publish_ms = sum(c["publish_us"] for c in consumes) / 1000
        total_write_ms = sum(c["crc_write_us"] for c in consumes) / 1000
        total_combine_ms = sum(c["combine_us"] for c in consumes) / 1000
        total_total_ms = sum(c["total_us"] for c in consumes) / 1000
        print(f"  Consumer per-chunk SELF times (sum across {len(consumes)} chunks):")
        print(f"    recv (blocking-get on decode): {total_recv_ms:.1f}ms")
        print(f"    publish_window:                {total_publish_ms:.1f}ms")
        print(f"    crc_write:                     {total_write_ms:.1f}ms")
        print(f"    combine_crc:                   {total_combine_ms:.1f}ms")
        print(f"    total (sum of above):          {total_total_ms:.1f}ms")

    # Consumer loop summary
    if loop_summ:
        l = loop_summ[0]
        iters = l["iters"]
        iter_sum_ms = l["iter_sum_us"] / 1000
        prefetch_ms = l["prefetch_us"] / 1000
        finder_ms = l["finder_us"] / 1000
        fetcher_get_ms = l["fetcher_get_us"] / 1000
        print(f"  Consumer loop summary:")
        print(f"    iters={iters} iter_sum={iter_sum_ms:.1f}ms prefetch={prefetch_ms:.1f}ms "
              f"finder={finder_ms:.1f}ms fetcher_get={fetcher_get_ms:.1f}ms")

    # Post-process stats (apply_window)
    if post_procs:
        total_mat_ms = sum(p["materialize_us"] for p in post_procs) / 1000
        total_apw_ms = sum(p["apply_window_us"] for p in post_procs) / 1000
        total_marker_bytes = sum(p["marker_bytes"] for p in post_procs)
        print(f"  Post-process ({len(post_procs)} spans):")
        print(f"    materialize: {total_mat_ms:.1f}ms  apply_window: {total_apw_ms:.1f}ms  marker_bytes: {total_marker_bytes:,}")

    # Decode duration distribution
    decode_durs_us = sorted([d["duration_us"] for d in decodes])
    if decode_durs_us:
        n = len(decode_durs_us)
        p50 = decode_durs_us[n // 2]
        p95 = decode_durs_us[min(n - 1, int(n * 0.95))]
        total_decode_ms = sum(decode_durs_us) / 1000
        print(f"  Decode durations ({n} chunks, SUM={total_decode_ms:.0f}ms):")
        print(f"    min={decode_durs_us[0]/1000:.0f}ms p50={p50/1000:.0f}ms "
              f"p95={p95/1000:.0f}ms max={decode_durs_us[-1]/1000:.0f}ms")

    # Decode NOT_STARTED analysis:
    # For each confirmed (non-speculative) submit, the decode was NOT prefetched.
    # The gap from when the prior chunk's window was published to when this decode
    # finishes is the stall window. We can compute:
    #   - Number of confirmed (non-speculative) submits (decode_NOT_STARTED proxy)
    #   - For each: decode duration (time the consumer had to wait after the window arrived)
    print(f"  Decode-NOT-STARTED proxy (confirmed=non-speculative submits):")
    print(f"    count={len(confirmed_submit)} (these are chunks that had no prefetched decode running)")
    for s in confirmed_submit:
        pidx = s.get("partition_idx", "?")
        # Find matching decode_ok
        matching = [d for d in decodes if d.get("partition_idx") == pidx]
        if matching:
            dur_ms = matching[0]["duration_us"] / 1000
            print(f"    partition {pidx}: decode_dur={dur_ms:.0f}ms (consumer blocked this long after confirmed offset)")
        else:
            print(f"    partition {pidx}: no matching decode_ok found")

    # Speculative mismatch details
    if mismatches:
        print(f"  Speculative mismatches ({len(mismatches)}):")
        for m in mismatches:
            offset_diff = abs(m.get("speculative_start", 0) - m.get("encoded_offset", 0))
            print(f"    partition {m['partition_idx']}: speculative_start={m['speculative_start']} "
                  f"encoded_offset={m['encoded_offset']} offset_diff={offset_diff}")


def main():
    logs = [
        ("/dev/shm/sm_T4.log", "bignasa T4"),
        ("/dev/shm/sm_T8.log", "bignasa T8"),
        ("/dev/shm/sm_sil8.log", "silesia T8"),
    ]
    for path, label in logs:
        if Path(path).exists():
            analyze_log(path, label)
            print()
        else:
            print(f"MISSING: {path}")


if __name__ == "__main__":
    main()
