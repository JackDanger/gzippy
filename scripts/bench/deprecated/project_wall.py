#!/usr/bin/env python3
"""Project the §3 (tier1-design-v2) coupled T8 wall from an engine-bench (ii)/(iii) ratio.

Inputs (from the guest engine-isolation bench, single-thread clean chunk):
  mbps_ii   = technique-stack (E2-E4) clean MB/s   (variant iv stacked)
  mbps_iii  = pure ISA-L clean MB/s                (oracle)
  mbps_i    = current scalar-u16 clean MB/s        (baseline; for context)

§3 anchors (tier1-design-v2.md, advisor-vetted):
  gzippy CURRENT clean per-chunk  = 92.7 ms  -> placement-perfect wall 0.6134s
  ISA-L-class clean per-chunk     = 39   ms  -> wall re-binds on pipeline floor ~0.54s
  same-sink tie bar (round-2)     = 0.604s    (rapidgzip same-sink; +12% room vs 0.54)
  39 chunks, T8, ramp 1.36 (descriptive).

Model: per-chunk clean time scales INVERSELY with clean MB/s. The §3 coupled step says:
  decode-bound wall = 39 chunks * per_chunk_ms / 8 / 1000 * 1.36
  total wall = max(decode_bound_wall, non_decode_floor)
where non_decode_floor is the shared pipeline floor (>=0.54s; advisor caveat: gzippy's
own non-decode consumer-serial floor may be materially above 0.54 — reported as a band).
"""
import sys

CHUNKS = 39
T = 8
RAMP = 1.36
GZIPPY_CURRENT_PER_CHUNK_MS = 92.7   # at gzippy current clean MB/s (~118-125)
ISAL_PER_CHUNK_MS = 39.0             # igzip-class
PIPELINE_FLOOR_LOW = 0.54           # rapidgzip own floor (optimistic equality)
PIPELINE_FLOOR_HIGH = 0.604         # round-2 same-sink bar
SPREAD = 0.02                       # ~ inter-run spread band (s)

def decode_bound_wall(per_chunk_ms):
    return CHUNKS * per_chunk_ms / T / 1000.0 * RAMP

def project(ratio_ii_over_iii):
    """ratio = mbps_ii / mbps_iii. ISA-L per-chunk = 39ms => stack per-chunk = 39/ratio."""
    per_chunk_ms = ISAL_PER_CHUNK_MS / ratio_ii_over_iii
    dbw = decode_bound_wall(per_chunk_ms)
    wall_low = max(dbw, PIPELINE_FLOOR_LOW)
    wall_high = max(dbw, PIPELINE_FLOOR_HIGH)
    return per_chunk_ms, dbw, wall_low, wall_high

def verdict(ratio):
    if ratio >= 0.85:
        return "PASS (within ~15% of ISA-L; decode stops binding, wall->pipeline floor)"
    if ratio <= 0.65:
        return "PLATEAU/FAIL (still >=1.5x slower than ISA-L; engine front NOT PROVEN)"
    return "NARROW-MISS / INCONCLUSIVE (0.65<ratio<0.85; report floor, STOP for supervisor)"

if __name__ == "__main__":
    # Usage: project_wall.py <mbps_ii> <mbps_iii> [mbps_i]
    if len(sys.argv) < 3:
        # demo across a sweep of ratios
        print("ratio  per_chunk_ms  decode_bound_wall  wall[floor 0.54]  wall[bar 0.604]  verdict")
        for r in [0.32, 0.50, 0.65, 0.75, 0.85, 0.95, 1.0]:
            pc, dbw, wl, wh = project(r)
            print(f"{r:.2f}   {pc:7.1f}      {dbw:.3f}s            {wl:.3f}s          {wh:.3f}s       {verdict(r)}")
        sys.exit(0)
    mii = float(sys.argv[1]); miii = float(sys.argv[2])
    ratio = mii / miii
    pc, dbw, wl, wh = project(ratio)
    print(f"mbps_ii={mii:.0f} mbps_iii={miii:.0f}  (ii)/(iii)={ratio:.3f}")
    print(f"projected per-chunk clean: {pc:.1f} ms")
    print(f"decode-bound wall: {dbw:.3f}s")
    print(f"projected T8 wall: {wl:.3f}s (floor 0.54) .. {wh:.3f}s (same-sink bar 0.604)")
    print(f"tie bar (round-2 same-sink) = 0.604s + spread {SPREAD}s")
    print(f"VERDICT: {verdict(ratio)}")
