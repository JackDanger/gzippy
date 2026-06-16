#!/usr/bin/env python3
"""optgate_build_artifact.py — assemble a fulcrum optgate.json from the guest
capture's per-arm sample files + meta.env.

The artifact shape is fulcrum src/optgate.rs::OptGateInput. Each sample line in
samples_<arm>.txt is "<cycles> <instructions> <bytes> <procs_running> <sha>".

Usage:
  optgate_build_artifact.py <art-dir> [--out <art-dir>/optgate.json]
"""
import json
import sys
from pathlib import Path


def read_meta(art: Path) -> dict:
    meta = {}
    for line in (art / "meta.env").read_text().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        meta[k] = v
    return meta


def read_samples(path: Path):
    samples, shas = [], []
    if not path.exists():
        return samples, shas
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        cyc, ins, byt, pr = (float(parts[0]), float(parts[1]),
                             float(parts[2]), float(parts[3]))
        sha = parts[4] if len(parts) >= 5 else None
        samples.append({"cycles": cyc, "instructions": ins,
                        "bytes": byt, "procs_running": pr})
        if sha:
            shas.append(sha)
    return samples, shas


def arm(art: Path, label: str, name: str, with_sha: bool):
    samples, shas = read_samples(art / f"samples_{name}.txt")
    a = {"label": label, "samples": samples}
    if with_sha and shas:
        # all gz output shas should be identical; take the most common.
        a["sha"] = max(set(shas), key=shas.count)
    return a


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    art = Path(sys.argv[1])
    out = art / "optgate.json"
    for i, a in enumerate(sys.argv):
        if a == "--out" and i + 1 < len(sys.argv):
            out = Path(sys.argv[i + 1])

    meta = read_meta(art)
    inp = {
        "base": arm(art, "base", "base", with_sha=True),
        "after": arm(art, "after", "after", with_sha=True),
        "rg": arm(art, "rapidgzip", "rg", with_sha=False),
        "reference_sha": meta.get("REFERENCE_SHA", ""),
        "clean_base": arm(art, "clean_base", "clean_base", with_sha=True),
        "clean_after": arm(art, "clean_after", "clean_after", with_sha=True),
        "k": float(meta.get("K", "1")),
        "clean_k": float(meta.get("CLEAN_K", "1")),
        "arch": meta.get("ARCH", "unknown"),
        "cross_arch_replicated": meta.get("CROSS_ARCH", "0") in ("1", "true", "True"),
        "base_commit": meta.get("BASE_COMMIT", ""),
        "after_commit": meta.get("AFTER_COMMIT", ""),
    }
    out.write_text(json.dumps(inp, indent=2))
    n = len(inp["base"]["samples"])
    print(f"wrote {out} (base N={n}, after N={len(inp['after']['samples'])}, "
          f"rg N={len(inp['rg']['samples'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
