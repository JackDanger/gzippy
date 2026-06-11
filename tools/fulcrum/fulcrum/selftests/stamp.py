"""Self-test stamp (SELF-TEST-OR-NO-TRUST enforcement data).

The stamp records a hash of the package's source. decide/analyze compare the
stamp hash against the live source hash: a missing or stale stamp means the
installed engine version has never passed its own self-tests, and every
rendered number carries the UNTESTED-ENGINE label until `fulcrum selftest`
passes again.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAMP_PATH = os.path.join(os.path.dirname(_PKG_DIR), ".selftest-stamp.json")


def version_hash(pkg_dir=_PKG_DIR):
    h = hashlib.sha256()
    for root, dirs, files in os.walk(pkg_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, pkg_dir).encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def write_stamp(counts, path=STAMP_PATH):
    data = {"version_hash": version_hash(),
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "counts": counts}
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    return path


def stamp_valid(path=STAMP_PATH):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("version_hash") == version_hash()


def trust_label(path=STAMP_PATH):
    """'' when the stamp is valid, else the loud label."""
    if stamp_valid(path):
        return ""
    return ("[SELF-TEST-OR-NO-TRUST] engine self-test stamp missing/stale for "
            "this source version — run `fulcrum selftest` before trusting any "
            "number below.")
