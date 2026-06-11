"""Append-only results ledger + automatic contradiction detection.

Scar (generalized here): the stale rg-anchor — a '0.98x' parity claim measured
against a banked comparator wall of 926.6ms while the live co-located
comparator ran ~810ms; ALL intermediate ratios needed a re-base lens. And the
cyc/iter clock confound — a banked TSC-cyc number contradicted a live one
because the captures' frequency states differed, not the code.

The ledger makes bank-drift detection automatic and fingerprint-aware:
  - every analyzed number is APPENDED (never rewritten) with its fingerprint;
  - before banking, the tool scans prior rows with a COMPATIBLE fingerprint and
    the same identity key and emits CONTRADICTS-LEDGER when the live number
    diverges beyond tolerance — either the tool or the bank is wrong, and the
    report says so instead of silently ranking;
  - rows with incompatible/unknown fingerprints are NEVER compared (that
    comparison is the phantom class itself).

Record schema (jsonl, one object per line):
  {"ts": iso8601, "runid": str, "project": str, "kind": "cell"|"knob",
   "key": str,                      # e.g. "silesia:T8:gz" or "silesia:T1:knob.slab_off"
   "value_ms": float, "n": int, "spread_pct": float,
   "tool": str,                     # "gzippy" | "rapidgzip" | knob name
   "fingerprint": {...}}
"""

import json
import os
from datetime import datetime, timezone

from .fingerprint import Fingerprint, compatible

# A live number contradicts a banked one when the relative divergence exceeds
# BOTH arms' spreads and this floor (so quiet cells don't false-positive).
REL_TOL_FLOOR = 0.03


class Ledger:
    def __init__(self, path):
        self.path = path

    # -- reading ----------------------------------------------------------
    def rows(self):
        if not self.path or not os.path.exists(self.path):
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # An append-only ledger may carry a torn last line after a
                    # crash; surface it as a row so the caller can warn.
                    out.append({"_corrupt": line[:80]})
        return out

    def has_run(self, runid):
        return any(r.get("runid") == runid for r in self.rows())

    # -- contradiction scan (the generalized bank-drift detector) ----------
    def contradictions(self, record):
        """Compare `record` against prior compatible rows with the same key.
        Returns a list of human-readable CONTRADICTS-LEDGER strings."""
        fp_new = Fingerprint.from_dict(record.get("fingerprint", {}))
        out = []
        for r in self.rows():
            if r.get("_corrupt") or r.get("key") != record.get("key"):
                continue
            if r.get("runid") == record.get("runid"):
                continue
            fp_old = Fingerprint.from_dict(r.get("fingerprint", {}))
            # Same-binary requirement for the tool-under-test (a code change
            # legitimately moves its numbers); comparators (whose binary is
            # pinned by version string in the key) compare across runs.
            same_bin = record.get("kind") == "cell" and \
                record.get("tool") not in ("comparator",)
            if not compatible(fp_old, fp_new, require_same_binary=same_bin):
                continue  # FINGERPRINT-OR-NO-COMPARE: never compare across
            v_old, v_new = r.get("value_ms"), record.get("value_ms")
            if not v_old or not v_new:
                continue
            rel = abs(v_new - v_old) / v_old
            tol = max(REL_TOL_FLOOR,
                      r.get("spread_pct", 0) / 100.0,
                      record.get("spread_pct", 0) / 100.0)
            if rel > tol:
                out.append(
                    f"CONTRADICTS-LEDGER: {record['key']} = {v_new:.1f}ms now "
                    f"vs {v_old:.1f}ms banked ({r.get('runid')}, "
                    f"{r.get('ts', '?')[:10]}) — {rel:.1%} divergence > "
                    f"tol {tol:.1%} under a COMPATIBLE fingerprint. Either the "
                    f"tool or the bank is wrong; reconcile before trusting "
                    f"either (the stale-anchor class).")
        return out

    # -- writing ------------------------------------------------------------
    def append(self, record):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        record = dict(record)
        record.setdefault("ts", datetime.now(timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ"))
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def make_record(runid, project, kind, key, value_ms, n, spread_pct, tool, fp):
    return {"runid": runid, "project": project, "kind": kind, "key": key,
            "value_ms": round(float(value_ms), 3), "n": int(n),
            "spread_pct": round(float(spread_pct), 2), "tool": tool,
            "fingerprint": fp.to_dict() if isinstance(fp, Fingerprint) else fp}
