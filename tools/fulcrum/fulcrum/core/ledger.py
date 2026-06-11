"""
THREAT MODEL (honest): the per-record hash chain is UNKEYED tamper-
EVIDENCE, not tamper-PROOF — it catches edit-without-rechain, reorder,
and truncation, but a full suffix re-forge recomputes cleanly. For
tamper-proofing use an HMAC with an out-of-band key or anchor the chain
head externally.
Append-only results ledger + automatic contradiction detection.

The ledger makes bank-drift detection automatic and fingerprint-aware:
  - every analyzed number is APPENDED (never rewritten) with its fingerprint;
  - before banking, the tool scans prior ACTIVE rows with a COMPATIBLE
    fingerprint and the same identity key and emits CONTRADICTS-LEDGER when
    the live number diverges beyond tolerance — either the tool or the bank
    is wrong, and the report says so instead of silently ranking;
  - a CONTRADICTING live number is NOT auto-banked as an anchor: it lands
    with status "pending-reconcile" and stays out of the anchor set until a
    `supersede` record resolves the conflict (see below);
  - rows with incompatible/unknown fingerprints are NEVER compared (that
    comparison is the phantom class itself — see docs/CASE-STUDIES.md for the
    stale-anchor and clock-confound histories that made these rules law).

Record kinds (jsonl, one object per line, append-only):

  measurement ("cell" | "knob"):
    {"ts": iso8601, "runid": str, "project": str, "kind": "cell"|"knob",
     "key": str,                  # e.g. "silesia:T8:gz" or "silesia:T1:knob.slab_off"
     "value_ms": float, "n": int, "spread_pct": float,
     "tool": str,                 # "gzippy" | "comparator" | knob name
     "fingerprint": {...},
     "status": "pending-reconcile"  # OPTIONAL; absent == active}

  supersede — retires a prior measurement row (it was honest then, but is no
  longer the anchor: comparator upgraded, protocol corrected, box changed);
  may simultaneously PROMOTE a pending-reconcile row to active:
    {"ts": ..., "kind": "supersede", "key": str, "retire_runid": str,
     "promote_runid": str|null, "reason": str}

  invalid — retires a prior measurement row that was NEVER right
  (measurement error, broken instrument); nothing is promoted by it:
    {"ts": ..., "kind": "invalid", "key": str, "target_runid": str,
     "reason": str}

APPEND-ONLY is a CONVENTION this class upholds (it only ever opens the file
in "a" mode and resolutions are new records, never edits) — it is NOT an OS
guarantee. For tamper evidence every appended record carries a hash-chain
field: chain = sha256(prev_chain + canonical_json(record_without_chain))[:16].
`verify_chain()` recomputes the chain and reports any edited/reordered/
removed-predecessor row. Rows predating the chain are tolerated (verification
covers chained rows only — state that honestly when auditing old ledgers).
"""

import hashlib
import json
import os
from datetime import datetime, timezone

from .fingerprint import Fingerprint, compatible

# A live number contradicts a banked one when the relative divergence exceeds
# BOTH arms' spreads and this floor (so quiet cells don't false-positive).
REL_TOL_FLOOR = 0.03

MEASUREMENT_KINDS = ("cell", "knob")
PENDING = "pending-reconcile"


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
        return any(r.get("runid") == runid for r in self.rows()
                   if r.get("kind") in MEASUREMENT_KINDS)

    # -- supersede / invalidate bookkeeping --------------------------------
    @staticmethod
    def _resolution_sets(rows):
        """(retired, promoted): sets of (key, runid) named by supersede /
        invalid records. Retired rows are out of the anchor set forever;
        promoted rows are pending-reconcile rows accepted as the new anchor."""
        retired, promoted = set(), set()
        for r in rows:
            if r.get("_corrupt"):
                continue
            if r.get("kind") == "supersede":
                retired.add((r.get("key"), r.get("retire_runid")))
                if r.get("promote_runid"):
                    promoted.add((r.get("key"), r.get("promote_runid")))
            elif r.get("kind") == "invalid":
                retired.add((r.get("key"), r.get("target_runid")))
        return retired, promoted

    def anchors(self, key=None):
        """Measurement rows usable as contradiction anchors: not corrupt, not
        retired by a supersede/invalid record, and not pending-reconcile
        (unless promoted). A pending row is a RECORD, never an anchor — using
        a contested number as the next run's truth is how a stale anchor
        becomes two stale anchors."""
        rows = self.rows()
        retired, promoted = self._resolution_sets(rows)
        out = []
        for r in rows:
            if r.get("_corrupt") or r.get("kind") not in MEASUREMENT_KINDS:
                continue
            if key is not None and r.get("key") != key:
                continue
            ident = (r.get("key"), r.get("runid"))
            if ident in retired:
                continue
            if r.get("status") == PENDING and ident not in promoted:
                continue
            out.append(r)
        return out

    # -- contradiction scan (the generalized bank-drift detector) ----------
    def contradictions(self, record):
        """Compare `record` against prior ACTIVE (anchor) rows with the same
        key. Returns a list of human-readable CONTRADICTS-LEDGER strings."""
        fp_new = Fingerprint.from_dict(record.get("fingerprint", {}))
        out = []
        for r in self.anchors(key=record.get("key")):
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
                    f"tool or the bank is wrong; the live row is banked "
                    f"PENDING-RECONCILE (never an anchor) until a `supersede` "
                    f"record resolves which one (the stale-anchor class).")
        return out

    # -- writing ------------------------------------------------------------
    def _last_chain(self):
        prev = ""
        for r in self.rows():
            if not r.get("_corrupt") and r.get("chain"):
                prev = r["chain"]
        return prev

    @staticmethod
    def _chain_hash(prev_chain, record):
        basis = json.dumps({k: v for k, v in record.items() if k != "chain"},
                           sort_keys=True)
        return hashlib.sha256((prev_chain + basis).encode()).hexdigest()[:16]

    def append(self, record):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        record = dict(record)
        record.setdefault("ts", datetime.now(timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ"))
        record["chain"] = self._chain_hash(self._last_chain(), record)
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def supersede(self, key, retire_runid, reason, promote_runid=None):
        if not str(reason).strip():
            raise ValueError("reason must be a non-empty justification")
        """Append a supersede record retiring (key, retire_runid) as an
        anchor, optionally promoting a pending-reconcile row to active."""
        self.append({"kind": "supersede", "key": key,
                     "retire_runid": retire_runid,
                     "promote_runid": promote_runid, "reason": reason})

    def invalidate(self, key, target_runid, reason):
        if not str(reason).strip():
            raise ValueError("reason must be a non-empty justification")
        """Append an invalid record retiring (key, target_runid) — the row
        was a measurement error and is never an anchor again."""
        self.append({"kind": "invalid", "key": key,
                     "target_runid": target_runid, "reason": reason})

    # -- tamper evidence ------------------------------------------------------
    def verify_chain(self):
        """Recompute the hash chain over chained rows. Returns a list of
        human-readable breaks (empty == chained rows intact). Rows without a
        chain field predate the chain and are skipped — verification only
        vouches for the chained rows."""
        breaks = []
        prev = ""
        for i, r in enumerate(self.rows()):
            if r.get("_corrupt"):
                breaks.append(f"row {i}: torn/corrupt line")
                continue
            if not r.get("chain"):
                continue  # pre-chain row: convention only, no evidence
            want = self._chain_hash(prev, r)
            if r["chain"] != want:
                breaks.append(
                    f"row {i} ({r.get('kind')} {r.get('key')} "
                    f"{r.get('runid', '')}): chain {r['chain']} != expected "
                    f"{want} — row edited, reordered, or a chained "
                    f"predecessor removed (append-only violated)")
            prev = r["chain"]
        return breaks

def make_record(runid, project, kind, key, value_ms, n, spread_pct, tool, fp):
    return {"runid": runid, "project": project, "kind": kind, "key": key,
            "value_ms": round(float(value_ms), 3), "n": int(n),
            "spread_pct": round(float(spread_pct), 2), "tool": tool,
            "fingerprint": fp.to_dict() if isinstance(fp, Fingerprint) else fp}
