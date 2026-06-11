"""Measurement fingerprints — FINGERPRINT-OR-NO-COMPARE (subsumes SINK-LAW).

Scars:
  - The 2026-06-11 HALF-PHANTOM matrix: rg re-based to a file sink while gz
    kept /dev/null numbers; "T1 0.973" was a phantom ratio across two sink
    protocols.
  - The stale rg-anchor: a "0.98x" claim measured against a banked 926.6ms
    while the live co-located comparator ran 810ms.
  - The cyc/iter "regression" that was a TSC frequency-state mismatch between
    captures (frozen no_turbo vs the bank's capture), not a code change.

Every stored number carries a Fingerprint: {sink, mask, freeze, binary sha,
corpus sha, protocol version}. Two numbers may form a ratio/delta ONLY if their
fingerprints are compatible. An unknown field is never compatible with
anything (unknown != unknown): refusing a comparison is cheap; un-publishing a
phantom is not.
"""

from dataclasses import asdict, dataclass

# Fields that must MATCH (and be known) for two measurements to be comparable.
COMPARE_FIELDS = ("sink", "mask", "freeze", "corpus_sha", "protocol")


@dataclass(frozen=True)
class Fingerprint:
    sink: str = "unknown"        # sink class, e.g. "regular-file" | "devnull" | "pipe"
    mask: str = "unknown"        # cpu pin mask, e.g. "0,2,4,6"
    freeze: str = "unknown"      # "frozen" | "acknowledged" | "thawed" | ...
    bin_sha: str = "unknown"     # binary identity (sha256)
    corpus_sha: str = "unknown"  # corpus content pin (decompressed sha256)
    # measurement-protocol version (fulcrum.PROTOCOL_VERSION)
    protocol: str = "unknown"

    def is_complete(self):
        return all(getattr(self, f) != "unknown" for f in COMPARE_FIELDS)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def incompatibilities(a, b, require_same_binary=False):
    """Return the list of human-readable reasons a and b may NOT be compared.

    Empty list == comparable. 'unknown' on either side of a compare-field is an
    incompatibility (never assume two unknowns match — that assumption IS the
    half-rebased-table phantom).
    """
    reasons = []
    for f in COMPARE_FIELDS:
        va, vb = getattr(a, f), getattr(b, f)
        if va == "unknown" or vb == "unknown":
            reasons.append(f"{f} unknown ({va!r} vs {vb!r}) — cannot certify "
                           f"identical {f} protocol")
        elif va != vb:
            reasons.append(f"{f} mismatch: {va!r} vs {vb!r}")
    if require_same_binary and a.bin_sha != b.bin_sha:
        reasons.append(f"bin_sha mismatch: {a.bin_sha[:12]} vs {b.bin_sha[:12]}")
    return reasons


def compatible(a, b, require_same_binary=False):
    return not incompatibilities(a, b, require_same_binary=require_same_binary)


def assert_comparable(a, b, what="ratio", require_same_binary=False):
    """Raise InvariantViolation unless a and b may form a ratio/delta.

    This is the enforcement point for SINK-LAW and FINGERPRINT-OR-NO-COMPARE:
    a mixed-sink or half-rebased comparison dies HERE, before any number is
    rendered.
    """
    from .invariants import InvariantViolation  # local import: no cycle at load
    reasons = incompatibilities(a, b, require_same_binary=require_same_binary)
    if reasons:
        name = ("SINK-LAW" if any(r.startswith("sink") for r in reasons)
                else "FINGERPRINT-OR-NO-COMPARE")
        raise InvariantViolation(
            name,
            f"REFUSING {what}: measurement fingerprints are not comparable — "
            + "; ".join(reasons))
