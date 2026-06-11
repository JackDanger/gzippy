"""Knob A/B verdicts — the causal core (CAUSAL-OR-HYPOTHESIS).

Scar: the 377ms pair-drain phantom, the per-EOB stop cost, the KEY-MISMATCH
re-key lever — attribution that never converted at the wall. The only causal
currency is a same-binary kill-switch A/B whose effect is counter-verified.

Convention mirrors the validated min-based ratio + max-spread margin (the
same-binary kill-switch instrument that separated layout wobble from behavior).
"""

from .stats import bimodal, resolution, sample_stats


def knob_verdict(base, knob):
    """base/knob: wall-sample lists (s). Returns dict with status in
    {CAUSAL-VERIFIED-COSTS, CAUSAL-VERIFIED-PAYS, CAUSAL-NULL} + numbers.
    knob arm = the FEATURE-ALTERED arm (kill-switch thrown / opt-in enabled).
    delta = knob_min - base_min: delta < 0 => altered arm faster => the shipped
    default COSTS wall in this cell (actionable)."""
    sb, sk = sample_stats(base), sample_stats(knob)
    if not sb or not sk:
        return {"status": "NO-DATA"}
    delta = sk["min"] - sb["min"]
    margin = max(sb["spread_pct"], sk["spread_pct"]) / 100.0 * sb["min"]
    res, n_need = resolution(delta, margin, margin, sb["n"])
    if abs(delta) > margin:
        status = "CAUSAL-VERIFIED-COSTS" if delta < 0 else "CAUSAL-VERIFIED-PAYS"
    else:
        status = "CAUSAL-NULL"
    return {"status": status, "delta_ms": delta * 1000.0,
            "margin_ms": margin * 1000.0, "base": sb, "knob": sk,
            "bimodal": bimodal(base) or bimodal(knob),
            "resolution": res, "n_needed": n_need}
