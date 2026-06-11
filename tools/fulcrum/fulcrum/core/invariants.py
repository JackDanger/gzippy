"""THE INVARIANT SET — first-class, enforced, each named for its scar.

These are not documentation. Each invariant is a rule the tool *executes*
(refusal or label) with a self-test proving the enforcement fires. Violations
raise InvariantViolation (an InstrumentError), so a contaminated comparison can
never silently produce a number that later gets quoted as truth.

`fulcrum invariants` renders this registry.
"""

from dataclasses import dataclass

from .trace import InstrumentError


class InvariantViolation(InstrumentError):
    """An enforced invariant fired. .invariant carries the scar-name."""

    def __init__(self, invariant, message):
        self.invariant = invariant
        super().__init__(f"[{invariant}] {message}")


@dataclass(frozen=True)
class Invariant:
    name: str   # the scar-name (stable identifier)
    rule: str   # what the tool enforces
    scar: str   # the historical failure that made the rule law
    enforcement: str  # where in the code the refusal/label lives


INVARIANTS = (
    Invariant(
        name="SINK-LAW",
        rule="Both arms of ANY comparison use identical regular-file sinks; the "
             "tool REFUSES mixed-sink or half-rebased comparisons. Non-file "
             "sinks (FIFO, /dev/null) are flagged on sight.",
        scar="The 2026-06-11 HALF-PHANTOM matrix: rg re-based to file-sink "
             "while gz kept /dev/null numbers — 'T1 0.973' was a phantom; the "
             "'gzippy is sink-insensitive' claim was falsified (~110ms@T1 real "
             "output cost). Earlier: the writev-phantom (a FIFO with a draining "
             "reader).",
        enforcement="fingerprint.assert_comparable (sink field); "
                    "decide.load_run sink fields; guest assert_regular_sink",
    ),
    Invariant(
        name="FROZEN-OR-LABELED",
        rule="A wall number from a thawed/loaded/readback-failed box is REFUSED "
             "for ranking; --allow-thaw downgrades refusal to an UNFROZEN label "
             "on every affected row. Freeze state is a fingerprint field.",
        scar="ocl_cf's 0.945<->0.989 drift from a thawed box (the freeze guard "
             "was WARN-only); a bench-lock TTL lapse mid-A/B caught only by "
             "absolute-level sanity.",
        enforcement="decide.analyze_run frozen gate; lib_decide_guest "
                    "freeze_readback (CONCRETE-WRONG never overridable)",
    ),
    Invariant(
        name="SHA-OR-VOID",
        rule="Every measured run's output is sha-verified against the corpus "
             "pin; any mismatch VOIDS the cell. A knob arm with wrong bytes is "
             "recorded as its own finding (switch not byte-transparent), never "
             "ranked.",
        scar="'A speed win with wrong bytes is a loss' (Rule 4); the read-slurp "
             "bug produced a false SHA DIVERGENCE — the check must be "
             "structural, not ad-hoc.",
        enforcement="guest per-run sha verify (decide_fail voids the cell); "
                    "decide.analyze_run sha_ok accounting + knob_sha_fail rows",
    ),
    Invariant(
        name="SPREAD-RESOLUTION",
        rule="Every verdict carries RESOLVED/UNRESOLVED with N-needed; a "
             "sub-spread delta is NEVER presented as a finding; bimodality is "
             "detected and flagged on every sample set.",
        scar="Sessions spent measuring TIEs; the N=21 silesia-T16 lesson — "
             "comparator distributions go bimodal/quantized and a median can "
             "sit on either mode.",
        enforcement="stats.resolution / stats.bimodal on every sample set; "
                    "causal.knob_verdict margins",
    ),
    Invariant(
        name="CAUSAL-OR-HYPOTHESIS",
        rule="No row is ranked as actionable without a tool-executed causal "
             "A/B; everything else is HYPOTHESIS + the exact pre-registered "
             "perturbation command. Attribution is a hypothesis generator, "
             "never the verdict.",
        scar="The 377ms pair-drain phantom, the per-EOB stop cost, the "
             "KEY-MISMATCH re-key lever — attribution that did NOT convert at "
             "the wall; 'contig_prof cycle-shares do NOT translate to wall "
             "share' (instrument-confirmed).",
        enforcement="decide.analyze_run tiering (tier 1 = causal only); "
                    "trace.print_bundle DESCRIPTIVE!=CAUSAL banner",
    ),
    Invariant(
        name="EFFECT-VERIFIED-OR-FLAGGED",
        rule="A kill-switch A/B is causal only if a counter predicate proves "
             "the switch engaged/disengaged; knobs without an in-tree counter "
             "are labeled EFFECT-UNVERIFIED; a failed predicate voids the A/B "
             "(EFFECT-CHECK-FAILED).",
        scar="The rpmalloc stats line printed in BOTH arms (line presence "
             "proves nothing — the predicate had to read slab-specific "
             "counters); oracle.sh built duplicate env keys (env last-wins => "
             "ZERO injection, silently).",
        enforcement="adapter effect_check predicates; decide.analyze_run "
                    "EFFECT-CHECK-FAILED tier demotion",
    ),
    Invariant(
        name="SELF-TEST-OR-NO-TRUST",
        rule="The analyzers carry synthetic-input self-tests with positive AND "
             "negative controls and assertion-fires-on-corruption tests; "
             "decide/analyze label their output untrusted when the engine's "
             "self-test stamp is missing or stale for the current code.",
        scar="Two instruments were silently broken (a clean-window oracle that "
             "re-ran the bootstrap; another that emitted EMPTY output); the "
             "busy+idle==span check was once a tautology.",
        enforcement="selftests package + cli stamp (selftest_stamp.py); "
                    "trace.py trust assertions",
    ),
    Invariant(
        name="FINGERPRINT-OR-NO-COMPARE",
        rule="Every stored number carries {sink, mask, freeze, binary sha, "
             "corpus sha, protocol version, comparator version, host "
             "identity}; ratios/deltas across incompatible or unknown "
             "fingerprints are REFUSED; ledger contradiction checks compare "
             "ONLY fingerprint-compatible rows.",
        scar="The cyc/iter 'regression' that was a TSC frequency-state mismatch "
             "between captures; the stale rg-anchor ('0.98x' vs a banked 926.6 "
             "when the live comparator ran 810).",
        enforcement="fingerprint.assert_comparable; ledger.contradictions "
                    "compatibility filter",
    ),
)


def render():
    lines = ["THE INVARIANT SET — each rule named for the scar that made it law",
             "=" * 72]
    for inv in INVARIANTS:
        lines.append(f"\n{inv.name}")
        lines.append(f"  rule        : {inv.rule}")
        lines.append(f"  scar        : {inv.scar}")
        lines.append(f"  enforcement : {inv.enforcement}")
    return "\n".join(lines)
