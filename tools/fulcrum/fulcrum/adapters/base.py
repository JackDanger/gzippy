"""ProjectAdapter — the plug surface between the fulcrum core and a project.

A project adapter supplies (data > code wherever possible):

  - binary/launch matrix: how the tool-under-test and its comparators run for a
    (corpus, threads) cell, and the expected production routing per cell. The
    actual launching lives in the project's environment-control policy (for
    gzippy: scripts/bench/decide.sh + _decide_guest.sh — freeze, canonical pin
    masks, regular-file sinks, per-run sha verification);
  - corpora/workloads with integrity pins (SHA-OR-VOID);
  - knob registry with effect predicates (EFFECT-VERIFIED-OR-FLAGGED);
  - comparator tools + banked comparator rows;
  - trace taxonomy (wait/compute/output classification + wall-critical frames);
  - routing/contamination guard (production vs oracle-seeded runs).
"""

from typing import NamedTuple

from ..core.trace import Taxonomy  # noqa: F401  (re-export for adapters)


class Knob(NamedTuple):
    """One same-binary kill-switch: env is the FEATURE-ALTERED arm, pred names
    the effect predicate proving the switch engaged, desc is human-readable.
    reverted marks a knob guarding a previously-shipped-then-reverted feature
    (the decision brief says 'reconcile with the prior revert' instead of
    'fix/condition' — structured, not string-matched from the desc)."""
    env: str
    pred: str
    desc: str
    reverted: bool = False


class ProjectAdapter:
    """Subclass per project. Attributes/methods the core engine consumes.

    Decision-table ROW CONTRACT (microprofile_rows and any adapter-supplied
    rows): each row dict carries component, cells, attrib, status, dist,
    verify, tier, rank_ms, PLUS the structured fields the brief builder
    keys on (never string-matched from component text):
      kind        : "knob" | "engine" | "pipeline" (row family)
      perturb_cmd : exact pre-registered perturbation command (tier-2
                    HYPOTHESIS rows; optional otherwise)
      reverted    : bool (knob rows: feature was previously shipped and
                    reverted — the brief says 'reconcile', not 'fix')
    """

    # -- identity / policy -------------------------------------------------
    name = "project"
    tie_bar = 0.99          # PASS bar for comparator ratio (project policy)
    taxonomy = Taxonomy()
    knobs = {}              # name -> Knob
    perturbations = {}      # trace-class -> suggested perturbation command

    # -- artifact loading -----------------------------------------------------
    def load_run(self, art_dir):
        """Load one measurement-run artifact directory into the run dict the
        decision engine consumes. The default implements the documented
        schema (tools/fulcrum/docs/SCHEMA.md); a project with its own
        artifact layout overrides this and maps to the same run-dict shape
        (also documented there)."""
        from ..core.decide import load_run_documented
        return load_run_documented(art_dir, self)

    # -- comparator identity ------------------------------------------------
    def comparator_version(self, manifest):
        """Normalized comparator tool version for the fingerprint's
        `comparator` field (e.g. "rapidgzip 0.16.0"). The default reads the
        documented manifest key; an adapter may parse its comparator's raw
        --version banner instead. Return "unknown" when it cannot be
        certified — an unknown comparator never compares (a comparator
        upgrade moves its numbers; cross-version ratios are different
        experiments)."""
        return manifest.get("comparator_version", "unknown")

    # -- counters / routing guard ------------------------------------------
    def parse_counters(self, text):
        """Counter-sidecar text -> {counter: int}."""
        return {}

    def routing_guard(self, counters, feature=None):
        """(is_production: bool|None, reason). False => the run is
        oracle-contaminated and its numbers are REFUSED; None => inconclusive
        (cannot certify production routing)."""
        return (None, "adapter provides no routing guard")

    def oracle_guard(self, counters, trace_self):
        """Removal-oracle contamination warnings (a handicapped contender must
        not be read as a ceiling). Returns a list of strings."""
        return []

    # -- knobs ----------------------------------------------------------------
    def effect_check(self, pred, base_txt, knob_txt):
        """(verified: bool|None, note). None = no in-tree counter =>
        EFFECT-UNVERIFIED label (never silently trusted)."""
        return (None, f"unknown predicate '{pred}'")

    # -- micro-profile (optional per-project engine counters) -----------------
    def parse_microprofile(self, text):
        """Profile-capture text -> opaque prof object, or None."""
        return None

    def microprofile_rows(self, ck, prof, gap_ms, run):
        """(rows, anomalies) for the decision table. Each row dict must carry:
        component, cells, attrib, status, dist, verify, tier, rank_ms."""
        return ([], [])

    # -- re-verify command surfaces -------------------------------------------
    def reverify_knob(self, ck, kname, run):
        return f"re-run the {kname} A/B on {ck[0]}:T{ck[1]}"

    def reverify_trace(self, ck, run, feature):
        return f"re-analyze the {ck[0]}:T{ck[1]} trace"
