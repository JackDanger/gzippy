"""Invariant-enforcement self-tests — the named regression cases.

Every scar in the invariant set gets a test proving the enforcement FIRES:
  - HALF-PHANTOM-2026-06-11 (SINK-LAW): a mixed-sink artifact (gz arm
    /dev/null, rg arm regular-file — the half-rebased matrix) is REFUSED.
  - STALE-ANCHOR / CLOCK-CONFOUND (FINGERPRINT-OR-NO-COMPARE): a ratio across
    freeze-state / protocol-version fingerprints is REFUSED; unknown is never
    compatible with anything.
  - BANK-DRIFT (ledger): a live number contradicting a banked number under a
    COMPATIBLE fingerprint raises CONTRADICTS-LEDGER; an INCOMPATIBLE
    fingerprint is never compared (no false contradiction).
  - SHA-OR-VOID: a cell whose manifest row lacks sha_ok=1 is voided, not
    ranked.
  - SELF-TEST-OR-NO-TRUST: the stamp goes stale when the source hash moves.
"""

import os
import tempfile

from ..adapters.gzippy import GzippyAdapter
from ..core.decide import analyze_run, load_run
from ..core.fingerprint import (
    Fingerprint,
    assert_comparable,
    compatible,
    incompatibilities,
)
from ..core.invariants import INVARIANTS, InvariantViolation
from ..core.ledger import Ledger, make_record
from . import Checker
from . import stamp as stamp_mod
from .test_decide import RG, make_artifact, write_samples

AD = GzippyAdapter()

FP = Fingerprint(sink="regular-file", mask="0", freeze="frozen",
                 bin_sha="deadbeef", corpus_sha="028bd002", protocol="fulcrum-v3")


def run():
    check = Checker()
    print("=== fulcrum selftest: invariant enforcement (the named scars) ===")

    # ------------------------------------------------------------------
    # SINK-LAW — regression case HALF-PHANTOM-2026-06-11: rg re-based to
    # file-sink while gz kept /dev/null. The mixed-sink artifact must be
    # REFUSED, not ranked.
    # ------------------------------------------------------------------
    d = tempfile.mkdtemp(prefix="fulcrum_inv_sink_")
    make_artifact(d, with_knobs=False, v3=True)
    man_path = os.path.join(d, "manifest.txt")
    txt = open(man_path).read().replace("sink_gz=regular-file",
                                        "sink_gz=devnull")
    open(man_path, "w").write(txt)
    raised = None
    try:
        analyze_run(load_run(d, AD), AD)
    except InvariantViolation as e:
        raised = e
    check(raised is not None and raised.invariant == "SINK-LAW",
          "HALF-PHANTOM-2026-06-11: mixed-sink artifact (gz=devnull, "
          "rg=regular-file) REFUSED with SINK-LAW")
    check(raised is not None and "sink" in str(raised),
          "SINK-LAW refusal names the sink mismatch verbatim")

    # Control: the same artifact with matching sinks ranks fine.
    d_ok = tempfile.mkdtemp(prefix="fulcrum_inv_sinkok_")
    make_artifact(d_ok, with_knobs=False, v3=True)
    rep_ok = analyze_run(load_run(d_ok, AD), AD)
    check(bool(rep_ok["scoreboard"]) and
          "FP-INCOMPLETE" not in rep_ok["scoreboard"][0],
          "control: same-sink v3 artifact ranks with a complete fingerprint")

    # Pre-v3 artifact (no sink/protocol fields): labeled, never refused,
    # never banked.
    d_old = tempfile.mkdtemp(prefix="fulcrum_inv_prev3_")
    make_artifact(d_old, with_knobs=False, v3=False)
    lpath_old = os.path.join(d_old, "ledger.jsonl")
    rep_old = analyze_run(load_run(d_old, AD), AD, ledger=Ledger(lpath_old))
    check("FP-INCOMPLETE" in rep_old["scoreboard"][0],
          "pre-v3 artifact: scoreboard labeled FP-INCOMPLETE (not refused)")
    check(not os.path.exists(lpath_old),
          "pre-v3 artifact: nothing banked to the ledger (incomplete "
          "fingerprint)")

    # ------------------------------------------------------------------
    # FINGERPRINT-OR-NO-COMPARE — regression cases STALE-ANCHOR and
    # CLOCK-CONFOUND: cross-fingerprint ratios are REFUSED.
    # ------------------------------------------------------------------
    fp_frozen = FP
    fp_thawed = Fingerprint(**{**FP.to_dict(), "freeze": "thawed"})
    raised2 = None
    try:
        assert_comparable(fp_frozen, fp_thawed, what="cross-freeze ratio")
    except InvariantViolation as e:
        raised2 = e
    check(raised2 is not None
          and raised2.invariant == "FINGERPRINT-OR-NO-COMPARE",
          "CLOCK-CONFOUND: frozen-vs-thawed fingerprint ratio REFUSED")
    fp_v2 = Fingerprint(**{**FP.to_dict(), "protocol": "fulcrum-v2"})
    check(not compatible(FP, fp_v2),
          "protocol-version mismatch => fingerprints incompatible")
    fp_unknown = Fingerprint()
    check(not compatible(fp_unknown, fp_unknown),
          "unknown fingerprints are NEVER compatible (even with themselves — "
          "two unknowns matching IS the half-rebased phantom)")
    check(compatible(FP, Fingerprint(**FP.to_dict())),
          "identical complete fingerprints ARE compatible (control)")
    fp_othermask = Fingerprint(**{**FP.to_dict(), "mask": "0,2,4,6"})
    check(any("mask" in r for r in incompatibilities(FP, fp_othermask)),
          "mask mismatch surfaced by name (free-placement numbers lie)")

    # SINK-LAW via assert_comparable: sink mismatch names SINK-LAW, not the
    # generic invariant.
    fp_null = Fingerprint(**{**FP.to_dict(), "sink": "devnull"})
    raised3 = None
    try:
        assert_comparable(FP, fp_null, what="mixed-sink ratio")
    except InvariantViolation as e:
        raised3 = e
    check(raised3 is not None and raised3.invariant == "SINK-LAW",
          "direct mixed-sink assert_comparable raises SINK-LAW by name")

    # ------------------------------------------------------------------
    # Ledger — BANK-DRIFT generalized (the stale-anchor + cyc/iter clock
    # confound detector).
    # ------------------------------------------------------------------
    ltmp = tempfile.mkdtemp(prefix="fulcrum_inv_ledger_")
    led = Ledger(os.path.join(ltmp, "ledger.jsonl"))
    banked = make_record("run_A", "gzippy", "cell", "silesia:T1:rg",
                         917.0, 7, 1.0, "comparator", FP)
    led.append(banked)
    live = make_record("run_B", "gzippy", "cell", "silesia:T1:rg",
                       810.0, 7, 1.0, "comparator", FP)
    contras = led.contradictions(live)
    check(len(contras) == 1 and "CONTRADICTS-LEDGER" in contras[0],
          "STALE-ANCHOR: live 810ms vs banked 917ms (compatible fingerprint) "
          "=> CONTRADICTS-LEDGER fires")
    check("917.0ms" in contras[0] and "810.0ms" in contras[0],
          "contradiction message carries both numbers verbatim")
    # Incompatible fingerprint (different freeze state): NEVER compared.
    live_thawed = make_record("run_C", "gzippy", "cell", "silesia:T1:rg",
                              810.0, 7, 1.0, "comparator", fp_thawed)
    check(led.contradictions(live_thawed) == [],
          "CLOCK-CONFOUND guard: incompatible fingerprint rows are never "
          "compared (no false contradiction)")
    # Within tolerance: no contradiction.
    live_close = make_record("run_D", "gzippy", "cell", "silesia:T1:rg",
                             925.0, 7, 2.0, "comparator", FP)
    check(led.contradictions(live_close) == [],
          "within-tolerance live number does NOT contradict the bank")
    # Same-binary requirement for tool-under-test rows: a gz cell banked under
    # a different bin_sha is not a contradiction (code changes move numbers).
    fp_newbin = Fingerprint(**{**FP.to_dict(), "bin_sha": "cafebabe"})
    led.append(make_record("run_E", "gzippy", "cell", "silesia:T1:gz",
                           1380.0, 7, 1.0, "gzippy", FP))
    live_newbin = make_record("run_F", "gzippy", "cell", "silesia:T1:gz",
                              900.0, 7, 1.0, "gzippy", fp_newbin)
    check(led.contradictions(live_newbin) == [],
          "tool-under-test rows compare only same-binary (a code change is "
          "not bank drift)")

    # Ledger end-to-end through analyze_run: bank, re-analyze (idempotent),
    # then a drifted second run contradicts.
    d_l = tempfile.mkdtemp(prefix="fulcrum_inv_lroundtrip_")
    make_artifact(d_l, with_knobs=False, v3=True)
    lpath = os.path.join(d_l, "ledger.jsonl")
    rep_l = analyze_run(load_run(d_l, AD), AD, ledger=Ledger(lpath))
    n_rows = len(Ledger(lpath).rows())
    check(n_rows == 2 and not any("CONTRADICTS" in a
                                  for a in rep_l["anomalies"]),
          f"e2e ledger: first run banks gz+rg rows (got {n_rows}), no "
          f"contradiction")
    analyze_run(load_run(d_l, AD), AD, ledger=Ledger(lpath))
    check(len(Ledger(lpath).rows()) == n_rows,
          "e2e ledger: re-analysis of the same runid appends NOTHING "
          "(append-only, idempotent)")
    # Second run, same fingerprint, rg wall drifted -21%: CONTRADICTS-LEDGER.
    d_l2 = tempfile.mkdtemp(prefix="fulcrum_inv_lroundtrip2_")
    make_artifact(d_l2, with_knobs=False, v3=True)
    man2 = os.path.join(d_l2, "manifest.txt")
    txt2 = open(man2).read().replace("runid=st", "runid=st2")
    open(man2, "w").write(txt2)
    write_samples(os.path.join(d_l2, "cell_silesia_T1", "wall_rg.txt"),
                  [x - 0.190 for x in RG])
    rep_l2 = analyze_run(load_run(d_l2, AD), AD, ledger=Ledger(lpath))
    check(any("CONTRADICTS-LEDGER" in a for a in rep_l2["anomalies"]),
          "e2e ledger: drifted comparator wall (the rg-anchor case) surfaces "
          "CONTRADICTS-LEDGER in the report")

    # ------------------------------------------------------------------
    # SHA-OR-VOID: a cell whose manifest row lacks sha_ok=1 is voided.
    # ------------------------------------------------------------------
    d_s = tempfile.mkdtemp(prefix="fulcrum_inv_sha_")
    make_artifact(d_s, with_knobs=False, v3=True)
    man_s = os.path.join(d_s, "manifest.txt")
    txt_s = open(man_s).read().replace(
        "cell_done=silesia:1:mask=0:sha_ok=1",
        "cell_done=silesia:1:mask=0:sha_ok=0")
    open(man_s, "w").write(txt_s)
    rep_s = analyze_run(load_run(d_s, AD), AD)
    check(rep_s["scoreboard"] == [] and
          any("SHA-OR-VOID" in a for a in rep_s["anomalies"]),
          "SHA-OR-VOID: sha_ok!=1 cell is VOID (anomaly, not a ranked row)")

    # ------------------------------------------------------------------
    # SELF-TEST-OR-NO-TRUST: stamp validity tracks the source hash.
    # ------------------------------------------------------------------
    stmp = os.path.join(tempfile.mkdtemp(prefix="fulcrum_inv_stamp_"),
                        "stamp.json")
    check(stamp_mod.stamp_valid(stmp) is False,
          "stamp: missing stamp => NOT valid (untested engine is untrusted)")
    stamp_mod.write_stamp({"t": {"checks": 1, "failures": 0}}, path=stmp)
    check(stamp_mod.stamp_valid(stmp) is True,
          "stamp: freshly written stamp validates")
    with open(stmp, "w") as f:
        f.write('{"version_hash": "stale"}')
    check(stamp_mod.stamp_valid(stmp) is False,
          "stamp: stale source hash => NOT valid")
    check("SELF-TEST-OR-NO-TRUST" in stamp_mod.trust_label(stmp),
          "stamp: trust_label carries the invariant name when invalid")

    # ------------------------------------------------------------------
    # Registry sanity: every charter invariant present with a scar.
    # ------------------------------------------------------------------
    names = {i.name for i in INVARIANTS}
    expected = {"SINK-LAW", "FROZEN-OR-LABELED", "SHA-OR-VOID",
                "SPREAD-RESOLUTION", "CAUSAL-OR-HYPOTHESIS",
                "EFFECT-VERIFIED-OR-FLAGGED", "SELF-TEST-OR-NO-TRUST",
                "FINGERPRINT-OR-NO-COMPARE"}
    check(expected <= names,
          f"invariant registry complete ({len(expected)} charter rules "
          f"present)")
    check(all(i.scar and i.rule and i.enforcement for i in INVARIANTS),
          "every invariant carries rule + scar + enforcement location")

    return check.finish("invariant-enforcement selftest")
