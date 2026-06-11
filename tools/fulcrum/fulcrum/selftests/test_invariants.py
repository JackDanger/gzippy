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
                 bin_sha="deadbeef", corpus_sha="028bd002",
                 protocol="fulcrum-v3", comparator="rapidgzip 0.16.0",
                 host="testcpu-13700T|6.0-test|abc123def456")


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
    txt = (open(man_path).read()
           .replace("sink_gz=regular-file", "sink_gz=devnull")
           .replace("sink_gz_derived=regular-file",
                    "sink_gz_derived=devnull"))
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

    # ------------------------------------------------------------------
    # DERIVED-NOT-SELF-REPORTED fields (P2 item 3): a lying manifest is
    # caught — the derivation governs the fingerprint, the self-report is
    # cross-checked and flagged.
    # ------------------------------------------------------------------
    from ..core.decide import canon_mask, derived_mismatches
    # Lying sink self-report (claim devnull, stat says regular-file): the
    # derived value governs (analysis proceeds, complete fingerprint), and
    # the lie is flagged verbatim.
    d_lie = tempfile.mkdtemp(prefix="fulcrum_inv_lying_")
    make_artifact(d_lie, with_knobs=False, v3=True)
    man_lie = os.path.join(d_lie, "manifest.txt")
    txt_lie = open(man_lie).read().replace("sink_gz=regular-file",
                                           "sink_gz=devnull")
    open(man_lie, "w").write(txt_lie)
    rep_lie = analyze_run(load_run(d_lie, AD), AD)
    check(any("DERIVED-MISMATCH" in a and "sink_gz=devnull" in a
              for a in rep_lie["anomalies"]),
          "lying manifest: self-reported sink contradicting the stat "
          "derivation is FLAGGED (DERIVED-MISMATCH, verbatim)")
    check(bool(rep_lie["scoreboard"]) and
          "FP-INCOMPLETE" not in rep_lie["scoreboard"][0],
          "lying manifest: the DERIVED value governs the fingerprint "
          "(cell still ranks, complete fingerprint)")
    # Lying freeze claim: frozen with NA sysfs readbacks is flagged.
    check(any("freeze_state=frozen claimed" in a for a in derived_mismatches(
        {"freeze_state": "frozen", "governor": "NA", "no_turbo": "1",
         "cell_meta": {}})),
          "lying manifest: freeze_state=frozen with an NA sysfs readback "
          "is FLAGGED (frozen requires READ values)")
    check(derived_mismatches(
        {"freeze_state": "frozen", "governor": "performance", "no_turbo": "1",
         "cell_meta": {}}) == [],
          "control: frozen with real sysfs readbacks passes the cross-check")
    # Mask: the taskset readback governs; a pin that did not take is flagged.
    mm = derived_mismatches({"cell_meta": {("silesia", 1):
                                           {"mask": "0", "maskd": "0-15"}}})
    check(any("pin did not take" in a for a in mm),
          "mask readback mismatch (requested 0, kernel says 0-15) FLAGGED — "
          "the pin did not take")
    check(derived_mismatches(
        {"cell_meta": {("silesia", 16):
                       {"mask": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                        "maskd": "0-15"}}}) == [],
          "mask canonical equivalence: '0-15' readback == the requested "
          "16-cpu list (formatting is not a lie)")
    check(canon_mask("0-15") == canon_mask(
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
        and canon_mask("0,2,4,6") == "0,2,4,6"
        and canon_mask("garbage") == "unknown",
          "canon_mask: range/list equivalence + unparseable => unknown")

    # ------------------------------------------------------------------
    # COMPARATOR-VERSION + HOST-IDENTITY fields (P2 item 2 — the
    # rapidgzip-version-drift and cross-host scenarios).
    # ------------------------------------------------------------------
    fp_rg17 = Fingerprint(**{**FP.to_dict(), "comparator": "rapidgzip 0.17.0"})
    check(any("comparator" in r for r in incompatibilities(FP, fp_rg17)),
          "comparator-version drift (0.16.0 vs 0.17.0) => incompatible, "
          "reason names the comparator field")
    fp_otherbox = Fingerprint(**{**FP.to_dict(),
                                 "host": "otherCPU|6.1|deadbeef0000"})
    check(any("host" in r for r in incompatibilities(FP, fp_otherbox)),
          "host-identity mismatch => incompatible, reason names host "
          "(same binary on a different box is a different experiment)")
    # The rapidgzip-version-drift ledger scenario: an rg wall banked under
    # 0.16.0 must NOT read as bank drift when 0.17.0 measures differently.
    led_v = Ledger(os.path.join(tempfile.mkdtemp(prefix="fulcrum_inv_vdrift_"),
                                "ledger.jsonl"))
    led_v.append(make_record("run_v16", "gzippy", "cell", "silesia:T1:rg",
                             917.0, 7, 1.0, "comparator", FP))
    live_v17 = make_record("run_v17", "gzippy", "cell", "silesia:T1:rg",
                           810.0, 7, 1.0, "comparator", fp_rg17)
    check(led_v.contradictions(live_v17) == [],
          "rapidgzip-version-drift: 0.17.0 wall differing from the banked "
          "0.16.0 wall is NOT a contradiction (different experiment, "
          "never compared)")
    live_same16 = make_record("run_v16b", "gzippy", "cell", "silesia:T1:rg",
                              810.0, 7, 1.0, "comparator", FP)
    check(len(led_v.contradictions(live_same16)) == 1,
          "control: the SAME comparator version drifting -21% DOES "
          "contradict (real bank drift still caught)")
    # comparator_version() probe: full banner, short form, absent.
    full_banner = ("rapidgzip, CLI to the parallelized, indexed, and seekable "
                   "gzip decoding library rapidgzip version 0.16.0")
    check(AD.comparator_version({"rg_version": full_banner})
          == "rapidgzip 0.16.0",
          "comparator_version probe normalizes the full --version banner")
    check(AD.comparator_version({"rg_version": "rapidgzip 0.16.0"})
          == "rapidgzip 0.16.0",
          "comparator_version probe accepts the short form")
    check(AD.comparator_version({}) == "unknown",
          "comparator_version probe: absent rg_version => unknown "
          "(never compares)")
    # host_identity(): all three fields or nothing.
    from ..core.decide import host_identity
    check(host_identity({"host_cpu_model": "c", "host_kernel": "k",
                         "host_id": "i"}) == "c|k|i",
          "host_identity composes cpu|kernel|id")
    check(host_identity({"host_cpu_model": "c", "host_kernel": "k"})
          == "unknown",
          "host_identity: partial identity => unknown (cannot certify "
          "same-host)")

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
    # The contradicting live row is NOT auto-banked as an anchor: it lands
    # pending-reconcile; the un-contradicted gz row of the same run banks
    # active.
    led_rt = Ledger(lpath)
    rg_rows = [r for r in led_rt.rows()
               if r.get("key") == "silesia:T1:rg" and r.get("runid") == "st2"]
    check(len(rg_rows) == 1 and rg_rows[0].get("status") == "pending-reconcile",
          "e2e ledger: contradicting live rg row banked pending-reconcile, "
          "not as an anchor")
    check(any("pending-reconcile" in a and "supersede" in a
              for a in rep_l2["anomalies"]),
          "e2e ledger: report names the pending-reconcile banking + the "
          "supersede resolution path")
    rg_anchors = led_rt.anchors(key="silesia:T1:rg")
    check([r.get("runid") for r in rg_anchors] == ["st"],
          "e2e ledger: anchor set for the contested key is still ONLY the "
          "original banked row (pending never anchors)")

    # ------------------------------------------------------------------
    # POISON-THEN-SUPERSEDE (the P2 gate case): a contested anchor is
    # retired by an explicit supersede record; the pending row is promoted;
    # subsequent runs compare against the NEW anchor only.
    # ------------------------------------------------------------------
    # A third drifted run still contradicts the ORIGINAL anchor — never the
    # pending row (a contested number must not become the next run's truth).
    live3 = make_record("st3", "gzippy", "cell", "silesia:T1:rg",
                        728.0, 7, 1.0, "comparator",
                        Fingerprint.from_dict(rg_anchors[0]["fingerprint"]))
    c3 = led_rt.contradictions(live3)
    check(len(c3) == 1 and "918.0ms banked" in c3[0],
          f"poison: 3rd drifted run contradicts ONLY the original anchor "
          f"(pending row never used as an anchor) — got {len(c3)} "
          f"contradiction(s)")
    # Resolve: supersede the original rg row, promoting the pending st2 row.
    led_rt.supersede("silesia:T1:rg", retire_runid="st",
                     reason="comparator rebuilt; old anchor measured a stale "
                            "binary", promote_runid="st2")
    rg_anchors2 = led_rt.anchors(key="silesia:T1:rg")
    check([r.get("runid") for r in rg_anchors2] == ["st2"],
          "supersede: retired row stops anchoring; promoted pending row IS "
          "the new anchor")
    check(led_rt.contradictions(live3) == [],
          "supersede: post-resolution drifted value agrees with the new "
          "anchor — no contradiction")
    # The file is append-only: the retired row is still IN the ledger.
    check(any(r.get("runid") == "st" and r.get("key") == "silesia:T1:rg"
              for r in led_rt.rows()),
          "supersede: retired row remains in the file (append-only — "
          "retired, not rewritten)")
    # has_run counts only measurement rows (a supersede carries no runid).
    check(led_rt.has_run("st2") and not led_rt.has_run("no-such-run"),
          "has_run: measurement rows only, unaffected by resolution records")

    # invalidate: a measurement-error row is retired outright, nothing
    # promoted.
    led_inv = Ledger(os.path.join(ltmp, "ledger_inv.jsonl"))
    led_inv.append(make_record("run_X", "gzippy", "cell", "model:T8:rg",
                               500.0, 7, 1.0, "comparator", FP))
    live_inv = make_record("run_Y", "gzippy", "cell", "model:T8:rg",
                           600.0, 7, 1.0, "comparator", FP)
    check(len(led_inv.contradictions(live_inv)) == 1,
          "invalidate setup: live contradicts the to-be-invalidated row")
    led_inv.invalidate("model:T8:rg", target_runid="run_X",
                       reason="mislabeled binary (native run as isal)")
    check(led_inv.contradictions(live_inv) == []
          and led_inv.anchors(key="model:T8:rg") == [],
          "invalidate: target row retired as an anchor; no promotion")

    # ------------------------------------------------------------------
    # Hash chain (tamper evidence for the append-only CONVENTION).
    # ------------------------------------------------------------------
    led_ch = Ledger(os.path.join(ltmp, "ledger_chain.jsonl"))
    led_ch.append(make_record("run_1", "gzippy", "cell", "k:T1:gz",
                              100.0, 7, 1.0, "gzippy", FP))
    led_ch.append(make_record("run_2", "gzippy", "cell", "k:T1:gz",
                              101.0, 7, 1.0, "gzippy", FP))
    led_ch.supersede("k:T1:gz", retire_runid="run_1", reason="test")
    check(all(r.get("chain") for r in led_ch.rows())
          and led_ch.verify_chain() == [],
          "hash chain: every appended record chained; verify_chain clean on "
          "an untampered ledger")
    # Tamper with a middle row's value (keeping its stored chain): caught.
    with open(led_ch.path) as f:
        lines = f.readlines()
    lines[1] = lines[1].replace('"value_ms": 101.0', '"value_ms": 90.0')
    with open(led_ch.path, "w") as f:
        f.writelines(lines)
    breaks = led_ch.verify_chain()
    check(len(breaks) >= 1 and "append-only violated" in breaks[0],
          "hash chain: edited row CAUGHT by verify_chain (tamper evidence)")
    # Pre-chain rows (legacy ledger) are tolerated; chained suffix verifies.
    led_old = Ledger(os.path.join(ltmp, "ledger_prechain.jsonl"))
    with open(led_old.path, "w") as f:
        f.write('{"kind": "cell", "key": "old:T1:gz", "runid": "r0", '
                '"value_ms": 5.0}\n')
    led_old.append(make_record("r1", "gzippy", "cell", "old:T1:gz",
                               6.0, 7, 1.0, "gzippy", FP))
    check(led_old.verify_chain() == [],
          "hash chain: pre-chain legacy rows tolerated (chain covers the "
          "chained suffix only — documented honestly)")

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
