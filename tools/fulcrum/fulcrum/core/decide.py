"""The ranked decision engine: artifact-dir -> decision table + brief.

Consumes the artifact directory produced by a project's measurement policy
(for gzippy: scripts/bench/decide.sh / _decide_guest.sh) and renders ONE ranked
component table where every row carries:
  - cells affected + wall-ms attribution (canonical-mask trace decomposition),
  - CAUSAL STATUS: tool-executed kill-switch A/B verdict for knob-covered
    components; HYPOTHESIS + the exact suggested perturbation for everything
    else — NEVER a recommendation without a knob (CAUSAL-OR-HYPOTHESIS),
  - DISTRIBUTION HEALTH: spread, bimodality, RESOLVED/UNRESOLVED + N-needed
    (SPREAD-RESOLUTION),
  - the EXACT re-verify command,
plus a DECISION BRIEF: top action + causal evidence + preconditions + command +
the result that would falsify it.

Every wall number is fingerprinted ({sink, mask, freeze, bin sha, corpus sha,
protocol}); ratios across incompatible fingerprints are REFUSED (SINK-LAW /
FINGERPRINT-OR-NO-COMPARE), and verdicts are banked to / cross-checked against
the append-only results ledger (CONTRADICTS-LEDGER).
"""

import os
import re

from .. import PROTOCOL_VERSION
from . import trace as tr
from .causal import knob_verdict
from .fingerprint import Fingerprint, assert_comparable, incompatibilities
from .ledger import PENDING, make_record
from .stats import bimodal, dist_health_str, read_samples, resolution, sample_stats

# ---------------------------------------------------------------------------
# Artifact-dir loading.
# ---------------------------------------------------------------------------

def parse_manifest(path):
    mf = {"cells_done": [], "knobs_done": [], "knob_sha_fail": [],
          "cell_meta": {}}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k == "cell_done":
                mf["cells_done"].append(v)
                # "corpus:T:mask=M:sha_ok=1" -> structured per-cell meta.
                parts = v.split(":")
                if len(parts) >= 2:
                    try:
                        ck = (parts[0], int(parts[1]))
                    except ValueError:
                        continue
                    meta = {}
                    for p in parts[2:]:
                        if "=" in p:
                            mk, mv = p.split("=", 1)
                            meta[mk] = mv
                    mf["cell_meta"][ck] = meta
            elif k == "knob_done":
                mf["knobs_done"].append(v)
            elif k == "knob_sha_fail":
                mf["knob_sha_fail"].append(v)
            else:
                mf[k] = v
    return mf


def cell_key(corpus, t):
    return (corpus, int(t))


def fmt_cell(ck):
    return f"{ck[0]}:T{ck[1]}"


def load_run(art_dir, adapter):
    """Load a run via the ADAPTER (pluggable): the default ProjectAdapter
    delegates back to load_run_documented below (the documented schema,
    docs/SCHEMA.md); a project with its own artifact layout overrides
    ProjectAdapter.load_run and maps to the same run-dict shape."""
    return adapter.load_run(art_dir)


def load_run_documented(art_dir, adapter):
    """The documented-schema loader (docs/SCHEMA.md): manifest.txt +
    cell_<corpus>_T<threads>/ sample files + knob_<name>/ A/B dirs +
    knob_effects_<corpus>_T<T>/ captures."""
    man_path = os.path.join(art_dir, "manifest.txt")
    if not os.path.exists(man_path):
        raise tr.InstrumentError(f"no manifest.txt in {art_dir} — not a decide "
                                 f"artifact dir")
    man = parse_manifest(man_path)
    run = {"manifest": man, "cells": {}, "dir": art_dir}
    for name in sorted(os.listdir(art_dir)):
        m = re.match(r"cell_([a-z0-9]+)_T(\d+)$", name)
        if not m:
            continue
        cdir = os.path.join(art_dir, name)
        ck = cell_key(m.group(1), m.group(2))
        cell = {"dir": cdir,
                "gz": read_samples(os.path.join(cdir, "wall_gz.txt")),
                "rg": read_samples(os.path.join(cdir, "wall_rg.txt")),
                "knobs": {}}
        ptxt = os.path.join(cdir, "prof.txt")
        cell["prof"] = None
        if os.path.exists(ptxt):
            with open(ptxt) as pf:
                cell["prof"] = adapter.parse_microprofile(pf.read())
        cell["trace"] = os.path.join(cdir, "trace.json")
        cell["verbose"] = os.path.join(cdir, "verbose.txt")
        for kn in sorted(os.listdir(cdir)):
            km = re.match(r"knob_(\w+)$", kn)
            if not km:
                continue
            kd = os.path.join(cdir, kn)
            meta = {}
            meta_path = os.path.join(kd, "meta.txt")
            if os.path.exists(meta_path):
                with open(meta_path) as mfh:
                    meta = dict(ln.strip().split("=", 1)
                                for ln in mfh if "=" in ln)
            cell["knobs"][km.group(1)] = {
                "base": read_samples(os.path.join(kd, "base.txt")),
                "knob": read_samples(os.path.join(kd, "knob.txt")),
                "meta": meta,
            }
        run["cells"][ck] = cell
    # knob effect captures
    run["effects"] = {}
    for name in sorted(os.listdir(art_dir)):
        m = re.match(r"knob_effects_([a-z0-9]+)_T(\d+)$", name)
        if not m:
            continue
        edir = os.path.join(art_dir, name)
        for f in os.listdir(edir):
            fm = re.match(r"effect_(base|knob)_(\w+)\.txt$", f)
            if fm:
                with open(os.path.join(edir, f)) as efh:
                    run["effects"].setdefault(
                        fm.group(2), {})[fm.group(1)] = efh.read()
    return run


# ---------------------------------------------------------------------------
# Fingerprints (SINK-LAW / FINGERPRINT-OR-NO-COMPARE enforcement points).
# ---------------------------------------------------------------------------

def canon_mask(mask):
    """Canonicalize a cpu-list string ('0-15', '0,2,4,6', '0,1-3,7') to a
    sorted comma list so a requested mask and a kernel readback compare by
    MEANING, not formatting. Unparseable/empty -> 'unknown'."""
    if not mask or mask == "unknown":
        return "unknown"
    cpus = set()
    try:
        for part in str(mask).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                cpus.update(range(int(lo), int(hi) + 1))
            else:
                cpus.add(int(part))
    except ValueError:
        return "unknown"
    return ",".join(str(c) for c in sorted(cpus)) if cpus else "unknown"


def derived_mismatches(man):
    """Cross-check self-reported manifest fields against their runner-DERIVED
    duplicates. A field that CAN be derived (sink class via stat, mask via
    taskset readback, freeze via sysfs) must be; a self-report contradicting
    the derivation is a lying/stale manifest and is flagged — the DERIVED
    value governs the fingerprint. Returns anomaly strings."""
    out = []
    for claim_key, derived_key in (("sink_gz", "sink_gz_derived"),
                                   ("sink_rg", "sink_rg_derived")):
        c, d = man.get(claim_key), man.get(derived_key)
        if c and d and c != d:
            out.append(
                f"DERIVED-MISMATCH: manifest self-reports {claim_key}={c} but "
                f"the runner derived {d} via stat — lying/stale manifest; the "
                f"DERIVED value governs the fingerprint")
    if man.get("freeze_state") == "frozen" and (
            man.get("governor") in (None, "", "NA")
            or man.get("no_turbo") in (None, "", "NA")):
        out.append(
            "DERIVED-MISMATCH: freeze_state=frozen claimed but the sysfs "
            "readbacks are NA (governor="
            f"{man.get('governor')!r}, no_turbo={man.get('no_turbo')!r}) — "
            "'frozen' requires READ values; treat the freeze claim as "
            "unverified")
    for ck, meta in man["cell_meta"].items():
        req, drv = meta.get("mask"), meta.get("maskd")
        if req and drv and drv != "unreadable" \
                and canon_mask(req) != canon_mask(drv):
            out.append(
                f"DERIVED-MISMATCH: {fmt_cell(ck)} requested mask={req} but "
                f"the taskset readback was {drv} — the pin did not take "
                f"(cpuset shrink / bad list); the READBACK governs the "
                f"fingerprint")
    return out


def host_identity(man):
    """Compose the host-identity fingerprint field from the manifest's
    derived host fields: "cpu-model|kernel|host-id". ALL three must be known
    (a partial identity cannot certify same-host); otherwise 'unknown'."""
    cpu = (man.get("host_cpu_model") or "").strip()
    kernel = (man.get("host_kernel") or "").strip()
    hid = (man.get("host_id") or "").strip()
    if cpu and kernel and hid:
        return f"{cpu}|{kernel}|{hid}"
    return "unknown"


def cell_fingerprints(man, ck, adapter):
    """Build the (tool-under-test, comparator) fingerprints for one cell from
    the manifest. Manifests predating a fingerprint field -> 'unknown'
    (label, never silently compatible)."""
    proto = man.get("protocol", "unknown")
    freeze = man.get("freeze_state", "unknown")
    corpus_sha = man.get(f"corpus_{ck[0]}_sha", "unknown")
    meta = man["cell_meta"].get(ck, {})
    # DERIVED values govern (taskset readback / stat); self-reports are the
    # fallback for artifacts predating derivation, cross-checked by
    # derived_mismatches().
    maskd = meta.get("maskd")
    mask = canon_mask(maskd if maskd and maskd != "unreadable"
                      else meta.get("mask", "unknown"))
    sink_default = man.get("sink_class", "unknown")
    sink_gz = man.get("sink_gz_derived") or man.get("sink_gz", sink_default)
    sink_rg = man.get("sink_rg_derived") or man.get("sink_rg", sink_default)
    comparator = adapter.comparator_version(man)
    host = host_identity(man)
    fp_gz = Fingerprint(sink=sink_gz, mask=mask,
                        freeze=freeze, bin_sha=man.get("bin_sha", "unknown"),
                        corpus_sha=corpus_sha, protocol=proto,
                        comparator=comparator, host=host)
    fp_rg = Fingerprint(sink=sink_rg, mask=mask,
                        freeze=freeze,
                        bin_sha="comparator:" + man.get("rg_version", "unknown"),
                        corpus_sha=corpus_sha, protocol=proto,
                        comparator=comparator, host=host)
    return fp_gz, fp_rg


def check_cell_comparable(fp_gz, fp_rg, ck):
    """SINK-LAW / FINGERPRINT-OR-NO-COMPARE for one cell's gz:rg ratio.

    A CONCRETE mismatch (both sides known, different — e.g. one arm /dev/null,
    one arm file) is REFUSED outright: that is the half-rebased-table phantom.
    Unknown-only gaps (an artifact predating a fingerprint field) downgrade to
    a label so old artifact dirs stay analyzable — but are never banked.
    Returns the label string ('' == fully comparable)."""
    inc = incompatibilities(fp_gz, fp_rg)
    if any("mismatch" in r for r in inc):
        assert_comparable(fp_gz, fp_rg, what=f"cell ratio {fmt_cell(ck)}")
    if inc:
        missing = ",".join(sorted({r.split()[0] for r in inc}))
        return f" FP-INCOMPLETE(fields unknown: {missing} — not banked)"
    return ""


# ---------------------------------------------------------------------------
# Freeze gate (FROZEN-OR-LABELED).
# ---------------------------------------------------------------------------

def frozen_ok(man):
    return (man.get("freeze_state") in ("frozen", "acknowledged")
            and man.get("quiet_state") == "quiet")


# ---------------------------------------------------------------------------
# The decision table.
# ---------------------------------------------------------------------------

def analyze_run(run, adapter, allow_thaw=False, feature=None, ledger=None):
    man = run["manifest"]
    feature = feature or man.get("feature")
    rows = []
    header = []
    anomalies = []

    ok_frozen = frozen_ok(man)
    if not ok_frozen and not allow_thaw:
        raise tr.InstrumentError(
            f"run NOT frozen/quiet (freeze_state={man.get('freeze_state')}, "
            f"quiet_state={man.get('quiet_state')}) — REFUSING to rank wall "
            f"numbers. Pass --allow-thaw to label instead. [FROZEN-OR-LABELED]")
    unfrozen_tag = "" if ok_frozen else " [UNFROZEN — ratio-only, do not bank]"

    # Derived-vs-self-reported cross-check (a lying manifest is caught here;
    # the DERIVED values are the ones the fingerprints below are built from).
    anomalies.extend(derived_mismatches(man))

    proto = man.get("protocol", "unknown")
    proto_tag = "" if proto == PROTOCOL_VERSION else \
        f" [protocol={proto} != analyzer {PROTOCOL_VERSION}]"
    header.append(f"run        : {man.get('runid')}  bin={man.get('bin')} "
                  f"sha={str(man.get('bin_sha'))[:16]} feature={feature}")
    header.append(f"box        : freeze={man.get('freeze_state')} "
                  f"quiet={man.get('quiet_state')} governor={man.get('governor')} "
                  f"no_turbo={man.get('no_turbo')} "
                  f"runnable_avg={man.get('runnable_avg')}{unfrozen_tag}")
    header.append(f"comparator : {man.get('rg_version')} "
                  f"[fingerprint: {adapter.comparator_version(man)}]")
    header.append(f"fingerprint: protocol={proto}{proto_tag} "
                  f"sink_gz={man.get('sink_gz', man.get('sink_class', 'unknown'))} "
                  f"sink_rg={man.get('sink_rg', man.get('sink_class', 'unknown'))} "
                  f"host={host_identity(man)} "
                  f"(per-cell mask + corpus pin in each row's fingerprint)")
    header.append(f"sha-verify : every measured run checked against the corpus "
                  f"pin (guest aborts on mismatch); "
                  f"cells_done={len(man['cells_done'])}")

    # ---- per-cell wall scoreboard (fingerprint-gated ratios) -------------------
    cell_walls = {}
    scoreboard = []
    for ck, cell in sorted(run["cells"].items()):
        sg, sr = sample_stats(cell["gz"]), sample_stats(cell["rg"])
        if not sg or not sr:
            continue
        # SHA-OR-VOID: a cell directory without its manifest completion row
        # (sha_ok recorded by the guest) is VOID — render the anomaly, skip.
        meta = man["cell_meta"].get(ck)
        if meta is not None and meta.get("sha_ok") != "1":
            anomalies.append(f"{fmt_cell(ck)}: cell present but sha_ok!=1 in "
                             f"manifest — VOID (SHA-OR-VOID), not ranked")
            continue
        fp_gz, fp_rg = cell_fingerprints(man, ck, adapter)
        fp_label = check_cell_comparable(fp_gz, fp_rg, ck)  # may raise
        ratio = sr["min"] / sg["min"] if sg["min"] else 0.0
        delta_s = sg["min"] - sr["min"]
        res, n_need = resolution(delta_s,
                                 sg["spread_pct"] / 100 * sg["min"],
                                 sr["spread_pct"] / 100 * sr["min"], sg["n"])
        verdict = "PASS" if ratio >= adapter.tie_bar else "FAIL"
        bm = ("gz" if bimodal(cell["gz"]) else "") + \
             ("+rg" if bimodal(cell["rg"]) else "")
        cell_walls[ck] = {"gz": sg, "rg": sr, "ratio": ratio,
                          "gap_ms": delta_s * 1000.0, "resolution": res,
                          "n_needed": n_need, "verdict": verdict,
                          "fp_gz": fp_gz, "fp_rg": fp_rg,
                          "fp_label": fp_label}
        scoreboard.append(
            f"  {fmt_cell(ck):13s} gz={sg['min']*1000:7.1f}ms "
            f"rg={sr['min']*1000:7.1f}ms "
            f"ratio={ratio:.3f} {verdict:4s} {res}"
            + (f"(N->{n_need})" if n_need else "")
            + f" spread gz={sg['spread_pct']:.1f}%/rg={sr['spread_pct']:.1f}%"
            + (f" BIMODAL[{bm}]" if bm else "")
            + fp_label)

    # ---- ledger: contradiction scan + banking (FINGERPRINT-aware) --------------
    ledger_notes = []
    if ledger is not None:
        already = ledger.has_run(man.get("runid"))
        n_banked = 0
        n_pending = 0
        for ck, w in sorted(cell_walls.items()):
            if w["fp_label"] or not ok_frozen:
                continue   # incomplete fingerprint / unfrozen: never banked
            recs = [
                make_record(man.get("runid"), adapter.name, "cell",
                            f"{fmt_cell(ck)}:gz", w["gz"]["min"] * 1000,
                            w["gz"]["n"], w["gz"]["spread_pct"],
                            adapter.name, w["fp_gz"]),
                make_record(man.get("runid"), adapter.name, "cell",
                            f"{fmt_cell(ck)}:rg", w["rg"]["min"] * 1000,
                            w["rg"]["n"], w["rg"]["spread_pct"],
                            "comparator", w["fp_rg"]),
            ]
            for rec in recs:
                contras = ledger.contradictions(rec)
                for c in contras:
                    anomalies.append(c)
                if already:
                    continue
                if contras:
                    # A contradicting live number is NEVER auto-banked as an
                    # anchor: it lands pending-reconcile until a supersede
                    # record names which side was wrong.
                    rec["status"] = PENDING
                    ledger.append(rec)
                    n_pending += 1
                    anomalies.append(
                        f"{rec['key']}: live row banked {PENDING} (not an "
                        f"anchor). After reconciling, resolve with: "
                        f"fulcrum ledger supersede --key '{rec['key']}' "
                        f"--retire <banked-runid> --promote {rec['runid']} "
                        f"--reason '<why the banked row is retired>'")
                else:
                    ledger.append(rec)
                    n_banked += 1
        ledger_notes.append(
            f"ledger     : {ledger.path} "
            + ("(run already banked — re-analysis, nothing appended)"
               if already else
               f"(banked {n_banked} rows"
               + (f", {n_pending} {PENDING}" if n_pending else "") + ")"))

    # ---- trace decomposition per cell (canonical mask) -------------------------
    trace_components = {}   # cls -> {cell: (ms, span_ms)}
    for ck, cell in sorted(run["cells"].items()):
        if not os.path.exists(cell["trace"]) or \
                os.path.getsize(cell["trace"]) == 0:
            anomalies.append(f"{fmt_cell(ck)}: trace absent/empty — attribution "
                             f"rows skipped for this cell")
            continue
        try:
            b = tr.analyze(cell["trace"], adapter,
                           counter_path=cell["verbose"],
                           declared_T=str(ck[1]), feature=feature)
        except tr.InstrumentError as e:
            anomalies.append(f"{fmt_cell(ck)}: trace engine REFUSED trace: {e}")
            continue
        if b["is_production"] is False:
            anomalies.append(f"{fmt_cell(ck)}: routing guard REFUSED "
                             f"({b['seed_reason']}) — attribution rows dropped")
            continue
        cons = b["consumer"]
        span = cons.get("span", 0)
        if not span:
            continue
        for cls in ("compute", "output", "wait", "idle"):
            ms = cons.get(cls, 0.0) / 1000.0
            trace_components.setdefault(cls, {})[ck] = (ms, span / 1000.0)

    # ---- knob rows (the causal tier) -------------------------------------------
    for ck, cell in sorted(run["cells"].items()):
        for kname, kdata in sorted(cell["knobs"].items()):
            kn = adapter.knobs.get(kname)
            envkv, pred, desc = (kn.env, kn.pred, kn.desc) if kn \
                else ("?", "none", kname)
            v = knob_verdict(kdata["base"], kdata["knob"])
            if v.get("status") == "NO-DATA":
                continue
            eff = run["effects"].get(kname)
            ev, enote = adapter.effect_check(
                pred, eff.get("base", "") if eff else "",
                eff.get("knob", "") if eff else "")
            if ev is False:
                status = f"EFFECT-CHECK-FAILED ({enote}) — A/B NOT causal"
                tier, rank = 5, 0.0
            else:
                d, mg = v["delta_ms"], v["margin_ms"]
                if v["status"] == "CAUSAL-VERIFIED-COSTS":
                    status = (f"CAUSAL-VERIFIED: shipped default COSTS "
                              f"{-d:.1f}ms max-arm-spread={mg:.1f}ms here "
                              f"(alt arm faster)")
                    tier, rank = 1, -d
                elif v["status"] == "CAUSAL-VERIFIED-PAYS":
                    status = (f"CAUSAL-VERIFIED: feature PAYS {d:.1f}ms "
                              f"max-arm-spread={mg:.1f}ms (disabling it loses)")
                    tier, rank = 3, d
                else:
                    status = (f"CAUSAL-NULL: |Δ|={abs(d):.1f}ms ≤ "
                              f"max-arm-spread={mg:.1f}ms (bounded)")
                    tier, rank = 4, abs(d)
                if ev is None:
                    status += f" [{enote}]"
                elif ev is True:
                    status += f" [effect-verified: {enote}]"
            dist = (f"base[{dist_health_str(kdata['base'])}] "
                    f"knob[{dist_health_str(kdata['knob'])}] "
                    f"{v.get('resolution', '')}"
                    + (f"(N->{v['n_needed']})" if v.get("n_needed") else ""))
            meta = kdata.get("meta", {})
            rss_base = meta.get("rss_base_mb", "")
            rss_knob = meta.get("rss_knob_mb", "")
            if rss_base and rss_knob:
                try:
                    rb_f, rk_f = float(rss_base), float(rss_knob)
                    pct = (rk_f - rb_f) / rb_f * 100 if rb_f else 0.0
                    sign = "+" if pct >= 0 else ""
                    rss_str = (f"rss base={rb_f:.0f}MB knob={rk_f:.0f}MB "
                               f"({sign}{pct:.0f}%)")
                except (ValueError, ZeroDivisionError):
                    rss_str = f"rss base={rss_base}MB knob={rss_knob}MB"
            else:
                rss_str = "rss N/A (pre-RSS capture run)"
            rows.append({
                "component": f"knob.{kname} ({desc})",
                "kind": "knob",
                "reverted": bool(kn.reverted) if kn else False,
                "cells": fmt_cell(ck),
                "attrib": f"Δ(alt-base)={v['delta_ms']:+.1f}ms @ canonical mask",
                "status": status + unfrozen_tag,
                "dist": dist,
                "rss": rss_str,
                "verify": adapter.reverify_knob(ck, kname, run),
                "tier": tier, "rank_ms": rank,
                "effect_verified": ev,
                "n_needed": v.get("n_needed"),
            })

    # ---- engine micro-profile rows (per corpus; HYPOTHESIS tier) ----------------
    for ck, cell in sorted(run["cells"].items()):
        w = cell_walls.get(ck)
        gap_ms = max(w["gap_ms"], 0.0) if w else 0.0
        prows, panoms = adapter.microprofile_rows(ck, cell.get("prof"),
                                                  gap_ms, run)
        rows.extend(prows)
        anomalies.extend(panoms)

    # ---- trace-component rows (HYPOTHESIS tier) ---------------------------------
    for cls, cells in sorted(trace_components.items()):
        worst_ck, (worst_ms, span_ms) = max(cells.items(),
                                            key=lambda kv: kv[1][0])
        cells_str = ",".join(fmt_cell(c) for c in sorted(cells))
        share = 100.0 * worst_ms / span_ms if span_ms else 0
        perturb = adapter.perturbations.get(cls, "design a knob first")
        rows.append({
            "component": f"pipeline.consumer.{cls}",
            "kind": "pipeline",
            "perturb_cmd": perturb,
            "cells": cells_str,
            "attrib": (f"worst {fmt_cell(worst_ck)}: {worst_ms:.1f}ms "
                       f"({share:.0f}% of wall-critical span)"),
            "status": (f"HYPOTHESIS (attribution only — NOT causal). Perturb: "
                       f"{perturb}"),
            "dist": "trace=1-shot (unfrozen-counters label)",
            "verify": adapter.reverify_trace(worst_ck, run, feature),
            "tier": 2,
            "rank_ms": worst_ms if cls != "wait" else worst_ms * 0.25,
            # wait is demoted: it is a SYMPTOM of the producer, not a lever site.
        })

    rows.sort(key=lambda r: (r["tier"], -r["rank_ms"]))

    # ---- DO THIS NEXT + the decision brief ----------------------------------------
    do_next, brief = build_brief(rows, cell_walls, man, adapter, ok_frozen)

    return {"header": header + ledger_notes, "scoreboard": scoreboard,
            "rows": rows, "anomalies": anomalies, "do_next": do_next,
            "brief": brief, "cell_walls": cell_walls}


def build_brief(rows, cell_walls, man, adapter, ok_frozen):
    """The decision brief: ACTION / WHY / PRECONDITIONS / COMMAND / FALSIFIER.
    Returns (do_next_line, brief_dict)."""
    failing = sorted(((ck, w) for ck, w in cell_walls.items()
                      if w["verdict"] == "FAIL"),
                     key=lambda kv: kv[1]["ratio"])
    precond_common = [
        f"box {'frozen+quiet' if ok_frozen else 'NOT frozen/quiet — label-only'}",
        f"binary sha {str(man.get('bin_sha'))[:16]} staged at {man.get('bin')}",
        "fingerprints complete on every banked cell (sink/mask/freeze/"
        "corpus/protocol)",
    ]
    if failing:
        worst = ", ".join(f"{fmt_cell(ck)} {w['ratio']:.3f}"
                          for ck, w in failing[:4])
        precond_common.append(f"failing cells (bar {adapter.tie_bar}): {worst}")

    for r in rows:
        if r["tier"] == 1:
            if r.get("reverted"):
                action = ("reconcile with the prior gated revert + check RSS "
                          "before flipping")
            else:
                action = "fix/condition the feature"
            do_next = (f"{r['component']} on {r['cells']} — the shipped default "
                       f"measurably COSTS wall "
                       f"({r['status'].split(':')[1].strip()}). "
                       f"Re-verify at N=21 then {action}: {r['verify']}")
            brief = {
                "action": f"{action} — {r['component']} on {r['cells']}",
                "evidence": (f"tool-executed same-binary A/B: {r['status']}; "
                             f"distribution: {r['dist']}; {r.get('rss', '')}"),
                "preconditions": precond_common + [
                    "effect predicate: "
                    + {True: "VERIFIED (switch engagement counter-proven)",
                       None: "UNVERIFIED (no in-tree counter — wall-only A/B)",
                       False: "FAILED"}[r.get("effect_verified")],
                ],
                "command": r["verify"],
                "falsifier": ("re-run at N=21 under the SAME fingerprint: if "
                              "|Δ| ≤ max-arm-spread (CAUSAL-NULL) or the sign "
                              "flips, this action is refuted; a knob arm sha "
                              "mismatch voids it (SHA-OR-VOID)."),
            }
            return do_next, brief
    for r in rows:
        if r["tier"] == 2 and r.get("kind") == "engine":
            # The exact perturbation: the row's own pre-registered command
            # (row contract), falling back to the adapter's compute-class
            # perturbation, then to the re-verify command — never a KeyError
            # on an adapter without a 'compute' perturbation.
            perturb = (r.get("perturb_cmd")
                       or adapter.perturbations.get("compute")
                       or r.get("verify", "design a perturbation knob first"))
            do_next = (f"{r['component']} on {r['cells']} — top bounded "
                       f"HYPOTHESIS ({r['attrib']}). Run the pre-registered "
                       f"perturbation BEFORE any work-stretch: {perturb}")
            brief = {
                "action": (f"causally test {r['component']} on {r['cells']} "
                           f"(top bounded HYPOTHESIS — not yet actionable)"),
                "evidence": (f"attribution only: {r['attrib']}; "
                             f"{r['status'].split('Perturb:')[0].strip()}"),
                "preconditions": precond_common + [
                    "CAUSAL-OR-HYPOTHESIS: no work-stretch before the "
                    "perturbation converts this to a causal verdict",
                ],
                "command": perturb,
                "falsifier": ("a flat (≤ inter-run spread) interleaved wall "
                              "response to the slow-injection — confirmed by "
                              "the frequency-neutral sleep control — refutes "
                              "this component as a wall binder; the bounded-ms "
                              "figure is a partition of the gap, not a "
                              "promise."),
            }
            return do_next, brief
    if rows:
        r = rows[0]
        do_next = (f"{r['component']} — see its row; no causal action surfaced.")
        brief = {
            "action": f"investigate {r['component']} (no causal action surfaced)",
            "evidence": r["status"],
            "preconditions": precond_common,
            "command": r["verify"],
            "falsifier": "n/a — no recommendation is being made",
        }
        return do_next, brief
    do_next = "no rankable rows (all captures refused?) — fix the run first."
    return do_next, {
        "action": "fix the run (no rankable rows)",
        "evidence": "all captures refused or absent",
        "preconditions": precond_common,
        "command": "re-run the measurement with captures enabled",
        "falsifier": "n/a",
    }
