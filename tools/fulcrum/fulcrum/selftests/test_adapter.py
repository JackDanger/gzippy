"""Adapter-pluggability self-tests (P2 item 4).

A SECOND project must be able to use fulcrum without touching gzippy code:
  - a toy adapter with its OWN artifact layout (a single run.json) overrides
    ProjectAdapter.load_run and flows end-to-end through analyze_run to a
    ranked table + decision brief;
  - an adapter with EMPTY perturbations never crashes the brief builder
    (the perturbations['compute'] KeyError class);
  - the brief builder keys on STRUCTURED row fields (kind / reverted), not
    component-name string matches (positive AND negative controls);
  - the default documented-schema loader is reachable through the adapter
    (core load_run delegates to ProjectAdapter.load_run).
"""

import json
import os
import tempfile

from ..adapters.base import Knob, ProjectAdapter
from ..core.decide import analyze_run, build_brief, load_run
from . import Checker
from .test_decide import GZ, RG, make_artifact


class ToyAdapter(ProjectAdapter):
    """A minimal second project: custom artifact layout (one run.json),
    own knob registry, NO perturbations, no micro-profile, no counters."""

    name = "toyproj"
    tie_bar = 0.95
    knobs = {"cache": Knob("TOY_CACHE=0", "none", "toy cache layer")}
    perturbations = {}   # deliberately empty — the KeyError regression case

    def load_run(self, art_dir):
        with open(os.path.join(art_dir, "run.json")) as f:
            raw = json.load(f)
        man = {"cells_done": [], "knobs_done": [], "knob_sha_fail": [],
               "cell_meta": {}}
        man.update(raw["manifest"])
        cells = {}
        for ckey, c in raw["cells"].items():
            corpus, t = ckey.split(":")
            ck = (corpus, int(t))
            man["cell_meta"][ck] = {"mask": "0", "sha_ok": "1"}
            man["cells_done"].append(ckey)
            cells[ck] = {
                "dir": art_dir,
                "gz": c["tool"],
                "rg": c["comparator"],
                "prof": None,
                "trace": os.path.join(art_dir, "no-trace.json"),
                "verbose": os.path.join(art_dir, "no-verbose.txt"),
                "knobs": {k: {"base": v["base"], "knob": v["knob"],
                              "meta": {"sha_ok": "1"}}
                          for k, v in c.get("knobs", {}).items()},
            }
        return {"manifest": man, "cells": cells, "dir": art_dir,
                "effects": {}}


TOY_MANIFEST = {
    "runid": "toy_run_1", "bin": "/opt/toy/bin", "bin_sha": "t0y5ha",
    "freeze_state": "frozen", "quiet_state": "quiet",
    "protocol": "fulcrum-v3",
    "sink_gz": "regular-file", "sink_rg": "regular-file",
    "corpus_toy_sha": "feedf00d",
    "comparator_version": "toycomp 1.0",     # the base-adapter probe key
    "host_cpu_model": "toycpu", "host_kernel": "1.0-toy",
    "host_id": "0123456789ab",
}


def write_toy_run(d, knob_delta_s=-0.060):
    run = {
        "manifest": TOY_MANIFEST,
        "cells": {
            "toy:1": {
                "tool": GZ,
                "comparator": RG,
                "knobs": {"cache": {"base": GZ,
                                    "knob": [x + knob_delta_s for x in GZ]}},
            },
        },
    }
    with open(os.path.join(d, "run.json"), "w") as f:
        json.dump(run, f)


def run():
    check = Checker()
    print("=== fulcrum selftest: adapter pluggability (toy project) ===")

    # 1. End-to-end: custom-layout run through load_run -> analyze_run ->
    #    brief, zero gzippy involvement.
    d = tempfile.mkdtemp(prefix="fulcrum_toy_")
    write_toy_run(d)
    toy = ToyAdapter()
    run_ = load_run(d, toy)   # core delegates to ToyAdapter.load_run
    check(("toy", 1) in run_["cells"] and len(run_["cells"][("toy", 1)]["gz"])
          == len(GZ),
          "toy adapter: custom run.json layout loads through core load_run "
          "(adapter delegation)")
    rep = analyze_run(run_, toy)
    check(bool(rep["scoreboard"])
          and "FP-INCOMPLETE" not in rep["scoreboard"][0],
          "toy adapter: cell ranks with a COMPLETE fingerprint (sink/mask/"
          "freeze/corpus/protocol/comparator/host all manifest-supplied)")
    check(any(r.get("kind") == "knob" and r["tier"] == 1
              for r in rep["rows"]),
          "toy adapter: toy knob CAUSAL-VERIFIED-COSTS lands tier 1")
    check("knob.cache" in rep["do_next"] and "fix/condition" in rep["do_next"],
          "toy adapter: DO-THIS-NEXT built from the toy knob (empty "
          "perturbations — no KeyError)")
    check(bool(rep.get("brief")) and rep["brief"]["command"],
          "toy adapter: decision brief complete with a command")

    # 2. STRUCTURED reverted flag (not a component-name string match).
    class ToyRevertedAdapter(ToyAdapter):
        knobs = {"cache": Knob("TOY_CACHE=0", "none",
                               "toy cache layer", reverted=True)}

    rep_rev = analyze_run(load_run(d, ToyRevertedAdapter()),
                          ToyRevertedAdapter())
    check("reconcile" in rep_rev["do_next"],
          "structured reverted=True => 'reconcile' phrasing (desc never "
          "contains the word)")

    class ToyDescTrapAdapter(ToyAdapter):
        # NEGATIVE control: the WORD 'reverted' in the desc must NOT trigger
        # the reconcile path (the old string-match bug).
        knobs = {"cache": Knob("TOY_CACHE=0", "none",
                               "cache layer (was reverted upstream, "
                               "unrelated)", reverted=False)}

    rep_trap = analyze_run(load_run(d, ToyDescTrapAdapter()),
                           ToyDescTrapAdapter())
    check("fix/condition" in rep_trap["do_next"]
          and "reconcile" not in rep_trap["do_next"],
          "negative control: 'reverted' in the DESC text does not trigger "
          "reconcile (structured field, not string match)")

    # 3. build_brief tier-2 robustness: engine-kind row, adapter without a
    #    'compute' perturbation -> falls back row perturb_cmd -> verify;
    #    never a KeyError.
    man = {"bin_sha": "t0y5ha", "bin": "/opt/toy/bin"}
    row = {"component": "engine.hotloop", "kind": "engine", "tier": 2,
           "rank_ms": 50.0, "cells": "toy:T1",
           "attrib": "50% of classed cycles",
           "status": "HYPOTHESIS: bounded. Perturb: <row-level>",
           "dist": "prof=1-shot", "verify": "rerun the toy profile"}
    do_next, brief = build_brief([row], {}, man, ToyAdapter(), True)
    check("rerun the toy profile" in brief["command"],
          "build_brief: empty perturbations + no perturb_cmd -> verify "
          "fallback (no KeyError)")
    row_p = dict(row, perturb_cmd="TOY_SLOW=50 toy-bench")
    _, brief_p = build_brief([row_p], {}, man, ToyAdapter(), True)
    check(brief_p["command"] == "TOY_SLOW=50 toy-bench",
          "build_brief: row-level perturb_cmd governs when present")

    # 4. The default documented-schema loader is reachable through a BARE
    #    ProjectAdapter (no override): loads the synthetic artifact dir.
    d2 = tempfile.mkdtemp(prefix="fulcrum_toy_default_")
    make_artifact(d2, with_knobs=True, v3=True)
    base = ProjectAdapter()
    run2 = load_run(d2, base)
    check(("silesia", 1) in run2["cells"]
          and "hit_drive" in run2["cells"][("silesia", 1)]["knobs"],
          "default ProjectAdapter.load_run delegates to the documented-"
          "schema loader (cells + knob dirs found)")

    return check.finish("adapter-pluggability selftest")
