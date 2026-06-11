"""Decision-engine self-tests (ported wholesale from scripts/fulcrum_decide.py
--selftest; every original check retained). Synthetic artifact dirs with KNOWN
structure validate the knob harness, distribution health, guard matrix, prof
parser + bank comparator, effect predicates, ranking and DO-THIS-NEXT."""

import os
import shutil
import tempfile

from ..adapters.gzippy import GzippyAdapter, bank_divergence, parse_prof
from ..core.causal import knob_verdict
from ..core.decide import analyze_run, load_run
from ..core.stats import bimodal, resolution
from ..core.trace import InstrumentError
from . import Checker

AD = GzippyAdapter()

PROF_TXT = (
    "[contig-prof] CONTIG (Block::decode_clean_into_contig):\n"
    "  calls=1768 total_cyc=1000000 classed_cyc=900000 (90.0% of total; "
    "rest=careful+entry/exit+unchained tail)\n"
    "  lit1   : iters=      100000 cyc=        90000   10.0% of classed,"
    "    0.9 cyc/iter\n"
    "  litpack: iters=       50000 cyc=        45000    5.0% of classed,"
    "    0.9 cyc/iter, lits=120000\n"
    "  litchn : iters=      200000 cyc=       206000   22.9% of classed,"
    "    1.0 cyc/iter, lits=500000\n"
    "  backref: iters=       16140 cyc=       563400   62.6% of classed,"
    "   34.9 cyc/iter, bytes=900000 dist_long=3\n"
    "  careful: cyc=50000 (5.0% of total) outer_iters=123\n"
    "  disttbl: builds=1765 reuses=3 (P3.4 dynamic-block dist_table "
    "amortization)\n"
    "[contig-prof] WRAPPER (decode_huffman_body_resumable):\n"
    "  calls=0 total_cyc=0 classed_cyc=0 (0.0%)\n")

GZ = [1.380, 1.382, 1.379, 1.385, 1.381, 1.383, 1.380]
RG = [0.920, 0.922, 0.918, 0.925, 0.921, 0.919, 0.923]

MANIFEST_V3_EXTRA = (
    "protocol=fulcrum-v3\nsink_gz=regular-file\nsink_rg=regular-file\n"
    "sink_gz_derived=regular-file\nsink_rg_derived=regular-file\n"
    "corpus_silesia_sha=028bd002c89c9a90\n"
    "host_cpu_model=testcpu-13700T\nhost_kernel=6.0-test\n"
    "host_id=abc123def456\n")


def write_manifest(d, freeze="acknowledged", extra=""):
    with open(os.path.join(d, "manifest.txt"), "w") as f:
        f.write("runid=st\nbin=/root/bin-test\nbin_sha=deadbeef\n"
                "feature=gzippy-native\nrg_version=rapidgzip 0.16.0\n"
                f"freeze_state={freeze}\nquiet_state=quiet\n"
                "governor=performance\nno_turbo=1\nrunnable_avg=1.0\n"
                "cell_done=silesia:1:mask=0:sha_ok=1\n" + extra)


def write_samples(path, xs):
    with open(path, "w") as f:
        f.write("\n".join(map(str, xs)))


def make_artifact(d, with_knobs=True, v3=False):
    """The canonical synthetic decide artifact dir (cell silesia:T1)."""
    cdir = os.path.join(d, "cell_silesia_T1")
    os.makedirs(cdir, exist_ok=True)
    write_manifest(d, extra=MANIFEST_V3_EXTRA if v3 else "")
    write_samples(os.path.join(cdir, "wall_gz.txt"), GZ)
    write_samples(os.path.join(cdir, "wall_rg.txt"), RG)
    with open(os.path.join(cdir, "prof.txt"), "w") as f:
        f.write(PROF_TXT)
    if not with_knobs:
        return cdir
    kdir = os.path.join(cdir, "knob_hit_drive")
    k2dir = os.path.join(cdir, "knob_dist_amort")
    os.makedirs(kdir, exist_ok=True)
    os.makedirs(k2dir, exist_ok=True)
    # hit_drive: disabling it makes the cell FASTER by 60ms (feature COSTS).
    write_samples(os.path.join(kdir, "base.txt"), GZ)
    write_samples(os.path.join(kdir, "knob.txt"), [x - 0.060 for x in GZ])
    with open(os.path.join(kdir, "meta.txt"), "w") as f:
        f.write("knob=hit_drive\nenv=GZIPPY_NO_HIT_DRIVE=1\npred=none\n"
                "cell=silesia:1\nmask=0\nsha_ok=1\n")
    # dist_amort: null.
    write_samples(os.path.join(k2dir, "base.txt"), GZ)
    write_samples(os.path.join(k2dir, "knob.txt"), GZ)
    with open(os.path.join(k2dir, "meta.txt"), "w") as f:
        f.write("knob=dist_amort\nenv=GZIPPY_DIST_AMORT=0\npred=prof_dist\n"
                "cell=silesia:1\nmask=0\nsha_ok=1\n")
    edir = os.path.join(d, "knob_effects_silesia_T1")
    os.makedirs(edir, exist_ok=True)
    with open(os.path.join(edir, "effect_base_dist_amort.txt"), "w") as f:
        f.write("disttbl: builds=2790 reuses=7 \n")
    with open(os.path.join(edir, "effect_knob_dist_amort.txt"), "w") as f:
        f.write("disttbl: builds=0 reuses=0 \n")
    return cdir


def run():
    check = Checker()
    print("=== fulcrum selftest: decision engine (decide) ===")

    # 1. knob harness: known-null knob => CAUSAL-NULL.
    base = [1.000, 1.002, 1.001, 1.003, 1.002, 1.001, 1.004]
    v = knob_verdict(base, list(base))
    check(v["status"] == "CAUSAL-NULL",
          f"known-null knob -> CAUSAL-NULL (got {v['status']})")
    v2 = knob_verdict(base, [x - 0.050 for x in base])
    check(v2["status"] == "CAUSAL-VERIFIED-COSTS" and v2["delta_ms"] < 0,
          f"-50ms shift -> CAUSAL-VERIFIED-COSTS (got {v2['status']})")
    v3 = knob_verdict(base, [x + 0.050 for x in base])
    check(v3["status"] == "CAUSAL-VERIFIED-PAYS",
          f"+50ms shift -> CAUSAL-VERIFIED-PAYS (got {v3['status']})")
    wide = [1.00, 1.05, 1.02, 1.08, 1.01, 1.06, 1.03]
    v4 = knob_verdict(wide, [x + 0.01 for x in wide])
    check(v4["status"] == "CAUSAL-NULL",
          f"sub-spread +10ms on 8% spread -> CAUSAL-NULL (got {v4['status']})")

    # 2. bimodality: two-mode flagged, unimodal control not.
    bi = [1.00, 1.01, 1.005, 1.30, 1.31, 1.305, 1.302]
    uni = [1.00, 1.01, 1.02, 1.03, 1.015, 1.025, 1.005]
    check(bimodal(bi) is True, "bimodal sample FLAGGED")
    check(bimodal(uni) is False, "unimodal control NOT flagged")
    deg = [1, 1, 1, 1, 1.01]
    check(bimodal(deg) is False,
          "degenerate [1,1,1,1,1.01] NOT flagged (right side has only 1 "
          "sample)")

    # 3. N-needed monotone: smaller delta -> larger N; resolved -> no N.
    r1 = resolution(0.001, 0.010, 0.010, 9)
    r2 = resolution(0.005, 0.010, 0.010, 9)
    r3 = resolution(0.020, 0.010, 0.010, 9)
    check(r1[0] == "UNRESOLVED" and r2[0] == "UNRESOLVED" and r1[1] > r2[1],
          f"N-needed monotone (Δ1ms->{r1[1]}, Δ5ms->{r2[1]})")
    check(r3 == ("RESOLVED", None),
          "supra-spread delta -> RESOLVED, no N-needed")

    # 4. guard matrix (the re-derived routing guard).
    g1 = AD.routing_guard({"window_seeded": 16, "finished_no_flip": 4,
                           "flip_to_clean": 12, "seeded_block": 16},
                          feature="gzippy-native")
    check(g1[0] is True, "guard: production-seeded native ACCEPT")
    g2 = AD.routing_guard({"window_seeded": 16, "seed_replay_hits": 16})
    check(g2[0] is False, "guard: SEED_WINDOWS replay REFUSE")
    g3 = AD.routing_guard({"isal_chunks": 14, "finished_no_flip": 2},
                          feature="gzippy-native")
    check(g3[0] is False, "guard: isal_chunks>0 on native REFUSE (oracle)")
    g4 = AD.routing_guard({"isal_chunks": 14, "finished_no_flip": 2,
                           "window_seeded": 9}, feature="gzippy-isal")
    check(g4[0] is True, "guard: isal_chunks>0 on isal ACCEPT (production)")
    g5 = AD.routing_guard({})
    check(g5[0] is None, "guard: no sidecar INCONCLUSIVE")

    # 5. prof parser on a synthetic dump in the exact binary format.
    p = parse_prof(PROF_TXT)
    check(p["classes"]["backref"]["cyc_iter"] == 34.9
          and p["classes"]["backref"]["share_pct"] == 62.6
          and p["disttbl"] == (1765, 3) and p["wrapper_calls"] == 0,
          "prof parser: classes/cyc-iter/disttbl/wrapper parsed exactly")
    check(bank_divergence(("silesia", 8), p) == [],
          "bank comparator: banked-consistent prof -> no divergence")
    p_moved = parse_prof(PROF_TXT.replace("  62.6% of classed,   34.9",
                                          "  30.0% of classed,   34.9"))
    check(any("DIVERGES-FROM-BANK" in dd
              for dd in bank_divergence(("silesia", 8), p_moved)),
          "bank comparator: >25% share move -> DIVERGES-FROM-BANK")
    p_small = parse_prof(PROF_TXT.replace("  62.6% of classed,   34.9",
                                          "  60.0% of classed,   34.9"))
    check(bank_divergence(("silesia", 8), p_small) == [],
          "bank comparator: 4% share move NOT flagged")

    # 6. effect predicates.
    ok, _ = AD.effect_check("verbose_seeded", "",
                            "seeded_block=0 seeded_wrapper=16 ")
    check(ok is True, "effect: seeded_block kill-switch verified")
    bad, _ = AD.effect_check("verbose_seeded", "",
                             "seeded_block=16 seeded_wrapper=0 ")
    check(bad is False, "effect: ineffective switch CAUGHT")
    okd, _ = AD.effect_check("prof_dist",
                             "disttbl: builds=2790 reuses=7 ",
                             "disttbl: builds=0 reuses=0 ")
    check(okd is True, "effect: dist_amort off => P3.4 counters dead "
                       "(builds=0/reuses=0; the counting sites live only in "
                       "the amortized arm, marker_inflate.rs:2262/2266)")
    badd, _ = AD.effect_check("prof_dist",
                              "disttbl: builds=2790 reuses=7 ",
                              "disttbl: builds=2790 reuses=7 ")
    check(badd is False,
          "effect: dist_amort knob arm still amortized => CAUGHT")
    vacd, _ = AD.effect_check("prof_dist",
                              "disttbl: builds=0 reuses=0 ",
                              "disttbl: builds=0 reuses=0 ")
    check(vacd is None,
          "effect: no dynamic blocks => predicate vacuous (None)")
    none_v, _ = AD.effect_check("none", "", "")
    check(none_v is None,
          "effect: knob without counter => EFFECT-UNVERIFIED (None)")
    rp_ok, rp_note = AD.effect_check(
        "rpmalloc_stats",
        "[rpmalloc final] slab_hits=0 slab_installs=0\n"
        "[rpmalloc final] mapped_peak=48M",
        "[rpmalloc final] slab_hits=14 slab_installs=15\n"
        "[rpmalloc final] mapped_peak=48M")
    check(rp_ok is True and "29" in rp_note,
          "effect: rpmalloc_stats knob-arm slab counters >0, base 0 => "
          "EFFECT-VERIFIED")
    rp_fail_base, _ = AD.effect_check(
        "rpmalloc_stats",
        "[rpmalloc final] slab_hits=3 slab_installs=2\n",
        "[rpmalloc final] slab_hits=14 slab_installs=15\n")
    check(rp_fail_base is False,
          "effect: rpmalloc_stats base arm engaged too => CAUGHT")
    rp_zero_knob, _ = AD.effect_check(
        "rpmalloc_stats",
        "[rpmalloc final] slab_hits=0 slab_installs=0\n",
        "[rpmalloc final] slab_hits=0 slab_installs=0\n")
    check(rp_zero_knob is False,
          "effect: rpmalloc_stats knob arm counters zero => INEFFECTIVE "
          "CAUGHT")
    rp_fail_missing, _ = AD.effect_check("rpmalloc_stats", "base output",
                                         "knob output")
    check(rp_fail_missing is False,
          "effect: rpmalloc_stats no slab counters in knob arm => CAUGHT")
    off_ok, off_note = AD.effect_check(
        "rpmalloc_stats_off",
        "[rpmalloc final] slab_hits=14 slab_installs=15\n",
        "[rpmalloc final] slab_hits=0 slab_installs=0\n")
    check(off_ok is True and "29" in off_note,
          "effect: rpmalloc_stats_off base auto-engaged, knob 0 => "
          "EFFECT-VERIFIED")
    off_vac, _ = AD.effect_check(
        "rpmalloc_stats_off",
        "[rpmalloc final] slab_hits=0 slab_installs=0\n",
        "[rpmalloc final] slab_hits=0 slab_installs=0\n")
    check(off_vac is None,
          "effect: rpmalloc_stats_off base not engaged => predicate vacuous "
          "(None)")
    off_fail, _ = AD.effect_check(
        "rpmalloc_stats_off",
        "[rpmalloc final] slab_hits=14 slab_installs=15\n",
        "[rpmalloc final] slab_hits=3 slab_installs=1\n")
    check(off_fail is False,
          "effect: rpmalloc_stats_off knob arm still engaged => CAUGHT")

    # 7. end-to-end on a synthetic artifact dir: ranked table + DO-THIS-NEXT.
    d = tempfile.mkdtemp(prefix="fulcrum_decide_st_")
    make_artifact(d)
    run_ = load_run(d, AD)
    rep = analyze_run(run_, AD)
    check(any("knob.hit_drive" in r["component"] and r["tier"] == 1
              for r in rep["rows"]),
          "e2e: hit_drive-COSTS row lands in tier 1")
    check(any("knob.dist_amort" in r["component"] and r["tier"] == 4
              and "CAUSAL-NULL" in r["status"] for r in rep["rows"]),
          "e2e: dist_amort null -> tier 4 CAUSAL-NULL")
    check(rep["rows"][0]["component"].startswith("knob.hit_drive"),
          "e2e: ranking puts the causal-COSTS row first")
    check("knob.hit_drive" in rep["do_next"],
          "e2e: DO-THIS-NEXT picks the top CAUSAL-VERIFIED-COSTS row")
    check(any(r["component"] == "engine.backref" for r in rep["rows"]),
          "e2e: engine.backref hypothesis row present from prof")
    check(bool(rep.get("brief")) and "falsifier" in rep["brief"]
          and "CAUSAL-NULL" in rep["brief"]["falsifier"],
          "e2e: decision brief present with a concrete falsifier")
    rep2 = analyze_run(load_run(d, AD), AD)
    check([r["component"] for r in rep["rows"]]
          == [r["component"] for r in rep2["rows"]],
          "e2e: ranked table deterministic")

    # 8. UNFROZEN refusal + --allow-thaw label.
    man_path = os.path.join(d, "manifest.txt")
    txt = open(man_path).read().replace("freeze_state=acknowledged",
                                        "freeze_state=thawed")
    open(man_path, "w").write(txt)
    raised = False
    try:
        analyze_run(load_run(d, AD), AD)
    except InstrumentError:
        raised = True
    check(raised, "UNFROZEN run REFUSED without --allow-thaw")
    rep3 = analyze_run(load_run(d, AD), AD, allow_thaw=True)
    check(any("UNFROZEN" in r["status"] for r in rep3["rows"]),
          "--allow-thaw labels every wall-derived row UNFROZEN")

    # 9. DO-THIS-NEXT without any causal action -> top engine hypothesis.
    shutil.rmtree(os.path.join(d, "cell_silesia_T1", "knob_hit_drive"))
    txt = open(man_path).read().replace("freeze_state=thawed",
                                        "freeze_state=frozen")
    open(man_path, "w").write(txt)
    rep4 = analyze_run(load_run(d, AD), AD)
    check("engine.backref" in rep4["do_next"],
          "no causal action -> DO-THIS-NEXT = top bounded engine HYPOTHESIS "
          "with its perturbation")
    check("falsifier" in rep4["brief"]
          and "flat" in rep4["brief"]["falsifier"],
          "hypothesis brief: falsifier = flat perturbation response")

    # 10. "reverted" knob DO-THIS-NEXT uses "reconcile" phrasing.
    d2 = tempfile.mkdtemp(prefix="fulcrum_decide_st10_")
    cdir2 = os.path.join(d2, "cell_silesia_T1")
    kslab = os.path.join(cdir2, "knob_slab_alloc")
    os.makedirs(kslab)
    write_manifest(d2, freeze="frozen")
    write_samples(os.path.join(cdir2, "wall_gz.txt"), GZ)
    write_samples(os.path.join(cdir2, "wall_rg.txt"), RG)
    write_samples(os.path.join(kslab, "base.txt"), GZ)
    write_samples(os.path.join(kslab, "knob.txt"), [x - 0.060 for x in GZ])
    with open(os.path.join(kslab, "meta.txt"), "w") as f:
        f.write("knob=slab_alloc\nenv=GZIPPY_SLAB_ALLOC=1\n"
                "pred=rpmalloc_stats\ncell=silesia:1\nmask=0\nsha_ok=1\n")
    edir2 = os.path.join(d2, "knob_effects_silesia_T1")
    os.makedirs(edir2)
    with open(os.path.join(edir2, "effect_base_slab_alloc.txt"), "w") as f:
        f.write("[rpmalloc final] slab_hits=0 slab_installs=0\n")
    with open(os.path.join(edir2, "effect_knob_slab_alloc.txt"), "w") as f:
        f.write("[rpmalloc final] slab_hits=22 slab_installs=23\n"
                "[rpmalloc final] mapped_peak=64M\n")
    rep10 = analyze_run(load_run(d2, AD), AD)
    check(any("slab_alloc" in r["component"] and r["tier"] == 1
              for r in rep10["rows"]),
          "e2e-reverted: slab_alloc CAUSAL-VERIFIED-COSTS lands tier 1")
    check("reconcile" in rep10["do_next"],
          "e2e-reverted: DO-THIS-NEXT for 'reverted' knob uses 'reconcile' "
          "phrasing")
    check("fix/condition" not in rep10["do_next"],
          "e2e-reverted: 'fix/condition' phrasing absent for reverted knob")
    check(any("slab engaged" in r.get("status", "") for r in rep10["rows"]
              if "slab_alloc" in r["component"]),
          "e2e-reverted: slab-engagement effect-verified note in slab_alloc "
          "status")

    return check.finish("decision-engine selftest")
