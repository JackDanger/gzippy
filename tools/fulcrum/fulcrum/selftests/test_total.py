"""Trace-engine self-tests (ported wholesale from scripts/fulcrum_total.py
--selftest; every original check retained). Synthetic traces with KNOWN
structure validate every guarantee, including assertion-fires-on-corruption
(non-tautology) tests."""

import json
import os
import tempfile

from ..adapters.gzippy import GzippyAdapter
from ..core.trace import (
    InstrumentError,
    analyze,
    assert_busy_plus_idle_equals_span,
    assert_no_double_count,
    consumer_tid,
    load_events,
    pair_spans,
    per_thread_busy_idle,
    self_time_by_name,
)
from . import Checker

AD = GzippyAdapter()
TAX = AD.taxonomy


def _ev(name, ph, ts, tid=1, pid=1, args=None):
    e = {"name": name, "ph": ph, "ts": ts, "pid": pid, "tid": tid}
    if args:
        e["args"] = args
    return e


def _synth_trace(stages, tid=1):
    """Flat (depth-0) sequence of named spans with given durations (us)."""
    ev = []
    t = 0.0
    for name, dur in stages:
        ev.append(_ev(name, "B", t, tid=tid))
        ev.append(_ev(name, "E", t + dur, tid=tid))
        t += dur
    return ev


def _synth_nested(parent, parent_dur, children, tid=1):
    """parent span [0, parent_dur] with children nested inside."""
    ev = [_ev(parent, "B", 0.0, tid=tid)]
    for name, start, dur in children:
        ev.append(_ev(name, "B", start, tid=tid))
        ev.append(_ev(name, "E", start + dur, tid=tid))
    ev.append(_ev(parent, "E", parent_dur, tid=tid))
    return ev


def _write_json(events, path):
    with open(path, "w") as f:
        f.write("[\n")
        for e in events:
            f.write(json.dumps(e) + ",\n")
        f.write("]\n")


def run():
    check = Checker()
    d = tempfile.mkdtemp(prefix="fulcrum_selftest_")
    print("=== fulcrum selftest: trace engine (total) ===")
    print(f"(scratch dir {d})")

    # --- 1. busy+idle==span holds on a clean flat trace ---
    flat = _synth_trace([("worker.decode", 1000.0),
                         ("consumer.writev", 200.0),
                         ("wait.future_recv", 300.0)])
    p = os.path.join(d, "trace_flat.json")
    _write_json(flat, p)
    spans, _ = pair_spans(load_events(p))
    bt = per_thread_busy_idle(spans, TAX)
    viol = assert_busy_plus_idle_equals_span(bt)
    check(not viol, "busy+idle==span on a clean flat trace")

    # --- 1b. the assertion is NON-TAUTOLOGICAL: corrupt 'covered' (simulate a
    #         leaf-sweep double-count) and the assert MUST fire. ---
    bt_ok = per_thread_busy_idle(spans, TAX)
    check(not assert_busy_plus_idle_equals_span(bt_ok),
          "busy+idle==span: clean bundle passes")
    bt_bad = {k: dict(v) for k, v in bt_ok.items()}
    any_key = next(iter(bt_bad))
    bt_bad[any_key]["covered"] += 500.0   # inject a phantom double-count
    check(bool(assert_busy_plus_idle_equals_span(bt_bad)),
          "busy+idle==span ASSERT FIRES on a corrupted 'covered' "
          "(non-tautological)")
    bt_bad2 = {k: dict(v) for k, v in bt_ok.items()}
    bt_bad2[any_key]["idle"] += 500.0     # inject a phantom idle gap
    check(bool(assert_busy_plus_idle_equals_span(bt_bad2)),
          "busy+idle==span ASSERT FIRES on a corrupted 'idle' (independent "
          "check)")

    # --- 2. no-double-count: nested children subtract from parent self-time ---
    nested = _synth_nested("consumer.combine_crc", 1000.0,
                           [("worker.decode", 100.0, 800.0)])
    p2 = os.path.join(d, "trace_nested.json")
    _write_json(nested, p2)
    spans2, _ = pair_spans(load_events(p2))
    sbn = self_time_by_name(spans2)
    crc_total, crc_self, _ = sbn["consumer.combine_crc"]
    # The combine_crc PHANTOM: total=1000us but self is only 200us.
    check(abs(crc_total - 1000.0) < 1e-6,
          "combine_crc TOTAL(SUM) = 1000us (the phantom)")
    check(abs(crc_self - 200.0) < 1e-6,
          "combine_crc SELF = 200us (phantom corrected -- no double-count)")
    dc = assert_no_double_count(spans2, sbn)
    check(not dc, "no negative self-time (no double-count) on nested trace")

    # --- 3. WAIT is classified as wait, NOT compute (the inversion guard) ---
    check(TAX.classify("consumer.wait_replaced_markers") == "wait",
          "consumer.wait_replaced_markers classified WAIT (not serial compute)")
    check(TAX.classify("consumer.dispatch_recv") == "wait",
          "consumer.dispatch_recv (blocking future recv) classified WAIT")
    check(TAX.classify("worker.decode") == "compute",
          "worker.decode classified COMPUTE")
    check(TAX.classify("consumer.writev") == "output",
          "consumer.writev classified OUTPUT")
    check(TAX.classify("totally.new.span") == "unknown",
          "unknown span surfaced (not silently bucketed)")

    # --- 4. POSITIVE control: inject +50% into ONE stage; that stage moves
    #        ~50%, others flat. ---
    base = _synth_trace([("worker.decode", 1000.0),
                         ("consumer.writev", 400.0)])
    slowed = _synth_trace([("worker.decode", 1500.0),   # +50%
                           ("consumer.writev", 400.0)])  # flat
    pb = os.path.join(d, "trace_base.json")
    ps = os.path.join(d, "trace_slow.json")
    _write_json(base, pb)
    _write_json(slowed, ps)
    sb = self_time_by_name(pair_spans(load_events(pb))[0])
    ss = self_time_by_name(pair_spans(load_events(ps))[0])
    dec_ratio = ss["worker.decode"][1] / sb["worker.decode"][1]
    wr_ratio = ss["consumer.writev"][1] / sb["consumer.writev"][1]
    check(abs(dec_ratio - 1.5) < 0.02,
          f"POSITIVE control: injected stage moved {dec_ratio:.2f}x "
          f"(~1.50 expected)")
    check(abs(wr_ratio - 1.0) < 0.02,
          f"POSITIVE control: other stage FLAT {wr_ratio:.2f}x "
          f"(~1.00 expected)")

    # --- 5. NEGATIVE control: identical trace twice -> all deltas ~0 ---
    nr = self_time_by_name(pair_spans(load_events(pb))[0])
    check(all(abs(sb[n][1] - nr[n][1]) < 1e-6 for n in sb),
          "NEGATIVE control: identical run -> zero delta on every stage")

    # --- 6. routing guard (RE-DERIVED): refuse only on ACTUAL oracle
    #        contamination; production-seeded routing (M3+) is ACCEPTED. ---
    is_prod, reason = AD.routing_guard(
        {"window_seeded": 17, "finished_no_flip": 4,
         "flip_to_clean": 12, "seeded_block": 16}, feature="gzippy-native")
    check(is_prod is True and "PRODUCTION-SEEDED" in reason,
          "guard ACCEPTS production-seeded run (window_seeded>0, no replay) "
          "-- the over-fire is fixed")
    is_prodb, rb = AD.routing_guard(
        {"window_seeded": 17, "finished_no_flip": 0, "seed_replay_hits": 17})
    check(is_prodb is False and "ORACLE-SEEDED" in rb,
          "guard REFUSES an oracle-seeded run (SEED_WINDOWS replay hits>0)")
    is_prodb2, rb2 = AD.routing_guard(
        {"finished_no_flip": 4, "bypass_replay_hits": 12})
    check(is_prodb2 is False and "BYPASS_DECODE" in rb2,
          "guard REFUSES BYPASS_DECODE replay run (hits>0, pre-computed "
          "results mask engine)")
    is_prod2, _ = AD.routing_guard(
        {"window_seeded": 0, "finished_no_flip": 16, "flip_to_clean": 1})
    check(is_prod2 is True,
          "guard ACCEPTS an unseeded window-absent production run")
    is_prod3, r3 = AD.routing_guard(
        {"isal_chunks": 16, "finished_no_flip": 4}, feature="gzippy-native")
    check(is_prod3 is False and "ORACLE" in r3,
          "guard REFUSES isal_chunks>0 on a NATIVE build (engine oracle)")
    is_prod3b, r3b = AD.routing_guard(
        {"isal_chunks": 16, "finished_no_flip": 4, "window_seeded": 12},
        feature="gzippy-isal")
    check(is_prod3b is True and "PRODUCTION clean-tail" in r3b,
          "guard ACCEPTS isal_chunks>0 on the ISAL build (production "
          "clean-tail)")
    is_prod3c, _ = AD.routing_guard({"isal_chunks": 16, "finished_no_flip": 4})
    check(is_prod3c is False,
          "guard refuses isal_chunks>0 with feature UNDECLARED (conservative)")
    is_prod3d, _ = AD.routing_guard({"isal_oracle_chunks": 16},
                                    feature="native")
    check(is_prod3d is False,
          "guard still REFUSES legacy isal_oracle_chunks label on native")
    is_prod4, r4 = AD.routing_guard({})
    check(is_prod4 is None,
          "guard is INCONCLUSIVE with no counter sidecar (refuses to certify)")
    is_prod5, _ = AD.routing_guard(
        {"window_seeded": 0, "finished_no_flip": 0, "flip_to_clean": 0})
    check(is_prod5 is None,
          "guard INCONCLUSIVE when no decode-path counter fired")
    # 6f. The real-sidecar label parses: `isal_chunks=` (what the binary
    #     emits) -- the OLD pattern (isal_oracle_chunks=) never matched a real
    #     sidecar; prove the fixed pattern does, disjointly.
    parsed = AD.parse_counters(
        "  Unified decoder: flip_to_clean=12 finished_no_flip=4 "
        "finish_decode=16 inflate_wrapper=0 window_seeded=2 seeded_block=16 "
        "seeded_wrapper=0 exact_block=3 exact_wrapper=0 bad_seed_resync=0 "
        "resumable_resync_calls=0 handoff_window_grows=8\n"
        "  ISA-L clean-tail engine (production on gzippy-isal): isal_chunks=14 "
        "isal_fallbacks=0 bfinal_exact_accepted=2 until_exact_fb=0 "
        "inexact_fb=0\n")
    check(parsed.get("isal_chunks") == 14 and parsed.get("window_seeded") == 2
          and parsed.get("seeded_block") == 16
          and "isal_oracle_chunks" not in parsed,
          "parse_counters reads the REAL binary labels (isal_chunks=, "
          "seeded_block=)")

    # --- 7. ORACLE contamination: fallbacks => impure-blend warning ---
    warns = AD.oracle_guard({"isal_oracle_chunks": 14,
                             "isal_oracle_fallbacks": 2}, {})
    check(any("IMPURE" in w for w in warns),
          "oracle contamination guard flags fallback-blended ceiling")

    # --- 8. EMPTY-OUTPUT failure class: empty trace RAISES ---
    pe = os.path.join(d, "trace_empty.json")
    with open(pe, "w") as f:
        f.write("")
    raised = False
    try:
        load_events(pe)
    except InstrumentError:
        raised = True
    check(raised, "EMPTY trace RAISES InstrumentError (empty-output class "
                  "caught)")

    # --- 9. contaminated run marked non-production by analyze() ---
    contam = _synth_trace([("worker.decode", 2000.0)], tid=1)
    pc = os.path.join(d, "trace_contam.json")
    pcc = os.path.join(d, "verbose_contam.txt")
    _write_json(contam, pc)
    with open(pcc, "w") as f:
        f.write("Unified decoder: flip_to_clean=0 finished_no_flip=0 "
                "window_seeded=17 bad_seed_resync=0\n"
                "SEED_WINDOWS replay: hits=17 misses=0\n")
    bundle = analyze(pc, AD, counter_path=pcc)
    check(bundle["is_production"] is False,
          "analyze() marks an oracle-seeded contaminated run NON-PRODUCTION")
    pcc2 = os.path.join(d, "verbose_contam_bypass.txt")
    with open(pcc2, "w") as f:
        f.write("Unified decoder: flip_to_clean=0 finished_no_flip=4 "
                "window_seeded=0 bad_seed_resync=0\n"
                "BYPASS_DECODE replay: hits=12 misses=0 (misses fall back to "
                "real decode)\n")
    bundle2 = analyze(pc, AD, counter_path=pcc2)
    check(bundle2["is_production"] is False,
          "analyze() marks BYPASS_DECODE replay run NON-PRODUCTION")

    # --- 10. end-to-end analyze() on a clean production-shaped trace ---
    prod_ev = (_synth_nested("consumer.iter", 3000.0,
                             [("consumer.wait_replaced_markers", 100.0, 500.0),
                              ("consumer.writev", 700.0, 300.0)], tid=1)
               + _synth_trace([("worker.decode", 2500.0)], tid=2))
    pp = os.path.join(d, "trace_prod.json")
    ppc = os.path.join(d, "verbose_prod.txt")
    _write_json(prod_ev, pp)
    with open(ppc, "w") as f:
        f.write("Unified decoder: flip_to_clean=1 finished_no_flip=16 "
                "window_seeded=0 bad_seed_resync=0\n")
    pb2 = analyze(pp, AD, counter_path=ppc)
    check(pb2["is_production"] is True,
          "analyze() certifies an unseeded window-absent run PRODUCTION")
    check(pb2["consumer"]["wait"] > 0,
          "analyze() finds WAIT time on the wall-critical thread (classified, "
          "not work)")

    # --- 11. consumer identified by FRAME OWNERSHIP, not max-span ---
    inv = []
    inv += _synth_nested("consumer.iter", 2000.0,
                         [("consumer.wait_replaced_markers", 100.0, 1800.0)],
                         tid=1)
    inv += _synth_trace([("worker.decode", 2500.0)], tid=2)  # spans WIDER
    pinv = os.path.join(d, "trace_inv.json")
    _write_json(inv, pinv)
    sp_inv, _ = pair_spans(load_events(pinv))
    bt_inv = per_thread_busy_idle(sp_inv, TAX)
    ct, method = consumer_tid(bt_inv, sp_inv, TAX)
    check(ct == (1, 1) and method == "consumer-frame-owner",
          "consumer picked by consumer.iter OWNERSHIP even when a worker "
          "spans wider")

    return check.finish("trace-engine selftest")
