#!/usr/bin/env python3
"""Unit tests for the decode-viz reducer.

The synthetic trace is hand-built so that a NAIVE reducer (one that ranks by
CPU-sum, or that ignores instrumentation coverage, or that calls every stall
rate-bound) produces a DIFFERENT, WRONG answer than the correct one.  Each test
asserts the correct answer AND asserts the naive answer would have been wrong.
"""
import json
import os
import tempfile
import unittest

import reduce as R


def write_trace(events):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(events, f)
    return path


def B(name, tid, ts, args=None, pid=1):
    return {"ph": "B", "name": name, "tid": tid, "ts": ts, "pid": pid,
            "args": args or {}}


def E(name, tid, ts, pid=1):
    return {"ph": "E", "name": name, "tid": tid, "ts": ts, "pid": pid}


class TestSpineIsWall(unittest.TestCase):
    """A fat overlapped worker decode must NOT read as weight; the consumer
    spine (small per-op but tiling the wall) IS the weight."""

    def build(self):
        # Consumer (tid1): two iters tiling [0,1000].  drive wraps them.
        # Worker (tid2): one HUGE decode 0..900 that overlaps but finishes
        #   before the second iter even starts its wait -> mostly SLACK.
        ev = []
        ev += [B("drive", 1, 0)]
        # iter 1: 0..500, contains a wait 100..400 (frontier stall)
        ev += [B("consumer.iter", 1, 0)]
        ev += [B("consumer.drain", 1, 100)]
        ev += [B("wait.future_recv", 1, 100, {"chunk_id": 1})]
        ev += [E("wait.future_recv", 1, 400)]
        ev += [E("consumer.drain", 1, 410)]
        ev += [B("consumer.window_publish_clean", 1, 410)]
        ev += [E("consumer.window_publish_clean", 1, 420)]
        ev += [E("consumer.iter", 1, 500)]
        # iter 2: 500..1000, contains wait 600..650 (short)
        ev += [B("consumer.iter", 1, 500)]
        ev += [B("consumer.drain", 1, 600)]
        ev += [B("wait.future_recv", 1, 600, {"chunk_id": 2})]
        ev += [E("wait.future_recv", 1, 650)]
        ev += [E("consumer.drain", 1, 660)]
        ev += [E("consumer.iter", 1, 1000)]
        ev += [E("drive", 1, 1000)]
        # Worker: giant decode 0..900 (CPU-sum says THIS is the bottleneck)
        ev += [B("worker.decode_chunk", 2, 0, {"start_bit": 0})]
        ev += [E("worker.decode_chunk", 2, 900)]
        return ev

    def test_spine_total_equals_wall_not_decode(self):
        path = write_trace(self.build())
        m = R.reduce_tool(path, "gz", None)
        os.unlink(path)
        spine = m["spine"]["spine_total_us"]
        decode = m["workers"]["total_decode_us"]
        wall = m["wall_us"]
        # CORRECT: spine tiles the wall (1000us)
        self.assertEqual(wall, 1000)
        self.assertEqual(spine, 1000)
        # NAIVE (CPU-sum) would call the 900us decode the heavyweight.
        self.assertEqual(decode, 900)
        # The honest model marks most of that decode as SLACK (overlaps only the
        # 300us+50us = 350us of consumer wait windows).
        self.assertGreater(m["workers"]["slack_decode_us"],
                           m["workers"]["wall_relevant_decode_us"])
        # i.e. a naive "decode is 90% of CPU => decode is the wall" is refuted:
        self.assertLess(m["workers"]["wall_relevant_decode_us"], wall * 0.5)


class TestRateVsPlacement(unittest.TestCase):
    """Two frontier stalls: one while a decode is RUNNING (rate-bound), one
    after all decodes closed (placement-bound).  A naive 'every stall is the
    decode' classifier would call both rate-bound."""

    def build(self):
        ev = [B("drive", 1, 0), B("consumer.iter", 1, 0)]
        # stall A: 100..200 while worker decode 50..300 is OPEN -> rate
        ev += [B("wait.future_recv", 1, 100, {"chunk_id": 1})]
        ev += [E("wait.future_recv", 1, 200)]
        # stall B: 500..560 ; all worker decode closed by 300 -> placement
        ev += [B("wait.future_recv", 1, 500, {"chunk_id": 2})]
        ev += [E("wait.future_recv", 1, 560)]
        ev += [E("consumer.iter", 1, 1000), E("drive", 1, 1000)]
        ev += [B("worker.decode_chunk", 2, 50, {}), E("worker.decode_chunk", 2, 300)]
        return ev

    def test_classification(self):
        path = write_trace(self.build())
        m = R.reduce_tool(path, "gz", None)
        os.unlink(path)
        st = m["stalls"]
        self.assertEqual(st["causal"], "HEURISTIC")  # never claims MEASURED
        self.assertEqual(st["n_rate"], 1)
        self.assertEqual(st["n_placement"], 1)
        cls = {s["chunk_id"]: s["cls"] for s in st["stalls"]}
        self.assertEqual(cls[1], "rate")
        self.assertEqual(cls[2], "placement")
        # NAIVE all-rate would give n_rate==2; assert we did NOT do that.
        self.assertNotEqual(st["n_rate"], 2)


class TestCoverageGap(unittest.TestCase):
    """A coarsely-instrumented tool leaves wall UNKNOWN, not idle.  A naive
    reducer that ignores coverage would claim ~100% coverage and read the gap
    as zero work."""

    def build(self):
        # drive spans the whole [0,1000] wall but only ONE iter covers [0,300].
        # The remaining 700us is uninstrumented -> coverage ~30%.
        ev = [B("drive", 1, 0)]
        ev += [B("consumer.iter", 1, 0), E("consumer.iter", 1, 300)]
        ev += [E("drive", 1, 1000)]
        ev += [B("worker.decode", 2, 0, {"mode": "clean"}), E("worker.decode", 2, 250)]
        return ev

    def test_coverage_below_half(self):
        path = write_trace(self.build())
        m = R.reduce_tool(path, "rg", None)
        os.unlink(path)
        cov = m["phases"]["coverage"]
        self.assertAlmostEqual(cov, 0.30, places=2)
        # NAIVE (max ts - min ts of instrumented spans, ignoring drive) would
        # compute coverage ~= 1.0 because every instrumented span is "busy".
        self.assertLess(cov, 0.5)


class TestBEMismatch(unittest.TestCase):
    """A dropped E must be COUNTED (it otherwise makes a fake giant bar)."""

    def build(self):
        # tid2: a dangling B (worker.block_body, never closed) -> unmatched_b=1.
        # tid3: a stray E with an empty stack -> unmatched_e=1.
        # These are the two failure modes that make a fake giant / phantom bar.
        ev = [B("drive", 1, 0), B("consumer.iter", 1, 0), E("consumer.iter", 1, 100),
              E("drive", 1, 100)]
        ev += [B("worker.block_body", 2, 0)]  # never closed -> unmatched_b
        ev += [E("worker.scan_run", 3, 50)]   # no open B on tid3 -> unmatched_e
        return ev

    def test_counts(self):
        path = write_trace(self.build())
        m = R.reduce_tool(path, "gz", None)
        os.unlink(path)
        mm = m["mismatch"]
        self.assertEqual(mm["unmatched_b"], 1)
        self.assertEqual(mm["unmatched_e"], 1)
        self.assertGreaterEqual(mm["total_mismatch"], 2)
        self.assertIn("worker.block_body", mm["affected_names"])
        self.assertIn("worker.scan_run", mm["affected_names"])

    def test_name_mismatch_counted(self):
        # An E whose name differs from the open B (interleaved/dropped frame)
        # must be counted as a name_mismatch (it produces a mis-attributed span).
        ev = [B("drive", 1, 0)]
        ev += [B("worker.block_body", 2, 0)]      # open
        ev += [E("worker.block_header", 2, 50)]   # pops block_body, name !=
        ev += [E("drive", 1, 100)]
        path = write_trace(ev)
        m = R.reduce_tool(path, "gz", None)
        os.unlink(path)
        mm = m["mismatch"]
        self.assertEqual(mm["name_mismatch"], 1)


class TestWallReconciliation(unittest.TestCase):
    def build(self):
        return [B("drive", 1, 0), B("consumer.iter", 1, 0),
                E("consumer.iter", 1, 1000), E("drive", 1, 1000)]

    def test_red_when_off(self):
        path = write_trace(self.build())
        # viz wall = 1000us; measured says 2000us -> 50% off -> not ok
        m = R.reduce_tool(path, "gz", 2000.0)
        os.unlink(path)
        self.assertFalse(m["reconciliation"]["ok"])
        self.assertGreater(m["reconciliation"]["delta_pct"], 10)

    def test_ok_when_close(self):
        path = write_trace(self.build())
        m = R.reduce_tool(path, "gz", 1050.0)  # within 10%
        os.unlink(path)
        self.assertTrue(m["reconciliation"]["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
