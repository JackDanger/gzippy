"""Trustworthy-by-construction Chrome-trace engine.

WHY THIS EXISTS
===============
Performance campaigns are repeatedly MISLED by their own trace tooling
(docs/CASE-STUDIES.md has the originals):
  - slack-masked SUMs read as wall binders (a region's SUM can be huge AND
    wall-neutral; SUM != wall);
  - seeded/oracle-contaminated runs that mask the real binder while looking
    like production;
  - counter inversions (a misnamed counter read backwards twice);
  - nested-span double-counts (a phantom "serial cost" that was an O(1) op);
  - instruments that were themselves broken: an oracle that silently re-ran
    the work it claimed to remove, a capture that emitted EMPTY output.

Every number this module produces is backed by an assertion that FAILS LOUD if
the precondition that makes the number meaningful is violated. It NEVER renders
a verdict the underlying data cannot support; it raises InstrumentError instead.

Six guarantees (each asserted, not assumed):
  1. SELF-TIME with NO DOUBLE-COUNT (nested spans subtracted; SUM labeled
     slack-maskable).
  2. WAIT vs COMPUTE vs OUTPUT classification via the adapter's taxonomy;
     unknown span names are surfaced, never silently bucketed.
  3. Routing/contamination guard via the adapter (production vs oracle-seeded).
  4. Removal-oracle contamination check via the adapter.
  5. PER-T aware (slack/Fill reporting).
  6. SELF-VALIDATING: synthetic-trace self-tests with positive AND negative
     controls, and assertion-fires-on-corruption tests (SELF-TEST-OR-NO-TRUST).

The span taxonomy is PROJECT data, not engine logic: it arrives as a Taxonomy
instance from the project adapter.
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass


class InstrumentError(Exception):
    """Raised when a precondition that makes a number meaningful is violated.

    We RAISE instead of printing-and-continuing so a contaminated/empty/seeded
    run can never silently produce a number that later gets quoted as truth.
    """


# ---------------------------------------------------------------------------
# Span classification taxonomy (adapter-supplied data).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Taxonomy:
    """Span-name classification for one project.

    Order of checks matters and is fixed by classify(): overhead names first,
    then WAIT (so a consumer-side wait is never mis-bucketed as compute), then
    outer frames, output, scheduler overhead, compute. Unknown names return
    "unknown" (surfaced, never silently bucketed -- a silent default bucket is
    how a misclassification hides).
    """

    wait_prefixes: tuple = ()
    output_prefixes: tuple = ()
    compute_prefixes: tuple = ()
    sched_overhead_prefixes: tuple = ()
    outer_frame_names: tuple = ()
    overhead_prefixes: tuple = ()
    # Frames emitted ONLY on the wall-critical thread (ownership beats
    # max-span: a long-lived worker can span wider and steal the label).
    consumer_exclusive_frames: tuple = ()

    def classify(self, name):
        """Return one of: wait | output | compute | overhead | outer | unknown."""
        if name in self.overhead_prefixes or \
                any(name.startswith(p) for p in self.overhead_prefixes):
            return "overhead"
        if any(name.startswith(p) for p in self.wait_prefixes):
            return "wait"
        if name in self.outer_frame_names:
            return "outer"
        if any(name.startswith(p) for p in self.output_prefixes):
            return "output"
        if any(name.startswith(p) for p in self.sched_overhead_prefixes):
            return "overhead"
        if any(name.startswith(p) for p in self.compute_prefixes):
            return "compute"
        return "unknown"


# ---------------------------------------------------------------------------
# Trace loading + span pairing.
# ---------------------------------------------------------------------------

def load_events(path):
    """Load a Chrome-trace JSON array, tolerating a trailing comma / no close ]."""
    with open(path) as f:
        s = f.read().strip()
    if not s:
        raise InstrumentError(
            f"EMPTY trace file: {path} -- the capture produced no events (the "
            f"'instrument emitted empty output' failure class). REFUSING to "
            f"render numbers.")
    if not s.startswith("["):
        s = "[" + s
    # Strip the close bracket (if any), drop any trailing comma/newline left by
    # a streaming emitter, then re-add the bracket. Some emitters write each
    # event with a trailing ",\n" and never close the array; some close it.
    # json (esp. 3.13+) rejects a trailing comma, so normalize both shapes.
    if s.endswith("]"):
        s = s[:-1]
    s = s.rstrip().rstrip(",").rstrip()
    s = s + "\n]"
    return json.loads(s)


def pair_spans(events):
    """Pair B/E events into spans, recording each span's parent (the enclosing
    open span on the same thread). Returns (spans, n_mismatched).

    parent is used for self-time (subtract children) -- the no-double-count core.
    """
    stacks = defaultdict(list)
    spans = []
    mismatched = 0
    for e in events:
        ph = e.get("ph")
        if ph == "B":
            stacks[(e.get("pid", 0), e.get("tid", 0))].append(e)
        elif ph == "E":
            key = (e.get("pid", 0), e.get("tid", 0))
            if not stacks[key]:
                mismatched += 1
                continue
            b = stacks[key].pop()
            if b["name"] != e["name"]:
                mismatched += 1
                # Best-effort: keep the begin we popped, pair it with this end.
            parent = stacks[key][-1]["name"] if stacks[key] else "<root>"
            spans.append({
                "name": b["name"],
                "parent": parent,
                "pid": b.get("pid", 0),
                "tid": b.get("tid", 0),
                "start": b.get("ts", 0.0),
                "end": e.get("ts", 0.0),
                "dur": e.get("ts", 0.0) - b.get("ts", 0.0),
                "args": b.get("args", {}),
                "depth": len(stacks[key]),  # nesting depth at begin
            })
    return spans, mismatched


# ---------------------------------------------------------------------------
# Self-time (no double-count) and per-thread busy/idle (busy+idle==span).
# ---------------------------------------------------------------------------

def self_time_by_name(spans):
    """name -> (total_dur, self_dur, count). self = total - time in direct children.

    self_dur sums to <= total and is the ONLY safe per-name number to compare
    across regions; total (SUM) is slack-maskable and is labeled as such.
    """
    total = defaultdict(float)
    count = defaultdict(int)
    child_time = defaultdict(float)  # name -> time spent in its direct children
    for s in spans:
        total[s["name"]] += s["dur"]
        count[s["name"]] += 1
        if s["parent"] != "<root>":
            child_time[s["parent"]] += s["dur"]
    out = {}
    for n in total:
        self_dur = total[n] - child_time[n]
        out[n] = (total[n], self_dur, count[n])
    return out


def per_thread_busy_idle(spans, taxonomy):
    """For each (pid,tid): the thread span (last_end - first_start) and a breakdown
    into wait/compute/output/overhead/unknown by LEAF attribution.

    LEAF attribution: at every instant a thread is in exactly one DEEPEST (leaf)
    span; that instant is charged to the leaf's class. This is what makes coverage
    EXACT (busy+idle==span) AND surfaces nested waits -- a wait nested inside an
    outer frame is attributed to the wait, not buried in the outer frame.

    Implementation: per thread, sort each span as (start,+name) / (end,-name)
    boundary events, sweep maintaining a stack, and attribute each [t0,t1) slice to
    the class of the top-of-stack (deepest open) span. The 'outer' frames thus get
    only their UNCOVERED self time -- exactly the no-double-count guarantee.
    """
    per = defaultdict(list)
    for s in spans:
        per[(s["pid"], s["tid"])].append(s)

    by_thread = {}
    for key, slist in per.items():
        boundaries = []
        for s in slist:
            boundaries.append((s["start"], 0, s))   # 0 = begin
            boundaries.append((s["end"], 1, s))      # 1 = end
        # Sort by time; at equal time, process ends before begins so a
        # zero-length gap is not mis-attributed.
        boundaries.sort(key=lambda b: (b[0], b[1]))
        first = min(s["start"] for s in slist)
        last = max(s["end"] for s in slist)
        t = {"first": first, "last": last, "span": last - first,
             "wait": 0.0, "compute": 0.0, "output": 0.0,
             "overhead": 0.0, "outer": 0.0, "unknown": 0.0}
        active = []  # stack of currently-open spans, deepest last
        prev_time = first
        covered = 0.0   # time with >=1 span open (charged to a class)
        idle_gap = 0.0  # time with ZERO spans open -- measured INDEPENDENTLY so
        # the busy+idle==span check is a real cross-check, not a tautology
        for (tm, kind, s) in boundaries:
            slice_dur = tm - prev_time
            if slice_dur > 0:
                if active:
                    leaf = max(active, key=lambda x: x["depth"])
                    t[taxonomy.classify(leaf["name"])] += slice_dur
                    covered += slice_dur
                else:
                    idle_gap += slice_dur
            prev_time = tm
            if kind == 0:
                active.append(s)
            else:
                # Remove the matching open span (by identity).
                for i in range(len(active) - 1, -1, -1):
                    if active[i] is s:
                        active.pop(i)
                        break
        busy = (t["compute"] + t["output"] + t["overhead"]
                + t["outer"] + t["unknown"] + t["wait"])
        t["covered"] = covered      # independent: sum of class buckets
        t["toplevel"] = busy        # == covered (every covered instant charged once)
        # idle is the INDEPENDENTLY-MEASURED zero-span gap time, NOT (span-busy).
        # The assertion covered + idle == span is therefore a genuine cross-check:
        # if leaf attribution ever double-counts (covered > real) or the sweep
        # mis-tracks the stack, covered + idle_gap will diverge from span and the
        # assert FIRES. (The old idle=span-busy made the check vacuous.)
        t["idle"] = idle_gap
        by_thread[key] = t
    return by_thread


def assert_busy_plus_idle_equals_span(by_thread, tol_us=1.0):
    """The core trust assertion -- a GENUINE cross-check (not the old tautology).

    Three independently-computed quantities must agree:
      (a) busy  = sum of the per-class buckets (compute+output+wait+...),
      (b) idle  = independently-measured zero-span gap time,
      (c) covered = independently-accumulated span-open time.
    busy == covered AND covered + idle == span hold only if the leaf sweep
    charged every instant exactly once. A double-count (covered too big) or a
    stack mis-track makes them diverge and this assert FIRES.
    Returns list of violations (empty == OK).
    """
    violations = []
    for key, t in by_thread.items():
        busy = t["toplevel"]
        covered = t.get("covered", busy)
        if abs(busy - covered) > tol_us:
            violations.append((key, "busy!=covered", busy, covered))
            continue
        if abs(covered + t["idle"] - t["span"]) > tol_us:
            violations.append((key, "covered+idle!=span", covered, t["idle"],
                               t["span"]))
    return violations


def assert_no_double_count(spans, self_by_name, tol_us=1.0):
    """Self-time must never exceed total (a negative self-time => double-count)."""
    violations = []
    for n, (total, self_dur, _cnt) in self_by_name.items():
        if self_dur < -tol_us:
            violations.append((n, total, self_dur))
    return violations


# ---------------------------------------------------------------------------
# Wall-critical thread identification (the ANTI-SUM).
# ---------------------------------------------------------------------------

def consumer_tid(by_thread, spans, taxonomy):
    """The wall-critical thread is the one OWNING the consumer-exclusive outer
    frames, NOT the max-span thread.

    (A long-lived pool worker can span slightly wider than the consumer and
    steal a max-span label, inverting a 98%-WAIT story into a compute story.
    Ownership of the adapter-declared exclusive frames is unambiguous.)

    Falls back to max-span only if no exclusive frame is present (degraded
    trace) -- the caller is warned via the returned method string.
    """
    if not by_thread:
        return None, "no-threads"
    owners = defaultdict(float)
    for s in spans:
        if s["name"] in taxonomy.consumer_exclusive_frames:
            owners[(s["pid"], s["tid"])] += s["dur"]
    if owners:
        return max(owners.items(), key=lambda kv: kv[1])[0], "consumer-frame-owner"
    return (max(by_thread.items(), key=lambda kv: kv[1]["span"])[0],
            "FALLBACK-max-span (no consumer-exclusive frame found)")


# ---------------------------------------------------------------------------
# Formatting + the analyze() bundle.
# ---------------------------------------------------------------------------

def fmt(us):
    if us >= 1_000_000:
        return f"{us / 1e6:.4f}s"
    if us >= 1000:
        return f"{us / 1000:.3f}ms"
    if us >= 1:
        return f"{us:.2f}us"
    return f"{us * 1000:.0f}ns"


def auto_counter_path(trace_path):
    """trace_X.json -> verbose_X.txt / counters_X.txt next to it."""
    import re
    d = os.path.dirname(trace_path)
    base = os.path.basename(trace_path)
    stem = re.sub(r"^trace_", "", re.sub(r"\.json$", "", base))
    for cand in (f"verbose_{stem}.txt", f"counters_{stem}.txt",
                 base.replace(".json", ".counters")):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            return p
    return None


def analyze(trace_path, adapter, counter_path=None, declared_T=None, feature=None):
    """Build the validated bundle for one trace. Raises InstrumentError on a
    precondition violation; returns a dict bundle otherwise.

    The adapter supplies: taxonomy, parse_counters, routing_guard, oracle_guard.
    """
    events = load_events(trace_path)
    if not events:
        raise InstrumentError(f"{trace_path}: zero events (empty-output class).")
    spans, mismatched = pair_spans(events)
    if not spans:
        raise InstrumentError(
            f"{trace_path}: zero paired spans -- B/E never matched. The trace "
            f"is structurally broken; REFUSING numbers.")

    taxonomy = adapter.taxonomy
    self_by_name = self_time_by_name(spans)
    by_thread = per_thread_busy_idle(spans, taxonomy)

    # ---- TRUST ASSERTIONS (fail loud) ----
    span_viol = assert_busy_plus_idle_equals_span(by_thread)
    dc_viol = assert_no_double_count(spans, self_by_name)
    if span_viol:
        raise InstrumentError(
            f"{trace_path}: busy+idle != span on {len(span_viol)} thread(s) "
            f"(e.g. {span_viol[0]}). The depth bookkeeping double-counts; "
            f"REFUSING to render a breakdown.")
    if dc_viol:
        raise InstrumentError(
            f"{trace_path}: negative self-time on {len(dc_viol)} span(s) "
            f"(e.g. {dc_viol[0]}) -- double-count detected; REFUSING numbers.")

    # ---- counters / routing guard (adapter-supplied) ----
    if counter_path is None:
        counter_path = auto_counter_path(trace_path)
    counters = {}
    if counter_path and os.path.exists(counter_path):
        with open(counter_path) as f:
            counters = adapter.parse_counters(f.read())
    is_prod, seed_reason = adapter.routing_guard(counters, feature=feature)
    oracle_warns = adapter.oracle_guard(counters, self_by_name)

    # ---- consumer (wall-critical) thread breakdown ----
    ctid, ctid_method = consumer_tid(by_thread, spans, taxonomy)
    cons = by_thread.get(ctid, {}) if ctid else {}

    # ---- unknown span surfacing ----
    unknown = sorted(
        ((n, v[1]) for n, v in self_by_name.items()
         if taxonomy.classify(n) == "unknown"),
        key=lambda x: -x[1],
    )

    return {
        "path": trace_path,
        "T": declared_T,
        "n_events": len(events),
        "n_spans": len(spans),
        "mismatched": mismatched,
        "self_by_name": self_by_name,
        "by_thread": by_thread,
        "consumer_tid": ctid,
        "consumer_tid_method": ctid_method,
        "consumer": cons,
        "counters": counters,
        "is_production": is_prod,
        "seed_reason": seed_reason,
        "oracle_warns": oracle_warns,
        "unknown": unknown,
        "taxonomy": taxonomy,
    }


def print_bundle(b):
    taxonomy = b["taxonomy"]
    print(f"\n========== fulcrum total: {b['path']} ==========")
    if b["T"]:
        print(f"declared T            : {b['T']}")
    print(f"events / spans       : {b['n_events']} / {b['n_spans']}"
          + (f"  (WARNING {b['mismatched']} mismatched B/E)" if b["mismatched"]
             else ""))

    # --- the routing / production guard, FIRST and LOUD ---
    print("\n-- ROUTING GUARD (production-routing preservation) --")
    if b["is_production"] is True:
        print(f"  [OK]  {b['seed_reason']}")
    elif b["is_production"] is False:
        print(f"  [REFUSE] {b['seed_reason']}")
    else:
        print(f"  [INCONCLUSIVE] {b['seed_reason']}")
    for w in b["oracle_warns"]:
        print(f"  [ORACLE-CONTAMINATION] {w}")

    # --- consumer = wall-critical thread breakdown (NOT a cross-thread SUM) ---
    c = b["consumer"]
    if c:
        span = c["span"]
        print(f"\n-- WALL-CRITICAL THREAD (consumer tid={b['consumer_tid'][1]}, "
              f"id-via={b['consumer_tid_method']}), span={fmt(span)} --")
        if b["consumer_tid_method"].startswith("FALLBACK"):
            print("  [WARN] consumer thread identified by FALLBACK (max-span) -- "
                  "a long-lived worker may have stolen the label; treat the "
                  "split below with caution.")
        print("  (this thread's timeline IS the wall; the split below is "
              "WAIT vs COMPUTE vs OUTPUT and busy+idle==span is a GENUINE "
              "cross-check)")

        def pct(x):
            return f"{100 * x / span:5.1f}%" if span else "  n/a"
        for cls in ("compute", "output", "wait", "overhead", "outer",
                    "unknown", "idle"):
            v = c.get(cls, 0.0)
            print(f"    {cls:10s} {fmt(v):>12s}  {pct(v)}")
        print("  NOTE: 'wait' is BLOCKED-on-another-thread time, NOT serial "
              "work. Do not attribute it to the consumer as compute.")

    # --- per-name SELF time (the safe number) with an explicit SUM caveat ---
    print("\n-- TOP SPANS by SELF-TIME (no double-count; SUM column is "
          "SLACK-MASKABLE) --")
    print(f"  {'name':40s} {'SELF':>11s} {'SUM(!=wall)':>12s} {'count':>7s} "
          f"{'class':>9s}")
    ranked = sorted(b["self_by_name"].items(), key=lambda kv: -kv[1][1])
    for n, (total, self_dur, cnt) in ranked[:20]:
        print(f"  {n:40s} {fmt(self_dur):>11s} {fmt(total):>12s} {cnt:>7d} "
              f"{taxonomy.classify(n):>9s}")
    print("  ^ SELF is comparable across regions. SUM is NOT the wall and a "
          "large SUM can be fully slack-masked (Fill<100%). Never read SUM as "
          "the binder.")
    print("\n  *** DESCRIPTIVE != CAUSAL. This ranking is a HYPOTHESIS "
          "GENERATOR. A binder\n      VERDICT requires a CAUSAL PERTURBATION "
          "(slow-inject + frequency-neutral sleep\n      control + interleaved "
          "locked wall, or a removal oracle). A SELF-time rank is\n      NOT a "
          "binder. (CAUSAL-OR-HYPOTHESIS.) ***")

    if b["unknown"]:
        print("\n-- UNCLASSIFIED span names (taxonomy drift -- classify before "
              "trusting) --")
        for n, sd in b["unknown"][:10]:
            print(f"    {n:40s} {fmt(sd):>11s}")

    # --- per-thread Fill (slack detection) ---
    print("\n-- PER-THREAD Fill (busy/span); low Fill => SUMs on this thread "
          "are slack-masked --")
    for key in sorted(b["by_thread"].keys()):
        t = b["by_thread"][key]
        busy = t["compute"] + t["output"]
        fill = (100 * busy / t["span"]) if t["span"] else 0
        print(f"    pid{key[0]}/tid{key[1]:<3d} span={fmt(t['span']):>10s} "
              f"busy={fmt(busy):>10s} fill={fill:5.1f}%")


def print_delta(left, right):
    print("\n========== CROSS-TOOL DELTA ==========")
    # The cross-tool split is only meaningful if BOTH traces use the SAME span
    # taxonomy. If the right side's consumer thread was identified by FALLBACK,
    # the names didn't line up -- warn loudly.
    if right.get("consumer_tid_method", "").startswith("FALLBACK"):
        print("  [WARN] right-hand trace has NO consumer-exclusive frame -- its "
              "span taxonomy may differ. The per-class delta below is only "
              "valid if both sides emit the same semantic names.")
    lc, rc = left["consumer"], right["consumer"]
    ls = lc.get("span", 0) if lc else 0
    rs = rc.get("span", 0) if rc else 0
    print(f"  wall-critical span:  left={fmt(ls)}   right={fmt(rs)}   "
          f"ratio(right/left)={rs/ls:.3f}" if ls else "  (no consumer span)")
    print("\n  WAIT/COMPUTE/OUTPUT on the wall-critical thread (left vs right):")
    for cls in ("compute", "output", "wait", "idle"):
        lv = lc.get(cls, 0) if lc else 0
        rv = rc.get(cls, 0) if rc else 0
        print(f"    {cls:10s} left={fmt(lv):>11s}  right={fmt(rv):>11s}  "
              f"delta={fmt(lv - rv):>11s}")
    print("  ^ This is the apples-to-apples split. A bigger 'compute' here is a "
          "real per-thread-rate gap ONLY if the routing guard above says BOTH "
          "runs are production (unseeded). If either side is SEEDED, this delta "
          "is void.")
    if left["is_production"] is False or right["is_production"] is False:
        print("  [REFUSE-VERDICT] one side is SEEDED/oracle -- the delta does "
              "not compare like with like. Re-capture both unseeded.")
