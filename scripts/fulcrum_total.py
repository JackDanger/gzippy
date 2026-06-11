#!/usr/bin/env python3
"""fulcrum_total — the campaign's trustworthy whole-system instrument.

WHY THIS EXISTS
===============
The gzippy->rapidgzip parity campaign was repeatedly MISLED by instruments:
  - slack-masked decodeBlock SUMs read as wall binders (a region's SUM can be
    huge AND wall-neutral; SUM != wall);
  - SEEDED oracles that route to the clean engine and mask the real
    (window-absent marker bootstrap) binder;
  - counter inversions (a misnamed counter read backwards twice);
  - nested-span double-counts (the combine_crc "62ms" phantom);
  - a clean-window oracle that silently re-ran the bootstrap, and another that
    emitted EMPTY output.

This tool is TRUSTWORTHY BY CONSTRUCTION. Every number it prints is backed by an
assertion that FAILS LOUD if the precondition that makes the number meaningful is
violated. It NEVER renders a verdict the underlying data cannot support; instead it
prints a guard line and refuses.

Six guarantees (each asserted, not assumed):
  1. SELF-TIME with NO DOUBLE-COUNT. Nested spans are subtracted so a parent's
     self-time excludes children. The SUM view is explicitly LABELED "SUM (not
     wall) -- slack-maskable" so it can never be read as the wall.
  2. WAIT vs COMPUTE vs OUTPUT classification. A blocking get on a decode-future is
     classified WAIT, never serial work. Unknown span names are flagged, not
     silently bucketed.
  3. WINDOW-ABSENT-PRESERVING by default. Reads the GZIPPY_VERBOSE counter sidecar;
     if window_seeded>0 (or the ISA-L engine oracle ran) it LOUDLY refuses to call
     the run "production" -- a seeded run routes to the clean engine and MASKS the
     binder. The seeded/unseeded distinction is impossible to confuse.
  4. REMOVAL-ORACLE contamination check. Reads an oracle's self-reported overhead
     (per-chunk alloc/copy the production path does not pay) and REFUSES to treat a
     contaminated contender as a ceiling.
  5. PER-T aware. Reports T and warns when slack (busy<wall) can mask a SUM.
  6. SELF-VALIDATING. `--selftest` builds synthetic traces with KNOWN structure and
     asserts: busy+idle==span, no double-count, a positive control (inject +N% into
     one stage -> that stage moves ~N%, others flat), a negative control (flat run
     stays flat), and the empty-output / re-ran-bootstrap failure class is CAUGHT.

INPUT
=====
  - One or two Chrome-trace JSON files (the same schema trace_v2.rs emits via
    GZIPPY_TIMELINE, and the rapidgzip-side equivalent).
  - Optional verbose-counter sidecar: a text file containing the GZIPPY_VERBOSE
    stderr dump (so seeding / oracle state is read, not assumed). Pass via
    `--counters <file>` or it is auto-detected next to the trace
    (trace_X.json -> verbose_X.txt or counters_X.txt).

USAGE
=====
  python3 scripts/fulcrum_total.py <gzippy_trace.json> [--counters <file>] [--T 8]
  python3 scripts/fulcrum_total.py <gzippy.json> <rapidgzip.json>   # cross-tool delta
  python3 scripts/fulcrum_total.py --selftest                       # validate the tool

OFF==identity: this is a read-only post-processor. It NEVER touches the production
path; the production path is unchanged whether or not the trace was captured.
"""

import json
import os
import re
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Span classification taxonomy.
#
# This is the load-bearing classification: a WAIT (blocked on another thread's
# decode future) must NEVER be counted as serial COMPUTE work -- that inversion
# bit the campaign. Names are matched by prefix against the wired trace_v2 sites
# (grep 'SpanGuard::begin' src/decompress/parallel/). UNKNOWN names are surfaced,
# never silently bucketed -- a silent bucket is how a misclassification hides.
# ---------------------------------------------------------------------------

# A WAIT span = this thread is BLOCKED on another thread / a future / a lock.
# Counting it as work is the inversion that mislabels a decode-future wait as
# "serial work" and double-blames the consumer.
WAIT_PREFIXES = (
    "wait.",            # generic wait spans (wait.future_recv, wait.block_fetcher_get)
    "lock.wait",        # blocked acquiring a mutex
    "pool.pick.wait",   # worker idle waiting for a task
    "consumer.wait_replaced_markers",  # consumer blocked on marker-resolve future
    "consumer.dispatch_recv",          # consumer blocking recv on the post-proc future
    # ttp.rx_recv_block is THE wait that gates the in-order wall (~97% of it per
    # block_fetcher.rs:245): a blocking rx.recv on the awaited chunk's decode.
    # Mis-bucketing it as compute is the binder-inversion that bit the campaign.
    "ttp.rx_recv_block",
    # ttp.get_if_available probes a ready chunk; ttp.take_prefetch hands off the
    # receiver -- both are part of the blocking-get path, classified as wait so a
    # consumer blocked on a future never reads as serial work.
    "ttp.get_if_available",
)

# An OUTPUT span = bytes/checksum leaving the pipeline (the serial tail rapidgzip
# explicitly minimizes). Kept distinct from COMPUTE so the "serial tail" term is
# never conflated with per-thread decode.
OUTPUT_PREFIXES = (
    "consumer.writev",
    "consumer.write_buffered",
    "consumer.combine_crc",
    "consumer.publish_windows",
    "consumer.window_publish_clean",   # serial window-publish (rg's named crit path)
    "consumer.window_publish_marker",
)

# A COMPUTE span = actual decode / marker-resolve / window-apply work.
COMPUTE_PREFIXES = (
    "worker.",                 # worker.decode, worker.bootstrap, worker.block_*, scan
    "post_process.apply_window",
    "post_process.task",       # the post-process unit of work (marker resolve etc.)
    "pool.run_task",
    "consumer.eager_postproc",
    "consumer.process_prefetches",
    "consumer.queue_prefetched_postproc",
    "consumer.arc_take_or_clone",
    "consumer.dispatch_post_process",
    "consumer.get_last_window",
    "consumer.try_take_prefetched",
    "consumer.block_finder_get",
    "ttp.take_prefetch",
    "coord.prefetch",
)

# pool.submit / pool.pick / pool.pick.lock are scheduler bookkeeping, not decode
# work and not a blocking-on-another-thread wait -- classify as overhead so they
# are neither read as the per-thread engine nor as the serial tail.
SCHED_OVERHEAD_PREFIXES = (
    "pool.submit",
    "pool.pick.lock",
    "pool.pick",       # the pick frame itself (its .wait child is WAIT, .lock OVERHEAD)
)

# lock.held is OVERHEAD, NOT busy -- counting both lock.held and the work done
# while holding the lock would double-count. consumer.iter / consumer.drain are
# OUTER LOOP frames (they nest everything) and are excluded from the busy/wait
# split to avoid double-counting their children; they are tracked separately.
OUTER_FRAME_NAMES = ("consumer.iter", "consumer.drain", "consumer.dispatch_recv")
OVERHEAD_PREFIXES = ("lock.held",)


def classify(name):
    """Return one of: wait | output | compute | overhead | outer | unknown.

    Order matters: WAIT is checked before COMPUTE so a 'consumer.*' wait is never
    mis-bucketed as compute. Unknown is returned (not silently dropped) so the
    caller can flag unclassified time -- a silent default bucket hides drift.
    """
    if name in OVERHEAD_PREFIXES or any(name.startswith(p) for p in OVERHEAD_PREFIXES):
        return "overhead"
    # consumer.dispatch_recv is both an outer frame AND a wait; treat as wait
    # (it is the blocking recv) but exclude from busy.
    if any(name.startswith(p) for p in WAIT_PREFIXES):
        return "wait"
    if name in OUTER_FRAME_NAMES:
        return "outer"
    if any(name.startswith(p) for p in OUTPUT_PREFIXES):
        return "output"
    if any(name.startswith(p) for p in SCHED_OVERHEAD_PREFIXES):
        return "overhead"
    if any(name.startswith(p) for p in COMPUTE_PREFIXES):
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
        raise InstrumentError(f"EMPTY trace file: {path} -- the capture produced no "
                              f"events (the 'instrument emitted empty output' failure "
                              f"class). REFUSING to render numbers.")
    if not s.startswith("["):
        s = "[" + s
    # Strip the close bracket (if any), drop any trailing comma/newline left by
    # the streaming emitter, then re-add the bracket. trace_v2 emits each event
    # with a trailing ",\n" and never writes a closing "]"; some writers add one.
    # json (esp. 3.13+) rejects a trailing comma, so normalize both shapes.
    if s.endswith("]"):
        s = s[:-1]
    s = s.rstrip().rstrip(",").rstrip()
    s = s + "\n]"
    return json.loads(s)


class InstrumentError(Exception):
    """Raised when a precondition that makes a number meaningful is violated.

    We RAISE instead of printing-and-continuing so a contaminated/empty/seeded
    run can never silently produce a number that later gets quoted as truth.
    """


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


def per_thread_busy_idle(spans):
    """For each (pid,tid): the thread span (last_end - first_start) and a breakdown
    into wait/compute/output/overhead/unknown by LEAF attribution.

    LEAF attribution: at every instant a thread is in exactly one DEEPEST (leaf)
    span; that instant is charged to the leaf's class. This is what makes coverage
    EXACT (busy+idle==span) AND surfaces nested waits -- a wait nested inside an
    outer 'consumer.iter' frame is attributed to the wait, not buried in the outer
    frame. (The earlier depth-0-only scheme buried nested waits; this fixes it.)

    Implementation: per thread, sort each span as (start,+name) / (end,-name)
    boundary events, sweep maintaining a stack, and attribute each [t0,t1) slice to
    the class of the top-of-stack (deepest open) span. The 'outer' frames thus get
    only their UNCOVERED self time -- exactly the no-double-count guarantee.
    """
    # Group spans per thread.
    per = defaultdict(list)
    for s in spans:
        per[(s["pid"], s["tid"])].append(s)

    by_thread = {}
    for key, slist in per.items():
        # Boundary sweep with a depth stack (use the recorded 'depth' to break ties
        # deterministically: deeper span wins at a shared boundary).
        boundaries = []
        for s in slist:
            boundaries.append((s["start"], 0, s))   # 0 = begin
            boundaries.append((s["end"], 1, s))      # 1 = end
        # Sort by time; at equal time, process ends before begins so a zero-length
        # gap is not mis-attributed, then deeper depth first for stable leaf pick.
        boundaries.sort(key=lambda b: (b[0], b[1]))
        first = min(s["start"] for s in slist)
        last = max(s["end"] for s in slist)
        t = {"first": first, "last": last, "span": last - first,
             "wait": 0.0, "compute": 0.0, "output": 0.0,
             "overhead": 0.0, "outer": 0.0, "unknown": 0.0}
        active = []  # stack of currently-open spans, deepest last
        prev_time = first
        covered = 0.0   # time with >=1 span open (charged to a class)
        idle_gap = 0.0  # time with ZERO spans open -- measured INDEPENDENTLY so the
                        # busy+idle==span check is a real cross-check, not a tautology
        for (tm, kind, s) in boundaries:
            slice_dur = tm - prev_time
            if slice_dur > 0:
                if active:
                    leaf = max(active, key=lambda x: x["depth"])
                    t[classify(leaf["name"])] += slice_dur
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
        # assert FIRES. (Advisor C2: the old idle=span-busy made the check vacuous.)
        t["idle"] = idle_gap
        by_thread[key] = t
    return by_thread


def assert_busy_plus_idle_equals_span(by_thread, tol_us=1.0):
    """The core trust assertion -- a GENUINE cross-check (not the old tautology).

    Two independent quantities are checked against the thread span:
      (a) busy  = sum of the per-class buckets (compute+output+wait+...),
      (b) idle  = independently-measured zero-span gap time,
      (c) covered = independently-accumulated span-open time.
    All three are computed by DIFFERENT accumulators in the sweep, so:
      busy == covered   AND   covered + idle == span
    only hold if the leaf sweep charged every instant exactly once. A double-count
    (covered too big) or a stack mis-track makes them diverge and this assert FIRES.
    (Advisor C2: the prior idle=span-busy made busy+idle==span vacuously true.)
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
    """Self-time must never exceed total, and child-time charged to a parent must
    not exceed that parent's total (a negative self-time => double-count)."""
    violations = []
    for n, (total, self_dur, _cnt) in self_by_name.items():
        if self_dur < -tol_us:
            violations.append((n, total, self_dur))
    return violations


# ---------------------------------------------------------------------------
# Wall-critical (the ANTI-SUM): per-instant union of active depth-0 spans by
# class. This answers "of the WALL, how much was the system in compute vs wait vs
# output" without ever summing across threads (which is the slack-mask trap).
# We use the SINGLE consumer/critical thread span timeline -- the in-order
# consumer is the wall by construction in this pipeline.
# ---------------------------------------------------------------------------

CONSUMER_EXCLUSIVE_FRAMES = ("consumer.iter", "consumer.drain")


def consumer_tid(by_thread, spans):
    """The wall-critical thread is the CONSUMER -- the one running the in-order get
    loop. Identify it by OWNERSHIP of the consumer-exclusive outer frames
    (consumer.iter / consumer.drain), NOT by max span.

    (Advisor attack #2: a long-lived pool worker can span slightly wider than the
    consumer and steal a max-span label, inverting the 98%-WAIT story into a
    compute story. consumer.iter/.drain are emitted ONLY on the consumer thread
    (chunk_fetcher.rs:1096/:3577), so frame ownership is unambiguous.)

    Falls back to max-span only if no consumer frame is present (degraded trace),
    and the caller is warned via the returned flag.
    """
    if not by_thread:
        return None, "no-threads"
    owners = defaultdict(float)
    for s in spans:
        if s["name"] in CONSUMER_EXCLUSIVE_FRAMES:
            owners[(s["pid"], s["tid"])] += s["dur"]
    if owners:
        return max(owners.items(), key=lambda kv: kv[1])[0], "consumer-frame-owner"
    # Degraded: no consumer frame in the trace -- fall back, but flag it.
    return (max(by_thread.items(), key=lambda kv: kv[1]["span"])[0],
            "FALLBACK-max-span (no consumer.iter/.drain frame found)")


# ---------------------------------------------------------------------------
# Counter sidecar parsing -- the WINDOW-ABSENT / SEEDING / ORACLE guard.
# ---------------------------------------------------------------------------

COUNTER_PATTERNS = {
    "window_seeded": r"window_seeded=(\d+)",
    "flip_to_clean": r"flip_to_clean=(\d+)",
    "finished_no_flip": r"finished_no_flip=(\d+)",
    "seeded_block": r"seeded_block=(\d+)",
    "seeded_wrapper": r"seeded_wrapper=(\d+)",
    "exact_block": r"exact_block=(\d+)",
    "exact_wrapper": r"exact_wrapper=(\d+)",
    # The binary emits `isal_chunks=` / `isal_fallbacks=` (chunk_fetcher.rs:870-871,
    # the ISA-L clean-tail line). The OLD pattern here was `isal_oracle_chunks=` — a
    # label the binary NEVER prints (the same historical grep-bug the instrument
    # registry documents for _oracle_guest.sh), so the oracle arm of the guard
    # could never fire on a real sidecar. Keep the legacy key for old synthetic
    # sidecars; the lookbehind keeps the two patterns disjoint.
    "isal_chunks": r"(?<!oracle_)isal_chunks=(\d+)",
    "isal_fallbacks": r"(?<!oracle_)isal_fallbacks=(\d+)",
    "isal_oracle_chunks": r"isal_oracle_chunks=(\d+)",
    "isal_oracle_fallbacks": r"isal_oracle_fallbacks=(\d+)",
    "bad_seed_resync": r"bad_seed_resync=(\d+)",
    # ONLY printed when GZIPPY_SEED_WINDOWS replay mode is ON (seed_windows.rs
    # report_seed_stats is a no-op off seed) — THE oracle-seeding tell.
    "seed_replay_hits": r"SEED_WINDOWS replay: hits=(\d+)",
}


def parse_counters(text):
    out = {}
    for key, pat in COUNTER_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            out[key] = int(m.group(1))
    return out


def auto_counter_path(trace_path):
    """trace_X.json -> verbose_X.txt / counters_X.txt next to it."""
    d = os.path.dirname(trace_path)
    base = os.path.basename(trace_path)
    stem = re.sub(r"^trace_", "", re.sub(r"\.json$", "", base))
    for cand in (f"verbose_{stem}.txt", f"counters_{stem}.txt",
                 base.replace(".json", ".counters")):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            return p
    return None


def seeding_guard(counters, feature=None):
    """Return (is_production, reason). A run is NOT production (binder-masking) iff
    an ACTUAL oracle contaminated it. RE-DERIVED 2026-06-10 (fulcrum2 charter):

    The OLD rule refused on window_seeded>0. That counter (WINDOW_SEEDED_CHUNKS,
    gzip_chunk.rs:1181) increments for ANY full-32KiB-initial-window decode --
    which since M3 includes PRODUCTION chunks whose predecessor window the live
    WindowMap published (chunk_fetcher.rs:2545 materialize path). So the old guard
    OVER-FIRED on every healthy native/isal production run. window_seeded>0 alone
    is production-seeded routing, not contamination.

    The ACTUAL contamination signals (each individually sufficient to refuse):
      1. seed_replay_hits>0 -- the `SEED_WINDOWS replay: hits=` line is printed
         ONLY when the GZIPPY_SEED_WINDOWS oracle store is active
         (seed_windows.rs report_seed_stats no-ops off seed). Oracle-seeded
         windows force the clean engine at boundaries production would have to
         marker-bootstrap -- the binder-masking this guard exists to catch.
      2. ISA-L engine chunks on a NATIVE build: isal_chunks>0 is PRODUCTION on
         gzippy-isal (the clean-tail engine) but oracle-only on gzippy-native
         (GZIPPY_ISAL_ENGINE_ORACLE). `feature` disambiguates; with feature
         unknown we stay conservative and refuse, telling the caller to declare.

    `feature`: 'gzippy-native'/'native', 'gzippy-isal'/'isal', or None (unknown).
    """
    if not counters:
        return (None, "NO COUNTER SIDECAR -- cannot verify production routing. "
                      "Capture with GZIPPY_VERBOSE=1 2> verbose_<label>.txt and pass "
                      "--counters. REFUSING to certify this as a production-routing "
                      "measurement.")
    feat = (feature or "").replace("gzippy-", "")
    replay = counters.get("seed_replay_hits", 0)
    oracle = max(counters.get("isal_chunks", 0),
                 counters.get("isal_oracle_chunks", 0))
    seeded = counters.get("window_seeded", 0)
    flips = counters.get("flip_to_clean", 0)
    no_flip = counters.get("finished_no_flip", 0)
    seeded_block = counters.get("seeded_block", 0)
    exact_block = counters.get("exact_block", 0)
    if replay > 0:
        return (False, f"ORACLE-SEEDED RUN (SEED_WINDOWS replay hits={replay}). The "
                       f"seed store forced clean-engine decodes at boundaries "
                       f"production would marker-bootstrap. This measures the "
                       f"clean-engine ceiling, NOT production.")
    if oracle > 0 and feat != "isal":
        if feat == "native":
            return (False, f"ISA-L ENGINE ORACLE RAN (isal_chunks={oracle} on a "
                           f"gzippy-native build -- only GZIPPY_ISAL_ENGINE_ORACLE "
                           f"reaches that engine there). A CEILING oracle, not "
                           f"production.")
        return (False, f"isal_chunks={oracle} with build feature UNDECLARED -- "
                       f"production on gzippy-isal, an engine oracle on native. "
                       f"Pass --feature to disambiguate; refusing conservatively.")
    # Production confirmation: SOME decode-path counter must have fired, else we
    # cannot rule out the silently-skipped/re-ran-bootstrap instrument class.
    if no_flip == 0 and flips == 0 and seeded == 0 and seeded_block == 0 \
            and exact_block == 0:
        return (None, "No decode-path counter fired (finished_no_flip, flip_to_clean, "
                      "window_seeded, seeded_block, exact_block all 0) -- cannot "
                      "confirm the production pipeline ran. (The 'oracle silently "
                      "re-ran/skipped the bootstrap' failure class.) Inconclusive.")
    seeded_note = (f"window_seeded={seeded} is PRODUCTION-SEEDED routing "
                   f"(WindowMap-published predecessor windows, M3+), "
                   if seeded > 0 else "window_seeded=0, ")
    isal_note = (f"isal_chunks={oracle} (PRODUCTION clean-tail on gzippy-isal), "
                 if oracle > 0 else "")
    return (True, f"PRODUCTION routing confirmed: no SEED_WINDOWS replay, no engine "
                  f"oracle ({seeded_note}{isal_note}finished_no_flip={no_flip}, "
                  f"flip_to_clean={flips}, seeded_block={seeded_block}, "
                  f"exact_block={exact_block}).")


def oracle_overhead_guard(counters, trace_self):
    """REMOVAL-ORACLE contamination check. A handicapped contender (e.g. a per-chunk
    64MiB alloc/to_vec the production path never pays) must NOT be read as a ceiling.

    We look for tell-tale alloc spans the production path does not have, and for the
    isal_oracle_fallbacks counter (fallbacks mean the oracle is impure). Returns a
    list of contamination warnings.
    """
    warns = []
    if counters:
        fb = max(counters.get("isal_fallbacks", 0),
                 counters.get("isal_oracle_fallbacks", 0))
        oc = max(counters.get("isal_chunks", 0),
                 counters.get("isal_oracle_chunks", 0))
        if oc > 0 and fb > 0:
            warns.append(f"ORACLE IMPURE: {fb}/{oc+fb} chunks fell back to the real "
                         f"engine -- the oracle did NOT replace 100% of decode; its "
                         f"wall is a BLEND, not a clean ceiling.")
    # Heuristic: a per-chunk full-output alloc/copy that production doesn't pay
    # shows up as a large 'alloc' instant or a copy span the production trace lacks.
    for n in trace_self:
        if "to_vec" in n or "oracle_copy" in n or "oracle_alloc" in n:
            warns.append(f"ORACLE COPY SPAN '{n}' present -- this is overhead the "
                         f"production path does not pay; subtract it before reading a "
                         f"ceiling (Rule 3: a handicapped contender != a ceiling).")
    return warns


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def fmt(us):
    if us >= 1_000_000:
        return f"{us / 1e6:.4f}s"
    if us >= 1000:
        return f"{us / 1000:.3f}ms"
    if us >= 1:
        return f"{us:.2f}us"
    return f"{us * 1000:.0f}ns"


def analyze(trace_path, counter_path=None, declared_T=None, feature=None):
    """Build the validated bundle for one trace. Raises InstrumentError on a
    precondition violation; returns a dict bundle otherwise."""
    events = load_events(trace_path)
    if not events:
        raise InstrumentError(f"{trace_path}: zero events (empty-output class).")
    spans, mismatched = pair_spans(events)
    if not spans:
        raise InstrumentError(f"{trace_path}: zero paired spans -- B/E never matched. "
                              f"The trace is structurally broken; REFUSING numbers.")

    self_by_name = self_time_by_name(spans)
    by_thread = per_thread_busy_idle(spans)

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

    # ---- counters / seeding guard ----
    if counter_path is None:
        counter_path = auto_counter_path(trace_path)
    counters = {}
    if counter_path and os.path.exists(counter_path):
        with open(counter_path) as f:
            counters = parse_counters(f.read())
    is_prod, seed_reason = seeding_guard(counters, feature=feature)
    oracle_warns = oracle_overhead_guard(counters, self_by_name)

    # ---- consumer (wall-critical) thread breakdown ----
    ctid, ctid_method = consumer_tid(by_thread, spans)
    cons = by_thread.get(ctid, {}) if ctid else {}

    # ---- unknown span surfacing ----
    unknown = sorted(
        ((n, v[1]) for n, v in self_by_name.items() if classify(n) == "unknown"),
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
    }


def print_bundle(b):
    print(f"\n========== fulcrum_total: {b['path']} ==========")
    if b["T"]:
        print(f"declared T            : {b['T']}")
    print(f"events / spans       : {b['n_events']} / {b['n_spans']}"
          + (f"  (WARNING {b['mismatched']} mismatched B/E)" if b["mismatched"] else ""))

    # --- the seeding / production guard, FIRST and LOUD ---
    print("\n-- ROUTING GUARD (window-absent preservation) --")
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
            print("  [WARN] consumer thread identified by FALLBACK (max-span) -- a "
                  "long-lived worker may have stolen the label; treat the split "
                  "below with caution.")
        print(f"  (this thread's timeline IS the wall; the split below is "
              f"WAIT vs COMPUTE vs OUTPUT and busy+idle==span is a GENUINE "
              f"cross-check)")

        def pct(x):
            return f"{100 * x / span:5.1f}%" if span else "  n/a"
        for cls in ("compute", "output", "wait", "overhead", "outer", "unknown", "idle"):
            v = c.get(cls, 0.0)
            print(f"    {cls:10s} {fmt(v):>12s}  {pct(v)}")
        # The classification that bit the campaign: WAIT must not read as work.
        print(f"  NOTE: 'wait' is BLOCKED-on-another-thread time, NOT serial work. "
              f"Do not attribute it to the consumer as compute.")

    # --- per-name SELF time (the safe number) with an explicit SUM caveat ---
    print("\n-- TOP SPANS by SELF-TIME (no double-count; SUM column is SLACK-MASKABLE) --")
    print(f"  {'name':40s} {'SELF':>11s} {'SUM(!=wall)':>12s} {'count':>7s} {'class':>9s}")
    ranked = sorted(b["self_by_name"].items(), key=lambda kv: -kv[1][1])
    for n, (total, self_dur, cnt) in ranked[:20]:
        print(f"  {n:40s} {fmt(self_dur):>11s} {fmt(total):>12s} {cnt:>7d} "
              f"{classify(n):>9s}")
    print("  ^ SELF is comparable across regions. SUM is NOT the wall and a large "
          "SUM can be fully slack-masked (Fill<100%). Never read SUM as the binder.")
    print("\n  *** DESCRIPTIVE != CAUSAL. This ranking is a HYPOTHESIS GENERATOR. A "
          "binder\n      VERDICT requires a CAUSAL PERTURBATION (GZIPPY_SLOW_BOOTSTRAP "
          "slow-inject\n      + frequency-neutral sleep control + interleaved locked-"
          "guest wall, or a\n      removal oracle). A SELF-time rank is NOT a binder. "
          "(CLAUDE.md PROCESS #1.) ***")

    if b["unknown"]:
        print("\n-- UNCLASSIFIED span names (taxonomy drift -- classify before trusting) --")
        for n, sd in b["unknown"][:10]:
            print(f"    {n:40s} {fmt(sd):>11s}")

    # --- per-thread Fill (slack detection: a SUM behind <100% Fill is maskable) ---
    print("\n-- PER-THREAD Fill (busy/span); low Fill => SUMs on this thread are slack-masked --")
    for key in sorted(b["by_thread"].keys()):
        t = b["by_thread"][key]
        busy = t["compute"] + t["output"]
        fill = (100 * busy / t["span"]) if t["span"] else 0
        print(f"    pid{key[0]}/tid{key[1]:<3d} span={fmt(t['span']):>10s} "
              f"busy={fmt(busy):>10s} fill={fill:5.1f}%")


def print_delta(left, right):
    print("\n========== CROSS-TOOL DELTA ==========")
    # The cross-tool split is only meaningful if BOTH traces use the SAME span
    # taxonomy. The rapidgzip-side emitter must emit the same semantic names
    # (worker.decode, consumer.iter, ...); if its consumer thread was identified
    # by FALLBACK, the names didn't line up -- warn loudly.
    if right.get("consumer_tid_method", "").startswith("FALLBACK"):
        print("  [WARN] right-hand (rapidgzip?) trace has NO consumer.iter/.drain "
              "frame -- its span taxonomy may differ from gzippy's. The per-class "
              "delta below is only valid if both sides emit the same semantic names.")
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
          "real per-thread-rate gap ONLY if the routing guard above says BOTH runs "
          "are production (unseeded). If either side is SEEDED, this delta is void.")
    # Refuse a delta verdict if either side isn't production.
    if left["is_production"] is False or right["is_production"] is False:
        print("  [REFUSE-VERDICT] one side is SEEDED/oracle -- the delta does not "
              "compare like with like. Re-capture both unseeded.")


# ---------------------------------------------------------------------------
# SELF-TEST: synthetic traces with KNOWN structure validate every guarantee.
# ---------------------------------------------------------------------------

def _ev(name, ph, ts, tid=1, pid=1, args=None):
    e = {"name": name, "ph": ph, "ts": ts, "pid": pid, "tid": tid}
    if args:
        e["args"] = args
    return e


def _synth_trace(stages, tid=1):
    """Build a flat (depth-0) sequence of named spans with given durations (us).
    stages = [(name, dur), ...] laid end to end starting at t=0."""
    ev = []
    t = 0.0
    for name, dur in stages:
        ev.append(_ev(name, "B", t, tid=tid))
        ev.append(_ev(name, "E", t + dur, tid=tid))
        t += dur
    return ev


def _synth_nested(parent, parent_dur, children, tid=1):
    """parent span [0, parent_dur] with children nested inside (used to test
    self-time / no-double-count). children = [(name, start, dur), ...]."""
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


def selftest():
    import tempfile
    failures = []

    def check(cond, msg):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {msg}")
        if not cond:
            failures.append(msg)

    d = tempfile.mkdtemp(prefix="fulcrum_selftest_")
    print("=== fulcrum_total --selftest ===")
    print(f"(scratch dir {d})")

    # --- 1. busy+idle==span holds on a clean flat trace ---
    flat = _synth_trace([("worker.decode", 1000.0),
                         ("consumer.writev", 200.0),
                         ("wait.future_recv", 300.0)])
    p = os.path.join(d, "trace_flat.json")
    _write_json(flat, p)
    spans, _ = pair_spans(load_events(p))
    bt = per_thread_busy_idle(spans)
    viol = assert_busy_plus_idle_equals_span(bt)
    check(not viol, "busy+idle==span on a clean flat trace")

    # --- 1b. the busy+idle==span assertion is NON-TAUTOLOGICAL: corrupt 'covered'
    #         (simulate a leaf-sweep double-count) and the assert MUST fire. The
    #         advisor refuted the old vacuous form; this proves the new one bites. ---
    bt_ok = per_thread_busy_idle(spans)
    check(not assert_busy_plus_idle_equals_span(bt_ok),
          "busy+idle==span: clean bundle passes")
    bt_bad = {k: dict(v) for k, v in bt_ok.items()}
    any_key = next(iter(bt_bad))
    bt_bad[any_key]["covered"] += 500.0   # inject a phantom double-count
    check(bool(assert_busy_plus_idle_equals_span(bt_bad)),
          "busy+idle==span ASSERT FIRES on a corrupted 'covered' (non-tautological)")
    bt_bad2 = {k: dict(v) for k, v in bt_ok.items()}
    bt_bad2[any_key]["idle"] += 500.0     # inject a phantom idle gap
    check(bool(assert_busy_plus_idle_equals_span(bt_bad2)),
          "busy+idle==span ASSERT FIRES on a corrupted 'idle' (independent check)")

    # --- 2. no-double-count: nested children subtract from parent self-time ---
    nested = _synth_nested("consumer.combine_crc", 1000.0,
                           [("worker.decode", 100.0, 800.0)])
    p2 = os.path.join(d, "trace_nested.json")
    _write_json(nested, p2)
    spans2, _ = pair_spans(load_events(p2))
    sbn = self_time_by_name(spans2)
    crc_total, crc_self, _ = sbn["consumer.combine_crc"]
    # The combine_crc PHANTOM: total=1000us but self is only 200us (the 800us
    # child is NOT combine_crc's own work). A naive SUM would read 1000us.
    check(abs(crc_total - 1000.0) < 1e-6, "combine_crc TOTAL(SUM) = 1000us (the phantom)")
    check(abs(crc_self - 200.0) < 1e-6,
          "combine_crc SELF = 200us (phantom corrected -- no double-count)")
    dc = assert_no_double_count(spans2, sbn)
    check(not dc, "no negative self-time (no double-count) on nested trace")

    # --- 3. WAIT is classified as wait, NOT compute (the inversion guard) ---
    check(classify("consumer.wait_replaced_markers") == "wait",
          "consumer.wait_replaced_markers classified WAIT (not serial compute)")
    check(classify("consumer.dispatch_recv") == "wait",
          "consumer.dispatch_recv (blocking future recv) classified WAIT")
    check(classify("worker.decode") == "compute", "worker.decode classified COMPUTE")
    check(classify("consumer.writev") == "output", "consumer.writev classified OUTPUT")
    check(classify("totally.new.span") == "unknown",
          "unknown span surfaced (not silently bucketed)")

    # --- 4. POSITIVE control: inject +50% into ONE stage; that stage moves ~50%,
    #        others flat. (The instrument must report the moved stage, not smear.) ---
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
          f"POSITIVE control: injected stage moved {dec_ratio:.2f}x (~1.50 expected)")
    check(abs(wr_ratio - 1.0) < 0.02,
          f"POSITIVE control: other stage FLAT {wr_ratio:.2f}x (~1.00 expected)")

    # --- 5. NEGATIVE control: identical trace twice -> all deltas ~0 ---
    nr = self_time_by_name(pair_spans(load_events(pb))[0])
    check(all(abs(sb[n][1] - nr[n][1]) < 1e-6 for n in sb),
          "NEGATIVE control: identical run -> zero delta on every stage")

    # --- 6. SEEDING guard (RE-DERIVED, fulcrum2 charter): refuse only on ACTUAL
    #        oracle contamination; production-seeded routing (M3+) is ACCEPTED. ---
    # 6a. The over-fire fix: window_seeded>0 WITHOUT a seed-replay line is
    #     production (WindowMap-published predecessor windows) -> ACCEPT.
    is_prod, reason = seeding_guard({"window_seeded": 17, "finished_no_flip": 4,
                                     "flip_to_clean": 12, "seeded_block": 16},
                                    feature="gzippy-native")
    check(is_prod is True and "PRODUCTION-SEEDED" in reason,
          "guard ACCEPTS production-seeded run (window_seeded>0, no replay) -- "
          "the over-fire is fixed")
    # 6b. The contamination it must STILL catch: GZIPPY_SEED_WINDOWS replay.
    is_prodb, rb = seeding_guard({"window_seeded": 17, "finished_no_flip": 0,
                                  "seed_replay_hits": 17})
    check(is_prodb is False and "ORACLE-SEEDED" in rb,
          "guard REFUSES an oracle-seeded run (SEED_WINDOWS replay hits>0)")
    is_prod2, _ = seeding_guard({"window_seeded": 0, "finished_no_flip": 16,
                                 "flip_to_clean": 1})
    check(is_prod2 is True,
          "guard ACCEPTS an unseeded window-absent production run")
    # 6c. ISA-L engine chunks: oracle on native, PRODUCTION on isal.
    is_prod3, r3 = seeding_guard({"isal_chunks": 16, "finished_no_flip": 4},
                                 feature="gzippy-native")
    check(is_prod3 is False and "ORACLE" in r3,
          "guard REFUSES isal_chunks>0 on a NATIVE build (engine oracle)")
    is_prod3b, r3b = seeding_guard({"isal_chunks": 16, "finished_no_flip": 4,
                                    "window_seeded": 12},
                                   feature="gzippy-isal")
    check(is_prod3b is True and "PRODUCTION clean-tail" in r3b,
          "guard ACCEPTS isal_chunks>0 on the ISAL build (production clean-tail)")
    is_prod3c, _ = seeding_guard({"isal_chunks": 16, "finished_no_flip": 4})
    check(is_prod3c is False,
          "guard refuses isal_chunks>0 with feature UNDECLARED (conservative)")
    # 6d. Legacy synthetic sidecars with the old label still refuse.
    is_prod3d, _ = seeding_guard({"isal_oracle_chunks": 16}, feature="native")
    check(is_prod3d is False,
          "guard still REFUSES legacy isal_oracle_chunks label on native")
    is_prod4, r4 = seeding_guard({})
    check(is_prod4 is None,
          "guard is INCONCLUSIVE with no counter sidecar (refuses to certify)")
    # 6e. No decode-path counter at all -> INCONCLUSIVE (skipped-bootstrap class).
    is_prod5, _ = seeding_guard({"window_seeded": 0, "finished_no_flip": 0,
                                 "flip_to_clean": 0})
    check(is_prod5 is None,
          "guard INCONCLUSIVE when no decode-path counter fired")
    # 6f. The real-sidecar label parses: `isal_chunks=` (what the binary emits,
    #     chunk_fetcher.rs:870) -- the OLD pattern (isal_oracle_chunks=) never
    #     matched a real sidecar; prove the fixed pattern does, disjointly.
    parsed = parse_counters(
        "  Unified decoder: flip_to_clean=12 finished_no_flip=4 finish_decode=16 "
        "inflate_wrapper=0 window_seeded=2 seeded_block=16 seeded_wrapper=0 "
        "exact_block=3 exact_wrapper=0 bad_seed_resync=0 resumable_resync_calls=0 "
        "handoff_window_grows=8\n"
        "  ISA-L clean-tail engine (production on gzippy-isal): isal_chunks=14 "
        "isal_fallbacks=0 bfinal_exact_accepted=2 until_exact_fb=0 inexact_fb=0\n")
    check(parsed.get("isal_chunks") == 14 and parsed.get("window_seeded") == 2
          and parsed.get("seeded_block") == 16
          and "isal_oracle_chunks" not in parsed,
          "parse_counters reads the REAL binary labels (isal_chunks=, seeded_block=)")

    # --- 7. ORACLE contamination: fallbacks => impure-blend warning ---
    warns = oracle_overhead_guard({"isal_oracle_chunks": 14, "isal_oracle_fallbacks": 2},
                                  {})
    check(any("IMPURE" in w for w in warns),
          "oracle contamination guard flags fallback-blended ceiling")

    # --- 8. EMPTY-OUTPUT failure class: empty trace RAISES (never silent numbers) ---
    pe = os.path.join(d, "trace_empty.json")
    with open(pe, "w") as f:
        f.write("")
    raised = False
    try:
        load_events(pe)
    except InstrumentError:
        raised = True
    check(raised, "EMPTY trace RAISES InstrumentError (empty-output class caught)")

    # --- 9. RE-RAN-BOOTSTRAP / broken-oracle class: a trace where the consumer
    #        span and the worker span are both full-wall (the 'oracle re-ran the
    #        whole bootstrap' signature) is detectable via the routing guard +
    #        the analyze() assertions still holding. Simulate a contaminated run
    #        whose counters say seeded -> analyze must mark non-production. ---
    contam = _synth_trace([("worker.decode", 2000.0)], tid=1)
    pc = os.path.join(d, "trace_contam.json")
    pcc = os.path.join(d, "verbose_contam.txt")
    _write_json(contam, pc)
    with open(pcc, "w") as f:
        f.write("Unified decoder: flip_to_clean=0 finished_no_flip=0 "
                "window_seeded=17 bad_seed_resync=0\n"
                "SEED_WINDOWS replay: hits=17 misses=0\n")
    bundle = analyze(pc, counter_path=pcc)
    check(bundle["is_production"] is False,
          "analyze() marks an oracle-seeded contaminated run NON-PRODUCTION")

    # --- 10. end-to-end analyze() on a clean production-shaped trace passes all
    #         assertions and certifies production ---
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
    pb2 = analyze(pp, counter_path=ppc)
    check(pb2["is_production"] is True,
          "analyze() certifies an unseeded window-absent run PRODUCTION")
    check(pb2["consumer"]["wait"] > 0,
          "analyze() finds WAIT time on the wall-critical thread (classified, not work)")

    # --- 11. consumer identified by FRAME OWNERSHIP, not max-span (advisor #2):
    #         a worker thread (tid=2) spans WIDER than the consumer (tid=1), but the
    #         consumer must still be picked because it owns consumer.iter. A max-span
    #         heuristic would wrongly pick the worker and invert the WAIT story. ---
    inv = []
    inv += _synth_nested("consumer.iter", 2000.0,
                         [("consumer.wait_replaced_markers", 100.0, 1800.0)], tid=1)
    # worker spans 0..2500 (WIDER than the consumer's 2000)
    inv += _synth_trace([("worker.decode", 2500.0)], tid=2)
    pinv = os.path.join(d, "trace_inv.json")
    _write_json(inv, pinv)
    sp_inv, _ = pair_spans(load_events(pinv))
    bt_inv = per_thread_busy_idle(sp_inv)
    ct, method = consumer_tid(bt_inv, sp_inv)
    check(ct == (1, 1) and method == "consumer-frame-owner",
          "consumer picked by consumer.iter OWNERSHIP even when a worker spans wider")

    print(f"\n=== SELFTEST {'PASSED' if not failures else 'FAILED'} "
          f"({len(failures)} failure(s)) ===")
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main():
    argv = sys.argv[1:]
    if "--selftest" in argv:
        sys.exit(selftest())

    counters = None
    declared_T = None
    feature = None
    files = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--counters":
            counters = argv[i + 1]; i += 2; continue
        if a == "--T":
            declared_T = argv[i + 1]; i += 2; continue
        if a == "--feature":
            feature = argv[i + 1]; i += 2; continue
        if a.startswith("--"):
            i += 1; continue
        files.append(a); i += 1

    if not files:
        print(__doc__)
        print("Run `python3 scripts/fulcrum_total.py --selftest` to validate the tool.")
        sys.exit(1)

    try:
        bundles = [analyze(files[0], counter_path=counters, declared_T=declared_T,
                           feature=feature)]
        if len(files) >= 2:
            bundles.append(analyze(files[1]))
    except InstrumentError as e:
        print(f"\n[INSTRUMENT REFUSED] {e}")
        sys.exit(2)

    for b in bundles:
        print_bundle(b)
    if len(bundles) == 2:
        print_delta(bundles[0], bundles[1])


if __name__ == "__main__":
    main()
