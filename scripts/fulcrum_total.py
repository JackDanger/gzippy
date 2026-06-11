#!/usr/bin/env python3
"""fulcrum_total — thin shim; the engine lives in the fulcrum OSS repo's
decide/ package (fulcrum.core.trace), located via $FULCRUM_HOME
(default: ~/www/fulcrum).

Byte-compatible CLI of the original monolith:
  python3 scripts/fulcrum_total.py <trace.json> [--counters <f>] [--T 8] [--feature F]
  python3 scripts/fulcrum_total.py <gzippy.json> <rapidgzip.json>   # cross-tool delta
  python3 scripts/fulcrum_total.py --selftest

Importers keep working too: `import fulcrum_total as ft` still exposes
InstrumentError, analyze, seeding_guard, parse_counters, classify and friends
(now backed by the package, gzippy-adapter-bound).
"""

import os
import sys

_FULCRUM_HOME = os.environ.get(
    "FULCRUM_HOME", os.path.join(os.path.expanduser("~"), "www", "fulcrum"))
_PKG = os.path.abspath(os.path.join(_FULCRUM_HOME, "decide"))
if not os.path.isdir(os.path.join(_PKG, "fulcrum")):
    sys.exit(f"fulcrum_total: decision engine not found at {_PKG} "
             "(set FULCRUM_HOME; default ~/www/fulcrum)")
sys.path.insert(0, _PKG)

from fulcrum.adapters.gzippy import COUNTER_PATTERNS, GzippyAdapter  # noqa: E402,F401
from fulcrum.cli import total_main as main  # noqa: E402
from fulcrum.core import trace as _trace  # noqa: E402
from fulcrum.core.trace import (  # noqa: E402,F401
    InstrumentError,
    Taxonomy,
    assert_busy_plus_idle_equals_span,
    assert_no_double_count,
    auto_counter_path,
    consumer_tid,
    fmt,
    load_events,
    pair_spans,
    print_bundle,
    print_delta,
    self_time_by_name,
)

_ADAPTER = GzippyAdapter()


def classify(name):
    return _ADAPTER.taxonomy.classify(name)


def parse_counters(text):
    return _ADAPTER.parse_counters(text)


def seeding_guard(counters, feature=None):
    return _ADAPTER.routing_guard(counters, feature=feature)


def oracle_overhead_guard(counters, trace_self):
    return _ADAPTER.oracle_guard(counters, trace_self)


def per_thread_busy_idle(spans, taxonomy=None):
    return _trace.per_thread_busy_idle(spans, taxonomy or _ADAPTER.taxonomy)


def analyze(trace_path, counter_path=None, declared_T=None, feature=None):
    return _trace.analyze(trace_path, _ADAPTER, counter_path=counter_path,
                          declared_T=declared_T, feature=feature)


if __name__ == "__main__":
    main()
