#!/usr/bin/env python3
"""fulcrum_decide — thin shim; the engine lives in the fulcrum OSS repo's
decide/ package (fulcrum.core.decide + fulcrum.adapters.gzippy), located
via $FULCRUM_HOME (default: ~/www/fulcrum).

Byte-compatible CLI of the original monolith:
  python3 scripts/fulcrum_decide.py <artifact-dir> [--allow-thaw]
  python3 scripts/fulcrum_decide.py --selftest
plus the v3 additions: [--feature F] [--ledger PATH] [--no-ledger].

The results ledger defaults to <repo>/artifacts/fulcrum/ledger.jsonl.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_FULCRUM_HOME = os.environ.get(
    "FULCRUM_HOME", os.path.join(os.path.expanduser("~"), "www", "fulcrum"))
_PKG = os.path.abspath(os.path.join(_FULCRUM_HOME, "decide"))
if not os.path.isdir(os.path.join(_PKG, "fulcrum")):
    sys.exit(f"fulcrum_decide: decision engine not found at {_PKG} "
             "(set FULCRUM_HOME; default ~/www/fulcrum)")
sys.path.insert(0, _PKG)

# Default the ledger next to the repo (decide.sh runs from anywhere).
os.environ.setdefault(
    "FULCRUM_LEDGER",
    os.path.abspath(os.path.join(_HERE, "..", "artifacts", "fulcrum",
                                 "ledger.jsonl")))

from fulcrum.cli import decide_main as main  # noqa: E402

if __name__ == "__main__":
    main()
