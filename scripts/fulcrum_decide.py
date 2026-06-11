#!/usr/bin/env python3
"""fulcrum_decide — thin shim; the engine lives in tools/fulcrum
(fulcrum.core.decide + fulcrum.adapters.gzippy).

Byte-compatible CLI of the original monolith:
  python3 scripts/fulcrum_decide.py <artifact-dir> [--allow-thaw]
  python3 scripts/fulcrum_decide.py --selftest
plus the v3 additions: [--feature F] [--ledger PATH] [--no-ledger].

The results ledger defaults to <repo>/artifacts/fulcrum/ledger.jsonl.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, "..", "tools", "fulcrum"))
sys.path.insert(0, _PKG)

# Default the ledger next to the repo (decide.sh runs from anywhere).
os.environ.setdefault(
    "FULCRUM_LEDGER",
    os.path.abspath(os.path.join(_HERE, "..", "artifacts", "fulcrum",
                                 "ledger.jsonl")))

from fulcrum.cli import decide_main as main  # noqa: E402

if __name__ == "__main__":
    main()
