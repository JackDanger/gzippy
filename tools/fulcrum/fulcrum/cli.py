"""fulcrum CLI.

Subcommands (also reachable through the host project's `scripts/fulcrum`
front door and the byte-compatible scripts/fulcrum_total.py /
scripts/fulcrum_decide.py shims):

  analyze <artifact-dir> [--allow-thaw] [--feature F] [--ledger PATH|--no-ledger]
      Render the ranked decision table + DECISION BRIEF from a pulled
      artifact dir (fingerprint-gated, ledger-cross-checked).
  total <trace.json> [<other.json>] [--counters F] [--T N] [--feature F]
      The validated whole-system trace analyzer (one trace or a cross-tool
      delta).
  selftest
      Run every suite (trace engine, decision engine, invariant enforcement);
      writes the SELF-TEST-OR-NO-TRUST stamp on success.
  invariants
      Render the enforced invariant set with scars.
  ledger [path]
      Summarize the results ledger.

Measurement runs themselves (freeze, masks, sinks, sha pins) live in the
project's environment-control policy — for gzippy, scripts/bench/decide.sh.
"""

import os
import sys

from .adapters.gzippy import GzippyAdapter
from .core import report as report_mod
from .core import trace as tr
from .core.decide import analyze_run, load_run
from .core.ledger import Ledger
from .selftests import stamp as stamp_mod


def _default_ledger_path():
    env = os.environ.get("FULCRUM_LEDGER")
    if env:
        return env
    return os.path.join(os.getcwd(), "artifacts", "fulcrum", "ledger.jsonl")


def _trust_banner():
    label = stamp_mod.trust_label()
    if label:
        print(label)


def total_main(argv=None):
    """Byte-compatible CLI of the legacy scripts/fulcrum_total.py."""
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        from .selftests import test_total
        rc, _, _ = test_total.run()
        sys.exit(rc)

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
        print("Run `fulcrum selftest` (or --selftest) to validate the tool.")
        sys.exit(1)

    _trust_banner()
    adapter = GzippyAdapter()
    try:
        bundles = [tr.analyze(files[0], adapter, counter_path=counters,
                              declared_T=declared_T, feature=feature)]
        if len(files) >= 2:
            bundles.append(tr.analyze(files[1], adapter))
    except tr.InstrumentError as e:
        print(f"\n[INSTRUMENT REFUSED] {e}")
        sys.exit(2)

    for b in bundles:
        tr.print_bundle(b)
    if len(bundles) == 2:
        tr.print_delta(bundles[0], bundles[1])


def decide_main(argv=None):
    """Byte-compatible CLI of the legacy scripts/fulcrum_decide.py, plus
    fingerprint + ledger options."""
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        from .selftests import test_decide, test_invariants
        rc1, _, f1 = test_decide.run()
        rc2, _, f2 = test_invariants.run()
        sys.exit(0 if (f1 + f2) == 0 else 1)

    allow_thaw = "--allow-thaw" in argv
    no_ledger = "--no-ledger" in argv
    feature = None
    ledger_path = None
    dirs = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--feature":
            feature = argv[i + 1]; i += 2; continue
        if a == "--ledger":
            ledger_path = argv[i + 1]; i += 2; continue
        if a.startswith("--"):
            i += 1; continue
        dirs.append(a); i += 1
    if not dirs:
        print(__doc__)
        sys.exit(1)

    _trust_banner()
    adapter = GzippyAdapter()
    ledger = None if no_ledger else Ledger(ledger_path
                                           or _default_ledger_path())
    try:
        run = load_run(dirs[0], adapter)
        rep = analyze_run(run, adapter, allow_thaw=allow_thaw,
                          feature=feature, ledger=ledger)
    except tr.InstrumentError as e:
        print(f"\n[INSTRUMENT REFUSED] {e}")
        sys.exit(2)
    report_mod.print_report(rep, tie_bar=adapter.tie_bar)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "help"
    rest = argv[1:]
    if cmd == "analyze":
        decide_main(rest)
    elif cmd == "total":
        total_main(rest)
    elif cmd == "selftest":
        from .selftests import run_all
        sys.exit(run_all())
    elif cmd == "invariants":
        from .core.invariants import render
        print(render())
    elif cmd == "ledger":
        path = rest[0] if rest else _default_ledger_path()
        rows = Ledger(path).rows()
        print(f"ledger: {path} ({len(rows)} rows)")
        for r in rows:
            if r.get("_corrupt"):
                print(f"  [TORN ROW] {r['_corrupt']}")
                continue
            fp = r.get("fingerprint", {})
            print(f"  {r.get('ts', '?'):20s} {r.get('runid', '?'):28s} "
                  f"{r.get('key', '?'):24s} {r.get('value_ms', 0):9.1f}ms "
                  f"n={r.get('n', 0):<3d} sink={fp.get('sink', '?')} "
                  f"freeze={fp.get('freeze', '?')} "
                  f"bin={str(fp.get('bin_sha', '?'))[:12]}")
    else:
        print(__doc__)
        sys.exit(0 if cmd in ("help", "-h", "--help") else 1)


if __name__ == "__main__":
    main()
