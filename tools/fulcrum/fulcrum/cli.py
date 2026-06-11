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
      Summarize the results ledger (anchors, pending-reconcile rows,
      supersede/invalid resolutions).
  ledger supersede --key K --retire RUNID [--promote RUNID] --reason R [path]
      Retire a banked row as an anchor (and optionally promote the
      pending-reconcile row that contradicted it). Append-only: the old row
      stays in the file, it just stops anchoring.
  ledger invalidate --key K --target RUNID --reason R [path]
      Retire a banked row that was a measurement error (never an anchor
      again; nothing is promoted).

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
        from .selftests import test_adapter, test_decide, test_invariants
        rc1, _, f1 = test_decide.run()
        rc2, _, f2 = test_invariants.run()
        rc3, _, f3 = test_adapter.run()
        sys.exit(0 if (f1 + f2 + f3) == 0 else 1)

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


def ledger_main(rest):
    """`fulcrum ledger [path]` listing + the supersede/invalidate verbs."""
    verb = rest[0] if rest and rest[0] in ("supersede", "invalidate") else None
    args = rest[1:] if verb else rest
    opts = {}
    positional = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--key", "--retire", "--promote", "--target", "--reason"):
            if i + 1 >= len(args):
                print(f"ledger {verb}: {a} needs a value")
                sys.exit(2)
            opts[a.lstrip("-")] = args[i + 1]; i += 2; continue
        if a.startswith("--"):
            print(f"ledger: unknown option {a}")
            sys.exit(2)
        positional.append(a); i += 1
    path = positional[0] if positional else _default_ledger_path()
    led = Ledger(path)

    if verb == "supersede":
        missing = [k for k in ("key", "retire", "reason") if k not in opts]
        if "reason" in opts and not opts["reason"].strip():
            print("error: --reason must be a non-empty justification", file=sys.stderr)
            return 2
        if missing:
            print(f"ledger supersede: missing --{' --'.join(missing)}")
            sys.exit(2)
        led.supersede(opts["key"], opts["retire"], opts["reason"],
                      promote_runid=opts.get("promote"))
        print(f"superseded: key={opts['key']} retired={opts['retire']}"
              + (f" promoted={opts['promote']}" if opts.get("promote") else "")
              + f" (appended to {path})")
        return
    if verb == "invalidate":
        missing = [k for k in ("key", "target", "reason") if k not in opts]
        if "reason" in opts and not opts["reason"].strip():
            print("error: --reason must be a non-empty justification", file=sys.stderr)
            return 2
        if missing:
            print(f"ledger invalidate: missing --{' --'.join(missing)}")
            sys.exit(2)
        led.invalidate(opts["key"], opts["target"], opts["reason"])
        print(f"invalidated: key={opts['key']} target={opts['target']} "
              f"(appended to {path})")
        return

    rows = led.rows()
    anchor_ids = {(r.get("key"), r.get("runid")) for r in led.anchors()}
    breaks = led.verify_chain()
    n_chained = sum(1 for r in rows
                    if not r.get("_corrupt") and r.get("chain"))
    chain_note = (f"chain BROKEN ({len(breaks)} break(s))" if breaks
                  else f"chain intact ({n_chained}/{len(rows)} rows chained; "
                       f"pre-chain rows are convention-only)")
    print(f"ledger: {path} ({len(rows)} rows, {len(anchor_ids)} anchors, "
          f"{chain_note})")
    for b in breaks:
        print(f"  !! TAMPER-EVIDENCE: {b}")
    for r in rows:
        if r.get("_corrupt"):
            print(f"  [TORN ROW] {r['_corrupt']}")
            continue
        kind = r.get("kind", "?")
        if kind == "supersede":
            print(f"  {r.get('ts', '?'):20s} [SUPERSEDE] {r.get('key', '?')} "
                  f"retired={r.get('retire_runid')} "
                  f"promoted={r.get('promote_runid') or '-'} "
                  f"reason={r.get('reason', '?')}")
            continue
        if kind == "invalid":
            print(f"  {r.get('ts', '?'):20s} [INVALID]   {r.get('key', '?')} "
                  f"target={r.get('target_runid')} "
                  f"reason={r.get('reason', '?')}")
            continue
        fp = r.get("fingerprint", {})
        ident = (r.get("key"), r.get("runid"))
        tag = ("ANCHOR " if ident in anchor_ids else
               ("PENDING" if r.get("status") == "pending-reconcile"
                else "RETIRED"))
        print(f"  {r.get('ts', '?'):20s} {tag:7s} {r.get('runid', '?'):28s} "
              f"{r.get('key', '?'):24s} {r.get('value_ms', 0):9.1f}ms "
              f"n={r.get('n', 0):<3d} sink={fp.get('sink', '?')} "
              f"freeze={fp.get('freeze', '?')} "
              f"bin={str(fp.get('bin_sha', '?'))[:12]}")


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
        ledger_main(rest)
    else:
        print(__doc__)
        sys.exit(0 if cmd in ("help", "-h", "--help") else 1)


if __name__ == "__main__":
    main()
