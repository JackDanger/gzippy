"""Self-tests (SELF-TEST-OR-NO-TRUST).

Instruments go silently broken — an oracle that re-ran the work it claimed
to remove, a capture that emitted empty output, a coverage assertion that
was a tautology (docs/CASE-STUDIES.md, "the broken instruments"). Every
guarantee therefore has a synthetic-input test with positive AND negative
controls, including tests that CORRUPT the data and assert the trust
assertion FIRES.

`run_all()` executes every suite and, on success, writes the self-test stamp
(keyed to a hash of the package source) that `fulcrum decide/analyze` check.
"""

from . import stamp as _stamp


def run_all(write_stamp=True):
    from . import test_adapter, test_decide, test_invariants, test_total
    total_fail = 0
    counts = {}
    for mod in (test_total, test_decide, test_invariants, test_adapter):
        rc, n_checks, n_fail = mod.run()
        counts[mod.__name__.rsplit(".", 1)[-1]] = {"checks": n_checks,
                                                   "failures": n_fail}
        total_fail += n_fail
    print(f"\n=== fulcrum selftest: "
          f"{sum(c['checks'] for c in counts.values())} checks, "
          f"{total_fail} failure(s) across {len(counts)} suites ===")
    if total_fail == 0 and write_stamp:
        path = _stamp.write_stamp(counts)
        print(f"=== self-test stamp written: {path} ===")
    return 0 if total_fail == 0 else 1


class Checker:
    """Tiny check harness shared by the suites (prints PASS/FAIL lines)."""

    def __init__(self):
        self.failures = []
        self.n = 0

    def __call__(self, cond, msg):
        self.n += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            self.failures.append(msg)

    def finish(self, name):
        ok = not self.failures
        print(f"\n=== {name} {'PASSED' if ok else 'FAILED'} "
              f"({len(self.failures)} failure(s)) ===")
        return (0 if ok else 1), self.n, len(self.failures)
