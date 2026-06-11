"""Rendering: the ranked table + the DECISION BRIEF."""


def print_report(rep, tie_bar=0.99):
    print("=" * 100)
    print("fulcrum decide — ONE-RUN decision table (plans/fulcrum-product.md)")
    print("=" * 100)
    for h in rep["header"]:
        print(h)
    print(f"\n-- CELL SCOREBOARD (wall, interleaved, sha-verified; "
          f"bar = {tie_bar}x EVERY T) --")
    for s in rep["scoreboard"]:
        print(s)
    print("\n-- RANKED COMPONENTS (tier 1 causal-COSTS > tier 2 hypotheses > "
          "tier 3 confirms > tier 4 null) --")
    for i, r in enumerate(rep["rows"], 1):
        print(f"\n[{i:2d}] {r['component']}   cells: {r['cells']}")
        print(f"     attribution : {r['attrib']}")
        print(f"     status      : {r['status']}")
        print(f"     distribution: {r['dist']}")
        if "rss" in r:
            print(f"     rss         : {r['rss']}")
        print(f"     re-verify   : {r['verify']}")
    if rep["anomalies"]:
        print("\n-- ANOMALIES (verbatim; investigate before trusting affected "
              "rows) --")
        for a in rep["anomalies"]:
            print(f"  !! {a}")
    print("\n" + "=" * 100)
    print(f"DO THIS NEXT: {rep['do_next']}")
    print("=" * 100)
    b = rep.get("brief")
    if b:
        print("DECISION BRIEF")
        print(f"  action       : {b['action']}")
        print(f"  evidence     : {b['evidence']}")
        print("  preconditions:")
        for p in b["preconditions"]:
            print(f"    - {p}")
        print(f"  command      : {b['command']}")
        print(f"  falsifier    : {b['falsifier']}")
        print("=" * 100)
