#!/usr/bin/env python3
"""holdout_report.py — turn the holdout/board TSVs into the overfit verdict.

Input rows (tab separated, written by holdout.sh):
    corpus_set  file  level  threads  rival  ours  rival_bytes  win

A "win" is ours <= rival at the same level (CLAUDE.md's per-label bar; a tie
passes). The verdict compares the holdout win-rate against the board win-rate
computed by THE SAME code on the same levels/threads/rivals in the same run, so
a gap cannot be a methodology artefact.

The verdict is an ALARM, not a ratchet: it never gates a merge and no threshold
here may be used to select a parameter. -6 points is the alarm line because the
board's own residual (37 failing of 660 post-#227) is ~5.6% — a holdout deficit
smaller than the board's own failure rate is not evidence of fit.
"""
import argparse
import collections
import sys

ALARM_POINTS = 6.0


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            s, f, lvl, thr, rival, ours, rb, win = line.split("\t")
            rows.append(
                dict(
                    set=s,
                    file=f,
                    level=int(lvl),
                    threads=int(thr),
                    rival=rival,
                    ours=int(ours),
                    rival_bytes=int(rb),
                    win=int(win),
                )
            )
    return rows


def rate(rows):
    if not rows:
        return None
    return 100.0 * sum(r["win"] for r in rows) / len(rows)


def by(rows, *keys):
    g = collections.defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def fmt_rate(rows):
    r = rate(rows)
    return "  n/a  " if r is None else f"{r:6.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--board")
    ap.add_argument("--commit", default="?")
    ap.add_argument("--binary-sha", default="?")
    ap.add_argument("--dirty", default="0")
    ap.add_argument("--absent-rivals", default="")
    a = ap.parse_args()

    hold = load(a.holdout)
    board = load(a.board) if a.board else []
    if not hold:
        print("holdout_report: EMPTY holdout TSV — a gate may only cite a dataset "
              "that exists; the run produced nothing.", file=sys.stderr)
        return 2

    out = []
    out.append("=" * 78)
    out.append("HOLDOUT vs BOARD — the overfit alarm")
    out.append(f"commit={a.commit[:12]} dirty={a.dirty} binary_sha={a.binary_sha[:16]}")
    if a.absent_rivals.strip():
        out.append(f"RIVALS ABSENT ON THIS BOX: {a.absent_rivals.strip()} "
                   "(their cells were not measured and are not in any rate below)")
    out.append(f"holdout cells={len(hold)}   board cells={len(board) if board else 0}")
    out.append("A win is ours <= rival at the SAME level; a tie counts as a win.")
    out.append("=" * 78)

    hr, br = rate(hold), rate(board)
    out.append("")
    out.append("OVERALL")
    out.append(f"  holdout win-rate : {hr:6.2f}%  ({sum(r['win'] for r in hold)}/{len(hold)})")
    if board:
        out.append(f"  board   win-rate : {br:6.2f}%  ({sum(r['win'] for r in board)}/{len(board)})")
        delta = hr - br
        out.append(f"  delta (holdout - board): {delta:+.2f} points")
        if delta <= -ALARM_POINTS:
            out.append(f"  VERDICT: OVERFIT ALARM — holdout is {-delta:.1f} points worse "
                       f"(alarm line {ALARM_POINTS:.0f}). The board is measuring fit, not "
                       "compression; the classes below name where.")
        elif delta < 0:
            out.append(f"  VERDICT: no alarm — holdout trails by {-delta:.2f} points, "
                       f"inside the {ALARM_POINTS:.0f}-point line.")
        else:
            out.append("  VERDICT: no alarm — holdout is at or above the board win-rate.")
    else:
        out.append("  board leg absent — this run reports the holdout rate only, and "
                   "cannot state an overfit verdict.")

    for label, keys, width in (
        ("BY RIVAL", ("rival",), 12),
        ("BY THREAD COUNT", ("threads",), 12),
        ("BY LEVEL", ("level",), 12),
        ("BY RIVAL x THREADS", ("rival", "threads"), 18),
    ):
        out.append("")
        out.append(label)
        gh, gb = by(hold, *keys), by(board, *keys)
        for k in sorted(set(gh) | set(gb), key=lambda t: tuple(str(x) for x in t)):
            name = ":".join(str(x) for x in k)
            h_rows, b_rows = gh.get(k, []), gb.get(k, [])
            line = f"  {name:<{width}} holdout {fmt_rate(h_rows)} (n={len(h_rows):4d})"
            if board:
                line += f"   board {fmt_rate(b_rows)} (n={len(b_rows):4d})"
                if h_rows and b_rows:
                    line += f"   delta {rate(h_rows) - rate(b_rows):+6.1f}"
            out.append(line)

    out.append("")
    out.append("HOLDOUT MEMBERS — worst first (a member is the archive TYPE at risk)")
    for k, rows in sorted(by(hold, "file").items(), key=lambda kv: rate(kv[1])):
        losses = [r for r in rows if not r["win"]]
        excess = sum(r["ours"] - r["rival_bytes"] for r in losses)
        worst = max(losses, key=lambda r: r["ours"] / r["rival_bytes"], default=None)
        line = (f"  {k[0]:<14} {fmt_rate(rows)} ({len(rows) - len(losses)}/{len(rows)})"
                f"  excess={excess:9d} B")
        if worst:
            line += (f"  worst={worst['rival']}:L{worst['level']}:T{worst['threads']}"
                     f" {worst['ours'] / worst['rival_bytes']:.4f}")
        out.append(line)

    if board:
        out.append("")
        out.append("BOARD FILES — worst first (the tuned population, same grading)")
        for k, rows in sorted(by(board, "file").items(), key=lambda kv: rate(kv[1])):
            nlost = sum(1 for r in rows if not r["win"])
            out.append(f"  {k[0]:<20} {fmt_rate(rows)} ({len(rows) - nlost}/{len(rows)})")

    out.append("")
    out.append("THE RULE: these files are NEVER tuned on. Reading a number here to "
               "choose a knob converts the only unbiased estimate we have into "
               "another tuning set.")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
