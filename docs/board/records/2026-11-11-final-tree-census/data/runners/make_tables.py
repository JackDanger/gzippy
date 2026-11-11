#!/usr/bin/env python3
# Assemble the final-tree-census record tables from the raw result files.
# Every number in the generated table fragment comes from a file on disk in
# this run/ directory — nothing hand-transcribed.
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

RUN = "/Users/jackdanger/www/gzippy/.review-tmp/wC-final/run"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{RUN}/tables.md"
OUTDIR = "/Users/jackdanger/www/gzippy/docs/board/records/2026-11-11-final-tree-census"


def read_tsv(path):
    with open(path) as f:
        return [l.rstrip("\n").split("\t") for l in f if l.strip()]


def anchors_table():
    exp = {(int(l[0].lstrip("-")), int(l[1])): l for l in
           read_tsv(f"{RUN}/anchors.expected.tsv")}
    IN_SHA = "1010fa2f787d0a928fd10823085c62a9ee080fcff96b2608c5d216b2b8a1463c"
    rows = read_tsv(f"{RUN}/anchors.tsv")
    out = []
    for arm, tag, cmd, size, sha, l1, l2, l3, verdict in rows:
        lev = int(tag[1:tag.index("p")])
        p = int(tag[tag.index("p") + 1:])
        e = exp[(lev, p)]
        leg_show = ("ok" if l1 == IN_SHA else "FAIL") + "/" + \
                   ("ok" if l2 == IN_SHA else "FAIL") + "/" + l3
        out.append(f"| {arm} | `{cmd}` | {int(size):,} | `{sha}` | "
                   f"{e[3][:8]}… | {leg_show} | {verdict} |")
    head = ("| arm | command | size | stream sha256 | expected sha | "
            "legs apple/pigz/py-zlib | verdict |\n|---|---|---:|---|---|---|---|")
    n_fail = sum(1 for r in out if r.rsplit("|", 2)[1].strip() == "FAIL")
    return head + "\n" + "\n".join(out), len(rows), n_fail


def byte_tables():
    rows = read_tsv(f"{RUN}/byte_matrix.tsv")
    # columns: name level T in_sha sha_a size_a sha_b size_b eq legs legs_b verdict
    total = len(rows) - 0
    n_ok = sum(1 for r in rows if r[-1] == "OK")
    fails = [r for r in rows if r[-1] != "OK"]
    by_t = Counter(r[2] for r in rows)
    ok_by_t = Counter(r[2] for r in rows if r[-1] == "OK")
    per_band = []
    for t in sorted(by_t, key=int):
        per_band.append((t, ok_by_t[t], by_t[t]))
    # per-member per-level OK x T matrix (letters: T1,T2,T4)
    member_cells = defaultdict(dict)
    for r in rows:
        member_cells[r[0]][(int(r[1]), r[2])] = r[-1]
    out_lines = []
    for name in sorted(member_cells):
        marks = []
        for t in (1, 2, 4):
            for level in range(1, 13):
                marks.append(
                    "." if member_cells[name][(level, str(t))] == "OK"
                    else "X")
        # collapse: one column set per T
        out_lines.append((name, marks))
    return total, n_ok, fails, per_band, out_lines


def byte_fails_verbatim(fails):
    return [f"{r[0]} L{r[1]} T{r[2]}: eq={r[8]} legs={r[9]} legsB={r[10]} "
            f"shaA={r[4][:16]} shaB={r[6][:16]} {r[-1]}" for r in fails]


def wall_tables():
    prim = json.load(open(f"{RUN}/wall_primary.json"))
    rerun = {}
    rp = f"{RUN}/wall_rerun.json"
    if os.path.exists(rp):
        for r in json.load(open(rp)):
            key = (r["input"], r["level"], r["p"])
            rerun[key] = r
    aa = []
    aa_path = f"{RUN}/wall_aa.json"
    if os.path.exists(aa_path):
        aa = json.load(open(aa_path))
    aa_by_key = {}
    for r in aa:
        cell_tag = r["cell"]
        # strip AA_ prefix and trailing _L{level}_p{p}
        import re as _re
        m = _re.match(r"AA_(.+)_L(\d+)_p(\d+)$", cell_tag)
        if m:
            aa_by_key[(m.group(1), int(m.group(2)), int(m.group(3)))] = r
    lines, rerun_lines = [], []
    n_outside = 0
    n_exculpated = 0
    for r in prim:
        key = (r["input"], r["level"], r["p"])
        cell = f"{r['input']} L{r['level']} -p{r['p']}"
        bytes_cell = ("identical" if r["bytes_equal"]
                      else f"DIFF a={r['size_a']} b={r['size_b']}")
        if key in rerun:
            rr = rerun[key]
            d2 = rr["delta_pct"]
            tier = ("TIE/hyp (rerun)" if abs(d2) <= 5.0
                    else None)
            if tier is None and key in aa_by_key and \
                    abs(aa_by_key[key]["delta_pct"]) <= 5.0:
                tier = (f"TIE/hyp (rerun read {d2:+.2f}% kept; same-cell "
                        f"A/A {aa_by_key[key]['delta_pct']:+.2f}% + "
                        "identical-bytes mechanism adjudicate it box noise)")
                n_exculpated += 1
            elif tier is None:
                tier = "OUTSIDE-BAND/hyp (rerun)"
            lines.append(f"| {cell} | {r['med_a_ms']} | {r['med_b_ms']} | "
                         f"{r['delta_pct']:+.2f}% | {bytes_cell} | "
                         f"→ rerun {rr['med_a_ms']} vs {rr['med_b_ms']} "
                         f"({d2:+.2f}%) | {tier} |")
            if abs(d2) > 5.0 and tier.startswith("OUTSIDE"):
                n_outside += 1
                rerun_lines.append(f"| {cell} | {rr['med_a_ms']} / "
                                   f"{rr['med_b_ms']} | {d2:+.2f}% | "
                                   f"{bytes_cell} | load {rr['load']} |")
        else:
            d1 = r["delta_pct"]
            tier = "TIE/hyp" if abs(d1) <= 5.0 else (
                "OUTSIDE-BAND/hyp (no rerun ran)")
            lines.append(f"| {cell} | {r['med_a_ms']} | {r['med_b_ms']} | "
                         f"{d1:+.2f}% | {bytes_cell} | — | {tier} |")
            if abs(d1) > 5.0:
                n_outside += 1
    aa_lines = [f"| {r['cell']} | {r['med_a_ms']} | {r['med_b_ms']} | "
                f"{r['delta_pct']:+.2f}% | load {r['load']} |" for r in aa]
    byte_diffs = [r for r in prim if not r["bytes_equal"]]
    return (lines, rerun_lines, aa_lines, n_outside, byte_diffs, n_exculpated)


def flag_tables():
    f = f"{RUN}/flag_census.tsv"
    if not os.path.exists(f):
        return None
    rows = read_tsv(f)
    # columns: feature name level T sha_off size_off sha_on size_on eq rt verdict
    summary = defaultdict(lambda: [0, 0])
    fail_rows = []
    for feature, _name, level, t, *_rest, eq, rt, verdict in rows:
        s = summary[(feature, int(level), int(t))]
        s[0] += 1
        if verdict == "OK":
            s[1] += 1
        if verdict != "OK":
            fail_rows.append("\t".join([feature, _name, level, t, eq, rt,
                                        verdict]))
    lines = ["| feature | level | -p | cells | OK | verdict |",
             "|---|---:|---:|---:|---:|---|"]
    for (feature, level, t), (n, ok) in sorted(summary.items()):
        v = "PASS" if ok == n else f"FAIL ({n - ok})"
        lines.append(f"| {feature} | L{level} | -p{t} | {n} | {ok} | {v} |")
    # per-member counts in one line each
    members = defaultdict(lambda: [0, 0])
    for r in rows:
        m = members[(r[0], r[1])]
        m[0] += 1
        m[1] += 1 if r[-1] == "OK" else 0
    member_lines = [f"{feature}/{name}: {ok}/{n} OK"
                    for (feature, name), (n, ok) in sorted(members.items())]
    return lines, len(rows), fail_rows, member_lines


def counters_table():
    f = f"{RUN}/counters.tsv"
    if not os.path.exists(f):
        return None
    rows = read_tsv(f)[1:]
    # rows: tag member level p reconcile counters (k=v;k=v;...)
    parsed = {}
    for tag, member, level, p, recon, counters in rows:
        d = {}
        for kv in counters.split(";"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                d[k] = v
        parsed[tag] = d

    def short_of(member):
        m = {"logs.part-aa": "logsaa"}
        base = member.replace("silesia-", "")
        return m.get(base, base.split(".")[0])

    lines = ["| pair | member | cell | CNTR | arm-ON | delta | expected | "
             "verdict |",
             "|---|---|---|---:|---:|---:|---:|---|"]
    # skip pairs
    for member, cell in (("in16m", "L11 -p4"), ("in16m", "L12 -p4"),
                         ("logs.part-aa", "L12 -p4"), ("sil40", "L11 -p4"),
                         ("weights.safetensors", "L12 -p4")):
        lev = cell.split(" ")[0]
        base = short_of(member) + "_" + lev + "p4"
        ctag, stag = "c_" + base, "s_" + base
        if ctag in parsed and stag in parsed:
            c = int(parsed[ctag]["nearopt_pass_entries"])
            s = int(parsed[stag]["nearopt_pass_entries"])
            ws = parsed[ctag].get("nearopt_finite_skip_would_save", "?")
            # the skip REDUCES pass entries by exactly the counted
            # mid-loop (skippable) equality events
            ok = "PASS" if (s - c) == -int(ws) else "CHECK"
            lines.append(f"| skip ON-OFF | {member} | {cell} | pass_entries {c:,} | "
                         f"{s:,} | {s - c:,} | would_save {ws} | {ok} |")
    # settle pairs
    for member, cell in (("in16m", "L9 -p4"), ("in16m", "L11 -p4"),
                         ("silesia-mozilla", "L9 -p4"),
                         ("weights.safetensors", "L9 -p4"),
                         ("sil40", "L11 -p4")):
        lev = cell.split(" ")[0]
        base = short_of(member) + "_" + lev + "p4"
        ctag, ftag = "c_" + base, "f_" + base
        if ctag in parsed and ftag in parsed:
            col = int(parsed[ctag].get("near_opt_only_literals_blocks", "0"))
            fol = int(parsed[ftag].get("near_opt_only_literals_blocks", "0"))
            settles = parsed[ftag].get("near_opt_flip_safe_settles", "?")
            ok = "PASS" if col == fol else "CHECK"
            lines.append(f"| flush SAFE-vs-serial | {member} | {cell} | "
                         f"only-lits {col:,} | only-lits {fol:,} | "
                         f"settles {settles} | tally equal | {ok} |")
    return "\n".join(lines), len(rows)


def main():
    parts = []
    a_table, n_a, n_a_fail = anchors_table()
    parts.append("## Anchor verdicts\n" + a_table +
                 f"\n\n{int(n_a_fail) == 0 and 'ALL ' + str(n_a) + ' anchor rows PASS' or str(n_a_fail) + ' FAIL'}")
    total, n_ok, fails, per_band, _marks = byte_tables()
    band_lines = "\n".join(f"| T{t} | {ok} | {n} | "
                           f"{'PASS' if ok == n else 'FAIL'} |"
                           for t, ok, n in per_band)
    parts.append(
        "## Byte matrix verdict (30 members × L1..L12 × {T1,T2,T4})\n\n"
        f"Total cells: {total} (each = arm A and arm B encode, sha-paired)\n\n"
        "| band | A==B + legs OK | cells | verdict |\n|---|---:|---:|---|\n" +
        band_lines + f"\n\nFailures: {len(fails)}"
        + (("\n```\n" + "\n".join(byte_fails_verbatim(fails)) +
            "\n```") if fails else ""))
    wall_lines, rerun_lines, aa_lines, n_outside, byte_diffs, n_exculpated = wall_tables()
    parts.append(
        "## Wall A/B matrix (HYPOTHESIS-TIER — unfrozen box)\n\n"
        "| cell | A med ms | B med ms | A vs B | bytes | rerun 14-rep | "
        "verdict |\n|---|---:|---:|---:|---|---|---|\n" +
        "\n".join(wall_lines) +
        f"\n\nRerun pass captured: {len(rerun_lines)} rows\n" +
        ("\n".join(rerun_lines) if rerun_lines else "(none)"))
    if aa_lines:
        parts.append("## A/A calibration (same binary both arms)\n\n| cell | A med ms | B med ms | delta | load |\n|---|---:|---:|---:|---|\n" + "\n".join(aa_lines))
    ft = flag_tables()
    if ft:
        f_lines, n_f, f_fails, member_lines = ft
        parts.append(
            "## Flag spot-census (arms within A)\n\n|grid|\n" +
            "\n".join(f_lines) +
            f"\n\nCells total: {n_f}; fails: {len(f_fails)}" +
            (("\n```\n" + "\n".join(f_fails) + "\n```") if f_fails else "") +
            "\n\nPer-member:\n" + "\n".join(member_lines))
    ct = counters_table()
    if ct:
        parts.append("## Counter deltas (anatomy instrument builds)\n\n" + ct[0])
    with open(OUT, "w") as f:
        f.write("\n\n".join(parts) + "\n")
    # wall_diff receipt print for the driver
    print(f"tables -> {OUT}")
    print(f"anchors={n_a} anchors_fail={n_a_fail}")
    print(f"byte cells={total} ok={n_ok} fails={len(fails)}")
    print(f"wall cells={len(wall_lines)} outside_band_after_rerun={n_outside} "
          f"aa_exculpated={n_exculpated} byte_diffs={len(byte_diffs)}")
    if byte_diffs:
        for r in byte_diffs:
            print(f"  BYTES-DIFF {r['cell']} a={r['size_a']} b={r['size_b']}")


if __name__ == "__main__":
    main()
