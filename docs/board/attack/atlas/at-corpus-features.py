#!/usr/bin/env python3
"""Byte-level feature panel for the gzippy SHAPE ATLAS (at-corpus-classes.md).

Computes, per corpus member / synthetic fixture:
  - repeat-distance histogram of 3-byte trigram "matches": for every position
    whose trigram occurred before, the distance to the NEAREST previous
    occurrence, bucketed around the libdeflate far-len-3 guards
    (<=64, 65..512, 513..2048, 2049..4096, 4097..8192, 8193..16384,
    16385..32768, >32768) and the first-occurrence ("never") share,
  - matched-position fraction (distance <= 32768, i.e. inside the gzip window),
  - near (<=4096, the greedy fixed guard) and far (>4096) mass split,
  - trigram-distribution entropy H3 (bits/trigram) and distinct-trigram density,
  - byte-value entropy H0, distinct byte count, top-8 byte-value mass share,
  - columnar stride periodicity: byte-equality autocorrelation at 4/8/16/32 B.

Fixtures are generated replicating src/fixtures.rs (XorShift64>>13/7/17,
same seeds, same append order) and verified against the frozen FNV-1a pins of
the fixtures_are_frozen test. Data files are the extracted Silesia members
(/tmp/atlas-silesia/silesia) and benchmark_data; every sample longer than the
cap is reduced to its first 4 MiB prefix (declared protocol).

python3 stdlib only. Emits a markdown table on stdout.
"""
import array
import math
import os
import sys

MASK64 = (1 << 64) - 1
LEN = 1 << 20
CAP = 4 << 20  # uniform prefix cap for corpus members

API = " "  # markdown table separator


class XorShift:
    """u64 back-shift xorshift, identical to src/fixtures.rs:33-43."""

    def __init__(self, seed):
        self.s = seed & MASK64

    def next(self):
        x = self.s
        x ^= (x << 13) & MASK64
        x ^= x >> 7
        x ^= (x << 17) & MASK64
        self.s = x
        return x


def fnv1a(data):
    """FNV-1a 64-bit, the hash the fixtures_are_frozen test pins."""
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & MASK64
    return h


# ---------------------------------------------------------------------------
# Fixture generators — byte-exact replicas of src/fixtures.rs:58-151
# ---------------------------------------------------------------------------

def gen_text(limit):
    words = ("the of and to in was it his that he her with for had is you not be "
             "she on at by which have from this him they were all are but said one "
             "when there them would been will who more no if out so what up their "
             "then time into little about could than like other some only over "
             "such down your").split()
    assert len(words) == 64
    rng = XorShift(0x7465787400000001)
    out = bytearray()
    words_in_sentence = 0
    while len(out) < limit:
        w = words[rng.next() % 64]
        if words_in_sentence == 0:
            c = w.encode()
            # .to_ascii_uppercase() on the first byte (fixtures.rs:78)
            first = c[0]
            c = bytes([first if not (97 <= first <= 122) else first - 32]) + c[1:]
            out += c
        else:
            out += w.encode()
        words_in_sentence += 1
        r = rng.next() % 100
        if r < 8 and words_in_sentence > 3:
            out += b". "
            words_in_sentence = 0
            if rng.next() % 4 == 0:
                out.append(10)
        elif r < 12:
            out += b", "
        else:
            out.append(32)
    return bytes(out[:limit])


def gen_tabular(limit):
    status = ("active", "inactive", "pending", "active", "active")
    rng = XorShift(0x7461627500000002)
    out = bytearray(b"id,timestamp,region,status,value,flag\n")
    v = 100_000
    while len(out) < limit:
        v += 1
        ts = 1_700_000_000 + rng.next() % 86_400
        region = rng.next() % 4
        st = status[rng.next() % 5]
        value = rng.next() % 100_000
        flag = rng.next() % 2
        out += ("{},{},region-{:02},{},{}.{:02},{}\n".format(
            v, ts, region, st, value // 100, value % 100, flag)).encode()
    return bytes(out[:limit])


def gen_binary(limit):
    rng = XorShift(0x62696E6100000003)
    out = bytearray()
    while len(out) < limit:
        out += bytes((0x7F, 0x45, 0x4C, 0x46, 0x02, 0x01, 0x01, 0x00))
        out += (rng.next() & 0xFFFFFFFF).to_bytes(4, "little")  # rng.next() as u32
        out += (((len(out) & 0xFFFFFFFF) ^ 0xDEADBEEF)).to_bytes(4, "little")
        for _ in range(rng.next() % 6 + 2):
            out += rng.next().to_bytes(8, "little")
        out += b"\x00" * (rng.next() % 48)
    return bytes(out[:limit])


def gen_noise(limit):
    rng = XorShift(0x6E6F697300000004)
    out = bytearray()
    while len(out) < limit:
        out += rng.next().to_bytes(8, "little")
    return bytes(out[:limit])


GENERATORS = {
    "text": gen_text,
    "tabular": gen_tabular,
    "binary": gen_binary,
    "noise": gen_noise,
}
FROZEN_FNV = {  # src/fixtures.rs:402-407
    "text": 0xD2AD5CB3D9F2AC83,
    "tabular": 0x8F132A1F79EC4511,
    "binary": 0xFE903199456D928D,
    "noise": 0xCDFD5FB185201167,
}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

DIST_BUCKETS = (64, 512, 2048, 4096, 8192, 16384, 32768)


def features(d):
    n = len(d)
    # ---- trigram repeat-distance pass: direct-addressed last-occurrence table
    # 2^24 buckets == every possible 3-byte key exactly once -> no collisions.
    tbl = array.array("i", bytes(4 * (1 << 24)))
    counts = [0] * (len(DIST_BUCKETS) + 2)  # 7 buckets + beyond + first
    last = 0
    d1 = d[1:]
    d2 = d[2:]
    i = 0
    for x, y, z in zip(d, d1, d2):
        k = (x << 16) | (y << 8) | z
        p = tbl[k]
        if p:
            dist = i - (p - 1)
            if dist <= DIST_BUCKETS[0]:
                counts[0] += 1
            elif dist <= DIST_BUCKETS[1]:
                counts[1] += 1
            elif dist <= DIST_BUCKETS[2]:
                counts[2] += 1
            elif dist <= DIST_BUCKETS[3]:
                counts[3] += 1
            elif dist <= DIST_BUCKETS[4]:
                counts[4] += 1
            elif dist <= DIST_BUCKETS[5]:
                counts[5] += 1
            elif dist <= DIST_BUCKETS[6]:
                counts[6] += 1
            else:
                counts[7] += 1
        else:
            counts[8] += 1
        tbl[k] = i + 1
        i += 1
    del tbl
    ntri = n - 2
    pct = [100.0 * c / ntri for c in counts]
    near3 = sum(counts[:4])      # <= 4096 (= greedy's fixed guard)
    far3 = counts[4] + counts[5] + counts[6]  # 4097..32768
    beyond = counts[7]
    matched = near3 + far3       # <= 32768, inside the DEFLATE window

    # ---- trigram distribution entropy (second pass over a count table)
    cnt = array.array("i", bytes(4 * (1 << 24)))
    for x, y, z in zip(d, d1, d2):
        cnt[(x << 16) | (y << 8) | z] += 1
    distinct = (1 << 24) - cnt.count(0)
    h3 = 0.0
    for c in cnt:
        if c:
            p = c / ntri
            h3 -= p * math.log2(p)
    del cnt

    # ---- byte-value stats (C-speed counting over full payload)
    counts8 = [0] * 256
    for b in d:
        counts8[b] += 1
    h0 = 0.0
    for c in counts8:
        if c:
            p = c / n
            h0 -= p * math.log2(p)
    nb = sum(1 for c in counts8 if c)
    top8 = 100.0 * sum(sorted(counts8, reverse=True)[:8]) / n

    # ---- stride autocorrelation, C-speed via bigint XOR + zero-byte count
    stride = {}
    for t in (4, 8, 16, 32):
        if n <= t:
            stride[t] = 0.0
            continue
        x = int.from_bytes(d[:-t], "big") ^ int.from_bytes(d[t:], "big")
        z = x.to_bytes(n - t, "big").count(0)
        stride[t] = 100.0 * z / (n - t)

    return {
        "bytes": n,
        "near3": pct[0:4],
        "far3": pct[4:7],
        "beyond": pct[7],
        "first": pct[8],
        "near3_tot": 100.0 * near3 / ntri,
        "far3_tot": 100.0 * far3 / ntri,
        "far_a": 100.0 * counts[4] / ntri,        # 4097-8192: lazy fixed guard
        "mid_tot": 100.0 * (counts[5] + counts[6]) / ntri,  # 8193-32768
        "matched": 100.0 * matched / ntri,
        "h3": h3,
        "h3_per_b": h3 / 3.0,
        "distinct_tri": distinct,
        "tri_density": distinct / ntri,
        "h0": h0,
        "nbytes": nb,
        "top8": top8,
        "stride": stride,
        "smax": max(stride.values()),
    }


# ---------------------------------------------------------------------------
# Shape-class assignment (the atlas's working taxonomy; see at-corpus-classes.md)
# ---------------------------------------------------------------------------

SHAPE_CLASSES = {
    "text (fixture)": "prose-text",
    "tabular (fixture)": "precision-columns",
    "binary (fixture)": "native-binary",
    "noise (fixture)": "incompressible",
    "dickens": "prose-text",
    "webster": "prose-text",
    "reymont": "prose-text",
    "mozilla": "native-binary",
    "samba": "archive-mix",
    "silesia.tar": "archive-mix",
    "software.archive": "archive-mix",
    "xml": "cadence-markup",
    "logs.txt": "local-log",
    "nci": "precision-columns",
    "osdb": "precision-columns",
    "ooffice": "already-compressed",
    "sao": "scientific-stride",
    "mr": "scientific-stride",
    "x-ray": "scientific-stride",
}


def row(name, src, f):
    vals = [
        "`{}`".format(name), src, "{:,}".format(f["bytes"]),
        *["{:.2f}".format(v) for v in f["near3"]],
        *["{:.2f}".format(v) for v in f["far3"]],
        "{:.3f}".format(f["beyond"]), "{:.1f}".format(f["first"]),
        "{:.3f}".format(f["h3"]), "{:.2f}".format(f["h3_per_b"]),
        "{:.2f}".format(f["tri_density"] * 100),
        "{:.2f}".format(f["h0"]), "{}".format(f["nbytes"]),
        "{:.1f}".format(f["top8"]),
        *["{:.2f}".format(f["stride"][t]) for t in (4, 8, 16, 32)],
        "{:.2f}".format(max(f["stride"].values())),
    ]
    assert len(vals) == 23, "cell count {} != 23".format(len(vals))
    return "| " + " | ".join(vals) + " |\n"


def main():
    results = []
    head = (
        "| member | source | trigram positions "
        "| dist <=64 | 65-512 | 513-2048 | 2049-4096 (%) "
        "| 4097-8192 | 8193-16384 | 16385-32768 (%) "
        "| >32768 (%) | first (%) | H3 bt/tri | H3/3 | trigram kinds % "
        "| H0 bb | byte vals | top-8 mass % "
        "| acorr4 % | acorr8 % | acorr16 % | acorr32 % | stride max % |"
    )
    print("### Panel 1 - full feature table\n")
    print(head)
    print("|---|" + "---:|" * 22)

    for name, gen in GENERATORS.items():
        d = gen(LEN)
        got = fnv1a(d)
        want = FROZEN_FNV[name]
        if got != want:
            sys.exit("fixture replica mismatch: " + name)
        src = "fixture replica 1 MiB, frozen FNV OK"
        results.append((name + " (fixture)", features(d), src))

    corpus = []
    silesia_dir = "/tmp/atlas-silesia/silesia"
    for f in sorted(os.listdir(silesia_dir)):
        corpus.append(("/tmp/atlas-silesia/silesia/" + f, f))
    for f in ("logs.txt", "software.archive", "silesia.tar"):
        corpus.append(("/Users/jackdanger/www/gzippy/benchmark_data/" + f, f))

    for path, name in corpus:
        with open(path, "rb") as fh:
            d = fh.read(CAP)
        src = "{:,} of {:,} B".format(len(d), os.path.getsize(path))
        results.append((name, features(d), src))

    for name, f, src in results:
        print(row(name, src, f), end="")

    # Panel 2: derived splits over the windows the len-3 policies speak in:
    # near <=4096 (greedy fixed guard, parse/greedy.rs:245), 4097-8192 (lazy
    # fixed guard), 8193-32768 (clean far), all <=32768 (matched), >window.
    print("\n| member | shape class | near<=4096 % | 4097-8192 % | 8193-32768 % "
          "| matched<=32768 % | >window % | acorr_max % |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, f, _ in results:
        print("| `{}` | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.3f} | {:.1f} |".format(
            name, SHAPE_CLASSES[name], f["near3_tot"], f["far_a"], f["mid_tot"],
            f["matched"], f["beyond"], f["smax"]))

    # Panel 3: per-shape-class means over the members of each class.
    print("\n| shape class | members | near mean % | far mean % | matched mean % "
          "| H3/3 mean | H0 mean | top-8 mean % | acorr max mean % |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    groups = {}
    for name, f, _ in results:
        groups.setdefault(SHAPE_CLASSES[name], []).append((name, f))
    for cls in sorted(groups):
        ms = groups[cls]
        def m(key):
            return sum(f[key] for _, f in ms) / len(ms)
        names = ", ".join("`{}`".format(n.split(" (")[0]) for n, _ in ms)
        print("| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.1f} | {:.1f} |".format(
            cls, names, m("near3_tot"), m("far3_tot"), m("matched"), m("h3_per_b"),
            m("h0"), m("top8"), m("smax")))


if __name__ == "__main__":
    main()
