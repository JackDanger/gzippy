# score/ SCHEMA — the cell file contract

This file is the AUTHORITATIVE format for every cell file under
`score/<arch-os>/<threadcount>/<corpus>.md`. Workers MUST follow it exactly so
`regen-index.sh` (a dumb `grep`) can derive every README without re-measuring.

A cell file records ONE frozen measurement of one (arch-os × threadcount ×
corpus) cell: how gzippy's two production builds (`gzippy-native`,
`gzippy-isal`) compare to the `rapidgzip-native` ELF comparator on that exact
workload, with all numbers pasted VERBATIM from `scripts/fulcrum`.

---

## Axes (the matrix)

| axis        | values                                                  |
|-------------|---------------------------------------------------------|
| arch-os     | `intel-x86_64` (neurotic), `amd-x86_64` (solvency)      |
| threadcount | `t1`, `t4`, `t8`, `t12`, `t16`                           |
| corpus      | `model`, `monorepo`, `storedheavy`, `silesia`, `bignasa`|

Mac is EXCLUDED — no trustworthy wall (no freeze).

Builds compared per cell (the 3-way capture):
- `rapidgzip-native` — the comparator ELF (NEVER the pip wheel).
  neurotic: `/root/oracle_c/rapidgzip-native`; solvency: `/root/gz-base/vendor/rapidgzip` build.
- `gzippy-native`  — `--no-default-features --features pure-rust-inflate` (C-FFI off the decode graph).
- `gzippy-isal`    — default features (ISA-L on the x86 decode path).

`ratio = rapidgzip-native wall / build wall`. **ratio >= 0.99 => PASS** (the
TIE bar: at-or-faster than rapidgzip). ratio < 0.99 => FAIL.

---

## Cell file structure (5 required parts, IN ORDER)

### 1. The `SCORE:` line (FIRST LINE, greppable, single line)

Exact format (one line, pipe-delimited):

```
SCORE: <arch-os> <tN> <corpus> | native=<ratio> <PASS|FAIL> | isal=<ratio> <PASS|FAIL> | rg=<wall>ms | N=<n> frozen <date> | blind:<tags>
```

- `<ratio>` = 2-decimal float, e.g. `0.88`.
- `rg=<wall>ms` = rapidgzip-native median wall in ms (the comparator anchor).
- `N=<n>` = sample count (>=3; <3 is itself a blindspot).
- `frozen <date>` = ISO date the freeze run was taken, e.g. `2026-06-13`.
- `blind:<tags>` = comma-separated derivation tags consumed by regen-index.sh.
  REQUIRED tags (no spaces):
  - `src=<sha7>`   git short-sha the binaries were built from (staleness vs HEAD).
  - `dist=<RESOLVED|BIMODAL|NOISY|...>` distribution verdict (untrustworthy if != RESOLVED).
  - `lever=<tag>`  the dominant cell lever (where gz exceeds/lags rg), e.g. `lever=finalize`, `lever=engine-W`, `lever=none`.
  - additional free tags allowed (e.g. `mask=8P4E`), comma-joined.
  - if genuinely nothing to flag beyond the required three, still emit all three.

Example:
```
SCORE: intel-x86_64 t8 silesia | native=0.74 FAIL | isal=0.99 PASS | rg=247ms | N=9 frozen 2026-06-13 | blind:src=1825d17,dist=RESOLVED,lever=engine-W,mask=8P
```

### 2. YAML header block

Fenced ```yaml block immediately after the SCORE line. Schema:

```yaml
cell: intel-x86_64/t8/silesia
date: 2026-06-13
box: neurotic
arch_os: intel-x86_64
threads: 8
thread_mask: "0-7"          # actual mask/affinity used; confirm via lscpu at fill time
corpus: silesia
corpus_pin:                 # the input the numbers are for
  path: /root/corpora/silesia.tar.gz
  sha256: <sha256 of the .gz input>
  decompressed_sha256: <sha256 of the gunzip output — the byte-exact target>
frozen:
  method: "/root/bench-lock.sh"   # or "no_turbo" / "source /root/bench-env.sh && freeze"
  readback: "<verbatim freeze readback line>"
samples: 9                  # 3 interleaved sweeps x 3 = 9
comparator: rapidgzip-native
bar: ">=0.99 ratio = PASS"
builds:
  rapidgzip-native:
    wall_ms: 247
    spread_ms: 4
    sha256: <binary sha256>
    ratio: 1.00
    verdict: COMPARATOR
    flavor: native-elf
  gzippy-native:
    wall_ms: 334
    spread_ms: 6
    sha256: <binary sha256>
    ratio: 0.74
    verdict: FAIL
    flavor: pure-rust-inflate      # MUST assert: no isal symbols
  gzippy-isal:
    wall_ms: 249
    spread_ms: 5
    sha256: <binary sha256>
    ratio: 0.99
    verdict: PASS
    flavor: isal                   # MUST assert: isal on decode path
parity:
  native_vs_rg: 0.74
  isal_vs_rg: 0.99
  native_vs_isal: 0.75
distribution: RESOLVED      # RESOLVED | BIMODAL | NOISY ; must match dist= tag
blindspots:
  - "<cell-specific caveat, free text>"
dominant_lever: engine-W    # must match lever= tag
```

### 3. `## VERDICT` — prose

2-5 sentences: what this cell says, which build passes, the binding constraint.

### 4. `## fulcrum decide` — VERBATIM paste

The deterministic source of truth. Paste the COMPLETE `scripts/fulcrum decide`
output (and the 3-way wall capture) verbatim, fenced. Every number in parts
1-3 must be traceable to this paste. NEVER hand-time; NEVER edit the paste.

### 5. `## FINDINGS` — cell-specific

Bullets unique to this cell (e.g. "native bimodal at t16 from E-core spill").

### 6. `## RE-VERIFY` — exact reproduction command

The exact shell to re-take this cell (box, freeze cmd, build cmds, fulcrum
invocation, sha-verify). Someone with box access must be able to copy-paste it.

---

## Re-measurement / archival

Re-measuring a cell MOVES the old file into:
```
score/archive/<arch-os>/<tN>/<corpus>/<date>-<binsha7>.md
```
(`<binsha7>` = first 7 of the gzippy-native binary sha256). Then write the new
cell at the canonical path. Never overwrite in place — history is the audit
trail.

---

## Discipline (banked, mandatory)

- A LEVER = where gzippy EXCEEDS/lags rapidgzip (the DELTA), never an absolute
  removal-oracle that also removes rg-shared work.
- instructions != wall on this workload — wall claims use fulcrum cycles/TMA.
- Byte-exact: every run sha-verified against `corpus_pin.decompressed_sha256`.
- STRIKE-5: assert input sha == pin BEFORE measuring; abort the cell otherwise.
- SINK LAW: decompress to a regular-file sink (never /dev/null shortcuts that
  skip write cost asymmetrically across tools).
- Numbers come ONLY from Fulcrum (`scripts/fulcrum decide`), pasted verbatim.
