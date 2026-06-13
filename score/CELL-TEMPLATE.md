SCORE: <arch-os> <tN> <corpus> | native=<r> <PASS|FAIL> | isal=<r> <PASS|FAIL> | rg=<wall>ms | N=<n> frozen <date> | blind:src=<sha7>,dist=<RESOLVED|...>,lever=<tag>

<!-- COPY this file to score/<arch-os>/<tN>/<corpus>.md and fill EVERY <...>. -->
<!-- The SCORE line MUST be line 1 (regen-index.sh greps it). See ../SCHEMA.md. -->

```yaml
cell: <arch-os>/<tN>/<corpus>
date: <YYYY-MM-DD>
box: <neurotic|solvency>
arch_os: <intel-x86_64|amd-x86_64>
threads: <N>
thread_mask: "<actual affinity/mask, confirmed via lscpu>"
corpus: <corpus>
corpus_pin:
  path: <abs path to .gz input on box>
  sha256: <sha256 of the .gz input>
  decompressed_sha256: <sha256 of gunzip output — the byte-exact target>
frozen:
  method: "<bench-lock.sh | no_turbo | source bench-env.sh && freeze>"
  readback: "<verbatim freeze readback>"
samples: <n>
comparator: rapidgzip-native
bar: ">=0.99 ratio = PASS"
builds:
  rapidgzip-native:
    wall_ms: <>
    spread_ms: <>
    sha256: <>
    ratio: 1.00
    verdict: COMPARATOR
    flavor: native-elf
  gzippy-native:
    wall_ms: <>
    spread_ms: <>
    sha256: <>
    ratio: <>
    verdict: <PASS|FAIL>
    flavor: pure-rust-inflate     # assert: NO isal symbols on decode path
  gzippy-isal:
    wall_ms: <>
    spread_ms: <>
    sha256: <>
    ratio: <>
    verdict: <PASS|FAIL>
    flavor: isal                  # assert: isal ON decode path
parity:
  native_vs_rg: <>
  isal_vs_rg: <>
  native_vs_isal: <>
distribution: <RESOLVED|BIMODAL|NOISY>   # must match dist= tag
blindspots:
  - "<cell-specific caveat>"
dominant_lever: <tag>                     # must match lever= tag
```

## VERDICT

<2-5 sentences: which build passes, the binding constraint, the headline.>

## fulcrum decide

```
<VERBATIM paste of `scripts/fulcrum decide` + the 3-way wall capture. Do not edit.>
```

## FINDINGS

- <cell-specific observations>

## RE-VERIFY

```bash
# exact reproduction: ssh, freeze, build all 3, fulcrum, sha-verify
```
