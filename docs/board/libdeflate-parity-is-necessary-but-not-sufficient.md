# Tying libdeflate everywhere CANNOT reach zero failing cells

**Measured 2026-08-01.** Deterministic size only — arch-invariant, load-immune, no
frozen box needed. Binaries and coordinates at the bottom.

## The claim

libdeflate is **not the smallest vendor at every (level, file)**. On 19 coordinates
spanning **every level 1-9**, libdeflate's own output is LARGER than gzip's or pigz's
at the same label. On those coordinates, being byte-identical to libdeflate means
inheriting its loss — the cell fails.

```
cells where libdeflate LOSES to gzip or pigz (T1, levels 1-9):  32
distinct (level, file) coordinates:                             19
levels affected:                                     1,2,3,4,5,6,7,8,9  (all of them)

by level:  L1=7  L2=3  L3=3  L4=2  L5=4  L6=6  L7=5  L8=1  L9=1
by file:   dd79_bin6 6   data.sqlite 4   minjs.min.js 4   aozora.txt 4
           monorepo.tar 4   photo.jpg 3   weights.safetensors 3
           armexe.elf 2    access.log 2
```

Largest gaps: `data.sqlite` at L1 (libdeflate is **5.6% bigger** than gzip),
`dd79_bin6` at L1 (4.0%), `access.log` at L5 (1.1%), `monorepo.tar` at L6 (0.59%),
`aozora.txt` at L6 (0.57%).

## Why this matters for the current phase

The standing directive is *"reach exact architectural parity with libdeflate… do not
[steal from other vendors] until every single cell is a tie and none a loss."* Those
two halves are not simultaneously satisfiable by parity alone: **at these 19
coordinates a tie WITH libdeflate IS a loss to gzip.**

This is not an argument to abandon parity. Parity is still the floor, it still closes
the whole libdeflate class, and the `ldx` port has now proven L0-L9 byte-exact. It is
an argument that **phase 1 has a known, bounded, enumerated residue** — 32 cells at T1
— which phase 2 must beat rather than tie, and that the residue should be named now
rather than discovered at the end.

## The receipt that makes it concrete: routing L1 to `ldx`

`src/compress/ldx` is byte-identical to libdeflate at L1 (23/23, sha256, verified
against BOTH the vendored library and a pristine upstream build). So routing L1 to it
is a change whose size effect is exactly computable without running the parser:

```
rival        fail now  fail after  closed  OPENED
gzip                1           4       0       3
pigz                0           3       0       3
libdeflate         15           0      15       0
TOTAL              16           7      15       6
```

**15 closed, 6 opened. Clause 3 is absolute, so this is NO-SHIP as it stands.** The six
that open are three files, each failing against both gzip and pigz:

| file | ours now | via ldx | gzip | excess |
|---|---|---|---|---|
| `armexe.elf` | 599,781 | 621,027 | 617,178 | +3,849 (0.62%) |
| `data.sqlite` | 13,201,103 | 15,698,314 | 14,863,748 | +834,566 (5.6%) |
| `dd79_bin6` | 4,480,440 | 4,667,981 | 4,486,452 | +181,529 (4.0%) |

Those three files are **exactly** the L1 coordinates in the table above where
libdeflate loses to gzip. They are all binaries.

## This independently confirms the record at `parse/mod.rs:540`

That record says attempt 1 died on size because *"`fast`'s `head3` LENGTH-3 table wins
on BINARIES and `ht_matchfinder` has no length-3 support"*. This measurement reaches
the same conclusion from the opposite direction — a FAITHFUL port of
`ht_matchfinder.h` rather than a derivative — and quantifies it: the mechanism costs
0.62-5.6% on three binaries, and it is not a porting defect. It is what
`ht_matchfinder.h` is.

**It also shows our 27 recorded T1 "WINS" over libdeflate are load-bearing, not
decoration.** On `armexe.elf`, `data.sqlite` and `dd79_bin6` our igzip-derived L1 is
what keeps those gzip and pigz cells PASSING. Any change that gives up a win over
libdeflate must check whether that win was the only thing holding another vendor's
cell.

## What this does NOT say

* It does not say the `ldx` port was wasted — it closes 15 L1 cells and it is the only
  exact oracle we have.
* It does not say routing L1 is dead. It says the lever is `ht_matchfinder` **plus** a
  length-3 table, which is attempt 2 in the binding record: that one PASSED its size
  leg and died on the **T1 wall** at 1.2662x. The wall is a solvency measurement and
  was not attempted here.
* It says nothing about T4, where 125 of the 200 failing cells live. This scan is T1.

## Provenance

* ours: `/Users/jackdanger/www/gzippy/target/release/gzippy`, tree `976d558e`,
  sha256 `54079f4306cc6723…`, invoked `-1 -c -p1`.
* ldx: `port/libdeflate-exact` `examples/ldxdump`; raw DEFLATE + 18 bytes of gzip
  container. That the `ldx` column equals the `libdeflate-gzip` column on all 22 files
  confirms both the byte-identity and that the container is exactly 18 bytes.
* rivals: homebrew `gzip`, `pigz` (`-p1`), `libdeflate-gzip`. **`igzip` is not
  installed on this box**, so igzip cells are NOT covered by this scan.
* corpus: all 22 files of `~/www/gzippy-bench/corpus`.
* Scripts: `scratchpad/ldxdiff/l1size.sh`, `vendorloss.sh`.

**Coverage limits, stated so they are not read as absent:** T1 only, levels 1-9 only,
gzip and pigz only (no igzip), size only (no wall).
