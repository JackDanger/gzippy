# After #227, 98% of the residual is L1 — and the coordinate escape does not exist

**Measured 2026-08-01.** Deterministic size, T4, **all nine levels**, 22 corpus files,
3 rivals (gzip, pigz, libdeflate) = 594 cells. main `120bfa9c` vs #227 `669e9a0c`, both
clean-worktree builds. **No igzip on this box. Not a wall claim.**

#227 itself: **91 closed, 0 opened.** What remains is **60 cells**, and they decompose
with **zero remainder**:

| class | cells | total excess | note |
|---|---|---|---|
| **A2 — real gap vs libdeflate** | **13** | **2,104,681 B** | **all thirteen are L1** |
| A1 — seam vs libdeflate | 35 | 43,744 B | ratio < 1.001, levels 2-9 |
| B — vs gzip/pigz, INHERITED | 12 | 201,196 B | libdeflate loses these too |
| **C — vs gzip/pigz, OUR OWN defect** | **0** | 0 B | |

**A2 is 98% of all residual excess bytes and it is entirely level 1.** One cell per
file, ratios 1.0028 to 1.1032:

```
L1 access.log   1.1032 (+341,529)   L1 monorepo.tar 1.0565 (+639,359)
L1 data.csv     1.0457 (+179,667)   L1 aozora.txt   1.0405 (+185,069)
L1 ecoli.fastq  1.0283              L1 markup.xml   1.0252
L1 minjs.min.js 1.0226              L1 dickens      1.0212
L1 dd79_text6   1.0197              L1 data.json    1.0181
L1 engine.wasm  1.0111              L1 weights.safetensors 1.0036
L1 data.parquet 1.0028
```

**Class C being zero is worth stating plainly: there is no cell at T4 where we lose to
gzip or pigz and libdeflate does not.** Every non-libdeflate failure is inherited.

## The hypothesis this measurement KILLED

`ldx` is byte-identical to libdeflate at L1, so routing L1 to it closes all 13 of class
A2. At T1 that was already measured as **14 closed / 6 opened** — blocked, because
`armexe.elf`, `data.sqlite` and `dd79_bin6` pass against gzip/pigz today and would fail.

CLAUDE.md says *"the budget is the tightest PASSING rival… an already-failing cell
cannot be flipped"*, and the L1 parked record's REOPEN contract points at re-testing the
**coordinate**. So the obvious move was: gate the routing to T>1, where those binary
cells might already be failing and clause 3 therefore could not be violated.

**That escape does not exist. Measured at T4:**

```
L1 @ T4 routed to ldx:   CLOSED 14   OPENED 6   unchanged 46
```

| opened at T4 | ours now (PASSES) | via ldx | rival |
|---|---|---|---|
| armexe.elf vs gzip | 600,310 | 621,027 | 617,178 |
| armexe.elf vs pigz | 600,310 | 621,027 | 615,561 |
| data.sqlite vs gzip | 13,198,414 | 15,698,314 | 14,863,748 |
| data.sqlite vs pigz | 13,198,414 | 15,698,314 | 14,830,275 |
| dd79_bin6 vs gzip | 4,484,295 | 4,667,981 | 4,486,452 |
| dd79_bin6 vs pigz | 4,484,295 | 4,667,981 | 4,491,293 |

Those three files **pass at T4 as well as at T1** — our igzip-derived L1 beats gzip and
pigz on them at both thread counts. **The same 6 cells open at either coordinate.** The
verdict is coordinate-INDEPENDENT, which is the opposite of what the parked record's
REOPEN contract invited us to hope.

## What that leaves, and it is now unambiguous

The L1 lever needs `ht_matchfinder` **plus length-3 match support**, so it keeps
libdeflate's text performance without giving up the binaries. That is exactly attempt 2
in the binding record at `src/compress/deflate/parse/mod.rs:540` — which **passed its
size leg and died on the T1 wall at 1.2662x**.

So the remaining question for the single largest residual class is a **wall** question
at T1, not a size question and not a coordinate question. It needs solvency.

## Coverage limits

T4 for the residual decomposition; the L1-routing check was run at **both** T1 and T4.
All nine levels. **gzip, pigz, libdeflate only — no igzip on this box.** Size only.
This is the residual after a PR that **has not merged**.

Scripts: `scripts/campaign/reattest227.sh`, `reattest227_ungraded.sh`.
