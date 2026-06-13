## (d1) staleness set incomplete — **CLOSED**

`input_fingerprint()` (the `find` line) enumerates `src build.rs Cargo.toml Cargo.lock vendor benches`, so `vendor/` and `benches/` — the two the old mtime set missed — are now part of the digest. The stamp is written post-build (`input_fingerprint > "$FPRINT"`) and re-checked on no-build (`CUR_FP="$(input_fingerprint)"` vs `STAMP_FP`). The set is complete to "everything rsync ships that can change the bin."

## (d2) cross-host mtime skew false-negative — **CLOSED**

The guard no longer touches mtime: it compares two content shas (`CUR_FP` vs `STAMP_FP`) computed by `sha256sum` over file contents, then `sha256sum | cut`. Both digests are clock-independent and computed on the same host at compare time, so owner/guest clock skew cannot produce a false "fresh." The absent-stamp branch also hard-fails (`no-build-fingerprint … re-run with --build`), closing the "can't prove freshness ⇒ measure anyway" gap.

## (f3) HOST_FROZEN overriding a readable-thawed host — **CLOSED**

State machine: `gov_state`/`trb_state` resolve to `MATCH` (equal), `NA` (literal `NA` ⇒ unreadable), else `WRONG`. The `case` order is decisive:
- `*WRONG*` arm precedes the NA arm, and `case` takes first match — so any concrete readable mismatch fails hard (`host-not-frozen … a READABLE thawed value cannot be overridden`) **regardless of HOST_FROZEN**.
- `HOST_FROZEN=1` is only consulted in the trailing `*)` arm, reachable solely when no component is `WRONG` (i.e. NA + MATCH). It rescues NA only.

An operator asserting frozen on a box whose sysfs reads `powersave`/`0` gets `WRONG` → exit 13, un-overridable.

## VERDICT

**CAN-REPORT-CONTAMINATED = NO** — all three residuals (d1, d2, f3) are closed; none of the three can route a contaminated number to a printed verdict.

(Non-blocking note, outside the three: ordering of guards is sound — env scrub → build/fingerprint → corpus sha → ParallelSM assertion → host-freeze → regular-file sink → per-run sha-verify, with VOID on any divergence. Nothing reopens d/f.)
