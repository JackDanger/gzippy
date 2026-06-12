# Native-rg Rescore: Comparator Correction

**Date:** 2026-06-12  
**Branch:** measure/native-rg-rescore  
**Commit HEAD:** 5e242e16 (no gzippy code changes; script fixes only)

---

## 1. Native rg Provenance

| Field | Value |
|-------|-------|
| Path | `/root/gzippy-p35/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip` |
| SHA256 | `b0397fca1bb1fd5f214a01c0b2f1fb3d8366279dd922920fca27a353ed1a9b49` |
| Version | `rapidgzip 0.16.0` |
| `--version` wall | **4 ms** (< 15 ms threshold → native ELF confirmed) |
| Wheel path | `/usr/local/bin/rapidgzip` |
| Wheel `--version` wall | **48 ms** (wheel-suspect confirmed) |
| Build origin | `cmake` in `/root/gzippy-p35/vendor/rapidgzip/librapidarchive/build/` |

Startup-cost self-check added to `_parity_guest.sh`: logs `--version` wall at every run and emits
WARN if >= 15 ms so a wheel-as-comparator regression is caught mechanically.

---

## 2. Script Changes Summary

**`scripts/bench/guest.env`** — comparison-target section replaced:
- Old: `RG=rapidgzip` (resolved via `which` → pip wheel at `/usr/local/bin/rapidgzip`, 48 ms startup)
- New: `RG_BIN=<pinned native path>` + `RG=${RG_BIN}` (primary comparator); `RG_WHEEL_BIN=/usr/local/bin/rapidgzip` (kept, labeled)
- Added: `RG_BIN_SHA` (sha256 of native binary, integrity-checked at runtime)
- `RG_TRACE` unchanged (oracle/fulcrum use only)

**`scripts/bench/parity.sh`** — `remote_env()` extended to pass `RG_BIN`, `RG_BIN_SHA`, `RG_WHEEL_BIN` to the guest runner.

**`scripts/bench/_parity_guest.sh`** — rapidgzip-presence section replaced:
1. Tries `$RG_BIN` first; sha-verifies against `$RG_BIN_SHA`; warns and falls through on mismatch.
2. Falls back to `$RG` (PATH), then `$RG_TRACE`.
3. Startup-cost self-check: measures `--version` wall; emits WARN if >= 15 ms (wheel-suspect), confirms native if < 15 ms.
4. Provenance line now includes `[${rg_label}]` and `path=` so every result self-documents which arm was used.

---

## 3. Re-Scored Matrix vs Native rg Bar

**Measurement protocol:**
- N=7 trials per cell (N=5 for T16); interleaved per-trial; warmup iter-0 dropped
- Regular-file sinks on `/dev/shm`; sha-verified every trial
- Masks: T1=0, T4=0,2,4,6, T8=0,2,4,6,8,10,12,14, T16=0-15 (minus core 15)
- Two gzippy builds: `gzippy-isal` (`parallel-sm+isal`) and `gzippy-native` (`parallel-sm+pure`)
- Comparator: native rg at `/root/gzippy-p35/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip`
- Wheel arm (T8 only): `/usr/local/bin/rapidgzip` for tax-delta labeling
- Verdict bar: ratio = rg_native_min / gz_min >= 0.99 → PASS; < 0.99 → LOSS

**Corpus availability:**  weights and monorepo returned HTTP 403 from `squishy.jackdanger.com` — those cells are UNAVAILABLE.

### Full Matrix

| corpus | T | build | gz_min (ms) | rg_native_min (ms) | ratio | verdict vs 0.99 |
|--------|---|-------|-------------|---------------------|-------|-----------------|
| silesia | 1 | gzippy-isal | 739 | 836 | **1.131** | PASS |
| silesia | 1 | gzippy-native | 1003 | 836 | **0.834** | LOSS |
| silesia | 4 | gzippy-isal | 513 | 419 | **0.816** | LOSS |
| silesia | 4 | gzippy-native | 521 | 419 | **0.804** | LOSS |
| silesia | 8 | gzippy-isal | 337 | 286 | **0.848** | LOSS |
| silesia | 8 | gzippy-native | 348 | 286 | **0.822** | LOSS |
| silesia | 16 | gzippy-isal | 364 | 278 | **0.765** | LOSS |
| silesia | 16 | gzippy-native | 325 | 278 | **0.856** | LOSS |
| bignasa | 8 | gzippy-isal | 930 | 828 | **0.891** | LOSS |
| bignasa | 8 | gzippy-native | 954 | 828 | **0.868** | LOSS |
| model | 1 | gzippy-isal | 2028 | 2246 | **1.108** | PASS |
| model | 1 | gzippy-native | 2992 | 2246 | **0.751** | LOSS |
| model | 8 | gzippy-isal | 439 | 337 | **0.768** | LOSS |
| model | 8 | gzippy-native | 518 | 337 | **0.651** | LOSS |
| storedheavy | 1 | gzippy-isal | 82 | 113 | **1.372** | PASS |
| storedheavy | 1 | gzippy-native | 102 | 113 | **1.099** | PASS |
| storedheavy | 8 | gzippy-isal | 82 | 72 | **0.884** | LOSS |
| storedheavy | 8 | gzippy-native | 75 | 72 | **0.961** | LOSS |
| storedmix | 8 | gzippy-isal | 83 | 74 | **0.891** | LOSS |
| storedmix | 8 | gzippy-native | 77 | 74 | **0.958** | LOSS |
| weights | 8 | — | UNAVAIL | — | — | — |
| weights | 16 | — | UNAVAIL | — | — | — |
| monorepo | 1 | — | UNAVAIL | — | — | — |
| monorepo | 8 | — | UNAVAIL | — | — | — |

### Wheel Tax-Delta Column (T8 cells only)

Tax = rg_wheel_min − rg_native_min. Wheel-era ratio = rg_wheel_min / gz_min (what was banked).

| corpus | T | build | rg_native (ms) | rg_wheel (ms) | tax (ms) | native_ratio | wheel_era_ratio | delta (native−wheel) | flip? |
|--------|---|-------|----------------|----------------|-----------|--------------|-----------------|----------------------|-------|
| silesia | 8 | gzippy-isal | 286 | 352 | **66** | 0.848 | 1.042 | **−0.194** | LOSS→PASS |
| silesia | 8 | gzippy-native | 286 | 352 | **66** | 0.822 | 1.010 | **−0.188** | LOSS→PASS |
| bignasa | 8 | gzippy-isal | 828 | 914 | **85** | 0.891 | 0.982 | **−0.091** | LOSS→LOSS (worse) |
| bignasa | 8 | gzippy-native | 828 | 914 | **85** | 0.868 | 0.957 | **−0.089** | LOSS→LOSS (worse) |
| model | 8 | gzippy-isal | 337 | 385 | **48** | 0.768 | 0.877 | **−0.109** | LOSS→LOSS (worse) |
| model | 8 | gzippy-native | 337 | 385 | **48** | 0.651 | 0.744 | **−0.093** | LOSS→LOSS (worse) |
| storedheavy | 8 | gzippy-isal | 72 | 115 | **42** | 0.884 | **1.399** | **−0.515** | LOSS→PASS (big flip) |
| storedheavy | 8 | gzippy-native | 72 | 115 | **42** | 0.961 | **1.522** | **−0.561** | LOSS→PASS (big flip) |
| storedmix | 8 | gzippy-isal | 74 | 116 | **42** | 0.891 | **1.405** | **−0.514** | LOSS→PASS (big flip) |
| storedmix | 8 | gzippy-native | 74 | 116 | **42** | 0.958 | **1.511** | **−0.553** | LOSS→PASS (big flip) |

*"flip?" column = what the wheel SHOWED → what native SHOWS*

---

## 4. Flip Analysis

### Cells that flip PASS→LOSS (wheel phantom wins now revealed as losses)

| corpus | T | build | wheel ratio | native ratio | change |
|--------|---|-------|-------------|--------------|--------|
| silesia | 8 | gzippy-isal | 1.042 (PASS) | 0.848 (LOSS) | −0.194 |
| silesia | 8 | gzippy-native | 1.010 (PASS) | 0.822 (LOSS) | −0.188 |
| storedheavy | 8 | gzippy-isal | 1.399 (PASS) | 0.884 (LOSS) | −0.515 |
| storedheavy | 8 | gzippy-native | 1.522 (PASS) | 0.961 (LOSS) | −0.561 |
| storedmix | 8 | gzippy-isal | 1.405 (PASS) | 0.891 (LOSS) | −0.514 |
| storedmix | 8 | gzippy-native | 1.511 (PASS) | 0.958 (LOSS) | −0.553 |

**Campaign-level flip:** The wheel-era table showed gzippy-isal WINNING storedheavy and storedmix at T8 (1.4-1.5x) and tying/winning silesia T8 (1.04x). Against native rg, all six of those cells are LOSSES. The "closed" status of these cells must be **re-opened**.

### Cells that remain LOSS (status unchanged, magnitude worse)

| corpus | T | build | wheel ratio | native ratio | change |
|--------|---|-------|-------------|--------------|--------|
| bignasa | 8 | gzippy-isal | 0.982 | 0.891 | −0.091 |
| bignasa | 8 | gzippy-native | 0.957 | 0.868 | −0.089 |
| model | 8 | gzippy-isal | 0.877 | 0.768 | −0.109 |
| model | 8 | gzippy-native | 0.744 | 0.651 | −0.093 |

These were already LOSSES and remain LOSSES; the native bar makes the gap larger by the tax amount.

### Cells that remain PASS (wins survive the honest bar)

| corpus | T | build | native ratio | notes |
|--------|---|-------|--------------|-------|
| silesia | 1 | gzippy-isal | 1.131 | ISA-L single-shot path; structural win |
| model | 1 | gzippy-isal | 1.108 | ISA-L single-shot path |
| storedheavy | 1 | gzippy-isal | 1.372 | Large win; stored-block fast path |
| storedheavy | 1 | gzippy-native | 1.099 | StoredParallel routing beats rg at T1 |

At T1 the 43 ms tax is ~5% of rg's silesia time (836 ms), so wheel-era T1 ratios are only ~0.05 inflated — wins that were large enough survive. All T1 ISA-L wins remain real.

### Wheel tax by corpus (T8, ms)

| corpus | rg_native (ms) | rg_wheel (ms) | tax (ms) | tax as % of native |
|--------|----------------|----------------|-----------|---------------------|
| silesia | 286 | 352 | 66 | 23% |
| bignasa | 828 | 914 | 85 | 10% |
| model | 337 | 385 | 48 | 14% |
| storedheavy | 72 | 115 | 42 | 58% |
| storedmix | 74 | 116 | 42 | 57% |

The tax is NOT a flat 43 ms. Faster corpora (stored blocks) show a ~42 ms tax close to the bare Python startup. Slower corpora (bignasa, silesia) show larger taxes (66-85 ms), likely because the Python process also incurs GIL/IO overhead that scales with data volume. The storedheavy/storedmix 58% tax explains why those cells looked like large wins with the wheel.

---

## 5. Anomalies (Undecorated)

1. **silesia T16 gzippy-native (325 ms) faster than T16 gzippy-isal (364 ms)**: Pure-Rust native build scales better than isal at T16. ISA-L's single-thread advantage reverses at high thread counts. Not expected from prior sessions.

2. **storedheavy T1 both builds WIN vs rg** (isal: 1.372x, native: 1.099x). Routing: isal → `IsalSingleShot`, native → `StoredParallel`. At T8, rg overtakes (isal 0.884, native 0.961). The stored-block T1 win is real on native rg — it was not a wheel phantom.

3. **storedmix T8 gzippy-native at 0.958**: Narrowly below the 0.99 bar. A 4% loss. With a 42 ms wheel tax on a 74 ms native rg time, the wheel made this look like a 1.51x win.

4. **model T1 gzippy-native at 0.751**: The engine-W gap is severe at T1 for the pure-Rust path (2992 ms vs 2246 ms rg). This is the funded engine-rewrite cell.

5. **bignasa T8 spread is large**: isal spread in this session was ~10% (med=1031 vs min=930). High variance cell; repeat may be warranted. The ratio is 0.891 (solid LOSS).

6. **Wheel tax for bignasa is 85 ms** (not 42-43 ms), larger than the bare Python startup. Reason is unclear — possibly Python IO overhead scales with uncompressed data size at 991 MB/s throughput.

---

## 6. Summary Verdict

**PASS cells (native rg bar):** 4 of the measurable cells (silesia T1 isal, model T1 isal, storedheavy T1 both builds).

**LOSS cells:** All T4/T8/T16 parallel-SM cells on silesia; all bignasa/model T8 cells; storedheavy T8; storedmix T8.

**Net change from wheel era:** 6 cells flip from PASS→LOSS (silesia T8 both builds, storedheavy T8 both builds, storedmix T8 both builds). No LOSS→PASS flips (native rg is strictly harder). The "campaign mostly closed" picture from the wheel era is materially wrong — the correct picture is that only T1 ISA-L cells and storedheavy/storedmix T1 pass.

**Weights and monorepo:** UNAVAILABLE (corpus not on guest; squishy 403). Status cannot be re-scored.
