"""gzippy adapter — the first (reference) ProjectAdapter.

Everything gzippy-specific that used to be hard-coded in scripts/fulcrum_*.py
lives here: the span taxonomy, the counter sidecar patterns, the re-derived
routing/seeding guard, the knob registry + effect predicates, the contig_prof
parser, the banked comparators, and the re-verify command surfaces.

The launch/environment policy (host freeze, canonical pin masks, regular-file
sinks, per-run sha verification) is the shell side: scripts/bench/decide.sh +
_decide_guest.sh + lib_decide_guest.sh in the host repo.
"""

import re

from .base import Knob, ProjectAdapter, Taxonomy

# ---------------------------------------------------------------------------
# Span classification taxonomy.
#
# A WAIT (blocked on another thread's decode future) must NEVER be counted as
# serial COMPUTE work -- that inversion bit the campaign. Names are matched by
# prefix against the wired trace_v2 sites (grep 'SpanGuard::begin'
# src/decompress/parallel/). UNKNOWN names are surfaced, never silently
# bucketed.
# ---------------------------------------------------------------------------

GZIPPY_TAXONOMY = Taxonomy(
    # A WAIT span = this thread is BLOCKED on another thread / future / lock.
    wait_prefixes=(
        "wait.",            # generic wait spans (wait.future_recv, ...)
        "lock.wait",        # blocked acquiring a mutex
        "pool.pick.wait",   # worker idle waiting for a task
        "consumer.wait_replaced_markers",  # blocked on marker-resolve future
        "consumer.dispatch_recv",          # blocking recv on post-proc future
        # ttp.rx_recv_block is THE wait that gates the in-order wall (~97% of
        # it per block_fetcher.rs:245). Mis-bucketing it as compute is the
        # binder-inversion that bit the campaign.
        "ttp.rx_recv_block",
        "ttp.get_if_available",
    ),
    # OUTPUT = bytes/checksum leaving the pipeline (the serial tail rapidgzip
    # explicitly minimizes); kept distinct from COMPUTE.
    output_prefixes=(
        "consumer.writev",
        "consumer.write_buffered",
        "consumer.combine_crc",
        "consumer.publish_windows",
        "consumer.window_publish_clean",
        "consumer.window_publish_marker",
    ),
    # COMPUTE = actual decode / marker-resolve / window-apply work.
    compute_prefixes=(
        "worker.",
        "post_process.apply_window",
        "post_process.task",
        "pool.run_task",
        "consumer.eager_postproc",
        "consumer.process_prefetches",
        "consumer.queue_prefetched_postproc",
        "consumer.arc_take_or_clone",
        "consumer.dispatch_post_process",
        "consumer.get_last_window",
        "consumer.try_take_prefetched",
        "consumer.block_finder_get",
        "ttp.take_prefetch",
        "coord.prefetch",
    ),
    # Scheduler bookkeeping: neither engine compute nor a blocking wait.
    sched_overhead_prefixes=(
        "pool.submit",
        "pool.pick.lock",
        "pool.pick",
    ),
    # Outer loop frames nest everything; excluded from the busy/wait split to
    # avoid double-counting children.
    outer_frame_names=("consumer.iter", "consumer.drain",
                       "consumer.dispatch_recv"),
    # lock.held is OVERHEAD, NOT busy -- counting both lock.held and the work
    # done while holding it would double-count.
    overhead_prefixes=("lock.held",),
    # Emitted ONLY on the consumer thread (chunk_fetcher.rs:1096/:3577).
    consumer_exclusive_frames=("consumer.iter", "consumer.drain"),
)


# ---------------------------------------------------------------------------
# Counter sidecar patterns -- the WINDOW-ABSENT / SEEDING / ORACLE guard data.
# ---------------------------------------------------------------------------

COUNTER_PATTERNS = {
    "window_seeded": r"window_seeded=(\d+)",
    "flip_to_clean": r"flip_to_clean=(\d+)",
    "finished_no_flip": r"finished_no_flip=(\d+)",
    "seeded_block": r"seeded_block=(\d+)",
    "seeded_wrapper": r"seeded_wrapper=(\d+)",
    "exact_block": r"exact_block=(\d+)",
    "exact_wrapper": r"exact_wrapper=(\d+)",
    # The binary emits `isal_chunks=` / `isal_fallbacks=` (chunk_fetcher.rs:870-871,
    # the ISA-L clean-tail line). The OLD pattern here was `isal_oracle_chunks=` — a
    # label the binary NEVER prints (the same historical grep-bug the instrument
    # registry documents for _oracle_guest.sh), so the oracle arm of the guard
    # could never fire on a real sidecar. Keep the legacy key for old synthetic
    # sidecars; the lookbehind keeps the two patterns disjoint.
    "isal_chunks": r"(?<!oracle_)isal_chunks=(\d+)",
    "isal_fallbacks": r"(?<!oracle_)isal_fallbacks=(\d+)",
    "isal_oracle_chunks": r"isal_oracle_chunks=(\d+)",
    "isal_oracle_fallbacks": r"isal_oracle_fallbacks=(\d+)",
    "bad_seed_resync": r"bad_seed_resync=(\d+)",
    # ONLY printed when GZIPPY_SEED_WINDOWS replay mode is ON (seed_windows.rs
    # report_seed_stats is a no-op off seed) — THE oracle-seeding tell.
    "seed_replay_hits": r"SEED_WINDOWS replay: hits=(\d+)",
    # ONLY printed when GZIPPY_BYPASS_DECODE replay is active
    # (decode_bypass.rs:628 report_replay_stats; no-ops when replay disabled).
    "bypass_replay_hits": r"BYPASS_DECODE replay: hits=(\d+)",
}


# ---------------------------------------------------------------------------
# contig_prof parser (the [contig-prof] stderr block, contig_prof.rs:195-299).
# ---------------------------------------------------------------------------

PROF_CLASS_RE = re.compile(
    r"^\s+(lit1|litpack|litchn|backref)\s*: iters=\s*(\d+) cyc=\s*(\d+)\s+"
    r"([\d.]+)% of classed,\s+([\d.]+) cyc/iter", re.M)
PROF_HEAD_RE = re.compile(
    r"calls=(\d+) total_cyc=(\d+) classed_cyc=(\d+) \(([\d.]+)% of total")
PROF_DIST_RE = re.compile(r"disttbl: builds=(\d+) reuses=(\d+)")
PROF_WRAPPER_CALLS_RE = re.compile(
    r"\[contig-prof\] WRAPPER.*?calls=(\d+)", re.S)


def parse_prof(text):
    out = {"classes": {}, "head": None, "disttbl": None, "wrapper_calls": None}
    m = PROF_HEAD_RE.search(text)
    if m:
        out["head"] = {"calls": int(m.group(1)), "total_cyc": int(m.group(2)),
                       "classed_cyc": int(m.group(3)),
                       "classed_pct": float(m.group(4))}
    for cm in PROF_CLASS_RE.finditer(text):
        out["classes"][cm.group(1)] = {
            "iters": int(cm.group(2)), "cyc": int(cm.group(3)),
            "share_pct": float(cm.group(4)), "cyc_iter": float(cm.group(5))}
    dm = PROF_DIST_RE.search(text)
    if dm:
        out["disttbl"] = (int(dm.group(1)), int(dm.group(2)))
    wm = PROF_WRAPPER_CALLS_RE.search(text)
    if wm:
        out["wrapper_calls"] = int(wm.group(1))
    return out


# ---------------------------------------------------------------------------
# Banked comparators (provenance-pinned). If a live measurement CONTRADICTS a
# bank row by more than the stated tolerance, the row is flagged
# DIVERGES-FROM-BANK instead of silently ranking. (The generalized,
# fingerprint-aware version of this is core/ledger.py; these rows predate the
# ledger and stay as the contig-prof structural comparator.)
# ---------------------------------------------------------------------------

BANK = {
    # plans/orchestrator-status.md "P3.5 OFFICIAL MATRIX @ a9fe662c" (2026-06-10),
    # CONTIG_PROF @ silesia T8 on bin-p35-native:
    "silesia_T8_backref_share": 62.6,   # % of classed cycles
    "silesia_T8_backref_cyc": 34.9,     # cyc/iter
    "silesia_T8_litchn_share": 22.9,    # % of classed cycles
    # NATIVE T1 trajectory (frozen): gz 1375ms vs rg ~914-921 => ~1.5x symbol rate.
    "silesia_T1_rg_ratio_band": (0.60, 0.72),
}
BANK_REL_TOL = 0.25  # >25% relative divergence => flag


def bank_divergence(cell, prof):
    """Compare a silesia T8 prof against the banked P3.5 comparator. Returns a
    list of human-readable divergence strings (empty == consistent)."""
    if cell != ("silesia", 8) or not prof["classes"]:
        return []
    div = []
    br = prof["classes"].get("backref")
    if br:
        share_ok = abs(br["share_pct"] - BANK["silesia_T8_backref_share"]) \
            / BANK["silesia_T8_backref_share"] <= BANK_REL_TOL
        for key, val in (("share_pct", BANK["silesia_T8_backref_share"]),
                         ("cyc_iter", BANK["silesia_T8_backref_cyc"])):
            got = br[key]
            if val and abs(got - val) / val > BANK_REL_TOL:
                msg = (f"backref.{key}={got:.1f} vs banked {val} "
                       f"(>±{BANK_REL_TOL:.0%}) — DIVERGES-FROM-BANK")
                if key == "cyc_iter" and share_ok:
                    msg += (" [shares MATCH the bank => structure consistent; "
                            "absolute TSC-cyc/iter scales with core-clock state "
                            "(TSC is fixed-rate) — suspect a frequency-state "
                            "mismatch between captures (frozen no_turbo here vs "
                            "the bank's capture), not a code change]")
                div.append(msg)
    lc = prof["classes"].get("litchn")
    if lc:
        val = BANK["silesia_T8_litchn_share"]
        if abs(lc["share_pct"] - val) / val > BANK_REL_TOL:
            div.append(f"litchn.share={lc['share_pct']:.1f} vs banked {val} "
                       f"— DIVERGES-FROM-BANK")
    return div


class GzippyAdapter(ProjectAdapter):
    name = "gzippy"
    # The binding TIE bar (project_tie_bar_99pct_all_threadcounts):
    # >=0.99x at EVERY thread count.
    tie_bar = 0.99
    taxonomy = GZIPPY_TAXONOMY

    # Knob registry (env of the FEATURE-ALTERED arm; MUST match the registry in
    # scripts/bench/_decide_guest.sh).
    knobs = {
        "dist_amort": Knob("GZIPPY_DIST_AMORT=0", "prof_dist",
                           "P3.4 DistTable amortization"),
        "stored_flip": Knob("GZIPPY_NO_STORED_FLIP=1", "none",
                            "M2b stored early-flip"),
        "seeded_block": Knob("GZIPPY_SEEDED_BLOCK=0", "verbose_seeded",
                             "M3 seeded chunks on Block"),
        "exact_block": Knob("GZIPPY_EXACT_BLOCK=0", "verbose_exact",
                            "M4 until-exact on Block"),
        "hit_drive": Knob("GZIPPY_NO_HIT_DRIVE=1", "none",
                          "confirmed-offset hit-drive prefetch"),
        "slab_alloc": Knob("GZIPPY_SLAB_ALLOC=1", "rpmalloc_stats",
                           "slab allocator force-on (the reverted lever, "
                           "reconciled: auto-ON at T<=GZIPPY_SLAB_MAX_T — "
                           "expect CAUSAL-NULL at default-ON cells)",
                           reverted=True),
        "slab_off": Knob("GZIPPY_SLAB_ALLOC=0", "rpmalloc_stats_off",
                         "slab force-OFF (gate proof: at T1 default-ON the "
                         "knob arm must lose the slab win and zero the slab "
                         "counters)"),
        "slab_bigbudget": Knob("GZIPPY_SLAB_BUDGET_MIB=600", "none",
                               "budget-shape probe (evidence trail): "
                               "admit-everything retention (~the original f2 "
                               "force-on class) vs the default T x largest "
                               "budget — separates budget-shape headroom from "
                               "state-dependence of the -99.9ms finding"),
        "eager_postproc": Knob("GZIPPY_EAGER_POSTPROC=1", "none",
                               "eager consumer post-processing (opt-in)"),
        # ISA-L-specific: always-small initial buffer (faithful rapidgzip
        # ALLOCATION_CHUNK_SIZE segment-append) vs the default ratio-informed
        # upfront reserve.  Effect predicate is "none" — no in-tree counter
        # tracks growth mode; A/B is wall-only.  Only meaningful on
        # gzippy-isal (the ISA-L clean-tail build); vacuous on gzippy-native
        # (GZIPPY_ISAL_INCREMENTAL_GROWTH is read only inside the isal
        # clean-tail code path).
        "isal_incremental_growth": Knob(
            "GZIPPY_ISAL_INCREMENTAL_GROWTH=1", "none",
            "ISA-L always-small initial buffer (vs ratio-informed reserve): "
            "faithfully ports rapidgzip ALLOCATION_CHUNK_SIZE=128KiB "
            "append loop; knob arm = always-small; base arm = production "
            "ratio-reserve"),
    }

    perturbations = {
        "compute": ("GZIPPY_SLOW_MODE=50 [GZIPPY_SLOW_KIND=sleep control] via "
                    "scripts/bench/oracle.sh --kind perturb (clean-loop "
                    "slow-inject, slow_knob.rs)"),
        "output": "GZIPPY_SKIP_WRITEV_SYSCALL=1 A/B (output-stage removal probe)",
        "wait": ("worker-side lever — perturb the ENGINE (slow_knob) and watch "
                 "this wait shrink/grow; the wait itself is not the cause"),
        "idle": ("scheduling-state probe: N=21 re-measure (bimodal check) "
                 "before anything"),
    }

    # ---- comparator identity --------------------------------------------------
    def comparator_version(self, manifest):
        """Normalize the rapidgzip --version banner recorded by the guest
        (`rg_version=`). Handles both the full banner ("rapidgzip, CLI to
        the ... library rapidgzip version 0.16.0") and the short
        "rapidgzip 0.16.0" form. Unknown stays unknown (never compares)."""
        raw = (manifest.get("rg_version") or "").strip()
        if not raw:
            return "unknown"
        m = re.search(r"(\d+\.\d+(?:\.\d+)*)\s*$", raw)
        if m:
            return f"rapidgzip {m.group(1)}"
        return raw  # unrecognized shape: keep verbatim (still a known value)

    # ---- counters / guards --------------------------------------------------
    def parse_counters(self, text):
        out = {}
        for key, pat in COUNTER_PATTERNS.items():
            m = re.search(pat, text)
            if m:
                out[key] = int(m.group(1))
        return out

    def routing_guard(self, counters, feature=None):
        """(is_production, reason). A run is NOT production (binder-masking)
        iff an ACTUAL oracle contaminated it. RE-DERIVED 2026-06-10 (fulcrum2
        charter):

        The OLD rule refused on window_seeded>0. That counter
        (WINDOW_SEEDED_CHUNKS, gzip_chunk.rs:1181) increments for ANY
        full-32KiB-initial-window decode -- which since M3 includes PRODUCTION
        chunks whose predecessor window the live WindowMap published
        (chunk_fetcher.rs:2545 materialize path). So the old guard OVER-FIRED
        on every healthy native/isal production run. window_seeded>0 alone is
        production-seeded routing, not contamination.

        The ACTUAL contamination signals (each individually sufficient):
          1. seed_replay_hits>0 -- the GZIPPY_SEED_WINDOWS oracle store is
             active; oracle-seeded windows force the clean engine at
             boundaries production would marker-bootstrap.
          2. bypass_replay_hits>0 -- pre-computed decode results replayed.
          3. ISA-L engine chunks on a NATIVE build: isal_chunks>0 is
             PRODUCTION on gzippy-isal (the clean-tail engine) but oracle-only
             on gzippy-native (GZIPPY_ISAL_ENGINE_ORACLE). `feature`
             disambiguates; unknown => conservative refuse.
        """
        if not counters:
            return (None,
                    "NO COUNTER SIDECAR -- cannot verify production routing. "
                    "Capture with GZIPPY_VERBOSE=1 2> verbose_<label>.txt and "
                    "pass --counters. REFUSING to certify this as a "
                    "production-routing measurement.")
        feat = (feature or "").replace("gzippy-", "")
        replay = counters.get("seed_replay_hits", 0)
        bypass = counters.get("bypass_replay_hits", 0)
        oracle = max(counters.get("isal_chunks", 0),
                     counters.get("isal_oracle_chunks", 0))
        seeded = counters.get("window_seeded", 0)
        flips = counters.get("flip_to_clean", 0)
        no_flip = counters.get("finished_no_flip", 0)
        seeded_block = counters.get("seeded_block", 0)
        exact_block = counters.get("exact_block", 0)
        if replay > 0:
            return (False,
                    f"ORACLE-SEEDED RUN (SEED_WINDOWS replay hits={replay}). "
                    f"The seed store forced clean-engine decodes at boundaries "
                    f"production would marker-bootstrap. This measures the "
                    f"clean-engine ceiling, NOT production.")
        if bypass > 0:
            return (False,
                    f"BYPASS_DECODE REPLAY ACTIVE (hits={bypass}). "
                    f"Pre-computed decode results replayed — real engine cost "
                    f"masked. This is a measurement contaminant, NOT "
                    f"production.")
        if oracle > 0 and feat != "isal":
            if feat == "native":
                return (False,
                        f"ISA-L ENGINE ORACLE RAN (isal_chunks={oracle} on a "
                        f"gzippy-native build -- only GZIPPY_ISAL_ENGINE_ORACLE "
                        f"reaches that engine there). A CEILING oracle, not "
                        f"production.")
            return (False,
                    f"isal_chunks={oracle} with build feature UNDECLARED -- "
                    f"production on gzippy-isal, an engine oracle on native. "
                    f"Pass --feature to disambiguate; refusing conservatively.")
        # Production confirmation: SOME decode-path counter must have fired,
        # else we cannot rule out the silently-skipped/re-ran-bootstrap class.
        if no_flip == 0 and flips == 0 and seeded == 0 and seeded_block == 0 \
                and exact_block == 0:
            return (None,
                    "No decode-path counter fired (finished_no_flip, "
                    "flip_to_clean, window_seeded, seeded_block, exact_block "
                    "all 0) -- cannot confirm the production pipeline ran. "
                    "(The 'oracle silently re-ran/skipped the bootstrap' "
                    "failure class.) Inconclusive.")
        seeded_note = (f"window_seeded={seeded} is PRODUCTION-SEEDED routing "
                       f"(WindowMap-published predecessor windows, M3+), "
                       if seeded > 0 else "window_seeded=0, ")
        isal_note = (f"isal_chunks={oracle} (PRODUCTION clean-tail on "
                     f"gzippy-isal), " if oracle > 0 else "")
        return (True,
                f"PRODUCTION routing confirmed: no SEED_WINDOWS replay, no "
                f"engine oracle ({seeded_note}{isal_note}"
                f"finished_no_flip={no_flip}, flip_to_clean={flips}, "
                f"seeded_block={seeded_block}, exact_block={exact_block}).")

    def oracle_guard(self, counters, trace_self):
        """REMOVAL-ORACLE contamination check: a handicapped contender (e.g. a
        per-chunk 64MiB alloc/copy production never pays) must NOT be read as
        a ceiling."""
        warns = []
        if counters:
            fb = max(counters.get("isal_fallbacks", 0),
                     counters.get("isal_oracle_fallbacks", 0))
            oc = max(counters.get("isal_chunks", 0),
                     counters.get("isal_oracle_chunks", 0))
            if oc > 0 and fb > 0:
                warns.append(
                    f"ORACLE IMPURE: {fb}/{oc+fb} chunks fell back to the real "
                    f"engine -- the oracle did NOT replace 100% of decode; its "
                    f"wall is a BLEND, not a clean ceiling.")
        for n in trace_self:
            if "to_vec" in n or "oracle_copy" in n or "oracle_alloc" in n:
                warns.append(
                    f"ORACLE COPY SPAN '{n}' present -- this is overhead the "
                    f"production path does not pay; subtract it before reading "
                    f"a ceiling (a handicapped contender != a ceiling).")
        return warns

    # ---- effect predicates ---------------------------------------------------
    def effect_check(self, pred, base_txt, knob_txt):
        """Prove the kill-switch actually flipped the feature. Returns
        (verified: bool|None, note). None = no in-tree counter =>
        EFFECT-UNVERIFIED label (never silently trusted)."""
        if pred == "none":
            return (None,
                    "no in-tree counter; A/B is wall-only (EFFECT-UNVERIFIED)")
        if pred == "verbose_seeded":
            m = re.search(r"seeded_block=(\d+) seeded_wrapper=(\d+)", knob_txt)
            if not m:
                return (False, "seeded_block counter line absent in knob arm")
            blk, wrp = int(m.group(1)), int(m.group(2))
            if blk == 0 and wrp > 0:
                return (True, f"knob arm: seeded_block=0, seeded_wrapper={wrp} "
                              f"(switch effective)")
            return (False, f"knob arm still seeded_block={blk} (switch "
                           f"INEFFECTIVE)")
        if pred == "verbose_exact":
            m = re.search(r"exact_block=(\d+) exact_wrapper=(\d+)", knob_txt)
            if not m:
                return (False, "exact_block counter line absent in knob arm")
            blk, wrp = int(m.group(1)), int(m.group(2))
            if blk == 0 and wrp > 0:
                return (True, f"knob arm: exact_block=0, exact_wrapper={wrp} "
                              f"(switch effective)")
            if blk == 0 and wrp == 0:
                return (None, "no until-exact chunks in this cell (predicate "
                              "vacuous)")
            return (False, f"knob arm still exact_block={blk} (switch "
                           f"INEFFECTIVE)")
        if pred == "prof_dist":
            # C_N_DISTBUILD / C_N_DISTREUSE are incremented ONLY in the
            # amortized (default) arm (marker_inflate.rs:2262/2266); the
            # GZIPPY_DIST_AMORT=0 kill-switch arm does fresh per-block builds
            # WITHOUT touching them. EFFECTIVE-switch signature: base arm
            # counters alive, knob arm counters DEAD.
            mb = re.search(r"disttbl: builds=(\d+) reuses=(\d+)", base_txt)
            mk = re.search(r"disttbl: builds=(\d+) reuses=(\d+)", knob_txt)
            if not (mb and mk):
                return (False, "disttbl prof line absent (capture without "
                               "GZIPPY_CONTIG_PROF?)")
            bb, br = map(int, mb.groups())
            kb, kr = map(int, mk.groups())
            if bb == 0 and br == 0:
                return (None, "base arm never hit the amortized build path "
                              "(no dynamic blocks?) — predicate vacuous")
            if kb == 0 and kr == 0:
                return (True, f"base builds={bb}/reuses={br} alive, knob arm "
                              f"counters dead (P3.4 path bypassed — switch "
                              f"effective)")
            return (False, f"knob arm still on the amortized path (builds={kb} "
                           f"reuses={kr}) — switch INEFFECTIVE")
        if pred == "rpmalloc_stats":
            # The [rpmalloc] stats dump prints in BOTH arms whenever
            # GZIPPY_RPMALLOC_STATS is set (line presence proves NOTHING — the
            # live functional check caught exactly that). Engagement proof is
            # the SLAB-SPECIFIC counters (rpmalloc_alloc.rs):
            #   "[rpmalloc {tag}] slab_hits=N slab_installs=M"
            kc = _slab_counts(knob_txt)
            bc = _slab_counts(base_txt)
            if kc is None:
                return (False,
                        "no slab_hits=/slab_installs= counters in knob arm — "
                        "binary predates the engagement counters or stats not "
                        "captured")
            if kc == 0:
                return (False,
                        "slab counters ZERO in knob arm — slab never engaged "
                        "(threshold/feature mismatch?) — switch INEFFECTIVE")
            if bc is None or bc > 0:
                return (False,
                        f"slab counters in BASE arm = {bc!r} (expected 0) — "
                        f"switch not exclusive or stats missing in base")
            return (True, f"knob arm slab engaged (hits+installs={kc}); base "
                          f"arm 0 — switch effective")
        if pred == "rpmalloc_stats_off":
            # INVERTED engagement proof for the force-off knob: on a
            # default-ON cell the BASE arm must show slab counters > 0 and the
            # knob arm (GZIPPY_SLAB_ALLOC=0) must zero them.
            kc = _slab_counts(knob_txt)
            bc = _slab_counts(base_txt)
            if bc is None:
                return (False,
                        "no slab_hits=/slab_installs= counters in base arm — "
                        "binary predates the engagement counters or stats not "
                        "captured")
            if bc == 0:
                return (None,
                        "slab counters ZERO in base arm — auto gate not "
                        "engaged on this cell (T above GZIPPY_SLAB_MAX_T?) — "
                        "predicate vacuous")
            if kc is None or kc > 0:
                return (False,
                        f"slab counters in KNOB arm = {kc!r} (expected 0) — "
                        f"GZIPPY_SLAB_ALLOC=0 did NOT disable the slab")
            return (True, f"base arm slab auto-engaged (hits+installs={bc}); "
                          f"force-off arm 0 — gate + kill-switch effective")
        return (False, f"unknown predicate '{pred}'")

    # ---- micro-profile --------------------------------------------------------
    def parse_microprofile(self, text):
        return parse_prof(text)

    def microprofile_rows(self, ck, prof, gap_ms, run):
        rows, anomalies = [], []
        if not prof or not prof["classes"]:
            return rows, anomalies
        for d in bank_divergence(ck, prof):
            anomalies.append(f"{ck[0]}:T{ck[1]}: {d}")
        for cls_name, c in sorted(prof["classes"].items(),
                                  key=lambda kv: -kv[1]["share_pct"]):
            bounded = gap_ms * c["share_pct"] / 100.0
            rows.append({
                "component": f"engine.{cls_name}",
                "kind": "engine",
                "perturb_cmd": self.perturbations["compute"],
                "cells": f"{ck[0]}:T{ck[1]}",
                "attrib": (f"{c['share_pct']:.1f}% of classed cyc, "
                           f"{c['cyc_iter']:.1f} cyc/iter, "
                           f"iters={c['iters']:,}"),
                "status": (f"HYPOTHESIS: bounded ≤{bounded:.0f}ms ESTIMATE "
                           f"(= cell rg-gap {gap_ms:.0f}ms × class share; a "
                           f"partition, not a promise). Perturb: "
                           f"{self.perturbations['compute']}"),
                "dist": "prof=1-shot counters (unfrozen-counters label)",
                "verify": (f"GZIPPY_CONTIG_PROF=1 GZIPPY_VERBOSE=1 taskset -c "
                           f"<mask> {run['manifest'].get('bin')} -d -c -p "
                           f"{ck[1]} /root/{ck[0]}.gz >/dev/null"),
                "tier": 2, "rank_ms": bounded,
            })
        if prof.get("wrapper_calls"):
            anomalies.append(
                f"{ck[0]}:T{ck[1]}: WRAPPER calls={prof['wrapper_calls']} "
                f"(expected 0 — contig should be the sole production engine)")
        return rows, anomalies

    # ---- re-verify command surfaces --------------------------------------------
    def reverify_knob(self, ck, kname, run):
        return (f"scripts/bench/decide.sh --cells {ck[0]}:{ck[1]} "
                f"--knob-cells {ck[0]}:{ck[1]} --knobs {kname} "
                f"--knob-n 21 --bin {run['manifest'].get('bin')}")

    def reverify_trace(self, ck, run, feature):
        return (f"python3 scripts/fulcrum_total.py "
                f"<artdir>/cell_{ck[0]}_T{ck[1]}/trace.json "
                f"--feature {feature}")


def _slab_counts(txt):
    m = re.search(r"slab_hits=(\d+) slab_installs=(\d+)", txt)
    return (int(m.group(1)) + int(m.group(2))) if m else None
