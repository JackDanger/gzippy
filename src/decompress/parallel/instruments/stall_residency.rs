//! STEP-0 discriminator (a): PARENT-CACHED-AT-STALL probe.
//!
//! See git history (campaign plan, removed). Answers the
//! [[project_confirmed_offset_prefetch_gap]] UNANSWERED discriminator: when the
//! consumer stalls at a CONFIRMED encoded-bit offset and falls through to a
//! synchronous COLD `get_with_prefetch`, is the chunk whose decoded range
//! CONTAINS that confirmed offset (the overshoot PARENT) currently
//! cached/in-flight (recoverable → interior-reuse / getIndexedChunk port is the
//! fix), or evicted/absent (→ cache-residency/consumer-pace re-scope)?
//!
//! COUNTERS ONLY — zero decode-behavior change. Env-gated by
//! `GZIPPY_STALL_RESIDENCY_PROBE`; OFF is an inlined early-return so OFF==identity
//! (proven by dual-sha). The probe takes read-only snapshots of the fetcher's
//! caches + in-flight key set at the stall site; it never mutates decode flow.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

static ENABLED: OnceLock<bool> = OnceLock::new();

#[inline]
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("GZIPPY_STALL_RESIDENCY_PROBE").is_some())
}

/// Total cold-get stalls observed (the `None` branch = no usable partition-keyed
/// prefetch; the consumer is about to block on a synchronous decode).
pub static STALLS_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Stalls at offset 0 (startup / first chunk — unavoidable, excluded from the
/// majority verdict).
pub static STALLS_STARTUP: AtomicU64 = AtomicU64::new(0);
/// A DECODED chunk resident in the prefetch_cache or main cache spans the
/// stalled `decode_start` in its ENCODED range `[encoded_offset_bits,
/// encoded_offset_bits + encoded_size_bits)` AND starts strictly before it
/// (decode_start > encoded_offset_bits) → interior reuse could emit
/// `[confirmed, end]` from it. (Advisor fix: this uses the encoded END, NOT
/// max_acceptable_start_bit which is the speculative start-tolerance window that
/// collapses to enc==max after re-anchor and could never match a real parent.)
pub static CONTAINING_CACHED: AtomicU64 = AtomicU64::new(0);
/// A resident chunk merely STARTS at/below decode_start (the NECESSARY condition
/// for a containing parent — the bug-free discriminating channel the advisor
/// endorsed). If this is 0 across non-startup stalls, no resident chunk could
/// possibly contain the stalled offset (parent never retained / consumed-ahead).
pub static HAS_NEAREST_LE_START: AtomicU64 = AtomicU64::new(0);
/// No decoded resident chunk contains it, but an IN-FLIGHT prefetch key ≤
/// decode_start exists whose partition spans it (the eventual range MAY contain
/// it; the Arc is not yet decoded so the range is unknowable — keyed estimate).
pub static CONTAINING_IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
/// Neither a resident decoded chunk nor a plausible in-flight prefetch contains
/// the stalled offset → EVICTED/absent. The re-scope signal.
pub static NOT_RESIDENT: AtomicU64 = AtomicU64::new(0);

// --- SATURATION vs HORIZON occupancy channels (git history (campaign plan, removed)) ---
// At each non-startup cold-get stall, classify WHY the decode hasn't started:
/// All workers busy at the stall instant (idle_capacity == 0) — no free slot to
/// start the marginal index's decode. SATURATION (engine) signal.
pub static OCC_SAT: AtomicU64 = AtomicU64::new(0);
/// Idle worker capacity existed AND the stalled index was NEVER enqueued to the
/// prefetcher (no in-flight key covers it). HORIZON-too-shallow (structural) signal.
pub static OCC_HORIZON_NOT_ENQUEUED: AtomicU64 = AtomicU64::new(0);
/// Idle worker capacity existed AND the stalled index WAS in-flight but not done
/// (a key covers it). The "in-flight-not-done" lead-length / engine-speed case the
/// prior 3 attempts hit — NOT a horizon-DEPTH lever (it was dispatched).
pub static OCC_HORIZON_ENQUEUED_NOT_DONE: AtomicU64 = AtomicU64::new(0);
/// Sum of busy-worker counts over non-startup stalls (for a mean-busy report).
pub static OCC_BUSY_SUM: AtomicU64 = AtomicU64::new(0);
/// Sum of idle_capacity over non-startup stalls (for a mean-idle report).
pub static OCC_IDLE_CAP_SUM: AtomicU64 = AtomicU64::new(0);

/// Classify one non-startup cold-get stall by worker occupancy + enqueued status.
///
/// `busy` = workers actively running a task (`spawned - idle`).
/// `idle_capacity` = parked idle workers + not-yet-spawned slots (lazy spawn).
/// `enqueued` = an in-flight prefetch key covers `decode_start` (the index WAS
/// dispatched, just not finished).
///
/// Counters only; the caller already gates on `enabled()` + non-startup.
pub fn classify_occupancy(busy: usize, idle_capacity: usize, enqueued: bool) {
    if !enabled() {
        return;
    }
    OCC_BUSY_SUM.fetch_add(busy as u64, Ordering::Relaxed);
    OCC_IDLE_CAP_SUM.fetch_add(idle_capacity as u64, Ordering::Relaxed);
    if idle_capacity == 0 {
        OCC_SAT.fetch_add(1, Ordering::Relaxed);
    } else if enqueued {
        OCC_HORIZON_ENQUEUED_NOT_DONE.fetch_add(1, Ordering::Relaxed);
    } else {
        OCC_HORIZON_NOT_ENQUEUED.fetch_add(1, Ordering::Relaxed);
    }
}

/// One read-only residency snapshot at a cold-get stall.
///
/// `cached`: `(encoded_offset_bits, max_acceptable_start_bit)` for every decoded
/// chunk currently in the prefetch_cache + main cache.
/// `in_flight_keys`: the encoded-bit keys of currently in-flight prefetches.
/// `decode_start`: the confirmed encoded-bit offset the consumer stalled on.
/// `partition_offset_for`: maps a key/offset to its partition key (to test
/// whether an in-flight prefetch's partition could span `decode_start`).
pub fn classify_stall(
    decode_start: usize,
    cached: &[(usize, usize, usize)], // (encoded_start, max_start_tolerance, encoded_end)
    in_flight_keys: &[usize],
    partition_span: usize,
) {
    if !enabled() {
        return;
    }
    STALLS_TOTAL.fetch_add(1, Ordering::Relaxed);
    if decode_start == 0 {
        STALLS_STARTUP.fetch_add(1, Ordering::Relaxed);
    }

    // NECESSARY-condition channel (advisor-endorsed, bug-free): does ANY resident
    // chunk start strictly before decode_start? A containing parent must.
    if cached.iter().any(|&(enc, _max, _end)| enc < decode_start) {
        HAS_NEAREST_LE_START.fetch_add(1, Ordering::Relaxed);
    }

    // (1) A decoded resident chunk whose ENCODED range spans decode_start as an
    // interior offset (enc < decode_start < encoded_end). Uses the encoded END,
    // not the speculative start-tolerance `max` (advisor fix).
    let resident_contains = cached
        .iter()
        .any(|&(enc, _max, end)| enc < decode_start && decode_start < end);
    if resident_contains {
        CONTAINING_CACHED.fetch_add(1, Ordering::Relaxed);
        emit_trace(decode_start, "CONTAINING_CACHED", cached, in_flight_keys);
        return;
    }

    // (2) An in-flight prefetch whose partition window could span decode_start.
    // The in-flight Arc is not decoded yet (range unknown), so estimate by key:
    // a key K is in-flight and K <= decode_start < K + partition_span (i.e. the
    // decode that started at K could, once done, cover decode_start).
    let in_flight_contains = in_flight_keys
        .iter()
        .any(|&k| k <= decode_start && decode_start < k.saturating_add(partition_span));
    if in_flight_contains {
        CONTAINING_IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
        emit_trace(decode_start, "CONTAINING_IN_FLIGHT", cached, in_flight_keys);
        return;
    }

    NOT_RESIDENT.fetch_add(1, Ordering::Relaxed);
    emit_trace(decode_start, "NOT_RESIDENT", cached, in_flight_keys);
}

fn emit_trace(
    decode_start: usize,
    class: &str,
    cached: &[(usize, usize, usize)],
    in_flight: &[usize],
) {
    if crate::decompress::parallel::trace::is_enabled() {
        // Dump the resident ranges (enc..end) + in-flight keys so a NOT_RESIDENT
        // verdict is auditable. nearest_le_start = nearest resident chunk that
        // STARTS ≤ decode_start (the candidate containing parent); -1 ⇒ no
        // resident chunk even starts at/below the stalled offset (never retained).
        let ranges: String = cached
            .iter()
            .map(|&(e, _m, end)| format!("[{e},{end})"))
            .collect::<Vec<_>>()
            .join(" ");
        let nearest = cached
            .iter()
            .filter(|&&(e, _, _)| e <= decode_start)
            .max_by_key(|&&(e, _, _)| e);
        let nearest_s = match nearest {
            Some(&(e, _m, end)) => format!(
                r#","nearest_le_start":{e},"nearest_le_end":{end},"contains":{}"#,
                e < decode_start && decode_start < end
            ),
            None => r#","nearest_le_start":-1"#.to_string(),
        };
        crate::decompress::parallel::trace::emit(
            "stall_residency",
            "classify",
            &format!(
                r#""decode_start":{decode_start},"class":"{class}","cached_count":{},"in_flight_count":{},"cached_ranges":"{ranges}","in_flight_keys":"{:?}"{nearest_s}"#,
                cached.len(),
                in_flight.len(),
                in_flight,
            ),
        );
    }
}

/// Print the tally to stderr at consumer-loop teardown (the deliverable line).
/// Conservation: startup + cached + in_flight + not_resident == total.
pub fn report() {
    if !enabled() {
        return;
    }
    let total = STALLS_TOTAL.load(Ordering::Relaxed);
    let startup = STALLS_STARTUP.load(Ordering::Relaxed);
    let cached = CONTAINING_CACHED.load(Ordering::Relaxed);
    let inflight = CONTAINING_IN_FLIGHT.load(Ordering::Relaxed);
    let absent = NOT_RESIDENT.load(Ordering::Relaxed);
    let has_le = HAS_NEAREST_LE_START.load(Ordering::Relaxed);
    let non_startup = total.saturating_sub(startup);
    let resident = cached + inflight;
    let resident_pct = if non_startup > 0 {
        100.0 * resident as f64 / non_startup as f64
    } else {
        0.0
    };
    // `startup` is an ORTHOGONAL annotation on a stall (the offset-0 first
    // chunk), NOT a separate bucket — every stall (startup or not) also gets
    // exactly one of {cached, inflight, absent}. Conservation: the three
    // classes sum to total.
    eprintln!(
        "  STALL_RESIDENCY_PROBE: total={total} startup={startup} \
         CONTAINING_CACHED={cached} CONTAINING_IN_FLIGHT={inflight} NOT_RESIDENT={absent} \
         has_nearest_le_start={has_le} \
         | non_startup={non_startup} resident(non-startup-est)={resident} ({resident_pct:.1}% of non-startup) \
         | conservation_ok={}",
        cached + inflight + absent == total,
    );

    // SATURATION vs HORIZON occupancy report (git history (campaign plan, removed)).
    let sat = OCC_SAT.load(Ordering::Relaxed);
    let hz_ne = OCC_HORIZON_NOT_ENQUEUED.load(Ordering::Relaxed);
    let hz_end = OCC_HORIZON_ENQUEUED_NOT_DONE.load(Ordering::Relaxed);
    let busy_sum = OCC_BUSY_SUM.load(Ordering::Relaxed);
    let idle_sum = OCC_IDLE_CAP_SUM.load(Ordering::Relaxed);
    let occ_total = sat + hz_ne + hz_end;
    let mean_busy = if occ_total > 0 {
        busy_sum as f64 / occ_total as f64
    } else {
        0.0
    };
    let mean_idle = if occ_total > 0 {
        idle_sum as f64 / occ_total as f64
    } else {
        0.0
    };
    let verdict = if occ_total == 0 {
        "NONE"
    } else if sat * 2 >= occ_total {
        "SATURATION"
    } else if hz_ne * 2 >= occ_total {
        "HORIZON"
    } else if hz_end >= sat && hz_end >= hz_ne {
        "IN-FLIGHT-NOT-DONE"
    } else {
        "MIXED-AMBIGUOUS"
    };
    eprintln!(
        "  STALL_OCCUPANCY_PROBE: non_startup_classified={occ_total} \
         SAT={sat} HORIZON_NOT_ENQUEUED={hz_ne} HORIZON_ENQUEUED_NOT_DONE={hz_end} \
         | mean_busy={mean_busy:.2} mean_idle_cap={mean_idle:.2} \
         | conservation_ok={} VERDICT={verdict}",
        occ_total == non_startup,
    );
}
