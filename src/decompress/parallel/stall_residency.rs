//! STEP-0 discriminator (a): PARENT-CACHED-AT-STALL probe.
//!
//! See plans/step0-discriminator-a-falsifier.md. Answers the
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
/// A DECODED chunk resident in the prefetch_cache or main cache has a range
/// `[encoded_offset_bits, max_acceptable_start_bit]` that STRICTLY CONTAINS the
/// stalled `decode_start` (decode_start > encoded_offset_bits) → interior reuse
/// could emit `[confirmed, end]` from it. The best case for the placement port.
pub static CONTAINING_CACHED: AtomicU64 = AtomicU64::new(0);
/// No decoded resident chunk contains it, but an IN-FLIGHT prefetch key ≤
/// decode_start exists whose partition spans it (the eventual range MAY contain
/// it; the Arc is not yet decoded so the range is unknowable — keyed estimate).
pub static CONTAINING_IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
/// Neither a resident decoded chunk nor a plausible in-flight prefetch contains
/// the stalled offset → EVICTED/absent. The re-scope signal.
pub static NOT_RESIDENT: AtomicU64 = AtomicU64::new(0);

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
    cached: &[(usize, usize)],
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

    // (1) A decoded resident chunk whose range strictly contains decode_start.
    let resident_contains = cached.iter().any(|&(enc, max)| {
        // matches_encoded_offset semantics: enc <= decode_start <= max,
        // AND strictly interior (decode_start > enc) — an exact-start match
        // would already have been accepted upstream, so the stall implies
        // either interior or absent.
        enc <= decode_start && decode_start <= max && decode_start > enc
    });
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

fn emit_trace(decode_start: usize, class: &str, cached: &[(usize, usize)], in_flight: &[usize]) {
    if crate::decompress::parallel::trace::is_enabled() {
        // Dump the resident ranges + in-flight keys so a NOT_RESIDENT verdict is
        // auditable: did a containing chunk genuinely not exist, or did the
        // containment test miss it? Also report the NEAREST resident chunk whose
        // start ≤ decode_start (the candidate parent had the consumer kept pace).
        let ranges: String = cached
            .iter()
            .map(|&(e, m)| format!("[{e},{m}]"))
            .collect::<Vec<_>>()
            .join(" ");
        let nearest = cached
            .iter()
            .filter(|&&(e, _)| e <= decode_start)
            .max_by_key(|&&(e, _)| e);
        let nearest_s = match nearest {
            Some(&(e, m)) => format!(
                r#","nearest_le_start":{e},"nearest_le_max":{m},"nearest_gap_to_max":{}"#,
                decode_start as i64 - m as i64
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
         | non_startup={non_startup} resident(non-startup-est)={resident} ({resident_pct:.1}% of non-startup) \
         | conservation_ok={}",
        cached + inflight + absent == total,
    );
}
