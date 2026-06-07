#![cfg(parallel_sm)]

//! CLEAN-ONLY ENGINE ORACLE (2026-06-07, campaign instrument).
//!
//! Purpose: bound the ENGINE (class-C) ceiling cleanly — the least-entangled
//! engine signal. Force EVERY chunk down the CLEAN (window-present) decode path
//! (`decode_chunk_with_until_exact`) instead of the markered speculative path,
//! by providing each worker the CORRECT predecessor 32 KiB window (captured from
//! a prior real decode) when the live `WindowMap` has not yet published it.
//!
//! This is the OPPOSITE of `decode_bypass` (Oracle-C): decode COMPUTE is FULLY
//! PRESERVED (every byte is really Huffman-decoded, once, via the clean path).
//! What is removed is only the MARKER/speculative premium — so the wall measures
//! gzippy's CLEAN per-chunk decode rate at T8 with the production publish-chain
//! INTACT (we do not touch the consumer publish / apply_window / ordering path;
//! the seeded window is only a clean-path fallback at the worker routing
//! decision when `window_map.get()` returned None).
//!
//! ## Capture (`GZIPPY_SEED_WINDOWS_CAPTURE=<file>`)
//! Run a normal decode and record aligned (start_bit → 32 KiB window) pairs at the
//! NATURAL clean-path worker decision (window present ⇒ the start_bit is a real
//! deflate boundary and the window is correctly aligned to it). A p=1 (sequential)
//! capture records EVERY non-zero chunk this way ⇒ a perfectly-aligned seed. At
//! `drive` end the accumulated pairs are written to `<file>`. (A WindowMap snapshot
//! is NOT used: its keys are partition GUESSES whose windows are misaligned — that
//! naive capture hit 0% / diverged; see the commit message.)
//!
//! ## Seed (`GZIPPY_SEED_WINDOWS=<file>`)
//! Load `<file>` at `drive` start. (1) Pre-seed the block_finder with the captured
//! aligned boundaries so every dispatch lands on a real boundary. (2) At the worker
//! routing decision (chunk_fetcher run_decode_task), when `window_map.get(start_bit)`
//! is None AND the side store has a window for `start_bit`, USE the seeded window
//! (clean path) instead of speculation. On a seed MISS, fall through to the normal
//! path (speculation) so output stays byte-correct. start_bit==0 unchanged.
//!
//! Output MUST be byte-identical to a normal decode (the seeded window is the
//! correct predecessor window). That is the correctness gate.
//!
//! ## Self-test counters
//! `seed_hits` / `seed_misses` (chunks that consulted the store and hit/missed)
//! are reported on the seeded run so the harness can prove the clean path was
//! actually forced (hit fraction → ~1, marker decode → ~0). Reported via
//! `report_seed_stats()` from `drive`.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

const MAGIC: &[u8; 8] = b"GZSEEDW2";

struct SeedData {
    /// encoded_offset_bits -> 32 KiB window bytes.
    windows: HashMap<usize, Vec<u8>>,
    /// Confirmed block start offsets (bits), sorted, from a prior real run.
    /// Parsed from the file for format completeness; the block_finder seed uses
    /// `seedable_chunk_starts()` (window keys) which is the same aligned set.
    #[allow(dead_code)]
    confirmed_offsets: Vec<usize>,
}

/// Side store loaded once.
static SEED_STORE: OnceLock<SeedData> = OnceLock::new();

static SEED_HITS: AtomicU64 = AtomicU64::new(0);
static SEED_MISSES: AtomicU64 = AtomicU64::new(0);

/// CAPTURE accumulator: (start_bit -> 32 KiB window bytes) recorded at the
/// worker decision when the NATURAL clean path is taken (window present, so the
/// start_bit is a real boundary and the window is correctly aligned to it).
/// A p=1 capture records every non-zero chunk this way → perfectly-aligned seed.
static CAPTURE_ACC: OnceLock<std::sync::Mutex<HashMap<usize, Vec<u8>>>> = OnceLock::new();

fn capture_acc() -> &'static std::sync::Mutex<HashMap<usize, Vec<u8>>> {
    CAPTURE_ACC.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Record an aligned (start_bit -> window) pair from the natural clean path.
/// No-op unless capture mode is on. `window` must be the 32 KiB predecessor
/// window actually used to clean-decode the chunk starting at `start_bit`.
pub fn record_clean_window(start_bit: usize, window: &[u8]) {
    if start_bit == 0 || !capture_enabled() || window.len() != 32768 {
        return;
    }
    capture_acc()
        .lock()
        .unwrap()
        .entry(start_bit)
        .or_insert_with(|| window.to_vec());
}

fn capture_path() -> Option<&'static str> {
    static P: OnceLock<Option<String>> = OnceLock::new();
    P.get_or_init(|| std::env::var("GZIPPY_SEED_WINDOWS_CAPTURE").ok())
        .as_deref()
}

fn seed_path() -> Option<&'static str> {
    static P: OnceLock<Option<String>> = OnceLock::new();
    P.get_or_init(|| std::env::var("GZIPPY_SEED_WINDOWS").ok())
        .as_deref()
}

/// DECOMPOSE knob (GZIPPY_SEED_NO_WINDOWS=1, measurement-only): suppress the
/// seeded-window fallback so `seed_window_for` always misses, forcing every chunk
/// onto the speculative/marker path even though the block_finder is pre-seeded with
/// the REAL boundaries. Isolates the boundary-ALIGNMENT sub-lever (chunks land on
/// real deflate boundaries but still pay the u16 marker decode). OFF==identity.
/// Byte-correct (the speculative path is the normal, correct production path).
pub fn seed_no_windows() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GZIPPY_SEED_NO_WINDOWS").is_ok_and(|v| v == "1"))
}

/// DECOMPOSE knob (GZIPPY_SEED_NO_BOUNDARIES=1, measurement-only): skip the
/// block_finder pre-seed so dispatch uses prod partition-GUESS boundaries while
/// windows ARE still seeded. Isolates the marker-COMPUTE sub-lever (correct windows
/// handed to chunks, but at partition-guess offsets so the head-of-line stalls /
/// re-decodes of the prod scheduling remain). OFF==identity.
pub fn seed_no_boundaries() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GZIPPY_SEED_NO_BOUNDARIES").is_ok_and(|v| v == "1"))
}

/// True if capture mode is active (record windows at drive end).
pub fn capture_enabled() -> bool {
    capture_path().is_some()
}

/// True if seed mode is active (force clean path from the side store).
pub fn seed_enabled() -> bool {
    seed_path().is_some()
}

/// Write the accumulated aligned (start_bit -> window) pairs to the capture
/// file. Called once from `drive` at the end of a normal decode. No-op if capture
/// off. The confirmed offsets stored are exactly the captured start_bits (real
/// aligned boundaries), used to pre-seed the block_finder on the seed run.
pub fn write_capture() {
    let Some(path) = capture_path() else { return };
    let acc = capture_acc().lock().unwrap();
    let mut windows: Vec<(usize, Vec<u8>)> = acc.iter().map(|(k, v)| (*k, v.clone())).collect();
    windows.sort_by_key(|(k, _)| *k);
    let confirmed_offsets: Vec<usize> = windows.iter().map(|(k, _)| *k).collect();
    let windows = &windows;
    let confirmed_offsets = &confirmed_offsets[..];
    match std::fs::File::create(path) {
        Ok(mut f) => {
            let mut buf: Vec<u8> = Vec::new();
            buf.extend_from_slice(MAGIC);
            buf.extend_from_slice(&(windows.len() as u64).to_le_bytes());
            for (off, bytes) in windows {
                buf.extend_from_slice(&(*off as u64).to_le_bytes());
                buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                buf.extend_from_slice(bytes);
            }
            buf.extend_from_slice(&(confirmed_offsets.len() as u64).to_le_bytes());
            for off in confirmed_offsets {
                buf.extend_from_slice(&(*off as u64).to_le_bytes());
            }
            if let Err(e) = f.write_all(&buf) {
                eprintln!("SEED_WINDOWS_CAPTURE write error: {e}");
            } else {
                eprintln!(
                    "SEED_WINDOWS_CAPTURE wrote {} windows + {} confirmed offsets to {} ({} bytes)",
                    windows.len(),
                    confirmed_offsets.len(),
                    path,
                    buf.len()
                );
            }
        }
        Err(e) => eprintln!("SEED_WINDOWS_CAPTURE open error: {e}"),
    }
}

fn empty_seed() -> SeedData {
    SeedData {
        windows: HashMap::new(),
        confirmed_offsets: Vec::new(),
    }
}

fn load_store() -> &'static SeedData {
    SEED_STORE.get_or_init(|| {
        let Some(path) = seed_path() else {
            return empty_seed();
        };
        let mut f = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("SEED_WINDOWS open error: {e}");
                return empty_seed();
            }
        };
        let mut buf = Vec::new();
        if let Err(e) = f.read_to_end(&mut buf) {
            eprintln!("SEED_WINDOWS read error: {e}");
            return empty_seed();
        }
        if buf.len() < 16 || &buf[0..8] != MAGIC {
            eprintln!("SEED_WINDOWS bad magic in {path}");
            return empty_seed();
        }
        let mut windows = HashMap::new();
        let n = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
        let mut p = 16usize;
        for _ in 0..n {
            if p + 16 > buf.len() {
                break;
            }
            let off = u64::from_le_bytes(buf[p..p + 8].try_into().unwrap()) as usize;
            let len = u64::from_le_bytes(buf[p + 8..p + 16].try_into().unwrap()) as usize;
            p += 16;
            if p + len > buf.len() {
                break;
            }
            windows.insert(off, buf[p..p + len].to_vec());
            p += len;
        }
        let mut confirmed_offsets = Vec::new();
        if p + 8 <= buf.len() {
            let m = u64::from_le_bytes(buf[p..p + 8].try_into().unwrap()) as usize;
            p += 8;
            for _ in 0..m {
                if p + 8 > buf.len() {
                    break;
                }
                confirmed_offsets
                    .push(u64::from_le_bytes(buf[p..p + 8].try_into().unwrap()) as usize);
                p += 8;
            }
        }
        eprintln!(
            "SEED_WINDOWS loaded {} windows + {} confirmed offsets from {}",
            windows.len(),
            confirmed_offsets.len(),
            path
        );
        SeedData {
            windows,
            confirmed_offsets,
        }
    })
}

/// Eagerly load the side store (call once from `drive` start in seed mode) so
/// the on-disk read does not show up in the first worker's decode time.
pub fn preload() {
    if seed_enabled() {
        let _ = load_store();
    }
}

/// The window-key offsets (sorted, deduped) — the ACTUAL per-chunk boundaries
/// that have a captured predecessor window, EXCLUDING 0. These are the offsets
/// the block_finder should be seeded with so every dispatched chunk start has a
/// matching seeded window (a clean decode is only possible where BOTH a real
/// boundary AND its window are known). Empty unless seed mode is on.
pub fn seedable_chunk_starts() -> Vec<usize> {
    if !seed_enabled() {
        return Vec::new();
    }
    let mut v: Vec<usize> = load_store()
        .windows
        .keys()
        .copied()
        .filter(|&k| k != 0)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Clean-path FALLBACK: if seed mode is on and the store has a window for
/// `start_bit`, return its bytes (and record a hit). Returns None on a miss
/// (caller falls through to speculation, keeping bytes correct). The caller
/// only consults this when `window_map.get(start_bit)` was None, so this never
/// shadows a real published window.
pub fn seed_window_for(start_bit: usize) -> Option<Vec<u8>> {
    if !seed_enabled() {
        return None;
    }
    // DECOMPOSE: seed-only-boundaries mode suppresses the window fallback so every
    // chunk takes the speculative/marker path at its (pre-seeded) real boundary.
    if seed_no_windows() {
        SEED_MISSES.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let store = load_store();
    match store.windows.get(&start_bit) {
        Some(b) => {
            SEED_HITS.fetch_add(1, Ordering::Relaxed);
            Some(b.clone())
        }
        None => {
            SEED_MISSES.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
}

/// Report seed hit/miss counters (the forced-clean self-test). No-op off seed.
pub fn report_seed_stats() {
    if !seed_enabled() {
        return;
    }
    let h = SEED_HITS.load(Ordering::Relaxed);
    let m = SEED_MISSES.load(Ordering::Relaxed);
    eprintln!("SEED_WINDOWS replay: hits={h} misses={m}");
}
