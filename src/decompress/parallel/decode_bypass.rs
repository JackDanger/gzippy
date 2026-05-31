#![cfg(parallel_sm)]

//! Decode-BYPASS experiment harness (2026-05-31, campaign instrument).
//!
//! Purpose: isolate gzippy's COORDINATION / orchestration overhead
//! (dispatch, in-order consumer, window publish, marker resolution,
//! buffer management, write, CRC combine) from the inner Huffman decode
//! CPU. The central unresolved campaign question is whether gzippy's
//! ~1.6× wall gap vs rapidgzip is DECODE (overlapped, sub-linear payoff
//! — all 8 isolated levers refuted) or COORDINATION (a lever independent
//! of decode).
//!
//! Mechanism: a two-pass capture/replay around the worker decode
//! primitive (`run_decode_task`).
//!
//!  - CAPTURE pass (`GZIPPY_BYPASS_CAPTURE=<file>`): run a NORMAL decode.
//!    Every decode-primitive result (a fully-built `ChunkData`) is
//!    recorded, keyed by `(start_bit, stop_hint_bit)` — the deterministic
//!    inputs to `decode_chunk_isal` / `speculative_decode_find_boundary`.
//!    At `drive` end, the map is serialized to `<file>`.
//!
//!  - REPLAY pass (`GZIPPY_BYPASS_DECODE=<file>`): the worker decode
//!    function, instead of running Huffman, looks up the precomputed
//!    `ChunkData` for its `(start_bit, stop_hint_bit)` and reconstructs
//!    it via a memcpy of the recorded `data_with_markers` / `data`
//!    segments + scalar metadata. Inner-Huffman CPU ≈ 0; the FULL
//!    coordination machinery downstream is unchanged. Output is
//!    byte-identical (the correctness gate).
//!
//! ## Why key on `(start_bit, stop_hint_bit)`
//!
//! Both decode primitives are DETERMINISTIC functions of
//! `(start_bit, stop_hint_bit)` — the decoded extent, the deflate-block
//! boundaries crossed, and the `data_with_markers` / `data` split are
//! fixed by the compressed input alone. Only WHICH primitive runs
//! (fast windowed `decode_chunk_isal` vs slow `speculative_*`, hence
//! whether the chunk carries markers) is timing-dependent. By keying on
//! the deterministic inputs and capturing whichever form the real run
//! produced, the replay faithfully reproduces BOTH form A (clean bytes,
//! windowed path) and form B (markers + tail, speculative path) in their
//! natural proportions — the most faithful "zero decode, full
//! coordination" experiment. See the module's design-review note in the
//! agent report.
//!
//! ## Form-A-only mode (`GZIPPY_BYPASS_FORCE_CLEAN=1`)
//!
//! Strips `data_with_markers` from every replayed chunk (folding it into
//! `data` as clean bytes — the capture stores a clean-resolved copy of
//! the marker prefix too). This bypasses the marker/apply_window/narrow
//! coordination, isolating the marker-coordination cost as the delta vs
//! the default (mixed A+B) replay.
//!
//! On a cache MISS (a `(start_bit, stop_hint_bit)` never captured),
//! replay returns `None` and the caller falls back to real decode so
//! output stays byte-correct.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Mutex, OnceLock};

use crate::decompress::parallel::chunk_data::{ChunkConfiguration, ChunkData, Footer, Subchunk};
use crate::decompress::parallel::crc32::CRC32Calculator;
use crate::decompress::parallel::rpmalloc_alloc::types;

/// Magic + version for the on-disk capture format.
const MAGIC: &[u8; 8] = b"GZBYPAS1";

/// One captured decode result, in a serialization-friendly shape.
/// Reconstructs a `ChunkData` via `to_chunk_data`.
#[derive(Clone)]
struct CapturedChunk {
    encoded_offset_bits: usize,
    max_acceptable_start_bit: usize,
    decode_origin_bit: usize,
    encoded_size_bits: usize,
    stopped_preemptively: bool,
    data_prefix_len: usize,
    /// Marker-tagged prefix (u16). Empty for clean/windowed chunks.
    data_with_markers: Vec<u16>,
    /// Clean byte suffix.
    data: Vec<u8>,
    /// (encoded_offset_bits, encoded_size_bits, decoded_offset, decoded_size)
    subchunks: Vec<(usize, usize, usize, usize)>,
    /// (crc32, uncompressed_size, end_bit_offset, decoded_end_offset)
    footers: Vec<(u32, u32, usize, usize)>,
}

impl CapturedChunk {
    fn from_chunk(c: &ChunkData) -> Self {
        CapturedChunk {
            encoded_offset_bits: c.encoded_offset_bits,
            max_acceptable_start_bit: c.max_acceptable_start_bit,
            decode_origin_bit: c.decode_origin_bit,
            encoded_size_bits: c.encoded_size_bits,
            stopped_preemptively: c.stopped_preemptively,
            data_prefix_len: c.data_prefix_len,
            data_with_markers: c.data_with_markers.clone(),
            data: c.data.as_slice().to_vec(),
            subchunks: c
                .subchunks
                .iter()
                .map(|s| {
                    (
                        s.encoded_offset_bits,
                        s.encoded_size_bits,
                        s.decoded_offset,
                        s.decoded_size,
                    )
                })
                .collect(),
            footers: c
                .footers
                .iter()
                .map(|f| {
                    (
                        f.crc32,
                        f.uncompressed_size,
                        f.end_bit_offset,
                        f.decoded_end_offset,
                    )
                })
                .collect(),
        }
    }

    /// Reconstruct a `ChunkData` whose downstream coordination behaves
    /// identically to a real decode result, but produced with ~zero
    /// inner-Huffman CPU (the segments are memcpy'd from the capture).
    ///
    /// CRC32 is RECOMPUTED here from the clean `data` bytes — this is
    /// genuine coordination work we WANT to keep (the real worker also
    /// CRCs `data` at append time), and it is not the inner-Huffman cost
    /// the experiment zeroes. The marker-prefix CRC (`narrowed_crc`) is
    /// computed later by `run_post_process_task`, exactly as in
    /// production.
    fn to_chunk_data(&self, configuration: ChunkConfiguration, force_clean: bool) -> ChunkData {
        // Decode whether force_clean can fold this chunk's whole marker
        // prefix into clean data. The captured marker prefix that survived
        // `clean_unmarked_data` ends in a real marker (>= MARKER_BASE) we
        // cannot resolve without the predecessor window, so folding is only
        // valid when the ENTIRE prefix is marker-free. In practice that is
        // rare; the meaningful A-vs-B delta comes from chunks captured CLEAN
        // (windowed path, empty data_with_markers). For form-B chunks with a
        // trailing marker, force_clean keeps them form B (no-op).
        use crate::decompress::parallel::replace_markers::MARKER_BASE;
        let fold = force_clean
            && !self.data_with_markers.is_empty()
            && !self.data_with_markers.iter().any(|&v| v >= MARKER_BASE);

        // `clean_lead`: marker-free prefix bytes folded to clean data
        // (force_clean fold case only). These ARE decoded output and must
        // be CRC'd. They are written before `self.data` to preserve output
        // order (output is markers | data).
        let mut data = types::u8_with_capacity(
            self.data.len()
                + if fold {
                    self.data_with_markers.len()
                } else {
                    0
                },
        );
        let data_with_markers: Vec<u16> = if fold {
            for &v in &self.data_with_markers {
                data.push(v as u8);
            }
            Vec::new()
        } else {
            self.data_with_markers.clone()
        };
        let folded_lead = data.len(); // clean bytes already pushed (== fold count)
        data.extend_from_slice(&self.data);

        // crc32s[0] covers the DECODED portion of `data` at decode-return
        // time — mirrors the post-finalize invariant (append_clean CRCs
        // data; clean_unmarked_data prepends the migrated-tail CRC). The
        // Option-A window-image prefix (`self.data[0..data_prefix_len]`) is
        // NOT CRC'd by the real decoder (not user output — the consumer
        // writes `data[prefix..]`), so we skip it. With folding, the layout
        // is: [folded_lead clean][prefix non-output][decoded] — CRC the
        // folded lead AND the post-prefix decoded bytes, skipping only the
        // prefix window image. For single-member silesia there is one stream.
        let mut crc0 = CRC32Calculator::new();
        if configuration.crc32_enabled {
            if folded_lead > 0 {
                crc0.update(&data[..folded_lead]);
            }
            let prefix_start = folded_lead;
            let prefix_end = (folded_lead + self.data_prefix_len).min(data.len());
            crc0.update(&data[prefix_end..]);
            let _ = prefix_start;
        }

        let subchunks: Vec<Subchunk> = self
            .subchunks
            .iter()
            .map(|&(eo, es, dofs, ds)| Subchunk {
                encoded_offset_bits: eo,
                encoded_size_bits: es,
                decoded_offset: dofs,
                decoded_size: ds,
                window: None,
            })
            .collect();
        let footers: Vec<Footer> = self
            .footers
            .iter()
            .map(|&(crc32, usz, eb, de)| Footer {
                crc32,
                uncompressed_size: usz,
                end_bit_offset: eb,
                decoded_end_offset: de,
            })
            .collect();

        ChunkData::from_bypass_parts(
            self.encoded_offset_bits,
            self.max_acceptable_start_bit,
            self.decode_origin_bit,
            self.encoded_size_bits,
            self.stopped_preemptively,
            self.data_prefix_len,
            data_with_markers,
            data,
            crc0,
            subchunks,
            footers,
            configuration,
        )
    }
}

// ── Mode detection (read once) ────────────────────────────────────────

fn capture_path() -> Option<&'static str> {
    static P: OnceLock<Option<String>> = OnceLock::new();
    P.get_or_init(|| std::env::var("GZIPPY_BYPASS_CAPTURE").ok())
        .as_deref()
}

fn replay_path() -> Option<&'static str> {
    static P: OnceLock<Option<String>> = OnceLock::new();
    P.get_or_init(|| std::env::var("GZIPPY_BYPASS_DECODE").ok())
        .as_deref()
}

fn force_clean() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var_os("GZIPPY_BYPASS_FORCE_CLEAN").is_some())
}

pub fn capture_enabled() -> bool {
    capture_path().is_some()
}

pub fn replay_enabled() -> bool {
    replay_path().is_some()
}

// ── Capture map (populated during a normal decode) ────────────────────

type Key = (usize, usize); // (start_bit, stop_hint_bit)

fn capture_map() -> &'static Mutex<HashMap<Key, CapturedChunk>> {
    static M: OnceLock<Mutex<HashMap<Key, CapturedChunk>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record a successfully-decoded chunk during the capture pass.
pub fn record(start_bit: usize, stop_hint_bit: usize, chunk: &ChunkData) {
    if !capture_enabled() {
        return;
    }
    let cap = CapturedChunk::from_chunk(chunk);
    capture_map()
        .lock()
        .unwrap()
        .insert((start_bit, stop_hint_bit), cap);
}

/// Serialize the capture map to the capture file. Called once at the end
/// of `drive`.
pub fn flush_capture() {
    let Some(path) = capture_path() else {
        return;
    };
    let map = capture_map().lock().unwrap();
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(MAGIC);
    write_usize(&mut buf, map.len());
    for ((sb, sh), c) in map.iter() {
        write_usize(&mut buf, *sb);
        write_usize(&mut buf, *sh);
        write_usize(&mut buf, c.encoded_offset_bits);
        write_usize(&mut buf, c.max_acceptable_start_bit);
        write_usize(&mut buf, c.decode_origin_bit);
        write_usize(&mut buf, c.encoded_size_bits);
        buf.push(c.stopped_preemptively as u8);
        write_usize(&mut buf, c.data_prefix_len);
        // data_with_markers
        write_usize(&mut buf, c.data_with_markers.len());
        for &v in &c.data_with_markers {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // data
        write_usize(&mut buf, c.data.len());
        buf.extend_from_slice(&c.data);
        // subchunks
        write_usize(&mut buf, c.subchunks.len());
        for &(a, b, d, e) in &c.subchunks {
            write_usize(&mut buf, a);
            write_usize(&mut buf, b);
            write_usize(&mut buf, d);
            write_usize(&mut buf, e);
        }
        // footers
        write_usize(&mut buf, c.footers.len());
        for &(crc, usz, eb, de) in &c.footers {
            buf.extend_from_slice(&crc.to_le_bytes());
            buf.extend_from_slice(&usz.to_le_bytes());
            write_usize(&mut buf, eb);
            write_usize(&mut buf, de);
        }
    }
    match std::fs::File::create(path) {
        Ok(mut f) => {
            let _ = f.write_all(&buf);
            eprintln!(
                "BYPASS_CAPTURE wrote {} chunks ({} bytes) to {}",
                map.len(),
                buf.len(),
                path
            );
        }
        Err(e) => eprintln!("BYPASS_CAPTURE failed to write {path}: {e}"),
    }
}

// ── Replay map (loaded lazily from the replay file) ───────────────────

fn replay_map() -> &'static HashMap<Key, CapturedChunk> {
    static M: OnceLock<HashMap<Key, CapturedChunk>> = OnceLock::new();
    M.get_or_init(|| {
        let Some(path) = replay_path() else {
            return HashMap::new();
        };
        let mut f = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("BYPASS_DECODE failed to open {path}: {e}");
                return HashMap::new();
            }
        };
        let mut bytes = Vec::new();
        if let Err(e) = f.read_to_end(&mut bytes) {
            eprintln!("BYPASS_DECODE failed to read {path}: {e}");
            return HashMap::new();
        }
        let map = parse_capture(&bytes);
        eprintln!("BYPASS_DECODE loaded {} chunks from {path}", map.len());
        map
    })
}

/// Replay: produce a `ChunkData` for `(start_bit, stop_hint_bit)` via
/// memcpy from the precomputed capture. Returns `None` on a cache miss
/// (caller must fall back to real decode for correctness).
pub fn replay(
    start_bit: usize,
    stop_hint_bit: usize,
    configuration: ChunkConfiguration,
) -> Option<ChunkData> {
    use std::sync::atomic::Ordering;
    // PREBUILT (default): each chunk's ChunkData is reconstructed ONCE at
    // load time and MOVED out on first request (take), so the per-call
    // replay cost is a HashMap lookup + Option::take — no fresh-Vec
    // allocation or memcpy on the worker, and no per-call page faults
    // (the buffers were faulted once at load). This is what isolates the
    // coordination floor: with decode AND reconstruction both off the
    // worker hot path, the wall is dominated by the in-order consumer
    // chain alone. Set GZIPPY_BYPASS_REBUILD=1 to use the per-call
    // reconstruction path instead (the confounded variant — measures
    // reconstruction allocation + faults too).
    if !rebuild_each_call() {
        let key = (start_bit, stop_hint_bit);
        let mut slot = prebuilt_map().lock().unwrap();
        if let Some(opt) = slot.get_mut(&key) {
            if let Some(chunk) = opt.take() {
                REPLAY_HITS.fetch_add(1, Ordering::Relaxed);
                return Some(chunk);
            }
        }
        REPLAY_MISSES.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    match replay_map().get(&(start_bit, stop_hint_bit)) {
        Some(c) => {
            REPLAY_HITS.fetch_add(1, Ordering::Relaxed);
            Some(c.to_chunk_data(configuration, force_clean()))
        }
        None => {
            REPLAY_MISSES.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
}

fn rebuild_each_call() -> bool {
    static R: OnceLock<bool> = OnceLock::new();
    *R.get_or_init(|| std::env::var_os("GZIPPY_BYPASS_REBUILD").is_some())
}

/// Prebuilt ChunkData per key, MOVED out on first request. Built once
/// from `replay_map()` at first access. Wrapped in a Mutex<Option<_>>
/// per slot so concurrent workers can take distinct chunks safely.
#[allow(clippy::type_complexity)]
fn prebuilt_map() -> &'static Mutex<HashMap<Key, Option<ChunkData>>> {
    static M: OnceLock<Mutex<HashMap<Key, Option<ChunkData>>>> = OnceLock::new();
    M.get_or_init(|| {
        // Use the production default configuration for reconstruction.
        let cfg = ChunkConfiguration::default();
        let fc = force_clean();
        let mut m = HashMap::new();
        for (k, c) in replay_map().iter() {
            m.insert(*k, Some(c.to_chunk_data(cfg, fc)));
        }
        Mutex::new(m)
    })
}

pub static REPLAY_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REPLAY_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Print replay hit/miss counters (called at drive end when replay on).
pub fn report_replay_stats() {
    use std::sync::atomic::Ordering;
    if !replay_enabled() {
        return;
    }
    eprintln!(
        "BYPASS_DECODE replay: hits={} misses={} (misses fall back to real decode)",
        REPLAY_HITS.load(Ordering::Relaxed),
        REPLAY_MISSES.load(Ordering::Relaxed),
    );
}

// ── Serialization helpers ─────────────────────────────────────────────

fn write_usize(buf: &mut Vec<u8>, v: usize) {
    buf.extend_from_slice(&(v as u64).to_le_bytes());
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn usize(&mut self) -> usize {
        let mut a = [0u8; 8];
        a.copy_from_slice(&self.b[self.pos..self.pos + 8]);
        self.pos += 8;
        u64::from_le_bytes(a) as usize
    }
    fn u32(&mut self) -> u32 {
        let mut a = [0u8; 4];
        a.copy_from_slice(&self.b[self.pos..self.pos + 4]);
        self.pos += 4;
        u32::from_le_bytes(a)
    }
    fn u8b(&mut self) -> u8 {
        let v = self.b[self.pos];
        self.pos += 1;
        v
    }
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        s
    }
}

fn parse_capture(bytes: &[u8]) -> HashMap<Key, CapturedChunk> {
    let mut map = HashMap::new();
    if bytes.len() < 8 || &bytes[..8] != MAGIC {
        eprintln!("BYPASS_DECODE: bad magic");
        return map;
    }
    let mut r = Reader { b: bytes, pos: 8 };
    let n = r.usize();
    for _ in 0..n {
        let sb = r.usize();
        let sh = r.usize();
        let encoded_offset_bits = r.usize();
        let max_acceptable_start_bit = r.usize();
        let decode_origin_bit = r.usize();
        let encoded_size_bits = r.usize();
        let stopped_preemptively = r.u8b() != 0;
        let data_prefix_len = r.usize();
        let dwm_len = r.usize();
        let mut data_with_markers = Vec::with_capacity(dwm_len);
        for _ in 0..dwm_len {
            let mut a = [0u8; 2];
            a.copy_from_slice(r.bytes(2));
            data_with_markers.push(u16::from_le_bytes(a));
        }
        let data_len = r.usize();
        let data = r.bytes(data_len).to_vec();
        let sc_n = r.usize();
        let mut subchunks = Vec::with_capacity(sc_n);
        for _ in 0..sc_n {
            subchunks.push((r.usize(), r.usize(), r.usize(), r.usize()));
        }
        let f_n = r.usize();
        let mut footers = Vec::with_capacity(f_n);
        for _ in 0..f_n {
            let crc = r.u32();
            let usz = r.u32();
            footers.push((crc, usz, r.usize(), r.usize()));
        }
        map.insert(
            (sb, sh),
            CapturedChunk {
                encoded_offset_bits,
                max_acceptable_start_bit,
                decode_origin_bit,
                encoded_size_bits,
                stopped_preemptively,
                data_prefix_len,
                data_with_markers,
                data,
                subchunks,
                footers,
            },
        );
    }
    map
}
