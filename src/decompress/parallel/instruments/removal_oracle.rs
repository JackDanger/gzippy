//! REMOVAL ORACLES for the contig clean loop (`Block::decode_clean_into_contig`)
//! — the campaign's Rule-3 instruments: a speed-up ceiling comes from REMOVING a
//! region and measuring, never from extrapolating a slow-down slope.
//!
//! Background (git history (campaign plan, removed), localization split 2026-06-11): the
//! frozen 9-arm T1 silesia perturbation found DECODE ≈ STORE (+576ms vs +526ms at
//! N=50 spin) — neither sub-region dominates, the gap is whole-loop. The next
//! pre-registered step is to size each side's REMOVAL ceiling. These two oracles
//! do that:
//!
//! ## 1. STORE-removal — `GZIPPY_ORACLE_NOSTORE=1`
//!
//! Decode everything normally (identical bit reads, identical Huffman table
//! lookups, identical dist decodes and range checks) but ELIDE the output
//! stores: the lit1/chain single-byte writes, the litpack packed-u64 write, and
//! the `emit_backref_contig` copy kernel (both its loads and stores — removing
//! the copy removes its dependent loads too; that is inherent to "remove the
//! store/copy half"). `*pos`/`emitted` advance identically so the loop
//! terminates at the same symbol and all length/CRC bookkeeping downstream runs
//! (over garbage bytes — same CPU cost). `baseline_wall − nostore_wall` is an
//! UPPER bound on what any store/copy optimization could ever recover.
//!
//! OUTPUT IS GARBAGE when this knob is ON: a loud banner is printed, CRC/ISIZE
//! verification is skipped (`sm_driver`), and writing to a REGULAR FILE is
//! refused — `gzippy -d -c file.gz > /dev/null` is the only sanctioned shape.
//!
//! ## 2. DECODE-removal — `GZIPPY_ORACLE_RECORD=<f>` / `GZIPPY_ORACLE_NODECODE=<f>`
//!
//! Replay-based, the honest shape (pre-registered): a RECORD pass runs a normal
//! decode and captures the contig loop's symbol stream per call — literal bytes
//! and (length, distance) pairs, plus the call's end bit-cursor state — keyed by
//! the call's deterministic entry state. The NODECODE pass replays the STORES
//! (single-byte literal writes + the production `emit_backref_contig` kernel)
//! WITHOUT Huffman decode, bit reads, or per-block LUT builds, then restores the
//! recorded bit-cursor end state so everything OUTSIDE the loop (header parses,
//! block transitions, CRC, coordination) runs genuinely. Output on a replay HIT
//! is BYTE-CORRECT (real literals + real copy positions), so the arm stays
//! sha-verifiable; a MISS falls back to real decode (counted, reported).
//!
//! Why not reuse `decode_bypass`: that instrument replays WHOLE CHUNKS as bulk
//! memcpy — it removes header parsing, LUT builds, AND the per-symbol store
//! stream along with the decode, so its delta is not comparable to the
//! STORE-removal arm and the shares-sum sanity check (store-share + decode-share
//! plus remainder ≈ whole) would be meaningless. This instrument removes ONLY
//! the decode half of the same region NOSTORE removes the store half of.
//!
//! Known approximation (documented, accepted): replay writes multi-literal
//! packets as single-byte stores while the production litpack arm uses one
//! packed-u64 store for up to 3 literals; the byte count is identical and the
//! lit1/chain single-byte arm dominates production literal traffic (~2.57
//! lits/iter via chained singles, P3.2).
//!
//! Both knobs are env-gated `OnceLock` reads (the `slow_knob`/`contig_prof`
//! pattern): OFF is a once-resolved bool behind a predictable branch at the few
//! hook sites; no production behavior changes when unset (proved by the
//! OFF-state sha grid).

// Instrument module: every item is reached only from the cfg-gated contig
// decode path (`decode_clean_into_contig` and the parallel-SM drive), so the
// default (empty-feature) lint build sees it as dead. Same rationale as the
// per-item allows in `slow_knob.rs`, applied module-wide because ALL entry
// points here are behind those cfgs.
#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

// ── STORE-removal knob ──────────────────────────────────────────────────────

/// `GZIPPY_ORACLE_NOSTORE=1` — elide the contig clean loop's output stores.
/// Read once; prints the loud banner on first resolution when ON.
#[inline]
pub fn nostore_enabled() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        let on = matches!(
            std::env::var("GZIPPY_ORACLE_NOSTORE").ok().as_deref(),
            Some("1")
        );
        if on {
            eprintln!(
                "\n████ ORACLE NOSTORE ACTIVE — OUTPUT BYTES ARE GARBAGE ████\n\
                 ████ measurement-only arm: contig-loop stores ELIDED;   ████\n\
                 ████ CRC/ISIZE verification SKIPPED; do NOT sha-verify; ████\n\
                 ████ regular-file output is REFUSED (use > /dev/null)   ████\n"
            );
        }
        on
    })
}

/// `GZIPPY_ORACLE_CRC_OFF=1` — Gate-2 CRC removal oracle. Sets the chunk
/// `CRC32Calculator` to disabled so `update()` early-returns (no `crc32fast`
/// folding work), while output bytes stay BYTE-CORRECT (only the checksum
/// accumulation is removed). Sizes CRC32's share of the T1 wall by AB against
/// the production-with-CRC arm. CRC verification is skipped in sm_driver when
/// this is on (the disabled calculator returns 0); ISIZE size verify still
/// runs, and the bytes are sha-verifiable (UNLIKE NOSTORE). Read once.
#[inline]
pub fn crc_off_enabled() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        let on = matches!(
            std::env::var("GZIPPY_ORACLE_CRC_OFF").ok().as_deref(),
            Some("1")
        );
        if on {
            eprintln!(
                "\n████ ORACLE CRC_OFF ACTIVE — CRC32 computation ELIDED ████\n\
                 ████ measurement-only arm: bytes are CORRECT (sha-ok);  ████\n\
                 ████ CRC verify SKIPPED (calculator disabled → 0).      ████\n"
            );
        }
        on
    })
}

// ── DECODE-removal record/replay ────────────────────────────────────────────

/// On-disk magic for the symbol-stream capture format.
const MAGIC: &[u8; 8] = b"GZSYMRP1";

/// Deterministic per-call key: (input slice len, absolute entry bit position,
/// 8-byte input fingerprint at the entry byte, output position at entry,
/// n_max_to_decode). All five are pure functions of decode state, so the same
/// call in a record run and a replay run produces the same key.
pub type Key = (u64, u64, u64, u64, u64);

#[inline]
pub fn call_key(data: &[u8], entry_bit: usize, out_pos: usize, n_max: usize) -> Key {
    let fp = if data.len() >= 8 {
        let i = (entry_bit / 8).min(data.len() - 8);
        u64::from_le_bytes(data[i..i + 8].try_into().unwrap())
    } else {
        0
    };
    (
        data.len() as u64,
        entry_bit as u64,
        fp,
        out_pos as u64,
        n_max as u64,
    )
}

fn record_path() -> Option<&'static str> {
    static P: OnceLock<Option<String>> = OnceLock::new();
    P.get_or_init(|| {
        let p = std::env::var("GZIPPY_ORACLE_RECORD").ok();
        if p.is_some() {
            assert!(
                !nostore_enabled(),
                "GZIPPY_ORACLE_RECORD requires a CORRECT decode; \
                 it cannot be combined with GZIPPY_ORACLE_NOSTORE"
            );
            eprintln!(
                "\n████ ORACLE RECORD ACTIVE — capturing contig symbol stream ████\n\
                 ████ (timing of this run is NOT a measurement)              ████\n"
            );
        }
        p
    })
    .as_deref()
}

fn replay_path() -> Option<&'static str> {
    static P: OnceLock<Option<String>> = OnceLock::new();
    P.get_or_init(|| {
        let p = std::env::var("GZIPPY_ORACLE_NODECODE").ok();
        if p.is_some() {
            eprintln!(
                "\n████ ORACLE NODECODE ACTIVE — contig Huffman decode REPLACED ████\n\
                 ████ by symbol-stream replay (measurement-only arm; output    ████\n\
                 ████ is byte-correct on replay hits, misses run real decode)  ████\n"
            );
        }
        p
    })
    .as_deref()
}

#[inline]
pub fn record_enabled() -> bool {
    record_path().is_some()
}

#[inline]
pub fn replay_enabled() -> bool {
    replay_path().is_some()
}

/// One op of the symbol stream: `lit_run` literal bytes (from the shared lit
/// byte stream), then — unless this is the trailing op — one back-reference of
/// (`len`, `dist`). `len == 0 && dist == 0` marks a trailing literal-only op.
/// Packed `lit_run | len<<32 | dist<<48` for a compact 8-byte record.
#[derive(Clone, Copy)]
pub struct Op(pub u64);

impl Op {
    #[inline]
    fn new(lit_run: u32, len: u16, dist: u16) -> Self {
        Op((lit_run as u64) | ((len as u64) << 32) | ((dist as u64) << 48))
    }
    #[inline]
    pub fn lit_run(self) -> usize {
        (self.0 & 0xFFFF_FFFF) as usize
    }
    #[inline]
    pub fn match_len(self) -> usize {
        ((self.0 >> 32) & 0xFFFF) as usize
    }
    #[inline]
    pub fn dist(self) -> usize {
        (self.0 >> 48) as usize
    }
}

/// Recorded end state + symbol stream of one `decode_clean_into_contig` call.
pub struct CallRec {
    pub end_pos: u64,
    pub end_bitbuf: u64,
    pub end_bitsleft: u32,
    pub emitted: u64,
    pub at_eob: bool,
    pub ops: Vec<Op>,
    pub lits: Vec<u8>,
}

/// Per-call recorder, held as a function-local in `decode_clean_into_contig`
/// (None when recording is off — one predictable branch per hook site).
pub struct Recorder {
    key: Key,
    pending_lits: u32,
    ops: Vec<Op>,
    lits: Vec<u8>,
}

impl Recorder {
    #[inline]
    pub fn lit(&mut self, b: u8) {
        self.lits.push(b);
        self.pending_lits += 1;
    }
    /// Record `n` packed literals from an ISA-L multi-symbol packet (byte i =
    /// `(sym >> 8*i) & 0xFF`, low-to-high == output order).
    #[inline]
    pub fn lits_from_packed(&mut self, sym: u32, n: usize) {
        for i in 0..n {
            self.lit((sym >> (8 * i)) as u8);
        }
    }
    #[inline]
    pub fn backref(&mut self, len: usize, dist: usize) {
        debug_assert!(len <= 258 && dist <= 32768);
        self.ops
            .push(Op::new(self.pending_lits, len as u16, dist as u16));
        self.pending_lits = 0;
    }
}

/// Begin recording one call. Returns `None` when recording is off.
#[inline]
pub fn record_begin(
    data: &[u8],
    entry_bit: usize,
    out_pos: usize,
    n_max: usize,
) -> Option<Recorder> {
    if !record_enabled() {
        return None;
    }
    Some(Recorder {
        key: call_key(data, entry_bit, out_pos, n_max),
        pending_lits: 0,
        ops: Vec::new(),
        lits: Vec::new(),
    })
}

#[allow(clippy::type_complexity)]
fn record_map() -> &'static Mutex<HashMap<Key, CallRec>> {
    static M: OnceLock<Mutex<HashMap<Key, CallRec>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Finish recording one call (Ok returns only — an erroring call is simply not
/// recorded; the replay run will miss it and fall back to real decode, which
/// reproduces the error path genuinely).
pub fn record_end(
    rec: Option<Recorder>,
    ok: bool,
    end_pos: usize,
    end_bitbuf: u64,
    end_bitsleft: u32,
    emitted: usize,
    at_eob: bool,
) {
    let Some(mut rec) = rec else { return };
    if !ok {
        return;
    }
    if rec.pending_lits > 0 {
        let p = rec.pending_lits;
        rec.ops.push(Op::new(p, 0, 0));
        rec.pending_lits = 0;
    }
    let call = CallRec {
        end_pos: end_pos as u64,
        end_bitbuf,
        end_bitsleft,
        emitted: emitted as u64,
        at_eob,
        ops: rec.ops,
        lits: rec.lits,
    };
    record_map().lock().unwrap().insert(rec.key, call);
}

/// Serialize the record map to `GZIPPY_ORACLE_RECORD`. Called at drive end
/// (next to `decode_bypass::flush_capture`).
pub fn flush_record() {
    let Some(path) = record_path() else { return };
    let map = record_map().lock().unwrap();
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(MAGIC);
    w64(&mut buf, map.len() as u64);
    for (k, c) in map.iter() {
        for v in [k.0, k.1, k.2, k.3, k.4] {
            w64(&mut buf, v);
        }
        w64(&mut buf, c.end_pos);
        w64(&mut buf, c.end_bitbuf);
        buf.extend_from_slice(&c.end_bitsleft.to_le_bytes());
        w64(&mut buf, c.emitted);
        buf.push(c.at_eob as u8);
        w64(&mut buf, c.ops.len() as u64);
        for op in &c.ops {
            w64(&mut buf, op.0);
        }
        w64(&mut buf, c.lits.len() as u64);
        buf.extend_from_slice(&c.lits);
    }
    match std::fs::File::create(path) {
        Ok(mut f) => {
            let r = f.write_all(&buf);
            eprintln!(
                "ORACLE_RECORD wrote {} calls ({} bytes) to {}{}",
                map.len(),
                buf.len(),
                path,
                if r.is_err() { " [WRITE FAILED]" } else { "" }
            );
        }
        Err(e) => eprintln!("ORACLE_RECORD failed to create {path}: {e}"),
    }
}

fn replay_map() -> &'static HashMap<Key, CallRec> {
    static M: OnceLock<HashMap<Key, CallRec>> = OnceLock::new();
    M.get_or_init(|| {
        let Some(path) = replay_path() else {
            return HashMap::new();
        };
        let mut bytes = Vec::new();
        match std::fs::File::open(path) {
            Ok(mut f) => {
                if let Err(e) = f.read_to_end(&mut bytes) {
                    eprintln!("ORACLE_NODECODE failed to read {path}: {e}");
                    return HashMap::new();
                }
            }
            Err(e) => {
                eprintln!("ORACLE_NODECODE failed to open {path}: {e}");
                return HashMap::new();
            }
        }
        let map = parse_capture(&bytes);
        eprintln!("ORACLE_NODECODE loaded {} calls from {path}", map.len());
        map
    })
}

/// Eagerly load + parse the replay map BEFORE the timed drive and report the
/// duration, so the one-time load is identifiable out-of-wall (same convention
/// as `decode_bypass::warm_prebuilt`). No-op unless replay is on.
pub fn warm_replay() {
    if !replay_enabled() {
        return;
    }
    let t0 = std::time::Instant::now();
    let n = replay_map().len();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("ORACLE_NODECODE warm_replay: loaded {n} calls in {ms:.1}ms (out-of-wall)");
}

pub static REPLAY_HITS: AtomicU64 = AtomicU64::new(0);
pub static REPLAY_MISSES: AtomicU64 = AtomicU64::new(0);

/// Look up the recorded call for `key`. Counts hits/misses. The map is
/// immutable after init, so lookups are lock-free shared reads.
#[inline]
pub fn replay_lookup(key: &Key) -> Option<&'static CallRec> {
    match replay_map().get(key) {
        Some(c) => {
            REPLAY_HITS.fetch_add(1, Ordering::Relaxed);
            Some(c)
        }
        None => {
            REPLAY_MISSES.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
}

/// Count a replay bail (room-guard refused the recorded call → real decode ran).
pub fn count_replay_bail() {
    REPLAY_MISSES.fetch_add(1, Ordering::Relaxed);
    REPLAY_HITS.fetch_sub(1, Ordering::Relaxed);
}

/// Print replay hit/miss counters at drive end. A NODECODE wall with a
/// non-trivial miss share UNDER-removes decode (misses run the real loop) —
/// the harness must read these.
pub fn report_replay_stats() {
    if !replay_enabled() {
        return;
    }
    eprintln!(
        "ORACLE_NODECODE replay: hits={} misses={} (misses ran the REAL contig decode)",
        REPLAY_HITS.load(Ordering::Relaxed),
        REPLAY_MISSES.load(Ordering::Relaxed),
    );
}

// ── serialization helpers ───────────────────────────────────────────────────

#[inline]
fn w64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

struct Rd<'a> {
    b: &'a [u8],
    pos: usize,
    overran: bool,
}

impl Rd<'_> {
    fn have(&self, n: usize) -> bool {
        self.pos + n <= self.b.len()
    }
    fn u64v(&mut self) -> u64 {
        if !self.have(8) {
            self.overran = true;
            return 0;
        }
        let v = u64::from_le_bytes(self.b[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        v
    }
    fn u32v(&mut self) -> u32 {
        if !self.have(4) {
            self.overran = true;
            return 0;
        }
        let v = u32::from_le_bytes(self.b[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn u8v(&mut self) -> u8 {
        if !self.have(1) {
            self.overran = true;
            return 0;
        }
        let v = self.b[self.pos];
        self.pos += 1;
        v
    }
    fn bytes(&mut self, n: usize) -> &[u8] {
        if !self.have(n) {
            self.overran = true;
            return &[];
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        s
    }
}

fn parse_capture(bytes: &[u8]) -> HashMap<Key, CallRec> {
    let mut map = HashMap::new();
    if bytes.len() < 8 || &bytes[..8] != MAGIC {
        eprintln!("ORACLE_NODECODE: bad magic in capture file");
        return map;
    }
    let mut r = Rd {
        b: bytes,
        pos: 8,
        overran: false,
    };
    let n = r.u64v();
    for _ in 0..n {
        let key = (r.u64v(), r.u64v(), r.u64v(), r.u64v(), r.u64v());
        let end_pos = r.u64v();
        let end_bitbuf = r.u64v();
        let end_bitsleft = r.u32v();
        let emitted = r.u64v();
        let at_eob = r.u8v() != 0;
        let n_ops = r.u64v() as usize;
        if r.overran || !r.have(n_ops.saturating_mul(8)) {
            eprintln!(
                "ORACLE_NODECODE: capture truncated/corrupt, parsed {} calls",
                map.len()
            );
            break;
        }
        let mut ops = Vec::with_capacity(n_ops);
        for _ in 0..n_ops {
            ops.push(Op(r.u64v()));
        }
        let n_lits = r.u64v() as usize;
        let lits = r.bytes(n_lits).to_vec();
        if r.overran {
            eprintln!(
                "ORACLE_NODECODE: capture truncated/corrupt, parsed {} calls",
                map.len()
            );
            break;
        }
        map.insert(
            key,
            CallRec {
                end_pos,
                end_bitbuf,
                end_bitsleft,
                emitted,
                at_eob,
                ops,
                lits,
            },
        );
    }
    map
}
