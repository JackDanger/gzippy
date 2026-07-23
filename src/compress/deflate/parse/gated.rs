//! DETECTOR-GATED LAZY-L3 (`l3-tune` feature; NOT a production strategy in
//! the default build — see `level.rs`'s level-3 arm).
//!
//! Mission: plain lazy-at-L3 (`2c7f9444`, `Strategy::Lazy` under `l3-tune`,
//! L3's knobs unchanged) is 19/21 strictly smaller than the shipped Greedy
//! default on the `~/www/gzippy-bench/corpus` 21-file breadth set, but
//! regresses `ecoli.fastq` (+0.3146%) and `weights.safetensors` (+0.0378%)
//! deterministically — the strict-Pareto re-gate (`992c5837`) FAILED leg (a)
//! on exactly those two files, cross-arch-replicated byte-for-byte. This
//! composes the SAME content-detector pattern that already shipped for L1
//! (`fast.rs`'s HASH3-GATE: literal fraction read for free off the
//! just-finished block's `Sink::litlen_freqs`, one-block lag, per-block
//! granularity — see `Hash3Cfg`'s doc comment) to gate the GREEDY-vs-LAZY
//! choice itself, per block, instead of running Lazy unconditionally for the
//! whole file.
//!
//! TWO-SIDED gate (unlike HASH3-GATE's one-sided `>= threshold`): the
//! mission's own framing (confirmed by the per-file breadth data) is that
//! `ecoli.fastq` (4-symbol DNA, extremely high match density — matches
//! dominate, so LOW literal fraction) and `weights.safetensors`
//! (near-incompressible floats — literals dominate, so HIGH literal
//! fraction) sit at OPPOSITE ends of the literal-fraction axis. A block is
//! therefore routed to GREEDY when the PRECEDING block's literal fraction is
//! `<= low_threshold_pct` (the ecoli-class signal) OR `>= high_threshold_pct`
//! (the weights-class signal); everything in the middle (the 19-file band)
//! stays LAZY, matching plain lazy's already-measured win there.
//!
//! Detection/emission granularity is a SEPARATE, independently-tunable knob
//! from libdeflate's own adaptive block-split (`parse::mod`'s
//! `SOFT_MAX_BLOCK_LENGTH` / `choose_max_block_end`, shared by plain
//! greedy/lazy/L5-9) — [`tune::L3GateTune::block_len`]. The mission's own
//! framing (mirroring the L1 HASH3-GATE precedent's 64KB block unit)
//! motivated trying something FINER than 300KB, on the theory that a
//! wrong-initial-guess "lag" block should be bounded to a small fraction of
//! a multi-MB file. MEASURED (2026-07-23 sweep) that theory does not hold
//! here: a 64KB unit correctly classifies both named files' content but
//! pays a real per-block Huffman-header SELF-TAX (many more, smaller blocks
//! than plain greedy/lazy's own adaptive split) that on its own made even a
//! 100%-correctly-classified `ecoli.fastq` slightly LARGER than the Greedy
//! baseline. Both named files are HOMOGENEOUS throughout (DNA / near-
//! incompressible floats), so with `initial_lazy=false` the very first
//! gate-block ALREADY guesses correctly regardless of block size — the
//! fine-granularity concern only bites on a MID-FILE content transition,
//! which neither named regression file has. Matching `parse::mod`'s own
//! `SOFT_MAX_BLOCK_LENGTH` (300_000, see [`L3_GATE_BLOCK_LEN`]) closes the
//! self-tax entirely while remaining fine-grained enough to lock onto the
//! correct strategy within 1-2 blocks (~87 gate-blocks on a 26MB file, per
//! the real gated trace in the commit that introduced this module) — bigger
//! gate-blocks WON here, an unintuitive result the sweep caught that a
//! straight port of the L1 precedent's block size would have missed.
//!
//! [`greedy::run_block`]/[`lazy::run_block`] (extracted, byte-identical logic
//! to `greedy::run`/`lazy::run`'s own inner block loop — pure code motion,
//! see those functions' doc comments) share ONE `HcMatchfinder`/`in_base`/
//! `next_hashes` across the WHOLE file (or gate-block-span), so switching
//! strategy between blocks never resets or duplicates matchfinder state —
//! similar in spirit to `fast.rs`'s CONTENT-ADAPTIVE CHAIN MATCHING lever's
//! `chain_hc`/`hc_catchup` composition, but simpler here because BOTH
//! strategies already share the IDENTICAL `HcMatchfinder` type end-to-end
//! (no `hc_catchup`-style resync needed — greedy and lazy at L3 already use
//! the same finder, unlike L1's chainless-vs-chain composition).

use super::super::bitstream::BitWriter;
use super::super::block_split::MIN_BLOCK_LENGTH;
use super::super::level::LevelParams;
use super::super::matchfinder::hc::HcMatchfinder;
use super::super::tables::DEFLATE_FIRST_LEN_SYM;
use super::{emit_block, greedy, lazy, Sink, StaticCodes, NUM_LITERALS};

/// Runtime-tunable knobs for the sweep (`l3-tune` feature, OFF by default).
/// Mirrors `fast::tune::L1Tune`'s env-var-overridable pattern so a search
/// driver can call [`set`] between candidates within one process (no rebuild
/// per candidate).
#[cfg(feature = "l3-tune")]
pub mod tune {
    use std::sync::{OnceLock, RwLock};

    #[derive(Clone, Copy, Debug)]
    pub struct L3GateTune {
        /// Master switch. `false` makes [`super::run`] behave EXACTLY like
        /// unconditional `Strategy::Lazy` (every block LAZY, from the first
        /// byte) — the control arm that reproduces the FAILED plain-lazy
        /// baseline (`992c5837`) through the SAME entry point, for
        /// apples-to-apples sweep A/Bs against the gated candidates.
        pub enabled: bool,
        /// Literal-fraction PERCENT (0-100): a block whose PRECEDING block's
        /// literal fraction is `<= low_threshold_pct` gates to GREEDY (the
        /// ecoli-class signal: match-dominated content).
        pub low_threshold_pct: u32,
        /// Literal-fraction PERCENT (0-100): a block whose PRECEDING block's
        /// literal fraction is `>= high_threshold_pct` gates to GREEDY (the
        /// weights-class signal: literal-dominated content).
        pub high_threshold_pct: u32,
        /// Starting state (before the first gate-block has produced a
        /// signal): `true` starts LAZY (the majority-class default — 19/21
        /// breadth files win under unconditional lazy); `false` starts
        /// GREEDY (the conservative choice: zero extra cost on the two named
        /// regressions from their very first byte, at the price of diluting
        /// the lazy win on the very first gate-block of the OTHER 19 files).
        /// See this module's top doc comment for the T>1 per-chunk framing
        /// this knob mirrors from `Hash3Cfg::gate_initial_active` — that
        /// lesson reversed T1's "start silent" call because `run()` restarts
        /// once per parallel-encode CHUNK, not once per file, so a
        /// wrong-guess cost paid at the top of every chunk can dominate the
        /// aggregate at T>1. Whichever direction is wrong here, the same
        /// mechanism applies; measure both under T1 AND T>1 before picking.
        pub initial_lazy: bool,
        /// Detection/emission granularity in bytes for THIS gated path only
        /// — independent of `parse::mod`'s `SOFT_MAX_BLOCK_LENGTH` (~300KB)
        /// used by plain greedy/lazy/L5-9. Smaller values respond to content
        /// changes faster (less lag-cost on a wrong initial guess) at the
        /// price of more per-block Huffman-header overhead (self-tax) — a
        /// real sweep axis, not assumed a priori.
        pub block_len: usize,
    }

    fn env_u32(name: &str, default: u32) -> u32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
    fn env_bool(name: &str, default: bool) -> bool {
        std::env::var(name)
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(default)
    }

    impl L3GateTune {
        fn from_env() -> Self {
            L3GateTune {
                enabled: env_bool("GZIPPY_L3TUNE_GATE_ENABLED", super::L3_GATE_ENABLED),
                low_threshold_pct: env_u32(
                    "GZIPPY_L3TUNE_GATE_LOW_PCT",
                    super::L3_GATE_LOW_THRESHOLD_PCT,
                ),
                high_threshold_pct: env_u32(
                    "GZIPPY_L3TUNE_GATE_HIGH_PCT",
                    super::L3_GATE_HIGH_THRESHOLD_PCT,
                ),
                initial_lazy: env_bool(
                    "GZIPPY_L3TUNE_GATE_INITIAL_LAZY",
                    super::L3_GATE_INITIAL_LAZY,
                ),
                block_len: env_usize("GZIPPY_L3TUNE_GATE_BLOCK_LEN", super::L3_GATE_BLOCK_LEN),
            }
        }
    }

    fn cell() -> &'static RwLock<L3GateTune> {
        static CELL: OnceLock<RwLock<L3GateTune>> = OnceLock::new();
        CELL.get_or_init(|| RwLock::new(L3GateTune::from_env()))
    }

    /// Current tune parameters (env-var defaults unless overridden by
    /// [`set`] in this process).
    pub fn get() -> L3GateTune {
        *cell().read().unwrap()
    }

    /// Override the tune parameters for every subsequent [`super::run`] call
    /// in THIS process — the search-only API an in-process sweep driver
    /// would use to try many candidates without a rebuild or respawn per
    /// candidate (mirrors `fast::tune::set`). Not called anywhere in this
    /// crate yet (this session's sweep drove candidates via the
    /// `GZIPPY_L3TUNE_GATE_*` env vars / subprocess-per-candidate instead —
    /// see `mod.rs`'s doc comment on `gated`) — kept for the next search
    /// session (an in-process fulcrum driver) rather than deleted.
    #[allow(dead_code)]
    pub fn set(t: L3GateTune) {
        *cell().write().unwrap() = t;
    }
}

/// Ship-candidate defaults for the DETECTOR-GATED LAZY-L3 composition
/// (2026-07-23 mission). Consumed directly by [`GateCfg::shipped`] (the
/// `not(l3-tune)` arm — unreachable in practice, since `level.rs` only ever
/// selects `Strategy::LazyGated` under `l3-tune`, but must still typecheck)
/// and as the env-var defaults for `tune::L3GateTune::from_env` (the
/// `l3-tune` arm actually exercised by the sweep).
///
/// SESSION VERDICT (2026-07-23) — NOT PROMOTED, wall leg FAILS: with these
/// exact defaults, the SIZE leg the mission set as the promotion gate CLEARS
/// for the first time in this campaign — real end-to-end compressed-size
/// comparisons (`GZIPPY_DEBUG`-free, `cargo build --release --features
/// l3-tune`) on the full `~/www/gzippy-bench/corpus` 21-file breadth set are
/// 21/21 strictly smaller than the shipped Greedy default, at T1, T4, AND
/// T16 (both `ecoli.fastq` and `weights.safetensors`, the two files that
/// blocked plain lazy's own strict-Pareto re-gate at `992c5837`, are FIXED:
/// 0.999922x/0.999987x vs Greedy at T1) — roundtrip-verified byte-exact on
/// every corpus file, L2/L4-12 confirmed byte-identical to the Greedy-only
/// baseline. The mission's next leg (`M1 N>=21` interleaved `hyperfine`,
/// `/dev/null` sink, `-p1`, `--warmup 2`, this exact shipped config, no env
/// overrides) does NOT clear: several middle-band (stays-LAZY) files exceed
/// the pre-registered `self-tax <= +10%` bar vs the Greedy `rebuild control`
/// — `aozora.txt` 1.25x (+25%), `dd79_bin6` 1.13x (+13%), `dickens` 1.11x
/// (+11%) — while the two gated-to-GREEDY files stay near-zero-tax as
/// designed (`ecoli.fastq` 1.00x, `weights.safetensors` 1.01x,
/// `data.csv` 1.01x). A control measurement (gate DISABLED, forcing pure
/// unconditional Lazy from block 0, `GZIPPY_L3TUNE_GATE_ENABLED=0`)
/// isolates the cause: `aozora.txt`/`dd79_bin6` come out statistically
/// IDENTICAL to the gated-shipped numbers (1.00x either way) — the
/// detector/dispatch machinery itself adds no measurable overhead; the tax
/// is `Strategy::Lazy`'s OWN inherent per-position lookahead-probe cost vs
/// Greedy, already present (and presumably already accepted, since L5-7 pay
/// it too) wherever Lazy runs, not something this composition introduces or
/// could fix by re-tuning its own thresholds. (`dickens` is the one
/// exception where the gate itself measurably HELPS wall — 1.18x FASTER
/// than pure-lazy — because real English text has enough low/high-`lit_pct`
/// stretches to trip the detector into GREEDY occasionally even though it
/// is a "stays LAZY" file on the size ledger; still not enough to clear the
/// bar vs Greedy.) Per the mission's own pre-registered escape hatch ("Any
/// leg fails -> keep default, record"), the expensive frozen-solvency gate
/// was correctly NOT run — this is a LOCAL M1 result only, not a Gate-1
/// N>=7-both-arches finding, but the size showing SO cleanly and the wall
/// failing SO clearly (25% >> 10%, on the Mac's own numbers, no cross-arch
/// replication needed to see a 2.5x-over-budget miss) makes running the box
/// leg pointless. Re-open trigger: a genuinely CHEAPER lazy-class parser
/// (or a wall-cost-aware gate that ALSO demotes expensive-but-marginal-win
/// blocks back to Greedy) would need to exist before this composition's
/// wall leg could pass — a structurally different, larger effort than a
/// threshold re-sweep. `L3_GATE_ENABLED` stays `true` (this IS the
/// interesting, not-obviously-dead configuration) but `level.rs`'s L3 arm
/// keeps `l3-tune` default-OFF, matching `992c5837`'s and `2c7f9444`'s own
/// disposition — a documented, correct, default-off experiment, not a
/// production default.
pub(super) const L3_GATE_ENABLED: bool = true;
/// Literal-fraction PERCENT threshold below which a block gates to GREEDY
/// (the ecoli-class, match-dominated signal).
///
/// DETECTOR-DESIGN NOTE: the initial signal-separation pass (step 1 of the
/// mission) explored TWO candidate signals off the free `Sink::litlen_freqs`
/// histogram — raw literal fraction (`lit_pct`, what ships) and the number
/// of DISTINCT literal symbol VALUES used in a block (a DNA-alphabet
/// detector: `ecoli.fastq` uses only ~15-18 distinct byte values per block
/// vs 40-256+ for every other breadth file). Distinct-symbol-count is the
/// CLEANER raw separator in isolation, but it is not what ships: `lit_pct`
/// alone turned out sufficient once combined with `initial_lazy=false`,
/// because of a self-reinforcing asymmetry this composition exploits rather
/// than fights — on `ecoli.fastq`, a block actually CODED WITH GREEDY comes
/// out at `lit_pct` ~16-22% (measured on the real gated trace, not the
/// uniform-lazy diagnostic dump), well below every other breadth file's
/// typical band, while a block coded with LAZY on the SAME bytes reads
/// ~29-40% (matching plain-lazy's own already-measured regression numbers).
/// So the very first correctly-greedy-coded block's LOW `lit_pct` is what
/// keeps the gate locked onto GREEDY for the rest of the file — the
/// literal-fraction signal is read AFTER the parse decision it also gates,
/// which is why this threshold (34) sits comfortably below EVERY other
/// breadth file's typical greedy-OR-lazy band while still catching
/// greedy-coded ecoli. MEASURED (2026-07-23 sweep, `~/www/gzippy-bench/
/// corpus` 21-file breadth, real end-to-end compressed-size comparisons,
/// not the diagnostic dump): 30 is the empirical cliff (`lit_pct<=30` misses
/// enough of the file's early transition to leave one whole block on the
/// wrong side, regressing 1.00005x vs Greedy); 31-38 all pass strict Pareto,
/// with the 21-file total ratio monotonically WORSENING above 34 (more of
/// `data.csv`'s genuine lazy win gets pulled into greedy for no benefit
/// once the threshold climbs past the content it needs to separate). `34`
/// sits 4 points clear of the empirical cliff (the same margin-not-
/// knife-edge lesson HASH3-GATE's `threshold=48`/cliff-at-50 re-check
/// taught) while giving up only ~0.035 aggregate percentage points vs the
/// knife-edge-optimal 31.
pub(super) const L3_GATE_LOW_THRESHOLD_PCT: u32 = 34;
/// Literal-fraction PERCENT threshold above which a block gates to GREEDY
/// (the weights-class, literal-dominated signal). MEASURED (2026-07-23
/// sweep): a WIDE plateau (90-97 all identical to the 5th decimal on the
/// 21-file breadth total) — unlike the low threshold, not a knife-edge in
/// either direction. `95` sits in the middle of the measured-flat range.
pub(super) const L3_GATE_HIGH_THRESHOLD_PCT: u32 = 95;
/// Starting state before the first gate-block has a signal — see
/// `tune::L3GateTune::initial_lazy`'s doc comment.
pub(super) const L3_GATE_INITIAL_LAZY: bool = false;
/// Detection/emission block length in bytes. MEASURED (2026-07-23 sweep,
/// `~/www/gzippy-bench/corpus` 21-file breadth): a 64KB unit (mirroring the
/// L1 HASH3-GATE precedent) correctly classifies every ecoli.fastq/
/// weights.safetensors block as GREEDY but pays a real per-block Huffman-
/// header SELF-TAX (more, smaller blocks than plain greedy/lazy's own
/// ~300KB adaptive split) that made even a 100%-correctly-classified
/// ecoli.fastq slightly LARGER than the Greedy baseline at T1 (ratio
/// 1.000036, block_len=262144). Matching `parse::mod`'s own
/// `SOFT_MAX_BLOCK_LENGTH` (300_000) closes that residual entirely (T1
/// ecoli.fastq ratio 0.999922) WITHOUT losing detector correctness: both
/// named files are homogeneous throughout (DNA / near-incompressible
/// floats), so `initial_lazy=false`'s first-block guess is already correct
/// from byte 0 regardless of block size — the fine-granularity concern the
/// L1 precedent motivated only matters for a MID-FILE content transition,
/// which neither named regression file has. `300_000` is used here as a
/// literal, not a re-export of `SOFT_MAX_BLOCK_LENGTH` (private to
/// `parse::mod`, not worth widening visibility for one shared constant);
/// [`gate_block_end`] otherwise mirrors `parse::mod`'s `choose_max_block_end`
/// exactly (down to the `MIN_BLOCK_LENGTH` tiny-trailing-block guard).
pub(super) const L3_GATE_BLOCK_LEN: usize = 300_000;

/// Resolved gate config for one [`run`] call — unifies the `l3-tune`
/// (env/search-tunable) and `not(l3-tune)` (const-bound, unreachable in
/// practice) build flavors into one small `Copy` struct, mirroring
/// `fast::Hash3Cfg`.
#[derive(Clone, Copy)]
struct GateCfg {
    enabled: bool,
    low_threshold_pct: u32,
    high_threshold_pct: u32,
    initial_lazy: bool,
    block_len: usize,
}

impl GateCfg {
    #[cfg(not(feature = "l3-tune"))]
    #[inline(always)]
    fn shipped() -> Self {
        GateCfg {
            enabled: L3_GATE_ENABLED,
            low_threshold_pct: L3_GATE_LOW_THRESHOLD_PCT,
            high_threshold_pct: L3_GATE_HIGH_THRESHOLD_PCT,
            initial_lazy: L3_GATE_INITIAL_LAZY,
            block_len: L3_GATE_BLOCK_LEN,
        }
    }

    #[cfg(feature = "l3-tune")]
    #[inline(always)]
    fn from_tune(t: tune::L3GateTune) -> Self {
        GateCfg {
            enabled: t.enabled,
            low_threshold_pct: t.low_threshold_pct,
            high_threshold_pct: t.high_threshold_pct,
            initial_lazy: t.initial_lazy,
            block_len: t.block_len,
        }
    }
}

/// This gate's OWN block-end chooser, parameterized by the tunable
/// `block_len` instead of `parse::mod`'s hardcoded `SOFT_MAX_BLOCK_LENGTH` —
/// otherwise an exact mirror of `super::choose_max_block_end`, including the
/// `MIN_BLOCK_LENGTH` "avoid a tiny trailing block" guard (measured to
/// matter: without it, a short final block cost a real, if small, self-tax
/// — see [`L3_GATE_BLOCK_LEN`]'s doc comment).
#[inline]
fn gate_block_end(block_begin: usize, in_end: usize, block_len: usize) -> usize {
    if in_end - block_begin < block_len + MIN_BLOCK_LENGTH {
        in_end
    } else {
        block_begin + block_len
    }
}

pub(super) fn run(
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    params: &LevelParams,
    statics: &StaticCodes,
    bw: &mut BitWriter,
    is_last: bool,
) {
    #[cfg(not(feature = "l3-tune"))]
    let cfg = GateCfg::shipped();
    #[cfg(feature = "l3-tune")]
    let cfg = GateCfg::from_tune(tune::get());

    let mut mf = HcMatchfinder::new();
    let mut in_base = 0usize;
    let mut next_hashes = [0u32; 2];
    let mut sink = Sink::new();

    if data_start > 0 {
        mf.skip_bytes(buf, &mut in_base, 0, in_end, data_start, &mut next_hashes);
    }

    let mut in_next = data_start;
    let block_len = cfg.block_len.max(1);
    let mut use_lazy_next = !cfg.enabled || cfg.initial_lazy;

    loop {
        // Start a new DEFLATE block, sized by THIS gate's own (independent)
        // block-length knob, not `SOFT_MAX_BLOCK_LENGTH`.
        let block_begin = in_next;
        let in_max_block_end = gate_block_end(block_begin, in_end, block_len);
        sink.begin();

        let use_lazy = use_lazy_next;
        in_next = if use_lazy {
            lazy::run_block(
                buf,
                in_next,
                block_begin,
                in_max_block_end,
                in_end,
                params,
                false, // lazy (not lazy2) — L3 never uses the lazy2 tie-break.
                &mut mf,
                &mut in_base,
                &mut next_hashes,
                &mut sink,
            )
        } else {
            greedy::run_block(
                buf,
                in_next,
                block_begin,
                in_max_block_end,
                in_end,
                params,
                &mut mf,
                &mut in_base,
                &mut next_hashes,
                &mut sink,
            )
        };

        emit_block(
            bw,
            buf,
            block_begin,
            &sink,
            statics,
            is_last && in_next == in_end,
        );

        // One-block-lag detector: decide the NEXT gate-block's strategy from
        // THIS block's already-populated `litlen_freqs` (free — no extra
        // scan), same mechanism as `Hash3Cfg`'s gate — but TWO-SIDED (see
        // the module doc comment).
        let literal_count: u32 = sink.litlen_freqs[..NUM_LITERALS].iter().sum();
        let match_count: u32 = sink.litlen_freqs[DEFLATE_FIRST_LEN_SYM..].iter().sum();
        let total = literal_count + match_count;
        use_lazy_next = !cfg.enabled
            || if total == 0 {
                // No signal (degenerate empty block) — hold the current
                // decision rather than guessing.
                use_lazy
            } else {
                let lit_pct = (literal_count as u64 * 100) / total as u64;
                !(lit_pct <= cfg.low_threshold_pct as u64
                    || lit_pct >= cfg.high_threshold_pct as u64)
            };

        if in_next == in_end {
            break;
        }
    }
}

/// Correctness tests for the DETECTOR-GATED LAZY-L3 composition — only
/// meaningfully exercised under `l3-tune` (level 3 is plain `Strategy::Greedy`
/// otherwise, already covered by the rest of the suite). Mirrors
/// `near_optimal.rs`'s own `compress_gzip` + `flate2::read::GzDecoder`
/// roundtrip pattern. Each fixture is sized to span SEVERAL
/// [`L3_GATE_BLOCK_LEN`] (300_000-byte) gate-blocks so a roundtrip failure
/// here would catch a real cross-block state bug (shared matchfinder handed
/// off between `greedy::run_block`/`lazy::run_block`, or the one-block-lag
/// detector itself) — not just "one strategy in isolation still works",
/// which the pre-existing greedy/lazy test suites already cover.
#[cfg(all(test, feature = "l3-tune"))]
mod tests {
    use super::super::super::{compress_gzip, compress_oneshot};
    use std::io::Read;

    fn decode(gz: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(gz)
            .read_to_end(&mut out)
            .expect("flate2 decode");
        out
    }

    /// A tiny xorshift PRNG — enough to produce incompressible-looking bytes
    /// deterministically (no external `rand` dependency needed).
    fn xorshift_fill(buf: &mut [u8], mut seed: u64) {
        for chunk in buf.chunks_mut(8) {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let bytes = seed.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }

    /// FASTQ-shaped low-alphabet content — the `ecoli.fastq`-class signal
    /// (high match density from a ~5-symbol alphabet: A/C/G/T/N + newline).
    fn dna_like(len: usize) -> Vec<u8> {
        const BASES: &[u8] = b"ACGTN";
        let mut out = Vec::with_capacity(len);
        let mut seed = 0x5EED_u64;
        while out.len() < len {
            for _ in 0..60 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                out.push(BASES[(seed % BASES.len() as u64) as usize]);
            }
            out.push(b'\n');
        }
        out.truncate(len);
        out
    }

    /// Near-incompressible content — the `weights.safetensors`-class signal
    /// (near-uniform high literal fraction).
    fn random_like(len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        xorshift_fill(&mut out, 0xC0FFEE);
        out
    }

    /// Ordinary redundant text — a stand-in for the 19-file "stays LAZY"
    /// middle band.
    fn text_like(len: usize) -> Vec<u8> {
        let phrase = b"the quick brown fox jumps over the lazy dog. ";
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            out.extend_from_slice(phrase);
        }
        out.truncate(len);
        out
    }

    #[test]
    fn gated_l3_roundtrips_dna_like_multi_block() {
        // ~2.4x L3_GATE_BLOCK_LEN so the gate must lock onto GREEDY and stay
        // there across a block boundary.
        let data = dna_like(720_000);
        let gz = compress_gzip(&data, 3);
        assert_eq!(decode(&gz), data);
    }

    #[test]
    fn gated_l3_roundtrips_random_like_multi_block() {
        let data = random_like(720_000);
        let gz = compress_gzip(&data, 3);
        assert_eq!(decode(&gz), data);
    }

    #[test]
    fn gated_l3_roundtrips_text_like_multi_block() {
        let data = text_like(720_000);
        let gz = compress_gzip(&data, 3);
        assert_eq!(decode(&gz), data);
    }

    /// The real stress case: content that SWITCHES class mid-stream, several
    /// times, across gate-block boundaries — exercises the shared
    /// `HcMatchfinder`/`in_base`/`next_hashes` handoff between
    /// `greedy::run_block` and `lazy::run_block` under repeated strategy
    /// flips, and the one-block-lag detector's actual transition behavior
    /// (not just "converges once and stays").
    #[test]
    fn gated_l3_roundtrips_mixed_content_multi_switch() {
        let mut data = Vec::new();
        data.extend(text_like(400_000));
        data.extend(dna_like(500_000));
        data.extend(random_like(500_000));
        data.extend(text_like(400_000));
        data.extend(dna_like(350_000));
        let gz = compress_gzip(&data, 3);
        assert_eq!(decode(&gz), data);
    }

    /// Sub-one-gate-block inputs (smaller than `L3_GATE_BLOCK_LEN`) — the
    /// degenerate single-block case, both content classes.
    #[test]
    fn gated_l3_roundtrips_small_inputs() {
        for data in [
            dna_like(500),
            random_like(500),
            text_like(500),
            vec![7u8; 1],
        ] {
            let gz = compress_gzip(&data, 3);
            assert_eq!(decode(&gz), data);
        }
    }

    /// `compress_oneshot` (the raw-DEFLATE, no gzip-framing entry point) —
    /// a second call path into the same `Strategy::LazyGated` dispatch, with
    /// its own inflate oracle rather than `flate2`'s gzip framing.
    #[test]
    fn gated_l3_roundtrips_via_compress_oneshot() {
        let data = dna_like(650_000);
        let compressed = compress_oneshot(&data, 3);
        let mut out = Vec::new();
        flate2::read::DeflateDecoder::new(&compressed[..])
            .read_to_end(&mut out)
            .expect("flate2 raw-deflate decode");
        assert_eq!(out, data);
    }
}
