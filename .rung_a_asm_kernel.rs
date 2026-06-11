//! ASM-campaign increment (a) — the litlen decode+consume micro-sequence as a
//! single `core::arch::asm!` block (plans/asm-campaign.md §2(a)).
//!
//! Scope: the P3.2 literal-chain arm of `Block::decode_clean_into_contig`
//! (`marker_inflate.rs`). One step = the fused short-table gather + chain-gate
//! + consume for ONE candidate chained literal:
//!
//! ```text
//!   entry = short_code_lookup[bitbuf & 0xFFF]          (F1 gather)
//!   gate: short entry AND bit_count!=0 AND sym_count==1
//!         AND code<=255 AND bit_count<=available
//!   hit:  bitbuf >>= bit_count; bitsleft -= bit_count  (consume)
//! ```
//!
//! The asm gate is a STRICT SUBSET of the Rust chain gate (the Rust path
//! refills when `available < 32` before gating; the asm never refills, so it
//! bails in exactly the cases where the Rust gate needs the refill or the
//! long-code path). On a bail the caller falls through to the unchanged Rust
//! body (`LutLitLenCode::decode` + the original gate), which re-reads the same
//! (L1-hot) entry from the SAME bit cursor — no bits are consumed on a bail.
//! On a hit the consume is bit-identical to the Rust arm's
//! (`Bits::consume(bit_count)`), and the prefix-free-code argument documented
//! at the Rust gate ("only CONSUME packets fully backed by real bits") makes
//! the no-refill lookup safe: an entry whose `bit_count <= available` is
//! determined entirely by real bits, because consumed `bitbuf` high bits are
//! zero and refills are append-only. Byte-exact by construction; the
//! differential tests below pin step-level state equality, and the corpus
//! suite (run with `--features asm-kernel`) pins whole-stream equality.
//!
//! Dispatch policy (charter §4): compiled only under
//! `feature = "asm-kernel"` + `target_arch = "x86_64"`; runtime-gated on
//! BMI2 detection (the block uses `shrx`) and the `GZIPPY_ASM_KERNEL=0`
//! kill-switch. The pure-Rust loop is always compiled and is the sole path
//! everywhere else. Effect verification (decide.sh EFFECT predicate):
//! `GZIPPY_ASM_STATS=1` + `GZIPPY_VERBOSE=1` prints hit/bail counters via
//! `dump_if_enabled` (chunk_fetcher.rs, next to the contig-prof dump);
//! counters are only accumulated when stats are enabled, so the production
//! hot loop carries zero atomics.

#![allow(dead_code)]

/// Entry-layout constants mirrored from `lut_huffman.rs` (compile-checked
/// against the source constants below so drift is impossible).
const LARGE_FLAG_BIT: u32 = super::lut_huffman::LARGE_FLAG_BIT;
const _: () = assert!(LARGE_FLAG_BIT == 1 << 25);
const _: () = assert!(super::lut_huffman::LARGE_SHORT_CODE_LEN_OFFSET == 28);
const _: () = assert!(super::lut_huffman::LARGE_SYM_COUNT_OFFSET == 26);
const _: () = assert!(super::lut_huffman::LARGE_SYM_COUNT_MASK == 3);
const _: () = assert!(super::lut_huffman::ISAL_DECODE_LONG_BITS == 12);

use crate::decompress::inflate::consume_first_decode::Bits;

/// Reference (pure-Rust) implementation of exactly the asm step's contract.
/// Compiled on every arch; the differential tests pin asm == ref, and the
/// ref's own equivalence to the production chain-arm gate is by inspection
/// (same constants, same order, same consume).
#[inline(always)]
pub fn litlen_chain_step_ref(tbl: &[u32; 1 << 12], bits: &mut Bits<'_>) -> Option<u8> {
    let entry = tbl[(bits.bitbuf & 0xFFF) as usize];
    if entry & LARGE_FLAG_BIT != 0 {
        return None; // long-code pointer — Rust path resolves it
    }
    let bit_count = entry >> super::lut_huffman::LARGE_SHORT_CODE_LEN_OFFSET;
    if bit_count == 0 {
        return None; // invalid code — Rust path produces the error
    }
    if (entry >> super::lut_huffman::LARGE_SYM_COUNT_OFFSET)
        & super::lut_huffman::LARGE_SYM_COUNT_MASK
        != 1
    {
        return None; // multi-literal packet — Rust packed-store arm
    }
    let code = entry & 0xFFFF;
    if code > 255 {
        return None; // length/EOB — Rust trailing handler
    }
    if bit_count > bits.available() {
        return None; // not fully backed by real bits — Rust refill+gate
    }
    bits.consume(bit_count);
    Some(code as u8)
}

#[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
mod imp {
    use super::Bits;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    pub static CHAIN_HITS: AtomicU64 = AtomicU64::new(0);
    pub static CHAIN_BAILS: AtomicU64 = AtomicU64::new(0);

    /// Runtime dispatch: ON when compiled in, unless `GZIPPY_ASM_KERNEL=0`
    /// (kill-switch) or the CPU lacks BMI2 (`shrx` in the block).
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            let killed = std::env::var("GZIPPY_ASM_KERNEL").is_ok_and(|v| v == "0");
            !killed && std::arch::is_x86_feature_detected!("bmi2")
        })
    }

    /// `GZIPPY_ASM_STATS=1` — effect-verification counters (decide.sh
    /// EFFECT predicate). OFF (default) keeps the hot loop atomics-free.
    pub fn stats_enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("GZIPPY_ASM_STATS").is_ok_and(|v| v == "1"))
    }

    /// The asm step. Contract identical to `litlen_chain_step_ref`.
    ///
    /// `ret` encoding: 0 = bail (no state change), `0x100 | code` = hit
    /// (bitbuf/bitsleft consumed by the entry's bit_count).
    ///
    /// # Safety (internal)
    /// The block only reads `tbl[0..4096]` (a valid, fully-initialized
    /// array reference) and registers; `options(readonly, nostack)`.
    #[inline(always)]
    pub fn litlen_chain_step(tbl: &[u32; 1 << 12], bits: &mut Bits<'_>, stats: bool) -> Option<u8> {
        let mut bitbuf = bits.bitbuf;
        let mut bitsleft: u64 = bits.bitsleft as u64;
        let ret: u64;
        unsafe {
            core::arch::asm!(
                // ret = 0 (speculative bail; overwritten on the hit path)
                "xor {ret:e}, {ret:e}",
                // F1 gather: entry = tbl[bitbuf & 0xFFF]
                "mov {idx}, {bitbuf}",
                "and {idx:e}, 0xFFF",
                "mov {e:e}, dword ptr [{tbl} + {idx}*4]",
                // long-code pointer? (LARGE_FLAG_BIT = 1<<25)
                "test {e:e}, 0x2000000",
                "jnz 9f",
                // bit_count = entry >> 28; 0 = invalid
                "mov {bc:e}, {e:e}",
                "shr {bc:e}, 28",
                "jz 9f",
                // sym_count == 1 ? (bits 26-27)
                "mov {t:e}, {e:e}",
                "shr {t:e}, 26",
                "and {t:e}, 3",
                "cmp {t:e}, 1",
                "jne 9f",
                // code = entry & 0xFFFF; literal ⇔ code <= 255
                "movzx {t:e}, {e:x}",
                "cmp {t:e}, 255",
                "ja 9f",
                // bit_count <= available (= low byte of bitsleft)
                "movzx {a:e}, {bitsleft:l}",
                "cmp {bc:e}, {a:e}",
                "ja 9f",
                // consume: bitbuf >>= bit_count; bitsleft -= bit_count
                "shrx {bitbuf}, {bitbuf}, {bc}",
                "sub {bitsleft:e}, {bc:e}",
                // ret = 0x100 | code
                "lea {ret:e}, [{t:e} + 0x100]",
                "9:",
                tbl = in(reg) tbl.as_ptr(),
                bitbuf = inout(reg) bitbuf,
                bitsleft = inout(reg) bitsleft,
                ret = out(reg) ret,
                idx = out(reg) _,
                e = out(reg) _,
                bc = out(reg) _,
                t = out(reg) _,
                a = out(reg) _,
                options(readonly, nostack),
            );
        }
        if ret == 0 {
            if stats {
                CHAIN_BAILS.fetch_add(1, Ordering::Relaxed);
            }
            return None;
        }
        bits.bitbuf = bitbuf;
        bits.bitsleft = bitsleft as u32;
        if stats {
            CHAIN_HITS.fetch_add(1, Ordering::Relaxed);
        }
        Some((ret & 0xFF) as u8)
    }

    pub fn dump_if_enabled() {
        if !stats_enabled() {
            return;
        }
        eprintln!(
            "[asm-kernel] enabled={} chain_hits={} chain_bails={}",
            enabled(),
            CHAIN_HITS.load(Ordering::Relaxed),
            CHAIN_BAILS.load(Ordering::Relaxed),
        );
    }
}

#[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
pub use imp::{dump_if_enabled, enabled, litlen_chain_step, stats_enabled};

/// Non-asm builds: the dispatch bool is constant-false and the dump is a
/// no-op, so every call site folds away.
#[cfg(not(all(feature = "asm-kernel", target_arch = "x86_64")))]
pub fn dump_if_enabled() {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::parallel::lut_huffman::LutLitLenCode;

    /// Fixed-Huffman litlen code lengths (RFC1951 §3.2.6).
    fn fixed_litlen_lens() -> Vec<u8> {
        let mut l = vec![8u8; 288];
        l[144..256].iter_mut().for_each(|x| *x = 9);
        l[256..280].iter_mut().for_each(|x| *x = 7);
        l
    }

    /// A skewed dynamic table: short codes for a few hot literals, long
    /// codes elsewhere — exercises multi-literal packing, long-code
    /// pointers, and a range of bit_counts.
    fn skewed_litlen_lens() -> Vec<u8> {
        // 286 = LIT_LEN (dynamic-table size). 288 would trigger the
        // builder's fixed-Huffman static-header correction (`count[8] -= 2`),
        // which requires the static code's 8-bit symbols.
        let mut l = vec![0u8; 286];
        // Kraft-complete: 2 codes of len 2 (2*1/4), 2 of len 3 (2*1/8),
        // 2 of len 4 (2*1/16), 1 of len 5, fill to complete with len 10s:
        // 1/2+1/4+1/8+1/32 used; remainder 3/32+... — easier: use a known
        // complete set: lengths {1: 'e'} no — keep it simple and VALID:
        // 0..=1 -> 2 bits, 2..=3 -> 3 bits, 4..=5 -> 4 bits, 256 -> 5 bits,
        // 6..=8 -> 6,7,7 bits? Validity is checked by rebuild_from's return;
        // assert it below. Use: two 2-bit, two 3-bit, two 4-bit, one 4-bit
        // (256), four 6-bit. Kraft: 2/4+2/8+3/16+4/64 = 0.5+0.25+0.1875+0.0625 = 1.0
        l[0] = 2;
        l[1] = 2;
        l[2] = 3;
        l[3] = 3;
        l[4] = 4;
        l[5] = 4;
        l[256] = 4;
        l[6] = 6;
        l[7] = 6;
        l[8] = 6;
        l[9] = 6;
        l
    }

    fn build(lens: &[u8]) -> LutLitLenCode {
        let mut c = LutLitLenCode::new_empty();
        assert!(c.rebuild_from(lens), "test table lens must be valid");
        c
    }

    /// Step differential: asm and ref must agree on result AND end state for
    /// every (table, bit-state) in a deterministic pseudo-random sweep.
    /// On non-x86_64 / non-bmi2 hosts the asm half is skipped (ref-only
    /// self-consistency still runs so the test is never silently empty).
    #[test]
    fn asm_step_matches_ref_on_random_states() {
        let tables = [build(&fixed_litlen_lens()), build(&skewed_litlen_lens())];
        // xorshift64* deterministic stream
        let mut x: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        };
        let data = [0u8; 8]; // step never refills; Bits::data is untouched
        for tbl in &tables {
            for _ in 0..200_000 {
                let bitbuf = next();
                // bitsleft: sweep real counts 0..=63 with the |56-style high
                // garbage the convention allows (low byte is the real value).
                let avail = (next() % 64) as u32;
                let bitsleft = avail | ((next() as u32) & 0xFFFF_FF00 & !0xFF);
                // Zero bits above `avail` (the loop invariant the production
                // cursor maintains: consumed high bits are zero).
                let bitbuf = if avail == 64 {
                    bitbuf
                } else {
                    bitbuf & ((1u64 << avail) - 1)
                };
                let mut b_ref = Bits {
                    data: &data,
                    pos: 0,
                    bitbuf,
                    bitsleft,
                };
                let mut b_asm = Bits {
                    data: &data,
                    pos: 0,
                    bitbuf,
                    bitsleft,
                };
                let r_ref = litlen_chain_step_ref(&tbl.table.short_code_lookup, &mut b_ref);
                #[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
                {
                    if std::arch::is_x86_feature_detected!("bmi2") {
                        let r_asm = super::litlen_chain_step(
                            &tbl.table.short_code_lookup,
                            &mut b_asm,
                            false,
                        );
                        assert_eq!(
                            r_ref, r_asm,
                            "result diverged at bitbuf={bitbuf:#x} avail={avail}"
                        );
                        assert_eq!(b_ref.bitbuf, b_asm.bitbuf, "bitbuf state diverged");
                        assert_eq!(b_ref.bitsleft, b_asm.bitsleft, "bitsleft state diverged");
                        assert_eq!(b_ref.pos, b_asm.pos);
                    }
                }
                // Ref self-checks: a hit must consume exactly the entry's
                // bit_count and never exceed availability.
                if let Some(_c) = r_ref {
                    let consumed = avail - b_ref.available();
                    assert!(consumed > 0 && consumed <= 15);
                    assert!(consumed <= avail);
                } else {
                    assert_eq!(b_ref.bitbuf, bitbuf);
                    assert_eq!(b_ref.bitsleft, bitsleft);
                }
            }
        }
    }

    /// The ref step's hit set is a subset of the production chain gate: any
    /// hit must be re-derivable from `decode()` on the same cursor with the
    /// same consume. (decode() may additionally refill; with our zero-padded
    /// high bits a hit's entry is fully determined by real bits, so decode()
    /// on the identical state returns the identical packet.)
    #[test]
    fn ref_hits_agree_with_production_decode() {
        let tables = [build(&fixed_litlen_lens()), build(&skewed_litlen_lens())];
        let mut x: u64 = 0xDEADBEEFCAFEF00D;
        let mut next = move || {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        };
        let data = [0u8; 8];
        for tbl in &tables {
            for _ in 0..100_000 {
                let avail = (next() % 64) as u32;
                let bitbuf = if avail == 64 {
                    next()
                } else {
                    next() & ((1u64 << avail) - 1)
                };
                let mut b = Bits {
                    data: &data,
                    pos: 0,
                    bitbuf,
                    bitsleft: avail,
                };
                let mut b2 = Bits {
                    data: &data,
                    pos: 0,
                    bitbuf,
                    bitsleft: avail,
                };
                if let Some(code) = litlen_chain_step_ref(&tbl.table.short_code_lookup, &mut b) {
                    let d = tbl.decode(&mut b2);
                    assert_eq!(d.sym_count, 1);
                    assert_eq!(d.symbol & 0xFFFF, code as u32);
                    assert_eq!(d.bit_count, avail - b.available());
                }
            }
        }
    }
}
