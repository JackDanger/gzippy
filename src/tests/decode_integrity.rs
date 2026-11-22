//! Container-integrity regressions from the 2026-11 decoder audit.
//!
//! Contract (gzip(1) parity): corrupt input that gzip rejects must be an
//! error, never Ok with wrong bytes; no input may panic; no allocation may be
//! sized by an untrusted header field beyond what the compressed bytes could
//! expand to. Each test names the audit finding it pins. Every fixture is a
//! few hundred bytes built in-test (or a byte literal of the audit's repro).

use crate::compress::zlib_container::adler32;
use crate::decompress::bgzf::{
    decompress_bgzf_parallel, decompress_bgzf_parallel_to_vec,
    decompress_multi_member_parallel_to_vec, gz_coverage_is_pure,
};
use crate::decompress::{decompress_gzip_to_vec, decompress_raw_bytes, decompress_zlib_turbo};
use crate::tests::alloc_counter::CountingAllocator;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// One final stored DEFLATE block holding `payload` (< 64 KiB).
fn stored_deflate(payload: &[u8]) -> Vec<u8> {
    let len = payload.len() as u16;
    let mut d = vec![0x01];
    d.extend_from_slice(&len.to_le_bytes());
    d.extend_from_slice(&(!len).to_le_bytes());
    d.extend_from_slice(payload);
    d
}

/// A plain gzip member: 10-byte header, `deflate`, CRC32 + ISIZE trailer.
fn plain_member(deflate: &[u8], crc: u32, isize: u32) -> Vec<u8> {
    let mut m = vec![0x1f, 0x8b, 0x08, 0, 0, 0, 0, 0, 0, 0x03];
    m.extend_from_slice(deflate);
    m.extend_from_slice(&crc.to_le_bytes());
    m.extend_from_slice(&isize.to_le_bytes());
    m
}

/// A gzippy "GZ"-FEXTRA member (4-byte whole-member size form), as the
/// parallel compressor writes them.
fn gz_member(deflate: &[u8], crc: u32, isize: u32) -> Vec<u8> {
    let total = (10 + 2 + 8 + deflate.len() + 8) as u32;
    let mut m = vec![0x1f, 0x8b, 0x08, 0x04, 0, 0, 0, 0, 0, 0x03, 8, 0];
    m.extend_from_slice(b"GZ");
    m.extend_from_slice(&4u16.to_le_bytes());
    m.extend_from_slice(&total.to_le_bytes());
    m.extend_from_slice(deflate);
    m.extend_from_slice(&crc.to_le_bytes());
    m.extend_from_slice(&isize.to_le_bytes());
    m
}

fn payload(seed: u8) -> Vec<u8> {
    (0..1500u32)
        .map(|i| b"the quick brown fox jumps over the lazy dog\n"[(i as usize) % 44] ^ seed)
        .collect()
}

/// Two stored members; when `corrupt`, member 2's payload has one byte
/// flipped AFTER its CRC was computed (ISIZE still right).
fn two_members(build: fn(&[u8], u32, u32) -> Vec<u8>, corrupt: bool) -> (Vec<u8>, Vec<u8>) {
    let (p1, p2) = (payload(0), payload(1));
    let mut d2 = stored_deflate(&p2);
    if corrupt {
        d2[100] ^= 0x20;
    }
    let mut file = build(&stored_deflate(&p1), crc32fast::hash(&p1), p1.len() as u32);
    file.extend(build(&d2, crc32fast::hash(&p2), p2.len() as u32));
    let mut expect = p1;
    expect.extend(p2);
    (file, expect)
}

// ── Finding 3: plain multi-member at T>1 verified ISIZE but never CRC32 ────

/// Audit repro `mm_badcrc.gz`: member 2's CRC field is wrong, its body valid.
const MM_BADCRC: &str = "1f8b08000000000000030bc94855282ccd4cce56482aca2fcf5348cbaf50c82acd2d2856c82f4b2d5228014ae72456552aa4e4a7eb29848c2a1e553caa7854f1a8e251c5c34b310078563412080700001f8b08000000000000030b484cce56c8ad5448caaf5028cf2cc95048cb2c4b5548c9af4acd53c8c92c2ccd2f52c82a4d2f565454540818553aaa7454e9a8d251a5a34a075a290020f9b4bab8060000";

#[test]
fn multi_member_parallel_rejects_bad_crc() {
    let data = hex(MM_BADCRC);
    for t in [2, 4, 8] {
        assert!(
            decompress_multi_member_parallel_to_vec(&data, t).is_err(),
            "T{t}"
        );
        assert!(
            decompress_gzip_to_vec(&data, t).is_err(),
            "decompress_gzip_to_vec T{t}"
        );
    }
}

#[test]
fn multi_member_parallel_rejects_corrupt_stored_byte() {
    let (data, _) = two_members(plain_member, true);
    for t in [1, 2, 4] {
        assert!(decompress_gzip_to_vec(&data, t).is_err(), "T{t}");
    }
    assert!(decompress_multi_member_parallel_to_vec(&data, 4).is_err());
}

#[test]
fn multi_member_parallel_valid_is_byte_identical() {
    let (data, expect) = two_members(plain_member, false);
    for t in [1, 2, 4] {
        assert_eq!(decompress_gzip_to_vec(&data, t).unwrap(), expect, "T{t}");
    }
    assert_eq!(
        decompress_multi_member_parallel_to_vec(&data, 4).unwrap(),
        expect
    );
}

// ── Finding 4: GZ-format blocks never verified CRC32; T1 skipped ISIZE;
//    ISIZE=0 blocks were skipped without decoding ─────────────────────────

fn bgzf_all_paths_err(data: &[u8], what: &str) {
    for t in [1, 4] {
        let mut sink = Vec::new();
        assert!(
            decompress_bgzf_parallel(data, &mut sink, t).is_err(),
            "{what}: decompress_bgzf_parallel T{t} accepted it"
        );
        assert!(
            decompress_bgzf_parallel_to_vec(data, t).is_err(),
            "{what}: decompress_bgzf_parallel_to_vec T{t} accepted it"
        );
        assert!(
            decompress_gzip_to_vec(data, t).is_err(),
            "{what}: decompress_gzip_to_vec T{t} accepted it"
        );
    }
}

#[test]
fn bgzf_rejects_bad_crc_corrupt_body_and_wrong_isize() {
    let p = payload(0);
    let crc = crc32fast::hash(&p);
    let n = p.len() as u32;

    let bad_crc = gz_member(&stored_deflate(&p), crc ^ 1, n);
    assert!(gz_coverage_is_pure(&bad_crc));
    bgzf_all_paths_err(&bad_crc, "bad CRC");

    let mut body = stored_deflate(&p);
    body[200] ^= 0x01;
    bgzf_all_paths_err(&gz_member(&body, crc, n), "corrupt stored byte");

    // ISIZE=0 on a block with content: must decode (and fail), not be skipped.
    bgzf_all_paths_err(&gz_member(&stored_deflate(&p), crc, 0), "ISIZE=0");
    // ISIZE larger than the real output: the T1 streaming path accepted it.
    bgzf_all_paths_err(&gz_member(&stored_deflate(&p), crc, n + 7), "ISIZE too big");

    // Second block bad, first fine (the pipelined writer must not emit it).
    let (two, _) = two_members(gz_member, true);
    assert!(gz_coverage_is_pure(&two));
    bgzf_all_paths_err(&two, "second block corrupt");
}

#[test]
fn bgzf_valid_is_byte_identical_including_empty_block() {
    let (mut data, expect) = two_members(gz_member, false);
    // A genuinely empty block (ISIZE=0, CRC=0, body decodes to nothing).
    data.extend(gz_member(&stored_deflate(&[]), 0, 0));
    assert!(gz_coverage_is_pure(&data));
    for t in [1, 4] {
        let mut sink = Vec::new();
        decompress_bgzf_parallel(&data, &mut sink, t).unwrap();
        assert_eq!(sink, expect, "T{t}");
        assert_eq!(decompress_bgzf_parallel_to_vec(&data, t).unwrap(), expect);
        assert_eq!(decompress_gzip_to_vec(&data, t).unwrap(), expect);
    }
}

// ── Finding 5: BSIZE shorter than the block's own header panicked ──────────

/// Audit repro `bgzf_bsize20x2.gz`: two 20-byte "blocks" whose BSIZE (20)
/// equals the header length, so `deflate_start (20) > start+len-8 (12)`.
const BGZF_BSIZE20X2: &str =
    "1f8b08040000000000030800475a0400140000001f8b08040000000000030800475a040014000000";

#[test]
fn bgzf_bsize_inside_header_is_not_our_format_and_does_not_panic() {
    let data = hex(BGZF_BSIZE20X2);
    // The router must not hand this to the GZ fast path ...
    assert!(!gz_coverage_is_pure(&data));
    // ... and the GZ entry points error cleanly if called directly.
    bgzf_all_paths_err(&data, "BSIZE=20");
    // gzip(1) rejects it ("invalid compressed data--format violated").
    let mut sink = Vec::new();
    assert!(crate::decompress::decompress_gzip_to_writer(&data, &mut sink).is_err());
}

// ── Finding 6: GZ-format buffers were sized from untrusted ISIZE ───────────

#[test]
fn bgzf_huge_isize_is_rejected_before_allocating() {
    let p = b"hello hello hello hello\n";
    let block = gz_member(&stored_deflate(p), crc32fast::hash(p), 0xffff_fff0);
    let data: Vec<u8> = (0..8).flat_map(|_| block.clone()).collect();
    assert!(gz_coverage_is_pure(&data));
    CountingAllocator::reset();
    let r = decompress_bgzf_parallel_to_vec(&data, 8);
    let bytes = CountingAllocator::bytes();
    assert!(r.is_err());
    // 8 x 4 GiB claimed; nothing near that may be allocated.
    assert!(
        bytes < 16 << 20,
        "allocated {bytes} bytes for a {}-byte input",
        data.len()
    );
    let mut sink = Vec::new();
    assert!(decompress_bgzf_parallel(&data, &mut sink, 1).is_err());
    assert!(decompress_bgzf_parallel(&data, &mut sink, 8).is_err());
}

// ── Finding 8: zlib header + Adler-32 unchecked; growth unbounded ─────────

fn zlib_wrap(deflate: &[u8], adler: u32) -> Vec<u8> {
    let mut z = vec![0x78, 0x9c];
    z.extend_from_slice(deflate);
    z.extend_from_slice(&adler.to_be_bytes());
    z
}

#[test]
fn zlib_rejects_bad_adler_corrupt_body_and_bad_header() {
    // Audit repro `zlib_badadler.zz` (last trailer byte flipped).
    let bad = hex("789ccb48cdc9c957a8cac94c5228cf2fca49e1ca181518151815205f00001eee3631");
    assert!(decompress_zlib_turbo(&bad, &mut Vec::new()).is_err());

    let p = payload(0);
    let mut body = stored_deflate(&p);
    body[300] ^= 0x01;
    assert!(decompress_zlib_turbo(&zlib_wrap(&body, adler32(&p)), &mut Vec::new()).is_err());

    let good = zlib_wrap(&stored_deflate(&p), adler32(&p));
    let mut fcheck = good.clone();
    fcheck[1] ^= 0x01; // (CMF<<8|FLG) % 31 != 0
    assert!(decompress_zlib_turbo(&fcheck, &mut Vec::new()).is_err());
    let mut fdict = good.clone();
    fdict[1] = 0xbb; // 0x78bb % 31 == 0, FDICT set
    assert!(decompress_zlib_turbo(&fdict, &mut Vec::new()).is_err());
    // Truncated trailer.
    assert!(decompress_zlib_turbo(&good[..good.len() - 2], &mut Vec::new()).is_err());
}

#[test]
fn zlib_valid_is_byte_identical() {
    let p = payload(3);
    let good = zlib_wrap(&stored_deflate(&p), adler32(&p));
    let mut out = Vec::new();
    decompress_zlib_turbo(&good, &mut out).unwrap();
    assert_eq!(out, p);
    // flate2's compressed form too (dynamic Huffman), and a high-ratio stream
    // that needs the buffer to grow past the 4x first guess.
    let zeros = vec![0u8; 3 << 20];
    for src in [&p[..], &zeros[..]] {
        let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut e, src).unwrap();
        let z = e.finish().unwrap();
        let mut out = Vec::new();
        decompress_zlib_turbo(&z, &mut out).unwrap();
        assert!(out == src);
    }
}

/// Audit repro `zlib_phantom.zz` (145 B): drove the 2 GiB doubling loop
/// (~900 MB RSS, 6.5 s) before failing.
#[test]
fn zlib_phantom_stream_is_bounded() {
    let mut z = hex("789c0de00190244992244922");
    z.resize(0x8b, 0);
    z.extend_from_slice(&hex("102100000000"));
    assert_eq!(z.len(), 145);
    CountingAllocator::reset();
    let r = decompress_zlib_turbo(&z, &mut Vec::new());
    let bytes = CountingAllocator::bytes();
    assert!(r.is_err());
    assert!(
        bytes < 16 << 20,
        "allocated {bytes} bytes for a 145-byte input"
    );
}

// ── Finding 9: decompress_raw (library) doubled to 2 GiB ───────────────────

/// Audit repro `fuzz_raw_oom_10b.bin` minus the fuzz target's selector byte:
/// libFuzzer reported `malloc(1073741824)`.
#[test]
fn raw_deflate_growth_is_bounded_by_input() {
    let raw = hex("151b1b1b1b1b1b1b7e");
    CountingAllocator::reset();
    let r = decompress_raw_bytes(&raw);
    let bytes = CountingAllocator::bytes();
    assert!(r.is_err());
    assert!(
        bytes < 16 << 20,
        "allocated {bytes} bytes for a 9-byte input"
    );
    // A valid high-ratio stream still decodes in full.
    let zeros = vec![0u8; 3 << 20];
    let mut e = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
    std::io::Write::write_all(&mut e, &zeros).unwrap();
    assert!(decompress_raw_bytes(&e.finish().unwrap()).unwrap() == zeros);
}

// ── Finding 10: a tampered index panicked `--seek` ─────────────────────────

#[test]
fn seek_with_tampered_index_errors_instead_of_panicking() {
    use crate::decompress::index::{build_index, load_index, seek_decompress, serialize_index};
    let data: Vec<u8> = (0..300 * 1024u32)
        .map(|i| (i % 251) as u8 ^ (i >> 9) as u8)
        .collect();
    let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    std::io::Write::write_all(&mut e, &data).unwrap();
    let gz = e.finish().unwrap();
    let mut idx = Vec::new();
    serialize_index(&build_index(&gz, 32 * 1024).unwrap(), &mut idx).unwrap();

    // Untampered: works.
    let mut out = Vec::new();
    seek_decompress(&gz, &load_index(&idx).unwrap(), 0, u64::MAX, &mut out).unwrap();
    assert!(out == data);

    // deflate_offset (bytes 6..10) and point 0's input_byte_pos (bytes 24..28).
    for (at, v) in [(6usize, 0xffff_fff0u32), (24, 0x7fff_fff0)] {
        let mut bad = idx.clone();
        bad[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let index = load_index(&bad).unwrap();
        let r =
            std::panic::catch_unwind(|| seek_decompress(&gz, &index, 0, u64::MAX, &mut Vec::new()));
        assert!(
            matches!(r, Ok(Err(_))),
            "field at byte {at}: must be a clean error"
        );
    }
}

// ── Finding 12: FHCRC and reserved/encrypted FLG bits (gzip 1.14 rejects) ──

#[test]
fn header_crc_and_reserved_flags_are_refused() {
    use crate::decompress::format::{classify_first_member, FirstMemberBoundary};
    // Audit repros hdr_fhcrc_bad.gz / hdr_reserved_flag.gz.
    let fhcrc_bad =
        hex("1f8b0802000000000003deadcb484d4c492d5228492d2e512848acccc94f4ce1ca188262008cd927fcc8000000");
    let reserved = hex(
        "1f8b0820000000000003cb484d4c492d5228492d2e512848acccc94f4ce1ca188262008cd927fcc8000000",
    );
    for d in [&fhcrc_bad, &reserved] {
        assert!(!matches!(
            classify_first_member(d),
            FirstMemberBoundary::Header
        ));
    }
    let mut flag40 = reserved.clone();
    flag40[3] = 0x40;
    assert!(!matches!(
        classify_first_member(&flag40),
        FirstMemberBoundary::Header
    ));

    // A CORRECT FHCRC is accepted.
    let mut ok = fhcrc_bad.clone();
    let hcrc = (crc32fast::hash(&ok[..10]) & 0xffff) as u16;
    ok[10..12].copy_from_slice(&hcrc.to_le_bytes());
    assert!(matches!(
        classify_first_member(&ok),
        FirstMemberBoundary::Header
    ));
}
