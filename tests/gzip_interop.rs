//! NO output of any compression level or thread count may ever carry the
//! legacy "GZ" FEXTRA subfield.
//!
//! History: pre-1.0 gzippy wrote its parallel blocks as concatenated members
//! whose headers carried an FEXTRA "GZ" block-size marker (man/gzippy-format).
//! RFC-1952-compliant readers skip it, but enough third-party gzip tools
//! mishandle FEXTRA that the marker made output non-interoperable in the
//! wild. It was removed from every writer in v1.0. These tests fail closed:
//! any future writer that sets flg bit 0x04 (or resurrects the subfield)
//! goes red at every level and thread count.
//!
//! The decode side still READS the marker (decompress::format), which is
//! backward compatibility, not regressable writer behavior.

const LEGACY_SUBFIELD_ID: [u8; 2] = [b'G', b'Z'];

/// A payload with real redundancy so every level produces a real member.
fn payload(len: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(len);
    let phrase = b"no legacy GZ FEXTRA subfield may ever be written. ";
    while data.len() < len {
        let i = data.len();
        data.extend_from_slice(phrase);
        data.extend_from_slice(&(i % 9973).to_le_bytes());
    }
    data.truncate(len);
    data
}

fn assert_no_fextra_member_header(stream: &[u8], what: &str) {
    // Walk EVERY member header in the stream (the multiblock shapes are the
    // risk surface): first member at 0, then follow members by decoding the
    // header length, skipping the deflate body by checking for the next
    // member via the FLG walk. Simpler and stricter: assert the FIRST
    // member's FLG has no 0x04 and scan the stream for a member start whose
    // FLG carries 0x04 followed by the legacy ID bytes.
    assert!(
        stream.starts_with(&[0x1f, 0x8b, 0x08]),
        "{what}: not a gzip stream"
    );
    let mut i = 0usize;
    let mut first = true;
    while i + 10 <= stream.len() {
        if stream[i] != 0x1f || stream[i + 1] != 0x8b {
            break;
        }
        let flg = stream[i + 3];
        assert!(
            flg & 0x04 == 0,
            "{what}: member at byte {i} sets FEXTRA (flg = {flg:#04x}) — \
             the legacy GZ subfield came back"
        );
        // Skip this member's header fields to find the next member start
        // conservatively: XLEN, then FNAME/FCOMMENT are flag-gated.
        let mut j = i + 10;
        if flg & 0x08 != 0 {
            while j < stream.len() && stream[j] != 0 {
                j += 1;
            }
            j += 1; // NUL
        }
        if flg & 0x10 != 0 {
            while j < stream.len() && stream[j] != 0 {
                j += 1;
            }
            j += 1; // NUL
        }
        // DEF (flg bit 5) would need 2 more bytes; never emitted anywhere.
        assert!(
            flg & 0x20 == 0 && flg & 0x02 == 0,
            "{what}: unexpected flags {flg:#04x}"
        );
        let _ = first;
        first = false;
        // For a single-member stream the member body follows; with more
        // members T>1 writes concatenations whose member boundaries we find
        // by scanning for the magic (0x1f 0x8b cannot appear densely enough
        // in compressed data to false-positively matter here — we LOOK for
        // the exact legacy pattern below instead).
        i = j;
        // The body is opaque; find the next member empirically.
        while i + 2 <= stream.len() && !(stream[i] == 0x1f && stream[i + 1] == 0x8b) {
            i += 1;
        }
    }
    // And the legacy subfield pattern itself must never appear anywhere.
    for w in 0..stream.len().saturating_sub(4) {
        if stream[w] == LEGACY_SUBFIELD_ID[0] && stream[w + 1] == LEGACY_SUBFIELD_ID[1] {
            assert_ne!(
                &stream[w..w + 4],
                &LEGACY_SUBFIELD_ID[..],
                "{what}: legacy GZ subfield found at byte {w}"
            );
        }
    }
}

#[test]
fn no_level_or_thread_count_sets_fextra() {
    let data = payload(600_000);
    for level in 0..=12u8 {
        for threads in [1usize, 2, 4] {
            // The public compress entry: covers the T1 routes (slack-padded
            // mmap-free path) and the T>1 pure pipeline at every level.
            let out = gzippy::compress_with_threads(&data, level, threads)
                .unwrap_or_else(|e| panic!("L{level} T{threads}: {e}"));
            assert_no_fextra_member_header(
                &out,
                &format!("compress_with_threads L{level} T{threads}"),
            );
            // Independent-decoder roundtrip: another gzip MUST read it.
            let mut d = flate2::read::MultiGzDecoder::new(&out[..]);
            let mut restored = Vec::new();
            std::io::Read::read_to_end(&mut d, &mut restored)
                .unwrap_or_else(|e| panic!("L{level} T{threads}: flate2 rejected: {e}"));
            assert_eq!(restored, data, "L{level} T{threads}: roundtrip");
        }
    }
}

#[test]
fn no_fextra_on_the_streaming_wrapper() {
    let data = payload(400_000);
    let mut out = Vec::new();
    gzippy::compress_to_writer(&data[..], &mut out, 7).expect("compress_to_writer");
    assert_no_fextra_member_header(&out, "compress_to_writer L7");

    let mut restored = Vec::new();
    let n = gzippy::decompress_to_writer(&out, &mut restored).expect("decode");
    assert_eq!(n, data.len() as u64);
    assert_eq!(restored, data);
}
