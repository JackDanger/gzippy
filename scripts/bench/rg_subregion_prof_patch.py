#!/usr/bin/env python3
"""R_WORKER SUB-PARTITION: extend the existing rapidgzip_region_prof namespace with a
SYMMETRIC table-vs-decode sub-split matched to gz's region_prof sub-partition:
  R_TABLE  = block->readHeader (readDynamicHuffmanCoding: precode + code-lengths + ALL
             Huffman LUTs, EAGER) — GzipChunk.hpp decodeChunkWithRapidgzip
  R_DECODE = the inner `while(!block->eob()){ block->read; result.append }` loop
             (decode + collect + interleaved clean-CRC)
  ring_other = WORKER - R_TABLE - R_DECODE (computed externally)
Own sub_depth counter (SUB_OVERLAP) — the sub-spans nest inside the existing WORKER Acc,
so they cannot share its `depth`. Gated RAPIDGZIP_REGION_PROF=1 (same env as the WORKER
partition → one run yields both). OFF == identity. Idempotent.

Depends on rg_region_prof_patch.py having already defined the namespace in ChunkData.hpp
and the WORKER Acc in GzipChunkFetcher.hpp.
"""
import sys, os

SRC_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/rg-build-src/librapidarchive/src/rapidgzip"
CHUNKDATA = os.path.join(SRC_DIR, "ChunkData.hpp")
GZIPCHUNK = os.path.join(SRC_DIR, "chunkdecoding", "GzipChunk.hpp")

# ── 1. Extend the namespace in ChunkData.hpp with sub-region state + SubAcc ──────
SUB_NS = r'''    // ===== R_WORKER SUB-PARTITION (table vs decode) =====
    inline std::atomic<uint64_t> t_cyc{0};
    inline std::atomic<uint64_t> t_calls{0};
    inline std::atomic<uint64_t> d_cyc{0};
    inline std::atomic<uint64_t> d_calls{0};
    inline std::atomic<uint64_t> sub_overlap{0};
    inline thread_local uint32_t sub_depth = 0;
    inline void sub_enter() { if (sub_depth > 0) sub_overlap.fetch_add(1, std::memory_order_relaxed); ++sub_depth; }
    inline void sub_leave() { if (sub_depth > 0) --sub_depth; }
    struct SubAcc { std::atomic<uint64_t>& cyc; std::atomic<uint64_t>& calls; bool on; uint64_t t0;
        SubAcc(std::atomic<uint64_t>& c, std::atomic<uint64_t>& n)
            : cyc(c), calls(n), on(enabled()), t0(0) {
            if (on) { sub_enter(); t0 = __builtin_ia32_rdtsc(); } }
        ~SubAcc() { if (on) { uint64_t dt = __builtin_ia32_rdtsc() - t0; sub_leave();
            cyc.fetch_add(dt, std::memory_order_relaxed);
            calls.fetch_add(1, std::memory_order_relaxed); } }
    };
    // ===== end sub-partition state =====
'''

def patch_chunkdata_ns():
    src = open(CHUNKDATA).read()
    if "R_WORKER SUB-PARTITION (table vs decode)" in src:
        print("CHUNKDATA SUB-NS ALREADY PATCHED"); return
    # Insert sub-state AFTER leave() (which is after enabled()/enter()/the depth
    # thread_local) so SubAcc's reference to enabled() resolves, and t_cyc et al.
    # are declared before the Dumper that prints them.
    anchor = "inline void leave() { if (depth > 0) --depth; }"
    ai = src.find(anchor)
    if ai < 0:
        raise SystemExit("FATAL: namespace leave() anchor not found (run rg_region_prof_patch first)")
    ins = ai + len(anchor)
    src = src[:ins] + "\n" + SUB_NS + src[ins:]

    # Extend the Dumper to print the sub-partition + self-test. Anchor on the WORKER
    # OVERLAP_VIOLATIONS fprintf already present.
    dump_anchor = '"[RG-REGION] OVERLAP_VIOLATIONS=%llu (must be 0) SELF-TEST: overlap==0:%d non-inert:%d -> %s\\n",'
    di = src.find(dump_anchor)
    if di < 0:
        raise SystemExit("FATAL: Dumper OVERLAP fprintf anchor not found")
    # find the end of that fprintf statement (the next ');')
    stmt_end = src.find(");", di)
    stmt_end = src.find("\n", stmt_end) + 1
    SUB_DUMP = r'''            const uint64_t tc=t_cyc.load(), tn=t_calls.load();
            const uint64_t dc=d_cyc.load(), dn=d_calls.load();
            const uint64_t ringOther = (wc >= tc + dc) ? (wc - tc - dc) : 0;
            const uint64_t subov = sub_overlap.load();
            const bool subNonInert = tn>0 && dn>0;
            std::fprintf(stderr,
              "[RG-SUBREGION] R_TABLE cyc=%llu hdr_calls=%llu | R_DECODE cyc=%llu calls=%llu | ring_other cyc=%llu | WORKER cyc=%llu\n",
              (unsigned long long)tc,(unsigned long long)tn,(unsigned long long)dc,(unsigned long long)dn,
              (unsigned long long)ringOther,(unsigned long long)wc);
            std::fprintf(stderr,
              "[RG-SUBREGION] SUB_OVERLAP_VIOLATIONS=%llu (must be 0) SELF-TEST: sub_overlap==0:%d non-inert:%d cons:%d -> %s\n",
              (unsigned long long)subov, subov==0, subNonInert, (wc>=tc+dc),
              (subov==0 && subNonInert && wc>=tc+dc) ? "PASS" : "FAIL");
'''
    src = src[:stmt_end] + SUB_DUMP + src[stmt_end:]
    open(CHUNKDATA, "w").write(src)
    print("CHUNKDATA SUB-NS + DUMPER PATCHED OK")


def patch_gzipchunk():
    src = open(GZIPCHUNK).read()
    if "rapidgzip_region_prof::SubAcc __rgt" in src:
        print("GZIPCHUNK ALREADY PATCHED"); return
    # R_TABLE: wrap block->readHeader (deflate block header = table build).
    t_anchor = "if ( auto error = block->readHeader( *bitReader ); error != Error::NONE ) {"
    ti = src.find(t_anchor)
    if ti < 0:
        raise SystemExit("FATAL: block->readHeader anchor not found")
    # Wrap just the readHeader call in a lambda so the SubAcc destructs right after it.
    t_repl = ("if ( auto error = [&]{ ::rapidgzip_region_prof::SubAcc __rgt{ "
              "::rapidgzip_region_prof::t_cyc, ::rapidgzip_region_prof::t_calls };\n"
              "                    return block->readHeader( *bitReader ); }(); error != Error::NONE ) {")
    src = src[:ti] + t_repl + src[ti + len(t_anchor):]

    # R_DECODE: wrap the inner `while(!block->eob())` loop. Open the scope AFTER the
    # `size_t blockBytesRead{ 0 };` declaration (so blockBytesRead stays in scope for
    # the post-loop `streamBytesRead += blockBytesRead;`), close before that line.
    d_anchor = "            size_t blockBytesRead{ 0 };\n"
    di = src.find(d_anchor)
    if di < 0:
        raise SystemExit("FATAL: blockBytesRead decl anchor not found")
    ins = di + len(d_anchor)
    open_guard = ("            { ::rapidgzip_region_prof::SubAcc __rgd{ "
                  "::rapidgzip_region_prof::d_cyc, ::rapidgzip_region_prof::d_calls };\n")
    src = src[:ins] + open_guard + src[ins:]
    # Close the scope right after the matching while loop. The loop is followed by
    # `streamBytesRead += blockBytesRead;` — insert the closing brace before it.
    close_anchor = "            streamBytesRead += blockBytesRead;"
    ci = src.find(close_anchor, ins)
    if ci < 0:
        raise SystemExit("FATAL: post-loop close anchor (streamBytesRead) not found")
    src = src[:ci] + "            } // end __rgd R_DECODE scope\n" + src[ci:]

    # R_DECODE (ISA-L tail): rg is built WITH_ISAL → the bulk CLEAN decode runs in
    # the ISA-L wrapper functions (inflateWrapper.readStream + result.append), which
    # BYPASS the deflate::Block inner loop above. Wrap their bodies into R_DECODE so
    # rg's decode region is MATCHED to gz (whose pure-Rust R_DECODE includes the clean
    # tail). RAII covers every return. ISA-L fuses table-build into readStream, so
    # these add no R_TABLE (matched: rg R_TABLE = only the deflate bootstrap headers;
    # both tools' R_TABLE+R_DECODE = all decode).
    isal1_anchor = "        inflateWrapper.setFileType( result.configuration.fileType );\n"
    i1 = src.find(isal1_anchor)
    if i1 < 0:
        raise SystemExit("FATAL: decodeChunkWithInexactOffset setFileType anchor not found")
    ins1 = i1 + len(isal1_anchor)
    g1 = ("        ::rapidgzip_region_prof::SubAcc __rgd_isalA{ "
          "::rapidgzip_region_prof::d_cyc, ::rapidgzip_region_prof::d_calls };\n")
    src = src[:ins1] + g1 + src[ins1:]

    isal2_anchor = "        bool stoppingPointReached{ false };\n"
    i2 = src.find(isal2_anchor)
    if i2 < 0:
        raise SystemExit("FATAL: finishDecodeChunkWithInexactOffset stoppingPointReached anchor not found")
    ins2 = i2 + len(isal2_anchor)
    g2 = ("        ::rapidgzip_region_prof::SubAcc __rgd_isalB{ "
          "::rapidgzip_region_prof::d_cyc, ::rapidgzip_region_prof::d_calls };\n")
    src = src[:ins2] + g2 + src[ins2:]

    open(GZIPCHUNK, "w").write(src)
    print("GZIPCHUNK R_TABLE + R_DECODE (+ISA-L tail) PATCHED OK")


if __name__ == "__main__":
    patch_chunkdata_ns()
    patch_gzipchunk()
    print("DONE")
