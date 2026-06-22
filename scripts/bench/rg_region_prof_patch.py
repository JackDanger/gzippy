#!/usr/bin/env python3
"""AMD WALL-DECOMP S2: patch rapidgzip with a SYMMETRIC, exclusive-leaf, 3-region
TSC partition matched to gz's region_prof (GZIPPY_REGION_PROF):
  WORKER   = GzipChunkFetcher::decodeBlock   (decode + clean-CRC(append) + collect)
  MARKERPP = ChunkData::applyWindow          (marker-resolve + marker-CRC)
  OUTPUT   = writeAll(...) in tools/rapidgzip.cpp (consumer write to fd)
ONE namespace, ONE shared per-thread depth counter -> OVERLAP_VIOLATIONS (cursor-agent
review fix #1: proves regions are exclusive non-nested leaves; conservation cannot pass
spuriously via region cancellation). Gated RAPIDGZIP_REGION_PROF=1; OFF == identity.
Idempotent. Run with the two source paths as needed; patches whichever files it finds.

Usage:
  rg_region_prof_patch.py <librapidarchive_src_dir> <tools_rapidgzip_cpp>
Defaults to the solvency box layout.
"""
import sys, re, os

SRC_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/rg-build-src/librapidarchive/src/rapidgzip"
CPP = sys.argv[2] if len(sys.argv) > 2 else \
    "/root/rg-build-src/src/tools/rapidgzip.cpp"

FETCHER = os.path.join(SRC_DIR, "GzipChunkFetcher.hpp")
CHUNKDATA = os.path.join(SRC_DIR, "ChunkData.hpp")

INC = ("#include <atomic>\n#include <cstdlib>\n#include <cstdio>\n"
       "#include <cstdint>\n#include <string>\n")

# Namespace definition lives in GzipChunkFetcher.hpp (included by the tool TU);
# the other sites reference it via the global namespace.
NS = r'''// ===== AMD WALL-DECOMP S2: symmetric 3-region exclusive-TSC partition =====
namespace rapidgzip_region_prof {
    inline std::atomic<uint64_t> w_cyc{0};
    inline std::atomic<uint64_t> w_calls{0};
    inline std::atomic<uint64_t> w_bytes{0};
    inline std::atomic<uint64_t> m_cyc{0};
    inline std::atomic<uint64_t> m_calls{0};
    inline std::atomic<uint64_t> m_bytes{0};
    inline std::atomic<uint64_t> o_cyc{0};
    inline std::atomic<uint64_t> o_calls{0};
    inline std::atomic<uint64_t> o_bytes{0};
    inline std::atomic<uint64_t> overlap{0};
    inline thread_local uint32_t depth = 0;
    inline bool enabled() {
        static const bool on = [](){ const char* e = std::getenv("RAPIDGZIP_REGION_PROF");
            return e && std::string(e) == "1"; }();
        return on;
    }
    inline void enter() { if (depth > 0) overlap.fetch_add(1, std::memory_order_relaxed); ++depth; }
    inline void leave() { if (depth > 0) --depth; }
    struct Acc { std::atomic<uint64_t>& cyc; std::atomic<uint64_t>& calls;
                 std::atomic<uint64_t>& bytes; uint64_t nb; bool on; uint64_t t0;
        Acc(std::atomic<uint64_t>& c, std::atomic<uint64_t>& n, std::atomic<uint64_t>& b, uint64_t bytes_)
            : cyc(c), calls(n), bytes(b), nb(bytes_), on(enabled()), t0(0) {
            if (on) { enter(); t0 = __builtin_ia32_rdtsc(); } }
        ~Acc() { if (on) { uint64_t dt = __builtin_ia32_rdtsc() - t0; leave();
            cyc.fetch_add(dt, std::memory_order_relaxed);
            calls.fetch_add(1, std::memory_order_relaxed);
            bytes.fetch_add(nb, std::memory_order_relaxed); } }
    };
    struct Dumper {
        ~Dumper() {
            if (!enabled()) return;
            const uint64_t wc=w_cyc.load(), wn=w_calls.load(), wb=w_bytes.load();
            const uint64_t mc=m_cyc.load(), mn=m_calls.load(), mb=m_bytes.load();
            const uint64_t oc=o_cyc.load(), on_=o_calls.load(), ob=o_bytes.load();
            const uint64_t ov=overlap.load();
            const bool nonInert = wn>0 && mn>0 && on_>0;
            std::fprintf(stderr,
              "[RG-REGION] WORKER cyc=%llu calls=%llu bytes=%llu | MARKERPP cyc=%llu calls=%llu mkbytes=%llu | OUTPUT cyc=%llu calls=%llu bytes=%llu\n",
              (unsigned long long)wc,(unsigned long long)wn,(unsigned long long)wb,
              (unsigned long long)mc,(unsigned long long)mn,(unsigned long long)mb,
              (unsigned long long)oc,(unsigned long long)on_,(unsigned long long)ob);
            std::fprintf(stderr,
              "[RG-REGION] OVERLAP_VIOLATIONS=%llu (must be 0) SELF-TEST: overlap==0:%d non-inert:%d -> %s\n",
              (unsigned long long)ov, ov==0, nonInert,
              (ov==0 && nonInert) ? "PASS" : "FAIL");
        }
    };
    inline Dumper g_dumper;
}
// ===== end AMD WALL-DECOMP S2 partition =====

'''


def add_includes(src):
    m = re.search(r'^#include[^\n]*\n', src, re.M)
    if not m:
        raise SystemExit("FATAL: no #include in source")
    return src[:m.end()] + INC + src[m.end():]


def patch_fetcher():
    src = open(FETCHER).read()
    if "rapidgzip_region_prof::Acc __rgw" in src:
        print("FETCHER ALREADY PATCHED"); return
    # WORKER guard: top of static decodeBlock body. Anchor = its unique first stmt.
    # Namespace is defined in ChunkData.hpp (included at line 27), so it is visible.
    anchor = "if ( chunkDataConfiguration.fileType == FileType::BZIP2 ) {"
    ai = src.find(anchor)
    if ai < 0:
        raise SystemExit("FATAL: decodeBlock body anchor not found")
    guard = "::rapidgzip_region_prof::Acc __rgw{ ::rapidgzip_region_prof::w_cyc, ::rapidgzip_region_prof::w_calls, ::rapidgzip_region_prof::w_bytes, 0 };\n        "
    src = src[:ai] + guard + src[ai:]
    open(FETCHER, "w").write(src)
    print("FETCHER PATCHED OK")


def patch_chunkdata():
    src = open(CHUNKDATA).read()
    if "rapidgzip_region_prof" in src:
        print("CHUNKDATA ALREADY PATCHED"); return
    # ChunkData.hpp is the lowest-level header (included by GzipChunkFetcher.hpp and
    # transitively by the tool) -> define the namespace HERE so all three sites see it.
    src = add_includes(src)
    idx = src.find(INC) + len(INC)
    src = src[:idx] + NS + src[idx:]
    # MARKERPP guard at top of applyWindow. Anchor on the unique first stmt.
    anchor = "const auto markerCount = dataWithMarkersSize();"
    ai = src.find(anchor)
    if ai < 0:
        raise SystemExit("FATAL: applyWindow anchor not found")
    ins_at = ai + len(anchor)
    guard = ("\n        ::rapidgzip_region_prof::Acc __rgm{ ::rapidgzip_region_prof::m_cyc, "
             "::rapidgzip_region_prof::m_calls, ::rapidgzip_region_prof::m_bytes, "
             "(uint64_t)markerCount };")
    src = src[:ins_at] + guard + src[ins_at:]
    open(CHUNKDATA, "w").write(src)
    print("CHUNKDATA PATCHED OK")


def patch_cpp():
    src = open(CPP).read()
    if "rapidgzip_region_prof::Acc __rgo" in src:
        print("CPP ALREADY PATCHED"); return
    anchor = "const auto errorCode = writeAll( chunkData, fileDescriptor, offsetInChunk, dataToWriteSize );"
    ai = src.find(anchor)
    if ai < 0:
        raise SystemExit("FATAL: writeAll anchor not found in tools/rapidgzip.cpp")
    # Lambda so the OUTPUT Acc destructs immediately after writeAll returns while the
    # errorCode value still escapes to the enclosing scope (exclusive leaf, no scope leak).
    repl = ("const auto errorCode = [&]{ ::rapidgzip_region_prof::Acc __rgo{ "
            "::rapidgzip_region_prof::o_cyc, ::rapidgzip_region_prof::o_calls, "
            "::rapidgzip_region_prof::o_bytes, (uint64_t)dataToWriteSize };\n                    "
            "return writeAll( chunkData, fileDescriptor, offsetInChunk, dataToWriteSize ); }();")
    src = src[:ai] + repl + src[ai:]
    open(CPP, "w").write(src)
    print("CPP PATCHED OK")


if __name__ == "__main__":
    patch_chunkdata()  # defines the namespace; must run first
    patch_fetcher()
    patch_cpp()
    print("DONE")
