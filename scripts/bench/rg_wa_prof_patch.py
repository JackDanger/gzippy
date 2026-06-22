#!/usr/bin/env python3
"""ZEN2-DECODE-MICROBENCH: patch rapidgzip deflate.hpp with a symmetric
window-absent (marker) decode cyc/B counter, gated on RAPIDGZIP_WA_PROF=1 and
`if constexpr(containsMarkerBytes)`. OFF == identity (one branch). Idempotent."""
import sys, re

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/rg-build-src/librapidarchive/src/rapidgzip/gzip/deflate.hpp"

src = open(PATH).read()
if "RAPIDGZIP_WA_PROF" in src:
    print("ALREADY PATCHED"); sys.exit(0)

# 1) ensure includes (after first #include line)
inc = "#include <atomic>\n#include <cstdlib>\n#include <cstdio>\n#include <cstdint>\n#include <string>\n"
m = re.search(r'^#include[^\n]*\n', src, re.M)
if not m:
    print("FATAL: no #include found"); sys.exit(2)
src = src[:m.end()] + inc + src[m.end():]

# 2) namespace + counters + atexit dumper, inserted at GLOBAL scope right after
#    the include block (before any `namespace` opens).
idx = src.find(inc) + len(inc)
block = (
"// ===== ZEN2-DECODE-MICROBENCH: window-absent (marker) decode cyc/B counter =====\n"
"namespace rapidgzip_wa_prof {\n"
"    inline std::atomic<uint64_t> g_cyc{0};\n"
"    inline std::atomic<uint64_t> g_bytes{0};\n"
"    inline std::atomic<uint64_t> g_calls{0};\n"
"    inline bool enabled() {\n"
"        static const bool on = [](){ const char* e = std::getenv(\"RAPIDGZIP_WA_PROF\");\n"
"            return e && std::string(e) == \"1\"; }();\n"
"        return on;\n"
"    }\n"
"    struct Guard {\n"
"        bool on; uint64_t t0; const size_t& nbr;\n"
"        ~Guard() {\n"
"            if (on) {\n"
"                g_cyc.fetch_add(__builtin_ia32_rdtsc() - t0, std::memory_order_relaxed);\n"
"                g_bytes.fetch_add(nbr, std::memory_order_relaxed);\n"
"                g_calls.fetch_add(1, std::memory_order_relaxed);\n"
"            }\n"
"        }\n"
"    };\n"
"    struct Dumper {\n"
"        ~Dumper() {\n"
"            if (!enabled()) return;\n"
"            const uint64_t c = g_cyc.load(), b = g_bytes.load(), n = g_calls.load();\n"
"            std::fprintf(stderr,\n"
"                \"[RG-WA-CYCB] calls=%llu cyc=%llu bytes=%llu rg_marker_decode_cyc_per_byte=%.4f\\n\",\n"
"                (unsigned long long)n, (unsigned long long)c, (unsigned long long)b,\n"
"                b ? (double)c / (double)b : 0.0);\n"
"        }\n"
"    };\n"
"    inline Dumper g_dumper;\n"
"}\n"
"// ===== end ZEN2-DECODE-MICROBENCH counter =====\n\n"
)
src = src[:idx] + block + src[idx:]

# 3) install the guard inside readInternalCompressedMultiCached. Anchor on the
#    UNIQUE line `auto [symbol, symbolCount] = coding.decode( bitReader );` and
#    insert the guard right after the preceding `size_t nBytesRead{ 0 };`.
uniq = "auto [symbol, symbolCount] = coding.decode( bitReader );"
ui = src.find(uniq)
if ui < 0:
    print("FATAL: multicached body anchor not found"); sys.exit(2)
nb_pat = "size_t nBytesRead{ 0 };"
nb = src.rfind(nb_pat, 0, ui)
if nb < 0:
    print("FATAL: nBytesRead decl not found before body"); sys.exit(2)
ins_at = nb + len(nb_pat)
guard = (
"\n    [[maybe_unused]] const bool waProf = containsMarkerBytes && ::rapidgzip_wa_prof::enabled();\n"
"    [[maybe_unused]] const uint64_t waT0 = waProf ? __builtin_ia32_rdtsc() : 0;\n"
"    [[maybe_unused]] ::rapidgzip_wa_prof::Guard waGuard{ waProf, waT0, nBytesRead };\n"
)
src = src[:ins_at] + guard + src[ins_at:]

open(PATH, "w").write(src)
print("PATCHED OK")
