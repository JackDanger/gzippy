#!/usr/bin/env python3
"""AMD-RESIDUAL S2: patch rapidgzip ChunkData.hpp with a symmetric TSC counter
around ChunkData::applyWindow (== gz R_MARKERPP: resolve/narrow + marker-CRC +
subchunk-window). Gated RAPIDGZIP_AW_PROF=1; OFF == identity. Idempotent."""
import sys, re

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/rg-build-src/librapidarchive/src/rapidgzip/ChunkData.hpp"
src = open(PATH).read()
if "RAPIDGZIP_AW_PROF" in src:
    print("ALREADY PATCHED"); sys.exit(0)

# 1) includes after first #include
inc = "#include <atomic>\n#include <cstdlib>\n#include <cstdio>\n#include <cstdint>\n#include <string>\n"
m = re.search(r'^#include[^\n]*\n', src, re.M)
if not m:
    print("FATAL: no #include"); sys.exit(2)
src = src[:m.end()] + inc + src[m.end():]

# 2) global namespace + counters + atexit dumper, after the include block
idx = src.find(inc) + len(inc)
block = (
"// ===== AMD-RESIDUAL S2: applyWindow (marker-postprocess) TSC counter =====\n"
"namespace rapidgzip_aw_prof {\n"
"    inline std::atomic<uint64_t> g_cyc{0};\n"
"    inline std::atomic<uint64_t> g_bytes{0};\n"
"    inline std::atomic<uint64_t> g_calls{0};\n"
"    inline bool enabled() {\n"
"        static const bool on = [](){ const char* e = std::getenv(\"RAPIDGZIP_AW_PROF\");\n"
"            return e && std::string(e) == \"1\"; }();\n"
"        return on;\n"
"    }\n"
"    struct Guard {\n"
"        bool on; uint64_t t0; const size_t& nb;\n"
"        ~Guard() {\n"
"            if (on) {\n"
"                g_cyc.fetch_add(__builtin_ia32_rdtsc() - t0, std::memory_order_relaxed);\n"
"                g_bytes.fetch_add(nb, std::memory_order_relaxed);\n"
"                g_calls.fetch_add(1, std::memory_order_relaxed);\n"
"            }\n"
"        }\n"
"    };\n"
"    struct Dumper {\n"
"        ~Dumper() {\n"
"            if (!enabled()) return;\n"
"            const uint64_t c = g_cyc.load(), b = g_bytes.load(), n = g_calls.load();\n"
"            std::fprintf(stderr,\n"
"                \"[RG-AW-CYCB] calls=%llu cyc=%llu mkbytes=%llu rg_applywindow_cyc_per_mkB=%.4f\\n\",\n"
"                (unsigned long long)n, (unsigned long long)c, (unsigned long long)b,\n"
"                b ? (double)c / (double)b : 0.0);\n"
"        }\n"
"    };\n"
"    inline Dumper g_dumper;\n"
"}\n"
"// ===== end AMD-RESIDUAL S2 counter =====\n\n"
)
src = src[:idx] + block + src[idx:]

# 3) guard at top of applyWindow — anchor on the unique first stmt.
anchor = "const auto markerCount = dataWithMarkersSize();"
ai = src.find(anchor)
if ai < 0:
    print("FATAL: applyWindow anchor not found"); sys.exit(2)
ins_at = ai + len(anchor)
guard = (
"\n        [[maybe_unused]] const bool awProf = ::rapidgzip_aw_prof::enabled();\n"
"        [[maybe_unused]] const uint64_t awT0 = awProf ? __builtin_ia32_rdtsc() : 0;\n"
"        [[maybe_unused]] ::rapidgzip_aw_prof::Guard awGuard{ awProf, awT0, markerCount };\n"
)
src = src[:ins_at] + guard + src[ins_at:]

open(PATH, "w").write(src)
print("PATCHED OK")
