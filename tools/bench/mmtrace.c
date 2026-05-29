/* mmtrace — LD_PRELOAD memcpy/memmove attribution.
 *
 * Which call sites copy the most BYTES? perf sampling + fat-LTO + a stripped
 * release binary make this nearly impossible to read; this shim answers it
 * exactly. Every memcpy/memmove of >= MMTRACE_MIN bytes (default 1024) is
 * attributed to the first stack frame that lies inside the main executable's
 * text (skipping libc internals), aggregated by that return address. At exit
 * it dumps "hexaddr bytes count" sorted by bytes; symbolize the addresses
 * with `addr2line -ife <binary> <addr>...` (inline-aware, so fat-LTO inline
 * chains resolve).
 *
 * Build:  gcc -O2 -fPIC -shared tools/bench/mmtrace.c -o tools/bench/mmtrace.so -ldl
 * Use:    MMTRACE_OUT=/tmp/mm.txt MMTRACE_MIN=4096 \
 *           LD_PRELOAD=tools/bench/mmtrace.so ./target/release/gzippy -d -c -p8 f.gz >/dev/null
 *         addr2line -ife ./target/release/gzippy $(awk '{print $1}' /tmp/mm.txt)
 *
 * Needs the binary NOT stripped (CARGO_PROFILE_RELEASE_STRIP=false) for
 * symbolization; attribution itself works on a stripped binary too.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>

static void *(*real_memmove)(void *, const void *, size_t);
static void *(*real_memcpy)(void *, const void *, size_t);
static uintptr_t exe_lo, exe_hi;
static size_t min_bytes = 1024;

#define NSLOT 16384
static struct { uintptr_t addr; uint64_t bytes; uint64_t count; } tbl[NSLOT];
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static __thread int in_hook = 0;

static void read_maps(void) {
    char exe[512];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n < 0) return;
    exe[n] = 0;
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return;
    char line[1024], perms[8], path[768];
    uintptr_t lo, hi;
    while (fgets(line, sizeof line, f)) {
        path[0] = 0;
        if (sscanf(line, "%lx-%lx %7s %*x %*x:%*x %*d %767[^\n]", &lo, &hi, perms, path) >= 3) {
            if (perms[2] == 'x' && path[0] && strstr(path, exe)) {
                if (exe_lo == 0 || lo < exe_lo) exe_lo = lo;
                if (hi > exe_hi) exe_hi = hi;
            }
        }
    }
    fclose(f);
}

__attribute__((constructor)) static void init(void) {
    real_memmove = dlsym(RTLD_NEXT, "memmove");
    real_memcpy = dlsym(RTLD_NEXT, "memcpy");
    const char *m = getenv("MMTRACE_MIN");
    if (m) min_bytes = strtoull(m, 0, 10);
    read_maps();
}

static uintptr_t first_exe_frame(void) {
    void *bt[24];
    int n = backtrace(bt, 24);
    for (int i = 2; i < n; i++) { /* skip record()+memcpy frames */
        uintptr_t a = (uintptr_t)bt[i];
        if (a >= exe_lo && a < exe_hi) return a;
    }
    return 0;
}

static void record(size_t n) {
    uintptr_t a = first_exe_frame();
    if (!a) return;
    pthread_mutex_lock(&lock);
    size_t h = (a >> 4) % NSLOT;
    for (size_t i = 0; i < NSLOT; i++) {
        size_t s = (h + i) % NSLOT;
        if (tbl[s].addr == 0) { tbl[s].addr = a; tbl[s].bytes = n; tbl[s].count = 1; break; }
        if (tbl[s].addr == a) { tbl[s].bytes += n; tbl[s].count++; break; }
    }
    pthread_mutex_unlock(&lock);
}

void *memmove(void *d, const void *s, size_t n) {
    if (!real_memmove) { unsigned char *dd = d; const unsigned char *ss = s;
        if (dd < ss) for (size_t i = 0; i < n; i++) dd[i] = ss[i];
        else for (size_t i = n; i-- > 0;) dd[i] = ss[i]; return d; }
    if (n >= min_bytes && !in_hook) { in_hook = 1; record(n); in_hook = 0; }
    return real_memmove(d, s, n);
}

void *memcpy(void *d, const void *s, size_t n) {
    if (!real_memcpy) { unsigned char *dd = d; const unsigned char *ss = s;
        for (size_t i = 0; i < n; i++) dd[i] = ss[i]; return d; }
    if (n >= min_bytes && !in_hook) { in_hook = 1; record(n); in_hook = 0; }
    return real_memcpy(d, s, n);
}

__attribute__((destructor)) static void dump(void) {
    const char *out = getenv("MMTRACE_OUT");
    FILE *f = out ? fopen(out, "w") : stderr;
    if (!f) return;
    for (int k = 0; k < 40; k++) {
        size_t best = NSLOT; uint64_t bb = 0;
        for (size_t i = 0; i < NSLOT; i++)
            if (tbl[i].addr && tbl[i].bytes > bb) { bb = tbl[i].bytes; best = i; }
        if (best == NSLOT) break;
        fprintf(f, "%016lx %12llu %10llu\n", (unsigned long)tbl[best].addr,
                (unsigned long long)tbl[best].bytes, (unsigned long long)tbl[best].count);
        tbl[best].bytes = 0; /* consumed; bb>0 guard skips it next pass */
    }
    if (out) fclose(f);
}
