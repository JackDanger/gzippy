/* memstress.c — co-resident memory-bandwidth STRESSOR for the dist-preload
 * bandwidth-confounder control.
 *
 * Spawns NTHREADS pthreads, each streaming memcpy over a per-thread buffer far
 * larger than LLC (so it hammers the shared L3 + memory controller, the exact
 * resource a speculative dist-entry PRELOAD would compete for). Runs until a
 * SIGTERM/SIGINT or until the optional duration (seconds) elapses. The caller
 * pins this process to sibling/other cores with taskset; the measured kernel
 * stays on its own isolated core, so any Δ change between unstressed and
 * stressed runs isolates bandwidth sensitivity.
 *
 *   build: gcc -O2 -pthread -o memstress memstress.c
 *   run  : taskset -c 0-3,5-15 ./memstress [NTHREADS] [BUF_MB] [SECONDS]
 *          (SECONDS<=0 or omitted => run until signalled)
 */
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdatomic.h>

static atomic_int g_stop = 0;
static size_t g_buf_bytes = 0;

static void on_sig(int s) { (void)s; atomic_store(&g_stop, 1); }

static void *worker(void *arg) {
    (void)arg;
    size_t n = g_buf_bytes;
    char *a = malloc(n), *b = malloc(n);
    if (!a || !b) { fprintf(stderr, "alloc fail\n"); return NULL; }
    memset(a, 1, n); memset(b, 2, n);
    volatile unsigned long sink = 0;
    while (!atomic_load(&g_stop)) {
        memcpy(b, a, n);
        sink += (unsigned long)b[(sink + 7) % n];
        memcpy(a, b, n);
        sink += (unsigned long)a[(sink + 13) % n];
    }
    free(a); free(b);
    return (void *)sink;
}

int main(int argc, char **argv) {
    int nthreads = argc > 1 ? atoi(argv[1]) : 8;
    int buf_mb   = argc > 2 ? atoi(argv[2]) : 96;   /* per-thread, >> 30MB LLC */
    int seconds  = argc > 3 ? atoi(argv[3]) : 0;
    if (nthreads < 1) nthreads = 1;
    g_buf_bytes = (size_t)buf_mb * 1024 * 1024;

    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);

    pthread_t th[256];
    if (nthreads > 256) nthreads = 256;
    fprintf(stderr, "memstress: %d threads x %d MB, %s\n",
            nthreads, buf_mb, seconds > 0 ? "timed" : "until-signal");
    for (int i = 0; i < nthreads; i++)
        pthread_create(&th[i], NULL, worker, NULL);

    if (seconds > 0) {
        sleep(seconds);
        atomic_store(&g_stop, 1);
    }
    for (int i = 0; i < nthreads; i++)
        pthread_join(th[i], NULL);
    return 0;
}
