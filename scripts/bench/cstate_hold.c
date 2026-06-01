/* cstate_hold.c — hold a PM-QoS CPU latency floor of 0us for the bench window.
 *
 * RUNS ON HOST. Compiled by host_lock_and_bench.sh with `cc`.
 *
 * Writing a 0 to /dev/cpu_dma_latency and KEEPING THE FD OPEN tells the kernel
 * "no CPU may enter a C-state deeper than 0us latency" for as long as the fd is
 * held — i.e. it pins cores out of deep idle so they don't pay exit latency
 * mid-measurement. Closing the fd (process exit / kill) releases the floor.
 * This MUST be a real held fd, not a `python -c` one-shot whose fd closes
 * immediately.
 *
 * Prints its own PID then sleeps forever; the parent records the PID and kills
 * it on restore.
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

int main(void) {
    int fd = open("/dev/cpu_dma_latency", O_RDWR);
    if (fd < 0) { perror("open /dev/cpu_dma_latency"); return 1; }
    int32_t zero = 0;
    if (write(fd, &zero, sizeof(zero)) != (ssize_t)sizeof(zero)) {
        perror("write cpu_dma_latency");
        return 1;
    }
    printf("%d\n", (int)getpid());
    fflush(stdout);
    /* Hold the fd open indefinitely. Kill releases it. */
    for (;;) pause();
    return 0;
}
