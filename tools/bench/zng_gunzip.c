/* zng_gunzip — minimal zlib-ng gzip decompressor front-end for the
 * competitive benchmark matrix (tools/bench/matrix.sh).
 *
 * Uses the classic zlib inflate API with windowBits = MAX_WBITS|16 (gzip),
 * looping inflateReset2 across members so multi-member / BGZF decode fully.
 * Compiled and linked against a zlib-ng build (header + libz.so from the
 * zlib-ng cmake build dir) so this measures zlib-ng, not stock zlib.
 *
 * Reads file into memory, streams inflate into a reused 1 MiB output buffer
 * written to stdout — same shape as the gzippy / rapidgzip CLIs.
 *
 * Build (against a zlib-ng build at $ZNG):
 *   gcc -O3 -march=native -I$ZNG zng_gunzip.c $ZNG/libz.a -o zng_gunzip
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include "zlib.h"

#define OUTCHUNK (1u << 20)

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: zng_gunzip <file.gz>\n"); return 2; }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct stat st; fstat(fd, &st);
    size_t in_len = (size_t)st.st_size;
    unsigned char *in = mmap(NULL, in_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (in == MAP_FAILED) { perror("mmap"); return 2; }

    unsigned char *out = malloc(OUTCHUNK);
    z_stream s; memset(&s, 0, sizeof(s));
    if (inflateInit2(&s, MAX_WBITS | 16) != Z_OK) { fprintf(stderr, "init\n"); return 1; }
    s.next_in = in; s.avail_in = (uInt)in_len;

    for (;;) {
        s.next_out = out; s.avail_out = OUTCHUNK;
        int r = inflate(&s, Z_NO_FLUSH);
        size_t produced = OUTCHUNK - s.avail_out;
        if (produced) (void)write(1, out, produced);
        if (r == Z_STREAM_END) {
            if (s.avail_in == 0) break;
            /* another concatenated member */
            if (inflateReset2(&s, MAX_WBITS | 16) != Z_OK) break;
            continue;
        }
        if (r != Z_OK && r != Z_BUF_ERROR) { fprintf(stderr, "inflate %d\n", r); return 1; }
        if (r == Z_BUF_ERROR && s.avail_in == 0) break;
    }
    inflateEnd(&s);
    return 0;
}
