/* ld_gunzip — minimal libdeflate gzip decompressor front-end for the
 * competitive benchmark matrix (tools/bench/matrix.sh).
 *
 * Reads a .gz file fully into memory, decompresses ALL members (gzip
 * multi-member / BGZF safe) via libdeflate_gzip_decompress_ex, writes the
 * decoded bytes to stdout. Apples-to-apples with the gzippy / rapidgzip
 * CLIs: a process that reads a file, decompresses, writes to (redirected)
 * stdout. Single-threaded — libdeflate has no threads; that is the point,
 * we must not be slower than it at ANY thread count, T1 included.
 *
 * libdeflate.h is not installed on the bench box; the ABI is stable, so we
 * declare the few prototypes we use. Build:
 *   gcc -O3 -march=native ld_gunzip.c -ldeflate -o ld_gunzip
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

struct libdeflate_decompressor;
struct libdeflate_decompressor *libdeflate_alloc_decompressor(void);
void libdeflate_free_decompressor(struct libdeflate_decompressor *);
/* result: 0 = SUCCESS, 1 = BAD_DATA, 2 = SHORT_OUTPUT, 3 = INSUFFICIENT_SPACE */
int libdeflate_gzip_decompress_ex(struct libdeflate_decompressor *d,
                                  const void *in, size_t in_nbytes,
                                  void *out, size_t out_nbytes_avail,
                                  size_t *actual_in_nbytes_ret,
                                  size_t *actual_out_nbytes_ret);

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: ld_gunzip <file.gz>\n"); return 2; }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct stat st; fstat(fd, &st);
    size_t in_len = (size_t)st.st_size;
    unsigned char *in = mmap(NULL, in_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (in == MAP_FAILED) { perror("mmap"); return 2; }

    /* Generous output buffer; grow on INSUFFICIENT_SPACE. */
    size_t out_cap = in_len * 8 + (1u << 20);
    unsigned char *out = malloc(out_cap);
    struct libdeflate_decompressor *d = libdeflate_alloc_decompressor();

    size_t in_pos = 0;
    while (in_pos < in_len) {
        size_t consumed = 0, produced = 0;
        int r = libdeflate_gzip_decompress_ex(d, in + in_pos, in_len - in_pos,
                                              out, out_cap, &consumed, &produced);
        if (r == 3) { /* INSUFFICIENT_SPACE */
            out_cap *= 2; out = realloc(out, out_cap); continue;
        }
        if (r != 0) { fprintf(stderr, "libdeflate error %d at in_pos %zu\n", r, in_pos); return 1; }
        if (consumed == 0) break; /* trailing padding / no progress */
        (void)write(1, out, produced);
        in_pos += consumed;
    }
    libdeflate_free_decompressor(d);
    return 0;
}
