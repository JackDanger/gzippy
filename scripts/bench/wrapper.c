/* Single-binary env-toggle oracle wrapper.
 * Sets ONE env var (or none, for the "change present" arm), then execs the
 * SAME gzippy binary with the SAME argv it received from fulcrum. This makes
 * the A/B a true removal-oracle: identical production binary, env flips the
 * one change. Compiled to a native Mach-O so fulcrum's Gate-0 accepts it.
 *
 * Compile-time -D defines:
 *   GZ        : absolute path to the (clean) gzippy binary  (required)
 *   ENV_NAME  : env var to set in the "removed" arm         (optional)
 *   ENV_VAL   : value for ENV_NAME                          (optional)
 * When ENV_NAME is undefined the wrapper sets nothing = the default/"present" arm.
 */
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char **argv) {
#ifdef ENV_NAME
    setenv(ENV_NAME, ENV_VAL, 1);
#endif
    argv[0] = GZ;           /* argv[1..] (= -d -c -pN corpus) forwarded as-is */
    execv(GZ, argv);
    return 127;             /* execv only returns on failure */
}
