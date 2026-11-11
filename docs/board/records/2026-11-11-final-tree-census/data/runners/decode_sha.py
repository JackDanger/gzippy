# Independent decode oracle: python3 zlib (NOT our decoder, NOT a shell
# gzippy alias). Chains every gzip member in the stream and sha256-checks the
# concatenated decompressed bytes against the original input's sha.
import sys
import zlib
import hashlib


def main(argv):
    stream_path, expect_sha = argv[1], argv[2]
    h = hashlib.sha256()
    with open(stream_path, "rb") as f:
        data = f.read()
    members = 0
    while data:
        d = zlib.decompressobj(31)
        try:
            out = d.decompress(data)
            out += d.flush()
        except zlib.error as exc:
            print("PYDECFAIL zlib error: %s" % exc, file=sys.stderr)
            return 1
        h.update(out)
        members += 1
        data = d.unused_data
        if not d.eof and data != b"":
            print("PYDECFAIL decoder did not report member end", file=sys.stderr)
            return 2
    got = h.hexdigest()
    if got != expect_sha:
        print(
            "PYDECFAIL sha mismatch got=%s want=%s (members=%d)"
            % (got[:16], expect_sha[:16], members),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
