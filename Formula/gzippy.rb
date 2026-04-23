class Gzippy < Formula
  desc "Fastest parallel gzip — drop-in replacement for gzip and gunzip"
  homepage "https://github.com/JackDanger/gzippy"
  version "0.1.1"
  license "Zlib"

  on_macos do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-apple-darwin.tar.gz"
      sha256 "288e883651ab3fc8e5585671199bac28c68c08576698c8c01f4fb0111164c699" # AARCH64_APPLE_DARWIN
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-apple-darwin.tar.gz"
      sha256 "7b6601c110e2c65d42a6fe02723670f91859e6bdefa6dc8548261f94e7f8cc2e" # X86_64_APPLE_DARWIN
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "41ae5a79b26b496ec97423d75295f67b069538a4422c89700a1b4b7464eacdc2" # AARCH64_UNKNOWN_LINUX_GNU
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "cf53108afe96682771894645e4d7d31fec0b43b5a9d722b07359849387fcbf24" # X86_64_UNKNOWN_LINUX_GNU
    end
  end

  # gzippy installs gzip and gunzip — cannot coexist with Homebrew's gzip package
  conflicts_with "gzip", because: "both install `gzip` and `gunzip` commands"

  def install
    bin.install "gzippy"
    bin.install_symlink "gzippy" => "gzip"
    bin.install_symlink "gzippy" => "gunzip"
    bin.install_symlink "gzippy" => "ungzippy"
    bin.install_symlink "gzippy" => "zcat"
  end

  def caveats
    <<~EOS
      gzippy is now your gzip and gunzip.

      If you had Homebrew's gzip installed, it was removed automatically due
      to the conflict. The macOS system gzip at /usr/bin/gzip is untouched.

      All four commands point to the same binary:
        gzip, gunzip, ungzippy, zcat → gzippy
    EOS
  end

  test do
    (testpath/"hello.txt").write "Hello, world!\n"
    system bin/"gzippy", "-k", "hello.txt"
    assert_predicate testpath/"hello.txt.gz", :exist?
    system bin/"gzip", "-d", "-f", "hello.txt.gz"
    assert_equal "Hello, world!\n", (testpath/"hello.txt").read

    # Verify all symlinks resolve
    assert_equal bin/"gzippy", (bin/"gzip").realpath
    assert_equal bin/"gzippy", (bin/"gunzip").realpath
    assert_equal bin/"gzippy", (bin/"ungzippy").realpath
  end
end
