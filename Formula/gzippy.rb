class Gzippy < Formula
  desc "Fastest parallel gzip — drop-in replacement for gzip and gunzip"
  homepage "https://github.com/JackDanger/gzippy"
  version "0.1.2"
  license "Zlib"

  on_macos do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-apple-darwin.tar.gz"
      sha256 "b5d85d8bf4d46337a872ae35e9b14a6fa2e53aece1d26424e9390fc8478977ab" # AARCH64_APPLE_DARWIN
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-apple-darwin.tar.gz"
      sha256 "edb6d6b8addc6c590411993cdfd457c9e8e143af791cc5c25afa5a92288aae1b" # X86_64_APPLE_DARWIN
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "1ed2ed106bfcc86ee4f71d39364c3bfa4599e319d422202789fd050b4d27a947" # AARCH64_UNKNOWN_LINUX_GNU
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "574ec915a9cc449431d7c08a55a8fae23111fafd4b68d8f42a659cc483ba046e" # X86_64_UNKNOWN_LINUX_GNU
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
