class Gzippy < Formula
  desc "Fastest parallel gzip — drop-in replacement for gzip and gunzip"
  homepage "https://github.com/JackDanger/gzippy"
  version "0.7.1"
  license "Zlib"

  on_macos do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-apple-darwin.tar.gz"
      sha256 "adfdff22b57436dd31cfa0804129ee7c0afa81697c203c06294ab390eafe0257" # AARCH64_APPLE_DARWIN
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-apple-darwin.tar.gz"
      sha256 "c2c3617670c6fe02f0b81bae43e0b8c2a990f20c2d24e6627c974a7059d301ce" # X86_64_APPLE_DARWIN
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "cf34ae3a7cbd1038bf46476439dec6c3668a643d5dd59dc9b3f3999895d61688" # AARCH64_UNKNOWN_LINUX_GNU
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "29e54d0189f89d6bd8c5be4b44e9a6a43a341bf68b2e53e18934f15f6b1f03fc" # X86_64_UNKNOWN_LINUX_GNU
    end
  end

  # gzippy installs gzip and gunzip — cannot coexist with Homebrew's gzip package
  conflicts_with "gzip", because: "both install `gzip` and `gunzip` commands"

  def install
    bin.install "gzippy"
    bin.install_symlink "gzippy" => "gzip"
    bin.install_symlink "gzippy" => "gunzip"
    bin.install_symlink "gzippy" => "gzcat"
    bin.install_symlink "gzippy" => "zcat"
    bin.install_symlink "gzippy" => "ungzippy"
    man1.install "man/gzippy.1" if File.exist?("man/gzippy.1")
    man5.install "man/gzippy-format.5" if File.exist?("man/gzippy-format.5")
    man7.install "man/gzippy-tuning.7" if File.exist?("man/gzippy-tuning.7")
  end

  def caveats
    <<~EOS
      gzip, gunzip, gzcat, zcat, ungzippy → gzippy

      Homebrew's bin precedes /usr/bin in PATH, so these shadow the system
      gzip automatically. The system /usr/bin/gzip is untouched.
    EOS
  end

  test do
    (testpath/"hello.txt").write "Hello, world!\n"
    system bin/"gzippy", "-k", "hello.txt"
    assert_predicate testpath/"hello.txt.gz", :exist?
    system bin/"gzip", "-d", "-f", "hello.txt.gz"
    assert_equal "Hello, world!\n", (testpath/"hello.txt").read

    assert_equal bin/"gzippy", (bin/"gzip").realpath
    assert_equal bin/"gzippy", (bin/"gunzip").realpath
    assert_equal bin/"gzippy", (bin/"gzcat").realpath
    assert_equal bin/"gzippy", (bin/"zcat").realpath
    assert_equal bin/"gzippy", (bin/"ungzippy").realpath
  end
end
