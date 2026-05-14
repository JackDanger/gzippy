class Gzippy < Formula
  desc "Fastest parallel gzip — drop-in replacement for gzip and gunzip"
  homepage "https://github.com/JackDanger/gzippy"
  version File.read(File.join(__dir__, "..", "VERSION"))
  license "Zlib"

  on_macos do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-apple-darwin.tar.gz"
      sha256 "fa9852822f56696c0b27fd28dbcf0db9c3692f1d1c721cae4ea41a6565b8e181" # AARCH64_APPLE_DARWIN
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-apple-darwin.tar.gz"
      sha256 "534d9a60ef4ce5811450fbc38b2d9a0e8053b3056de9bf8391b6899f3ba5098d" # X86_64_APPLE_DARWIN
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "346b583faccd54a0c46f1666df83946cb796391dc1a5f4c9997ce258e661483b" # AARCH64_UNKNOWN_LINUX_GNU
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "14d51eef71145e75f2fdcb47f3eab2e50449e915dbf2d2ddb61882db37328ac6" # X86_64_UNKNOWN_LINUX_GNU
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
