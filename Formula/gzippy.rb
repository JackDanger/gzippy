class Gzippy < Formula
  desc "Fastest parallel gzip — drop-in replacement for gzip and gunzip"
  homepage "https://github.com/JackDanger/gzippy"
  version File.read(File.join(__dir__, "..", "VERSION")).strip
  license "Zlib"

  on_macos do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-apple-darwin.tar.gz"
      sha256 "84239ae7d440f75615c98b70eb56643b215a4f932c90867a01ad25166662198a" # AARCH64_APPLE_DARWIN
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-apple-darwin.tar.gz"
      sha256 "8f99ac90b2d33a66fc3ea027c10256effc05e8f67a7e7360c2dfdc4ac5be433b" # X86_64_APPLE_DARWIN
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "177e5ccd3abab14c066b01f74eb6e3de2e6dfba83cd2bbd87d77dbd96c5e5b81" # AARCH64_UNKNOWN_LINUX_GNU
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "42cfe9c75e7b510ef7643e74fe5a771433a594f4c9d5ab56b4ad9980e0fb7e40" # X86_64_UNKNOWN_LINUX_GNU
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
