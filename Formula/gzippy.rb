class Gzippy < Formula
  desc "Fastest parallel gzip — drop-in replacement for gzip and gunzip"
  homepage "https://github.com/JackDanger/gzippy"
  version "0.6.0"
  license "Zlib"

  on_macos do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-apple-darwin.tar.gz"
      sha256 "69b1e97abb674dd4489f8bfd3c36ae0a37d05c32d7571aff8c48c0d3f6c28311" # AARCH64_APPLE_DARWIN
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-apple-darwin.tar.gz"
      sha256 "1cc2961f0f4670c5e207fea978341a24fad35800ce52bb7190ac611eb653037a" # X86_64_APPLE_DARWIN
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "ea48f31b5b6014b0a160ec78f5fe95dcd9b7468c5aad04d3c2eabd8deed5596f" # AARCH64_UNKNOWN_LINUX_GNU
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "5d8f08dc76eeda7ace346ce1892648c3479008202b9d0457a516ab30d28d5e58" # X86_64_UNKNOWN_LINUX_GNU
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
