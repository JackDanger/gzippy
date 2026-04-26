class Gzippy < Formula
  desc "Fastest parallel gzip — drop-in replacement for gzip and gunzip"
  homepage "https://github.com/JackDanger/gzippy"
  version File.read(File.join(__dir__, "..", "VERSION")).strip rescue "0.3.0"
  license "Zlib"

  on_macos do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-apple-darwin.tar.gz"
      sha256 "eec908f517cbda2250d0f84ca9af3b6fe9a8f61c323b08e6707bfee2c8647ed0" # AARCH64_APPLE_DARWIN
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-apple-darwin.tar.gz"
      sha256 "5e223e8180bbf230b781290489b8ec64b1d7e173f547b1a312a6dd37ef9dd945" # X86_64_APPLE_DARWIN
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "8c9431612db7de6de3ae9ec501df5cfdafccc2d6e71910a176b9bc95d33652b0" # AARCH64_UNKNOWN_LINUX_GNU
    end
    on_intel do
      url "https://github.com/JackDanger/gzippy/releases/download/v#{version}/gzippy-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "cce4c3904f46feb529de518527058263d84b5db373eca798656d029b98d005f0" # X86_64_UNKNOWN_LINUX_GNU
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
