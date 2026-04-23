#!/usr/bin/env bash
# Install gzippy — detects platform and uses the right package manager.
# Usage: curl -fsSL https://raw.githubusercontent.com/JackDanger/gzippy/main/install.sh | bash
set -euo pipefail

REPO="JackDanger/gzippy"
APT_REPO="https://jackdanger.github.io/gzippy"

info()  { printf '%s\n' "$*"; }
die()   { printf 'error: %s\n' "$*" >&2; exit 1; }

install_macos() {
    command -v brew &>/dev/null || die "Homebrew not found. Install from https://brew.sh"
    brew tap jackdanger/gzippy "https://github.com/$REPO" 2>/dev/null || true
    brew install jackdanger/gzippy/gzippy
}

install_apt() {
    command -v curl &>/dev/null || sudo apt-get install -y curl gpg
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL "$APT_REPO/gzippy-signing-key.asc" \
        | gpg --dearmor \
        | sudo tee /etc/apt/keyrings/gzippy.gpg >/dev/null
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/gzippy.gpg] $APT_REPO stable main" \
        | sudo tee /etc/apt/sources.list.d/gzippy.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y gzippy
}

install_binary() {
    local arch os target
    arch=$(uname -m)
    os=$(uname -s)
    case "$arch" in
        x86_64)        arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *) die "Unsupported architecture: $arch" ;;
    esac
    case "$os" in
        Linux)  target="${arch}-unknown-linux-gnu" ;;
        Darwin) target="${arch}-apple-darwin" ;;
        *) die "Unsupported OS: $os" ;;
    esac

    local latest
    latest=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
        | grep '"tag_name"' | head -1 | cut -d'"' -f4)
    [ -n "$latest" ] || die "Could not determine latest release"

    local url="https://github.com/$REPO/releases/download/$latest/gzippy-$target.tar.gz"
    info "Downloading gzippy $latest ($target)..."

    local tmp
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    curl -fsSL "$url" | tar -xz -C "$tmp"

    local dir="$HOME/.local/bin"
    mkdir -p "$dir"
    mv "$tmp/gzippy" "$dir/"
    info "Installed to $dir/gzippy"
    echo ":$PATH:" | grep -q ":$dir:" \
        || info "  Add to PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
}

OS=$(uname -s)
if [ "$OS" = "Darwin" ]; then
    install_macos
elif [ -f /etc/debian_version ] \
     || grep -qi "debian\|ubuntu" /etc/os-release 2>/dev/null; then
    install_apt
else
    install_binary
fi

info ""
info "✓ $(gzippy --version 2>/dev/null || echo 'gzippy installed')"
