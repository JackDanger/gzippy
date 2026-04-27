#!/usr/bin/env bash
# Install gzippy — detects platform and uses the right package manager.
# Usage: curl -fsSL https://raw.githubusercontent.com/JackDanger/gzippy/main/scripts/install.sh | bash
set -euo pipefail

REPO="JackDanger/gzippy"
APT_REPO="https://jackdanger.github.io/gzippy"

info() { printf '%s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

install_macos() {
    if command -v brew &>/dev/null; then
        # Remove conflicting Homebrew gzip if present
        if brew list gzip &>/dev/null 2>&1; then
            info "Removing Homebrew gzip (gzippy replaces it)..."
            brew uninstall gzip
        fi
        brew tap jackdanger/gzippy "https://github.com/$REPO" 2>/dev/null || true
        brew install jackdanger/gzippy/gzippy
    else
        info "Homebrew not found — installing binary to /usr/local/bin..."
        install_macos_binary
    fi
}

install_macos_binary() {
    local target
    case $(uname -m) in
        arm64|aarch64) target="aarch64-apple-darwin" ;;
        x86_64)        target="x86_64-apple-darwin" ;;
        *) die "Unsupported architecture: $(uname -m)" ;;
    esac

    local latest
    latest=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
        | grep '"tag_name"' | head -1 | cut -d'"' -f4)
    [ -n "$latest" ] || die "Could not determine latest release"

    local url="https://github.com/$REPO/releases/download/$latest/gzippy-$target.tar.gz"
    info "Downloading gzippy $latest ($target)..."

    local tmp; tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

    curl -fsSL "$url"          -o "$tmp/gzippy.tar.gz"
    curl -fsSL "${url}.sha256" -o "$tmp/expected.sha256"

    local expected actual
    expected=$(cat "$tmp/expected.sha256")
    actual=$(shasum -a 256 "$tmp/gzippy.tar.gz" | awk '{print $1}')
    [ "$expected" = "$actual" ] || die "SHA256 mismatch — download may be corrupted"

    tar -xz -C "$tmp" -f "$tmp/gzippy.tar.gz"

    # /usr/local/bin precedes /usr/bin in macOS's default PATH (/etc/paths),
    # so installing here shadows the system gzip without touching PATH.
    local dir=/usr/local/bin
    if [ ! -w "$dir" ]; then
        info "Installing to $dir (requires sudo)..."
        sudo mkdir -p "$dir"
        sudo mv    "$tmp/gzippy" "$dir/gzippy"
        sudo ln -sf gzippy "$dir/gzip"
        sudo ln -sf gzippy "$dir/gunzip"
        sudo ln -sf gzippy "$dir/zcat"
        sudo ln -sf gzippy "$dir/gzcat"
        sudo ln -sf gzippy "$dir/ungzippy"
    else
        mv    "$tmp/gzippy" "$dir/gzippy"
        ln -sf gzippy "$dir/gzip"
        ln -sf gzippy "$dir/gunzip"
        ln -sf gzippy "$dir/zcat"
        ln -sf gzippy "$dir/gzcat"
        ln -sf gzippy "$dir/ungzippy"
    fi

    info "Installed to $dir — gzip, gunzip, gzcat, zcat, ungzippy → gzippy"
    info "System /usr/bin/gzip is untouched"
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

install_arch() {
    if command -v yay &>/dev/null; then
        yay -S --noconfirm gzippy-bin
    elif command -v paru &>/dev/null; then
        paru -S --noconfirm gzippy-bin
    else
        die "No AUR helper found. Install yay or paru, then: yay -S gzippy-bin
Or see: https://aur.archlinux.org/packages/gzippy-bin"
    fi
}

install_linux_binary() {
    local target
    case $(uname -m) in
        x86_64)        target="x86_64-unknown-linux-gnu" ;;
        aarch64|arm64) target="aarch64-unknown-linux-gnu" ;;
        *) die "Unsupported architecture: $(uname -m)" ;;
    esac

    local latest
    latest=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
        | grep '"tag_name"' | head -1 | cut -d'"' -f4)
    [ -n "$latest" ] || die "Could not determine latest release"

    local url="https://github.com/$REPO/releases/download/$latest/gzippy-$target.tar.gz"
    info "Downloading gzippy $latest ($target)..."

    local tmp; tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

    curl -fsSL "$url"          -o "$tmp/gzippy.tar.gz"
    curl -fsSL "${url}.sha256" -o "$tmp/expected.sha256"

    local expected actual
    expected=$(cat "$tmp/expected.sha256")
    actual=$(sha256sum "$tmp/gzippy.tar.gz" | awk '{print $1}')
    [ "$expected" = "$actual" ] || die "SHA256 mismatch — download may be corrupted"

    tar -xz -C "$tmp" -f "$tmp/gzippy.tar.gz"

    local dir="$HOME/.local/bin"
    mkdir -p "$dir"
    mv "$tmp/gzippy" "$dir/gzippy"
    info "Installed to $dir/gzippy"
    echo ":$PATH:" | grep -q ":$dir:" \
        || info "  Add to PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
}

OS=$(uname -s)
if [ "$OS" = "Darwin" ]; then
    install_macos
elif [ -f /etc/arch-release ] \
     || grep -qi "arch linux" /etc/os-release 2>/dev/null; then
    install_arch
elif [ -f /etc/debian_version ] \
     || grep -qi "debian\|ubuntu" /etc/os-release 2>/dev/null; then
    install_apt
else
    install_linux_binary
fi

info ""
info "✓ $(gzippy --version 2>/dev/null || echo 'gzippy installed')"
