#!/usr/bin/env bash
# Update AUR packages and push to aur.archlinux.org.
# Usage: update-aur.sh [<version>] [<x86_64-sha256>] [<aarch64-sha256>]
#
# If version not provided, reads from VERSION file.
# Requires: AUR_SSH_KEY env var (path to SSH private key registered on aur.archlinux.org)
# Or called from CI where ~/.ssh/aur is already configured.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-$(cat "$REPO_ROOT/VERSION")}"
X86_SHA="${2:?x86_64 sha256 required}"
ARM_SHA="${3:?aarch64 sha256 required}"

update_and_push() {
    local pkgname="$1"
    local workdir="/tmp/aur-${pkgname}"

    rm -rf "$workdir"
    git clone "ssh://aur@aur.archlinux.org/${pkgname}.git" "$workdir"
    cp "$REPO_ROOT/aur/${pkgname}/PKGBUILD" "$workdir/PKGBUILD"

    cd "$workdir"

    # Stamp version + URLs
    sed -i "s/pkgver=.*/pkgver=$VERSION/" PKGBUILD
    sed -i "s|/vVERSION/|/v$VERSION/|g" PKGBUILD

    # Stamp hashes
    sed -i "s/sha256sums_x86_64=.*/sha256sums_x86_64=('$X86_SHA')/" PKGBUILD
    sed -i "s/sha256sums_aarch64=.*/sha256sums_aarch64=('$ARM_SHA')/" PKGBUILD

    # Generate .SRCINFO via Docker (works on any host).
    # makepkg refuses to run as root, so we create an unprivileged "builder"
    # user inside the container and run as that user. Pass the host UID/GID
    # in so the container can chown files BACK before we exit — otherwise
    # the host-side `git config` step that follows hits permission-denied
    # on `.git/config` (it's owned by the container's builder UID, not the
    # runner's user).
    local host_uid host_gid
    host_uid=$(id -u)
    host_gid=$(id -g)
    docker run --rm \
        -v "$workdir":/pkg \
        -w /pkg \
        -e HOST_UID="$host_uid" \
        -e HOST_GID="$host_gid" \
        archlinux:latest \
        bash -c '
            pacman -Sy --noconfirm base-devel 2>/dev/null
            useradd -m builder
            chown -R builder /pkg
            su builder -c "makepkg --printsrcinfo" > /pkg/.SRCINFO
            chown -R "$HOST_UID:$HOST_GID" /pkg
        '

    git config user.email "gzippy@jackdanger.com"
    git config user.name "gzippy-bot"
    git add PKGBUILD .SRCINFO
    git diff --cached --quiet && echo "$pkgname: nothing changed" && return
    git commit -m "Update to v$VERSION"
    git push
    echo "$pkgname: pushed v$VERSION"
}

update_and_push gzippy-bin
update_and_push gzippy-replace-gzip
