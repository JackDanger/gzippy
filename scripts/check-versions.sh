#!/usr/bin/env bash
# Verify VERSION file matches Cargo.toml and Formula/gzippy.rb
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VERSION_FILE=$(cat VERSION)
CARGO_VERSION=$(grep "^version = " Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
FORMULA_VERSION=$(grep "version \"" Formula/gzippy.rb | sed 's/.*version "\(.*\)".*/\1/')

echo "Checking version consistency:"
echo "  VERSION file: $VERSION_FILE"
echo "  Cargo.toml:   $CARGO_VERSION"
echo "  Formula:      $FORMULA_VERSION"

ERRORS=0

if [ "$VERSION_FILE" != "$CARGO_VERSION" ]; then
    echo "❌ VERSION file ($VERSION_FILE) doesn't match Cargo.toml ($CARGO_VERSION)"
    ERRORS=$((ERRORS + 1))
fi

if [ "$VERSION_FILE" != "$FORMULA_VERSION" ]; then
    echo "❌ VERSION file ($VERSION_FILE) doesn't match Formula ($FORMULA_VERSION)"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -eq 0 ]; then
    echo "✅ All versions match"
    exit 0
else
    echo ""
    echo "To fix, update VERSION file then run:"
    echo "  cargo build  # updates Cargo.toml"
    echo "  sed -i 's/version \".*\"/version \"$VERSION_FILE\"/' Formula/gzippy.rb"
    exit 1
fi
