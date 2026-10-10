#!/usr/bin/env bash
# Build the Arch package for AI Car Race.
#
#   ./build.sh            # create the source tarball from the working tree, then makepkg
#   ./build.sh -i         # build and install with pacman -U (needs sudo)
#
# The generated source tarball, pkg/, src/ and the *.pkg.tar.* output are all
# git-ignored; only this folder (PKGBUILD, build.sh, gen_srcinfo.py) is
# tracked.  For an AUR-style build from a published tag, edit PKGBUILD's
# `source` to point at the GitHub archive instead and run `makepkg` directly.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"

pkgver="$(sed -n "s/^pkgver=//p" "$here/PKGBUILD" | head -1)"
pkgname="$(sed -n "s/^pkgname=//p" "$here/PKGBUILD" | head -1)"
tarball="$here/${pkgname}-${pkgver}.tar.gz"

echo ">> repo:    $root"
echo ">> package: ${pkgname}-${pkgver}"

# --- 1. source tarball from the working tree --------------------------------
# Include uncommitted changes (unlike `git archive HEAD`) so the package
# matches what is being developed, but never ship the VCS, caches, runtime
# data or previous build output.
rm -f "$tarball"
tar -C "$root" -czf "$tarball" \
    --exclude='./.git' \
    --exclude='./.git/*' \
    --exclude='__pycache__' \
    --exclude='*.py[co]' \
    --exclude='./.venv' \
    --exclude='./venv' \
    --exclude='./build' \
    --exclude='./dist' \
    --exclude='*.egg-info' \
    --exclude='./anticheat/target' \
    --exclude='./config' \
    --exclude='./projects' \
    --exclude='./runtime' \
    --exclude='*.pkg.tar' \
    --exclude='*.pkg.tar.*' \
    --exclude='./arch' \
    --transform "s,^\./,${pkgname}-${pkgver}/," \
    .
echo ">> wrote $tarball"

# --- 2. makepkg -------------------------------------------------------------
# -f: overwrite an existing package.  -d: skip dependency checks -- the app is
# pure Python and none of its runtime dependencies are needed to assemble the
# package (installing it still pulls them in).
cd "$here"
makepkg -fd "$@"
