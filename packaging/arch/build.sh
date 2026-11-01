#!/usr/bin/env bash
# Build the Arch package (.pkg.tar.zst) from the frozen Nuitka tree.
#
#   ./build.sh                # compile the tree if needed, then makepkg
#   ./build.sh --no-build     # reuse an existing build/nuitka/*.dist
#   PKGVER=1.2.3 ./build.sh   # stamp another version (CI passes the tag)
#
# The result is copied to <repo>/dist/ so every packaging format drops its
# artifacts in one place.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=packaging/common.sh
source "$here/../common.sh"

build=1
[[ "${1:-}" == "--no-build" ]] && build=0

command -v makepkg >/dev/null || die "需要 makepkg (pacman)"
command -v fakeroot >/dev/null || die "需要 fakeroot"
command -v zstd >/dev/null || die "需要 zstd"

pkgver="${PKGVER:-$(sed -n 's/^pkgver=//p' "$here/PKGBUILD" | head -1)}"
pkgver="${pkgver//-/_}"
if [[ "$(sed -n 's/^pkgver=//p' "$here/PKGBUILD" | head -1)" != "$pkgver" ]]; then
    echo ">> PKGBUILD pkgver -> $pkgver"
    sed -i "s/^pkgver=.*/pkgver=$pkgver/" "$here/PKGBUILD"
fi

if [[ $build -eq 1 ]]; then
    VERSION="$pkgver" "$here/../nuitka/build.sh"
fi
dist="$(pkg_find_dist)" || die "没有可用的 build/nuitka/*.dist"

# --- source tarball for the PKGBUILD ------------------------------------
tarball="$here/$PKG_NAME-$pkgver-nuitka.tar.zst"
base="$(basename "$dist")"
echo ">> 生成打包源 $tarball ..."
tar -C "$(dirname "$dist")" --zstd -cf "$tarball" \
    --transform "s,^${base//./\\.},${PKG_NAME}," "$base"

# --- build ---------------------------------------------------------------
cd "$here"
makepkg -fd --noconfirm

dest="$(pkg_dist_dir)"
rm -f "$dest/$PKG_NAME-$pkgver"-*.pkg.tar.*
cp -v "$here"/*.pkg.tar.zst "$dest/"

echo
echo ">> 完成:"
ls -lh "$dest/"*.pkg.tar.*
