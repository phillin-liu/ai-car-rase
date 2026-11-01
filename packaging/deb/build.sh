#!/usr/bin/env bash
# Build a Debian/Ubuntu package (.deb) from the frozen Nuitka tree.
#
#   ./build.sh                # compile the tree if needed, then package
#   ./build.sh --no-build     # reuse an existing build/nuitka/*.dist
#   VERSION=1.2.3 ./build.sh  # stamp another version (CI passes the tag)
#
# No debhelper: the payload is a single pre-built tree, so the package is
# staged by hand and handed to dpkg-deb.  Output goes to <repo>/dist/.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=packaging/common.sh
source "$here/../common.sh"

build=1
[[ "${1:-}" == "--no-build" ]] && build=0

command -v dpkg-deb >/dev/null || die "需要 dpkg-deb (dpkg)"

version="$(pkg_version)"
# Debian versions may not contain '-', and '~' sorts before the release, so
# "1.2.3-rc1" becomes "1.2.3~rc1-1" instead of a bogus revision.
upstream="${version//-/\~}"
debversion="$upstream-1"
glibc="$(pkg_glibc_version)"
libdir="/usr/lib/$PKG_NAME"

if [[ $build -eq 1 ]]; then
    dist="$(pkg_ensure_dist)"
else
    dist="$(pkg_find_dist)" || die "没有可用的 build/nuitka/*.dist"
fi

root="$(pkg_root)"
stage="$root/build/deb/root"
rm -rf "$stage"
mkdir -p "$stage/DEBIAN" "$stage$libdir" "$stage/usr/bin" \
         "$stage/usr/share/applications" "$stage/usr/share/doc/$PKG_NAME"

# --- payload: the frozen tree -------------------------------------------
echo ">> 复制冻结树 $(basename "$dist") ..."
cp -a "$dist/." "$stage$libdir/"

# --- launcher + desktop entry -------------------------------------------
pkg_render "$here/../common/launcher.sh.in" "LIBDIR=$libdir" \
    > "$stage/usr/bin/$PKG_NAME"
chmod 0755 "$stage/usr/bin/$PKG_NAME"
pkg_render "$here/../common/ai-car-rase.desktop.in" \
    > "$stage/usr/share/applications/$PKG_NAME.desktop"
chmod 0644 "$stage/usr/share/applications/$PKG_NAME.desktop"

# --- docs ----------------------------------------------------------------
cp LICENSE "$stage/usr/share/doc/$PKG_NAME/copyright"
cp README.md "$stage/usr/share/doc/$PKG_NAME/README.md"

# --- control (needs the payload size) -----------------------------------
installed_size=$(( $(du -sk "$stage" | cut -f1) - $(du -sk "$stage/DEBIAN" | cut -f1) ))
pkg_render "$here/control.in" \
    "DEBVERSION=$debversion" \
    "GLIBC=$glibc" \
    "INSTALLED_SIZE=$installed_size" \
    > "$stage/DEBIAN/control"
chmod 0644 "$stage/DEBIAN/control"

# --- build ---------------------------------------------------------------
dest="$(pkg_dist_dir)"
out="$dest/${PKG_NAME}_${debversion}_amd64.deb"
rm -f "$out"
# zstd keeps the ~300 MB tree small and is the Debian 12+/Ubuntu 22.04+
# default; fall back to xz on older dpkg.
if dpkg-deb --help 2>&1 | grep -q zstd; then
    compression=(-Zzstd -z19)
else
    compression=(-Zxz -z6)
fi
dpkg-deb --root-owner-group "${compression[@]}" --build "$stage" "$out"

echo
echo ">> 完成:"
ls -lh "$out"
dpkg-deb --info "$out" | sed -n '1,12p'
