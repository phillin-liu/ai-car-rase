#!/usr/bin/env bash
# Build an RPM (Fedora / RHEL / openSUSE) from the frozen Nuitka tree.
#
#   ./build.sh                # compile the tree if needed, then rpmbuild
#   ./build.sh --no-build     # reuse an existing build/nuitka/*.dist
#   VERSION=1.2.3 ./build.sh  # stamp another version (CI passes the tag)
#
# rpmbuild works in a throw-away _topdir under build/rpm/; the resulting RPM
# is copied to <repo>/dist/ like every other format.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=packaging/common.sh
source "$here/../common.sh"

build=1
[[ "${1:-}" == "--no-build" ]] && build=0

command -v rpmbuild >/dev/null || die "需要 rpmbuild (rpm-build)"
command -v rpm >/dev/null || die "需要 rpm"

version="$(pkg_version)"
# RPM Version may not contain '-'; the tag "1.2.3-rc1" becomes "1.2.3_rc1".
rpm_version="${version//-/_}"
glibc="$(pkg_glibc_version)"
libdir="$(rpm --eval '%{_libdir}')/$PKG_NAME"
root="$(pkg_root)"

if [[ $build -eq 1 ]]; then
    dist="$(pkg_ensure_dist)"
else
    dist="$(pkg_find_dist)" || die "没有可用的 build/nuitka/*.dist"
fi

work="$root/build/rpm"
rm -rf "$work"
mkdir -p "$work/BUILD" "$work/BUILDROOT" "$work/RPMS" "$work/SOURCES" \
         "$work/SPECS" "$work/SRPMS" "$work/stage"

# --- Source0: frozen tree + docs, one top-level directory ----------------
src_dir="$work/stage/$PKG_NAME-$rpm_version"
echo ">> 复制冻结树 $(basename "$dist") ..."
mkdir -p "$src_dir"
cp -a "$dist/." "$src_dir/"
cp LICENSE README.md "$src_dir/"
tar -C "$work/stage" -czf "$work/SOURCES/$PKG_NAME-$rpm_version-nuitka.tar.gz" \
    "$PKG_NAME-$rpm_version"

# --- Source1/2: launcher + desktop entry --------------------------------
pkg_render "$here/../common/launcher.sh.in" "LIBDIR=$libdir" \
    > "$work/SOURCES/$PKG_NAME.launcher"
chmod 0755 "$work/SOURCES/$PKG_NAME.launcher"
pkg_render "$here/../common/ai-car-rase.desktop.in" \
    > "$work/SOURCES/$PKG_NAME.desktop"
chmod 0644 "$work/SOURCES/$PKG_NAME.desktop"

# --- spec ----------------------------------------------------------------
pkg_render "$here/ai-car-rase.spec.in" \
    "RPMVERSION=$rpm_version" \
    "GLIBC=$glibc" \
    "DATE=$(LC_ALL=C date '+%a %b %d %Y')" \
    > "$work/SPECS/$PKG_NAME.spec"

# --- build ---------------------------------------------------------------
rpmbuild -bb "$work/SPECS/$PKG_NAME.spec" \
    --define "_topdir $work" \
    --define "_sourcedir $work/SOURCES" \
    --define "_rpmdir $work/RPMS" \
    --define "_builddir $work/BUILD" \
    --define "_buildrootdir $work/BUILDROOT"

dest="$(pkg_dist_dir)"
cp -v "$work"/RPMS/*/*.rpm "$dest/"

echo
echo ">> 完成:"
ls -lh "$dest/"*.rpm
rpm -qip "$work"/RPMS/*/*.rpm | sed -n '1,12p'
