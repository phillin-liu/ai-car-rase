#!/usr/bin/env bash
# Shared helpers for the packaging scripts.  Source this file, never run it:
#
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"
#
# Every packaging format (Arch, deb, rpm) ships the *same* frozen Nuitka tree
# (build/nuitka/*.dist, see ../nuitka/build.sh); the per-format scripts only
# stage, add metadata and compress it.  Windows has its own script because it
# has to be compiled on Windows.

set -euo pipefail

PKG_NAME="ai-car-rase"

# Repository root -- packaging/ lives directly below it.
pkg_root() {
    (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
}

die() {
    echo "$PKG_NAME: $*" >&2
    exit 1
}

# Version stamped into the packages.  $VERSION (what CI passes from the git
# tag) wins; otherwise the pkgver in the Arch PKGBUILD is the single source of
# truth for the release version.
pkg_version() {
    local v="${VERSION:-}"
    if [ -z "$v" ]; then
        v="$(sed -n 's/^pkgver=//p' "$(pkg_root)/packaging/arch/PKGBUILD" | head -1)"
    fi
    v="${v#v}"
    [ -n "$v" ] || die "无法确定版本号（请设置 VERSION）"
    printf '%s\n' "$v"
}

# Frozen Nuitka tree to package: $DIST, else the newest build/nuitka/*.dist.
# Returns non-zero when nothing has been built yet.
pkg_find_dist() {
    local root dist
    root="$(pkg_root)"
    if [ -n "${DIST:-}" ]; then
        printf '%s\n' "$DIST"
        return 0
    fi
    dist="$(find "$root/build/nuitka" -mindepth 1 -maxdepth 1 -type d -name '*.dist' \
        -printf '%T@\t%p\n' 2>/dev/null | sort -rn | head -1 | cut -f2-)"
    [ -n "$dist" ] && [ -d "$dist" ] || return 1
    printf '%s\n' "$dist"
}

# Build the frozen tree unless it already exists, then echo its path.
# `--no-build` callers must use pkg_find_dist directly.
pkg_ensure_dist() {
    local dist
    if dist="$(pkg_find_dist)"; then
        printf '%s\n' "$dist"
        return 0
    fi
    echo ">> 没有找到 build/nuitka/*.dist，先运行 Nuitka 编译 ..." >&2
    "$(pkg_root)/packaging/nuitka/build.sh" >&2
    dist="$(pkg_find_dist)" || die "Nuitka 编译结束但没有生成 *.dist"
    printf '%s\n' "$dist"
}

# glibc version of the build host: the frozen tree is linked against it, so it
# becomes the minimum version in the package metadata.
pkg_glibc_version() {
    local v
    v="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
    if [ -z "$v" ]; then
        v="$(ldd --version 2>/dev/null | head -1 | awk '{print $NF}')"
    fi
    printf '%s\n' "${v:-2.17}"
}

# Render a packaging template (@@NAME@@, @@VERSION@@, @@LIBDIR@@ ...).
pkg_render() {
    local tpl="$1"
    shift
    local ver expr
    ver="$(pkg_version)"
    expr=(-e "s|@@NAME@@|$PKG_NAME|g" -e "s|@@VERSION@@|$ver|g")
    local pair
    for pair in "$@"; do
        expr+=(-e "s|@@${pair%%=*}@@|${pair#*=}|g")
    done
    sed "${expr[@]}" "$tpl"
}

# Artifact directory shared by all formats (uploaded by CI, git-ignored).
pkg_dist_dir() {
    local d
    d="$(pkg_root)/dist"
    mkdir -p "$d"
    printf '%s\n' "$d"
}
