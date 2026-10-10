#!/usr/bin/env bash
# Build the AI Car Race Nuitka *standalone* distribution.
#
#   ./build.sh            # compile into ../../build/nuitka/
#   ./build.sh --onefile  # single self-extracting executable instead
#
# The standalone ``*.dist`` tree is what arch/nuitka/PKGBUILD ships: it
# contains its own Python interpreter and every Python/extension dependency
# (numpy, pygame, PyOpenGL, PyQt5, Pillow, openai ...), so the target machine
# only needs the base system libraries (glibc, libGL, libX11 ...), never any
# ``python-*`` package.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="$root/build/nuitka"
run_file="$root/main.py"

mode="standalone"
[[ "${1:-}" == "--onefile" ]] && mode="onefile"

mkdir -p "$out"

pkgname="$(sed -n 's/^pkgname=//p' "$here/PKGBUILD" | head -1)"
pkgver="$(sed -n 's/^pkgver=//p' "$here/PKGBUILD" | head -1)"
tarball="$here/${pkgname}-${pkgver}-nuitka.tar.zst"

# Prefer the distro's system Python (the one carrying numpy/pygame/PyQt5).
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "找不到 python3" >&2; exit 1; }
"$PY" -c 'import nuitka' 2>/dev/null || {
    echo ">> 安装 Nuitka ..."
    "$PY" -m pip install --break-system-packages --upgrade nuitka ordered-set zstandard
}

# Keep the bundled detector weights-import path optional --saves ~200 MB and
# the code has an explicit pure-Python fallback.
NOFOLLOW=(--nofollow-import-to=scipy
          --nofollow-import-to=ultralytics
          --nofollow-import-to=torch
          --nofollow-import-to=torchvision
          --nofollow-import-to=matplotlib
          --nofollow-import-to=tkinter)

echo ">> Nuitka $("$PY" -m nuitka --version | head -1) / mode=$mode"

"$PY" -m nuitka \
    --"$mode" \
    --enable-plugin=pyqt5 \
    --include-package=car_game \
    --include-package-data=car_game \
    --include-data-file="$root/anticheat/target/release/ac-validate=anticheat/target/release/ac-validate" \
    --output-dir="$out" \
    --output-filename=ai-car-rase \
    --assume-yes-for-downloads \
    --company-name="AI Car Race" \
    --product-name="AI Car Race" \
    --file-version=0.1.0 \
    --product-version=0.1.0 \
    --report="$out/report.xml" \
    "${NOFOLLOW[@]}" \
    "$run_file"

dist="$out/main.dist"
for extra in "$out"/*.dist; do [[ -d "$extra" ]] && dist="$extra"; done
if [[ "$mode" == "standalone" ]]; then
    # --- make it self-contained (Qt5, SDL2, X11, ...) -------------------
    python3 "$here/bundle_libs.py" "$dist"

    # --- source tarball for PKGBUILD -----------------------------------
    echo ">> 生成打包源 $tarball ..."
    base="$(basename "$dist")"
    tar -C "$out" --zstd -cf "$tarball" \
        --transform "s,^${base//./\\.},${pkgname}," "$base"
    ls -lh "$tarball"
fi

echo
echo ">> 完成:"
ls -d "$out"/*.dist "$out"/*.onefile 2>/dev/null || ls -lh "$out"
