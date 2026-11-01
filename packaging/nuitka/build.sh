#!/usr/bin/env bash
# Compile main.py into a frozen Nuitka tree under build/nuitka/.
#
#   ./build.sh            # standalone tree -> build/nuitka/<exe>.dist
#   ./build.sh --onefile  # single self-extracting executable instead
#
# The standalone tree is the *single* input for packaging/arch, packaging/deb
# and packaging/rpm: it carries its own Python interpreter, every extension
# module, and -- after bundle_libs.py -- every native library it may ship, so
# the target machine needs no python-* / Qt5 / SDL2 package at all.
#
# Windows is compiled on Windows: packaging/windows/build.ps1.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="$root/build/nuitka"
main_py="$root/main.py"

# shellcheck source=packaging/common.sh
source "$here/../common.sh"
version="$(pkg_version)"

mode="standalone"
[[ "${1:-}" == "--onefile" ]] && mode="onefile"

mkdir -p "$out"

# --- Python carrying numpy/pygame/PyQt5 -----------------------------------
# Prefer $PYTHON (CI points it at a venv), else the system python3.  A frozen
# build needs the *same* interpreter that has the dependencies installed.
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || die "找不到 $PY"
if ! "$PY" -c 'import nuitka' 2>/dev/null; then
    echo ">> 安装 Nuitka ..."
    "$PY" -m pip install --upgrade nuitka ordered-set zstandard
fi

# --- anti-cheat validator -------------------------------------------------
# build.sh embeds the Rust binary as data; the pure-Python validator is used
# when it is missing, so a failed build here only degrades, never breaks the
# bundle.
ac_bin="$root/anticheat/target/release/ac-validate"
if [[ ! -x "$ac_bin" ]] && command -v cargo >/dev/null; then
    echo ">> 编译 Rust 反作弊校验器 ..."
    cargo build --release --locked --manifest-path "$root/anticheat/Cargo.toml"
fi
ac_args=()
if [[ -x "$ac_bin" ]]; then
    ac_args+=("--include-data-file=$ac_bin=anticheat/target/release/ac-validate")
else
    echo "!! 未找到 $ac_bin，冻结树将使用纯 Python 反作弊校验" >&2
fi

# The detector weights import path is optional -- skipping it saves ~200 MB
# and the code has an explicit pure-Python fallback.
NOFOLLOW=(
    --nofollow-import-to=scipy
    --nofollow-import-to=ultralytics
    --nofollow-import-to=torch
    --nofollow-import-to=torchvision
    --nofollow-import-to=matplotlib
    --nofollow-import-to=tkinter
)

echo ">> Nuitka $("$PY" -m nuitka --version | head -1) / mode=$mode / version=$version"

"$PY" -m nuitka \
    --"$mode" \
    --enable-plugin=pyqt5 \
    --include-package=car_game \
    --include-package-data=car_game \
    "${ac_args[@]}" \
    --output-dir="$out" \
    --output-filename=ai-car-rase \
    --assume-yes-for-downloads \
    --company-name="AI Car Race" \
    --product-name="AI Car Race" \
    --file-version="$version" \
    --product-version="$version" \
    --report="$out/report.xml" \
    "${NOFOLLOW[@]}" \
    "$main_py"

if [[ "$mode" == "standalone" ]]; then
    dist="$(find "$out" -mindepth 1 -maxdepth 1 -type d -name '*.dist' | head -1)"
    [[ -n "$dist" ]] || die "Nuitka 没有生成 *.dist 目录"
    # Make the tree self-contained: pull the ldd() closure of the distro
    # libraries (Qt5, SDL2, X11, ICU, OpenBLAS ...) into <dist>/lib/.
    "$PY" "$here/bundle_libs.py" "$dist"
    # Record the tree so the per-format scripts can find it without guessing.
    printf '%s\n' "$dist" > "$out/DIST"
else
    # A one-file build is not a tree the packaging scripts can ship.
    rm -f "$out/DIST"
fi

echo
echo ">> 完成:"
if [[ "$mode" == "standalone" ]]; then
    echo "   dist  = $(cat "$out/DIST")"
else
    echo "   exe   = $(find "$out" -maxdepth 1 -type f -name 'ai-car-rase*' | head -1)"
fi
find "$out" -maxdepth 2 -name 'ai-car-rase' -o -maxdepth 1 -name 'ai-car-rase.*' | head -5
