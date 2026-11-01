#!/usr/bin/env bash
# Create the source archive(s) attached to a GitHub release.
#
#   make-source-archive.sh <version> <out.tar.gz> [<out.zip>]
#
# The archive contains the full working tree (README, LICENSE, sources,
# packaging config, ...) under a single ai-car-rase-<version>/ directory.
# Generated build output, caches and local runtime data are skipped.
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 <version> <out.tar.gz> [<out.zip>]" >&2
    exit 2
fi

version="$1"
tar_out="$2"
zip_out="${3:-}"
prefix="ai-car-rase-${version}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

excludes=(
    --exclude='./.git'
    --exclude='./.venv'
    --exclude='./venv'
    --exclude='./env'
    --exclude='./build'
    --exclude='./dist'
    --exclude='./export'
    --exclude='./logs'
    --exclude='./runtime'
    --exclude='./projects'
    --exclude='./config'
    --exclude='./__pycache__'
    --exclude='*/__pycache__'
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='./.pytest_cache'
    --exclude='./.mypy_cache'
    --exclude='./.ruff_cache'
    --exclude='./.tox'
    --exclude='./.nox'
    --exclude='./htmlcov'
    --exclude='./anticheat/target'
    --exclude='./packaging/arch/pkg'
    --exclude='./packaging/arch/src'
    --exclude='./packaging/arch/.SRCINFO'
    --exclude='./packaging/arch/ai-car-rase-*.tar.zst'
    --exclude='./packaging/arch/*.pkg.tar.zst'
    --exclude='*.pkg.tar'
    --exclude='*.pkg.tar.*'
    --exclude='*.deb'
    --exclude='*.rpm'
    --exclude='*.log'
    --exclude='*.pdf'
    --exclude='.DS_Store'
    --exclude='Thumbs.db'
)

mkdir -p "$(dirname "$tar_out")"
tar -C "$root" -czf "$tar_out" \
    --transform "s,^\./,${prefix}/," \
    "${excludes[@]}" .
echo ">> wrote $tar_out"

if [ -n "$zip_out" ]; then
    zip_out="$(cd "$(dirname "$zip_out")" && pwd)/$(basename "$zip_out")"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    tar -xzf "$tar_out" -C "$tmp"
    (cd "$tmp" && "${PYTHON:-python3}" -m zipfile -c "$zip_out" "$prefix")
    echo ">> wrote $zip_out"
fi
