#!/usr/bin/env bash
# Create a *stub* frozen Nuitka tree.
#
#   make-stub-dist.sh <out-dir>
#
# The packaging scripts (packaging/{deb,rpm}/build.sh --no-build) only stage
# and compress whatever tree they are pointed at, so CI can validate the
# package metadata in seconds instead of waiting for a real Nuitka build:
#
#   DIST=$(./make-stub-dist.sh /tmp/stub.dist) packaging/deb/build.sh --no-build
set -euo pipefail

out="${1:?usage: $0 <out-dir>}"

rm -rf "$out"
mkdir -p "$out/lib" "$out/anticheat/target/release"

# Stand-in for the frozen executable: the packaging code only needs an ELF
# file of the right name to exist (it never runs it).
cat > "$out/ai-car-rase" <<'EOF'
#!/bin/sh
echo "stub ai-car-rase"
EOF
chmod 0755 "$out/ai-car-rase"

printf 'stub\n' > "$out/lib/libstub.so.1"
printf 'stub\n' > "$out/anticheat/target/release/ac-validate"
chmod 0755 "$out/anticheat/target/release/ac-validate"

# Sparse payload so the archives exercise multi-file handling without
# spending seconds on compression.
: > "$out/stub-payload.bin"

printf '%s\n' "$out"
