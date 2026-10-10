#!/usr/bin/env python3
"""Make a Nuitka standalone tree self-contained.

Nuitka copies the Python interpreter, every extension module and some C
libraries, but deliberately leaves "system" libraries (Qt5, SDL2, X11, ...)
to the distro.  This script walks the ``ldd`` closure of the whole ``*.dist``
tree and copies everything that is missing into ``<dist>/lib/`` so the launcher
can point ``LD_LIBRARY_PATH`` at it and the package needs no runtime
``python-*`` / ``qt5`` / ``sdl2`` packages.

Deliberately *not* bundled (and therefore the only real system requirements):

* the C library / dynamic loader (``libc``, ``libm``, ``libpthread``, ...);
* the OpenGL / GPU stack (``libGL``, ``libEGL``, ``libGLX``, ``libgbm``,
  ``libdrm``, ``libvulkan``, vendor drivers) -- these *must* come from the
  machine's graphics drivers, bundling Mesa would break NVIDIA/AMD setups.

Usage::

    bundle_libs.py <path-to-*.dist>
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

# --- libraries that must stay on the host system -------------------------
# Patterns are prefixes (no trailing ``$``), because glibc sonames carry a
# version (``libc.so.6``); ``libm\\.so`` must not accidentally match
# ``libmount.so`` -- the literal dot already prevents that.
BLOCK = re.compile(
    r"^(?:"
    r"ld-linux.*|"
    # glibc core
    r"libc\.so|libm\.so|libpthread|libdl\.so|librt\.so|libresolv\.so|"
    r"libnsl\.so|libutil\.so|libanl\.so|libthread_db\.so|"
    # OpenGL / GPU -- must come from the machine's drivers
    r"libGL\.so|libEGL\.so|libGLX\.so|libGLdispatch|libOpenGL\.so|"
    r"libGLESv[12]|libgbm\.so|libdrm|libvulkan\.so|libglapi\.so|"
    r"libGLX_mesa|libEGL_mesa|libgallium|libLLVM|"
    r"libcuda\.so|libnvidia|libnvcuvid|libnvoptix"
    r")"
)

# Qt plugins that drag in desktop-specific stacks we do not want to freeze
# (GTK themes, CUPS).  Qt falls back cleanly when they are absent.
DROP_PLUGINS = (
    "platformthemes/libqgtk3.so",
    "printsupport/libcupsprintersupport.so",
)


def is_elf(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"\x7fELF"
    except OSError:
        return False


def ldd_deps(path: str) -> list[tuple[str, str]]:
    try:
        out = subprocess.run(["ldd", path], capture_output=True, text=True,
                             timeout=60).stdout
    except Exception:
        return []
    deps: list[tuple[str, str]] = []
    for line in out.splitlines():
        line = line.strip()
        m = re.match(r"^(\S+)\s+=>\s+(\S+)", line)
        if m and m.group(2).startswith("/"):
            # group(1) is usually a soname, but the loader line can print a
            # path there too (``/lib64/ld-linux-x86-64.so.2 => ...``).
            deps.append((os.path.basename(m.group(1)), m.group(2)))
        elif line.startswith("/") and "=>" not in line:
            # interpreter line: "/lib64/ld-linux-x86-64.so.2 (0x...)"
            path = line.split()[0]
            deps.append((os.path.basename(path), path))
    return deps


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    dist = os.path.abspath(sys.argv[1])
    if not os.path.isdir(dist):
        print(f"不是目录: {dist}", file=sys.stderr)
        return 1

    for rel in DROP_PLUGINS:
        p = os.path.join(dist, "PyQt5", "qt-plugins", rel)
        if os.path.exists(p):
            os.remove(p)

    libdir = os.path.join(dist, "lib")
    os.makedirs(libdir, exist_ok=True)

    elfs: list[str] = []
    provided: set[str] = set()
    for root, _dirs, files in os.walk(dist):
        for fn in files:
            p = os.path.join(root, fn)
            if is_elf(p):
                elfs.append(p)
                provided.add(os.path.basename(p))

    need: dict[str, str] = {}          # soname -> source path
    queue = list(elfs)
    seen: set[str] = set()
    while queue:
        p = queue.pop()
        if p in seen:
            continue
        seen.add(p)
        for soname, path in ldd_deps(p):
            if soname in provided or BLOCK.match(soname):
                continue
            if not (path.startswith("/usr/lib") or path.startswith("/lib")):
                continue
            if soname not in need:
                need[soname] = path
                queue.append(path)

    copied = 0
    total = 0
    for soname, src in sorted(need.items()):
        dst = os.path.join(libdir, soname)
        try:
            shutil.copy2(os.path.realpath(src), dst)
            os.chmod(dst, 0o755)
            copied += 1
            total += os.path.getsize(dst)
        except OSError as exc:
            print(f"  !! 无法复制 {src}: {exc}", file=sys.stderr)

    print(f">> 打包 {copied} 个系统库到 {libdir} ({total / 1048576:.1f} MB)")
    print(f">> 仍由系统提供: glibc / 动态加载器 / OpenGL-GPU 栈")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
