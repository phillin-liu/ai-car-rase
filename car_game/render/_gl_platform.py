"""Pick the PyOpenGL platform that matches the window SDL is about to create.

PyOpenGL chooses its platform (GLX / EGL / ...) from the environment when
``OpenGL`` is first imported.  On Linux its default is EGL, but pygame/SDL on
an X11 session creates a *GLX* context -- so PyOpenGL reports "no valid
context" on the very first ``glVertexAttribPointer`` and the 3D window dies
before the first frame.  The failure does not happen on Wayland, where SDL
does create an EGL context, which is why it looked like a "some machines"
bug.

This module only sets ``PYOPENGL_PLATFORM`` when the caller has not chosen one
already, and only on Linux.  Importing it must happen before ``OpenGL.GL``.
"""
from __future__ import annotations

import os
import sys


def ensure_pyopengl_platform() -> str:
    """Set ``PYOPENGL_PLATFORM`` if it is unset and the driver is known."""
    current = os.environ.get("PYOPENGL_PLATFORM")
    if current:
        return current
    if not sys.platform.startswith("linux"):
        return ""

    driver = os.environ.get("SDL_VIDEODRIVER", "").strip().lower()
    if driver.startswith("wayland"):
        platform = "egl"
    elif driver.startswith("x11"):
        platform = "glx"
    elif os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        platform = "glx"
    else:
        # Wayland (or a driver that supplies its own context): keep PyOpenGL's
        # default.  Headless imports never reach the renderer.
        return ""

    os.environ["PYOPENGL_PLATFORM"] = platform
    return platform


ensure_pyopengl_platform()
