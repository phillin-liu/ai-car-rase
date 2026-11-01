"""OpenGL rendering package (geometry, meshes, scene, camera, HUD)."""
# Import first: it selects the PyOpenGL platform (GLX for an X11 SDL window)
# before ``renderer`` imports OpenGL.
from . import _gl_platform  # noqa: F401  (imported for its side effect)
from .renderer import Renderer

__all__ = ["Renderer"]
