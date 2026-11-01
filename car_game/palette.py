"""Scene palette and lighting constants shared by both renderers.

The human watches the race through the OpenGL renderer (:mod:`car_game.render`);
the AI's "camera" is a pure-Pillow software rasteriser
(:mod:`car_game.vision.fp_render`).  Each used to carry its own hand-copied
copy of these values and the copies had already drifted apart -- the asphalt,
the sky-ground tone and the distant-hill colour all disagreed, so the AI was
never quite looking at the world the player saw.

Keeping them in one module is what lets the AI's frame match the window.

Colours are linear floats in ``0..1``, matching what the GL side feeds its
shaders; :func:`rgb255` converts one for Pillow.

This module must stay dependency-free (standard library only).
``car_game.render`` pulls in OpenGL at import time and ``car_game.vision`` has
to keep working in a head-less process with no GL context at all, so the shared
values cannot live under ``car_game/render/``.
"""
from __future__ import annotations

# --- sky and atmosphere ------------------------------------------------
SKY_TOP = (0.26, 0.46, 0.78)
SKY_HORIZON = (0.76, 0.84, 0.92)
SKY_GROUND = (0.26, 0.30, 0.26)
SKY_POW = 0.62                    # gradient exponent, matches SKY_FS
FOG_RANGE = (150.0, 620.0)        # metres, matches LIT_FS uFogRange

# --- sun ---------------------------------------------------------------
LIGHT_DIR = (0.36, 0.84, 0.40)    # points towards the sun (not normalised)
AMBIENT = 0.52
DIFFUSE = 0.78

# --- ground and road ---------------------------------------------------
GROUND = (0.255, 0.395, 0.225)
ROAD = (0.185, 0.185, 0.205)
LINE = (0.88, 0.88, 0.85)
CURB_A = (0.78, 0.16, 0.14)
CURB_B = (0.90, 0.90, 0.88)
HILL = (0.21, 0.32, 0.23)

# --- roadside props ----------------------------------------------------
RAIL = (0.60, 0.63, 0.68)
POST = (0.42, 0.45, 0.50)
WINDOW = (0.10, 0.14, 0.22)
# facade colours, cycled per building so the skyline is not one flat grey
BUILDINGS = [(0.60, 0.64, 0.70), (0.74, 0.60, 0.50), (0.50, 0.58, 0.66),
             (0.82, 0.74, 0.60), (0.44, 0.48, 0.56), (0.70, 0.54, 0.56)]
TRUNK = (0.30, 0.21, 0.13)
FOLIAGE = [(0.10, 0.42, 0.16), (0.13, 0.48, 0.19), (0.09, 0.36, 0.14),
           (0.16, 0.44, 0.18)]

# Obstacle bodies.  The *detector* keys on its own flat palette in
# ``car_game.vision.render`` -- these are the lit scene colours and must not be
# confused with it.
OBSTACLE_COLORS = {
    "cone": (0.95, 0.36, 0.14),
    "barrel": (0.92, 0.74, 0.18),
    "crate": (0.58, 0.40, 0.23),
}

# --- cars --------------------------------------------------------------
# The body colours the player sees.  The AI's *detection* frame deliberately
# uses its own high-contrast palette (``car_game.vision.render.RIVAL_CAR`` /
# ``OWN_CAR``) because the built-in detector segments on exact colour; the
# realistic frame the model is shown uses these instead.
CAR_COLORS = ((0.95, 0.25, 0.2), (0.25, 0.55, 0.95))

# contact shadows
SHADOW = (0.0, 0.0, 0.0)
SHADOW_ALPHA = 0.30


def rgb255(color):
    """Convert a ``0..1`` float colour to an 8-bit RGB tuple for Pillow."""
    return tuple(int(max(0, min(255, round(c * 255.0)))) for c in color)
