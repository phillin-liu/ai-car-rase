"""Frame orchestration: sky, world, cars, shadows and HUD.

Draw order per frame:
  1. sky gradient (no depth test / write),
  2. static world mesh (sun + ambient + fog),
  3. glossy decals (lane markings, windows) with polygon offset,
  4. contact shadows (blend, no depth write),
  5. hazards, cars (paint / trim / wheels), car blob shadows,
  6. HUD texture as a full-screen overlay.
"""
from __future__ import annotations

import time

import numpy as np
from OpenGL.GL import *

from ..mathutil import (identity, multiply, rotation_y, rotation_z, scaling,
                        translation)
from ..palette import (AMBIENT, DIFFUSE, FOG_RANGE, LIGHT_DIR as _LIGHT_DIR,
                       SKY_GROUND, SKY_HORIZON, SKY_TOP)
from .camera import Camera
from .gl import (LIT_FS, LIT_VS, OV_FS, OV_VS, SKY_FS, SKY_VS, Mesh,
                 OverlayQuad, Program, apply_common_state, create_texture,
                 normal_matrix)
from .hud import Hud, countdown_label
from .models import WHEEL_R, WHEELS, build_blob_shadow, build_paint, build_trim, build_wheel
from .scene import build_scene

# Sky / fog / sun come from car_game.palette so the software renderer that
# produces the AI's frame agrees with this one; only the array form is local.
LIGHT_DIR = np.array(_LIGHT_DIR, dtype=np.float64)

# ortho mapping the 0..1 overlay quad onto the full clip square
_OVERLAY_ORTHO = np.array([
    [2.0, 0.0, 0.0, -1.0],
    [0.0, 2.0, 0.0, -1.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)


def car_model(x, y, z, heading, scale=1.0) -> np.ndarray:
    """Car local frame: +X forward, +Y up, +Z left."""
    return multiply(translation(x, y, z), rotation_y(-heading), scaling(scale))


class Renderer:
    # The HUD is a full-window RGBA texture: at 1080p, repainting it and
    # pushing 8 MB to the GPU is easily the most expensive thing in the frame,
    # and it is what makes the 3D view hitch.  A speed readout does not need
    # 60 Hz, so it is rebuilt at this rate -- plus immediately whenever
    # something discrete changes (see ``_hud_signature``), so the countdown,
    # the pause veil and the finish panel never visibly trail the race.
    HUD_HZ = 25.0

    # Backstop for gaps that are certainly not frames (a window drag, a
    # minimise, a disk hiccup): averaging one of those in would poison the
    # readout for the next second.  The between-match stall is handled exactly,
    # by resetting in ``set_track``, so this only has to catch the pathological
    # case -- it is deliberately far above any frame rate worth reporting.
    FRAME_GAP_RESET = 2.0

    def __init__(self, width: int, height: int, msaa: bool = True):
        self.width = int(width)
        self.height = int(height)
        self.msaa = bool(msaa)

        self.lit = Program(LIT_VS, LIT_FS)
        self.sky_prog = Program(SKY_VS, SKY_FS)
        self.overlay = Program(OV_VS, OV_FS)
        self.quad = OverlayQuad()

        self.camera = Camera()
        self.hud = Hud(width, height)
        self._hud_tex = None
        self._hud_size = (0, 0)
        self._hud_t = 0.0
        self._hud_sig = None
        # smoothed frame time, shown by the debug HUD so a slow machine can be
        # diagnosed from the overlay instead of guessed at
        self._last_draw = None
        self.frame_ms = 0.0

        self.car_paint = self._upload(build_paint())
        self.car_trim = self._upload(build_trim())
        self.wheel = self._upload(build_wheel())
        self.blob = self._upload(build_blob_shadow(1.0))

        self.scene = None
        self.track = None
        self.scene_mesh = None
        self.gloss_mesh = None
        self.scene_shadow = None

        apply_common_state()
        glClearColor(SKY_HORIZON[0], SKY_HORIZON[1], SKY_HORIZON[2], 1.0)

    # ------------------------------------------------------------------
    @staticmethod
    def _upload(buffer) -> Mesh:
        m = Mesh()
        m.upload(buffer.verts, buffer.indices)
        return m

    def set_track(self, track) -> None:
        sc = build_scene(track)
        self.scene = sc
        self.track = track
        self.scene_mesh = self._upload(sc.mesh)
        self.gloss_mesh = self._upload(sc.gloss)
        self.scene_shadow = self._upload(sc.shadow)
        self.camera.set_bounds(sc.center, sc.extent)
        # A new match: whatever happened between the previous frame and this
        # rebuild is setup work, not a frame time.  Start the average afresh so
        # the debug overlay reports the race that is about to run.
        self._last_draw = None

    def warmup(self, game, camera="chase", watch=0, show_watch=True,
               dt=1.0 / 60.0, countdown=0.0, paused=False, detail=False,
               debug=False) -> None:
        """Draw and present one frame *outside* the frame-time accounting.

        The first frame after :meth:`set_track` is by far the most expensive of
        a match, and none of it is the race: the GL driver links the programs
        for their first real use, validates the scene VBOs that were just
        uploaded, allocates the full-window HUD texture and pushes ~8 MB into
        it, and compiles the first-frame pipeline.  Paid inside the loop that
        single frame ran into the hundreds of milliseconds, and because the
        smoothed readout folds every frame in, the debug overlay then needed
        about a second to climb back from the "7 fps" it showed at the start.

        Drawing the same frame here -- with the cars still on the line, during
        the between-match work the runner is already doing -- starts the race
        with a warm pipeline.  ``_last_draw`` is cleared afterwards so the time
        this costs is not charged to the next frame either.
        """
        import pygame
        self.draw(game, camera=camera, watch=watch, show_watch=show_watch,
                  dt=dt, countdown=countdown, paused=paused, detail=detail,
                  debug=debug)
        pygame.display.flip()
        self._last_draw = None

    # ------------------------------------------------------------------
    def _account_frame(self, gap: float) -> None:
        """Fold one inter-frame gap into the smoothed frame time.

        A gap past :attr:`FRAME_GAP_RESET` is not a frame -- it is the
        between-match work (scoring, saving, generating the next track,
        rebuilding the scene) -- so it restarts the average instead of being
        averaged in as one enormous frame.
        """
        if gap > self.FRAME_GAP_RESET:
            self.frame_ms = 0.0
            return
        ms = gap * 1000.0
        self.frame_ms = (ms if self.frame_ms <= 0.0
                         else self.frame_ms * 0.9 + ms * 0.1)

    def _lighting_uniforms(self, eye):
        p = self.lit
        p.set_vec3("uLightDir", LIGHT_DIR / np.linalg.norm(LIGHT_DIR))
        p.set_vec3("uCamPos", eye)
        p.set_vec3("uSkyColor", SKY_TOP)
        p.set_vec3("uGroundColor", SKY_GROUND)
        p.set_vec3("uFogColor", SKY_HORIZON)
        p.set_vec2("uFogRange", *FOG_RANGE)
        p.set_float("uAmbient", AMBIENT)
        p.set_float("uDiffuse", DIFFUSE)
        p.set_float("uAlpha", 1.0)

    def _draw_mesh(self, mesh, model, pv, tint=(1.0, 1.0, 1.0),
                   specular=0.06, shininess=24.0, alpha=1.0):
        self.lit.set_mat4("uMVP", pv @ model)
        self.lit.set_mat4("uModel", model)
        self.lit.set_mat3("uNormalMat", normal_matrix(model))
        self.lit.set_vec3("uTint", tint)
        self.lit.set_float("uSpecular", specular)
        self.lit.set_float("uShininess", shininess)
        self.lit.set_float("uAlpha", alpha)
        mesh.draw()

    def _draw_sky(self):
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glDisable(GL_CULL_FACE)
        self.sky_prog.use()
        self.sky_prog.set_vec3("uTopColor", SKY_TOP)
        self.sky_prog.set_vec3("uHorizonColor", SKY_HORIZON)
        self.quad.draw()
        glEnable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)

    def _draw_shadows(self, game, pv):
        glDepthMask(GL_FALSE)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-2.0, -2.0)
        eye = self.camera.eye
        self._lighting_uniforms(eye)
        if self.scene_shadow is not None:
            self._draw_mesh(self.scene_shadow, identity(), pv, specular=0.0,
                            shininess=1.0, alpha=0.34)
        for i, car in enumerate(game.cars):
            model = multiply(
                translation(float(car.pos[0]), float(car.y) + 0.03,
                            float(car.pos[1])),
                scaling(2.6, 1.0, 1.5))
            self._draw_mesh(self.blob, model, pv, specular=0.0, shininess=1.0,
                            alpha=0.30)
        for hz in game.hazards:
            hy = hz.get("y")
            if hy is None:
                hy = self._height_at(hz["x"], hz["z"])
            r = hz.get("radius", 2.5)
            model = multiply(translation(hz["x"], float(hy) + 0.03, hz["z"]),
                             scaling(r, 1.0, r))
            self._draw_mesh(self.blob, model, pv, tint=(0.35, 0.28, 0.20),
                            specular=0.4, shininess=40.0, alpha=0.92)
        glDisable(GL_POLYGON_OFFSET_FILL)
        glDepthMask(GL_TRUE)

    def _draw_car(self, game, idx, car, pv):
        tint = car.color
        model = car_model(float(car.pos[0]), float(car.y) + 0.05,
                          float(car.pos[1]), float(car.heading))
        self._draw_mesh(self.car_paint, model, pv, tint=tint,
                        specular=0.35, shininess=48.0)
        self._draw_mesh(self.car_trim, model, pv, specular=0.55, shininess=72.0)
        steer = 0.0
        act = game.last_action[idx] if idx < len(game.last_action) else None
        if act is not None:
            steer = float(act.steer) * 0.45
        spin = -float(car.distance) / max(WHEEL_R, 1e-3)
        for (wx, wy, wz, front) in WHEELS:
            wm = multiply(model, translation(wx, wy, wz),
                          rotation_y(steer if front else 0.0),
                          rotation_z(spin))
            self._draw_mesh(self.wheel, wm, pv, specular=0.30, shininess=40.0)

    def _height_at(self, x, z):
        if self.track is None:
            return 0.0
        try:
            i, t, p, d, lat, tang = self.track.nearest((x, z), None)
            return self.track.height_at_index(i, t)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def draw(self, game, camera="chase", watch=0, show_watch=True, dt=1 / 60.0,
             countdown=0.0, paused=False, detail=False, debug=False):
        now = time.perf_counter()
        if self._last_draw is not None:
            self._account_frame(now - self._last_draw)
        self._last_draw = now
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        proj, view = self.camera.update(game, camera, watch, dt,
                                        self.width, self.height)
        pv = multiply(proj, view)
        self._draw_sky()

        self.lit.use()
        self._lighting_uniforms(self.camera.eye)
        if self.scene_mesh is not None:
            self._draw_mesh(self.scene_mesh, identity(), pv, specular=0.05,
                            shininess=18.0)
        if self.gloss_mesh is not None:
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-3.0, -3.0)
            self._draw_mesh(self.gloss_mesh, identity(), pv, specular=0.45,
                            shininess=64.0)
            glDisable(GL_POLYGON_OFFSET_FILL)

        self._draw_shadows(game, pv)

        for i, car in enumerate(game.cars):
            self._draw_car(game, i, car, pv)

        glUseProgram(0)
        self._draw_hud(game, camera, watch, show_watch, countdown, paused,
                       detail, debug)

    # ------------------------------------------------------------------
    @staticmethod
    def _hud_signature(game, camera, watch, show_watch, countdown, paused,
                       detail, debug):
        """Everything the HUD shows that changes *discretely*.

        Speed and the clock change continuously and can ride the refresh rate;
        these must appear on the frame they happen or the overlay visibly
        trails the race, so they force an immediate rebuild.
        """
        # The countdown goes in as the *caption* it draws ("3", "2", "1",
        # "GO!"), not as the raw float.  The float changes on every frame of
        # the 3 s intro, and each change used to repaint the whole HUD and
        # re-upload 8 MB of texture -- a per-frame cost that existed only
        # during the countdown.
        return (countdown_label(countdown), bool(paused), bool(detail),
                bool(debug), game.state, camera, watch, show_watch,
                getattr(game, "winner", None), getattr(game, "end_reason", ""),
                len(getattr(game, "events", ()) or ()))

    def _draw_hud(self, game, camera, watch, show_watch, countdown, paused,
                  detail, debug):
        import pygame
        resized = self._hud_tex is None or self._hud_size != (self.width,
                                                              self.height)
        if resized:
            if self._hud_tex is not None:
                glDeleteTextures([self._hud_tex])
            self._hud_tex = create_texture(self.width, self.height)
            self._hud_size = (self.width, self.height)
            self._hud_sig = None

        now = time.time()
        sig = self._hud_signature(game, camera, watch, show_watch, countdown,
                                  paused, detail, debug)
        if sig != self._hud_sig or now - self._hud_t >= 1.0 / self.HUD_HZ:
            self._hud_sig = sig
            self._hud_t = now
            self.hud.detail = bool(detail)
            self.hud.frame_ms = self.frame_ms
            surf = self.hud.paint(game, camera=camera, watch=watch,
                                  show_watch=show_watch, countdown=countdown,
                                  paused=paused, debug=debug)
            # flipped=False: one straight copy beats a per-row flip of 8 MB;
            # OV_VS flips V so the quad still comes out the right way up
            data = pygame.image.tobytes(surf, "RGBA")
            glBindTexture(GL_TEXTURE_2D, self._hud_tex)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                            GL_RGBA, GL_UNSIGNED_BYTE, data)

        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glDisable(GL_CULL_FACE)
        self.overlay.use()
        self.overlay.set_mat4("uProj", _OVERLAY_ORTHO)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._hud_tex)
        self.overlay.set_int("uTex", 0)
        self.quad.draw()
        glEnable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glUseProgram(0)

    # ------------------------------------------------------------------
    def toggle_detail(self) -> bool:
        self.hud.detail = not self.hud.detail
        return self.hud.detail

    def resize(self, width: int, height: int) -> None:
        self.width, self.height = int(width), int(height)
        self.hud.resize(width, height)

    def close(self):
        for prog in (self.lit, self.sky_prog, self.overlay):
            prog.close()
        for mesh in (self.car_paint, self.car_trim, self.wheel, self.blob,
                     self.scene_mesh, self.gloss_mesh, self.scene_shadow):
            if mesh is not None:
                mesh.close()
        try:
            if self._hud_tex is not None:
                glDeleteTextures([self._hud_tex])
        except Exception:
            pass
