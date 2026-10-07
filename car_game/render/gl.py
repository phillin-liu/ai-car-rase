"""Low level OpenGL helpers: shaders, meshes, textures, GL state.

The lighting model is Blinn-Phong (directional sun + hemispheric ambient +
specular) with distance fog; the sky is a full-screen gradient drawn before
the scene so the horizon and the fog colour match seamlessly.
"""
from __future__ import annotations

import ctypes

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders


# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------
LIT_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aColor;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMat;
out vec3 vNormal;
out vec3 vColor;
out vec3 vWorld;
void main(){
    vec4 wp = uModel * vec4(aPos, 1.0);
    vWorld = wp.xyz;
    vNormal = uNormalMat * aNormal;
    vColor = aColor;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

LIT_FS = """
#version 330 core
in vec3 vNormal;
in vec3 vColor;
in vec3 vWorld;
uniform vec3 uLightDir;      // unit vector pointing TOWARDS the sun
uniform vec3 uCamPos;
uniform vec3 uSkyColor;
uniform vec3 uGroundColor;
uniform vec3 uFogColor;
uniform vec2 uFogRange;      // (start, end) metres
uniform vec3 uTint;
uniform float uAmbient;
uniform float uDiffuse;
uniform float uSpecular;
uniform float uShininess;
uniform float uAlpha;
out vec4 FragColor;
void main(){
    vec3 n = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uCamPos - vWorld);
    vec3 H = normalize(L + V);
    float diff = max(dot(n, L), 0.0);
    float hemi = 0.5 + 0.5 * n.y;
    vec3 base = vColor * uTint;
    vec3 ambient = base * mix(uGroundColor, uSkyColor, hemi) * uAmbient;
    vec3 col = ambient + base * diff * uDiffuse;
    float spec = pow(max(dot(n, H), 0.0), max(uShininess, 1.0)) * uSpecular;
    col += vec3(spec) * diff;
    float dist = length(uCamPos - vWorld);
    col = mix(col, uFogColor, smoothstep(uFogRange.x, uFogRange.y, dist));
    FragColor = vec4(col, uAlpha);
}
"""

SKY_VS = """
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main(){
    vUV = aPos;
    gl_Position = vec4(aPos * 2.0 - 1.0, 1.0, 1.0);
}
"""

SKY_FS = """
#version 330 core
in vec2 vUV;
uniform vec3 uTopColor;
uniform vec3 uHorizonColor;
out vec4 FragColor;
void main(){
    vec3 col = mix(uHorizonColor, uTopColor, pow(clamp(vUV.y, 0.0, 1.0), 0.62));
    FragColor = vec4(col, 1.0);
}
"""

OV_VS = """
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
uniform mat4 uProj;
out vec2 vUV;
void main(){
    // the HUD surface is uploaded top-down (a straight copy instead of a
    // per-row flip of the whole window), so flip V here to match
    vUV = vec2(aUV.x, 1.0 - aUV.y);
    gl_Position = uProj * vec4(aPos, 0.0, 1.0);
}
"""

OV_FS = """
#version 330 core
in vec2 vUV;
uniform sampler2D uTex;
out vec4 FragColor;
void main(){
    vec4 c = texture(uTex, vUV);
    FragColor = c;
}
"""


# ---------------------------------------------------------------------------
def ctypes_void(offset: int):
    return ctypes.c_void_p(offset)


def compile_program(vs: str, fs: str) -> int:
    return shaders.compileProgram(
        shaders.compileShader(vs, GL_VERTEX_SHADER),
        shaders.compileShader(fs, GL_FRAGMENT_SHADER))


class Program:
    """A compiled shader program with cached uniform locations."""

    def __init__(self, vs: str, fs: str):
        self.id = compile_program(vs, fs)
        self._loc: dict = {}

    def use(self):
        glUseProgram(self.id)

    def loc(self, name: str) -> int:
        if name not in self._loc:
            self._loc[name] = glGetUniformLocation(self.id, name)
        return self._loc[name]

    def set_mat4(self, name: str, m):
        glUniformMatrix4fv(self.loc(name), 1, GL_TRUE,
                           np.asarray(m, dtype=np.float32))

    def set_mat3(self, name: str, m):
        glUniformMatrix3fv(self.loc(name), 1, GL_TRUE,
                           np.asarray(m, dtype=np.float32))

    def set_vec3(self, name: str, v):
        glUniform3f(self.loc(name), float(v[0]), float(v[1]), float(v[2]))

    def set_vec2(self, name: str, a: float, b: float):
        glUniform2f(self.loc(name), float(a), float(b))

    def set_float(self, name: str, v: float):
        glUniform1f(self.loc(name), float(v))

    def set_int(self, name: str, v: int):
        glUniform1i(self.loc(name), int(v))

    def close(self):
        try:
            glDeleteProgram(self.id)
        except Exception:
            pass


class Mesh:
    """Static indexed triangle mesh (pos3 / normal3 / color3)."""

    STRIDE = 9 * 4

    def __init__(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        self.count = 0

    def upload(self, verts, indices):
        arr = np.asarray(verts, dtype=np.float32).reshape(-1, 9)
        idx = np.asarray(indices, dtype=np.uint32).reshape(-1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
        for loc, off in ((0, 0), (1, 12), (2, 24)):
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, self.STRIDE,
                                  ctypes_void(off))
        glBindVertexArray(0)
        self.count = len(idx)

    def draw(self):
        if self.count == 0:
            return
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def close(self):
        try:
            glDeleteBuffers(1, [self.vbo])
            glDeleteBuffers(1, [self.ebo])
            glDeleteVertexArrays(1, [self.vao])
        except Exception:
            pass


class OverlayQuad:
    """Unit quad with 0..1 UVs, used for the sky and the HUD texture."""

    def __init__(self):
        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        verts = np.array([
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
        ], dtype=np.float32)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes_void(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes_void(8))
        glBindVertexArray(0)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)


def create_texture(w: int, h: int) -> int:
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, None)
    return tex


def apply_common_state():
    """Depth + culling + blending state shared by the whole frame."""
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    try:
        glEnable(GL_MULTISAMPLE)
    except Exception:
        pass


def normal_matrix(model) -> np.ndarray:
    """Inverse-transpose of the upper-left 3x3 (correct under non-uniform scale)."""
    m = np.asarray(model, dtype=np.float64)[:3, :3]
    try:
        return np.linalg.inv(m).T.astype(np.float32)
    except np.linalg.LinAlgError:
        return m.astype(np.float32)
