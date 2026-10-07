"""Vision-capability probe used by the "test connection" button.

A connection test is no longer just "can I reach the API": the program only
races vision models, so the probe sends a tiny image with known shapes/colours
and only accepts the model when it both accepts image input and reacts to it.
A model that is reachable but text-only is reported as *not* a vision model,
so the user can pick a different one instead of racing blind.

The provider clients come from :mod:`car_game.vision.client`, the same
factories ``ai/llm.py`` uses, so the probe and the live driver cannot drift
apart.
"""
from __future__ import annotations

from .render import encode_png_b64

# shapes/colours drawn into the probe image; a real vision model will mention
# at least one of these when asked to describe the picture
_PROBE_TERMS = ("red", "红", "circle", "圆", "green", "绿", "square", "方",
                "blue", "蓝", "triangle", "三角", "shape", "图形")
_VISION_ERROR_HINTS = ("vision", "multimodal", "image", "content type",
                       "unsupported", "not support", "does not support",
                       "invalid content", "text only", "expected text",
                       "content_type", "image_url", "media_type")


def make_probe_image():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (160, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([18, 38, 62, 82], fill=(220, 30, 30))          # red circle
    draw.rectangle([70, 40, 112, 82], fill=(30, 170, 60))       # green square
    draw.polygon([(80, 108), (110, 108), (95, 88)], fill=(30, 60, 220))
    return img


# ---------------------------------------------------------------------------
def openai_client_for(cfg):
    """Kept as the historical name; the factory lives in vision.client."""
    from .client import openai_client
    return openai_client(cfg)


def anthropic_client_for(cfg):
    from .client import anthropic_client
    return anthropic_client(cfg)


# ---------------------------------------------------------------------------
def _usable_as_vision(text: str) -> bool:
    low = (text or "").lower()
    return any(t in low for t in _PROBE_TERMS)


def _looks_like_vision_error(text: str) -> bool:
    low = (text or "").lower()
    return any(h in low for h in _VISION_ERROR_HINTS)


def _probe_openai(cfg):
    client = openai_client_for(cfg)
    b64 = encode_png_b64(make_probe_image())
    resp = client.chat.completions.create(
        model=cfg.model, max_tokens=60,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "用一句话描述这张图里的形状和颜色。"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}])
    return resp.choices[0].message.content or ""


def _probe_anthropic(cfg):
    client = anthropic_client_for(cfg)
    b64 = encode_png_b64(make_probe_image())
    resp = client.messages.create(
        model=cfg.model, max_tokens=60,
        messages=[{"role": "user", "content": [
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": "用一句话描述这张图里的形状和颜色。"},
        ]}])
    return "".join(b.text for b in resp.content
                   if getattr(b, "type", "") == "text")


def probe_model(cfg) -> tuple[bool, bool | None, str]:
    """``(usable, is_vision, message)``.

    ``usable`` is True only when the side can actually drive with vision.
    ``is_vision`` is True / False / None (connection failed, unknown).
    """
    spec = cfg.spec()
    if cfg.needs_key():
        return False, None, "缺少 API 密钥 / 认证令牌"
    try:
        text = (_probe_anthropic(cfg) if spec.kind == "anthropic"
                else _probe_openai(cfg))
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if _looks_like_vision_error(msg):
            return False, False, "该模型不支持图像输入（非视觉模型）"
        return False, None, f"连接失败: {msg[:90]}"
    if _usable_as_vision(text):
        return True, True, f"连接成功：视觉模型（支持图像输入）{text.strip()[:24]!r}"
    return False, False, "连接成功但未识别图像内容，疑似非视觉模型"


def test_connection(cfg) -> tuple[bool, str]:
    """Backwards-compatible ``(ok, msg)`` wrapper around :func:`probe_model`."""
    ok, _is_vision, msg = probe_model(cfg)
    return ok, msg
