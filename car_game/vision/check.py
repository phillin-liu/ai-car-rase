"""Connection / vision-capability probe used by the "test connection" button.

For the vision providers a connection test is not just "can I reach the API":
the probe sends a tiny image with known shapes/colours and only accepts the
model when it both accepts image input and reacts to it.  A model that is
reachable but text-only is reported as *not* a vision model, so the user can
pick a different one instead of racing blind.

The ``sidecar`` provider is not a vision model: its probe checks the served
policy's ``/info`` contract (observation layout and action table) instead.

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


def _probe_sidecar(cfg):
    """A sidecar policy is not a vision model: probe its HTTP contract."""
    from ..ai.sidecar import default_policy_name, fetch_info, policy_name

    info = fetch_info(cfg)
    obs = info.get("observation")
    if not isinstance(obs, dict):
        raise RuntimeError("sidecar-serve 未返回 observation 描述，请升级 sidecar")
    dim = int(info.get("input_dim") or 0)
    described = int(obs.get("dim") or 0)
    if described and described != dim:
        raise RuntimeError(f"观测描述 {described} 维与模型 {dim} 维不一致")
    actions = len(info.get("actions") or [])
    mode = obs.get("mode")
    name = policy_name(cfg, info)
    if name.startswith("http"):       # /info has no name; ask /health
        name = default_policy_name(cfg) or "默认策略"
    return f"{name}（{dim} 维观测 / {actions} 个动作" + (
        f"，mode={mode}）" if mode else "）")


def probe_model(cfg) -> tuple[bool, bool | None, str]:
    """``(usable, is_vision, message)``.

    ``usable`` is True when the side can actually drive.
    ``is_vision`` is True / False / None (connection failed, unknown / not a
    vision provider).
    A sidecar policy reports ``None``: it drives, but from LIDAR/telemetry
    rather than from an image.
    """
    spec = cfg.spec()
    if spec.kind == "sidecar":
        try:
            detail = _probe_sidecar(cfg)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            # a 404 from sidecar-serve names the models it *does* host; that
            # list is the actual fix, so it must not be cut off
            available = getattr(exc, "available", None)
            names = available() if callable(available) else []
            if names:
                msg = f"模型 {cfg.model} 不存在；可用：{'、'.join(names)}"
            return False, None, f"连接失败: {msg[:200]}"
        return True, None, f"连接成功：sidecar 策略模型 {detail}"
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
