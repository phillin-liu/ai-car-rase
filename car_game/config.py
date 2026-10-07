"""Configuration dataclasses and persistence helpers."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from .providers import (DEFAULT_MODELS, PROVIDER_LABELS, PROVIDERS,
                        ProviderSpec, get as get_provider, is_known,
                        normalize_provider, provider_ids)

if getattr(sys, "frozen", False):
    # Frozen build (PyInstaller): keep config/, projects/ and runtime/ next to
    # the executable so settings, API keys and match history survive restarts.
    # ``sys._MEIPASS`` is a throw-away temp dir in one-file builds, so it must
    # never be used as the data root.
    ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "config")
PROJECTS_DIR = os.path.join(ROOT, "projects")
RUNTIME_DIR = os.path.join(ROOT, "runtime")
KEYS_FILE = os.path.join(CONFIG_DIR, "keys.json")


@dataclass
class ModelConfig:
    provider: str = "ollama"         # one of car_game.providers.PROVIDERS
    model: str = ""
    display_name: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.3
    max_tokens: int = 300
    timeout: float = 20.0
    decision_interval: float = 1.0   # seconds between LLM queries
    # --- local vision (program-side perception, never text) -------------
    vision_image_size: int = 256     # rendered frame side length (px)
    vision_ahead_m: float = 80.0     # how far ahead the frame reaches
    vision_view: str = "first"       # "first" (3D driver view) | "topdown"
    vision_fov: float = 58.0         # vertical field of view for the 3D view
    # Anti-aliasing for the realistic frame shown to the model: the frame is
    # rendered at N x and downsampled.  1 disables it.  Never applied to the
    # flat frame the built-in detector segments on.
    vision_supersample: int = 2
    # Optional learned *detector* weights (.pt/.pth/.onnx) for the companion
    # detector that reads the frame alongside a remote model.  Empty = the
    # built-in colour detector.  Loading these needs ``ultralytics``; there is
    # no longer any UI for this field -- it is a settings.json-only override.
    vision_weights: str = ""

    def spec(self) -> ProviderSpec:
        return get_provider(self.provider)

    def resolved_key(self) -> str:
        if self.api_key:
            return self.api_key
        spec = self.spec()
        return os.environ.get(spec.key_env, "") if spec.key_env else ""

    def resolved_auth_token(self) -> str:
        """Bearer auth token (``ANTHROPIC_AUTH_TOKEN``).

        This is *not* an API key: the Anthropic SDK takes it as ``auth_token``
        and it must never be passed as ``api_key``.
        """
        if self.spec().kind == "anthropic":
            return os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        return ""

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        env = {"openai": "OPENAI_BASE_URL",
               "anthropic": "ANTHROPIC_BASE_URL"}.get(self.spec().kind, "")
        from_env = os.environ.get(env, "") if env else ""
        return from_env or self.spec().base_url

    def needs_key(self) -> bool:
        """True when this side cannot run without an API key."""
        spec = self.spec()
        return spec.key_required and not (self.resolved_key()
                                          or self.resolved_auth_token())

    def label(self) -> str:
        if self.display_name:
            return self.display_name
        return f"{self.provider}:{self.model}"


@dataclass
class MatchConfig:
    mode: str = "ai_vs_ai"            # "ai_vs_ai" | "human_vs_ai"
    human_player: int = 0             # which car the human controls (0 or 1)
    laps: int = 2
    matches: int = 5                  # 0 == infinite until stopped
    render: bool = True
    headless: bool = False
    view: str = "chase"               # chase | first | overhead | free
    spectate: bool = True             # allow watching AI POV in ai_vs_ai
    items_enabled: bool = True
    window_width: int = 1280
    window_height: int = 720
    fps: int = 60
    realtime: bool = True             # headless: False => run as fast as possible
    track_width: float = 16.0
    difficulty: float = 1.0
    ai_decision_interval: float = 1.0
    debug: bool = False               # show AI decisions / telemetry overlay
    msaa: bool = True                 # 4x multisampling when the driver allows it
    auto_unstick: bool = True         # put a car back on the centre line if stuck
    auto_unstick_delay: float = 2.5   # seconds without progress before that happens

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MatchConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


def _default_side(display: str) -> ModelConfig:
    """Default side.

    Ollama, because it is the one provider that needs no key: the program no
    longer ships an AI of its own, so "works out of the box" now means "talks
    to a local Ollama server", not "drives itself".
    """
    return ModelConfig(provider="ollama", model=DEFAULT_MODELS["ollama"],
                       display_name=display)


@dataclass
class GameConfig:
    match: MatchConfig = field(default_factory=MatchConfig)
    p1: ModelConfig = field(default_factory=lambda: _default_side("Ollama P1"))
    p2: ModelConfig = field(default_factory=lambda: _default_side("Ollama P2"))

    def to_dict(self) -> dict:
        return {
            "match": self.match.to_dict(),
            "p1": dataclasses.asdict(self.p1),
            "p2": dataclasses.asdict(self.p2),
        }

    def to_public_dict(self) -> dict:
        """Like :meth:`to_dict` but with API keys stripped.

        Used for anything written to disk (settings, session files, per-match
        result payloads): the key lives only in ``config/keys.json`` and is
        re-hydrated by :func:`apply_saved_keys` on load.
        """
        data = self.to_dict()
        for key in ("p1", "p2"):
            if isinstance(data.get(key), dict):
                data[key]["api_key"] = ""
        return data

    @classmethod
    def from_dict(cls, d: dict) -> "GameConfig":
        d = d or {}
        g = cls()
        if "match" in d:
            g.match = MatchConfig.from_dict(d["match"])
        for key in ("p1", "p2"):
            if key in d and isinstance(d[key], dict):
                names = {f.name for f in dataclasses.fields(ModelConfig)}
                side = ModelConfig(**{k: v for k, v in d[key].items()
                                      if k in names})
                # legacy configs may name providers this build no longer
                # offers (local / rule / mock) -- never load a non-vision
                # backend.  Such a config also carries that provider's model
                # name and label, both of which are meaningless on the
                # provider it now maps to, so they are re-derived instead of
                # being carried over ("ollama / local-vision-v1" would fail
                # on the first request).
                raw = str(side.provider or "")
                side.provider = normalize_provider(raw)
                if not side.model or not is_known(raw):
                    side.model = DEFAULT_MODELS.get(side.provider, "")
                    side.display_name = ""
                setattr(g, key, side)
        return g

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_public_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "GameConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# API key storage
# ---------------------------------------------------------------------------
def load_keys() -> dict:
    try:
        with open(KEYS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_keys(keys: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(KEYS_FILE, "w", encoding="utf-8") as fh:
        json.dump(keys, fh, ensure_ascii=False, indent=2)


def apply_saved_keys(cfg: GameConfig) -> GameConfig:
    keys = load_keys()
    for side in (cfg.p1, cfg.p2):
        if not side.api_key and side.provider in keys:
            side.api_key = keys.get(side.provider, "")
    return cfg


def ensure_dirs() -> None:
    for d in (CONFIG_DIR, PROJECTS_DIR, RUNTIME_DIR):
        os.makedirs(d, exist_ok=True)


def default_settings_path() -> str:
    return os.path.join(CONFIG_DIR, "settings.json")


def load_settings() -> GameConfig:
    path = default_settings_path()
    if os.path.exists(path):
        try:
            return apply_saved_keys(GameConfig.load(path))
        except Exception:
            pass
    return apply_saved_keys(GameConfig())


def save_settings(cfg: GameConfig) -> None:
    ensure_dirs()
    cfg.save(default_settings_path())
