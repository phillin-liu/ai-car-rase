"""The vision-model provider registry.

The game only races *vision* models, so this module is the single source of
truth for "which AI backends exist, how do I reach them, and what do they
need from the user".  Everything else (config, driver factory, connection
probe, model scan, UI combos) reads from here instead of carrying its own
hard-coded list of provider strings.

Two kinds of provider:

* ``ollama``    -- a local Ollama server, spoken to over its OpenAI-compatible
                   endpoint.  No network, no key.  This is the offline option:
                   the program no longer ships an AI of its own, because a
                   hand-written driver would not be a model.
* ``openai``    -- every cloud vendor that speaks the OpenAI wire protocol,
                   which is all of them except Anthropic.  Adding a domestic
                   vendor is one line in :data:`PROVIDERS`.

Adding a vendor therefore means editing *one* dict entry, not eleven
hard-coded tuples.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    """Everything the program needs to know about one AI backend."""

    id: str                     # stable key: stored in config, keys.json
    label: str                  # shown in the UI
    kind: str                   # local | ollama | openai | anthropic
    base_url: str = ""          # built-in endpoint (user may override)
    default_model: str = ""
    key_required: bool = True   # False -> never ask for an API key
    key_env: str = ""           # env-var fallback for the key
    models_path: str = "/models"        # where the model list lives
    signup_hint: str = ""       # where to get a key (UI tooltip)
    note: str = ""              # extra UI hint (e.g. 豆包 needs an endpoint id)

    @property
    def listed(self) -> bool:
        """Can the model list be fetched from the provider?"""
        return self.kind in ("ollama", "openai", "anthropic")


# ---------------------------------------------------------------------------
# The registry.  Order matters: it is the order the UI shows.
PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="ollama", label="Ollama（本地服务）", kind="ollama",
        base_url="http://localhost:11434/v1", default_model="llava",
        key_required=False, key_env="OLLAMA_API_KEY",
        signup_hint="无需密钥；先在本机运行 ollama serve 并 pull 一个视觉模型",
        note="需要本机安装并启动 Ollama。",
    ),
    ProviderSpec(
        id="openai", label="OpenAI", kind="openai",
        base_url="https://api.openai.com/v1", default_model="gpt-4o-mini",
        key_env="OPENAI_API_KEY",
        signup_hint="platform.openai.com 申请，或设置 OPENAI_API_KEY",
    ),
    ProviderSpec(
        id="anthropic", label="Anthropic", kind="anthropic",
        default_model="claude-sonnet-5-5", key_env="ANTHROPIC_API_KEY",
        signup_hint="console.anthropic.com 申请，或设置 ANTHROPIC_API_KEY",
    ),
    # --- 国内厂商：均为 OpenAI 兼容接口，只需填 API 密钥 ---------------
    ProviderSpec(
        id="deepseek", label="DeepSeek（深度求索）", kind="openai",
        base_url="https://api.deepseek.com/v1", default_model="deepseek-chat",
        key_env="DEEPSEEK_API_KEY",
        signup_hint="platform.deepseek.com 申请密钥",
    ),
    ProviderSpec(
        id="zhipu", label="智谱 GLM", kind="openai",
        base_url="https://open.bigmodel.cn/api/paas/v4", default_model="glm-4v",
        key_env="ZHIPU_API_KEY",
        signup_hint="open.bigmodel.cn 申请密钥",
    ),
    ProviderSpec(
        id="qwen", label="通义千问（DashScope）", kind="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-vl-plus", key_env="DASHSCOPE_API_KEY",
        signup_hint="dashscope.console.aliyun.com 申请密钥",
    ),
    ProviderSpec(
        id="moonshot", label="Kimi（月之暗面）", kind="openai",
        base_url="https://api.moonshot.cn/v1", default_model="moonshot-v1-8k-vision-preview",
        key_env="MOONSHOT_API_KEY",
        signup_hint="platform.moonshot.cn 申请密钥",
    ),
    ProviderSpec(
        id="doubao", label="豆包（火山引擎）", kind="openai",
        base_url="https://ark.cn-beijing.volces.com/api/v3", default_model="",
        key_env="ARK_API_KEY",
        signup_hint="console.volcengine.com/ark 申请密钥",
        note="模型名称要填方舟的「接入点 ID」（ep-…），不是模型显示名。",
    ),
    ProviderSpec(
        id="siliconflow", label="硅基流动 SiliconFlow", kind="openai",
        base_url="https://api.siliconflow.cn/v1", default_model="Qwen/Qwen2-VL-72B-Instruct",
        key_env="SILICONFLOW_API_KEY",
        signup_hint="siliconflow.cn 申请密钥",
    ),
    ProviderSpec(
        id="minimax", label="MiniMax", kind="openai",
        base_url="https://api.minimax.chat/v1", default_model="abab6.5s-chat",
        key_env="MINIMAX_API_KEY",
        signup_hint="platform.minimaxi.com 申请密钥",
    ),
    ProviderSpec(
        id="yi", label="零一万物 Yi", kind="openai",
        base_url="https://api.lingyiwanwu.com/v1", default_model="yi-vision",
        key_env="YI_API_KEY",
        signup_hint="platform.lingyiwanwu.com 申请密钥",
    ),
    ProviderSpec(
        id="compatible", label="自定义（兼容 OpenAI 接口）", kind="openai",
        base_url="", default_model="",
        key_env="OPENAI_API_KEY",
        signup_hint="填写任意 OpenAI 兼容服务的地址与密钥",
        note="自建 vLLM / LM Studio / 中转站请用这一项。",
    ),
)

_BY_ID = {spec.id: spec for spec in PROVIDERS}

# The fallback has to be a provider that can actually see and that costs
# nothing to try: Ollama, the local server.  ``local`` itself is here because
# the program no longer ships a driver of its own -- an old config naming it
# has to land on something raceable.
_FALLBACK = "ollama"

# provider ids this game no longer offers; old configs/session files may still
# name them, so they are mapped onto something that can actually see.
_LEGACY = {"rule": _FALLBACK, "mock": _FALLBACK, "human-fallback": _FALLBACK,
           "local": _FALLBACK}


def get(provider_id: str | None) -> ProviderSpec:
    """Spec for ``provider_id``; unknown/legacy ids fall back to Ollama."""
    pid = (provider_id or "").strip().lower()
    pid = _LEGACY.get(pid, pid)
    return _BY_ID.get(pid) or _BY_ID[_FALLBACK]


def normalize_provider(provider_id: str | None) -> str:
    """Canonical provider id (never an unknown string)."""
    return get(provider_id).id


def is_known(provider_id: str | None) -> bool:
    return (provider_id or "").strip().lower() in _BY_ID


def provider_ids() -> tuple[str, ...]:
    return tuple(spec.id for spec in PROVIDERS)


# ---------------------------------------------------------------------------
# Convenience tables (same names the rest of the code already imports).
PROVIDER_LABELS: dict[str, str] = {spec.id: spec.label for spec in PROVIDERS}
DEFAULT_MODELS: dict[str, str] = {spec.id: spec.default_model for spec in PROVIDERS}
