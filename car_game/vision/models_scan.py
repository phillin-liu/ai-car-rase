"""Model discovery -- "which models can this provider actually run?".

Powers the 「扫描模型」 button.  Only remote providers are scanned: they expose
a models endpoint, the OpenAI SDK covers the OpenAI-compatible ones, and a
plain ``GET {base_url}/models`` is the fallback for servers whose SDK support
lags (and for Ollama's native tag list).  The ``local`` side runs the
built-in detector and has no model list at all.

Pure functions, no UI: the caller turns exceptions into messages.
"""
from __future__ import annotations


class ScanError(RuntimeError):
    """Model discovery failed; the message is user-facing (Chinese)."""


# ---------------------------------------------------------------------------
def _remote_ids_from_openai_sdk(cfg) -> list[str]:
    from .client import client_for
    client = client_for(cfg)
    page = client.models.list()
    ids = []
    for item in page:
        mid = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        if mid:
            ids.append(str(mid))
    return ids


def _remote_ids_via_http(cfg) -> list[str]:
    """Plain ``GET {base_url}/models`` -- works even where the SDK chokes."""
    import requests
    base = cfg.resolved_base_url().rstrip("/")
    if not base:
        raise ScanError("该提供方还没有接口地址，请先填写 API Base URL")
    headers = {}
    key = cfg.resolved_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        r = requests.get(base + cfg.spec().models_path, headers=headers,
                         timeout=min(float(cfg.timeout), 30.0))
    except Exception as exc:  # noqa: BLE001
        hint = ("请确认本机已启动 Ollama（ollama serve）"
                if cfg.spec().kind == "ollama" else "请检查接口地址与网络")
        raise ScanError(f"无法连接 {base}：{hint}（{str(exc)[:80]}）") from exc
    if r.status_code >= 400:
        raise ScanError(f"接口返回 {r.status_code}: {r.text[:120]}")
    try:
        return _ids_from_payload(r.json())
    except ValueError as exc:
        raise ScanError(f"接口返回的不是模型列表：{r.text[:80]}") from exc


def _ids_from_payload(data) -> list[str]:
    """Accept both OpenAI (``data[].id``) and Ollama (``models[].name``)."""
    ids: list[str] = []
    if isinstance(data, dict):
        for item in data.get("data") or []:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
        for item in data.get("models") or []:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model") or item.get("id")
                if name:
                    ids.append(str(name))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, dict):
                name = item.get("id") or item.get("name")
                if name:
                    ids.append(str(name))
    return ids


def list_remote_models(cfg) -> list[str]:
    """Ask a remote provider which models it offers."""
    if cfg.needs_key() and cfg.spec().kind != "ollama":
        raise ScanError("该提供方需要 API 密钥，请先填写密钥再扫描")
    ids: list[str] = []
    try:
        ids = _remote_ids_from_openai_sdk(cfg)
    except Exception:
        ids = []
    if not ids:
        ids = _remote_ids_via_http(cfg)      # raises ScanError with a reason
    if not ids and cfg.spec().kind == "ollama":
        ids = _ollama_native_tags(cfg)
    # dedupe, keep the provider's own ordering
    seen: set[str] = set()
    out = [m for m in ids if not (m in seen or seen.add(m))]
    if not out:
        raise ScanError("接口没有返回任何模型")
    return out


def _ollama_native_tags(cfg) -> list[str]:
    """``/api/tags`` on the Ollama host (the OpenAI shim hides some models)."""
    import requests
    base = cfg.resolved_base_url().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        r = requests.get(base + "/api/tags", timeout=min(float(cfg.timeout), 30.0))
        if r.status_code >= 400:
            return []
        return _ids_from_payload(r.json())
    except Exception:
        return []


def list_models(cfg) -> list[str]:
    """All model ids ``cfg`` could drive with.

    The local side has no model list: it always runs the built-in detector.
    """
    return list_remote_models(cfg)
