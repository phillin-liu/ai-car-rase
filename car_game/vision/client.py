"""HTTP clients for the cloud vision providers.

Shared by the live driver (``ai/llm.py``) and the connection probe
(``vision/check.py``) so the two can never drift apart.  Everything except
Anthropic speaks the OpenAI wire protocol -- that includes Ollama and every
domestic vendor in :mod:`car_game.providers` -- so there is exactly one
OpenAI client factory and one Anthropic client factory.
"""
from __future__ import annotations


def openai_client(cfg):
    """OpenAI-SDK client for any OpenAI-compatible endpoint."""
    from openai import OpenAI
    kwargs: dict = {"timeout": cfg.timeout}
    base = cfg.resolved_base_url()
    if base:
        kwargs["base_url"] = base
    # a keyless endpoint (Ollama) still needs *some* string here
    kwargs["api_key"] = cfg.resolved_key() or "sk-none"
    return OpenAI(**kwargs)


def anthropic_client(cfg):
    """Anthropic SDK client (``auth_token`` and ``api_key`` are distinct)."""
    import anthropic
    kwargs: dict = {"timeout": cfg.timeout}
    base = cfg.resolved_base_url()
    if base:
        kwargs["base_url"] = base
    token = cfg.resolved_auth_token()
    if token:
        # a bearer auth token is NOT an api key -- the SDK takes it as
        # auth_token, and passing it as api_key would be rejected
        kwargs["auth_token"] = token
    else:
        key = cfg.resolved_key()
        if key:
            kwargs["api_key"] = key
    return anthropic.Anthropic(**kwargs)


def client_for(cfg):
    """Build the client matching the provider's wire protocol."""
    if cfg.spec().kind == "anthropic":
        return anthropic_client(cfg)
    return openai_client(cfg)
