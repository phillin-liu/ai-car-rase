"""Driver factory.

Every provider in the game is a vision provider and every one of them is a
remote model, so there is exactly one driver shape: the vision-LLM pipeline in
:mod:`car_game.ai.llm`.

The program has no driving logic of its own -- there is deliberately no
rule-based or built-in fallback, because a hand-written driver is not a model
and would make a run against a real model meaningless.  A side that cannot get
a decision waits rather than being driven by the program.

Legacy provider strings (``local``, ``rule``, ``mock``) are normalised to a
real vision provider rather than silently becoming a non-vision AI.
"""
from __future__ import annotations

from ..config import DEFAULT_MODELS, ModelConfig
from ..providers import get as get_provider, normalize_provider
from .base import BaseDriver
from .llm import LLMDriver


def build_driver(cfg: ModelConfig, track, name: str, difficulty: float = 1.0,
                 stagger: float = 0.0) -> BaseDriver:
    """``stagger`` offsets this side's *later* queries (fraction of the
    decision interval) so two sides do not render their frames on one tick."""
    cfg.provider = normalize_provider(cfg.provider)
    return LLMDriver(cfg, track, name, difficulty, stagger)


def normalize_model_config(cfg: ModelConfig) -> ModelConfig:
    cfg.provider = normalize_provider(cfg.provider)
    if not cfg.model:
        cfg.model = DEFAULT_MODELS.get(cfg.provider, "unknown")
    return cfg
