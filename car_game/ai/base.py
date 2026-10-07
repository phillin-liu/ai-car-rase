"""Driver base class and item/racing helpers shared by AI implementations."""
from __future__ import annotations

import math

from ..config import ModelConfig
from ..core_types import Action, Observation


class BaseDriver:
    """Interface every controller (rule based / LLM / human) implements."""

    kind = "base"

    def __init__(self, cfg: ModelConfig, track, name: str):
        self.cfg = cfg
        self.track = track
        self.name = name
        self.policy = None            # optional high level policy dict (LLM)
        self.raw_reason = ""

    # -- lifecycle ------------------------------------------------------
    def reset(self, track) -> None:
        self.track = track

    def act(self, obs: Observation) -> Action:  # pragma: no cover - abstract
        raise NotImplementedError

    def prime(self, obs: Observation) -> None:
        """Take a first look before the race clock starts (default: nothing).

        Only drivers whose first decision is not instantaneous need this; the
        engine calls it while the field is held at the line so a slow opening
        look never costs race distance, and then waits for
        :attr:`has_decision` before the countdown.
        """

    @property
    def has_decision(self) -> bool:
        """True once the driver can drive without falling back to coasting."""
        return True

    def close(self) -> None:
        pass

    def stats(self) -> dict:
        return {"driver": self.kind, "model": self.cfg.model}

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi
