"""Lightweight rollout types shared by CPU pull workers (no SkyRL import)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

TrainingPhase: TypeAlias = str


@dataclass
class TrajectoryID:
    instance_id: str
    repetition_id: int
