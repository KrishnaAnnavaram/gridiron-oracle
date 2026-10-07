"""Settings from environment variables. Neo4j credentials come ONLY from the environment."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field, SecretStr


def _env(name: str, default: str) -> str:
    v = os.environ.get(name, "")
    return v.strip() or default


class EloConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    k: float = Field(20.0, gt=0)
    home_advantage: float = Field(55.0, ge=0)  # Elo points
    revert: float = Field(1 / 3, ge=0, le=1)  # share of the distance to the mean removed between seasons
    mean: float = 1505.0
    start: float = 1500.0
    mov: bool = True  # margin-of-victory multiplier


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int = 7
    window: int = Field(8, ge=1, le=32)
    elo: EloConfig = EloConfig()
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: SecretStr = SecretStr("")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            seed=int(_env("GRIDIRON_SEED", "7")),
            window=int(_env("GRIDIRON_WINDOW", "8")),
            elo=EloConfig(k=float(_env("GRIDIRON_ELO_K", "20")), home_advantage=float(_env("GRIDIRON_ELO_HFA", "55")),
                          revert=float(_env("GRIDIRON_ELO_REVERT", str(1 / 3)))),
            neo4j_uri=_env("NEO4J_URI", ""),
            neo4j_user=_env("NEO4J_USER", ""),
            neo4j_password=SecretStr(_env("NEO4J_PASSWORD", "")),
        )
