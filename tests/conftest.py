import os

import pytest

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

from gridiron_oracle.features import build_features  # noqa: E402
from gridiron_oracle.games import game_table, team_game_table  # noqa: E402
from gridiron_oracle.synthetic import make_pbp  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("GRIDIRON_", "NEO4J_")):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session")
def pbp():
    return make_pbp(range(2012, 2017), weeks=8, seed=4)


@pytest.fixture(scope="session")
def tables(pbp):
    return game_table(pbp), team_game_table(pbp)


@pytest.fixture(scope="session")
def feats(tables):
    games, long = tables
    return build_features(games, long, window=4)
