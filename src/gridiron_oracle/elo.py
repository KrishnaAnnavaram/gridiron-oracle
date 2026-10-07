"""A plain-Python Elo engine over date-ordered games. It records the PRE-GAME ratings of every game.

The prototype ran Elo over all games (also the test games) in an unordered graph query and then
predicted the test games from the final ratings. Here the engine walks the games in date order, and
the prediction for a game uses only the ratings before that game.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .config import EloConfig


@dataclass
class EloResult:
    games: pd.DataFrame  # input games plus elo_home_pre, elo_away_pre, elo_p_home
    ratings: dict[str, float]  # ratings after the last game


def expected_home(home: float, away: float, home_advantage: float) -> float:
    return 1 / (1 + 10 ** (-(home - away + home_advantage) / 400))


def mov_multiplier(margin: float, winner_elo_diff: float) -> float:
    """Margin-of-victory multiplier: grows with the log of the margin, shrinks for expected blowouts."""
    if margin == 0:
        return 1.0
    return math.log(abs(margin) + 1) * 2.2 / (winner_elo_diff * 0.001 + 2.2)


def run_elo(games: pd.DataFrame, cfg: EloConfig = EloConfig()) -> EloResult:
    need = {"game_id", "season", "game_date", "home_team", "away_team", "home_score", "away_score"}
    missing = need - set(games.columns)
    if missing:
        raise ValueError(f"games miss columns {sorted(missing)}")
    g = games.sort_values(["game_date", "game_id"]).reset_index(drop=True).copy()
    ratings: dict[str, float] = {}
    season = None
    pre_h, pre_a, p_home = [], [], []
    for row in g.itertuples(index=False):
        if season is not None and row.season != season:
            for t in ratings:  # regression to the mean between seasons
                ratings[t] = ratings[t] + cfg.revert * (cfg.mean - ratings[t])
        season = row.season
        rh = ratings.setdefault(row.home_team, cfg.start)
        ra = ratings.setdefault(row.away_team, cfg.start)
        e = expected_home(rh, ra, cfg.home_advantage)
        pre_h.append(rh)
        pre_a.append(ra)
        p_home.append(e)
        margin = row.home_score - row.away_score
        s = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        mult = 1.0
        if cfg.mov and margin != 0:
            diff = (rh + cfg.home_advantage - ra) if margin > 0 else (ra - rh - cfg.home_advantage)
            mult = mov_multiplier(margin, diff)
        delta = cfg.k * mult * (s - e)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta
    g["elo_home_pre"], g["elo_away_pre"], g["elo_p_home"] = pre_h, pre_a, p_home
    return EloResult(g, ratings)
