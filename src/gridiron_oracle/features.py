"""Point-in-time features: each value uses only games that ended BEFORE the game date.

- Rolling form: for each team, the MEAN of the previous ``window`` games (``shift(1)`` then
  ``rolling``). A mean, not the sum of a home mean and an away mean.
- Season to date: the mean of the earlier games of the same season.
- Head to head: earlier meetings of the two teams, with the date filter on BOTH directions.
- Elo: the pre-game ratings from ``elo.run_elo``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EloConfig
from .elo import run_elo

FORM_STATS = ("points_for", "points_against", "yards", "yards_allowed", "giveaways", "takeaways", "third_rate",
              "pass_yards", "rush_yards", "win")


def team_form(long: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Add ``form_*`` (last ``window`` games), ``std_*`` (season to date) and ``rest_days`` to the long table."""
    df = long.sort_values(["team", "game_date", "game_id"]).copy()
    by_team = df.groupby("team", sort=False)
    for stat in FORM_STATS:
        prev = by_team[stat].shift(1)
        df[f"form_{stat}"] = prev.groupby(df.team).transform(lambda s: s.rolling(window, min_periods=1).mean())
        prev_season = df.groupby(["team", "season"])[stat].shift(1)
        df[f"std_{stat}"] = prev_season.groupby([df.team, df.season]).transform(lambda s: s.expanding().mean())
    df["form_games"] = by_team.cumcount().clip(upper=window)
    df["rest_days"] = by_team["game_date"].diff().dt.days
    return df


def head_to_head(games: pd.DataFrame) -> pd.DataFrame:
    """For each game: earlier meetings of the two teams and the mean margin for today's home team.

    A meeting counts only if its date is strictly earlier, in BOTH home/away directions.
    """
    g = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    pair = [tuple(sorted(x)) for x in zip(g.home_team, g.away_team)]
    history: dict[tuple, list[tuple]] = {}
    out = []
    for k, row in enumerate(g.itertuples(index=False)):
        past = [m for m in history.get(pair[k], []) if m[0] < row.game_date]
        if past:
            margins = [m[3] if m[1] == row.home_team else -m[3] for m in past]
            out.append((row.game_id, len(past), float(np.mean(margins))))
        else:
            out.append((row.game_id, 0, np.nan))
        history.setdefault(pair[k], []).append((row.game_date, row.home_team, row.away_team, row.margin))
    return pd.DataFrame(out, columns=["game_id", "h2h_games", "h2h_margin"])


def build_features(games: pd.DataFrame, long: pd.DataFrame, window: int = 8, elo: EloConfig = EloConfig()) -> pd.DataFrame:
    """One row per game: target, Elo, home and away form, and home minus away differences."""
    form = team_form(long, window)
    cols = [c for c in form.columns if c.startswith(("form_", "std_"))] + ["rest_days"]
    home = form[form.is_home == 1][["game_id"] + cols].add_prefix("home_").rename(columns={"home_game_id": "game_id"})
    away = form[form.is_home == 0][["game_id"] + cols].add_prefix("away_").rename(columns={"away_game_id": "game_id"})
    elo_games = run_elo(games, elo).games[["game_id", "elo_home_pre", "elo_away_pre", "elo_p_home"]]
    df = (games.merge(elo_games, on="game_id", validate="one_to_one")
          .merge(home, on="game_id", validate="one_to_one")
          .merge(away, on="game_id", validate="one_to_one")
          .merge(head_to_head(games), on="game_id", validate="one_to_one"))
    for c in cols:
        df[f"diff_{c}"] = df[f"home_{c}"] - df[f"away_{c}"]
    df["elo_diff"] = df.elo_home_pre - df.elo_away_pre
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: everything known before kickoff. Never the score, the margin or the result."""
    banned = {"home_score", "away_score", "margin", "home_win", "overtime"}
    return [c for c in df.columns if c.startswith(("home_form", "away_form", "home_std", "away_std", "diff_",
                                                  "home_rest", "away_rest", "elo_", "h2h_")) and c not in banned]
