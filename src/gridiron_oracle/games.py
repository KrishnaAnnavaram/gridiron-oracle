"""Play-by-play -> one row per game and one row per team per game.

Every aggregation is a ``groupby`` on keys, and every join is a merge on ``game_id`` (and ``team``).
No column is copied between frames by position, so an overtime period or a missing play type
cannot shift values onto another game.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PBP_COLUMNS = (
    "season", "game_id", "game_date", "home_team", "away_team", "posteam", "defteam", "qtr", "play_type",
    "yards_gained", "interception", "fumble_lost", "third_down_converted", "third_down_failed",
    "total_home_score", "total_away_score",
)
OPTIONAL = {"week": np.nan, "touchdown": 0, "field_goal_result": None, "penalty_yards": 0}


class SchemaError(ValueError):
    pass


def validate_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PBP_COLUMNS if c not in pbp.columns]
    if missing:
        raise SchemaError(f"play-by-play misses columns {missing}")
    df = pbp.copy()
    for col, default in OPTIONAL.items():
        if col not in df:
            df[col] = default
    df = df[df.posteam.notna() & df.posteam.astype(str).str.len().gt(0)]
    df["game_date"] = pd.to_datetime(df["game_date"])
    teams = df.groupby("game_id")[["home_team", "away_team"]].nunique()
    if (teams > 1).any().any():
        raise SchemaError("a game_id has more than one home or away team")
    return df


def game_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per game: teams, date, final score, overtime flag."""
    df = validate_pbp(pbp)
    g = df.groupby("game_id").agg(
        season=("season", "first"), week=("week", "first"), game_date=("game_date", "first"),
        home_team=("home_team", "first"), away_team=("away_team", "first"),
        home_score=("total_home_score", "max"), away_score=("total_away_score", "max"),
        overtime=("qtr", lambda q: int((q > 4).any())),
    ).reset_index()
    g["margin"] = g.home_score - g.away_score
    g["home_win"] = np.where(g.margin > 0, 1.0, np.where(g.margin < 0, 0.0, np.nan))  # NaN = tie
    return g.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def team_game_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, with offence stats, the stats allowed and the result."""
    df = validate_pbp(pbp)
    df["is_pass"] = (df.play_type == "pass").astype(int)
    df["is_run"] = (df.play_type == "run").astype(int)
    df["scrimmage"] = df.is_pass | df.is_run
    df["pass_yards"] = df.yards_gained.where(df.is_pass == 1, 0)
    df["rush_yards"] = df.yards_gained.where(df.is_run == 1, 0)
    df["scrimmage_yards"] = df.yards_gained.where(df.scrimmage == 1, 0)
    df["giveaway"] = df.interception.fillna(0) + df.fumble_lost.fillna(0)
    off = df.groupby(["game_id", "posteam"]).agg(
        plays=("scrimmage", "sum"), yards=("scrimmage_yards", "sum"),
        pass_yards=("pass_yards", "sum"), rush_yards=("rush_yards", "sum"), giveaways=("giveaway", "sum"),
        third_conv=("third_down_converted", "sum"), third_fail=("third_down_failed", "sum"),
        penalty_yards=("penalty_yards", "sum"),
    ).reset_index().rename(columns={"posteam": "team"})
    games = game_table(pbp)
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        part = games[["game_id", "season", "week", "game_date", f"{side}_team", f"{other}_team", f"{side}_score",
                      f"{other}_score"]].rename(columns={f"{side}_team": "team", f"{other}_team": "opponent",
                                                        f"{side}_score": "points_for", f"{other}_score": "points_against"})
        part["is_home"] = int(side == "home")
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    long = long.merge(off, on=["game_id", "team"], how="left", validate="one_to_one")
    allowed = off.rename(columns={"team": "opponent"})[["game_id", "opponent", "yards", "giveaways"]].rename(
        columns={"yards": "yards_allowed", "giveaways": "takeaways"})
    long = long.merge(allowed, on=["game_id", "opponent"], how="left", validate="one_to_one")
    att = long.third_conv + long.third_fail
    long["third_rate"] = np.where(att > 0, long.third_conv / att.where(att > 0, 1), np.nan)
    long["win"] = np.where(long.points_for > long.points_against, 1.0,
                           np.where(long.points_for < long.points_against, 0.0, 0.5))
    counts = long.groupby("game_id").size()
    if (counts != 2).any():
        raise SchemaError(f"games without exactly two team rows: {list(counts[counts != 2].index[:3])}")
    return long.sort_values(["game_date", "game_id", "is_home"]).reset_index(drop=True)
