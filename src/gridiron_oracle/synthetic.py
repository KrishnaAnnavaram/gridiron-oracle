"""A synthetic NFL-like league with play-by-play rows in the nflverse column style.

Teams have hidden offence and defence strengths that drift inside a season and regress between
seasons. Each game is a sequence of drives. A drive ends in a touchdown, a field goal, a turnover or
a punt, with probabilities from the strengths and a home advantage. No real team, player or game.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = tuple(f"T{k:02d}" for k in range(32))


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def _schedule(rng: np.random.Generator, weeks: int) -> list[tuple[int, str, str]]:
    games = []
    for week in range(1, weeks + 1):
        order = rng.permutation(len(TEAMS))
        for a, b in zip(order[::2], order[1::2]):
            home, away = (TEAMS[a], TEAMS[b]) if rng.random() < 0.5 else (TEAMS[b], TEAMS[a])
            games.append((week, home, away))
    return games


def make_pbp(seasons=range(2009, 2019), weeks: int = 16, seed: int = 7, home_edge: float = 0.18) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    off = {t: rng.normal(0, 0.6) for t in TEAMS}
    dfn = {t: rng.normal(0, 0.6) for t in TEAMS}
    cols: dict[str, list] = {c: [] for c in (
        "season", "week", "game_id", "game_date", "home_team", "away_team", "play_id", "posteam", "defteam", "qtr",
        "play_type", "yards_gained", "touchdown", "field_goal_result", "interception", "fumble_lost",
        "third_down_converted", "third_down_failed", "penalty_yards", "total_home_score", "total_away_score")}
    for season in seasons:
        for t in TEAMS:  # regression between seasons
            off[t] = 0.6 * off[t] + rng.normal(0, 0.45)
            dfn[t] = 0.6 * dfn[t] + rng.normal(0, 0.45)
        start = pd.Timestamp(f"{season}-09-08")
        for week, home, away in _schedule(rng, weeks):
            for t in (home, away):  # drift inside the season
                off[t] += rng.normal(0, 0.05)
                dfn[t] += rng.normal(0, 0.05)
            date = (start + pd.Timedelta(days=7 * (week - 1))).strftime("%Y-%m-%d")
            gid = f"{season}_{week:02d}_{away}_{home}"
            score = {home: 0, away: 0}
            n_drives = int(rng.integers(20, 25))
            play_id = 0
            drives = [(home if d % 2 == 0 else away) for d in range(n_drives)]
            qtrs = [min(4, 1 + d * 4 // n_drives) for d in range(n_drives)]
            k = 0
            while k < len(drives):
                pos = drives[k]
                de = away if pos == home else home
                edge = off[pos] - dfn[de] + (home_edge if pos == home else -home_edge)
                p_td = _sigmoid(-1.35 + 0.8 * edge)
                p_to = _sigmoid(-2.0 - 0.5 * edge)
                u = rng.random()
                outcome = "td" if u < p_td else "to" if u < p_td + p_to else "fg" if u < p_td + p_to + 0.16 else "punt"
                n_plays = int(rng.integers(3, 10))
                for p in range(n_plays):
                    play_id += 1
                    last = p == n_plays - 1
                    ptype = "pass" if rng.random() < 0.57 else "run"
                    yards = int(np.clip(rng.normal(5.2 + 1.5 * edge, 7.0 if ptype == "pass" else 4.0), -10, 60))
                    td = ic = fl = 0
                    fg = None
                    if last and outcome == "td":
                        td = 1
                        score[pos] += 7
                    elif last and outcome == "fg":
                        ptype, yards, fg = "field_goal", 0, "made"
                        score[pos] += 3
                    elif last and outcome == "to":
                        if ptype == "pass":
                            ic = 1
                        else:
                            fl = 1
                    elif last and outcome == "punt":
                        ptype, yards = "punt", 0
                    third = (p == n_plays - 2) and n_plays >= 3
                    conv = int(third and outcome in ("td", "fg") and rng.random() < 0.8)
                    fail = int(third and not conv)
                    pen = int(rng.random() < 0.06) * int(rng.choice([5, 10, 15]))
                    vals = (season, week, gid, date, home, away, play_id, pos, de, qtrs[min(k, len(qtrs) - 1)] if k < n_drives else 5,
                            ptype, yards, td, fg, ic, fl, conv, fail, pen, score[home], score[away])
                    for c, v in zip(cols, vals):
                        cols[c].append(v)
                k += 1
                if k == len(drives) and score[home] == score[away] and len(drives) < n_drives + 4:
                    drives += [home if rng.random() < 0.5 else away]  # overtime drive (qtr 5)
    return pd.DataFrame(cols)
