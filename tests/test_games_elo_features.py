import numpy as np
import pandas as pd
import pytest

from gridiron_oracle.config import EloConfig, Settings
from gridiron_oracle.elo import expected_home, mov_multiplier, run_elo
from gridiron_oracle.features import build_features, feature_columns, head_to_head, team_form
from gridiron_oracle.games import SchemaError, game_table, team_game_table


def test_game_table_scores_and_ties(pbp, tables):
    games, _ = tables
    assert len(games) == pbp.game_id.nunique()
    one = pbp[pbp.game_id == games.game_id.iloc[0]]
    assert games.home_score.iloc[0] == one.total_home_score.max()
    ties = games[games.margin == 0]
    assert ties.home_win.isna().all()
    assert set(games.home_win.dropna().unique()) <= {0.0, 1.0}


def test_team_rows_join_on_keys(pbp, tables):
    # Problem 4: values belong to the right game, also for overtime games.
    games, long = tables
    assert (long.groupby("game_id").size() == 2).all()
    ot = games[games.overtime == 1]
    assert len(ot) > 0
    gid = ot.game_id.iloc[0]
    row = long[(long.game_id == gid) & (long.is_home == 1)].iloc[0]
    g = games.set_index("game_id").loc[gid]
    assert row.points_for == g.home_score and row.points_against == g.away_score
    plays = pbp[(pbp.game_id == gid) & (pbp.posteam == row.team)]
    assert row.giveaways == plays.interception.sum() + plays.fumble_lost.sum()
    opp = long[(long.game_id == gid) & (long.is_home == 0)].iloc[0]
    assert row.yards_allowed == opp.yards and row.takeaways == opp.giveaways


def test_schema_errors(pbp):
    with pytest.raises(SchemaError, match="misses columns"):
        game_table(pbp.drop(columns=["qtr"]))
    bad = pbp.copy()
    bad.loc[bad.index[0], "home_team"] = "ZZ"
    with pytest.raises(SchemaError, match="more than one"):
        game_table(bad)


def test_elo_uses_pre_game_ratings_in_date_order(tables):
    # Problem 1: the prediction of a game uses only the ratings before it.
    games, _ = tables
    shuffled = games.sample(frac=1, random_state=0)
    res = run_elo(shuffled)
    assert res.games.game_date.is_monotonic_increasing
    first = res.games.iloc[0]
    assert first.elo_home_pre == first.elo_away_pre == 1500
    # Changing the score of the LAST game does not change any pre-game value.
    changed = games.copy()
    last = changed.index[-1]
    changed.loc[last, "home_score"] += 30
    again = run_elo(changed)
    assert np.allclose(again.games.elo_p_home, res.games.elo_p_home)
    assert again.ratings != res.ratings


def test_elo_math():
    assert expected_home(1500, 1500, 0) == 0.5
    assert expected_home(1500, 1500, 55) > 0.5
    assert mov_multiplier(0, 0) == 1.0
    assert mov_multiplier(21, 0) > mov_multiplier(3, 0)
    assert mov_multiplier(21, 300) < mov_multiplier(21, 0)


def test_elo_zero_sum_and_season_regression():
    g = pd.DataFrame({
        "game_id": ["a", "b"], "season": [2010, 2011],
        "game_date": pd.to_datetime(["2010-09-10", "2011-09-10"]),
        "home_team": ["X", "X"], "away_team": ["Y", "Y"], "home_score": [30, 10], "away_score": [0, 10],
    })
    cfg = EloConfig(revert=0.5, mean=1500, mov=False, home_advantage=0)
    res = run_elo(g, cfg)
    after_first = 1500 + 20 * 0.5
    assert res.games.elo_home_pre.iloc[1] == pytest.approx(1500 + 0.5 * (after_first - 1500))
    assert sum(res.ratings.values()) == pytest.approx(3000)
    with pytest.raises(ValueError):
        run_elo(g.drop(columns=["season"]))


def test_rolling_form_uses_only_earlier_games(tables):
    # Problem 5 and leakage: the form is a MEAN of earlier games.
    _, long = tables
    form = team_form(long, window=3)
    team = form.team.iloc[0]
    t = form[form.team == team].sort_values("game_date")
    k = 5
    expected = t.points_for.iloc[k - 3 : k].mean()
    assert t.form_points_for.iloc[k] == pytest.approx(expected)
    assert np.isnan(t.form_points_for.iloc[0])
    assert t.form_win.between(0, 1).all() or t.form_win.isna().any()


def test_features_do_not_change_when_the_future_changes(tables):
    games, long = tables
    base = build_features(games, long, window=4)
    cut = games.game_date.quantile(0.6)
    g2, l2 = games.copy(), long.copy()
    future = g2.game_date >= cut
    g2.loc[future, "home_score"] += 50
    g2["margin"] = g2.home_score - g2.away_score
    l2.loc[l2.game_date >= cut, "points_for"] += 50
    changed = build_features(g2, l2, window=4)
    cols = feature_columns(base)
    past = base.game_date < cut
    pd.testing.assert_frame_equal(base.loc[past, cols].reset_index(drop=True),
                                  changed.loc[past, cols].reset_index(drop=True))
    on_cut = base.game_date == cut
    if on_cut.any():  # a game on the cut date also must not see its own result
        pd.testing.assert_frame_equal(base.loc[on_cut, cols].reset_index(drop=True),
                                      changed.loc[on_cut, cols].reset_index(drop=True))


def test_head_to_head_date_filter_on_both_directions():
    # Problem 6: a later meeting in either direction is never counted.
    d = pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01"])
    g = pd.DataFrame({"game_id": ["g1", "g2", "g3"], "game_date": d, "home_team": ["A", "B", "A"],
                      "away_team": ["B", "A", "B"], "margin": [7, 3, -10]})
    h = head_to_head(g).set_index("game_id")
    assert h.loc["g1", "h2h_games"] == 0
    assert h.loc["g2", "h2h_games"] == 1 and h.loc["g2", "h2h_margin"] == -7
    assert h.loc["g3", "h2h_games"] == 2 and h.loc["g3", "h2h_margin"] == pytest.approx((7 - 3) / 2)


def test_feature_columns_exclude_outcomes(feats):
    cols = feature_columns(feats)
    for banned in ("home_score", "away_score", "margin", "home_win", "overtime"):
        assert banned not in cols
    assert "elo_p_home" in cols and "diff_form_points_for" in cols


def test_settings_secrets(monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "very-secret-pw")
    monkeypatch.setenv("GRIDIRON_ELO_K", "25")
    s = Settings.from_env()
    assert s.elo.k == 25 and "very-secret-pw" not in repr(s)
