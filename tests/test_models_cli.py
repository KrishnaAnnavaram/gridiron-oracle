import json

import numpy as np
import pandas as pd
import pytest

from gridiron_oracle.cli import main
from gridiron_oracle.graph import cypher_script
from gridiron_oracle.elo import run_elo
from gridiron_oracle.models import (
    LABELS,
    EloOnly,
    HomeTeam,
    accuracy,
    bootstrap_diff,
    brier,
    calibration_table,
    default_models,
    ece,
    log_loss,
    logistic,
    walk_forward,
)


def test_metrics():
    y = np.array([1, 0, 1, 1])
    p = np.array([0.9, 0.2, 0.6, 0.4])
    assert accuracy(y, p) == 0.75
    assert brier(y, np.full(4, 0.5)) == 0.25
    assert log_loss(y, np.full(4, 0.5)) == pytest.approx(np.log(2))
    assert log_loss(np.array([1]), np.array([0.0])) < 20  # clipped
    t = calibration_table(y, p, bins=2)
    assert t.n.sum() == 4 and 0 <= ece(y, p, bins=2) <= 1


def test_labels_are_explicit():
    # Problem 7: the binary target has one fixed mapping.
    assert LABELS == {1: "home win", 0: "away win"}


def test_pipeline_fits_scaler_and_selection_on_train_only(feats):
    # Problem 2: scaling and feature selection live inside the pipeline.
    from gridiron_oracle.features import feature_columns

    cols = feature_columns(feats)
    played = feats[feats.home_win.notna()]
    train = played[played.season <= 2014]
    pipe = logistic(k=10, C=1.0).fit(train[cols], train.home_win)
    kept = pipe.named_steps["drop_constant"].transform(
        pipe.named_steps["impute"].transform(train[cols]))
    assert np.allclose(pipe.named_steps["scale"].mean_, kept.mean(axis=0))
    assert pipe.named_steps["select"].get_support().sum() == 10


def test_walk_forward_order_and_selection(feats):
    # Problem 3: hyperparameters are chosen on the season before the test season, never on the test season.
    wf = walk_forward(feats, [2015, 2016])
    names = {r.model for r in wf.rows}
    assert names == {"home-team", "elo", "logistic", "boosting"}
    for r in wf.rows:
        assert 0 <= r.accuracy <= 1 and r.log_loss > 0
        if r.model == "logistic":
            assert set(r.params) == {"k", "C"}
    pooled = wf.pooled().set_index("model")
    assert pooled.loc["elo", "log_loss"] < pooled.loc["home-team", "log_loss"]
    assert pooled.loc["logistic", "log_loss"] < pooled.loc["home-team", "log_loss"]
    d, lo, hi = bootstrap_diff(wf.predictions, "logistic", "home-team", n_boot=200)
    assert lo <= d <= hi and d < 0


def test_walk_forward_needs_history(feats):
    with pytest.raises(ValueError):
        walk_forward(feats, [2013])


def test_ties_are_left_out_and_counted(feats):
    wf = walk_forward(feats, [2016], lambda: [HomeTeam(), EloOnly()])
    ties = int(feats[(feats.season == 2016)].home_win.isna().sum())
    assert all(r.ties_left_out == ties for r in wf.rows)
    assert wf.predictions.home_win.notna().all()


def test_home_team_baseline(feats):
    m = HomeTeam().fit(feats[feats.home_win.notna()])
    assert 0.5 < m.rate < 0.75
    assert len(default_models()) == 4


def test_gru_on_real_sequences(tables, feats):
    # Problem 8: the sequence model sees each team's previous games, not one time step.
    pytest.importorskip("torch")
    from gridiron_oracle.sequence import GRUModel, build_sequences

    games, long = tables
    seqs = build_sequences(long, n_last=4)
    first = long.sort_values("game_date").iloc[0]
    assert seqs[(first.game_id, first.team)][:, -1].sum() == 0  # no history before the first game
    played = feats[feats.home_win.notna()]
    m = GRUModel(long, n_last=4, epochs=2, seed=0).fit(played[played.season <= 2014], played[played.season == 2015])
    p = m.predict_proba(played[played.season == 2016])
    assert p.shape == ((played.season == 2016).sum(),) and ((p > 0) & (p < 1)).all()


def test_cypher_export(tables):
    games, _ = tables
    script = cypher_script(run_elo(games).games.head(3))
    assert script.count("MERGE (h)-[:PLAYED") == 3 and "elo_p_home" in script
    assert "password" not in script.lower()


def test_cli_end_to_end(tmp_path, capsys):
    pbp = tmp_path / "pbp.csv"
    assert main(["synth", "--out", str(pbp), "--first", "2012", "--last", "2015", "--seed", "2"]) == 0
    games, long = tmp_path / "games.csv", tmp_path / "long.csv"
    assert main(["build", "--pbp", str(pbp), "--out", str(games), "--long-out", str(long), "--window", "4"]) == 0
    assert main(["elo", "--games", str(games), "--top", "3"]) == 0
    capsys.readouterr()
    assert main(["evaluate", "--games", str(games), "--test-seasons", "2015", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {r["model"] for r in data["pooled"]} == {"home-team", "elo", "logistic", "boosting"}
    assert main(["evaluate", "--games", str(games), "--n-test", "1"]) == 0
    assert "calibration" in capsys.readouterr().out
    out = tmp_path / "g.cypher"
    assert main(["export-neo4j", "--games", str(games), "--out", str(out)]) == 0
    assert out.read_text().startswith("CREATE CONSTRAINT")
