"""The ``gridiron-oracle`` command line."""

from __future__ import annotations

import os

# joblib (used by scikit-learn) prints a long warning on some Windows hosts when it counts CPU cores.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import Settings
from .elo import run_elo
from .features import build_features
from .games import game_table, team_game_table
from .graph import cypher_script, push
from .models import LABELS, bootstrap_diff, calibration_table, default_models, walk_forward
from .synthetic import make_pbp


def _read_games(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def cmd_synth(args) -> int:
    pbp = make_pbp(range(args.first, args.last + 1), seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pbp.to_csv(out, index=False)
    print(f"wrote {out}: {len(pbp)} plays, {pbp.game_id.nunique()} games, seasons {args.first}-{args.last}")
    return 0


def cmd_fetch(args) -> int:  # pragma: no cover - network
    try:
        import nfl_data_py as nfl
    except ImportError:
        print('this command needs: pip install -e ".[nflverse]"', file=sys.stderr)
        return 1
    pbp = nfl.import_pbp_data(list(range(args.first, args.last + 1)))
    pbp = pbp[pbp.season_type == "REG"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pbp.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(pbp)} plays")
    return 0


def cmd_build(args) -> int:
    s = Settings.from_env()
    pbp = pd.read_csv(args.pbp, low_memory=False)
    games, long = game_table(pbp), team_game_table(pbp)
    feats = build_features(games, long, args.window or s.window, s.elo)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out, index=False)
    if args.long_out:
        long.to_csv(args.long_out, index=False)
    print(f"wrote {out}: {len(feats)} games, {feats.shape[1]} columns, {int(feats.home_win.isna().sum())} ties")
    return 0


def cmd_elo(args) -> int:
    s = Settings.from_env()
    res = run_elo(_read_games(args.games), s.elo)
    table = sorted(res.ratings.items(), key=lambda kv: -kv[1])[: args.top]
    for rank, (team, r) in enumerate(table, start=1):
        print(f"{rank:2d}. {team:6s} {r:7.1f}")
    return 0


def cmd_evaluate(args) -> int:
    s = Settings.from_env()
    df = _read_games(args.games)
    seasons = sorted(int(x) for x in df.season.unique())
    test = [int(x) for x in args.test_seasons.split(",")] if args.test_seasons else seasons[-args.n_test:]
    factory = default_models
    if args.sequence:
        from .sequence import GRUModel

        long = pd.read_csv(args.long)
        long["game_date"] = pd.to_datetime(long.game_date)

        def factory():
            return default_models(s.seed) + [GRUModel(long, seed=s.seed)]
    wf = walk_forward(df, test, factory)
    pooled = wf.pooled()
    if args.json:
        print(json.dumps({"seasons": [asdict(r) for r in wf.rows], "pooled": pooled.to_dict(orient="records")}, indent=2))
        return 0
    print(f"target: home_win (1 = {LABELS[1]}, 0 = {LABELS[0]}); test seasons {test}; "
          f"ties left out: {sum(r.ties_left_out for r in wf.rows if r.model == 'home-team')}")
    print(f"{'season':>6s} {'model':10s} {'n':>4s} {'accuracy':>8s} {'log-loss':>8s} {'Brier':>6s}  params")
    for r in wf.rows:
        print(f"{r.season:6d} {r.model:10s} {r.n:4d} {r.accuracy:8.3f} {r.log_loss:8.4f} {r.brier:6.4f}  {r.params or ''}")
    print("\npooled over the test seasons")
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for name in [m for m in pooled.model if m not in ("home-team", "elo")]:
        d, lo, hi = bootstrap_diff(wf.predictions, name, "elo")
        print(f"log-loss({name}) - log-loss(elo): {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    best = pooled.sort_values("log_loss").model.iloc[0]
    g = wf.predictions[wf.predictions.model == best]
    print(f"\ncalibration of {best}:")
    print(calibration_table(g.home_win.to_numpy(), g.p_home.to_numpy()).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def cmd_export_neo4j(args) -> int:
    s = Settings.from_env()
    games = run_elo(_read_games(args.games), s.elo).games
    if args.push:  # pragma: no cover - needs a server
        print(f"pushed {push(games, s)} games to {s.neo4j_uri}")
        return 0
    Path(args.out).write_text(cypher_script(games), encoding="utf-8")
    print(f"wrote {args.out} ({len(games)} games). Run it with cypher-shell -f {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gridiron-oracle", description="Leak-free NFL home-win probabilities")
    sub = p.add_subparsers(dest="command", required=True)

    for name, func, helptext in (("synth", cmd_synth, "write a synthetic play-by-play CSV"),
                                 ("fetch", cmd_fetch, "download nflverse play-by-play (extra nflverse)")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--out", default="data/pbp.csv")
        sp.add_argument("--first", type=int, default=2009)
        sp.add_argument("--last", type=int, default=2018)
        sp.add_argument("--seed", type=int, default=7)
        sp.set_defaults(func=func)

    sp = sub.add_parser("build", help="play-by-play -> game table with point-in-time features and Elo")
    sp.add_argument("--pbp", default="data/pbp.csv")
    sp.add_argument("--out", default="data/games.csv")
    sp.add_argument("--long-out", default="data/team_games.csv", help="also write the team-game table")
    sp.add_argument("--window", type=int)
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("elo", help="show the Elo table after the last game")
    sp.add_argument("--games", default="data/games.csv")
    sp.add_argument("--top", type=int, default=10)
    sp.set_defaults(func=cmd_elo)

    sp = sub.add_parser("evaluate", help="walk-forward evaluation by season")
    sp.add_argument("--games", default="data/games.csv")
    sp.add_argument("--test-seasons", help="comma list (default: the last --n-test seasons)")
    sp.add_argument("--n-test", type=int, default=4)
    sp.add_argument("--sequence", action="store_true", help="also train the GRU (extra sequence)")
    sp.add_argument("--long", default="data/team_games.csv", help="team-game table for the GRU")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("export-neo4j", help="write a Cypher script, or push to Neo4j with --push")
    sp.add_argument("--games", default="data/games.csv")
    sp.add_argument("--out", default="outputs/graph.cypher")
    sp.add_argument("--push", action="store_true")
    sp.set_defaults(func=cmd_export_neo4j)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
