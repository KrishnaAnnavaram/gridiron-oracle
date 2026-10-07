"""Optional Neo4j export: teams as nodes, games as PLAYED relationships with pre-game Elo values.

The graph is an export for exploration. The Elo computation happens in ``elo.py``, in date order.
``cypher_script`` writes a parameter-free script for ``cypher-shell``. ``push`` uses the ``neo4j``
driver (extra ``neo4j``) with credentials from ``NEO4J_URI``, ``NEO4J_USER`` and ``NEO4J_PASSWORD``.
"""

from __future__ import annotations

import json

import pandas as pd

from .config import Settings

_GAME_FIELDS = ("game_id", "season", "game_date", "home_team", "away_team", "home_score", "away_score",
                "elo_home_pre", "elo_away_pre", "elo_p_home")


def _records(games: pd.DataFrame) -> list[dict]:
    g = games.sort_values(["game_date", "game_id"])[list(_GAME_FIELDS)].copy()
    g["game_date"] = pd.to_datetime(g.game_date).dt.strftime("%Y-%m-%d")
    return json.loads(g.to_json(orient="records"))


def cypher_script(games: pd.DataFrame) -> str:
    lines = ["CREATE CONSTRAINT team_name IF NOT EXISTS FOR (t:Team) REQUIRE t.name IS UNIQUE;"]
    for r in _records(games):
        props = ", ".join(f"{k}: {json.dumps(r[k])}" for k in _GAME_FIELDS if k not in ("home_team", "away_team"))
        lines.append(
            f"MERGE (h:Team {{name: {json.dumps(r['home_team'])}}}) MERGE (a:Team {{name: {json.dumps(r['away_team'])}}}) "
            f"MERGE (h)-[:PLAYED {{{props}, home: true}}]->(a);"
        )
    return "\n".join(lines) + "\n"


def push(games: pd.DataFrame, settings: Settings) -> int:  # pragma: no cover - needs a Neo4j server
    if not (settings.neo4j_uri and settings.neo4j_user and settings.neo4j_password.get_secret_value()):
        raise RuntimeError("set NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD in the environment")
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError('the Neo4j export needs: pip install -e ".[neo4j]"') from exc
    query = (
        "UNWIND $rows AS r MERGE (h:Team {name: r.home_team}) MERGE (a:Team {name: r.away_team}) "
        "MERGE (h)-[p:PLAYED {game_id: r.game_id}]->(a) SET p += r"
    )
    rows = _records(games)
    auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())
    with GraphDatabase.driver(settings.neo4j_uri, auth=auth) as driver:
        driver.execute_query(query, rows=rows)
    return len(rows)
