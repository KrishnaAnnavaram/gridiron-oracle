# data/

Git ignores everything in this folder except this file. Do not commit play-by-play files or built tables.

## Real data: nflverse play-by-play

- Source: the open `nflverse` project, <https://github.com/nflverse/nflverse-data>
  (Python access with `nfl_data_py`, <https://github.com/nflverse/nfl_data_py>).
- Licence: nflverse data is published under CC-BY 4.0. Check the current terms of the repository and
  credit the project.
- Download the regular seasons 2009 to 2018 (about 450,000 plays):
  `pip install -e ".[nflverse]"` and then `gridiron-oracle fetch --first 2009 --last 2018 --out data/pbp.csv`.
- The Kaggle set "Detailed NFL Play-by-Play Data 2009-2018" (nflscrapR) has the same column names.

Columns that `gridiron-oracle build` needs:

| Column | Meaning |
|---|---|
| `season`, `game_id`, `game_date` | Game keys and date |
| `home_team`, `away_team`, `posteam`, `defteam` | Teams of the game and of the play |
| `qtr` | Quarter (5 = overtime) |
| `play_type` | `pass`, `run`, `punt`, `field_goal`, and others |
| `yards_gained` | Yards of the play |
| `interception`, `fumble_lost` | Turnovers (0 or 1) |
| `third_down_converted`, `third_down_failed` | Third-down result (0 or 1) |
| `total_home_score`, `total_away_score` | Score after the play |
| `week`, `penalty_yards` (optional) | Week number and penalty yards |

## Built files

| File | Command | Contents |
|---|---|---|
| `data/pbp.csv` | `synth` or `fetch` | Play-by-play |
| `data/games.csv` | `build` | One row per game: scores, target, Elo, form features, differences |
| `data/team_games.csv` | `build` | One row per team per game (input of the GRU) |
| `outputs/graph.cypher` | `export-neo4j` | Cypher script for a Neo4j graph |

## Synthetic data

`gridiron-oracle synth --out data/pbp.csv` writes a synthetic league: 32 teams (`T00` to `T31`),
16 weeks for each season, about 340,000 plays for 10 seasons. Team strengths are hidden and change
between seasons. No real team, player or game is in this file.
