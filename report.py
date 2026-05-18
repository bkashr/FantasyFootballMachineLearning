"""CLI viewer over the scored fantasy data.

A first pass at the terminal UI step in PLANNING.md. Stays text-only
(no curses / textual) — print tables to stdout, easy to pipe and grep.

Subcommands:
  draft-board   Players ordered by current ADP with their value score
  values        Top N positive-value players
  reaches       Top N negative-value players (most overpriced)
  player NAME   Single-player profile: stats, ADP, recent injuries
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from database import DB_PATH, connect
from scoring import DEFAULT_MODEL_VERSION


def _print_table(rows: list[dict], columns: list[tuple[str, str, int]]):
    """columns: list of (key, header, width)."""
    if not rows:
        print("(no rows)")
        return
    header = "  ".join(f"{h:<{w}}" for _, h, w in columns)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "  ".join(_fmt(r.get(k), w) for k, _, w in columns)
        print(line)


def _fmt(v, width: int) -> str:
    if v is None:
        s = ""
    elif isinstance(v, float):
        s = f"{v:.1f}"
    else:
        s = str(v)
    if len(s) > width:
        s = s[: width - 1] + "…"
    return f"{s:<{width}}"


def cmd_draft_board(args):
    with connect(args.db) as conn:
        rows = conn.execute(
            """
            WITH latest AS (
                SELECT player_id, adp, adp_rank
                FROM adp_snapshots
                WHERE (player_id, captured_at) IN (
                    SELECT player_id, MAX(captured_at) FROM adp_snapshots
                    WHERE draft_format = ? GROUP BY player_id
                )
                  AND draft_format = ?
            )
            SELECT
                latest.adp_rank,
                latest.adp,
                p.full_name,
                p.position,
                p.team,
                s.score
            FROM latest
            JOIN players p USING (player_id)
            LEFT JOIN player_scores s
              ON s.player_id = p.player_id
                 AND s.season = ?
                 AND s.model_version = ?
            ORDER BY latest.adp_rank
            LIMIT ?
            """,
            (args.draft_format, args.draft_format, args.season, args.model_version, args.limit),
        ).fetchall()
    _print_table(
        [dict(r) for r in rows],
        columns=[
            ("adp_rank", "ADP#", 5),
            ("adp", "ADP", 6),
            ("full_name", "Player", 28),
            ("position", "Pos", 4),
            ("team", "Tm", 4),
            ("score", "Score", 8),
        ],
    )


def _ranked(direction: str, args):
    from scoring import top_reaches, top_values

    fn = top_values if direction == "values" else top_reaches
    rows = fn(args.season, args.model_version, args.limit, args.db)
    _print_table(
        [dict(r) for r in rows],
        columns=[
            ("full_name", "Player", 28),
            ("position", "Pos", 4),
            ("team", "Tm", 4),
            ("latest_adp", "ADP", 6),
            ("score", "Score", 8),
        ],
    )


def cmd_values(args):
    _ranked("values", args)


def cmd_reaches(args):
    _ranked("reaches", args)


def cmd_player(args):
    name = args.name.strip().lower()
    with connect(args.db) as conn:
        player = conn.execute(
            """
            SELECT player_id, full_name, position, team, birth_date,
                   underdog_id, gsis_id
            FROM players
            WHERE LOWER(full_name) = ? OR LOWER(full_name) LIKE ?
            ORDER BY (LOWER(full_name) = ?) DESC
            LIMIT 1
            """,
            (name, f"%{name}%", name),
        ).fetchone()
        if not player:
            print(f"No player matches {args.name!r}", file=sys.stderr)
            sys.exit(1)

        print(f"{player['full_name']} ({player['position']}, {player['team'] or '?'})")
        if player["birth_date"]:
            print(f"  DOB: {player['birth_date']}")
        if player["underdog_id"]:
            print(f"  underdog_id={player['underdog_id']}  gsis_id={player['gsis_id']}")

        print("\nSeason stats:")
        season_rows = list(conn.execute(
            """
            SELECT season, games_played, fantasy_points_ppr,
                   pass_yards, pass_tds,
                   rush_yards, rush_tds,
                   rec_yards, rec_tds
            FROM player_season_stats WHERE player_id = ?
            ORDER BY season DESC
            """,
            (player["player_id"],),
        ))
        _print_table(
            [dict(r) for r in season_rows],
            columns=[
                ("season", "Year", 5),
                ("games_played", "G", 3),
                ("fantasy_points_ppr", "PPR", 7),
                ("pass_yards", "PassY", 6),
                ("pass_tds", "PassTD", 6),
                ("rush_yards", "RushY", 6),
                ("rush_tds", "RushTD", 6),
                ("rec_yards", "RecY", 6),
                ("rec_tds", "RecTD", 6),
            ],
        )

        print("\nADP history:")
        adp_rows = list(conn.execute(
            """
            SELECT captured_at, draft_format, adp, adp_rank
            FROM adp_snapshots WHERE player_id = ?
            ORDER BY captured_at DESC LIMIT 10
            """,
            (player["player_id"],),
        ))
        _print_table(
            [dict(r) for r in adp_rows],
            columns=[
                ("captured_at", "Captured", 20),
                ("draft_format", "Format", 12),
                ("adp", "ADP", 6),
                ("adp_rank", "Rank", 5),
            ],
        )

        print("\nRecent injuries:")
        inj_rows = list(conn.execute(
            """
            SELECT season, week, status, body_part, reported_at
            FROM injuries WHERE player_id = ?
            ORDER BY reported_at DESC LIMIT 10
            """,
            (player["player_id"],),
        ))
        _print_table(
            [dict(r) for r in inj_rows],
            columns=[
                ("season", "Year", 5),
                ("week", "Wk", 3),
                ("status", "Status", 26),
                ("body_part", "Body", 14),
                ("reported_at", "Reported", 20),
            ],
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--season", type=int, default=2024)
    p.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    p.add_argument("--draft-format", default="best_ball")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_board = sub.add_parser("draft-board", help="Players ordered by current ADP")
    p_board.add_argument("--limit", type=int, default=50)
    p_board.set_defaults(func=cmd_draft_board)

    p_val = sub.add_parser("values", help="Top N positive-value players")
    p_val.add_argument("--limit", type=int, default=20)
    p_val.set_defaults(func=cmd_values)

    p_rch = sub.add_parser("reaches", help="Top N reaches")
    p_rch.add_argument("--limit", type=int, default=20)
    p_rch.set_defaults(func=cmd_reaches)

    p_pl = sub.add_parser("player", help="Single-player profile")
    p_pl.add_argument("name", help="Player name (case-insensitive substring OK)")
    p_pl.set_defaults(func=cmd_player)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
