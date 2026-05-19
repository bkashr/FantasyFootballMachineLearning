"""Phase 5: portfolio tracker.

Tracks the user's active best-ball drafts and the picks they've made.
Feeds exposure-aware scoring ('a player's value to me drops as my
exposure rises') and stack analysis ('already roster the QB? bump
the WRs/TEs on his team').

Manual entry via CSV — simplest format that covers what we need:

    draft_name,season,my_slot,round,pick_overall,player_name,position
    "BBM6 #12",2026,7,1,7,"Bijan Robinson","RB"
    "BBM6 #12",2026,7,2,18,"Drake London","WR"
    ...

draft_name is the key — all rows with the same name become one draft.
season + my_slot are taken from the first row per name. position is
optional but helps disambiguate name collisions in the crosswalk.

Once we know Underdog's actual draft-export format, we can add a
direct ingest that reads it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
from pathlib import Path

from database import DB_PATH, connect, init_db
from ingest_underdog import _build_lookups, _resolve_player

log = logging.getLogger(__name__)


def import_drafts_from_csv(path: str | Path, db_path: str = DB_PATH) -> dict:
    """Parse the manual-entry CSV and write drafts + picks. Each unique
    draft_name becomes one row in my_drafts; each data row becomes a
    pick. Idempotent — re-running on the same file is a no-op via
    INSERT OR IGNORE on UNIQUE (draft_id, pick_overall)."""
    init_db(db_path)
    required = {"draft_name", "season", "my_slot", "round", "pick_overall", "player_name"}
    rows: list[dict] = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Portfolio CSV missing columns: {sorted(missing)}. "
                f"Got: {reader.fieldnames}"
            )
        rows = [dict(r) for r in reader]

    with connect(db_path) as conn:
        by_uid, by_name_pos, by_name = _build_lookups(conn)

        # Group rows by draft_name to create my_drafts entries
        drafts_by_name: dict[str, dict] = {}
        for r in rows:
            name = (r.get("draft_name") or "").strip()
            if not name:
                continue
            if name not in drafts_by_name:
                drafts_by_name[name] = {
                    "draft_name": name,
                    "season": int(r["season"]),
                    "my_slot": int(r["my_slot"]) if r.get("my_slot") else None,
                    "draft_format": (r.get("draft_format") or "best_ball").strip(),
                    "teams": int(r["teams"]) if r.get("teams") else 12,
                    "rounds": int(r["rounds"]) if r.get("rounds") else 18,
                    "drafted_at": (r.get("drafted_at") or "").strip() or None,
                }

        # Upsert drafts; build name -> draft_id lookup
        draft_ids: dict[str, int] = {}
        for name, d in drafts_by_name.items():
            conn.execute(
                """
                INSERT INTO my_drafts (
                    draft_name, season, my_slot, draft_format, teams, rounds, drafted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (
                    d["draft_name"], d["season"], d["my_slot"], d["draft_format"],
                    d["teams"], d["rounds"], d["drafted_at"],
                ),
            )
            row = conn.execute(
                "SELECT draft_id FROM my_drafts WHERE draft_name = ? AND season = ?",
                (d["draft_name"], d["season"]),
            ).fetchone()
            draft_ids[name] = row["draft_id"]

        # Now the picks
        picks_written = 0
        unmatched: list[str] = []
        for r in rows:
            name = (r.get("draft_name") or "").strip()
            if not name or name not in draft_ids:
                continue
            player_name = (r.get("player_name") or "").strip()
            position = (r.get("position") or "").strip() or None
            rec = {
                "underdog_id": None,
                "full_name": player_name,
                "position": position,
            }
            pid = _resolve_player(rec, by_uid, by_name_pos, by_name)
            if pid is None:
                unmatched.append(player_name)
                continue
            try:
                round_no = int(r["round"])
                pick_overall = int(r["pick_overall"])
            except (TypeError, ValueError):
                continue
            try:
                conn.execute(
                    """
                    INSERT INTO my_picks (draft_id, player_id, round, pick_overall)
                    VALUES (?, ?, ?, ?)
                    """,
                    (draft_ids[name], pid, round_no, pick_overall),
                )
                picks_written += 1
            except Exception:
                # Likely UNIQUE violation on (draft_id, pick_overall) — already
                # imported. Idempotent re-run.
                pass

    if unmatched:
        log.warning(
            "%d picks had no players-table match: %s%s",
            len(unmatched),
            ", ".join(unmatched[:5]),
            " ..." if len(unmatched) > 5 else "",
        )
    log.info(
        "Imported %d drafts and %d picks from %s",
        len(drafts_by_name),
        picks_written,
        path,
    )
    return {
        "drafts_imported": len(drafts_by_name),
        "picks_imported": picks_written,
        "unmatched": len(unmatched),
    }


def player_exposure(season: int, db_path: str = DB_PATH) -> list[dict]:
    """Return {player, position, team, n_rosters, pct_of_drafts}, ordered
    by exposure descending."""
    with connect(db_path) as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM my_drafts WHERE season = ?", (season,)
        ).fetchone()["n"]
        if total == 0:
            return []
        rows = list(conn.execute(
            """
            SELECT p.player_id, p.full_name, p.position, p.team,
                   COUNT(DISTINCT m.draft_id) AS n_rosters
            FROM my_picks m
            JOIN my_drafts d USING (draft_id)
            JOIN players p USING (player_id)
            WHERE d.season = ?
            GROUP BY p.player_id
            ORDER BY n_rosters DESC, p.full_name
            """,
            (season,),
        ))
        return [
            {
                "player_id": r["player_id"],
                "full_name": r["full_name"],
                "position": r["position"],
                "team": r["team"],
                "n_rosters": r["n_rosters"],
                "pct_of_drafts": r["n_rosters"] / total * 100.0,
            }
            for r in rows
        ]


def team_exposure(season: int, db_path: str = DB_PATH) -> list[dict]:
    """Exposure by NFL team — how concentrated am I on each offense?"""
    with connect(db_path) as conn:
        rows = list(conn.execute(
            """
            SELECT p.team,
                   COUNT(*) AS picks_total,
                   COUNT(DISTINCT m.draft_id) AS drafts_with_team,
                   AVG(picks_per_draft) AS avg_picks_per_draft_when_present
            FROM (
                SELECT m.draft_id, p.team, COUNT(*) AS picks_per_draft
                FROM my_picks m
                JOIN my_drafts d USING (draft_id)
                JOIN players p USING (player_id)
                WHERE d.season = ? AND p.team IS NOT NULL
                GROUP BY m.draft_id, p.team
            ) per_team_per_draft
            JOIN my_picks m USING (draft_id)
            JOIN players p USING (player_id)
            WHERE per_team_per_draft.team = p.team
            GROUP BY p.team
            ORDER BY drafts_with_team DESC, picks_total DESC
            """,
            (season,),
        ))
        return [dict(r) for r in rows]


def draft_count(season: int, db_path: str = DB_PATH) -> int:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT COUNT(*) AS n FROM my_drafts WHERE season = ?", (season,)
        ).fetchone()["n"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from-csv",
        help="Path to the portfolio CSV (see module docstring for format).",
    )
    p.add_argument("--season", type=int, default=None)
    p.add_argument(
        "--report",
        choices=["players", "teams"],
        help="Print an exposure report instead of importing.",
    )
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.from_csv:
        result = import_drafts_from_csv(args.from_csv, db_path=args.db)
        print(
            f"Imported {result['drafts_imported']} drafts, "
            f"{result['picks_imported']} picks "
            f"({result['unmatched']} unmatched names)"
        )

    if args.report:
        season = args.season or 2026
        n_drafts = draft_count(season, db_path=args.db)
        print(f"\nSeason {season}: {n_drafts} drafts")
        if args.report == "players":
            print("\nPlayer exposure (top by # rosters):")
            print(f"  {'Player':28s} {'Pos':3s} {'Tm':3s}   N    %")
            for r in player_exposure(season, db_path=args.db)[: args.limit]:
                print(
                    f"  {r['full_name'][:28]:28s} "
                    f"{r['position'] or '':3s} {r['team'] or '':3s}  "
                    f"{r['n_rosters']:3d}  {r['pct_of_drafts']:5.1f}%"
                )
        else:
            print("\nTeam exposure:")
            print(f"  {'Tm':3s}   drafts_with   total_picks")
            for r in team_exposure(season, db_path=args.db)[: args.limit]:
                print(
                    f"  {r['team'] or '?':3s}   {r['drafts_with_team']:11d}   {r['picks_total']:11d}"
                )


if __name__ == "__main__":
    main()
