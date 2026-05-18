"""Pull rosters + season + weekly stats from nflverse via nfl_data_py and
upsert them into our SQLite database.

Idempotent: safe to re-run for the same seasons. Re-runs update changed
values (e.g. corrected stats, team changes) but won't create duplicates."""

from __future__ import annotations

import argparse
import logging
from typing import Iterable

import nfl_data_py as nfl
import pandas as pd

from database import DB_PATH, connect, init_db, upsert_many

log = logging.getLogger(__name__)

# nflverse weekly/season column -> our schema column
STAT_COLUMN_MAP = {
    "attempts": "pass_attempts",
    "completions": "pass_completions",
    "passing_yards": "pass_yards",
    "passing_tds": "pass_tds",
    "interceptions": "interceptions",
    "carries": "carries",
    "rushing_yards": "rush_yards",
    "rushing_tds": "rush_tds",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "rec_yards",
    "receiving_tds": "rec_tds",
    "fantasy_points": "fantasy_points",
    "fantasy_points_ppr": "fantasy_points_ppr",
}


def _none_if_nan(v):
    """SQLite doesn't grok numpy NaN, pandas Timestamps, or numpy scalars —
    coerce to None / native python types."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if hasattr(v, "item"):
        return v.item()
    return v


def _row_to_dict(row: pd.Series, columns: Iterable[str]) -> dict:
    return {c: _none_if_nan(row[c]) if c in row else None for c in columns}


def _sum_fumbles(row: pd.Series) -> int:
    """Combine the three fumble-lost columns nflverse splits across passing /
    rushing / receiving into one total."""
    total = 0
    for c in ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"):
        v = _none_if_nan(row.get(c))
        if v is not None:
            total += int(v)
    return total


def _gsis_to_player_id(conn) -> dict[str, int]:
    """Build a {gsis_id: player_id} lookup from the players table."""
    return {
        r["gsis_id"]: r["player_id"]
        for r in conn.execute(
            "SELECT player_id, gsis_id FROM players WHERE gsis_id IS NOT NULL"
        )
    }


def ingest_rosters(seasons: list[int], db_path: str = DB_PATH) -> int:
    """Upsert players from nflverse seasonal rosters. Returns row count."""
    log.info("Pulling rosters for seasons=%s", seasons)
    df = nfl.import_seasonal_rosters(seasons)
    # latest season wins for fields that change year to year (team, position)
    df = df.sort_values("season").drop_duplicates("player_id", keep="last")

    rows = []
    for _, r in df.iterrows():
        gsis = r.get("player_id")
        if not isinstance(gsis, str) or not gsis:
            continue
        rows.append(
            {
                "full_name": _none_if_nan(r.get("player_name")),
                "position": _none_if_nan(r.get("position")),
                "team": _none_if_nan(r.get("team")),
                "birth_date": _none_if_nan(r.get("birth_date")),
                "rookie_year": _none_if_nan(r.get("rookie_year")),
                "gsis_id": gsis,
                "sleeper_id": _none_if_nan(r.get("sleeper_id")),
                "pfr_id": _none_if_nan(r.get("pfr_id")),
                "espn_id": _none_if_nan(r.get("espn_id")),
            }
        )

    rows = _null_duplicate_source_ids(
        rows, columns=("sleeper_id", "pfr_id", "espn_id")
    )

    with connect(db_path) as conn:
        upsert_many(conn, "players", rows, conflict_cols=["gsis_id"])
    log.info("Upserted %d players", len(rows))
    return len(rows)


def _null_duplicate_source_ids(
    rows: list[dict], columns: tuple[str, ...]
) -> list[dict]:
    """Null out duplicate non-null source IDs so we don't trip the UNIQUE
    constraint. Real-world data has occasional collisions (e.g. nflverse
    sometimes lists two players sharing a pfr_id)."""
    for col in columns:
        seen: dict[object, int] = {}
        for i, row in enumerate(rows):
            v = row.get(col)
            if v is None or v == "":
                continue
            if v in seen:
                log.warning(
                    "Duplicate %s=%r on players %r and %r — nulling later occurrence",
                    col,
                    v,
                    rows[seen[v]].get("gsis_id"),
                    row.get("gsis_id"),
                )
                row[col] = None
            else:
                seen[v] = i
    return rows


def ingest_season_stats(seasons: list[int], db_path: str = DB_PATH) -> int:
    log.info("Pulling seasonal stats for seasons=%s", seasons)
    df = nfl.import_seasonal_data(seasons)
    df = df[df["season_type"] == "REG"]

    with connect(db_path) as conn:
        lookup = _gsis_to_player_id(conn)

        rows = []
        missing = 0
        for _, r in df.iterrows():
            pid = lookup.get(r["player_id"])
            if pid is None:
                missing += 1
                continue
            row = {"player_id": pid, "season": int(r["season"])}
            for src, dst in STAT_COLUMN_MAP.items():
                row[dst] = _none_if_nan(r.get(src))
            row["games_played"] = _none_if_nan(r.get("games"))
            row["fumbles_lost"] = _sum_fumbles(r)
            row["team"] = None  # seasonal aggregate doesn't carry a team
            rows.append(row)

        upsert_many(
            conn,
            "player_season_stats",
            rows,
            conflict_cols=["player_id", "season"],
        )

    if missing:
        log.warning("Skipped %d season rows with no players-table match", missing)
    log.info("Upserted %d season-stat rows", len(rows))
    return len(rows)


def ingest_weekly_stats(seasons: list[int], db_path: str = DB_PATH) -> int:
    log.info("Pulling weekly stats for seasons=%s", seasons)
    df = nfl.import_weekly_data(seasons)
    df = df[df["season_type"] == "REG"]

    with connect(db_path) as conn:
        lookup = _gsis_to_player_id(conn)

        rows = []
        missing = 0
        for _, r in df.iterrows():
            pid = lookup.get(r["player_id"])
            if pid is None:
                missing += 1
                continue
            row = {
                "player_id": pid,
                "season": int(r["season"]),
                "week": int(r["week"]),
                "team": _none_if_nan(r.get("recent_team")),
                "opponent": _none_if_nan(r.get("opponent_team")),
            }
            for src, dst in STAT_COLUMN_MAP.items():
                row[dst] = _none_if_nan(r.get(src))
            row["fumbles_lost"] = _sum_fumbles(r)
            rows.append(row)

        upsert_many(
            conn,
            "player_weekly_stats",
            rows,
            conflict_cols=["player_id", "season", "week"],
        )

    if missing:
        log.warning("Skipped %d weekly rows with no players-table match", missing)
    log.info("Upserted %d weekly-stat rows", len(rows))
    return len(rows)


def ingest_injuries(seasons: list[int], db_path: str = DB_PATH) -> int:
    """Append injury reports. Append-only — each report is a status
    snapshot at a point in time. Matches players via gsis_id."""
    log.info("Pulling injuries for seasons=%s", seasons)
    df = nfl.import_injuries(seasons)

    with connect(db_path) as conn:
        lookup = _gsis_to_player_id(conn)

        rows = []
        missing = 0
        for _, r in df.iterrows():
            pid = lookup.get(r.get("gsis_id"))
            if pid is None:
                missing += 1
                continue
            rows.append(
                {
                    "player_id": pid,
                    "season": _none_if_nan(r.get("season")),
                    "week": _none_if_nan(r.get("week")),
                    "status": _none_if_nan(
                        r.get("report_status") or r.get("practice_status")
                    ),
                    "body_part": _none_if_nan(
                        r.get("report_primary_injury")
                        or r.get("practice_primary_injury")
                    ),
                    "notes": _none_if_nan(r.get("report_secondary_injury")),
                    "reported_at": _none_if_nan(r.get("date_modified")),
                }
            )

        # Delete-and-replace per season so re-runs don't duplicate.
        conn.execute(
            f"DELETE FROM injuries WHERE season IN ({','.join(['?'] * len(seasons))})",
            seasons,
        )

        if rows:
            cols = ["player_id", "season", "week", "status", "body_part", "notes", "reported_at"]
            conn.executemany(
                f"INSERT INTO injuries ({','.join(cols)}) VALUES ({','.join(['?'] * len(cols))})",
                [[r[c] for c in cols] for r in rows],
            )

    if missing:
        log.warning("Skipped %d injury rows with no players-table match", missing)
    log.info("Appended %d injury rows", len(rows))
    return len(rows)


def ingest_all(seasons: list[int], db_path: str = DB_PATH) -> dict:
    """Run rosters -> season -> weekly -> injuries in order. Rosters
    first so the crosswalk is populated before everything else looks
    up player_id."""
    init_db(db_path)
    return {
        "players": ingest_rosters(seasons, db_path),
        "season": ingest_season_stats(seasons, db_path),
        "weekly": ingest_weekly_stats(seasons, db_path),
        "injuries": ingest_injuries(seasons, db_path),
    }


def _parse_seasons(s: str) -> list[int]:
    """Accept '2022,2023' or '2020-2023'."""
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seasons",
        default="2023",
        help="comma list or range, e.g. '2022,2023' or '2020-2024'",
    )
    p.add_argument("--db", default=DB_PATH)
    p.add_argument(
        "--step",
        choices=["all", "rosters", "season", "weekly", "injuries"],
        default="all",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    seasons = _parse_seasons(args.seasons)
    init_db(args.db)
    if args.step in ("all", "rosters"):
        ingest_rosters(seasons, args.db)
    if args.step in ("all", "season"):
        ingest_season_stats(seasons, args.db)
    if args.step in ("all", "weekly"):
        ingest_weekly_stats(seasons, args.db)
    if args.step in ("all", "injuries"):
        ingest_injuries(seasons, args.db)


if __name__ == "__main__":
    main()
