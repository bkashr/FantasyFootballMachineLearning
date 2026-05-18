"""Tests for nflverse ingest helpers that don't require network access."""

import numpy as np
import pandas as pd
import pytest

import ingest_nflverse
from database import connect
from ingest_nflverse import (
    _none_if_nan,
    _null_duplicate_source_ids,
    _sum_fumbles,
    ingest_injuries,
    ingest_rosters,
    ingest_season_stats,
    ingest_weekly_stats,
)


def test_none_if_nan_passes_through_real_values():
    assert _none_if_nan("hello") == "hello"
    assert _none_if_nan(0) == 0
    assert _none_if_nan(0.0) == 0.0
    assert _none_if_nan(False) is False


def test_none_if_nan_collapses_nan_and_none():
    assert _none_if_nan(None) is None
    assert _none_if_nan(float("nan")) is None
    assert _none_if_nan(np.nan) is None
    assert _none_if_nan(pd.NA) is None


def test_none_if_nan_coerces_timestamp_to_isoformat():
    ts = pd.Timestamp("2024-03-04")
    assert _none_if_nan(ts) == "2024-03-04"


def test_none_if_nan_unwraps_numpy_scalars():
    assert _none_if_nan(np.int64(7)) == 7
    assert _none_if_nan(np.float64(1.5)) == 1.5
    # And the result is a plain Python scalar, not a numpy one.
    assert type(_none_if_nan(np.int64(7))) is int


def test_sum_fumbles_treats_missing_as_zero():
    row = pd.Series({"sack_fumbles_lost": 1, "rushing_fumbles_lost": np.nan})
    # receiving_fumbles_lost missing entirely
    assert _sum_fumbles(row) == 1


def test_sum_fumbles_adds_three_components():
    row = pd.Series(
        {
            "sack_fumbles_lost": 1,
            "rushing_fumbles_lost": 2,
            "receiving_fumbles_lost": 3,
        }
    )
    assert _sum_fumbles(row) == 6


def test_null_duplicate_source_ids_keeps_first_nulls_rest():
    rows = [
        {"gsis_id": "G1", "pfr_id": "DupePFR"},
        {"gsis_id": "G2", "pfr_id": "DupePFR"},
        {"gsis_id": "G3", "pfr_id": "UniquePFR"},
    ]
    cleaned = _null_duplicate_source_ids(rows, columns=("pfr_id",))
    assert cleaned[0]["pfr_id"] == "DupePFR"
    assert cleaned[1]["pfr_id"] is None
    assert cleaned[2]["pfr_id"] == "UniquePFR"


def test_null_duplicate_source_ids_treats_empty_string_as_null():
    rows = [
        {"gsis_id": "G1", "pfr_id": ""},
        {"gsis_id": "G2", "pfr_id": ""},
    ]
    cleaned = _null_duplicate_source_ids(rows, columns=("pfr_id",))
    # Empty strings are skipped from the dup-tracking entirely
    assert cleaned[0]["pfr_id"] == ""
    assert cleaned[1]["pfr_id"] == ""


def test_ingest_rosters_with_stubbed_nflverse(tmp_db, monkeypatch):
    # Build a small DataFrame mirroring nflverse's seasonal_rosters shape
    fake = pd.DataFrame(
        [
            {
                "season": 2023,
                "player_id": "00-0000001",
                "player_name": "Alice Adams",
                "position": "WR",
                "team": "DAL",
                "birth_date": pd.Timestamp("1998-04-12"),
                "sleeper_id": "s1",
                "pfr_id": "PFRalice",
                "espn_id": "1001",
            },
            {
                "season": 2023,
                "player_id": "00-0000002",
                "player_name": "Bob Baker",
                "position": "RB",
                "team": "NYG",
                "birth_date": pd.Timestamp("1995-09-30"),
                "sleeper_id": "s2",
                "pfr_id": "PFRbob",
                "espn_id": "1002",
            },
        ]
    )
    monkeypatch.setattr(
        ingest_nflverse.nfl, "import_seasonal_rosters", lambda seasons: fake
    )

    n = ingest_rosters([2023], db_path=tmp_db)
    assert n == 2

    with connect(tmp_db) as conn:
        rows = list(
            conn.execute(
                "SELECT full_name, position, team, birth_date, gsis_id, pfr_id FROM players ORDER BY full_name"
            )
        )
    assert rows[0]["full_name"] == "Alice Adams"
    assert rows[0]["birth_date"] == "1998-04-12"
    assert rows[0]["gsis_id"] == "00-0000001"
    assert rows[0]["pfr_id"] == "PFRalice"
    assert rows[1]["team"] == "NYG"


def _seed_minimal_roster(tmp_db, monkeypatch):
    """Insert two known players via the roster ingest path so subsequent
    stat-ingest tests have something to join against."""
    roster_df = pd.DataFrame(
        [
            {"season": 2023, "player_id": "G1", "player_name": "P1", "position": "WR", "team": "DAL"},
            {"season": 2023, "player_id": "G2", "player_name": "P2", "position": "RB", "team": "NYG"},
        ]
    )
    monkeypatch.setattr(ingest_nflverse.nfl, "import_seasonal_rosters", lambda seasons: roster_df)
    ingest_rosters([2023], db_path=tmp_db)


def test_ingest_season_stats_maps_columns_and_skips_unknown_players(tmp_db, monkeypatch):
    _seed_minimal_roster(tmp_db, monkeypatch)

    season_df = pd.DataFrame(
        [
            {
                "player_id": "G1",
                "season": 2023,
                "season_type": "REG",
                "attempts": 0,
                "completions": 0,
                "passing_yards": 0,
                "passing_tds": 0,
                "interceptions": 0,
                "carries": 5,
                "rushing_yards": 30,
                "rushing_tds": 0,
                "targets": 120,
                "receptions": 90,
                "receiving_yards": 1200,
                "receiving_tds": 8,
                "fantasy_points": 200.0,
                "fantasy_points_ppr": 290.0,
                "games": 16,
                "sack_fumbles_lost": 0,
                "rushing_fumbles_lost": 1,
                "receiving_fumbles_lost": 0,
            },
            # Player with a gsis_id NOT in our players table — should be skipped
            {
                "player_id": "G_GHOST",
                "season": 2023,
                "season_type": "REG",
                "fantasy_points_ppr": 50.0,
            },
            # Postseason row — should be filtered out by season_type
            {
                "player_id": "G1",
                "season": 2023,
                "season_type": "POST",
                "fantasy_points_ppr": 30.0,
            },
        ]
    )
    monkeypatch.setattr(ingest_nflverse.nfl, "import_seasonal_data", lambda seasons: season_df)

    n = ingest_season_stats([2023], db_path=tmp_db)
    assert n == 1

    with connect(tmp_db) as conn:
        rows = list(
            conn.execute(
                "SELECT p.full_name, s.rec_yards, s.fantasy_points_ppr, s.fumbles_lost "
                "FROM player_season_stats s JOIN players p USING (player_id)"
            )
        )
    assert len(rows) == 1
    assert rows[0]["full_name"] == "P1"
    assert rows[0]["rec_yards"] == 1200
    assert rows[0]["fantasy_points_ppr"] == 290.0
    assert rows[0]["fumbles_lost"] == 1


def test_ingest_weekly_stats_records_team_and_opponent(tmp_db, monkeypatch):
    _seed_minimal_roster(tmp_db, monkeypatch)

    weekly_df = pd.DataFrame(
        [
            {
                "player_id": "G1",
                "season": 2023,
                "week": 4,
                "season_type": "REG",
                "recent_team": "DAL",
                "opponent_team": "NE",
                "attempts": 0, "completions": 0, "passing_yards": 0, "passing_tds": 0, "interceptions": 0,
                "carries": 1, "rushing_yards": 5, "rushing_tds": 0,
                "targets": 9, "receptions": 7, "receiving_yards": 110, "receiving_tds": 1,
                "fantasy_points": 19.5, "fantasy_points_ppr": 26.5,
                "sack_fumbles_lost": 0, "rushing_fumbles_lost": 0, "receiving_fumbles_lost": 0,
            }
        ]
    )
    monkeypatch.setattr(ingest_nflverse.nfl, "import_weekly_data", lambda seasons: weekly_df)

    n = ingest_weekly_stats([2023], db_path=tmp_db)
    assert n == 1

    with connect(tmp_db) as conn:
        row = conn.execute(
            "SELECT team, opponent, rec_yards FROM player_weekly_stats WHERE season=2023 AND week=4"
        ).fetchone()
    assert row["team"] == "DAL"
    assert row["opponent"] == "NE"
    assert row["rec_yards"] == 110


def test_ingest_injuries_is_idempotent(tmp_db, monkeypatch):
    _seed_minimal_roster(tmp_db, monkeypatch)

    inj_df = pd.DataFrame(
        [
            {
                "season": 2023, "week": 1, "gsis_id": "G1",
                "report_status": "Questionable", "report_primary_injury": "Knee",
                "report_secondary_injury": None,
                "practice_status": None, "practice_primary_injury": None,
                "date_modified": pd.Timestamp("2023-09-08T18:00:00"),
            },
            {
                "season": 2023, "week": 2, "gsis_id": "G2",
                "report_status": "Out", "report_primary_injury": "Hamstring",
                "report_secondary_injury": None,
                "practice_status": None, "practice_primary_injury": None,
                "date_modified": pd.Timestamp("2023-09-15T18:00:00"),
            },
        ]
    )
    monkeypatch.setattr(ingest_nflverse.nfl, "import_injuries", lambda seasons: inj_df)

    ingest_injuries([2023], db_path=tmp_db)
    with connect(tmp_db) as conn:
        first = conn.execute("SELECT COUNT(*) FROM injuries").fetchone()[0]

    # Re-run with the same fixture — counts should match (no duplication)
    ingest_injuries([2023], db_path=tmp_db)
    with connect(tmp_db) as conn:
        second = conn.execute("SELECT COUNT(*) FROM injuries").fetchone()[0]
    assert first == second == 2


def test_ingest_rosters_handles_duplicate_pfr_ids(tmp_db, monkeypatch):
    # Two players share a pfr_id (a real nflverse quirk). Both should
    # still be inserted; the second loses its pfr_id to satisfy UNIQUE.
    fake = pd.DataFrame(
        [
            {"season": 2023, "player_id": "G1", "player_name": "P1", "position": "WR", "team": "DAL", "pfr_id": "Dupe"},
            {"season": 2023, "player_id": "G2", "player_name": "P2", "position": "WR", "team": "NYG", "pfr_id": "Dupe"},
        ]
    )
    monkeypatch.setattr(ingest_nflverse.nfl, "import_seasonal_rosters", lambda seasons: fake)

    ingest_rosters([2023], db_path=tmp_db)

    with connect(tmp_db) as conn:
        rows = list(conn.execute("SELECT full_name, pfr_id FROM players ORDER BY full_name"))
    assert len(rows) == 2
    pfrs = [r["pfr_id"] for r in rows]
    assert pfrs.count("Dupe") == 1
    assert pfrs.count(None) == 1
