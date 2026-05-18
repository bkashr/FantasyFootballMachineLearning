import sqlite3

import pytest

from database import connect, init_db, upsert, upsert_many


def test_init_creates_expected_tables(tmp_db):
    with connect(tmp_db) as conn:
        names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    expected = {
        "players",
        "adp_snapshots",
        "player_season_stats",
        "player_weekly_stats",
        "player_projections",
        "team_season_stats",
        "injuries",
        "player_scores",
    }
    assert expected.issubset(names)


def test_foreign_keys_enforced(tmp_db):
    # Inserting a stat row for a nonexistent player_id should fail
    # because connect() enables foreign_keys.
    with pytest.raises(sqlite3.IntegrityError):
        with connect(tmp_db) as conn:
            conn.execute(
                "INSERT INTO player_season_stats (player_id, season) VALUES (?, ?)",
                (9999, 2023),
            )


def test_upsert_inserts_then_updates(tmp_db):
    with connect(tmp_db) as conn:
        upsert(
            conn,
            "players",
            {"full_name": "Jane Doe", "position": "WR", "gsis_id": "G1"},
            conflict_cols=["gsis_id"],
        )
        upsert(
            conn,
            "players",
            {"full_name": "Jane D.", "position": "WR", "gsis_id": "G1", "team": "DAL"},
            conflict_cols=["gsis_id"],
        )
        rows = list(conn.execute("SELECT full_name, team FROM players WHERE gsis_id='G1'"))
    assert len(rows) == 1
    assert rows[0]["full_name"] == "Jane D."
    assert rows[0]["team"] == "DAL"


def test_upsert_many_round_trips(tmp_db):
    rows = [
        {"full_name": f"P{i}", "position": "WR", "gsis_id": f"G{i}"}
        for i in range(5)
    ]
    with connect(tmp_db) as conn:
        upsert_many(conn, "players", rows, conflict_cols=["gsis_id"])
        count = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    assert count == 5


def test_connect_rolls_back_on_exception(tmp_db):
    with pytest.raises(RuntimeError):
        with connect(tmp_db) as conn:
            conn.execute(
                "INSERT INTO players (full_name, gsis_id) VALUES (?, ?)",
                ("Should Not Persist", "G_ROLLBACK"),
            )
            raise RuntimeError("boom")

    with connect(tmp_db) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM players WHERE gsis_id='G_ROLLBACK'"
        ).fetchone()[0]
    assert n == 0
