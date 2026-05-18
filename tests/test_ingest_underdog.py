import json

import pytest

from database import connect, upsert_many
from ingest_underdog import (
    _normalize_name,
    _resolve_player,
    fetch_adp_from_file,
    ingest_adp,
    normalize_adp_response,
)


def test_normalize_name_strips_punctuation_and_case():
    assert _normalize_name("D'Andre Swift") == "dandre swift"
    assert _normalize_name("A.J. Brown") == "aj brown"
    assert _normalize_name("Amon-Ra St. Brown") == "amon ra st brown"


def test_normalize_adp_response_handles_alternate_keys():
    raw = {
        "players": [
            {"id": "u1", "name": "A B", "position": "WR", "adp": 1.0, "adp_rank": 1},
            {"player_id": "u2", "display_name": "C D", "team_abbr": "DAL", "adp": 2.0, "rank": 2},
        ]
    }
    out = normalize_adp_response(raw)
    assert [r["underdog_id"] for r in out] == ["u1", "u2"]
    assert out[1]["team"] == "DAL"
    assert out[1]["adp_rank"] == 2


def test_resolve_player_prefers_underdog_id():
    by_uid = {"u1": 100}
    by_name = {"jane doe": 200}
    rec = {"underdog_id": "u1", "full_name": "Jane Doe"}
    assert _resolve_player(rec, by_uid, by_name) == 100


def test_resolve_player_falls_back_to_name():
    by_uid: dict[str, int] = {}
    by_name = {"jane doe": 200}
    rec = {"underdog_id": "u_new", "full_name": "Jane Doe"}
    assert _resolve_player(rec, by_uid, by_name) == 200


def test_resolve_player_returns_none_on_miss():
    rec = {"underdog_id": None, "full_name": "Nobody"}
    assert _resolve_player(rec, {}, {}) is None


def test_ingest_adp_writes_snapshots_and_backfills_underdog_id(tmp_db, tmp_path):
    with connect(tmp_db) as conn:
        upsert_many(
            conn,
            "players",
            [
                {"full_name": "Jane Doe", "position": "WR", "gsis_id": "G_JD"},
                {"full_name": "Bob Smith", "position": "RB", "gsis_id": "G_BS"},
            ],
            conflict_cols=["gsis_id"],
        )

    fixture = tmp_path / "adp.json"
    fixture.write_text(
        json.dumps(
            [
                {"underdog_id": "u_jd", "full_name": "Jane Doe", "adp": 12.4, "adp_rank": 12},
                {"underdog_id": "u_bs", "full_name": "Bob Smith", "adp": 30.1, "adp_rank": 30},
                {"underdog_id": "u_x", "full_name": "Unknown Player", "adp": 99.0, "adp_rank": 200},
            ]
        )
    )

    n = ingest_adp(
        fetch=lambda: fetch_adp_from_file(fixture),
        db_path=tmp_db,
    )
    assert n == 2

    with connect(tmp_db) as conn:
        snaps = list(conn.execute("SELECT player_id, adp FROM adp_snapshots ORDER BY adp"))
        uids = dict(
            conn.execute(
                "SELECT full_name, underdog_id FROM players WHERE underdog_id IS NOT NULL"
            )
        )
    assert len(snaps) == 2
    assert uids == {"Jane Doe": "u_jd", "Bob Smith": "u_bs"}


def test_ingest_adp_respects_min_age_hours(tmp_db, tmp_path):
    with connect(tmp_db) as conn:
        upsert_many(
            conn,
            "players",
            [{"full_name": "Jane Doe", "position": "WR", "gsis_id": "G_JD"}],
            conflict_cols=["gsis_id"],
        )

    fixture = tmp_path / "adp.json"
    fixture.write_text(
        json.dumps([{"underdog_id": "u_jd", "full_name": "Jane Doe", "adp": 5.0, "adp_rank": 5}])
    )

    # First run — empty DB, no min-age constraint -> writes 1
    assert ingest_adp(fetch=lambda: fetch_adp_from_file(fixture), db_path=tmp_db) == 1
    # Second run with a 24h threshold -> no new snapshot
    assert (
        ingest_adp(
            fetch=lambda: fetch_adp_from_file(fixture),
            min_age_hours=24.0,
            db_path=tmp_db,
        )
        == 0
    )
