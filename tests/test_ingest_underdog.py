import json

import pytest

from database import connect, upsert_many
from ingest_underdog import (
    _build_lookups,
    _infer_year_from_filename,
    _normalize_name,
    _parse_adp_date_label,
    _resolve_player,
    fetch_adp_from_file,
    ingest_adp,
    ingest_adp_from_4for4_csv,
    normalize_adp_response,
)


def test_normalize_name_strips_punctuation_and_case():
    assert _normalize_name("D'Andre Swift") == "dandre swift"
    assert _normalize_name("A.J. Brown") == "aj brown"
    assert _normalize_name("Amon-Ra St. Brown") == "amon ra st brown"


def test_normalize_name_drops_generational_suffixes():
    # Jr/Sr/II/III/IV should not split a player into two crosswalk entries
    assert _normalize_name("Marvin Harrison Jr.") == _normalize_name("Marvin Harrison")
    assert _normalize_name("Odell Beckham Jr.") == _normalize_name("Odell Beckham")
    assert _normalize_name("Calvin Ridley III") == "calvin ridley"
    assert _normalize_name("Cedric Tillman, Sr.") == "cedric tillman"


def test_normalize_name_collapses_whitespace():
    assert _normalize_name("  Patrick   Mahomes  ") == "patrick mahomes"


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
    assert _resolve_player(rec, by_uid, {}, by_name) == 100


def test_resolve_player_falls_back_to_name():
    by_uid: dict[str, int] = {}
    by_name = {"jane doe": 200}
    rec = {"underdog_id": "u_new", "full_name": "Jane Doe"}
    assert _resolve_player(rec, by_uid, {}, by_name) == 200


def test_resolve_player_uses_position_to_break_ties():
    # Two players named "Josh Allen" — QB and LB. With position in the
    # record, we should match the QB; without, name-only fallback might
    # pick either, so we don't assert on that case here.
    by_uid: dict[str, int] = {}
    by_name_pos = {("josh allen", "QB"): 100, ("josh allen", "LB"): 200}
    by_name: dict = {}  # ambiguous, so collapsed out
    rec = {"underdog_id": None, "full_name": "Josh Allen", "position": "QB"}
    assert _resolve_player(rec, by_uid, by_name_pos, by_name) == 100


def test_resolve_player_returns_none_on_miss():
    rec = {"underdog_id": None, "full_name": "Nobody"}
    assert _resolve_player(rec, {}, {}, {}) is None


def test_build_lookups_prefers_fantasy_position_on_collision(tmp_db):
    from database import connect, upsert_many

    with connect(tmp_db) as conn:
        upsert_many(
            conn,
            "players",
            [
                {"full_name": "Josh Allen", "position": "QB", "gsis_id": "G_QB"},
                {"full_name": "Josh Allen", "position": "LB", "gsis_id": "G_LB"},
                {"full_name": "Bijan Robinson", "position": "RB", "gsis_id": "G_BR"},
            ],
            conflict_cols=["gsis_id"],
        )
        qb_pid = conn.execute(
            "SELECT player_id FROM players WHERE gsis_id='G_QB'"
        ).fetchone()[0]
        by_uid, by_name_pos, by_name = _build_lookups(conn)

    # Name-only lookup must resolve to the QB on a name collision
    assert by_name["josh allen"] == qb_pid
    # Name+pos lookup has both entries
    assert ("josh allen", "QB") in by_name_pos
    assert ("josh allen", "LB") in by_name_pos


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


def test_parse_adp_date_label_handles_month_day():
    assert _parse_adp_date_label("April 25", 2026) == "2026-04-25T00:00:00"
    assert _parse_adp_date_label("May 19", 2026) == "2026-05-19T00:00:00"
    assert _parse_adp_date_label("December 1", 2024) == "2024-12-01T00:00:00"


def test_parse_adp_date_label_rejects_bad_input():
    import pytest
    with pytest.raises(ValueError):
        _parse_adp_date_label("Smarch 7", 2026)
    with pytest.raises(ValueError):
        _parse_adp_date_label("garbage", 2026)


def test_infer_year_from_filename():
    from pathlib import Path
    assert _infer_year_from_filename(Path("Underdog_Draft_Table_20260519.csv")) == 2026
    assert _infer_year_from_filename(Path("foo_20240801_bar.csv")) == 2024
    assert _infer_year_from_filename(Path("no_date_here.csv")) is None


def test_ingest_4for4_creates_one_snapshot_per_date_column(tmp_db, tmp_path):
    from database import connect, upsert_many

    # Seed two players matching by (name, position)
    with connect(tmp_db) as conn:
        upsert_many(
            conn,
            "players",
            [
                {"full_name": "Bijan Robinson", "position": "RB", "gsis_id": "G1"},
                {"full_name": "Ja'Marr Chase",  "position": "WR", "gsis_id": "G2"},
            ],
            conflict_cols=["gsis_id"],
        )

    csv_path = tmp_path / "Underdog_Draft_Table_20260519.csv"
    csv_path.write_text(
        '"Rank","Player","Position","Position Rank","ADP on April 25","ADP on May 19","ADP Change"\n'
        '1,"Bijan Robinson","RB","RB1",1.5,1.5,0\n'
        '2,"Ja\'Marr Chase","WR","WR1",3.1,3.0,-0.1\n'
        '500,"Tail Guy","WR","WR99",215.5,216,0.5\n'   # undrafted sentinel -> filtered
    )
    summary = ingest_adp_from_4for4_csv(csv_path, db_path=tmp_db)

    # Two date columns, two real players => 4 snapshots written
    assert summary["snapshots_written"] == 4
    assert summary["by_date"]["2026-04-25T00:00:00"]["written"] == 2
    assert summary["by_date"]["2026-05-19T00:00:00"]["written"] == 2

    with connect(tmp_db) as conn:
        # Tail Guy never made it in (filtered + unmatched, both reasons)
        rows = list(conn.execute(
            "SELECT captured_at, source, adp, adp_rank FROM adp_snapshots "
            "ORDER BY captured_at, adp"
        ))
    assert len(rows) == 4
    # adp_rank derived from sort within each snapshot
    assert rows[0]["adp_rank"] == 1
    assert rows[1]["adp_rank"] == 2
    # All rows tagged with the 4for4 source
    assert all(r["source"] == "4for4_underdog" for r in rows)


def test_ingest_4for4_rejects_csv_with_no_date_columns(tmp_db, tmp_path):
    import pytest
    csv_path = tmp_path / "no_dates.csv"
    csv_path.write_text('"Rank","Player","Position"\n1,"Bijan","RB"\n')
    with pytest.raises(ValueError, match="No 'ADP on"):
        ingest_adp_from_4for4_csv(csv_path, db_path=tmp_db, year=2026)


def test_ingest_4for4_is_idempotent_on_overlap(tmp_db, tmp_path):
    from database import connect, upsert_many

    with connect(tmp_db) as conn:
        upsert_many(
            conn,
            "players",
            [{"full_name": "Bijan Robinson", "position": "RB", "gsis_id": "G1"}],
            conflict_cols=["gsis_id"],
        )

    csv_path = tmp_path / "Underdog_Draft_Table_20260519.csv"
    csv_path.write_text(
        '"Rank","Player","Position","Position Rank","ADP on April 25","ADP on May 19","ADP Change"\n'
        '1,"Bijan Robinson","RB","RB1",1.5,1.5,0\n'
    )

    # First ingest: writes 2 rows (Apr 25 + May 19)
    s1 = ingest_adp_from_4for4_csv(csv_path, db_path=tmp_db)
    assert s1["snapshots_written"] == 2
    assert s1["skipped_existing"] == 0

    # Second ingest of same file: nothing new
    s2 = ingest_adp_from_4for4_csv(csv_path, db_path=tmp_db)
    assert s2["snapshots_written"] == 0
    assert s2["skipped_existing"] == 2

    # Now an overlapping file (May 19 + June 3) — only June 3 should be new
    csv2 = tmp_path / "Underdog_Draft_Table_20260603.csv"
    csv2.write_text(
        '"Rank","Player","Position","Position Rank","ADP on May 19","ADP on June 3","ADP Change"\n'
        '1,"Bijan Robinson","RB","RB1",1.5,1.4,-0.1\n'
    )
    s3 = ingest_adp_from_4for4_csv(csv2, db_path=tmp_db)
    assert s3["snapshots_written"] == 1
    assert s3["skipped_existing"] == 1

    with connect(tmp_db) as conn:
        dates = [
            r["captured_at"]
            for r in conn.execute(
                "SELECT DISTINCT captured_at FROM adp_snapshots ORDER BY captured_at"
            )
        ]
    assert dates == [
        "2026-04-25T00:00:00",
        "2026-05-19T00:00:00",
        "2026-06-03T00:00:00",
    ]
