import pytest

from database import connect, upsert_many
from portfolio import (
    draft_count,
    import_drafts_from_csv,
    player_exposure,
    team_exposure,
)


def _seed_players(db_path, players):
    """players: list of (name, position, team)."""
    with connect(db_path) as conn:
        upsert_many(
            conn,
            "players",
            [
                {"full_name": n, "position": p, "team": t, "gsis_id": f"G_{i}"}
                for i, (n, p, t) in enumerate(players)
            ],
            conflict_cols=["gsis_id"],
        )


def _write_csv(tmp_path, rows, name="portfolio.csv"):
    f = tmp_path / name
    f.write_text(rows)
    return f


def test_import_creates_drafts_and_picks(tmp_db, tmp_path):
    _seed_players(tmp_db, [
        ("Bijan Robinson", "RB", "ATL"),
        ("CeeDee Lamb",    "WR", "DAL"),
    ])
    csv_path = _write_csv(tmp_path, (
        "draft_name,season,my_slot,round,pick_overall,player_name,position\n"
        '"Draft A",2026,5,1,5,"Bijan Robinson","RB"\n'
        '"Draft A",2026,5,2,20,"CeeDee Lamb","WR"\n'
    ))
    result = import_drafts_from_csv(csv_path, db_path=tmp_db)
    assert result == {"drafts_imported": 1, "picks_imported": 2, "unmatched": 0}
    assert draft_count(2026, db_path=tmp_db) == 1


def test_import_is_idempotent(tmp_db, tmp_path):
    _seed_players(tmp_db, [("Bijan Robinson", "RB", "ATL")])
    csv_path = _write_csv(tmp_path, (
        "draft_name,season,my_slot,round,pick_overall,player_name,position\n"
        '"Draft A",2026,5,1,5,"Bijan Robinson","RB"\n'
    ))
    import_drafts_from_csv(csv_path, db_path=tmp_db)
    second = import_drafts_from_csv(csv_path, db_path=tmp_db)
    assert second["picks_imported"] == 0
    # And still only 1 pick in the DB
    with connect(tmp_db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM my_picks").fetchone()[0]
    assert n == 1


def test_import_flags_unmatched_names(tmp_db, tmp_path):
    _seed_players(tmp_db, [("Bijan Robinson", "RB", "ATL")])
    csv_path = _write_csv(tmp_path, (
        "draft_name,season,my_slot,round,pick_overall,player_name,position\n"
        '"Draft A",2026,5,1,5,"Bijan Robinson","RB"\n'
        '"Draft A",2026,5,2,20,"Made Up Player","WR"\n'
    ))
    result = import_drafts_from_csv(csv_path, db_path=tmp_db)
    assert result["picks_imported"] == 1
    assert result["unmatched"] == 1


def test_player_exposure_percentages_track_draft_count(tmp_db, tmp_path):
    _seed_players(tmp_db, [
        ("Bijan Robinson", "RB", "ATL"),
        ("CeeDee Lamb",    "WR", "DAL"),
    ])
    # Bijan in 2 of 3 drafts (66%), Lamb in 1 (33%)
    csv_path = _write_csv(tmp_path, (
        "draft_name,season,my_slot,round,pick_overall,player_name,position\n"
        '"A",2026,5,1,5,"Bijan Robinson","RB"\n'
        '"B",2026,5,1,7,"Bijan Robinson","RB"\n'
        '"B",2026,5,2,18,"CeeDee Lamb","WR"\n'
        '"C",2026,5,1,9,"CeeDee Lamb","WR"\n'
    ))
    import_drafts_from_csv(csv_path, db_path=tmp_db)

    exposure = player_exposure(2026, db_path=tmp_db)
    by_name = {e["full_name"]: e for e in exposure}
    assert by_name["Bijan Robinson"]["n_rosters"] == 2
    assert by_name["Bijan Robinson"]["pct_of_drafts"] == pytest.approx(2 / 3 * 100)
    assert by_name["CeeDee Lamb"]["n_rosters"] == 2
    # All 3 drafts have one of these two -> ordering is exposure-desc
    assert exposure[0]["n_rosters"] == 2


def test_team_exposure_counts_drafts_with_each_team(tmp_db, tmp_path):
    _seed_players(tmp_db, [
        ("Bijan Robinson", "RB", "ATL"),
        ("Drake London",   "WR", "ATL"),
        ("CeeDee Lamb",    "WR", "DAL"),
    ])
    # Draft A: both Falcons + a Cowboy. Draft B: just a Cowboy.
    csv_path = _write_csv(tmp_path, (
        "draft_name,season,my_slot,round,pick_overall,player_name,position\n"
        '"A",2026,5,1,5,"Bijan Robinson","RB"\n'
        '"A",2026,5,2,20,"Drake London","WR"\n'
        '"A",2026,5,3,29,"CeeDee Lamb","WR"\n'
        '"B",2026,5,1,9,"CeeDee Lamb","WR"\n'
    ))
    import_drafts_from_csv(csv_path, db_path=tmp_db)
    teams = {r["team"]: r for r in team_exposure(2026, db_path=tmp_db)}
    # ATL is in draft A only (1 draft, 2 total picks)
    assert teams["ATL"]["drafts_with_team"] == 1
    assert teams["ATL"]["picks_total"] == 2
    # DAL is in both drafts (2 drafts, 2 total picks)
    assert teams["DAL"]["drafts_with_team"] == 2


def test_import_rejects_csv_missing_required_columns(tmp_db, tmp_path):
    _seed_players(tmp_db, [("Bijan", "RB", "ATL")])
    csv_path = _write_csv(tmp_path, (
        "draft_name,season,round,player_name\n"
        '"A",2026,1,"Bijan"\n'
    ))
    with pytest.raises(ValueError, match="missing columns"):
        import_drafts_from_csv(csv_path, db_path=tmp_db)
