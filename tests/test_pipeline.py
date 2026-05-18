"""End-to-end pipeline test: roster ingest -> stat ingest -> ADP ingest
-> scoring -> rankings. Uses stubbed nflverse data and an in-memory ADP
fixture so the test runs offline."""

import json

import pandas as pd

import ingest_nflverse
from database import connect
from ingest_nflverse import (
    ingest_rosters,
    ingest_season_stats,
)
from ingest_underdog import fetch_adp_from_file, ingest_adp
from scoring import score_players, top_reaches, top_values


def test_pipeline_ingests_scores_and_ranks(tmp_db, tmp_path, monkeypatch):
    # 1. Roster: 4 players with prior-season stats
    roster_df = pd.DataFrame(
        [
            {"season": 2023, "player_id": "G_STAR",  "player_name": "Big Star",  "position": "WR", "team": "DAL"},
            {"season": 2023, "player_id": "G_VAL",   "player_name": "Late Bloomer", "position": "RB", "team": "DET"},
            {"season": 2023, "player_id": "G_REACH", "player_name": "Overdrafted", "position": "WR", "team": "NYG"},
            {"season": 2023, "player_id": "G_MID",   "player_name": "Mid Guy",   "position": "TE", "team": "SF"},
        ]
    )
    monkeypatch.setattr(ingest_nflverse.nfl, "import_seasonal_rosters", lambda seasons: roster_df)
    ingest_rosters([2023], db_path=tmp_db)

    # 2. Season stats: very different prior PPR levels
    def _stat_row(pid, ppr):
        return {
            "player_id": pid, "season": 2023, "season_type": "REG",
            "attempts": 0, "completions": 0, "passing_yards": 0, "passing_tds": 0, "interceptions": 0,
            "carries": 0, "rushing_yards": 0, "rushing_tds": 0,
            "targets": 0, "receptions": 0, "receiving_yards": 0, "receiving_tds": 0,
            "fantasy_points": ppr * 0.7, "fantasy_points_ppr": ppr, "games": 17,
            "sack_fumbles_lost": 0, "rushing_fumbles_lost": 0, "receiving_fumbles_lost": 0,
        }

    season_df = pd.DataFrame(
        [
            _stat_row("G_STAR", 350),
            _stat_row("G_VAL", 280),
            _stat_row("G_REACH", 100),
            _stat_row("G_MID", 200),
        ]
    )
    monkeypatch.setattr(ingest_nflverse.nfl, "import_seasonal_data", lambda seasons: season_df)
    ingest_season_stats([2023], db_path=tmp_db)

    # 3. ADP fixture: Star at pick 1, Reach at pick 2 (overpriced),
    # Mid at pick 30, Value at pick 80 (underpriced relative to 280 PPR).
    fixture = tmp_path / "adp.json"
    fixture.write_text(json.dumps([
        {"underdog_id": "u_star",  "full_name": "Big Star",     "adp": 1.0,  "adp_rank": 1,  "draft_format": "best_ball"},
        {"underdog_id": "u_reach", "full_name": "Overdrafted",  "adp": 2.0,  "adp_rank": 2,  "draft_format": "best_ball"},
        {"underdog_id": "u_mid",   "full_name": "Mid Guy",      "adp": 30.0, "adp_rank": 30, "draft_format": "best_ball"},
        {"underdog_id": "u_val",   "full_name": "Late Bloomer", "adp": 80.0, "adp_rank": 80, "draft_format": "best_ball"},
    ]))
    n = ingest_adp(fetch=lambda: fetch_adp_from_file(fixture), db_path=tmp_db)
    assert n == 4

    # 4. Score with a clean curve: descending by rank.
    curve = {0: 350, 1: 320, 2: 290, 3: 260, 4: 230, 5: 200, 6: 170, 7: 140, 8: 110, 9: 80, 10: 60, 11: 40, 12: 30, 13: 25, 14: 20}
    score_players(season=2024, db_path=tmp_db, bucket_size=6, curve=curve)

    # 5. Rankings
    values = top_values(2024, limit=4, db_path=tmp_db)
    reaches = top_reaches(2024, limit=4, db_path=tmp_db)

    assert values[0]["full_name"] == "Late Bloomer"   # cheapest, big PPR
    assert reaches[0]["full_name"] == "Overdrafted"   # priced top but low PPR
    # Crosswalk backfill worked
    with connect(tmp_db) as conn:
        uids = dict(conn.execute("SELECT full_name, underdog_id FROM players WHERE underdog_id IS NOT NULL"))
    assert uids["Big Star"] == "u_star"
    assert uids["Late Bloomer"] == "u_val"
