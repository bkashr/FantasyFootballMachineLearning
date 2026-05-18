import pytest

from database import connect, upsert_many
from projections import (
    _age_at_season,
    _age_factor,
    _injury_factor,
    _weighted_per_game_baseline,
    AGE_CURVES,
    DEFAULT_YEAR_WEIGHTS,
    backtest,
    project_all,
    project_player,
)


def _seed_player_with_seasons(db_path, name, position, birth_date, season_rows):
    """season_rows: list of (season, ppr, games_played)."""
    with connect(db_path) as conn:
        upsert_many(
            conn,
            "players",
            [
                {
                    "full_name": name,
                    "position": position,
                    "birth_date": birth_date,
                    "gsis_id": f"G_{name.replace(' ', '_')}",
                }
            ],
            conflict_cols=["gsis_id"],
        )
        pid = conn.execute(
            "SELECT player_id FROM players WHERE full_name = ?", (name,)
        ).fetchone()[0]
        upsert_many(
            conn,
            "player_season_stats",
            [
                {
                    "player_id": pid,
                    "season": season,
                    "fantasy_points_ppr": ppr,
                    "fantasy_points": ppr * 0.7,
                    "games_played": games,
                }
                for season, ppr, games in season_rows
            ],
            conflict_cols=["player_id", "season"],
        )
        return pid


def test_age_at_season_handles_iso_and_missing():
    assert _age_at_season("1995-04-12", 2024) == 29
    assert _age_at_season("1990-12-31", 2024) == 34
    assert _age_at_season(None, 2024) is None
    assert _age_at_season("garbage", 2024) is None


def test_age_factor_is_flat_under_peak():
    # At and below peak age, no decline
    for pos in AGE_CURVES:
        peak = AGE_CURVES[pos]["peak_age"]
        assert _age_factor(peak, pos) == 1.0
        assert _age_factor(peak - 5, pos) == 1.0


def test_age_factor_declines_above_peak():
    # RBs decline fastest; QBs slowest
    rb_over_3 = _age_factor(AGE_CURVES["RB"]["peak_age"] + 3, "RB")
    qb_over_3 = _age_factor(AGE_CURVES["QB"]["peak_age"] + 3, "QB")
    assert rb_over_3 < qb_over_3 < 1.0


def test_age_factor_respects_floor():
    # 10 years past peak should hit the floor, not go below
    assert _age_factor(AGE_CURVES["RB"]["peak_age"] + 20, "RB") == AGE_CURVES["RB"]["floor"]


def test_age_factor_unknown_position_is_identity():
    assert _age_factor(35, "LB") == 1.0
    assert _age_factor(35, None) == 1.0


def test_injury_factor_clean_player_is_one():
    assert _injury_factor(games_missed=0, possible_games=51) == 1.0


def test_injury_factor_floors_at_constant():
    # Missing every single game still floors at 0.7
    assert _injury_factor(games_missed=51, possible_games=51) == pytest.approx(0.70)


def test_injury_factor_scales_linearly():
    # 50% games missed → factor = 1 - 0.5 * 0.5 = 0.75
    assert _injury_factor(games_missed=25, possible_games=50) == pytest.approx(0.75)


def test_injury_factor_no_possible_games_is_identity():
    assert _injury_factor(games_missed=0, possible_games=0) == 1.0


def test_weighted_per_game_baseline_uses_recency_weights():
    # Same per-game across years → output equals that per-game
    rows = [
        {"season": 2022, "fantasy_points_ppr": 170, "games_played": 17},
        {"season": 2021, "fantasy_points_ppr": 170, "games_played": 17},
        {"season": 2020, "fantasy_points_ppr": 170, "games_played": 17},
    ]
    pg, n = _weighted_per_game_baseline(rows, target_season=2023)
    assert pg == pytest.approx(10.0)
    assert n == 3


def test_weighted_per_game_baseline_skips_zero_games():
    rows = [
        {"season": 2022, "fantasy_points_ppr": 0, "games_played": 0},
        {"season": 2021, "fantasy_points_ppr": 170, "games_played": 17},
    ]
    pg, n = _weighted_per_game_baseline(rows, target_season=2023)
    assert pg == pytest.approx(10.0)
    assert n == 1


def test_weighted_per_game_baseline_weights_recent_year_more():
    # Recent year per-game = 15, older = 5. Recent weighted more
    # so weighted value should be closer to 15 than to 5.
    rows = [
        {"season": 2022, "fantasy_points_ppr": 255, "games_played": 17},  # 15/g
        {"season": 2021, "fantasy_points_ppr": 85, "games_played": 17},   # 5/g
    ]
    pg, n = _weighted_per_game_baseline(rows, target_season=2023)
    # weights 0.55 / 0.30 → normalized 0.647 / 0.353
    # weighted = 15 * 0.647 + 5 * 0.353 ≈ 11.47
    assert 10.5 < pg < 12.5


def test_project_player_returns_none_for_no_history(tmp_db):
    _seed_player_with_seasons(tmp_db, "Rookie Guy", "WR", "2002-01-01", [])
    with connect(tmp_db) as conn:
        pid = conn.execute(
            "SELECT player_id FROM players WHERE full_name='Rookie Guy'"
        ).fetchone()[0]
        proj = project_player(conn, pid, target_season=2024)
    assert proj is None


def test_project_player_combines_factors(tmp_db):
    # 26-year-old WR averaging 10 PPR/g for 17 games over 3 seasons
    # No injuries, in peak age → projection should be ~170
    _seed_player_with_seasons(
        tmp_db,
        "Healthy Peak",
        "WR",
        "1998-01-01",
        [(2021, 170, 17), (2022, 170, 17), (2023, 170, 17)],
    )
    with connect(tmp_db) as conn:
        pid = conn.execute(
            "SELECT player_id FROM players WHERE full_name='Healthy Peak'"
        ).fetchone()[0]
        proj = project_player(conn, pid, target_season=2024)
    assert proj is not None
    assert proj.age_factor == 1.0  # 26 yo WR, well under peak (29)
    assert proj.injury_factor == 1.0  # 51/51 games played
    assert proj.projected_points_ppr == pytest.approx(170.0)
    assert proj.n_prior_seasons == 3


def test_project_player_discounts_age_for_old_rb(tmp_db):
    # 32-year-old RB averaging 10 PPR/g — age factor should bite
    _seed_player_with_seasons(
        tmp_db,
        "Old RB",
        "RB",
        "1992-01-01",
        [(2021, 170, 17), (2022, 170, 17), (2023, 170, 17)],
    )
    with connect(tmp_db) as conn:
        pid = conn.execute(
            "SELECT player_id FROM players WHERE full_name='Old RB'"
        ).fetchone()[0]
        proj = project_player(conn, pid, target_season=2024)
    # 2024 - 1992 = 32 → 5 years past RB peak of 27 → 0.08 * 5 = 0.40 off
    # factor = 1 - 0.40 = 0.60
    assert proj.age_factor == pytest.approx(0.60)
    assert proj.projected_points_ppr < proj.base_ppr


def test_project_player_discounts_injury_prone(tmp_db):
    # Healthy peak-age WR but missed half his games
    _seed_player_with_seasons(
        tmp_db,
        "Injury Prone",
        "WR",
        "1998-01-01",
        [(2021, 85, 9), (2022, 85, 8), (2023, 85, 9)],
    )
    with connect(tmp_db) as conn:
        pid = conn.execute(
            "SELECT player_id FROM players WHERE full_name='Injury Prone'"
        ).fetchone()[0]
        proj = project_player(conn, pid, target_season=2024)
    assert proj is not None
    # Missed games: 51 - 26 = 25 of 51 possible → 49% miss rate → factor 0.755
    assert 0.74 < proj.injury_factor < 0.77
    # Base per-game ≈ 9.86 / game * 17 ≈ 167 — but age 26 WR so age_factor 1.0
    # Injured but productive → still projects > 100
    assert 110 < proj.projected_points_ppr < 140


def test_project_all_writes_to_player_projections(tmp_db):
    _seed_player_with_seasons(tmp_db, "Alice", "WR", "1998-01-01", [(2023, 170, 17)])
    _seed_player_with_seasons(tmp_db, "Bob", "RB", "1996-01-01", [(2023, 200, 16)])
    summary = project_all(target_season=2024, db_path=tmp_db)
    assert summary["projected"] == 2

    with connect(tmp_db) as conn:
        rows = list(conn.execute(
            "SELECT p.full_name, pr.projected_points_ppr, pr.source "
            "FROM player_projections pr JOIN players p USING (player_id) "
            "WHERE pr.season=2024"
        ))
    names = {r["full_name"] for r in rows}
    assert names == {"Alice", "Bob"}
    assert all(r["source"] == "internal_v2" for r in rows)


def test_project_all_is_idempotent(tmp_db):
    _seed_player_with_seasons(tmp_db, "Alice", "WR", "1998-01-01", [(2023, 170, 17)])
    project_all(target_season=2024, db_path=tmp_db)
    project_all(target_season=2024, db_path=tmp_db)
    with connect(tmp_db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM player_projections").fetchone()[0]
    assert n == 1


def test_backtest_compares_projection_to_actual(tmp_db):
    # Player who scores exactly the same per-game each year — projection
    # should hit actual cleanly.
    _seed_player_with_seasons(
        tmp_db,
        "Predictable",
        "WR",
        "1998-01-01",
        [(2020, 170, 17), (2021, 170, 17), (2022, 170, 17), (2023, 170, 17)],
    )
    result = backtest(target_season=2023, db_path=tmp_db, min_actual_games=4)
    assert result["n"] == 1
    assert abs(result["mae"]) < 1e-6  # perfect projection
    assert result["correlation"] == pytest.approx(0.0) or result["correlation"] >= 0
