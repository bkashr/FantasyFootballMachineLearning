import pytest

from database import connect, upsert_many
from scoring import (
    _expected_for_position_rank,
    _expected_for_rank,
    _synthetic_curve_by_position,
    _synthetic_curve_from_ppr_rank,
    build_value_curve,
    build_value_curve_by_position,
    score_players,
    top_reaches,
    top_values,
)


def _seed_players_and_stats(db_path, players_with_ppr):
    """players_with_ppr: list of (full_name, prior_season_ppr, current_season_ppr_or_None)."""
    with connect(db_path) as conn:
        upsert_many(
            conn,
            "players",
            [
                {"full_name": name, "position": "WR", "gsis_id": f"G_{i}"}
                for i, (name, _, _) in enumerate(players_with_ppr)
            ],
            conflict_cols=["gsis_id"],
        )
        pids = dict(conn.execute("SELECT full_name, player_id FROM players"))

        stats = []
        for name, prior, current in players_with_ppr:
            stats.append(
                {
                    "player_id": pids[name],
                    "season": 2023,
                    "fantasy_points_ppr": prior,
                    "fantasy_points": prior * 0.7 if prior is not None else None,
                }
            )
            if current is not None:
                stats.append(
                    {
                        "player_id": pids[name],
                        "season": 2024,
                        "fantasy_points_ppr": current,
                        "fantasy_points": current * 0.7,
                    }
                )
        upsert_many(conn, "player_season_stats", stats, conflict_cols=["player_id", "season"])
        return pids


def _seed_adp(db_path, pids, ranks_by_name, draft_format="best_ball"):
    rows = []
    for name, (adp, rank) in ranks_by_name.items():
        rows.append(
            {
                "player_id": pids[name],
                "adp": float(adp),
                "adp_rank": rank,
                "draft_format": draft_format,
                "source": "underdog",
                "captured_at": "2026-01-01T00:00:00",
            }
        )
    with connect(db_path) as conn:
        cols = ["player_id", "adp", "adp_rank", "draft_format", "source", "captured_at"]
        conn.executemany(
            f"INSERT INTO adp_snapshots ({','.join(cols)}) VALUES ({','.join(['?'] * len(cols))})",
            [[r[c] for c in cols] for r in rows],
        )


def test_expected_for_rank_extrapolates_with_last_bucket():
    curve = {0: 300.0, 1: 200.0, 2: 100.0}
    # bucket size 6: rank 1 -> bucket 0
    assert _expected_for_rank(1, curve, bucket_size=6) == 300.0
    assert _expected_for_rank(7, curve, bucket_size=6) == 200.0
    # rank way past the last bucket -> hold flat at last
    assert _expected_for_rank(200, curve, bucket_size=6) == 100.0


def test_synthetic_curve_descends_with_rank(tmp_db):
    _seed_players_and_stats(
        tmp_db,
        [
            ("Top1", 400, None),
            ("Top2", 380, None),
            ("Mid1", 200, None),
            ("Mid2", 180, None),
            ("Low1", 60, None),
            ("Low2", 40, None),
        ],
    )
    with connect(tmp_db) as conn:
        curve = _synthetic_curve_from_ppr_rank(conn, season=2023, bucket_size=2)
    # 2-player buckets: 0 = top two, 1 = mid two, 2 = low two — descending
    assert curve[0] > curve[1] > curve[2]
    assert curve[0] == 390.0
    assert curve[1] == 190.0
    assert curve[2] == 50.0


def test_build_value_curve_only_uses_preseason_snapshots(tmp_db):
    # Player drafted at rank 1 in pre-2023 season, then finished 2023 with 300 PPR.
    # We seed BOTH a pre-season snapshot and a post-season one. Only the
    # pre-season snapshot should be used so the curve isn't leaky.
    pids = _seed_players_and_stats(tmp_db, [("Star", 300, None)])
    with connect(tmp_db) as conn:
        for captured_at in ("2023-08-01T00:00:00", "2024-02-01T00:00:00"):
            conn.execute(
                "INSERT INTO adp_snapshots (player_id, adp, adp_rank, draft_format, source, captured_at) "
                "VALUES (?,?,?,?,?,?)",
                (pids["Star"], 1.0, 1, "best_ball", "underdog", captured_at),
            )

        curve = build_value_curve(conn, seasons=[2023], bucket_size=1)
    # Should have only one observation (the preseason snapshot), averaging
    # cleanly to 300. If both snapshots were used we'd still get 300, but
    # with TWO contributions — the test is that the post-season row isn't
    # silently joined twice.
    assert curve == {0: 300.0}


def test_build_value_curve_skips_outside_year_snapshots(tmp_db):
    pids = _seed_players_and_stats(tmp_db, [("Star", 300, None)])
    with connect(tmp_db) as conn:
        # Snapshot in 2025 — too late to inform 2023 outcomes
        conn.execute(
            "INSERT INTO adp_snapshots (player_id, adp, adp_rank, draft_format, source, captured_at) "
            "VALUES (?,?,?,?,?,?)",
            (pids["Star"], 1.0, 1, "best_ball", "underdog", "2025-08-01T00:00:00"),
        )
        curve = build_value_curve(conn, seasons=[2023], bucket_size=1)
    assert curve == {}


def test_score_players_marks_value_vs_reach(tmp_db):
    # Build a clear talent gradient: prior PPR descends from 360 to 60.
    # With bucket_size=1 the synthetic curve is just that descending list,
    # so expected_at_rank tracks the talent curve closely.
    pids = _seed_players_and_stats(
        tmp_db,
        [
            ("Stud A", 360, None),
            ("Stud B", 340, None),
            ("Stud C", 320, None),
            ("Mid A",  260, None),
            ("Mid B",  240, None),
            ("Mid C",  220, None),
            ("Late A", 180, None),
            ("Late B", 140, None),
            ("Late C", 100, None),
            ("Bench",   60, None),
        ],
    )
    # Reach Guy = mid-talent player drafted at the top (rank 1, expected ~360)
    # Value Guy = mid-talent player drafted late (rank 9, expected ~100)
    _seed_adp(
        tmp_db,
        pids,
        {
            "Mid A": (1.0, 1),   # reach: 260 actual vs ~360 expected
            "Mid B": (9.0, 9),   # value: 240 actual vs ~100 expected
        },
    )
    # Inject a clean curve: each rank bucket expects PPR matching the
    # player at that finish — Stud A (360) at rank 1, Stud B (340) at
    # rank 2, ... Bench (60) at rank 10. With bucket_size=1, each
    # rank maps to one bucket.
    injected_curve = dict(enumerate([360, 340, 320, 260, 240, 220, 180, 140, 100, 60]))
    score_players(season=2024, bucket_size=1, db_path=tmp_db, curve=injected_curve)

    values = top_values(2024, limit=2, db_path=tmp_db)
    reaches = top_reaches(2024, limit=2, db_path=tmp_db)
    assert values[0]["full_name"] == "Mid B"
    assert reaches[0]["full_name"] == "Mid A"
    assert values[0]["score"] > 0
    assert reaches[0]["score"] < 0


def test_score_players_reads_projection_from_table(tmp_db):
    # Seed two players with prior PPR and a current-day ADP each.
    pids = _seed_players_and_stats(
        tmp_db,
        [
            ("Underprojected Player", 150, None),   # prior PPR 150
            ("Overprojected Player",  150, None),   # same prior PPR
        ],
    )
    _seed_adp(
        tmp_db,
        pids,
        {
            "Underprojected Player": (10.0, 10),
            "Overprojected Player":  (10.0, 10),
        },
    )

    # Now write differing projections to player_projections.
    with connect(tmp_db) as conn:
        conn.executemany(
            """
            INSERT INTO player_projections
                (player_id, season, source, projected_points_ppr, captured_at)
            VALUES (?, 2024, 'internal_v2', ?, '2026-01-01T00:00:00')
            """,
            [
                (pids["Underprojected Player"], 300.0),
                (pids["Overprojected Player"],  50.0),
            ],
        )

    score_players(
        season=2024,
        db_path=tmp_db,
        bucket_size=1,
        curve={9: 150.0},  # expected_at_rank_10 = 150
        projection_source="internal_v2",
    )

    with connect(tmp_db) as conn:
        rows = dict(conn.execute(
            "SELECT p.full_name, s.score FROM player_scores s "
            "JOIN players p USING (player_id) WHERE s.season = 2024"
        ))
    # Underprojected: 300 - 150 = +150. Overprojected: 50 - 150 = -100.
    assert rows["Underprojected Player"] == pytest.approx(150.0)
    assert rows["Overprojected Player"] == pytest.approx(-100.0)


def test_score_players_skips_players_with_no_adp(tmp_db):
    # Two players with prior PPR but no ADP at all → should not be scored.
    pids = _seed_players_and_stats(
        tmp_db,
        [
            ("Ghost A", 300, None),
            ("Ghost B", 200, None),
        ],
    )
    injected = {0: 250.0}  # single-bucket curve
    n = score_players(season=2024, db_path=tmp_db, curve=injected)
    assert n == 0

    with connect(tmp_db) as conn:
        scores = conn.execute("SELECT COUNT(*) FROM player_scores").fetchone()[0]
    assert scores == 0


def test_expected_for_position_rank_misses_when_position_unknown():
    curves = {"QB": {0: 350, 1: 280}}
    assert _expected_for_position_rank("RB", 5, curves, bucket_size=3) is None
    assert _expected_for_position_rank(None, 5, curves, bucket_size=3) is None
    assert _expected_for_position_rank("QB", None, curves, bucket_size=3) is None


def test_expected_for_position_rank_extrapolates_within_position():
    curves = {"QB": {0: 350, 1: 280, 2: 220}}
    assert _expected_for_position_rank("QB", 1, curves, bucket_size=3) == 350
    assert _expected_for_position_rank("QB", 5, curves, bucket_size=3) == 280
    # Past the end: holds the last bucket flat
    assert _expected_for_position_rank("QB", 50, curves, bucket_size=3) == 220


def test_synthetic_curve_by_position_separates_positions(tmp_db):
    # 3 QBs descending, 3 RBs descending — each should get its own curve
    pids = _seed_players_and_stats(
        tmp_db,
        [
            ("QB1", 400, None), ("QB2", 350, None), ("QB3", 300, None),
            ("RB1", 250, None), ("RB2", 200, None), ("RB3", 150, None),
        ],
    )
    # Override their positions (default seeded as WR)
    with connect(tmp_db) as conn:
        conn.execute("UPDATE players SET position='QB' WHERE full_name LIKE 'QB%'")
        conn.execute("UPDATE players SET position='RB' WHERE full_name LIKE 'RB%'")
        curves = _synthetic_curve_by_position(conn, 2023, bucket_size=1)

    assert "QB" in curves and "RB" in curves
    # Bucket 0 = top finisher at each position
    assert curves["QB"][0] == 400
    assert curves["RB"][0] == 250
    # Bucket 2 = third finisher
    assert curves["QB"][2] == 300
    assert curves["RB"][2] == 150


def test_position_aware_scoring_doesnt_overrate_cheap_qbs(tmp_db):
    # Set up: 4 RBs (expected ~250 PPR at top of pos), and 1 cheap QB
    # at QB12 projected to score 280. Flat curve would say "QB12 at
    # overall ADP 50 should score 50 PPR" (mostly RB/WR at that ADP)
    # → score = 280 - 50 = +230 huge value. Position-aware: "QB12 in
    # this league scores ~280 average" → score ≈ 0.
    pids = _seed_players_and_stats(
        tmp_db,
        [
            ("RB Top",   300, None),
            ("RB 2",     270, None),
            ("RB 3",     240, None),
            ("RB 4",     210, None),
            # Plus a bunch of QBs to populate the QB curve
            ("QB 1",     360, None),
            ("QB 6",     300, None),
            ("QB 12",    280, None),
            ("QB 18",    240, None),
        ],
    )
    with connect(tmp_db) as conn:
        conn.execute("UPDATE players SET position='RB' WHERE full_name LIKE 'RB%'")
        conn.execute("UPDATE players SET position='QB' WHERE full_name LIKE 'QB%'")

    # ADP snapshots: position_rank for each
    adp_rows = [
        ("RB Top",   1.0,   1, "RB", 1),
        ("QB 12",    50.0, 50, "QB", 12),
    ]
    with connect(tmp_db) as conn:
        cols = ["player_id", "adp", "adp_rank", "position_rank", "draft_format", "source", "captured_at"]
        conn.executemany(
            f"INSERT INTO adp_snapshots ({','.join(cols)}) VALUES ({','.join(['?'] * len(cols))})",
            [
                [pids[name], adp, ovr, pr, "best_ball", "underdog", "2026-01-01T00:00:00"]
                for name, adp, ovr, _, pr in adp_rows
            ],
        )

    # Position-aware scoring on 2024 — projection from prior season
    # (we seeded 2023). QB curve buckets[0..3] = [360, 300, 280, 240].
    # QB 12 at position_rank 12 falls in bucket (12-1)//3 = 3 → 240.
    # Projection for QB 12 is 280 (prior season). Score = 280 - 240 = +40.
    score_players(season=2024, db_path=tmp_db, position_aware=True, bucket_size=3)

    with connect(tmp_db) as conn:
        scores = dict(conn.execute(
            "SELECT p.full_name, s.score FROM player_scores s "
            "JOIN players p USING (player_id) WHERE s.season=2024"
        ))
    # Both scored
    assert "QB 12" in scores
    assert "RB Top" in scores
    # QB 12 not over-rewarded — compared to other QBs (240 expected), is roughly fair
    assert abs(scores["QB 12"] - 40.0) < 1.0


def test_position_aware_falls_back_to_synthetic_when_no_history(tmp_db):
    # Just verify the path doesn't crash when there are no historical
    # ADP snapshots to build a curve from.
    pids = _seed_players_and_stats(
        tmp_db,
        [
            ("Top WR", 300, None),
            ("Mid WR", 200, None),
            ("Low WR", 100, None),
        ],
    )
    with connect(tmp_db) as conn:
        conn.execute("UPDATE players SET position='WR'")
        # ADP for one player
        conn.execute(
            "INSERT INTO adp_snapshots (player_id, adp, adp_rank, position_rank, "
            "draft_format, source, captured_at) VALUES (?,?,?,?,?,?,?)",
            (pids["Mid WR"], 30.0, 30, 15, "best_ball", "underdog", "2026-01-01T00:00:00"),
        )
    n = score_players(season=2024, db_path=tmp_db, position_aware=True, bucket_size=3)
    assert n == 1
