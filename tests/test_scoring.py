from database import connect, upsert_many
from scoring import (
    _expected_for_rank,
    _synthetic_curve_from_ppr_rank,
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
