import pytest

from database import connect, upsert_many
from recommend import recommend


def _seed(tmp_db, players, scores, picks_by_draft=None):
    """players: [(name, position, team)]
    scores: {name: score}
    picks_by_draft: {draft_name: [name, ...]}"""
    with connect(tmp_db) as conn:
        upsert_many(
            conn,
            "players",
            [
                {"full_name": n, "position": p, "team": t, "gsis_id": f"G_{i}"}
                for i, (n, p, t) in enumerate(players)
            ],
            conflict_cols=["gsis_id"],
        )
        pids = dict(conn.execute("SELECT full_name, player_id FROM players"))

        # player_scores under model_version='test_v1'
        score_rows = [
            (pids[name], 2026, "test_v1", score, None, "2026-05-01T00:00:00")
            for name, score in scores.items()
        ]
        conn.executemany(
            "INSERT INTO player_scores (player_id, season, model_version, score, rank, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            score_rows,
        )

        # Optionally add drafts + picks
        if picks_by_draft:
            for draft_name, names in picks_by_draft.items():
                conn.execute(
                    "INSERT INTO my_drafts (draft_name, season, my_slot) VALUES (?, ?, ?)",
                    (draft_name, 2026, 5),
                )
                draft_id = conn.execute(
                    "SELECT draft_id FROM my_drafts WHERE draft_name = ?",
                    (draft_name,),
                ).fetchone()["draft_id"]
                for i, name in enumerate(names, start=1):
                    conn.execute(
                        "INSERT INTO my_picks (draft_id, player_id, round, pick_overall) "
                        "VALUES (?, ?, ?, ?)",
                        (draft_id, pids[name], i, i),
                    )

        return pids


def test_recommend_returns_base_scores_when_no_portfolio(tmp_db):
    _seed(
        tmp_db,
        players=[("A", "WR", "DAL"), ("B", "RB", "BUF")],
        scores={"A": 50.0, "B": 30.0},
    )
    recs = recommend(2026, "test_v1", db_path=tmp_db)
    assert recs[0]["full_name"] == "A"
    assert recs[0]["adjusted_score"] == 50.0
    assert recs[0]["exposure_penalty"] == 0.0
    assert recs[0]["stack_bonus"] == 0.0


def test_recommend_applies_exposure_penalty(tmp_db):
    # A is in 2 of 2 drafts (100% exposure), B is in 0 of 2 (0%)
    _seed(
        tmp_db,
        players=[("A", "WR", "DAL"), ("B", "RB", "BUF"), ("C", "WR", "NYG")],
        scores={"A": 50.0, "B": 30.0, "C": 20.0},
        picks_by_draft={
            "Draft1": ["A"],
            "Draft2": ["A"],
        },
    )
    # 1.0 point per percent → A: 50 - 100 = -50, B: 30 - 0 = 30
    recs = recommend(
        2026, "test_v1", exposure_penalty_per_pct=1.0, db_path=tmp_db
    )
    by_name = {r["full_name"]: r for r in recs}
    assert by_name["A"]["adjusted_score"] == pytest.approx(-50.0)
    assert by_name["B"]["adjusted_score"] == 30.0
    # B should now rank ahead of A
    assert recs[0]["full_name"] == "B"


def test_recommend_applies_stack_bonus(tmp_db):
    # Roster a QB on DAL → DAL WRs get +10 stack
    _seed(
        tmp_db,
        players=[
            ("DAL QB", "QB", "DAL"),
            ("DAL WR", "WR", "DAL"),
            ("NYG WR", "WR", "NYG"),  # no stack, control
        ],
        scores={"DAL QB": 100.0, "DAL WR": 50.0, "NYG WR": 50.0},
        picks_by_draft={"D1": ["DAL QB"]},
    )
    recs = recommend(
        2026, "test_v1", stack_bonus=10.0, exposure_penalty_per_pct=0.0,
        db_path=tmp_db,
    )
    by_name = {r["full_name"]: r for r in recs}
    # DAL WR got the stack, NYG WR didn't
    assert by_name["DAL WR"]["adjusted_score"] == 60.0
    assert by_name["DAL WR"]["stack_bonus"] == 10.0
    assert by_name["NYG WR"]["adjusted_score"] == 50.0
    assert by_name["NYG WR"]["stack_bonus"] == 0.0


def test_recommend_excludes_already_picked_when_draft_id_set(tmp_db):
    pids = _seed(
        tmp_db,
        players=[("A", "WR", "DAL"), ("B", "RB", "BUF")],
        scores={"A": 50.0, "B": 30.0},
        picks_by_draft={"D1": ["A"]},
    )
    with connect(tmp_db) as conn:
        draft_id = conn.execute(
            "SELECT draft_id FROM my_drafts WHERE draft_name = 'D1'"
        ).fetchone()["draft_id"]

    # Without draft_id, both are returned
    all_recs = recommend(2026, "test_v1", db_path=tmp_db)
    assert {r["full_name"] for r in all_recs} == {"A", "B"}

    # With draft_id, A is excluded (already picked in this draft)
    in_draft = recommend(2026, "test_v1", draft_id=draft_id, db_path=tmp_db)
    assert {r["full_name"] for r in in_draft} == {"B"}


def test_recommend_returns_empty_when_no_scores(tmp_db):
    assert recommend(2026, "test_v1", db_path=tmp_db) == []
