"""First-pass player scoring.

Baseline model (`baseline_v1`):
  projection  = player's PPR points from the most recent prior season
  expected    = mean PPR points historically scored by players drafted
                at a similar ADP slot (from the value curve)
  score       = projection - expected_at_adp

Positive score = good value at current ADP, negative = reach.

This is intentionally crude. The README's longer-term plan adds projected
offense/defense ratings, injury context, and an ML layer trained on
historical drafts. Slot those in as `baseline_v2`, etc.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math

from database import DB_PATH, connect, init_db

log = logging.getLogger(__name__)

DEFAULT_MODEL_VERSION = "baseline_v1"


def _latest_adp(conn, draft_format: str = "best_ball"):
    """Return {player_id: (adp, adp_rank)} for the freshest snapshot of
    each player in the given format."""
    rows = conn.execute(
        """
        SELECT a.player_id, a.adp, a.adp_rank
        FROM adp_snapshots a
        JOIN (
            SELECT player_id, MAX(captured_at) AS ts
            FROM adp_snapshots
            WHERE draft_format = ?
            GROUP BY player_id
        ) latest ON latest.player_id = a.player_id AND latest.ts = a.captured_at
        WHERE a.draft_format = ?
        """,
        (draft_format, draft_format),
    ).fetchall()
    return {r["player_id"]: (r["adp"], r["adp_rank"]) for r in rows}


def _projection_from_prior_season(conn, season: int) -> dict[int, float]:
    """Use the prior season's PPR as the projection."""
    rows = conn.execute(
        """
        SELECT player_id, fantasy_points_ppr
        FROM player_season_stats
        WHERE season = ? AND fantasy_points_ppr IS NOT NULL
        """,
        (season - 1,),
    ).fetchall()
    return {r["player_id"]: float(r["fantasy_points_ppr"]) for r in rows}


def _projection_from_table(conn, season: int, source: str) -> dict[int, float]:
    """Read projections from player_projections. Source tag picks the
    model — 'internal_v2' is the Phase-2 projection."""
    rows = conn.execute(
        """
        SELECT player_id, projected_points_ppr
        FROM player_projections
        WHERE season = ? AND source = ? AND projected_points_ppr IS NOT NULL
        """,
        (season, source),
    ).fetchall()
    return {r["player_id"]: float(r["projected_points_ppr"]) for r in rows}


def build_value_curve(
    conn,
    seasons: list[int],
    draft_format: str = "best_ball",
    bucket_size: int = 6,
) -> dict[int, float]:
    """Return {adp_bucket: expected_ppr_points}.

    Joins each season's stats with ADP snapshots captured BEFORE that
    season's results — i.e. snapshots from the same calendar year, taken
    pre-Sept. This avoids leakage where a current-day snapshot gets
    matched to historical PPR and trivially predicts it.

    Bucket size groups adjacent ADP slots so a single noisy outcome
    doesn't dominate. With bucket_size=6 (≈ half a round in a 12-team
    draft), bucket 0 covers ADP 1-6, bucket 1 covers 7-12, etc.
    """
    if not seasons:
        return {}

    bucket_totals: dict[int, list[float]] = {}
    for season in seasons:
        # Snapshots captured during the offseason / early-season for `season`.
        # We bracket on Jan-Sep of that year to exclude in-season redrafts
        # and any post-season cleanup.
        lo = f"{season}-01-01T00:00:00"
        hi = f"{season}-09-15T00:00:00"
        rows = conn.execute(
            """
            SELECT a.adp_rank, s.fantasy_points_ppr
            FROM adp_snapshots a
            JOIN player_season_stats s
              ON s.player_id = a.player_id AND s.season = ?
            WHERE a.draft_format = ?
              AND a.adp_rank IS NOT NULL
              AND s.fantasy_points_ppr IS NOT NULL
              AND a.captured_at >= ?
              AND a.captured_at < ?
            """,
            (season, draft_format, lo, hi),
        ).fetchall()
        for r in rows:
            bucket = (r["adp_rank"] - 1) // bucket_size
            bucket_totals.setdefault(bucket, []).append(float(r["fantasy_points_ppr"]))

    return {b: sum(v) / len(v) for b, v in bucket_totals.items() if v}


def _synthetic_curve_from_ppr_rank(
    conn, season: int, bucket_size: int = 6
) -> dict[int, float]:
    """Stand-in for the value curve before we have historical ADP. Ranks
    last season's PPR finishers and buckets them — answers 'what does
    the Nth-best fantasy player typically score?'"""
    rows = conn.execute(
        """
        SELECT fantasy_points_ppr
        FROM player_season_stats
        WHERE season = ? AND fantasy_points_ppr IS NOT NULL
        ORDER BY fantasy_points_ppr DESC
        """,
        (season,),
    ).fetchall()
    buckets: dict[int, list[float]] = {}
    for i, r in enumerate(rows):
        b = i // bucket_size
        buckets.setdefault(b, []).append(float(r["fantasy_points_ppr"]))
    return {b: sum(v) / len(v) for b, v in buckets.items()}


def _expected_for_rank(rank: int | None, curve: dict[int, float], bucket_size: int) -> float | None:
    if rank is None or not curve:
        return None
    bucket = (rank - 1) // bucket_size
    if bucket in curve:
        return curve[bucket]
    # extrapolate by holding the last bucket flat
    max_bucket = max(curve)
    return curve[max_bucket] if bucket > max_bucket else None


def score_players(
    season: int,
    model_version: str = DEFAULT_MODEL_VERSION,
    draft_format: str = "best_ball",
    bucket_size: int = 6,
    db_path: str = DB_PATH,
    curve: dict[int, float] | None = None,
    projection_source: str | None = None,
) -> int:
    """Compute and store scores for `season`. Returns rows written.

    `curve` lets callers inject a pre-built value curve (rank-bucket ->
    expected PPR). Useful for tests and for plugging in a better curve
    once we have proper historical ADP. If None, we try a real curve
    from prior-season ADP and fall back to a synthetic curve derived
    from prior-season PPR finish.

    `projection_source`: if set, pull projections from
    player_projections WHERE source = projection_source (e.g.
    'internal_v2'). If None, fall back to using the prior season's
    PPR directly — that's the baseline_v1 behavior.
    """
    init_db(db_path)
    with connect(db_path) as conn:
        if projection_source:
            projections = _projection_from_table(conn, season, projection_source)
            log.info(
                "Using %d projections from source=%s",
                len(projections),
                projection_source,
            )
        else:
            projections = _projection_from_prior_season(conn, season)
        adp = _latest_adp(conn, draft_format=draft_format)

        if curve is None:
            curve = build_value_curve(
                conn,
                seasons=[season - 1],
                draft_format=draft_format,
                bucket_size=bucket_size,
            )
            if not curve:
                log.info(
                    "No historical ADP yet — using synthetic curve from prior PPR ranks"
                )
                curve = _synthetic_curve_from_ppr_rank(
                    conn, season - 1, bucket_size=bucket_size
                )

        computed_at = dt.datetime.utcnow().isoformat(timespec="seconds")
        rows = []
        skipped_no_adp = 0
        for player_id, proj in projections.items():
            adp_rank = adp.get(player_id, (None, None))[1]
            expected = _expected_for_rank(adp_rank, curve, bucket_size)
            if expected is None:
                # No ADP or off the curve — score = projection - expected is
                # undefined. Skip rather than reporting a misleading number.
                skipped_no_adp += 1
                continue
            rows.append(
                {
                    "player_id": player_id,
                    "season": season,
                    "model_version": model_version,
                    "score": proj - expected,
                    "rank": None,
                    "computed_at": computed_at,
                }
            )

        rows.sort(key=lambda r: r["score"], reverse=True)
        for i, r in enumerate(rows, start=1):
            r["rank"] = i
        if skipped_no_adp:
            log.info(
                "Skipped %d players with no current ADP (can't compute value)",
                skipped_no_adp,
            )

        if rows:
            cols = ["player_id", "season", "model_version", "score", "rank", "computed_at"]
            conn.executemany(
                f"""
                INSERT INTO player_scores ({','.join(cols)})
                VALUES ({','.join(['?'] * len(cols))})
                ON CONFLICT(player_id, season, model_version) DO UPDATE SET
                    score = excluded.score,
                    rank = excluded.rank,
                    computed_at = excluded.computed_at
                """,
                [[r[c] for c in cols] for r in rows],
            )
    log.info("Scored %d players for season=%d model=%s", len(rows), season, model_version)
    return len(rows)


_RANKING_SQL = """
    WITH latest AS (
        SELECT player_id, adp, adp_rank
        FROM adp_snapshots
        WHERE (player_id, captured_at) IN (
            SELECT player_id, MAX(captured_at) FROM adp_snapshots GROUP BY player_id
        )
    )
    SELECT p.full_name, p.position, p.team,
           s.score, s.rank,
           latest.adp AS latest_adp,
           latest.adp_rank AS latest_adp_rank
    FROM player_scores s
    JOIN players p USING (player_id)
    JOIN latest USING (player_id)
    WHERE s.season = ? AND s.model_version = ?
    ORDER BY s.score {direction}
    LIMIT ?
"""


def top_values(
    season: int,
    model_version: str = DEFAULT_MODEL_VERSION,
    limit: int = 20,
    db_path: str = DB_PATH,
):
    """Players whose score exceeds expectation at their ADP — best buys."""
    with connect(db_path) as conn:
        return conn.execute(
            _RANKING_SQL.format(direction="DESC"),
            (season, model_version, limit),
        ).fetchall()


def top_reaches(
    season: int,
    model_version: str = DEFAULT_MODEL_VERSION,
    limit: int = 20,
    db_path: str = DB_PATH,
):
    """Players whose score is well below expectation at their ADP."""
    with connect(db_path) as conn:
        return conn.execute(
            _RANKING_SQL.format(direction="ASC"),
            (season, model_version, limit),
        ).fetchall()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", type=int, required=True, help="Season to score (uses prior season's stats as projection)")
    p.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    p.add_argument("--draft-format", default="best_ball")
    p.add_argument(
        "--projection-source",
        default=None,
        help="Read projections from player_projections WHERE source=<this>. "
        "If omitted, falls back to prior-season PPR (baseline_v1).",
    )
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--show", type=int, default=10, help="Print top N values and reaches after scoring")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    score_players(
        season=args.season,
        model_version=args.model_version,
        draft_format=args.draft_format,
        db_path=args.db,
        projection_source=args.projection_source,
    )

    if args.show:
        print(f"\nTop {args.show} VALUES:")
        for r in top_values(args.season, args.model_version, args.show, args.db):
            print(f"  {r['full_name']:30s} {r['position']:3s} {r['team'] or '':3s}  ADP {r['latest_adp']:>5.1f}  score {r['score']:+7.1f}")
        print(f"\nTop {args.show} REACHES:")
        for r in top_reaches(args.season, args.model_version, args.show, args.db):
            print(f"  {r['full_name']:30s} {r['position']:3s} {r['team'] or '':3s}  ADP {r['latest_adp']:>5.1f}  score {r['score']:+7.1f}")


if __name__ == "__main__":
    main()
