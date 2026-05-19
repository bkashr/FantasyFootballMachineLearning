"""Phase 6 (v1): personalized draft recommendations.

Composes the three layers from ROADMAP.md into a single 'who should I
take next?' view:

  projection ── (already in player_projections)
       │
       ▼
  value      ── score_players writes player_scores (marketwide)
       │
       ▼
  recommend  ── this module: layer in YOUR portfolio
              - exposure penalty: a player's value to YOU drops as
                your exposure rises across drafts
              - stack bonus: lift pass-catchers on teams where you've
                already rostered the QB

The marketwide value score stays in player_scores (good for anyone
looking at the same DB). The personalized adjustment is computed at
read time so it stays current without recomputing.
"""

from __future__ import annotations

import argparse
import logging

from database import DB_PATH, connect
from portfolio import draft_count, player_exposure

log = logging.getLogger(__name__)

PASS_CATCHER_POSITIONS = ("WR", "TE")


def _qb_stacks(conn, season: int, draft_id: int | None) -> set[str]:
    """Return the set of NFL teams where I already roster the QB.
    If draft_id is provided, only that draft's QBs count (in-draft
    mode). Otherwise, any team I've rostered a QB on across the
    season's portfolio."""
    sql = """
        SELECT DISTINCT p.team
        FROM my_picks mp
        JOIN my_drafts d USING (draft_id)
        JOIN players p USING (player_id)
        WHERE p.position = 'QB' AND d.season = ? AND p.team IS NOT NULL
    """
    params: list = [season]
    if draft_id is not None:
        sql += " AND mp.draft_id = ?"
        params.append(draft_id)
    return {r["team"] for r in conn.execute(sql, params)}


def _already_rostered(conn, draft_id: int) -> set[int]:
    """Player IDs I've already picked in this specific draft."""
    return {
        r["player_id"]
        for r in conn.execute(
            "SELECT player_id FROM my_picks WHERE draft_id = ?",
            (draft_id,),
        )
    }


def recommend(
    season: int,
    model_version: str,
    draft_id: int | None = None,
    exposure_penalty_per_pct: float = 0.5,
    stack_bonus: float = 10.0,
    db_path: str = DB_PATH,
    limit: int = 30,
) -> list[dict]:
    """Return personalized recommendations.

    Args:
      season: scoring season (e.g. 2026)
      model_version: which player_scores row to read ('position_v1',
                     'baseline_v2', etc)
      draft_id: if set, exclude already-picked players and use
                this draft's QB roster for stack bonuses (in-draft mode);
                otherwise use the season-wide portfolio
      exposure_penalty_per_pct: subtract this many points per 1% of
                exposure (default 0.5 → 20% exposure = -10 points)
      stack_bonus: PPR points added to pass-catchers on teams I roster
                a QB on (default 10)
    """
    with connect(db_path) as conn:
        # Marketwide value scores
        base_rows = list(conn.execute(
            """
            WITH latest_adp AS (
                SELECT player_id, adp, position_rank
                FROM adp_snapshots
                WHERE (player_id, captured_at) IN (
                    SELECT player_id, MAX(captured_at) FROM adp_snapshots
                    GROUP BY player_id
                )
            )
            SELECT
                p.player_id, p.full_name, p.position, p.team,
                s.score AS base_score,
                la.adp, la.position_rank
            FROM player_scores s
            JOIN players p USING (player_id)
            LEFT JOIN latest_adp la USING (player_id)
            WHERE s.season = ? AND s.model_version = ?
            """,
            (season, model_version),
        ))
        if not base_rows:
            log.warning(
                "No player_scores rows for season=%d model=%s — run scoring.py first",
                season, model_version,
            )
            return []

        # Exposure
        exposure = {
            e["player_id"]: e["pct_of_drafts"]
            for e in player_exposure(season, db_path=db_path)
        }
        # Stacks
        stacked_teams = _qb_stacks(conn, season, draft_id)
        already = _already_rostered(conn, draft_id) if draft_id else set()

    results: list[dict] = []
    for r in base_rows:
        if r["player_id"] in already:
            continue
        base = float(r["base_score"]) if r["base_score"] is not None else 0.0
        exp_pct = exposure.get(r["player_id"], 0.0)
        exp_penalty = exp_pct * exposure_penalty_per_pct
        stack = stack_bonus if (
            r["position"] in PASS_CATCHER_POSITIONS
            and r["team"] in stacked_teams
        ) else 0.0
        adjusted = base - exp_penalty + stack
        results.append({
            "player_id": r["player_id"],
            "full_name": r["full_name"],
            "position": r["position"],
            "team": r["team"],
            "adp": r["adp"],
            "base_score": base,
            "exposure_pct": exp_pct,
            "exposure_penalty": exp_penalty,
            "stack_bonus": stack,
            "adjusted_score": adjusted,
        })

    results.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return results[:limit]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--model-version", default="position_v1")
    p.add_argument(
        "--draft-id", type=int, default=None,
        help="Specific draft to recommend FOR (excludes already-picked players, "
        "uses that draft's QBs for stack bonuses). Without it, season-wide.",
    )
    p.add_argument("--exposure-penalty-per-pct", type=float, default=0.5)
    p.add_argument("--stack-bonus", type=float, default=10.0)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    n_drafts = draft_count(args.season, db_path=args.db)
    print(
        f"Season {args.season}: {n_drafts} drafts in portfolio  "
        f"(model={args.model_version}"
        + (f", draft_id={args.draft_id}" if args.draft_id else "")
        + ")"
    )
    print()
    print(f"  {'Player':28s} {'Pos':3s} {'Tm':3s} {'ADP':>5s}  "
          f"{'base':>7s}  {'exp%':>5s} {'-pen':>5s} {'+stk':>4s}  {'adj':>7s}")
    for r in recommend(
        season=args.season,
        model_version=args.model_version,
        draft_id=args.draft_id,
        exposure_penalty_per_pct=args.exposure_penalty_per_pct,
        stack_bonus=args.stack_bonus,
        db_path=args.db,
        limit=args.limit,
    ):
        adp = f"{r['adp']:.1f}" if r["adp"] is not None else "-"
        print(
            f"  {r['full_name'][:28]:28s} {r['position'] or '':3s} "
            f"{r['team'] or '':3s} {adp:>5s}  "
            f"{r['base_score']:+7.1f}  {r['exposure_pct']:5.1f} "
            f"{-r['exposure_penalty']:+5.1f} {r['stack_bonus']:+4.0f}  "
            f"{r['adjusted_score']:+7.1f}"
        )


if __name__ == "__main__":
    main()
