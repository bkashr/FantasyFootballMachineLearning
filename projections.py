"""Phase-2 player projections (internal_v2).

Replaces the baseline-v1 'projection = last year's PPR' with a real
per-player projection. Components, in order of contribution:

1. Multi-year weighted PPR baseline. Recent seasons weighted more.
2. Per-game rate, not raw totals — so productive-but-injured players
   aren't punished by games missed.
3. Position-specific age curve adjustment.
4. Injury-risk discount based on historical games missed.

Limitations of this version (v2):

- Injury detail is "games missed" only — we don't yet have ACL vs
  Achilles vs ankle granularity. The ROADMAP.md flags this for v3.
- No coaching-change / team-context adjustment yet.
- Rookies with no prior NFL data can't be projected from this model —
  they're returned in the 'no_history' bucket for manual handling.

Backtest plan: project a past season using only data from seasons
before it, compare to actuals. The `backtest` function does this.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
from dataclasses import dataclass

from database import DB_PATH, connect, init_db

log = logging.getLogger(__name__)

DEFAULT_SOURCE = "internal_v2"
EXPECTED_GAMES = 17  # NFL regular season since 2021

# Weight schedule: years_back -> weight. Normalized at use time for
# players with fewer years of data. Most-recent year carries the most
# signal but we don't ignore older seasons entirely — they smooth out
# one-year flukes.
DEFAULT_YEAR_WEIGHTS = {1: 0.55, 2: 0.30, 3: 0.15}

# Position-specific age adjustment. Flat 1.0 at and below peak; linear
# decline above. The constants are rough approximations of published
# fantasy age curves (RBs decline earliest, QBs latest) and will be
# refined empirically once we have a backtest loop running.
AGE_CURVES = {
    "QB": {"peak_age": 33, "decline_per_year": 0.04, "floor": 0.70},
    "RB": {"peak_age": 27, "decline_per_year": 0.08, "floor": 0.40},
    "WR": {"peak_age": 29, "decline_per_year": 0.04, "floor": 0.60},
    "TE": {"peak_age": 29, "decline_per_year": 0.05, "floor": 0.55},
}

# Injury discount: how much do we knock down a player whose recent
# games-missed rate is high? A player who misses 30% of recent games
# loses 15% off the projection (0.30 * 0.5). Floored at 0.7 so we don't
# zero out a player who happened to have a bad injury year — recovery
# is real.
INJURY_DISCOUNT_FACTOR = 0.5
INJURY_DISCOUNT_FLOOR = 0.70


@dataclass
class Projection:
    player_id: int
    season: int
    base_ppr: float       # multi-year weighted, per-game extrapolated
    age_factor: float
    injury_factor: float
    projected_points_ppr: float
    n_prior_seasons: int

    def to_db_row(self, source: str, captured_at: str) -> dict:
        return {
            "player_id": self.player_id,
            "season": self.season,
            "source": source,
            "projected_points_ppr": self.projected_points_ppr,
            # We don't yet split standard from PPR. Same value for now;
            # can split if/when we generate a non-PPR projection.
            "projected_points": self.projected_points_ppr * 0.75,
            "pass_yards": None,
            "pass_tds": None,
            "rush_yards": None,
            "rush_tds": None,
            "receptions": None,
            "rec_yards": None,
            "rec_tds": None,
            "captured_at": captured_at,
        }


def _age_at_season(birth_date_iso: str | None, season: int) -> int | None:
    """Age the player will turn during this NFL season. Approximated as
    season_year - birth_year — close enough for an age-curve multiplier."""
    if not birth_date_iso:
        return None
    try:
        birth_year = int(birth_date_iso.split("-")[0])
    except (ValueError, IndexError):
        return None
    return season - birth_year


def _age_factor(age: int | None, position: str | None) -> float:
    """Multiplier on the base projection from the position-specific age
    curve. Players at or below peak get 1.0. Above peak, decline linearly
    until hitting the position's floor."""
    if age is None or position not in AGE_CURVES:
        return 1.0
    curve = AGE_CURVES[position]
    if age <= curve["peak_age"]:
        return 1.0
    over = age - curve["peak_age"]
    factor = 1.0 - over * curve["decline_per_year"]
    return max(factor, curve["floor"])


def _injury_factor(games_missed: int, possible_games: int) -> float:
    """Discount based on games missed over the recent window. A clean
    bill of health = 1.0. Heavy injury history floors at INJURY_DISCOUNT_FLOOR."""
    if possible_games <= 0:
        return 1.0
    miss_rate = games_missed / possible_games
    factor = 1.0 - miss_rate * INJURY_DISCOUNT_FACTOR
    return max(factor, INJURY_DISCOUNT_FLOOR)


def _weighted_per_game_baseline(
    season_stats: list[dict],
    target_season: int,
    weights: dict[int, float] = DEFAULT_YEAR_WEIGHTS,
) -> tuple[float, int]:
    """Given a list of season stat rows {season, fantasy_points_ppr,
    games_played}, return (per_game_ppr_weighted, n_seasons_used).

    Seasons with games_played <= 0 are dropped — we'd divide by zero
    otherwise, and a zero-game season carries no signal."""
    contributions: list[tuple[float, float]] = []
    for row in season_stats:
        ppr = row.get("fantasy_points_ppr")
        games = row.get("games_played")
        season = row.get("season")
        if ppr is None or games is None or games <= 0 or season is None:
            continue
        years_back = target_season - season
        if years_back not in weights:
            continue
        per_game = ppr / games
        contributions.append((per_game, weights[years_back]))

    if not contributions:
        return 0.0, 0

    total_weight = sum(w for _, w in contributions)
    weighted = sum(pg * w for pg, w in contributions) / total_weight
    return weighted, len(contributions)


def project_player(
    conn,
    player_id: int,
    target_season: int,
    expected_games: int = EXPECTED_GAMES,
    weights: dict[int, float] = DEFAULT_YEAR_WEIGHTS,
) -> Projection | None:
    """Build a single player's projection. Returns None when there's
    no prior-season data to project from (typically rookies)."""
    seasons_window = [target_season - yb for yb in weights.keys()]
    placeholders = ",".join("?" * len(seasons_window))
    rows = list(conn.execute(
        f"""
        SELECT season, fantasy_points_ppr, games_played
        FROM player_season_stats
        WHERE player_id = ? AND season IN ({placeholders})
        """,
        (player_id, *seasons_window),
    ))
    season_stats = [dict(r) for r in rows]
    per_game, n_seasons = _weighted_per_game_baseline(
        season_stats, target_season, weights
    )
    if n_seasons == 0:
        return None

    # Injury discount uses the same window — games missed vs games
    # possible across the seasons we actually have data for.
    games_played = sum(
        (r["games_played"] or 0) for r in season_stats
        if r.get("season") is not None
        and (target_season - r["season"]) in weights
    )
    possible_games = expected_games * n_seasons
    games_missed = max(possible_games - games_played, 0)
    injury_f = _injury_factor(games_missed, possible_games)

    # Age factor needs birth_date + position from the players table
    player_row = conn.execute(
        "SELECT birth_date, position FROM players WHERE player_id = ?",
        (player_id,),
    ).fetchone()
    age = _age_at_season(
        player_row["birth_date"] if player_row else None, target_season
    )
    age_f = _age_factor(age, player_row["position"] if player_row else None)

    base = per_game * expected_games
    projection = base * age_f * injury_f
    return Projection(
        player_id=player_id,
        season=target_season,
        base_ppr=base,
        age_factor=age_f,
        injury_factor=injury_f,
        projected_points_ppr=projection,
        n_prior_seasons=n_seasons,
    )


def project_all(
    target_season: int,
    source: str = DEFAULT_SOURCE,
    db_path: str = DB_PATH,
    expected_games: int = EXPECTED_GAMES,
    weights: dict[int, float] = DEFAULT_YEAR_WEIGHTS,
) -> dict:
    """Project every player with prior-season data. Writes results to
    player_projections (upserts on player_id/season/source). Returns
    a summary dict {projected, no_history}."""
    init_db(db_path)
    captured_at = dt.datetime.utcnow().isoformat(timespec="seconds")

    with connect(db_path) as conn:
        # Candidate pool: any player with at least one season of stats
        # in our weight window.
        seasons_window = [target_season - yb for yb in weights.keys()]
        placeholders = ",".join("?" * len(seasons_window))
        candidate_ids = [
            r["player_id"]
            for r in conn.execute(
                f"""
                SELECT DISTINCT player_id
                FROM player_season_stats
                WHERE season IN ({placeholders})
                  AND fantasy_points_ppr IS NOT NULL
                """,
                seasons_window,
            )
        ]

        projections: list[Projection] = []
        for pid in candidate_ids:
            proj = project_player(
                conn, pid, target_season, expected_games, weights
            )
            if proj is not None:
                projections.append(proj)

        rows = [p.to_db_row(source, captured_at) for p in projections]
        if rows:
            cols = list(rows[0].keys())
            conn.executemany(
                f"""
                INSERT INTO player_projections ({','.join(cols)})
                VALUES ({','.join(['?'] * len(cols))})
                ON CONFLICT(player_id, season, source) DO UPDATE SET
                  projected_points_ppr = excluded.projected_points_ppr,
                  projected_points = excluded.projected_points,
                  captured_at = excluded.captured_at
                """,
                [[r[c] for c in cols] for r in rows],
            )

    log.info(
        "Projected %d players for season=%d source=%s",
        len(projections),
        target_season,
        source,
    )
    return {"projected": len(projections), "candidates": len(candidate_ids)}


def backtest(
    target_season: int,
    db_path: str = DB_PATH,
    weights: dict[int, float] = DEFAULT_YEAR_WEIGHTS,
    min_actual_games: int = 4,
) -> dict:
    """Project target_season using only earlier-season data, then compare
    to the season's actual PPR for players who played enough games.

    Returns summary stats: mean absolute error, mean error (bias),
    correlation, and the row count compared. Players with fewer than
    min_actual_games are excluded — they're noisy on actuals."""
    init_db(db_path)
    backtest_source = f"backtest_{target_season}"
    project_all(
        target_season=target_season,
        source=backtest_source,
        db_path=db_path,
        weights=weights,
    )

    with connect(db_path) as conn:
        rows = list(conn.execute(
            """
            SELECT
                p.full_name,
                p.position,
                pr.projected_points_ppr,
                s.fantasy_points_ppr AS actual_ppr,
                s.games_played
            FROM player_projections pr
            JOIN player_season_stats s
              ON s.player_id = pr.player_id AND s.season = pr.season
            JOIN players p ON p.player_id = pr.player_id
            WHERE pr.source = ? AND pr.season = ?
              AND s.fantasy_points_ppr IS NOT NULL
              AND s.games_played >= ?
            """,
            (backtest_source, target_season, min_actual_games),
        ))

    if not rows:
        log.warning("No backtest rows for season=%d", target_season)
        return {"n": 0}

    diffs = [r["projected_points_ppr"] - r["actual_ppr"] for r in rows]
    abs_diffs = [abs(d) for d in diffs]
    mae = sum(abs_diffs) / len(abs_diffs)
    bias = sum(diffs) / len(diffs)

    # Pearson correlation
    n = len(rows)
    proj = [r["projected_points_ppr"] for r in rows]
    actual = [r["actual_ppr"] for r in rows]
    proj_mean = sum(proj) / n
    actual_mean = sum(actual) / n
    cov = sum((p - proj_mean) * (a - actual_mean) for p, a in zip(proj, actual)) / n
    proj_var = sum((p - proj_mean) ** 2 for p in proj) / n
    actual_var = sum((a - actual_mean) ** 2 for a in actual) / n
    correlation = (
        cov / math.sqrt(proj_var * actual_var)
        if proj_var > 0 and actual_var > 0
        else 0.0
    )

    return {
        "n": n,
        "mae": mae,
        "bias": bias,
        "correlation": correlation,
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument(
        "--backtest",
        action="store_true",
        help="Project the season then compare to actuals.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.backtest:
        result = backtest(args.season, db_path=args.db)
        if result["n"] == 0:
            print("No data to backtest")
            return
        print(
            f"Backtest season={args.season}: n={result['n']}  "
            f"MAE={result['mae']:.1f}  bias={result['bias']:+.1f}  "
            f"corr={result['correlation']:.3f}"
        )
        # Show top hits and misses by absolute error
        rows = result["rows"]
        worst = sorted(
            rows,
            key=lambda r: abs(r["projected_points_ppr"] - r["actual_ppr"]),
            reverse=True,
        )[:10]
        print("\nLargest projection errors:")
        for r in worst:
            diff = r["projected_points_ppr"] - r["actual_ppr"]
            print(
                f"  {r['full_name']:28s} {r['position']:3s}  "
                f"proj {r['projected_points_ppr']:5.0f}  "
                f"actual {r['actual_ppr']:5.0f}  diff {diff:+5.0f}"
            )
    else:
        summary = project_all(
            target_season=args.season, source=args.source, db_path=args.db
        )
        print(f"Projected {summary['projected']} of {summary['candidates']} candidates")


if __name__ == "__main__":
    main()
