"""Pull ADP from Underdog Fantasy and append snapshots.

Endpoint is undocumented — this module is structured around a pluggable
`fetch_adp` callable so the real HTTP call can be slotted in after the
recon described in PLANNING.md. Until then, callers can pass a fixture
file path or a stub function.

Crosswalk: incoming players are matched to our internal player_id by
underdog_id first, falling back to case-insensitive name match. Misses
are logged but don't fail the run — they're the cleanup work the
PLANNING.md flags as inevitable for new rookies and name collisions."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import re
from pathlib import Path
from typing import Callable, Iterable

from database import DB_PATH, connect, init_db

log = logging.getLogger(__name__)

# Underdog's CSV download endpoint. Path is
# /rankings/download/{slate_id}/{?}/{contest_style_id}?product=fantasy&...
# Behind Cloudflare; easier to download via the web "Download" button
# and feed the file through --from-underdog-csv than to replay the HTTP
# call from Python.
UNDERDOG_RANKINGS_DOWNLOAD = (
    "https://app.underdogsports.com/rankings/download/{slate_id}/{x}/{contest_style_id}"
)

# Map Underdog's full team names to nflverse's 3-letter abbreviations
# so the crosswalk lines up across sources. nflverse uses the standard
# NFL.com abbrs: LAR / LAC / LV / WAS / JAX, etc.
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


def fetch_adp_from_underdog(draft_format: str = "best_ball") -> list[dict]:
    """Live HTTP fetcher. Underdog's rankings page is behind Cloudflare
    bot protection and uses a 10-min JWT, so direct replay from Python
    is fragile. Use the CSV download path instead — see
    `fetch_adp_from_underdog_csv`."""
    raise NotImplementedError(
        "Live HTTP fetch is fragile due to Cloudflare + 10-min JWT. "
        "Download the rankings CSV from Underdog and use "
        "fetch_adp_from_underdog_csv() instead."
    )


def fetch_adp_from_file(path: str | Path) -> list[dict]:
    """Test/dev fetcher. Reads a JSON file of the shape produced by
    `normalize_adp_response` so we can exercise the pipeline before the
    real endpoint is known."""
    return json.loads(Path(path).read_text())


def fetch_adp_from_underdog_csv(
    path: str | Path,
    draft_format: str = "best_ball",
) -> list[dict]:
    """Parse Underdog's rankings CSV download.

    Expected header (as of May 2026):
        id, firstName, lastName, adp, projectedPoints, salary,
        positionRank, slotName, teamName, lineupStatus, byeWeek

    Returns normalized records with both ADP and projection fields so
    the same pass can populate adp_snapshots and player_projections."""
    required = {"id", "firstName", "lastName", "adp", "slotName", "teamName"}
    out: list[dict] = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Underdog CSV missing expected columns: {sorted(missing)}. "
                f"Got: {reader.fieldnames}"
            )
        # rank from the CSV's natural sort order (rows are returned
        # ordered by ADP ascending — Bijan at 1.5 first, Gibbs at 1.6
        # second, etc.)
        for rank, row in enumerate(reader, start=1):
            adp_raw = row.get("adp")
            if not adp_raw:
                continue
            try:
                adp = float(adp_raw)
            except ValueError:
                continue
            team_full = (row.get("teamName") or "").strip()
            out.append(
                {
                    "underdog_id": (row.get("id") or "").strip() or None,
                    "full_name": f"{row.get('firstName','').strip()} "
                                 f"{row.get('lastName','').strip()}".strip(),
                    "position": (row.get("slotName") or "").strip() or None,
                    "team": TEAM_NAME_TO_ABBR.get(team_full, team_full),
                    "adp": adp,
                    "adp_rank": rank,
                    "draft_format": draft_format,
                    # Bonus projection field — ingest_adp ignores it,
                    # ingest_underdog_projections picks it up.
                    "projected_points_ppr": _maybe_float(
                        row.get("projectedPoints")
                    ),
                    "position_rank": (row.get("positionRank") or "").strip()
                    or None,
                    "bye_week": _maybe_int(row.get("byeWeek")),
                }
            )
    return out


def _maybe_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_position_rank(s: str | None) -> int | None:
    """Extract the integer from a label like 'RB15' or 'WR1'."""
    if not s:
        return None
    digits = "".join(c for c in s if c.isdigit())
    return int(digits) if digits else None


def _maybe_int(s: str | None) -> int | None:
    if s is None or s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def ingest_underdog_projections(
    records: list[dict],
    source: str = "underdog",
    season: int | None = None,
    db_path: str = DB_PATH,
) -> int:
    """Write Underdog's projectedPoints column into player_projections.
    Same crosswalk as ingest_adp — match by underdog_id, then by
    (name, position), then name alone.

    `season` defaults to the next NFL season relative to today; pass
    explicitly if the CSV is for a different one."""
    init_db(db_path)
    if season is None:
        today = dt.date.today()
        season = today.year if today.month >= 3 else today.year - 1

    captured_at = dt.datetime.utcnow().isoformat(timespec="seconds")

    with connect(db_path) as conn:
        by_uid, by_name_pos, by_name = _build_lookups(conn)

        rows: list[dict] = []
        unmatched = 0
        for rec in records:
            proj = rec.get("projected_points_ppr")
            if proj is None:
                continue
            pid = _resolve_player(rec, by_uid, by_name_pos, by_name)
            if pid is None:
                unmatched += 1
                continue
            rows.append(
                {
                    "player_id": pid,
                    "season": season,
                    "source": source,
                    "projected_points_ppr": float(proj),
                    "projected_points": float(proj) * 0.75,
                    "captured_at": captured_at,
                }
            )

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
        "Wrote %d Underdog projections (season=%d, source=%s); %d unmatched",
        len(rows),
        season,
        source,
        unmatched,
    )
    return len(rows)


_4FOR4_DATE_COL_RE = re.compile(r"^ADP on (.+)$")
_FILENAME_DATE_RE = re.compile(r"(\d{8})")
_MONTH_NAMES = {
    name: i for i, name in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}


def _infer_year_from_filename(path: Path) -> int | None:
    """Pull the year from a filename like 'Underdog_Draft_Table_20260519.csv'."""
    m = _FILENAME_DATE_RE.search(path.name)
    if m:
        return int(m.group(1)[:4])
    return None


def _parse_adp_date_label(label: str, year: int) -> str:
    """'April 25' + 2026 -> '2026-04-25T00:00:00'."""
    parts = label.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Can't parse 4for4 date label: {label!r}")
    month_name, day = parts
    month = _MONTH_NAMES.get(month_name)
    if month is None:
        raise ValueError(f"Unknown month in 4for4 date label: {month_name!r}")
    return f"{year:04d}-{month:02d}-{int(day):02d}T00:00:00"


def ingest_adp_from_4for4_csv(
    path: str | Path,
    year: int | None = None,
    draft_format: str = "best_ball",
    source: str = "4for4_underdog",
    skip_undrafted_threshold: float = 215.0,
    db_path: str = DB_PATH,
) -> dict:
    """Parse 4for4's Underdog ADP CSV. Each 'ADP on <date>' column
    becomes its own snapshot in adp_snapshots, so a single download
    can backfill multiple historical points.

    Args:
      year: year for the 'Month Day' date labels. Inferred from the
            filename's YYYYMMDD if present; otherwise defaults to now.
      skip_undrafted_threshold: 4for4 caps undrafted players at ~216;
            drop those — they're not real ADP, just position-holders.
    """
    init_db(db_path)
    path = Path(path)
    if year is None:
        year = _infer_year_from_filename(path) or dt.datetime.utcnow().year

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        date_columns: list[tuple[str, str]] = []
        for col in fieldnames:
            m = _4FOR4_DATE_COL_RE.match(col)
            if m:
                date_columns.append((col, _parse_adp_date_label(m.group(1), year)))
        if not date_columns:
            raise ValueError(
                f"No 'ADP on <date>' columns in 4for4 CSV. Got: {fieldnames}"
            )
        rows = list(reader)

    snapshots_by_date: dict[str, list[dict]] = {ts: [] for _, ts in date_columns}
    unmatched_by_date: dict[str, list[str]] = {ts: [] for _, ts in date_columns}

    with connect(db_path) as conn:
        by_uid, by_name_pos, by_name = _build_lookups(conn)

        # Idempotency: pre-load the (player_id, captured_at) pairs we
        # already have for this source so re-ingesting an overlapping
        # file is a no-op for the dates that already exist.
        existing_keys: set[tuple[int, str]] = {
            (r["player_id"], r["captured_at"])
            for r in conn.execute(
                "SELECT player_id, captured_at FROM adp_snapshots WHERE source = ?",
                (source,),
            )
        }
        skipped_existing = 0

        for row in rows:
            name = (row.get("Player") or "").strip()
            position = (row.get("Position") or "").strip() or None
            pos_rank = _parse_position_rank(row.get("Position Rank"))
            if not name:
                continue
            rec = {
                "underdog_id": None,
                "full_name": name,
                "position": position,
            }
            pid = _resolve_player(rec, by_uid, by_name_pos, by_name)

            for col, ts in date_columns:
                raw = (row.get(col) or "").strip()
                if not raw:
                    continue
                try:
                    adp = float(raw)
                except ValueError:
                    continue
                if adp >= skip_undrafted_threshold:
                    continue
                if pid is None:
                    unmatched_by_date[ts].append(name)
                    continue
                if (pid, ts) in existing_keys:
                    skipped_existing += 1
                    continue
                snapshots_by_date[ts].append(
                    {
                        "player_id": pid,
                        "adp": adp,
                        "adp_rank": None,
                        "position_rank": pos_rank,
                        "draft_format": draft_format,
                        "source": source,
                        "captured_at": ts,
                    }
                )

        # Derive per-snapshot adp_rank from sorting ADP ascending.
        for batch in snapshots_by_date.values():
            batch.sort(key=lambda s: s["adp"])
            for i, snap in enumerate(batch, start=1):
                snap["adp_rank"] = i

        all_snapshots = [s for batch in snapshots_by_date.values() for s in batch]
        if all_snapshots:
            cols = [
                "player_id", "adp", "adp_rank", "position_rank",
                "draft_format", "source", "captured_at",
            ]
            conn.executemany(
                f"INSERT INTO adp_snapshots ({','.join(cols)}) "
                f"VALUES ({','.join(['?'] * len(cols))})",
                [[s[c] for c in cols] for s in all_snapshots],
            )

    summary: dict = {
        "snapshots_written": len(all_snapshots),
        "skipped_existing": skipped_existing,
        "by_date": {},
    }
    for col, ts in date_columns:
        summary["by_date"][ts] = {
            "written": len(snapshots_by_date[ts]),
            "unmatched": len(unmatched_by_date[ts]),
        }

    log.info(
        "4for4 ingest: %d snapshot rows across %d dates (skipped %d already in DB)",
        summary["snapshots_written"],
        len(date_columns),
        skipped_existing,
    )
    for ts, stats in summary["by_date"].items():
        log.info(
            "  %s: %d written, %d unmatched", ts, stats["written"], stats["unmatched"]
        )
    return summary


def normalize_adp_response(raw: dict) -> list[dict]:
    """Translate Underdog's raw payload into our internal shape.

    Expected output records: {underdog_id, full_name, position, team,
    adp, adp_rank, draft_format}.

    This is a placeholder — the actual mapping depends on what fields
    the real response uses. Adjust once we have a captured response."""
    out: list[dict] = []
    players = raw.get("players") or raw.get("data") or []
    for p in players:
        out.append(
            {
                "underdog_id": str(p.get("id") or p.get("player_id") or ""),
                "full_name": p.get("name") or p.get("display_name"),
                "position": p.get("position"),
                "team": p.get("team") or p.get("team_abbr"),
                "adp": p.get("adp"),
                "adp_rank": p.get("adp_rank") or p.get("rank"),
                "draft_format": p.get("draft_format") or "best_ball",
            }
        )
    return out


FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST", "FB"}


def _build_lookups(conn):
    """Return three lookups into the players table:
    - by_underdog_id: {underdog_id: player_id}
    - by_name_pos:    {(normalized_name, position): player_id}
    - by_name:        {normalized_name: player_id}

    Name+position is preferred over name alone — that's how we keep
    'Josh Allen QB' from matching 'Josh Allen LB'."""
    by_uid: dict[str, int] = {}
    by_name_pos: dict[tuple[str, str], int] = {}
    by_name: dict[str, list[int]] = {}
    for r in conn.execute(
        "SELECT player_id, full_name, position, underdog_id FROM players"
    ):
        if r["underdog_id"]:
            by_uid[r["underdog_id"]] = r["player_id"]
        if r["full_name"]:
            nm = _normalize_name(r["full_name"])
            if r["position"]:
                by_name_pos[(nm, r["position"])] = r["player_id"]
            by_name.setdefault(nm, []).append(r["player_id"])
    # Collapse name-only lookup, but prefer fantasy-relevant positions
    # when there's a collision so 'Josh Allen' alone still lands on the QB.
    name_singletons: dict[str, int] = {}
    for nm, pids in by_name.items():
        if len(pids) == 1:
            name_singletons[nm] = pids[0]
            continue
        fantasy_pids = [
            pid for (nm2, pos), pid in by_name_pos.items()
            if nm2 == nm and pos in FANTASY_POSITIONS
        ]
        if len(fantasy_pids) == 1:
            name_singletons[nm] = fantasy_pids[0]
        # else: leave it unresolved — caller will see a miss and log it
    return by_uid, by_name_pos, name_singletons


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, drop common
    generational suffixes (Jr., Sr., II/III/IV). This handles the
    crosswalk edge cases PLANNING.md flags — Underdog's 'AJ Brown'
    vs nflverse's 'A.J. Brown', or 'Marvin Harrison Jr.' vs 'Marvin
    Harrison' when one source omits the suffix."""
    s = name.lower().replace(".", "").replace("'", "").replace(",", "")
    s = s.replace("-", " ")
    tokens = [t for t in s.split() if t and t not in _NAME_SUFFIXES]
    return " ".join(tokens)


def _resolve_player(
    rec: dict,
    by_uid: dict,
    by_name_pos: dict | None = None,
    by_name: dict | None = None,
) -> int | None:
    """Resolution order:
    1. underdog_id exact match
    2. (normalized name, position) match
    3. normalized name alone, only if it's unambiguous after filtering
       to fantasy-relevant positions
    """
    uid = rec.get("underdog_id")
    if uid and uid in by_uid:
        return by_uid[uid]
    name = rec.get("full_name")
    if not name:
        return None
    nm = _normalize_name(name)
    pos = rec.get("position")
    if pos and by_name_pos and (nm, pos) in by_name_pos:
        return by_name_pos[(nm, pos)]
    if by_name and nm in by_name:
        return by_name[nm]
    return None


def _last_snapshot_age_hours(conn, draft_format: str) -> float | None:
    row = conn.execute(
        "SELECT MAX(captured_at) AS ts FROM adp_snapshots WHERE draft_format=?",
        (draft_format,),
    ).fetchone()
    if not row or not row["ts"]:
        return None
    last = dt.datetime.fromisoformat(row["ts"])
    return (dt.datetime.utcnow() - last).total_seconds() / 3600.0


def ingest_adp(
    fetch: Callable[[], list[dict]] | None = None,
    draft_format: str = "best_ball",
    min_age_hours: float = 0.0,
    backfill_underdog_id: bool = True,
    db_path: str = DB_PATH,
) -> int:
    """Pull ADP and append a snapshot row per matched player.

    Args:
      fetch: callable returning normalized records (see normalize_adp_response).
             Defaults to the live Underdog fetcher.
      draft_format: tag for the snapshot, e.g. 'best_ball' or 'half_ppr'.
      min_age_hours: skip if the most recent snapshot for this format
                     is younger than this. 0 = always pull.
      backfill_underdog_id: when a name-match resolves a player, fill in
                            players.underdog_id so future runs hit the
                            faster ID path.
    """
    init_db(db_path)
    if fetch is None:
        fetch = lambda: normalize_adp_response_from_live(draft_format)

    with connect(db_path) as conn:
        if min_age_hours > 0:
            age = _last_snapshot_age_hours(conn, draft_format)
            if age is not None and age < min_age_hours:
                log.info(
                    "Last %s snapshot is %.1fh old (< %.1fh) — skipping",
                    draft_format,
                    age,
                    min_age_hours,
                )
                return 0

        records = fetch()
        by_uid, by_name_pos, by_name = _build_lookups(conn)
        captured_at = dt.datetime.utcnow().isoformat(timespec="seconds")

        snapshots: list[dict] = []
        unmatched: list[str] = []
        backfills: list[tuple[str, int]] = []

        for rec in records:
            pid = _resolve_player(rec, by_uid, by_name_pos, by_name)
            if pid is None:
                unmatched.append(rec.get("full_name") or "?")
                continue
            if rec.get("adp") is None:
                continue
            snapshots.append(
                {
                    "player_id": pid,
                    "adp": float(rec["adp"]),
                    "adp_rank": rec.get("adp_rank"),
                    "position_rank": _parse_position_rank(
                        rec.get("position_rank")
                    ),
                    "draft_format": rec.get("draft_format") or draft_format,
                    "source": "underdog",
                    "captured_at": captured_at,
                }
            )
            uid = rec.get("underdog_id")
            if (
                backfill_underdog_id
                and uid
                and uid not in by_uid
            ):
                backfills.append((uid, pid))
                by_uid[uid] = pid

        if snapshots:
            cols = [
                "player_id",
                "adp",
                "adp_rank",
                "position_rank",
                "draft_format",
                "source",
                "captured_at",
            ]
            conn.executemany(
                f"INSERT INTO adp_snapshots ({','.join(cols)}) VALUES ({','.join(['?'] * len(cols))})",
                [[s[c] for c in cols] for s in snapshots],
            )

        for uid, pid in backfills:
            conn.execute(
                "UPDATE players SET underdog_id=? WHERE player_id=? AND underdog_id IS NULL",
                (uid, pid),
            )

    if unmatched:
        log.warning(
            "%d ADP records had no players-table match: %s%s",
            len(unmatched),
            ", ".join(unmatched[:5]),
            " ..." if len(unmatched) > 5 else "",
        )
    log.info("Appended %d ADP snapshots", len(snapshots))
    return len(snapshots)


def normalize_adp_response_from_live(draft_format: str) -> list[dict]:
    """Convenience wrapper: hit Underdog, normalize, return records.
    Currently delegates to the real fetcher (which is not implemented)."""
    raw = fetch_adp_from_underdog(draft_format=draft_format)
    return normalize_adp_response(raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from-underdog-csv",
        help="Path to the rankings CSV downloaded from Underdog's web app. "
        "Populates both adp_snapshots and player_projections (source='underdog').",
    )
    p.add_argument(
        "--from-4for4-csv",
        help="Path to 4for4's Underdog Draft Table CSV. Each 'ADP on <date>' "
        "column becomes its own snapshot in adp_snapshots (source='4for4_underdog').",
    )
    p.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year for 4for4's 'Month Day' date labels. Inferred from filename "
        "(YYYYMMDD) when not provided.",
    )
    p.add_argument(
        "--from-file",
        help="Path to a JSON file of pre-normalized records (see fetch_adp_from_file). "
        "Useful for fixture-driven tests.",
    )
    p.add_argument("--draft-format", default="best_ball")
    p.add_argument(
        "--min-age-hours",
        type=float,
        default=0.0,
        help="Skip if last snapshot is younger than this many hours.",
    )
    p.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season tag for projections (defaults to current/next NFL season).",
    )
    p.add_argument(
        "--no-projections",
        action="store_true",
        help="With --from-underdog-csv, skip writing the projectedPoints "
        "column to player_projections.",
    )
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.from_underdog_csv:
        records = fetch_adp_from_underdog_csv(
            args.from_underdog_csv, draft_format=args.draft_format
        )
        ingest_adp(
            fetch=lambda: records,
            draft_format=args.draft_format,
            min_age_hours=args.min_age_hours,
            db_path=args.db,
        )
        if not args.no_projections:
            ingest_underdog_projections(
                records, season=args.season, db_path=args.db
            )
    elif args.from_4for4_csv:
        ingest_adp_from_4for4_csv(
            args.from_4for4_csv,
            year=args.year,
            draft_format=args.draft_format,
            db_path=args.db,
        )
    elif args.from_file:
        ingest_adp(
            fetch=lambda: fetch_adp_from_file(args.from_file),
            draft_format=args.draft_format,
            min_age_hours=args.min_age_hours,
            db_path=args.db,
        )
    else:
        # Falls through to the live HTTP fetcher, which raises a clear
        # NotImplementedError pointing back to the CSV path.
        ingest_adp(
            fetch=None,
            draft_format=args.draft_format,
            min_age_hours=args.min_age_hours,
            db_path=args.db,
        )


if __name__ == "__main__":
    main()
