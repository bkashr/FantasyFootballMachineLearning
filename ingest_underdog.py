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
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Callable, Iterable

from database import DB_PATH, connect, init_db

log = logging.getLogger(__name__)

UNDERDOG_ADP_URL = "https://api.underdogfantasy.com/v1/over_under_lines"  # PLACEHOLDER — confirm via devtools recon


def fetch_adp_from_underdog(draft_format: str = "best_ball") -> list[dict]:
    """Real HTTP fetcher. Not wired up yet — recon Underdog's network
    calls (see PLANNING.md step 1) to fill in the URL, headers, and
    response parsing. Until then this raises so we don't silently no-op."""
    raise NotImplementedError(
        "Underdog ADP endpoint not yet identified. Do the devtools recon "
        "in PLANNING.md step 1 and wire it up here."
    )


def fetch_adp_from_file(path: str | Path) -> list[dict]:
    """Test/dev fetcher. Reads a JSON file of the shape produced by
    `normalize_adp_response` so we can exercise the pipeline before the
    real endpoint is known."""
    return json.loads(Path(path).read_text())


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
        "--from-file",
        help="Path to a JSON file of pre-normalized records (see fetch_adp_from_file). "
        "Useful before the live endpoint is wired up.",
    )
    p.add_argument("--draft-format", default="best_ball")
    p.add_argument(
        "--min-age-hours",
        type=float,
        default=0.0,
        help="Skip if last snapshot is younger than this many hours.",
    )
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.from_file:
        fetch = lambda: fetch_adp_from_file(args.from_file)
    else:
        fetch = None  # uses the live (not-yet-implemented) fetcher

    ingest_adp(
        fetch=fetch,
        draft_format=args.draft_format,
        min_age_hours=args.min_age_hours,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
