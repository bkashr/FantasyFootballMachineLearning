import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = "fantasy.db"


@contextmanager
def connect(db_path: str = DB_PATH):
    """Yield a sqlite3 connection with foreign keys enabled. Commits on
    clean exit, rolls back on exception, always closes."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    -- Crosswalk: one row per real-world player, with IDs from each source.
    -- player_id is our internal canonical ID; source IDs are nullable so we
    -- can add a player before we have every source's ID for them.
    CREATE TABLE IF NOT EXISTS players (
        player_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name     TEXT NOT NULL,
        position      TEXT,
        team          TEXT,
        birth_date    TEXT,
        rookie_year   INTEGER,
        underdog_id   TEXT UNIQUE,
        gsis_id       TEXT UNIQUE,
        sleeper_id    TEXT UNIQUE,
        pfr_id        TEXT UNIQUE,
        espn_id       TEXT UNIQUE
    );

    -- Append-only ADP snapshots. One row per (player, pull). Keep the full
    -- history so we can see how ADP moved over time.
    CREATE TABLE IF NOT EXISTS adp_snapshots (
        snapshot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id     INTEGER NOT NULL,
        adp           REAL NOT NULL,
        adp_rank      INTEGER,
        draft_format  TEXT,
        source        TEXT NOT NULL DEFAULT 'underdog',
        captured_at   TEXT NOT NULL,
        FOREIGN KEY (player_id) REFERENCES players(player_id)
    );
    CREATE INDEX IF NOT EXISTS idx_adp_player_time
        ON adp_snapshots(player_id, captured_at);

    -- Season-level aggregate stats.
    CREATE TABLE IF NOT EXISTS player_season_stats (
        player_id          INTEGER NOT NULL,
        season             INTEGER NOT NULL,
        team               TEXT,
        games_played       INTEGER,
        pass_attempts      INTEGER,
        pass_completions   INTEGER,
        pass_yards         INTEGER,
        pass_tds           INTEGER,
        interceptions      INTEGER,
        carries            INTEGER,
        rush_yards         INTEGER,
        rush_tds           INTEGER,
        targets            INTEGER,
        receptions         INTEGER,
        rec_yards          INTEGER,
        rec_tds            INTEGER,
        fumbles_lost       INTEGER,
        fantasy_points     REAL,
        fantasy_points_ppr REAL,
        PRIMARY KEY (player_id, season),
        FOREIGN KEY (player_id) REFERENCES players(player_id)
    );

    -- Weekly box scores.
    CREATE TABLE IF NOT EXISTS player_weekly_stats (
        player_id          INTEGER NOT NULL,
        season             INTEGER NOT NULL,
        week               INTEGER NOT NULL,
        team               TEXT,
        opponent           TEXT,
        pass_attempts      INTEGER,
        pass_completions   INTEGER,
        pass_yards         INTEGER,
        pass_tds           INTEGER,
        interceptions      INTEGER,
        carries            INTEGER,
        rush_yards         INTEGER,
        rush_tds           INTEGER,
        targets            INTEGER,
        receptions         INTEGER,
        rec_yards          INTEGER,
        rec_tds            INTEGER,
        fumbles_lost       INTEGER,
        fantasy_points     REAL,
        fantasy_points_ppr REAL,
        PRIMARY KEY (player_id, season, week),
        FOREIGN KEY (player_id) REFERENCES players(player_id)
    );

    -- Projected stats. Source-tagged so we can store multiple projections
    -- side by side (FantasyPros consensus, our own model, etc.).
    CREATE TABLE IF NOT EXISTS player_projections (
        player_id            INTEGER NOT NULL,
        season               INTEGER NOT NULL,
        source               TEXT NOT NULL,
        projected_points     REAL,
        projected_points_ppr REAL,
        pass_yards           INTEGER,
        pass_tds             INTEGER,
        rush_yards           INTEGER,
        rush_tds             INTEGER,
        receptions           INTEGER,
        rec_yards            INTEGER,
        rec_tds              INTEGER,
        captured_at          TEXT NOT NULL,
        PRIMARY KEY (player_id, season, source),
        FOREIGN KEY (player_id) REFERENCES players(player_id)
    );

    -- Team-level season stats. Feeds the offense/defense ratings the README
    -- calls out, and supports schedule/strength-of-opponent adjustments.
    CREATE TABLE IF NOT EXISTS team_season_stats (
        team               TEXT NOT NULL,
        season             INTEGER NOT NULL,
        games_played       INTEGER,
        points_for         INTEGER,
        points_against     INTEGER,
        pass_yards_for     INTEGER,
        rush_yards_for     INTEGER,
        pass_yards_against INTEGER,
        rush_yards_against INTEGER,
        offense_rating     REAL,
        defense_rating     REAL,
        PRIMARY KEY (team, season)
    );

    -- Injury history. One row per reported status change.
    CREATE TABLE IF NOT EXISTS injuries (
        injury_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id     INTEGER NOT NULL,
        season        INTEGER,
        week          INTEGER,
        status        TEXT,
        body_part     TEXT,
        notes         TEXT,
        reported_at   TEXT,
        FOREIGN KEY (player_id) REFERENCES players(player_id)
    );
    CREATE INDEX IF NOT EXISTS idx_injuries_player ON injuries(player_id);

    -- Computed player scores, source-tagged so we can iterate on the
    -- scoring model without losing prior runs.
    CREATE TABLE IF NOT EXISTS player_scores (
        player_id     INTEGER NOT NULL,
        season        INTEGER NOT NULL,
        model_version TEXT NOT NULL,
        score         REAL NOT NULL,
        rank          INTEGER,
        computed_at   TEXT NOT NULL,
        PRIMARY KEY (player_id, season, model_version),
        FOREIGN KEY (player_id) REFERENCES players(player_id)
    );
    """)

    conn.commit()
    conn.close()


def upsert(conn, table: str, row: dict, conflict_cols: list[str]):
    """Insert or update a single row keyed on conflict_cols. Used by
    ingest scripts so re-runs don't duplicate."""
    cols = list(row.keys())
    placeholders = ",".join(["?"] * len(cols))
    col_list = ",".join(cols)
    update_cols = [c for c in cols if c not in conflict_cols]
    if update_cols:
        update_clause = ",".join(f"{c}=excluded.{c}" for c in update_cols)
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT({','.join(conflict_cols)}) DO UPDATE SET {update_clause}"
        )
    else:
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT({','.join(conflict_cols)}) DO NOTHING"
        )
    conn.execute(sql, [row[c] for c in cols])


def upsert_many(conn, table: str, rows: list[dict], conflict_cols: list[str]):
    """Bulk version of upsert. Assumes every row has the same keys."""
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ",".join(["?"] * len(cols))
    col_list = ",".join(cols)
    update_cols = [c for c in cols if c not in conflict_cols]
    if update_cols:
        update_clause = ",".join(f"{c}=excluded.{c}" for c in update_cols)
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT({','.join(conflict_cols)}) DO UPDATE SET {update_clause}"
        )
    else:
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT({','.join(conflict_cols)}) DO NOTHING"
        )
    conn.executemany(sql, [[r[c] for c in cols] for r in rows])


if __name__ == "__main__":
    init_db()
    print(f"Initialized {DB_PATH}")
