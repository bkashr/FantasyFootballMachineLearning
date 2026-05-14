# Planning Document

## Overview

Build a backend-first system that ingests fantasy football data, scores players, and compares those scores against Underdog ADP to surface values and reaches. Frontend (terminal UI) comes after the data layer is solid.

## Why backend first

The frontend is a viewer over whatever schema we land on. Building it before knowing the data shape creates rework. Lock down sources and schema, then build UI against real data.

## Storage decision

SQLite. Fantasy football data is small (hundreds of players x ~20 years of weekly stats is well under a million rows), single-writer, no server to run. Easy to swap to Postgres later if needed.

## Data domains

Two distinct domains with different sources and refresh cadences:

### 1. Historical / static stats
- **What:** career numbers, age, position, team, prior-year fantasy points, weekly box scores, injuries (later)
- **Source:** `nfl_data_py` (Python wrapper around nflverse). Free, comprehensive, includes player ID crosswalks
- **Cadence:** rarely — once at season start, then after each week during the season

### 2. ADP (Underdog)
- **What:** current ADP per player, timestamped so we can track movement over time
- **Source:** Underdog's internal API (reverse-engineered from their web app's network calls)
- **Cadence:** lazy refresh — pull on demand when the user loads the tool, cache with a timestamp, only re-fetch if older than X hours. Append-only writes so we keep the full history of how ADP moved.

### Future data (not first pass)
- NFL news with timestamps, so ADP swings can be correlated with events (trades, injury reveals, etc.)
- Projected offense / defense ratings
- Injury history

## The ID crosswalk problem

Every source uses different player IDs:
- nflverse: GSIS / PFR / Sleeper IDs
- Underdog: their own internal IDs + display name only in some responses

We need a `players` table that acts as the crosswalk:
- `underdog_id`, `gsis_id`, `sleeper_id`, canonical name, position, team

This will need some semi-manual cleanup for edge cases — rookies that don't exist in historical sources yet, name collisions (there are two Michael Carters), suffixes (Jr./Sr.), etc. Building this once saves constant headaches downstream.

## ADP source: why Underdog's internal API over FantasyPros

FantasyPros' "Underdog ADP" is sometimes lagged or smoothed. Going direct to Underdog avoids that. The tradeoff is the endpoint is undocumented and can change — accepted, since manually updating the endpoint string when it breaks is cheap for a personal project.

## First concrete step (before any code)

Recon on Underdog's API:

1. Open Underdog Fantasy in the browser
2. Open devtools -> Network tab
3. Load a best-ball draft lobby or the player pool page
4. Find the call that returns the player list with ADP
5. Note:
   - The URL
   - Headers — especially auth. Is there a bearer token? Is it tied to a logged-in session? Or is it open?
   - Response shape — what fields come back, what IDs, what does ADP look like (raw number? rank?)

The answer determines whether ingest is ~10 lines or ~100, and shapes the schema. Don't design tables before seeing the real response.

## Sequenced plan

1. **Recon Underdog API** (manual, devtools) — answer the auth + response shape questions
2. **Sketch schema** around the join keys the real responses give us
   - `players` (crosswalk)
   - `adp_snapshots` (append-only, timestamped)
   - `player_season_stats`, `player_weekly_stats` (from nflverse)
3. **Write ingest scripts** (idempotent — safe to re-run)
   - `ingest_nflverse.py` — historical stats
   - `ingest_underdog_adp.py` — on-demand ADP pull, writes a new snapshot row
4. **Player scoring** — first pass can be simple (projected points vs ADP), iterate from there
5. **Frontend terminal UI** — viewer over the scored data
6. **ML layer** (later) — train on historical drafts to find strategy patterns

## Open questions to revisit

- Where do we get *projected* stats from? (FantasyPros consensus? Roll our own from historicals?)
- How do we handle rookies with no historical data?
- For the ML draft strategy piece — where do we get historical draft results to train on? Underdog draft history per user is accessible; broader datasets may need to be assembled.
