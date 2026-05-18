# Roadmap

Long-term phased plan for the Fantasy Football best-ball draft assistant.
`PLANNING.md` covers the tactical Phase-1 sequencing; this document covers
the broader vision and the phases beyond foundation.

## Vision

A best-ball draft assistant that, given your current draft slot and your
active portfolio across every live draft, recommends the best pick
available — combining per-player projections, value-vs-ADP analysis,
position-fall-off awareness, stacking and team-build heuristics, exposure
management, and a measured dose of uniqueness / contrarianism. Long-term,
the system learns directly from historical winning teams instead of
relying on hand-coded rules.

## Three concentric layers

```
   ┌─ Projection (will player X score Y points?)
   │
   ├─── Value (is player X a buy at their current ADP?)
   │
   └────── Recommendation (given my portfolio + draft state, who do I take HERE?)
```

Each layer feeds the next. The value layer is meaningless without
projections; the recommendation layer is meaningless without value scores.

## Phased build

### Phase 1 — Foundation (DONE)

Schema, nflverse ingest (rosters / season / weekly / injuries), Underdog
ADP stub, `baseline_v1` scoring, CLI viewer (`report.py`). 36 tests
passing. Underdog live endpoint still pending devtools recon — fixture
JSON works end-to-end in the meantime.

### Phase 2 — Projection v2

Replace "projection = last year's PPR" with something closer to reality.
Components:

- **Multi-year weighted PPR baseline.** Recent seasons weighted more.
- **Per-game rate, not raw totals.** Productive-but-injured players
  shouldn't be punished by games missed.
- **Age curve adjustment per position.** RBs decline around 28, WRs hold
  longer, QBs peak late.
- **Team-context adjustment.** Coaching change, new system, depth-chart
  change.
- **Injury-detail-aware risk model.** Not just "missed 6 games last
  year" — the type of injury and recovery trajectory:
  - Specific injury: ACL tear, Achilles tear, ankle fracture, hamstring
    soft tissue, etc. Different recovery curves and reinjury rates.
  - Time since injury (rookie-year ACL is different from one in week 17).
  - Age at injury (older players recover slower).
  - Position (QB knee injury vs WR knee injury — different mobility
    demands).
  - **Play style.** Athletic / explosion-dependent players (Saquon-type)
    are hit harder by lower-body injuries than technique-dependent
    players (Kupp / Davante Adams types). Likely classified from advanced
    stats — average separation, YAC-over-expected, broken-tackle rate.

**Backtest plan.** Project 2023 using only 2020-2022 data, compare to
actual 2023 PPR. Iterate weights using this feedback loop. Same approach
for the injury-recovery curves: bucket post-injury seasons by injury
type / age / position / play-style and measure actual production.

### Phase 3 — Position-aware value

Per-position value curves (TE15 cliff vs QB12-18 plateau). Computes
positional VBD: points above replacement at the position rather than vs
overall draft slot.

**Note: prefer borrowing over re-deriving.** Most of the positional
fall-off and value-curve work already exists in the public fantasy
community (4for4, FantasyPros, Establish The Run, Sharp Football
Analysis, RotoViz). Import or replicate their published curves where
possible. Only build our own analysis for things we can't find
off-the-shelf.

### Phase 4 — Best-ball strategy patterns

Sits next to Phase 3 in spirit: stay optimal to best-ball best practices
most of the time. Items to capture:

- Position run timing — when drafters reach for each position.
- Winning team build distributions — QB/RB/WR/TE counts on advancing
  teams.
- Stack value — QB + pass-catcher correlation, bringback patterns.
- Late-round hit-rate zones by position.

**Same note as Phase 3:** prefer importing established best-ball research
(Establish The Run, RotoViz, Underdog's own Best Ball Mania data drops)
over original analysis where good work already exists.

### Phase 5 — Portfolio layer

Two things to figure out before coding.

**Research questions (no math yet, just notes):**

- **Optimal exposure caps.** Is it 12% max on an R1 player? 15% when we
  believe they're undervalued? Lower on perceived busts? Find empirical
  evidence in BBM advancement data rather than guess.
- **Hard avoids.** When do we *categorically* skip a player we think is
  a bust to concentrate hit-rate on alternatives?
- **Diversification axes.** Exposure by player, by team-offense, by
  archetype (rookie vs vet, ceiling vs floor)?
- **Uniqueness / contrarianism.** Same logic as unconventional play in
  poker: deviating from norms can be profitable, not because uniqueness
  is intrinsically good but because (a) the field is anti-correlated to
  your unique builds and (b) when a contrarian play hits, it crushes the
  field. Examples: two QBs back-to-back; grabbing a falling player for a
  rare R1/R2 combination. Hard to encode cleanly — defer the math. A
  reasonable measurable proxy: *what fraction of advancing teams in BBM
  had this combo?* Low frequency + high advance rate = profitable
  contrarian play.

**Coding components (after research):**

- Portfolio tracker tables: `my_drafts`, `my_picks`.
- Exposure-aware scoring — a player's value to *you* discounts as
  exposure rises, lifts if you believe they're undervalued.
- Stack-aware boosts — already roster the QB? Lift his pass-catchers.
- Uniqueness/contrarianism scoring (eventually).

### Phase 6 — Live recommender (frontend)

The user-facing tool. "You're picking 7.4 in a draft, here are the top 5
options ranked for this draft + your portfolio + best-ball best practices
+ your contrarian preferences." Composes everything from Phases 2-5.
Largely frontend / wiring work — the heavy lifting is upstream.

### Phase 7 — ML layer

Train on historical best-ball drafts to learn winning-team patterns
directly instead of hand-coding them.

**Which prior phases does this serve?** Multiple, in different ways:

- **Phase 3 (position-aware value).** ML can validate or refine the
  curves — does the rank → PPR relationship look the same in winning
  teams as in the population?
- **Phase 4 (strategy patterns).** ML can learn position-run and
  team-build patterns we don't see by eye. This is the most natural fit.
- **Phase 5 (portfolio).** ML can suggest exposure caps and uniqueness
  weights from data instead of intuition — e.g. "teams with >15%
  exposure to any one R1 player underperformed in BBM 2023".
- **Phase 2 (projections).** ML *could* fit projections too, but that's
  a separate model with a different target (player points, not roster
  outcomes). Keep projection a stats model and ML a draft-strategy
  model — different problems shouldn't share one model.

**Data.** 2024 and 2025 Best Ball Mania final-round rosters (Underdog
publishes these annually). Possibly scrape advancing-round rosters from
public profiles for more samples. For each draft we need: roster, draft
slot, finish, advancement rate.

## Cross-cutting open questions

1. **Format scope.** Best-ball only, or also redraft / dynasty / DFS?
   Assuming best-ball for now; format-specific scoring weights and
   roster constructions would need to be added throughout if expanded.
2. **Portfolio sync.** Does Underdog expose a "my drafts" endpoint? If
   not, v1 of Phase 5 is manual entry; v2 is auto-sync.
3. **Underdog endpoint recon.** Still blocks live ADP. Until then,
   fixture-driven testing only.

## Factors to model (reference list)

**Per-player**
- Multi-year PPR, weighted recent
- Age × position (decline curves)
- Team change / coaching change
- Depth chart position
- Opportunity metrics: carries + targets
- Injury history with type, recency, age-at-injury, play-style sensitivity
- Schedule strength

**Per-position**
- Replacement-level PPR (13th QB, 25th RB, 37th WR, 13th TE)
- Where the curve cliffs
- Positional variance / upside

**Per-team-build (best-ball-specific)**
- 18 roster spots
- Typical winning build distribution
- QB stacks + bringbacks
- Position runs

**Per-portfolio**
- Exposure % per player across all drafts
- Team-offense exposure
- Risk profile (rookies vs vets, ceiling vs floor)
- Uniqueness / contrarianism score
