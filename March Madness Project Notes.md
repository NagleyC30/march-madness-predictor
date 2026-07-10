March Madness Project Notes



Bet sizing — something like the Kelly Criterion, which tells you how much of your bankroll to wager based on your estimated edge over the sportsbook. On a -50,000 line you essentially have no edge worth betting, so Kelly would tell you to bet nearly nothing (or nothing at all).



Moneyline vs. spread — spreads tend to offer better value on heavy favorites because instead of accepting -50,000 to pick Michigan to win, you'd take Michigan -30.5 at roughly -110, meaning you risk $110 to win $100. Much better return for the same correct pick. The tradeoff is you now need Michigan to win by 31+, not just win.



Model features — a trained model could incorporate things like seed matchup history, KenPom efficiency ratings, injury reports, line movement, and public betting percentages to estimate the true probability of each outcome, then compare that to the implied probability in the odds to find spots where you actually have an edge.





Spread Cutoff at -200 odds

Would have won \~$700 versus a loss of \~$70





Run the experiment after having the model build and test on every year from 2008-2025. 

1\. Have the model learn on data from each year (Start with learning from 2008 and predicting 2009) -- **COMPLETE**

2\. Predict the following year's bracket using the knowledge (the first couple of years might be rough) -- **COMPLETE**



**NEXT**

3\. If the odds can be tracked down, figure out how much money would have been won using the bracket and testing different spread ranges (i.e., -200, -300, -350, etc.) -- **COMPLETE**

- Real closing moneylines pulled from the Sportsbook Reviews Online archive (`fetch_data`-style script `fetch_odds.py` → `data/odds.csv`), tournaments 2008–2019 + 2021 (2020 no tourney; 2022+ not published). 100% of tournament teams name-matched.

- `backtest_odds.py` settles the model's walk-forward picks at the real closing line for every game actually played, sweeping the -200/-250/-300/-350 confidence thresholds. Result: **every** training-window × threshold combo loses money (~ -2% to -7% ROI). The "confident" picks are heavy favorites (avg -200 bet is priced near -2800) — they pay pennies on a win and cost a full unit on a loss. No edge over the closing line, exactly the Kelly point from the notes above. (The earlier "~$700 vs -$70" figure was against model-implied odds, not real lines.)

- Shown in the app's **Betting Simulation** page via a "Real sportsbook odds" vs "Model-implied odds" toggle.

3b\. Flip heavy chalk to spread bets: when the real moneyline on a pick is shorter (more favored) than the threshold, the moneyline pays too little — so instead take the point spread at standard -110 juice (per the Kelly/Michigan -30.5 note above). Bet moneyline when the real line is at or above the threshold, flip to the spread when it's below. Compare the flip strategy's P&L against moneyline-only. -- **COMPLETE**

- Added real closing spreads + final scores to `data/odds.csv` (spread = smaller of the two SBR Close cells, favorite lays it; validated at a 48% home cover rate). `backtest_odds.py` now settles both strategies in one run → `data/betting_simulation_spreadflip.csv`.

- Result: flipping to the spread **doesn't help — it's worse** at 19 of 20 window×threshold combos (~ -5% to -11% ROI vs -2% to -7% for moneyline). The closing spread is efficient: a near-certain moneyline win becomes a ~coin-flip cover (1165–1170 ATS at -200) and you eat the -110 vig every time. The naive "take the points on a huge favorite" intuition doesn't survive contact with real, efficient closing lines. Shown as a third strategy toggle on the **Betting Simulation** page.

4\. Apply each model to the 2027 tournament. -- **IN PROGRESS / blocked on data**

- 2027 field data doesn't exist yet (Selection Sunday is March 2027). The dataset currently ends at the 2026 field, which is the only forecast-eligible target.

- Built `predict_all_windows.py`: applies **all five training-window models** (all_prior / last_1 / last_3 / last_5 / last_10) to the latest forecast field → `data/bracket_all_windows.csv`. It auto-detects the target year, so `python predict_all_windows.py` will predict 2027 the moment that field is added (or `python predict_all_windows.py 2027`).

- Demonstrated on **2026**: all five models agree on the champion (**Duke #1**) and on 31/32 first-round games; they diverge on the deep mid-seed run (Final Four's 3rd/4th spots swing across Arizona/Houston/Florida/Illinois/Nebraska/Purdue). Shown on the **Bracket Predictions** page as a champion+Final-Four-by-model table plus a per-window full bracket.

- TODO when 2027 field is published: re-run `predict_all_windows.py` (and add the 2027 rows to `KenPom Barttorvik.csv`).

5\. Build a current-season predictor for the upcoming year's games: pull the live/preseason Barttorvik ratings for the in-progress 2026–27 season and let the app predict upcoming regular-season and tournament games as they're scheduled (updating as ratings refresh through the season). Extends the general Game Predictor, which currently covers completed seasons 2008–2026. -- **COMPLETE (head-to-head); scheduled-games deferred on data**

- Barttorvik already publishes **2027 preseason ratings** (`2027_team_results.csv`), and they align exactly with the model's 34-feature schema (0 missing/extra, 0 nulls, 365 teams). `update_current_season.py` fetches them and merges/refreshes them into `data/ratings.csv` (now 2008–2027) — re-run any time to pull updated ratings through the season.

- The Game Predictor now offers **2027** (defaults to it) with an "in progress" banner: any 2026–27 head-to-head predicts immediately (e.g. Air Force vs Abilene Christian, neutral → 72.5%). The trained model is unchanged (2008–2026 games); it just forecasts from the new season's ratings.

- Scheduled-game prediction ("upcoming games as they're scheduled") is **deferred**: Barttorvik's `2027_super_sked.csv` (the schedule) is still 404 — it publishes closer to tip-off (Nov 2026). `update_current_season.py` reports when it appears; a schedule-driven "upcoming games" view can be layered on then.



**REFINED ROADMAP** (reprioritized 2026-07-09)

Direction: make **betting the centerpiece** of the project, then broaden the
**models**, then **polish the UI** for real users. Data-limited items wait until
the data is published.

---

**PHASE 1 — Betting front and center (multiple strategies).** *(current focus)*

6\. **Make betting the flagship of the app.** Elevate it in the information
architecture — lead the navigation / Overview with the betting analysis, frame
the whole project around "can the model beat the market," and consolidate the
betting views into one strong dashboard instead of a single buried page.

7\. **Build a strategy lab — try many betting strategies, compared side by side.** -- **DONE (first pass)**
On top of the three current settlements (moneyline, flip-to-spread,
model-implied), add:
- **Value / +EV betting**: bet only where the model's win probability beats the
  market's implied probability (from the real closing line) — the genuine "find
  spots with an edge" test. Report whether *any* slice (underdogs, line ranges,
  round) beats the close.
- **Staking schemes**: flat vs **fractional Kelly** (sized by the estimated edge).
- **Selection variants**: underdog-only, by-seed, by-round, favorites-only.
- Compare all strategies on one dashboard: ROI, win rate, **max drawdown**,
  bankroll **equity curve** over time.

- Built: `backtest_odds.py` now emits `data/bet_games.csv` (one settled
  game-prediction per row: model prob + real closing moneylines both sides +
  result). `betting_strategies.py` post-processes it into strategies (fast, no
  retraining; the app computes them live). Strategy-lab section on the **Betting
  Simulation** page: comparison table (bets, win%, ROI%, avg edge%, end bankroll,
  max drawdown) + bankroll-over-time chart, per training window.
- **Result — nuanced, and a great Phase-2 hook.** Flat +EV betting turns a
  *profit* on the longer windows (all_prior +4.2%, edge≥3% +11.8%, underdogs
  +7.2%) — the opposite of the chalk result. **But it's fragile**, not a proven
  edge: it's negative on the short windows (last_1/last_3), and on all_prior only
  5/12 tournaments win (2012 & 2018 carry it). The model is **overconfident** on
  favorites (rates games 84%/94% that it wins 78%/88%), so much of its "edge" is
  illusory. Decisive tell: **¼-Kelly loses on every window and craters the
  bankroll ($1000→$254, -89% drawdown)** — if the edges were real, Kelly would
  compound them. → strong motivation for Phase 2 (calibrate the model, then see
  if the +EV edges survive).
- **By-seed / by-round slices — DONE.** `betting_strategies.py` now emits
  `data/betting_strategies_slices.csv` (Value +EV flat, broken down by round and
  by the seed of the backed team); the **Betting Simulation** page shows both as
  a "Where does the edge live?" pair of tables. **Finding:** the edge isn't
  uniform — for `all_prior` it concentrates in the **first round (+10% ROI, 326
  bets)** and on **double-digit-seed underdogs (9–12: +18%, 13–16: +15%)**, and
  the favorite tiers (1–4, 5–8) lose — consistent with the model being
  overconfident on chalk. Deep-round slices (F4 +52% on 8 bets, Chip on 4) are
  noise. Reinforces the Phase-2 calibration motivation.

---

**PHASE 2 — Model bake-off (expand beyond RandomForest / Bagging).**

8\. **Add more ML models and compare them.** Broaden the tournament pipeline past
RandomForest + BaggingClassifier to e.g. **Logistic Regression** (a naturally
calibrated baseline), **Gradient/HistGradient Boosting**, **SVM**, **MLP**, and
optionally **XGBoost/LightGBM** (weigh the extra deploy dependency). Compare on:
- walk-forward **accuracy** and **Brier / log-loss**,
- **calibration** (reliability diagrams; boosting/forests are often
  miscalibrated → add isotonic/Platt where it helps — the general game model
  already has a calibration page to mirror),
- **betting P&L** (which model actually loses the least / finds the most value),
- and **bracket-pool points** (standard 10/20/40/80/160/320-by-round scoring
  across 2009–2025) — a concrete "which model fills the best bracket" metric.
Surface a model-comparison view in the app.

- **Bake-off + calibration — DONE (2026-07-10).** New `model_bakeoff.py`
  walk-forwards (window `all_prior`, 2009+) a **sklearn-only** roster —
  Logistic Regression, Random Forest, Bagging, HistGradientBoosting, **MLP (the
  "neural net")**, SVM — producing one out-of-sample P(high-seed-wins) per real
  tournament game, plus an **isotonic-calibrated** variant of each. Writes
  `data/model_bakeoff_{summary,reliability,meta}.csv`; new **Model Bake-off**
  page (metrics table shaded by Brier/log-loss/ECE + a raw-vs-calibrated
  reliability diagram). No new deploy dependency (avoided matplotlib — the
  Styler shading is a hand-rolled green→red `.apply`).
- **Findings (pooled over 16 tournaments):**
  - **Accuracy is nearly flat (0.66–0.72)** across all 12 variants — accuracy
    can't tell these apart, exactly why probabilistic metrics are needed.
  - **Overconfidence is real and quantified.** Raw HistGradientBoosting ECE
    **0.187**, raw **MLP the worst** (ECE **0.270**, log-loss **1.69** — classic
    small-tabular-data overfitting). Isotonic calibration slashes both (HGB→0.025,
    MLP→0.036); avg log-loss over the roster **0.841 → 0.596** with calibration.
  - **Random Forest raw is already the best-calibrated** (Brier **0.192**) —
    bagging averages its trees' votes. Nuance: isotonic *slightly hurts* RF
    (0.192→0.196) — calibration isn't free on an already-calibrated model with
    limited data.
- **Sub-step b — do the +EV edges survive calibration? — DONE (2026-07-10).**
  `backtest_calibrated.py` re-settles the *same* walk-forward bets at the *same*
  real closing lines, but wraps the trained model in **isotonic calibration**
  (fit on training seasons only) → `data/bet_games_calibrated.csv` (same schema
  as `bet_games.csv`, so the strategy lab runs on it unchanged). New **"Does the
  edge survive calibration?"** table on the Betting Simulation page puts raw vs
  calibrated ROI/edge side by side per strategy.
- **Finding — nuanced, and *stronger* than "the edge was fake":**
  - The +EV profit **does NOT collapse.** On `all_prior`, Value-flat even nudges
    up (+4.2→+5.2%), underdogs improve (+7.2→+10.4%); only edge≥3% craters
    (+11.8→+3.5%). On `last_3` several strategies flip from negative to slightly
    positive. So the longer-window edge is **partly robust** to calibration.
  - **¼-Kelly improves on every window** (all_prior −8.7→−4.1, last_3 −13.9→−6.2,
    last_1 −19.0→−8.6) — calibration curbs the overconfident sizing that cratered
    the bankroll, exactly as theory predicts (though Kelly still loses).
  - **The claimed edge % barely moves** — because the backtest's chosen model is
    the RF/Bagging the bake-off already found *well-calibrated*. The dramatic
    overconfidence lived in the models the backtest doesn't use (MLP, raw
    boosting). Consistent story across both sub-steps.
- **Sub-step c — bracket-pool points per model — DONE (2026-07-10).**
  `bracket_pool.py` walk-forwards each roster model's *full* bracket
  (`simulate_full_bracket`), scores it 10/20/40/80/160/320-by-round via the app's
  cascade scorer, and sums over 2009–2025 → `data/bracket_pool_{summary,by_year}.csv`.
  Includes a **Chalk (always higher seed)** baseline. New "Which model fills the
  best bracket?" section on the Model Bake-off page (bar chart vs the chalk line +
  per-round table).
- **Finding:** **Random Forest is the only model that clearly beats chalk** (8740
  vs 8170 pts over 16 yrs); **the MLP is worst by far (6510)** — the same
  overfitting that wrecked its calibration busts its brackets. HGB and LogReg also
  trail chalk. **Caveat surfaced in-app:** the *Championship* column is 0 for every
  model because the cascade convention only credits a game when *both* teams in the
  predicted matchup actually reached it — near-impossible by the final, so deep-
  round credit is rare (but the comparison stays apples-to-apples).
- **Still TODO / optional in Phase 2:**
  - Optional: run the bake-off across all five windows (currently `all_prior`
    only); consider XGBoost/LightGBM iff a real gain justifies the dep.
  - **New follow-up (from the overfitting review):** train the classifier on the
    ~103k regular-season games in `data/games.csv` (the tournament model only sees
    ~1k tournament games) — the structural fix for overfitting and the real
    prerequisite for a genuine deep neural net. Log as a post-Phase-2 item.

---

**PHASE 3 — UI / UX polish (make it nice for people to use).**

9\. **Design pass on the app.** Custom Streamlit theme (colors, fonts via
`.streamlit/config.toml`), cleaner navigation and a real landing/Overview page,
consistent chart styling, readable copy, and mobile-friendly layout. Goal: it
looks polished and is easy for a non-technical visitor to explore.

---

**PHASE 4 — Parked until the data is published.**

10\. **Complete item 4 (2027 field):** add 2027 rows to `KenPom Barttorvik.csv`
and re-run `predict_all_windows.py` once Selection Sunday 2027 sets the field.

11\. **Complete item 5 (scheduled games):** build the schedule-driven "upcoming
2026–27 games" view (+ optional auto-refresh of ratings) when
`2027_super_sked.csv` publishes (~Nov 2026).

12\. **Extend real odds to 2022–2025:** find a recent-lines source so the betting
backtests cover the full model era, not just through 2021. (Not strictly
time-gated — a sourcing task that can happen whenever.)

---

**NEW REQUESTS (added 2026-07-10)** — with feasibility notes + a reprioritized
order. See "REORDERED PLAN" at the bottom for the cascade-aware sequence.

13\. **Rename the project to B.O.B. (Betting on Basketball).** -- **DONE (2026-07-10,
with the branding pass — items 13 + 16 shipped together)**
- Renamed across `app.py` (`page_title` → "B.O.B. — Betting on Basketball",
  `page_icon`, sidebar title, module docstring), the **Overview** brand header +
  framing copy ("can it beat the market?"), and `README.md`. Centralized in
  `branding.py` (`APP_NAME`/`APP_TAGLINE`/`APP_ICON`) so it's defined once.

14\. **"Me vs. the Machine" contest page — my predictions vs. the models'.** For
every game the model predicts, I also predict; we run an everlasting scoreboard
of who's better. Show W-L-D for the model and for me, plus how much each of us
would have made under **every betting strategy** (item 7's lab). Must be
automated for 2027-forward games not yet played, with a recurring prompt to lock
my picks *before* tip-off (no score leakage), and I must not see the model's pick
when I make mine. -- **IN PROGRESS — manual-entry mode DONE (2026-07-10); auto-schedule deferred on data**
- **Manual-entry mode shipped.** New `contest.py` (store + scoring) + **"Me vs
  Machine"** page: pick a season/two teams/venue, lock your pick, and *then* the
  model's pick is revealed (blind entry enforced — the model is only queried
  inside the lock handler, never rendered before you commit). Both picks are
  written to `data/contest_picks.csv` with a UTC timestamp; you settle each game
  later by entering the real winner. Scoreboard: model vs. you W-L, agreement,
  and the headline **"when you disagree, who's right?"** stat. Optional market
  moneylines enable a **symmetric flat-bet P&L** (each side flat-bets its own
  pick — the fair wager when the human gives a pick but not a probability).
  Verified end-to-end live (lock → pending → settle → scoreboard). `contest_picks.csv`
  is gitignored (user runtime data); dtype gotcha fixed (empty text cols read as
  float64 rejected timestamp writes).
- **Still deferred (auto phase):** the automated weekly slate (needs the 2027
  schedule / `2027_super_sked.csv`, ~Nov 2026 — same blocker as items 5/11), the
  recurring "make your picks" prompt (scheduled-tasks/cron), and richer betting
  strategies (Kelly/+EV) once real closing lines exist. Durable storage (not an
  ephemeral-cloud CSV) also needed before it's truly "everlasting."
- **Feasibility: yes, with an honor-system caveat.** The head-to-head engine
  (`game_model`) and the strategy lab already exist; this page is mostly a
  pick-capture + settle + tally layer on top. The hard part is *integrity*, not
  modelling.
- **Slate = a curated weekly set of ~10 games, selected automatically** (must be
  data-only so it self-runs for 2027-forward). *Gate → score → constrain:*
  - **Sizing is data-backed (measured on 2023–2025 games.csv + ratings.csv):** a
    typical regular-season week has ~285–310 D1-vs-D1 games, ~76–84 pass the gate,
    and **~17–24 are genuinely "good"** (both teams strong *and* competitive).
    Only ~5/week are both-top-25, so a slate can't be all marquee clashes — the
    composite is what keeps quality up. **≥10 qualifying games in ~95–100% of
    weeks**, and the 10th-best game's competitiveness stays high (median
    0.75–0.90). → **10/week is safe.** Early-Nov / some conf-tourney weeks are
    thinner: apply a **quality floor** and let the slate shrink below 10 rather
    than padding with blowouts.
  - **Eligibility gate (all required):** both teams D1 with current ratings; game
    not yet played; ≥1 team in the national top ~60 by BARTHAG (kills
    two-bad-teams games).
  - **Composite "good matchup" score** (weights are tunable knobs). For a
    *contest* a slate of blowouts is a bad test (we'd both go 8/8), so
    **competitiveness is weighted highest**:
    - **Competitiveness (40%):** `1 − 2·|p − 0.5|`, `p` = model win prob. Pick'em
      = 1.0; a 90/10 game = 0.2.
    - **Quality (35%):** reward *both* teams good — `min(BARTHAG_a, BARTHAG_b)`
      (or inverted avg national rank), so one great + one weak doesn't qualify.
    - **Salience (25%), additive bonuses:** both ranked top-25; conference/rivalry
      game; **cross-window model disagreement** (the five windows split); model-
      vs-market disagreement where odds exist.
  - **Slate constraints:** ~8 games/week; no team twice in a week; per-day cap;
    keep a mix of favorites/underdogs (don't collapse to all toss-ups).
  - In **manual-entry mode** the same score doubles as a "suggest good matchups"
    helper over whatever games I paste in.
- **Design rules to keep it honest (commit-then-reveal):**
  - **Lock before tip-off.** Persist my pick with a timestamp (e.g.
    `data/contest_picks.csv`, optionally git-committed for an audit trail) and
    freeze it at the scheduled tip. Settle only after the official final.
  - **Lock the model's pick at the same time**, from the ratings-as-of that
    moment — so the model can't be (even accidentally) retrained on a result
    before the game is scored. No look-ahead for either side.
  - **Blind entry.** The pick form must not render the model's choice, prob, or
    the market line until after I submit.
  - **No score leakage.** The prompt fires from a schedule, not from any view
    that shows finals; the settle step is the only place scores enter.
  - **Missed-deadline policy** (pick one): auto-forfeit that game to the model, or
    drop the game from both records. Decide up front.
  - **Define the slate:** which games count (all D1? a curated weekly slate?
    tournament only?) and the tie/"draw" definition — **straight-up basketball
    games don't draw**, so "draw" only means an **ATS push** on the spread bets;
    the moneyline W-L has no draws.
- **Automation:** the recurring "make your picks" prompt can run via the
  scheduled-tasks / cron infra (or a calendar event). After finals post, a settle
  job updates both records and re-runs the strategy P&L for the contest slate.
- **Data dependency:** "games yet to happen in 2027" needs the schedule —
  `2027_super_sked.csv` is still 404 (same blocker as items 5/11, ~Nov 2026).
  **Un-blocked path:** ship a **manual-entry mode first** (I paste/enter an
  upcoming matchup, both sides predict, settle when I enter the final), then wire
  the auto-schedule feed when it publishes.
- **Cascade rule:** this page *displays model probabilities and betting P&L*, so
  Phase-2 calibration/new-models change every number on it. **Build it once,
  after Phase 2**, or accept re-work.

15\. **"How the models work" learning page.** Explain each classifier —
RandomForest, Bagging, XGBoost, etc. — what it's doing and why. Assess turning
the predictor into a **neural network**: how, and what constraints block it. --
**DONE (2026-07-10)**
- Built the **"How the Models Work"** page: how the model sees a game (feature
  diffs → `HIGH_SEED_WINS`), a **live feature-importance chart** and a **real
  depth-3 decision tree** (`model_explain.py`, precomputed to
  `data/model_explain_{importance.csv,tree.txt}` so the page loads instantly —
  live compute was ~26s), plain-English expanders for Bagging/RF, boosting,
  LogReg, SVM, MLP, and a "Could this be a neural network?" section that ties the
  MLP's bake-off overfitting to the data-size limit + the ~103k regular-season
  follow-up. No matplotlib (tree via `export_text`, not `plot_tree`). Explainers
  reference the **Model Bake-off** numbers instead of duplicating them, so the two
  pages compose. TODO when re-run: `python model_explain.py` after data changes.
- **Feasibility: easy for the explainers, nuanced for the NN.**
- **Explainer content** can be live, not just prose: feature-importance bars from
  the trained model, a single rendered decision tree, bagging/boosting
  animations, a reliability/calibration diagram (already computed). Building it
  *after* the Phase-2 bake-off means it documents models that actually exist
  (XGBoost/MLP) instead of hypotheticals — otherwise the page gets written twice.
- **Neural network — how & constraints:**
  - **Cheapest real NN = `sklearn.MLPClassifier`**, which is *already on the
    Phase-2 list (item 8)* and adds **no new dependency** and no scaler change
    (StandardScaler is already in the pipeline). That's the pragmatic "make it a
    neural net" path.
  - **A deep NN (PyTorch/Keras)** is possible but low-upside here: the data is
    **small tabular** (~34 features; tournament games ≈ dozens/yr, the general
    game model has more but still modest). On small tabular data, gradient-boosted
    trees (XGBoost) typically **beat** NNs, and a deep net mainly adds overfitting
    risk + a heavy deploy dependency (torch wheels) for the Streamlit host.
  - **Constraints, ranked:** (1) dataset size / overfitting — the real limiter,
    not tooling; (2) deploy weight if we go beyond sklearn; (3) calibration — NNs
    are also miscalibrated out of the box, same isotonic/Platt fix as the trees.
  - **Verdict to record:** add MLP in the bake-off, report it honestly against the
    trees; treat a full deep-learning rewrite as a labelled experiment, not the
    default.

16\. **Branding UI: theme selector + human-designed logo (by my gf).** New UI to
pick a theme and attach the logo. -- **DONE (2026-07-10) — logo slot awaits the real asset**
- **`branding.py`** centralizes it all: brand name/tagline/icon, 5 accent
  **themes** (Hardwood default, Baseline Blue, Net Green, Buzzer Purple, Classic
  Red), `apply_theme` (runtime CSS injection), `render_logo`, `render_brand_header`.
- **Theme selector** in the sidebar swaps the accent at runtime (recolours primary
  buttons, links, headers, brand mark, accent bar). Native light/dark stays the
  Streamlit menu toggle; `.streamlit/config.toml` primaryColor set to the Hardwood
  default so native widgets stay in step. Switching verified via Streamlit
  `AppTest` (Hardwood→Net Green→Buzzer Purple all inject the right accent).
- **Logo slot** via `st.logo()` — loads `assets/logo.svg`→`logo.png`→placeholder
  (same for the square icon). Ships a placeholder wordmark + icon; **`assets/README.md`
  has the drop-in spec for gf's real logo** (SVG + transparent PNG, light+dark
  variants, square crop). Drop the real files in and they take over — no code change.
- **Still needs:** gf's actual logo files (the one external dependency); optional
  richer theming (full palettes / dark presets) later.

---

**REORDERED PLAN (cascade-aware, 2026-07-10)** — sequenced to minimize rework.
Guiding rule: anything that changes **model probabilities** (calibration, new
models) ripples into betting numbers, the contest scoreboard, and every prob shown
— so do model work *before* building new surfaces that display those numbers; and
do all **branding** (rename + theme + logo) as one final sweep after the new pages
exist. Start the gf's **logo** now regardless (external lead time).

1. **Finish Phase-1 betting backlog** — by-seed / by-round slices in the strategy
   lab (small, already scaffolded). -- **DONE (2026-07-10)**
2. **Phase 2: model bake-off + calibration (item 8)** — the pivotal unblocker.
   Calibrating/adding models (incl. **MLP** = the "neural net") changes every
   downstream number, so it comes before the new display pages. *In parallel:
   email gf the logo spec (item 16).*
   - **Sub-step a — bake-off + calibration metrics + app page: DONE (2026-07-10).**
   - **Sub-step b — calibrated probs → betting backtest: DONE (2026-07-10).**
     Edges partly survive; ¼-Kelly improves everywhere; claimed edge unmoved
     because the backtest model was already well-calibrated.
   - **Sub-step c — bracket-pool points per model: DONE (2026-07-10).** Random
     Forest is the only model that beats a chalk bracket; MLP is worst.
3. **Learning page (item 15)** — write it against the models that now exist.
   -- **DONE (2026-07-10).** "How the Models Work" page; explainers reference the
   bake-off numbers.
4. **Me-vs-Machine contest (item 14)** — build once on calibrated probs. Ship
   manual-entry mode first; wire the schedule feed when `2027_super_sked.csv`
   lands (~Nov 2026). -- **Manual-entry mode DONE (2026-07-10)**; auto-schedule +
   recurring prompt + Kelly/+EV betting deferred to the data phase.
5. **Branding pass (items 13 + 16 + Phase-3 item 9)** — rename to **B.O.B.**,
   theme selector, drop in gf's logo; theme every page in one sweep.
   -- **DONE (2026-07-10).** Renamed to B.O.B.; `branding.py` with 5 accent themes +
   a sidebar theme selector; logo slot with placeholder awaiting gf's real asset
   (spec in `assets/README.md`).
6. **Parked on data** (items 10/11/12): 2027 field, scheduled-games feed, extend
   odds to 2022–2025.

---

**NEXT SESSION — START HERE (state as of 2026-07-10)**

The whole reordered plan (items 1–5) is **shipped and merged to `main`** (PRs
#2–#6). The app is renamed **B.O.B.** Everything left is data-gated or external —
concrete resumable actions, roughly in order of "can do now" → "waits on data":

1. **Biggest modelling lever (no blocker) — train the tournament model on the
   ~103k regular-season games** in `data/games.csv`. Today the tournament model
   (`mm_model`) only sees ~1k tournament games; `game_model` already uses the big
   log. This is the structural fix for the overfitting the bake-off exposed (esp.
   the MLP, log-loss 1.69) and the real prerequisite for a genuine deep neural
   net. Everything downstream (betting numbers, contest, bake-off) would re-run
   off the better probabilities — treat like Phase 2's cascade.
2. **Drop in the real logo** whenever gf's asset is ready: add
   `assets/logo.svg`/`.png` + `assets/icon.svg`/`.png` per `assets/README.md`;
   they auto-override the placeholders, no code change.
3. **Extend real odds to 2022–2025** (item 12) — source recent closing lines so
   the betting backtests cover the full model era, not just through 2021. Not
   time-gated, just a sourcing task.
4. **~Nov 2026, when `2027_super_sked.csv` publishes:** the schedule-driven
   "upcoming games" view (item 11) **and** the **contest auto-phase** (item 14) —
   automated weekly slate (selection criteria already specced in item 14),
   recurring pick prompt, Kelly/+EV betting on real lines, and durable (non-CSV)
   contest storage.
5. **After Selection Sunday 2027:** add the 2027 field rows to
   `KenPom Barttorvik.csv` and re-run `predict_all_windows.py` (item 10).

**Regenerate the app's data after any source change:** first the fetchers —
`fetch_data.py` (games.csv), `fetch_odds.py` (odds.csv), `update_current_season.py`
(ratings.csv) — then the analysis scripts, which each read those and write their
own `data/*.csv` (order-independent among themselves): `precompute.py`,
`backtest.py`, `backtest_pit.py`, `backtest_odds.py`, `backtest_calibrated.py`,
`model_bakeoff.py`, `bracket_pool.py`, `model_explain.py`, `predict_all_windows.py`.
Smoke-test with `python test_app.py`.

