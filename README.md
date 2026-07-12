# 🏀 B.O.B. — Betting on Basketball

**B.O.B.** is a machine-learning model that predicts NCAA men's basketball games
from **KenPom** and **Barttorvik** team-efficiency ratings, wrapped in an
interactive [Streamlit](https://streamlit.io) dashboard — built to ask one
question: **can it beat the market?**

The model is validated **walk-forward**: to predict any game, it only trains on
games that happened *before* it — no looking into the future. Across 2009–2025 it
reaches roughly **71% per-game accuracy**. The dashboard has a swappable colour
**theme** (pick one in the sidebar) and a logo slot (`assets/` — see its README).

## ✨ Features

- **Bracket Predictions** — the upcoming tournament's bracket predicted by **all
  five training-window models**, with their champions and Final Fours compared
  side by side, plus each model's full bracket.
- **Head-to-Head** — pick any two teams from a season for a live prediction with
  win probabilities and an implied moneyline.
- **Game Predictor** — predict *any* D1 game (regular season or tournament) from
  a separate model trained on ~100k games since 2008. Covers completed seasons
  **and the in-progress 2026–27 season** using Barttorvik's live/preseason
  ratings (refreshed with `python update_current_season.py`).
- **Model Accuracy** — walk-forward accuracy by training window, year, and round.
- **Betting Lab** (the flagship) — hypothetical P&L against **real historical
  closing lines** (2009–2019, 2021): a confidence-threshold sweep (moneyline vs
  flip-to-spread vs model-implied) staking **$100** a bet, plus a **strategy lab**
  that bets **value / +EV** spots (where the model's probability beats the price,
  on either side) with flat vs **fractional-Kelly** staking and compares ROI,
  drawdown and bankroll curves. Educational, not betting advice.
- **Me vs Machine** — an everlasting, blind-locked contest of your picks vs. the
  model's, with a symmetric flat-bet P&L.
- **Custom Metric** — upload your own metric (`YEAR, TEAM, <numeric columns>`) and
  the app retrains with and without it, reporting the accuracy change and the
  metric's **permutation importance** ranked against the built-in features.
- **Data Explorer** — browse the underlying team-season ratings.

## 🗂️ Project layout

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard (the deployed app). |
| `mm_model.py` | Importable, side-effect-free modeling core. |
| `precompute.py` | Runs the heavy walk-forward once, writes result CSVs to `data/`. |
| `predict_all_windows.py` | Forecasts the upcoming bracket with every training window → `data/bracket_all_windows.csv`. |
| `update_current_season.py` | Pulls the in-progress season's Barttorvik ratings into `data/ratings.csv` for the Game Predictor. |
| `fetch_odds.py` | Downloads & parses real historical sportsbook moneylines → `data/odds.csv`. |
| `fetch_cbb_lines.py` | Fetches real spreads/moneylines/totals from CollegeBasketballData.com (free API key) → `data/lines_cbbd.csv`. Real book lines, 2022–present. |
| `backtest_odds.py` | Settles the model's picks at those real odds (moneyline + flip-to-spread); also emits the per-game table `data/bet_games.csv`. |
| `betting_strategies.py` | Strategy lab — value/+EV & Kelly strategies over `bet_games.csv` (imported live by the app). |
| `margin_model.py` | Regression on point margin; bets the spread (ATS) vs real closing lines, point-in-time vs season-aggregate. |
| `data/` | Precomputed results the app loads instantly. |
| `KenPom Barttorvik.csv` | Source team-season efficiency ratings (2008–2026). |
| `requirements.txt` | Python dependencies for Streamlit Community Cloud. |

The dashboards read precomputed CSVs so the app loads instantly on Streamlit
Cloud's limited resources. The **Head-to-Head** predictor and **Custom Metric**
page train a model live (cached per selection).

## 🚀 Run locally

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

python precompute.py             # generate data/*.csv  (one-time, a few minutes)
streamlit run app.py
```

Then open http://localhost:8501.

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (must be **public** for the free tier).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **Create app → Deploy a public app from GitHub**.
4. Select this repo, branch `main`, main file `app.py`, and click **Deploy**.

Streamlit installs `requirements.txt` and serves `app.py` automatically. Commit
the `data/` CSVs (produced by `precompute.py`) so the dashboards work without
re-running the heavy training in the cloud.

## 🔬 How the model works

1. **Reconstruct history** — every tournament game in the dataset is rebuilt into
   a matchup (higher seed vs. lower seed) with the actual winner.
2. **Feature differences** — each game becomes the *difference* between the two
   teams across 96 metrics (adjusted offense/defense, tempo, shooting,
   rebounding, experience, strength of schedule, …).
3. **Walk-forward training** — a `GridSearchCV` over `RandomForestClassifier` and
   `BaggingClassifier` selects the best model using only prior seasons.
4. **Bracket simulation** — the model advances winners round by round and reports
   each pick's confidence as an implied American moneyline.

## 💸 Real-odds betting backtest

Did the model's picks actually make money at real sportsbook prices? To find
out honestly:

```bash
python fetch_odds.py        # data/odds.csv — real closing moneylines
python backtest_odds.py     # data/betting_simulation_real.csv — P&L
```

- **Odds source** — the free [Sportsbook Reviews Online](https://www.sportsbookreviewsonline.com)
  NCAA basketball archive, one row per game with the closing moneyline. It
  covers tournaments **2008–2019 and 2021** (2020 had no tournament; later
  seasons aren't published there). Team names are reconciled to Barttorvik's
  with a normalizer + verified alias map (**100%** of tournament teams matched).
- **Method** — walk-forward with no leakage (same training as `precompute.py`):
  for every game a tournament actually played, the model picks a winner and a
  confidence; at each threshold (−200/−250/−300/−350) it stakes $100 on picks it
  is at least that confident about, **settled at the real closing line and the
  real result**. Games with no matchable line are excluded and reported.
- **Result** — every training-window × threshold combination loses money
  (roughly −2% to −7% ROI). The model's "confident" picks are heavy favorites
  (the average −200 bet is priced near −2800): they pay pennies when they win
  and cost a full unit when they don't, and the model has no edge over the
  closing line. This is the Kelly intuition made concrete — on a huge favorite
  there is essentially nothing worth betting.
- **Flip to spreads?** (`betting_simulation_spreadflip.csv`) — the natural next
  idea is to *take the points* on those heavy favorites: when a pick's real
  moneyline is shorter than the threshold, bet the real closing spread instead
  (at its real price when `odds.csv` carries one, otherwise standard −110). It
  doesn't help — it's **worse** at 19 of 20 window × threshold
  combinations (−5% to −11% ROI). The closing spread is efficient, so a
  near-certain moneyline win becomes a ~coin-flip cover (1165–1170 against the
  spread at −200) and you pay the −110 vig on every one. The market has already
  priced the favorite fairly on both markets.
- **Value / +EV betting** (`betting_strategies.py` over `bet_games.csv`) — the
  real edge test: bet only where the model's probability beats the price (on
  either side), varying the staking. Flat +EV turns a *profit* on the longer
  training windows (up to +12% ROI, concentrated in underdogs and higher-edge
  spots) — the opposite of the chalk result — **but it doesn't hold up**: it's
  negative on the short windows, only ~half the individual tournaments win, the
  model is measurably overconfident on favorites, and fractional-Kelly staking
  (which trusts the edge sizes) loses on every window and craters the bankroll.
  Read as a *lead to validate with a better-calibrated model*, not a proven edge.

## ⚠️ Disclaimer

This is an educational project. The betting figures — both the real-odds
backtest and the implied-odds calibration view — are historical analysis and
are **not** a betting recommendation.
