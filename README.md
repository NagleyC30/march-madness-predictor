# 🏀 March Madness Bracket Predictor

A machine-learning model that predicts NCAA men's basketball tournament games
from **KenPom** and **Barttorvik** team-efficiency ratings, wrapped in an
interactive [Streamlit](https://streamlit.io) dashboard.

The model is validated **walk-forward**: to predict any tournament, it only
trains on seasons that happened *before* it — no looking into the future. Across
2009–2025 it reaches roughly **71% per-game accuracy**.

## ✨ Features

- **Bracket Predictions** — the model's full predicted 2026 bracket and champion.
- **Head-to-Head** — pick any two teams from a season for a live prediction with
  win probabilities and an implied moneyline.
- **Model Accuracy** — walk-forward accuracy by training window, year, and round.
- **Betting Simulation** — hypothetical P&L at several confidence thresholds
  (educational; model-implied odds, not real sportsbook lines).
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
   teams across ~95 metrics (adjusted offense/defense, tempo, shooting,
   rebounding, experience, strength of schedule, …).
3. **Walk-forward training** — a `GridSearchCV` over `RandomForestClassifier` and
   `BaggingClassifier` selects the best model using only prior seasons.
4. **Bracket simulation** — the model advances winners round by round and reports
   each pick's confidence as an implied American moneyline.

## ⚠️ Disclaimer

This is an educational project. The betting figures use the model's own implied
odds as a proxy and are **not** a betting recommendation.
