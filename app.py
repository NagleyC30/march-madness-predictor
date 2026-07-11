# app.py — Streamlit dashboard for B.O.B. (Betting on Basketball).
#
# Dashboards (Predictions / Accuracy / Betting / Data) read precomputed CSVs in
# ./data so they load instantly on Streamlit Community Cloud. The interactive
# Head-to-Head predictor trains a model live, cached per (year, window).

import os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

import mm_model as mm
import game_model as gm
import betting_strategies as bs
import model_explain as me
import contest as ct
import branding as brand

st.set_page_config(
    page_title=f"{brand.APP_NAME} — {brand.APP_TAGLINE}",
    page_icon=brand.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
brand.render_logo()

DATA_DIR = "data"


# ──────────────────────────────────────────────────────────────
# CACHED LOADERS
# ──────────────────────────────────────────────────────────────

@st.cache_data
def load_raw():
    df = mm.load_data()
    return df, mm.get_features(df)


@st.cache_data
def load_results():
    """Load precomputed result CSVs (may be missing before precompute runs)."""
    out = {}
    for key, fname in {
        "wf": "walk_forward_accuracy.csv",
        "bet": "betting_simulation.csv",
        "bet_real": "betting_simulation_real.csv",
        "bet_flip": "betting_simulation_spreadflip.csv",
        "bet_real_meta": "betting_real_meta.csv",
        "summary": "window_summary.csv",
        "bracket2026": "bracket_prediction_2026.csv",
        "bracket_windows": "bracket_all_windows.csv",
        "bracket_windows_meta": "bracket_all_windows_meta.csv",
        "meta": "meta.csv",
    }.items():
        path = os.path.join(DATA_DIR, fname)
        out[key] = pd.read_csv(path) if os.path.exists(path) else None
    return out


@st.cache_data
def load_strategy_lab():
    """Load the per-game predictions table and compute every betting strategy
    live (cheap post-processing). Returns None if the table isn't built yet."""
    path = os.path.join(DATA_DIR, "bet_games.csv")
    if not os.path.exists(path):
        return None
    games = bs.load_games(path)
    summary, equity = bs.run_all(games)
    slices = pd.concat([bs.run_slices(games, "round"),
                        bs.run_slices(games, "seed")], ignore_index=True)
    out = {"games": games, "summary": summary, "equity": equity, "slices": slices}

    # Calibrated twin (backtest_calibrated.py): same bets, isotonic-calibrated
    # probabilities. Lets the lab put raw vs calibrated ROI side by side.
    cal_path = os.path.join(DATA_DIR, "bet_games_calibrated.csv")
    if os.path.exists(cal_path):
        cal_summary, _ = bs.run_all(bs.load_games(cal_path))
        out["cal_summary"] = cal_summary
    return out


@st.cache_data
def load_explainers():
    """Teaching artifacts for the 'How the Models Work' page — precomputed to disk
    by `python model_explain.py` (instant), or computed live as a fallback."""
    return me.load_artifacts()


@st.cache_data
def load_bakeoff():
    """Load the precomputed model bake-off (model_bakeoff.py). Returns None if it
    hasn't been generated yet."""
    s_path = os.path.join(DATA_DIR, "model_bakeoff_summary.csv")
    r_path = os.path.join(DATA_DIR, "model_bakeoff_reliability.csv")
    if not (os.path.exists(s_path) and os.path.exists(r_path)):
        return None
    out = {"summary": pd.read_csv(s_path), "reliability": pd.read_csv(r_path)}
    m_path = os.path.join(DATA_DIR, "model_bakeoff_meta.csv")
    out["meta"] = pd.read_csv(m_path).iloc[0] if os.path.exists(m_path) else None
    p_path = os.path.join(DATA_DIR, "bracket_pool_summary.csv")
    out["pool"] = pd.read_csv(p_path) if os.path.exists(p_path) else None
    rs_path = os.path.join(DATA_DIR, "regseason_compare_summary.csv")
    out["regseason"] = pd.read_csv(rs_path) if os.path.exists(rs_path) else None
    return out


@st.cache_data
def get_matchups():
    df, _ = load_raw()
    rows, years = mm.build_all_matchups(df)
    return pd.DataFrame(rows), years


@st.cache_resource(show_spinner="Training model… (first run only)")
def train_model_cached(target_year, window_name):
    """Train the best model for a given test year + training window. Cached so
    the live predictor only pays the GridSearch cost once per combo."""
    df, features = load_raw()
    rows, years = mm.build_all_matchups(df)
    window_size = mm.TRAINING_WINDOWS[window_name]
    train_years = mm.get_train_years_for_window(years, target_year, window_size)
    train_rows = [r for r in rows if r["YEAR"] in train_years]
    df_train = mm.build_model_dataset(train_rows, df, features)
    model, model_name, cv = mm.train_best_model(df_train, features)
    return model, model_name, cv, train_years


@st.cache_data(show_spinner=False)
def eval_custom_metric_cached(clean_df, metric_cols, target_year, window_name):
    """Train with/without the uploaded metric and score it, cached per upload +
    (year, window). Streamlit hashes the DataFrame, so re-uploading the same file
    reuses the result."""
    return mm.evaluate_custom_metric(
        df_raw, clean_df, tuple(metric_cols), target_year, window_name)


@st.cache_data
def load_game_backtest():
    """Load the precomputed general-model backtest CSVs (from backtest.py)."""
    out = {}
    for key, fname in {
        "summary": "game_backtest_summary.csv",
        "calib": "game_calibration.csv",
        "meta": "game_backtest_meta.csv",
    }.items():
        path = os.path.join(DATA_DIR, fname)
        out[key] = pd.read_csv(path) if os.path.exists(path) else None
    return out


@st.cache_data
def load_game_backtest_pit():
    """Load the point-in-time vs. season-aggregate comparison (backtest_pit.py)."""
    out = {}
    for key, fname in {
        "summary": "game_backtest_pit_summary.csv",
        "calib": "game_calibration_pit.csv",
        "meta": "game_backtest_pit_meta.csv",
    }.items():
        path = os.path.join(DATA_DIR, fname)
        out[key] = pd.read_csv(path) if os.path.exists(path) else None
    return out


@st.cache_resource(show_spinner="Loading game model…")
def load_game_predictor():
    """Load the pre-trained general game model + its ratings table (both built by
    fetch_data.py / game_model.py). Returns (artifact, ratings) or (None, None)."""
    artifact = gm.load_model()
    if artifact is None or not os.path.exists(gm.RATINGS_FILE):
        return None, None
    return artifact, gm.load_ratings()


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:.1%}"


def round_label(code):
    return mm.ROUND_LABELS.get(code, code)


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────

df_raw, FEATURES = load_raw()
results = load_results()
all_years = sorted(df_raw["YEAR"].unique())
completed_years = [y for y in all_years if y not in (2026, 2027)]

st.sidebar.title(f"{brand.APP_ICON} {brand.APP_NAME}")
st.sidebar.caption(
    f"**{brand.APP_TAGLINE}** — can the model beat the market?  \n"
    "Walk-forward ML on KenPom + Barttorvik efficiency ratings, "
    f"{completed_years[0]}–{completed_years[-1]}."
)
brand.theme_selector()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Bracket Predictions", "Head-to-Head", "Game Predictor",
     "Me vs Machine", "Backtest & Calibration", "Model Accuracy", "Model Bake-off",
     "How the Models Work", "Betting Simulation", "Custom Metric",
     "Data Explorer"],
)

if results["meta"] is not None:
    meta = results["meta"].iloc[0]
    st.sidebar.divider()
    st.sidebar.caption(
        f"Best forecasting window: **{meta['best_window']}**  \n"
        f"Historical matchups: **{int(meta['n_historical_matchups'])}**"
    )

st.sidebar.divider()
st.sidebar.caption(
    "⚠️ Educational project. Betting figures — including the real-odds backtest — "
    "are historical analysis, **not** a betting recommendation."
)


# ──────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ──────────────────────────────────────────────────────────────

if page == "Overview":
    brand.render_brand_header()
    st.markdown(
        "**B.O.B. — Betting on Basketball.** A machine-learning model that predicts "
        "NCAA basketball games from **KenPom** and **Barttorvik** team-efficiency "
        "ratings, built to ask one question: **can it beat the market?** It's "
        "validated **walk-forward** — to predict any year, it only ever trains on "
        "games that happened *before* it, no peeking at the future."
    )

    if results["summary"] is not None:
        s = results["summary"].copy()
        best = s.loc[s["indep_acc"].idxmax()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best window accuracy", f"{best['indep_acc']:.1%}",
                  help="Independent per-game accuracy of the strongest training window.")
        c2.metric("Games evaluated", f"{int(best['indep_total']):,}")
        c3.metric("Seasons tested", f"{len([y for y in completed_years if y >= 2009])}")
        c4.metric("Features per game", f"{len(FEATURES)}")
    else:
        st.info("Run `python precompute.py` to generate the results dashboards.")

    st.subheader("How it works")
    st.markdown(
        """
1. **Reconstruct history** — every tournament game from the dataset is rebuilt
   into a matchup (higher seed vs. lower seed) with the actual winner.
2. **Feature differences** — each game becomes the *difference* between the two
   teams across ~95 efficiency metrics (adjusted offense/defense, tempo,
   shooting, rebounding, experience, strength of schedule, …).
3. **Train walk-forward** — a `GridSearchCV` over `RandomForest` and
   `Bagging` classifiers picks the best model using only prior seasons.
4. **Simulate the bracket** — the model advances winners round by round, and
   reports each pick's confidence as an implied moneyline.
        """
    )

    st.subheader("The pages")
    st.markdown(
        """
- **Bracket Predictions** — the model's full predicted bracket (2026) and champion.
- **Head-to-Head** — pick any two *tournament* teams and get a live prediction.
- **Game Predictor** — predict *any* game (regular season or tournament) for any
  two D1 teams, with home-court advantage.
- **Backtest & Calibration** — walk-forward accuracy and calibration of the
  general game model across ~87k games.
- **Model Accuracy** — walk-forward accuracy by training window and tournament round.
- **Betting Simulation** — hypothetical P&L at various confidence thresholds.
- **Custom Metric** — upload your own metric and see how much it helps the model.
- **Data Explorer** — browse the underlying team-season stats.
        """
    )


# ──────────────────────────────────────────────────────────────
# PAGE: BRACKET PREDICTIONS
# ──────────────────────────────────────────────────────────────

elif page == "Bracket Predictions":

    def _round_table(rnd):
        """Matchup/Pick/Implied-line table for one round of one bracket."""
        show = rnd.copy()
        show["Matchup"] = (
            show["TEAM_HIGH"] + " (#" + show["SEED_HIGH"].astype(int).astype(str)
            + ")  vs  " + show["TEAM_LOW"] + " (#"
            + show["SEED_LOW"].astype(int).astype(str) + ")")
        show["Pick"] = (show["PRED_WINNER"] + " (#"
                        + show["PRED_SEED"].astype(int).astype(str) + ")")
        show["Implied line"] = show["IMPLIED_LINE"].apply(lambda v: f"{int(v):+d}")
        st.dataframe(
            show[["REGION", "Matchup", "Pick", "Implied line"]]
            .rename(columns={"REGION": "Region"}),
            hide_index=True, width="stretch",
        )

    bw = results["bracket_windows"]
    bmeta = results["bracket_windows_meta"]

    if bw is not None:
        year = int(bmeta.iloc[0]["target_year"]) if bmeta is not None \
            else int(bw["YEAR"].iloc[0])
        best = bmeta.iloc[0]["best_window"] if bmeta is not None else "all_prior"
        windows = [w for w in mm.TRAINING_WINDOWS if w in set(bw["WINDOW"])]

        st.title(f"Predicted Bracket — {year}")
        st.markdown(
            "Every one of the five **training-window** models applied to the same "
            f"{year} field. Shorter windows react to recent seasons; `all_prior` "
            "uses everything. Where they disagree is where the pick is least certain."
        )

        # ---- Champion & Final Four by model ----
        champ_all = bw[bw["ROUND"] == "Championship"]
        f4_all = bw[bw["ROUND"] == "F4"]
        rows = []
        for w in windows:
            c = champ_all[champ_all["WINDOW"] == w]
            ff = f4_all[f4_all["WINDOW"] == w]
            ff_teams = sorted(set(ff["TEAM_HIGH"]) | set(ff["TEAM_LOW"]))
            rows.append({
                "Training window": w,
                "Model": c["MODEL"].iloc[0] if not c.empty else "—",
                "Predicted champion": (
                    f"{c.iloc[0]['PRED_WINNER']} (#{int(c.iloc[0]['PRED_SEED'])})"
                    if not c.empty else "—"),
                "Final Four": ", ".join(ff_teams),
            })
        st.subheader("Champion & Final Four by model")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        n_champs = champ_all.groupby("WINDOW").first()["PRED_WINNER"].nunique()
        if n_champs == 1:
            st.success(f"All five models agree on the champion: "
                       f"**{champ_all.iloc[0]['PRED_WINNER']}**.")
        else:
            st.info(f"The five models name **{n_champs} different champions** — "
                    "see how each fills the bracket below.")

        # ---- Full bracket for a chosen window ----
        st.subheader("Full bracket by model")
        wsel = st.selectbox(
            "Training window", windows,
            index=windows.index(best) if best in windows else 0,
            help="`" + best + "` has the best walk-forward accuracy.")
        sub = bw[bw["WINDOW"] == wsel]
        csel = sub[sub["ROUND"] == "Championship"]
        if not csel.empty:
            r = csel.iloc[0]
            st.success(f"### 🏆 {wsel} champion: **{r['PRED_WINNER']}** "
                       f"(#{int(r['PRED_SEED'])} seed)")
        st.caption(f"Model: {sub['MODEL'].iloc[0]} · "
                   f"trained on {int(sub['N_TRAIN_YEARS'].iloc[0])} prior seasons")
        for code in mm.ALL_ROUND_NAMES:
            rnd = sub[sub["ROUND"] == code]
            if rnd.empty:
                continue
            with st.expander(f"{round_label(code)}  ({len(rnd)} games)",
                             expanded=code in ("F4", "Championship")):
                _round_table(rnd)
        st.download_button(
            "⬇️ Download all brackets (CSV)",
            bw.to_csv(index=False).encode(),
            "bracket_all_windows.csv", "text/csv",
        )

    elif results["bracket2026"] is not None:
        # Fallback: single best-window bracket from precompute.py.
        bracket = results["bracket2026"]
        st.title("Predicted Bracket — 2026")
        champ = bracket[bracket["ROUND"] == "Championship"]
        if not champ.empty:
            row = champ.iloc[0]
            st.success(
                f"### 🏆 Predicted National Champion: **{row['PRED_WINNER']}** "
                f"(#{int(row['PRED_SEED'])} seed)")
        st.caption(
            f"Model: {bracket['MODEL'].iloc[0]} · training window: "
            f"{bracket['BEST_WINDOW'].iloc[0]}  ·  run "
            "`python predict_all_windows.py` to compare all five windows.")
        for code in mm.ALL_ROUND_NAMES:
            rnd = bracket[bracket["ROUND"] == code]
            if rnd.empty:
                continue
            with st.expander(f"{round_label(code)}  ({len(rnd)} games)",
                             expanded=code in ("F4", "Championship")):
                _round_table(rnd)
        st.download_button(
            "⬇️ Download bracket (CSV)",
            bracket.to_csv(index=False).encode(),
            "bracket_prediction_2026.csv", "text/csv",
        )
    else:
        st.warning("No precomputed bracket found. Run "
                   "`python predict_all_windows.py` first.")


# ──────────────────────────────────────────────────────────────
# PAGE: HEAD-TO-HEAD
# ──────────────────────────────────────────────────────────────

elif page == "Head-to-Head":
    st.title("Head-to-Head Predictor")
    st.markdown(
        "Pick any two teams from a season and the model predicts the winner. "
        "The model is trained walk-forward using seasons **before** the one you pick."
    )

    c1, c2 = st.columns([1, 1])
    year = c1.selectbox("Season", all_years, index=all_years.index(2026)
                        if 2026 in all_years else len(all_years) - 1)
    window = c2.selectbox("Training window", list(mm.TRAINING_WINDOWS.keys()),
                          index=0,
                          help="Which prior seasons the model learns from.")

    year_df = df_raw[df_raw["YEAR"] == year].copy()
    teams = sorted(year_df["TEAM"].unique())
    if len(teams) < 2:
        st.warning(f"Not enough teams in {year}.")
    else:
        c3, c4 = st.columns(2)
        team_a = c3.selectbox("Team A", teams, index=0)
        team_b = c4.selectbox("Team B", teams, index=1)

        if team_a == team_b:
            st.info("Pick two different teams.")
        else:
            seed_a = int(year_df[year_df["TEAM"] == team_a]["SEED"].iloc[0])
            seed_b = int(year_df[year_df["TEAM"] == team_b]["SEED"].iloc[0])

            train_years = mm.get_train_years_for_window(
                completed_years, year, mm.TRAINING_WINDOWS[window])
            if not train_years:
                st.warning("No prior seasons available to train on for this selection.")
            else:
                model, model_name, cv, used_years = train_model_cached(year, window)
                winner, w_seed, p_high = mm.predict_game_proba(
                    team_a, seed_a, team_b, seed_b, year_df, model, FEATURES)

                # Map probability back to each team
                high_team = team_a if seed_a <= seed_b else team_b
                p_a = p_high if team_a == high_team else 1 - p_high
                p_b = 1 - p_a

                st.divider()
                st.subheader(f"Prediction: **{winner}** advances")
                cc1, cc2 = st.columns(2)
                cc1.metric(f"{team_a} (#{seed_a})", f"{p_a:.1%}",
                           delta="WIN" if winner == team_a else None)
                cc2.metric(f"{team_b} (#{seed_b})", f"{p_b:.1%}",
                           delta="WIN" if winner == team_b else None)

                line = mm.prob_to_american_odds(max(p_a, p_b))
                st.caption(
                    f"Model implied moneyline on **{winner}**: `{line:+d}` · "
                    f"{model_name} · CV accuracy {cv:.1%} · "
                    f"trained on {used_years[0]}–{used_years[-1]}"
                )

                chart_df = pd.DataFrame({
                    "Team": [team_a, team_b],
                    "Win probability": [p_a, p_b],
                })
                st.altair_chart(
                    alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X("Win probability:Q", scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(format="%")),
                        y=alt.Y("Team:N", sort="-x"),
                        color=alt.Color("Team:N", legend=None),
                        tooltip=["Team", alt.Tooltip("Win probability:Q", format=".1%")],
                    ).properties(height=140),
                    width="stretch",
                )


# ──────────────────────────────────────────────────────────────
# PAGE: GAME PREDICTOR  (any matchup, regular season or tournament)
# ──────────────────────────────────────────────────────────────

elif page == "Game Predictor":
    st.title("Game Predictor")
    st.markdown(
        "Predict **any** college-basketball game — regular season or tournament, "
        "including the **in-progress season**. Pick a season and two teams, choose "
        "who's at home, and the model gives each side's win probability. It's "
        "trained on **every D1 game since 2008** (~100k games) using Barttorvik "
        "efficiency ratings plus home court."
    )

    artifact, ratings = load_game_predictor()
    if artifact is None:
        st.warning(
            "No trained game model found. Generate the data and model locally, "
            "then redeploy:\n\n"
            "```\npython fetch_data.py 2008 2026\npython game_model.py\n```"
        )
    else:
        model = artifact["model"]
        meta = artifact["meta"]
        features = meta["features"]
        seasons = sorted(ratings["YEAR"].unique(), reverse=True)
        # Any season beyond the model's training range is the in-progress one —
        # its ratings are Barttorvik's live/preseason projections, not final.
        in_progress_year = (seasons[0] if seasons and seasons[0] > meta["year_max"]
                            else None)

        def _season_label(y):
            return f"{y} — in progress" if y == in_progress_year else str(y)

        c1, c2 = st.columns([1, 2])
        season = c1.selectbox("Season", seasons, index=0,
                              format_func=_season_label)
        ryear = ratings[ratings["YEAR"] == season]

        if season == in_progress_year:
            st.info(
                f"**{season - 1}-{str(season)[-2:]} is in progress.** These are "
                "Barttorvik's live/preseason projected ratings, which refresh as "
                "games are played — re-run `python update_current_season.py` to pull "
                "the latest. The model itself is unchanged (trained on completed "
                f"seasons through {meta['year_max']}); it's forecasting from the new "
                "season's ratings. Head-to-head works now; game-by-game schedule "
                "prediction will follow once Barttorvik publishes the schedule."
            )
        teams = sorted(ryear["TEAM"].unique())

        c3, c4 = st.columns(2)
        team_a = c3.selectbox("Team A", teams, index=0)
        team_b = c4.selectbox("Team B", teams,
                              index=1 if len(teams) > 1 else 0)

        loc_label = st.radio(
            "Where is it played?",
            [f"🏠 {team_a} home", "⚖️ Neutral court", f"🏠 {team_b} home"],
            index=1, horizontal=True,
        )
        location = ("A" if loc_label.endswith(f"{team_a} home")
                    else "B" if loc_label.endswith(f"{team_b} home") else "N")

        if team_a == team_b:
            st.info("Pick two different teams.")
        else:
            probs = gm.win_probabilities(
                team_a, team_b, location, ryear, model, features)
            if probs is None:
                st.warning("One of these teams has no ratings for this season.")
            else:
                p_a, p_b = probs
                winner = team_a if p_a >= p_b else team_b
                line = mm.prob_to_american_odds(max(p_a, p_b))

                st.divider()
                st.subheader(f"Prediction: **{winner}** wins")
                cc1, cc2 = st.columns(2)
                cc1.metric(team_a, fmt_pct(p_a),
                           delta="WIN" if winner == team_a else None)
                cc2.metric(team_b, fmt_pct(p_b),
                           delta="WIN" if winner == team_b else None)

                venue = ("neutral court" if location == "N"
                         else f"{team_a if location == 'A' else team_b} at home")
                st.caption(
                    f"Implied moneyline on **{winner}**: `{line:+d}` · {venue} · "
                    f"model CV accuracy {fmt_pct(meta['cv_accuracy'])} on "
                    f"{meta['n_games']:,} games ({meta['year_min']}–{meta['year_max']})"
                )

                chart_df = pd.DataFrame(
                    {"Team": [team_a, team_b], "Win probability": [p_a, p_b]})
                st.altair_chart(
                    alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X("Win probability:Q",
                                scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(format="%")),
                        y=alt.Y("Team:N", sort="-x"),
                        color=alt.Color("Team:N", legend=None),
                        tooltip=["Team",
                                 alt.Tooltip("Win probability:Q", format=".1%")],
                    ).properties(height=140),
                    width="stretch",
                )
                st.caption(
                    "Flip the home/away toggle to see how much home court moves "
                    "the line — it's worth roughly 3–4 points on a neutral game."
                )


# ──────────────────────────────────────────────────────────────
# PAGE: ME VS MACHINE  (the everlasting contest — manual entry)
# ──────────────────────────────────────────────────────────────

elif page == "Me vs Machine":
    st.title("🥊 Me vs. the Machine")
    st.markdown(
        "An everlasting contest: for every game, **you** pick a winner and so does "
        "the **model** — then we track who's better. This is the **manual-entry** "
        "version (you enter games and results by hand); an automated weekly slate "
        "arrives once the 2026–27 schedule publishes."
    )
    with st.expander("The rules (why this stays honest)", expanded=False):
        st.markdown(
            "- **Blind entry.** You lock your pick *before* the model's pick is "
            "shown — the machine's choice is revealed only after you commit.\n"
            "- **Locked together, timestamped.** Both picks are written to "
            "`data/contest_picks.csv` the instant you lock in, so neither can be "
            "edited after the fact.\n"
            "- **Settle only after the game.** You enter the real winner later; "
            "scores never touch a pick before it's locked.\n"
            "- **Betting is a fair, symmetric flat bet.** If you enter the market "
            "moneylines, each side flat-bets *its own* pick — the only "
            "apples-to-apples wager when you give a pick but not a probability. "
            "Richer strategies (Kelly, +EV) come with the real-odds feed later."
        )

    artifact, ratings = load_game_predictor()
    if artifact is None:
        st.warning(
            "The contest needs the trained game model. Run `python -c \"import "
            "game_model; game_model.save_model()\"` locally to build it."
        )
    else:
        model = artifact["model"]
        meta = artifact["meta"]
        features = meta["features"]

        # ── Make your pick (blind) ──────────────────────────────────
        st.subheader("Make your pick")
        seasons = sorted(ratings["YEAR"].unique(), reverse=True)
        in_progress_year = (seasons[0] if seasons and seasons[0] > meta["year_max"]
                            else None)
        cA, cB = st.columns([1, 2])
        season = cA.selectbox(
            "Season", seasons, index=0,
            format_func=lambda y: f"{y} (in progress)" if y == in_progress_year else str(y))
        ryear = ratings[ratings["YEAR"] == season]
        teams = sorted(ryear["TEAM"].unique())

        c1, c2 = st.columns(2)
        team_a = c1.selectbox("Team A", teams, index=0, key="mvm_a")
        team_b = c2.selectbox("Team B", teams, index=1 if len(teams) > 1 else 0,
                              key="mvm_b")
        loc_label = st.radio(
            "Where is it played?",
            [f"🏠 {team_a} home", "⚖️ Neutral court", f"🏠 {team_b} home"],
            index=1, horizontal=True, key="mvm_loc")
        location = ("A" if loc_label.endswith(f"{team_a} home")
                    else "B" if loc_label.endswith(f"{team_b} home") else "N")

        if team_a == team_b:
            st.info("Pick two different teams.")
        else:
            user_pick = st.radio("**Your pick to win**", [team_a, team_b],
                                 horizontal=True, key="mvm_pick")
            with st.expander("Optional: confidence & market moneylines"):
                user_conf = st.slider("How confident are you?", 0.50, 0.99, 0.60,
                                      help="Stored for later; doesn't affect the W-L record.")
                mcols = st.columns(2)
                ml_a = mcols[0].number_input(f"{team_a} moneyline", value=0, step=5,
                                             help="0 = no line. Enter the real line to settle bets.")
                ml_b = mcols[1].number_input(f"{team_b} moneyline", value=0, step=5)

            if st.button("🔒 Lock in my pick — then reveal the model's", type="primary"):
                probs = gm.win_probabilities(team_a, team_b, location, ryear,
                                             model, features)
                if probs is None:
                    st.warning("One of these teams has no ratings for this season.")
                else:
                    p_a, p_b = probs
                    model_pick = team_a if p_a >= p_b else team_b
                    model_prob = max(p_a, p_b)
                    ct.add_pick(
                        season, team_a, team_b, location, user_pick,
                        model_pick, model_prob, user_conf=round(user_conf, 2),
                        ml_a=(ml_a or None), ml_b=(ml_b or None))
                    agree = model_pick == user_pick
                    st.success(f"Locked **{user_pick}**. 🤖 The model picked "
                               f"**{model_pick}** ({fmt_pct(model_prob)}) — "
                               f"{'you agree 🤝' if agree else 'you disagree ⚔️'}.")
                    st.caption("Enter the result below once the game is played.")

        # ── Awaiting results ────────────────────────────────────────
        picks = ct.load_picks()
        pending = picks[picks["status"] == "pending"]
        st.divider()
        st.subheader(f"Awaiting results ({len(pending)})")
        if pending.empty:
            st.caption("No open games. Lock in a pick above.")
        else:
            for r in pending.itertuples(index=False):
                venue = ("neutral" if r.location == "N"
                         else f"{r.team_a} home" if r.location == "A"
                         else f"{r.team_b} home")
                gc1, gc2, gc3 = st.columns([3, 2, 1])
                gc1.markdown(f"**{r.team_a}** vs **{r.team_b}**  \n"
                             f"<small>{r.season} · {venue} · your pick: "
                             f"{r.user_pick}</small>", unsafe_allow_html=True)
                winner = gc2.selectbox("Winner", [r.team_a, r.team_b],
                                       key=f"win_{r.id}", label_visibility="collapsed")
                if gc3.button("Settle", key=f"settle_{r.id}"):
                    ct.settle_pick(r.id, winner)
                    st.rerun()

        # ── Scoreboard ──────────────────────────────────────────────
        sb = ct.scoreboard(picks)
        st.divider()
        st.subheader("🏆 Scoreboard")
        if sb["n_settled"] == 0:
            st.info("No settled games yet — lock in some picks and record the "
                    "results to start the contest.")
        else:
            m1, m2 = st.columns(2)
            m1.metric("🤖 Model record",
                      f"{sb['model_w']}–{sb['model_l']}",
                      f"{sb['model_w'] / sb['n_settled']:.0%} correct")
            m2.metric("🧑 Your record",
                      f"{sb['user_w']}–{sb['user_l']}",
                      f"{sb['user_w'] / sb['n_settled']:.0%} correct")

            if sb["disagree"] > 0:
                lead = ("**you** are" if sb["disagree_user_right"] > sb["disagree_model_right"]
                        else "**the model** is" if sb["disagree_model_right"] > sb["disagree_user_right"]
                        else "you're **even**")
                st.markdown(
                    f"**The stat that matters — when you disagree** "
                    f"({sb['disagree']} game{'s' if sb['disagree'] != 1 else ''}): "
                    f"you were right **{sb['disagree_user_right']}**, the model "
                    f"**{sb['disagree_model_right']}**. On the games you saw "
                    f"differently, {lead} ahead. (You agreed on {sb['agree']}.)")
            else:
                st.caption("You've agreed with the model on every settled game so "
                           "far — no head-to-head disagreements yet.")

            if sb["n_user_bets"] or sb["n_model_bets"]:
                b1, b2 = st.columns(2)
                b1.metric(f"🤖 Model flat-bet P&L ({sb['n_model_bets']} bets)",
                          f"${sb['model_pnl']:+,.2f}")
                b2.metric(f"🧑 Your flat-bet P&L ({sb['n_user_bets']} bets)",
                          f"${sb['user_pnl']:+,.2f}")
                st.caption(f"Flat ${ct.STAKE:,.0f} on each pick, at the moneylines "
                           "you entered. Only games with a line count.")

        settled = picks[picks["status"] == "settled"]
        if not settled.empty:
            with st.expander(f"History ({len(settled)} settled)"):
                hist = settled[["season", "team_a", "team_b", "user_pick",
                                "model_pick", "actual_winner"]].rename(columns={
                    "season": "Season", "team_a": "Team A", "team_b": "Team B",
                    "user_pick": "Your pick", "model_pick": "Model pick",
                    "actual_winner": "Winner"})
                st.dataframe(hist[::-1], hide_index=True, width="stretch")

        st.caption("Picks are stored in `data/contest_picks.csv`. On an ephemeral "
                   "cloud host this resets on restart — durable storage is a later "
                   "upgrade.")


# ──────────────────────────────────────────────────────────────
# PAGE: BACKTEST & CALIBRATION  (general game model)
# ──────────────────────────────────────────────────────────────

elif page == "Backtest & Calibration":
    st.title("Game Model — Backtest & Calibration")
    bt = load_game_backtest()
    if bt["meta"] is None:
        st.warning(
            "No backtest data found. Generate it locally, then redeploy:\n\n"
            "```\npython fetch_data.py 2008 2026\npython backtest.py\n```"
        )
    else:
        meta = bt["meta"].iloc[0]
        summary = bt["summary"]
        calib = bt["calib"]

        st.markdown(
            "How well does the Game Predictor actually do? This is a "
            "**walk-forward** backtest: for each season the model is trained "
            "*only on earlier seasons*, then asked to predict that season's "
            "games — so it never sees the future it's tested on."
        )

        c1, c2, c3, c4 = st.columns(4)
        delta = meta["accuracy"] - meta["base_home_acc"]
        c1.metric("Accuracy", fmt_pct(meta["accuracy"]),
                  delta=f"{delta:+.1%} vs. always-home",
                  help="Share of games where the model's favorite actually won.")
        c2.metric("Brier score", f"{meta['brier']:.3f}",
                  help="Mean squared error of the probabilities (lower is better; "
                       "0 is perfect, 0.25 is a coin flip).")
        c3.metric("Log loss", f"{meta['log_loss']:.3f}",
                  help="Penalizes confident wrong calls (lower is better).")
        c4.metric("Games evaluated", f"{int(meta['n_games']):,}",
                  help=f"{int(meta['season_min'])}–{int(meta['season_max'])} "
                       f"({int(meta['n_seasons'])} seasons)")

        st.info(
            "⚠️ Ratings are **season-aggregate**, so a game's features already "
            "reflect that game — in-season accuracy is a little optimistic. "
            "Point-in-time ratings would tighten this; walk-forward training "
            "already removes the separate 'training on the future' bias."
        )

        st.subheader("Accuracy over time")
        by_year = summary.groupby("YEAR")[["correct", "n"]].sum()
        by_year["accuracy"] = by_year["correct"] / by_year["n"]
        by_year = by_year.reset_index()
        line = alt.Chart(by_year).mark_line(point=True).encode(
            x=alt.X("YEAR:O", title="Season"),
            y=alt.Y("accuracy:Q", title="Accuracy",
                    scale=alt.Scale(domain=[0.5, 0.9]), axis=alt.Axis(format="%")),
            tooltip=["YEAR", alt.Tooltip("accuracy:Q", format=".1%"), "n"],
        )
        base = alt.Chart(pd.DataFrame({"y": [meta["base_home_acc"]]})).mark_rule(
            strokeDash=[4, 4], color="gray").encode(y="y:Q")
        st.altair_chart((line + base).properties(height=320), width="stretch")
        st.caption("Dashed line = always-pick-home baseline. The dip around "
                   "2021 reflects the empty-arena COVID season.")

        st.subheader("Accuracy by game type")
        by_type = summary.groupby("GAME_TYPE")[["correct", "n"]].sum()
        by_type["accuracy"] = by_type["correct"] / by_type["n"]
        by_type = by_type.reset_index()
        type_labels = {"nonconf": "Non-conference", "conf": "Conference",
                       "conf_tourney": "Conf. tournament", "postseason": "Postseason"}
        by_type["label"] = by_type["GAME_TYPE"].map(type_labels)
        st.altair_chart(
            alt.Chart(by_type).mark_bar().encode(
                x=alt.X("accuracy:Q", title="Accuracy",
                        scale=alt.Scale(domain=[0.5, 0.9]), axis=alt.Axis(format="%")),
                y=alt.Y("label:N", title=None, sort="-x"),
                tooltip=["label", alt.Tooltip("accuracy:Q", format=".1%"), "n"],
                color=alt.value("#E8590C"),
            ).properties(height=200), width="stretch")
        st.caption("Non-conference games (more mismatches) are easiest; "
                   "postseason games (evenly matched) are hardest.")

        st.subheader("Calibration")
        st.markdown(
            "When the model says a team has a **70% chance**, does it win about "
            "70% of the time? Points on the diagonal mean the probabilities are "
            "trustworthy."
        )
        group_labels = {"all": "All games", "home": "Home/away games only",
                        "neutral": "Neutral-court games only",
                        "nonconf": "Non-conference", "conf": "Conference",
                        "conf_tourney": "Conf. tournament", "postseason": "Postseason"}
        avail = [g for g in group_labels if g in set(calib["group"])]
        pick = st.selectbox("Show calibration for", avail,
                            format_func=lambda g: group_labels.get(g, g))
        cg = calib[calib["group"] == pick].copy()
        cg["Predicted"] = cg["sum_pred"] / cg["n"]
        cg["Actual"] = cg["sum_actual"] / cg["n"]

        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            strokeDash=[5, 5], color="gray").encode(x="x:Q", y="y:Q")
        pts = alt.Chart(cg).mark_circle(opacity=0.85).encode(
            x=alt.X("Predicted:Q", title="Model predicted P(home win)",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Actual:Q", title="Observed home win rate",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            size=alt.Size("n:Q", title="Games", scale=alt.Scale(range=[20, 500])),
            tooltip=[alt.Tooltip("Predicted:Q", format=".1%"),
                     alt.Tooltip("Actual:Q", format=".1%"), "n"],
            color=alt.value("#1C6DD0"),
        )
        st.altair_chart((diag + pts).properties(height=400), width="stretch")
        st.caption("Bubble size = number of games in that probability range.")

        # ── Point-in-time vs. season-aggregate comparison ──────────────
        pit = load_game_backtest_pit()
        if pit["meta"] is not None:
            pm = pit["meta"].iloc[0]
            st.divider()
            st.subheader("Point-in-time reality check")
            st.markdown(
                "The accuracy above uses **season-aggregate** ratings, which "
                "already reflect each game's result. Below is the *same* "
                "walk-forward backtest on the *same* well-covered games, but "
                "comparing **point-in-time** ratings (as they stood the morning "
                "of each game, from Barttorvik's time machine) against "
                "season-aggregate ratings. The gap is the optimism."
            )
            gap = pm["acc_agg"] - pm["acc_pit"]
            g1, g2, g3 = st.columns(3)
            g1.metric("Point-in-time accuracy", fmt_pct(pm["acc_pit"]),
                      help="Honest: ratings as they were before each game.")
            g2.metric("Season-aggregate accuracy", fmt_pct(pm["acc_agg"]),
                      help="Optimistic: end-of-season ratings.")
            g3.metric("Optimism gap", f"{gap:+.1%}",
                      delta=f"{gap:+.1%}", delta_color="inverse",
                      help="How much season-aggregate ratings inflate accuracy.")
            st.caption(
                f"On {int(pm['n_games']):,} games, {int(pm['season_min'])}–"
                f"{int(pm['season_max'])}. Brier {pm['brier_pit']:.3f} "
                f"(point-in-time) vs {pm['brier_agg']:.3f} (season-aggregate)."
            )

            summ = pit["summary"]
            by_year = summ.groupby("YEAR")[
                ["correct_pit", "correct_agg", "n"]].sum()
            by_year["Point-in-time"] = by_year["correct_pit"] / by_year["n"]
            by_year["Season-aggregate"] = by_year["correct_agg"] / by_year["n"]
            long = by_year.reset_index().melt(
                id_vars="YEAR", value_vars=["Point-in-time", "Season-aggregate"],
                var_name="Ratings", value_name="accuracy")
            st.altair_chart(
                alt.Chart(long).mark_line(point=True).encode(
                    x=alt.X("YEAR:O", title="Season"),
                    y=alt.Y("accuracy:Q", title="Accuracy",
                            scale=alt.Scale(domain=[0.6, 0.85]),
                            axis=alt.Axis(format="%")),
                    color=alt.Color("Ratings:N", title=None,
                                    scale=alt.Scale(
                                        domain=["Point-in-time", "Season-aggregate"],
                                        range=["#1C6DD0", "#E8590C"])),
                    tooltip=["YEAR", "Ratings",
                             alt.Tooltip("accuracy:Q", format=".1%")],
                ).properties(height=320), width="stretch")
            st.caption(
                "Season-aggregate (red) sits above point-in-time (blue) every "
                "year — that vertical gap is the leakage from letting a game see "
                "its team's final rating."
            )


# ──────────────────────────────────────────────────────────────
# PAGE: MODEL ACCURACY
# ──────────────────────────────────────────────────────────────

elif page == "Model Accuracy":
    st.title("Walk-Forward Model Accuracy")
    wf = results["wf"]
    summary = results["summary"]
    if wf is None or summary is None:
        st.warning("No precomputed accuracy data. Run `python precompute.py` first.")
    else:
        st.markdown(
            "**Independent accuracy** asks the model about every game that actually "
            "happened. **Cascade accuracy** only counts games where the simulated "
            "bracket sent the same two teams as reality (so it compounds earlier "
            "mistakes). Both are shown."
        )

        st.subheader("Accuracy by training window")
        s = summary.copy().sort_values("indep_acc", ascending=False)
        s_disp = s[["window", "indep_acc", "indep_correct", "indep_total",
                    "cascade_acc"]].rename(columns={
            "window": "Training window", "indep_acc": "Independent acc",
            "indep_correct": "Correct", "indep_total": "Games",
            "cascade_acc": "Cascade acc"})
        st.dataframe(
            s_disp.style.format({"Independent acc": "{:.1%}", "Cascade acc": "{:.1%}"}),
            hide_index=True, width="stretch",
        )

        st.altair_chart(
            alt.Chart(s).mark_bar().encode(
                x=alt.X("indep_acc:Q", title="Independent accuracy",
                        scale=alt.Scale(domain=[0.5, 0.8]), axis=alt.Axis(format="%")),
                y=alt.Y("window:N", title="Training window", sort="-x"),
                tooltip=[alt.Tooltip("indep_acc:Q", format=".1%")],
                color=alt.value("#E8590C"),
            ).properties(height=200),
            width="stretch",
        )

        st.subheader("Accuracy over time")
        by_year = (wf.groupby(["window", "test_year"])
                   .apply(lambda g: g["indep_correct"].sum() / g["indep_total"].sum()
                          if g["indep_total"].sum() > 0 else np.nan)
                   .reset_index(name="accuracy"))
        sel = st.multiselect("Windows", list(mm.TRAINING_WINDOWS.keys()),
                             default=["all_prior", "last_5_years"])
        plot_df = by_year[by_year["window"].isin(sel)] if sel else by_year
        line = alt.Chart(plot_df).mark_line(point=True).encode(
            x=alt.X("test_year:O", title="Tournament year"),
            y=alt.Y("accuracy:Q", title="Independent accuracy",
                    scale=alt.Scale(domain=[0.4, 0.95]), axis=alt.Axis(format="%")),
            color=alt.Color("window:N", title="Window"),
            tooltip=["window", "test_year", alt.Tooltip("accuracy:Q", format=".1%")],
        )
        baseline = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
            strokeDash=[4, 4], color="gray").encode(y="y:Q")
        st.altair_chart((line + baseline).properties(height=350),
                        width="stretch")

        st.subheader("Accuracy by tournament round")
        by_round = (wf.groupby("round").apply(lambda g: pd.Series({
            "accuracy": g["indep_correct"].sum() / g["indep_total"].sum()
                        if g["indep_total"].sum() > 0 else np.nan,
            "games": int(g["indep_total"].sum()),
        })).reindex(mm.ALL_ROUND_NAMES).reset_index())
        by_round["round_label"] = by_round["round"].map(mm.ROUND_LABELS)
        st.altair_chart(
            alt.Chart(by_round.dropna()).mark_bar().encode(
                x=alt.X("round_label:N", title="Round",
                        sort=[mm.ROUND_LABELS[r] for r in mm.ALL_ROUND_NAMES]),
                y=alt.Y("accuracy:Q", title="Independent accuracy",
                        axis=alt.Axis(format="%")),
                tooltip=["round_label",
                         alt.Tooltip("accuracy:Q", format=".1%"), "games"],
                color=alt.value("#1C6DD0"),
            ).properties(height=300),
            width="stretch",
        )
        st.caption(
            "Early rounds (more games, bigger seed gaps) are easiest; later "
            "rounds have few samples so accuracy is noisier."
        )


# ──────────────────────────────────────────────────────────────
# PAGE: MODEL BAKE-OFF
# ──────────────────────────────────────────────────────────────

elif page == "Model Bake-off":
    st.title("Model Bake-off")
    bake = load_bakeoff()
    if bake is None:
        st.info(
            "The model bake-off hasn't been generated yet. Run "
            "`python model_bakeoff.py` to build it."
        )
    else:
        win = bake["meta"]["window"] if bake["meta"] is not None else "all_prior"
        st.markdown(
            "The tournament model picks between Random Forest and Bagging on "
            "cross-validated **accuracy**. But the betting work showed the model "
            "is **overconfident** — its *picks* can be fine while its "
            "*probabilities* are wrong, and accuracy can't see that. This bakes a "
            "wider roster of classifiers (all scikit-learn, no new dependency) "
            "against each other on **probability quality** — Brier score, "
            "log-loss and calibration error — and tests whether post-hoc "
            "**isotonic calibration** fixes the overconfidence.\n\n"
            f"Walk-forward from 2009, training window **{win}**. Metrics pooled "
            "over every actual tournament game."
        )

        summ = bake["summary"].copy()
        best = summ.sort_values("brier").iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Best Brier score", f"{best['brier']:.4f}",
                  f"{best['model']} ({best['calibrated']})")
        raw_ll = summ[summ["calibrated"] == "raw"]["log_loss"].mean()
        cal_ll = summ[summ["calibrated"] == "isotonic"]["log_loss"].mean()
        c2.metric("Avg log-loss, raw → calibrated",
                  f"{raw_ll:.3f} → {cal_ll:.3f}",
                  f"{(cal_ll - raw_ll):+.3f}", delta_color="inverse")
        c3.metric("Accuracy spread (all variants)",
                  f"{summ['accuracy'].min():.3f}–{summ['accuracy'].max():.3f}",
                  "picks barely differ")

        st.subheader("Every model, ranked by Brier score")
        st.caption("Lower Brier / log-loss / ECE = better-calibrated probabilities. "
                   "**ECE** is the average gap between confidence and reality.")
        disp = summ.rename(columns={
            "model": "Model", "calibrated": "Calibration", "games": "Games",
            "accuracy": "Accuracy", "brier": "Brier", "log_loss": "Log-loss",
            "ece": "ECE"})

        def _shade_low_good(col):
            """Green (best/low) → red (worst/high) shading, no matplotlib."""
            lo, hi = float(col.min()), float(col.max())
            rng = (hi - lo) or 1.0
            out = []
            for v in col:
                t = (float(v) - lo) / rng
                r = int(46 + (214 - 46) * t)
                g = int(160 + (39 - 160) * t)
                b = int(44 + (40 - 44) * t)
                out.append(f"background-color: rgba({r},{g},{b},0.25)")
            return out

        st.dataframe(
            disp.style.format({"Accuracy": "{:.3f}", "Brier": "{:.4f}",
                               "Log-loss": "{:.4f}", "ECE": "{:.4f}"})
            .apply(_shade_low_good, subset=["Brier", "Log-loss", "ECE"]),
            hide_index=True, width="stretch",
        )

        st.subheader("Reliability — is a stated 70% really a 70%?")
        rel = bake["reliability"]
        model_pick = st.selectbox("Model", sorted(rel["model"].unique()),
                                  index=sorted(rel["model"].unique()).index(best["model"])
                                  if best["model"] in set(rel["model"]) else 0)
        rsub = rel[rel["model"] == model_pick].copy()
        rsub["Calibration"] = rsub["calibrated"]
        diag = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
        line = alt.Chart(rsub).mark_line(point=True).encode(
            x=alt.X("mean_pred:Q", title="Model's predicted probability",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("obs_freq:Q", title="Actual win frequency",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Calibration:N"),
            tooltip=["Calibration", alt.Tooltip("mean_pred:Q", format=".2f"),
                     alt.Tooltip("obs_freq:Q", format=".2f"), "count"],
        )
        perfect = alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray").encode(
            x="x:Q", y="y:Q")
        st.altair_chart((perfect + line).properties(height=400), width="stretch")
        st.caption(
            "The dashed line is perfect calibration. Points **below** it mean the "
            "model is overconfident (it says 80% but wins less often); calibration "
            "should pull the curve toward the diagonal."
        )
        st.info(
            "**Takeaway.** The models barely differ on *accuracy*, but a lot on "
            "*calibration*: raw boosting and the MLP (a small neural net) are badly "
            "overconfident on this small, tabular dataset — the MLP is the worst raw "
            "model — while Random Forest is already well-calibrated because bagging "
            "averages its trees' votes. **Isotonic calibration** sharply improves the "
            "overconfident models. Next: feed these calibrated probabilities back "
            "into the betting backtest and see whether the +EV edges survive."
        )

        # ── Which model fills the best bracket? ─────────────────────
        pool = bake.get("pool")
        if pool is not None and not pool.empty:
            st.subheader("Which model fills the best bracket?")
            st.markdown(
                "Accuracy and Brier weight every game equally, but a bracket "
                "**pool** doesn't: calling the champion is worth **320** points, a "
                "first-round game only **10**. Each model's full walk-forward "
                "bracket, scored 10/20/40/80/160/320 by round and summed over "
                "2009–2025 — against a **chalk** baseline that always picks the "
                "higher seed."
            )
            chalk_total = float(pool.loc[pool["model"].str.startswith("Chalk"),
                                         "total"].iloc[0]) if pool["model"].str.startswith("Chalk").any() else None
            pool_disp = pool.copy()
            pool_disp["beats_chalk"] = pool_disp["total"] >= (chalk_total or 0)
            bar = alt.Chart(pool_disp).mark_bar().encode(
                x=alt.X("total:Q", title="Total pool points (2009–2025)"),
                y=alt.Y("model:N", sort="-x", title=None),
                color=alt.Color("beats_chalk:N", title="Beats chalk?",
                                scale=alt.Scale(domain=[True, False],
                                                range=["#2F9E44", "#D6336C"])),
                tooltip=["model", "total", "avg_per_year", "pct_of_max"],
            )
            rule = None
            if chalk_total is not None:
                rule = alt.Chart(pd.DataFrame({"x": [chalk_total]})).mark_rule(
                    strokeDash=[4, 4], color="gray").encode(x="x:Q")
            st.altair_chart((bar + rule) if rule is not None else bar,
                            width="stretch")

            round_cols = ["R64", "R32", "S16", "E8", "F4", "Championship"]
            pd_disp = pool[["model", "total", "avg_per_year", "pct_of_max"] + round_cols].rename(
                columns={"model": "Model", "total": "Total", "avg_per_year": "Avg/yr",
                         "pct_of_max": "% of max"})
            st.dataframe(
                pd_disp.style.format({"Total": "{:.0f}", "Avg/yr": "{:.1f}",
                                      "% of max": "{:.1f}",
                                      **{c: "{:.0f}" for c in round_cols}}),
                hide_index=True, width="stretch",
            )
            st.info(
                "**Takeaway.** Only **Random Forest** clearly beats a chalk bracket; "
                "the **MLP is the worst by far** — the same overfitting that wrecked "
                "its calibration also busts its brackets. The **Championship column "
                "is 0 for everyone**: under the app's cascade scoring a game only "
                "counts if *both* teams in your predicted matchup actually reached it, "
                "which essentially never holds by the final — so deep-round credit is "
                "rare, but the comparison is apples-to-apples across models."
            )

        # ── Does more training data help? ───────────────────────────
        rs = bake.get("regseason")
        if rs is not None and not rs.empty:
            st.subheader("Does more training data help? (the ~90k-game test)")
            st.markdown(
                "The overconfidence above comes partly from training on only ~1,000 "
                "tournament games — so the obvious fix is *more data*. We tested that "
                "directly: same feature space, same model (HistGradientBoosting), same "
                "test games — only the **training set** changes, from ~1,000 "
                "tournament games to the **~90,000-game regular-season log**."
            )
            rd = rs.rename(columns={
                "model": "Training set", "calibrated": "Calibration",
                "games": "Test games", "accuracy": "Accuracy", "brier": "Brier",
                "log_loss": "Log-loss", "ece": "ECE"})
            st.dataframe(
                rd.style.format({"Accuracy": "{:.3f}", "Brier": "{:.4f}",
                                 "Log-loss": "{:.4f}", "ECE": "{:.4f}"})
                .apply(_shade_low_good, subset=["Brier", "Log-loss"]),
                hide_index=True, width="stretch")
            st.warning(
                "**Surprise: more data made it *worse*.** Training on the full "
                "regular-season log **lowered** accuracy (0.80 vs 0.86) and worsened "
                "Brier and log-loss. Regular-season games are a **different "
                "distribution** — blowouts, home courts, mismatched teams — so the "
                "extra volume dilutes the tournament-specific signal instead of "
                "reducing variance. **More data only helps when it's on-distribution.** "
                "(Both regimes share the season-aggregate ratings leakage, so the "
                "*absolute* numbers are optimistic — but the *relative* comparison, "
                "which is the point, holds.)"
            )


# ──────────────────────────────────────────────────────────────
# PAGE: HOW THE MODELS WORK
# ──────────────────────────────────────────────────────────────

elif page == "How the Models Work":
    st.title("How the Models Work")
    st.markdown(
        "A plain-English tour of the classifiers behind the predictions — what "
        "each one actually *does*, why we use several, and whether a neural "
        "network would help. The numbers that rank these models live on the "
        "**Model Bake-off** page; this page explains the machinery."
    )

    st.header("How the model sees a game")
    st.markdown(
        "Every game is reduced to one row of **differences**. For each of the "
        "~90 KenPom/Barttorvik ratings we subtract the lower seed's value from the "
        "higher seed's, and the model learns to predict **`HIGH_SEED_WINS`** (1 if "
        "the better seed wins). So a *positive* number always means \"the "
        "favorite has more of this stat.\" The model never sees team names — only "
        "these rating gaps, which is what lets it generalize to matchups it has "
        "never seen."
    )

    expl = load_explainers()

    st.header("What the model actually looks at")
    st.markdown(
        "Impurity-based feature importance from a Random Forest trained on every "
        "historical tournament game — how often each rating gap is the thing that "
        "decides a split. The efficiency margins dominate; team names, seeds and "
        "conference never enter."
    )
    imp = expl["importance"].copy()
    chart = alt.Chart(imp).mark_bar(color="#1C6DD0").encode(
        x=alt.X("importance:Q", title="Importance"),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
    )
    st.altair_chart(chart.properties(height=420), width="stretch")
    st.caption(
        "**BADJ EM / KADJ EM** are Barttorvik's and KenPom's adjusted efficiency "
        "margins (points per 100 possessions vs. an average team); **WAB** is "
        "\"wins above bubble\"; **BARTHAG** is a power rating. In short: the model "
        "leans on how much better one team is at scoring and preventing scoring."
    )

    st.header("A single decision tree")
    st.markdown(
        "The simplest building block. A tree asks yes/no questions about the "
        "rating gaps, funneling each game down to a prediction. Here's a **real** "
        "depth-3 tree trained on the data (shortened for readability) — read "
        "`BADJ EM <= 11.50` as \"is the favorite's efficiency-margin edge 11.5 or "
        "less?\":"
    )
    st.code(expl["tree"], language="text")
    st.markdown(
        "One tree is easy to read but **brittle** — nudge the data and the splits "
        "jump around (high variance). Every model below is a strategy for taming "
        "that brittleness."
    )

    st.header("The roster, in plain English")
    with st.expander("🌲 Bagging & Random Forest — averaging many trees", expanded=True):
        st.markdown(
            "**Bagging** trains many trees, each on a random resample of the games, "
            "and averages their votes. Averaging cancels the individual trees' "
            "random errors, so the ensemble is far steadier than any one tree. "
            "**Random Forest** adds one twist: at each split a tree may only "
            "consider a random subset of the features, which *decorrelates* the "
            "trees so the averaging helps even more.\n\n"
            "Because the prediction is a **vote share** across hundreds of trees, "
            "Random Forest's probabilities come out naturally well-behaved — which "
            "is exactly why it was the best-**calibrated** and best **bracket-pool** "
            "model in the bake-off."
        )
    with st.expander("🚀 Boosting — HistGradientBoosting / XGBoost"):
        st.markdown(
            "Boosting builds trees **in sequence**, each one focused on the games "
            "the previous trees got wrong. This chases accuracy aggressively and "
            "often ranks games very well — but it also tends to become "
            "**overconfident**, pushing probabilities toward 0 or 1. In the "
            "bake-off, raw HistGradientBoosting was badly miscalibrated until "
            "isotonic calibration reined it in. **XGBoost/LightGBM** are faster, "
            "regularized cousins of the same idea; we've left them out for now to "
            "avoid an extra deploy dependency, since HistGradientBoosting already "
            "represents the family."
        )
    with st.expander("📈 Logistic Regression — the calibrated baseline"):
        st.markdown(
            "A straight-line model: it weights each rating gap and squashes the sum "
            "into a probability. It can't capture interactions the way trees do, so "
            "its accuracy is a touch lower — but its probabilities are honest by "
            "construction, which makes it a valuable sanity-check baseline."
        )
    with st.expander("🧭 SVM — drawing the widest boundary"):
        st.markdown(
            "A Support Vector Machine finds the boundary that separates "
            "favorite-wins from upset games with the **widest possible margin**, "
            "and (with an RBF kernel) can bend that boundary into curves. Solid "
            "accuracy, but its raw scores aren't true probabilities without extra "
            "calibration."
        )
    with st.expander("🧠 MLP — a (small) neural network"):
        st.markdown(
            "A multi-layer perceptron stacks layers of weighted sums and "
            "non-linearities, learning its own feature combinations instead of "
            "being handed them. In principle the most flexible model here — in "
            "practice, on the bake-off it was the **worst**: badly overconfident "
            "and the weakest bracket-filler. See the next section for why, and "
            "whether it's worth pursuing."
        )

    st.header("Could this be a neural network?")
    st.markdown(
        "Yes — and it already partly is (the **MLP** above is a small neural net, "
        "and it needs no extra dependency). The real question is whether a "
        "*bigger, deeper* net would help. On this data, probably not yet:"
    )
    st.markdown(
        "- **The blocker is data size, not tooling.** The tournament model trains "
        "on only ~1,000 games (with ~90 rating features). Deep nets are hungry; on "
        "small, tabular data they **overfit** — exactly what we saw (the MLP's "
        "log-loss blew up to 1.69). Gradient-boosted trees usually *beat* neural "
        "nets in this regime.\n"
        "- **But \"just add more data\" isn't a free lunch.** We tested it: training "
        "the same model on the **~90,000-game regular-season log** instead of ~1,000 "
        "tournament games made tournament predictions **worse**, not better "
        "(accuracy 0.80 vs 0.86; higher Brier and log-loss). Regular-season games "
        "are a *different distribution* — blowouts, home courts, mismatches — so the "
        "extra volume dilutes the tournament signal. More data only helps if it's "
        "**on-distribution**; see the Bake-off's *\"more data\"* section.\n"
        "- **Calibration still applies.** Neural nets are miscalibrated out of the "
        "box too, so they'd need the same isotonic/Platt step the trees got.\n"
        "- **Deploy cost.** A real deep net means PyTorch/Keras — heavy for this "
        "Streamlit app — whereas the MLP rides along inside scikit-learn for free."
    )
    st.info(
        "**Bottom line.** More flexible ≠ better on a small dataset, and — as we "
        "found — more *data* isn't automatically better either when it comes from a "
        "different distribution. Random Forest quietly wins here because averaging "
        "keeps it honest. A genuine deep net would need *on-distribution* data at "
        "scale (e.g. many seasons of neutral-site games between tournament-quality "
        "teams), which we don't have — so it stays a labelled experiment, not the "
        "default."
    )


# ──────────────────────────────────────────────────────────────
# PAGE: BETTING SIMULATION
# ──────────────────────────────────────────────────────────────

elif page == "Betting Simulation":
    st.title("Betting Simulation")

    def _render_bet_table(summ, window, extra_cols=()):
        """Threshold P&L table for one training window."""
        wsub = summ[summ["window"] == window].sort_values("threshold")
        base = ["threshold", "placed", "won", "lost"]
        if "push" in extra_cols:
            base.append("push")
        cols = base + ["net_pnl", "roi_pct", *(c for c in extra_cols if c != "push")]
        names = {"threshold": "Confidence line", "placed": "Bets", "won": "Won",
                 "lost": "Lost", "push": "Push", "net_pnl": "Net P&L ($)",
                 "roi_pct": "ROI %", "avg_ml": "Avg real line", "no_odds": "No line",
                 "spread_bets": "Spread bets", "ml_bets": "ML bets"}
        disp = wsub[cols].rename(columns=names)
        st.dataframe(
            disp.style.format({"Net P&L ($)": "{:+.2f}", "ROI %": "{:+.1f}",
                               "Avg real line": "{:+.0f}"}, na_rep="—")
            .map(lambda v: "color: #2F9E44" if isinstance(v, (int, float)) and v > 0
                 else ("color: #D6336C" if isinstance(v, (int, float)) and v < 0 else ""),
                 subset=["Net P&L ($)", "ROI %"]),
            hide_index=True, width="stretch",
        )

    def _render_cum_chart(bet, window):
        """Cumulative P&L by threshold across tournament years."""
        cum = bet[bet["window"] == window].sort_values("test_year").copy()
        cum["cum_pnl"] = cum.groupby("threshold")["net_pnl"].cumsum()
        cum["threshold"] = cum["threshold"].astype(str)
        line = alt.Chart(cum).mark_line(point=True).encode(
            x=alt.X("test_year:O", title="Tournament year"),
            y=alt.Y("cum_pnl:Q", title="Cumulative net P&L ($)"),
            color=alt.Color("threshold:N", title="Confidence line"),
            tooltip=["test_year", "threshold",
                     alt.Tooltip("cum_pnl:Q", format="+.2f")],
        )
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            strokeDash=[4, 4], color="gray").encode(y="y:Q")
        st.altair_chart((line + zero).properties(height=350), width="stretch")

    bet_real, bet_flip, bet_impl = (
        results["bet_real"], results["bet_flip"], results["bet"])
    sources = []
    if bet_real is not None:
        sources.append("Real odds — moneyline")
    if bet_flip is not None:
        sources.append("Real odds — flip chalk to spread")
    if bet_impl is not None:
        sources.append("Model-implied odds (calibration)")

    if not sources:
        st.warning("No precomputed betting data. Run `python backtest_odds.py` "
                   "(and `python precompute.py`) first.")
    else:
        source = st.radio("Strategy", sources, horizontal=True)
        st.markdown(
            "At each **confidence threshold** the model bets $10 on its pick only "
            "when it is at least that confident (e.g. `-300` = only very strong "
            "favorites). Higher thresholds → fewer, chalkier bets."
        )
        meta = results["bet_real_meta"]
        yrs = meta.iloc[0]["odds_years"] if meta is not None else "2009–2019, 2021"

        if source == "Real odds — moneyline":
            st.info(
                "Every bet is settled at the game's **real closing moneyline** "
                "from the sportsbook archive and against the real result — this is "
                "actual profit/loss, not a calibration proxy. Tournaments "
                f"backtested: **{yrs}** (the seasons with published odds and a prior "
                "season to train on). Bets on games with no matchable line are "
                "excluded and counted under **No line**."
            )
            summ = (bet_real.groupby(["window", "threshold"]).agg(
                placed=("placed", "sum"), won=("won", "sum"), lost=("lost", "sum"),
                net_pnl=("net_pnl", "sum"), wagered=("total_wagered", "sum"),
                no_odds=("no_odds", "sum"),
                avg_ml=("avg_ml", "mean"),
            ).reset_index())
            summ["roi_pct"] = (summ["net_pnl"] / summ["wagered"] * 100).round(1)
            summ["avg_ml"] = summ["avg_ml"].round()

            window = st.selectbox("Training window",
                                  sorted(summ["window"].unique()), index=0)
            _render_bet_table(summ, window, extra_cols=("avg_ml", "no_odds"))
            st.subheader("Cumulative P&L over time")
            _render_cum_chart(bet_real, window)
            st.caption(
                "Flat-staking the model's confident picks at real prices is a "
                "losing game: heavy favorites pay pennies when they win but cost a "
                "full unit when they don't, and the model has no true edge over the "
                "closing line. This is exactly the Kelly intuition — on a huge "
                "favorite there is almost nothing worth betting."
            )
        elif source == "Real odds — flip chalk to spread":
            st.info(
                "Same picks and thresholds, but when the pick's **real moneyline is "
                "shorter than the threshold** (heavy chalk that pays pennies), the "
                "bet is placed on the **real point spread at −110** instead — the "
                "'take the points on a huge favorite' idea. Softer lines still bet "
                "the moneyline. **Spread bets** / **ML bets** show the split; "
                "**Push** = the favorite landed exactly on the number."
            )
            summ = (bet_flip.groupby(["window", "threshold"]).agg(
                placed=("placed", "sum"), won=("won", "sum"), lost=("lost", "sum"),
                push=("push", "sum"), net_pnl=("net_pnl", "sum"),
                wagered=("total_wagered", "sum"),
                spread_bets=("spread_bets", "sum"), ml_bets=("ml_bets", "sum"),
            ).reset_index())
            summ["roi_pct"] = (summ["net_pnl"] / summ["wagered"] * 100).round(1)

            window = st.selectbox("Training window",
                                  sorted(summ["window"].unique()), index=0)
            _render_bet_table(summ, window,
                              extra_cols=("push", "spread_bets", "ml_bets"))
            st.subheader("Cumulative P&L over time")
            _render_cum_chart(bet_flip, window)
            st.caption(
                "Flipping to the spread doesn't rescue it — arguably makes it "
                "worse. The closing spread is efficient, so a near-certain "
                "moneyline win becomes a ~coin-flip cover, and you pay −110 vig on "
                "each one. The market has already priced the favorite fairly on "
                "both the moneyline and the spread."
            )
        else:
            st.warning(
                "This uses the **model's own implied odds** as a proxy for "
                "sportsbook lines, so it measures calibration — not real profit."
            )
            summ = (bet_impl.groupby(["window", "threshold"]).agg(
                placed=("placed", "sum"), won=("won", "sum"), lost=("lost", "sum"),
                net_pnl=("net_pnl", "sum"), wagered=("total_wagered", "sum"),
            ).reset_index())
            summ["roi_pct"] = (summ["net_pnl"] / summ["wagered"] * 100).round(1)

            window = st.selectbox("Training window",
                                  sorted(summ["window"].unique()), index=0)
            _render_bet_table(summ, window)
            st.subheader("Cumulative P&L over time")
            _render_cum_chart(bet_impl, window)
            st.caption(
                "Negative ROI across thresholds is expected: betting at fair "
                "implied odds with no sportsbook edge tends to lose to variance."
            )

    # ── Strategy lab ─────────────────────────────────────────────
    lab = load_strategy_lab()
    if lab is not None:
        st.divider()
        st.header("🧪 Strategy lab")
        st.markdown(
            "The sweep above bets the model's pick and just varies *how confident* "
            "it must be. The strategy lab asks the sharper question: decide **per "
            "game** whether the model's probability beats the **real price** "
            "(+EV) — on *either* side — and vary **how you stake**. "
            f"Bankroll **${bs.START_BANKROLL:,.0f}**, flat unit **${bs.FLAT_UNIT:,.0f}**, "
            f"**{bs.KELLY_FRACTION:g}-Kelly** for the Kelly strategy."
        )
        lab_windows = sorted(lab["summary"]["window"].unique())
        lab_window = st.selectbox(
            "Model / training window", lab_windows,
            index=lab_windows.index("all_prior") if "all_prior" in lab_windows else 0,
            key="lab_window")
        s = lab["summary"][lab["summary"]["window"] == lab_window]

        disp = s[["strategy", "bets", "win_rate", "roi_pct", "avg_edge_pct",
                  "final_bankroll", "max_drawdown_pct"]].rename(columns={
            "strategy": "Strategy", "bets": "Bets", "win_rate": "Win %",
            "roi_pct": "ROI %", "avg_edge_pct": "Avg edge %",
            "final_bankroll": "End bankroll ($)", "max_drawdown_pct": "Max drawdown %"})
        st.dataframe(
            disp.style.format({"Win %": "{:.1f}", "ROI %": "{:+.1f}",
                               "Avg edge %": "{:+.1f}", "End bankroll ($)": "{:,.0f}",
                               "Max drawdown %": "{:.1f}"})
            .map(lambda v: "color: #2F9E44" if isinstance(v, (int, float)) and v > 0
                 else ("color: #D6336C" if isinstance(v, (int, float)) and v < 0 else ""),
                 subset=["ROI %"]),
            hide_index=True, width="stretch",
        )

        st.subheader("Bankroll over time")
        eq = lab["equity"][lab["equity"]["window"] == lab_window].copy()
        base = alt.Chart(eq).mark_line().encode(
            x=alt.X("bet:Q", title="Bet number (chronological)"),
            y=alt.Y("bankroll:Q", title="Bankroll ($)",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("strategy:N", title="Strategy"),
            tooltip=["strategy", "bet", "year",
                     alt.Tooltip("bankroll:Q", format="$,.0f")],
        )
        start_rule = alt.Chart(pd.DataFrame({"y": [bs.START_BANKROLL]})).mark_rule(
            strokeDash=[4, 4], color="gray").encode(y="y:Q")
        st.altair_chart((base + start_rule).properties(height=380),
                        width="stretch")
        st.caption(
            "**Avg edge %** is how much the model's probability exceeds the "
            "de-vigged market price on the bets it places — its *claimed* edge."
        )
        st.warning(
            "**Read this skeptically.** Flat +EV betting turns a profit on the "
            "longer-training windows — but not the shorter ones, and even there "
            "only about half the individual tournaments win (a couple of outlier "
            "years carry it). The model is also measurably **overconfident** on "
            "favorites, so some of its 'edge' is illusory. The tell: **¼-Kelly**, "
            "which sizes bets by the claimed edge, loses on every window and "
            "craters the bankroll — if the edges were real, Kelly would compound "
            "them, not destroy them. Treat this as a **lead to validate with a "
            "better-calibrated model**, not a proven way to beat the market."
        )

        # ── Where does the edge live? (by round / by seed) ──────────
        sl_all = lab.get("slices")
        if sl_all is not None and not sl_all.empty:
            st.subheader("Where does the edge live?")
            st.markdown(
                "The Value (+EV, flat) strategy, broken down by the **round** of "
                "the game and the **seed of the team it backs** — to see whether "
                "any slice actually beats the closing line, or whether the overall "
                "profit hides pockets that don't."
            )

            def _slice_table(dim, dim_label):
                d = sl_all[(sl_all["window"] == lab_window) &
                           (sl_all["slice_by"] == dim)]
                d = d[["slice", "bets", "win_rate", "roi_pct", "avg_edge_pct"]].rename(
                    columns={"slice": dim_label, "bets": "Bets", "win_rate": "Win %",
                             "roi_pct": "ROI %", "avg_edge_pct": "Avg edge %"})
                st.dataframe(
                    d.style.format({"Win %": "{:.1f}", "ROI %": "{:+.1f}",
                                    "Avg edge %": "{:+.1f}"})
                    .map(lambda v: "color: #2F9E44" if isinstance(v, (int, float)) and v > 0
                         else ("color: #D6336C" if isinstance(v, (int, float)) and v < 0 else ""),
                         subset=["ROI %"]),
                    hide_index=True, width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                st.caption("**By round**")
                _slice_table("round", "Round")
            with c2:
                st.caption("**By seed of the backed team**")
                _slice_table("seed", "Seed")
            st.caption(
                "Read the deep-round rows skeptically — the Final Four and "
                "Championship slices are only a handful of bets each, so their "
                "eye-popping ROI is mostly noise. The steadier signal: the edge "
                "concentrates in the **first round** and on **double-digit-seed "
                "underdogs**, and evaporates on the favorites — consistent with "
                "the model being overconfident on chalk."
            )

        # ── Does the edge survive calibration? (raw vs calibrated probs) ──
        cal_summary = lab.get("cal_summary")
        if cal_summary is not None:
            st.subheader("Does the edge survive calibration?")
            st.markdown(
                "The Bake-off showed the model is **overconfident** — so some of "
                "the +EV \"edge\" above may just be the model overstating its win "
                "probabilities, not a real market inefficiency. Here the **same "
                "bets at the same closing lines** are re-settled with the model's "
                "probabilities **isotonic-calibrated** (fit on the training "
                "seasons only). If the profit shrinks toward zero, the edge was "
                "mostly overconfidence."
            )
            raw_w = lab["summary"][lab["summary"]["window"] == lab_window]
            cal_w = cal_summary[cal_summary["window"] == lab_window]
            comp = raw_w[["strategy", "roi_pct", "avg_edge_pct"]].merge(
                cal_w[["strategy", "roi_pct", "avg_edge_pct"]],
                on="strategy", suffixes=(" raw", " calib"))
            comp["ROI Δ"] = (comp["roi_pct calib"] - comp["roi_pct raw"]).round(1)
            comp = comp.rename(columns={
                "strategy": "Strategy", "roi_pct raw": "ROI % (raw)",
                "roi_pct calib": "ROI % (calib)", "avg_edge_pct raw": "Edge % (raw)",
                "avg_edge_pct calib": "Edge % (calib)"})
            comp = comp[["Strategy", "ROI % (raw)", "ROI % (calib)", "ROI Δ",
                         "Edge % (raw)", "Edge % (calib)"]]
            st.dataframe(
                comp.style.format({"ROI % (raw)": "{:+.1f}", "ROI % (calib)": "{:+.1f}",
                                   "ROI Δ": "{:+.1f}", "Edge % (raw)": "{:+.1f}",
                                   "Edge % (calib)": "{:+.1f}"})
                .map(lambda v: "color: #2F9E44" if isinstance(v, (int, float)) and v > 0
                     else ("color: #D6336C" if isinstance(v, (int, float)) and v < 0 else ""),
                     subset=["ROI % (raw)", "ROI % (calib)"]),
                hide_index=True, width="stretch",
            )
            st.caption(
                "**Avg edge %** is how much the model's probability exceeds the "
                "de-vigged market price — calibration should shrink the *claimed* "
                "edge toward what's real. Watch whether the +EV strategies' ROI "
                "collapses once that happens."
            )


# ──────────────────────────────────────────────────────────────
# PAGE: CUSTOM METRIC
# ──────────────────────────────────────────────────────────────

elif page == "Custom Metric":
    st.title("Test Your Own Metric")
    st.markdown(
        "Upload a metric of your own — coaching tenure, tournament history, "
        "travel distance, anything numeric — and see **how much it helps the "
        "model** predict real tournament games. The app trains the model twice "
        "(with and without your metric) and ranks your metric against the ~95 "
        "existing features by **permutation importance**."
    )

    st.subheader("1 · Get the format right")
    st.markdown(
        "Your CSV needs a **`YEAR`** column, a **`TEAM`** column, and one or more "
        "**numeric** columns (your metrics). Team names must match the dataset "
        "exactly — browse them on the **Data Explorer** page. Rows you don't "
        "provide are filled neutrally, so partial coverage is fine."
    )
    template = (df_raw[["YEAR", "TEAM"]]
                .drop_duplicates().sort_values(["YEAR", "TEAM"]))
    template["my_metric"] = ""
    st.download_button(
        "⬇️ Download blank template (every YEAR, TEAM to fill in)",
        template.to_csv(index=False).encode(),
        "custom_metric_template.csv", "text/csv",
    )

    st.subheader("2 · Upload and evaluate")
    upload = st.file_uploader("Upload your metric CSV", type=["csv"])

    if upload is None:
        st.info("Waiting for a CSV with YEAR, TEAM, and at least one numeric column.")
    else:
        try:
            raw_up = pd.read_csv(upload)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
            raw_up = None

        if raw_up is not None:
            clean, metric_cols, err = mm.prepare_custom_metric(raw_up)
            if err:
                st.error(err)
            else:
                cov = mm.custom_metric_coverage(df_raw, clean)
                st.success(f"Detected metric column(s): **{', '.join(metric_cols)}**")

                c1, c2 = st.columns(2)
                c1.metric("Team-seasons covered", f"{cov['coverage']:.0%}",
                          help=f"{cov['n_matched']:,} of {cov['n_base']:,} "
                               "tournament team-seasons matched your upload.")
                c2.metric("Upload rows that matched nothing",
                          f"{len(cov['unmatched_upload']):,}")

                if cov["unmatched_upload"]:
                    with st.expander("Rows that didn't match a team-season "
                                     "(check YEAR + spelling)"):
                        st.dataframe(
                            pd.DataFrame(cov["unmatched_upload"],
                                         columns=["YEAR", "TEAM"]),
                            hide_index=True, width="stretch", height=240,
                        )

                if cov["coverage"] == 0:
                    st.warning(
                        "No rows matched. Team names must match the dataset "
                        "exactly — compare against the Data Explorer page."
                    )
                else:
                    cc1, cc2 = st.columns(2)
                    sel_year = cc1.selectbox(
                        "Evaluate on season", completed_years[::-1], index=0,
                        help="Only completed tournaments can be scored.")
                    sel_window = cc2.selectbox(
                        "Training window", list(mm.TRAINING_WINDOWS.keys()),
                        index=0, help="Which prior seasons the model learns from.")

                    if st.button("Run model with my metric", type="primary"):
                        with st.spinner("Training with and without your metric… "
                                        "(first run per selection)"):
                            res = eval_custom_metric_cached(
                                clean, metric_cols, sel_year, sel_window)

                        if res.get("error"):
                            st.error(res["error"])
                        else:
                            st.divider()
                            st.subheader("Does your metric improve accuracy?")
                            d = res["acc_delta"]
                            m1, m2, m3 = st.columns(3)
                            m1.metric("With your metric", fmt_pct(res["acc_with"]))
                            m2.metric("Baseline (without)", fmt_pct(res["acc_without"]))
                            m3.metric("Change", f"{d:+.1%}",
                                      delta=f"{d:+.1%}",
                                      delta_color="normal" if d != 0 else "off")
                            st.caption(
                                f"Scored on **{res['n_test_games']} actual "
                                f"{sel_year} games** · trained on "
                                f"{res['train_years'][0]}–{res['train_years'][-1]} · "
                                f"model with metric: {res['model_with']}"
                            )
                            if d > 0:
                                st.success("Your metric improved accuracy on this "
                                           "season. 🎯")
                            elif d < 0:
                                st.info("Your metric lowered accuracy on this "
                                        "season — it may add noise here.")
                            else:
                                st.info("No change in accuracy on this season.")

                            st.subheader("How important is your metric?")
                            imp = res["importance"]
                            ranks = imp[imp["is_custom"]][
                                ["feature", "rank", "importance"]]
                            for _, r in ranks.iterrows():
                                st.markdown(
                                    f"- **{r['feature']}** ranks "
                                    f"**#{int(r['rank'])} of {res['n_features']}** "
                                    f"features (permutation importance "
                                    f"{r['importance']:+.4f})."
                                )

                            top = imp.head(20).copy()
                            top["kind"] = np.where(top["is_custom"],
                                                   "Your metric", "Existing feature")
                            st.altair_chart(
                                alt.Chart(top).mark_bar().encode(
                                    x=alt.X("importance:Q",
                                            title="Permutation importance "
                                                  "(accuracy drop when shuffled)"),
                                    y=alt.Y("feature:N", sort="-x", title=None),
                                    color=alt.Color(
                                        "kind:N", title=None,
                                        scale=alt.Scale(
                                            domain=["Your metric", "Existing feature"],
                                            range=["#E8590C", "#8A7355"])),
                                    tooltip=["feature",
                                             alt.Tooltip("importance:Q", format="+.4f"),
                                             "rank"],
                                ).properties(height=460),
                                width="stretch",
                            )
                            st.caption(
                                "Permutation importance measures how much test "
                                "accuracy drops when a feature's values are randomly "
                                "shuffled — higher means the model relies on it more. "
                                "Top 20 features shown."
                            )


# ──────────────────────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ──────────────────────────────────────────────────────────────

elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown(
        f"Underlying team-season ratings: **{len(df_raw):,} rows**, "
        f"**{len(df_raw.columns)} columns**, {completed_years[0]}–{all_years[-1]}."
    )
    c1, c2 = st.columns([1, 2])
    year = c1.selectbox("Season", all_years, index=len(all_years) - 1)
    yr_df = df_raw[df_raw["YEAR"] == year]
    search = c2.text_input("Filter teams (substring)", "")
    if search:
        yr_df = yr_df[yr_df["TEAM"].str.contains(search, case=False, na=False)]

    default_cols = [c for c in
                    ["TEAM", "SEED", "CONF", "KADJ EM", "KADJ O", "KADJ D", "BARTHAG"]
                    if c in yr_df.columns]
    cols = st.multiselect("Columns", list(df_raw.columns), default=default_cols)
    st.dataframe(yr_df[cols] if cols else yr_df, hide_index=True,
                 width="stretch", height=500)
    st.download_button(
        "⬇️ Download this season (CSV)",
        yr_df.to_csv(index=False).encode(),
        f"teams_{year}.csv", "text/csv",
    )
