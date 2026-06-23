# app.py — Streamlit dashboard for the March Madness predictor.
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

st.set_page_config(
    page_title="March Madness Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        "summary": "window_summary.csv",
        "bracket2026": "bracket_prediction_2026.csv",
        "meta": "meta.csv",
    }.items():
        path = os.path.join(DATA_DIR, fname)
        out[key] = pd.read_csv(path) if os.path.exists(path) else None
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

st.sidebar.title("🏀 March Madness Predictor")
st.sidebar.caption(
    "Walk-forward ML on KenPom + Barttorvik efficiency ratings, "
    f"{completed_years[0]}–{completed_years[-1]}."
)

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Bracket Predictions", "Head-to-Head", "Model Accuracy",
     "Betting Simulation", "Data Explorer"],
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
    "⚠️ Educational project. Betting figures use model-implied odds as a proxy "
    "and are **not** a betting recommendation."
)


# ──────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ──────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("🏀 March Madness Bracket Predictor")
    st.markdown(
        "A machine-learning model that predicts NCAA tournament games from "
        "**KenPom** and **Barttorvik** team-efficiency ratings. It's validated "
        "**walk-forward**: to predict any year, it only ever trains on tournaments "
        "that happened *before* that year — no peeking at the future."
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
- **Head-to-Head** — pick any two teams from a season and get a live prediction.
- **Model Accuracy** — walk-forward accuracy by training window and tournament round.
- **Betting Simulation** — hypothetical P&L at various confidence thresholds.
- **Data Explorer** — browse the underlying team-season stats.
        """
    )


# ──────────────────────────────────────────────────────────────
# PAGE: BRACKET PREDICTIONS
# ──────────────────────────────────────────────────────────────

elif page == "Bracket Predictions":
    st.title("Predicted Bracket — 2026")
    bracket = results["bracket2026"]
    if bracket is None:
        st.warning("No precomputed 2026 bracket found. Run `python precompute.py` first.")
    else:
        champ = bracket[bracket["ROUND"] == "Championship"]
        if not champ.empty:
            row = champ.iloc[0]
            st.success(
                f"### 🏆 Predicted National Champion: **{row['PRED_WINNER']}** "
                f"(#{int(row['PRED_SEED'])} seed)"
            )

        st.caption(
            f"Model: {bracket['MODEL'].iloc[0]} · training window: "
            f"{bracket['BEST_WINDOW'].iloc[0]}"
        )

        for code in mm.ALL_ROUND_NAMES:
            rnd = bracket[bracket["ROUND"] == code]
            if rnd.empty:
                continue
            with st.expander(f"{round_label(code)}  ({len(rnd)} games)",
                             expanded=code in ("F4", "Championship")):
                show = rnd.copy()
                show["Matchup"] = (
                    show["TEAM_HIGH"] + " (#" + show["SEED_HIGH"].astype(int).astype(str)
                    + ")  vs  " + show["TEAM_LOW"] + " (#"
                    + show["SEED_LOW"].astype(int).astype(str) + ")"
                )
                show["Pick"] = show["PRED_WINNER"] + " (#" + show["PRED_SEED"].astype(int).astype(str) + ")"
                show["Implied line"] = show["IMPLIED_LINE"].apply(lambda v: f"{int(v):+d}")
                st.dataframe(
                    show[["REGION", "Matchup", "Pick", "Implied line"]]
                    .rename(columns={"REGION": "Region"}),
                    hide_index=True, width="stretch",
                )
        st.download_button(
            "⬇️ Download bracket (CSV)",
            bracket.to_csv(index=False).encode(),
            "bracket_prediction_2026.csv", "text/csv",
        )


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
                color=alt.value("#FF4B4B"),
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
                color=alt.value("#1f77b4"),
            ).properties(height=300),
            width="stretch",
        )
        st.caption(
            "Early rounds (more games, bigger seed gaps) are easiest; later "
            "rounds have few samples so accuracy is noisier."
        )


# ──────────────────────────────────────────────────────────────
# PAGE: BETTING SIMULATION
# ──────────────────────────────────────────────────────────────

elif page == "Betting Simulation":
    st.title("Betting Simulation")
    st.warning(
        "Educational only. This uses the **model's own implied odds** as a proxy "
        "for sportsbook lines, so it measures model calibration — not real-world "
        "profit. It is not betting advice."
    )
    bet = results["bet"]
    if bet is None:
        st.warning("No precomputed betting data. Run `python precompute.py` first.")
    else:
        st.markdown(
            "At each **confidence threshold** the model only 'bets' on games where "
            "its implied moneyline is at least that confident (e.g. `-300` = only "
            "very strong favorites). $10 per bet."
        )

        summ = (bet.groupby(["window", "threshold"]).agg(
            placed=("placed", "sum"), won=("won", "sum"), lost=("lost", "sum"),
            net_pnl=("net_pnl", "sum"), wagered=("total_wagered", "sum"),
        ).reset_index())
        summ["roi_pct"] = (summ["net_pnl"] / summ["wagered"] * 100).round(1)

        window = st.selectbox("Training window", sorted(summ["window"].unique()),
                              index=0)
        wsub = summ[summ["window"] == window].sort_values("threshold")

        disp = wsub[["threshold", "placed", "won", "lost", "net_pnl", "roi_pct"]]
        disp = disp.rename(columns={
            "threshold": "Confidence line", "placed": "Bets", "won": "Won",
            "lost": "Lost", "net_pnl": "Net P&L ($)", "roi_pct": "ROI %"})
        st.dataframe(
            disp.style.format({"Net P&L ($)": "{:+.2f}", "ROI %": "{:+.1f}"})
            .map(lambda v: "color: #2ca02c" if isinstance(v, (int, float)) and v > 0
                 else ("color: #d62728" if isinstance(v, (int, float)) and v < 0 else ""),
                 subset=["Net P&L ($)", "ROI %"]),
            hide_index=True, width="stretch",
        )

        st.subheader("Cumulative P&L over time")
        bsub = bet[bet["window"] == window].copy()
        cum = bsub.sort_values("test_year").copy()
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
        st.altair_chart((line + zero).properties(height=350),
                        width="stretch")
        st.caption(
            "Negative ROI across thresholds is expected: betting at fair implied "
            "odds with no sportsbook edge tends to lose to variance and vig."
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
