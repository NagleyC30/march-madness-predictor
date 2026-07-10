# predict_all_windows.py — Apply EVERY training-window model to the upcoming
# tournament (project checklist item 4).
#
# precompute.py forecasts the bracket with a single "best" window. This script
# instead trains all five windows (all_prior / last_1 / last_3 / last_5 /
# last_10) and simulates the full bracket with each, so their picks — and their
# predicted champions and Final Fours — can be compared side by side.
#
# The target is auto-detected as the latest season that has a seeded field but
# no played results yet (the forecast year — 2026 today). Pass a year to force
# it; once the 2027 field is published this predicts 2027 with no code change:
#
#   python predict_all_windows.py            # latest forecast field
#   python predict_all_windows.py 2027       # a specific year
#
# Output (committed, consumed by app.py):
#   data/bracket_all_windows.csv   long format: one row per (window, game) with
#                                   the pick and its implied moneyline.
#   data/bracket_all_windows_meta.csv  target year, best window, generated time.

import os
import sys

import pandas as pd

import mm_model as mm

OUT_DIR = "data"


def forecast_years(df, tour_years):
    """Seasons that have a field but weren't trained on (no results) — the
    genuine forecast targets."""
    return sorted(set(df["YEAR"].unique()) - set(tour_years))


def predict_bracket(df, features, all_rows, tour_years, target_year, window_name):
    """Simulate the full bracket for one training window; return a DataFrame of
    its games with the pick's implied moneyline attached."""
    window_size = mm.TRAINING_WINDOWS[window_name]
    train_years = mm.get_train_years_for_window(tour_years, target_year, window_size)
    train_rows = [r for r in all_rows if r["YEAR"] in train_years]
    df_train = mm.build_model_dataset(train_rows, df, features)
    model, model_name, cv = mm.train_best_model(df_train, features)

    df_target = df[df["YEAR"] == target_year]
    pred_rows = mm.simulate_full_bracket(df_target, model, target_year, features)

    out = pd.DataFrame(pred_rows)
    cat = pd.CategoricalDtype(mm.ALL_ROUND_NAMES, ordered=True)
    out["ROUND"] = out["ROUND"].astype(cat)
    out = out.sort_values(["ROUND", "REGION"]).reset_index(drop=True)
    out["IMPLIED_LINE"] = out.apply(
        lambda r: mm.prob_to_american_odds(
            r["PROB_HIGH_WINS"] if r["PRED_WINNER"] == r["TEAM_HIGH"]
            else 1 - r["PROB_HIGH_WINS"]), axis=1)
    out.insert(0, "WINDOW", window_name)
    out["MODEL"] = model_name
    out["CV_SCORE"] = round(cv, 4)
    out["N_TRAIN_YEARS"] = len(train_years)
    return out, model_name


def main(target_year=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = mm.load_data()
    features = mm.get_features(df)
    all_rows, tour_years = mm.build_all_matchups(df)

    candidates = forecast_years(df, tour_years)
    if target_year is None:
        if not candidates:
            print("No forecast-eligible field found (a season with seeds but no "
                  "results). Nothing to predict.")
            return
        target_year = max(candidates)
    if target_year not in df["YEAR"].unique():
        print(f"No field data for {target_year} yet — cannot predict it. "
              f"Available forecast fields: {candidates or 'none'}.")
        return

    print(f"Applying every training window to the {target_year} bracket "
          f"(forecast fields available: {candidates})\n")

    parts, champions = [], []
    for window_name in mm.TRAINING_WINDOWS:
        out, model_name = predict_bracket(
            df, features, all_rows, tour_years, target_year, window_name)
        parts.append(out)
        champ = out[out["ROUND"] == "Championship"]
        f4 = out[out["ROUND"] == "F4"]
        champ_team = champ.iloc[0]["PRED_WINNER"] if not champ.empty else "—"
        champ_seed = int(champ.iloc[0]["PRED_SEED"]) if not champ.empty else 0
        champions.append({
            "window": window_name, "model": model_name,
            "champion": champ_team, "champion_seed": champ_seed,
            "final_four": " / ".join(sorted(
                set(f4["TEAM_HIGH"]) | set(f4["TEAM_LOW"]))) if not f4.empty else "",
        })
        print(f"  {window_name:13} ({model_name:17}) -> champion: "
              f"{champ_team} (#{champ_seed})", flush=True)

    all_brackets = pd.concat(parts, ignore_index=True)
    all_brackets.to_csv(os.path.join(OUT_DIR, "bracket_all_windows.csv"), index=False)

    # "Best" window = the one with the highest walk-forward accuracy, if that
    # summary exists; else fall back to all_prior. Only used to pick a default.
    best_window = "all_prior"
    summ_path = os.path.join(OUT_DIR, "window_summary.csv")
    if os.path.exists(summ_path):
        summ = pd.read_csv(summ_path)
        if "indep_acc" in summ.columns and not summ.empty:
            best_window = summ.loc[summ["indep_acc"].idxmax(), "window"]

    pd.DataFrame([{
        "target_year": target_year,
        "best_window": best_window,
        "n_windows": len(mm.TRAINING_WINDOWS),
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
    }]).to_csv(os.path.join(OUT_DIR, "bracket_all_windows_meta.csv"), index=False)

    print(f"\nChampion by window:")
    print(pd.DataFrame(champions)[["window", "champion", "champion_seed"]]
          .to_string(index=False))
    n_agree = pd.Series([c["champion"] for c in champions]).nunique()
    print(f"\n{n_agree} distinct champion(s) across the 5 windows.")
    print(f"\nWrote data/bracket_all_windows.csv ({len(all_brackets)} rows).")


if __name__ == "__main__":
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(yr)
