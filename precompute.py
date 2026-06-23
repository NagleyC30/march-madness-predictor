# precompute.py — Run the heavy walk-forward validation ONCE and write result
# CSVs that the Streamlit app loads instantly. Re-run this whenever the source
# data (KenPom Barttorvik.csv) changes.
#
#   python precompute.py
#
# Outputs (committed to the repo, consumed by app.py):
#   data/walk_forward_accuracy.csv   per-year, per-window, per-round accuracy
#   data/betting_simulation.csv      per-year, per-window, per-threshold P&L
#   data/bracket_prediction_2026.csv predicted 2026 bracket (best window)
#   data/window_summary.csv          overall accuracy by training window

import os
import pandas as pd
import numpy as np

import mm_model as mm

OUT_DIR = 'data'
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = mm.load_data()
    features = mm.get_features(df)

    print("Building historical matchup dataset...")
    all_matchup_rows, tournament_years = mm.build_all_matchups(df)
    df_all_matchups = pd.DataFrame(all_matchup_rows)
    print(f"Total historical matchups: {len(df_all_matchups)}")

    test_years = [y for y in tournament_years if y >= 2009]
    print(f"Walk-forward over {test_years[0]}-{test_years[-1]}, "
          f"windows: {list(mm.TRAINING_WINDOWS.keys())}")

    all_wf_results  = []
    all_bet_results = []

    total_iters = len(mm.TRAINING_WINDOWS) * len(test_years)
    done_iters = 0

    for window_name, window_size in mm.TRAINING_WINDOWS.items():
        print(f"\n=== WINDOW: {window_name} ===", flush=True)
        for test_year in test_years:
            done_iters += 1
            train_years = mm.get_train_years_for_window(
                tournament_years, test_year, window_size)
            if not train_years:
                continue

            train_rows = [r for r in all_matchup_rows if r['YEAR'] in train_years]
            df_train = mm.build_model_dataset(train_rows, df, features)
            if len(df_train) < 10:
                print(f"  {test_year}: insufficient training data — skip")
                continue

            model, model_name, cv_score = mm.train_best_model(df_train, features)

            df_test_year = df[df['YEAR'] == test_year]
            pred_rows = mm.simulate_full_bracket(df_test_year, model, test_year, features)

            cascade = mm.score_predictions_cascade(pred_rows, df_all_matchups, test_year)
            indep = mm.score_predictions_independent(
                df_all_matchups, test_year, df_test_year, model, features)

            casc_c = sum(v['correct'] for v in cascade.values())
            casc_t = sum(v['total'] for v in cascade.values())
            ind_c = sum(v['correct'] for v in indep.values())
            ind_t = sum(v['total'] for v in indep.values())
            print(f"  [{done_iters}/{total_iters}] {test_year} | {model_name} | "
                  f"cv={cv_score:.3f} | cascade {casc_c}/{casc_t} | "
                  f"indep {ind_c}/{ind_t}", flush=True)

            for rnd in mm.ALL_ROUND_NAMES:
                cs = cascade.get(rnd, {'correct': 0, 'total': 0, 'accuracy': None})
                isc = indep.get(rnd, {'correct': 0, 'total': 0, 'accuracy': None})
                all_wf_results.append({
                    'window': window_name, 'test_year': test_year,
                    'train_years': str(train_years), 'model': model_name,
                    'cv_score': cv_score, 'round': rnd,
                    'cascade_correct': cs['correct'], 'cascade_total': cs['total'],
                    'cascade_accuracy': cs['accuracy'],
                    'indep_correct': isc['correct'], 'indep_total': isc['total'],
                    'indep_accuracy': isc['accuracy'],
                })

            bet_sim = mm.simulate_betting(
                pred_rows, df_all_matchups, test_year,
                thresholds=mm.BETTING_THRESHOLDS)
            for thresh, bstats in bet_sim.items():
                all_bet_results.append({
                    'window': window_name, 'test_year': test_year,
                    'threshold': thresh, **bstats,
                })

    df_wf = pd.DataFrame(all_wf_results)
    df_betting = pd.DataFrame(all_bet_results)

    df_wf.to_csv(f'{OUT_DIR}/walk_forward_accuracy.csv', index=False)
    df_betting.to_csv(f'{OUT_DIR}/betting_simulation.csv', index=False)

    # Overall accuracy by window
    summary = (df_wf.groupby('window')
               .apply(lambda g: pd.Series({
                   'cascade_correct': g['cascade_correct'].sum(),
                   'cascade_total':   g['cascade_total'].sum(),
                   'cascade_acc':     g['cascade_correct'].sum() / g['cascade_total'].sum()
                                      if g['cascade_total'].sum() > 0 else 0,
                   'indep_correct':   g['indep_correct'].sum(),
                   'indep_total':     g['indep_total'].sum(),
                   'indep_acc':       g['indep_correct'].sum() / g['indep_total'].sum()
                                      if g['indep_total'].sum() > 0 else 0,
               })).reset_index())
    summary.to_csv(f'{OUT_DIR}/window_summary.csv', index=False)

    # Predict 2026 with the best (highest independent accuracy) window
    best_window = summary.loc[summary['indep_acc'].idxmax(), 'window']
    best_size = mm.TRAINING_WINDOWS[best_window]
    print(f"\nBest window for forecasting: {best_window} "
          f"(indep acc {summary['indep_acc'].max():.1%})")

    for target_year in (2026,):
        if target_year not in df['YEAR'].unique():
            print(f"[skip] no data for {target_year}")
            continue
        train_yrs = mm.get_train_years_for_window(tournament_years, target_year, best_size)
        train_rows = [r for r in all_matchup_rows if r['YEAR'] in train_yrs]
        df_train = mm.build_model_dataset(train_rows, df, features)
        model, model_name, cv_score = mm.train_best_model(df_train, features)
        df_target = df[df['YEAR'] == target_year]
        pred_rows = mm.simulate_full_bracket(df_target, model, target_year, features)

        df_pred = pd.DataFrame(pred_rows)
        cat = pd.CategoricalDtype(mm.ALL_ROUND_NAMES, ordered=True)
        df_pred['ROUND'] = df_pred['ROUND'].astype(cat)
        df_pred = df_pred.sort_values(['ROUND', 'REGION']).reset_index(drop=True)
        df_pred['IMPLIED_LINE'] = df_pred.apply(
            lambda r: mm.prob_to_american_odds(
                r['PROB_HIGH_WINS'] if r['PRED_WINNER'] == r['TEAM_HIGH']
                else 1 - r['PROB_HIGH_WINS']), axis=1)
        df_pred['BEST_WINDOW'] = best_window
        df_pred['MODEL'] = model_name
        df_pred.to_csv(f'{OUT_DIR}/bracket_prediction_{target_year}.csv', index=False)
        champ = df_pred[df_pred['ROUND'] == 'Championship']
        if not champ.empty:
            print(f"{target_year} predicted champion: "
                  f"{champ.iloc[0]['PRED_WINNER']} (#{champ.iloc[0]['PRED_SEED']})")

    # Persist metadata for the app footer
    pd.DataFrame([{
        'best_window': best_window,
        'generated_utc': pd.Timestamp.utcnow().isoformat(),
        'n_historical_matchups': len(df_all_matchups),
        'test_years': f"{test_years[0]}-{test_years[-1]}",
    }]).to_csv(f'{OUT_DIR}/meta.csv', index=False)

    print("\nDone. Wrote result CSVs to ./data/")


if __name__ == '__main__':
    main()
