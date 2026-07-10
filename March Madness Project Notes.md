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

4\. Apply each model to the 2027 tournament. 

5\. Build a current-season predictor for the upcoming year's games: pull the live/preseason Barttorvik ratings for the in-progress 2026–27 season and let the app predict upcoming regular-season and tournament games as they're scheduled (updating as ratings refresh through the season). Extends the general Game Predictor, which currently covers completed seasons 2008–2026.

