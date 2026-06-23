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

3\. If the odds can be tracked down, figure out how much money would have been won using the bracket and testing different spread ranges (i.e., -200, -300, -350, etc.)

4\. Apply each model to the 2027 tournament. 

