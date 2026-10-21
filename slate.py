# slate.py — "Me vs. the Machine" slate-selection criteria (project item 14).
#
# Decides WHICH games are worth putting in the contest. A slate of blowouts is a
# bad test (we'd both go 8/8), so this is gate -> score -> constrain:
#   1. Eligibility gate  — both teams rated, game not yet played, and at least one
#                          team in the national top-N by BARTHAG.
#   2. Composite score   — competitiveness (40%) + quality (35%) + salience (25%).
#   3. Slate constraints — no team twice, per-day cap, quality floor (the slate
#                          shrinks rather than padding with blowouts), and a
#                          favorite/underdog mix (not all toss-ups).
#
# Data-only and side-effect free, so it can self-run on the 2027 schedule feed
# later and, today, doubles as a "suggest good matchups" helper over games pasted
# into the manual-entry page.
#
# BLIND-ENTRY RULE: the tuple these functions return carries the model's win
# probability for internal use. The UI must never render `p_a`, `p_max`, the
# competitiveness component or the favorite from a suggestion — see
# `public_view()`, which strips them.

import re

import pandas as pd

import game_model as gm

# ── Tunable knobs ───────────────────────────────────────────────
W_COMPETITIVE = 0.40
W_QUALITY = 0.35
W_SALIENCE = 0.25

TOP_N_GATE = 60          # at least one team must be in the national top N
TOP_N_MARQUEE = 25       # "ranked" for the salience bonus
QUALITY_FLOOR = 0.35     # composite below this never makes a slate
SLATE_SIZE = 10
PER_DAY_CAP = 4
LEAN_PROB = 0.65         # a game with max(p, 1-p) >= this counts as "lopsided"
MIN_LEAN = 2             # keep at least this many non-toss-ups when available

# Salience bonuses (summed, capped at 1.0).
SAL_BOTH_RANKED = 0.50
SAL_SAME_CONF = 0.25
SAL_WINDOW_SPLIT = 0.25      # cross-window model disagreement (0..1 input)
SAL_MARKET_SPLIT = 0.25      # model-vs-market disagreement
MARKET_SPLIT_FULL = 0.20     # a 20-pt model/market gap earns the full bonus

GAME_COLUMNS = ["team_a", "team_b", "location"]   # location: 'A' | 'B' | 'N'


def _rank_table(ratings_year):
    """BARTHAG rank (1 = best) per team for one season."""
    r = ratings_year[["TEAM", "CONF", "barthag"]].copy()
    r["rank"] = r["barthag"].rank(ascending=False, method="min").astype(int)
    return r.set_index("TEAM")


def competitiveness(p):
    """1 for a pick'em, 0 for a certainty: 1 - 2|p - 0.5|."""
    return 1.0 - 2.0 * abs(p - 0.5)


def score_games(games, ratings_year, model, features, top_n=TOP_N_GATE):
    """Gate and score a candidate games frame.

    ``games`` needs team_a, team_b, location ('A'/'B'/'N'); optional columns:
    ``played`` (bool), ``date`` (any sortable/hashable day label),
    ``window_split`` (0..1 cross-window disagreement) and ``market_p_a``
    (market-implied P(team_a wins)). Returns the eligible games with every
    component plus the composite ``score``, best first. Ineligible games are
    dropped.
    """
    ranks = _rank_table(ratings_year)
    rows = []
    for g in games.itertuples(index=False):
        a, b = g.team_a, g.team_b
        if a == b or a not in ranks.index or b not in ranks.index:
            continue
        played = getattr(g, "played", False)
        if pd.notna(played) and bool(played):
            continue
        ra, rb = ranks.loc[a], ranks.loc[b]
        if min(ra["rank"], rb["rank"]) > top_n:
            continue
        probs = gm.win_probabilities(a, b, g.location, ratings_year, model, features)
        if probs is None:
            continue
        p_a = probs[0]

        comp = competitiveness(p_a)
        quality = float(min(ra["barthag"], rb["barthag"]))

        sal = 0.0
        if max(ra["rank"], rb["rank"]) <= TOP_N_MARQUEE:
            sal += SAL_BOTH_RANKED
        if ra["CONF"] == rb["CONF"]:
            sal += SAL_SAME_CONF
        split = getattr(g, "window_split", None)
        if split is not None and pd.notna(split):
            sal += SAL_WINDOW_SPLIT * float(min(max(split, 0.0), 1.0))
        mkt = getattr(g, "market_p_a", None)
        if mkt is not None and pd.notna(mkt):
            sal += SAL_MARKET_SPLIT * min(abs(p_a - float(mkt)) / MARKET_SPLIT_FULL, 1.0)
        sal = min(sal, 1.0)

        rows.append({
            "team_a": a, "team_b": b, "location": g.location,
            "date": getattr(g, "date", None),
            "p_a": p_a, "p_max": max(p_a, 1.0 - p_a),
            "competitiveness": comp, "quality": quality, "salience": sal,
            "both_ranked": max(ra["rank"], rb["rank"]) <= TOP_N_MARQUEE,
            "same_conf": ra["CONF"] == rb["CONF"],
            "score": W_COMPETITIVE * comp + W_QUALITY * quality + W_SALIENCE * sal,
        })
    cols = ["team_a", "team_b", "location", "date", "p_a", "p_max",
            "competitiveness", "quality", "salience", "both_ranked",
            "same_conf", "score"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("score", ascending=False, ignore_index=True)


def build_slate(scored, size=SLATE_SIZE, floor=QUALITY_FLOOR,
                per_day_cap=PER_DAY_CAP, min_lean=MIN_LEAN):
    """Pick the slate from ``score_games`` output under the constraints.

    May return fewer than ``size`` games: anything under ``floor`` is never
    padded in. No team appears twice; at most ``per_day_cap`` games share a
    ``date``; and up to ``min_lean`` slots are reserved for non-toss-ups so the
    slate isn't all coin flips (only if enough such games clear the floor).
    """
    cand = scored[scored["score"] >= floor]
    chosen, teams, per_day = [], set(), {}

    def _take(row):
        if row.team_a in teams or row.team_b in teams:
            return False
        if pd.notna(row.date) and per_day.get(row.date, 0) >= per_day_cap:
            return False
        chosen.append(row.Index)
        teams.update((row.team_a, row.team_b))
        if pd.notna(row.date):
            per_day[row.date] = per_day.get(row.date, 0) + 1
        return True

    lean = cand[cand["p_max"] >= LEAN_PROB]
    n_lean = 0
    for row in lean.itertuples():
        if n_lean >= min_lean or len(chosen) >= size:
            break
        n_lean += _take(row)
    for row in cand.itertuples():
        if len(chosen) >= size:
            break
        if row.Index not in chosen:
            _take(row)
    return cand.loc[chosen].sort_values("score", ascending=False, ignore_index=True)


def public_view(slate):
    """The only columns safe to show before the user locks a pick: who plays
    where, why it's interesting, and the composite — never the model's
    probability, favorite, or competitiveness."""
    out = slate[["team_a", "team_b", "location", "score"]].copy()
    tags = []
    for r in slate.itertuples():
        t = []
        if r.both_ranked:
            t.append(f"both top-{TOP_N_MARQUEE}")
        if r.same_conf:
            t.append("conference game")
        tags.append(", ".join(t))
    out["why"] = tags
    return out


# ── Manual-entry helper ─────────────────────────────────────────

_AT = re.compile(r"\s+(?:at|@)\s+", re.I)
_VS = re.compile(r"\s+(?:vs\.?|v\.?|-)\s+", re.I)


def parse_games(text, teams):
    """Parse pasted lines into a games frame. ``A vs B`` = neutral court,
    ``A at B`` / ``A @ B`` = B is home. Team names are matched exactly
    (case-insensitive) against ``teams``. Returns (games_df, bad_lines)."""
    lookup = {t.lower(): t for t in teams}
    rows, bad = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _AT.search(line):
            parts, loc = _AT.split(line, maxsplit=1), "B"
        elif _VS.search(line):
            parts, loc = _VS.split(line, maxsplit=1), "N"
        else:
            bad.append(line)
            continue
        a, b = lookup.get(parts[0].strip().lower()), lookup.get(parts[1].strip().lower())
        if a is None or b is None:
            bad.append(line)
        else:
            rows.append({"team_a": a, "team_b": b, "location": loc})
    return pd.DataFrame(rows, columns=GAME_COLUMNS), bad
