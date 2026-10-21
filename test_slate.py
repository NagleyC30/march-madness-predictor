# Checks for slate.py: gate, constraints, blind-entry safety, parsing, and that
# the Me vs Machine page renders and the suggest button works end to end.
import pandas as pd
import game_model as gm
import slate as sl

art = gm.load_model()
ratings = gm.load_ratings()
model, features = art["model"], art["meta"]["features"]
ry = ratings[ratings["YEAR"] == 2026]
top = ry.sort_values("barthag", ascending=False)["TEAM"].tolist()
bottom = ry.sort_values("barthag")["TEAM"].tolist()

# parsing
g, bad = sl.parse_games(f"{top[0]} vs {top[1]}\n{top[2]} at {top[3]}\nnonsense\nZzz vs {top[0]}", ry["TEAM"])
assert list(g["location"]) == ["N", "B"] and len(bad) == 2, (g, bad)

# gate: two bottom teams excluded; unknown team / played excluded
games = pd.DataFrame([
    {"team_a": bottom[0], "team_b": bottom[1], "location": "N"},
    {"team_a": top[0], "team_b": top[1], "location": "N", "played": True},
    {"team_a": top[0], "team_b": top[1], "location": "N"},
])
sc = sl.score_games(games, ry, model, features)
assert len(sc) == 1 and sc.iloc[0].team_a == top[0]

# constraints: no team twice, per-day cap, size, floor, lean mix
pairs = [{"team_a": top[i], "team_b": top[j], "location": "N", "date": f"d{(i+j) % 2}"}
         for i in range(0, 30) for j in range(i + 1, 30)]
sc = sl.score_games(pd.DataFrame(pairs), ry, model, features)
sl_ = sl.build_slate(sc)
assert 0 < len(sl_) <= sl.SLATE_SIZE
used = list(sl_.team_a) + list(sl_.team_b)
assert len(used) == len(set(used)), "team appears twice"
assert sl_.groupby("date").size().max() <= sl.PER_DAY_CAP
assert (sl_["score"] >= sl.QUALITY_FLOOR).all()
assert (sl_["p_max"] >= sl.LEAN_PROB).sum() >= min(sl.MIN_LEAN, (sc[sc.score >= sl.QUALITY_FLOOR].p_max >= sl.LEAN_PROB).sum())
assert sl.build_slate(sc, floor=2.0).empty            # shrinks, never pads

# blind entry: public view leaks no probability / favorite
pv = sl.public_view(sl_)
assert not {"p_a", "p_max", "competitiveness"} & set(pv.columns)

# UI end-to-end
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=180).run()
at.sidebar.radio[0].set_value("Me vs Machine").run()
assert not at.exception, at.exception
at.text_area(key="mvm_candidates").set_value(
    f"{top[0]} vs {top[1]}\n{top[2]} vs {top[3]}\n{top[4]} at {top[5]}\nbogus line").run()
at.button(key="mvm_suggest").click().run()
assert not at.exception, at.exception
assert len(at.dataframe) >= 1, "suggestion table not shown"
print("slate tests OK")
