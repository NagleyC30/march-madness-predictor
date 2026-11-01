# test_weekly_slate.py — offline checks for weekly_slate.py (no network).
#
#   python test_weekly_slate.py
#
# Uses the committed model + data/ratings.csv, and a synthetic super_sked so it
# runs anywhere (including CI) without touching barttorvik.com.

import itertools
import os
import tempfile
from datetime import date
from unittest import mock

import numpy as np
import pandas as pd

import game_model as gm
import weekly_slate as ws


def _sked_row(d, away, home, neutral="0", gtype="1", winner=""):
    r = [""] * 30
    r[1], r[6], r[7], r[8], r[14], r[29] = d, gtype, neutral, away, home, winner
    return ",".join(f'"{x}"' for x in r)



# Two rows copied verbatim from barttorvik.com/schedule.php?date=20261102
# (2026-11-01): a neutral-site D1 game and a home game vs a non-D1 opponent.
REAL_ROWS = """<tr>
		<th style="text-align:left" class="mobileout" id="timeheader"><a href="?sort=time&amp;date=20261102&amp;conlimit=">Time</a></th>		
		<th style="text-align:left">Matchup</th>
		<th style="text-align:left"><a href="?sort=close&amp;date=20261102&amp;conlimit=">T-Rank Line</a></th>
		<th class="mobileout"><a title="Torvik Thrill Quotient" href="?date=20261102&amp;conlimit=">TTQ</a></th>
		<th style="text-align:left">Result</th>
		</tr>
<tr style=" ">
<td title="Florida -4.5 o/u 161.5" style="text-align:left;white-space:nowrap;color:black;font-weight: normal" class="mobileout"> <span class="gametime">05:00 PM<span></span></span></td>

<td style="text-align:left" title="Benchmark International Arena — Tampa, FL"><span class="mobileout" style="font-size:8px;">3</span> <a href="team.php?team=Florida&amp;year=2027">Florida</a> <span class="mobileonly"><br></span>vs <span class="mobileout" style="font-size:8px;">12</span> <a href="team.php?team=Miami+FL&amp;year=2027">Miami FL</a><span style="color:#ebb609;font-size:10px;font-style:italic"><span class="mobileout" style="color:gray"> ESPN2</span><span></span></span></td>
<td style="text-align:left; "><a href="trank.php?year=2027&amp;t1l=Florida&amp;t2l=Miami+FL&amp;">Florida -2.8<span class="mobileonly">,<br></span> 82-79 (60%)</a></td>
<td class="mobileout" style="text-align:center;background-color:#A5D9B4">90</td>
<td style="text-align:left;"><a style="" href="box.php?muid=FloridaMiami+FL11-2&amp;year=2027"></a></td></tr>
<tr style=" background-color:#eee">
<td style="text-align:left;white-space:nowrap;color:black;font-weight: normal" class="mobileout"> <span class="gametime">06:00 PM<span></span></span></td>

<td style="text-align:left" title="UWF Field House — Pensacola, FL"><span class="mobileout" style="font-size:8px;">0</span> <a href="team.php?team=Southeastern+Baptist&amp;year=2027">Southeastern Baptist</a> <span class="mobileonly"><br></span>at <span class="mobileout" style="font-size:8px;">363</span> <a href="team.php?team=West+Florida&amp;year=2027">West Florida</a><span style="color:#ebb609;font-size:10px;font-style:italic"><span class="mobileout" style="color:gray"> ESPN+</span><span></span></span></td>
<td style="text-align:left; "><a href="trank.php?year=2027&amp;t1l=Southeastern+Baptist&amp;t2l=West+Florida&amp;">West Florida (100%)</a></td>
<td class="mobileout" style="text-align:center;background-color:#F9A2A5">-16</td>
<td style="text-align:left;"><a style="" href="box.php?muid=Southeastern+BaptistWest+Florida11-2&amp;year=2027"></a></td></tr>
"""


def _page_row(away, home, neutral=False, tip="07:00 PM", ra=10, rh=20, result=""):
    """A row in the same markup as REAL_ROWS, for any two teams."""
    q = lambda t: t.replace(" ", "+").replace("&", "%26")
    return (f'<tr style=" "><td class="mobileout"> <span class="gametime">{tip}<span></span></span></td>'
            f'<td style="text-align:left" title="Some Arena — City, ST"><span class="mobileout" style="font-size:8px;">{ra}</span> '
            f'<a href="team.php?team={q(away)}&amp;year=2027">{away.replace("&", "&amp;")}</a> <span class="mobileonly"><br></span>'
            f'{"vs" if neutral else "at"} <span class="mobileout" style="font-size:8px;">{rh}</span> '
            f'<a href="team.php?team={q(home)}&amp;year=2027">{home.replace("&", "&amp;")}</a>'
            f'<span style="color:#ebb609"><span class="mobileout" style="color:gray"> ESPN+</span></span></td>'
            f'<td><a href="trank.php">x</a></td><td class="mobileout">50</td>'
            f'<td style="text-align:left;"><a href="box.php?muid=x">{result}</a></td></tr>')


def test_parse_real_schedule_page():
    g = ws.parse_schedule_page(REAL_ROWS, "2026-11-02")
    assert len(g) == 1, "header skipped, non-D1 game dropped"
    r = g.iloc[0]
    assert (r.AWAY, r.HOME, bool(r.NEUTRAL)) == ("Florida", "Miami FL", True)
    assert r.TIP_ET == "05:00 PM" and r.TV == "ESPN2" and r.YEAR == 2027
    assert r.VENUE.startswith("Benchmark International Arena") and not r.PLAYED
    h = ws.parse_schedule_page(_page_row("Texas A&M", "Iowa St."), "2026-11-03").iloc[0]
    assert (h.AWAY, h.HOME, bool(h.NEUTRAL)) == ("Texas A&M", "Iowa St.", False)
    done = ws.parse_schedule_page(_page_row("A", "B", result="W, 70-60"), "2026-11-03")
    assert done.PLAYED.all()


def test_entropy():
    assert abs(ws.entropy_bits(0.5) - 1.0) < 1e-9
    assert ws.entropy_bits(0.0) < 1e-9 and ws.entropy_bits(1.0) < 1e-9
    assert ws.entropy_bits(0.6) == ws.entropy_bits(0.4)


def test_weeks_and_seasons():
    assert ws.week_bounds(date(2026, 11, 12)) == (date(2026, 11, 9), date(2026, 11, 15))
    assert ws.next_week_start(date(2026, 11, 8)) == date(2026, 11, 9)   # Sun -> Mon
    assert ws.next_week_start(date(2026, 11, 9)) == date(2026, 11, 9)   # Mon -> same
    assert ws.season_for(date(2026, 11, 9)) == 2027
    assert ws.season_for(date(2027, 3, 1)) == 2027


def test_parse_schedule_keeps_unplayed():
    text = "\n".join([
        _sked_row("11/10/26", "Duke", "Kansas", winner="Kansas"),         # played
        _sked_row("11/12/26", "Houston", "Florida"),                      # unplayed
        _sked_row("11/12/26", "Illinois", "Virginia", neutral="1"),       # neutral
        _sked_row("11/13/26", "Duke", "Some D2", gtype="99"),             # non-D1
        _sked_row("not a date", "A", "B"),                                # junk
    ])
    s = ws.parse_schedule(2027, text=text)
    assert len(s) == 3
    assert s.loc[s.HOME == "Kansas", "PLAYED"].item()
    assert not s.loc[s.HOME == "Florida", "PLAYED"].item()
    assert s.loc[s.HOME == "Virginia", "NEUTRAL"].item()


def _real_scored():
    art = gm.load_model()
    feats = art["meta"]["features"]
    r = gm.load_ratings()
    r = r[r.YEAR == r.YEAR.max()]
    top = r.sort_values("barthag", ascending=False).TEAM.head(40).tolist()
    games = pd.DataFrame([{"YEAR": 2027, "DATE": "2026-11-12", "HOME": h,
                           "AWAY": a, "NEUTRAL": False, "GAME_TYPE": "nonconf"}
                          for h, a in itertools.permutations(top[:12], 2)])
    return ws.score_games(games, r, art["model"], feats), art


def test_score_and_select():
    scored, _ = _real_scored()
    assert len(scored) == 12 * 11
    assert scored.P_HOME.between(0, 1).all()
    # home court should help: P(A home vs B) > P(A away at B) on average
    a = scored.set_index(["HOME", "AWAY"]).P_HOME
    flipped = np.array([1 - a[(aw, h)] for h, aw in a.index])
    assert (a.values > flipped).mean() > 0.8

    slate = ws.select_slate(scored, k=5)
    assert len(slate) == 5
    teams = list(slate.HOME) + list(slate.AWAY)
    assert len(set(teams)) == 10, "no team twice"
    assert slate.ENTROPY.is_monotonic_decreasing
    # the slate's least-uncertain game is still at least as uncertain as the
    # median game in the pool
    assert slate.ENTROPY.min() >= scored.ENTROPY.median()


def test_gate():
    scored, _ = _real_scored()
    scored = scored.copy()
    scored["BARTHAG_RANK_A"] = 200                 # every away team is weak
    assert ws.select_slate(scored, k=5, both_top=75).empty
    assert len(ws.select_slate(scored, k=5, both_top=None)) == 5


def test_frozen_slate_not_regenerated():
    scored, art = _real_scored()
    slate = ws.select_slate(scored, k=5)
    rows = ws.to_slate_rows(slate, 2027, date(2026, 11, 9), art["meta"], "2026-11-08")
    assert list(rows.columns) == ws.SLATE_COLUMNS
    assert (rows.model_prob_pick >= 0.5).all()
    assert rows.model_training_data.eq("point-in-time").all()
    with tempfile.TemporaryDirectory() as d, mock.patch.object(ws, "SLATE_DIR", d):
        path = os.path.join(d, "2027_2026-11-09.csv")
        rows.to_csv(path, index=False)
        with mock.patch.object(ws, "parse_schedule",
                               side_effect=AssertionError("must not refetch")):
            again = ws.build_slate(date(2026, 11, 11))
        assert again.model_pick.tolist() == rows.model_pick.tolist()


def test_end_to_end_mocked_network():
    r = gm.load_ratings()
    r27 = r[r.YEAR == r.YEAR.max()]
    teams = r27.sort_values("barthag", ascending=False).TEAM.head(90).tolist()
    rng = np.random.default_rng(1)
    pages = {}
    for day in range(9, 16):                       # Mon Nov 9 .. Sun Nov 15
        rows = []
        for _ in range(40):
            a, h = rng.choice(teams, 2, replace=False)
            rows.append(_page_row(a, h, neutral=rng.random() < .25))
        pages[f"202611{day:02d}"] = "<table>" + "".join(rows) + "</table>"
    fetch = lambda url: pages.get(url.split("date=")[1][:8], "<table></table>")
    with tempfile.TemporaryDirectory() as d, \
            mock.patch.object(ws, "SLATE_DIR", d), \
            mock.patch.object(ws, "_get", fetch), \
            mock.patch.object(ws.fd, "parse_team_results", lambda y: r27):
        empty = ws.build_slate(date(2026, 11, 3), today=date(2026, 11, 1))   # no games that week
        assert empty.empty and not os.listdir(d), "empty slate must not be frozen"
        out = ws.build_slate(date(2026, 11, 9), today=date(2026, 11, 8))
        assert len(out) == 5
        assert os.path.exists(os.path.join(d, "2027_2026-11-09.csv"))
        assert out.date.between("2026-11-09", "2026-11-15").all()
        assert (out.tip_et == "07:00 PM").all()
        # built mid-week (Wed), the slate skips Mon-Wed games
        mid = ws.build_slate(date(2026, 11, 9), today=date(2026, 11, 11), overwrite=True)
        assert (mid.date >= "2026-11-12").all()


def test_cli_parses():
    # the Sunday GitHub Action runs this exact entry point
    with mock.patch.object(ws, "build_slate") as b:
        ws.main(["--week", "2026-11-11"])
    (wk,), kw = b.call_args
    assert wk == date(2026, 11, 11) and kw["k"] == 5 and kw["both_top"] == 75


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
