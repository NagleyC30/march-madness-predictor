# test_slate_results.py — offline checks for slate_results.py + auto_settle.
#
#   python test_slate_results.py

import os
import tempfile
from datetime import date

import pandas as pd

import contest as ct
import slate_contest as sc
import slate_results as sr
import test_slate_contest as tc
from test_weekly_slate import _page_row


def check_parse_result():
    p = sr.parse_result
    assert p("W, 78-70", "Houston", "VCU") == "Houston"
    assert p("L, 70-78", "Houston", "VCU") == "VCU"
    assert p("W 78-70", "Houston", "VCU") == "Houston"
    assert p("Houston 78, VCU 70", "Houston", "VCU") == "Houston"
    assert p("VCU 80, Houston 70", "Houston", "VCU") == "VCU"
    for bad in ("", None, "W, 60-70", "L, 70-60", "W, 70-70", "Postponed", "Duke 80, UNC 70"):
        assert p(bad, "Houston", "VCU") is None, bad


def _fetcher(pages):
    return lambda url: pages.get(url.split("date=")[1][:8], "<html></html>")


def check_collect_and_save():
    with tempfile.TemporaryDirectory() as d:
        slates = tc._slate(d)                      # Nov 10-15 2026, 5 games
        res = sr.load_results(os.path.join(d, "slates"))
        assert res.empty
        # Tue game final; Wed game on the page but not played; nothing else listed
        pages = {"20261110": _page_row("Houston", "VCU", result="W, 78-70"),
                 "20261111": _page_row("Kansas", "Michigan")}
        today = date(2026, 11, 11)
        due = sr.due_games(slates, res, today)
        assert len(due) == 2
        new, report = sr.collect(slates, res, today, fetch=_fetcher(pages), pause=0,
                                 now="2026-11-11T05:00:00+00:00")
        assert new.game_id.tolist() == ["2026-11-10|Houston@VCU"]
        assert new.winner.tolist() == ["Houston"]
        assert [r[2] for r in report] == ["Houston", None]
        sr.save(new, os.path.join(d, "slates"))
        # recorded results are never overwritten and drop out of the due list
        again = sr.load_results(os.path.join(d, "slates"))
        assert again.winner.tolist() == ["Houston"]
        flipped = new.assign(winner="VCU")
        sr.save(flipped, os.path.join(d, "slates"))
        assert sr.load_results(os.path.join(d, "slates")).winner.tolist() == ["Houston"]
        assert len(sr.due_games(slates, again, today)) == 1


def check_auto_settle():
    with tempfile.TemporaryDirectory() as d, tc._Store(d):
        slates = tc._slate(d)
        g0, g1 = slates.iloc[0].to_dict(), slates.iloc[1].to_dict()
        sc.lock_pick(g0, "Houston", tc.at("2026-11-10T18:30:00-05:00"))
        sc.lock_pick(g1, "Michigan", tc.at("2026-11-11T18:30:00-05:00"))
        results = pd.DataFrame([
            {"game_id": g0["game_id"], "winner": "Houston"},
            {"game_id": g1["game_id"], "winner": "Duke"}])        # not in the game
        now = tc.at("2026-11-12T12:00:00-05:00")
        board = sc.game_board(slates, sc.slate_picks(), at=now)
        assert sc.auto_settle(board, results) == 1
        board = sc.game_board(slates, sc.slate_picks(), at=now)
        assert sc.auto_settle(board, results) == 0              # idempotent
        st = board.set_index("game_id").state
        assert st[g0["game_id"]] == "final" and st[g1["game_id"]] == "picked"
        assert board.set_index("game_id").loc[g0["game_id"], "user_correct"]
        # a game that hasn't tipped is never settled, even with a result row
        early = sc.game_board(slates, sc.slate_picks(), at=tc.at("2026-11-11T12:00:00-05:00"))
        assert sc.auto_settle(early, pd.DataFrame([
            {"game_id": g1["game_id"], "winner": "Kansas"}])) == 0


if __name__ == "__main__":
    for t in (check_parse_result, check_collect_and_save, check_auto_settle):
        t(); print(f"ok  {t.__name__}")
    print("\nall passed")
