# test_slate_contest.py — offline checks for the weekly-slate contest
# (slate_contest.py + slate_page.py on top of contest.py).
#
#   python test_slate_contest.py                       # CSV storage
#   TEST_DB=postgresql://… python test_slate_contest.py  # also Postgres (wipes
#                                                         # the contest_picks table)

import os
import tempfile
from datetime import datetime, timezone
from unittest import mock

import pandas as pd

import contest as ct
import slate_contest as sc

UTC = timezone.utc


def at(s):
    return datetime.fromisoformat(s).astimezone(UTC)


def _slate(tmp):
    """Mon Nov 9 - Sun Nov 15 2026; one game has no listed tip time."""
    games = [("2026-11-10", "07:00 PM", "Houston", "VCU", False, "VCU", 0.5004),
             ("2026-11-11", "08:30 PM", "Kansas", "Michigan", False, "Kansas", 0.5111),
             ("2026-11-12", "06:00 PM", "SMU", "TCU", True, "TCU", 0.5074),
             ("2026-11-14", "", "St. John's", "Louisville", False, "Louisville", 0.5167),
             ("2026-11-15", "02:00 PM", "Miami FL", "Utah", False, "Miami FL", 0.5029)]
    rows = [{"slate_id": "2027_2026-11-09", "season": 2027, "week_start": "2026-11-09",
             "week_end": "2026-11-15", "game_id": f"{d}|{a}@{h}", "date": d,
             "home": h, "away": a, "neutral": n, "game_type": "",
             "model_p_home": p if mp == h else 1 - p, "model_pick": mp,
             "model_prob_pick": p, "entropy_bits": 0.999,
             "home_barthag_rank": 10, "away_barthag_rank": 20, "tip_et": t,
             "generated_utc": "2026-11-08T14:00:00+00:00",
             "model_training_data": "point-in-time", "ratings_as_of": "2026-11-08"}
            for d, t, a, h, n, mp, p in games]
    os.makedirs(os.path.join(tmp, "slates"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(tmp, "slates", "2027_2026-11-09.csv"),
                              index=False)
    return sc.load_slates(os.path.join(tmp, "slates"))


class _Store:
    """Point contest.py at a fresh CSV, or at the TEST_DB Postgres (wiped)."""

    def __init__(self, tmp, db_url=None):
        self.tmp, self.db = tmp, db_url

    def __enter__(self):
        self.patches = [mock.patch.object(ct, "DATA_DIR", self.tmp),
                        mock.patch.object(ct, "PICKS_FILE",
                                          os.path.join(self.tmp, "picks.csv"))]
        for p in self.patches:
            p.start()
        ct._engine = ct._engine_url = None
        if self.db:
            os.environ["SUPABASE_DB_URL"] = self.db
            import sqlalchemy as sa
            eng = sa.create_engine(self.db.replace("postgresql://", "postgresql+psycopg2://"))
            with eng.begin() as c:
                c.execute(sa.text("DROP TABLE IF EXISTS contest_picks"))
        else:
            os.environ.pop("SUPABASE_DB_URL", None)
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()
        os.environ.pop("SUPABASE_DB_URL", None)
        ct._engine = ct._engine_url = None


def check_lock_time():
    assert sc.lock_time("2026-11-10", "07:00 PM").astimezone(UTC).isoformat() \
        == "2026-11-11T00:00:00+00:00"                     # 7 p.m. EST
    assert sc.lock_time("2026-11-10", "").astimezone(UTC).hour == 5   # midnight EST
    assert sc.lock_time("2026-10-20", "").astimezone(UTC).hour == 4   # midnight EDT
    assert sc.lock_time("2026-11-10", "junk").astimezone(UTC).hour == 5
    assert not sc.is_locked("2026-11-10", "07:00 PM", at("2026-11-10T18:59:00-05:00"))
    assert sc.is_locked("2026-11-10", "07:00 PM", at("2026-11-10T19:00:00-05:00"))


def check_lock_pick_rules(db=None):
    with tempfile.TemporaryDirectory() as d, _Store(d, db):
        slate = _slate(d)
        g = slate.iloc[0].to_dict()                       # Tue 7 p.m.
        before = at("2026-11-10T18:30:00-05:00")
        try:
            sc.lock_pick(g, "Duke", before)
            raise AssertionError("team not in game")
        except ValueError as e:
            assert "isn't playing" in str(e)
        pid = sc.lock_pick(g, "Houston", before)
        try:
            sc.lock_pick(g, "VCU", before)
            raise AssertionError("second pick must be refused")
        except ct.AlreadyPicked:
            pass
        try:
            sc.lock_pick(g, "Houston", at("2026-11-10T19:00:01-05:00"))
            raise AssertionError("locked game must be refused")
        except ValueError as e:
            assert "locked" in str(e)
        p = ct.load_picks()
        assert len(p) == 1
        row = p.iloc[0]
        assert (row.team_a, row.team_b, row.location) == ("Houston", "VCU", "B")
        assert row.model_pick == "VCU" and row.slate_game_id == g["game_id"]
        assert sc.slate_picks().pick_id.tolist() == [pid]


def check_board_states_and_scoreboard(db=None):
    with tempfile.TemporaryDirectory() as d, _Store(d, db):
        slate = _slate(d)
        ids = slate.game_id.tolist()
        early = at("2026-11-09T12:00:00-05:00")
        sc.lock_pick(slate.iloc[0].to_dict(), "Houston", early)     # right, model wrong
        sc.lock_pick(slate.iloc[1].to_dict(), "Michigan", early)    # wrong, model right
        sc.lock_pick(slate.iloc[2].to_dict(), "TCU", early)         # agree, no result yet
        for pid, w in zip(sc.slate_picks().pick_id[:2], ["Houston", "Kansas"]):
            assert sc.record_winner(pid, w)

        now = at("2026-11-14T12:00:00-05:00")   # Sat noon: Sat game (no tip) locked
        b = sc.game_board(slate, sc.slate_picks(), now).set_index("game_id")
        assert b.loc[ids[0], "state"] == "final" and b.loc[ids[0], "user_correct"]
        assert not b.loc[ids[0], "model_correct"]
        assert b.loc[ids[1], "state"] == "final" and b.loc[ids[1], "model_correct"]
        assert b.loc[ids[2], "state"] == "picked" and b.loc[ids[2], "locked"]
        assert b.loc[ids[3], "state"] == "missed" and b.loc[ids[3], "reveal"]
        assert b.loc[ids[4], "state"] == "open" and not b.loc[ids[4], "reveal"]

        # a pick stamped after the lock is ignored, even if it got stored
        late = sc.slate_picks()
        late.loc[late.game_id == ids[2], "picked_utc"] = "2026-11-12T23:30:00+00:00"
        b2 = sc.game_board(slate, late, now).set_index("game_id")
        assert b2.loc[ids[2], "state"] == "missed"

        sb = sc.scoreboard(b.reset_index())
        assert sb["n"] == 2 and sb["user_w"] == 1 and sb["model_w"] == 1
        assert sb["disagree"] == 2 and sb["dis_user"] == 1 and sb["dis_model"] == 1
        assert sb["missed"] == 1


def check_board_with_no_picks(db=None):
    """The very first visit: a slate exists but nothing is picked yet."""
    with tempfile.TemporaryDirectory() as d, _Store(d, db):
        slate = _slate(d)
        b = sc.game_board(slate, sc.slate_picks(), at("2026-11-11T12:00:00-05:00"))
        assert b.state.tolist() == ["missed", "open", "open", "open", "open"]
        assert not b.user_correct.any() and not b.model_correct.any()
        assert sc.scoreboard(b) == {"n": 0}


def check_existing_db_upgrades(db):
    """A contest_picks table created before slates existed gains the column."""
    import sqlalchemy as sa
    with tempfile.TemporaryDirectory() as d, _Store(d, db):
        eng = sa.create_engine(db.replace("postgresql://", "postgresql+psycopg2://"))
        with eng.begin() as c:
            c.execute(sa.text(ct._SCHEMA[0]))           # the original table only
            c.execute(sa.text("""INSERT INTO contest_picks (season, team_a, team_b,
                location, user_pick, model_pick, model_prob_pick)
                VALUES (2027, 'Duke', 'Kansas', 'N', 'Duke', 'Kansas', 0.6)"""))
        assert ct.check_storage() == (True, None)
        p = ct.load_picks()
        assert len(p) == 1 and p.slate_game_id.iloc[0] == ""
        with eng.begin() as c:                           # guard covers the new column
            try:
                c.execute(sa.text("UPDATE contest_picks SET slate_game_id = 'x'"))
                raise AssertionError("slate_game_id must be immutable")
            except sa.exc.DBAPIError as e:
                assert "locked pick fields" in str(e)


def check_current_slate_id():
    with tempfile.TemporaryDirectory() as d:
        slate = _slate(d)
        from datetime import date
        assert sc.current_slate_id(slate, date(2026, 11, 12)) == "2027_2026-11-09"
        assert sc.current_slate_id(slate, date(2026, 11, 7)) == "2027_2026-11-09"
        assert sc.current_slate_id(slate, date(2026, 12, 1)) == "2027_2026-11-09"
        assert sc.current_slate_id(pd.DataFrame()) is None


def _page_app():
    import slate_page
    slate_page.render()


def check_page(db=None):
    from streamlit.testing.v1 import AppTest
    with tempfile.TemporaryDirectory() as d, _Store(d, db):
        slate = _slate(d)
        sc.lock_pick(slate.iloc[0].to_dict(), "Houston", at("2026-11-09T12:00:00-05:00"))
        frozen = at("2026-11-10T20:00:00-05:00")       # Tue 8 p.m.: game 1 tipped
        with mock.patch.object(sc, "SLATE_DIR", os.path.join(d, "slates")), \
                mock.patch.object(sc, "now_utc", lambda: frozen):
            app = AppTest.from_function(_page_app)
            app.run(timeout=60)
            assert not app.exception, app.exception
            text = " ".join(m.value for m in app.markdown)
            assert "Model: **VCU**" in text, "locked game reveals the model"
            for hidden in ("Model: **Kansas**", "Model: **TCU**",
                           "Model: **Louisville**", "Model: **Miami FL**"):
                assert hidden not in text, f"leaked {hidden}"
            assert "Next lock: Wed Nov 11, 8:30 PM ET" in text, text
            gid = slate.game_id
            app.radio(key=f"pick_{gid[1]}").set_value("Michigan").run()
            app.button(key=f"lock_{gid[1]}").click().run()
            assert not app.exception, app.exception
            text = " ".join(m.value for m in app.markdown)
            assert "Model: **Kansas**" in text, "revealed right after locking"
            # game 1 has tipped: record the winner on its card
            app.selectbox(key=f"win_{gid[0]}").set_value("Houston").run()
            app.button(key=f"settle_{gid[0]}").click().run()
            assert not app.exception, app.exception
            p = ct.load_picks().set_index("slate_game_id")
            assert p.loc[gid[0], "status"] == "settled"
            assert p.loc[gid[1], "user_pick"] == "Michigan"
            assert "🧑 You" in " ".join(m.label for m in app.metric)


if __name__ == "__main__":
    db = os.environ.get("TEST_DB")
    plain = [check_lock_time, check_current_slate_id]
    stored = [check_board_with_no_picks, check_lock_pick_rules,
              check_board_states_and_scoreboard, check_page]
    for t in plain:
        t(); print(f"ok  {t.__name__}")
    for t in stored:
        t(); print(f"ok  {t.__name__} (csv)")
    if db:
        for t in stored:
            t(db); print(f"ok  {t.__name__} (postgres)")
        check_existing_db_upgrades(db); print("ok  check_existing_db_upgrades (postgres)")
    print("\nall passed")
