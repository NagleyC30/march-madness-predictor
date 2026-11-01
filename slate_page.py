# slate_page.py — The "This Week's Slate" page (Streamlit). Called from app.py.
#
# Shows the week's 5 model-chosen toss-ups, lets you lock a pick for each game
# before tip-off, reveals the model's pick only after you've locked (or the game
# has), lets you record the winner once it's played, and keeps the season
# scoreboard. Logic lives in slate_contest.py; picks are stored by contest.py.

from datetime import date

import streamlit as st

import contest as ct
import slate_contest as sc

STATE_BADGE = {
    "open": "🟢 Open",
    "picked": "🔒 Locked in",
    "missed": "⏭️ Missed — dropped",
    "final": "🏁 Final",
}


def _day(d):
    d = date.fromisoformat(str(d))       # no %-d: it crashes strftime on Windows
    return f"{d:%a %b} {d.day}"


def _week_label(week_start):
    d = date.fromisoformat(week_start)
    return f"Week of {d:%b} {d.day}, {d.year}"


def _when(g):
    tip = g.tip_et.lstrip("0") if g.tip_et else ""
    return f"{_day(g.date)}, {tip} ET" if tip else f"{_day(g.date)} (time TBA)"


def _matchup(g):
    if g.neutral:
        return f"**{g.away}** vs **{g.home}** (neutral)"
    return f"**{g.away}** @ **{g.home}**"


def _game_card(g):
    with st.container(border=True):
        top, badge = st.columns([4, 2])
        lock_note = ("picks close at tip-off" if g.tip_et
                     else "picks close at 12:00 a.m. ET that day")
        top.markdown(f"{_matchup(g)}  \n<small>{_when(g)} · {lock_note} · "
                     f"model uncertainty {g.entropy_bits:.3f} bits</small>",
                     unsafe_allow_html=True)
        badge.markdown(f"**{STATE_BADGE[g.state]}**")

        if g.state == "open":
            choice = st.radio("Your pick", [g.away, g.home], horizontal=True,
                              key=f"pick_{g.game_id}", index=None)
            if st.button("🔒 Lock my pick", key=f"lock_{g.game_id}",
                         disabled=choice is None, type="primary"):
                try:
                    sc.lock_pick(g._asdict(), choice)
                    st.rerun()
                except ValueError as e:          # incl. contest.AlreadyPicked
                    st.error(str(e))
            st.caption("🤖 The model's pick stays hidden until you lock yours.")
            return

        model_line = f"🤖 Model: **{g.model_pick}** ({g.model_prob_pick:.1%})"
        if g.state in ("picked", "final"):
            you = f"🧑 You: **{g.user_pick}**"
            if g.state == "final":
                you += " ✅" if g.user_correct else " ❌"
                model_line += " ✅" if g.model_correct else " ❌"
            st.markdown(f"{you} &nbsp;·&nbsp; {model_line}")
        else:
            st.markdown(model_line)

        if g.state == "picked" and g.locked:
            c1, c2 = st.columns([3, 1])
            winner = c1.selectbox("Who won?", [g.away, g.home], index=None,
                                  key=f"win_{g.game_id}", placeholder="Who won?",
                                  label_visibility="collapsed")
            if c2.button("Record result", key=f"settle_{g.game_id}",
                         disabled=winner is None):
                sc.record_winner(g.pick_id, winner)
                st.rerun()
        elif g.state == "final":
            st.caption(f"Winner: {g.winner}")


def _scoreboard(board):
    sb = sc.scoreboard(board)
    st.subheader("🏆 Slate scoreboard")
    if sb["n"] == 0:
        st.info("No results yet. Lock your picks, then record each winner on its "
                "card after the game.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("🧑 You", f"{sb['user_w']}–{sb['n'] - sb['user_w']}",
              f"{sb['user_w'] / sb['n']:.0%} correct")
    c2.metric("🤖 Model", f"{sb['model_w']}–{sb['n'] - sb['model_w']}",
              f"{sb['model_w'] / sb['n']:.0%} correct")
    lead = sb["user_w"] - sb["model_w"]
    c3.metric("Lead", "Even" if lead == 0 else (f"You +{lead}" if lead > 0
                                                 else f"Model +{-lead}"))
    if sb["disagree"]:
        st.markdown(f"**When you disagreed** ({sb['disagree']} games): you were right "
                    f"**{sb['dis_user']}**, the model **{sb['dis_model']}**. "
                    f"You agreed on {sb['agree']}.")
    st.caption(f"Missed picks ({sb['missed']}) don't count for either side. "
               "Slate games also appear in the Me vs Machine page's overall record.")


def render():
    st.title("🗓️ This Week's Slate")
    st.markdown(
        "Every week the model picks the **5 games it's least sure about** "
        "(highest entropy, both teams in the national top 75) and freezes its "
        "picks. You pick the same games blind. Each game locks at **tip-off**; a "
        "game you don't pick is dropped from both records.")

    ok, hint = ct.check_storage()
    if not ok:
        st.error("**Can't reach the picks database, so the slate is paused.**  \n" + hint)
        return

    slates = sc.load_slates()
    if slates.empty:
        st.info("No slate yet. Run `python weekly_slate.py` on your PC, then "
                "commit and push `data/slates/` (see the setup guide).")
        return

    board = sc.game_board(slates, sc.slate_picks())
    weeks = (board.drop_duplicates("slate_id")[["slate_id", "week_start"]]
             .sort_values("week_start", ascending=False))
    current = sc.current_slate_id(slates)
    ids = weeks.slate_id.tolist()
    starts = weeks.set_index("slate_id").week_start
    sel = st.selectbox("Week", ids, index=ids.index(current) if current in ids else 0,
                       format_func=lambda s: _week_label(starts[s])
                       + ("  (this week)" if s == current else ""))

    week = board[board.slate_id == sel].sort_values(["lock_at", "game_id"])
    n_open = int((week.state == "open").sum())
    if n_open:
        nxt = week[week.state == "open"].iloc[0]
        st.markdown(f"**{n_open} game{'s' if n_open != 1 else ''} still open.** "
                    f"Next lock: {_when(nxt)}.")
    for g in week.itertuples(index=False):
        _game_card(g)

    st.divider()
    _scoreboard(board)

    past = board[board.state.isin(["final", "missed"])]
    if not past.empty:
        with st.expander(f"All past slate games ({len(past)})"):
            t = past.assign(
                Game=past.away + " at " + past.home,
                You=past.user_pick.fillna("—"),
                Result=past.winner.fillna("—"),
                Status=past.state.map(STATE_BADGE),
            )[["date", "Game", "You", "model_pick", "Result", "Status"]].rename(
                columns={"date": "Date", "model_pick": "Model"})
            st.dataframe(t[::-1], hide_index=True, width="stretch")
