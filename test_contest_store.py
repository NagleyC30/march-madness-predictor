# Exercises contest.py's storage layer. CSV mode always; Postgres mode when
# SUPABASE_DB_URL is set (point it at a scratch database — the table is wiped).
import os, tempfile
import sqlalchemy as sa
import contest as ct

def lifecycle():
    pid = ct.add_pick(2026, "Duke", "Kansas", "N", "Duke", "Kansas", 0.61,
                      user_conf=0.7, ml_a=-150, ml_b=130)
    pid2 = ct.add_pick(2026, "Houston", "Baylor", "A", "Baylor", "Houston", 0.7)
    df = ct.load_picks()
    assert list(df["status"]) == ["pending", "pending"] and df["created_utc"].str.len().min() > 0
    assert ct.settle_pick(pid, "Kansas")
    assert ct.settle_pick(pid2, "Houston")
    sb = ct.scoreboard()
    assert sb["n_settled"] == 2 and sb["model_w"] == 2 and sb["user_w"] == 0, sb
    assert sb["n_user_bets"] == 1            # only the game with a line
    d = ct.load_picks()
    assert (d["settled_utc"] != "").all() and (d["actual_winner"] != "").all()
    pid3 = ct.add_pick(2026, "A", "B", "N", "A", "B", 0.5)
    ct.delete_pick(pid3)
    assert len(ct.load_picks()) == 2
    return pid

os.environ.pop("SUPABASE_DB_URL", None) if not os.environ.get("TEST_DB") else None
url = os.environ.get("TEST_DB")

# ---- CSV mode
os.environ.pop("SUPABASE_DB_URL", None)
with tempfile.TemporaryDirectory() as d:
    ct.DATA_DIR, ct.PICKS_FILE = d, os.path.join(d, "p.csv")
    assert ct.storage_backend() == "csv"
    lifecycle(); print("csv OK")

# ---- Postgres mode
if url:
    os.environ["SUPABASE_DB_URL"] = url
    eng = sa.create_engine(url.replace("postgresql://", "postgresql+psycopg2://"))
    with eng.begin() as c: c.execute(sa.text("DROP TABLE IF EXISTS contest_picks"))
    assert ct.storage_backend() == "supabase"
    pid = lifecycle()
    # integrity: settled rows are final, locked fields immutable, no re-settle
    assert not ct.settle_pick(pid, "Duke"), "re-settle must be refused"
    for sql in ["UPDATE contest_picks SET user_pick='Kansas' WHERE id=:i",
                "UPDATE contest_picks SET actual_winner='Duke' WHERE id=:i",
                "DELETE FROM contest_picks WHERE id=:i"]:
        try:
            with eng.begin() as c: c.execute(sa.text(sql), {"i": pid})
            raise SystemExit(f"guard failed: {sql}")
        except sa.exc.DBAPIError:
            pass
    pend = ct.add_pick(2026, "X", "Y", "N", "X", "Y", 0.5)
    for sql in ["UPDATE contest_picks SET user_pick='Y' WHERE id=:i",
                "UPDATE contest_picks SET created_utc=now()-interval '1 day' WHERE id=:i"]:
        try:
            with eng.begin() as c: c.execute(sa.text(sql), {"i": pend})
            raise SystemExit(f"guard failed: {sql}")
        except sa.exc.DBAPIError:
            pass
    ct.delete_pick(pend)
    print("postgres OK (lifecycle + immutability guard)")
