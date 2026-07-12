# Smoke test: run every page of app.py via Streamlit's AppTest and assert no
# exception is raised. Not committed as a unit test suite — just a boot check.
from streamlit.testing.v1 import AppTest

PAGES = ["Overview", "Betting Lab", "Me vs Machine", "Bracket Predictions",
         "Head-to-Head", "Game Predictor", "Model Accuracy",
         "Backtest & Calibration", "Model Bake-off", "How the Models Work",
         "Custom Metric", "Data Explorer"]

for page in PAGES:
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    assert not at.exception, f"Overview load failed: {at.exception}"
    # switch sidebar radio to the target page
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page} raised: {at.exception}"
    print(f"  OK: {page}")

print("ALL PAGES RENDER WITHOUT EXCEPTION")
