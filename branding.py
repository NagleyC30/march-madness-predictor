# branding.py — B.O.B. brand identity, runtime theme selector, and logo slot
# (project checklist items 13 + 16, the Phase-3 branding pass).
#
# The project is renamed **B.O.B. (Betting on Basketball)** and framed around the
# real question: can the model beat the market? This module centralizes the name,
# the swappable colour themes, and the logo so every page picks them up from one
# place.
#
# Theme mechanism: the *base* palette (hardwood cream background, basketball-
# orange primary, warm text/borders/charts) lives natively in
# .streamlit/config.toml so it themes every surface Streamlit renders. On top of
# that we let the user pick an **accent palette** at runtime by injecting CSS —
# it recolours the accent-driven surfaces (primary buttons, links, the brand
# mark, the accent bar, sidebar selection) so a different accent stays cohesive.
# The default "Hardwood" accent matches config.toml's primaryColor exactly, so
# with no change the runtime layer is a no-op and native widgets stay in step.

import os

import streamlit as st

APP_NAME = "B.O.B."
APP_TAGLINE = "Betting on Basketball"
APP_ICON = "🏀"

ASSETS_DIR = "assets"
# The real logo (dropped in later) wins; otherwise the shipped placeholder.
LOGO_CANDIDATES = ("logo.svg", "logo.png", "logo_placeholder.svg")
ICON_CANDIDATES = ("icon.svg", "icon.png", "icon_placeholder.svg")

# name -> (accent hex, short blurb). The first is the default and matches
# .streamlit/config.toml's primaryColor so native widgets stay in step.
THEMES = {
    "Hardwood": ("#E8590C", "warm basketball orange"),
    "Baseline Blue": ("#1C6DD0", "cool court blue"),
    "Net Green": ("#2F9E44", "nothing-but-net green"),
    "Buzzer Purple": ("#7048E8", "overtime purple"),
    "Classic Red": ("#FF4B4B", "the Streamlit red"),
}
DEFAULT_THEME = "Hardwood"


def _first_existing(names):
    for n in names:
        p = os.path.join(ASSETS_DIR, n)
        if os.path.exists(p):
            return p
    return None


def render_logo():
    """Show the brand logo top-left (and in the sidebar) if an asset exists.
    Falls back silently to the text/emoji branding when none is present."""
    logo = _first_existing(LOGO_CANDIDATES)
    if logo is None:
        return
    icon = _first_existing(ICON_CANDIDATES) or logo
    try:
        st.logo(logo, icon_image=icon, link=None, size="large")
    except TypeError:                      # older Streamlit signature
        st.logo(logo, icon_image=icon)


def apply_theme(accent):
    """Inject CSS so the chosen accent recolours the app's reachable surfaces.

    The hardwood base (background, text, borders, charts, sidebar) comes from
    config.toml; this only overrides the accent-driven bits so a non-default
    accent choice stays cohesive on top of the native theme."""
    st.markdown(
        f"""
        <style>
          :root {{ --bob-accent: {accent}; }}

          /* Primary buttons (cover old + new Streamlit selectors) */
          .stButton > button[kind="primary"],
          .stButton > button[data-testid="baseButton-primary"],
          [data-testid="stBaseButton-primary"] {{
              background-color: var(--bob-accent) !important;
              border-color: var(--bob-accent) !important;
          }}
          .stButton > button[kind="primary"]:hover,
          [data-testid="stBaseButton-primary"]:hover {{
              filter: brightness(0.92);
              border-color: var(--bob-accent) !important;
          }}

          /* Links + the brand mark */
          a, .bob-brand-name {{ color: var(--bob-accent) !important; }}

          /* The accent bar under the brand header — a painted baseline stripe */
          div.bob-accent-bar {{
              height: 5px; border-radius: 3px; margin: .1rem 0 1.1rem 0;
              background: linear-gradient(90deg, var(--bob-accent), #C77D2E);
          }}

          /* Section headings get an accent left-rule, like a court sideline */
          .main h2, .main h3 {{
              border-left: 4px solid var(--bob-accent);
              padding-left: .55rem;
          }}

          /* Metric values in the accent so KPIs pop off the hardwood */
          [data-testid="stMetricValue"] {{ color: var(--bob-accent); }}

          /* Sidebar nav: highlight the selected page in the accent */
          section[data-testid="stSidebar"] label[data-baseweb="radio"] div:first-child {{
              border-color: var(--bob-accent);
          }}

          /* Tabs: underline the active tab in the accent */
          .stTabs [data-baseweb="tab-highlight"] {{
              background-color: var(--bob-accent) !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header():
    """The B.O.B. wordmark + tagline for the top of the Overview page."""
    st.markdown(
        f"""
        <div style="display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;">
          <span class="bob-brand-name" style="font-size:2.4rem;font-weight:800;
              letter-spacing:.02em;">{APP_ICON} {APP_NAME}</span>
          <span style="font-size:1.1rem;opacity:.75;">— {APP_TAGLINE}</span>
        </div>
        <div class="bob-accent-bar"></div>
        """,
        unsafe_allow_html=True,
    )


def theme_selector(container=None):
    """Sidebar (or given container) theme picker. Returns the accent hex and
    applies the theme. Persists across reruns via session state."""
    box = container or st.sidebar
    names = list(THEMES)
    # Seed the default ONCE, then let the widget own its value through `key`
    # alone. Passing both `index` (derived from session_state) and `key` is the
    # bug that crashed the app right after the first theme change: once
    # `bob_theme` is in session_state, current Streamlit raises rather than
    # letting a default coexist with a keyed value.
    if "bob_theme" not in st.session_state:
        st.session_state["bob_theme"] = DEFAULT_THEME
    choice = box.selectbox(
        "🎨 Theme", names,
        format_func=lambda n: f"{n} — {THEMES[n][1]}",
        key="bob_theme",
    )
    accent = THEMES[choice][0]
    apply_theme(accent)
    return accent
