# branding.py — B.O.B. brand identity, runtime theme selector, and logo slot
# (project checklist items 13 + 16, the Phase-3 branding pass).
#
# The project is renamed **B.O.B. (Betting on Basketball)** and framed around the
# real question: can the model beat the market? This module centralizes the name,
# the swappable colour themes, and the logo so every page picks them up from one
# place.
#
# Theme mechanism: Streamlit's base light/dark toggle stays native (the ⋮ menu).
# On top of that we let the user pick an **accent palette** at runtime by
# injecting CSS — it recolours the things CSS can reach cohesively (primary
# buttons, links, headers, the brand mark, focus rings). Native widget accents
# (slider/checkbox) follow the config.toml primaryColor and aren't runtime-
# swapped; the palettes are built around that default so it stays coherent.

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
    """Inject CSS so the chosen accent recolours the app's reachable surfaces."""
    st.markdown(
        f"""
        <style>
          :root {{ --bob-accent: {accent}; }}
          /* Primary buttons */
          .stButton > button[kind="primary"],
          .stButton > button[data-testid="baseButton-primary"] {{
              background-color: var(--bob-accent);
              border-color: var(--bob-accent);
          }}
          .stButton > button[kind="primary"]:hover {{
              filter: brightness(0.92);
              border-color: var(--bob-accent);
          }}
          /* Links + the brand mark */
          a, .bob-brand-name {{ color: var(--bob-accent); }}
          /* Page + section headings get an accent-tinted top rule feel */
          h1 span, h2 span {{ color: inherit; }}
          div.bob-accent-bar {{
              height: 4px; border-radius: 2px; margin: .1rem 0 1rem 0;
              background-color: var(--bob-accent);
          }}
          /* Sidebar nav: highlight the selected page in the accent */
          section[data-testid="stSidebar"] label[data-baseweb="radio"] div:first-child {{
              border-color: var(--bob-accent);
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
    choice = box.selectbox(
        "🎨 Theme", names,
        index=names.index(st.session_state.get("bob_theme", DEFAULT_THEME)),
        format_func=lambda n: f"{n} — {THEMES[n][1]}",
        key="bob_theme",
    )
    accent = THEMES[choice][0]
    apply_theme(accent)
    return accent
