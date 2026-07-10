# Brand assets

The app (`branding.py`) looks for a logo here and falls back to the shipped
placeholders if the real files aren't present yet.

## Drop-in files (replace the placeholders)

| File | Used for | Notes |
|------|----------|-------|
| `logo.svg` **or** `logo.png` | Full wordmark, top-left of the app + sidebar | Wide wordmark, ~300×76 aspect. SVG preferred; transparent-background PNG fallback. |
| `icon.svg` **or** `icon.png` | Collapsed-sidebar mark / small icon | Square (e.g. 64×64 or 512×512), transparent background. |

`branding.py` prefers `logo.svg` → `logo.png` → `logo_placeholder.svg` (same for
the icon), so just add the real files and they take over automatically — no code
change needed.

## Spec for the designed logo (the gf brief)

- **Formats:** SVG (vector, ideal) **and** a transparent-background PNG fallback.
- **Light + dark variants:** the app supports the native Streamlit light/dark
  toggle, so provide a version that reads on **both** — or two files and we can
  wire variant-switching.
- **Square icon crop:** a standalone square mark for the favicon / collapsed
  sidebar (`icon.*`), not just the wide wordmark.
- **Name + tagline:** **B.O.B.** — *Betting on Basketball*.

The placeholders (`logo_placeholder.svg`, `icon_placeholder.svg`) show a basketball
+ wordmark and are intentionally simple — they're meant to be replaced.
