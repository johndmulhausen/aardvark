"""In-app font size switcher (CCv2 component).

Owns one zero-height CCv2 component (declared at module import) plus the
public :func:`mount_font_size_switcher` helper that the entry script
calls on every rerun. The component renders nothing visible — its inline
JS injects (or updates) a single ``<style>`` tag in ``document.head``
that overrides the root ``html`` font size. Streamlit's ``baseFontSize``
is set via ``.streamlit/config.toml`` and is only read at app boot, so
the only way to change body / heading / caption / code sizes at runtime
is to override the CSS — which is exactly what this component does.

Why a CCv2 component instead of ``st.html``?
--------------------------------------------
``st.html`` injects its content inside Streamlit's per-element iframe
(per the host-document anchor pattern documented in
:mod:`app_pages.diff`), which means a ``<style>`` block written through
``st.html`` only styles the iframe's body, not the host page. CCv2
components share the host page's ``document`` (no iframe — see the CCv2
contract under "CCv2 model"), so the component's JS can mutate
``document.head`` directly and have the rule apply globally.

Why not ``localStorage``?
-------------------------
Unlike :mod:`theme_switcher` (where Streamlit's frontend reads
``localStorage`` on app boot to pick a theme), Streamlit has no
runtime hook for font size. The font size only ever applies because we
inject the CSS rule, so persisting in ``localStorage`` would be
redundant — the Python-side value (loaded from
``~/.wb_coding_agent/preferences.json`` on every session start) is the
source of truth, and we hand that directly to the component on every
mount. The component is idempotent (uses a known style-tag id), so
re-mounting it on every page is cheap.

Why ``!important``?
-------------------
Streamlit's bundled CSS sets the root ``html`` font-size at higher
specificity than a plain rule. ``!important`` is the smallest hammer
that reliably wins; nothing else needs ``!important`` because every
downstream Streamlit token cascades from the root size in ``rem``.
"""
from __future__ import annotations

import streamlit as st

# Inline HTML: the component's own DOM is intentionally empty. The host
# div is set to ``display:none`` so the component takes up no vertical
# space in the layout regardless of where it's mounted (we mount it from
# the top-level entry script, before any page content renders).
_HTML = "<div data-wb-font-size-switcher-host style='display:none'></div>"

_JS = r"""
export default function (component) {
  const data = component.data || {};
  const sizePx = String(data.sizePx || "").trim();
  const STYLE_ID = "wb-font-size-override";

  // The component runs in shadow DOM by default, but ``document`` here
  // is the host page's document - we can mutate ``document.head``
  // directly (mirroring ``chat_input.py``). The style tag is keyed by
  // a known id so repeated mounts update one tag rather than appending
  // a stack across reruns.
  let style = document.getElementById(STYLE_ID);
  if (!style) {
    style = document.createElement("style");
    style.id = STYLE_ID;
    document.head.appendChild(style);
  }

  // Empty sizePx means "use config.toml default" - clear our override
  // so the page renders at whatever ``baseFontSize`` the bundle was
  // built with.
  if (!sizePx) {
    if (style.textContent) style.textContent = "";
    return;
  }

  // Apply to the html root so rem-based Streamlit CSS scales
  // accordingly. !important wins against the bundled root-size rule.
  const rule = "html { font-size: " + sizePx + " !important; }";
  if (style.textContent !== rule) style.textContent = rule;
}
"""

_COMPONENT = st.components.v2.component(
    "wb_font_size_switcher",
    html=_HTML,
    js=_JS,
)


# Mapping from the user-facing label persisted in ``profile.font_size``
# to the actual pixel value injected into CSS. The empty string is
# treated as "no override - use config.toml's baseFontSize"; ``Small``
# resolves to the same 12px the bundled config.toml ships with, so
# picking it explicitly is visually a no-op but makes the user's choice
# concrete on disk. The order of this dict is the order the segmented
# control renders the options in (smallest to largest). Keep the
# ``Small`` -> 12px row aligned with ``[theme] baseFontSize`` in
# ``.streamlit/config.toml`` (and its mirror in
# ``scripts/build_desktop.py``) so the "no override" path and the
# "Small" pick render identically.
_LABEL_TO_PX: dict[str, str] = {
    "Extra small": "10px",
    "Small": "12px",
    "Medium": "14px",
    "Large": "16px",
    "Extra large": "18px",
}

# Public: the canonical option list for the segmented control on the
# Settings page. Defined here (rather than re-listed in the page module)
# so adding a new size only requires editing one place.
FONT_SIZE_OPTIONS: tuple[str, ...] = tuple(_LABEL_TO_PX.keys())


def label_to_px(label: str) -> str:
    """Map a font-size label to its pixel value, or ``""`` when unset.

    Returns the empty string for any label that isn't in
    :data:`FONT_SIZE_OPTIONS`, which the component interprets as "do
    not apply an override".
    """
    return _LABEL_TO_PX.get(label, "")


def mount_font_size_switcher(font_size: str) -> None:
    """Mount the font size switcher on the current page.

    Should be called once per script run from the top-level entry
    script (so the override is applied on every page in the multi-page
    app).

    Args:
        font_size: One of ``""``, ``"Extra small"``, ``"Small"``,
            ``"Medium"``, ``"Large"``, or ``"Extra large"``. Empty string
            means "use the config.toml default" - the component clears
            any previously applied override so the page renders at
            Streamlit's configured base font size.
    """
    px = _LABEL_TO_PX.get(font_size, "")
    # The key persists the component's state under
    # ``st.session_state["wb_font_size_switcher"]`` so Streamlit
    # short-circuits remounting the component when neither the data
    # payload nor the layout changed. The component itself is
    # idempotent regardless, but keying it keeps the resulting
    # session_state slot stable across reruns.
    _COMPONENT(
        key="wb_font_size_switcher",
        data={"sizePx": px},
    )
