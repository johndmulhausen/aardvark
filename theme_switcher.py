"""In-app Light / Dark / System theme switcher.

Owns one zero-height CCv2 component (declared at module import) plus the
public :func:`mount_theme_switcher` helper that the entry script calls on
every rerun. The component renders nothing visible — its inline JS reads
and writes the same ``localStorage`` key Streamlit's frontend reads at
boot to pick a theme (``stActiveTheme-${window.location.pathname}-v2``,
JSON-stringified ``"System"`` / ``"Light"`` / ``"Dark"``). Applying a new
theme requires a ``window.location.reload()`` because Streamlit only
reads that key once during app boot — there is no React listener that
re-paints on storage changes.

Why a component instead of ``st.html`` with a ``<script>`` tag?
---------------------------------------------------------------
``st.html`` injects the script at the position where it's rendered and
re-runs it on every Streamlit rerun, but it has no way to send anything
back to Python. We need bidirectional communication for the migration
case: a user who set Dark via Streamlit's old built-in toolbar toggle
(before this switcher existed) has ``"Dark"`` in localStorage but ``""``
in their on-disk profile. On first render we want to *adopt* that value,
not blow it away with our default. The component does that by calling
``setStateValue("detected", current)`` whenever localStorage holds a
valid value that differs from the explicit Python preference; Streamlit
fires ``on_detected_change`` once per actual value change so the
``actions.theme_detected`` callback can persist the migrated choice and
re-render the segmented control on the Settings page with the right
selection.

The Settings page's segmented control writes to ``profile.theme`` and to
``st.session_state.theme_pref``; on the next rerun the entry script
re-mounts this component with ``is_explicit=True`` and the new
``theme=...`` value, the JS notices the localStorage mismatch, writes
the new value, and reloads. After the reload the component sees
``current === desired`` and no-ops.
"""
from __future__ import annotations

import streamlit as st

# Inline HTML: the component's own DOM is intentionally empty. The host
# div is set to ``display:none`` so the component takes up no vertical
# space in the layout regardless of where it's mounted (we mount it from
# the top-level entry script, before any page content renders).
_HTML = "<div data-wb-theme-switcher-host style='display:none'></div>"

_JS = r"""
export default function (component) {
  const data = component.data || {};
  const desired = String(data.theme || "System");
  const isExplicit = !!data.isExplicit;
  const valid = ["System", "Light", "Dark"];
  if (!valid.includes(desired)) return;

  // The localStorage key is namespaced by pathname (Streamlit's own
  // convention) and version-suffixed. Keep this in sync with Streamlit's
  // frontend - inspect ``utils.*.js`` in ``streamlit/static/static/js/``
  // for ``ACTIVE_THEME`` if it changes.
  const KEY = "stActiveTheme-" + window.location.pathname + "-v2";

  let current = null;
  try {
    const raw = window.localStorage.getItem(KEY);
    current = raw ? JSON.parse(raw) : null;
  } catch (e) {
    current = null;
  }
  if (!valid.includes(current)) current = null;

  if (!isExplicit) {
    // Migration path: the user has not yet picked via our switcher. If
    // localStorage already has a non-default value (set by Streamlit's
    // legacy toolbar toggle), adopt it server-side so the switcher's UI
    // matches what's actually being rendered. Don't write or reload.
    if (current && current !== desired) {
      component.setStateValue("detected", current);
    }
    return;
  }

  // Explicit branch: apply the Python-side preference.
  if (current === desired) return;
  // Treat empty localStorage as equivalent to "System" to skip a noop
  // reload on first launch when the user's first explicit pick is
  // System. (Streamlit's ``getActiveTheme()`` falls back to System when
  // the key is missing; writing "System" is harmless but a needless
  // reload.)
  if (current === null && desired === "System") return;

  try {
    window.localStorage.setItem(KEY, JSON.stringify(desired));
  } catch (e) {
    return;
  }
  // Streamlit reads the theme key once at app boot - the only way to
  // apply the new value at runtime is to reload the document. The user
  // perceives this as the expected outcome of clicking the switcher.
  window.location.reload();
}
"""

_COMPONENT = st.components.v2.component(
    "wb_theme_switcher",
    html=_HTML,
    js=_JS,
)


def mount_theme_switcher(theme: str, *, on_detected) -> None:
    """Mount the theme switcher on the current page.

    Should be called once per script run from the top-level entry
    script (so it runs on every page in the multi-page app and applies
    the persisted preference regardless of which page the user is on).

    Args:
        theme: The user's persisted Light / Dark / System preference, or
            an empty string when no preference has been chosen yet. An
            empty string puts the component in "migration" mode: it
            reads localStorage and reports any pre-existing value back
            to Python via the ``on_detected`` callback so the app can
            adopt it, but does not write to localStorage and does not
            reload the page.
        on_detected: Callback invoked when the JS-side detects a
            non-empty localStorage value that differs from the current
            Python-side preference. Always pass this even when you
            don't expect to need it; per the CCv2 docs, the result
            attribute only exists on the mount result when the matching
            ``on_<key>_change`` callback is wired.
    """
    is_explicit = theme in ("System", "Light", "Dark")
    desired = theme if is_explicit else "System"
    # The key persists the component's state under
    # ``st.session_state["wb_theme_switcher"]`` so Streamlit only fires
    # ``on_detected_change`` when the JS-reported value actually changes.
    # Without a key, the state is reset every rerun and the callback
    # would fire repeatedly on every render where localStorage holds a
    # non-default value.
    _COMPONENT(
        key="wb_theme_switcher",
        data={"theme": desired, "isExplicit": is_explicit},
        on_detected_change=on_detected,
    )
