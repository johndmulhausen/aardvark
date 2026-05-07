"""HTML5 Fullscreen API trigger CCv2 component.

Used by the Phase 5 lightbox modal in ``app_pages/chat.py`` to give
images a true OS-level fullscreen experience comparable to the native
``<video controls>`` fullscreen control. Mounted only when the
lightbox is open and showing an image; ``st.video`` already provides
fullscreen via the browser's native player UI, so video doesn't need
this component.

The component is zero-height (``display:none``) so it takes up no
layout space. Each call to :func:`mount_fullscreen_trigger` bumps a
``request_id`` counter inside the data payload — the inline JS only
fires the fullscreen call when ``request_id`` differs from the last
seen value, which lets a button's ``on_click`` callback "request"
fullscreen by re-mounting with an incremented counter without us
having to fight Streamlit's component-state model.

Mirrors the structure of :mod:`theme_switcher` and
:mod:`font_size_switcher`: zero-height host element, one inline JS
default-export function, persistent state via the component ``key``
so reruns don't reset the last-fired counter.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

_HTML = "<div data-wb-fullscreen-trigger-host style='display:none'></div>"

_JS = r"""
export default function (component) {
  const data = component.data || {};
  const requestId = Number(data.requestId || 0);
  const targetSelector = String(data.targetSelector || "");

  // The component's persistent state holds the last-seen request id
  // so we only fire fullscreen on transitions (avoids re-firing on
  // unrelated reruns where the same target is still mounted).
  const last = component.getStateValue ? component.getStateValue("last_request_id") : null;
  const lastNum = Number(last || 0);
  if (requestId <= lastNum || requestId === 0) {
    return;
  }

  if (!targetSelector) {
    component.setStateValue("last_request_id", requestId);
    return;
  }

  // Look up the target element in the parent document. Streamlit
  // hosts components in iframes, so we have to reach across via
  // ``window.parent.document`` — the lightbox dialog renders into
  // the parent so this works as long as the iframe was mounted from
  // the same origin (which Streamlit guarantees).
  let target = null;
  try {
    target = window.parent.document.querySelector(targetSelector);
  } catch (e) {
    target = null;
  }
  if (!target) {
    component.setStateValue("last_request_id", requestId);
    return;
  }

  // Modern browsers expose ``requestFullscreen`` on every Element;
  // Safari (until very recent versions) and older WebKit variants
  // need the prefixed call. The ``then``/``catch`` chain keeps a
  // rejection (e.g. user denied fullscreen permission) from
  // bubbling into the console as an unhandled promise rejection.
  const fn =
    target.requestFullscreen ||
    target.webkitRequestFullscreen ||
    target.mozRequestFullScreen ||
    target.msRequestFullscreen;
  if (fn) {
    try {
      const result = fn.call(target);
      if (result && typeof result.then === "function") {
        result.catch(() => {});
      }
    } catch (e) {
      // Silently swallow — a user-permission denial isn't an error
      // from our perspective.
    }
  }
  component.setStateValue("last_request_id", requestId);
}
"""

_COMPONENT = st.components.v2.component(
    "wb_fullscreen_trigger",
    html=_HTML,
    js=_JS,
)


def mount_fullscreen_trigger(target_selector: str, *, request_id: int) -> None:
    """Mount the fullscreen trigger.

    Called from inside the lightbox dialog (``app_pages/chat.py``)
    every render. To actually invoke fullscreen, the caller bumps
    ``request_id`` (typically stored in session state) before re-
    mounting; the inline JS fires the Fullscreen API call only on
    a request_id increase, so a render that re-mounts with the same
    counter is a no-op.

    Args:
        target_selector: A CSS selector for the parent-document
            element to enter fullscreen on. Typically the rendered
            ``<img>`` element inside the lightbox modal — the chat
            page wires this up via a stable id attribute it injects
            with ``st.html`` before the image renders.
        request_id: A monotonically-increasing counter the caller
            bumps on each user click of the Fullscreen button. The
            component compares against its persistent state and
            fires only on a strictly-increasing id.
    """
    _COMPONENT(
        key="wb_fullscreen_trigger",
        data={"targetSelector": target_selector, "requestId": int(request_id)},
    )
