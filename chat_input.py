"""Chat-input enhancer: slash-command autocomplete + draft persistence.

Owns a single CCv2 component (``mount_slash_autocomplete``) that latches
onto the page's existing ``st.chat_input`` ``<textarea>`` and adds two
otherwise-unavailable behaviors:

1. **Slash-command autocomplete** — typing ``/`` pops a floating
   keyboard-navigable palette of skill suggestions over the textarea.
   The dropdown is rendered into ``document.body`` (not into the
   component's shadow root) so it can overlay the chat input from
   anywhere on the page without z-index battles.
2. **Draft persistence per chat** — every keystroke is debounced
   (~200ms) and forwarded to Python via ``setStateValue("draft", {...})``
   so the active chat's ``draft_text`` field is saved to disk on the
   next Streamlit rerun (which any natural interaction triggers — chat
   switch, tab navigation, button click, etc.). The component also
   *restores* the saved draft into the textarea whenever the active
   chat changes, so navigating Chat → Settings → Chat (or clicking a
   different chat in the sidebar) brings the user's in-progress text
   back instead of silently dropping it.

Why a side-by-side enhancer instead of a chat-input replacement?
----------------------------------------------------------------
``st.chat_input`` ships with sticky-to-bottom layout, theme-aware styling,
and built-in submit + file/audio plumbing. Re-implementing those would
quadruple the size of this module for no user-visible gain. Instead the
enhancer keeps ``st.chat_input`` exactly as it is and adds the two
concerns above through a small, scoped piece of JS that:

1. Locates the chat input's ``<textarea>`` via ``document.querySelector``
   (the component runs in shadow DOM but shares the page's ``document``).
2. Attaches a single set of listeners per textarea, idempotent across
   Streamlit reruns thanks to a marker property on the element.
3. Maintains per-tab singleton state under
   ``window.__wbSlashAutocomplete`` so reruns / multi-mount sequences
   don't pile up duplicate listeners or duplicate dropdown elements.

Draft persistence contract:
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each ``setStateValue("draft", payload)`` call sends a payload of shape
``{"chat_id": <str>, "text": <str>}`` rather than a bare string. Callers
on the Python side resolve the chat by ``chat_id`` (not by the active
chat) so a typed-then-immediately-switched flow saves the draft to the
*chat the user was typing in*, even if ``ss.active_chat_id`` has
already moved on by the time the ``on_draft_change`` callback fires.

Restoration is gated on chat-id changes: the JS singleton tracks the
last-applied ``chat_id``, and only force-rewrites the textarea value
when the next render passes a different ``chat_id`` (or on the very
first attach). This avoids overwriting the user's in-progress typing
on every Streamlit rerun.

The skill list is delivered every run via ``data={"skills": [...]}`` so
edits to ``AGENTS.md`` or new ``.cursor/skills/`` / ``.claude/skills/``
directories show up on the next rerun without a page refresh.
"""
from __future__ import annotations

import json
from collections.abc import Callable

import streamlit as st

# Inline HTML: the component's own DOM is intentionally empty. We render
# a zero-height container so the component takes up no vertical space in
# the app layout. All visible UI (the dropdown) is appended to
# ``document.body`` from JS so it can overlay the chat input.
_HTML = "<div data-wb-slash-autocomplete-host style='display:none'></div>"

# CSS injected into ``document.head`` by the JS. Lives outside the
# component's shadow root so it can style the dropdown elements that we
# also add to ``document.body``. Uses ``--st-*`` theme variables so the
# dropdown automatically matches the user's Streamlit theme (light, dark,
# or custom).
_DROPDOWN_CSS = """
#wb-slash-autocomplete {
  position: fixed;
  z-index: 999999;
  display: none;
  flex-direction: column;
  background: var(--st-secondary-background-color, #f0f2f6);
  color: var(--st-text-color, #31333f);
  border: 1px solid var(--st-border-color, rgba(49,51,63,0.2));
  border-radius: var(--st-base-radius, 0.5rem);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18), 0 2px 6px rgba(0, 0, 0, 0.08);
  font-family: var(--st-font, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif);
  font-size: 0.875rem;
  overflow: hidden;
  max-height: 320px;
}
#wb-slash-autocomplete.is-open { display: flex; }
#wb-slash-autocomplete .wb-header {
  padding: 0.5rem 0.75rem;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--st-text-color, #31333f);
  opacity: 0.6;
  border-bottom: 1px solid var(--st-border-color-light, rgba(49,51,63,0.1));
  background: var(--st-background-color, #ffffff);
}
#wb-slash-autocomplete .wb-list {
  overflow-y: auto;
  max-height: 280px;
}
#wb-slash-autocomplete .wb-empty {
  padding: 0.75rem;
  opacity: 0.6;
  font-style: italic;
}
#wb-slash-autocomplete .wb-item {
  padding: 0.5rem 0.75rem;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
  border-left: 2px solid transparent;
}
#wb-slash-autocomplete .wb-item:hover,
#wb-slash-autocomplete .wb-item.is-active {
  background: var(--st-background-color, #ffffff);
  border-left-color: var(--st-primary-color, #0080ff);
}
#wb-slash-autocomplete .wb-row {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  flex-wrap: wrap;
}
#wb-slash-autocomplete .wb-slug {
  font-family: var(--st-code-font, "Source Code Pro", monospace);
  color: var(--st-primary-color, #0080ff);
  font-weight: 600;
  font-size: 0.85rem;
}
#wb-slash-autocomplete .wb-scope {
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 0.05rem 0.35rem;
  border-radius: var(--st-base-radius, 0.5rem);
  background: var(--st-blue-background-color, rgba(28,131,225,0.12));
  color: var(--st-blue-text-color, #1c83e1);
}
#wb-slash-autocomplete .wb-scope.is-user {
  background: var(--st-gray-background-color, rgba(120,120,128,0.12));
  color: var(--st-gray-text-color, #525252);
}
#wb-slash-autocomplete .wb-desc {
  font-size: 0.75rem;
  opacity: 0.75;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
#wb-slash-autocomplete .wb-footer {
  padding: 0.35rem 0.75rem;
  font-size: 0.65rem;
  opacity: 0.55;
  border-top: 1px solid var(--st-border-color-light, rgba(49,51,63,0.1));
  background: var(--st-background-color, #ffffff);
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}
#wb-slash-autocomplete .wb-footer kbd {
  font-family: var(--st-code-font, "Source Code Pro", monospace);
  background: var(--st-secondary-background-color, #f0f2f6);
  border: 1px solid var(--st-border-color-light, rgba(49,51,63,0.1));
  border-radius: 0.25rem;
  padding: 0 0.25rem;
  font-size: 0.65rem;
}
"""

# The component JS. Runs as an ES module on every script run; the global
# state held under ``window.__wbSlashAutocomplete`` keeps it idempotent so
# that mounting the component twenty times in a row doesn't pile up
# duplicate listeners or duplicate dropdown elements. The textarea hookup
# is also self-healing: we re-run ``attach`` on every script run, so if
# the previous textarea was removed from the DOM (e.g. the user
# disconnected and the chat input was unmounted), the next mount finds
# the new one.
#
# Draft persistence:
# ------------------
# On every render, ``data.chatId`` + ``data.draft`` come down from
# Python. We track the last-applied chat id in ``state.appliedChatId``;
# when the next render's chat id differs (or on the very first attach),
# we force-rewrite the textarea value to ``data.draft`` so navigating
# away and coming back, or switching to a different chat in the
# sidebar, restores the user's in-progress text. On every keystroke
# we debounce ~200ms and then call ``setStateValue("draft", payload)``
# where ``payload = {chat_id, text}`` — sending the chat id alongside
# the text lets the Python ``on_draft_change`` callback save to the
# right chat even if ``ss.active_chat_id`` has already moved on by the
# time the rerun fires the callback (e.g. user typed in chat A, then
# clicked chat B in the sidebar before the debounce fired). Before
# applying a draft from a different chat we flush any pending debounce
# for the previous chat so fast switches don't lose the last few
# characters of typing.
_JS = r"""
export default function (component) {
  const data = component.data || {};
  const skills = Array.isArray(data.skills) ? data.skills : [];
  const placeholderHint = typeof data.placeholderHint === "string" ? data.placeholderHint : "";
  const chatId = typeof data.chatId === "string" ? data.chatId : "";
  const draft = typeof data.draft === "string" ? data.draft : "";

  const win = window;
  const doc = document;
  // Streamlit injects --st-* theme variables into the CCv2 component's
  // shadow root, NOT onto document.body. The dropdown lives outside the
  // shadow root, so we have to copy those variables onto the dropdown
  // every time the theme might have changed (i.e. every script run).
  // ``themeHost`` is the element to read computed CSS variables from.
  const themeHost =
    (component.parentElement && component.parentElement.host) ||
    component.parentElement ||
    doc.body;

  if (!win.__wbSlashAutocomplete) {
    win.__wbSlashAutocomplete = {
      skills: [],
      textarea: null,
      dropdown: null,
      stylesInjected: false,
      activeIndex: 0,
      matches: [],
      queryRange: null,
      onInput: null,
      onKeyDown: null,
      onBlur: null,
      onScroll: null,
      onResize: null,
      cleanupHooked: false,
      // --- Draft persistence state ---------------------------------
      // Latest values pushed down from Python (refreshed every render
      // by the assignments at the top of the function below).
      expectedChatId: "",
      expectedDraft: "",
      // The chat id whose draft we most recently wrote into the
      // textarea. Used to detect chat switches: when expectedChatId
      // diverges from this we restore the new chat's draft.
      appliedChatId: "",
      // True once we've applied a draft at least once. Distinguishes
      // "first attach" (apply even if expected matches applied) from
      // "subsequent renders within the same chat" (do nothing).
      draftEverApplied: false,
      // The last value we sent up via setStateValue OR pushed down
      // from Python. Used to skip redundant setStateValue calls and
      // to skip overwriting the textarea when nothing changed.
      lastSentDraft: "",
      // Pending debounce timer id, or null. Cleared on flush / fresh
      // typing / chat switch.
      draftDebounce: null,
      // The setStateValue function from the latest render. Stored
      // because input listeners attached on first attach fire on
      // every keystroke later, when the original render's `component`
      // closure is stale.
      setStateValueFn: null,
    };
  }
  const state = win.__wbSlashAutocomplete;
  state.skills = skills;
  state.placeholderHint = placeholderHint;
  state.expectedChatId = chatId;
  state.expectedDraft = draft;
  // Refresh the bound setStateValue every render so it's always the
  // latest one (each render gets a fresh ``component`` object).
  state.setStateValueFn = component.setStateValue || state.setStateValueFn;

  function injectStyles() {
    if (state.stylesInjected) return;
    if (doc.querySelector("#wb-slash-autocomplete-styles")) {
      state.stylesInjected = true;
      return;
    }
    const style = doc.createElement("style");
    style.id = "wb-slash-autocomplete-styles";
    style.textContent = __DROPDOWN_CSS__;
    doc.head.appendChild(style);
    state.stylesInjected = true;
  }

  function findThemeRoot() {
    // Streamlit injects --st-* CSS variables onto specific app-shell
    // elements, not onto document.body. Append the dropdown under one of
    // those so the theme variables (and therefore dark mode) propagate.
    // Falls back to document.body so the dropdown still mounts in unusual
    // environments where these testids change.
    const candidates = [
      '[data-testid="stAppViewContainer"]',
      '[data-testid="stApp"]',
      ".stApp",
      "section.main",
    ];
    for (const sel of candidates) {
      const el = doc.querySelector(sel);
      if (el) return el;
    }
    return doc.body;
  }

  function ensureDropdown() {
    const root = findThemeRoot();
    if (!state.dropdown) {
      const dd = doc.createElement("div");
      dd.id = "wb-slash-autocomplete";
      dd.setAttribute("role", "listbox");
      dd.addEventListener("mousedown", (e) => {
        // Stop the click from blurring the textarea before our handler runs.
        e.preventDefault();
      });
      state.dropdown = dd;
    }
    if (!root.contains(state.dropdown)) {
      if (state.dropdown.parentElement) {
        state.dropdown.parentElement.removeChild(state.dropdown);
      }
      root.appendChild(state.dropdown);
    }
    copyThemeVarsToDropdown();
    return state.dropdown;
  }

  function copyThemeVarsToDropdown() {
    // Forward every --st-* variable from the component's shadow root onto
    // the dropdown element so the dropdown's CSS rules (which reference
    // var(--st-...) values) resolve to the active Streamlit theme. This
    // is what gives us automatic dark-mode + custom-theme support without
    // an explicit theme prop.
    if (!state.dropdown || !themeHost) return;
    let cs;
    try {
      cs = getComputedStyle(themeHost);
    } catch (e) {
      return;
    }
    for (let i = 0; i < cs.length; i++) {
      const name = cs[i];
      if (name.indexOf("--st-") === 0) {
        const value = cs.getPropertyValue(name);
        if (value) state.dropdown.style.setProperty(name, value);
      }
    }
  }

  function findChatTextarea() {
    // Try the modern testid first, then a couple of fallbacks. Streamlit's
    // chat input has historically used different testids and attribute
    // shapes; keep the selector list defensive so we keep working when it
    // changes again.
    const selectors = [
      '[data-testid="stChatInput"] textarea',
      '[data-testid="stChatInputTextArea"]',
      'textarea[data-testid="stChatInputTextArea"]',
    ];
    for (const sel of selectors) {
      const el = doc.querySelector(sel);
      if (el) return el;
    }
    // Last-resort: look for a textarea whose closest ancestor has a
    // testid that contains "ChatInput".
    const all = doc.querySelectorAll("textarea");
    for (const ta of all) {
      const anc = ta.closest('[data-testid*="ChatInput"]');
      if (anc) return ta;
    }
    return null;
  }

  function getQueryRange(textarea) {
    // Walk backwards from the cursor until we hit a "/" that's at the
    // start of the message or preceded by whitespace. If we hit
    // whitespace before finding a "/", there's no active query.
    const value = textarea.value || "";
    const cursor = textarea.selectionStart == null ? value.length : textarea.selectionStart;
    let i = cursor - 1;
    while (i >= 0) {
      const ch = value.charAt(i);
      if (ch === "/") {
        if (i === 0 || /\s/.test(value.charAt(i - 1))) {
          const query = value.slice(i + 1, cursor);
          // Reject embedded whitespace: "/foo bar" stops at the space.
          if (/\s/.test(query)) return null;
          return { start: i, end: cursor, query };
        }
        return null;
      }
      if (/\s/.test(ch)) return null;
      i -= 1;
    }
    return null;
  }

  function scoreSkill(skill, query) {
    if (!query) return 1; // empty query: show all skills, original order
    const q = query.toLowerCase();
    const slug = (skill.slug || "").toLowerCase();
    if (slug === q) return 100;
    if (slug.startsWith(q)) return 80;
    if (slug.includes(q)) return 60;
    const triggers = Array.isArray(skill.triggers) ? skill.triggers : [];
    for (const t of triggers) {
      const tl = String(t).toLowerCase();
      if (tl === q) return 50;
      if (tl.startsWith(q)) return 40;
      if (tl.includes(q)) return 20;
    }
    const desc = (skill.description || "").toLowerCase();
    if (desc.includes(q)) return 5;
    return 0;
  }

  function computeMatches(query) {
    const scored = (state.skills || [])
      .map((s) => ({ skill: s, score: scoreSkill(s, query) }))
      .filter((m) => m.score > 0);
    scored.sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return (a.skill.slug || "").localeCompare(b.skill.slug || "");
    });
    return scored.slice(0, 8).map((m) => m.skill);
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[c]));
  }

  function renderDropdown() {
    const dd = ensureDropdown();
    dd.innerHTML = "";
    const header = doc.createElement("div");
    header.className = "wb-header";
    header.textContent = state.matches.length
      ? "Skills - press Enter or Tab to insert"
      : "No matching skills";
    dd.appendChild(header);

    if (!state.matches.length) {
      const empty = doc.createElement("div");
      empty.className = "wb-empty";
      empty.textContent = state.placeholderHint || "Try a different prefix.";
      dd.appendChild(empty);
    } else {
      const list = doc.createElement("div");
      list.className = "wb-list";
      state.matches.forEach((skill, idx) => {
        const item = doc.createElement("div");
        item.className = "wb-item" + (idx === state.activeIndex ? " is-active" : "");
        item.setAttribute("role", "option");
        item.setAttribute("data-idx", String(idx));
        const scopeClass = skill.scope === "user" ? " is-user" : "";
        item.innerHTML =
          '<div class="wb-row">' +
          '<span class="wb-slug">/' + escapeHtml(skill.slug) + '</span>' +
          '<span class="wb-scope' + scopeClass + '">' + escapeHtml(skill.scope || "workspace") + '</span>' +
          '</div>' +
          (skill.description
            ? '<div class="wb-desc">' + escapeHtml(skill.description) + '</div>'
            : "");
        item.addEventListener("mousedown", (e) => {
          e.preventDefault();
          state.activeIndex = idx;
          acceptSelection();
        });
        list.appendChild(item);
      });
      dd.appendChild(list);
    }

    const footer = doc.createElement("div");
    footer.className = "wb-footer";
    footer.innerHTML =
      '<span><kbd>Up</kbd> <kbd>Down</kbd> navigate</span>' +
      '<span><kbd>Tab</kbd> / <kbd>Enter</kbd> insert</span>' +
      '<span><kbd>Esc</kbd> close</span>';
    dd.appendChild(footer);
  }

  function positionDropdown() {
    const dd = state.dropdown;
    const ta = state.textarea;
    if (!dd || !ta) return;
    const rect = ta.getBoundingClientRect();
    const width = Math.min(rect.width, 520);
    dd.style.width = width + "px";
    dd.style.left = rect.left + "px";
    dd.style.bottom = (win.innerHeight - rect.top + 8) + "px";
  }

  function showDropdown() {
    const dd = ensureDropdown();
    dd.classList.add("is-open");
    dd.style.display = "flex";
    positionDropdown();
  }

  function hideDropdown() {
    if (!state.dropdown) return;
    state.dropdown.classList.remove("is-open");
    state.dropdown.style.display = "none";
    state.queryRange = null;
    state.matches = [];
  }

  function isOpen() {
    return !!state.dropdown && state.dropdown.classList.contains("is-open");
  }

  function setTextareaValue(textarea, newValue) {
    // React's controlled inputs cache the previous value; bypass the cache
    // by going through the native HTMLTextAreaElement value setter, then
    // dispatch an input event so React picks up the change.
    const proto = win.HTMLTextAreaElement.prototype;
    const nativeSetter = Object.getOwnPropertyDescriptor(proto, "value").set;
    nativeSetter.call(textarea, newValue);
    // Mark the next 'input' event as programmatic so our own listener
    // doesn't echo the value back to Python via setStateValue (and so
    // the autocomplete doesn't hide/reshow).
    state.suppressNextInput = true;
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function sendDraft(text) {
    // Push the textarea contents up to Python under the chat id we
    // most recently restored. Skips the websocket round-trip when
    // nothing changed since the last send, and refuses to send when
    // we have no chat id to attribute the typing to (e.g. the chat
    // page hasn't finished its first render yet).
    if (!state.appliedChatId) return;
    if (text === state.lastSentDraft) return;
    state.lastSentDraft = text;
    if (state.setStateValueFn) {
      state.setStateValueFn("draft", { chat_id: state.appliedChatId, text: text });
    }
  }

  function flushPendingDraft() {
    // Cancel the in-flight debounce and immediately push the textarea
    // contents up. Used when the chat is about to switch under us so
    // the previous chat's last few keystrokes still get attributed to
    // it instead of silently dropped.
    if (state.draftDebounce) {
      clearTimeout(state.draftDebounce);
      state.draftDebounce = null;
    }
    const ta = state.textarea;
    if (!ta) return;
    sendDraft(ta.value || "");
  }

  function applyDraftIfNeeded() {
    // Force the textarea value to match the draft Python sent down,
    // but only when (a) we've never applied a draft yet (first attach)
    // or (b) the chat id changed since we last applied. Otherwise the
    // user is mid-typing inside the same chat and the textarea is the
    // source of truth — overwriting it would jump the cursor.
    const ta = state.textarea;
    if (!ta) return;
    const expectedChatId = state.expectedChatId || "";
    const expectedDraft = state.expectedDraft || "";
    const chatChanged = expectedChatId !== state.appliedChatId;
    if (!chatChanged && state.draftEverApplied) {
      return;
    }
    // Flush any pending typing for the previous chat before we
    // overwrite the textarea with the new chat's draft. ``sendDraft``
    // is a no-op when nothing changed, so this is cheap when there's
    // no pending typing.
    if (state.draftEverApplied && state.appliedChatId && state.appliedChatId !== expectedChatId) {
      flushPendingDraft();
    }
    if (ta.value !== expectedDraft) {
      setTextareaValue(ta, expectedDraft);
    }
    state.appliedChatId = expectedChatId;
    state.draftEverApplied = true;
    state.lastSentDraft = expectedDraft;
  }

  function acceptSelection() {
    const ta = state.textarea;
    const range = state.queryRange;
    const skill = state.matches[state.activeIndex];
    if (!ta || !range || !skill) {
      hideDropdown();
      return;
    }
    const value = ta.value || "";
    const before = value.slice(0, range.start);
    const after = value.slice(range.end);
    const insert = "/" + skill.slug;
    const needSpace = after.length === 0 || after.charAt(0) !== " ";
    const tail = needSpace ? " " : "";
    const newValue = before + insert + tail + after;
    setTextareaValue(ta, newValue);
    const cursor = (before + insert + tail).length;
    ta.setSelectionRange(cursor, cursor);
    ta.focus();
    hideDropdown();
  }

  function refreshMatches() {
    const ta = state.textarea;
    if (!ta) {
      hideDropdown();
      return;
    }
    // No skills configured at all -> don't ever show the dropdown.
    // Mirrors the historical "only mount the component when there are
    // skills" behavior; we now mount unconditionally (so draft
    // persistence still works in skill-less projects), but the
    // autocomplete still degrades to a no-op visually.
    if (!state.skills || state.skills.length === 0) {
      hideDropdown();
      return;
    }
    const range = getQueryRange(ta);
    if (!range) {
      hideDropdown();
      return;
    }
    state.queryRange = range;
    state.matches = computeMatches(range.query);
    state.activeIndex = 0;
    if (!state.matches.length && range.query.length === 0) {
      // Bare "/" with no skills configured at all - still hide.
      hideDropdown();
      return;
    }
    showDropdown();
    renderDropdown();
  }

  function detach() {
    const ta = state.textarea;
    if (!ta) return;
    if (state.onInput) ta.removeEventListener("input", state.onInput);
    if (state.onKeyDown) ta.removeEventListener("keydown", state.onKeyDown, true);
    if (state.onBlur) ta.removeEventListener("blur", state.onBlur);
    delete ta.__wbSlashAutocompleteAttached;
    state.textarea = null;
    state.onInput = state.onKeyDown = state.onBlur = null;
    if (state.draftDebounce) {
      clearTimeout(state.draftDebounce);
      state.draftDebounce = null;
    }
    // Clear the "applied" tracking so the next attach (which is a
    // fresh DOM element) re-applies the current draft. Without this,
    // a Chat -> Settings -> Chat round-trip would leave the new
    // textarea empty even though we still have a draft on file.
    state.draftEverApplied = false;
    state.appliedChatId = "";
    state.lastSentDraft = "";
  }

  function attach(textarea) {
    if (textarea.__wbSlashAutocompleteAttached) {
      state.textarea = textarea;
      return;
    }
    textarea.__wbSlashAutocompleteAttached = true;
    state.textarea = textarea;

    state.onInput = () => {
      // Programmatic textarea updates (slash insert, draft restore,
      // submit clear) dispatch an 'input' event we don't want to
      // round-trip back to Python. ``setTextareaValue`` flips the
      // suppress flag for exactly one event.
      if (state.suppressNextInput) {
        state.suppressNextInput = false;
        // Slash autocomplete still needs to recompute against the
        // new value (a slash insert leaves the textarea in a state
        // where the dropdown should hide).
        refreshMatches();
        return;
      }
      refreshMatches();
      // Debounced draft save. Each keystroke restarts the timer; we
      // only push to Python after the user pauses typing for ~200ms.
      // The debounce keeps websocket traffic tiny without leaving the
      // draft persistently stale — any Streamlit rerun (chat switch,
      // tab navigation, button click) flushes the latest text via
      // applyDraftIfNeeded -> flushPendingDraft below.
      if (state.draftDebounce) clearTimeout(state.draftDebounce);
      state.draftDebounce = setTimeout(() => {
        state.draftDebounce = null;
        const ta = state.textarea;
        if (!ta) return;
        sendDraft(ta.value || "");
      }, 200);
    };
    state.onKeyDown = (e) => {
      if (!isOpen()) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        state.activeIndex = (state.activeIndex + 1) % Math.max(1, state.matches.length);
        renderDropdown();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        state.activeIndex = (state.activeIndex - 1 + Math.max(1, state.matches.length)) % Math.max(1, state.matches.length);
        renderDropdown();
      } else if (e.key === "Enter" || e.key === "Tab") {
        if (state.matches.length === 0) {
          hideDropdown();
          return;
        }
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        acceptSelection();
      } else if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        hideDropdown();
      }
    };
    state.onBlur = () => {
      // Defer so a click on a dropdown item still has a chance to fire
      // before we tear it down.
      setTimeout(() => {
        if (doc.activeElement !== textarea) hideDropdown();
      }, 150);
    };

    textarea.addEventListener("input", state.onInput);
    // Capture phase so we run before React's delegated listeners and can
    // intercept Enter without triggering chat-input submission.
    textarea.addEventListener("keydown", state.onKeyDown, true);
    textarea.addEventListener("blur", state.onBlur);
  }

  function tryAttachLoop(remaining) {
    const ta = findChatTextarea();
    if (ta) {
      // If we previously had a different textarea reference, drop it.
      if (state.textarea && state.textarea !== ta) detach();
      attach(ta);
      // Restore the active chat's draft into the (possibly fresh)
      // textarea. ``applyDraftIfNeeded`` is safe to call repeatedly
      // — it only writes when the chat id changed (or on first
      // attach), so re-renders within the same chat don't disturb
      // the user's in-progress typing.
      applyDraftIfNeeded();
      return;
    }
    if (remaining > 0) setTimeout(() => tryAttachLoop(remaining - 1), 150);
  }

  // Self-healing reattach: if the cached textarea was removed from the DOM
  // (Streamlit unmounted the chat input), drop the reference and look
  // again from scratch.
  if (state.textarea && !doc.body.contains(state.textarea)) detach();
  tryAttachLoop(20);
  // The textarea was already attached on a previous render; we still
  // need to re-check the draft on every render in case the chat id
  // changed since last time (e.g. user clicked a different chat in the
  // sidebar — the textarea is the same DOM node, but ``data.chatId``
  // / ``data.draft`` are now for a different chat).
  if (state.textarea) applyDraftIfNeeded();

  injectStyles();

  if (!state.cleanupHooked) {
    win.addEventListener("scroll", () => { if (isOpen()) positionDropdown(); }, true);
    win.addEventListener("resize", () => { if (isOpen()) positionDropdown(); });
    state.cleanupHooked = true;
  }

  // Don't tear down on rerun - the user may still be typing, and the next
  // mount immediately reattaches anyway. Real cleanup happens implicitly
  // when the textarea is removed from the DOM (handled above).
  return () => {};
}
""".replace("__DROPDOWN_CSS__", json.dumps(_DROPDOWN_CSS))


_COMPONENT = st.components.v2.component(
    "wb_slash_autocomplete",
    html=_HTML,
    js=_JS,
)

# Python-side session-state key under which Streamlit stores the
# component instance's state. Exposed as a module constant so the chat
# page's ``on_draft_change`` callback can read the latest draft payload
# (``st.session_state[STATE_KEY].draft`` -> ``{"chat_id": str,
# "text": str}``) without hard-coding the string in two places.
STATE_KEY = "wb_slash_autocomplete"


def mount_slash_autocomplete(
    skills: list[dict[str, object]],
    *,
    placeholder_hint: str = "",
    chat_id: str = "",
    draft: str = "",
    on_draft_change: Callable[[], None] | None = None,
) -> None:
    """Mount the chat-input enhancer on the page's ``st.chat_input``.

    Should be rendered on every script run (the JS is idempotent across
    reruns, but Streamlit needs to re-mount the component each time so it
    receives the latest ``skills`` list, ``chat_id``, and ``draft``).

    Two responsibilities (see this module's docstring for the full
    contract):

    1. Slash-command autocomplete keyed off ``skills``.
    2. Per-chat draft persistence keyed off ``chat_id`` + ``draft``: the
       JS restores ``draft`` into the textarea on chat switch / first
       attach, and pushes typing back up via
       ``setStateValue("draft", {chat_id, text})`` after a ~200ms
       debounce. The Python callback ``on_draft_change`` (when wired)
       fires on the next rerun to persist the typing to disk.

    Args:
        skills: Skill dicts produced by :func:`project_context.summary`,
            each with at least ``slug``, ``description``, ``scope``, and
            optionally ``triggers``. The dropdown filters this list as the
            user types after a ``/``.
        placeholder_hint: Short text shown in the dropdown body when the
            current query has no matches. Defaults to a generic hint.
        chat_id: Identifier of the chat the textarea is currently
            displaying. The JS uses this to detect chat switches (so a
            sidebar click restores the new chat's draft) and to tag
            outgoing draft-save events so they're attributed to the
            right chat even when ``ss.active_chat_id`` has already
            moved on by the time the rerun fires.
        draft: The chat's currently-saved draft text — the JS restores
            this into the textarea on chat switch / first attach.
            Empty string means "leave the textarea empty" (used after
            a submit when the chat's ``draft_text`` was cleared).
        on_draft_change: Optional callback fired by Streamlit on the
            next rerun whenever the JS pushed up a new draft via
            ``setStateValue``. Read the payload from the component
            instance's session-state entry (keyed
            ``"wb_slash_autocomplete"``) and persist it.
    """
    payload_skills: list[dict[str, object]] = []
    for s in skills or []:
        payload_skills.append(
            {
                "slug": str(s.get("slug", "")),
                "description": str(s.get("description", "") or ""),
                "scope": str(s.get("scope", "workspace") or "workspace"),
                "triggers": list(s.get("triggers", []) or [])[:24],
            }
        )
    kwargs: dict[str, object] = {
        # Stable Python-side key so the on-change callback only fires
        # when the JS-reported value actually changes, and so callers
        # can look up the latest payload via
        # ``st.session_state[STATE_KEY]["draft"]``. Without a key the
        # component instance state resets every rerun and the callback
        # would refuse to fire (or fire spuriously, depending on
        # Streamlit version).
        "key": STATE_KEY,
        "data": {
            "skills": payload_skills,
            "placeholderHint": placeholder_hint,
            "chatId": chat_id or "",
            "draft": draft or "",
        },
    }
    # Wiring ``on_draft_change`` is what guarantees Streamlit fires the
    # callback when the JS-side ``setStateValue("draft", ...)`` lands a
    # new payload — without it the value still arrives in session
    # state, but the callback wouldn't run and the disk write would
    # be deferred until some other code path noticed the new value.
    if on_draft_change is not None:
        kwargs["on_draft_change"] = on_draft_change
    _COMPONENT(**kwargs)
