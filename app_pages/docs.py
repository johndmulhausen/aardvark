"""Docs page: the in-app user guide for the W&B Coding Agent.

Pure-render module — a single :func:`render` called at module scope, just
like the other pages under ``app_pages/``. The page is the user-facing
companion to ``AGENTS.md``: every change to a user-visible surface in
the app should also land here in the same commit. AGENTS.md is for
people who hack on the app; this page is for people who use it.

Voice rules (locked in by ``AGENTS.md``):

- Plain English, roughly an 8th-grade reading level. Short sentences.
  Active voice. Two-syllable words over four-syllable ones when both
  work.
- Define every jargon term inline at first use ("API key (a long
  secret string that proves you're you ...)").
- Lead with the user's task, then the mechanic.
- No developer jargon ("session state", "background thread", "OpenAI
  tool schema", "atomic JSON write", etc.). That stays in AGENTS.md.

The page deliberately does not import :mod:`streamlit_app`,
:mod:`agent`, :mod:`tools`, :mod:`mcp_servers`, :mod:`chats`,
:mod:`account`, :mod:`usage`, :mod:`git_ops`, :mod:`commit_ai`, or
:mod:`wb_client`. The docs are static prose: nothing here reads or
writes the user's filesystem, talks to the inference service, or
mutates session state.
"""
from __future__ import annotations

import streamlit as st


def _render_get_started() -> None:
    """Three-card "Get started in 3 steps" block.

    Reused at the top of the Overview tab and in the chat page's
    welcome card so first-run users see the same path no matter where
    they land.
    """
    cols = st.columns(3, border=True)
    with cols[0]:
        st.markdown(":material/key: **1. Add your W&B API key**")
        st.caption(
            "Open the **Settings** tab in the top nav and paste a key from "
            "[wandb.ai/settings](https://wandb.ai/settings). Tick "
            "**Remember on this machine** to skip this step next time."
        )
    with cols[1]:
        st.markdown(":material/folder_open: **2. Pick a folder**")
        st.caption(
            "Below the chat box, choose a project folder on your computer. "
            "The agent can only read and edit files inside that folder."
        )
    with cols[2]:
        st.markdown(":material/chat: **3. Send your first message**")
        st.caption(
            "Ask the agent to read your code, fix a bug, or write a new "
            "feature. You'll see every step it takes, with green-and-red "
            "diffs for any change."
        )


def _render_overview_tab() -> None:
    """Tab 1 — Overview. One-paragraph pitch, get-started, on-disk files, safety."""
    st.subheader("What is this app?")
    st.markdown(
        "This is a coding assistant powered by W&B Inference (Weights & "
        "Biases' service for running open-source AI models in the cloud). "
        "Point it at a folder on your computer, ask a question, and it "
        "works as an agent (an AI assistant that can run small actions, "
        "not just chat) — reading your code, suggesting changes, running "
        "commands, and showing you exactly what it did."
    )

    st.subheader("Get started in 3 steps")
    _render_get_started()

    st.subheader("What happens when you send a message")
    st.markdown(
        "1. **You type a question** and hit enter.  \n"
        "2. **The AI model picks a tool** (one of the small actions the "
        "agent can take — read a file, run a command, and so on) for "
        "what it needs to do next.  \n"
        "3. **The app runs the tool** in your folder and shows you the "
        "result, including any file changes as a green-and-red diff.  \n"
        "4. **The model writes a reply** with what it found or did. If "
        "it needs another tool, it loops back to step 2 on its own."
    )
    st.caption(
        "The whole back-and-forth is called a **turn**. Some turns are "
        "one round of step 2-3-4; complex tasks can take several rounds "
        "before the model is satisfied with its answer."
    )

    st.subheader("Where things are saved on your computer")
    st.markdown(
        "Everything is kept under a folder called `~/.wb_coding_agent/` "
        "in your home directory. You don't have to touch any of these "
        "by hand — they're just here so you know what's stored:"
    )
    st.markdown(
        "- **Your saved API key and GitHub token** — only when you tick "
        "the Remember box on the Settings page. Saved with permissions "
        "set so only you can read the file.\n"
        "- **Your settings** — theme, font size, GitHub identity, "
        "and so on.\n"
        "- **Your recent folders** — so the working-directory dropdown "
        "remembers projects you've used.\n"
        "- **Your usage log** — one line per agent reply with the token "
        "count and cost. The Usage tab reads from here.\n"
        "- **Your MCP servers** — list of external tool servers you've "
        "added on the Settings page. Saved so only you can read it.\n"
        "- **Your chat history** — one file per conversation, so chats "
        "(including any half-typed message you haven't sent yet) "
        "stick around between launches."
    )

    st.subheader("What's safe and what isn't")
    st.markdown(
        "- The agent can read and change anything inside the folder you "
        "pick. **Pick a folder you're comfortable letting it touch.**\n"
        "- When the agent runs a shell command, it stops itself after "
        "**30 seconds** so a runaway command can't hang.\n"
        "- **Ask only** mode (covered on the Agent page tab) lets the "
        "model look at your files but not change them or run commands. "
        "Use it for code reviews, explanations, or just questions.\n"
        "- The app never sends your files to anywhere except the W&B "
        "Inference service the agent is talking to. Your code does not "
        "leave the connection you set up."
    )


def _render_agent_tab() -> None:
    """Tab 2 — a top-down tour of the Agent page in screen order."""
    st.subheader("A tour of the Agent page")
    st.caption(
        "Walking through the page in the order you see it, from the top "
        "down."
    )

    st.markdown("##### The page title")
    st.markdown(
        "The big heading at the top is the name of the chat (one "
        "back-and-forth conversation with the agent) you're currently "
        "on. New chats start out called **New chat**. Once you send "
        "your first message, the AI gives the chat a short title on its "
        "own so you can find it again later in the sidebar."
    )

    st.markdown("##### The conversation area")
    st.markdown(
        "The scrolling box right under the title is where your messages "
        "and the agent's replies show up, oldest at the top. It scrolls "
        "on its own as new content arrives, so the controls at the "
        "bottom of the page stay in view."
    )
    st.markdown(
        "Every agent reply ends with a small footer line like "
        "`13.2k tokens · $0.014 · 3 rounds · DeepSeek V3.1 · View trace`. "
        "That tells you:"
    )
    st.markdown(
        "- **Tokens** — a token is a chunk of text (usually a word or "
        "part of a word) that the model bills against. The number adds "
        "up your message and the reply.\n"
        "- **Cost** — what that reply cost, in US dollars.\n"
        "- **Rounds** — how many times the agent had to use a tool "
        "before it was done.\n"
        "- **Model** — which AI model produced this reply.\n"
        "- **View trace** — a clickable link that opens this reply's "
        "trace (a recording of one full turn — what you asked, what the "
        "agent did, and how many tokens it used) on W&B Weave (a tool "
        "that records every step the agent takes so you can review it "
        "later)."
    )

    st.markdown("##### The chat input")
    st.markdown(
        "The text box near the bottom of the page is where you type. "
        "It's locked until you've connected to W&B Inference and "
        "picked a folder. Hit Enter to send."
    )
    st.markdown(
        "**Your draft is saved as you type.** If you switch to another "
        "chat, jump to the Settings or Usage tab, or close and reopen "
        "the app, the text you typed (but didn't send) comes back the "
        "next time you open that chat. Each chat keeps its own draft, "
        "so you can have one half-written message in one conversation "
        "and a different one in another without them mixing up."
    )
    st.caption(
        "Tip: type `/` to pop up a list of skills (covered below)."
    )

    st.markdown("##### The Stop button")
    st.markdown(
        "While the agent is working on a reply, a red **Stop** button "
        "appears right above the chat input and the input itself "
        "locks. Click **Stop** any time you want to interrupt the "
        "model — the connection to the AI service drops right away "
        "so you don't get charged for any more tokens."
    )
    st.markdown(
        "Anything the model already wrote stays in the chat (with a "
        "small grey **Stopped by user** caption underneath so you can "
        "see where it cut off), and the chat input unlocks so you "
        "can send a new message right away."
    )
    st.caption(
        "The Stop button is only visible while a reply is in flight. "
        "Once the chat is idle again it disappears on its own."
    )

    st.markdown("##### The actions row")
    st.markdown(
        "Just below the chat input, every chat shows a single strip of "
        "controls for picking a folder and working with git. From left "
        "to right:"
    )
    st.markdown(
        "- **Working directory** dropdown — picks the folder the agent "
        "is allowed to read and edit. Two helper options live at the "
        "top of the list, both prefixed with a small **›** so they "
        "stand apart from real folders below: **Browse for a folder...** "
        "opens a normal OS file picker, and **Start a new project...** "
        "opens a small window that creates a fresh project. Picking "
        "either option runs the action and then sets the dropdown to "
        "the folder you ended up on, so the helper labels never stick "
        "around as the chosen folder.\n"
        "- **Branch picker** — switches between branches (a branch is "
        "an independent line of work in git). Two helper options live "
        "at the top of the list, both prefixed with the same **›** "
        "marker as the working-directory helpers: **New branch...** "
        "opens a small window to create a new branch off the current "
        "one (any unsaved changes come with you onto the new branch), "
        "and **Fetch upstream branches** runs `git fetch` so the "
        "branch list and behind/ahead status reflect the latest remote "
        "state (without changing any of your files). Picking either "
        "option runs the action and then snaps the dropdown back to "
        "your current branch, so the helper labels never stick around "
        "as the chosen branch. The rest of the list shows your local "
        "branches followed by remote-only branches; picking a remote-"
        "only entry checks it out as a new local branch in one click. "
        "If you have unsaved changes, switching branches tries to "
        "bring them with you; git only complains when the target "
        "branch would overwrite a dirty file.\n"
        "- **Changes** — opens a window listing every changed file. "
        "Each file has its own collapsible section with a green-and-"
        "red view of what was added or removed.\n"
        "- **Sync** — the big primary button on the right. Commits "
        "anything dirty (with an AI-drafted message), pulls down "
        "upstream changes if there are any, pushes up your local "
        "commits, and always fetches the latest list of branches "
        "from the remote so the branch dropdown above stays current. "
        "Covered in depth in the **Git** tab."
    )
    st.caption(
        "Until you pick a folder, only the **Working directory** "
        "dropdown shows up — the git controls are hidden because "
        "there's nothing for them to act on yet. Once you pick a "
        "folder, the three git controls appear. From that point on "
        "they stay in place even when they're not usable — they're "
        "just disabled (not hidden) when your folder isn't a git "
        "project or git isn't installed, so the row doesn't shift "
        "around as you change folders."
    )

    st.markdown("##### Picking a folder")
    st.markdown(
        "The dropdown labeled **Working directory** picks the folder "
        "the agent is allowed to read and edit (also called the "
        "**working directory**). You can:"
    )
    st.markdown(
        "- Pick a recent folder from the dropdown.\n"
        "- Paste any path and hit Enter.\n"
        "- Pick **Browse for a folder...** at the top of the list "
        "(the **›** prefix marks it as a helper action rather than a "
        "real folder) to open a normal file picker.\n"
        "- Pick **Start a new project...** right below it (same **›** "
        "marker) to create a brand-new project."
    )
    with st.expander("What does \"Start a new project\" do?"):
        st.markdown(
            "Picking **Start a new project...** at the top of the "
            "Working directory dropdown opens a small window that "
            "helps you set up a fresh project. You can:"
        )
        st.markdown(
            "- **Make an empty folder** with git already set up.\n"
            "- **Clone one of your GitHub projects** — pick from a list "
            "of your repos.\n"
            "- **Create a brand-new project on GitHub** — name it, "
            "choose private or public, and the app makes the folder, "
            "wires it up, and you're ready to go."
        )
        st.caption(
            "The last two options need a GitHub personal access token "
            "(a separate password that lets apps act on your behalf on "
            "GitHub) verified on the Settings page first."
        )

    st.markdown("##### The Project context button")
    st.markdown(
        "If your folder has special files like `AGENTS.md`, "
        "`CLAUDE.md`, `CONVENTIONS.md`, or a `.cursor/rules/` folder, "
        "the app finds them and quietly hands them to the AI as "
        "background context on every message. To the right of the "
        "Model dropdown is a **Project context** button — click it "
        "to open a small window that shows you exactly what was found."
    )
    st.markdown(
        "The window also lists any **skills** (a small file with "
        "extra instructions the AI loads when your message matches "
        "it) the app discovered, both in your folder and globally on "
        "your computer. Close the window with the X, the Esc key, or "
        "the Close button — your chat input and the rest of the page "
        "stay exactly where they were, since the window is an overlay "
        "rather than an inline panel."
    )
    st.caption(
        "When your folder has no detected guidance files or skills, "
        "the Project context button is disabled and its tooltip "
        "explains how to add some (drop an `AGENTS.md` or a "
        "`.cursor/skills/` folder into the working directory)."
    )

    st.markdown("##### Mode")
    st.markdown(
        "The **Mode** dropdown picks how much the AI is allowed to do:"
    )
    st.markdown(
        "- **Agent** — the AI can read files, edit files, write new "
        "files, and run commands. Use this for normal coding work.\n"
        "- **Ask only** — the AI can read your files but cannot change "
        "them or run any commands. Use this for code reviews, "
        "explanations, and questions where you don't want the AI to "
        "touch anything."
    )

    st.markdown("##### Model")
    st.markdown(
        "The **Model** dropdown lists every AI model your W&B account "
        "can use. Each one has different strengths and a different "
        "price per million tokens. The card right under the dropdown "
        "shows you, for the model you've picked:"
    )
    st.markdown(
        "- **Context** — how much text the model can read at once.\n"
        "- **Params** — a rough size number for the model.\n"
        "- **Price** — in US dollars per one million input tokens / "
        "one million output tokens.\n"
        "- **Description** — a short note from the W&B docs about what "
        "the model is good at."
    )
    st.markdown(
        "On some models you'll also see an **orange warning** right "
        "under the card. That means the model is known to *describe* "
        "edits in plain text without actually making them — you'll get "
        "a reply like \"I'll update the file...\" but no diff and no "
        "real change on disk. The warning links to the public bug "
        "report for that model so you can read the source and decide "
        "for yourself. If you hit it, just pick a different model "
        "from the dropdown."
    )


    st.subheader("Attaching files to a message")
    st.markdown(
        "The chat input has a paperclip button on the right, sitting "
        "right next to the send button. Click the paperclip (or drag "
        "files onto the chat input) to attach files to your next "
        "message. Supported types:"
    )
    st.markdown(
        "- **Images** (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`) — "
        "the agent can describe them, transcribe text, compare them, "
        "or use them as references when writing code. Only models "
        "that support images can read them; if you've picked a "
        "text-only model, the agent will tell you it didn't see the "
        "image.\n"
        "- **PDFs** — Anthropic Claude 3.5+ and Google Gemini 1.5+ "
        "read PDFs natively, including tables and figures. For the "
        "other providers, the app extracts text from the PDF before "
        "sending so the agent at least sees the words. A small chip "
        "next to the file shows which path was used (`native PDF` "
        "vs `text-extracted`).\n"
        "- **Plain text and code files** (`.py`, `.md`, `.json`, "
        "`.csv`, etc.) — inlined as a code block in your message so "
        "the agent can read the contents directly."
    )
    st.markdown(
        "Attached files are saved into a folder called "
        "`.wb_artifacts/<chat-id>/inbox/` inside your project. The "
        "first time the app uses that folder, it adds it to your "
        "project's `.gitignore` so you don't accidentally commit "
        "uploads. Files stay on disk until you delete them yourself."
    )

    st.subheader("Media output: image, audio, and video generation")
    st.markdown(
        "Some models can generate images, audio, or video instead of "
        "(or in addition to) text. The agent has three built-in tools "
        "it can call when it makes sense:"
    )
    st.markdown(
        "- **`generate_image`** — text-to-image. Saves PNG files "
        "into `.wb_artifacts/<chat-id>/`. The reply shows the image "
        "right inside the chat with a **View** button (opens a "
        "lightbox modal) and a **Download** button.\n"
        "- **`generate_speech`** — text-to-speech. Saves an MP3 (by "
        "default; WAV / Opus / FLAC also supported) into the same "
        "folder. The chat shows an HTML5 audio player so you can "
        "listen without leaving the page.\n"
        "- **`generate_video`** — text-to-video (Sora, Veo). Long-"
        "running operations show a progress caption while they run. "
        "The reply embeds the video with native browser playback "
        "(including fullscreen via the player's own button)."
    )
    st.markdown(
        "When you click **View** on an image, a lightbox modal opens "
        "with the image at full dialog width and a **Fullscreen** "
        "button that uses your browser's true fullscreen mode. "
        "Video has the same Fullscreen control built into the "
        "player. All generated files live in your project folder, "
        "not on a cloud somewhere — they're yours to keep, move, "
        "or delete."
    )

    st.subheader("Skills and slash commands")
    st.markdown(
        "Skills are bundles of extra instructions for the AI. You can "
        "load one in two ways:"
    )
    st.markdown(
        "- **Type a slash command** (typing `/` followed by a name to "
        "load a specific instruction set). For example, `/refactor` "
        "would load a skill called `refactor`. As you type, a small "
        "menu pops up over the chat input — use the arrow keys to move, "
        "Tab or Enter to pick, Escape to close.\n"
        "- **Just say a keyword** from the skill's description. The app "
        "watches your message and turns on any skill whose keywords "
        "match. Up to 5 skills can load this way at once."
    )
    st.markdown(
        "**How to tell the AI actually used the skill.** When a skill "
        "loads, two things appear in your reply: first, a small caption "
        "above the AI's message that says `Loaded N skill(s)` — that's "
        "the app confirming it sent the skill's instructions to the AI. "
        "Second, the AI's reply should start with a single line like "
        "`Following: /refactor` listing every active skill — that's the "
        "AI itself confirming it read them. Both lines together give "
        "you a strong signal the skill was applied. If you ever see the "
        "first caption but not the second line, the AI may have skimmed "
        "past the skill (some smaller models do this); try a stronger "
        "model from the model picker."
    )
    st.caption(
        "One small detail: when you type something like `/refactor make "
        "this faster`, the AI receives just `make this faster` — the "
        "slash command itself is treated as a label and stripped before "
        "your message is sent. You'll still see your full message "
        "(including the `/refactor` part) in your own chat history; "
        "only the AI's view of it is cleaned up. Any slash that doesn't "
        "match a real skill (like a path `/usr/local/bin`) is left "
        "alone, so you don't have to worry about command-line snippets "
        "or file paths getting corrupted."
    )
    st.caption(
        "Skills can live in two places: inside your project folder "
        "(under `.cursor/skills/` or `.claude/skills/`) so they're "
        "shared with anyone who clones it, or in your home directory "
        "so they're available in every project you open."
    )

    st.subheader("The tools the AI can use")
    st.caption(
        "Each tool is one small action. The agent picks which to run "
        "and you'll see the action and its result inline in the chat."
    )
    with st.expander(":material/folder_open: List files"):
        st.markdown(
            "Shows the agent a tree of files in your folder, like the "
            "output of `tree`. The agent uses this to figure out where "
            "things live before it reads or changes anything."
        )
    with st.expander(":material/description: Read a file"):
        st.markdown(
            "Opens a file in your folder and shows the agent its "
            "contents, with line numbers. The agent can ask for the "
            "whole file or just a slice."
        )
    with st.expander(":material/edit_note: Write a file"):
        st.markdown(
            "Saves a brand-new file or completely overwrites an "
            "existing one. The chat shows you the difference in green "
            "and red so you can see exactly what changed."
        )
    with st.expander(":material/edit: Edit a file"):
        st.markdown(
            "Finds a unique piece of text in a file and replaces it "
            "with something else. Same green-and-red diff. Useful for "
            "small targeted edits without rewriting the whole file."
        )
    with st.expander(":material/terminal: Run a shell command"):
        st.markdown(
            "Runs a command in your folder, like `npm test` or "
            "`pytest`. The chat shows you the exit code and anything "
            "the command printed to the terminal. The command stops "
            "itself after 30 seconds. **Only available in Agent mode.**"
        )
    with st.expander(":material/extension: External tools (MCP)"):
        st.markdown(
            "If you've connected any MCP servers in Settings, their "
            "tools show up here too. **MCP** stands for **Model "
            "Context Protocol** — an open standard that lets external "
            "programs hand the agent extra tools. For example, you "
            "could connect an MCP server that knows how to search "
            "Slack, query a database, or control a web browser."
        )
        st.caption(
            "MCP tools are only available in Agent mode."
        )


def _render_chats_tab() -> None:
    """Tab 3 — the multi-chat sidebar."""
    st.subheader("The chat history sidebar")
    st.markdown(
        "On the left side of every page, you'll see a list of every "
        "conversation you've had with the agent. The app remembers "
        "chats between launches — closing and reopening the app does "
        "not lose your history."
    )

    st.markdown("##### The \"New chat\" button")
    st.markdown(
        "At the top of the sidebar. Click it to start a fresh "
        "conversation. If you already have an empty chat that you "
        "never used, clicking the button brings you back to it "
        "instead of stacking up empty rows. If you click it from a "
        "non-Agent tab (Settings, Usage, or Docs), the app jumps you "
        "back to the **Agent** tab so you can start typing right "
        "away."
    )

    st.markdown("##### Each row in the list")
    st.markdown(
        "Every row has, from left to right: a small icon for the "
        "chat's state, the chat's title, an **archive** button, and a "
        "**delete** button. The chat you're currently looking at is "
        "shown with an outlined box around it and can't be clicked "
        "(since you're already on it)."
    )

    st.markdown("##### What the icons mean")
    st.markdown(
        "- :material/chat_bubble_outline: **Empty bubble** — brand-new "
        "chat, no messages yet.\n"
        "- :material/progress_activity: **Spinner** — the agent is "
        "working on a reply right now. Open this chat and use the "
        "**Stop** button above the chat input if you want to "
        "interrupt it without spending more tokens.\n"
        "- :material/check_circle: **Check mark** — the chat is done "
        "with its last reply and waiting for your next message.\n"
        "- :material/cancel: **Red X** — something went wrong with the "
        "last reply. Hover the row to see the error message."
    )

    st.markdown("##### Archive vs delete")
    st.markdown(
        "- **Archive** hides the chat in the **Archive** section near "
        "the bottom of the sidebar. The chat is still there if you "
        "want it back — click the unarchive icon to bring it out.\n"
        "- **Delete** removes the chat for good. The app asks you to "
        "confirm first. Once it's deleted, it's gone."
    )
    st.caption(
        "If you delete a chat while it's still working on a reply (the "
        "spinner icon is showing), the app warns you and the confirm "
        "button changes to **Stop and delete**. Clicking it stops the "
        "AI right away so you don't get charged for any more tokens, "
        "then removes the chat."
    )

    st.subheader("How chats get titled")
    st.markdown(
        "After you send your first message, the AI picks a short title "
        "for the chat on its own (about five words) and that title "
        "stays for the rest of the conversation."
    )

    st.subheader("Switching between chats")
    st.markdown(
        "Click any row in the sidebar to load that chat. If you're "
        "on a tab other than **Agent** (say, **Settings** or "
        "**Usage**), the app jumps you back to the Agent tab so you "
        "can see the chat's history right away. Each chat remembers "
        "its own folder, mode, and model — so you can have one chat "
        "working in your frontend project on a fast model and another "
        "chat working in a backend folder on a smarter model. Picking "
        "a new model in one chat does not change the others."
    )
    st.caption(
        "Each chat also remembers any half-typed message in its chat "
        "input. Switch chats, jump tabs, or close the app — when you "
        "come back to that chat the draft is still there. Sending the "
        "message clears it."
    )

    st.subheader("Two chats at the same time")
    st.markdown(
        "You can ask one chat a question, switch to a different chat "
        "while the first is still thinking, and ask something there "
        "too. Both replies run in the background. When you click back "
        "to the first chat, you'll see however much of its reply has "
        "arrived so far."
    )

    st.subheader("If the app crashes mid-reply")
    st.markdown(
        "If the app shuts down or crashes while a reply was being "
        "written, the next time you open it that chat is marked with "
        "the red X icon and a clear note saying the reply was "
        "interrupted. You can keep going in the same chat (just send a "
        "new message) or start a fresh one."
    )


def _render_git_tab() -> None:
    """Tab 4 — the git workflow on the chat page."""
    st.subheader("Git, in plain words")
    st.markdown(
        "**Git** is a tool that tracks every change to your code. The "
        "app uses it so you can save your work, share it, and undo "
        "mistakes. **If you've never used git, you don't have to "
        "learn the commands** — the buttons in the app cover what you "
        "need."
    )

    st.subheader("The actions row at the bottom of the page")
    st.markdown(
        "Every chat shows a single strip of file and git controls just "
        "below the chat input. Until you pick a folder, only the "
        "**Working directory** dropdown shows up — the git controls "
        "appear once a folder is picked. From that point on the git "
        "cells stay in place even when they're not usable; they're "
        "just disabled (not hidden) when your folder isn't a git "
        "project, so the row layout stays stable as you change "
        "folders:"
    )

    st.markdown("##### The branch picker")
    st.markdown(
        "A **branch** is a separate copy of your work where you can "
        "try things out. Two helper options sit at the top of the "
        "dropdown, both prefixed with a small **›** so they read as "
        "helper actions rather than as real branches:"
    )
    st.markdown(
        "- **New branch...** — opens a small window that asks for a "
        "name, creates a new branch based on whatever you have right "
        "now (any unsaved changes come along), and switches you to it.\n"
        "- **Fetch upstream branches** — pulls down the latest list of "
        "branches and changes from the remote (a copy of the project "
        "hosted somewhere else, like GitHub), **without** changing any "
        "of your files. Use it when you want to see what's been added "
        "on the remote since you last looked. After it runs, the "
        "dropdown above refreshes so any new remote branches show up "
        "(and any deleted on the remote disappear)."
    )
    st.markdown(
        "Below those two helpers, the dropdown shows the branches that "
        "exist on your computer first, then any branches that only "
        "exist on the remote. Pick a branch and the app switches to it."
    )
    st.caption(
        "Picking a remote-only branch automatically makes a matching "
        "local branch on your computer first, then switches to it. "
        "Switching with unsaved changes brings them along when there's "
        "no conflict; if a conflict would happen, git refuses with a "
        "clear message and you stay on the current branch. Picking a "
        "helper option (**New branch...** or **Fetch upstream "
        "branches**) runs the action and then snaps the dropdown back "
        "to your current branch, so the helper labels never stick "
        "around as the chosen branch."
    )

    st.markdown("##### The Sync button")
    st.markdown(
        "The **Sync** button is the big one. It's a **two-way sync** "
        "with the remote — it brings down anyone else's changes *and* "
        "sends up your work, in one click. Here's exactly what it "
        "does:"
    )
    st.markdown(
        "1. **If you have unsaved changes**, it asks an AI model to "
        "write a short commit message describing them (a **commit** "
        "is one saved snapshot of your code, with a message that "
        "says what changed), then stages every changed file and "
        "saves the commit.\n"
        "2. **Always fetches** the latest list of branches and "
        "commits from the remote, and refreshes the branch dropdown "
        "above so any new remote branches show up (and any branches "
        "deleted on the remote disappear).\n"
        "3. **If the remote has new commits you don't have**, pulls "
        "them down and replays your local commits on top — this is "
        "called a **rebase** (replay your local commits on top of "
        "the latest remote changes).\n"
        "4. **If you have local commits the remote doesn't have**, "
        "pushes them up.\n"
        "5. **If neither side has new commits**, it just confirms "
        "you're already in sync — but the fetch in step 2 still "
        "ran, so the branch list is up to date."
    )
    st.markdown(
        "The model that writes the commit message is **whichever "
        "one you've picked for your chat** (in the model picker "
        "below the chat input). So if your chat is using Claude, "
        "Claude writes the commit message; if it's using Gemini, "
        "Gemini does; and so on. The same goes for the pull-request "
        "title and description on the **Push and open a pull request** "
        "path below."
    )
    st.markdown(
        "Sync stays enabled whenever you're inside a git project on "
        "a real branch (not a half-finished merge or rebase), even "
        "when there's nothing local to commit and nothing on the "
        "remote to pull — clicking it then is the easiest way to "
        "ask \"what new branches has anyone pushed since I last "
        "looked?\" without leaving the app. A pull-only or "
        "fetch-only sync doesn't need a model at all, so it works "
        "even if you haven't picked one yet."
    )
    st.caption(
        "You'll see toast notifications as each step finishes."
    )

    st.subheader("First push: publishing a branch")
    st.markdown(
        "When you sync a branch that has never been pushed before, "
        "you'll see a small window asking you to pick:"
    )
    st.markdown(
        "- **Just push the branch** — uploads the branch to the "
        "remote. Nothing else happens.\n"
        "- **Push and open a pull request** — uploads the branch, "
        "then opens a page in your web browser with a draft pull "
        "request (a proposal to merge your branch into the main one, "
        "reviewed on GitHub) already filled in. The model you've "
        "picked for your chat writes the title and description for "
        "you. You just review and click submit."
    )
    st.caption(
        "Pull requests are supported on GitHub, GitLab, and Bitbucket."
    )

    st.subheader("Merge conflicts (when two changes collide)")
    st.markdown(
        "A **merge conflict** is when two people changed the same "
        "lines in a file, and git can't tell which version to keep. "
        "It shows up most often during the rebase step of a Sync."
    )
    st.markdown(
        "When this happens, you'll see a yellow warning in the sidebar "
        "listing the files with conflicts, and a "
        "**:material/auto_fix_high: Resolve with DeepSeek** button."
    )
    st.markdown(
        "Click that button and the AI takes over: it reads each "
        "conflicted file, picks a sensible merge for each one, saves "
        "them, and finishes the rebase for you. Every step shows up "
        "in the chat just like a normal reply, so you can see exactly "
        "what it decided."
    )

    st.subheader("The Changes window")
    st.markdown(
        "The **Changes** button at the right of the actions row opens "
        "a modal listing every file that has changed in your folder "
        "since the last commit. Each file has its own collapsible "
        "section with a green-and-red diff. Small badges flag files "
        "that are:"
    )
    st.markdown(
        "- **Untracked** — brand-new files git hasn't seen yet.\n"
        "- **Deleted** — files you removed.\n"
        "- **Renamed** — files you moved.\n"
        "- **Staged only** — already marked for the next commit.\n"
        "- **Unstaged only** — changed but not yet marked."
    )

    st.markdown("##### Throwing changes away")
    st.markdown(
        "If the agent (or you) made a change you don't want to keep, "
        "the Changes window has two ways to throw it away — both work "
        "before you commit, and neither asks the agent to do anything:"
    )
    st.markdown(
        "- **Discard** (next to each file) — undoes just that one "
        "file's changes. The file goes back to whatever was last "
        "committed. Brand-new files (untracked) get deleted from your "
        "folder.\n"
        "- **Discard all** (top-right of the window) — does the same "
        "for every file in the list, in one click."
    )
    st.markdown(
        "Clicking either button opens a small popover that asks you "
        "to confirm — there's no in-app undo, so the popover is your "
        "last chance to back out. Once you confirm, the changes are "
        "gone for good. (If you want to keep the changes but stop "
        "showing them in the diff, commit them or stash them with "
        "regular git instead.)"
    )

    st.subheader("How commits get your name on them")
    st.markdown(
        "When you sign in with GitHub on the Settings page, the app "
        "uses your GitHub name and email on every commit it makes for "
        "you. So when someone looks at the commit on GitHub, they'll "
        "see it was authored by you — not a generic robot account."
    )

    st.subheader("Cloning and creating GitHub projects")
    st.markdown(
        "Picking **Start a new project...** at the top of the "
        "Working directory dropdown can clone one of your GitHub "
        "repos or create a brand-new one for you. Both options need "
        "a GitHub personal access token verified on the Settings "
        "page first."
    )
    st.caption(
        "Your token is sent only when needed — it never gets saved "
        "into the cloned project's settings, so the project is safe "
        "to share."
    )


def _render_settings_tab() -> None:
    """Tab 5 — a tour of the Settings page."""
    st.subheader("A tour of the Settings page")

    st.markdown("##### The page header")
    st.markdown(
        "When you've signed in with GitHub, you'll see your avatar, "
        "username, and email at the top. Otherwise it just says "
        "**Settings**."
    )

    st.subheader("GitHub identity")
    st.markdown(
        "This is where you connect your GitHub account, so commits "
        "the agent makes are signed with your name."
    )
    st.markdown(
        "You'll need a **personal access token** (PAT) — a separate "
        "password that lets apps act on your behalf on GitHub. To set "
        "one up:"
    )
    st.markdown(
        "1. Click the **Generate a PAT on GitHub** button. It opens "
        "GitHub's token page in your browser.\n"
        "2. Pick a fine-grained token. Recommended permissions: "
        "**`read:user`** and **`user:email`**. Add **`repo`** if you "
        "want the agent to push code to GitHub repos you own.\n"
        "3. Copy the token GitHub gives you (it starts with `ghp_` or "
        "`github_pat_`).\n"
        "4. Paste it back into the **Personal access token** field "
        "here.\n"
        "5. Click **Verify and save**. The app checks the token works "
        "and shows your avatar."
    )
    st.caption(
        "To unlink your GitHub account later, click **Sign out of "
        "GitHub** on this card."
    )

    st.subheader("Appearance")
    st.markdown(
        "The Appearance card has two segmented controls:"
    )
    st.markdown(
        "- **Theme** — pick **System** (matches your computer's "
        "light/dark setting), **Light**, or **Dark**. The page "
        "reloads once when you switch so the new colors apply "
        "everywhere.\n"
        "- **Font size** — pick from **Extra small**, **Small**, "
        "**Medium**, **Large**, or **Extra large**. Useful if the "
        "default is too small to read or too big on your screen. "
        "Changes apply right away."
    )
    st.caption(
        "Both choices are saved on this computer so they stay between "
        "launches."
    )

    st.subheader("Providers")
    st.markdown(
        "The Settings page lists 7 AI providers, all visible above "
        "the fold. You only need to add a key for the providers you "
        "actually want to use — adding even one is enough to start "
        "chatting."
    )
    st.markdown(
        "Each provider card has the same pieces: an **API key** field "
        "(paste your key from that provider's website), a **Get a key** "
        "button that opens the provider's key page in your browser, a "
        "**Remember on this machine** checkbox so the app can re-use "
        "the key next time, and **Connect / Disconnect / Forget** "
        "buttons."
    )
    st.markdown("##### The 7 providers")
    st.markdown(
        "- **W&B Inference** — Weights & Biases' open-model service, "
        "and the source of **traces** (a recorded timeline of every "
        "step the agent takes). Tracing only works when W&B is "
        "connected, even if you use a different provider for the "
        "actual chat. So most users want a W&B key whatever else "
        "they pick. Get one at "
        "[wandb.ai/settings](https://wandb.ai/settings). The optional "
        "**Project** field tags your usage to a specific W&B team.\n"
        "- **OpenAI** — GPT-4o, o-series reasoning models, "
        "GPT-Image-1, native text-to-speech, and Sora video. When "
        "you pick an o-series model (o1, o3, o3-mini, o3-pro, ...) "
        "the chat page shows a **Reasoning effort** control under "
        "the model card so you can dial chain-of-thought up or down.\n"
        "- **Anthropic** — the Claude family. **Prompt caching** "
        "(automatic, no toggle needed) cuts the cost of every turn "
        "after the first by about 90% — a big deal for a coding "
        "agent that re-sends a long system prompt every turn.\n"
        "- **Google Gemini** — Gemini 3.1 Pro / 3 Flash / 2.5 Pro / "
        "Flash, Imagen for images, Veo for video. The chat page "
        "shows a **Google Search grounding** toggle under the model "
        "card when a Gemini model is selected (lets the model hit "
        "real-time web at query time; adds per-search costs above "
        "Google's free tier).\n"
        "- **Mistral** — Mistral's own models (Mistral Large 3, "
        "Codestral for code, Magistral Medium / Small for "
        "reasoning).\n"
        "- **xAI** — the Grok family (Grok 4, Grok 4 Fast). The "
        "chat page shows a **Live Search** toggle under the model "
        "card when a Grok model is selected (lets Grok hit "
        "real-time web at query time; adds per-search costs above "
        "the per-token rate).\n"
        "- **OpenRouter** — labeled **marked-up gateway**. OpenRouter "
        "is one key that reaches hundreds of models from many other "
        "providers, but they add 5–10% on top of the native price. "
        "Pick this when you want broad model coverage without "
        "managing a separate key per model lab."
    )
    st.caption(
        "Pricing in this app is always **direct from the provider** "
        "— we never mark up tokens. The only exception is OpenRouter, "
        "which adds the markup itself; we label it clearly so you "
        "know what you're paying for."
    )

    st.subheader("W&B Inference")
    st.markdown(
        "This is where you connect to W&B Inference (the cloud "
        "service that runs the AI models). The card has four pieces:"
    )
    st.markdown(
        "- **API key** — your **API key** is a long secret string "
        "that proves you're you when the app talks to W&B. Paste it "
        "here. Get one from "
        "[wandb.ai/settings](https://wandb.ai/settings). The key is "
        "kept only in memory unless you tick the box below.\n"
        "- **Project (optional)** — fill in `team/project` if you "
        "want your usage and traces to show up under a specific W&B "
        "team's project page. Leave it blank to use a default project "
        "called `wandb-coding-agent` under your account.\n"
        "- **Remember on this machine** — when ticked, your API key "
        "is saved on this computer in a file only you can read. Untick "
        "it if you'd rather paste the key fresh every session.\n"
        "- **Connect / Disconnect / Forget saved API key** — Connect "
        "logs you in and downloads the list of available models. It "
        "also auto-adds the **W&B Official** entry to your MCP "
        "servers list (using the same key as the bearer token), so "
        "the agent can query your runs, Weave traces, and ask the "
        "W&B SupportBot. Disconnect signs you out for this session "
        "and disables that MCP entry. **Forget saved API key** "
        "erases the saved key from your computer and removes the "
        "MCP entry too."
    )
    st.caption(
        "The next time you open the app, it tries to connect on its "
        "own using your saved key. If that fails (an expired key, no "
        "internet, etc.), you'll see a message and a Reconnect button "
        "on the Agent page."
    )

    st.subheader("What \"Tracing turns to W&B Weave\" means")
    st.markdown(
        "Once you connect, you may see a small caption like "
        "*Tracing turns to W&B Weave at `team/project`*. **Weave** is "
        "a tool that records every step the agent takes so you can "
        "review it later in your web browser. It's automatic and free "
        "with your W&B account."
    )
    st.markdown(
        "Click the project name in that caption (or the **View "
        "trace** link in any agent reply) to open the trace in your "
        "browser. You'll see the full timeline of the reply — every "
        "tool the agent used, how long each step took, and how many "
        "tokens it spent."
    )

    st.subheader("MCP servers")
    st.markdown(
        "**MCP** is the **Model Context Protocol** — an open "
        "standard that lets external programs hand the agent extra "
        "tools. An MCP server is just a program that exposes some "
        "tools through that standard. For example:"
    )
    st.markdown(
        "- A server that knows how to search Slack.\n"
        "- A server that controls a web browser.\n"
        "- A server that lets the agent query a database.\n"
        "- A server that exposes your company's internal API."
    )
    st.markdown(
        "The MCP servers card lists every server you've added and "
        "lets you turn each on or off."
    )

    st.markdown("##### The W&B Official server (added for you)")
    st.markdown(
        "When you connect **W&B Inference**, the app automatically "
        "adds an entry called **W&B Official** to your MCP servers "
        "list. It uses the same API key as the bearer token, so you "
        "don't need to copy-paste anything else. It's marked with a "
        "small **auto-configured** badge."
    )
    st.markdown(
        "This server gives the agent tools for working with your "
        "W&B account directly — looking up runs, querying Weave "
        "traces, creating reports, asking the W&B SupportBot for "
        "documentation help. If you'd rather not have those tools "
        "available, untick its **Enabled** box; the W&B Inference "
        "connection itself stays live. **Disconnect** on the W&B "
        "Inference card disables this entry; **Forget key** removes "
        "it entirely."
    )

    st.markdown("##### Adding a server")
    st.markdown(
        "Click **Add server**. The window asks for a name and a "
        "transport type. Two transport types in plain words:"
    )
    st.markdown(
        "- **Stdio (local subprocess)** — pick this for tools you "
        "run on your own computer. You give it a command to run, like "
        "`npx`, plus any arguments and environment variables.\n"
        "- **HTTP (remote)** — pick this for an MCP service that "
        "lives on the internet. You give it a URL and any "
        "authentication headers it needs (like `Authorization: Bearer "
        "...`)."
    )
    st.caption(
        "Auth headers you add are saved on this computer in a file "
        "only you can read. They're stored as plain text, so only "
        "enter credentials you're comfortable having on disk."
    )

    st.markdown("##### A note on modes")
    st.markdown(
        "MCP tools are only available in **Agent** mode. In **Ask "
        "only** mode they're hidden, since Ask mode is for read-only "
        "code review."
    )


def _render_usage_tab() -> None:
    """Tab 6 — the Usage dashboard."""
    st.subheader("Token usage and cost")
    st.markdown(
        "The **Usage** tab shows you how much you've used the AI and "
        "how much it's cost. Open it any time you want to check in."
    )

    st.subheader("The four cards at the top")
    st.markdown(
        "- **Tokens today** — how many tokens (chunks of text the "
        "model bills against) you've used since midnight, your "
        "computer's time. The little arrow underneath compares it to "
        "yesterday.\n"
        "- **Cost today** — your spend so far today, in US dollars.\n"
        "- **Tokens (7 days)** — total over the last seven days. The "
        "arrow compares it to the seven days before that.\n"
        "- **Cost (7 days)** — same idea, in dollars."
    )

    st.subheader("Tokens per day chart")
    st.markdown(
        "A 30-day line chart with two lines:"
    )
    st.markdown(
        "- **Prompt tokens** — text you sent to the model.\n"
        "- **Completion tokens** — text the model sent back."
    )

    st.subheader("Cost per day chart")
    st.markdown(
        "A 30-day line chart in US dollars. Add up all the agent "
        "replies for each day and you get this line."
    )

    st.subheader("Cost by model chart")
    st.markdown(
        "A bar chart showing how much each AI model has cost you, "
        "sorted from most to least. Useful for spotting which model "
        "is your most expensive habit."
    )

    st.subheader("Recent turns table")
    st.markdown(
        "The last 100 agent replies, with:"
    )
    st.markdown(
        "- **Time** — when you sent the message.\n"
        "- **Model** — which AI model produced the reply.\n"
        "- **Mode** — Agent or Ask only.\n"
        "- **Prompt / Completion / Total tokens** — how much text was "
        "involved.\n"
        "- **Cost (USD)** — the price of that reply. Models without "
        "a published price show **—** instead of a number.\n"
        "- **Latency (s)** — how long the reply took, in seconds.\n"
        "- **Rounds** — how many tool-calling rounds the agent did "
        "before it was done."
    )

    st.subheader("Where do these numbers come from?")
    st.markdown(
        "Every time the AI replies, the W&B Inference service tells "
        "the app exactly how much text it processed (in tokens). The "
        "app saves that for you, locally on this computer. Prices "
        "come from the rates W&B publishes at "
        "[wandb.ai/site/pricing/inference](https://wandb.ai/site/pricing/inference)."
    )

    st.subheader("Where do my traces live?")
    st.markdown(
        "Click the **View trace** link in any agent reply on the "
        "Agent page, or click the **W&B Weave** project name in the "
        "Settings page caption. Both open the W&B Weave page in your "
        "web browser, where you can see every step of every reply you've ever run."
    )


def render() -> None:
    """The Docs page body. One title, one caption, six tabs."""
    st.title("Docs")
    st.caption(
        "Plain-English answers to \"what does this button do?\" and "
        "\"how do I get started?\". Open the tab that matches what "
        "you're looking at on screen."
    )

    overview, agent_tab, chats_tab, git_tab, settings_tab, usage_tab = st.tabs(
        [
            ":material/lightbulb: Overview",
            ":material/auto_awesome: Agent page",
            ":material/forum: Chats",
            ":material/commit: Git",
            ":material/settings: Settings",
            ":material/insights: Usage",
        ]
    )
    with overview:
        _render_overview_tab()
    with agent_tab:
        _render_agent_tab()
    with chats_tab:
        _render_chats_tab()
    with git_tab:
        _render_git_tab()
    with settings_tab:
        _render_settings_tab()
    with usage_tab:
        _render_usage_tab()


render()
