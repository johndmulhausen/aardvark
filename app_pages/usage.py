"""Usage page: token + cost dashboards backed by ``~/.wb_coding_agent/usage.jsonl``.

Replaces the chat interface in the main window when the user clicks the
**Usage** entry in ``st.navigation``. Reads the entire usage log on every
render (the file is small — one line per turn) and aggregates via the
helpers in :mod:`usage`.

Layout:

1. Four KPI cards across the top: today's tokens, today's cost, 7-day
   tokens, 7-day cost. Each compares against the prior period so the user
   sees a trend delta.
2. Daily token volume — a stacked-area-ish line chart of prompt vs
   completion tokens over the last 30 days.
3. Daily cost — a line chart in USD over the last 30 days.
4. Cost-by-model — a bar chart of total cost per model for the current
   month.
5. Recent turns — a dataframe of the last 100 turns with timestamp, model,
   prompt/completion tokens, total tokens, cost, latency, mode.

The empty state (no usage log yet) renders an info card pointing the user
back to the Chat page.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import streamlit as st

import usage as usage_log
from models import model_label


def _kpi_delta(current: float, previous: float, *, money: bool = False) -> str | None:
    """Format a KPI delta string. Returns ``None`` when there's no prior period."""
    if previous == 0 and current == 0:
        return None
    if previous == 0:
        return None
    diff = current - previous
    if money:
        return f"{diff:+.4f}" if abs(diff) < 0.01 else f"{diff:+.2f}"
    return f"{int(diff):+d}"


def _build_daily_token_chart_data(
    entries: list[dict[str, Any]],
    days: int = 30,
) -> dict[str, list[Any]]:
    """Build a {date, prompt, completion} dict suitable for ``st.line_chart``.

    Backfills missing days with zeros so the chart x-axis reads continuously
    even when the user skips a day. Returns a dict-of-lists shape that
    Streamlit's chart elements accept directly (no pandas dependency).
    """
    today = datetime.now(timezone.utc).date()
    cutoff = today - timedelta(days=days - 1)
    by_day = {row["date"]: row for row in usage_log.aggregate_by_day(entries)}
    dates: list[str] = []
    prompt: list[int] = []
    completion: list[int] = []
    for i in range(days):
        d = (cutoff + timedelta(days=i)).isoformat()
        dates.append(d)
        row = by_day.get(d, {})
        prompt.append(int(row.get("prompt_tokens") or 0))
        completion.append(int(row.get("completion_tokens") or 0))
    return {"date": dates, "prompt": prompt, "completion": completion}


def _build_daily_cost_chart_data(
    entries: list[dict[str, Any]],
    days: int = 30,
) -> dict[str, list[Any]]:
    today = datetime.now(timezone.utc).date()
    cutoff = today - timedelta(days=days - 1)
    by_day = {row["date"]: row for row in usage_log.aggregate_by_day(entries)}
    dates: list[str] = []
    cost: list[float] = []
    for i in range(days):
        d = (cutoff + timedelta(days=i)).isoformat()
        dates.append(d)
        row = by_day.get(d, {})
        cost.append(float(row.get("cost_usd") or 0.0))
    return {"date": dates, "cost_usd": cost}


def _format_iso_for_table(ts: str) -> str:
    """Render an ISO-8601 timestamp in the user's local time, sortable form."""
    parsed = usage_log._parse_ts(ts)
    if parsed is None:
        return ts
    local = parsed.astimezone()
    return local.strftime("%Y-%m-%d %H:%M:%S")


def render() -> None:
    """Page body for the Usage dashboard (called by ``st.navigation``)."""
    st.title("Usage and cost")
    st.caption(
        "Token usage and cost across every turn you've run, captured from the "
        "live ``usage`` chunks the W&B Inference service emits and priced via "
        "the per-model rates published at "
        "[wandb.ai/site/pricing/inference](https://wandb.ai/site/pricing/inference)."
    )

    entries = usage_log.load_usage()
    if not entries:
        st.info(
            "No usage recorded yet. Head to the **Chat** page and run a turn "
            "to start populating this dashboard.",
            icon=":material/insights:",
        )
        return

    now = datetime.now(timezone.utc)
    today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    yesterday_start = today_start - timedelta(days=1)
    seven_start = today_start - timedelta(days=6)
    fourteen_start = today_start - timedelta(days=13)

    today_entries = [e for e in entries if (ts := usage_log._parse_ts(e.get("ts"))) and ts >= today_start]
    yesterday_entries = [
        e for e in entries
        if (ts := usage_log._parse_ts(e.get("ts"))) and yesterday_start <= ts < today_start
    ]
    seven_entries = [e for e in entries if (ts := usage_log._parse_ts(e.get("ts"))) and ts >= seven_start]
    prior_seven_entries = [
        e for e in entries
        if (ts := usage_log._parse_ts(e.get("ts"))) and fourteen_start <= ts < seven_start
    ]

    today_t = usage_log.totals(today_entries)
    yesterday_t = usage_log.totals(yesterday_entries)
    seven_t = usage_log.totals(seven_entries)
    prior_seven_t = usage_log.totals(prior_seven_entries)

    cols = st.columns(4, border=True)
    with cols[0]:
        st.metric(
            label="Tokens today",
            value=usage_log.format_tokens(today_t["total_tokens"]),
            delta=_kpi_delta(today_t["total_tokens"], yesterday_t["total_tokens"]),
            help="Total prompt + completion tokens billed for turns started today (UTC).",
        )
    with cols[1]:
        st.metric(
            label="Cost today",
            value=usage_log.format_cost(today_t["cost_usd"]),
            delta=_kpi_delta(today_t["cost_usd"], yesterday_t["cost_usd"], money=True),
            help="Estimated USD cost of today's turns at published per-model rates.",
        )
    with cols[2]:
        st.metric(
            label="Tokens (7 days)",
            value=usage_log.format_tokens(seven_t["total_tokens"]),
            delta=_kpi_delta(seven_t["total_tokens"], prior_seven_t["total_tokens"]),
            help="Trailing 7-day total. Delta compares against the prior 7 days.",
        )
    with cols[3]:
        st.metric(
            label="Cost (7 days)",
            value=usage_log.format_cost(seven_t["cost_usd"]),
            delta=_kpi_delta(seven_t["cost_usd"], prior_seven_t["cost_usd"], money=True),
            help="Trailing 7-day cost. Delta compares against the prior 7 days.",
        )

    # ----- Phase 5: Media KPI strip -----
    media_today = [
        e for e in today_entries
        if e.get("model_mode") and e.get("model_mode") != "chat"
    ]
    if media_today:
        # Per-mode aggregations for the strip.
        img_count = sum(int(e.get("unit_count") or 0) for e in media_today if e.get("kind") == "image")
        audio_seconds = sum(int(e.get("unit_count") or 0) for e in media_today if e.get("kind") == "audio")
        video_seconds = sum(int(e.get("unit_count") or 0) for e in media_today if e.get("kind") == "video")
        media_cost = sum(
            float(e["cost_usd"]) for e in media_today
            if isinstance(e.get("cost_usd"), (int, float))
        )
        st.subheader("Media today")
        media_cols = st.columns(4, border=True)
        with media_cols[0]:
            st.metric(
                label="Images",
                value=str(img_count),
                help="Images generated in the current UTC day across all providers.",
            )
        with media_cols[1]:
            st.metric(
                label="Audio (sec)",
                value=str(audio_seconds),
                help="Audio seconds synthesized today.",
            )
        with media_cols[2]:
            st.metric(
                label="Video (sec)",
                value=str(video_seconds),
                help="Video seconds generated today.",
            )
        with media_cols[3]:
            st.metric(
                label="Media cost",
                value=usage_log.format_cost(media_cost),
                help="Total USD spent on media generation today.",
            )

    st.subheader("Tokens per day")
    st.caption("Last 30 days. Stacked: prompt tokens vs completion tokens.")
    daily_tok = _build_daily_token_chart_data(entries, days=30)
    st.line_chart(
        daily_tok,
        x="date",
        y=["prompt", "completion"],
        x_label="Date",
        y_label="Tokens",
        height=240,
    )

    st.subheader("Cost per day")
    st.caption("Last 30 days. USD, computed at the per-model rates above.")
    daily_cost = _build_daily_cost_chart_data(entries, days=30)
    st.line_chart(
        daily_cost,
        x="date",
        y="cost_usd",
        x_label="Date",
        y_label="USD",
        height=240,
    )

    st.subheader("Cost by model")
    st.caption("Total cost per model across every turn in the log, sorted high-to-low.")
    by_model = usage_log.aggregate_by_model(entries)
    if any(r["cost_usd"] > 0 for r in by_model):
        chart_rows = {
            "model": [model_label(r["model"]) for r in by_model],
            "cost_usd": [float(r["cost_usd"]) for r in by_model],
        }
        st.bar_chart(
            chart_rows,
            x="model",
            y="cost_usd",
            x_label="Model",
            y_label="USD",
            height=max(140, 32 * len(by_model)),
            horizontal=True,
        )
    else:
        st.caption(
            "All recorded turns used a model without published pricing — "
            "switch to a priced model in the chat to see cost data here."
        )

    st.subheader("Recent turns")
    recent = list(reversed(entries))[:100]
    table_rows = []
    for e in recent:
        cost = e.get("cost_usd")
        # Phase 5: media-mode rows. ``model_mode`` defaults to "chat"
        # for back-compat (older entries lack the field entirely).
        model_mode = e.get("model_mode") or "chat"
        unit_count = e.get("unit_count")
        unit = e.get("unit") or ""
        table_rows.append({
            "Time": _format_iso_for_table(e.get("ts", "")),
            "Model": model_label(e.get("model", "")),
            "Mode": e.get("mode") or "",
            "Output": (
                f"{unit_count} {unit}".strip()
                if isinstance(unit_count, (int, float)) and model_mode != "chat"
                else f"{int(e.get('total_tokens') or 0)} tokens"
            ),
            "Prompt tokens": int(e.get("prompt_tokens") or 0),
            "Completion tokens": int(e.get("completion_tokens") or 0),
            "Total tokens": int(e.get("total_tokens") or 0),
            "Cost (USD)": cost if isinstance(cost, (int, float)) else None,
            "Latency (s)": (
                round(float(e["duration_seconds"]), 2)
                if isinstance(e.get("duration_seconds"), (int, float))
                else None
            ),
            "Rounds": int(e.get("rounds") or 0),
        })
    st.dataframe(
        table_rows,
        width="stretch",
        hide_index=True,
        column_config={
            "Cost (USD)": st.column_config.NumberColumn(format="$%.4f"),
            "Latency (s)": st.column_config.NumberColumn(format="%.2fs"),
        },
    )


render()
