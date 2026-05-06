"""Token-usage capture, cost computation, and aggregation.

Single source of truth for everything the **Usage** dashboard needs. Pricing
data is read from :mod:`models` so we don't duplicate the dollar amounts
across modules. The on-disk log lives at
``~/.wb_coding_agent/usage.jsonl`` (one JSON object per line, append-only)
so the dashboard stays usable across app restarts.

Schema of one log entry::

    {
        "ts": "2026-05-06T14:32:11.123456+00:00",  # ISO-8601 UTC, "Z" or +00:00
        "model": "openai/gpt-oss-120b",
        "prompt_tokens": 1842,
        "completion_tokens": 312,
        "total_tokens": 2154,
        "cost_usd": 0.000463,         # nullable when pricing is not yet known
        "input_cost_usd": 0.000276,   # split for per-axis charts
        "output_cost_usd": 0.000187,
        "rounds": 3,                  # number of inference rounds in the turn
        "duration_seconds": 4.812,    # wall-clock latency of the turn
        "mode": "agent"               # or "ask"
    }

This module has no Streamlit imports so non-UI code (the agent loop, future
CLIs) can record turns without dragging in Streamlit.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models import MODEL_METADATA

CONFIG_DIR = Path.home() / ".wb_coding_agent"
USAGE_FILE = CONFIG_DIR / "usage.jsonl"


def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def get_pricing(model: str) -> tuple[float | None, float | None]:
    """Return ``(input_price_per_1m, output_price_per_1m)`` for ``model``.

    Both elements are ``None`` when the model is not in :data:`MODEL_METADATA`
    or when its pricing is missing (experimental previews). Callers should
    treat that case as "cost unknown" and render a placeholder, not zero.
    """
    meta = MODEL_METADATA.get(model)
    if not meta:
        return None, None
    inp = meta.get("input_price_per_1m")
    out = meta.get("output_price_per_1m")
    return (
        float(inp) if isinstance(inp, (int, float)) else None,
        float(out) if isinstance(out, (int, float)) else None,
    )


def compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> tuple[float | None, float | None, float | None]:
    """Compute ``(input_cost, output_cost, total_cost)`` in USD.

    Each component is ``None`` when its corresponding price is unknown so
    the dashboard can distinguish "no pricing data" from "literally $0.00".
    The total is ``None`` only when both per-axis prices are unknown — if
    either is known we still return a partial total so the user sees
    something meaningful instead of a blank cell.
    """
    inp_price, out_price = get_pricing(model)
    input_cost = (
        prompt_tokens * inp_price / 1_000_000 if inp_price is not None else None
    )
    output_cost = (
        completion_tokens * out_price / 1_000_000 if out_price is not None else None
    )
    if input_cost is None and output_cost is None:
        total = None
    else:
        total = (input_cost or 0.0) + (output_cost or 0.0)
    return input_cost, output_cost, total


def record_usage(entry: dict[str, Any]) -> None:
    """Append ``entry`` to :data:`USAGE_FILE` as one JSON line.

    Best-effort: if the file cannot be opened for write the failure is
    swallowed so the chat turn doesn't fail just because the disk is full
    or read-only. The agent loop runs unaffected.
    """
    try:
        _ensure_config_dir()
        with USAGE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")))
            f.write("\n")
    except OSError:
        pass


def build_entry(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int | None = None,
    rounds: int = 1,
    duration_seconds: float | None = None,
    mode: str | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build a fully-populated log entry from raw counts.

    Centralizes the schema so callers don't accidentally drop fields. The
    returned dict is what the dashboard expects to find in
    :data:`USAGE_FILE`; pass it straight to :func:`record_usage`.
    """
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    inp_cost, out_cost, total_cost = compute_cost(model, prompt_tokens, completion_tokens)
    ts = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    return {
        "ts": ts,
        "model": model,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "cost_usd": total_cost,
        "input_cost_usd": inp_cost,
        "output_cost_usd": out_cost,
        "rounds": int(rounds),
        "duration_seconds": (
            float(duration_seconds) if duration_seconds is not None else None
        ),
        "mode": mode or "",
    }


def load_usage(since: datetime | None = None) -> list[dict[str, Any]]:
    """Read every entry from :data:`USAGE_FILE`, optionally filtering by time.

    Lines that fail to parse are silently skipped — the log is append-only
    and we don't want one mangled write to poison the dashboard. ``since``,
    when given, returns only entries with ``ts >= since`` (interpreted as
    UTC if naive).
    """
    if not USAGE_FILE.exists():
        return []
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    out: list[dict[str, Any]] = []
    try:
        with USAGE_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if since is not None:
                    ts = _parse_ts(obj.get("ts"))
                    if ts is None or ts < since:
                        continue
                out.append(obj)
    except OSError:
        return []
    return out


def _parse_ts(raw: Any) -> datetime | None:
    """Best-effort ISO-8601 parser that tolerates trailing ``Z``."""
    if not isinstance(raw, str) or not raw:
        return None
    candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def aggregate_by_day(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group entries by UTC date and return a list of daily totals.

    Output is sorted oldest-first and contains a row for every day that has
    activity. Each row has ``date`` (ISO date string), ``prompt_tokens``,
    ``completion_tokens``, ``total_tokens``, ``cost_usd``, ``turns``.
    """
    by_day: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "turns": 0,
        }
    )
    for entry in entries:
        ts = _parse_ts(entry.get("ts"))
        if ts is None:
            continue
        key = ts.date().isoformat()
        bucket = by_day[key]
        bucket["prompt_tokens"] += int(entry.get("prompt_tokens") or 0)
        bucket["completion_tokens"] += int(entry.get("completion_tokens") or 0)
        bucket["total_tokens"] += int(entry.get("total_tokens") or 0)
        cost = entry.get("cost_usd")
        if isinstance(cost, (int, float)):
            bucket["cost_usd"] += float(cost)
        bucket["turns"] += 1
    return [
        {"date": day, **bucket}
        for day, bucket in sorted(by_day.items())
    ]


def aggregate_by_model(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group entries by model id and return a list of per-model totals.

    Output is sorted by total cost descending so the bar chart reads top
    contributors first. Each row has ``model``, ``prompt_tokens``,
    ``completion_tokens``, ``total_tokens``, ``cost_usd``, ``turns``.
    """
    by_model: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "turns": 0,
        }
    )
    for entry in entries:
        model = str(entry.get("model") or "")
        if not model:
            continue
        bucket = by_model[model]
        bucket["prompt_tokens"] += int(entry.get("prompt_tokens") or 0)
        bucket["completion_tokens"] += int(entry.get("completion_tokens") or 0)
        bucket["total_tokens"] += int(entry.get("total_tokens") or 0)
        cost = entry.get("cost_usd")
        if isinstance(cost, (int, float)):
            bucket["cost_usd"] += float(cost)
        bucket["turns"] += 1
    rows = [{"model": m, **bucket} for m, bucket in by_model.items()]
    rows.sort(key=lambda r: r["cost_usd"], reverse=True)
    return rows


def totals(entries: list[dict[str, Any]]) -> dict[str, float]:
    """Sum tokens / cost / turn count across ``entries``.

    Used to drive the dashboard's KPI cards. Returns zeros for empty input
    so the cards always render even in the empty state.
    """
    out = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "turns": 0,
    }
    for entry in entries:
        out["prompt_tokens"] += int(entry.get("prompt_tokens") or 0)
        out["completion_tokens"] += int(entry.get("completion_tokens") or 0)
        out["total_tokens"] += int(entry.get("total_tokens") or 0)
        cost = entry.get("cost_usd")
        if isinstance(cost, (int, float)):
            out["cost_usd"] += float(cost)
        out["turns"] += 1
    return out


def format_tokens(n: int) -> str:
    """Compact human-readable token count: 1234 -> ``1.2k``, 1_400_000 -> ``1.4M``."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


def format_cost(c: float | None) -> str:
    """Render a USD cost with sensible precision, or ``-`` when unknown."""
    if c is None:
        return "-"
    if c == 0:
        return "$0.00"
    if c < 0.01:
        return f"${c:.4f}"
    if c < 1:
        return f"${c:.3f}"
    return f"${c:,.2f}"
