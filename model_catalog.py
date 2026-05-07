"""Model catalog: live model lists + enriched pricing / descriptions / tags.

The single source of truth for everything the model picker shows. Owns:

- Per-provider ``/v1/models`` adapters (the only place in the codebase
  that hits a provider's listing endpoint).
- The LiteLLM pricing-registry fetch + bundled offline snapshot.
- The OpenRouter ``/api/v1/models`` description fetch.
- The merge logic that enforces the **strict completeness gate**: a
  model only enters :func:`available_qualified_ids` if it has a real
  description, provider-accurate pricing, and a known context window.
  Models missing any of those are tracked in
  :func:`hidden_models_summary` so the picker can surface a count
  caption — but they are never instantiated as :class:`ModelInfo` and
  never appear in the UI.

Hard rules (mirrored in ``AGENTS.md``):

- **No Streamlit imports** — runs from background threads safely.
- **Pricing for ``<provider>:<raw>`` MUST come from a source that priced
  that exact provider's offering** (curated > LiteLLM-with-
  ``litellm_provider``-assertion > OpenRouter for ``openrouter:*`` only
  > the provider's own ``/v1/models`` if it carries pricing). No cross-
  provider substitution.
- **No auto-generated description floors** — if no real description is
  available, the model is hidden, not faked.
- **Disk cache files
  (``~/.wb_coding_agent/{model_catalog,litellm_registry,openrouter_catalog}.json``)
  are written ONLY by this module.**

Source precedence per field
---------------------------
- **Pricing**: curated ``MODEL_METADATA`` entry with both
  ``input_price_per_1m`` and ``output_price_per_1m`` set →
  LiteLLM entry (``litellm_provider`` field asserted to match the
  qualified provider) → for ``openrouter:*`` ids only, OpenRouter's
  ``pricing.prompt`` / ``pricing.completion`` → provider's own API
  response if it carries pricing (Together does, OpenRouter does).
- **Description**: curated ``MODEL_METADATA`` → OpenRouter
  ``description`` → provider's API description (Google's
  ``description`` field, Anthropic's ``display_name``).
- **Context**: curated → LiteLLM's ``max_input_tokens`` →
  provider's API response if it carries one.

Auto-derived tags
-----------------
Applied to entries that survive the gate when curated ``tags`` is
empty (the curated tags ``coding`` / ``reasoning`` / ``frontier``
stay curated-only):

- ``long_context`` when ``context >= 200_000``.
- ``cheap`` when ``output_price_per_1m <= 0.5``.
- ``fast`` when the provider is Groq or Cerebras (latency-tier hardware).
- ``multimodal`` when LiteLLM's ``supports_vision`` flag is true.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from providers import PROVIDERS, Provider, get_provider


# ---------------------------------------------------------------------------
# On-disk cache layout
# ---------------------------------------------------------------------------
CONFIG_DIR = Path.home() / ".wb_coding_agent"
LIVE_CATALOG_FILE = CONFIG_DIR / "model_catalog.json"
LITELLM_CACHE_FILE = CONFIG_DIR / "litellm_registry.json"
OPENROUTER_CACHE_FILE = CONFIG_DIR / "openrouter_catalog.json"

# Repo-bundled offline-floor snapshot. Used when we have no fresh disk
# cache and the runtime LiteLLM fetch fails (offline first-launch). The
# refresh procedure for this file is documented in
# ``data/litellm_model_registry.json``'s top-level comment field.
_REPO_DIR = Path(__file__).resolve().parent
BUNDLED_LITELLM_SNAPSHOT = _REPO_DIR / "data" / "litellm_model_registry.json"

# How long a per-source disk cache entry is considered fresh before the
# next ``refresh()`` call refetches. 24h matches the "models change
# frequently but not constantly" rhythm of the LLM ecosystem; users can
# also click the modal Refresh button to force a refetch outside this
# window.
CATALOG_TTL_SECONDS = 24 * 3600

# Upstream URLs.
LITELLM_REGISTRY_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
OPENROUTER_CATALOG_URL = "https://openrouter.ai/api/v1/models"


# Provider-id → LiteLLM-prefix mapping. Mirrored from
# ``providers.PROVIDERS[*].litellm_prefix`` so this module never
# reaches back into ``providers``-mutation territory; the values must
# stay in lockstep with the ``Provider.litellm_prefix`` field.
PROVIDER_LITELLM_PREFIX: dict[str, str] = {
    pid: provider.litellm_prefix for pid, provider in PROVIDERS.items()
}


# Curated allowlist of qualified ids known to support PDF input. Used
# by the multimodal translator (Phase 6) to decide whether to send
# PDFs as native ``document`` blocks (Anthropic) / ``Part.from_data``
# (Google) versus running ``pypdf`` text extraction first. Limited to
# the explicitly-known cases — bringing a model in here means we've
# verified the provider's API accepts PDF input for that model.
PDF_SUPPORT_ALLOWLIST: frozenset[str] = frozenset({
    "anthropic:claude-3-5-sonnet-20241022",
    "anthropic:claude-3-5-haiku-20241022",
    "anthropic:claude-3-opus-20240229",
    "gemini:gemini-2.5-pro",
    "gemini:gemini-2.5-flash",
    "gemini:gemini-2.0-flash",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
PricingSource = Literal["curated", "litellm", "openrouter", "live"]
DescriptionSource = Literal["curated", "openrouter", "live"]
ModelMode = Literal[
    "chat",
    "image_generation",
    "audio_speech",        # TTS (text → audio)
    "audio_transcription",  # STT (audio → text)
    "video_generation",
    "embedding",
]


@dataclass(frozen=True)
class ModelInfo:
    """One enriched model entry surfaced in the picker.

    Every ``ModelInfo`` represents a model that has cleared the strict
    completeness gate **for its mode**: chat models guarantee
    ``input_price_per_1m`` + ``output_price_per_1m`` + a real
    description + a known context window; media-mode models guarantee
    the mode-specific pricing field + a description (context is
    optional for media). Fields that were optional in the upstream
    source dict are normalized away before construction so consumers
    don't have to defensively check ``Optional[float]`` everywhere.

    Mode-specific pricing slots:

    - ``image_pricing``: dict keyed by ``"<size>/<quality>"`` (e.g.
      ``"1024x1024/standard"``) → USD per image. Populated only for
      ``mode == "image_generation"`` models.
    - ``tts_pricing_per_1m_chars``: USD per 1M input characters.
      Populated only for ``mode == "audio_speech"`` models.
    - ``stt_pricing_per_1m_seconds``: USD per 1M audio seconds.
      Populated only for ``mode == "audio_transcription"`` models.
    - ``video_pricing_per_second``: USD per generated video second.
      Populated only for ``mode == "video_generation"`` models.
    - ``embedding_pricing_per_1m``: USD per 1M tokens for embeddings.
      Populated only for ``mode == "embedding"`` models.

    Chat-pricing fields (``input_price_per_1m`` / ``output_price_per_1m``)
    are kept at the top level for chat models so existing UI / cost
    code paths continue to work without a polymorphism shim — the
    mode discriminator tells consumers which slot is meaningful.

    ``pricing_source`` and ``description_source`` carry per-field
    provenance so the UI can explain where a particular row's data
    came from (e.g. for debugging an unexpected price).
    """

    qualified_id: str
    provider_id: str
    raw_id: str
    label: str
    description: str
    context: int
    input_price_per_1m: float
    output_price_per_1m: float
    cache_hit_price_per_1m: float | None
    supports_tools: bool
    supports_vision: bool
    supports_pdf_input: bool
    tags: list[str]
    pricing_source: PricingSource
    description_source: DescriptionSource
    # Phase 5: mode discriminator + per-mode pricing slots.
    mode: ModelMode = "chat"
    image_pricing: dict[str, float] | None = None
    tts_pricing_per_1m_chars: float | None = None
    stt_pricing_per_1m_seconds: float | None = None
    video_pricing_per_second: float | None = None
    embedding_pricing_per_1m: float | None = None


@dataclass
class _CatalogState:
    """Module-level mutable catalog state, mutex-protected.

    Kept as a single dataclass so the lock guards a coherent snapshot:
    all reads + writes go through the same module-level
    ``_state_lock`` so a UI render against ``info_by_qualified_id`` and
    a refresh thread merging new data never observe a half-updated
    view.
    """

    # Merged catalog: qualified_id -> ModelInfo. Only entries that
    # cleared the completeness gate live here.
    info_by_qualified_id: dict[str, ModelInfo] = field(default_factory=dict)
    # Per-provider counts of models that we *saw* but had to drop
    # because metadata was incomplete. The picker surfaces this as a
    # "N models hidden — pricing or description not yet verified"
    # caption.
    hidden_per_provider: dict[str, int] = field(default_factory=dict)
    # Per-provider raw-id lists (the unfiltered ``/v1/models`` output).
    # Populated by ``refresh()`` regardless of completeness — useful for
    # the Settings card status caption ("N models available") which
    # cares about availability, not picker readiness.
    raw_ids_per_provider: dict[str, list[str]] = field(default_factory=dict)
    # Last successful refresh time per provider (for the "Last
    # refreshed Nm ago" caption).
    last_refreshed: dict[str, datetime] = field(default_factory=dict)
    # Last successful LiteLLM registry fetch.
    last_litellm: datetime | None = None
    # Last successful OpenRouter catalog fetch.
    last_openrouter: datetime | None = None
    # Per-source error messages (for the modal warning chip).
    errors: dict[str, str] = field(default_factory=dict)


_state_lock = threading.Lock()
_state = _CatalogState()


def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically (tmp + os.replace) with default permissions."""
    _ensure_config_dir()
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _read_json_if_fresh(path: Path, *, ttl_seconds: int = CATALOG_TTL_SECONDS) -> dict[str, Any] | None:
    """Return the parsed JSON when ``path`` exists and is younger than ``ttl_seconds``."""
    try:
        st = path.stat()
    except OSError:
        return None
    age = time.time() - st.st_mtime
    if age > ttl_seconds:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_json_any_age(path: Path) -> dict[str, Any] | None:
    """Return parsed JSON regardless of age (offline-floor read)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Per-provider /v1/models adapters
# ---------------------------------------------------------------------------
def _list_via_openai_compat(api_key: str, base_url: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """List raw model ids + (when present) per-model live metadata via OpenAI-compat.

    Returns ``(ids, live_meta)`` where ``live_meta[id]`` is a dict
    carrying any extra fields the provider's response surfaces. Most
    OpenAI-compat providers return only ``id`` + ``owned_by``; Together
    + OpenRouter additionally carry pricing / context info that we
    fold into the merge step (lowest priority — curated and LiteLLM win).
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.models.list()
    ids: list[str] = []
    live_meta: dict[str, dict[str, Any]] = {}
    for m in response.data:
        ids.append(m.id)
        # ``model_dump()`` is the Pydantic v2 method on openai SDK
        # response objects. Falling back to a manual extraction keeps
        # us safe against SDK schema drift.
        try:
            payload = m.model_dump()
        except Exception:
            payload = {"id": m.id}
        live_meta[m.id] = payload
    return sorted(set(ids)), live_meta


def _list_via_openai_native(client: Any) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """List raw model ids + minimal live metadata from native OpenAI client."""
    response = client.models.list()
    ids: list[str] = []
    live_meta: dict[str, dict[str, Any]] = {}
    for m in response.data:
        ids.append(m.id)
        try:
            payload = m.model_dump()
        except Exception:
            payload = {"id": m.id}
        live_meta[m.id] = payload
    return sorted(set(ids)), live_meta


def _list_via_anthropic_native(client: Any) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """List raw model ids + display_name from native Anthropic client.

    Drains pagination via ``has_more`` + ``after_id`` defensively even
    though Anthropic's ~10 active models fit on one page today.
    """
    out_ids: set[str] = set()
    live_meta: dict[str, dict[str, Any]] = {}
    response = client.models.list(limit=100)
    for m in response.data:
        out_ids.add(m.id)
        live_meta[m.id] = {
            "id": m.id,
            "display_name": getattr(m, "display_name", None) or "",
            "type": getattr(m, "type", None) or "",
        }
    while getattr(response, "has_more", False):
        last = response.data[-1].id if response.data else None
        if not last:
            break
        response = client.models.list(limit=100, after_id=last)
        for m in response.data:
            out_ids.add(m.id)
            live_meta[m.id] = {
                "id": m.id,
                "display_name": getattr(m, "display_name", None) or "",
                "type": getattr(m, "type", None) or "",
            }
    return sorted(out_ids), live_meta


def _list_via_google_native(client: Any) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """List raw model ids + description / context from native Google client.

    The ``google-genai`` SDK names models like ``"models/gemini-2.5-pro"``
    and exposes ``displayName``, ``description``, ``inputTokenLimit``,
    ``outputTokenLimit``, and ``supportedActions`` on each entry. We
    strip the ``models/`` prefix so the raw id matches what LiteLLM
    expects under its ``"gemini/"`` namespace.
    """
    out_ids: set[str] = set()
    live_meta: dict[str, dict[str, Any]] = {}
    for m in client.models.list():
        name = getattr(m, "name", None) or ""
        raw = name[len("models/"):] if name.startswith("models/") else name
        if not raw:
            continue
        out_ids.add(raw)
        live_meta[raw] = {
            "id": raw,
            "display_name": getattr(m, "display_name", None) or "",
            "description": getattr(m, "description", None) or "",
            "input_token_limit": getattr(m, "input_token_limit", None),
            "output_token_limit": getattr(m, "output_token_limit", None),
            "supported_actions": list(getattr(m, "supported_actions", []) or []),
        }
    return sorted(out_ids), live_meta


def _list_provider_models_with_meta(
    provider_id: str,
    client: Any,
    api_key: str,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Dispatch to the right per-kind adapter."""
    provider = PROVIDERS.get(provider_id)
    if provider is None:
        raise ValueError(f"Unknown provider id: {provider_id!r}")
    if provider.kind == "openai_native":
        return _list_via_openai_native(client)
    if provider.kind == "anthropic_native":
        return _list_via_anthropic_native(client)
    if provider.kind == "google_native":
        return _list_via_google_native(client)
    if provider.kind == "litellm_compat":
        return _list_via_openai_compat(api_key, provider.base_url)
    raise ValueError(f"Unhandled provider kind: {provider.kind!r}")


def list_raw_models(provider_id: str, client: Any, api_key: str) -> list[str]:
    """Return raw model ids reachable for ``provider_id``.

    Convenience wrapper around the listing adapters that returns just
    the sorted id list (no live metadata). Used by
    :func:`actions.connect_provider` for the connectivity check; the
    full enrichment path goes through :func:`refresh` which keeps the
    live metadata for the merge.
    """
    ids, _ = _list_provider_models_with_meta(provider_id, client, api_key)
    return ids


# ---------------------------------------------------------------------------
# LiteLLM pricing registry fetch
# ---------------------------------------------------------------------------
def _http_get_json(url: str, *, timeout: float = 15.0) -> Any:
    """Plain-stdlib JSON GET with a User-Agent. Raises on non-2xx / parse errors."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "wb-coding-agent",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def _load_litellm_registry() -> dict[str, Any]:
    """Read the LiteLLM registry from the freshest available source.

    Order: fresh disk cache → runtime fetch → repo-bundled snapshot.
    Returns ``{}`` when even the bundled snapshot is missing (fresh
    install, no network) so the merge step still runs.
    """
    cached = _read_json_if_fresh(LITELLM_CACHE_FILE)
    if isinstance(cached, dict) and isinstance(cached.get("registry"), dict):
        return cached["registry"]
    try:
        registry = _http_get_json(LITELLM_REGISTRY_URL)
        if isinstance(registry, dict):
            try:
                _atomic_write_json(
                    LITELLM_CACHE_FILE,
                    {
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "registry": registry,
                    },
                )
            except OSError:
                pass
            with _state_lock:
                _state.last_litellm = datetime.now(timezone.utc)
                _state.errors.pop("litellm", None)
            return registry
    except Exception as e:  # noqa: BLE001 — surfaced verbatim
        with _state_lock:
            _state.errors["litellm"] = f"LiteLLM fetch failed: {e}"
    # Fall through: stale disk cache, then repo-bundled snapshot.
    stale = _read_json_any_age(LITELLM_CACHE_FILE)
    if isinstance(stale, dict) and isinstance(stale.get("registry"), dict):
        return stale["registry"]
    bundled = _read_json_any_age(BUNDLED_LITELLM_SNAPSHOT)
    if isinstance(bundled, dict):
        # The bundled file is the raw upstream payload (not wrapped
        # with ``{fetched_at, registry}``) so we use it directly.
        return bundled
    return {}


def _fetch_openrouter_catalog() -> dict[str, dict[str, Any]]:
    """Read OpenRouter's full model catalog, cache to disk, return id-keyed dict.

    The OpenRouter response is ``{"data": [{...}, ...]}`` where each
    entry has ``id`` (e.g. ``"anthropic/claude-3-5-sonnet"``),
    ``description``, ``context_length``, ``pricing.prompt``,
    ``pricing.completion``, ``supported_parameters``, ``architecture``.
    We re-key by ``id`` for fast lookup during enrichment.
    """
    cached = _read_json_if_fresh(OPENROUTER_CACHE_FILE)
    if isinstance(cached, dict) and isinstance(cached.get("by_id"), dict):
        return cached["by_id"]
    try:
        body = _http_get_json(OPENROUTER_CATALOG_URL)
    except Exception as e:  # noqa: BLE001
        with _state_lock:
            _state.errors["openrouter"] = f"OpenRouter fetch failed: {e}"
        # Fall through to stale cache if present.
        stale = _read_json_any_age(OPENROUTER_CACHE_FILE)
        if isinstance(stale, dict) and isinstance(stale.get("by_id"), dict):
            return stale["by_id"]
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for entry in (body.get("data") if isinstance(body, dict) else []) or []:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            by_id[entry["id"]] = entry
    try:
        _atomic_write_json(
            OPENROUTER_CACHE_FILE,
            {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "by_id": by_id,
            },
        )
    except OSError:
        pass
    with _state_lock:
        _state.last_openrouter = datetime.now(timezone.utc)
        _state.errors.pop("openrouter", None)
    return by_id


# ---------------------------------------------------------------------------
# Per-source resolvers
# ---------------------------------------------------------------------------
def _curated_meta(qualified_id: str) -> dict[str, Any] | None:
    """Return the curated metadata dict for ``qualified_id`` if present."""
    from models import MODEL_METADATA
    return MODEL_METADATA.get(qualified_id)


def _litellm_lookup(
    registry: dict[str, Any],
    provider: Provider,
    raw_id: str,
) -> dict[str, Any] | None:
    """Look up ``provider.litellm_prefix + raw_id`` in the LiteLLM registry.

    Returns the raw entry when the lookup succeeds AND the
    ``litellm_provider`` field on the entry asserts the same provider
    family we expect. The provider-family check is the integrity
    safeguard that keeps us from substituting (say) Together's
    Llama-3.3 entry for Groq's identically-named model — different
    inference offerings, different prices.

    Returns ``None`` when the entry isn't there or the provider
    assertion fails.
    """
    if not isinstance(registry, dict) or not registry:
        return None
    key = f"{provider.litellm_prefix}{raw_id}"
    entry = registry.get(key)
    if not isinstance(entry, dict):
        return None
    expected = _LITELLM_PROVIDER_FOR.get(provider.id)
    if expected is None:
        # We don't have a provider-family expectation, so accept the
        # entry as-is. This covers ``openai`` / ``anthropic`` whose
        # entries don't always carry ``litellm_provider``.
        return entry
    actual = entry.get("litellm_provider")
    if actual is not None and actual != expected:
        return None
    return entry


# Mapping from our provider id to the value LiteLLM's registry uses in
# its ``litellm_provider`` field. Where None, we don't enforce the
# assertion (no false-positive risk because the prefix is unique). Kept
# private because it's an implementation detail of the LiteLLM-fetch
# layer.
_LITELLM_PROVIDER_FOR: dict[str, str | None] = {
    "wandb": None,
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "mistral": "mistral",
    "xai": "xai",
    "together": "together_ai",
    "fireworks": "fireworks_ai",
    "groq": "groq",
    "openrouter": "openrouter",
    "deepinfra": "deepinfra",
    "cerebras": "cerebras",
}


def _openrouter_lookup(
    catalog: dict[str, dict[str, Any]],
    provider: Provider,
    raw_id: str,
) -> dict[str, Any] | None:
    """Find an OpenRouter entry that describes ``raw_id`` from this provider.

    OpenRouter ids look like ``"anthropic/claude-3-5-sonnet"`` so we try
    a few candidate keys: the bare raw id, ``provider/raw``, and
    ``provider_label/raw``. The match is best-effort — OpenRouter's
    purpose for us is descriptions only (and pricing for the
    ``openrouter:*`` namespace), so a missed match just means the
    catalog falls through to the next source.
    """
    if not catalog:
        return None
    candidates = [raw_id]
    # OpenRouter prefixes ids with the upstream provider's slug.
    or_provider_slug = {
        "wandb": "wandb",
        "openai": "openai",
        "anthropic": "anthropic",
        "gemini": "google",
        "mistral": "mistralai",
        "xai": "x-ai",
        "together": "meta-llama",  # not a real prefix; raw_id often already has owner
        "fireworks": "fireworks",
        "groq": "groq",
        "openrouter": "",
        "deepinfra": "deepinfra",
        "cerebras": "cerebras",
    }
    slug = or_provider_slug.get(provider.id, "")
    if slug and "/" not in raw_id:
        candidates.append(f"{slug}/{raw_id}")
    # Some Anthropic ids have date suffixes; try the prefix-only form too.
    if provider.id == "anthropic" and "-202" in raw_id:
        # ``claude-3-5-sonnet-20241022`` → try ``anthropic/claude-3.5-sonnet``
        # since OpenRouter aliases differ; this is best-effort only.
        bare = raw_id.split("-202", 1)[0]
        candidates.append(f"anthropic/{bare.replace('claude-3-', 'claude-3.')}")
    for cand in candidates:
        entry = catalog.get(cand)
        if entry is not None:
            return entry
    return None


def _to_per_1m(per_token: Any) -> float | None:
    """Convert a LiteLLM ``input_cost_per_token`` (per-token USD) to per-1M.

    LiteLLM's pricing fields are floats like ``0.0000025`` (per token)
    or strings the registry occasionally produces for newer entries.
    We coerce conservatively and return ``None`` on any parse failure.
    """
    if per_token is None:
        return None
    try:
        v = float(per_token)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return round(v * 1_000_000, 6)


def _to_int(val: Any) -> int | None:
    """Best-effort int coercion. Returns ``None`` on failure or non-positive."""
    if val is None:
        return None
    try:
        n = int(val)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


# ---------------------------------------------------------------------------
# Merge logic + completeness gate
# ---------------------------------------------------------------------------
_LITELLM_MODE_TO_OURS: dict[str, ModelMode] = {
    "chat": "chat",
    "completion": "chat",
    "image_generation": "image_generation",
    "audio_speech": "audio_speech",
    "audio_transcription": "audio_transcription",
    "video_generation": "video_generation",
    "embedding": "embedding",
}


def _detect_mode(litellm_entry: dict[str, Any]) -> ModelMode:
    """Map LiteLLM's ``mode`` field to our :data:`ModelMode` enum.

    LiteLLM uses ``"chat"`` / ``"completion"`` interchangeably for
    text models — we normalize both to ``"chat"`` since
    ``client.chat.completions.create`` is the only call shape we
    serve in v1.
    """
    raw = (litellm_entry.get("mode") or "").strip()
    return _LITELLM_MODE_TO_OURS.get(raw, "chat")


def _merge_one(
    provider: Provider,
    raw_id: str,
    *,
    live_meta: dict[str, Any] | None,
    litellm_registry: dict[str, Any],
    openrouter_catalog: dict[str, dict[str, Any]],
) -> ModelInfo | None:
    """Build a :class:`ModelInfo` for one ``(provider, raw_id)`` pair.

    Returns ``None`` if the entry fails the strict completeness gate
    (missing pricing, description, or context). Callers count those
    failures into ``hidden_per_provider``.

    Mode detection: LiteLLM's ``mode`` field on the entry tells us
    whether this is a chat / image / audio / video / embedding model.
    The completeness gate runs per-mode — chat needs both input and
    output prices, image_generation needs at least one
    ``input_cost_per_image`` rate, etc. Models without a recognizable
    mode-specific pricing field are dropped from the picker (same
    silence-beats-wrong-data principle as the chat path).
    """
    qualified = f"{provider.id}:{raw_id}"
    curated = _curated_meta(qualified) or {}
    litellm = _litellm_lookup(litellm_registry, provider, raw_id) or {}
    openrouter = _openrouter_lookup(openrouter_catalog, provider, raw_id) or {}
    live = live_meta or {}

    mode: ModelMode = _detect_mode(litellm)
    # Curated entries don't currently carry an explicit ``mode`` field
    # (every curated entry is a chat model in v1); a future curated
    # image / audio / video entry would override here.
    cur_mode = curated.get("mode")
    if isinstance(cur_mode, str) and cur_mode in _LITELLM_MODE_TO_OURS.values():
        mode = cur_mode  # type: ignore[assignment]

    # ---- Pricing (curated > litellm > openrouter[only if openrouter:*] > live)
    in_price: float | None = None
    out_price: float | None = None
    cache_hit_price: float | None = None
    pricing_source: PricingSource | None = None

    cur_in = curated.get("input_price_per_1m")
    cur_out = curated.get("output_price_per_1m")
    if isinstance(cur_in, (int, float)) and isinstance(cur_out, (int, float)):
        in_price = float(cur_in)
        out_price = float(cur_out)
        pricing_source = "curated"
        cache_hit_price = curated.get("cache_hit_price_per_1m")

    if pricing_source is None:
        ll_in = _to_per_1m(litellm.get("input_cost_per_token"))
        ll_out = _to_per_1m(litellm.get("output_cost_per_token"))
        if ll_in is not None and ll_out is not None:
            in_price = ll_in
            out_price = ll_out
            pricing_source = "litellm"
            cache_hit_price = _to_per_1m(litellm.get("cache_read_input_token_cost"))

    if pricing_source is None and provider.id == "openrouter":
        # OpenRouter pricing only used for openrouter:* qualified ids
        # (the user is paying OpenRouter directly, so OpenRouter's
        # pricing IS the direct rate from their perspective).
        pricing = openrouter.get("pricing") or {}
        or_in = _to_per_1m(pricing.get("prompt"))
        or_out = _to_per_1m(pricing.get("completion"))
        if or_in is not None and or_out is not None:
            in_price = or_in
            out_price = or_out
            pricing_source = "openrouter"

    if pricing_source is None:
        # Live API pricing fallthrough — Together + OpenRouter
        # historically include pricing in their /v1/models response.
        live_pricing = live.get("pricing") or {}
        live_in = _to_per_1m(live_pricing.get("input")) or _to_per_1m(live_pricing.get("prompt"))
        live_out = _to_per_1m(live_pricing.get("output")) or _to_per_1m(live_pricing.get("completion"))
        if live_in is not None and live_out is not None:
            in_price = live_in
            out_price = live_out
            pricing_source = "live"

    # Per-mode pricing (image / audio / video / embedding). Only
    # populated for non-chat modes; chat models stay on
    # input_price_per_1m / output_price_per_1m above.
    image_pricing: dict[str, float] | None = None
    tts_pricing: float | None = None
    stt_pricing: float | None = None
    video_pricing: float | None = None
    embedding_pricing: float | None = None

    if mode == "image_generation":
        # LiteLLM ships per-image image-generation prices in
        # ``input_cost_per_pixel`` (legacy) or via mode-specific
        # ``input_cost_per_image`` (current). Fall through to a
        # default 1024x1024/standard slot.
        per_image = litellm.get("input_cost_per_image")
        if isinstance(per_image, (int, float)) and per_image > 0:
            image_pricing = {"1024x1024/standard": float(per_image)}
        if image_pricing is None:
            return None
        # Image-gen models don't have meaningful in/out token prices;
        # zero them out so the chat-pricing path's strict gate doesn't
        # accidentally fail an image-generation entry.
        in_price = 0.0
        out_price = 0.0
        pricing_source = pricing_source or "litellm"

    elif mode == "audio_speech":
        per_char = _to_per_1m(litellm.get("input_cost_per_character"))
        if per_char is not None:
            tts_pricing = per_char
        else:
            return None
        in_price = 0.0
        out_price = 0.0
        pricing_source = pricing_source or "litellm"

    elif mode == "audio_transcription":
        # LiteLLM uses ``input_cost_per_second`` for STT.
        per_sec = litellm.get("input_cost_per_second")
        if isinstance(per_sec, (int, float)) and per_sec > 0:
            stt_pricing = round(float(per_sec) * 1_000_000, 6)
        else:
            return None
        in_price = 0.0
        out_price = 0.0
        pricing_source = pricing_source or "litellm"

    elif mode == "video_generation":
        per_vsec = litellm.get("input_cost_per_video_per_second")
        if isinstance(per_vsec, (int, float)) and per_vsec > 0:
            video_pricing = float(per_vsec)
        else:
            return None
        in_price = 0.0
        out_price = 0.0
        pricing_source = pricing_source or "litellm"

    elif mode == "embedding":
        # Embeddings only have an "input" cost.
        emb = _to_per_1m(litellm.get("input_cost_per_token"))
        if emb is not None:
            embedding_pricing = emb
            in_price = emb
            out_price = 0.0
            pricing_source = pricing_source or "litellm"
        else:
            return None

    elif pricing_source is None or in_price is None or out_price is None:
        # Chat models still require both axes.
        return None

    # ---- Description (curated > openrouter > live)
    description: str | None = None
    description_source: DescriptionSource | None = None

    cur_desc = curated.get("description")
    if isinstance(cur_desc, str) and cur_desc.strip():
        description = cur_desc.strip()
        description_source = "curated"

    if description is None:
        or_desc = openrouter.get("description")
        if isinstance(or_desc, str) and or_desc.strip():
            description = or_desc.strip()
            description_source = "openrouter"

    if description is None:
        # Live API description (Google's `description`, Anthropic's
        # `display_name` as a soft floor).
        live_desc = live.get("description") or live.get("display_name")
        if isinstance(live_desc, str) and live_desc.strip():
            description = live_desc.strip()
            description_source = "live"

    if description is None or description_source is None:
        return None

    # ---- Context (curated > litellm > live)
    context: int | None = None
    cur_ctx = curated.get("context")
    if isinstance(cur_ctx, str):
        # Curated entries store context as e.g. ``"128k"`` or ``"1000k"`` — parse.
        s = cur_ctx.strip().lower().rstrip("k")
        try:
            n = int(round(float(s)))
            context = n * 1000
        except ValueError:
            context = None
    elif isinstance(cur_ctx, (int, float)):
        context = int(cur_ctx)

    if context is None:
        ll_ctx = _to_int(litellm.get("max_input_tokens")) or _to_int(litellm.get("max_tokens"))
        if ll_ctx:
            context = ll_ctx

    if context is None:
        live_ctx = _to_int(live.get("context_length")) or _to_int(live.get("input_token_limit"))
        if live_ctx:
            context = live_ctx
        else:
            or_ctx = _to_int(openrouter.get("context_length"))
            if or_ctx:
                context = or_ctx

    if context is None or context <= 0:
        if mode == "chat":
            return None
        # For non-chat modes a context window isn't meaningful; use 0
        # as a sentinel so the field stays non-None.
        context = 0

    # ---- Capability flags
    supports_tools = bool(litellm.get("supports_function_calling", True))
    # Pull OpenRouter's modality arrays for both the supports_vision
    # check and the new auto-tag derivation below. They look like
    # ``["text", "image", "file"]`` etc.; the ``modality`` short
    # string (``"text+image->text"``) is a fallback for older entries.
    arch = openrouter.get("architecture") or {}
    input_modalities = [m.lower() for m in (arch.get("input_modalities") or []) if isinstance(m, str)]
    output_modalities = [m.lower() for m in (arch.get("output_modalities") or []) if isinstance(m, str)]
    short_modality = (arch.get("modality") or "").lower()
    supports_vision = bool(litellm.get("supports_vision", False))
    if not supports_vision:
        if "image" in input_modalities or "image" in short_modality:
            supports_vision = True
    supports_pdf = qualified in PDF_SUPPORT_ALLOWLIST

    # ---- Auto-derived modality flags (used by the tag derivation
    # below + ALSO surfaced as picker tabs so users can filter to
    # "models that take images" / "models that produce audio" etc.)
    is_image_gen = (
        mode == "image_generation"
        or "image" in output_modalities
    )
    is_audio_gen = (
        mode == "audio_speech"
        or "audio" in output_modalities
        or bool(litellm.get("supports_audio_output", False))
    )
    is_audio_in = (
        mode == "audio_transcription"
        or "audio" in input_modalities
        or bool(litellm.get("supports_audio_input", False))
    )
    is_video_gen = (
        mode == "video_generation"
        or "video" in output_modalities
    )
    is_video_in = "video" in input_modalities

    # ---- Tags: curated wins for opinionated tags (coding, reasoning,
    # frontier); auto-derived layer adds the objective ones
    # (long_context, cheap, fast, multimodal) plus modality flags
    # (vision, image_gen, audio_gen, audio_in, video_gen, video_in).
    # Auto-tags merge with curated rather than replace, so a curated
    # ``coding`` entry on a vision-capable model gets both ``coding``
    # and ``vision`` tags.
    tags = list(curated.get("tags") or [])
    auto_tags: list[str] = []
    if context >= 200_000:
        auto_tags.append("long_context")
    if out_price <= 0.5:
        auto_tags.append("cheap")
    if provider.id in ("groq", "cerebras"):
        auto_tags.append("fast")
    if supports_vision:
        auto_tags.append("multimodal")
        # ``vision`` is the picker-tab tag (chat models that accept
        # image input). ``multimodal`` stays for back-compat with the
        # original tag set.
        if mode == "chat":
            auto_tags.append("vision")
    if is_image_gen:
        auto_tags.append("image_gen")
    if is_audio_gen:
        auto_tags.append("audio_gen")
    if is_audio_in and not is_audio_gen:
        # Don't double-tag a model that does both — audio_gen wins
        # because that's the more useful filter (TTS models).
        auto_tags.append("audio_in")
    if is_video_gen:
        auto_tags.append("video_gen")
    if is_video_in and not is_video_gen:
        auto_tags.append("video_in")
    for tag in auto_tags:
        if tag not in tags:
            tags.append(tag)

    # ---- Label
    label = curated.get("label") or live.get("display_name") or raw_id.split("/")[-1]

    return ModelInfo(
        qualified_id=qualified,
        provider_id=provider.id,
        raw_id=raw_id,
        label=str(label),
        description=description,
        context=int(context),
        input_price_per_1m=float(in_price),
        output_price_per_1m=float(out_price),
        cache_hit_price_per_1m=(
            float(cache_hit_price) if isinstance(cache_hit_price, (int, float)) else None
        ),
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        supports_pdf_input=supports_pdf,
        tags=tags,
        pricing_source=pricing_source,
        description_source=description_source,
        mode=mode,
        image_pricing=image_pricing,
        tts_pricing_per_1m_chars=tts_pricing,
        stt_pricing_per_1m_seconds=stt_pricing,
        video_pricing_per_second=video_pricing,
        embedding_pricing_per_1m=embedding_pricing,
    )


def _all_curated_for_provider(provider_id: str) -> list[str]:
    """Return curated raw ids for ``provider_id`` (so curated entries surface
    even when the provider's /v1/models doesn't list them, e.g. an alias).

    We always merge in curated raw ids regardless of whether the live
    listing mentioned them, so the picker shows our hand-picked rows
    even if the provider's listing surface is incomplete.
    """
    from models import MODEL_METADATA
    out: list[str] = []
    prefix = f"{provider_id}:"
    for qid in MODEL_METADATA:
        if qid.startswith(prefix):
            out.append(qid[len(prefix):])
    return out


def _persist_live_catalog() -> None:
    """Snapshot the merged catalog to disk for offline-first launches.

    Stores ``ModelInfo`` payloads as plain dicts via ``asdict`` so the
    file is human-inspectable. The next launch reads it via
    :func:`_hydrate_from_disk` to populate the in-memory catalog
    before any refresh runs.
    """
    with _state_lock:
        snapshot = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "models": [asdict(mi) for mi in _state.info_by_qualified_id.values()],
            "hidden_per_provider": dict(_state.hidden_per_provider),
            "raw_ids_per_provider": {k: list(v) for k, v in _state.raw_ids_per_provider.items()},
            "last_refreshed": {k: v.isoformat() for k, v in _state.last_refreshed.items()},
            "last_litellm": _state.last_litellm.isoformat() if _state.last_litellm else None,
            "last_openrouter": _state.last_openrouter.isoformat() if _state.last_openrouter else None,
        }
    try:
        _atomic_write_json(LIVE_CATALOG_FILE, snapshot)
    except OSError:
        pass


def _hydrate_from_disk() -> None:
    """Read the previously-persisted catalog into in-memory state.

    Used at module import + on Streamlit startup so the picker has
    something to show before the first network refresh completes.
    The TTL check is bypassed deliberately — any cached data is
    better than an empty picker, and a stale entry will be replaced
    by the next ``refresh()``.
    """
    raw = _read_json_any_age(LIVE_CATALOG_FILE)
    if not isinstance(raw, dict):
        return
    models = raw.get("models")
    if not isinstance(models, list):
        return
    hydrated: dict[str, ModelInfo] = {}
    for m in models:
        if not isinstance(m, dict):
            continue
        try:
            mi = ModelInfo(
                qualified_id=m["qualified_id"],
                provider_id=m["provider_id"],
                raw_id=m["raw_id"],
                label=m["label"],
                description=m["description"],
                context=int(m["context"]),
                input_price_per_1m=float(m["input_price_per_1m"]),
                output_price_per_1m=float(m["output_price_per_1m"]),
                cache_hit_price_per_1m=(
                    float(m["cache_hit_price_per_1m"])
                    if m.get("cache_hit_price_per_1m") is not None
                    else None
                ),
                supports_tools=bool(m.get("supports_tools", True)),
                supports_vision=bool(m.get("supports_vision", False)),
                supports_pdf_input=bool(m.get("supports_pdf_input", False)),
                tags=list(m.get("tags") or []),
                pricing_source=m.get("pricing_source", "curated"),
                description_source=m.get("description_source", "curated"),
            )
        except (KeyError, ValueError, TypeError):
            continue
        hydrated[mi.qualified_id] = mi
    with _state_lock:
        _state.info_by_qualified_id.update(hydrated)
        hp = raw.get("hidden_per_provider") or {}
        if isinstance(hp, dict):
            _state.hidden_per_provider.update({k: int(v) for k, v in hp.items() if isinstance(v, int)})
        rid = raw.get("raw_ids_per_provider") or {}
        if isinstance(rid, dict):
            _state.raw_ids_per_provider.update(
                {k: list(v) for k, v in rid.items() if isinstance(v, list)}
            )
        lr = raw.get("last_refreshed") or {}
        if isinstance(lr, dict):
            for k, v in lr.items():
                if isinstance(v, str):
                    try:
                        _state.last_refreshed[k] = datetime.fromisoformat(v)
                    except ValueError:
                        pass
        if isinstance(raw.get("last_litellm"), str):
            try:
                _state.last_litellm = datetime.fromisoformat(raw["last_litellm"])
            except ValueError:
                pass
        if isinstance(raw.get("last_openrouter"), str):
            try:
                _state.last_openrouter = datetime.fromisoformat(raw["last_openrouter"])
            except ValueError:
                pass


# Hydrate at import so callers get a sensible catalog before any refresh.
_hydrate_from_disk()


# ---------------------------------------------------------------------------
# Public API: refresh / get_info / available_qualified_ids / hidden_summary
# ---------------------------------------------------------------------------
def refresh(provider_id: str, client: Any, api_key: str = "") -> list[ModelInfo]:
    """Refresh the catalog for ``provider_id``. Synchronous; blocks the caller.

    Steps:

    1. List raw model ids via the per-kind ``/v1/models`` adapter.
    2. Fold in any curated raw ids that the live listing missed (so
       hand-picked recommended rows always appear even if the
       provider's listing surface doesn't mention them).
    3. Pull the LiteLLM registry + OpenRouter catalog (with disk-cache
       fall-through on network failure).
    4. Run the merge step on every (provider, raw_id) candidate. The
       completeness gate returns ``None`` for incomplete entries —
       count those into ``hidden_per_provider`` for the modal caption.
    5. Atomically replace this provider's slice of the catalog (so a
       provider that briefly loses connectivity doesn't drop ALL its
       models — it keeps the prior cached set until the next
       successful refresh).

    Returns the list of :class:`ModelInfo` entries that survived the
    completeness gate. Errors are surfaced through ``_state.errors`` —
    the caller (typically ``actions.connect_provider``) reads them
    out of ``ss.connect_errors`` to render in the Settings card.
    """
    provider = get_provider(provider_id)
    if provider is None:
        return []

    # 1. Live listing.
    try:
        raw_ids, live_meta = _list_provider_models_with_meta(provider_id, client, api_key)
    except Exception as e:  # noqa: BLE001
        with _state_lock:
            _state.errors[provider_id] = f"{provider.label} list-models failed: {e}"
        # Don't drop existing entries on transient failure — keep the
        # last-known-good slice. Just bail.
        with _state_lock:
            return [
                mi for mi in _state.info_by_qualified_id.values()
                if mi.provider_id == provider_id
            ]

    # 2. Fold in curated ids the live listing missed.
    curated_ids = _all_curated_for_provider(provider_id)
    candidate_set: set[str] = set(raw_ids) | set(curated_ids)

    # 3. LiteLLM + OpenRouter blobs.
    litellm_registry = _load_litellm_registry()
    openrouter_catalog = _fetch_openrouter_catalog()

    # 4. Merge each candidate.
    fresh: dict[str, ModelInfo] = {}
    hidden = 0
    for raw in candidate_set:
        meta = live_meta.get(raw, {})
        info = _merge_one(
            provider,
            raw,
            live_meta=meta,
            litellm_registry=litellm_registry,
            openrouter_catalog=openrouter_catalog,
        )
        if info is None:
            hidden += 1
            continue
        fresh[info.qualified_id] = info

    # 5. Replace this provider's slice of the catalog atomically.
    with _state_lock:
        # Drop the prior slice for this provider only.
        for qid in list(_state.info_by_qualified_id.keys()):
            if _state.info_by_qualified_id[qid].provider_id == provider_id:
                del _state.info_by_qualified_id[qid]
        _state.info_by_qualified_id.update(fresh)
        _state.hidden_per_provider[provider_id] = hidden
        _state.raw_ids_per_provider[provider_id] = list(raw_ids)
        _state.last_refreshed[provider_id] = datetime.now(timezone.utc)
        _state.errors.pop(provider_id, None)

    _persist_live_catalog()
    return sorted(fresh.values(), key=lambda mi: mi.label.casefold())


def refresh_all(clients: dict[str, Any], api_keys: dict[str, str]) -> dict[str, list[ModelInfo]]:
    """Refresh every provider for which we have either a client or a key.

    Skips providers that have neither a live client nor a saved API
    key — those simply aren't connected, and refreshing them would
    just hit the same auth error every time. The catch-all per-
    provider exception handling in :func:`refresh` means a single
    failing provider never blocks the others.
    """
    results: dict[str, list[ModelInfo]] = {}
    for pid, provider in PROVIDERS.items():
        client = clients.get(pid)
        api_key = (api_keys.get(pid) or "").strip()
        if provider.kind == "litellm_compat" and not api_key:
            continue
        if provider.kind != "litellm_compat" and client is None:
            continue
        try:
            results[pid] = refresh(pid, client, api_key)
        except Exception as e:  # noqa: BLE001 — surfaced via _state.errors
            with _state_lock:
                _state.errors[pid] = f"{provider.label} refresh raised: {e}"
            results[pid] = []
    return results


def refresh_all_async(
    clients: dict[str, Any],
    api_keys: dict[str, str],
    on_done: Callable[[], None] | None = None,
) -> threading.Thread:
    """Spawn a daemon thread to run :func:`refresh_all` in the background.

    The Streamlit chat page polls a 0.5s ``@st.fragment`` while
    ``ss.model_catalog_refreshing`` is True; this function flips that
    flag back to False (via ``on_done``) once the refresh completes
    so the modal re-renders with the new catalog.

    The thread is daemon-marked so it doesn't block app shutdown when
    the user closes the window mid-refresh.
    """
    def _runner() -> None:
        try:
            refresh_all(clients, api_keys)
        except Exception as e:  # noqa: BLE001
            with _state_lock:
                _state.errors["__refresh_all__"] = f"refresh_all failed: {e}"
        if on_done is not None:
            try:
                on_done()
            except Exception:  # noqa: BLE001 — never let on_done crash the thread
                pass

    t = threading.Thread(target=_runner, daemon=True, name="model-catalog-refresh")
    t.start()
    return t


def get_info(qualified_id: str) -> ModelInfo | None:
    """Return the :class:`ModelInfo` for ``qualified_id`` or ``None`` if hidden / unknown."""
    with _state_lock:
        return _state.info_by_qualified_id.get(qualified_id)


def all_qualified_ids() -> list[str]:
    """Return every qualified id in the in-memory catalog (sorted)."""
    with _state_lock:
        return sorted(_state.info_by_qualified_id.keys())


def available_qualified_ids(clients: dict[str, Any] | None = None) -> list[str]:
    """Qualified ids the user can currently call — i.e. on a connected provider.

    Two-step filter:

    1. **Connectivity** — the provider must be connected this session.
       ``litellm_compat`` providers count as connected when the
       connect flow successfully listed any models (LiteLLM is
       stateless, so we use the raw-id list as the connectivity
       signal). Native providers (``openai_native`` / ``anthropic_native``
       / ``google_native``) require a non-None client object in
       ``clients[pid]``.
    2. **Reachability** — the model's raw id MUST appear in the live
       ``/v1/models`` listing for that provider. This catches the
       case where ``MODEL_METADATA`` carries a curated entry for, say,
       ``openai:o1`` but the user's API key doesn't have access to
       o1; we'd otherwise let them pick it and crash on the first
       chat call with a 401/403. The strict completeness gate
       (curated > LiteLLM > OpenRouter > live) guarantees real
       *metadata* but the *reachability* check is a separate axis
       enforced here.

    When ``clients`` is None, both filters are bypassed and every
    catalogued model is returned (used by the Settings page when no
    provider has been connected yet so users can preview what's
    available).
    """
    with _state_lock:
        all_ids = sorted(_state.info_by_qualified_id.keys())
    if clients is None:
        return all_ids
    out: list[str] = []
    for qid in all_ids:
        with _state_lock:
            info = _state.info_by_qualified_id.get(qid)
            raw_ids = _state.raw_ids_per_provider.get(info.provider_id, []) if info else []
        if info is None:
            continue
        provider = PROVIDERS.get(info.provider_id)
        if provider is None:
            continue
        # Connectivity check.
        if provider.kind == "litellm_compat":
            connected = bool(raw_ids)
        else:
            connected = clients.get(info.provider_id) is not None
        if not connected:
            continue
        # Reachability check: the live listing must have confirmed
        # this raw id. Without this, a curated MODEL_METADATA entry
        # for a model the user's key can't actually call would land
        # in the picker and crash on the first turn.
        if info.raw_id not in raw_ids:
            continue
        out.append(qid)
    return out


def hidden_models_summary(clients: dict[str, Any] | None = None) -> dict[str, int]:
    """Per-provider counts of models hidden by the completeness gate.

    Limited to providers that are currently connected when ``clients``
    is provided so the picker doesn't surface "5 hidden on Mistral"
    when the user hasn't even connected Mistral.
    """
    with _state_lock:
        full = dict(_state.hidden_per_provider)
    if clients is None:
        return full
    out: dict[str, int] = {}
    for pid, n in full.items():
        provider = PROVIDERS.get(pid)
        if provider is None:
            continue
        with _state_lock:
            connected = bool(_state.raw_ids_per_provider.get(pid))
        if provider.kind == "litellm_compat":
            if connected:
                out[pid] = n
        else:
            if clients.get(pid) is not None:
                out[pid] = n
    return out


def last_refreshed_at(provider_id: str) -> datetime | None:
    """Return the wall-clock time we last successfully refreshed ``provider_id``."""
    with _state_lock:
        return _state.last_refreshed.get(provider_id)


def newest_refresh() -> datetime | None:
    """Return the most-recent wall-clock refresh time across all providers."""
    with _state_lock:
        if not _state.last_refreshed:
            return None
        return max(_state.last_refreshed.values())


def last_litellm_refresh() -> datetime | None:
    """Return the wall-clock time we last successfully fetched LiteLLM."""
    with _state_lock:
        return _state.last_litellm


def errors() -> dict[str, str]:
    """Snapshot of per-source / per-provider error strings."""
    with _state_lock:
        return dict(_state.errors)


def models_with_mode(
    mode: ModelMode,
    *,
    available: set[str] | list[str] | None = None,
) -> list[ModelInfo]:
    """Return :class:`ModelInfo` entries with ``info.mode == mode``.

    When ``available`` is provided (typically the qualified-id set
    returned by :func:`available_qualified_ids`), the result is
    intersected so callers can render "image-gen models reachable
    from a connected provider" instead of the catalog-global view.
    """
    with _state_lock:
        all_infos = [mi for mi in _state.info_by_qualified_id.values() if mi.mode == mode]
    if available is None:
        return all_infos
    avail = set(available)
    return [mi for mi in all_infos if mi.qualified_id in avail]


def default_model_for_mode(
    mode: ModelMode,
    clients: dict[str, Any],
    *,
    prefer_cheapest: bool = True,
) -> ModelInfo | None:
    """Pick a sensible default model for ``mode`` from connected providers.

    Used by the Phase 5 media tools (``generate_image`` /
    ``generate_speech`` / ``generate_video``) when the user hasn't
    explicitly named a model. ``prefer_cheapest=True`` sorts by the
    lowest mode-specific price; ``False`` returns the first match.

    Returns ``None`` when no connected provider exposes a model in
    the requested mode — the tool should surface a clear "no
    provider supports this" error to the user.
    """
    available = set(available_qualified_ids(clients))
    candidates = models_with_mode(mode, available=available)
    if not candidates:
        return None
    if not prefer_cheapest:
        return candidates[0]
    if mode == "chat":
        candidates.sort(key=lambda mi: mi.output_price_per_1m)
    elif mode == "image_generation":
        candidates.sort(key=lambda mi: min((mi.image_pricing or {0: 0}).values()))  # type: ignore[arg-type]
    elif mode == "audio_speech":
        candidates.sort(key=lambda mi: mi.tts_pricing_per_1m_chars or float("inf"))
    elif mode == "audio_transcription":
        candidates.sort(key=lambda mi: mi.stt_pricing_per_1m_seconds or float("inf"))
    elif mode == "video_generation":
        candidates.sort(key=lambda mi: mi.video_pricing_per_second or float("inf"))
    elif mode == "embedding":
        candidates.sort(key=lambda mi: mi.embedding_pricing_per_1m or float("inf"))
    return candidates[0]


__all__ = [
    "BUNDLED_LITELLM_SNAPSHOT",
    "CATALOG_TTL_SECONDS",
    "LITELLM_REGISTRY_URL",
    "LIVE_CATALOG_FILE",
    "ModelInfo",
    "OPENROUTER_CATALOG_URL",
    "PDF_SUPPORT_ALLOWLIST",
    "PROVIDER_LITELLM_PREFIX",
    "all_qualified_ids",
    "available_qualified_ids",
    "errors",
    "get_info",
    "hidden_models_summary",
    "last_litellm_refresh",
    "last_refreshed_at",
    "list_raw_models",
    "newest_refresh",
    "refresh",
    "refresh_all",
    "refresh_all_async",
]
