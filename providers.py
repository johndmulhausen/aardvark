"""Provider catalog and client factory for the multi-provider inference layer.

Single source of truth for which inference providers the app supports, what
kind of dispatch each one needs, and how to build a client for the ones
that own persistent client objects.

The 12-entry ``PROVIDERS`` dict is split four ways by ``kind``:

- ``"openai_native"`` (1 provider): OpenAI. Uses the ``openai`` SDK direct
  to ``api.openai.com`` so users get first-class access to features
  LiteLLM only passes through generically — ``reasoning_effort`` for the
  o-series, typed structured outputs via
  ``client.beta.chat.completions.parse(...)``, ``prediction`` for predicted
  outputs, and the Files / Batch / Vector Stores APIs.
- ``"anthropic_native"`` (1 provider): Anthropic. Uses the ``anthropic``
  SDK so users get prompt caching via ``cache_control`` blocks (~90%
  repeat-cost savings on every turn after the first), extended thinking
  with typed thinking blocks, computer use / bash / text-editor pre-built
  tools, the Message Batches API (50% off async work), and
  ``client.messages.count_tokens(...)`` for pre-flight cost estimation.
- ``"google_native"`` (1 provider): Google Gemini. Uses the
  ``google-genai`` SDK so users get grounding with Google Search,
  code-execution, ``client.caches.create(...)`` for context caching, and
  native Imagen / Veo image / video gen.
- ``"litellm_compat"`` (9 providers): W&B Inference, Mistral, xAI,
  Together, Fireworks, Groq, OpenRouter, DeepInfra, Cerebras. All nine
  are OpenAI-compatible chat-completions endpoints with no significantly
  differentiating native features for our use case. Calls dispatch via
  ``litellm.completion(model=f"{prefix}{raw_id}", api_key=..., base_url=..., ...)``
  with no persistent client object — LiteLLM is stateless. The catalog-
  fetch path in ``model_catalog.py`` does construct an ``openai.OpenAI``
  instance per LiteLLM-routed provider for ``client.models.list()`` calls
  because LiteLLM's listing surface is per-provider, but those clients
  live there and aren't exposed via session state.

**Important: the LiteLLM library does not mark up tokens.** It is the
MIT-licensed open source ``litellm`` PyPI package and ``litellm.completion``
makes direct ``httpx`` calls to the provider's endpoint using the user's
API key. The ``litellm.ai`` hosted proxy (LiteLLM Cloud) is a separate
commercial product we never use. The pricing in LiteLLM's
``model_prices_and_context_window.json`` is direct-from-provider rates
because the library makes direct-from-provider calls.

The ``tier`` field drives the Settings page card layout:
``"primary"`` cards render in the default body of the Settings page;
``"more"`` cards render inside an ``st.expander("More providers")``.
OpenRouter is intentionally placed in ``"more"`` and carries a
non-empty ``notes`` string about its 5–10% markup so the UI can warn
users.

The ``litellm_prefix`` field is consulted by ``chat_streams.py``
(Phase 3) to build the ``model=`` argument passed to
``litellm.completion(...)`` and by ``model_catalog.py`` (Phase 2) to
look up entries in LiteLLM's pricing registry under the right namespace.
For ``openai_native`` / ``anthropic_native`` / ``google_native`` the
prefix is unused at call time but kept for completeness so
``model_catalog.py`` can resolve LiteLLM pricing entries for those
providers too (LiteLLM ships their pricing under ``""`` / ``""`` /
``"gemini/"`` namespaces respectively).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ProviderKind = Literal[
    "openai_native",
    "anthropic_native",
    "google_native",
    "litellm_compat",
]
ProviderTier = Literal["primary", "more"]


@dataclass(frozen=True)
class Provider:
    """One inference provider entry.

    ``id`` is the canonical, lower-case slug used everywhere — as the dict
    key in ``PROVIDERS``, as the prefix in qualified model ids
    (``<id>:<raw_model_id>``), as the dict key in ``ss.provider_keys`` /
    ``ss.clients`` / ``ss.provider_models`` / ``ss.connect_errors``, and
    as the LiteLLM ``litellm_provider`` field for the cross-provider
    pricing-substitution check.

    ``litellm_prefix`` follows LiteLLM's own naming convention:

    - OpenAI / Anthropic / W&B's "wandb/..." entries are bare (no prefix).
    - Together is ``"together_ai/"``, Groq is ``"groq/"``, Fireworks is
      ``"fireworks_ai/"``, Mistral is ``"mistral/"``, etc.

    The string is concatenated with the model's raw id at call time:
    ``model=f"{provider.litellm_prefix}{raw_id}"``. When LiteLLM ships
    no entry for a particular ``<prefix><raw_id>`` combination, the
    catalog falls through to other sources (curated > OpenRouter > the
    provider's own ``/v1/models``); pricing is never substituted across
    providers.

    ``base_url`` is consumed only by ``litellm_compat`` providers (passed
    as ``base_url=`` to ``litellm.completion``). Native providers leave
    it as the empty string because their SDKs hard-code their endpoints.

    ``key_url`` is rendered as an ``st.link_button`` in the provider's
    Settings card so users can jump to the provider's API-key console.

    ``extra_fields`` is a free-form dict for per-provider quirks. Today
    only W&B Inference uses it (``{"project": ""}``) for the team/project
    string the user can paste in addition to the API key.
    """

    id: str
    label: str
    kind: ProviderKind
    base_url: str = ""
    key_url: str = ""
    tier: ProviderTier = "primary"
    notes: str = ""
    litellm_prefix: str = ""
    extra_fields: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The 12-provider catalog.
#
# Tier rationale (5 primary + 7 more): the default Settings body shows
# the providers that together give you almost everything you'd want to
# do in this app with the smallest number of API keys to manage.
#
# Primary (5):
#
# - **W&B Inference** — first because it's the home provider AND
#   because Weave tracing only kicks in when W&B is connected,
#   regardless of which provider you use for inference. So even users
#   who plan to call OpenAI / Anthropic / Google directly want a W&B
#   key for observability.
# - **OpenAI**, **Anthropic**, **Google Gemini** — the three frontier
#   providers, each with native-SDK features the LiteLLM library
#   only passes through generically (Anthropic prompt caching is the
#   killer feature for an agent that re-sends a long system prompt
#   every turn).
# - **OpenRouter** — labeled as a marked-up gateway, but elevated to
#   the primary tier because it's the cheapest "everything else"
#   option for users who don't want to manage seven separate keys
#   for the open-model clouds. Together / Groq / Fireworks / etc.
#   models are all reachable through OpenRouter at a 5-10% premium.
#
# More (7): Together, Groq, Fireworks, Mistral, xAI, Cerebras,
# DeepInfra. All direct-host, no markup. Worth a direct key when you
# use that provider heavily; otherwise OpenRouter covers them.
# ---------------------------------------------------------------------------
PROVIDERS: dict[str, Provider] = {
    # -- Primary tier (5) ---------------------------------------------------
    "wandb": Provider(
        id="wandb",
        label="W&B Inference",
        kind="litellm_compat",
        base_url="https://api.inference.wandb.ai/v1",
        key_url="https://wandb.ai/settings",
        tier="primary",
        # LiteLLM uses ``wandb/...`` for the W&B Inference namespace.
        litellm_prefix="wandb/",
        # The optional ``team/project`` field the user pastes alongside the
        # API key — used for usage attribution and Weave tracing. The
        # widget reads this dict to decide whether to render the field.
        extra_fields={"project": ""},
    ),
    "openai": Provider(
        id="openai",
        label="OpenAI",
        kind="openai_native",
        # Native SDK hard-codes the endpoint; base_url left empty.
        key_url="https://platform.openai.com/api-keys",
        tier="primary",
        # OpenAI ids in LiteLLM are bare (``gpt-4o``, ``o3-mini``, etc.).
        litellm_prefix="",
    ),
    "anthropic": Provider(
        id="anthropic",
        label="Anthropic",
        kind="anthropic_native",
        key_url="https://console.anthropic.com/settings/keys",
        tier="primary",
        # Anthropic ids in LiteLLM are bare (``claude-3-5-sonnet-...``).
        litellm_prefix="",
    ),
    "gemini": Provider(
        id="gemini",
        label="Google Gemini",
        kind="google_native",
        key_url="https://aistudio.google.com/app/apikey",
        tier="primary",
        # LiteLLM ships Gemini entries under ``gemini/`` for the public
        # AI Studio endpoint (vs ``vertex_ai/`` for Vertex).
        litellm_prefix="gemini/",
    ),
    "openrouter": Provider(
        id="openrouter",
        label="OpenRouter",
        kind="litellm_compat",
        base_url="https://openrouter.ai/api/v1",
        key_url="https://openrouter.ai/keys",
        tier="primary",
        # The Settings card surfaces this string as a caption next to a
        # ``:gray-badge[marked-up gateway]`` so users know what they're
        # signing up for. Every other provider's ``notes`` stays empty —
        # it's an OpenRouter-specific affordance.
        notes=(
            "OpenRouter adds ~5–10% over native rates. Pick this when you "
            "want one key that reaches hundreds of models. For popular "
            "models you use heavily, prefer the provider's own card under "
            "**More providers** for native pricing."
        ),
        litellm_prefix="openrouter/",
    ),
    # -- More tier (7) ------------------------------------------------------
    "together": Provider(
        id="together",
        label="Together AI",
        kind="litellm_compat",
        base_url="https://api.together.xyz/v1",
        key_url="https://api.together.xyz/settings/api-keys",
        tier="more",
        litellm_prefix="together_ai/",
    ),
    "groq": Provider(
        id="groq",
        label="Groq",
        kind="litellm_compat",
        base_url="https://api.groq.com/openai/v1",
        key_url="https://console.groq.com/keys",
        tier="more",
        litellm_prefix="groq/",
    ),
    "fireworks": Provider(
        id="fireworks",
        label="Fireworks AI",
        kind="litellm_compat",
        base_url="https://api.fireworks.ai/inference/v1",
        key_url="https://fireworks.ai/account/api-keys",
        tier="more",
        litellm_prefix="fireworks_ai/",
    ),
    "mistral": Provider(
        id="mistral",
        label="Mistral",
        kind="litellm_compat",
        base_url="https://api.mistral.ai/v1",
        key_url="https://console.mistral.ai/api-keys",
        tier="more",
        litellm_prefix="mistral/",
    ),
    "xai": Provider(
        id="xai",
        label="xAI (Grok)",
        kind="litellm_compat",
        base_url="https://api.x.ai/v1",
        key_url="https://console.x.ai/",
        tier="more",
        litellm_prefix="xai/",
    ),
    "cerebras": Provider(
        id="cerebras",
        label="Cerebras",
        kind="litellm_compat",
        base_url="https://api.cerebras.ai/v1",
        key_url="https://cloud.cerebras.ai/platform/",
        tier="more",
        litellm_prefix="cerebras/",
    ),
    "deepinfra": Provider(
        id="deepinfra",
        label="DeepInfra",
        kind="litellm_compat",
        base_url="https://api.deepinfra.com/v1/openai",
        key_url="https://deepinfra.com/dash/api_keys",
        tier="more",
        litellm_prefix="deepinfra/",
    ),
}


def get_provider(provider_id: str) -> Provider | None:
    """Return the :class:`Provider` for ``provider_id`` or ``None`` if unknown.

    This is the standard read accessor used everywhere outside the
    catalog itself; consumers should never reach into ``PROVIDERS``
    directly with ``[]`` because that would raise on ids loaded from a
    forward-compatible chat file (e.g. a chat that recorded a provider
    we've since renamed). ``None`` is the explicit "we don't know that
    provider" signal.
    """
    return PROVIDERS.get(provider_id)


def primary_providers() -> list[Provider]:
    """Return the 7 ``tier == "primary"`` providers, in catalog order."""
    return [p for p in PROVIDERS.values() if p.tier == "primary"]


def more_providers() -> list[Provider]:
    """Return the 5 ``tier == "more"`` providers, in catalog order."""
    return [p for p in PROVIDERS.values() if p.tier == "more"]


def make_provider_client(
    provider_id: str,
    api_key: str,
    **extra: Any,
) -> Any | None:
    """Build the persistent client object for ``provider_id`` if any.

    Returns:

    - For ``openai_native``: an ``openai.OpenAI(api_key=...)`` configured
      direct to ``api.openai.com`` (no ``base_url`` override) so the typed
      ``client.beta.chat.completions.parse(...)`` surface works as
      published by OpenAI.
    - For ``anthropic_native``: an ``anthropic.Anthropic(api_key=...)``.
    - For ``google_native``: a ``google.genai.Client(api_key=...)``.
    - For ``litellm_compat``: ``None``. LiteLLM is stateless — calls
      happen via ``litellm.completion(model=..., api_key=..., base_url=...)``
      with no persistent client object. ``ss.clients[provider_id]``
      holds ``None`` for these and the API key is read from
      ``ss.provider_keys[provider_id]`` at call time.

    ``extra`` is reserved for per-provider quirks. Today nothing in
    ``extra_fields`` flows through to native client construction (W&B's
    ``project`` is forwarded by the LiteLLM call layer at request time,
    not at client-construction time).

    Raises:

    - ``ValueError`` if ``provider_id`` is not in :data:`PROVIDERS`.
    - ``ValueError`` if ``api_key`` is empty.
    - ``ImportError`` (re-raised) if a native SDK isn't installed in the
      environment. The deps in ``pyproject.toml`` cover this; the
      explicit re-raise is a safety net for misbuilt desktop bundles.
    """
    provider = PROVIDERS.get(provider_id)
    if provider is None:
        raise ValueError(f"Unknown provider id: {provider_id!r}")
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError(f"API key for {provider.label} is empty.")

    if provider.kind == "openai_native":
        # Lazy import so the module imports cleanly even if openai is
        # somehow missing at load time (it shouldn't be — pinned in
        # pyproject — but the same pattern keeps the import-time blast
        # radius small).
        from openai import OpenAI
        return OpenAI(api_key=api_key)

    if provider.kind == "anthropic_native":
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)

    if provider.kind == "google_native":
        from google import genai
        return genai.Client(api_key=api_key)

    if provider.kind == "litellm_compat":
        # Stateless — the call layer reads the API key from session
        # state on every request. Returning None makes the contract
        # explicit ("there is no persistent client to disconnect").
        return None

    # Defensive: an exhaustive Literal would catch this at typecheck
    # time, but a stray future ``kind`` value should land here loudly.
    raise ValueError(f"Unhandled provider kind: {provider.kind!r}")


__all__ = [
    "PROVIDERS",
    "Provider",
    "ProviderKind",
    "ProviderTier",
    "get_provider",
    "make_provider_client",
    "more_providers",
    "primary_providers",
]
