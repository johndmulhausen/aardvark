"""Provider catalog and client factory for the multi-provider inference layer.

Single source of truth for which inference providers the app supports, what
kind of dispatch each one needs, and how to build a client for each.

The 7-entry ``PROVIDERS`` dict is split five ways by ``kind``:

- ``"openai_native"`` (1 provider): OpenAI. Uses the ``openai`` SDK direct
  to ``api.openai.com``. Wires native OpenAI features the OpenAI-compat
  branch can't model — ``reasoning_effort`` for the o-series (live
  per-chat toggle: low / medium / high) is the active one today; typed
  structured outputs via ``client.beta.chat.completions.parse(...)``,
  ``prediction`` for predicted outputs, and the Files / Batch / Vector
  Stores APIs are hooks-ready for future expansion. Identical wire
  format to the ``openai_compat`` providers below; the kind
  discriminator exists so OpenAI-specific kwargs land in one branch
  without affecting the other HTTP-compatible providers.
- ``"anthropic_native"`` (1 provider): Anthropic. Uses the ``anthropic``
  SDK so users get prompt caching via ``cache_control`` blocks
  (auto-applied to the system prompt by
  ``chat_streams._openai_messages_to_anthropic`` — ~90% repeat-cost
  savings on every turn after the first when the long system prompt is
  cached), extended thinking with typed thinking blocks, computer use /
  bash / text-editor pre-built tools, the Message Batches API (50% off
  async work), and ``client.messages.count_tokens(...)`` for pre-flight
  cost estimation.
- ``"google_native"`` (1 provider): Google Gemini. Uses the
  ``google-genai`` SDK so users get grounding with Google Search
  (per-chat opt-in toggle in the chat page), code-execution,
  ``client.caches.create(...)`` for context caching, and native Imagen /
  Veo image / video gen.
- ``"mistral_native"`` (1 provider): Mistral. Uses the ``mistralai`` SDK
  so users get Mistral's full surface — FIM completion, embeddings,
  classifiers, the agents API, and Magistral reasoning options —
  alongside chat completions. The chat path itself is OpenAI-shape on
  the wire, but the dispatch route stays separate so future
  Mistral-only kwargs can land in one branch without affecting other
  providers.
- ``"xai_native"`` (1 provider): xAI (Grok). Uses the ``openai`` SDK with
  ``base_url=https://api.x.ai/v1`` because xAI's REST endpoint is
  officially OpenAI-compatible (per docs.x.ai). The dedicated dispatch
  route is what unlocks the per-chat **Live Search** toggle (real-time
  web access at query time), passed through as
  ``extra_body={"search_parameters": {...}}`` on the OpenAI-SDK call.
  We deliberately do NOT pull in the gRPC-based ``xai-sdk`` package
  because the REST surface covers everything we need for chat + Live
  Search and avoids the gRPC / OpenTelemetry dependency tail.
- ``"openai_compat"`` (2 providers): W&B Inference, OpenRouter. Both
  expose OpenAI-shape ``/v1/chat/completions`` endpoints, so they share
  one code path: ``openai.OpenAI(base_url=provider.base_url,
  api_key=...)`` with ``client.chat.completions.create(stream=True,
  ...)`` and no provider-specific kwargs. W&B Inference is here because
  it's the home provider whose OSS model lineup is OpenAI-shape;
  OpenRouter is here because it's the marked-up gateway that aggregates
  hundreds of upstream models without exposing per-upstream native
  features.

The ``tier`` field drives the Settings page card layout:
``"primary"`` cards render in the default body of the Settings page;
``"more"`` cards render inside an ``st.expander("More providers")``.
OpenRouter sits in ``"primary"`` despite its 5–10% markup because a
single OpenRouter key reaches hundreds of models — the persistent
``:gray-badge[marked-up gateway]`` label + the ``notes`` caption keep
the trade-off visible without burying the option.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ProviderKind = Literal[
    "openai_native",
    "anthropic_native",
    "google_native",
    "mistral_native",
    "xai_native",
    "openai_compat",
]
ProviderTier = Literal["primary", "more"]


@dataclass(frozen=True)
class Provider:
    """One inference provider entry.

    ``id`` is the canonical, lower-case slug used everywhere — as the dict
    key in ``PROVIDERS``, as the prefix in qualified model ids
    (``<id>:<raw_model_id>``), and as the dict key in ``ss.provider_keys`` /
    ``ss.clients`` / ``ss.provider_models`` / ``ss.connect_errors``.

    ``base_url`` is consumed only by ``openai_compat`` providers (passed
    as ``base_url=`` to ``openai.OpenAI(...)``). Native providers leave
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
    extra_fields: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The 7-provider catalog (all primary tier).
#
# Every entry earns its top-of-page card on a different axis:
#
# - **W&B Inference** — first because it's the home provider AND
#   because Weave tracing only kicks in when W&B is connected,
#   regardless of which provider you use for inference. So even users
#   who plan to call OpenAI / Anthropic / Google directly want a W&B
#   key for observability.
# - **OpenAI**, **Anthropic**, **Google Gemini** — three frontier
#   providers, each with native-SDK feature wiring (OpenAI
#   ``reasoning_effort``, Anthropic auto-prompt-caching, Google
#   Search grounding) that OpenAI-compat HTTP can't model.
# - **Mistral**, **xAI** — two more frontier labs (they train and
#   release their own foundation models — Mistral Large, Codestral,
#   Grok-4, Grok 4 Fast). Worth a direct key for users who lean on
#   those labs' models heavily; OpenRouter also routes to both at a
#   5-10% markup.
# - **OpenRouter** — labeled as a marked-up gateway, but kept in the
#   primary tier because it's the cheapest "everything else" option
#   for users who don't want to manage a separate key per open-model
#   lab. Llama / DeepSeek / Qwen / etc. are all reachable through
#   OpenRouter at a 5-10% premium.
#
# We do NOT list pure inference servers (Together, Groq, Fireworks,
# DeepInfra, Cerebras, etc.) because OpenRouter already routes to
# those backends and the direct-key value-add over OpenRouter is
# purely the markup — simpler to point users at OpenRouter than to
# maintain a card per inference cloud. If a future use case needs a
# specific inference cloud that OpenRouter doesn't carry, add a
# Provider entry here.
# ---------------------------------------------------------------------------
PROVIDERS: dict[str, Provider] = {
    # -- Primary tier (5) ---------------------------------------------------
    "wandb": Provider(
        id="wandb",
        label="W&B Inference",
        kind="openai_compat",
        base_url="https://api.inference.wandb.ai/v1",
        key_url="https://wandb.ai/settings",
        tier="primary",
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
    ),
    "anthropic": Provider(
        id="anthropic",
        label="Anthropic",
        kind="anthropic_native",
        key_url="https://console.anthropic.com/settings/keys",
        tier="primary",
    ),
    "gemini": Provider(
        id="gemini",
        label="Google Gemini",
        kind="google_native",
        key_url="https://aistudio.google.com/app/apikey",
        tier="primary",
    ),
    "openrouter": Provider(
        id="openrouter",
        label="OpenRouter",
        kind="openai_compat",
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
    ),
    "mistral": Provider(
        id="mistral",
        label="Mistral",
        kind="mistral_native",
        # Native ``mistralai`` SDK hard-codes the endpoint internally;
        # ``base_url`` is documented for reference only and not used at
        # client-construction time.
        base_url="https://api.mistral.ai/v1",
        key_url="https://console.mistral.ai/api-keys",
        tier="primary",
    ),
    "xai": Provider(
        id="xai",
        label="xAI (Grok)",
        kind="xai_native",
        # Used at client construction time — xAI's REST endpoint is
        # OpenAI-compatible per docs.x.ai, and ``make_provider_client``
        # builds an ``openai.OpenAI(base_url=...)`` against this URL.
        # The native dispatch path adds Live Search via ``extra_body``
        # without affecting the openai_compat 2.
        base_url="https://api.x.ai/v1",
        key_url="https://console.x.ai/",
        tier="primary",
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
    """Return the ``tier == "primary"`` providers, in catalog order."""
    return [p for p in PROVIDERS.values() if p.tier == "primary"]


def more_providers() -> list[Provider]:
    """Return the ``tier == "more"`` providers, in catalog order."""
    return [p for p in PROVIDERS.values() if p.tier == "more"]


def make_provider_client(
    provider_id: str,
    api_key: str,
    **extra: Any,
) -> Any:
    """Build the persistent client object for ``provider_id``.

    Returns:

    - For ``openai_native``: an ``openai.OpenAI(api_key=...)`` configured
      direct to ``api.openai.com`` (no ``base_url`` override) so the typed
      ``client.beta.chat.completions.parse(...)`` surface works as
      published by OpenAI.
    - For ``anthropic_native``: an ``anthropic.Anthropic(api_key=...)``.
    - For ``google_native``: a ``google.genai.Client(api_key=...)``.
    - For ``mistral_native``: a ``mistralai.Mistral(api_key=...)``.
    - For ``xai_native``: an ``openai.OpenAI(base_url="https://api.x.ai/v1",
      api_key=...)``. xAI's REST endpoint is officially OpenAI-compatible
      per docs.x.ai, so we reuse the OpenAI SDK; the dispatch route is
      kept separate from ``openai_compat`` so xAI-specific kwargs
      (Live Search ``search_parameters``, future reasoning kwargs)
      can land in one branch without affecting other providers.
    - For ``openai_compat``: an ``openai.OpenAI(base_url=provider.base_url,
      api_key=...)`` pointed at the provider's OpenAI-shape
      ``/v1/chat/completions`` endpoint. Same SDK as ``openai_native``,
      different ``base_url`` per provider — currently W&B Inference and
      OpenRouter.

    ``extra`` is reserved for per-provider quirks. Today nothing in
    ``extra_fields`` flows through to client construction (W&B's
    ``project`` is forwarded to Weave init at connect time, not to the
    OpenAI client constructor).

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

    if provider.kind == "openai_compat":
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=provider.base_url)

    if provider.kind == "anthropic_native":
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)

    if provider.kind == "google_native":
        from google import genai
        return genai.Client(api_key=api_key)

    if provider.kind == "mistral_native":
        # Lazy import — keeps the rest of the app importable even if
        # ``mistralai`` isn't installed (e.g. lean dev shells). The
        # SDK exposes its main client at ``mistralai.client.Mistral``;
        # the top-level ``mistralai`` package has no ``__init__``
        # re-export so the dotted import path is required.
        from mistralai.client import Mistral
        return Mistral(api_key=api_key)

    if provider.kind == "xai_native":
        # Use the OpenAI SDK against xAI's OpenAI-compatible REST
        # endpoint. The dedicated dispatch path in ``chat_streams``
        # adds Live Search via ``extra_body``; the gRPC ``xai-sdk`` is
        # not used — the REST surface covers everything we need.
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=provider.base_url)

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
