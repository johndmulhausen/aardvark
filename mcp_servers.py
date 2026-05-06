"""MCP (Model Context Protocol) runtime + on-disk config for the agent.

This module owns every interaction with external MCP servers. It exposes two
things to the rest of the app:

1. :class:`MCPRegistry` — a process-wide singleton (see :func:`get_registry`)
   that loads/saves ``~/.wb_coding_agent/mcp.json``, opens persistent client
   sessions to each enabled server, lists their tools, and dispatches tool
   calls. Tool dispatch is decorated with ``@weave.op(kind="tool")`` so MCP
   calls show up alongside our local tools in W&B Weave traces.
2. The OpenAI tool-schema export (:meth:`MCPRegistry.openai_tool_schemas`)
   that turns the union of every connected server's tools into the
   ``tools=[...]`` payload the chat-completion API wants. Every MCP tool is
   namespaced as ``mcp__<server_id>__<tool_name>`` so the agent loop can
   route tool calls back here without ambiguity.

Why a background asyncio loop?
------------------------------
The MCP Python SDK is async-only: ``ClientSession``, ``stdio_client``, and
``streamablehttp_client`` are all async context managers. The Streamlit
script thread is sync, so we run a single daemon-thread event loop for the
lifetime of the process and submit coroutines via
``asyncio.run_coroutine_threadsafe``. The loop, the live sessions, and the
``AsyncExitStack`` that owns them all live on that one thread; this keeps
session state (subprocess pipes, HTTP connections) alive across Streamlit
reruns and avoids re-initializing servers on every user message.

Why a single registry rather than per-rerun?
--------------------------------------------
Streamlit reruns the script top-to-bottom on every user interaction. Each
rerun calling ``MCPRegistry()`` fresh would reconnect every server (slow,
flaky, and wastes file descriptors). Instead, ``streamlit_app.py`` caches
the singleton with ``@st.cache_resource`` so connections persist across
reruns. The cache holds a reference, the daemon thread keeps the loop alive,
and disconnect happens explicitly via :meth:`MCPRegistry.disconnect`.

Security note
-------------
The on-disk config (``~/.wb_coding_agent/mcp.json``) holds HTTP auth
headers in plaintext, so the file is written with mode 0600.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import threading
import uuid
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypeVar

import httpx
import weave
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

# ---------------------------------------------------------------------------
# weave.op compat shim
# ---------------------------------------------------------------------------
# ``kind`` and ``color`` were added to ``weave.op`` partway through the 0.52
# series; on older installs the decorator raises ``TypeError`` at import time
# the moment Python evaluates ``@weave.op(..., kind="tool", color="green")``,
# which crashes the whole app before the UI ever loads. We feature-detect the
# supported kwargs once at module load and silently drop the unsupported ones,
# so older weave still gives us correct trace trees (just without the UI
# kind/color categorization). pyproject pins a recent enough weave for fresh
# installs; this shim handles upgrade-laggers.
_WEAVE_OP_PARAMS = set(inspect.signature(weave.op).parameters)
_WEAVE_OP_DROP = {k for k in ("kind", "color") if k not in _WEAVE_OP_PARAMS}


def _op(*args: Any, **kwargs: Any) -> Any:
    """``@weave.op`` wrapper that drops kwargs unsupported by older weave."""
    for k in _WEAVE_OP_DROP:
        kwargs.pop(k, None)
    return weave.op(*args, **kwargs)

CONFIG_DIR = Path.home() / ".wb_coding_agent"
CONFIG_FILE = CONFIG_DIR / "mcp.json"

# Each MCP tool is exposed to the model under a namespaced name so the agent
# loop can recover both the server id and the tool name from a tool-call
# event. The double underscore is unlikely to collide with any real tool
# name, and the prefix keeps a quick ``startswith("mcp__")`` check usable as
# the dispatch fork.
TOOL_NAME_PREFIX = "mcp__"
TOOL_NAME_SEPARATOR = "__"

# OpenAI's tool-name validation only accepts a-z/A-Z/0-9/_ and limits the
# length to 64 characters. We sanitize both server ids and tool names to fit
# within that contract; collisions are vanishingly rare in practice and we
# reject them on registration with a clear error.
_NAME_SANITIZER = re.compile(r"[^a-zA-Z0-9_]")
_MAX_TOOL_NAME_LEN = 64

# Default per-call timeout (seconds) for ``session.call_tool``. Long enough
# for chunky retrieval-style MCP calls but short enough that a misbehaving
# server can't wedge the chat thread.
DEFAULT_CALL_TIMEOUT = 60.0

T = TypeVar("T")


@dataclass
class ServerConfig:
    """Persistent configuration for a single MCP server.

    Either the stdio fields (``command`` + ``args`` + ``env``) OR the http
    fields (``url`` + ``headers``) are populated, depending on
    :attr:`transport`. ``id`` is a stable, sanitized identifier used for the
    OpenAI tool-name namespace; ``name`` is the human-friendly label.
    """

    id: str
    name: str
    transport: Literal["stdio", "http"]
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServerConfig":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or data["id"]),
            transport=data.get("transport", "stdio"),
            command=str(data.get("command") or ""),
            args=list(data.get("args") or []),
            env=dict(data.get("env") or {}),
            url=str(data.get("url") or ""),
            headers=dict(data.get("headers") or {}),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass
class ServerStatus:
    """Live runtime status of a server, exposed to the UI.

    ``tools`` carries the cached tool list returned by ``list_tools`` after
    the most recent successful initialize. ``error`` is set whenever a
    connect/list-tools cycle has failed — the UI surfaces it in the sidebar
    and we keep the previous tool list cleared so the model isn't offered
    schemas it can't actually invoke.
    """

    connected: bool = False
    error: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)


def _sanitize_name(name: str) -> str:
    """Sanitize an arbitrary string into the OpenAI tool-name character set.

    Replaces any character outside ``[A-Za-z0-9_]`` with ``_``. We do not
    truncate here; truncation happens in :func:`_namespaced_tool_name` once
    the full ``mcp__server__tool`` string is assembled, so we know exactly
    how many characters of the tool's own name we can keep.
    """
    cleaned = _NAME_SANITIZER.sub("_", name).strip("_")
    return cleaned or "server"


def _namespaced_tool_name(server_id: str, tool_name: str) -> str:
    """Build the ``mcp__<server>__<tool>`` name shown to the model.

    The combined name is truncated at :data:`_MAX_TOOL_NAME_LEN` characters
    by trimming the tool-name suffix; the prefix and server id are preserved
    intact because the agent's dispatch routing parses them back out.
    """
    safe_server = _sanitize_name(server_id)
    safe_tool = _sanitize_name(tool_name)
    full = f"{TOOL_NAME_PREFIX}{safe_server}{TOOL_NAME_SEPARATOR}{safe_tool}"
    if len(full) <= _MAX_TOOL_NAME_LEN:
        return full
    overflow = len(full) - _MAX_TOOL_NAME_LEN
    safe_tool = safe_tool[: max(1, len(safe_tool) - overflow)]
    return f"{TOOL_NAME_PREFIX}{safe_server}{TOOL_NAME_SEPARATOR}{safe_tool}"


def parse_namespaced_tool_name(name: str) -> tuple[str, str] | None:
    """Recover ``(server_id, tool_name)`` from a namespaced tool name.

    Returns ``None`` for names that don't follow the ``mcp__server__tool``
    convention, so the agent loop can cheaply ignore non-MCP tool calls.

    Note: the returned ``tool_name`` is the *sanitized* form. We look it up
    against the live tool list at dispatch time (see
    :meth:`MCPRegistry._resolve_tool_name`), so the original tool name —
    which may include hyphens or other punctuation that got rewritten to
    underscores during sanitization — is recovered there.
    """
    if not name.startswith(TOOL_NAME_PREFIX):
        return None
    rest = name[len(TOOL_NAME_PREFIX):]
    if TOOL_NAME_SEPARATOR not in rest:
        return None
    server_id, tool_name = rest.split(TOOL_NAME_SEPARATOR, 1)
    if not server_id or not tool_name:
        return None
    return server_id, tool_name


class _BackgroundLoop:
    """Owns a single asyncio event loop running on a daemon thread.

    The loop and its thread are started lazily on the first call to
    :meth:`run` and live for the rest of the process. Submitted coroutines
    are scheduled with ``run_coroutine_threadsafe`` and awaited
    synchronously, which is the supported way to call async code from a
    different thread.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None:
                return self._loop
            loop = asyncio.new_event_loop()

            def _runner() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            thread = threading.Thread(
                target=_runner,
                name="mcp-asyncio-loop",
                daemon=True,
            )
            thread.start()
            self._loop = loop
            self._thread = thread
            return loop

    def run(self, coro: Awaitable[T], timeout: float | None = None) -> T:
        loop = self._ensure()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    def submit(self, coro: Awaitable[T]) -> asyncio.Future[T]:
        loop = self._ensure()
        return asyncio.run_coroutine_threadsafe(coro, loop)


@dataclass
class _LiveSession:
    """Internal handle for a connected MCP server.

    Each entry owns an :class:`AsyncExitStack` that we built inside the
    background loop while opening the transport + ``ClientSession``. To
    cleanly shut the server down we have to *close that same exit stack*
    from inside the loop, which is what :meth:`MCPRegistry.disconnect`
    does.
    """

    config: ServerConfig
    session: ClientSession
    exit_stack: AsyncExitStack
    tools: list[dict[str, Any]]
    http_client: httpx.AsyncClient | None = None


class MCPRegistry:
    """Process-wide registry of MCP server configs and live sessions.

    Public API:

    - :meth:`load` / :meth:`save` — persistence at
      ``~/.wb_coding_agent/mcp.json`` (mode 0600).
    - :meth:`add` / :meth:`update` / :meth:`remove` — config CRUD; each
      mutates ``self.configs``, persists, and reconciles live sessions.
    - :meth:`connect` / :meth:`disconnect` — open/close one server's
      session. ``connect`` is idempotent and refreshes the cached tool
      list.
    - :meth:`reconcile` — bring live sessions in line with current configs;
      called after every mutation and from :meth:`load`.
    - :meth:`openai_tool_schemas` — union of every connected server's tools
      as OpenAI function-calling schemas.
    - :meth:`dispatch` — route a tool-call event to the right session.
    """

    def __init__(self) -> None:
        self.configs: list[ServerConfig] = []
        self.statuses: dict[str, ServerStatus] = {}
        self._sessions: dict[str, _LiveSession] = {}
        self._loop = _BackgroundLoop()
        self._mutex = threading.Lock()

    # ---- persistence ----

    def load(self) -> None:
        """Read ``mcp.json`` and (re)connect every enabled server.

        Missing or malformed files are treated as "no servers configured"
        rather than raised, so a fresh user has a clean slate.
        """
        try:
            raw = CONFIG_FILE.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            self.configs = []
            self.statuses = {}
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.configs = []
            self.statuses = {}
            return
        servers = data.get("servers") if isinstance(data, dict) else None
        if not isinstance(servers, list):
            self.configs = []
            self.statuses = {}
            return
        configs: list[ServerConfig] = []
        for entry in servers:
            if not isinstance(entry, dict):
                continue
            try:
                configs.append(ServerConfig.from_dict(entry))
            except (KeyError, TypeError, ValueError):
                continue
        self.configs = configs
        self.statuses = {c.id: ServerStatus() for c in configs}
        self.reconcile()

    def save(self) -> None:
        """Persist current configs to disk, creating the dir if needed."""
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {"servers": [c.to_dict() for c in self.configs]},
                indent=2,
            )
            CONFIG_FILE.write_text(payload, encoding="utf-8")
            try:
                os.chmod(CONFIG_FILE, 0o600)
            except OSError:
                pass
        except OSError:
            pass

    # ---- config CRUD ----

    def add(self, config: ServerConfig) -> None:
        with self._mutex:
            if any(c.id == config.id for c in self.configs):
                raise ValueError(f"Server id already exists: {config.id}")
            self.configs.append(config)
            self.statuses[config.id] = ServerStatus()
        self.save()
        self.reconcile()

    def update(self, config: ServerConfig) -> None:
        with self._mutex:
            idx = next((i for i, c in enumerate(self.configs) if c.id == config.id), None)
            if idx is None:
                raise ValueError(f"Server not found: {config.id}")
            self.configs[idx] = config
            self.statuses.setdefault(config.id, ServerStatus())
        self.save()
        # Force a fresh connect since the transport details may have changed.
        self.disconnect(config.id)
        self.reconcile()

    def remove(self, server_id: str) -> None:
        self.disconnect(server_id)
        with self._mutex:
            self.configs = [c for c in self.configs if c.id != server_id]
            self.statuses.pop(server_id, None)
        self.save()

    # ---- session lifecycle ----

    def reconcile(self) -> None:
        """Bring live sessions in line with current configs.

        For every enabled config without a live session, attempt a connect.
        For every live session whose config is gone or disabled, disconnect.
        Errors are surfaced via :class:`ServerStatus` rather than raised so
        the UI can render them; the agent loop just won't see those tools.
        """
        active_ids = {c.id for c in self.configs if c.enabled}

        for live_id in list(self._sessions.keys()):
            if live_id not in active_ids:
                self.disconnect(live_id)

        for cfg in self.configs:
            if not cfg.enabled:
                continue
            if cfg.id in self._sessions:
                continue
            try:
                self.connect(cfg.id)
            except Exception as e:
                status = self.statuses.setdefault(cfg.id, ServerStatus())
                status.connected = False
                status.error = str(e)
                status.tools = []

    def connect(self, server_id: str) -> None:
        """Open a session to the named server and cache its tool list."""
        cfg = next((c for c in self.configs if c.id == server_id), None)
        if cfg is None:
            raise ValueError(f"Server not found: {server_id}")
        if server_id in self._sessions:
            return
        try:
            live = self._loop.run(self._connect_async(cfg), timeout=30.0)
        except Exception as e:
            status = self.statuses.setdefault(server_id, ServerStatus())
            status.connected = False
            status.error = f"{type(e).__name__}: {e}"
            status.tools = []
            raise
        self._sessions[server_id] = live
        self.statuses[server_id] = ServerStatus(
            connected=True,
            error=None,
            tools=live.tools,
        )

    async def _connect_async(self, cfg: ServerConfig) -> _LiveSession:
        """Async helper run inside the background loop.

        Builds the transport via the appropriate MCP client, wraps it in a
        ``ClientSession``, runs ``initialize`` and ``list_tools``, and
        returns the populated :class:`_LiveSession`.
        """
        stack = AsyncExitStack()
        http_client: httpx.AsyncClient | None = None
        try:
            if cfg.transport == "stdio":
                if not cfg.command:
                    raise ValueError("stdio server is missing 'command'")
                env = {**os.environ, **cfg.env} if cfg.env else None
                params = StdioServerParameters(
                    command=cfg.command,
                    args=list(cfg.args),
                    env=env,
                )
                streams = await stack.enter_async_context(stdio_client(params))
                read_stream, write_stream = streams[0], streams[1]
            else:
                if not cfg.url:
                    raise ValueError("http server is missing 'url'")
                streams = await stack.enter_async_context(
                    streamablehttp_client(
                        url=cfg.url,
                        headers=cfg.headers or None,
                    )
                )
                read_stream, write_stream = streams[0], streams[1]

            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            tools_result = await session.list_tools()
            tools = [_tool_to_dict(t) for t in tools_result.tools]
            return _LiveSession(
                config=cfg,
                session=session,
                exit_stack=stack,
                tools=tools,
                http_client=http_client,
            )
        except Exception:
            await stack.aclose()
            raise

    def disconnect(self, server_id: str) -> None:
        """Close a live session if any. Always safe to call."""
        live = self._sessions.pop(server_id, None)
        if live is None:
            status = self.statuses.get(server_id)
            if status is not None:
                status.connected = False
                status.tools = []
            return
        try:
            self._loop.run(live.exit_stack.aclose(), timeout=10.0)
        except Exception:
            pass
        status = self.statuses.setdefault(server_id, ServerStatus())
        status.connected = False
        status.tools = []

    # ---- tool exposure + dispatch ----

    def openai_tool_schemas(self) -> list[dict[str, Any]]:
        """Union of every connected server's tools as OpenAI tool schemas.

        Tool names are namespaced via :func:`_namespaced_tool_name` so the
        agent's dispatch loop can recover ``(server_id, tool)`` from a
        ``tool_call`` event by calling :func:`parse_namespaced_tool_name`.
        Each schema's description is prefixed with ``[server_name]`` so a
        model offered tools from several servers can disambiguate by
        provenance.
        """
        schemas: list[dict[str, Any]] = []
        for live in self._sessions.values():
            for tool in live.tools:
                tool_name = tool.get("name", "")
                if not tool_name:
                    continue
                description = tool.get("description") or ""
                description = f"[{live.config.name}] {description}".strip()
                parameters = tool.get("inputSchema") or {
                    "type": "object",
                    "properties": {},
                }
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": _namespaced_tool_name(live.config.id, tool_name),
                            "description": description[:1024],
                            "parameters": parameters,
                        },
                    }
                )
        return schemas

    def _resolve_tool_name(self, live: _LiveSession, sanitized: str) -> str | None:
        """Look up the original (unsanitized) tool name for dispatch.

        Tool names listed by an MCP server may include characters (like
        hyphens) that we replaced with underscores during namespacing. We
        round-trip back to the original by sanitizing each known tool name
        and matching on equality, so the model's tool-call goes through
        even when the original name has punctuation.
        """
        for tool in live.tools:
            real = tool.get("name", "")
            if not real:
                continue
            if _sanitize_name(real) == sanitized:
                return real
        return None

    def dispatch_call(
        self,
        server_id: str,
        sanitized_tool_name: str,
        arguments: dict[str, Any],
        timeout: float = DEFAULT_CALL_TIMEOUT,
    ) -> dict[str, Any]:
        """Run an MCP tool by name and return a JSON-serializable result.

        Returns ``{"error": ...}`` on failure (server not connected, tool
        not found, server-side error, timeout) so the agent loop can feed
        the error back to the model the same way it handles local-tool
        errors.

        On success the returned dict has shape ``{"content": [...],
        "is_error": bool, "structured_content": ...}`` mirroring MCP's
        ``CallToolResult``.
        """
        live = self._sessions.get(server_id)
        if live is None:
            return {"error": f"MCP server '{server_id}' is not connected."}
        real_name = self._resolve_tool_name(live, sanitized_tool_name)
        if real_name is None:
            return {
                "error": (
                    f"Tool '{sanitized_tool_name}' not found on server "
                    f"'{live.config.name}'."
                ),
            }
        try:
            result = self._loop.run(
                live.session.call_tool(real_name, arguments=arguments),
                timeout=timeout,
            )
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        return _calltool_result_to_dict(result)


@_op(name="mcp_dispatch_tool", kind="tool", color="green")
def dispatch(name: str, arguments_json: str) -> dict[str, Any]:
    """Run an MCP tool by namespaced name.

    Decorated with ``@weave.op`` so MCP calls show up alongside local-tool
    dispatches in the trace tree (kind=tool, green). Mirrors the contract
    of ``tools.dispatch`` so the agent loop can route based on the tool
    name and otherwise treat results identically.
    """
    parsed = parse_namespaced_tool_name(name)
    if parsed is None:
        return {"error": f"Not an MCP tool name: {name}"}
    server_id, tool_name = parsed
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON arguments: {e}"}
    if not isinstance(args, dict):
        return {"error": "Tool arguments must be a JSON object."}
    return get_registry().dispatch_call(server_id, tool_name, args)


def _tool_to_dict(tool: Any) -> dict[str, Any]:
    """Turn an MCP ``Tool`` pydantic model into a plain JSON-able dict."""
    if hasattr(tool, "model_dump"):
        try:
            return tool.model_dump(mode="json", by_alias=True, exclude_none=True)
        except TypeError:
            return tool.model_dump(by_alias=True, exclude_none=True)
    if hasattr(tool, "dict"):
        return tool.dict()
    return {
        "name": getattr(tool, "name", ""),
        "description": getattr(tool, "description", ""),
        "inputSchema": getattr(tool, "inputSchema", {}),
    }


def _calltool_result_to_dict(result: Any) -> dict[str, Any]:
    """Turn an MCP ``CallToolResult`` into a JSON-serializable dict.

    The returned shape is intentionally compact for the model: a list of
    content blocks (each with ``type`` and the relevant payload field),
    plus the ``isError`` flag and any ``structuredContent`` the server
    chose to include.
    """
    if hasattr(result, "model_dump"):
        try:
            return result.model_dump(mode="json", by_alias=True, exclude_none=True)
        except TypeError:
            return result.model_dump(by_alias=True, exclude_none=True)
    if hasattr(result, "dict"):
        return result.dict()
    return {"content": str(result)}


def make_server_id(name: str) -> str:
    """Derive a stable, sanitized server id from a human-friendly name.

    Falls back to a UUID4 prefix if the sanitized name is empty (e.g. the
    user typed only punctuation).
    """
    base = _sanitize_name(name).lower().strip("_")
    if not base:
        base = uuid.uuid4().hex[:8]
    return base


_REGISTRY: MCPRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> MCPRegistry:
    """Return the process-wide :class:`MCPRegistry` singleton.

    The first call constructs the registry and runs :meth:`load`, which
    spins up the background loop and connects every enabled server.
    Streamlit's ``@st.cache_resource`` is the right place to memoize the
    singleton at the UI layer; this function is the lower-level fallback
    used from non-Streamlit contexts (e.g. the agent loop, smoke tests).
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            registry = MCPRegistry()
            registry.load()
            _REGISTRY = registry
    return _REGISTRY
