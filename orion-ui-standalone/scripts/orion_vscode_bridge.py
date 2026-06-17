"""Minimal MCP bridge that forwards VS Code chat prompts into Orion Forge.

This server is intentionally isolated from the main web app so it can be
enabled, disabled, or removed without touching Orion's core runtime.

Transport:
    MCP over stdio using newline-delimited JSON-RPC messages.

Tool:
    orion_chat

The tool accepts a prompt, optional context, and an optional agent override.
It forwards the request to /api/chat/send and returns the normalized response.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from collections.abc import Iterable
from typing import Any, Dict, Optional

import requests


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "vscode_bridge.json")
DEFAULT_ALLOWED_AGENTS = ["orion_cannon", "elysia_cannon", "k_os"]
RETRYABLE_MODEL_STATUS_CODES = frozenset({400, 404, 408, 409, 422, 429, 500, 502, 503, 504})

# Friendly aliases that can be used via model_override. The bridge will try
# candidates in order until one succeeds (or non-retryable failure occurs).
MODEL_ALIAS_CANDIDATES: dict[str, list[str]] = {
    "deepseek-reasoner": [
        "deepseek/deepseek-r1",
        "deepseek-reasoner",
    ],
    "gpt-5.2": [
        "openai/gpt-5.2",
        "gpt-5.2",
    ],
    "claude-sonnet-latest": [
        "anthropic/claude-sonnet-4-6",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4-20250514",
    ],
    "claude-opus-latest": [
        "anthropic/claude-opus-4-8",
        "claude-opus-4-8",
        "anthropic/claude-opus-4-6",
        "claude-opus-4-6",
    ],
}


@dataclasses.dataclass(frozen=True)
class BridgeConfig:
    base_url: str = "http://127.0.0.1:8989"
    default_agent: str = "orion_cannon"
    allowed_agents: tuple[str, ...] = tuple(DEFAULT_ALLOWED_AGENTS)
    timeout_seconds: int = 120
    default_mode: str = "chat"
    default_max_steps: int = 10
    default_burst_ticks: int = 3
    default_model_override: str = ""
    fallback_model_override: str = ""
    auth_mode: str = "disabled"
    cookie_name: str = "sb_access_token"
    cookie_value: str = ""
    bearer_token: str = ""
    bridge_key: str = ""


def _load_json_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read().strip()
    return json.loads(raw) if raw else {}


def _as_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _dedupe_models(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        model = _as_str(value, "")
        if not model:
            continue
        key = model.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(model)
    return out


def _expand_model_candidates(value: Any) -> list[str]:
    model = _as_str(value, "")
    if not model:
        return []
    candidates = MODEL_ALIAS_CANDIDATES.get(model.lower())
    if not candidates:
        return [model]
    return _dedupe_models([*candidates, model])


def load_config() -> BridgeConfig:
    file_config = _load_json_file(os.environ.get("ORION_VSCODE_BRIDGE_CONFIG", DEFAULT_CONFIG_PATH))
    auth_config = file_config.get("auth", {}) if isinstance(file_config.get("auth", {}), dict) else {}

    allowed_agents = file_config.get("allowed_agents", DEFAULT_ALLOWED_AGENTS)
    if not isinstance(allowed_agents, Iterable) or isinstance(allowed_agents, (str, bytes)):
        allowed_agents = DEFAULT_ALLOWED_AGENTS

    config = BridgeConfig(
        base_url=_as_str(os.environ.get("ORION_VSCODE_BRIDGE_BASE_URL") or file_config.get("base_url"), "http://127.0.0.1:8989"),
        default_agent=_as_str(os.environ.get("ORION_VSCODE_BRIDGE_DEFAULT_AGENT") or file_config.get("default_agent"), "orion_cannon"),
        allowed_agents=tuple(str(agent).strip() for agent in allowed_agents if str(agent).strip()),
        timeout_seconds=_as_int(os.environ.get("ORION_VSCODE_BRIDGE_TIMEOUT_SECONDS") or file_config.get("timeout_seconds"), 120),
        default_mode=_as_str(os.environ.get("ORION_VSCODE_BRIDGE_DEFAULT_MODE") or file_config.get("default_mode"), "chat"),
        default_max_steps=_as_int(os.environ.get("ORION_VSCODE_BRIDGE_DEFAULT_MAX_STEPS") or file_config.get("default_max_steps"), 10),
        default_burst_ticks=_as_int(os.environ.get("ORION_VSCODE_BRIDGE_DEFAULT_BURST_TICKS") or file_config.get("default_burst_ticks"), 3),
        default_model_override=_as_str(
            os.environ.get("ORION_VSCODE_BRIDGE_DEFAULT_MODEL") or file_config.get("default_model_override"),
            "",
        ),
        fallback_model_override=_as_str(
            os.environ.get("ORION_VSCODE_BRIDGE_FALLBACK_MODEL") or file_config.get("fallback_model_override"),
            "",
        ),
        auth_mode=_as_str(os.environ.get("ORION_VSCODE_BRIDGE_AUTH_MODE") or auth_config.get("mode"), "disabled"),
        cookie_name=_as_str(os.environ.get("ORION_VSCODE_BRIDGE_COOKIE_NAME") or auth_config.get("cookie_name"), "sb_access_token"),
        cookie_value=_as_str(os.environ.get("ORION_VSCODE_BRIDGE_COOKIE_VALUE") or auth_config.get("cookie_value"), ""),
        bearer_token=_as_str(os.environ.get("ORION_VSCODE_BRIDGE_BEARER_TOKEN") or auth_config.get("bearer_token"), ""),
        bridge_key=_as_str(os.environ.get("ORION_VSCODE_BRIDGE_KEY") or auth_config.get("bridge_key"), ""),
    )

    if config.default_agent not in config.allowed_agents:
        config = dataclasses.replace(config, default_agent=config.allowed_agents[0] if config.allowed_agents else config.default_agent)

    return config


def _make_headers(config: BridgeConfig) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.auth_mode == "cookie" and config.cookie_value:
        headers["Cookie"] = f"{config.cookie_name}={config.cookie_value}"
    elif config.auth_mode == "bearer" and config.bearer_token:
        headers["Authorization"] = f"Bearer {config.bearer_token}"
    elif config.auth_mode == "bridge_key" and config.bridge_key:
        headers["X-Bridge-Key"] = config.bridge_key
    return headers


def _normalize_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "response": payload.get("response", ""),
        "chat_id": payload.get("chat_id"),
        "model": payload.get("model"),
        "task_type": payload.get("task_type"),
        "router_tier": payload.get("router_tier"),
        "usage": payload.get("usage", {}),
        "cost": payload.get("cost", {}),
        "layers": payload.get("layers", []),
        "tool_calls": payload.get("tool_calls", []),
        "saved_memories": payload.get("saved_memories", []),
        "generated_image": payload.get("generated_image"),
        "generated_video": payload.get("generated_video"),
        "credits": payload.get("credits", {}),
    }


def _post_orion(config: BridgeConfig, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = config.base_url.rstrip("/") + "/api/chat/send"
    response = requests.post(url, headers=_make_headers(config), json=payload, timeout=config.timeout_seconds)
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Orion response was not a JSON object")
    return data


def call_orion(config: BridgeConfig, arguments: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _as_str(arguments.get("prompt"), "")
    if not prompt:
        raise ValueError("prompt is required")

    agent = _as_str(arguments.get("agent") or config.default_agent, config.default_agent)
    if agent not in config.allowed_agents:
        raise ValueError(f"agent '{agent}' is not allowed; allowed agents: {', '.join(config.allowed_agents)}")

    context = _as_str(arguments.get("context"), "")
    if context:
        stimulus = f"Context:\n{context}\n\nTask:\n{prompt}"
    else:
        stimulus = prompt

    requested_model = _as_str(arguments.get("model_override"), "")
    if requested_model:
        model_candidates: list[Optional[str]] = [*(_expand_model_candidates(requested_model) or [requested_model])]
    else:
        default_candidates = _expand_model_candidates(config.default_model_override)
        fallback_candidates = _expand_model_candidates(config.fallback_model_override)
        merged = _dedupe_models([*default_candidates, *fallback_candidates])
        model_candidates = [*merged] if merged else [None]

    payload = {
        "agent": agent,
        "stimulus": stimulus,
        "mode": _as_str(arguments.get("mode"), config.default_mode),
        "burst_ticks": _as_int(arguments.get("burst_ticks"), config.default_burst_ticks),
        "max_steps": _as_int(arguments.get("max_steps"), config.default_max_steps),
        "chat_id": arguments.get("chat_id"),
        "model_override": None,
    }

    last_http_error: Optional[requests.HTTPError] = None
    for index, candidate in enumerate(model_candidates):
        payload["model_override"] = candidate or None
        try:
            data = _post_orion(config, payload)
            normalized = _normalize_response(data)
            normalized["agent"] = agent
            normalized["request"] = {
                "mode": payload["mode"],
                "burst_ticks": payload["burst_ticks"],
                "max_steps": payload["max_steps"],
                "chat_id": payload["chat_id"],
                "model_override": payload["model_override"],
                "model_candidates": model_candidates,
            }
            normalized["model_fallback_used"] = index > 0
            return normalized
        except requests.HTTPError as exc:
            last_http_error = exc
            status_code = exc.response.status_code if exc.response is not None else 500
            has_more_candidates = index < len(model_candidates) - 1
            if has_more_candidates and status_code in RETRYABLE_MODEL_STATUS_CODES:
                continue
            raise

    if last_http_error is not None:
        raise last_http_error
    raise ValueError("No model candidates available")


def _tool_schema(config: BridgeConfig) -> Dict[str, Any]:
    return {
        "name": "orion_chat",
        "description": "Route a prompt through Orion Forge using Orion Cannon, Elysia Cannon, or K-OS.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Task or question to send to Orion Forge."},
                "agent": {
                    "type": "string",
                    "enum": list(config.allowed_agents),
                    "description": f"Persona to use. Defaults to {config.default_agent}.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional working context, code summary, or chat summary to prepend before the prompt.",
                },
                "chat_id": {"type": "string", "description": "Reuse an existing Orion chat thread."},
                "model_override": {
                    "type": "string",
                    "description": (
                        "Force a specific model for this request. "
                        "Supports aliases: deepseek-reasoner, gpt-5.2, "
                        "claude-sonnet-latest, claude-opus-latest."
                    ),
                },
                "mode": {"type": "string", "enum": ["chat", "burst"], "default": config.default_mode},
                "burst_ticks": {"type": "integer", "minimum": 1, "maximum": 25, "default": config.default_burst_ticks},
                "max_steps": {"type": "integer", "minimum": 1, "maximum": 50, "default": config.default_max_steps},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    }


def _json_rpc_result(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _json_rpc_error(request_id: Any, code: int, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    error: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _write_message(message: Dict[str, Any]) -> None:
    # MCP stdio transport uses newline-delimited JSON (one JSON object per
    # line, no embedded newlines). Write raw bytes + "\n" to avoid Windows
    # text-mode translating "\n" into "\r\n" and corrupting the framing.
    body = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(body + b"\n")
    sys.stdout.buffer.flush()


def _read_message() -> Optional[Dict[str, Any]]:
    # Read one newline-delimited JSON message. Blank lines are skipped and
    # malformed lines are ignored rather than crashing the server loop.
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue


def _tool_result_text(result: Dict[str, Any]) -> str:
    response_text = result.get("response") or ""
    meta = {
        "agent": result.get("agent"),
        "model": result.get("model"),
        "task_type": result.get("task_type"),
        "router_tier": result.get("router_tier"),
        "chat_id": result.get("chat_id"),
        "usage": result.get("usage", {}),
        "cost": result.get("cost", {}),
    }
    return json.dumps({"meta": meta, "response": response_text, "raw": result}, ensure_ascii=False, indent=2)


def serve() -> None:
    config = load_config()
    while True:
        message = _read_message()
        if message is None:
            return

        method = message.get("method")
        request_id = message.get("id")

        if method == "notifications/initialized":
            continue

        if method == "initialize":
            params = message.get("params", {}) if isinstance(message.get("params", {}), dict) else {}
            protocol_version = params.get("protocolVersion", "2024-11-05")
            result = {
                "protocolVersion": protocol_version,
                "serverInfo": {"name": "orion-vscode-bridge", "version": "1.0.0"},
                "capabilities": {
                    "tools": {
                        "listChanged": False,
                    }
                },
            }
            _write_message(_json_rpc_result(request_id, result))
            continue

        if method == "tools/list":
            _write_message(_json_rpc_result(request_id, {"tools": [_tool_schema(config)]}))
            continue

        if method == "tools/call":
            params = message.get("params", {}) if isinstance(message.get("params", {}), dict) else {}
            tool_name = params.get("name")
            arguments = params.get("arguments", {}) if isinstance(params.get("arguments", {}), dict) else {}
            if tool_name != "orion_chat":
                _write_message(_json_rpc_error(request_id, -32601, f"Unknown tool '{tool_name}'"))
                continue

            try:
                result = call_orion(config, arguments)
                _write_message(
                    _json_rpc_result(
                        request_id,
                        {
                            "content": [
                                {
                                    "type": "text",
                                    "text": _tool_result_text(result),
                                }
                            ],
                            "isError": False,
                        },
                    )
                )
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else 500
                detail: Dict[str, Any]
                try:
                    detail = exc.response.json() if exc.response is not None else {}
                except Exception:
                    detail = {"body": exc.response.text if exc.response is not None else ""}
                _write_message(
                    _json_rpc_error(
                        request_id,
                        -32000,
                        f"Orion request failed with HTTP {status_code}",
                        detail,
                    )
                )
            except (requests.RequestException, ValueError) as exc:
                _write_message(_json_rpc_error(request_id, -32000, str(exc)))
            continue

        if request_id is not None:
            _write_message(_json_rpc_error(request_id, -32601, f"Unknown method '{method}'"))


if __name__ == "__main__":
    try:
        serve()
    except KeyboardInterrupt:
        pass