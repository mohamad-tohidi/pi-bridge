"""PiSession: Python ↔ Pi Agent bridge via JSONL subprocess protocol."""

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Iterator

from .errors import BridgeError
from .types import (
    AgentEndEvent,
    CustomTool,
    ErrorEvent,
    Model,
    Provider,
    ResponseEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
)

# Path to the bridge server script (sibling of this package)
_BRIDGE_DIR = Path(__file__).parent.parent / "bridge"
_BRIDGE_SERVER = _BRIDGE_DIR / "server.mjs"

_PI_AGENT_PACKAGE = "@earendil-works/pi-coding-agent"


def _discover_pi_agent_base() -> str:
    env_base = os.environ.get("PI_AGENT_BASE")
    if env_base:
        return env_base

    try:
        npm_root = subprocess.check_output(
            ["npm", "root", "-g"],
            text=True,
        ).strip()
    except Exception:
        return ""

    return str(Path(npm_root) / _PI_AGENT_PACKAGE)


def _parse_event(raw: dict) -> ResponseEvent | None:
    """Convert a raw bridge JSON dict to a typed ResponseEvent, or None to skip."""
    t = raw.get("type")
    if t == "text_delta":
        return TextDeltaEvent(delta=raw["delta"])
    if t == "thinking_delta":
        return ThinkingDeltaEvent(delta=raw["delta"])
    if t == "tool_call":
        return ToolCallEvent(
            tool_call_id=raw["tool_call_id"],
            tool_name=raw["tool_name"],
            arguments=raw.get("arguments", {}),
        )
    if t == "tool_result":
        return ToolResultEvent(
            tool_call_id=raw["tool_call_id"],
            tool_name=raw["tool_name"],
            content=raw.get("content", ""),
            is_error=raw.get("is_error", False),
        )
    if t == "turn_end":
        return TurnEndEvent()
    if t == "agent_end":
        return AgentEndEvent(stop_reason=raw.get("stop_reason", "stop"))
    if t == "error":
        return ErrorEvent(message=raw.get("message", "unknown error"))
    return None  # skip unknown / internal events


class PiSession:
    """
    A live session with a Pi agent.

    Owns a single Node.js bridge subprocess for its lifetime.
    Multiple send() calls accumulate context in the underlying Pi AgentSession.
    """

    def __init__(
        self,
        provider: Provider,
        model: Model,
        cwd: str = ".",
        system_prompt: str = "",
        tools: list[str] | None = None,
        custom_tools: list[CustomTool] | None = None,
        persist: bool = False,
        bridge_path: str = "",
    ):
        self._provider = provider
        self._model = model
        self._custom_tools: list[CustomTool] = custom_tools or []
        self._closed = False

        bridge_script = bridge_path or str(_BRIDGE_SERVER)

        env = os.environ.copy()
        env["PI_AGENT_BASE"] = _discover_pi_agent_base()

        self._proc = subprocess.Popen(
            ["node", bridge_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=os.path.abspath(cwd),
        )

        # Background thread to drain stderr (prevents deadlock)
        self._stderr_lines: list[str] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

        # Lock protecting stdin writes (used by send() and tool result callbacks)
        self._stdin_lock = threading.Lock()

        # Send init message
        self._write({
            "type": "init",
            "provider": {
                "base_url": provider.base_url,
                "api_key": provider.api_key,
            },
            "model": {
                "name": model.name,
                "api_format": model.api_format,
                "thinking": model.thinking,
            },
            "cwd": os.path.abspath(cwd),
            "system_prompt": system_prompt,
            "tools": tools,
            "custom_tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "prompt_snippet": t.prompt_snippet,
                    "prompt_guidelines": t.prompt_guidelines,
                }
                for t in self._custom_tools
            ],
            "persist": persist,
        })

        # Wait for ready
        ready = self._read_line()
        if ready is None:
            stderr = "\n".join(self._stderr_lines)
            raise BridgeError(f"Bridge process exited before ready. Stderr:\n{stderr}")
        if ready.get("type") == "error":
            raise ValueError(f"Bridge init error: {ready.get('message')}")
        if ready.get("type") != "ready":
            raise BridgeError(f"Expected ready, got: {ready}")

    # ------------------------------------------------------------------
    # Internal I/O helpers
    # ------------------------------------------------------------------

    def _write(self, obj: dict) -> None:
        """Write a JSON line to bridge stdin."""
        if self._closed:
            raise BridgeError("Session is closed")
        line = json.dumps(obj) + "\n"
        with self._stdin_lock:
            self._proc.stdin.write(line.encode())
            self._proc.stdin.flush()

    def _read_line(self) -> dict | None:
        """Read one JSON line from bridge stdout. Returns None on EOF."""
        while True:
            raw = self._proc.stdout.readline()
            if not raw:
                return None
            line = raw.decode().strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # Ignore malformed lines (shouldn't happen in normal operation)
                continue

    def _drain_stderr(self) -> None:
        for line in self._proc.stderr:
            self._stderr_lines.append(line.decode().rstrip())

    def _check_alive(self) -> None:
        if self._closed:
            raise BridgeError("Session is closed")
        if self._proc.poll() is not None:
            stderr = "\n".join(self._stderr_lines)
            raise BridgeError(f"Bridge process has exited (code {self._proc.returncode}). Stderr:\n{stderr}")

    # ------------------------------------------------------------------
    # Custom tool dispatch (called inline from send_stream)
    # ------------------------------------------------------------------

    def _dispatch_tool_request(self, raw: dict) -> None:
        """
        Execute a custom tool call synchronously and send back the result.
        Called from within send_stream() when a tool_request event arrives.
        """
        tool_id = raw["id"]
        tool_name = raw["tool"]
        args = raw.get("args", {})

        # Find matching custom tool
        fn = None
        for t in self._custom_tools:
            if t.name == tool_name:
                fn = t.fn
                break

        if fn is None:
            self._write({
                "type": "tool_error",
                "id": tool_id,
                "message": f"No Python handler for tool: {tool_name}",
            })
            return

        try:
            result = fn(**args)
            self._write({
                "type": "tool_result",
                "id": tool_id,
                "content": str(result),
                "is_error": False,
            })
        except Exception as exc:
            self._write({
                "type": "tool_result",
                "id": tool_id,
                "content": str(exc),
                "is_error": True,
            })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_stream(self, message: str) -> Iterator[ResponseEvent]:
        """
        Stream events for one agent turn.
        Handles custom tool_request events inline.
        Yields ResponseEvents until agent_end is received.
        """
        self._check_alive()
        self._write({"type": "prompt", "message": message})

        while True:
            self._check_alive()
            raw = self._read_line()
            if raw is None:
                raise BridgeError("Bridge process closed stdout unexpectedly")

            # Custom tool forwarding
            if raw.get("type") == "tool_request":
                self._dispatch_tool_request(raw)
                continue

            # Response to a query command (get_messages / get_state) — skip
            if raw.get("type") == "response":
                continue

            event = _parse_event(raw)
            if event is None:
                continue

            yield event

            if isinstance(event, AgentEndEvent):
                break

    def send(self, message: str) -> list[ResponseEvent]:
        """Block until agent_end, return all events from this turn."""
        return list(self.send_stream(message))

    @property
    def messages(self) -> list[dict]:
        """Fetch full message history from the bridge."""
        self._check_alive()
        self._write({"type": "get_messages"})
        while True:
            raw = self._read_line()
            if raw is None:
                raise BridgeError("Bridge closed before get_messages response")
            if raw.get("type") == "response" and raw.get("command") == "get_messages":
                return raw.get("data", {}).get("messages", [])
            # Skip other events that may arrive (e.g. if called after send finishes)

    @property
    def state(self) -> dict:
        """Fetch current session state from the bridge."""
        self._check_alive()
        self._write({"type": "get_state"})
        while True:
            raw = self._read_line()
            if raw is None:
                raise BridgeError("Bridge closed before get_state response")
            if raw.get("type") == "response" and raw.get("command") == "get_state":
                return raw.get("data", {})

    def set_model(self, provider: Provider, model: Model) -> None:
        """Switch model and/or provider at runtime."""
        self._write({
            "type": "set_model",
            "provider": {
                "base_url": provider.base_url,
                "api_key": provider.api_key,
            },
            "model": {
                "name": model.name,
                "api_format": model.api_format,
                "thinking": model.thinking,
            },
        })

    def set_thinking_level(self, level: str) -> None:
        """Switch thinking level at runtime."""
        self._write({"type": "set_thinking_level", "level": level})

    def compact(self, instructions: str = "") -> None:
        """Manually trigger context compaction."""
        self._write({"type": "compact", "instructions": instructions})

    def abort(self) -> None:
        """Abort the current operation."""
        self._write({"type": "abort"})

    def close(self) -> None:
        """Shut down the bridge process."""
        if self._closed:
            return
        self._closed = True
        try:
            self._write({"type": "shutdown"})
        except Exception:
            pass
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        self._proc.wait(timeout=5)
