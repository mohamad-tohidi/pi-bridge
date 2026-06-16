# pi-bridge

MY TODO's:
1- Add Hallucinate Detector.
2- Add Skills. 


---
Python wrapper for the [Pi Agent SDK](https://github.com/earendil-works/pi). Pi is an autonomous coding agent that reads, edits, and runs code in a working directory using tools like `bash`, `read`, and `edit`. pi-bridge lets you drive it from Python via a local Node.js bridge process.

## Dependencies

- **Node.js** ≥ 18
- **Pi Agent** installed globally:
  ```bash
  npm install -g @earendil-works/pi-coding-agent
  ```
- **Python** ≥ 3.11

## Installation

```bash
git clone <this-repo>
cd pi-bridge
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```python
from pi_bridge import PiSession, Provider, Model

session = PiSession(
    provider=Provider(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-...",
    ),
    model=Model(
        name="deepseek-chat",
        api_format="completion",  # "completion" | "response" | "anthropic"
    ),
    cwd="/your/project",          # working directory for the agent
)

# Send a message and get all events at once
events = session.send("列出当前目录的文件")
for e in events:
    if e.type == "text_delta":
        print(e.delta, end="", flush=True)

# Or stream events one by one
for e in session.send_stream("解释一下 main.py"):
    if e.type == "text_delta":
        print(e.delta, end="", flush=True)

session.close()
```

Multiple `send()` calls on the same session share context — the agent remembers the full conversation history automatically.

## Custom tools

```python
from pi_bridge import PiSession, Provider, Model, CustomTool

def web_search(query: str) -> str:
    return "..."  # your implementation

session = PiSession(
    provider=Provider(...),
    model=Model(...),
    custom_tools=[
        CustomTool(
            name="web_search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            fn=web_search,
        )
    ],
)
```

## API reference

### Input types

These are the types you construct and pass into `PiSession`.

#### `Provider`

| Field | Type | Description |
|-------|------|-------------|
| `base_url` | `str` | API base URL, e.g. `"https://api.anthropic.com/v1"` |
| `api_key` | `str` | API key (default: `""`) |

Known hosts are mapped to their Pi provider name automatically. Any other hostname is used as-is.

#### `Model`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Model ID, e.g. `"claude-sonnet-4-5"`, `"deepseek-chat"` |
| `api_format` | `str` | API call format (see table below) |
| `thinking` | `str \| None` | Thinking level: `None` / `"off"` / `"minimal"` / `"low"` / `"medium"` / `"high"` / `"xhigh"` |

| `api_format` | For |
|--------------|-----|
| `"anthropic"` | Anthropic Claude |
| `"completion"` | OpenAI Chat Completions, DeepSeek, Groq, … |
| `"response"` | OpenAI Responses API |

#### `CustomTool`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Tool name (letters, digits, underscores only) |
| `description` | `str` | Shown to the model to decide when to call the tool |
| `parameters` | `dict` | JSON Schema object describing the tool's arguments |
| `fn` | `Callable` | Python function to call; receives keyword args matching the schema |

`fn` should return a string. Raised exceptions are caught and sent back to the agent as an error result.

---

### `PiSession`

```python
PiSession(
    provider: Provider,
    model: Model,
    cwd: str = ".",
    system_prompt: str = "",
    tools: list[str] | None = None,
    custom_tools: list[CustomTool] | None = None,
    persist: bool = False,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `provider` | — | API endpoint and credentials |
| `model` | — | Model to use |
| `cwd` | `"."` | Working directory for the agent |
| `system_prompt` | `""` | Override the default system prompt |
| `tools` | `None` | Built-in tool allowlist; `None` = Pi defaults (`read`, `bash`, `edit`, `write`), `[]` = none, explicit list = only those tools. All available names: `read`, `bash`, `edit`, `write`, `grep`, `find`, `ls` |
| `custom_tools` | `None` | Python-side tools exposed to the agent |
| `persist` | `False` | Persist session history to disk under `cwd/.pi/` |

**Methods**

| Method | Returns | Description |
|--------|---------|-------------|
| `send(msg)` | `list[ResponseEvent]` | Send a message, block until the agent finishes, return all events |
| `send_stream(msg)` | `Iterator[ResponseEvent]` | Same but yield events as they arrive |
| `set_model(provider, model)` | `None` | Hot-swap the model mid-session. Conversation history is preserved; new provider credentials take effect immediately. |
| `set_thinking_level(level)` | `None` | Change thinking intensity without switching models |
| `compact(instructions="")` | `None` | Manually trigger context compaction. `instructions` is an optional string telling the model what to focus on when summarizing. |
| `abort()` | `None` | Abort the current operation |
| `close()` | `None` | Shut down the bridge process |

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `messages` | `list[dict]` | Full conversation history. Each dict has a `role` field (`"user"`, `"assistant"`, `"tool_result"`) plus role-specific fields (see below). |
| `state` | `dict` | Current session state: model, message count, streaming flag, etc. |

`messages` entry shapes:

```python
{"role": "user", "content": "your message text"}
{"role": "assistant", "content": [{"type": "text", "text": "..."}], "stop_reason": "tool_use"}
{"role": "tool_result", "tool_call_id": "...", "tool_name": "...", "content": "...", "is_error": False}
```

---

### Response events

`send()` and `send_stream()` return/yield `ResponseEvent`, which is a union of the following:

| Type | Fields | Description |
|------|--------|-------------|
| `TextDeltaEvent` | `delta: str` | Incremental text chunk from the model |
| `ThinkingDeltaEvent` | `delta: str` | Incremental thinking/reasoning chunk |
| `ToolCallEvent` | `tool_call_id`, `tool_name`, `arguments` | Agent is invoking a tool |
| `ToolResultEvent` | `tool_call_id`, `tool_name`, `content`, `is_error` | Tool execution completed |
| `TurnEndEvent` | — | One LLM inference turn finished (may be multiple per `send()`) |
| `AgentEndEvent` | `stop_reason: str` | Agent finished the full response; last event in every `send()` |
| `ErrorEvent` | `message: str` | Something went wrong (API error, timeout, etc.) |

All events have a `type` field matching the class name in snake_case (e.g. `"text_delta"`, `"agent_end"`).

---

### Error handling

`ErrorEvent` is yielded for recoverable errors (e.g. API errors). The session remains usable afterwards.

`BridgeError` is raised as a Python exception when the bridge process crashes or the protocol breaks down. Once raised, the session is dead — create a new `PiSession`.

```python
from pi_bridge import BridgeError

try:
    events = session.send("...")
    for e in events:
        if e.type == "error":
            print("Agent error:", e.message)  # recoverable
        elif e.type == "text_delta":
            print(e.delta, end="")
except BridgeError as e:
    print("Bridge crashed:", e)  # session is dead
```
