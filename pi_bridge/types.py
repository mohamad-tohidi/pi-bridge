from typing import Callable, Protocol, Iterator
from dataclasses import dataclass, field


@dataclass
class Provider:
    base_url: str
    api_key: str = ""


@dataclass
class Model:
    name: str
    api_format: str          # "completion" | "response" | "anthropic"
    thinking: str | None = None


@dataclass
class CustomTool:
    name: str
    description: str
    parameters: dict
    fn: Callable
    prompt_snippet: str | None = None
    prompt_guidelines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Response events
# ---------------------------------------------------------------------------

@dataclass
class TextDeltaEvent:
    delta: str
    type: str = "text_delta"


@dataclass
class ThinkingDeltaEvent:
    delta: str
    type: str = "thinking_delta"


@dataclass
class ToolCallEvent:
    tool_call_id: str
    tool_name: str
    arguments: dict
    type: str = "tool_call"


@dataclass
class ToolResultEvent:
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    type: str = "tool_result"


@dataclass
class TurnEndEvent:
    type: str = "turn_end"


@dataclass
class AgentEndEvent:
    stop_reason: str
    type: str = "agent_end"


@dataclass
class ErrorEvent:
    message: str
    type: str = "error"


ResponseEvent = (
    TextDeltaEvent
    | ThinkingDeltaEvent
    | ToolCallEvent
    | ToolResultEvent
    | TurnEndEvent
    | AgentEndEvent
    | ErrorEvent
)


class ResponseEventTransformer(Protocol):
    """A protocol for objects that can transform a stream of ResponseEvents."""
    def transform(self, event_iterator: Iterator[ResponseEvent]) -> Iterator[ResponseEvent]:
        ...
