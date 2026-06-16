from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class ToolStatus(str, Enum):
    pending = "pending"
    valid   = "valid"
    invalid = "invalid"


class DynamicTool(BaseModel):
    name:        str
    description: str
    parameters:  Dict[str, Any]
    code:        str
    entry_point: str = "run"
    status:      ToolStatus = ToolStatus.pending
    error:       Optional[str] = None


class ToolCreateRequest(BaseModel):
    name:        str
    description: str
    parameters:  Dict[str, Any]
    code:        str
    entry_point: str = "run"


class ToolUpdateRequest(BaseModel):
    description: Optional[str]         = None
    parameters:  Optional[Dict[str, Any]] = None
    code:        Optional[str]          = None
    entry_point: Optional[str]          = None


# ---------------------------------------------------------------------------
# Built-in tool definition (returned by /tools for hardcoded tools)
# ---------------------------------------------------------------------------

class ToolDefinition(BaseModel):
    name:        str
    description: str
    parameters:  Dict[str, Any]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class AgentCreateRequest(BaseModel):
    name:            str
    system_prompt:   str
    tool_types:      List[str]
    behavior_config: Dict[str, Any] = Field(default_factory=dict)


class AgentUpdateRequest(BaseModel):
    system_prompt:   Optional[str]            = None
    tool_types:      Optional[List[str]]       = None
    behavior_config: Optional[Dict[str, Any]]  = None


class AgentResponse(BaseModel):
    name:            str
    system_prompt:   str
    tool_types:      List[str]
    behavior_config: Dict[str, Any]


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class SessionInfo(BaseModel):
    session_id: str
    agent_name: str


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    agent_name: str
    query:      str
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    response:   str
    session_id: str
