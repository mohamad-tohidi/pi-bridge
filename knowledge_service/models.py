from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# LLMs
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    name:       str
    base_url:   str
    api_key:    str
    model_name: str
    api_format: str = "completion"  # completion | response | anthropic


class LLMCreateRequest(BaseModel):
    name:       str
    base_url:   str
    api_key:    str
    model_name: str
    api_format: str = "completion"


class LLMUpdateRequest(BaseModel):
    base_url:   Optional[str] = None
    api_key:    Optional[str] = None
    model_name: Optional[str] = None
    api_format: Optional[str] = None


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
    description: Optional[str]            = None
    parameters:  Optional[Dict[str, Any]] = None
    code:        Optional[str]            = None
    entry_point: Optional[str]            = None


# ---------------------------------------------------------------------------
# Built-in tool definition
# ---------------------------------------------------------------------------

class ToolDefinition(BaseModel):
    name:        str
    description: str
    parameters:  Dict[str, Any]


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

class Skill(BaseModel):
    name:          str
    description:   str
    content:       str              # markdown instructions injected into system prompt
    allowed_tools: List[str] = Field(default_factory=list)  # empty = all tools


class SkillCreateRequest(BaseModel):
    name:          str
    description:   str
    content:       str
    allowed_tools: List[str] = Field(default_factory=list)


class SkillUpdateRequest(BaseModel):
    description:   Optional[str]       = None
    content:       Optional[str]       = None
    allowed_tools: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class AgentCreateRequest(BaseModel):
    name:            str
    system_prompt:   str
    tool_types:      List[str]
    skill_names:     List[str] = Field(default_factory=list)
    llm_name:        Optional[str] = None  # None = use env default
    behavior_config: Dict[str, Any] = Field(default_factory=dict)


class AgentUpdateRequest(BaseModel):
    system_prompt:   Optional[str]           = None
    tool_types:      Optional[List[str]]      = None
    skill_names:     Optional[List[str]]      = None
    llm_name:        Optional[str]            = None  # set to "" to revert to default
    behavior_config: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    name:            str
    system_prompt:   str
    tool_types:      List[str]
    skill_names:     List[str] = Field(default_factory=list)
    llm_name:        Optional[str]  # None = using env default
    behavior_config: Dict[str, Any]


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class SessionInfo(BaseModel):
    session_id: str
    agent_name: str
    llm_name:   Optional[str]  # the LLM this session was built with


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
