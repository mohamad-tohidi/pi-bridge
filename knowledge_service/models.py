from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class AgentCreateRequest(BaseModel):
    name: str
    system_prompt: str
    tool_types: List[str]
    behavior_config: Dict[str, Any] = Field(default_factory=dict)


class AgentUpdateRequest(BaseModel):
    system_prompt: Optional[str] = None
    tool_types: Optional[List[str]] = None
    behavior_config: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    name: str
    system_prompt: str
    tool_types: List[str]
    behavior_config: Dict[str, Any]


class AskRequest(BaseModel):
    agent_name: str
    query: str
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    response: str
    session_id: str
