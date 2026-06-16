from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class AgentCreateRequest(BaseModel):
    name: str
    system_prompt: str
    tool_types: List[str]  # e.g., ["parsa", "haditha"]
    behavior_config: Dict[str, Any] = Field(default_factory=dict)

class AgentResponse(BaseModel):
    name: str
    system_prompt: str
    tool_types: List[str]
    behavior_config: Dict[str, Any]

class AskRequest(BaseModel):
    agent_name: str
    query: str

class AskResponse(BaseModel):
    response: str
