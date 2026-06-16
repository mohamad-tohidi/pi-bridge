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

class AgentResponse(BaseModel):
    name: str
    system_prompt: str
    tool_types: List[str]
    behavior_config: Dict[str, Any]

class AskRequest(BaseModel):
    agent_name: str
    query: str
    session_id: Optional[str] = None  # None = stateless one-shot

class AskResponse(BaseModel):
    response: str
    session_id: str  # always returned so client can continue the conversation
