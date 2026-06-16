from __future__ import annotations

from typing import Dict, Optional, Tuple
import os
import uuid

from dotenv import load_dotenv; load_dotenv()

from pi_bridge.session import PiSession
from pi_bridge.types import Provider, Model, CustomTool
from .models import AgentResponse, AgentCreateRequest, AgentUpdateRequest, SessionInfo
from .tools import BUILTIN_TOOLS, make_sync, BUILTIN_TOOL_DEFINITIONS
from .storage import storage
from .tool_storage import dynamic_tool_storage, build_tool_callable
from .models import ToolStatus

API_KEY  = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_API_BASE", "")
MODEL_NAME = os.environ.get("OPENAI_MODEL", "")

DEFAULT_PROVIDER = Provider(base_url=BASE_URL, api_key=API_KEY)
DEFAULT_MODEL    = Model(name=MODEL_NAME, api_format="completion")


class AgentManager:
    def __init__(self):
        # session_id -> (PiSession, agent_name)
        self._sessions: Dict[str, Tuple[PiSession, str]] = {}

    # ------------------------------------------------------------------
    # Agent CRUD
    # ------------------------------------------------------------------

    def create_agent(self, request: AgentCreateRequest) -> AgentResponse:
        system_prompt = request.system_prompt
        if request.behavior_config.get("return_links"):
            system_prompt += (
                "\n\nIMPORTANT: When you use information from a tool, you MUST cite it. "
                "Use the format [[LINK:ID]] where ID is the ID in the tool output."
            )
        agent = AgentResponse(
            name=request.name,
            system_prompt=system_prompt,
            tool_types=request.tool_types,
            behavior_config=request.behavior_config,
        )
        storage.add_agent(agent)
        return agent

    def get_agent(self, name: str) -> Optional[AgentResponse]:
        return storage.get_agent(name)

    def list_agents(self) -> list[AgentResponse]:
        return storage.list_agents()

    def update_agent(self, name: str, request: AgentUpdateRequest) -> Optional[AgentResponse]:
        agent = storage.get_agent(name)
        if not agent:
            return None
        updated = AgentResponse(
            name=name,
            system_prompt=request.system_prompt   if request.system_prompt   is not None else agent.system_prompt,
            tool_types=request.tool_types          if request.tool_types      is not None else agent.tool_types,
            behavior_config=request.behavior_config if request.behavior_config is not None else agent.behavior_config,
        )
        storage.update_agent(name, updated)
        self._invalidate_sessions_for_agent(name)
        return updated

    def delete_agent(self, name: str) -> bool:
        self._invalidate_sessions_for_agent(name)
        return storage.delete_agent(name)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[SessionInfo]:
        return [
            SessionInfo(session_id=sid, agent_name=agent_name)
            for sid, (_, agent_name) in self._sessions.items()
        ]

    def get_or_create_session(self, agent_name: str, session_id: Optional[str]) -> Tuple[PiSession, str]:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id][0], session_id

        session = self._build_session(agent_name)
        sid = session_id or str(uuid.uuid4())

        if session_id:
            self._sessions[sid] = (session, agent_name)

        return session, sid

    def close_session(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        self._sessions[session_id][0].close()
        del self._sessions[session_id]
        return True

    def close_all(self):
        for sid in list(self._sessions.keys()):
            self.close_session(sid)

    def _invalidate_sessions_for_agent(self, agent_name: str):
        stale = [sid for sid, (_, name) in self._sessions.items() if name == agent_name]
        for sid in stale:
            self._sessions[sid][0].close()
            del self._sessions[sid]

    def invalidate_sessions_for_tool(self, tool_name: str):
        """Close all sessions whose agent uses the given tool."""
        stale = [
            sid for sid, (_, agent_name) in self._sessions.items()
            if tool_name in (storage.get_agent(agent_name) or AgentResponse(
                name="", system_prompt="", tool_types=[], behavior_config={}
            )).tool_types
        ]
        for sid in stale:
            self._sessions[sid][0].close()
            del self._sessions[sid]

    # ------------------------------------------------------------------
    # Internal: build session
    # ------------------------------------------------------------------

    def _build_session(self, agent_name: str) -> PiSession:
        agent = storage.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")

        custom_tools: list[CustomTool] = []

        for tool_type in agent.tool_types:
            # 1. built-in tool?
            if tool_type in BUILTIN_TOOLS:
                fn = make_sync(BUILTIN_TOOLS[tool_type])
                tool_def = next((t for t in BUILTIN_TOOL_DEFINITIONS if t["name"] == tool_type), None)
                if tool_def:
                    custom_tools.append(CustomTool(
                        name=tool_type,
                        description=tool_def["description"],
                        parameters=tool_def["parameters"],
                        fn=fn,
                    ))
                continue

            # 2. dynamic tool?
            dyn = dynamic_tool_storage.get(tool_type)
            if dyn and dyn.status == ToolStatus.valid:
                custom_tools.append(CustomTool(
                    name=dyn.name,
                    description=dyn.description,
                    parameters=dyn.parameters,
                    fn=build_tool_callable(dyn),
                ))

        return PiSession(
            provider=DEFAULT_PROVIDER,
            model=DEFAULT_MODEL,
            system_prompt=agent.system_prompt,
            custom_tools=custom_tools,
        )


agent_manager = AgentManager()
