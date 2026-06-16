from typing import Dict, Optional, Tuple
import os
import uuid

from dotenv import load_dotenv; load_dotenv()

from pi_bridge.session import PiSession
from pi_bridge.types import Provider, Model, CustomTool
from .models import AgentResponse, AgentCreateRequest
from .tools import TOOLS, get_tool_definitions
from .storage import storage

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_API_BASE", "")
MODEL_NAME = os.environ.get("OPENAI_MODEL", "")

DEFAULT_PROVIDER = Provider(base_url=BASE_URL, api_key=API_KEY)
DEFAULT_MODEL = Model(name=MODEL_NAME, api_format="completion")


class AgentManager:
    def __init__(self):
        # keyed by session_id, not agent name
        self._sessions: Dict[str, PiSession] = {}

    def create_agent(self, request: AgentCreateRequest) -> AgentResponse:
        system_prompt = request.system_prompt
        if request.behavior_config.get("return_links"):
            system_prompt += (
                "\n\nIMPORTANT: When you use information from a tool, you MUST cite it. "
                "Instead of writing the full URL, use the format [[LINK:ID]] where ID is the "
                "ID provided in the tool output (e.g., [[LINK:123]]). Do not include the actual URL in your text."
            )
        agent = AgentResponse(
            name=request.name,
            system_prompt=system_prompt,
            tool_types=request.tool_types,
            behavior_config=request.behavior_config
        )
        storage.add_agent(agent)
        return agent

    def get_agent(self, name: str) -> Optional[AgentResponse]:
        return storage.get_agent(name)

    def list_agents(self) -> list[AgentResponse]:
        return storage.list_agents()

    def _build_session(self, agent_name: str) -> PiSession:
        """Create a fresh PiSession for a given agent."""
        agent = storage.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} not found")

        custom_tools = []
        for tool_type in agent.tool_types:
            if tool_type in TOOLS:
                fn = TOOLS[tool_type]

                import asyncio
                from concurrent.futures import ThreadPoolExecutor

                def sync_wrapper(**kwargs):
                    def run_async():
                        return asyncio.run(fn(**kwargs))
                    with ThreadPoolExecutor() as executor:
                        return executor.submit(run_async).result()

                tool_def = next((t for t in get_tool_definitions() if t["name"] == tool_type), None)
                if tool_def:
                    custom_tools.append(CustomTool(
                        name=tool_type,
                        description=tool_def["description"],
                        parameters=tool_def["parameters"],
                        fn=sync_wrapper
                    ))

        return PiSession(
            provider=DEFAULT_PROVIDER,
            model=DEFAULT_MODEL,
            system_prompt=agent.system_prompt,
            custom_tools=custom_tools,
        )

    def get_or_create_session(self, agent_name: str, session_id: Optional[str]) -> Tuple[PiSession, str]:
        """
        - session_id=None  → one-shot: fresh session, not stored
        - session_id given → reuse if exists, else create and store under that id
        """
        if session_id and session_id in self._sessions:
            return self._sessions[session_id], session_id

        session = self._build_session(agent_name)
        sid = session_id or str(uuid.uuid4())

        if session_id:
            # caller explicitly provided an id → they want a persistent conversation
            self._sessions[sid] = session

        return session, sid

    def close_session(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].close()
            del self._sessions[session_id]

    def close_all(self):
        for sid in list(self._sessions.keys()):
            self.close_session(sid)


agent_manager = AgentManager()
