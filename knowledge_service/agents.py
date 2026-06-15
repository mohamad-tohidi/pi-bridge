from typing import Dict, Optional, Iterator
import os

from dotenv import load_dotenv; load_dotenv()

from pi_bridge.session import PiSession
from pi_bridge.types import Provider, Model, CustomTool
from .models import AgentResponse, AgentCreateRequest
from .tools import TOOLS, get_tool_definitions
from .storage import storage
from .transformers import LinkEnforcementTransformer

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_API_BASE", "")
MODEL_NAME = os.environ.get("OPENAI_MODEL", "")

# Mocked provider and model for demonstration
DEFAULT_PROVIDER = Provider(base_url=BASE_URL, api_key=API_KEY)
DEFAULT_MODEL = Model(name=MODEL_NAME, api_format="completion")

class AgentManager:
    def __init__(self):
        self._sessions: Dict[str, PiSession] = {}

    def create_agent(self, request: AgentCreateRequest) -> AgentResponse:
        # Construct the system prompt based on behavior config
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

    def get_session(self, agent_name: str) -> PiSession:
        if agent_name in self._sessions:
            return self._sessions[agent_name]

        agent = storage.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} not found")

        custom_tools = []
        for tool_type in agent.tool_types:
            if tool_type in TOOLS:
                fn = TOOLS[tool_type]
                
                import asyncio
                import threading
                from concurrent.futures import ThreadPoolExecutor

                def sync_wrapper(**kwargs):
                    def run_async():
                        return asyncio.run(fn(**kwargs))
                    
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(run_async)
                        return future.result()

                tool_def = next((t for t in get_tool_definitions() if t["name"] == tool_type), None)
                if tool_def:
                    custom_tools.append(CustomTool(
                        name=tool_type,
                        description=tool_def["description"],
                        parameters=tool_def["parameters"],
                        fn=sync_wrapper
                    ))

        transformers = []
        if agent.behavior_config.get("return_links"):
            transformers.append(LinkEnforcementTransformer())

        session = PiSession(
            provider=DEFAULT_PROVIDER,
            model=DEFAULT_MODEL,
            system_prompt=agent.system_prompt,
            custom_tools=custom_tools,
            transformers=transformers
        )
        self._sessions[agent_name] = session
        return session

    def close_session(self, agent_name: str):
        if agent_name in self._sessions:
            self._sessions[agent_name].close()
            del self._sessions[agent_name]

    def close_all(self):
        for agent_name in list(self._sessions.keys()):
            self.close_session(agent_name)

agent_manager = AgentManager()
