from typing import Dict
from .models import AgentResponse

class AgentStorage:
    def __init__(self):
        self._agents: Dict[str, AgentResponse] = {}

    def add_agent(self, agent: AgentResponse):
        self._agents[agent.name] = agent

    def get_agent(self, name: str) -> AgentResponse | None:
        return self._agents.get(name)

    def list_agents(self) -> list[AgentResponse]:
        return list(self._agents.values())

# Singleton instance
storage = AgentStorage()
