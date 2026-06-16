from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

from .models import AgentResponse


def _to_dict(agent: AgentResponse) -> dict:
    if hasattr(agent, "model_dump"):
        return agent.model_dump()
    return vars(agent)


def _from_dict(data: dict) -> AgentResponse:
    return AgentResponse(**data)


class AgentBackend(ABC):
    @abstractmethod
    def add(self, agent: AgentResponse) -> None: ...

    @abstractmethod
    def get(self, name: str) -> AgentResponse | None: ...

    @abstractmethod
    def list_all(self) -> list[AgentResponse]: ...

    @abstractmethod
    def update(self, name: str, agent: AgentResponse) -> AgentResponse | None: ...

    @abstractmethod
    def delete(self, name: str) -> bool: ...


class SQLiteAgentBackend(AgentBackend):
    def __init__(self, db_path: str = "agents.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    name TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def add(self, agent: AgentResponse) -> None:
        data = _to_dict(agent)
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO agents (name, payload)
                VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET payload = excluded.payload
                """,
                (agent.name, json.dumps(data, ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, name: str) -> AgentResponse | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload FROM agents WHERE name = ?", (name,)
            ).fetchone()
            return _from_dict(json.loads(row[0])) if row else None
        finally:
            conn.close()

    def list_all(self) -> list[AgentResponse]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT payload FROM agents").fetchall()
            return [_from_dict(json.loads(row[0])) for row in rows]
        finally:
            conn.close()

    def update(self, name: str, agent: AgentResponse) -> AgentResponse | None:
        if not self.get(name):
            return None
        self.add(agent)
        return agent

    def delete(self, name: str) -> bool:
        conn = self._connect()
        try:
            cursor = conn.execute("DELETE FROM agents WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()


class JSONLAgentBackend(AgentBackend):
    def __init__(self, file_path: str = "agents.jsonl"):
        self.file_path = Path(file_path)
        self.file_path.touch(exist_ok=True)
        self._agents: Dict[str, AgentResponse] = {}
        self._load()

    def _load(self) -> None:
        self._agents.clear()
        with self.file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self._agents[_from_dict(data).name] = _from_dict(data)

    def _flush(self) -> None:
        with self.file_path.open("w", encoding="utf-8") as f:
            for agent in self._agents.values():
                f.write(json.dumps(_to_dict(agent), ensure_ascii=False) + "\n")

    def add(self, agent: AgentResponse) -> None:
        self._agents[agent.name] = agent
        self._flush()

    def get(self, name: str) -> AgentResponse | None:
        return self._agents.get(name)

    def list_all(self) -> list[AgentResponse]:
        return list(self._agents.values())

    def update(self, name: str, agent: AgentResponse) -> AgentResponse | None:
        if name not in self._agents:
            return None
        self._agents[name] = agent
        self._flush()
        return agent

    def delete(self, name: str) -> bool:
        if name not in self._agents:
            return False
        del self._agents[name]
        self._flush()
        return True


class AgentStorage:
    def __init__(self, backend: AgentBackend | None = None):
        self._backend = backend or SQLiteAgentBackend()

    def add_agent(self, agent: AgentResponse) -> None:
        self._backend.add(agent)

    def get_agent(self, name: str) -> AgentResponse | None:
        return self._backend.get(name)

    def list_agents(self) -> list[AgentResponse]:
        return self._backend.list_all()

    def update_agent(self, name: str, agent: AgentResponse) -> AgentResponse | None:
        return self._backend.update(name, agent)

    def delete_agent(self, name: str) -> bool:
        return self._backend.delete(name)


storage = AgentStorage()
