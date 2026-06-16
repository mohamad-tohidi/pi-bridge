"""SQLite-backed storage for LLM configurations."""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

from .models import LLMConfig


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS llms (
    name        TEXT PRIMARY KEY,
    base_url    TEXT NOT NULL,
    api_key     TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    api_format  TEXT NOT NULL DEFAULT 'completion'
)
"""


class LLMStorage:
    def __init__(self, db_path: str = "agents.db"):
        self.db_path = db_path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init(self):
        with self._conn() as c:
            c.execute(_CREATE_TABLE)

    def _row_to_llm(self, row: tuple) -> LLMConfig:
        name, base_url, api_key, model_name, api_format = row
        return LLMConfig(
            name=name,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            api_format=api_format,
        )

    def add(self, llm: LLMConfig) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO llms (name, base_url, api_key, model_name, api_format)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    base_url   = excluded.base_url,
                    api_key    = excluded.api_key,
                    model_name = excluded.model_name,
                    api_format = excluded.api_format
                """,
                (llm.name, llm.base_url, llm.api_key, llm.model_name, llm.api_format),
            )

    def get(self, name: str) -> Optional[LLMConfig]:
        with self._conn() as c:
            row = c.execute(
                "SELECT name, base_url, api_key, model_name, api_format FROM llms WHERE name = ?",
                (name,),
            ).fetchone()
        return self._row_to_llm(row) if row else None

    def list_all(self) -> list[LLMConfig]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT name, base_url, api_key, model_name, api_format FROM llms"
            ).fetchall()
        return [self._row_to_llm(r) for r in rows]

    def update(self, name: str, llm: LLMConfig) -> Optional[LLMConfig]:
        if not self.get(name):
            return None
        self.add(llm)
        return llm

    def delete(self, name: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM llms WHERE name = ?", (name,))
        return cur.rowcount > 0


llm_storage = LLMStorage()
