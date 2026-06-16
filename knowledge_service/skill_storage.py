"""SQLite-backed storage for skills."""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

from .models import Skill

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS skills (
    name          TEXT PRIMARY KEY,
    description   TEXT NOT NULL,
    content       TEXT NOT NULL,
    allowed_tools TEXT NOT NULL DEFAULT '[]'
)
"""


class SkillStorage:
    def __init__(self, db_path: str = "agents.db"):
        self.db_path = db_path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init(self):
        with self._conn() as c:
            c.execute(_CREATE_TABLE)

    def _row_to_skill(self, row: tuple) -> Skill:
        name, description, content, allowed_tools = row
        return Skill(
            name=name,
            description=description,
            content=content,
            allowed_tools=json.loads(allowed_tools),
        )

    def add(self, skill: Skill) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO skills (name, description, content, allowed_tools)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description   = excluded.description,
                    content       = excluded.content,
                    allowed_tools = excluded.allowed_tools
                """,
                (
                    skill.name,
                    skill.description,
                    skill.content,
                    json.dumps(skill.allowed_tools, ensure_ascii=False),
                ),
            )

    def get(self, name: str) -> Optional[Skill]:
        with self._conn() as c:
            row = c.execute(
                "SELECT name, description, content, allowed_tools FROM skills WHERE name = ?",
                (name,)
            ).fetchone()
        return self._row_to_skill(row) if row else None

    def list_all(self) -> list[Skill]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT name, description, content, allowed_tools FROM skills"
            ).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def update(self, name: str, skill: Skill) -> Optional[Skill]:
        if not self.get(name):
            return None
        self.add(skill)
        return skill

    def delete(self, name: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM skills WHERE name = ?", (name,))
        return cur.rowcount > 0


skill_storage = SkillStorage()
