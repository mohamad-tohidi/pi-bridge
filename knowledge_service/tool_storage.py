"""SQLite-backed storage for dynamic user-submitted tools."""
from __future__ import annotations

import ast
import json
import sqlite3
import sys
from typing import Optional

from .models import DynamicTool, ToolStatus

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_BLOCKED_MODULES = {"os", "subprocess", "sys", "shutil", "pathlib", "socket",
                    "ctypes", "multiprocessing", "threading", "importlib",
                    "builtins", "__builtin__", "eval", "exec", "compile",
                    "open", "io", "tempfile", "glob", "signal", "pty",
                    "resource", "mmap", "pickle", "shelve"}


class ValidationError(Exception):
    pass


def _check_imports(tree: ast.AST) -> None:
    """Reject any import of a blocked module."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if node.module else [])
            )
            for name in names:
                root = name.split(".")[0]
                if root in _BLOCKED_MODULES:
                    raise ValidationError(f"Import of '{name}' is not allowed")


def _check_no_open_calls(tree: ast.AST) -> None:
    """Reject direct calls to open(), exec(), eval()."""
    blocked_calls = {"open", "exec", "eval", "compile", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in blocked_calls:
                raise ValidationError(f"Call to '{name}()' is not allowed")


def validate_tool_code(code: str, entry_point: str) -> str:
    """
    Parse and validate user-submitted tool code.
    Returns a clean error message on failure, empty string on success.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    try:
        _check_imports(tree)
        _check_no_open_calls(tree)
    except ValidationError as e:
        return str(e)

    # entry_point function must exist
    fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if entry_point not in fn_names:
        return f"Function '{entry_point}' not found in submitted code"

    # Dry-run: attempt to exec in isolated namespace
    safe_ns: dict = {}
    try:
        import requests as _requests
        safe_ns["requests"] = _requests
        exec(compile(tree, "<tool>", "exec"), safe_ns)  # noqa: S102
    except Exception as e:
        return f"Execution error during validation: {e}"

    return ""  # no error


def build_tool_callable(tool: DynamicTool):
    """Return a callable (sync) that executes the stored tool code."""
    import requests as _requests
    import json as _json
    import re as _re
    import math as _math
    import datetime as _datetime
    import collections as _collections
    import itertools as _itertools
    import functools as _functools
    import string as _string
    import hashlib as _hashlib
    import base64 as _base64
    import urllib as _urllib
    import html as _html

    safe_ns = {
        "requests": _requests,
        "json": _json,
        "re": _re,
        "math": _math,
        "datetime": _datetime,
        "collections": _collections,
        "itertools": _itertools,
        "functools": _functools,
        "string": _string,
        "hashlib": _hashlib,
        "base64": _base64,
        "urllib": _urllib,
        "html": _html,
    }
    exec(compile(tool.code, f"<tool:{tool.name}>", "exec"), safe_ns)  # noqa: S102
    fn = safe_ns[tool.entry_point]

    def wrapper(**kwargs):
        return fn(**kwargs)

    wrapper.__name__ = tool.name
    return wrapper


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS dynamic_tools (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    parameters  TEXT NOT NULL,
    code        TEXT NOT NULL,
    entry_point TEXT NOT NULL DEFAULT 'run',
    status      TEXT NOT NULL DEFAULT 'pending',
    error       TEXT
)
"""


class DynamicToolStorage:
    def __init__(self, db_path: str = "agents.db"):
        self.db_path = db_path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init(self):
        with self._conn() as c:
            c.execute(_CREATE_TABLE)

    def _row_to_tool(self, row: tuple) -> DynamicTool:
        name, description, parameters, code, entry_point, status, error = row
        return DynamicTool(
            name=name,
            description=description,
            parameters=json.loads(parameters),
            code=code,
            entry_point=entry_point,
            status=ToolStatus(status),
            error=error,
        )

    def add(self, tool: DynamicTool) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO dynamic_tools (name, description, parameters, code, entry_point, status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description  = excluded.description,
                    parameters   = excluded.parameters,
                    code         = excluded.code,
                    entry_point  = excluded.entry_point,
                    status       = excluded.status,
                    error        = excluded.error
                """,
                (
                    tool.name, tool.description,
                    json.dumps(tool.parameters, ensure_ascii=False),
                    tool.code, tool.entry_point,
                    tool.status.value, tool.error,
                ),
            )

    def get(self, name: str) -> Optional[DynamicTool]:
        with self._conn() as c:
            row = c.execute(
                "SELECT name, description, parameters, code, entry_point, status, error "
                "FROM dynamic_tools WHERE name = ?", (name,)
            ).fetchone()
        return self._row_to_tool(row) if row else None

    def list_all(self) -> list[DynamicTool]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT name, description, parameters, code, entry_point, status, error "
                "FROM dynamic_tools"
            ).fetchall()
        return [self._row_to_tool(r) for r in rows]

    def update(self, name: str, tool: DynamicTool) -> Optional[DynamicTool]:
        if not self.get(name):
            return None
        self.add(tool)
        return tool

    def delete(self, name: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM dynamic_tools WHERE name = ?", (name,))
        return cur.rowcount > 0


dynamic_tool_storage = DynamicToolStorage()
