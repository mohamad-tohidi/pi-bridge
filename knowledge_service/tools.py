"""Built-in (hardcoded) tools + helpers to merge with dynamic tools."""
from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger(__name__)


async def parsa_tool(query: str) -> str:
    """
    Searches the internal semantic search API for religious content.
    Best used with Persian queries.
    """
    logger.info(f"Parsa tool called with query: {query}")
    url = "http://172.30.0.112:8181/api/user/question/search"
    params = {
        "type": "semantic",
        "query": query,
        "page_size": 10,
        "page": 1,
        "is_send_answer": "true",
    }
    try:
        response = requests.get(url, params=params, headers={"accept": "application/json"}, timeout=10)
        response.raise_for_status()
        data = response.json()
        search_results = data.get("data", {}).get("result", [])
        items = []
        for res in search_results:
            highlight = res.get("highlight", "")
            match = re.search(r"\u067e\u0627\u0633\u062e:\s*(.*)", highlight)
            snippet = match.group(1).strip() if match else highlight
            items.append(
                f"[ID:{res.get('id')}] {res.get('content')} | {snippet[:600]} | {res.get('source_link', '')}"
            )
        return "\n\n".join(items) if items else "No results found."
    except Exception as e:
        return f"Error: {e}"


async def haditha_tool(query: str) -> str:
    """Search for Ahadith from AhlulBait."""
    return f"HADITHA RESULT: (Mock) Searching for Ahadith related to '{query}'..."


# Hardcoded tools registry
BUILTIN_TOOLS = {
    "parsa": parsa_tool,
    "haditha": haditha_tool,
}

BUILTIN_TOOL_DEFINITIONS = [
    {
        "name": "parsa",
        "description": "Searches for religious content. Best used with Persian queries.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "haditha",
        "description": "Searches for Ahadith from AhlulBait.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
]


def make_sync(async_fn):
    """Wrap an async function into a sync callable."""
    def wrapper(**kwargs):
        with ThreadPoolExecutor() as pool:
            return pool.submit(lambda: asyncio.run(async_fn(**kwargs))).result()
    wrapper.__name__ = getattr(async_fn, "__name__", "tool")
    return wrapper
