#!/usr/bin/env python3
"""
pi-bridge demo: session continuity + custom tool calling.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pi_bridge import PiSession, Provider, Model, CustomTool
from dotenv import load_dotenv ; load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_API_BASE", "")
MODEL_NAME = os.environ.get("OPENAI_MODEL", "")

if not API_KEY:
    print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Custom tool: returns a secret number hidden from the agent
# ---------------------------------------------------------------------------

SECRET = 37

def get_secret_number() -> str:
    print(f"  [tool] get_secret_number() → {SECRET}")
    return str(SECRET)


# ---------------------------------------------------------------------------
# Session setup
# ---------------------------------------------------------------------------

print("Initializing Pi session...")
session = PiSession(
    provider=Provider(
        base_url=BASE_URL,
        api_key=API_KEY,
    ),
    model=Model(
        name=MODEL_NAME,
        api_format="completion",
    ),
    tools=[],
    custom_tools=[
        CustomTool(
            name="get_secret_number",
            description="Returns a secret number. Call this tool whenever you need the secret number.",
            parameters={"type": "object", "properties": {}},
            fn=get_secret_number,
        )
    ],
)


def print_events(events):
    for event in events:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
        elif event.type == "tool_call":
            print(f"  [agent calling tool: {event.tool_name}]")
        elif event.type == "agent_end":
            print(f"\n[stop_reason={event.stop_reason}]")
        elif event.type == "error":
            print(f"\n[ERROR: {event.message}]", file=sys.stderr)


# ---------------------------------------------------------------------------
# Round 1: agent must call the tool to get the number
# ---------------------------------------------------------------------------

print("\n=== 第一轮：工具调用 ===")
print_events(session.send("请调用 get_secret_number 工具获取秘密数字，然后告诉我它是多少。"))

# ---------------------------------------------------------------------------
# Round 2: agent already knows the number from the previous turn
# ---------------------------------------------------------------------------

print("\n=== 第二轮：会话连续性（无需再次调用工具）===")
print_events(session.send("把刚才那个秘密数字乘以2，只回答结果。"))

# ---------------------------------------------------------------------------
# Message history
# ---------------------------------------------------------------------------

print("\n=== 消息历史 ===")
for msg in session.messages:
    role = msg.get("role", "?")
    if role == "assistant":
        texts = [c["text"] for c in msg.get("content", []) if c.get("type") == "text"]
        if texts:
            print(f"assistant: {''.join(texts).strip()}")
    elif role == "user":
        print(f"user: {str(msg.get('content', '')).strip()}")
    elif role == "tool_result":
        print(f"tool_result({msg.get('tool_name')}): {msg.get('content', '').strip()}")

session.close()
print("\nDemo complete!")
