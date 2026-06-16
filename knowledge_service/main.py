from __future__ import annotations

from typing import List, Optional, Union
import json
import threading
import logging
from pathlib import Path
from queue import Queue

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    LLMConfig, LLMCreateRequest, LLMUpdateRequest,
    DynamicTool, ToolCreateRequest, ToolUpdateRequest, ToolStatus,
    AgentResponse, AgentCreateRequest, AgentUpdateRequest,
    SessionInfo,
    AskRequest, AskResponse,
)
from .tools import BUILTIN_TOOL_DEFINITIONS
from .tool_storage import dynamic_tool_storage, validate_tool_code
from .llm_storage import llm_storage
from .agents import agent_manager
from pi_bridge.types import TextDeltaEvent, AgentEndEvent, ErrorEvent

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Pi Knowledge Service",
    openapi_tags=[
        {
            "name": "LLMs",
            "description": "Manage LLM configurations. If no LLM is assigned to an agent, "
                           "the service falls back to the environment variables "
                           "(`OPENAI_API_BASE`, `OPENAI_API_KEY`, `OPENAI_MODEL`).",
        },
        {
            "name": "Tools",
            "description": "Register, inspect, update, and delete dynamic tools. "
                           "Built-in tools are read-only and always appear in GET responses.",
        },
        {
            "name": "Agents",
            "description": "Create and manage agents. Each agent has a system prompt, "
                           "a list of tools, and an optional LLM assignment.",
        },
        {
            "name": "Ask",
            "description": "Send a query to an agent. Use `/ask` for a blocking response "
                           "or `/ask/stream` for a real-time SSE stream.",
        },
        {
            "name": "Sessions",
            "description": "Inspect and close active conversation sessions.",
        },
    ],
)

# Serve static files (CSS, JS if ever split out)
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ===========================================================================
# UI
# ===========================================================================

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the terminal UI."""
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(index))


# ===========================================================================
# LLMs
# ===========================================================================

@app.get("/llms", response_model=Union[LLMConfig, List[LLMConfig]], tags=["LLMs"])
async def get_llms(name: Optional[str] = None):
    """Get one LLM config by name, or list all registered LLMs."""
    if name:
        llm = llm_storage.get(name)
        if not llm:
            raise HTTPException(status_code=404, detail=f"LLM '{name}' not found")
        return llm
    return llm_storage.list_all()


@app.post("/llms", response_model=LLMConfig, status_code=201, tags=["LLMs"])
async def create_llm(request: LLMCreateRequest):
    """Register a new LLM configuration."""
    llm = LLMConfig(**request.model_dump())
    llm_storage.add(llm)
    return llm


@app.patch("/llms", response_model=LLMConfig, tags=["LLMs"])
async def update_llm(name: str, request: LLMUpdateRequest):
    """Partially update an LLM config. Invalidates sessions using this LLM."""
    existing = llm_storage.get(name)
    if not existing:
        raise HTTPException(status_code=404, detail=f"LLM '{name}' not found")
    updated = LLMConfig(
        name=name,
        base_url=request.base_url     if request.base_url    is not None else existing.base_url,
        api_key=request.api_key       if request.api_key     is not None else existing.api_key,
        model_name=request.model_name if request.model_name  is not None else existing.model_name,
        api_format=request.api_format if request.api_format  is not None else existing.api_format,
    )
    llm_storage.update(name, updated)
    agent_manager.invalidate_sessions_for_llm(name)
    return updated


@app.delete("/llms", tags=["LLMs"])
async def delete_llm(name: str):
    """Delete an LLM config. Sessions using it are closed."""
    if not llm_storage.delete(name):
        raise HTTPException(status_code=404, detail=f"LLM '{name}' not found")
    agent_manager.invalidate_sessions_for_llm(name)
    return {"status": "deleted", "name": name}


# ===========================================================================
# TOOLS
# ===========================================================================

@app.get("/tools", response_model=Union[DynamicTool, List[DynamicTool]], tags=["Tools"])
async def get_tools(name: Optional[str] = None):
    """Get one tool by name, or list all (built-in + dynamic)."""
    all_tools: list[DynamicTool] = [
        DynamicTool(
            name=d["name"],
            description=d["description"],
            parameters=d["parameters"],
            code="# built-in",
            status=ToolStatus.valid,
        )
        for d in BUILTIN_TOOL_DEFINITIONS
    ] + dynamic_tool_storage.list_all()

    if name:
        tool = next((t for t in all_tools if t.name == name), None)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
        return tool
    return all_tools


@app.post("/tools", response_model=DynamicTool, status_code=201, tags=["Tools"])
async def create_tool(request: ToolCreateRequest):
    """Register a new dynamic tool. Code is validated immediately on submission."""
    error = validate_tool_code(request.code, request.entry_point)
    status = ToolStatus.invalid if error else ToolStatus.valid
    tool = DynamicTool(
        name=request.name,
        description=request.description,
        parameters=request.parameters,
        code=request.code,
        entry_point=request.entry_point,
        status=status,
        error=error or None,
    )
    dynamic_tool_storage.add(tool)
    return tool


@app.patch("/tools", response_model=DynamicTool, tags=["Tools"])
async def update_tool(name: str, request: ToolUpdateRequest):
    """Partially update a dynamic tool. Code is re-validated if changed."""
    existing = dynamic_tool_storage.get(name)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    updated = DynamicTool(
        name=name,
        description=request.description if request.description is not None else existing.description,
        parameters=request.parameters   if request.parameters  is not None else existing.parameters,
        code=request.code               if request.code        is not None else existing.code,
        entry_point=request.entry_point if request.entry_point is not None else existing.entry_point,
    )
    error = validate_tool_code(updated.code, updated.entry_point)
    updated.status = ToolStatus.invalid if error else ToolStatus.valid
    updated.error  = error or None
    dynamic_tool_storage.update(name, updated)
    agent_manager.invalidate_sessions_for_tool(name)
    return updated


@app.delete("/tools", tags=["Tools"])
async def delete_tool(name: str):
    """Delete a dynamic tool and invalidate any sessions that use it."""
    if not dynamic_tool_storage.delete(name):
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    agent_manager.invalidate_sessions_for_tool(name)
    return {"status": "deleted", "name": name}


# ===========================================================================
# AGENTS
# ===========================================================================

@app.get("/agents", response_model=Union[AgentResponse, List[AgentResponse]], tags=["Agents"])
async def get_agents(name: Optional[str] = None):
    """Get one agent by name, or list all agents."""
    if name:
        agent = agent_manager.get_agent(name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
        return agent
    return agent_manager.list_agents()


@app.post("/agents", response_model=AgentResponse, status_code=201, tags=["Agents"])
async def create_agent(request: AgentCreateRequest):
    """Create a new agent. Set `llm_name` to assign a specific LLM, or omit to use the env default."""
    try:
        return agent_manager.create_agent(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/agents", response_model=AgentResponse, tags=["Agents"])
async def update_agent(name: str, request: AgentUpdateRequest):
    """
    Partially update an agent.

    - Set `llm_name` to a registered LLM name to switch the agent's LLM.
    - Set `llm_name` to `""` (empty string) to revert to the env default.
    - Omit `llm_name` to leave it unchanged.
    """
    updated = agent_manager.update_agent(name, request)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return updated


@app.delete("/agents", tags=["Agents"])
async def delete_agent(name: str):
    """Delete an agent and close all its active sessions."""
    if not agent_manager.delete_agent(name):
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return {"status": "deleted", "name": name}


# ===========================================================================
# ASK
# ===========================================================================

@app.post("/ask", response_model=AskResponse, tags=["Ask"])
async def ask(request: AskRequest):
    """
    Send a query to an agent and get a full response.

    - Omit `session_id` for a stateless one-shot question.
    - Provide a `session_id` to start or continue a multi-turn conversation.
    """
    try:
        session, sid = agent_manager.get_or_create_session(request.agent_name, request.session_id)
        full_response = ""
        for event in session.send(request.query):
            if isinstance(event, TextDeltaEvent):
                full_response += event.delta
            elif isinstance(event, ErrorEvent):
                raise HTTPException(status_code=500, detail=f"Agent error: {event.message}")
            elif isinstance(event, AgentEndEvent):
                break
        return AskResponse(response=full_response, session_id=sid)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream", tags=["Ask"])
async def ask_stream(request: AskRequest):
    """
    Send a query and stream the response as Server-Sent Events.

    First event is always `{"type": "session_id", "session_id": "..."}`.
    """
    async def event_generator():
        try:
            session, sid = agent_manager.get_or_create_session(request.agent_name, request.session_id)
            q: Queue = Queue()

            def worker():
                try:
                    for event in session.send_stream(request.query):
                        q.put(event)
                    q.put(None)
                except Exception as e:
                    q.put(e)

            threading.Thread(target=worker, daemon=True).start()
            yield f"data: {json.dumps({'type': 'session_id', 'session_id': sid})}\n\n"

            while True:
                event = q.get()
                if event is None:
                    break
                if isinstance(event, Exception):
                    yield f"data: {json.dumps({'type': 'error', 'message': str(event)})}\n\n"
                    break
                if isinstance(event, TextDeltaEvent):
                    yield f"data: {json.dumps({'type': 'text_delta', 'delta': event.delta})}\n\n"
                elif isinstance(event, AgentEndEvent):
                    yield f"data: {json.dumps({'type': 'agent_end', 'stop_reason': event.stop_reason, 'session_id': sid})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps(event.__dict__)}\n\n"

        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ===========================================================================
# SESSIONS
# ===========================================================================

@app.get("/sessions", response_model=Union[SessionInfo, List[SessionInfo]], tags=["Sessions"])
async def get_sessions(session_id: Optional[str] = None):
    """List all active sessions, or get one by session_id."""
    if session_id:
        match = next((s for s in agent_manager.list_sessions() if s.session_id == session_id), None)
        if not match:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return match
    return agent_manager.list_sessions()


@app.delete("/sessions", tags=["Sessions"])
async def close_session(session_id: str):
    """Close and destroy an active session."""
    if not agent_manager.close_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"status": "closed", "session_id": session_id}


# ===========================================================================
# Lifecycle
# ===========================================================================

@app.on_event("shutdown")
def shutdown_event():
    agent_manager.close_all()
