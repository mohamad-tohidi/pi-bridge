from typing import List, Optional
import json
import threading
import logging
from queue import Queue

from .models import ToolDefinition, AgentResponse, AgentCreateRequest, AgentUpdateRequest, AskRequest, AskResponse
from .tools import get_tool_definitions
from .agents import agent_manager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pi_bridge.types import TextDeltaEvent, AgentEndEvent, ErrorEvent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pi Knowledge Service")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@app.get("/tools", response_model=List[ToolDefinition])
async def get_tools():
    return [ToolDefinition(**d) for d in get_tool_definitions()]


# ---------------------------------------------------------------------------
# Agents  (single route, optional name param)
# ---------------------------------------------------------------------------

@app.get("/agents", response_model=AgentResponse | List[AgentResponse])
async def get_agents(name: Optional[str] = None):
    if name:
        agent = agent_manager.get_agent(name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
        return agent
    return agent_manager.list_agents()


@app.post("/agents", response_model=AgentResponse, status_code=201)
async def create_agent(request: AgentCreateRequest):
    try:
        return agent_manager.create_agent(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/agents", response_model=AgentResponse)
async def update_agent(name: str, request: AgentUpdateRequest):
    updated = agent_manager.update_agent(name, request)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return updated


@app.delete("/agents")
async def delete_agent(name: str):
    deleted = agent_manager.delete_agent(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return {"status": "deleted", "name": name}


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        session, sid = agent_manager.get_or_create_session(request.agent_name, request.session_id)
        events = session.send(request.query)

        full_response = ""
        for event in events:
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


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    async def event_generator():
        try:
            session, sid = agent_manager.get_or_create_session(request.agent_name, request.session_id)
            event_queue: Queue = Queue()

            def worker():
                try:
                    for event in session.send_stream(request.query):
                        event_queue.put(event)
                    event_queue.put(None)
                except Exception as e:
                    event_queue.put(e)

            threading.Thread(target=worker).start()

            yield f"data: {json.dumps({'type': 'session_id', 'session_id': sid})}\n\n"

            while True:
                event = event_queue.get()
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


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@app.delete("/sessions")
async def close_session(session_id: str):
    agent_manager.close_session(session_id)
    return {"status": "closed", "session_id": session_id}


@app.on_event("shutdown")
def shutdown_event():
    agent_manager.close_all()
