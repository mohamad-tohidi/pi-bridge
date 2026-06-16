from typing import List
import json
import threading
import logging
from queue import Queue

from .models import ToolDefinition, AgentResponse, AgentCreateRequest, AskRequest, AskResponse
from .tools import get_tool_definitions
from .agents import agent_manager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pi_bridge.types import TextDeltaEvent, AgentEndEvent, ErrorEvent, ToolResultEvent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pi Knowledge Service")


@app.get("/tools", response_model=List[ToolDefinition])
async def get_tools():
    defs = get_tool_definitions()
    return [ToolDefinition(**d) for d in defs]


@app.get("/agents", response_model=List[AgentResponse])
async def list_agents():
    return agent_manager.list_agents()


@app.get("/agents/{name}", response_model=AgentResponse)
async def get_agent(name: str):
    agent = agent_manager.get_agent(name)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@app.post("/agents", response_model=AgentResponse)
async def create_agent(request: AgentCreateRequest):
    try:
        return agent_manager.create_agent(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                raise HTTPException(status_code=500, detail=f"Agent Error: {event.message}")
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
            event_queue = Queue()

            def worker():
                try:
                    for event in session.send_stream(request.query):
                        event_queue.put(event)
                    event_queue.put(None)
                except Exception as e:
                    event_queue.put(e)

            thread = threading.Thread(target=worker)
            thread.start()

            # emit session_id first so client can grab it immediately
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

            thread.join()

        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    agent_manager.close_session(session_id)
    return {"status": "closed"}


@app.on_event("shutdown")
def shutdown_event():
    agent_manager.close_all()
