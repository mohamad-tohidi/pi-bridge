from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import List
import json
import asyncio
import threading
from queue import Queue

from .models import ToolDefinition, AgentResponse, AgentCreateRequest, AskRequest, AskResponse
from .tools import get_tool_definitions
from .agents import agent_manager
from .storage import storage

# Import types for instance checking
from pi_bridge.types import TextDeltaEvent, AgentEndEvent, ErrorEvent

app = FastAPI(title="Pi Knowledge Service")

@app.get("/tools", response_model=List[ToolDefinition])
async def get_tools():
    defs = get_tool_definitions()
    return [ToolDefinition(**d) for d in defs]

@app.get("/agents", response_model=List[AgentResponse])
async def list_agents():
    agents = agent_manager.list_agents()
    return agents

@app.get("/agents/{name}", response_model=AgentResponse)
async def get_agent(name: str):
    agent = agent_manager.get_agent(name)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.post("/agents", response_model=AgentResponse)
async def create_agent(request: AgentCreateRequest):
    try:
        agent = agent_manager.create_agent(request)
        return agent
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        session = agent_manager.get_session(request.agent_name)
        events = session.send(request.query)
        
        full_response = ""
        for event in events:
            # Log events to terminal for debugging
            print(f"DEBUG: Received event type: {type(event)} - content: {event}")
            
            if isinstance(event, TextDeltaEvent):
                full_response += event.delta
            elif isinstance(event, ErrorEvent):
                # If the agent returns an error, raise it as an HTTP exception
                raise HTTPException(status_code=500, detail=f"Agent Error: {event.message}")
            elif isinstance(event, AgentEndEvent):
                break
        
        return AskResponse(response=full_response)
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
            session = agent_manager.get_session(request.agent_name)
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

            while True:
                event = event_queue.get()
                if event is None:
                    break
                if isinstance(event, Exception):
                    yield f"data: {json.dumps({'type': 'error', 'message': str(event)})}\n\n"
                    break
                
                print(f"DEBUG STREAM: {event}")
                yield f"data: {json.dumps(event.__dict__)}\n\n"
            
            thread.join()

        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.on_event("shutdown")
def shutdown_event():
    agent_manager.close_all()
