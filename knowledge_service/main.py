from typing import List
import json

import threading
import logging
from queue import Queue

from .models import ToolDefinition, AgentResponse, AgentCreateRequest, AskRequest, AskResponse
from .tools import get_tool_definitions
from .agents import agent_manager

from .utils import extract_link_map, replace_tokens_with_links

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pi_bridge.types import TextDeltaEvent, AgentEndEvent, ErrorEvent, ToolResultEvent

# Configure logging
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
        tool_results_content = []
        
        for event in events:
            logger.info(f"Received event type: {type(event)}")
            
            if isinstance(event, TextDeltaEvent):
                full_response += event.delta
            elif isinstance(event, ToolResultEvent):
                tool_results_content.append(event.content)
            elif isinstance(event, ErrorEvent):
                raise HTTPException(status_code=500, detail=f"Agent Error: {event.message}")
            elif isinstance(event, AgentEndEvent):
                break
        
        link_map = extract_link_map(tool_results_content)
        final_response = replace_tokens_with_links(full_response, link_map)
        
        return AskResponse(response=final_response)
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

            link_map = {}

            while True:
                event = event_queue.get()
                if event is None:
                    break
                if isinstance(event, Exception):
                    yield f"data: {json.dumps({'type': 'error', 'message': str(event)})}\n\n"
                    break
                
                if isinstance(event, ToolResultEvent):
                    new_map = extract_link_map([event.content])
                    link_map.update(new_map)
                    continue

                if isinstance(event, TextDeltaEvent):
                    processed_delta = replace_tokens_with_links(event.delta, link_map)
                    yield f"data: {json.dumps({'type': 'text_delta', 'delta': processed_delta})}\n\n"
                
                elif isinstance(event, AgentEndEvent):
                    yield f"data: {json.dumps(event.__dict__)}\n\n"
                    break
                else:
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
