"""
API Server for Agentic AI Demo
This module provides the FastAPI server that exposes the agent's capabilities.
"""
import logging
import os
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import agent modules
from agent.perception.text_processor import TextProcessor
from agent.reasoning.inference_engine import InferenceEngine
from agent.planning.planner import Planner
from agent.action.executor import ActionExecutor
from agent.exact_recall.memory_manager import MemoryManager

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Agentic AI Demo API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent components
text_processor = TextProcessor()
inference_engine = InferenceEngine()
planner = Planner()
action_executor = ActionExecutor()
memory_manager = MemoryManager(memory_file="agent_memories.json")

# Define API models
class UserInput(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None

class MemoryInput(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    perception_result: Dict[str, Any]
    reasoning_result: Dict[str, Any]
    planning_result: Dict[str, Any]
    action_result: Dict[str, Any]
    recall_result: Dict[str, Any]
    final_response: str

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# API endpoints
@app.get("/")
async def root():
    return {"message": "Agentic AI Demo API is running"}

@app.post("/process", response_model=AgentResponse)
async def process_input(user_input: UserInput):
    """Process user input through the agent pipeline."""
    logger.info(f"Processing input: {user_input.text}")

    # Step 1: Perception - Process the input
    perception_result = text_processor.process(user_input.text, user_input.context)

    # Step 2: Exact Recall - Retrieve relevant memories
    recall_result = {
        "query": user_input.text,
        "memories": memory_manager.retrieve(user_input.text),
        "timestamp": memory_manager._get_timestamp() if hasattr(memory_manager, '_get_timestamp') else None
    }

    # Step 3: Reasoning - Analyze the processed data with memories
    reasoning_result = inference_engine.analyze(perception_result)

    # Step 4: Planning - Generate action plan
    planning_result = planner.create_plan(reasoning_result)

    # Step 5: Action - Execute the plan
    action_result = action_executor.execute(planning_result)

    # Store the interaction in memory
    memory_content = f"User: {user_input.text}\nAgent: {action_result.get('response', 'No response generated')}"
    memory_manager.store(memory_content, {
        "user_input": user_input.text,
        "response": action_result.get('response', 'No response generated'),
        "interaction_type": perception_result.get('intent', {}).get('type', 'unknown')
    })

    # Prepare the final response
    response = AgentResponse(
        perception_result=perception_result,
        reasoning_result=reasoning_result,
        planning_result=planning_result,
        action_result=action_result,
        recall_result=recall_result,
        final_response=action_result.get("response", "No response generated")
    )

    return response

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            # Process the input through the agent pipeline
            user_input = UserInput(text=data)

            # Step 1: Perception
            perception_result = text_processor.process(user_input.text)
            await websocket.send_json({"module": "perception", "data": perception_result})

            # Step 2: Exact Recall
            recall_result = {
                "query": user_input.text,
                "memories": memory_manager.retrieve(user_input.text),
                "timestamp": memory_manager._get_timestamp() if hasattr(memory_manager, '_get_timestamp') else None
            }
            await websocket.send_json({"module": "exact_recall", "data": recall_result})

            # Step 3: Reasoning
            reasoning_result = inference_engine.analyze(perception_result)
            await websocket.send_json({"module": "reasoning", "data": reasoning_result})

            # Step 4: Planning
            planning_result = planner.create_plan(reasoning_result)
            await websocket.send_json({"module": "planning", "data": planning_result})

            # Step 5: Action
            action_result = action_executor.execute(planning_result)
            await websocket.send_json({"module": "action", "data": action_result})

            # Store the interaction in memory
            memory_content = f"User: {user_input.text}\nAgent: {action_result.get('response', 'No response generated')}"
            memory_manager.store(memory_content, {
                "user_input": user_input.text,
                "response": action_result.get('response', 'No response generated'),
                "interaction_type": perception_result.get('intent', {}).get('type', 'unknown')
            })

            # Send final response
            await websocket.send_json({
                "module": "final",
                "data": {
                    "response": action_result.get("response", "No response generated")
                }
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Memory management endpoints
@app.get("/memories")
async def get_memories():
    """Get all stored memories."""
    return {"memories": memory_manager.get_all()}

@app.post("/memories")
async def create_memory(memory_input: MemoryInput):
    """Store a new memory."""
    memory_id = memory_manager.store(memory_input.content, memory_input.metadata)
    return {"memory_id": memory_id, "status": "success"}

@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory."""
    success = memory_manager.delete(memory_id)
    return {"status": "success" if success else "not_found"}

@app.delete("/memories")
async def clear_memories():
    """Clear all memories."""
    memory_manager.clear()
    return {"status": "success"}

@app.get("/memories/search")
async def search_memories(query: str, limit: int = 5):
    """Search memories based on a query."""
    memories = memory_manager.retrieve(query, limit)
    return {"memories": memories}

def start_api_server():
    """Start the API server."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("api.app:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    start_api_server()
