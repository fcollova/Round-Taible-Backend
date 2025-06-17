from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import httpx
import configparser
import asyncio
import json
from typing import List, Dict, Any, Optional
import os
import logging
from datetime import datetime
from websocket_manager import debate_manager
from llm_queue_manager import llm_queue, MessageRequest, MessagePriority

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = configparser.ConfigParser()
config.read('config.conf')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await llm_queue.start(num_workers=3)
    yield
    # Shutdown
    await llm_queue.stop()

app = FastAPI(title="Round TAIble Backend", version="1.0.0", lifespan=lifespan)

# CORS middleware
allowed_origins = []
try:
    allowed_origins = config.get('cors', 'allowed_origins').split(',')
except:
    # Fallback per produzione
    allowed_origins = [
        "http://localhost:3000",
        "https://*.vercel.app",
        "https://*.railway.app"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter configuration
import os
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY') or config.get('openrouter', 'api_key', fallback='')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL') or config.get('openrouter', 'base_url', fallback='https://openrouter.ai/api/v1')
TIMEOUT = int(os.getenv('OPENROUTER_TIMEOUT', '60')) or int(config.get('openrouter', 'timeout', fallback='60'))

# Model mappings - load all models from config
MODELS = {}
for model_name in config.options('models'):
    MODELS[model_name] = config.get('models', model_name)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    id: str
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

async def call_openrouter(model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """Make API call to OpenRouter"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Round TAIble"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        **kwargs
    }
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            logger.error(f"Response status: {response.status_code}")
            logger.error(f"Response text: {response.text}")
            logger.error(f"Request payload: {json.dumps(payload, indent=2)}")
            raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Round TAIble Backend API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models": list(MODELS.keys())}

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Main chat endpoint that interfaces with OpenRouter"""
    if request.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not supported")
    
    openrouter_model = MODELS[request.model]
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    try:
        response = await call_openrouter(
            model=openrouter_model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debate/generate")
async def generate_debate_message(request: Dict[str, Any]):
    """Generate a debate message for a specific AI personality"""
    model = request.get('model')
    topic = request.get('topic')
    context = request.get('context', '')
    personality = request.get('personality', '')
    message_count = request.get('message_count', 0)
    
    if not model or not topic:
        raise HTTPException(status_code=400, detail="Model and topic are required")
    
    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model {model} not supported")
    
    # Create system prompt based on personality and debate stage
    if message_count == 0:
        # First message - opening statement
        system_prompt = f"""You are starting a live debate about: {topic}

        Your personality and expertise: {personality}
        
        As the first speaker, provide an opening statement that:
        1. Clearly states your position on the topic
        2. Presents 2-3 key arguments
        3. Sets the tone for an engaging debate
        
        Keep it concise (2-3 paragraphs max) and compelling. This is your chance to make a strong first impression."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Start the debate on: {topic}"}
        ]
    else:
        # Continuing debate - response to previous messages
        system_prompt = f"""You are participating in an ongoing live debate about: {topic}

        Your personality and debate style: {personality}
        
        Previous discussion context:
        {context}
        
        Generate a thoughtful response that:
        1. Directly addresses specific points made by other participants
        2. Builds upon or challenges their arguments with evidence
        3. Maintains your unique perspective while engaging constructively
        4. Advances the debate with new insights or counterpoints
        5. If there is moderator guidance in the context, you MUST acknowledge and address it in your response
        
        Guidelines:
        - Keep it concise (1-2 paragraphs max) and engaging
        - Reference specific points from the previous discussion
        - Stay true to your personality and expertise
        - Be respectful but intellectually rigorous
        - Avoid repeating what others have already said
        - IMPORTANT: If the moderator has provided guidance, incorporate their direction into your response while maintaining your unique perspective"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Based on the discussion above, provide your next contribution to the debate on: {topic}"}
        ]
    
    try:
        response = await call_openrouter(
            model=MODELS[model],
            messages=messages,
            max_tokens=350 if message_count == 0 else 300,
            temperature=0.8
        )
        
        content = response['choices'][0]['message']['content']
        return {
            "model": model,
            "content": content,
            "topic": topic,
            "message_count": message_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debate/continue")
async def continue_debate(request: Dict[str, Any]):
    """Generate next message in ongoing debate"""
    debate_id = request.get('debate_id')
    models = request.get('models', [])
    topic = request.get('topic')
    recent_messages = request.get('recent_messages', [])
    
    if not debate_id or not models or not topic:
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    logger.info(f"Debate {debate_id}: Continue with {len(models)} participants on topic: '{topic}'")
    
    # Fetch model information from the database
    model_info = {}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:3000/api/models")
            if response.status_code == 200:
                model_info = response.json()
    except Exception as e:
        print(f"Warning: Could not fetch model info from database: {e}")
    
    # Determine next model to speak (strict round-robin alternation)
    last_speaker = recent_messages[-1].get('model') if recent_messages else None
    next_model = None
    
    if last_speaker and len(models) > 1:
        try:
            current_index = models.index(last_speaker)
            next_index = (current_index + 1) % len(models)
            next_model = models[next_index]
        except ValueError:
            # If last speaker not in current models, start from first
            next_model = models[0]
    else:
        # Start with first model or continue if only one model
        next_model = models[0]
    
    # Map model ID to internal model name - use the ID directly since it matches MODELS keys
    model_name = next_model
    logger.info(f"Debate {debate_id}: Next speaker is {next_model}, mapped to {model_name}")
    
    if model_name not in MODELS:
        logger.error(f"Model {model_name} not found in MODELS: {list(MODELS.keys())}")
        raise HTTPException(status_code=400, detail=f"Model {model_name} not supported")
    
    # Create enhanced context from recent messages with model names and moderator interventions
    context_messages = []
    moderator_guidance = ""
    
    for msg in recent_messages[-5:]:  # Last 5 messages for better context
        if msg.get('type') == 'moderator':
            # Handle moderator messages specially
            moderator_guidance = f"\n\n**MODERATOR GUIDANCE**: {msg.get('content', '')}\n*Please address and incorporate this guidance in your response.*"
            context_messages.append(f"[MODERATORE]: {msg.get('content', '')}")
        else:
            model_id = msg.get('model', 'Unknown')
            model_display_name = model_id
            if model_id in model_info:
                model_display_name = model_info[model_id].get('name', model_id)
            context_messages.append(f"{model_display_name}: {msg.get('content', '')}")
    
    context = "\n".join(context_messages) + moderator_guidance
    
    # Get personality from database
    personality = f"AI model {next_model}"
    if next_model in model_info:
        personality = model_info[next_model].get('description', personality)
    
    # Crea richiesta per la coda LLM
    message_request = MessageRequest(
        debate_id=debate_id,
        model_id=model_name,
        context=context,
        topic=topic,
        personality=personality,
        message_count=len(recent_messages),
        priority=MessagePriority.NORMAL
    )
    
    # Enqueue la richiesta
    request_id = await llm_queue.enqueue_message(message_request)
    
    return {
        "status": "queued",
        "request_id": request_id,
        "debate_id": debate_id,
        "next_model": next_model,
        "queue_stats": llm_queue.get_queue_stats()
    }

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    return {"models": MODELS}

@app.websocket("/ws/debates/{debate_id}")
async def websocket_endpoint(websocket: WebSocket, debate_id: str):
    """WebSocket endpoint per dibattiti real-time"""
    await debate_manager.connect(websocket, debate_id)
    try:
        while True:
            # Ricevi messaggi dal client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Gestisci azione utente
            response = await debate_manager.handle_user_action(debate_id, message)
            
            # Invia risposta se necessario
            if response:
                await websocket.send_text(json.dumps(response))
                
    except WebSocketDisconnect:
        debate_manager.disconnect(websocket, debate_id)
    except Exception as e:
        logger.error(f"WebSocket error for debate {debate_id}: {e}")
        debate_manager.disconnect(websocket, debate_id)

@app.get("/ws/stats")
async def get_websocket_stats():
    """Endpoint per statistiche WebSocket"""
    return debate_manager.get_connection_stats()

@app.get("/queue/stats")
async def get_queue_statistics():
    """Endpoint per monitorare stato delle code LLM"""
    return llm_queue.get_queue_stats()

@app.post("/queue/priority")
async def set_message_priority(request: Dict[str, Any]):
    """Endpoint per impostare priorità alta a una richiesta"""
    request_id = request.get('request_id')
    priority = request.get('priority', 'NORMAL')
    
    try:
        priority_enum = MessagePriority[priority.upper()]
        # Logica per aggiornare priorità se implementata
        return {"status": "updated", "request_id": request_id, "priority": priority}
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid priority level")

if __name__ == "__main__":
    import uvicorn
    
    host = config.get('server', 'host')
    port = int(config.get('server', 'port'))
    debug = config.getboolean('server', 'debug')
    
    if debug:
        uvicorn.run("main:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port)