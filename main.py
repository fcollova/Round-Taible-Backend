from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
import requests
import httpx
import asyncio
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import time
from websocket_manager import debate_manager
from llm_queue_manager import llm_queue, MessageRequest, MessagePriority
from logging_config import setup_logging, get_context_logger, performance_metrics
from config_manager import get_config, ConfigurationError

# Initialize configuration manager
try:
    config = get_config()
    logger_initialized = False
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    exit(1)

# Setup structured logging using configuration
if not logger_initialized:
    setup_logging(
        debug=config.get_logging_debug(),
        level=config.get_logging_level(),
        console_output=config.get_logging_console_output(),
        file_output=config.get_logging_file_output(),
        log_file=config.get_logging_file_path()
    )
    logger_initialized = True

logger = get_context_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - ridotto numero workers per evitare sovraccarico API
    await llm_queue.start(num_workers=2)
    yield
    # Shutdown
    await llm_queue.stop()

app = FastAPI(title="Round TAIble Backend", version="1.0.0", lifespan=lifespan)

# Logging middleware per tutte le richieste HTTP
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log richiesta in entrata
        logger.info("HTTP request started",
                   method=request.method,
                   url=str(request.url),
                   client_ip=request.client.host if request.client else "unknown",
                   user_agent=request.headers.get("user-agent", "unknown"))
        
        # Processa richiesta
        try:
            response = await call_next(request)
            response_time = time.time() - start_time
            
            # Log risposta
            performance_metrics.log_api_call(
                endpoint=str(request.url.path),
                method=request.method,
                response_time=response_time,
                status_code=response.status_code
            )
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error("HTTP request failed",
                        method=request.method,
                        url=str(request.url),
                        response_time=response_time,
                        error=str(e),
                        error_type=type(e).__name__)
            raise

app.add_middleware(LoggingMiddleware)

# CORS and Frontend configuration from config manager
allowed_origins = config.get_cors_allowed_origins()
FRONTEND_URL = config.get_frontend_url()

if not allowed_origins:
    allowed_origins = ['http://localhost:3000']
    
logger.info("CORS and Frontend configuration loaded", 
           origins=allowed_origins, 
           frontend_url=FRONTEND_URL,
           environment=config.get_environment())

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter and Frontend configuration from config manager
OPENROUTER_API_KEY = config.get_openrouter_api_key()
OPENROUTER_BASE_URL = config.get_openrouter_base_url()
TIMEOUT = config.get_openrouter_timeout()
FRONTEND_TIMEOUT = config.get_frontend_timeout()

# Model mappings from config manager
MODELS = config.get_models()
logger.info("Models loaded from configuration", 
           model_count=len(MODELS), 
           models=list(MODELS.keys()),
           environment=config.get_environment())

# Simple data structures instead of Pydantic models for Render compatibility

def call_openrouter(model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """Make API call to OpenRouter with comprehensive logging and rate limiting"""
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    
    logger.info("Starting OpenRouter API call", 
               model_id=model, 
               request_id=request_id,
               message_count=len(messages))
    
    # Add configurable rate limiting delay to prevent API abuse detection
    import random
    min_delay = config.get_openrouter_min_delay()
    max_delay = config.get_openrouter_max_delay()
    delay = random.uniform(min_delay, max_delay)
    time.sleep(delay)
    
    logger.debug("Applied rate limiting delay", 
                request_id=request_id, 
                delay_seconds=delay)
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": FRONTEND_URL,
        "X-Title": "Round TAIble - AI Debate Platform",
        "User-Agent": "Round-TAIble/1.0 (AI-Debate-Platform)",
        "Accept": "application/json",
        "X-Request-ID": request_id
    }
    
    payload = {
        "model": model,
        "messages": messages,
        **kwargs
    }
    
    # Log conversation details in debug mode
    if config.get_logging_debug():
        logger.debug("OpenRouter request payload",
                    model_id=model,
                    request_id=request_id,
                    messages=messages,
                    payload_params=kwargs)
    
    try:
        # Use configurable timeout instead of hardcoded 30s
        timeout = config.get_openrouter_timeout()
        
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response_time = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        
        # Log successful request
        tokens_used = result.get('usage', {}).get('total_tokens', 0)
        
        # Check for rate limit warnings in response headers
        rate_limit_remaining = response.headers.get('x-ratelimit-remaining')
        rate_limit_reset = response.headers.get('x-ratelimit-reset')
        
        if rate_limit_remaining:
            remaining = int(rate_limit_remaining)
            if remaining < 5:  # Warn when getting close to limit
                logger.warning("OpenRouter rate limit getting low",
                              model_id=model,
                              request_id=request_id,
                              remaining_requests=remaining,
                              reset_time=rate_limit_reset)
        
        performance_metrics.log_llm_request(
            model=model,
            response_time=response_time,
            tokens_used=tokens_used,
            request_id=request_id
        )
        
        logger.info("OpenRouter API call successful",
                   model_id=model,
                   request_id=request_id,
                   response_time=response_time,
                   tokens_used=tokens_used,
                   status_code=response.status_code)
        
        # Log conversation response in debug mode
        if config.get_logging_debug():
            response_content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            logger.debug("OpenRouter response content",
                        model_id=model,
                        request_id=request_id,
                        response_content=response_content,
                        full_response=result)
        
        return result
        
    except requests.RequestException as e:
        response_time = time.time() - start_time
        status_code = getattr(response, 'status_code', None) if 'response' in locals() else None
        
        # Handle specific HTTP status codes
        if status_code == 429:
            # Rate limit exceeded - this might indicate why key gets disabled
            logger.warning("OpenRouter rate limit exceeded - backing off",
                          model_id=model,
                          request_id=request_id,
                          response_time=response_time,
                          retry_after=getattr(response, 'headers', {}).get('retry-after', 'unknown'))
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
        
        elif status_code == 401:
            # API key issues
            logger.error("OpenRouter API authentication failed - check API key",
                        model_id=model,
                        request_id=request_id,
                        api_key_prefix=OPENROUTER_API_KEY[:10] + "..." if OPENROUTER_API_KEY else "None")
            raise HTTPException(status_code=401, detail="API authentication failed")
        
        elif status_code == 403:
            # API key disabled/banned
            logger.critical("OpenRouter API key disabled or banned",
                           model_id=model,
                           request_id=request_id,
                           response_time=response_time)
            raise HTTPException(status_code=403, detail="API key disabled or insufficient permissions")
        
        logger.error("OpenRouter API request failed",
                    model_id=model,
                    request_id=request_id,
                    response_time=response_time,
                    status_code=status_code,
                    error=str(e))
        
        if status_code:
            logger.error("API error response details",
                        request_id=request_id,
                        status_code=status_code,
                        response_text=getattr(response, 'text', 'N/A')[:1000],
                        response_headers=dict(getattr(response, 'headers', {})))
        
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")
        
    except Exception as e:
        response_time = time.time() - start_time
        logger.error("Unexpected error in OpenRouter call",
                    model_id=model,
                    request_id=request_id,
                    response_time=response_time,
                    error=str(e),
                    error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Round TAIble Backend API", "status": "running"}

@app.get("/health")
async def health_check():
    logger.debug("Health check requested")
    queue_stats = llm_queue.get_queue_stats()
    ws_stats = debate_manager.get_connection_stats()
    
    health_data = {
        "status": "healthy", 
        "models": list(MODELS.keys()),
        "active_debates": ws_stats['active_debates'],
        "queue_active": queue_stats['active_requests'],
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info("Health check completed", **health_data)
    return health_data

@app.post("/chat/completions")
async def chat_completions(request: dict):
    """Main chat endpoint that interfaces with OpenRouter"""
    start_time = time.time()
    model = request.get("model")
    
    logger.info("Chat completion request started", model_id=model)
    
    if not model or model not in MODELS:
        logger.warning("Invalid model requested", model_id=model, available_models=list(MODELS.keys()))
        raise HTTPException(status_code=400, detail=f"Model {model} not supported")
    
    openrouter_model = MODELS[model]
    messages = request.get("messages", [])
    
    try:
        response = call_openrouter(
            model=openrouter_model,
            messages=messages,
            max_tokens=request.get("max_tokens", 1000),
            temperature=request.get("temperature", 0.7),
            stream=request.get("stream", False)
        )
        
        response_time = time.time() - start_time
        performance_metrics.log_api_call(
            endpoint="/chat/completions",
            method="POST",
            response_time=response_time,
            status_code=200,
            model_id=model
        )
        
        return response
        
    except Exception as e:
        response_time = time.time() - start_time
        logger.error("Chat completion failed",
                    model_id=model,
                    response_time=response_time,
                    error=str(e))
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
    
    logger.info("Continuing debate with participants", 
               debate_id=debate_id,
               participant_count=len(models),
               participants=models,
               topic=topic,
               message_count=len(recent_messages))
    
    # Fetch model information from the database
    model_info = {}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{FRONTEND_URL}/api/models")
            if response.status_code == 200:
                model_info = response.json()
    except Exception as e:
        print(f"Warning: Could not fetch model info from database: {e}")
    
    # Debug: Log dei recent_messages per verificare il formato
    logger.info("Analyzing recent messages for turn determination",
               debate_id=debate_id,
               total_recent_messages=len(recent_messages),
               recent_messages_sample=recent_messages[-2:] if len(recent_messages) > 1 else recent_messages)
    
    # Determine next model to speak (strict round-robin alternation)
    last_speaker = recent_messages[-1].get('model') if recent_messages else None
    next_model = None
    
    logger.info("Turn analysis",
               debate_id=debate_id,
               last_speaker=last_speaker,
               available_models=models,
               last_message_full=recent_messages[-1] if recent_messages else None)
    
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
    logger.info("Next debate speaker determined",
               debate_id=debate_id,
               next_speaker=next_model,
               mapped_model=model_name,
               last_speaker=last_speaker)
    
    if model_name not in MODELS:
        logger.error("Model not found in available models",
                    model_id=model_name,
                    available_models=list(MODELS.keys()),
                    debate_id=debate_id)
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
    
    logger.info("Priority change requested",
               request_id=request_id,
               new_priority=priority)
    
    try:
        priority_enum = MessagePriority[priority.upper()]
        # Logica per aggiornare priorità se implementata
        return {"status": "updated", "request_id": request_id, "priority": priority}
    except KeyError:
        logger.warning("Invalid priority level requested",
                      request_id=request_id,
                      invalid_priority=priority,
                      valid_priorities=list(MessagePriority.__members__.keys()))
        raise HTTPException(status_code=400, detail="Invalid priority level")

@app.get("/logs/recent")
async def get_recent_logs(lines: int = 100):
    """Endpoint per ottenere log recenti"""
    import os
    
    log_file = "./logs/backend.log"
    
    if not os.path.exists(log_file):
        logger.warning("Log file not found", log_file=log_file)
        return {"logs": [], "message": "No log file found"}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
        logger.debug("Recent logs requested",
                    requested_lines=lines,
                    returned_lines=len(recent_lines),
                    total_lines=len(all_lines))
        
        return {
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines)
        }
        
    except Exception as e:
        logger.error("Failed to read log file",
                    log_file=log_file,
                    error=str(e))
        raise HTTPException(status_code=500, detail="Failed to read logs")

@app.get("/system/metrics")
async def get_system_metrics():
    """Endpoint per metriche di sistema complete"""
    import psutil
    import os
    
    try:
        # Metriche sistema
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Metriche applicazione
        queue_stats = llm_queue.get_queue_stats()
        ws_stats = debate_manager.get_connection_stats()
        
        # Info processo
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()
        
        metrics = {
            "system": {
                "cpu_percent": cpu_percent,
                "memory_total_mb": memory.total // (1024*1024),
                "memory_used_mb": memory.used // (1024*1024),
                "memory_percent": memory.percent,
                "disk_total_gb": disk.total // (1024*1024*1024),
                "disk_used_gb": disk.used // (1024*1024*1024),
                "disk_percent": (disk.used / disk.total) * 100
            },
            "process": {
                "memory_rss_mb": process_memory.rss // (1024*1024),
                "memory_vms_mb": process_memory.vms // (1024*1024),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads()
            },
            "application": {
                "queue_stats": queue_stats,
                "websocket_stats": ws_stats,
                "models_available": len(MODELS)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.debug("System metrics requested", **metrics["system"])
        
        return metrics
        
    except Exception as e:
        logger.error("Failed to get system metrics",
                    error=str(e),
                    error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail="Failed to get system metrics")

if __name__ == "__main__":
    import uvicorn
    
    # Use configuration manager
    host = config.get_server_host()
    port = config.get_server_port()
    reload = config.get_server_reload()
    
    logger.info("Starting FastAPI server", 
               host=host, 
               port=port, 
               environment=config.get_environment(),
               reload=reload,
               models_available=len(MODELS))
    
    if reload:
        uvicorn.run("main:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port)