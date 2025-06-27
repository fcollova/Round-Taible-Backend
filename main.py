"""
Round TAIble Backend - Refactored Main Application

Questo file contiene solo la configurazione dell'app FastAPI e la registrazione degli endpoints.
Tutta la logica business è stata spostata nei moduli dedicati:

- debate_manager.py: Gestione dibattiti
- admin_endpoints.py: Operazioni amministrative  
- system_endpoints.py: Monitoraggio e sistema
- openrouter_client.py: Client OpenRouter
- websocket_manager.py: Gestione WebSocket (esistente)
- llm_queue_manager.py: Gestione coda LLM (esistente)
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any
import time
import json
from datetime import datetime

# Import dei moduli esistenti
from websocket_manager import debate_manager as ws_manager
from llm_queue_manager import llm_queue
from logging_config import setup_logging, get_context_logger, performance_metrics
from config_manager import get_config, ConfigurationError

# Import dei nuovi moduli refactorizzati
from debate_manager import get_debate_service
from admin_endpoints import get_admin_service
from system_endpoints import system_service
from openrouter_client import get_openrouter_client

# Initialize configuration
try:
    config = get_config()
    logger_initialized = False
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    exit(1)

# Setup logging
if not logger_initialized:
    setup_logging(
        debug=config.get_logging_debug(),
        level=config.get_logging_level(),
        console_output=config.get_logging_console_output(),
        file_output=config.get_logging_file_output(),
        log_file=config.get_logging_file_path(),
        module_levels=config.get_module_logging_levels()
    )
    logger_initialized = True

logger = get_context_logger(__name__)

# Configuration
FRONTEND_URL = config.get_frontend_url()
OPENROUTER_API_KEY = config.get_openrouter_api_key()
OPENROUTER_BASE_URL = config.get_openrouter_base_url()
TIMEOUT = config.get_openrouter_timeout()

# Models are now loaded dynamically from frontend API

# Initialize services  
debate_service = get_debate_service(FRONTEND_URL)
admin_service = get_admin_service(FRONTEND_URL)
openrouter_client = get_openrouter_client(OPENROUTER_API_KEY, OPENROUTER_BASE_URL, TIMEOUT)

logger.info("Application initialized", 
           frontend_url=FRONTEND_URL,
           environment=config.get_environment())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Round TAIble Backend")
    await llm_queue.start(num_workers=2)
    yield
    # Shutdown
    logger.info("Shutting down Round TAIble Backend")
    await llm_queue.stop()


# FastAPI app
app = FastAPI(
    title="Round TAIble Backend", 
    version="1.0.0", 
    description="AI Debate Platform Backend - Refactored Architecture",
    lifespan=lifespan
)


# Middleware
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        logger.info("HTTP request started",
                   method=request.method,
                   url=str(request.url),
                   client_ip=request.client.host if request.client else "unknown",
                   user_agent=request.headers.get("user-agent", "unknown"))
        
        try:
            response = await call_next(request)
            response_time = time.time() - start_time
            
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

# CORS
allowed_origins = config.get_cors_allowed_origins()
if not allowed_origins:
    allowed_origins = ['http://localhost:3000']

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Middleware configured", origins=allowed_origins)


# ========================================
# BASIC ENDPOINTS
# ========================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Round TAIble Backend API - Refactored",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return await system_service.health_check()


# ========================================
# OPENROUTER ENDPOINTS  
# ========================================

@app.options("/chat/completions")
async def chat_completions_options():
    """Handle CORS preflight for chat/completions"""
    return {"status": "ok"}

@app.post("/chat/completions")
async def chat_completions(request: Request):
    """OpenRouter-compatible chat completions endpoint"""
    try:
        request_data = await request.json()
        model = request_data.get("model")
        messages = request_data.get("messages", [])
        
        if not model or not messages:
            raise HTTPException(status_code=400, detail="Model and messages are required")
        
        # Extract additional parameters
        kwargs = {k: v for k, v in request_data.items() if k not in ["model", "messages"]}
        
        logger.info("Chat completion requested", 
                   model=model,
                   message_count=len(messages),
                   max_tokens=kwargs.get('max_tokens', 'default'))
        
        # Use OpenRouter client
        result = await openrouter_client.chat_completion(model, messages, **kwargs)
        
        return result
        
    except Exception as e:
        logger.error("Chat completion failed", error=str(e), model=request.get("model"))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset-model-state/{model_id}")
async def reset_model_state(model_id: str):
    """Reset the state of a specific model (useful when model is blocked due to errors)"""
    try:
        openrouter_client.reset_model_state(model_id)
        logger.info("Model state reset", model=model_id)
        return {"status": "success", "message": f"Model {model_id} state has been reset"}
    except Exception as e:
        logger.error("Failed to reset model state", model=model_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# DEBATE ENDPOINTS
# ========================================

@app.options("/debate/continue")
async def debate_continue_options():
    """Handle CORS preflight for debate/continue"""
    return {"status": "ok"}

@app.post("/debate/continue")
async def continue_debate(request: Request):
    """Continue an ongoing debate"""
    request_data = await request.json()
    return await debate_service.continue_debate(request_data)


# ========================================
# WEBSOCKET ENDPOINTS
# ========================================

@app.websocket("/ws/debates/{debate_id}")
async def websocket_endpoint(websocket: WebSocket, debate_id: str):
    """WebSocket endpoint for real-time debate communication"""
    connection_established = False
    try:
        await ws_manager.connect(websocket, debate_id)
        connection_established = True
        logger.info("WebSocket connected", 
                   debate_id=debate_id,
                   client_ip=websocket.client.host if websocket.client else "unknown")
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await ws_manager.handle_user_action(debate_id, message)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON received via WebSocket",
                             debate_id=debate_id,
                             error=str(e))
                continue
                
    except WebSocketDisconnect as e:
        logger.info("WebSocket disconnected",
                   debate_id=debate_id,
                   disconnect_code=e.code,
                   reason=e.reason if hasattr(e, 'reason') else 'unknown')
    except Exception as e:
        logger.error("WebSocket error",
                    debate_id=debate_id,
                    error=str(e),
                    error_type=type(e).__name__)
    finally:
        # Only cleanup if connection was established
        if connection_established:
            try:
                await ws_manager.disconnect(websocket, debate_id)
            except Exception as cleanup_error:
                logger.warning("Error during WebSocket cleanup",
                              debate_id=debate_id,
                              error=str(cleanup_error))


# ========================================
# MONITORING ENDPOINTS
# ========================================

@app.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket statistics"""
    return await system_service.get_websocket_stats()


@app.get("/queue/stats")
async def get_queue_stats():
    """Get LLM queue statistics"""
    return await system_service.get_queue_stats()


@app.get("/models")
async def get_models():
    """Get available models"""
    return await system_service.get_available_models()


@app.get("/system/metrics")
async def get_system_metrics():
    """Get detailed system metrics"""
    return await admin_service.get_system_metrics()


# ========================================
# ADMIN ENDPOINTS
# ========================================

@app.get("/admin/status")
async def get_admin_status():
    """Get system status for admin dashboard"""
    return await admin_service.get_system_status()


@app.post("/admin/debate/pause")
async def admin_pause_debate(request: Request, admin_request: dict):
    """Pause a debate (admin only)"""
    return await admin_service.pause_debate(request, admin_request)


@app.post("/admin/debate/resume")
async def admin_resume_debate(request: Request, admin_request: dict):
    """Resume a debate (admin only)"""
    return await admin_service.resume_debate(request, admin_request)


@app.post("/admin/debate/finish")
async def admin_finish_debate(request: Request, admin_request: dict):
    """Finish a debate (admin only)"""
    return await admin_service.finish_debate(request, admin_request)


@app.post("/admin/debate/delete")
async def admin_delete_debate(request: Request, admin_request: dict):
    """Delete a debate (admin only)"""
    return await admin_service.delete_debate(request, admin_request)


@app.post("/admin/debate/permanent_delete")
async def admin_permanent_delete_debate(request: Request, admin_request: dict):
    """Permanently delete a debate (admin only)"""
    return await admin_service.permanent_delete_debate(request, admin_request)


@app.post("/queue/priority")
async def set_queue_priority(priority_request: dict):
    """Set priority for LLM queue (admin only)"""
    return await admin_service.set_queue_priority(priority_request)


@app.get("/logs/recent")
async def get_recent_logs(limit: int = 100):
    """Get recent system logs (admin only)"""
    return await admin_service.get_recent_logs(limit)


@app.post("/ws/admin/broadcast")
async def admin_broadcast(admin_request: dict):
    """Send admin broadcast message"""
    return await admin_service.admin_broadcast(admin_request)


# ========================================
# STARTUP MESSAGE
# ========================================

logger.info("Round TAIble Backend started successfully",
           endpoints_registered=len(app.routes),
           services=["debate_manager", "admin_service", "system_service", "openrouter_client"],
           frontend_url=FRONTEND_URL)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)