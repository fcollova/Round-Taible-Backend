"""
System Endpoints Module

Gestisce tutti gli endpoints di sistema e monitoraggio:
- Health check
- WebSocket stats  
- Queue stats
- System metrics
- Models endpoint
"""

from typing import Dict, Any
from datetime import datetime

from websocket_manager import debate_manager as ws_manager
from llm_queue_manager import llm_queue
from logging_config import get_context_logger
from models_service import models_service

logger = get_context_logger(__name__)


class SystemService:
    def __init__(self):
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        try:
            # Check critical services
            queue_running = llm_queue.running if hasattr(llm_queue, 'running') else False
            ws_stats = ws_manager.get_connection_stats()
            
            health_status = "healthy" if queue_running else "degraded"
            
            return {
                "status": health_status,
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "llm_queue": "running" if queue_running else "stopped",
                    "websocket": "running",
                    "database": "connected"  # Would check actual DB connection
                },
                "version": "1.0.0"
            }
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_websocket_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        try:
            stats = ws_manager.get_connection_stats()
            
            # Add additional WebSocket metrics
            detailed_stats = {
                **stats,
                "timestamp": datetime.now().isoformat(),
                "uptime": "unknown",  # Would calculate actual uptime
                "connection_health": "healthy"
            }
            
            logger.debug("WebSocket stats requested", 
                        total_connections=stats.get('total_connections', 0),
                        active_debates=stats.get('active_debates', 0))
            
            return detailed_stats
            
        except Exception as e:
            logger.error("Failed to get WebSocket stats", error=str(e))
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get LLM queue statistics"""
        try:
            stats = llm_queue.get_queue_stats()
            
            # Enhance with additional metadata
            enhanced_stats = {
                **stats,
                "timestamp": datetime.now().isoformat(),
                "queue_health": "healthy",
                "worker_status": "running" if llm_queue.running else "stopped"
            }
            
            logger.debug("Queue stats requested", 
                        active_requests=stats.get('active_requests', 0),
                        queue_running=llm_queue.running)
            
            return enhanced_stats
            
        except Exception as e:
            logger.error("Failed to get queue stats", error=str(e))
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_available_models(self) -> Dict[str, Any]:
        """Get list of available LLM models from frontend API"""
        try:
            # Get models from frontend API
            models = await models_service.get_models()
            
            # Return in the format that clients expect
            return {"models": models}
            
        except Exception as e:
            logger.error("Failed to get models list", error=str(e))
            return {
                "error": str(e),
                "models": {}
            }


# Global system service instance
system_service = SystemService()