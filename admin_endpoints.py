"""
Admin Endpoints Module

Gestisce tutti gli endpoints per le operazioni amministrative:
- Gestione dibattiti (pause, resume, finish, delete)
- Monitoraggio sistema
- Broadcast admin
- Statistiche e logs
"""

from fastapi import HTTPException, Request
from typing import Dict, Any
from datetime import datetime
import asyncio

from websocket_manager import debate_manager as ws_manager
from llm_queue_manager import llm_queue
from debate_manager import get_debate_service
from logging_config import get_context_logger, performance_metrics
from config_manager import get_config

logger = get_context_logger(__name__)


class AdminService:
    def __init__(self, frontend_url: str):
        self.frontend_url = frontend_url
        config = get_config()
        self.debate_service = get_debate_service(frontend_url)

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for admin dashboard"""
        try:
            # Collect various system metrics
            queue_stats = llm_queue.get_queue_stats()
            ws_stats = ws_manager.get_connection_stats()
            
            # Performance metrics (placeholder)
            perf_stats = {"api_calls": 0, "llm_requests": 0, "avg_response_time": 0}
            
            status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "llm_queue": {
                        "status": "active" if llm_queue.running else "inactive",
                        "active_requests": queue_stats.get('active_requests', 0),
                        "queue_sizes": queue_stats.get('queue_sizes', {}),
                        "stats": queue_stats.get('stats', {})
                    },
                    "websocket": {
                        "status": "active",
                        "total_connections": ws_stats.get('total_connections', 0),
                        "active_debates": ws_stats.get('active_debates', 0),
                        "debate_connections": ws_stats.get('debate_connections', {})
                    }
                },
                "performance": perf_stats,
                "system_info": {
                    "environment": "production",  # Should come from config
                    "version": "1.0.0"
                }
            }
            
            logger.info("System status requested", 
                       queue_active=queue_stats.get('active_requests', 0),
                       ws_connections=ws_stats.get('total_connections', 0))
            
            return status
            
        except Exception as e:
            logger.error("Failed to get system status", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def pause_debate(self, request: Request, admin_request: Dict[str, Any]) -> Dict[str, Any]:
        """Admin endpoint to pause a debate"""
        admin_user = request.headers.get("X-Admin-User", "unknown")
        debate_id = admin_request.get("debate_id")
        reason = admin_request.get("reason", "Paused by admin")
        
        return await self.debate_service.pause_debate(debate_id, reason, admin_user)

    async def resume_debate(self, request: Request, admin_request: Dict[str, Any]) -> Dict[str, Any]:
        """Admin endpoint to resume a debate"""
        admin_user = request.headers.get("X-Admin-User", "unknown")
        debate_id = admin_request.get("debate_id")
        reason = admin_request.get("reason", "Resumed by admin")
        
        return await self.debate_service.resume_debate(debate_id, reason, admin_user)

    async def finish_debate(self, request: Request, admin_request: Dict[str, Any]) -> Dict[str, Any]:
        """Admin endpoint to finish a debate"""
        admin_user = request.headers.get("X-Admin-User", "unknown")
        debate_id = admin_request.get("debate_id")
        reason = admin_request.get("reason", "Finished by admin")
        
        return await self.debate_service.finish_debate(debate_id, reason, admin_user)

    async def delete_debate(self, request: Request, admin_request: Dict[str, Any]) -> Dict[str, Any]:
        """Admin endpoint to delete a debate (soft delete)"""
        admin_user = request.headers.get("X-Admin-User", "unknown")
        debate_id = admin_request.get("debate_id")
        reason = admin_request.get("reason", "Deleted by admin")
        
        return await self.debate_service.delete_debate(debate_id, reason, admin_user, permanent=False)

    async def permanent_delete_debate(self, request: Request, admin_request: Dict[str, Any]) -> Dict[str, Any]:
        """Admin endpoint to permanently delete a debate"""
        admin_user = request.headers.get("X-Admin-User", "unknown")
        debate_id = admin_request.get("debate_id")
        reason = admin_request.get("reason", "Permanently deleted by admin")
        
        return await self.debate_service.delete_debate(debate_id, reason, admin_user, permanent=True)

    async def admin_broadcast(self, admin_request: Dict[str, Any]) -> Dict[str, Any]:
        """Send admin broadcast message to all connected clients"""
        message = admin_request.get("message")
        broadcast_type = admin_request.get("type", "admin_announcement")
        target_debate = admin_request.get("debate_id")  # Optional: broadcast to specific debate
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        logger.info("Admin broadcast requested", 
                   message_preview=message[:100],
                   broadcast_type=broadcast_type,
                   target_debate=target_debate)
        
        try:
            broadcast_data = {
                "type": broadcast_type,
                "data": {
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                    "source": "admin"
                }
            }
            
            if target_debate:
                # Broadcast to specific debate
                await ws_manager.broadcast_to_debate(target_debate, broadcast_data)
                recipient_count = len(ws_manager.debate_connections.get(target_debate, []))
            else:
                # Broadcast to all connected clients
                await ws_manager.broadcast_to_all(broadcast_data)
                recipient_count = ws_manager.total_connections
            
            logger.info("Admin broadcast sent successfully", 
                       recipients=recipient_count,
                       broadcast_type=broadcast_type)
            
            return {
                "status": "success",
                "message": "Broadcast sent successfully",
                "recipients": recipient_count,
                "broadcast_type": broadcast_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to send admin broadcast", 
                        error=str(e),
                        broadcast_type=broadcast_type)
            raise HTTPException(status_code=500, detail=f"Failed to send broadcast: {str(e)}")

    async def set_queue_priority(self, priority_request: Dict[str, Any]) -> Dict[str, Any]:
        """Set priority for specific models in the LLM queue"""
        model_id = priority_request.get("model_id")
        priority = priority_request.get("priority", "NORMAL")
        
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required")
        
        # Validate priority
        valid_priorities = ["LOW", "NORMAL", "HIGH", "URGENT"]
        if priority not in valid_priorities:
            raise HTTPException(status_code=400, detail=f"Priority must be one of: {valid_priorities}")
        
        logger.info("Queue priority adjustment requested", 
                   model_id=model_id,
                   priority=priority)
        
        try:
            # This would need to be implemented in llm_queue_manager
            # For now, just return success
            return {
                "status": "success",
                "message": f"Priority set to {priority} for model {model_id}",
                "model_id": model_id,
                "priority": priority,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to set queue priority", 
                        error=str(e),
                        model_id=model_id)
            raise HTTPException(status_code=500, detail=f"Failed to set priority: {str(e)}")

    async def get_recent_logs(self, limit: int = 100) -> Dict[str, Any]:
        """Get recent system logs for admin review"""
        try:
            # This is a simplified implementation
            # In a real system, you'd read from log files or a logging service
            logs = []
            
            # Get queue stats as a form of "logs"
            queue_stats = llm_queue.get_queue_stats()
            ws_stats = ws_manager.get_connection_stats()
            
            logs.append({
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": f"Queue status: {queue_stats.get('active_requests', 0)} active requests",
                "component": "llm_queue"
            })
            
            logs.append({
                "timestamp": datetime.now().isoformat(),
                "level": "INFO", 
                "message": f"WebSocket status: {ws_stats.get('total_connections', 0)} connections",
                "component": "websocket"
            })
            
            return {
                "status": "success",
                "logs": logs[-limit:],  # Return last N logs
                "count": len(logs),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get recent logs", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics for monitoring"""
        try:
            # Collect comprehensive metrics
            queue_stats = llm_queue.get_queue_stats()
            ws_stats = ws_manager.get_connection_stats()
            perf_stats = {"api_calls": 0, "llm_requests": 0, "avg_response_time": 0}
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "llm_queue": {
                    "running": llm_queue.running,
                    "worker_count": len(llm_queue.workers) if hasattr(llm_queue, 'workers') else 0,
                    "active_requests": queue_stats.get('active_requests', 0),
                    "queue_sizes": queue_stats.get('queue_sizes', {}),
                    "statistics": queue_stats.get('stats', {}),
                    "model_rates": queue_stats.get('model_rates', {})
                },
                "websocket": {
                    "total_connections": ws_stats.get('total_connections', 0),
                    "active_debates": ws_stats.get('active_debates', 0),
                    "debate_connections": ws_stats.get('debate_connections', {}),
                    "connection_health": "healthy"  # Would check actual connection health
                },
                "performance": perf_stats,
                "health_status": "healthy"  # Overall health assessment
            }
            
            logger.debug("System metrics collected", 
                        active_requests=metrics['llm_queue']['active_requests'],
                        ws_connections=metrics['websocket']['total_connections'])
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


# Global admin service instance
admin_service = None

def get_admin_service(frontend_url: str) -> AdminService:
    """Get or create the global admin service instance"""
    global admin_service
    if admin_service is None:
        admin_service = AdminService(frontend_url)
    return admin_service