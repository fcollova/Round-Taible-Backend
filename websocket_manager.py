from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import json
import asyncio
from datetime import datetime
import logging
from logging_config import get_context_logger, performance_metrics

logger = get_context_logger(__name__)

class DebateWebSocketManager:
    def __init__(self):
        # debate_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # debate_id -> debate_state
        self.debate_states: Dict[str, dict] = {}
        
    async def connect(self, websocket: WebSocket, debate_id: str):
        try:
            await websocket.accept()
            
            if debate_id not in self.active_connections:
                self.active_connections[debate_id] = set()
                logger.info("New debate WebSocket session started", debate_id=debate_id)
            
            self.active_connections[debate_id].add(websocket)
            connection_count = len(self.active_connections[debate_id])
            
            # Invia stato attuale del dibattito
            if debate_id in self.debate_states:
                state_message = {
                    "type": "debate_state",
                    "data": self.debate_states[debate_id]
                }
                await websocket.send_text(json.dumps(state_message))
                logger.debug("Sent current debate state to new connection", 
                           debate_id=debate_id,
                           state_keys=list(self.debate_states[debate_id].keys()))
            
            logger.info("Client connected to debate WebSocket", 
                       debate_id=debate_id,
                       total_connections=connection_count,
                       websocket_id=id(websocket))
            
            # Aggiorna contatore viewer
            await self.update_viewer_count(debate_id)
            
            # Log metriche generali
            performance_metrics.log_websocket_metrics(
                total_connections=sum(len(conns) for conns in self.active_connections.values()),
                active_debates=len(self.active_connections),
                debate_id=debate_id
            )
            
        except Exception as e:
            logger.error("Failed to establish WebSocket connection",
                        debate_id=debate_id,
                        error=str(e),
                        error_type=type(e).__name__)
            raise
    
    async def disconnect(self, websocket: WebSocket, debate_id: str):
        websocket_id = id(websocket)
        was_connected = False
        
        if debate_id in self.active_connections:
            if websocket in self.active_connections[debate_id]:
                was_connected = True
                self.active_connections[debate_id].discard(websocket)
                
                remaining_connections = len(self.active_connections[debate_id])
                
                if not self.active_connections[debate_id]:
                    del self.active_connections[debate_id]
                    logger.info("Last client disconnected, cleaning up debate session",
                              debate_id=debate_id)
                    
                    # Cleanup stato dibattito se nessuno connesso
                    if debate_id in self.debate_states:
                        del self.debate_states[debate_id]
                        logger.debug("Debate state cleaned up", debate_id=debate_id)
                else:
                    logger.info("Client disconnected from debate WebSocket",
                              debate_id=debate_id,
                              websocket_id=websocket_id,
                              remaining_connections=remaining_connections)
        
        if was_connected:
            # Aggiorna contatore viewer
            await self.update_viewer_count(debate_id)
            
            # Log metriche aggiornate
            await self._log_disconnect_metrics(debate_id)
        else:
            logger.warning("Attempted to disconnect non-existent WebSocket",
                          debate_id=debate_id,
                          websocket_id=websocket_id)
    
    async def _log_disconnect_metrics(self, debate_id: str):
        """Log metriche dopo disconnessione"""
        try:
            performance_metrics.log_websocket_metrics(
                total_connections=sum(len(conns) for conns in self.active_connections.values()),
                active_debates=len(self.active_connections),
                debate_id=debate_id
            )
        except Exception as e:
            logger.error("Failed to log disconnect metrics", error=str(e))
    
    async def broadcast_to_debate(self, debate_id: str, message: dict):
        if debate_id not in self.active_connections:
            logger.debug("No connections for debate broadcast", debate_id=debate_id)
            return
        
        connection_count = len(self.active_connections[debate_id])
        disconnected = set()
        message_json = json.dumps(message)
        message_type = message.get('type', 'unknown')
        
        logger.debug("Broadcasting message to debate connections",
                    debate_id=debate_id,
                    message_type=message_type,
                    connection_count=connection_count)
        
        successful_sends = 0
        for connection in self.active_connections[debate_id]:
            try:
                await connection.send_text(message_json)
                successful_sends += 1
            except WebSocketDisconnect:
                logger.debug("WebSocket disconnected during broadcast",
                           debate_id=debate_id,
                           websocket_id=id(connection))
                disconnected.add(connection)
            except Exception as e:
                logger.error("Error sending WebSocket message",
                           debate_id=debate_id,
                           websocket_id=id(connection),
                           error=str(e),
                           error_type=type(e).__name__)
                disconnected.add(connection)
        
        # Rimuovi connessioni morte
        for conn in disconnected:
            self.active_connections[debate_id].discard(conn)
        
        if disconnected:
            logger.info("Removed dead WebSocket connections",
                       debate_id=debate_id,
                       disconnected_count=len(disconnected),
                       remaining_count=len(self.active_connections[debate_id]))
        
        logger.debug("Broadcast completed",
                    debate_id=debate_id,
                    message_type=message_type,
                    successful_sends=successful_sends,
                    failed_sends=len(disconnected))
    
    async def update_debate_state(self, debate_id: str, state_update: dict):
        was_new_state = debate_id not in self.debate_states
        
        if was_new_state:
            self.debate_states[debate_id] = {
                "status": "active",
                "participants": [],
                "message_count": 0,
                "viewers": 0,
                "last_speaker": None
            }
            logger.info("New debate state initialized", debate_id=debate_id)
        
        previous_state = self.debate_states[debate_id].copy()
        self.debate_states[debate_id].update(state_update)
        
        logger.debug("Debate state updated",
                    debate_id=debate_id,
                    updated_fields=list(state_update.keys()),
                    new_state=self.debate_states[debate_id])
        
        message = {
            "type": "state_update",
            "data": state_update,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast_to_debate(debate_id, message)
    
    async def update_viewer_count(self, debate_id: str):
        viewer_count = len(self.active_connections.get(debate_id, set()))
        
        await self.update_debate_state(debate_id, {
            "viewers": viewer_count
        })
    
    async def send_new_message(self, debate_id: str, message: dict):
        await self.broadcast_to_debate(debate_id, {
            "type": "new_message",
            "data": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Aggiorna contatore messaggi
        if debate_id in self.debate_states:
            current_count = self.debate_states[debate_id].get("message_count", 0)
            last_speaker = None
            
            # Determina last_speaker in base al tipo di messaggio
            if message.get("type") == "moderator":
                last_speaker = "moderator"
            else:
                last_speaker = message.get("ai") or message.get("modelId")
            
            await self.update_debate_state(debate_id, {
                "message_count": current_count + 1,
                "last_speaker": last_speaker
            })
    
    async def send_vote_update(self, debate_id: str, message_id: str, votes: dict):
        await self.broadcast_to_debate(debate_id, {
            "type": "vote_update",
            "data": {
                "message_id": message_id,
                "votes": votes
            },
            "timestamp": datetime.now().isoformat()
        })
    
    async def handle_user_action(self, debate_id: str, action: dict):
        """Gestisce azioni utente ricevute via WebSocket"""
        action_type = action.get("type")
        
        logger.debug("Handling user action",
                    debate_id=debate_id,
                    action_type=action_type)
        
        try:
            if action_type == "heartbeat":
                # Risposta heartbeat
                logger.debug("Heartbeat received", debate_id=debate_id)
                return {"type": "heartbeat_ack"}
            
            elif action_type == "status_change":
                # Propaga cambio stato
                new_status = action.get("status")
                logger.info("Debate status change requested",
                           debate_id=debate_id,
                           new_status=new_status)
                
                await self.update_debate_state(debate_id, {
                    "status": new_status
                })
            
            elif action_type == "vote":
                # Propaga voto in real-time
                message_id = action.get("message_id")
                votes = action.get("votes")
                
                logger.info("Vote update received",
                           debate_id=debate_id,
                           message_id=message_id,
                           vote_counts=votes)
                
                await self.send_vote_update(debate_id, message_id, votes)
            
            elif action_type == "debate_started":
                timestamp = action.get("timestamp")
                logger.info("Debate started event",
                           debate_id=debate_id,
                           started_at=timestamp)
                
                await self.update_debate_state(debate_id, {
                    "status": "live",
                    "started_at": timestamp
                })
            
            elif action_type == "new_message":
                # Messaggio del moderatore da propagare
                message_data = action.get("data", {})
                message_type = message_data.get("type", "ai")
                
                logger.info("New message broadcast received",
                           debate_id=debate_id,
                           message_type=message_type,
                           has_content=bool(message_data.get("content")))
                
                # Propaga il messaggio a tutti i client connessi
                await self.send_new_message(debate_id, message_data)
            
            elif action_type == "sync_debate_state":
                # Il frontend invia lo stato del dibattito per sincronizzazione
                debate_data = action.get("debate", {})
                logger.debug("Debate state sync received",
                           debate_id=debate_id,
                           debate_status=debate_data.get("status"))
                
                # Aggiorna lo stato locale se necessario
                if debate_data:
                    await self.update_debate_state(debate_id, {
                        "status": debate_data.get("status", "unknown"),
                        "participants": debate_data.get("participants", []),
                        "viewers": debate_data.get("viewers", 0)
                    })
            
            else:
                logger.warning("Unknown user action type",
                              debate_id=debate_id,
                              action_type=action_type)
            
            return {"type": "action_processed", "action": action_type}
            
        except Exception as e:
            logger.error("Error handling user action",
                        debate_id=debate_id,
                        action_type=action_type,
                        error=str(e),
                        error_type=type(e).__name__)
            return {"type": "action_error", "action": action_type, "error": str(e)}
    
    def get_connection_stats(self):
        """Restituisce statistiche connessioni con logging"""
        total_connections = sum(len(connections) for connections in self.active_connections.values())
        active_debates = len(self.active_connections)
        
        debate_stats = {
            debate_id: len(connections) 
            for debate_id, connections in self.active_connections.items()
        }
        
        stats = {
            "total_connections": total_connections,
            "active_debates": active_debates,
            "debates": debate_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.debug("WebSocket connection stats requested", **stats)
        
        return stats

    def disconnect_all_from_debate(self, debate_id: str):
        """Disconnect all WebSocket connections from a specific debate (admin action)"""
        if debate_id not in self.active_connections:
            logger.warning("Attempted to disconnect all from non-existent debate", 
                         debate_id=debate_id)
            return
        
        connections = self.active_connections[debate_id].copy()
        connection_count = len(connections)
        
        logger.info("Disconnecting all connections from debate", 
                   debate_id=debate_id, 
                   connection_count=connection_count)
        
        # Send final notification to all connected clients before disconnection
        disconnect_message = {
            "type": "admin_disconnect",
            "message": "This debate has been closed by an administrator",
            "debate_id": debate_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send disconnect notification and close all connections
        for websocket in connections:
            try:
                # Send final message
                asyncio.create_task(websocket.send_text(json.dumps(disconnect_message)))
                # Close connection
                asyncio.create_task(websocket.close(code=1000, reason="Debate closed by admin"))
                logger.debug("Closed WebSocket connection for debate", 
                           debate_id=debate_id, 
                           websocket_id=id(websocket))
            except Exception as e:
                logger.warning("Error closing WebSocket connection", 
                             debate_id=debate_id, 
                             websocket_id=id(websocket),
                             error=str(e))
        
        # Clear all connections for this debate
        self.active_connections.pop(debate_id, None)
        
        logger.info("All connections disconnected from debate", 
                   debate_id=debate_id, 
                   disconnected_count=connection_count)

    async def broadcast_global(self, message: dict):
        """Broadcast a message to all connected clients across all debates"""
        total_sent = 0
        total_failed = 0
        
        logger.info("Starting global broadcast", 
                   total_debates=len(self.active_connections),
                   message_type=message.get("type", "unknown"))
        
        # Iterate through all debates and their connections
        for debate_id, connections in self.active_connections.items():
            for websocket in connections.copy():  # Use copy to avoid modification during iteration
                try:
                    await websocket.send_text(json.dumps(message))
                    total_sent += 1
                    logger.debug("Global message sent", 
                               debate_id=debate_id,
                               websocket_id=id(websocket))
                except Exception as e:
                    total_failed += 1
                    logger.warning("Failed to send global message", 
                                 debate_id=debate_id,
                                 websocket_id=id(websocket),
                                 error=str(e))
                    # Remove dead connection
                    connections.discard(websocket)
        
        logger.info("Global broadcast completed", 
                   messages_sent=total_sent,
                   messages_failed=total_failed,
                   message_type=message.get("type", "unknown"))
        
        return {
            "sent": total_sent,
            "failed": total_failed,
            "total_attempts": total_sent + total_failed
        }

    def remove_debate_state(self, debate_id: str):
        """Forcibly remove debate state from memory (admin action)"""
        if debate_id in self.debate_states:
            old_state = self.debate_states.pop(debate_id)
            logger.warning("Debate state forcibly removed by admin", 
                         debate_id=debate_id,
                         old_status=old_state.get("status", "unknown"),
                         state_keys=list(old_state.keys()))
        else:
            logger.warning("Attempted to remove non-existent debate state", 
                         debate_id=debate_id)
        
        # Also ensure no active connections remain
        if debate_id in self.active_connections:
            connection_count = len(self.active_connections[debate_id])
            self.active_connections.pop(debate_id, None)
            logger.info("Removed active connections during state removal", 
                       debate_id=debate_id,
                       removed_connections=connection_count)

# Istanza globale
debate_manager = DebateWebSocketManager()