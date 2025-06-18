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
    
    def disconnect(self, websocket: WebSocket, debate_id: str):
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
            asyncio.create_task(self.update_viewer_count(debate_id))
            
            # Log metriche aggiornate
            asyncio.create_task(self._log_disconnect_metrics(debate_id))
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
            await self.update_debate_state(debate_id, {
                "message_count": current_count + 1,
                "last_speaker": message.get("ai")
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
                    action_type=action_type,
                    user_action=action)
        
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

# Istanza globale
debate_manager = DebateWebSocketManager()