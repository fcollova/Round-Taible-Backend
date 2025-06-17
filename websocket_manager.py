from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import json
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DebateWebSocketManager:
    def __init__(self):
        # debate_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # debate_id -> debate_state
        self.debate_states: Dict[str, dict] = {}
        
    async def connect(self, websocket: WebSocket, debate_id: str):
        await websocket.accept()
        
        if debate_id not in self.active_connections:
            self.active_connections[debate_id] = set()
        
        self.active_connections[debate_id].add(websocket)
        
        # Invia stato attuale del dibattito
        if debate_id in self.debate_states:
            await websocket.send_text(json.dumps({
                "type": "debate_state",
                "data": self.debate_states[debate_id]
            }))
        
        logger.info(f"Client connesso a dibattito {debate_id}. Totale: {len(self.active_connections[debate_id])}")
        
        # Aggiorna contatore viewer
        await self.update_viewer_count(debate_id)
    
    def disconnect(self, websocket: WebSocket, debate_id: str):
        if debate_id in self.active_connections:
            self.active_connections[debate_id].discard(websocket)
            if not self.active_connections[debate_id]:
                del self.active_connections[debate_id]
                # Cleanup stato dibattito se nessuno connesso
                if debate_id in self.debate_states:
                    del self.debate_states[debate_id]
        
        logger.info(f"Client disconnesso da dibattito {debate_id}")
        
        # Aggiorna contatore viewer
        asyncio.create_task(self.update_viewer_count(debate_id))
    
    async def broadcast_to_debate(self, debate_id: str, message: dict):
        if debate_id not in self.active_connections:
            return
        
        disconnected = set()
        message_json = json.dumps(message)
        
        for connection in self.active_connections[debate_id]:
            try:
                await connection.send_text(message_json)
            except WebSocketDisconnect:
                disconnected.add(connection)
            except Exception as e:
                logger.error(f"Errore invio messaggio WebSocket: {e}")
                disconnected.add(connection)
        
        # Rimuovi connessioni morte
        for conn in disconnected:
            self.active_connections[debate_id].discard(conn)
    
    async def update_debate_state(self, debate_id: str, state_update: dict):
        if debate_id not in self.debate_states:
            self.debate_states[debate_id] = {
                "status": "active",
                "participants": [],
                "message_count": 0,
                "viewers": 0,
                "last_speaker": None
            }
        
        self.debate_states[debate_id].update(state_update)
        
        await self.broadcast_to_debate(debate_id, {
            "type": "state_update",
            "data": state_update,
            "timestamp": datetime.now().isoformat()
        })
    
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
        
        if action_type == "heartbeat":
            # Risposta heartbeat
            return {"type": "heartbeat_ack"}
        
        elif action_type == "status_change":
            # Propaga cambio stato
            await self.update_debate_state(debate_id, {
                "status": action.get("status")
            })
        
        elif action_type == "vote":
            # Propaga voto in real-time
            await self.send_vote_update(
                debate_id, 
                action.get("message_id"), 
                action.get("votes")
            )
        
        elif action_type == "debate_started":
            await self.update_debate_state(debate_id, {
                "status": "live",
                "started_at": action.get("timestamp")
            })
        
        return {"type": "action_processed", "action": action_type}
    
    def get_connection_stats(self):
        """Restituisce statistiche connessioni"""
        total_connections = sum(len(connections) for connections in self.active_connections.values())
        active_debates = len(self.active_connections)
        
        return {
            "total_connections": total_connections,
            "active_debates": active_debates,
            "debates": {
                debate_id: len(connections) 
                for debate_id, connections in self.active_connections.items()
            }
        }

# Istanza globale
debate_manager = DebateWebSocketManager()