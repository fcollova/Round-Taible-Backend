"""
Debate Management Module

Gestisce tutte le operazioni relative ai dibattiti:
- Continuazione dibattiti 
- Operazioni admin (pause, resume, finish, delete)
- Logica di gestione del turno e speaker selection
"""

from fastapi import HTTPException
from typing import Dict, List, Any
import httpx
from datetime import datetime

from llm_queue_manager import llm_queue, MessageRequest, MessagePriority
from context_manager import context_manager
from websocket_manager import debate_manager as ws_manager
from logging_config import get_context_logger
from config_manager import get_config
from models_service import models_service

logger = get_context_logger(__name__)


class DebateManager:
    def __init__(self, frontend_url: str, frontend_timeout: int = 10):
        self.frontend_url = frontend_url.rstrip('/')
        self.frontend_timeout = frontend_timeout

    async def continue_debate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate next message in ongoing debate"""
        debate_id = request.get('debate_id')
        models = request.get('models', [])
        topic = request.get('topic')
        recent_messages = request.get('recent_messages', [])
        starting_model = request.get('starting_model')
        
        if not debate_id or not models or not topic:
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        logger.info("Continuing debate with participants", 
                   debate_id=debate_id,
                   participant_count=len(models),
                   participants=models,
                   topic=topic,
                   message_count=len(recent_messages))
        
        # Model information is now handled directly via models_service
        
        # Debug: Log dei recent_messages per verificare il formato
        logger.info("Analyzing recent messages for turn determination",
                   debate_id=debate_id,
                   recent_messages_count=len(recent_messages),
                   recent_message_sample=[{
                       'id': msg.get('id', 'no-id'),
                       'modelId': msg.get('modelId', msg.get('ai', 'no-model')),
                       'turnNumber': msg.get('turnNumber', 'no-turn'),
                       'type': msg.get('type', 'no-type')
                   } for msg in recent_messages[:3]])
        
        # Recupera i messaggi reali dal database tramite API
        actual_messages = await self._get_debate_messages(debate_id)
        
        # Recupera informazioni del dibattito per il starting model (se non già fornito)
        if not starting_model:
            starting_model = await self._get_debate_starting_model(debate_id)
        
        logger.info("Retrieved actual messages from database",
                   debate_id=debate_id,
                   actual_messages_count=len(actual_messages),
                   frontend_messages_count=len(recent_messages),
                   starting_model=starting_model)
        
        # Usa context_manager per costruire il contesto con i messaggi reali
        debate_context = await context_manager.build_debate_context(
            debate_id=debate_id,
            topic=topic,
            recent_messages=actual_messages
        )
        
        # Determina chi deve parlare (round-robin dalla lista dei modelli)
        next_model_id = await self._determine_next_speaker(models, actual_messages, starting_model)
        
        # Ottieni informazioni complete del modello
        next_model_info = await models_service.get_model(next_model_id)
        if not next_model_info:
            raise HTTPException(status_code=400, detail=f"Model {next_model_id} not found")
        
        # Formatta il contesto per il prompt
        context = context_manager.format_context_for_prompt(debate_context)
        
        # Get personality from model info
        personality = next_model_info.get('description', f"AI model {next_model_info['name']}")
        
        # Crea richiesta per la coda LLM
        message_request = MessageRequest(
            debate_id=debate_id,
            model_id=next_model_info['openrouterId'],  # ID per OpenRouter
            context=context,
            topic=topic,
            personality=personality,
            message_count=debate_context.message_count,
            priority=MessagePriority.NORMAL
        )
        
        # Aggiungi informazioni del modello originale per il frontend
        message_request.original_model_id = next_model_id  # ID originale del database
        message_request.model_info = next_model_info  # Informazioni complete
        
        # Log context summary per debugging
        context_summary = context_manager.get_context_summary(debate_context)
        logger.debug("Debate context summary", **context_summary)
        
        # Enqueue la richiesta
        request_id = await llm_queue.enqueue_message(message_request)
        
        return {
            "status": "queued",
            "request_id": request_id,
            "debate_id": debate_id,
            "next_model": next_model_id,
            "next_model_name": next_model_info['name'],
            "message_count": debate_context.message_count,
            "context_messages": len(debate_context.messages)
        }

    async def pause_debate(self, debate_id: str, reason: str, admin_user: str) -> Dict[str, Any]:
        """Admin endpoint to pause an active debate"""
        logger.info("Admin pause debate requested", 
                   admin_user=admin_user, 
                   debate_id=debate_id, 
                   reason=reason)
        
        if not debate_id:
            raise HTTPException(status_code=400, detail="debate_id is required")
        
        try:
            # Update debate state to paused
            await ws_manager.update_debate_state(debate_id, {
                "status": "paused",
                "last_admin_action": datetime.now().isoformat(),
                "admin_user": admin_user,
                "admin_reason": reason
            })
            
            # Broadcast pause event to all connected clients
            await ws_manager.broadcast_to_debate(debate_id, {
                "type": "debate_paused",
                "data": {
                    "reason": reason,
                    "admin_user": admin_user,
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            logger.info("Debate paused successfully", 
                       debate_id=debate_id,
                       admin_user=admin_user)
            
            return {
                "status": "success",
                "action": "paused",
                "debate_id": debate_id,
                "admin_user": admin_user,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to pause debate",
                        debate_id=debate_id,
                        admin_user=admin_user,
                        error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to pause debate: {str(e)}")

    async def resume_debate(self, debate_id: str, reason: str, admin_user: str) -> Dict[str, Any]:
        """Admin endpoint to resume a paused debate"""
        logger.info("Admin resume debate requested", 
                   admin_user=admin_user, 
                   debate_id=debate_id, 
                   reason=reason)
        
        if not debate_id:
            raise HTTPException(status_code=400, detail="debate_id is required")
        
        try:
            # Update debate state to active
            await ws_manager.update_debate_state(debate_id, {
                "status": "active",
                "last_admin_action": datetime.now().isoformat(),
                "admin_user": admin_user,
                "admin_reason": reason
            })
            
            # Broadcast resume event to all connected clients
            await ws_manager.broadcast_to_debate(debate_id, {
                "type": "debate_resumed",
                "data": {
                    "reason": reason,
                    "admin_user": admin_user,
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            logger.info("Debate resumed successfully", 
                       debate_id=debate_id,
                       admin_user=admin_user)
            
            return {
                "status": "success",
                "action": "resumed",
                "debate_id": debate_id,
                "admin_user": admin_user,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to resume debate",
                        debate_id=debate_id,
                        admin_user=admin_user,
                        error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to resume debate: {str(e)}")

    async def finish_debate(self, debate_id: str, reason: str, admin_user: str) -> Dict[str, Any]:
        """Admin endpoint to finish an active debate"""
        logger.info("Admin finish debate requested", 
                   admin_user=admin_user, 
                   debate_id=debate_id, 
                   reason=reason)
        
        if not debate_id:
            raise HTTPException(status_code=400, detail="debate_id is required")
        
        try:
            # Update debate state to finished
            await ws_manager.update_debate_state(debate_id, {
                "status": "finished",
                "last_admin_action": datetime.now().isoformat(),
                "admin_user": admin_user,
                "admin_reason": reason
            })
            
            # Broadcast finish event to all connected clients
            await ws_manager.broadcast_to_debate(debate_id, {
                "type": "debate_finished",
                "data": {
                    "reason": reason,
                    "admin_user": admin_user,
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            logger.info("Debate finished successfully", 
                       debate_id=debate_id,
                       admin_user=admin_user)
            
            return {
                "status": "success",
                "action": "finished",
                "debate_id": debate_id,
                "admin_user": admin_user,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to finish debate",
                        debate_id=debate_id,
                        admin_user=admin_user,
                        error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to finish debate: {str(e)}")

    async def delete_debate(self, debate_id: str, reason: str, admin_user: str, permanent: bool = False) -> Dict[str, Any]:
        """Admin endpoint to delete a debate (soft or permanent)"""
        action_type = "permanent_delete" if permanent else "soft_delete"
        
        logger.info(f"Admin {action_type} debate requested", 
                   admin_user=admin_user, 
                   debate_id=debate_id, 
                   reason=reason)
        
        if not debate_id:
            raise HTTPException(status_code=400, detail="debate_id is required")
        
        try:
            if permanent:
                # Broadcast deletion warning
                await ws_manager.broadcast_to_debate(debate_id, {
                    "type": "debate_permanent_delete_warning",
                    "data": {
                        "message": "Questo dibattito verrà eliminato definitivamente tra 10 secondi",
                        "admin_user": admin_user,
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
                # Update to deleted status
                status = "deleted"
            else:
                # Soft delete - just mark as deleted
                status = "deleted"
            
            # Update debate state
            await ws_manager.update_debate_state(debate_id, {
                "status": status,
                "last_admin_action": datetime.now().isoformat(),
                "admin_user": admin_user,
                "admin_reason": reason
            })
            
            # Broadcast delete event
            await ws_manager.broadcast_to_debate(debate_id, {
                "type": "debate_deleted",
                "data": {
                    "reason": reason,
                    "admin_user": admin_user,
                    "permanent": permanent,
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            # Disconnect all clients from this debate
            await ws_manager.disconnect_all_from_debate(debate_id)
            
            logger.info(f"Debate {action_type} successfully", 
                       debate_id=debate_id,
                       admin_user=admin_user)
            
            return {
                "status": "success",
                "action": action_type,
                "debate_id": debate_id,
                "admin_user": admin_user,
                "reason": reason,
                "permanent": permanent,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to {action_type} debate",
                        debate_id=debate_id,
                        admin_user=admin_user,
                        error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to {action_type} debate: {str(e)}")


    async def _get_debate_messages(self, debate_id: str) -> List[Dict]:
        """Recupera i messaggi del dibattito dal database tramite API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.frontend_url}/api/debates/{debate_id}/messages",
                    timeout=self.frontend_timeout
                )
                
                if response.status_code == 200:
                    messages = response.json()
                    logger.debug("Successfully retrieved messages from database",
                               debate_id=debate_id,
                               message_count=len(messages))
                    return messages
                else:
                    logger.warning("Failed to retrieve messages from database",
                                 debate_id=debate_id,
                                 status_code=response.status_code,
                                 error_response=response.text[:200])
                    return []
                    
        except Exception as e:
            logger.error("Exception retrieving messages from database",
                        debate_id=debate_id,
                        error=str(e),
                        error_type=type(e).__name__)
            return []

    async def _get_debate_starting_model(self, debate_id: str) -> str:
        """Recupera il modello di partenza scelto per il dibattito"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.frontend_url}/api/debates/{debate_id}",
                    timeout=self.frontend_timeout
                )
                
                if response.status_code == 200:
                    debate_data = response.json()
                    starting_model = debate_data.get('startingModel')
                    logger.debug("Retrieved starting model from database",
                               debate_id=debate_id,
                               starting_model=starting_model)
                    return starting_model
                else:
                    logger.warning("Failed to retrieve debate data for starting model",
                                 debate_id=debate_id,
                                 status_code=response.status_code)
                    return None
                    
        except Exception as e:
            logger.error("Exception retrieving starting model from database",
                        debate_id=debate_id,
                        error=str(e),
                        error_type=type(e).__name__)
            return None

    async def _determine_next_speaker(self, models: List[str], recent_messages: List[Dict], starting_model: str = None) -> str:
        """Determina quale modello deve parlare successivamente usando round-robin"""
        if not models:
            raise HTTPException(status_code=400, detail="No models provided")
        
        if not recent_messages:
            # Se non ci sono messaggi, inizia con il modello scelto o il primo
            if starting_model and starting_model in models:
                next_speaker = starting_model
                logger.info("No previous messages, starting with chosen model", 
                           next_speaker=next_speaker,
                           starting_model=starting_model)
            else:
                next_speaker = models[0]
                logger.info("No previous messages, starting with first model", 
                           next_speaker=next_speaker,
                           starting_model_missing=starting_model)
            return next_speaker
        
        # Trova l'ultimo messaggio AI (non moderatore)
        last_ai_message = None
        for msg in reversed(recent_messages):
            if msg.get('type') != 'moderator' and msg.get('type') != 'fallback':
                last_ai_message = msg
                break
        
        if not last_ai_message:
            # Se non ci sono messaggi AI precedenti, inizia con il primo modello
            next_speaker = models[0]
            logger.info("No previous AI messages found, starting with first model", 
                       next_speaker=next_speaker)
            return next_speaker
        
        # Determina l'ultimo speaker
        last_speaker = last_ai_message.get('modelId') or last_ai_message.get('ai')
        
        if not last_speaker:
            next_speaker = models[0]
            logger.info("Could not determine last speaker, starting with first model", 
                       next_speaker=next_speaker)
            return next_speaker
        
        # Trova l'indice del modello che ha parlato per ultimo
        try:
            last_speaker_index = models.index(last_speaker)
            # Passa al prossimo modello nella lista (round-robin)
            next_speaker_index = (last_speaker_index + 1) % len(models)
            next_speaker = models[next_speaker_index]
            
            logger.info("Round-robin speaker selection", 
                       last_speaker=last_speaker,
                       last_speaker_index=last_speaker_index,
                       next_speaker=next_speaker,
                       next_speaker_index=next_speaker_index,
                       total_models=len(models))
            
            return next_speaker
            
        except ValueError:
            # Se l'ultimo speaker non è nella lista attuale, inizia con il primo
            next_speaker = models[0]
            logger.warning("Last speaker not in current model list, starting with first model", 
                          last_speaker=last_speaker,
                          available_models=models,
                          next_speaker=next_speaker)
            return next_speaker



# Istanza globale del debate manager
debate_service = None

def get_debate_service(frontend_url: str, frontend_timeout: int = 10) -> DebateManager:
    """Get or create the global debate service instance"""
    global debate_service
    if debate_service is None:
        debate_service = DebateManager(frontend_url, frontend_timeout)
    return debate_service