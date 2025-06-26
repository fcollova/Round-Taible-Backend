"""
Modulo per la gestione del contesto dei dibattiti.
Gestisce la costruzione, formattazione e persistenza del contesto delle conversazioni.
"""

import httpx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from config_manager import ConfigManager
from logging_config import get_context_logger

logger = get_context_logger(__name__)
config = ConfigManager()


@dataclass
class ContextMessage:
    """Rappresenta un messaggio nel contesto del dibattito"""
    model_id: str
    content: str
    timestamp: str
    message_type: str  # 'ai', 'moderator', 'user'
    turn_number: Optional[int] = None
    speaker_name: Optional[str] = None


@dataclass 
class DebateContext:
    """Contesto completo di un dibattito"""
    debate_id: str
    topic: str
    messages: List[ContextMessage]
    moderator_guidance: str
    message_count: int
    last_speaker: Optional[str] = None
    context_window_size: int = 5


class ContextManager:
    """Manager per la gestione del contesto dei dibattiti"""
    
    def __init__(self):
        self.frontend_url = config.get_frontend_url()
        self.frontend_timeout = config.get_frontend_timeout()
        self.default_context_window = 5
        
        logger.info("ContextManager initialized",
                   frontend_url=self.frontend_url,
                   default_window_size=self.default_context_window)
    
    async def build_debate_context(self, debate_id: str, topic: str, 
                                 recent_messages: List[Dict[str, Any]]) -> DebateContext:
        """
        Costruisce il contesto completo di un dibattito.
        
        Args:
            debate_id: ID del dibattito
            topic: Argomento del dibattito
            recent_messages: Lista dei messaggi recenti
            
        Returns:
            DebateContext: Contesto completo del dibattito
        """
        logger.debug("Building debate context",
                    debate_id=debate_id,
                    topic=topic,
                    message_count=len(recent_messages))
        
        # Carica informazioni sui modelli per nomi user-friendly
        model_info = await self._get_model_info()
        
        # Converte i messaggi in ContextMessage objects
        context_messages = []
        moderator_guidance = ""
        
        # Usa sliding window per ottimizzare le performance
        recent_window = recent_messages[-self.default_context_window:] if recent_messages else []
        
        for msg in recent_window:
            message_type = msg.get('type', 'ai')
            model_id = msg.get('model', 'Unknown')
            content = msg.get('content', '')
            
            # Determina il nome del speaker
            speaker_name = self._get_speaker_name(model_id, message_type, model_info)
            
            # Gestione speciale per messaggi moderatore
            if message_type == 'moderator':
                moderator_guidance = self._extract_moderator_guidance(content)
            
            context_message = ContextMessage(
                model_id=model_id,
                content=content,
                timestamp=msg.get('timestamp', ''),
                message_type=message_type,
                turn_number=msg.get('turnNumber'),
                speaker_name=speaker_name
            )
            
            context_messages.append(context_message)
        
        # Determina l'ultimo speaker
        last_speaker = None
        if context_messages:
            last_message = context_messages[-1]
            last_speaker = last_message.model_id if last_message.message_type != 'moderator' else 'moderator'
        
        debate_context = DebateContext(
            debate_id=debate_id,
            topic=topic,
            messages=context_messages,
            moderator_guidance=moderator_guidance,
            message_count=len(recent_messages),
            last_speaker=last_speaker,
            context_window_size=len(context_messages)
        )
        
        logger.info("Debate context built",
                   debate_id=debate_id,
                   total_messages=len(recent_messages),
                   context_window_messages=len(context_messages),
                   has_moderator_guidance=bool(moderator_guidance),
                   last_speaker=last_speaker)
        
        return debate_context
    
    def format_context_for_prompt(self, debate_context: DebateContext) -> str:
        """
        Formatta il contesto per l'inserimento nel prompt.
        
        Args:
            debate_context: Contesto del dibattito
            
        Returns:
            str: Contesto formattato come stringa
        """
        if not debate_context.messages:
            return ""
        
        context_lines = []
        
        # Formatta ogni messaggio nel contesto
        for msg in debate_context.messages:
            if msg.message_type == 'moderator':
                context_lines.append(f"[MODERATORE]: {msg.content}")
            else:
                speaker_display = msg.speaker_name or msg.model_id
                context_lines.append(f"{speaker_display}: {msg.content}")
        
        # Unisce i messaggi
        formatted_context = "\n".join(context_lines)
        
        # Aggiunge guidance del moderatore se presente
        if debate_context.moderator_guidance:
            formatted_context += f"\n\n**MODERATOR GUIDANCE**: {debate_context.moderator_guidance}\n*Please address and incorporate this guidance in your response.*"
        
        logger.debug("Context formatted for prompt",
                    debate_id=debate_context.debate_id,
                    context_length=len(formatted_context),
                    message_count=len(debate_context.messages),
                    has_moderator_guidance=bool(debate_context.moderator_guidance))
        
        return formatted_context
    
    def determine_next_speaker(self, debate_context: DebateContext, 
                             available_models: List[str]) -> Optional[str]:
        """
        Determina il prossimo speaker nel dibattito usando logica round-robin.
        
        Args:
            debate_context: Contesto del dibattito
            available_models: Lista dei modelli disponibili
            
        Returns:
            Optional[str]: ID del prossimo modello o None
        """
        if not available_models:
            logger.warning("No available models for next speaker determination",
                          debate_id=debate_context.debate_id)
            return None
        
        if len(available_models) == 1:
            next_speaker = available_models[0]
            logger.debug("Single model available",
                        debate_id=debate_context.debate_id,
                        next_speaker=next_speaker)
            return next_speaker
        
        # Logica round-robin
        last_speaker = debate_context.last_speaker
        
        if last_speaker and last_speaker != 'moderator' and last_speaker in available_models:
            try:
                current_index = available_models.index(last_speaker)
                next_index = (current_index + 1) % len(available_models)
                next_speaker = available_models[next_index]
                
                logger.debug("Next speaker determined via round-robin",
                           debate_id=debate_context.debate_id,
                           last_speaker=last_speaker,
                           next_speaker=next_speaker,
                           current_index=current_index,
                           next_index=next_index)
                
                return next_speaker
            except ValueError:
                logger.warning("Last speaker not in available models",
                             debate_id=debate_context.debate_id,
                             last_speaker=last_speaker,
                             available_models=available_models)
        
        # Fallback: primo modello disponibile
        next_speaker = available_models[0]
        logger.debug("Using fallback first speaker",
                    debate_id=debate_context.debate_id,
                    next_speaker=next_speaker,
                    reason="no_valid_last_speaker")
        
        return next_speaker
    
    def get_context_summary(self, debate_context: DebateContext) -> Dict[str, Any]:
        """
        Genera un riassunto del contesto per logging e debugging.
        
        Args:
            debate_context: Contesto del dibattito
            
        Returns:
            Dict[str, Any]: Riassunto del contesto
        """
        message_types = {}
        speakers = set()
        
        for msg in debate_context.messages:
            message_types[msg.message_type] = message_types.get(msg.message_type, 0) + 1
            if msg.message_type != 'moderator':
                speakers.add(msg.model_id)
        
        return {
            'debate_id': debate_context.debate_id,
            'topic': debate_context.topic,
            'total_messages': debate_context.message_count,
            'context_window_messages': len(debate_context.messages),
            'message_types': message_types,
            'unique_speakers': list(speakers),
            'last_speaker': debate_context.last_speaker,
            'has_moderator_guidance': bool(debate_context.moderator_guidance),
            'context_length': len(self.format_context_for_prompt(debate_context))
        }
    
    async def _get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Recupera informazioni sui modelli dal frontend per nomi user-friendly.
        
        Returns:
            Dict[str, Dict[str, Any]]: Informazioni sui modelli
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.frontend_url}/api/models",
                    timeout=self.frontend_timeout
                )
                
                if response.status_code == 200:
                    model_info = response.json()
                    logger.debug("Model info retrieved",
                               model_count=len(model_info))
                    return model_info
                else:
                    logger.warning("Failed to retrieve model info",
                                 status_code=response.status_code)
                    return {}
        except Exception as e:
            logger.warning("Exception retrieving model info",
                          error=str(e),
                          error_type=type(e).__name__)
            return {}
    
    def _get_speaker_name(self, model_id: str, message_type: str, 
                         model_info: Dict[str, Dict[str, Any]]) -> str:
        """
        Determina il nome user-friendly del speaker.
        
        Args:
            model_id: ID del modello
            message_type: Tipo di messaggio
            model_info: Informazioni sui modelli
            
        Returns:
            str: Nome user-friendly del speaker
        """
        if message_type == 'moderator':
            return 'MODERATORE'
        
        if model_id in model_info:
            return model_info[model_id].get('name', model_id)
        
        return model_id
    
    def _extract_moderator_guidance(self, moderator_content: str) -> str:
        """
        Estrae e formatta la guidance del moderatore.
        
        Args:
            moderator_content: Contenuto del messaggio del moderatore
            
        Returns:
            str: Guidance formattata
        """
        # Per ora restituisce il contenuto completo
        # In futuro si potrebbero applicare trasformazioni specifiche
        return moderator_content.strip()
    
    async def save_context_to_database(self, debate_context: DebateContext) -> bool:
        """
        Salva il contesto nel database per persistenza (opzionale).
        
        Args:
            debate_context: Contesto da salvare
            
        Returns:
            bool: True se salvato con successo
        """
        try:
            # Implementazione futura per salvare snapshot del contesto
            # utile per analytics e debugging
            logger.debug("Context save requested",
                        debate_id=debate_context.debate_id,
                        message_count=len(debate_context.messages))
            return True
        except Exception as e:
            logger.error("Failed to save context to database",
                        debate_id=debate_context.debate_id,
                        error=str(e))
            return False


# Istanza globale del ContextManager
context_manager = ContextManager()