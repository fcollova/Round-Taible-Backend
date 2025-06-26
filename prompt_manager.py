"""
Modulo per la gestione dei prompt dei dibattiti.
Gestisce il caricamento, la cache, il fallback e la formattazione dei prompt di sistema.
"""

import httpx
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
from config_manager import ConfigManager
from logging_config import get_context_logger

logger = get_context_logger(__name__)
config = ConfigManager()


@dataclass
class PromptConfig:
    """Configurazione di un prompt di sistema"""
    name: str
    system_template: str
    user_template: str
    parameters: Dict[str, Any]
    placeholders: Dict[str, str]
    version: int
    category: str
    priority: int


class PromptManager:
    """Manager per la gestione dei prompt dei dibattiti"""
    
    def __init__(self):
        self.system_prompts: Dict[str, PromptConfig] = {}
        self.prompts_cache_time: Optional[datetime] = None
        self.cache_ttl_minutes = 5
        self.frontend_url = config.get_frontend_url()
        self.frontend_timeout = config.get_frontend_timeout()
        
        logger.info("PromptManager initialized",
                   frontend_url=self.frontend_url,
                   cache_ttl_minutes=self.cache_ttl_minutes)
    
    async def get_prompt_config(self, prompt_name: str) -> PromptConfig:
        """
        Recupera la configurazione di un prompt specifico.
        
        Args:
            prompt_name: Nome del prompt ('opening_statement', 'continuing_debate', etc.)
            
        Returns:
            PromptConfig: Configurazione del prompt
            
        Raises:
            Exception: Se il prompt non viene trovato
        """
        # Assicurati che i prompt siano caricati e aggiornati
        await self._ensure_prompts_loaded()
        
        if prompt_name not in self.system_prompts:
            logger.error("Prompt not found",
                        prompt_name=prompt_name,
                        available_prompts=list(self.system_prompts.keys()))
            raise Exception(f"Prompt '{prompt_name}' not found")
        
        prompt_config = self.system_prompts[prompt_name]
        
        logger.debug("Prompt config retrieved",
                    prompt_name=prompt_name,
                    version=prompt_config.version,
                    placeholders_count=len(prompt_config.placeholders))
        
        return prompt_config
    
    def format_prompt_template(self, template: str, **kwargs) -> str:
        """
        Formatta un template di prompt con i parametri forniti.
        
        Args:
            template: Template del prompt con placeholder {nome}
            **kwargs: Parametri per sostituire i placeholder
            
        Returns:
            str: Template formattato
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            missing_param = str(e).strip("'\"")
            logger.warning("Missing parameter in prompt template",
                          missing_param=missing_param,
                          template_preview=template[:100],
                          available_params=list(kwargs.keys()))
            
            # Sostituisci i parametri disponibili
            formatted = template
            for key, value in kwargs.items():
                formatted = formatted.replace(f"{{{key}}}", str(value))
            
            # Sostituisci parametri mancanti con placeholder visibile
            remaining_placeholders = re.findall(r'{([^}]+)}', formatted)
            for placeholder in remaining_placeholders:
                formatted = formatted.replace(f"{{{placeholder}}}", f"[MISSING_{placeholder.upper()}]")
                
            logger.warning("Template formatted with missing parameters",
                          missing_placeholders=remaining_placeholders,
                          formatted_preview=formatted[:200])
            
            return formatted
    
    def build_template_params(self, prompt_config: PromptConfig, **dynamic_values) -> Dict[str, str]:
        """
        Costruisce i parametri per la formattazione del template.
        
        Args:
            prompt_config: Configurazione del prompt
            **dynamic_values: Valori dinamici (topic, personality, context, etc.)
            
        Returns:
            Dict[str, str]: Parametri per la formattazione
        """
        # 1. Inizia con placeholder configurabili dal prompt
        template_params = prompt_config.placeholders.copy()
        
        # 2. Sovrascrivi con valori dinamici che hanno precedenza
        system_template = prompt_config.system_template
        user_template = prompt_config.user_template
        
        # Topic è sempre dinamico
        if 'topic' in dynamic_values:
            template_params['topic'] = dynamic_values['topic']
        
        # Personality: usa valore dinamico se presente, altrimenti mantieni placeholder del database
        if '{personality}' in system_template or '{personality}' in user_template:
            if 'personality' in dynamic_values and dynamic_values['personality']:
                template_params['personality'] = dynamic_values['personality']
            # Se non c'è valore dinamico, usa il placeholder del database (già caricato)
            
        # Context: sempre dinamico dai messaggi recenti del dibattito
        if '{context}' in system_template or '{context}' in user_template:
            template_params['context'] = dynamic_values.get('context', '')
        
        logger.debug("Template parameters built",
                    prompt_name=prompt_config.name,
                    template_params=list(template_params.keys()),
                    topic=template_params.get('topic', 'N/A'),
                    has_personality=bool(template_params.get('personality')),
                    has_context=bool(template_params.get('context')),
                    other_placeholders=[k for k in template_params.keys() 
                                      if k not in ['topic', 'personality', 'context']])
        
        return template_params
    
    def format_system_prompt(self, prompt_config: PromptConfig, **dynamic_values) -> str:
        """
        Formatta il system prompt utilizzando la configurazione e i valori dinamici.
        
        Args:
            prompt_config: Configurazione del prompt
            **dynamic_values: Valori dinamici
            
        Returns:
            str: System prompt formattato
        """
        template_params = self.build_template_params(prompt_config, **dynamic_values)
        return self.format_prompt_template(prompt_config.system_template, **template_params)
    
    def format_user_prompt(self, prompt_config: PromptConfig, **dynamic_values) -> str:
        """
        Formatta il user prompt utilizzando la configurazione e i valori dinamici.
        
        Args:
            prompt_config: Configurazione del prompt
            **dynamic_values: Valori dinamici
            
        Returns:
            str: User prompt formattato
        """
        template_params = self.build_template_params(prompt_config, **dynamic_values)
        fallback_user_template = f"Continua il dibattito su: {dynamic_values.get('topic', '[ARGOMENTO]')}"
        user_template = prompt_config.user_template or fallback_user_template
        return self.format_prompt_template(user_template, **template_params)
    
    def _get_priority_for_prompt(self, prompt_name: str) -> int:
        """
        Determina la priorità basata sul nome del prompt.
        
        Args:
            prompt_name: Nome del prompt
            
        Returns:
            int: Priorità del prompt
        """
        priorities = {
            'opening_statement': 1,
            'continuing_debate': 2, 
            'moderator_response': 3
        }
        return priorities.get(prompt_name, 0)
    
    async def _ensure_prompts_loaded(self):
        """Assicura che i prompt siano caricati e aggiornati"""
        now = datetime.now()
        
        # Controlla se i prompt devono essere ricaricati
        if (self.prompts_cache_time is None or 
            now - self.prompts_cache_time > timedelta(minutes=self.cache_ttl_minutes)):
            
            logger.info("Loading system prompts from database",
                       cache_expired=self.prompts_cache_time is not None,
                       last_cache_time=self.prompts_cache_time.isoformat() if self.prompts_cache_time else None)
            
            await self._load_system_prompts()
    
    async def _load_system_prompts(self) -> Dict[str, PromptConfig]:
        """
        Carica i prompt di sistema dal database tramite API del frontend.
        
        Returns:
            Dict[str, PromptConfig]: Dizionario dei prompt caricati
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.frontend_url}/api/prompts",
                    timeout=self.frontend_timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    raw_prompts = data.get('prompts', {})
                    
                    # Converte i dati in PromptConfig objects
                    self.system_prompts = {}
                    for prompt_name, prompt_data in raw_prompts.items():
                        # Determina categoria basata sul nome del prompt
                        category = 'debate'
                        if prompt_name == 'moderator_response':
                            category = 'moderation'
                            
                        self.system_prompts[prompt_name] = PromptConfig(
                            name=prompt_name,
                            system_template=prompt_data.get('systemTemplate', ''),
                            user_template=prompt_data.get('userTemplate', ''),
                            parameters=prompt_data.get('parameters', {}),
                            placeholders=prompt_data.get('placeholders', {}),
                            version=prompt_data.get('version', 1),
                            category=category,
                            priority=self._get_priority_for_prompt(prompt_name)
                        )
                    
                    self.prompts_cache_time = datetime.now()
                    
                    logger.info("System prompts loaded from database",
                               prompt_count=len(self.system_prompts),
                               cache_time=self.prompts_cache_time.isoformat(),
                               prompt_names=list(self.system_prompts.keys()))
                    
                    return self.system_prompts
                else:
                    logger.error("Failed to load system prompts",
                               status_code=response.status_code,
                               error_response=response.text[:200])
                    return self._get_fallback_prompts()
                    
        except Exception as e:
            logger.error("Exception loading system prompts",
                        error=str(e),
                        error_type=type(e).__name__)
            return self._get_fallback_prompts()
    
    def _get_fallback_prompts(self) -> Dict[str, PromptConfig]:
        """
        Prompt di fallback se il database non è disponibile.
        
        Returns:
            Dict[str, PromptConfig]: Prompt di fallback
        """
        logger.warning("Using fallback prompts - database unavailable")
        
        fallback_prompts = {
            'opening_statement': PromptConfig(
                name='opening_statement',
                system_template="""Stai iniziando un dibattito live su: {topic}

La tua personalità ed esperienza: {personality}
Stile comunicativo: {tone}
Lunghezza della risposta: {response_length}

Come primo oratore, fornisci una dichiarazione di apertura che:
1. Espone chiaramente la tua posizione sull'argomento
2. Presenta 2-3 argomenti chiave
3. Imposta il tono per un dibattito coinvolgente

{additional_instructions}""",
                user_template="Inizia il dibattito su: {topic}",
                parameters={'max_tokens': 350, 'temperature': 0.8},
                placeholders={
                    'personality': "Assistente AI esperto con capacità analitiche",
                    'tone': "professionale ma coinvolgente", 
                    'response_length': "massimo 2-3 paragrafi",
                    'additional_instructions': "Mantieni concisione e coinvolgimento. Questa è la tua opportunità per fare una forte prima impressione."
                },
                version=1,
                category='debate',
                priority=1
            ),
            'continuing_debate': PromptConfig(
                name='continuing_debate',
                system_template="""Stai partecipando a un dibattito live in corso su: {topic}

La tua personalità e stile di dibattito: {personality}

Contesto della discussione precedente:
{context}

Genera una risposta ponderata che:
1. Affronta direttamente punti specifici sollevati da altri partecipanti
2. Costruisce o sfida i loro argomenti con prove
3. Mantiene la tua prospettiva unica mentre coinvolge in modo costruttivo
4. Fa avanzare il dibattito con nuove intuizioni o controargomentazioni
5. Se c'è una guida del moderatore nel contesto, DEVI riconoscerla e affrontarla

Linee guida:
- Mantieni concisione (massimo 1-2 paragrafi) e coinvolgimento
- Fai riferimento a punti specifici della discussione precedente
- Rimani fedele alla tua personalità ed esperienza
- Sii rispettoso ma intellettualmente rigoroso
- Evita di ripetere quello che altri hanno già detto
- IMPORTANTE: Se il moderatore ha fornito una guida, incorpora la loro direzione nella tua risposta mantenendo la tua prospettiva unica

{additional_instructions}""",
                user_template="Basandoti sulla discussione sopra, fornisci il tuo prossimo contributo al dibattito su: {topic}",
                parameters={'max_tokens': 300, 'temperature': 0.8},
                placeholders={
                    'personality': "Assistente AI esperto con capacità analitiche",
                    'additional_instructions': "Ricorda di fare riferimento a punti specifici dei messaggi precedenti e mantenere rigore intellettuale."
                },
                version=1,
                category='debate',
                priority=2
            ),
            'moderator_response': PromptConfig(
                name='moderator_response',
                system_template="""Stai moderando un dibattito live su: {topic}

Il tuo ruolo come moderatore: {personality}

Contesto della discussione:
{context}

Come moderatore, fornisci una risposta che:
1. Mantiene il dibattito focalizzato e produttivo
2. Gestisce eventuali tensioni o deviazioni
3. Pone domande che stimolano approfondimenti
4. Assicura che tutti i partecipanti abbiano voce
5. Riassume punti chiave quando necessario

{additional_instructions}""",
                user_template="Come moderatore, intervieni nel dibattito su: {topic}",
                parameters={'max_tokens': 250, 'temperature': 0.7},
                placeholders={
                    'personality': "Moderatore esperto, imparziale e professionale",
                    'additional_instructions': "Mantieni neutralità e guida la discussione verso conclusioni costruttive."
                },
                version=1,
                category='moderation',
                priority=3
            )
        }
        
        self.system_prompts = fallback_prompts
        self.prompts_cache_time = datetime.now()
        
        return fallback_prompts


# Istanza globale del PromptManager
prompt_manager = PromptManager()