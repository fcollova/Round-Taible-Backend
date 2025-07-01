import asyncio
from asyncio import Queue, Lock
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import logging
from enum import Enum
import time
import requests
import httpx
from logging_config import get_context_logger, performance_metrics
from config_manager import get_config
from prompt_manager import prompt_manager
from context_manager import context_manager

logger = get_context_logger(__name__)

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class MessageRequest:
    debate_id: str
    model_id: str
    context: str
    topic: str
    personality: str
    message_count: int
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    max_retries: int = 3
    retry_count: int = 0
    callback: Optional[Callable] = None
    original_model_id: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None

class LLMQueueManager:
    def __init__(self, max_concurrent_requests: int = 3, frontend_url: str = "http://localhost:3000", frontend_timeout: int = 10):
        # Code per priorità
        self.priority_queues: Dict[MessagePriority, Queue] = {
            priority: Queue() for priority in MessagePriority
        }
        
        # Semaforo per limitare richieste concorrenti
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Tracking delle richieste attive
        self.active_requests: Dict[str, MessageRequest] = {}
        self.request_lock = Lock()
        
        # Rate limiting per modello
        self.model_last_request: Dict[str, datetime] = {}
        self.model_rate_limits: Dict[str, timedelta] = {
            'gpt4': timedelta(seconds=1),
            'claude': timedelta(seconds=0.5),
            'gemini': timedelta(seconds=0.8),
            'llama': timedelta(seconds=0.3)
        }
        
        # Cache rimossa - gestita dal PromptManager
        
        # Statistiche
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'queue_wait_times': []
        }
        
        # Frontend configuration - ora gestita da PromptManager e ContextManager
        self.frontend_url = frontend_url.rstrip('/')  # Remove trailing slash
        self.frontend_timeout = frontend_timeout
        
        # Workers
        self.workers = []
        self.running = False
    
    async def start(self, num_workers: int = 2):
        """Avvia i worker per processare le code"""
        self.running = True
        
        logger.info("Starting LLM Queue Manager", 
                   worker_count=num_workers,
                   max_concurrent=self.semaphore._value)
        
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Worker per cleanup richieste scadute
        cleanup_worker = asyncio.create_task(self._cleanup_expired_requests())
        self.workers.append(cleanup_worker)
        
        logger.info("LLM Queue Manager started successfully", 
                   total_workers=len(self.workers),
                   active_workers=num_workers,
                   cleanup_worker=True)
    
    async def stop(self):
        """Ferma tutti i workers"""
        logger.info("Stopping LLM Queue Manager", 
                   active_workers=len(self.workers),
                   pending_requests=len(self.active_requests))
        
        self.running = False
        cancelled_count = 0
        
        for worker in self.workers:
            worker.cancel()
            cancelled_count += 1
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("LLM Queue Manager stopped", 
                   cancelled_workers=cancelled_count,
                   final_stats=self.stats)
    
    async def enqueue_message(self, request: MessageRequest) -> str:
        """Aggiungi richiesta alla coda appropriata"""
        request_id = f"{request.debate_id}_{request.model_id}_{int(time.time() * 1000)}"
        
        logger.info("Enqueuing LLM message request",
                   request_id=request_id,
                   debate_id=request.debate_id,
                   model_id=request.model_id,
                   priority=request.priority.name,
                   message_count=request.message_count)
        
        async with self.request_lock:
            self.active_requests[request_id] = request
        
        await self.priority_queues[request.priority].put((request_id, request))
        
        # Log stato delle code
        queue_stats = self.get_queue_stats()
        performance_metrics.log_queue_metrics(
            queue_sizes=queue_stats['queue_sizes'],
            active_requests=queue_stats['active_requests'],
            avg_wait_time=queue_stats['stats']['average_queue_wait_time']
        )
        
        logger.debug("Request successfully enqueued",
                    request_id=request_id,
                    queue_size=queue_stats['queue_sizes'][request.priority.name],
                    total_active=queue_stats['active_requests'])
        
        return request_id
    
    async def _worker(self, worker_name: str):
        """Worker che processa le richieste dalle code"""
        while self.running:
            try:
                # Controlla code in ordine di priorità
                for priority in reversed(list(MessagePriority)):
                    queue = self.priority_queues[priority]
                    
                    if not queue.empty():
                        try:
                            # Timeout per evitare blocking indefinito
                            request_id, request = await asyncio.wait_for(
                                queue.get(), timeout=0.1
                            )
                            
                            logger.debug("Worker picked up request",
                                       worker_name=worker_name,
                                       request_id=request_id,
                                       priority=priority.name,
                                       debate_id=request.debate_id)
                            
                            # Processa la richiesta
                            await self._process_request(worker_name, request_id, request)
                            break
                            
                        except asyncio.TimeoutError:
                            continue
                
                # Pausa breve se nessuna richiesta da processare
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("Critical error in worker",
                           worker_name=worker_name,
                           error=str(e),
                           error_type=type(e).__name__)
                await asyncio.sleep(1)
    
    async def _process_request(self, worker_name: str, request_id: str, request: MessageRequest):
        """Processa una singola richiesta"""
        async with self.semaphore:  # Limita concorrenza
            start_time = datetime.now()
            queue_wait_time = (start_time - request.created_at).total_seconds()
            
            try:
                # Rate limiting per modello
                await self._apply_rate_limit(request.model_id)
                
                logger.info("Worker processing LLM request",
                           worker_name=worker_name,
                           request_id=request_id,
                           debate_id=request.debate_id,
                           model_id=request.model_id,
                           queue_wait_time=queue_wait_time,
                           retry_count=request.retry_count)
                
                # Genera risposta LLM
                response = await self._generate_llm_response(request_id, request)
                
                if response:  # Se la risposta è valida
                    # Calcola tempo di risposta
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    # Invia risposta via WebSocket
                    message_data = {
                        'modelId': request.original_model_id or request.model_id,  # Usa l'ID originale se disponibile
                        'ai': request.original_model_id or request.model_id,       # Retrocompatibilità 
                        'content': response['content'],
                        'timestamp': datetime.now().isoformat(),
                        'type': 'ai',
                        'id': f"msg_{int(time.time() * 1000)}",
                        'turnNumber': request.message_count + 1
                    }
                    
                    # Salva il messaggio nel database
                    await self._save_message_to_database(request.debate_id, message_data)
                    
                    # Invia il messaggio via WebSocket
                    await self._send_response_to_debate(request.debate_id, {
                        'type': 'new_message',
                        'data': message_data
                    })
                    
                    # Programma automaticamente il prossimo messaggio (gestione autonoma backend)
                    await self._schedule_next_message(request)
                    
                    # Callback se presente
                    if request.callback:
                        await request.callback(request_id, response, None)
                    
                    # Aggiorna statistiche
                    self.stats['successful_requests'] += 1
                    self._update_average_response_time(response_time)
                    self.stats['queue_wait_times'].append(queue_wait_time)
                    
                    # Mantieni solo ultime 100 metriche per memoria
                    if len(self.stats['queue_wait_times']) > 100:
                        self.stats['queue_wait_times'] = self.stats['queue_wait_times'][-100:]
                    
                    logger.info("Worker completed LLM request successfully",
                               worker_name=worker_name,
                               request_id=request_id,
                               debate_id=request.debate_id,
                               model_id=request.model_id,
                               response_time=response_time,
                               content_length=len(response.get('content', '')),
                               queue_wait_time=queue_wait_time)
                
            except Exception as e:
                await self._handle_request_error(request_id, request, e)
                
            finally:
                # Rimuovi richiesta dalle attive
                async with self.request_lock:
                    self.active_requests.pop(request_id, None)
                
                self.stats['total_requests'] += 1
    
    async def _apply_rate_limit(self, model_id: str):
        """Applica rate limiting per modello"""
        if model_id in self.model_last_request:
            time_limit = self.model_rate_limits.get(model_id, timedelta(seconds=1))
            time_since_last = datetime.now() - self.model_last_request[model_id]
            
            if time_since_last < time_limit:
                wait_time = (time_limit - time_since_last).total_seconds()
                await asyncio.sleep(wait_time)
        
        self.model_last_request[model_id] = datetime.now()
    
    async def _load_system_prompts(self) -> Dict[str, dict]:
        """Carica i prompt di sistema dal database tramite API"""
        try:
            # Controlla se la cache è ancora valida
            if (self.prompts_cache_time and 
                datetime.now() - self.prompts_cache_time < self.prompts_cache_ttl):
                return self.system_prompts
            
            # Carica prompt dal database tramite API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.frontend_url}/api/prompts",
                    timeout=self.frontend_timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.system_prompts = data.get('prompts', {})
                    self.prompts_cache_time = datetime.now()
                    
                    logger.info("System prompts loaded from database",
                               prompt_count=len(self.system_prompts),
                               cache_time=self.prompts_cache_time.isoformat())
                    
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
    
    def _get_fallback_prompts(self) -> Dict[str, dict]:
        """Prompt di fallback se il database non è disponibile"""
        return {
            'opening_statement': {
                'systemTemplate': """You are starting a live debate about: {topic}

Your personality and expertise: {personality}
Speaking style: {tone}
Response length: {response_length}

As the first speaker, provide an opening statement that:
1. Clearly states your position on the topic
2. Presents 2-3 key arguments
3. Sets the tone for an engaging debate

{additional_instructions}""",
                'userTemplate': "Start the debate on: {topic}",
                'parameters': {'max_tokens': 350, 'temperature': 0.8},
                'placeholders': {
                    'personality': "Expert AI assistant with analytical capabilities",
                    'tone': "professional yet engaging", 
                    'response_length': "2-3 paragraphs maximum",
                    'additional_instructions': "Keep it concise and compelling. This is your chance to make a strong first impression."
                }
            },
            'continuing_debate': {
                'systemTemplate': """You are participating in an ongoing live debate about: {topic}

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
- IMPORTANT: If the moderator has provided guidance, incorporate their direction into your response while maintaining your unique perspective

{additional_instructions}""",
                'userTemplate': "Based on the discussion above, provide your next contribution to the debate on: {topic}",
                'parameters': {'max_tokens': 300, 'temperature': 0.8},
                'placeholders': {
                    'personality': "Expert AI assistant with analytical capabilities",
                    'additional_instructions': "Remember to reference specific points from previous messages and maintain intellectual rigor."
                }
            }
        }
    
    async def _generate_llm_response(self, request_id: str, request: MessageRequest) -> dict:
        """Genera risposta dall'LLM utilizzando i nuovi moduli dedicati"""
        try:
            # Import dinamico per evitare circular imports
            from openrouter_client import get_openrouter_client
            from config_manager import get_config
            
            config = get_config()
            openrouter_client = get_openrouter_client(
                config.get_openrouter_api_key(),
                config.get_openrouter_base_url(),
                config.get_openrouter_timeout()
            )
            
            logger.debug("Generating LLM response",
                        model_id=request.model_id,
                        topic=request.topic,
                        message_count=request.message_count,
                        personality=request.personality)
            
            # Determina quale prompt utilizzare
            prompt_name = 'opening_statement' if request.message_count == 0 else 'continuing_debate'
            
            # Ottieni configurazione prompt dal PromptManager
            prompt_config = await prompt_manager.get_prompt_config(prompt_name)
            
            # Prepara valori dinamici per i template
            dynamic_values = {
                'topic': request.topic,
                'personality': request.personality,
                'context': request.context if hasattr(request, 'context') else ''
            }
            
            # Aggiungi parametri specifici per tipo di prompt
            if prompt_name == 'opening_statement':
                dynamic_values.update({
                    'tone': "professional yet engaging",
                    'response_length': "2-3 paragraphs",
                    'additional_instructions': "Stay focused on the debate topic and provide reasoned arguments."
                })
            elif prompt_name == 'continuing_debate':
                dynamic_values.update({
                    'response_approach': "analytical and engaging",
                    'guidelines': """Linee guida:
- Mantieni concisione (massimo 1-2 paragrafi) e coinvolgimento
- Fai riferimento a punti specifici della discussione precedente
- Rimani fedele alla tua personalità ed esperienza
- Sii rispettoso ma intellettualmente rigoroso
- Evita di ripetere quello che altri hanno già detto
- IMPORTANTE: Se il moderatore ha fornito una guida, incorpora la loro direzione nella tua risposta mantenendo la tua prospettiva unica"""
                })
            
            # Formatta i prompt utilizzando il PromptManager
            system_prompt = prompt_manager.format_system_prompt(prompt_config, **dynamic_values)
            user_prompt = prompt_manager.format_user_prompt(prompt_config, **dynamic_values)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Ottieni parametri LLM dalla configurazione del prompt
            max_tokens = prompt_config.parameters.get('max_tokens', 300)
            temperature = prompt_config.parameters.get('temperature', 0.8)
            
            # Log detailed prompt information in debug mode
            if config.get_logging_debug():
                logger.debug("LLM request details",
                            model_id=request.model_id,
                            topic=request.topic,
                            prompt_name=prompt_name,
                            prompt_version=prompt_config.version,
                            system_prompt_preview=system_prompt[:200],
                            user_prompt=user_prompt,
                            context_length=len(dynamic_values['context']),
                            max_tokens=max_tokens,
                            temperature=temperature,
                            placeholders_count=len(prompt_config.placeholders))
            
            # Chiama OpenRouter per generare la risposta
            response = await openrouter_client.chat_completion(
                model=request.model_id,  # Usa direttamente il model_id
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            content = response['choices'][0]['message']['content']
            
            logger.debug("LLM response generated successfully",
                        model_id=request.model_id,
                        topic=request.topic,
                        message_count=request.message_count,
                        response_length=len(content),
                        response_preview=content[:100] + "..." if len(content) > 100 else content)
            
            return {
                "model": request.model_id,
                "content": content,
                "topic": request.topic,
                "message_count": request.message_count
            }
            
        except Exception as e:
            # Incrementa retry count
            request.retry_count += 1
            
            if request.retry_count < request.max_retries:
                # Re-enqueue con priorità più alta se possibile
                new_priority = MessagePriority.HIGH if request.priority == MessagePriority.NORMAL else request.priority
                request.priority = new_priority
                
                await asyncio.sleep(2 ** request.retry_count)  # Exponential backoff
                await self.enqueue_message(request)
                
                logger.warning("Retrying LLM request",
                              request_id=request_id,
                              debate_id=request.debate_id,
                              model_id=request.model_id,
                              retry_count=request.retry_count,
                              max_retries=request.max_retries,
                              new_priority=new_priority.name,
                              backoff_delay=2 ** request.retry_count)
                return None
            else:
                raise e
    
    async def _handle_request_error(self, request_id: str, request: MessageRequest, error: Exception):
        """Gestisce errori nelle richieste usando messaggi di fallback dal database"""
        logger.error("Error processing LLM request",
                    request_id=request_id,
                    debate_id=request.debate_id,
                    model_id=request.model_id,
                    error=str(error),
                    error_type=type(error).__name__,
                    retry_count=request.retry_count,
                    max_retries=request.max_retries)
        
        # Determina quale messaggio di fallback utilizzare in base al tipo di errore
        fallback_message_name = self._determine_fallback_message_type(error)
        
        try:
            # Ottieni messaggio di fallback dal database
            fallback_content = await self._get_fallback_message(
                fallback_message_name, 
                request.topic, 
                request.model_id
            )
            
            logger.info("Using fallback message from database",
                       message_name=fallback_message_name,
                       request_id=request_id,
                       debate_id=request.debate_id)
            
        except Exception as fallback_error:
            # Se anche il fallback del database fallisce, usa messaggio di emergenza minimo
            logger.error("Fallback message retrieval failed, using emergency message",
                        fallback_error=str(fallback_error),
                        fallback_message_attempted=fallback_message_name,
                        request_id=request_id)
            fallback_content = "Sistema temporaneamente non disponibile. Riproverò a breve."
            fallback_message_name = "emergency"
        
        # Crea messaggio di fallback con turnNumber corretto
        message_data = {
            'id': f"{request_id}_fallback",
            'modelId': request.original_model_id or request.model_id,  # Usa l'ID originale se disponibile
            'ai': request.original_model_id or request.model_id,       # Retrocompatibilità
            'content': fallback_content,
            'timestamp': datetime.now().isoformat(),
            'type': 'fallback',
            'turnNumber': request.message_count + 1,
            'is_fallback': True,
            'error_type': type(error).__name__,
            'fallback_message_used': fallback_message_name
        }
        
        # Salva il messaggio nel database
        await self._save_message_to_database(request.debate_id, message_data)
        
        # Invia il messaggio via WebSocket
        await self._send_response_to_debate(request.debate_id, {
            'type': 'new_message',
            'data': message_data
        })
        
        # Callback con errore
        if request.callback:
            await request.callback(request_id, None, error)
        
        self.stats['failed_requests'] += 1
    
    async def _verify_fallback_messages_available(self) -> bool:
        """Verifica che i messaggi di fallback siano disponibili nel database"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.frontend_url}/api/fallback-messages",
                    timeout=self.frontend_timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    messages = data.get('messages', {})
                    
                    # Verifica che almeno i messaggi di base siano presenti
                    required_messages = ['technical', 'rate_limit', 'emergency']
                    available = all(msg in messages for msg in required_messages)
                    
                    if available:
                        logger.info("Fallback messages available in database",
                                   available_count=len(messages))
                    else:
                        logger.warning("Some required fallback messages missing",
                                     missing=[msg for msg in required_messages if msg not in messages])
                    
                    return available
                else:
                    logger.warning("Failed to verify fallback messages availability",
                                 status_code=response.status_code)
                    return False
                    
        except Exception as e:
            logger.warning("Fallback messages not available in database",
                          error=str(e))
            return False
    
    def _determine_fallback_message_type(self, error: Exception) -> str:
        """Determina quale messaggio di fallback utilizzare in base al tipo di errore"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Errori di sovraccarico/rate limiting
        if ('429' in error_message or 'rate limit' in error_message or 
            'too many requests' in error_message or 'overload' in error_message):
            return 'rate_limit'
        
        # Errori di timeout
        if ('timeout' in error_message or 'timed out' in error_message):
            return 'timeout'
        
        # Errori di rete
        if ('network' in error_message or 'connection' in error_message or 
            'unreachable' in error_message or 'dns' in error_message):
            return 'network_error'
        
        # Errori API del provider
        if ('api' in error_message or 'provider' in error_message or 
            'service unavailable' in error_message):
            return 'api_error'
        
        # Errori tecnici generici
        return 'technical'
    
    async def _get_fallback_message(self, message_name: str, topic: str, model_id: str) -> str:
        """Ottiene un messaggio di fallback dal database e sostituisce i placeholder"""
        try:
            # Carica messaggi di fallback dal database tramite API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.frontend_url}/api/fallback-messages",
                    timeout=self.frontend_timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    messages = data.get('messages', {})
                    
                    if message_name in messages:
                        message_template = messages[message_name]['message']
                        
                        # Sostituisce i placeholder nel messaggio
                        fallback_content = message_template.format(
                            topic=topic,
                            model=model_id
                        )
                        
                        logger.info("Fallback message retrieved successfully",
                                   message_name=message_name,
                                   content_length=len(fallback_content))
                        
                        return fallback_content
                    else:
                        raise Exception(f"Fallback message '{message_name}' not found in database")
                else:
                    raise Exception(f"Failed to fetch fallback messages: {response.status_code}")
                    
        except Exception as e:
            logger.error("Failed to get fallback message",
                        message_name=message_name,
                        error=str(e))
            # Fallback al messaggio di emergenza
            return f"Sistema temporaneamente non disponibile per il dibattito su \"{topic}\". Il modello {model_id} riproverà a breve."
    
    async def _save_message_to_database(self, debate_id: str, message_data: dict):
        """Salva messaggio nel database tramite API Next.js"""
        try:
            # Costruisci URL API Next.js
            api_url = f"{self.frontend_url}/api/debates/{debate_id}/messages"
            
            # Prepara payload per API
            payload = {
                'modelId': message_data.get('modelId'),
                'content': message_data.get('content'),
                'timestamp': message_data.get('timestamp'),
                'turnNumber': message_data.get('turnNumber'),
                'type': message_data.get('type', 'ai')
            }
            
            logger.info("Saving message to database",
                       debate_id=debate_id,
                       model_id=payload['modelId'],
                       turn_number=payload.get('turnNumber'),
                       content_length=len(payload['content']) if payload['content'] else 0)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(api_url, json=payload, timeout=self.frontend_timeout)
                
                if response.status_code == 200:
                    saved_message = response.json()
                    logger.info("Message saved successfully",
                               debate_id=debate_id,
                               message_id=saved_message.get('id'),
                               turn_number=saved_message.get('turnNumber'))
                else:
                    error_text = response.text
                    logger.error("Failed to save message to database",
                                debate_id=debate_id,
                                status_code=response.status_code,
                                error_response=error_text[:200])
                                
        except Exception as e:
            logger.error("Exception saving message to database",
                        debate_id=debate_id,
                        error=str(e),
                        error_type=type(e).__name__)
    
    async def _send_response_to_debate(self, debate_id: str, message: dict):
        """Invia risposta al dibattito via WebSocket"""
        try:
            from websocket_manager import debate_manager
            await debate_manager.broadcast_to_debate(debate_id, message)
        except Exception as e:
            logger.error("Failed to send WebSocket response",
                    debate_id=debate_id,
                    error=str(e),
                    error_type=type(e).__name__)
    
    async def _schedule_next_message(self, current_request: MessageRequest):
        """Programma automaticamente il prossimo messaggio nel dibattito"""
        try:
            # Verifica che il dibattito sia ancora attivo
            debate_id = current_request.debate_id
            
            # Recupera lo stato del dibattito
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.frontend_url}/api/debates/{debate_id}",
                    timeout=self.frontend_timeout
                )
                
                if response.status_code != 200:
                    logger.warning("Cannot get debate status for auto-continuation",
                                 debate_id=debate_id,
                                 status_code=response.status_code)
                    return
                
                debate_data = response.json()
                
                # Solo continua se il dibattito è ancora live
                if debate_data.get('status') != 'live':
                    logger.info("Debate not live, stopping auto-continuation",
                              debate_id=debate_id,
                              status=debate_data.get('status'))
                    return
                
                # Recupera lista partecipanti
                participants = debate_data.get('participants', [])
                if len(participants) < 2:
                    logger.warning("Not enough participants for auto-continuation",
                                 debate_id=debate_id,
                                 participant_count=len(participants))
                    return
                
                logger.info("Scheduling next message for debate auto-continuation",
                           debate_id=debate_id,
                           participants=participants,
                           current_turn=current_request.message_count + 1)
                
                # Programma il prossimo messaggio con un delay
                async def continue_debate():
                    await asyncio.sleep(5)  # Attesa 5 secondi tra messaggi
                    
                    # Richiama il debate manager per continuare
                    try:
                        from debate_manager import get_debate_service
                        from config_manager import get_config
                        
                        config = get_config()
                        debate_service = get_debate_service(
                            config.get_frontend_url(),
                            config.get_frontend_timeout()
                        )
                        
                        # Continua il dibattito automaticamente
                        await debate_service.continue_debate({
                            'debate_id': debate_id,
                            'models': participants,
                            'topic': debate_data.get('title', ''),
                            'recent_messages': []  # Il debate_service recupererà i messaggi dal DB
                        })
                        
                    except Exception as e:
                        logger.error("Failed to auto-continue debate",
                                   debate_id=debate_id,
                                   error=str(e),
                                   error_type=type(e).__name__)
                
                # Avvia task asincrono per continuazione
                asyncio.create_task(continue_debate())
                
        except Exception as e:
            logger.error("Failed to schedule next message",
                        debate_id=current_request.debate_id,
                        error=str(e),
                        error_type=type(e).__name__)
    
    async def _cleanup_expired_requests(self):
        """Rimuove richieste scadute"""
        while self.running:
            try:
                current_time = datetime.now()
                expired_timeout = timedelta(minutes=5)
                
                async with self.request_lock:
                    expired_requests = [
                        req_id for req_id, req in self.active_requests.items()
                        if current_time - req.created_at > expired_timeout
                    ]
                    
                    for req_id in expired_requests:
                        expired_req = self.active_requests[req_id]
                        del self.active_requests[req_id]
                        logger.warning("Expired request removed",
                                     request_id=req_id,
                                     debate_id=expired_req.debate_id,
                                     model_id=expired_req.model_id,
                                     age_minutes=(current_time - expired_req.created_at).total_seconds() / 60)
                
                await asyncio.sleep(60)  # Cleanup ogni minuto
                
            except Exception as e:
                logger.error("Error in cleanup worker",
                           error=str(e),
                           error_type=type(e).__name__)
                await asyncio.sleep(60)
    
    def _update_average_response_time(self, response_time: float):
        """Aggiorna tempo di risposta medio"""
        current_avg = self.stats['average_response_time']
        total_successful = self.stats['successful_requests']
        
        if total_successful == 1:
            self.stats['average_response_time'] = response_time
        else:
            # Media pesata
            self.stats['average_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
    
    def get_queue_stats(self) -> dict:
        """Restituisce statistiche delle code"""
        queue_sizes = {
            priority.name: queue.qsize() 
            for priority, queue in self.priority_queues.items()
        }
        
        avg_queue_wait = (
            sum(self.stats['queue_wait_times']) / len(self.stats['queue_wait_times'])
            if self.stats['queue_wait_times'] else 0
        )
        
        return {
            'queue_sizes': queue_sizes,
            'active_requests': len(self.active_requests),
            'stats': {
                **self.stats,
                'average_queue_wait_time': avg_queue_wait
            },
            'model_rates': {
                model: last_req.isoformat() if last_req else None
                for model, last_req in self.model_last_request.items()
            }
        }

# Get configuration from config manager
config = get_config()
frontend_url = config.get_frontend_url()
frontend_timeout = config.get_frontend_timeout()

logger.info("Frontend configuration loaded from config manager", 
           url=frontend_url, 
           timeout=frontend_timeout,
           environment=config.get_environment())

# Istanza globale con concorrenza configurabile per evitare ban API key
llm_queue = LLMQueueManager(
    max_concurrent_requests=config.get_openrouter_max_concurrent(),
    frontend_url=frontend_url,
    frontend_timeout=frontend_timeout
)